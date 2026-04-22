# Integration Recipes

!!! tip "TL;DR"
    DMuon composes with HuggingFace Trainer, torchtitan, and custom training loops —
    3 lines of setup after the usual FSDP2 wrap. Call `dmuon.dedicate_params` before
    `fully_shard`, then use `dmuon.Muon` as the optimizer. The training loop itself
    does not change.

---

## Design principle

DMuon integrates through two mechanisms:

1. **Monkey-patch on import** — `import dmuon` patches `fully_shard` so it
   automatically skips parameters that carry the `_dedicated_owner_rank` attribute.
   No modification to FSDP2 internals is required.
2. **Forward/backward hooks** — `dedicate_params` registers pre/post forward hooks
   on the chosen layer modules. These hooks issue the shard broadcast and gradient
   reduce on dedicated CUDA streams.

As long as the training loop calls `loss.backward()` followed by `optimizer.step()`,
DMuon slots in without further changes. Any framework that follows this contract works.

---

## HuggingFace Transformers and Accelerate

```python title="hf_dmuon.py"
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM
import dmuon

dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
mesh = init_device_mesh("cuda", (dist.get_world_size(),))

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B", torch_dtype=torch.bfloat16,
).cuda()

# Step 1: mark dedicated params BEFORE fully_shard.
dmuon.dedicate_params(
    model, mesh,
    predicate=lambda n, p: p.ndim == 2 and any(
        k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj")
    ),
)
# Step 2: apply FSDP2 — dedicated params are skipped automatically.
for layer in model.model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

# Step 3: use dmuon.Muon.
optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    outputs = model(**batch)
    outputs.loss.backward()
    optimizer.step()
```

### HuggingFace Trainer

Pass the optimizer directly via the `optimizers` argument:

```python title="hf_trainer_dmuon.py"
from transformers import Trainer, TrainingArguments
import dmuon

# ... model setup, dedicate_params, and fully_shard as above ...

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    fsdp="",           # disable Trainer's built-in FSDP wrapping
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    optimizers=(optimizer, None),   # (optimizer, lr_scheduler)
)
trainer.train()
```

!!! warning "Disable Trainer's built-in FSDP"
    Pass `fsdp=""` to `TrainingArguments` when you have already applied `fully_shard`
    manually. Applying `fully_shard` twice will raise an error or produce incorrect
    behaviour.

For Qwen-VL and other nested multi-modal models, pass `hook_boundary_predicate` to
`dedicate_params` before applying `fully_shard` (see
[Custom Hook Boundaries](custom-hook-boundaries.md)).

---

## torchtitan

torchtitan has its own parallel wrapping logic. Apply `dedicate_params` before
torchtitan's `parallelize_model` call, and DMuon's monkey-patch ensures `fully_shard`
skips the dedicated parameters:

```python title="torchtitan_dmuon.py"
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import dmuon

mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
model = build_model(config)

# Apply DMuon BEFORE torchtitan wraps the model.
dmuon.dedicate_params(
    model, mesh["dp"],
    predicate=lambda n, p: p.ndim == 2 and "proj" in n,
)
parallelize_model(model, mesh, config)   # torchtitan applies fully_shard + TP here

optimizer = dmuon.Muon(model, lr=config.lr, adamw_lr=config.adamw_lr)

for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

When using HSDP async mode, drain pending broadcasts before torchtitan's DCP checkpoint
save: `dmuon.wait_all_replicate_broadcasts(model)`. First-class torchtitan integration
is on the roadmap; the manual path above is fully supported today.

---

## DeepSpeed ZeRO

DMuon's current implementation targets FSDP2. The dedicated ownership primitive is
designed to be runtime-portable: parameters tagged by `dedicate_params` carry
`_dedicated_owner_rank`, and a DeepSpeed adapter could skip them during ZeRO
partitioning.

Current status:

- **ZeRO-0 / ZeRO-1 / ZeRO-2**: compatible in principle; no adapter exists yet.
- **ZeRO-3**: DeepSpeed ZeRO-3 bucket-based storage would need adapter-level work.

**This integration is on the roadmap, not yet shipped.** Use FSDP2 today.

---

## Custom training loops

The minimal required call sequence:

```python title="custom_loop.py"
import dmuon
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (world_size,))
model = MyModel().cuda()

dmuon.dedicate_params(
    model, mesh, predicate=lambda n, p: p.ndim == 2 and "proj" in n
)
for layer in model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()
    # optimizer.step() handles wait_all_reduces + NS + AdamW +
    # broadcast_all_updates internally.
    optimizer.step()
```

Optional manual overrides for specific situations:

```python title="manual_overrides.py"
import dmuon
from contextlib import nullcontext

# Before checkpoint save in HSDP async mode: drain pending broadcasts.
dmuon.wait_all_replicate_broadcasts(model)
model_sd = dmuon.get_model_state_dict(model)

# Gradient accumulation.
for i, batch in enumerate(dataloader):
    ctx = dmuon.no_sync(model) if (i + 1) % accum_steps != 0 else nullcontext()
    with ctx:
        loss = model(batch).loss / accum_steps
        loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Checkpointing across integrations

DMuon's checkpoint API is framework-agnostic (see [Checkpointing](checkpoint.md)):

```python title="checkpoint_save.py"
import torch
import torch.distributed as dist
import dmuon

model_sd = dmuon.get_model_state_dict(model)      # drains async broadcast first
optim_sd = dmuon.get_optimizer_state_dict(model, optimizer)
if dist.get_rank() == 0:
    torch.save({"model": model_sd, "optim": optim_sd}, "checkpoint.pt")
dist.barrier()

ckpt = torch.load("checkpoint.pt", map_location="cpu")
dmuon.set_model_state_dict(model, ckpt["model"])
dmuon.set_optimizer_state_dict(model, optimizer, ckpt["optim"])
```

For HuggingFace Trainer, add a training callback that calls
`dmuon.wait_all_replicate_broadcasts(model)` before each checkpoint save, or override
`_save_checkpoint` to use `dmuon.get_model_state_dict` directly.

---

## See also

- [Training Guide](training.md) — full single-node workflow
- [Checkpointing](checkpoint.md) — state-dict semantics and HSDP restart
- [HSDP Guide](hsdp.md) — multi-node setup with replicate mesh
- [Custom Hook Boundaries](custom-hook-boundaries.md) — for nested model structures
