# Checkpointing

DMuon provides checkpoint functions that handle both dedicated parameters (DMuon-managed) and symmetric parameters (FSDP2-managed) in a single unified state dict.

---

## Why Special Handling?

Dedicated parameters are stored only on the owner rank — `model.state_dict()` would only see empty placeholders on non-owner ranks. DMuon's checkpoint functions gather all parameters to produce a **standard state dict** that is compatible with single-GPU loading and HuggingFace.

## Save

```python
import torch
import torch.distributed as dist
import dmuon

# Gather state dicts (all ranks must call)
model_sd = dmuon.get_model_state_dict(model)
optim_sd = dmuon.get_optimizer_state_dict(model, optimizer)

# Only rank 0 writes to disk
if dist.get_rank() == 0:
    torch.save({"model": model_sd, "optim": optim_sd}, "checkpoint.pt")
dist.barrier()
```

!!! info "All ranks must call"
    `get_model_state_dict()` and `get_optimizer_state_dict()` are collective operations — all ranks must call them, even though only rank 0 saves the result.

## Load (Resume Training)

```python
# All ranks load the checkpoint
ckpt = torch.load("checkpoint.pt", map_location="cpu")

dmuon.set_model_state_dict(model, ckpt["model"])
dmuon.set_optimizer_state_dict(model, optimizer, ckpt["optim"])
```

This restores:

- **Model weights** — both dedicated and FSDP2-managed parameters
- **Optimizer state** — momentum buffers (Muon) and Adam moments (AdamW)
- **Step counters** — for correct AdamW bias correction

## Load Pretrained (No Optimizer State)

Loading a pretrained model (without optimizer state) works the same way:

```python
pretrained_sd = torch.load("pretrained_model.pt", map_location="cpu")
dmuon.set_model_state_dict(model, pretrained_sd)
```

This is compatible with:

- Single-GPU `torch.save(model.state_dict(), ...)` checkpoints
- HuggingFace `model.save_pretrained()` checkpoints (use `safetensors` or `bin` format)

## Full Example

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
import dmuon

def setup_model(mesh):
    """Build and wrap model."""
    model = MyModel().cuda()
    dmuon.dedicate_params(model, mesh, predicate=lambda n, p: "proj" in n and p.ndim == 2)
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)
    return model, optimizer

def save_checkpoint(model, optimizer, step, path="checkpoint.pt"):
    model_sd = dmuon.get_model_state_dict(model)
    optim_sd = dmuon.get_optimizer_state_dict(model, optimizer)
    if dist.get_rank() == 0:
        torch.save({"model": model_sd, "optim": optim_sd, "step": step}, path)
    dist.barrier()

def load_checkpoint(model, optimizer, path="checkpoint.pt"):
    ckpt = torch.load(path, map_location="cpu")
    dmuon.set_model_state_dict(model, ckpt["model"])
    dmuon.set_optimizer_state_dict(model, optimizer, ckpt["optim"])
    return ckpt.get("step", 0)

# --- Main ---
dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (dist.get_world_size(),))
model, optimizer = setup_model(mesh)

# Resume if checkpoint exists
start_step = 0
if os.path.exists("checkpoint.pt"):
    start_step = load_checkpoint(model, optimizer)

for step in range(start_step, total_steps):
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()
    optimizer.step()

    if (step + 1) % save_interval == 0:
        save_checkpoint(model, optimizer, step + 1)
```

## State Dict Format

### Model State Dict

The model state dict is in **standard PyTorch format** — a flat dict mapping fully-qualified parameter names to tensors:

```python
{
    "layers.0.self_attn.q_proj.weight": tensor(...),
    "layers.0.self_attn.k_proj.weight": tensor(...),
    "layers.0.ln.weight": tensor(...),
    ...
}
```

### Optimizer State Dict

The optimizer state dict has DMuon-specific structure with separate sections:

```python
{
    "fsdp2": { ... },        # FSDP2 param states (Adam moments)
    "dedicated": {            # Dedicated param states (momentum buffers)
        "layers.0.self_attn.q_proj.weight": {
            "momentum_buffer": tensor(...)
        },
        ...
    }
}
```
