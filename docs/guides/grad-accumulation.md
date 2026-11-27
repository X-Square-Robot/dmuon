# Gradient Accumulation

!!! tip "TL;DR"
    Wrap accumulation micro-batches with `dmuon.no_sync(model)` to suppress gradient communication on both dedicated and symmetric params. Call `optimizer.step()` + `optimizer.zero_grad()` only on the last micro-batch. Divide loss by `accum_steps` before `.backward()` so the accumulated gradient matches a single large-batch gradient.

---

## Basic Usage

```python
from contextlib import nullcontext
import dmuon

accum_steps = 4

for i, batch in enumerate(dataloader):
    # Skip reduce on accumulation steps
    is_accumulating = (i + 1) % accum_steps != 0
    ctx = dmuon.no_sync(model) if is_accumulating else nullcontext()

    with ctx:
        loss = model(batch).loss / accum_steps
        loss.backward()

    # Step only after accumulating all micro-batches
    if not is_accumulating:
        optimizer.step()
        optimizer.zero_grad()
```

## How It Works

### Keeping dedicated weights resident during accumulation

For a pure dedicated-ownership model configured with
`reshard_after_forward=False`, opt in around the complete accumulation window:

```python
optimizer.zero_grad()
with dmuon.retain_unsharded_params(model):
    for micro in range(accum_steps):
        ctx = dmuon.no_sync(model) if micro < accum_steps - 1 else nullcontext()
        with ctx:
            loss = model(batches[micro]).loss / accum_steps
            loss.backward()
optimizer.step()
```

This defers weight release after the `no_sync` backward passes. Gradient
accumulation and live-gradient clearing still run on every microbatch; the
last synchronized backward reduces and releases weights normally. Context
exit releases remaining weights, including prefetched groups, even if the
window is incomplete or raises. It does not discard accumulated gradients.

The context is off by default and requires the complete ownership model.
DDP and mixed FSDP2 models are unsupported. Full compute weights may remain
resident on every rank alongside gradients and activations, so validate peak
memory before enabling it. All ranks must follow the same schedule. Nested
retention contexts are supported; nested `no_sync` contexts are unsupported.

Call optimizer step, `zero_grad`, checkpoint save/load and update publication
outside the context. DMuon's public entry points reject these operations
while retention is active. Do not mutate parameters or call native
`load_state_dict` inside the context. If cleanup fails, rebuild the model
before continuing; no update or communication is replayed automatically.
Weight cleanup does not roll back a failed backward pass or guarantee that
its partially computed gradients can be reused.

### Gradient synchronization

`dmuon.no_sync(model)` disables gradient communication for **both** parameter types:

- **Dedicated params**: Skips the reduce to owner. Gradients accumulate locally on each rank in `_accumulated_grad`.
- **Symmetric params**: Calls `model.set_requires_gradient_sync(False)` to skip FSDP2's reduce-scatter.

On the next backward **outside** `no_sync()`:

- **Dedicated params**: The accumulated gradient is merged with the new gradient before reduce.
- **Symmetric params**: FSDP2 automatically handles accumulated gradients.

After `optimizer.step()`, calling `optimizer.zero_grad()` clears both `_reduced_grad` and `_accumulated_grad`.

!!! note "FP32 accumulation memory"
    With an explicit `grad_dtype=torch.float32` parameter policy, `no_sync()`
    stores dedicated gradients in an FP32 accumulator. For BF16 parameters,
    this buffer uses 4 bytes per element instead of 2 bytes for BF16
    accumulation, so its payload is twice as large. The accumulator is
    unsharded across the data-parallel mesh on each rank; with tensor
    parallelism, it covers only that rank's local TP slice. This 2x comparison
    applies to the accumulator alone. Count dtype-cast and merge temporaries,
    communication buffers, model gradients, and optimizer state separately
    when estimating peak memory. Without an explicit `grad_dtype`, the
    accumulator retains the raw gradient dtype.

## Example: Full Training Loop with Accumulation

```python
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
import dmuon

dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (dist.get_world_size(),))
model = MyModel().cuda()

dmuon.dedicate_params(model, mesh, predicate=lambda n, p: "proj" in n and p.ndim == 2)
for layer in model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)

accum_steps = 4
global_step = 0

for i, batch in enumerate(dataloader):
    is_accumulating = (i + 1) % accum_steps != 0
    ctx = dmuon.no_sync(model) if is_accumulating else nullcontext()

    with ctx:
        loss = model(batch).loss / accum_steps
        loss.backward()

    if not is_accumulating:
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

        if dist.get_rank() == 0:
            print(f"step {global_step}: loss={loss.item() * accum_steps:.4f}")
```

!!! tip "Loss scaling"
    Divide the loss by `accum_steps` before `.backward()` so the accumulated gradient is equivalent to a single large-batch gradient.

!!! note "DMuon-Z2 and gradient accumulation"
    `reshard_after_forward=False` (DMuon-Z2) retains packed buffers through the same microbatch's forward and backward. Cross-microbatch retention requires the opt-in context above; see [Z2 vs Z3 Modes](z2-z3-modes.md) for the memory implications of the default modes.

## See also

- [Training Guide](training.md) — full DMuon training workflow
- [Z2 vs Z3 Modes](z2-z3-modes.md) — packed-buffer lifecycle and its effect on memory during accumulation
