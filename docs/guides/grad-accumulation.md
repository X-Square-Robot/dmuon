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

`dmuon.no_sync(model)` disables gradient communication for **both** parameter types:

- **Dedicated params**: Skips the reduce to owner. Gradients accumulate locally on each rank in `_accumulated_grad`.
- **Symmetric params**: Calls `model.set_requires_gradient_sync(False)` to skip FSDP2's reduce-scatter.

On the next backward **outside** `no_sync()`:

- **Dedicated params**: The accumulated gradient is merged with the new gradient before reduce.
- **Symmetric params**: FSDP2 automatically handles accumulated gradients.

After `optimizer.step()`, calling `optimizer.zero_grad()` clears both `_reduced_grad` and `_accumulated_grad`.

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
    Note that `reshard_after_forward=False` (DMuon-Z2) interacts with gradient accumulation by keeping packed buffers resident across micro-batches — see [Z2 vs Z3 Modes](z2-z3-modes.md) for the memory implication.

## See also

- [Training Guide](training.md) — full DMuon training workflow
- [Z2 vs Z3 Modes](z2-z3-modes.md) — packed-buffer lifecycle and its effect on memory during accumulation
