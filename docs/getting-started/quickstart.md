# Quick Start

This guide gets you from zero to a running distributed training with DMuon in 5 minutes.

## Prerequisites

- DMuon installed ([Installation](installation.md))
- 2+ GPUs available on a single node

## Minimal Training Script

Create `train_minimal.py`:

```python
"""Minimal DMuon training example."""
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

import dmuon  # (1)!


# Simple model
class TinyMLP(nn.Module):
    def __init__(self, d=512, ff=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "gate_proj": nn.Linear(d, ff, bias=False),  # (2)!
                "up_proj": nn.Linear(d, ff, bias=False),
                "down_proj": nn.Linear(ff, d, bias=False),
                "ln": nn.LayerNorm(d),
            })
            for _ in range(4)
        ])
        self.head = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            h = layer["ln"](x)
            x = x + layer["down_proj"](layer["gate_proj"](h) * layer["up_proj"](h))
        return self.head(x).sum()


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    mesh = init_device_mesh("cuda", (world_size,))

    torch.manual_seed(42)
    model = TinyMLP().cuda()

    # --- DMuon setup (3 lines) ---
    dmuon.dedicate_params(  # (3)!
        model, mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)  # (4)!
    fully_shard(model, mesh=mesh)

    optimizer = dmuon.Muon(  # (5)!
        model, lr=0.02, momentum=0.95, ns_steps=5,
        adamw_lr=1e-3,
    )

    # --- Training loop ---
    for step in range(20):
        optimizer.zero_grad()
        x = torch.randn(4, 512, device="cuda")
        loss = model(x)
        loss.backward()
        optimizer.step()

        if rank == 0 and step % 5 == 0:
            print(f"step {step:3d}  loss={loss.item():.4f}")

    dist.destroy_process_group()
    if rank == 0:
        print("Done!")


if __name__ == "__main__":
    main()
```

1. `import dmuon` auto-patches FSDP2 so that `fully_shard()` skips dedicated params.
2. The `proj` layers are 2D matrix parameters — these will use Muon with Newton-Schulz. LayerNorm is 1D — it will use AdamW.
3. Mark which parameters get dedicated ownership. The `predicate` selects 2D projection layers.
4. Apply FSDP2 as usual. Dedicated params are automatically skipped.
5. `dmuon.Muon` manages both parameter types: Newton-Schulz for dedicated params, AdamW for the rest.

## Run It

```bash
torchrun --nproc_per_node=4 train_minimal.py
```

Expected output:

```
step   0  loss=3.2145
step   5  loss=1.0832
step  10  loss=0.4217
step  15  loss=0.1583
Done!
```

## What Just Happened?

In those 3 setup lines, DMuon did the following:

1. **`dedicate_params()`** — Assigned each projection layer to an owner rank using balanced partition. The owner stores the full parameter; other ranks hold empty placeholders.

2. **`fully_shard()`** — FSDP2 shards all *non-dedicated* parameters (LayerNorm) as usual. Dedicated params are skipped automatically.

3. **`dmuon.Muon()`** — Created an optimizer that runs Newton-Schulz on each owner's dedicated params and AdamW on each rank's FSDP2 shards.

During training, every forward/backward step:

- **Forward**: Owner broadcasts full params to all ranks
- **Backward**: Gradients are reduced back to the owner
- **Step**: Owner runs Newton-Schulz on its params; all ranks run AdamW on their FSDP2 shards

No all-gather. No redundant NS compute.

## Next

To understand *why* this works and *how* it composes with FSDP2, read [Core Concepts](concepts.md).
