# DMuon

> Dedicated ownership for [Muon](https://arxiv.org/abs/2502.16982) on PyTorch FSDP2.
> **One owner. One Newton-Schulz. Zero optimizer all-gather.**

<p align="center">
  <img src="assets/dmuon-banner.png" alt="DMuon" width="100%" />
</p>

<p align="center">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-enabled-76B900?logo=nvidia&logoColor=white">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-black"></a>
</p>

DMuon assigns each matrix parameter to a single **owner rank**: the owner broadcasts the
full parameter in forward, gradients are reduced to the owner in backward, and Newton-Schulz
runs **only on the owner** in the optimizer step. This removes the all-gather that standard
FSDP2 + Muon needs and cuts redundant NS compute from `R` ranks to `1`. Composes with FSDP2,
HSDP (multi-node), DDP, and Tensor Parallel.

## Install

```bash
git clone git@github.com:X-Square-Robot/dmuon.git && cd dmuon
pip install -e .
```

## 3-Line Integration

```python
import dmuon  # auto-patches FSDP2

# Mark which params get dedicated ownership (auto-balanced across ranks)
dmuon.dedicate_params(model, dp_mesh, predicate=lambda n, p: "proj" in n and p.ndim == 2)

# Use FSDP2 as usual — dedicated params are handled automatically
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)
```

Forward broadcast, backward reduce, and owner-only optimizer execution are handled by hooks.

## Full Training Example

```python
import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
import dmuon

# Setup
mesh = init_device_mesh("cuda", (world_size,))
model = MyModel().cuda()

# Apply DMuon + FSDP2
dmuon.dedicate_params(model, mesh, predicate=lambda n, p: "proj" in n and p.ndim == 2)
for layer in model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

# Muon for dedicated matrix params, AdamW for the rest — handled automatically
optimizer = dmuon.Muon(model, lr=0.02, ns_steps=5, adamw_lr=1e-3)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
```

For multi-node, pass a 2D `(replicate, shard)` mesh to `dedicate_params(..., replicate_mesh=...)`
and `fully_shard(..., mesh=hsdp)`; everything else is identical.

## Benchmark

Measured on a 128-node cluster, DMuon runs the matrix optimizer at roughly AdamW step time.

| Model | AdamW step | DMuon step | Δ vs AdamW |
|:------|-----------:|-----------:|-----------:|
| WallX| 1492 ms | 1522 ms | +2.0% |
| PI05| xxx ms | xxx ms | +1.9 % |
