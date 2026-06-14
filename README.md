<p align="center">
  <img src="assets/dmuon_icon.jpg" alt="DMuon icon" width="120" />
</p>

<h3 align="center">DMuon</h3>

<p align="center">
  <em>Drop-in Distributed Muon optimizer implementation in Near-AdamW cost</em>
</p>

---

<p align="center">
  <img src="assets/dmuon-banner.png" alt="DMuon" width="100%" />
</p>

<p align="center">
  <img alt="Tech Report" src="https://img.shields.io/badge/Tech%20Report-Coming%20Soon-lightgrey">
  <img alt="Wiki" src="https://img.shields.io/badge/Wiki-WIP-lightgrey">
</p>

**DMuon** is a high-performance distributed implementation of the Muon optimizer that drops
into any existing training pipeline in **just 3 lines of code**. Through fine-grained kernel
tuning, load-balanced work scheduling, and a redesigned distributed communication path, DMuon
delivers **near-AdamW step time** while keeping Muon's optimization benefits — fully
plug-and-play, with no changes to your model.

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

Measured on a 16-node cluster, DMuon runs the matrix optimizer at roughly AdamW step time.

| Model | AdamW step | DMuon step | Δ vs AdamW |
|:------|-----------:|-----------:|-----------:|
| WallX| 1259 ms | 1285 ms | +2.1% |
| Pi0| 1617 ms | 1645 ms | +1.7 % |
| Wall-WM| 3309 ms | 3424 ms | 3.4 % |


## Acknowledgments

DMuon builds upon the ideas and engineering of several excellent prior works:

- **[Muon](https://kellerjordan.github.io/posts/muon/)** by Keller Jordan et al. — the original Muon optimizer, which orthogonalizes momentum updates via Newton-Schulz iteration.
- **[Moonlight](https://github.com/MoonshotAI/Moonlight)** ([*Muon is Scalable for LLM Training*](https://arxiv.org/abs/2502.16982)) by the Moonshot AI (Kimi) team — which demonstrated Muon's scalability to large-scale LLM training and introduced the weight-decay and update-scale adjustments that make it practical as a drop-in AdamW replacement. Our distributed design is heavily inspired by their ZeRO-1 Muon implementation.
- **[Gram Newton-Schulz](https://github.com/Dao-AILab/gram-newton-schulz)** ([blog post](https://tridao.me/blog/2026/gram-newton-schulz/)) by Jack Zhang, Noah Amsel, Berlin Chen, and Tri Dao — a hardware-aware reformulation of Newton-Schulz that iterates on the Gram matrix and exploits its symmetry with dedicated CuTeDSL GEMM kernels. Our SYRK-based kernel design follows this line of work.

We thank the authors of these projects for open-sourcing their code and insights.