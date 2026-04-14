# DMuon

> Dedicated parameter ownership for distributed training with matrix optimizers.

DMuon makes [Muon](https://arxiv.org/abs/2502.16982) work efficiently with PyTorch FSDP2. Each parameter is assigned to a single owner rank — the owner runs Newton-Schulz locally with **zero extra communication** and **1/R compute**.

## Why DMuon?

Matrix optimizers (Muon, Shampoo, SOAP) need the **full gradient matrix** for Newton-Schulz orthogonalization. With standard FSDP2, this means either:

- All-gathering the full gradient to every rank (**extra communication**)
- Every rank running NS independently (**R times redundant compute**)

DMuon eliminates both: the owner already has the complete gradient after `reduce`, and only the owner runs NS.

```
                    Standard FSDP2 + Muon          DMuon
Optimizer comm      all-gather full gradient        zero
NS compute          R times (every rank)            1 time (owner only)
```

## Getting Started

### Install

```bash
git clone https://github.com/user/dmuon && cd dmuon
pip install -e .
```

### 3-Line Integration

```python
import dmuon  # auto-patches FSDP2

# Mark which params get dedicated ownership (auto-balanced across ranks)
dmuon.dedicate_params(model, dp_mesh, predicate=lambda n, p: "proj" in n)

# Use FSDP2 as usual — dedicated params are handled automatically
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)
```

That's it. Forward, backward, and optimizer communication are handled by hooks.

### Full Training Example

```python
import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
import dmuon

# Setup
mesh = init_device_mesh("cuda", (world_size,))
model = MyModel().cuda()

# Apply DMuon + FSDP2
dmuon.dedicate_params(model, mesh, predicate=lambda n, p: "proj" in n)
for layer in model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

# Get this rank's owned params for optimizer
owned = dmuon.get_owned_params(model, mesh.get_local_rank())

# Training loop
for batch in dataloader:
    loss = model(batch).loss
    loss.backward()

    # Muon step — only on owned params, zero communication
    for dp in owned:
        if dp._reduced_grad is not None:
            update = newton_schulz(dp._reduced_grad)
            dp._owned_data.add_(update, alpha=-lr)
            dp._reduced_grad = None
```

## How It Works

### Two Parallel Systems

DMuon runs **alongside** FSDP2 — each manages a disjoint set of parameters:

```
Model Parameters
    |
    +-- proj layers (q/k/v/o/gate/up/down_proj)  -->  DMuon
    |     storage:   owner has full param, others empty
    |     forward:   broadcast from owner
    |     backward:  reduce to owner
    |     optimizer: owner runs NS (zero comm)
    |
    +-- other params (layernorm, embedding)  ---------> FSDP2
          storage:   every rank has 1/R shard
          forward:   all-gather
          backward:  reduce-scatter
          optimizer: every rank updates shard (SGD/AdamW)
```

### Composition via `ignored_params`

DMuon integrates with FSDP2 the same way Activation Checkpointing does — through hooks on the same module, with no modifications to FSDP2 internals:

1. `dedicate_params()` marks parameters with `_dedicated_owner_rank`
2. On `import dmuon`, a monkey-patch makes `fully_shard()` auto-skip marked params
3. DMuon registers its own forward/backward hooks for broadcast/reduce

### Balanced Partition

`dedicate_params` uses LPT (Longest Processing Time) with two constraints:

- **Global balance**: each rank owns ~model_size/R total parameters
- **Same-layer concurrency**: parameters in the same layer go to different ranks, enabling concurrent broadcasts
- **Small param packing**: k_proj + v_proj in the same layer share one owner for packed broadcast

### TP Compatibility

Works with Tensor Parallelism — apply TP first, then DMuon:

```python
parallelize_module(layer.mlp, tp_mesh, {...})   # TP first
dmuon.dedicate_params(model, dp_mesh, ...)      # DMuon second
fully_shard(layer, mesh=dp_mesh)                # FSDP2 third
```

Within a DP group, all ranks share the same TP position, so broadcasting a TP shard is correct.

## Benchmarks

**8 x A800-SXM4-80GB, bf16, seq_len=64**

### vs DDP + Muon

| Model | DDP + Muon | DMuon | Speedup |
|:------|----------:|------:|--------:|
| Qwen2.5-1.5B | 2,444 ms | 219 ms | **11.2x** |
| Llama-3.2-3B | 2,840 ms | 267 ms | **10.6x** |

### vs FSDP2 + Muon

| Model | FSDP2 + Muon | DMuon | Speedup |
|:------|------------:|------:|--------:|
| Qwen2.5-7B | 21,672 ms | 427 ms | **50.7x** |
| Llama-3.1-8B | 13,424 ms | 461 ms | **29.1x** |

All benchmarks verified: every rank produces identical loss values.

### Communication Primitives

| Data Size | AllGather | 5x Broadcast | Winner |
|:----------|--------:|--------:|:-------|
| 3.7 MB | 0.08 ms | 0.18 ms | AllGather |
| 25.7 MB | 0.53 ms | 0.66 ms | AllGather |
| 135.8 MB | 1.85 ms | 1.65 ms | **Broadcast** |
| 466 MB | 5.75 ms | 3.80 ms | **Broadcast** |

Broadcast wins for large parameters (>100 MB) and scales better with DP group size.

## Roadmap

- [x] Core ownership model
- [x] Balanced partition with concurrency constraints
- [x] FSDP2 composition
- [x] TP compatibility
- [x] LLM benchmarks (Qwen2.5, Llama-3)
- [x] Gram NS with TP SYRK decomposition (O(m^2) TP comm instead of O(mn))
- [x] CuteDSL SYRK kernel (5/7 Gram NS ops, 1.4-1.5x E2E speedup)
- [x] Prefetch pipeline (forward + backward)
- [ ] Gradient accumulation (`no_sync` context manager)
- [ ] State dict save/load

See [TODO.md](TODO.md) for detailed implementation plans.

## Acknowledgments

The Gram Newton-Schulz iteration in DMuon is adapted from
[Dao-AILab/gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz),
including per-step coefficients, restart mechanism, and the SYRK symmetry optimization.

The CuteDSL SYRK kernel is adapted from [quack](https://github.com/Dao-AILab/quack)
by Tri Dao et al.

## Citation

If you use DMuon, please also cite:

```bibtex
@misc{GramNewtonSchulz,
  title   = {Gram Newton-Schulz},
  author  = {Jack Zhang and Noah Amsel and Berlin Chen and Tri Dao},
  year    = {2026},
  url     = {https://dao-ailab.github.io/blog/2026/gram-newton-schulz/}
}
```

## License

Apache 2.0
