# DMuon

**Dedicated ownership for Muon on PyTorch FSDP2.**

*One owner. One Newton-Schulz. Zero optimizer all-gather.*

---

## The Problem

Matrix optimizers like [Muon](https://arxiv.org/abs/2502.16982) need the **full gradient matrix** for Newton-Schulz orthogonalization. But FSDP2 shards everything — each rank holds only 1/R of the gradient.

The naive fix is expensive:

1. **All-gather** the full gradient to every rank — O(mn) extra communication
2. **Every rank** runs the same Newton-Schulz — R times redundant compute

For a 7B model on 8 GPUs, this adds **3-4x overhead** vs AdamW.

## The Solution

DMuon assigns each matrix parameter to a single **owner rank**. The owner is the only rank that stores the full parameter and runs Newton-Schulz — no all-gather, no redundant compute.

```
Standard FSDP2 + Muon           DMuon
========================        ========================
all-gather full gradient        reduce gradient to owner
  O(mn) communication             O(mn/R) communication
every rank runs NS              owner runs NS alone
  R times redundant                1 time total
```

| | Standard FSDP2 + Muon | DMuon |
|---|---|---|
| Optimizer comm | all-gather full gradient | **zero** |
| NS compute | R times (every rank) | **1 time** (owner only) |
| Total overhead vs AdamW | 200-400% | **4-13%** |

## Quick Preview

```python
import dmuon  # auto-patches FSDP2

# 1. Mark parameters for dedicated ownership
dmuon.dedicate_params(model, dp_mesh, predicate=lambda n, p: "proj" in n)

# 2. FSDP2 as usual — dedicated params are handled automatically
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)

# 3. Train with Muon
optimizer = dmuon.Muon(model, lr=0.02, momentum=0.95)
```

That's it. Forward broadcast, backward reduce, and owner-only Newton-Schulz are all automatic.

## Features

- **Zero optimizer communication** — owner already has the full gradient after reduce
- **1/R NS compute** — only the owner runs Newton-Schulz, not every rank
- **FSDP2 native** — works alongside `fully_shard()` with no modifications to FSDP2 internals
- **HSDP native** — 2D `(replicate, shard)` mesh with two-stage reduce and **async forward-hidden broadcast** out of the box. See [HSDP guide](guides/hsdp.md)
- **TP compatible** — Gram Newton-Schulz with TP SYRK decomposition, O(d_model^2) TP comm
- **Checkpoint compatible** — standard state dicts, works with HuggingFace and single-GPU loading
- **Gradient accumulation** — `no_sync()` context manager, same pattern as FSDP2

## Benchmarks

**8 x A800-SXM4-80GB, bf16, seq=2048**

| Model | FSDP2+AdamW | DMuon | Overhead |
|:------|----------:|------:|------:|
| Qwen2.5-1.5B | 328 ms | 340 ms | +4% |
| Llama-3.2-3B | 599 ms | 660 ms | +10% |
| Qwen2.5-7B | 1,108 ms | 1,222 ms | +10% |
| Llama-3.1-8B | 1,188 ms | 1,349 ms | +13% |

Optimizer step is **12-15x faster** than naive FSDP2+Muon.

## Next Steps

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    Install DMuon and verify your setup.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    Run your first distributed training in 5 minutes.

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-head-lightbulb:{ .lg .middle } **Core Concepts**

    Understand how dedicated ownership works and composes with FSDP2.

    [:octicons-arrow-right-24: Core Concepts](getting-started/concepts.md)

-   :material-server-network:{ .lg .middle } **HSDP (Multi-Node)**

    2D mesh training with async forward-hidden broadcast.

    [:octicons-arrow-right-24: HSDP Guide](guides/hsdp.md)

-   :material-book-open-variant:{ .lg .middle } **API Reference**

    Complete function signatures and parameter documentation.

    [:octicons-arrow-right-24: API Reference](reference/api.md)

</div>
