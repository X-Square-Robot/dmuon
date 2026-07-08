# DMuon

**Dedicated ownership for matrix optimizers on PyTorch DDP, FSDP2, and HSDP.**

*One owner. One Newton-Schulz. Zero optimizer all-gather.*

---

<div class="dmuon-hero" markdown>

DMuon assigns each matrix parameter to a single **owner rank**. The owner stores the full parameter, reduces gradients from peers, and runs Newton-Schulz alone — eliminating the all-gather and redundant compute that make naive FSDP2+Muon 3–4× slower than AdamW.

Scale from a single node to multi-node HSDP clusters with a two-line API change. The 2D mesh, two-stage reduce, and async forward-hidden broadcast are all handled internally.

</div>

---

## What DMuon Can Do

??? abstract "LLM Pretraining — Llama, Qwen, Mistral on FSDP2/HSDP"

    Train transformer language models with Muon at near-AdamW cost.
    Dedicated ownership routes each projection parameter to a single owner;
    Newton-Schulz runs once per step with zero optimizer all-gather.
    Tested on Qwen2.5 (1.5B–7B) and Llama-3 (3B–8B) on 8×A800, with step
    overhead of only 4–13% vs FSDP2+AdamW.

    ```python
    import dmuon
    from torch.distributed.fsdp import fully_shard

    dmuon.dedicate_params(
        model, mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    optimizer = dmuon.Muon(model, lr=0.02, ns_steps=5, adamw_lr=1e-3)
    ```

??? abstract "Multi-Node HSDP — 2D mesh with async broadcast hiding"

    Scale beyond one node using a `(replicate, shard)` 2D device mesh.
    DMuon performs a two-stage gradient reduce (shard → replicate) and
    dispatches the post-step replicate broadcast on a dedicated CUDA stream,
    hiding it behind the next iteration's forward compute. Bit-identical to
    the synchronous baseline; falls back automatically if the broadcast
    cannot hide.

    ```python
    hsdp = init_device_mesh(
        "cuda", (replicate_size, shard_size),
        mesh_dim_names=("replicate", "shard"),
    )
    dmuon.dedicate_params(
        model, hsdp["shard"],
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
        replicate_mesh=hsdp["replicate"],
    )
    optimizer = dmuon.Muon(model, lr=0.02, replicate_async=True)
    ```

??? abstract "VLA and VLM — Vision-Language-Action and Vision-Language Models"

    DMuon's predicate-based selection works with any architecture.
    For VLMs and VLAs, start by applying the predicate to trainable attention
    and MLP projection layers that need Muon. Parameters not selected by the
    predicate, such as embeddings, frozen vision towers, or task heads, remain
    under standard FSDP2. If the vision encoder is trainable and uses
    compatible projection layers, include it by extending the predicate and hook
    boundaries.
    TP compatibility via Gram Newton-Schulz (O(d_model²) communication)
    keeps DMuon usable with column/row-parallel tensor parallelism.

??? abstract "MoE — Mixture-of-Experts with expert-parallel layouts"

    Hook boundaries can be set to align with expert modules using
    `hook_boundary_predicate`. Each expert's projection parameters are
    independently assigned to an owner rank; balanced partition ensures
    no single rank becomes a straggler across expert groups.

---

## Key Features

<div class="grid cards" markdown>

-   :material-server-network:{ .lg .middle } **HSDP Native**

    2D `(replicate, shard)` mesh with two-stage reduce and async
    forward-hidden broadcast. Single API change from 1D shard-only.

-   :material-layers-triple:{ .lg .middle } **DMuon-Z2 / DMuon-Z3**

    Mirror FSDP2's `reshard_after_forward` for Muon-target parameters.
    Z3 (default) is memory-optimal; Z2 saves one broadcast per layer.

-   :material-transit-connection-variant:{ .lg .middle } **Hook Boundary Control**

    `hook_boundary_predicate` decouples hook attachment from partition.
    Align exactly with your `fully_shard()` boundaries for any architecture.

-   :material-check-decagram:{ .lg .middle } **Bit-Identical Correctness**

    Async and sync HSDP paths produce identical loss trajectories.
    Validated on 4-GPU (G=2, R=2) and tested via checkpoint restart.

-   :material-puzzle:{ .lg .middle } **FSDP2 Compatible**

    No modifications to FSDP2 internals. A lightweight monkey-patch
    makes `fully_shard()` skip dedicated params automatically on import.

-   :material-scale-balance:{ .lg .middle } **Apache 2.0**

    Permissive license. Use in research or production without restriction.

</div>

---

## Benchmarks

Current snapshot: A800-SXM4-80GB, bf16, seq=4096 for LLM runs, random initialization, synthetic data. MFU is computed from step-start intervals so cross-step communication overlap is not double-counted. These rows are point-in-time research-preview summaries from controlled A800 runs; use them as relative performance context rather than a public reproduction recipe.

### LLM Z2/Z3 Scaling

| Model / 128GPU | AdamW MFU range | DMuon MFU range | Best DMuon topology |
|:---------------|---------------:|---------------:|:--------------------|
| Qwen2.5-1.5B | 36.3–43.6% | 38.4–43.1% | HSDP-Z2, 43.1% |
| Qwen2.5-7B | 43.0–48.3% | 39.7–48.0% | FSDP-Z2, 48.0% |
| Llama-3.2-3B | 46.1–48.1% | 46.4–48.6% | HSDP-Z2, 48.6% |
| Llama-3.1-8B | 47.0–49.9% | 41.1–46.2% | FSDP-Z2, 46.2% |

The 128GPU rows cover FSDP-Z2, FSDP-Z3, HSDP-Z2, and HSDP-Z3.


---

## Getting Started

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    Install DMuon from source and verify your CUDA environment.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    Running scripts for DDP-style, FSDP2, and HSDP — pick your topology.

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-head-lightbulb:{ .lg .middle } **Core Concepts**

    Dedicated ownership, Z2/Z3 modes, hook boundaries, and HSDP design.

    [:octicons-arrow-right-24: Core Concepts](getting-started/concepts.md)

-   :material-server-network:{ .lg .middle } **HSDP Guide**

    Complete walkthrough: 2D mesh, async mode, and checkpointing.

    [:octicons-arrow-right-24: HSDP Guide](guides/hsdp.md)

</div>

---

DMuon builds on dedicated ownership pioneered by ZeRO-1 (Rajbhandari et al., 2020) and Distributed Shampoo (Shi et al., 2023). Gram Newton-Schulz kernel adapted from Dao et al., 2026.

GitHub: [X-Square-Robot/dmuon](https://github.com/X-Square-Robot/dmuon) &nbsp;·&nbsp; arXiv preprint: [TBD]
