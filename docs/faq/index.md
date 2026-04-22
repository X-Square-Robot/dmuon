# FAQ

!!! tip "TL;DR"
    Quick answers to common adoption questions.  Each entry cross-links to the
    relevant guide.  If your question is not here, open a GitHub Discussion or
    check [Troubleshooting](../troubleshooting.md).

---

??? question "Why isn't DMuon just ZeRO-1?"
    **ZeRO-1** shards optimizer state across all ranks, so each rank manages a
    1/N slice of the state.  For Adam this works well: each rank updates its
    own parameter slice independently.

    **The problem with matrix optimizers** is that Newton-Schulz cannot run on
    a matrix shard — it needs the complete (m, n) gradient to compute a
    meaningful orthogonal update.  A ZeRO-1 rank holding 1/N rows cannot
    orthogonalize correctly without first all-gathering the full matrix.

    **Dedicated ownership** goes further: one rank owns the *entire* parameter
    and runs NS locally.  No all-gather.  The price is that other ranks must
    receive a broadcast of the updated parameter — but that broadcast hides
    inside the next forward pass.

    See [Core Concepts](../getting-started/concepts.md) for a full walk-through.

---

??? question "Do I need HSDP?"
    **Single-node multi-GPU** — a 1D shard-only mesh is sufficient and simpler.
    Pass just `mesh` to `dedicate_params`; omit `replicate_mesh`.

    **Multi-node training** (`replicate_size ≥ 2`) — HSDP's two-stage reduce
    (shard → replicate) combined with DMuon's async post-step broadcast pays off:
    the replicate broadcast overlaps with the next forward pass, amortizing the
    inter-node IB cost.  At 16+ GPUs across two nodes, the async hide is
    typically worthwhile.

    Pass `replicate_mesh=hsdp["replicate"]` to `dedicate_params` and
    `replicate_async=True` (default) to `Muon` to get the full HSDP benefit.

    See [HSDP guide](../guides/hsdp.md).

---

??? question "When do I need hook_boundary_predicate?"
    The default heuristic looks for `layers.N` or `blocks.N` in the parameter's
    fully-qualified name to find the layer module for hook registration.  This
    works for standard Llama / Qwen `model.layers.N.mlp.*_proj` structures and
    standard ViT `visual.blocks.N.attn.*_proj` structures.

    You need `hook_boundary_predicate` when your model deviates:

    - **VLA models**: action heads sit outside the main layer stack
    - **MoE models**: expert parameters have different FQN patterns
    - **Nested multi-modal models**: vision encoder + LLM have separate layer
      numbering hierarchies
    - **Custom adapters / LoRA layers**: adapter names don't match `layers.N`

    Example for a VLA action head:

    ```python
    import dmuon

    dmuon.dedicate_params(
        model,
        mesh,
        predicate=lambda n, p: "proj" in n,
        hook_boundary_predicate=lambda m: hasattr(m, "_is_action_layer"),
        hook_boundary_strict=True,
    )
    ```

    See [Custom Hook Boundaries](../guides/custom-hook-boundaries.md).

---

??? question "Z2 or Z3?"
    **Default Z3** (`reshard_after_forward=True`) for any model with
    Muon-target parameters larger than available per-rank memory budget.
    Communication cost is `3(N-1)/N · P_M` bytes/step; peak memory per rank
    is low because broadcast buffers are freed after each forward.

    **Z2** (`reshard_after_forward=False`) when you can afford to keep P_M
    elements resident on every rank across forward + backward.  Communication
    cost drops to `2(N-1)/N · P_M` — the theoretical ring all-reduce lower
    bound.  Best for smaller models (≤ 3B params) where GPU memory is not the
    bottleneck.

    As a rough rule of thumb: use Z2 when the total Muon-target parameter bytes
    fit in less than 20 % of per-GPU VRAM after accounting for activations and
    optimizer state.  At 7B+ parameters, Z3 is almost always required.

    See [Z2 vs Z3 Modes](../guides/z2-z3-modes.md).

---

??? question "Can I mix DMuon with DeepSpeed?"
    **Short answer:** not today for ZeRO-3.  DeepSpeed ZeRO-3 uses a different
    parameter storage mechanism (`deepspeed.zero.Init` + custom hooks) that is
    incompatible with DMuon's `dedicate_params` + `fully_shard` contract.

    **ZeRO-0/1/2 with DeepSpeed** is on the roadmap — the dedicated-ownership
    primitive is compatible in principle since parameters are not fragmented at
    the storage level.  Contribution welcome; see
    [Integration Recipes](../guides/integration-recipes.md) for the current
    approach.

    **Current recommendation:** pair DMuon with **PyTorch FSDP2**.  This is the
    primary tested and supported configuration.

---

??? question "Bit-identical convergence guarantees?"
    Yes.  DMuon validates bit-identical outputs on a 4-GPU test harness
    (`tests/distributed/test_hsdp_correctness.py`) across three axes:

    1. **HSDP vs. shard-only:** DMuon-HSDP (G=2, R=2) produces identical loss
       values to shard-only DMuon (G=4) over 10 training steps.
    2. **Async vs. sync:** `replicate_async=True` (default) produces identical
       outputs to `replicate_async=False` (Phase B sync path).
    3. **Checkpoint restart:** resuming from a checkpoint produces identical
       loss values to an uninterrupted run over the same steps.

    These tests run on every PR.  If you observe divergence from a single-GPU
    baseline, check [Troubleshooting](../troubleshooting.md).

---

??? question "Does DMuon work with Tensor Parallelism?"
    Yes for **1D FSDP + TP**.  Apply TP first, then DMuon, then FSDP2:

    ```python
    from torch.distributed.tensor.parallel import parallelize_module
    import dmuon
    from torch.distributed.fsdp import fully_shard

    for layer in model.layers:
        parallelize_module(layer.mlp, tp_mesh, {...})   # TP first

    dmuon.dedicate_params(model, dp_mesh, ...)          # DMuon second

    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)                # FSDP2 third
    fully_shard(model, mesh=dp_mesh)
    ```

    Within a DP group all ranks share the same TP position, so broadcasting a
    TP shard is correct.  DMuon uses TP-aware Gram Newton-Schulz with O(d²)
    TP communication via all-reduce of Gram matrices.

    **2D HSDP × TP** (3D parallelism) is not yet validated.  See
    [Tensor Parallelism](../guides/tp-support.md).

---

??? question "How do I cite DMuon?"
    **DMuon itself:**

    ```bibtex
    @misc{DMuon,
      title   = {DMuon: Dedicated Parameter Ownership for Distributed Muon Training},
      author  = {Xingchen Liu},
      year    = {2026},
      url     = {https://github.com/StarrickLiu/dmuon}
    }
    ```

    **Gram Newton-Schulz** (if using the default `"gram"` backend):

    ```bibtex
    @misc{GramNewtonSchulz,
      title  = {Gram Newton-Schulz},
      author = {Jack Zhang and Noah Amsel and Berlin Chen and Tri Dao},
      year   = {2026},
      url    = {https://dao-ailab.github.io/blog/2026/gram-newton-schulz/}
    }
    ```

    **Muon optimizer** (Jordan et al., 2024):
    arXiv:2502.16982

    **Distributed Shampoo** (Shi et al., 2023) and **ZeRO-1**
    (Rajbhandari et al., 2020) pioneered the dedicated-ownership primitive that
    DMuon extends.

---

## See also

- [Troubleshooting](../troubleshooting.md)
- [Core Concepts](../getting-started/concepts.md)
- [API Reference](../reference/api.md)
- [Contributing](../contributing.md)
