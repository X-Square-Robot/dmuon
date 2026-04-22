# Troubleshooting

!!! tip "TL;DR"
    Most issues fall into four categories: import / installation, training setup
    (bad predicate or mesh shape), runtime correctness (NaN, divergence), or
    performance (slow step, no overlap, OOM).  Check the symptom table for your
    error, apply the fix, then verify with a single-GPU smoke test before
    re-running distributed.

---

## Installation

??? warning "ImportError: No module named 'dmuon'"
    **Cause:** DMuon is not installed in the current Python environment.

    **Fix:**
    ```bash
    git clone https://github.com/StarrickLiu/dmuon && cd dmuon
    pip install -e .
    ```
    Verify: `python -c "import dmuon; print(dmuon.__version__)"`.

---

??? warning "ImportError: cannot import name 'fully_shard' from 'torch.distributed.fsdp'"
    **Cause:** PyTorch version is too old.  FSDP2 (`fully_shard` from
    `torch.distributed.fsdp`) requires PyTorch 2.4+.

    **Fix:** upgrade PyTorch.
    ```bash
    pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cu121
    ```

---

??? warning "CUDA extension fails to load / CuteDSL SYRK not available"
    **Cause:** CUDA version mismatch or missing CuteDSL dependencies.

    **Fix:** DMuon automatically falls back to `@torch.compile` PyTorch for NS
    when the SYRK kernel is unavailable.  Verify the active backend:
    ```python
    import dmuon
    print(dmuon.get_ns_backend())  # "compiled" is acceptable
    ```
    For SM80+ SYRK, ensure CUDA 11.8+ and that the `cutedsl` wheel in the repo
    is compiled against your CUDA toolkit.

---

## Training setup

??? warning "TypeError: dedicate_params() got an unexpected keyword argument '...'"
    **Cause:** version mismatch between user code and the installed DMuon.
    Common case: user code uses `replicate_mesh=` or `hook_boundary_predicate=`
    from a newer API, but the installed package is older.

    **Fix:** pull latest and reinstall:
    ```bash
    git pull && pip install -e .
    ```
    Check `dmuon.__version__` matches what you expect.

---

??? warning "Warning: 'dedicate_params: no parameters matched the predicate'"
    **Cause:** the `predicate` function returned `False` for every parameter.
    Common causes: wrong string in predicate (e.g., `"proj"` when your model
    uses `"linear"`), or the model structure does not have 2D projection
    parameters.

    **Fix:** debug your predicate interactively:
    ```python
    for name, param in model.named_parameters():
        if param.ndim == 2:
            print(name, param.shape)
    ```
    Adjust the predicate to match the names you see.

---

??? warning "Wrong mesh shape — HSDP mesh must be 2D with names ('replicate', 'shard')"
    **Cause:** HSDP setup requires a 2D `DeviceMesh` with `mesh_dim_names=
    ("replicate", "shard")`.  Passing an unnamed or 1D mesh to `replicate_mesh`
    will fail.

    **Fix:**
    ```python
    from torch.distributed.device_mesh import init_device_mesh

    hsdp = init_device_mesh(
        "cuda",
        (replicate_size, shard_size),
        mesh_dim_names=("replicate", "shard"),
    )
    dmuon.dedicate_params(model, hsdp["shard"], replicate_mesh=hsdp["replicate"])
    ```

---

## Runtime correctness

??? warning "NaN in loss after a few steps"
    **Cause (most common):** upstream — not DMuon.  Check whether the same NaN
    appears with AdamW only.  If yes, the issue is in data loading, model
    architecture, or dtype mismatch.

    **If NaN appears only with DMuon:** check for mixed-precision mismatch.
    Ensure `compute_dtype` in `dedicate_params` matches the model's autocast
    dtype, or leave it as `None` to inherit the parameter dtype.

    **Persistent NaN in Gram NS:** if NaN appears only with the `"gram"`
    backend, try `deterministic=True` to isolate whether it is the SYRK
    kernel:
    ```python
    ns = dmuon.NewtonSchulz(deterministic=True)
    optimizer = dmuon.Muon(model, ns_backend=ns)
    ```

---

??? warning "'forward output type mismatch' / ModelOutput attribute access lost"
    **Cause:** DMuon's forward hook wraps the module output; in older versions
    `HuggingFace ModelOutput` namedtuple attribute access was lost after wrapping.

    **Fix:** this is resolved in the latest DMuon.  If you see it on the
    current `main`, open a GitHub issue with your model class and PyTorch version.

---

??? warning "Loss diverges from single-GPU or AdamW baseline"
    **Cause — coefficient mismatch:** if you switched from `"gram"` to
    `"direct"`, the learning rate may need tuning.  The two backends have
    different effective step sizes.

    **Cause — LR too high:** Muon's internal scaling is `0.2 * sqrt(max(m, n))`.
    Start with `lr=0.02` and reduce if divergence appears.

    **Cause — NS backend mismatch:** ensure every rank uses the same
    `ns_backend` configuration; mixing deterministic and non-deterministic
    modes is unsupported.

    **Debug:** compare loss curves between `"gram"` and `"direct"` backends
    on a small model first.

---

## Performance

??? warning "Optimizer step is slow (>>100 ms for a small model)"
    **Cause:** a group may have tripped the async→sync fallback.  Enable
    profiling and check the per-group wait table:
    ```bash
    DMUON_REPLICATE_PROFILE=1 torchrun --nproc_per_node=... train.py
    ```
    Then call `dmuon.replicate_profile_report(model)` at the end of training.

    **Fix:**
    ```python
    dmuon.reset_replicate_fallback(model)
    import dmuon.group as g
    g.REPLICATE_WAIT_THRESHOLD_US = 500  # raise threshold if IB is slow
    ```
    See [Profiling & Fallback](guides/profiling-and-fallback.md).

---

??? warning "Broadcast never overlaps with forward / no async speedup observed"
    **Cause 1:** network bandwidth is the bottleneck — replicate broadcast
    saturates IB before the forward pass can hide it.  Typical on NVLink-only
    nodes sharing a slow uplink.

    **Cause 2:** the forward pass is too fast relative to the broadcast
    (small model, short sequence length).  There is no compute to hide the
    communication behind.

    **Fix:** switch to sync mode to avoid unnecessary async book-keeping
    overhead:
    ```python
    optimizer = dmuon.Muon(model, lr=0.02, replicate_async=False)
    ```

---

??? warning "OOM on owner ranks"
    **Cause:** LPT (Longest Processing Time) partition may assign too many
    large parameters to a few owner ranks, causing memory imbalance.

    **Fix:** profile the balance:
    ```bash
    DMUON_PROFILE_BALANCE=1 torchrun --nproc_per_node=... train.py
    ```
    If the log shows high imbalance, verify that `_extract_layer_id` correctly
    identifies your model's layer structure.  For ViT-style models with
    `blocks.N` paths, ensure `blocks.N` appears in the FQN — otherwise all
    parameters may collapse to the same "layer" key.  See
    [Design / Architecture](design/architecture.md) and the ViT partition
    bug report in the internal notes.

---

## HSDP-specific

??? warning "Dangling async event on abrupt shutdown (KeyboardInterrupt / OOM)"
    **Cause:** the replicate-broadcast stream has a pending event that was not
    consumed before process exit.  **Fix:** benign — CUDA cleans up on exit.
    For a graceful handler: `dmuon.wait_all_replicate_broadcasts(model)`.

---

??? warning "Checkpoint save/load fails across different world sizes or mesh topologies"
    **Cause:** owner assignments are relative to the shard coordinate system.
    A G=8 checkpoint cannot load into a G=4 run.

    **Fix:** known limitation.  Save via `get_model_state_dict` (full unsharded
    tensors) and use `set_model_state_dict` to reload.  Do not reuse optimizer
    state dicts across topology changes — restart from model weights only.

---

??? warning "Fallback protocol stuck in sync mode after network improvement"
    **Cause:** a group has permanently tripped `_replicate_sync_fallback=True`
    and will not re-enable async automatically.

    **Fix:**
    ```python
    dmuon.reset_replicate_fallback(model)
    ```
    Call once per model after resolving the network issue.

---

## Tensor Parallelism

??? warning "HSDP + TP (3D parallelism) produces incorrect results"
    **Cause:** 2D HSDP × TP combination is not yet validated.  The TP Gram
    all-reduce and the HSDP replicate all-reduce may run on overlapping process
    groups in ways not tested.

    **Fix:** use 1D FSDP + TP instead.  Apply TP, then DMuon, then `fully_shard`
    on a 1D mesh.  See [Tensor Parallelism](guides/tp-support.md).

---

## See also

- [FAQ](faq/index.md)
- [Profiling & Fallback](guides/profiling-and-fallback.md)
- [HSDP guide](guides/hsdp.md)
- [API Reference](reference/api.md)
