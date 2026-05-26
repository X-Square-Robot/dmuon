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
    pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu121
    ```

---

??? warning "CUDA extension fails to load / CuteDSL SYRK not available"
    **Cause:** CUDA version mismatch or missing CuteDSL dependencies.

    **Fix:** DMuon automatically falls back to cuBLAS (`torch.mm` / `torch.addmm`)
    for Gram-NS SYRK ops when no CuteDSL kernel is available.  Verify the active
    backend:
    ```python
    import dmuon
    print(dmuon.get_ns_backend())
    # "Gram NS · kernel=cublas (SM80, universal fallback)" is an acceptable state
    # — correctness is preserved, only SYRK acceleration is off.
    ```
    For the A-card `cute_sm80` fast path, install the `[syrk]` extras.  For
    SM90+ machines, install `dmuon[quack]` to pick up Tri Dao's quack SYRK
    automatically via `kernel="auto"`.  See
    [Backend dispatch](reference/newton-schulz.md#backend-dispatch) for the
    full auto-detection ladder.

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
    backend, switch to the cuBLAS reference kernel to isolate whether the
    fast SYRK path is the culprit:
    ```python
    ns = dmuon.NewtonSchulz(kernel="cublas")  # same as deterministic=True
    optimizer = dmuon.Muon(model, ns_backend=ns)
    ```
    If cuBLAS also NaNs, the problem is in the Gram iteration itself
    (coefficients, restart positions, input scale) rather than the kernel.

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

    **Cause — NS kernel mismatch across ranks:** ensure every rank uses
    the same `ns_backend` / `kernel=` setting; mixing different SYRK
    kernels (e.g. `cute_sm80` on some ranks and `cublas` on others) can
    accumulate numerical drift across the DP / replicate axes.  Run
    `dmuon.get_ns_backend()` on every rank and cross-check.

    **Debug:** compare loss curves between `"gram"` and `"direct"` backends
    on a small model first.

---

## Performance

??? warning "Optimizer step is slow (>>100 ms for a small model)"
    **Cause:** owner load may be imbalanced, or the post-step publish may be
    too large to hide behind the next forward pass.

    **Fix:** first compare sync and async timing with
    `dmuon.Muon(..., replicate_async=False/True)`.  If only a few owner ranks
    are slow, inspect the dedicated parameter assignment and consider a more
    even hook boundary or owner strategy.

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

    **Fix:** verify that `_extract_layer_id` correctly
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

## Tensor Parallelism

??? warning "`ValueError: DMuon requires named DeviceMesh for TP detection`"
    **Cause:** you passed a mesh without `mesh_dim_names` to
    `parallelize_module` / `fully_shard` / `dedicate_params`.  DMuon
    identifies the TP axis by subtracting DP dim names from each
    parameter's `DTensor.mesh_dim_names`, so names are mandatory.

    **Fix:** construct the mesh with names:
    ```python
    mesh = init_device_mesh("cuda", (dp_size, tp_size),
                            mesh_dim_names=("dp", "tp"))
    ```

??? warning "`RuntimeError: tp_scatter_delta_async: previous event still pending`"
    **Cause:** two consecutive `optimizer.step()` calls without an
    intervening forward (which is what drains the async scatter event).
    Usually a bug in a custom training loop that calls `step()` twice
    per iteration, or calls `step()` then saves a checkpoint without
    doing a forward first.

    **Fix:** do one forward between any two `step()` calls, OR switch
    to sync post-step to avoid the cross-call event:
    ```python
    optimizer = dmuon.Muon(model, lr=0.02, replicate_async=False)
    ```

??? info "HSDP × TP (3D mesh) — supported"
    The 3D mesh `(replicate, shard, tp)` is validated (see
    [TP support guide](guides/tp-support.md) and the internal reports
    `tp_design.md`, `tp_alignment_report.md`).  Sync and async post-step
    paths produce bit-identical loss trajectories.  The order is:

    1. `parallelize_module(model, mesh["tp"], plan)`
    2. `dmuon.dedicate_params(model, mesh["shard"], replicate_mesh=mesh["replicate"], ...)`
    3. `fully_shard(model, mesh=mesh["replicate","shard"])`

---

## See also

- [FAQ](faq/index.md)
- [HSDP guide](guides/hsdp.md)
- [API Reference](reference/api.md)
