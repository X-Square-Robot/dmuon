# Installation

!!! tip "TL;DR"
    Install from source with `pip install -e .`. SYRK acceleration requires
    `pip install -e ".[syrk]"` and an SM80+ GPU (A100/A800/H100). Takes under
    2 minutes on a standard setup.

---

## Requirements

| Requirement | Minimum version | Notes |
|-------------|----------------|-------|
| Python | 3.10 | `match` syntax used internally |
| PyTorch | 2.6 | FSDP2 (`fully_shard`) API required |
| CUDA | 11.8 / 12.1 / 12.4 | All three variants tested |
| GPU SM | SM80+ | Required for SYRK kernel (optional) |
| NCCL | bundled with PyTorch | No separate install needed |

!!! note "SM80+ for SYRK"
    The CuteDSL SYRK kernel targets SM80+ (A100, A800, H100, H200).
    On older GPUs (SM70 / V100) DMuon still works — it falls back to
    `@torch.compile`'d pure PyTorch for Newton-Schulz, which is
    fully correct but ~1.5× slower on the optimizer step.

---

## Install Methods

=== "From source (recommended)"

    ```bash
    git clone https://github.com/X-Square-Robot/dmuon
    cd dmuon
    pip install -e .
    ```

    This installs the core library in editable mode. The SYRK kernel
    extension is **not** built; Newton-Schulz uses the compiled PyTorch fallback.

=== "pip (coming soon)"

    ```bash
    # Coming soon — not yet on PyPI
    pip install dmuon
    ```

    PyPI release is planned after the research preview. Until then,
    install from source (see the "From source" tab).

=== "Development (editable + test deps)"

    ```bash
    git clone https://github.com/X-Square-Robot/dmuon
    cd dmuon
    pip install -e ".[dev]"
    ```

    Installs the library in editable mode plus test, packaging, and
    documentation dependencies. Run the unit test suite to confirm everything
    works:

    ```bash
    pytest tests/unit/ -v
    ```

---

## Optional: SYRK Kernel Acceleration

The SYRK kernel exploits Gram matrix symmetry for ~1.5× speedup on
Newton-Schulz. It requires SM80+ hardware and additional build dependencies:

```bash
pip install -e ".[syrk]"
```

This pulls in:

- `nvidia-cutlass-dsl >= 4.4.2`
- `apache-tvm-ffi`
- `torch-c-dlpack-ext`

Build time is typically 1–3 minutes on first use (JIT compilation).
The compiled artifact is cached in `~/.cache/dmuon/`.

---

## Optional: Fast Gradient Clipping (CUDA)

DMuon ships an optional CUDA kernel that fuses **segmented gradient clipping**
— the per-bucket norm, clip coefficient, and in-place scaling for the
`regular` / `muon` / `adamw` gradient groups — into a single pass. The
training semantics are identical to the pure-Python path: each bucket still
gets its own norm and its own clip coefficient. Only the arithmetic moves to
the GPU.

`torch` is intentionally **not** a build dependency (pinning it would force an
isolated build to download a multi-GB generic torch and link the kernel against
it — an ABI-mismatch risk). So to compile the kernel, build **without isolation**
in an environment that already has torch, with a CUDA toolchain on `PATH`:

```bash
# nvcc / CUDA_HOME must be visible; torch must already be installed
pip install -e . --no-build-isolation
```

- With `--no-build-isolation` and `CUDA_HOME` present, `dmuon._fast_clip_cuda`
  is built against your real torch and used automatically.
- A plain `pip install -e .` (isolated build) has no torch in the build env, so
  `setup.py` skips the extension and DMuon uses the equivalent pure-Python clip
  at runtime. Nothing breaks — clipping is just computed on the host side.
- If the extension is **built but fails to load** (e.g. a later torch/CUDA
  upgrade breaks its ABI), DMuon warns once and falls back to Python. Set
  `DMUON_FAST_CLIP_VERBOSE=1` to raise the underlying error instead.

### Build & runtime toggles

| Variable | Effect |
|----------|--------|
| `DMUON_BUILD_FAST_CLIP=0` | Skip building the CUDA extension at install time. |
| `DMUON_FAST_CLIP=0` | Disable the fast path at runtime (use pure Python). |
| `DMUON_FAST_CLIP_CHUNK_SIZE` | Per-tensor chunk size for the kernel (default `262144`). |
| `DMUON_FAST_CLIP_VERBOSE=1` | Raise the import error instead of silently falling back — use when a build "should" have worked. |

The runtime path also falls back to Python automatically for inputs outside the
kernel contract (non-contiguous, sparse, unsupported dtype) or when a
non-finite bucket norm is detected, so a missing or stale extension never
changes results.

!!! note "The build needs a compiler, not a specific GPU"
    Compiling the clip kernels only requires a host CUDA compiler (`nvcc`), not
    an SM80+ GPU — the kernels are architecture-agnostic. Match the CUDA
    toolkit to your PyTorch CUDA build (11.8 / 12.1 / 12.4).

---

## Verify Installation

```python title="verify_install.py"
import dmuon
import torch

print(f"DMuon version : {dmuon.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")
print(f"NS backend     : {dmuon.get_ns_backend()}")
```

Expected output (SYRK installed, SM80+ GPU):

```
DMuon version : 0.2.0
PyTorch version: 2.6.0
CUDA available : True
NS backend     : Gram NS · kernel=cute_sm80 (SM80, DMuon internal)
```

Expected output (SYRK not installed or older GPU):

```
DMuon version : 0.2.0
PyTorch version: 2.6.0
CUDA available : True
NS backend     : Gram NS · kernel=cublas (SM80, universal fallback)
```

---

## Troubleshooting

**`ImportError: cannot import name 'fully_shard' from torch.distributed.fsdp`**
: PyTorch is older than 2.6. The `fully_shard` API in FSDP2 was stabilised
  in 2.6. Run `pip install --upgrade torch` and confirm `torch.__version__`
  reports at least `2.6.0`.

**`RuntimeError: NCCL error: unhandled system error`**
: Usually a process-group init issue before the first collective. Confirm
  `MASTER_ADDR` and `MASTER_PORT` are set, and that `dist.init_process_group`
  is called before `dmuon.dedicate_params`. See
  [Troubleshooting](../troubleshooting.md) for a checklist.

**SYRK build fails with `cutlass-dsl` not found**
: Confirm you installed the `[syrk]` extras: `pip install -e ".[syrk]"`.
  If the build still fails, the compiled fallback is used automatically —
  Newton-Schulz will still run correctly, just slightly slower.

**Fast-clip kernel not used (`fastpath=False` in clip stats)**
: The `dmuon._fast_clip_cuda` extension was not built (no `CUDA_HOME` at install
  time) or was disabled via `DMUON_FAST_CLIP=0`. Reinstall with a CUDA toolchain
  on `PATH`, or set `DMUON_FAST_CLIP_VERBOSE=1` to surface the import error.
  Gradient clipping stays correct via the Python path in the meantime.

---

## See Also

- [Quick Start](quickstart.md) — run your first distributed training
- [Core Concepts](concepts.md) — understand dedicated ownership before training
- [Troubleshooting](../troubleshooting.md) — runtime errors and common issues
