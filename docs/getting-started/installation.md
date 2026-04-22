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
    git clone https://github.com/StarrickLiu/dmuon
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
    git clone https://github.com/StarrickLiu/dmuon
    cd dmuon
    pip install -e ".[dev]"
    ```

    Installs the library in editable mode plus test dependencies
    (`pytest`, `pytest-dist`, and related tooling). Run the unit
    test suite to confirm everything works:

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
NS backend     : syrk_sm80
```

Expected output (SYRK not installed or older GPU):

```
DMuon version : 0.2.0
PyTorch version: 2.6.0
CUDA available : True
NS backend     : compiled
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

---

## See Also

- [Quick Start](quickstart.md) — run your first distributed training
- [Core Concepts](concepts.md) — understand dedicated ownership before training
- [Troubleshooting](../troubleshooting.md) — runtime errors and common issues
