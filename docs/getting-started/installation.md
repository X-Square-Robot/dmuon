# Installation

## Requirements

- Python >= 3.10
- PyTorch >= 2.4.0
- CUDA-capable GPUs (multi-GPU for distributed training)

## Install from Source

```bash
git clone https://github.com/StarrickLiu/dmuon && cd dmuon
pip install -e .
```

## Optional: SYRK Kernel Acceleration

DMuon includes a custom [CuteDSL](https://github.com/NVIDIA/cutlass) SYRK kernel that exploits Gram matrix symmetry for ~1.5x speedup on Newton-Schulz iterations. This requires SM80+ GPUs (A100, A800, H100, etc.) and additional dependencies:

```bash
pip install -e ".[syrk]"
```

This installs:

- `nvidia-cutlass-dsl >= 4.4.2`
- `apache-tvm-ffi`
- `torch-c-dlpack-ext`

!!! note
    The SYRK kernel is optional. Without it, DMuon falls back to `@torch.compile`'d pure PyTorch, which is fully functional but slightly slower for the Gram NS iterations.

## Verify Installation

```python
import dmuon
print(f"DMuon {dmuon.__version__}")
print(f"NS backend: {dmuon.get_ns_backend()}")
```

Expected output:
```
DMuon 0.2.0
NS backend: syrk_sm80    # or "compiled" if SYRK deps not installed
```

## Next

[Quick Start](quickstart.md) — Run your first distributed training with DMuon.
