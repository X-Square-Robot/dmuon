"""Internal utilities borrowed from FSDP2.

Verbatim copies from ``torch.distributed.fsdp._fully_shard._fsdp_param`` —
kept here because they are private APIs not re-exported by PyTorch.

Each function is documented with its upstream source location so the
provenance is explicit and we can keep it in sync with PyTorch upgrades.
"""

import torch
import torch.nn as nn


def alloc_storage(tensor: torch.Tensor) -> None:
    """Resize tensor's storage to ``numel * itemsize`` if not already.

    Mirrors ``torch/distributed/fsdp/_fully_shard/_fsdp_param.py:alloc_storage``.
    Used to re-materialize a tensor whose storage was freed via
    :func:`free_storage`. Idempotent.
    """
    size = tensor.numel() * tensor.itemsize
    if (storage := tensor.untyped_storage()).size() != size:
        storage.resize_(size)


def free_storage(tensor: torch.Tensor) -> None:
    """Resize tensor's storage to 0 to free GPU memory without destroying the
    Python ``Tensor`` / ``nn.Parameter`` object.

    Mirrors ``torch/distributed/fsdp/_fully_shard/_fsdp_param.py:free_storage``.
    Idempotent.
    """
    if (storage := tensor.untyped_storage()).size() != 0:
        storage.resize_(0)


def unsafe_setattr_param(module: nn.Module, param_name: str, param: nn.Parameter) -> None:
    """Fast-path setattr that bypasses ``nn.Module.__setattr__``.

    Mirrors ``torch/distributed/fsdp/_fully_shard/_fsdp_param.py:unsafe_setattr_param``.

    ``nn.Module.__setattr__`` has non-trivial CPU overhead (``remove_from`` +
    ``register_parameter``) that we can skip when the module has not overridden
    ``__setattr__``. For most layers (``nn.Linear``, ``nn.LayerNorm``, ...)
    this is safe and saves several microseconds per call.

    Falls back to regular :func:`setattr` when the module overrides
    ``__setattr__`` (e.g., some quantization wrappers).
    """
    if getattr(module.__setattr__, "__func__", None) is nn.Module.__setattr__:
        module._parameters[param_name] = param
    else:
        setattr(module, param_name, param)


def set_requires_grad_if_needed(src_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> None:
    """Copy ``requires_grad`` from src to dst only if it differs.

    Mirrors ``torch/distributed/fsdp/_fully_shard/_fsdp_param.py:set_requires_grad_if_needed``.
    Avoids the Python↔C++ context switch of ``requires_grad_`` in the common
    case where the flag already matches.
    """
    if src_tensor.requires_grad != dst_tensor.requires_grad:
        dst_tensor.requires_grad_(src_tensor.requires_grad)
