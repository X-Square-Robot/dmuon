"""Select the owner-local arithmetic used by the optimizer."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from dmuon._core.device import get_device_module

if TYPE_CHECKING:
    import torch


_COMPUTE_BACKENDS = ("auto", "torch", "tilelang")


def normalize_compute_backend(compute_backend: str) -> str:
    """Validate and return an owner-local compute backend name."""
    if compute_backend not in _COMPUTE_BACKENDS:
        raise ValueError(
            f"compute_backend must be one of {_COMPUTE_BACKENDS}, got {compute_backend!r}"
        )
    return compute_backend


@lru_cache(maxsize=None)
def _is_950dt(index: int) -> bool:
    name = get_device_module("npu").get_device_name(index)
    return name.lower().replace(" ", "").startswith("ascend950dt")


@lru_cache(maxsize=1)
def _tilelang_available() -> bool:
    try:
        import_module("tilelang")
        import_module("tilelang.ascend.language")
    except ImportError:
        return False
    return True


def use_tilelang(device: torch.device, compute_backend: str = "auto") -> bool:
    """Return whether the TileLang compute backend should handle a device.

    Args:
        device: Device that owns the optimizer tensors.
        compute_backend: ``"auto"``, ``"torch"``, or ``"tilelang"``.

    Returns:
        True only for an eligible Ascend 950DT with the required TileLang modules.
        Automatic selection falls back to Torch when TileLang is unavailable.

    Raises:
        ValueError: If the backend name is invalid or TileLang is requested on
            unsupported Ascend hardware.
        RuntimeError: If TileLang is explicitly requested but not installed.
    """
    choice = normalize_compute_backend(compute_backend)
    if device.type != "npu" or choice == "torch":
        return False
    index = device.index if device.index is not None else get_device_module("npu").current_device()
    supported = _is_950dt(index)
    if choice == "tilelang" and not supported:
        raise ValueError("compute_backend='tilelang' requires an Ascend 950DT")
    if not supported:
        return False
    available = _tilelang_available()
    if choice == "tilelang" and not available:
        raise RuntimeError(
            "compute_backend='tilelang' requires TileLang with Ascend support. "
            "Build the unmodified TileLang v0.1.15 sources with USE_ASCEND=ON; "
            "see docs/getting-started/installation.md."
        )
    return available


def resolve_compute_backend(device: torch.device, compute_backend: str = "auto") -> ModuleType:
    """Resolve the arithmetic module for an owner-local device.

    Args:
        device: Device that owns the optimizer tensors.
        compute_backend: ``"auto"``, ``"torch"``, or ``"tilelang"``.

    Returns:
        The TileLang compute module when eligible, otherwise the Torch module.
    """
    if use_tilelang(device, compute_backend):
        from .tilelang import compute
    else:
        from . import torch as compute
    return compute


def supports_tilelang_clip(
    device: torch.device,
    grads: Iterable[torch.Tensor],
    compute_backend: str = "auto",
) -> bool:
    """Check the whole local bucket before launching a norm or an in-place scale.

    Args:
        device: Common device required for every gradient.
        grads: Local gradients whose layout and dtype determine eligibility.
        compute_backend: ``"auto"``, ``"torch"``, or ``"tilelang"``.

    Returns:
        True when every gradient can use the TileLang clipping implementation.
    """
    if not use_tilelang(device, compute_backend):
        return False
    from torch import bfloat16, float16, float32, strided
    from torch_npu import get_npu_format

    return all(
        grad.device == device
        and grad.dtype in (float32, float16, bfloat16)
        and grad.layout == strided
        and grad.numel() > 0
        and get_npu_format(grad) in (0, 2, 30)
        for grad in grads
    )
