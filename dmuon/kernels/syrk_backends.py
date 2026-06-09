"""Unified SYRK backend dispatch — the single entry point for Newton-Schulz.

Sits above the concrete kernels:

    syrk_dispatch()
      ├── SyrkBackend.QUACK      → syrk_quack.syrk      (SM90+, soft dep)
      ├── SyrkBackend.CUTE_SM80  → syrk_sm80 kernel     (SM80/87)
      ├── SyrkBackend.CUBLAS     → torch.mm / torch.addmm  (universal)
      └── SyrkBackend.COMPILE    → torch.compile wrap   (reserved, post-MVP)

``kernel="auto"`` picks the best backend for the current device at
import time; users may override via the ``backend=`` argument or the
``DMUON_NS_KERNEL`` env var.

This module owns backend selection and keeps optional kernels behind soft
dependency checks.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional

import torch
from torch import Tensor

from dmuon.kernels import syrk_quack

_logger = logging.getLogger(__name__)


class SyrkBackend(str, Enum):
    """Level-2 kernel backend selector (see NewtonSchulz ``kernel=`` arg)."""

    AUTO = "auto"
    QUACK = "quack"
    CUTE_SM80 = "cute_sm80"
    CUBLAS = "cublas"
    COMPILE = "compile"  # reserved: torch.compile-wrapped cublas; not MVP


# ---------------------------------------------------------------------------
# Hardware detection (filled at import time)
# ---------------------------------------------------------------------------
_SM_VERSION: int = 0
if torch.cuda.is_available():
    _cap = torch.cuda.get_device_capability()
    _SM_VERSION = _cap[0] * 10 + _cap[1]

# Try importing the SM80 CuteDSL kernel.  Matches the probe pattern that
# ``syrk_dispatch`` (legacy) uses so we stay bit-compatible during the
# B-series transition.
_HAS_CUTE_SM80 = False
_syrk_sm80_fn = None
try:
    from dmuon.kernels.syrk_sm80 import syrk_sm80 as _syrk_sm80_fn

    _HAS_CUTE_SM80 = True
except ImportError:  # pragma: no cover — env-dependent
    pass


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
def detect_best_backend(sm_version: Optional[int] = None) -> SyrkBackend:
    """Pick the best available backend for ``sm_version``.

    Priority:
        * SM90+ with quack installed     → QUACK
        * SM80 / 87 with cute_sm80 built → CUTE_SM80
        * anything else                   → CUBLAS
    """
    if sm_version is None:
        sm_version = _SM_VERSION
    if sm_version >= 90:
        if syrk_quack.is_supported(sm_version):
            return SyrkBackend.QUACK
        return SyrkBackend.CUBLAS
    if 80 <= sm_version < 90:
        if _HAS_CUTE_SM80:
            return SyrkBackend.CUTE_SM80
        return SyrkBackend.CUBLAS
    return SyrkBackend.CUBLAS


def _quack_unsupport_hint() -> str:
    """Return a user-facing hint explaining why quack can't run right now."""
    if _SM_VERSION < 90:
        return (
            "quack requires SM90+ (H100/B200).  On an SM80 device (A100/A800) "
            "use kernel='cute_sm80' or kernel='auto'."
        )
    if not syrk_quack.HAS_QUACK:
        return "Install via `pip install dmuon[quack]`."
    if not syrk_quack.ADAPTER_READY:
        return (
            "The quack adapter has been disabled via the ADAPTER_READY "
            "circuit breaker (see dmuon.kernels.syrk_quack).  Flip it "
            "back to True to re-enable, or pass kernel='auto' for a "
            "working cuBLAS fallback."
        )
    return "Unknown quack unavailability reason."


def resolve_backend(user_choice: SyrkBackend) -> SyrkBackend:
    """Resolve ``SyrkBackend.AUTO`` to a concrete backend; pass others through.

    If the caller explicitly requested a backend that isn't available on
    this device (e.g. ``QUACK`` on SM80, or ``QUACK`` with quack not
    installed), we raise a clear error here so the failure doesn't creep
    into the SYRK hot path.
    """
    if user_choice == SyrkBackend.AUTO:
        return detect_best_backend()
    if user_choice == SyrkBackend.QUACK and not syrk_quack.is_supported(_SM_VERSION):
        hint = _quack_unsupport_hint()
        raise RuntimeError(
            f"kernel='quack' requested but not usable: SM={_SM_VERSION}, "
            f"HAS_QUACK={syrk_quack.HAS_QUACK}, "
            f"ADAPTER_READY={syrk_quack.ADAPTER_READY}.\n{hint}"
        )
    if user_choice == SyrkBackend.CUTE_SM80 and not _HAS_CUTE_SM80:
        raise RuntimeError(
            "kernel='cute_sm80' requested but dmuon.kernels.syrk_sm80 "
            "failed to import.  Check your CuteDSL installation."
        )
    if user_choice == SyrkBackend.COMPILE:
        raise NotImplementedError(
            "kernel='compile' is reserved for post-MVP; use 'cublas' for now."
        )
    return user_choice


def get_backend_status() -> dict:
    """Introspection: all the facts a user might want to see about NS kernels.

    Returns a plain dict so it JSON-serialises cleanly for logging.
    """
    return {
        "sm_version": _SM_VERSION,
        "auto_choice": detect_best_backend().value,
        "quack_available": syrk_quack.HAS_QUACK,
        "cute_sm80_available": _HAS_CUTE_SM80,
        "cublas_always_available": True,
    }


# ---------------------------------------------------------------------------
# Dispatch — the single SYRK call-point
# ---------------------------------------------------------------------------
def _cublas_syrk(
    A: Tensor,
    D: Tensor,
    *,
    B: Optional[Tensor] = None,
    C: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    diag_add: float = 0.0,
) -> None:
    """``D = alpha · A @ Bᵀ + beta · C + diag_add · I`` via ``torch.mm`` /
    ``torch.addmm``.  ``B is None`` means ``B = A`` (true SYRK semantics).

    ``torch.mm`` ignores alpha, so the no-C path applies alpha in-place
    after the matmul — the previous ``syrk_or_cublas`` silently dropped
    alpha here, which was safe because callers only ever pass alpha=1.0
    without C.  The new contract is stricter.
    """
    BT = A.T if B is None else B.T
    if C is not None:
        torch.addmm(C, A, BT, alpha=alpha, beta=beta, out=D)
    else:
        torch.mm(A, BT, out=D)
        if alpha != 1.0:
            D.mul_(alpha)
    if diag_add != 0.0:
        D.diagonal().add_(diag_add)


def syrk_dispatch(
    A: Tensor,
    D: Tensor,
    *,
    B: Optional[Tensor] = None,
    C: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    diag_add: float = 0.0,
    backend: SyrkBackend = SyrkBackend.AUTO,
    # cute_sm80-specific tile knobs (forwarded by the autotune layer)
    tile_m: Optional[int] = None,
    tile_k: Optional[int] = None,
    num_stages: Optional[int] = None,
    _symmetric: bool = False,
) -> None:
    """Dispatch ``D = alpha · A @ Bᵀ + beta · C + diag_add · I`` to the
    configured backend.

    The autotune layer (``dmuon.optim.syrk_dispatch``) is responsible for
    choosing the tile config for CuteDSL backends; the raw dispatch
    function forwards whatever it receives.  When ``backend=AUTO`` we
    resolve on every call — the detection result itself is cached at
    module import, so this is effectively free.
    """
    resolved = resolve_backend(backend)

    if resolved == SyrkBackend.CUBLAS:
        _cublas_syrk(A, D, B=B, C=C, alpha=alpha, beta=beta, diag_add=diag_add)
        return

    if resolved == SyrkBackend.CUTE_SM80:
        assert _syrk_sm80_fn is not None  # gated by _HAS_CUTE_SM80
        _syrk_sm80_fn(
            A,
            D,
            B=B,
            C=C,
            alpha=alpha,
            beta=beta,
            diag_add=diag_add,
            tile_m=tile_m,
            tile_k=tile_k,
            num_stages=num_stages,
            _symmetric=_symmetric,
        )
        return

    if resolved == SyrkBackend.QUACK:
        syrk_quack.syrk(
            A, D, B=B, C=C, alpha=alpha, beta=beta, diag_add=diag_add
        )
        return

    raise RuntimeError(f"Unreachable backend: {resolved!r}")


# ---------------------------------------------------------------------------
# Env-var override for the auto choice (consumed by NewtonSchulz in B4)
# ---------------------------------------------------------------------------
def resolve_env_kernel() -> Optional[SyrkBackend]:
    """If ``DMUON_NS_KERNEL`` is set, return the parsed backend (or raise
    ``ValueError`` for an unknown value).  Returns ``None`` when unset."""
    raw = os.environ.get("DMUON_NS_KERNEL")
    if raw is None:
        return None
    try:
        return SyrkBackend(raw)
    except ValueError as exc:  # pragma: no cover — user error path
        valid = ", ".join(b.value for b in SyrkBackend)
        raise ValueError(
            f"DMUON_NS_KERNEL={raw!r} is not a valid backend; "
            f"choose one of: {valid}"
        ) from exc
