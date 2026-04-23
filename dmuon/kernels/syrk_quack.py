"""Quack (Tri Dao) SYRK wrapper for SM90+ — soft dependency.

``quack-kernels`` is installed via ``pip install dmuon[quack]`` and is
only exercised on SM90+ hardware.  The adapter maps DMuon's SYRK
contract (``D = α · A @ Bᵀ + β · C + diag_add · I``) to quack's
``gemm_symmetric`` (``out = α · A @ B + β · C``) — DMuon passes ``B.T``
as quack's ``B`` and adds the diagonal bias as a post-step.

The mapping is validated on B300 (SM103) in B7/B8; see
``docs/internal/benchmarks/quack_smoke_b300.md`` for the correctness
matrix and perf crossover.

Design (Phase B-H):
  * Uses the 2D high-level interface ``quack.gemm_interface.gemm_symmetric``
    which auto-tunes tile / cluster / swizzle internally.  DMuon's
    per-backend autotune layer therefore stores no tile keys for
    ``backend="quack"``.
  * Passes ``A.T`` as a stride view (not ``.contiguous()``); B7 O1
    confirmed CUTLASS-DSL handles the column-major input natively.
  * Output tensor ``D`` is always fully symmetric (B7 O3: quack writes
    both triangles), so DMuon's ``_symmetric`` flag is ignored here.
  * ``alpha=0``/``beta=0`` short-circuit correctly (B7 O4).
"""

from __future__ import annotations

import logging
from typing import Optional

from torch import Tensor

_logger = logging.getLogger(__name__)

try:  # soft dependency
    import quack  # type: ignore  # noqa: F401
    from quack.gemm_interface import gemm_symmetric as _quack_gemm_symmetric

    HAS_QUACK = True
except Exception as _quack_import_exc:  # pragma: no cover — env-dependent
    # Catch broadly: quack registers torch operators at import time and
    # incompatible PyTorch builds surface as ``RuntimeError`` (not
    # ``ImportError``).  Any failure here is treated the same as
    # "package not usable" — the dispatch layer falls back cleanly.
    HAS_QUACK = False
    _quack_gemm_symmetric = None
    _logger.debug("quack soft-dep unavailable: %r", _quack_import_exc)


#: Phase-gate flag.  Flipped to ``True`` in B8 after the quack→DMuon
#: adapter is implemented and validated on SM100 (B300).  When
#: ``False``, :func:`is_supported` always returns ``False`` so the
#: dispatch layer never routes into :func:`syrk` — useful as a
#: circuit-breaker if the adapter needs to be disabled quickly (e.g.
#: quack-kernels breaking change).
ADAPTER_READY = True


def is_supported(sm_version: int) -> bool:
    """Whether the quack SYRK backend is usable on this GPU.

    Returns True only if (1) :data:`ADAPTER_READY` is set, (2) ``quack``
    is importable, and (3) the device compute capability is ≥ SM90.
    SM80 devices always return False — quack's SYRK path is SM90+-only
    by design.
    """
    return ADAPTER_READY and HAS_QUACK and sm_version >= 90


def syrk(
    A: Tensor,
    D: Tensor,
    *,
    B: Optional[Tensor] = None,
    C: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    diag_add: float = 0.0,
) -> None:
    """DMuon SYRK contract routed to quack's symmetric GEMM.

    Computes ``D = alpha · A @ Bᵀ + beta · C + diag_add · I`` in-place
    on ``D``.  When ``B is None`` we take ``B = A`` (true SYRK
    semantics, ``A @ Aᵀ``).

    Implementation is a thin wrapper around
    ``quack.gemm_interface.gemm_symmetric``.  Three adapter responsibilities:

    1. **Transpose**: quack computes ``A @ B`` (no implicit transpose),
       DMuon's contract is ``A @ Bᵀ`` — we pass ``A.T`` (or ``B.T``) as
       quack's ``B``.  A stride view is fine; no ``.contiguous()`` copy.
    2. **Diag bias**: quack has no ``diag_add`` knob, so we apply it
       post-call as ``D.diagonal().add_(diag_add)``.  Skipped when zero.
    3. **Caller guarantees**: ``A @ Bᵀ`` must be symmetric (quack's
       symmetric-GEMM precondition).  For ``B is None`` this is
       trivially true; for other call sites DMuon already guarantees
       this via ``_core/newton_schulz.py`` (Gram-space intermediates).
    """
    if not HAS_QUACK or _quack_gemm_symmetric is None:
        raise RuntimeError(
            "dmuon.kernels.syrk_quack.syrk called but quack is not "
            "installed; install via `pip install dmuon[quack]`."
        )

    # Map DMuon ``A @ Bᵀ`` to quack ``A @ B_quack`` via transpose.  Stride
    # view is fine per B7 O1 — avoids a materialised copy.
    B_quack = (A if B is None else B).T

    _quack_gemm_symmetric(A, B_quack, C=C, out=D, alpha=alpha, beta=beta)

    if diag_add != 0.0:
        D.diagonal().add_(diag_add)
