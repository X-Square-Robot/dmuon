"""Newton-Schulz orthogonalization with tiered hardware backends.

Backend selection (auto-detected at import time):
- **SM80+ & CuteDSL SYRK**: lower-triangle + mirror-write kernel (50% tile savings)
- **Fallback**: ``@torch.compile`` pure PyTorch

Two NS modes:
- ``newton_schulz(G)``: Standard NS on full matrix (non-TP)
- ``gram_newton_schulz(G_shard, tp_group)``: Gram NS with TP SYRK decomposition

Gram NS iteration logic is adapted from Dao-AILab/gram-newton-schulz, including
per-step coefficients, restart mechanism, and mixed-precision pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

from dmuon.optim.syrk_dispatch import (
    HAS_SYRK as _HAS_SYRK,
    get_ns_backend,
    syrk_or_cublas as _syrk_or_cublas,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coefficients — per-step, from Dao-AILab/gram-newton-schulz
# ---------------------------------------------------------------------------

# https://x.com/YouJiacheng/status/1905861218138804534
YOU_COEFFICIENTS = [
    [4.0848, -6.8946, 2.9270],
    [3.9505, -6.3029, 2.6377],
    [3.7418, -5.5913, 2.3037],
    [2.8769, -3.1427, 1.2046],
    [2.8366, -3.0525, 1.2012],
]

# https://arxiv.org/pdf/2505.16932 — with safety factor 1.05
_SAFETY_FACTOR = 1.05
_UNMODIFIED_POLAR_EXPRESS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
]
POLAR_EXPRESS_COEFFICIENTS = [
    (a / _SAFETY_FACTOR, b / _SAFETY_FACTOR**3, c / _SAFETY_FACTOR**5)
    for (a, b, c) in _UNMODIFIED_POLAR_EXPRESS
]

# Default coefficients and restart positions
DEFAULT_COEFFICIENTS = POLAR_EXPRESS_COEFFICIENTS
DEFAULT_RESTART_ITERATIONS = [2]  # optimal for POLAR_EXPRESS per autotune



# ---------------------------------------------------------------------------
# Compiled PyTorch fallback (uses per-step coefficients)
# ---------------------------------------------------------------------------
@torch.compile
def _compiled_newton_schulz(
    G: Tensor, coefficients: list[list[float]], eps: float = 1e-7
) -> Tensor:
    """Compiled standard NS with per-step coefficients."""
    X = G.float()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    X = X.half()
    for a, b, c in coefficients:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


# ---------------------------------------------------------------------------
# Standard NS — public API
# ---------------------------------------------------------------------------
def newton_schulz(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: Optional[list[list[float]]] = None,
    restart_iterations: Optional[list[int]] = None,
) -> Tensor:
    """Newton-Schulz orthogonalization with automatic backend selection.

    Uses Gram NS (Dao-AILab algorithm) by default for better precision
    (per-step coefficients, restart mechanism, mixed-precision pipeline).
    When CuteDSL SYRK is available, symmetric ops are accelerated.

    Args:
        G: Gradient matrix (m, n), any dtype.
        steps: Ignored (determined by len(coefficients)).
        eps: Normalization epsilon.
        coefficients: Per-step coefficients. Defaults to POLAR_EXPRESS_COEFFICIENTS.
        restart_iterations: Restart positions. Defaults to [2].

    Returns:
        Orthogonalized update.
    """
    return gram_newton_schulz_local(
        G, steps=steps, eps=eps,
        coefficients=coefficients,
        restart_iterations=restart_iterations,
    )


# ---------------------------------------------------------------------------
# Gram NS — TP-correct, adapted from Dao-AILab/gram-newton-schulz
# ---------------------------------------------------------------------------
def gram_newton_schulz(
    G_shard: Tensor,
    tp_group: dist.ProcessGroup,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: Optional[list[list[float]]] = None,
    restart_iterations: Optional[list[int]] = None,
) -> Tensor:
    """Gram Newton-Schulz with TP SYRK decomposition.

    Adapted from Dao-AILab/gram-newton-schulz. Iterates on the (m, m) Gram
    matrix instead of the full (m, n) matrix. Uses per-step coefficients and
    restart mechanism for numerical stability.

    For SYRK operations (``R = X @ X^T`` and ``Z = c*R^2 + b*R``), uses
    CuteDSL SYRK kernel when available (50% tile savings from symmetry).

    Args:
        G_shard: TP-sharded gradient (m, n/T) on this rank.
        tp_group: TP process group for all-reduce.
        steps: Ignored (determined by len(coefficients)).
        eps: Normalization epsilon.
        coefficients: Per-step coefficients. Defaults to POLAR_EXPRESS_COEFFICIENTS.
        restart_iterations: Iteration indices for restart. Defaults to [2].

    Returns:
        Orthogonalized update shard (m, n/T).
    """
    if coefficients is None:
        coefficients = DEFAULT_COEFFICIENTS
    if restart_iterations is None:
        restart_iterations = DEFAULT_RESTART_ITERATIONS

    original_dtype = G_shard.dtype

    # --- fp32 normalization (Dao-AILab precision strategy) ---
    X = G_shard.float()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    X = X.half()

    # --- Initial SYRK: R = X @ X^T ---
    m = X.shape[0]
    _use_syrk = _HAS_SYRK and X.is_cuda
    if _use_syrk:
        R = torch.empty(m, m, device=X.device, dtype=X.dtype)
        _syrk_or_cublas(X, R)
    else:
        R = X @ X.T

    # --- TP all-reduce: exact Gram = Σ G_i @ G_i^T ---
    dist.all_reduce(R, group=tp_group)

    # --- Gram NS iterations with restarts (Dao-AILab algorithm) ---
    I = torch.eye(m, device=X.device, dtype=X.dtype) if not _use_syrk else None
    Q: Optional[Tensor] = None
    Z = torch.empty_like(R) if _use_syrk else None
    Q_bufs = [torch.empty_like(R), torch.empty_like(R)] if _use_syrk else [None, None]
    q_idx = 0
    RZ_buf = torch.empty_like(R) if _use_syrk else None
    R_new = torch.empty_like(R) if _use_syrk else None

    for i, (a, b, c) in enumerate(coefficients):
        if i in restart_iterations and i != 0:
            X = Q @ X
            if _use_syrk:
                _syrk_or_cublas(X, R)
            else:
                R = X @ X.T
            dist.all_reduce(R, group=tp_group)
            Q = None

        if _use_syrk:
            _syrk_or_cublas(R, Z, C=R, alpha=c, beta=b)

            if Q is None:
                need_R_evolve = i < len(coefficients) - 1 and (i + 1) not in restart_iterations
                if not need_R_evolve:
                    _syrk_or_cublas(R, Q_bufs[q_idx], C=R, alpha=c, beta=b, diag_add=a)
                else:
                    Q_bufs[q_idx].copy_(Z)
                    Q_bufs[q_idx].diagonal().add_(a)
                Q = Q_bufs[q_idx]
            else:
                q_next = 1 - q_idx
                _syrk_or_cublas(Q, Q_bufs[q_next], B=Z, C=Q, beta=a)
                Q = Q_bufs[q_next]
                q_idx = q_next

            if i < len(coefficients) - 1 and (i + 1) not in restart_iterations:
                _syrk_or_cublas(R, RZ_buf, B=Z, C=R, beta=a)
                _syrk_or_cublas(Z, R_new, B=RZ_buf, C=RZ_buf, beta=a)
                R = R_new
        else:
            Z_t = torch.baddbmm(R, R, R, alpha=c, beta=b)
            if Q is None:
                Q = Z_t + a * I
            else:
                Q = torch.baddbmm(Q, Z_t, Q, beta=a)
            if i < len(coefficients) - 1 and (i + 1) not in restart_iterations:
                RZ_t = torch.baddbmm(R, R, Z_t, beta=a)
                R = torch.baddbmm(RZ_t, Z_t, RZ_t, beta=a)

    # --- Project back ---
    X = Q @ X

    if transposed:
        X = X.T
    return X.to(original_dtype)


def gram_newton_schulz_local(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: Optional[list[list[float]]] = None,
    restart_iterations: Optional[list[int]] = None,
) -> Tensor:
    """Gram NS without TP (single-rank). Uses Gram iteration for its precision
    benefits (restarts, per-step coefficients) even without TP sharding.

    Args:
        G: Full gradient matrix (m, n).
        steps: Ignored (determined by len(coefficients)).
        eps: Normalization epsilon.
        coefficients: Per-step coefficients.
        restart_iterations: Iteration indices for restart.

    Returns:
        Orthogonalized update (m, n).
    """
    if coefficients is None:
        coefficients = DEFAULT_COEFFICIENTS
    if restart_iterations is None:
        restart_iterations = DEFAULT_RESTART_ITERATIONS

    original_dtype = G.dtype

    X = G.float()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    X = X.half()

    # Initial SYRK: R = X @ X^T
    m = X.shape[0]
    _use_syrk = _HAS_SYRK and X.is_cuda
    if _use_syrk:
        R = torch.empty(m, m, device=X.device, dtype=X.dtype)
        _syrk_or_cublas(X, R)
    else:
        R = X @ X.T

    # Pre-allocate buffers for SYRK path (ping-pong for Q to avoid aliasing)
    I = torch.eye(m, device=X.device, dtype=X.dtype) if not _use_syrk else None
    Q: Optional[Tensor] = None
    Z = torch.empty_like(R) if _use_syrk else None
    Q_bufs = [torch.empty_like(R), torch.empty_like(R)] if _use_syrk else [None, None]
    q_idx = 0  # ping-pong index
    RZ_buf = torch.empty_like(R) if _use_syrk else None
    R_new = torch.empty_like(R) if _use_syrk else None

    for i, (a, b, c) in enumerate(coefficients):
        if i in restart_iterations and i != 0:
            X = Q @ X
            if _use_syrk:
                _syrk_or_cublas(X, R)
            else:
                R = X @ X.T
            Q = None

        if _use_syrk:
            # Z = c*R² + b*R (always needed for R evolve)
            _syrk_or_cublas(R, Z, C=R, alpha=c, beta=b)

            if Q is None:
                # First/restart: Q = Z + a*I (fuse diag_add if last step or before restart)
                need_R_evolve = i < len(coefficients) - 1 and (i + 1) not in restart_iterations
                if not need_R_evolve:
                    # No R evolve needed, fuse Z+aI into single SYRK
                    _syrk_or_cublas(R, Q_bufs[q_idx], C=R, alpha=c, beta=b, diag_add=a)
                else:
                    # Need Z for R evolve, compute Q = Z + a*I via diag add on Z copy
                    Q_bufs[q_idx].copy_(Z)
                    Q_bufs[q_idx].diagonal().add_(a)
                Q = Q_bufs[q_idx]
            else:
                # Q_new = Q@Z + a*Q (write to OTHER buffer to avoid alias)
                q_next = 1 - q_idx
                _syrk_or_cublas(Q, Q_bufs[q_next], B=Z, C=Q, beta=a)
                Q = Q_bufs[q_next]
                q_idx = q_next

            if i < len(coefficients) - 1 and (i + 1) not in restart_iterations:
                # RZ = R@Z + a*R (symmetric)
                _syrk_or_cublas(R, RZ_buf, B=Z, C=R, beta=a)
                # R = Z@RZ + a*RZ (symmetric)
                _syrk_or_cublas(Z, R_new, B=RZ_buf, C=RZ_buf, beta=a)
                R = R_new
        else:
            Z_t = torch.addmm(R, R, R, alpha=c, beta=b)
            if Q is None:
                Q = Z_t + a * I
            else:
                Q = torch.addmm(Q, Z_t, Q, beta=a)
            if i < len(coefficients) - 1 and (i + 1) not in restart_iterations:
                RZ_t = torch.addmm(R, R, Z_t, beta=a)
                R = torch.addmm(RZ_t, Z_t, RZ_t, beta=a)

    X = Q @ X

    if transposed:
        X = X.T
    return X.to(original_dtype)
