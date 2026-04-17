"""Algebraic accuracy metrics for the polar factor of a matrix.

Adapted from Noah Amsel / Tri Dao's gram-NS diagnostic reference. All metrics
are pure algebraic identities — they do NOT require an SVD ground truth, and
can be computed from ``(A, X_est)`` alone.

For a perfect polar factor ``X* = U @ V^T`` of ``A = U Σ V^T``:

    * ``X* X*^T = I``                (orthogonality of rows)
    * ``X* H = A``  where H = sym(X*^T A)   (polar reconstruction)
    * ``H = V Σ V^T``                (H is PSD)
    * ``<A, X*> = ‖A‖_*``            (nuclear norm Lagrange dual)
    * ``σ_max(X*) = 1``              (spectral norm bound)

Running a truncated NS gives X_est; deviation from the identities is the
error. All metrics return scalars; float('inf') on non-finite input.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def polar_accuracy(A: Tensor, X_est: Tensor) -> dict[str, float]:
    """Five algebraic polar-factor error metrics.

    Args:
        A: original matrix (m, n), typically fp64.
        X_est: estimated polar factor (m, n), any dtype; cast to ``A.dtype``.

    Returns:
        Dict with keys ``orth_error``, ``residual_error``, ``psd_error``,
        ``dual_obj``, ``bound_violation``. All floats; inf if X_est has NaN/Inf.
    """
    out = dict(
        orth_error=float("inf"),
        residual_error=float("inf"),
        psd_error=float("inf"),
        dual_obj=float("inf"),
        bound_violation=float("inf"),
    )
    if not X_est.isfinite().all():
        return out

    X = X_est.to(A.dtype)
    H = X.mT @ A
    H = (H + H.mT) / 2
    Heigs = torch.linalg.eigvalsh(H)
    nuc = torch.linalg.matrix_norm(A, ord="nuc")
    sigmax = torch.linalg.matrix_norm(X, ord=2)
    m = X.shape[-2]
    I = torch.eye(m, device=X.device, dtype=X.dtype)

    out["orth_error"] = ((X @ X.mT - I).norm() / I.norm()).item()
    out["residual_error"] = ((X @ H - A).norm() / A.norm()).item()
    pos = Heigs[Heigs > 0].norm()
    neg = Heigs[Heigs < 0].norm()
    out["psd_error"] = (neg / pos).item() if pos.item() > 0 else float("inf")
    out["dual_obj"] = ((nuc - torch.inner(A.flatten(), X.flatten())) / nuc).item()
    out["bound_violation"] = max((sigmax - 1).item(), 0.0)
    return out


def direct_svd_error(X_ref: Tensor, X_est: Tensor) -> float:
    """Relative Frobenius error vs reference polar factor.

    When ``X_ref`` is the ground-truth polar (from fp64 SVD), this is the
    most interpretable single number: "how far from truth, relative to truth".
    """
    if not X_est.isfinite().all():
        return float("inf")
    X = X_est.to(X_ref.dtype)
    denom = X_ref.norm().item()
    if denom == 0:
        return float("inf")
    return ((X - X_ref).norm() / X_ref.norm()).item()


def diagonalizability_error(
    M: Tensor,
    U0: Tensor,
    V0: Optional[Tensor] = None,
    symmetric: bool = False,
) -> float:
    """Off-diagonal Frobenius ratio after rotating into the starting basis.

    For an exact NS iterate, ``M_t = U_0 f(Σ_0) V_0^T`` so ``U_0^T M_t V_0``
    is diagonal. Non-zero off-diagonal mass measures eigenvector drift.

    Args:
        M: iterate to analyze, any dtype (cast to fp64 internally).
        U0: starting left singular vectors (m, m) in fp64.
        V0: starting right singular vectors (n, n) in fp64. Ignored when
            ``symmetric=True`` (R_t, Q_t are symmetric → diagonalize with U0).
        symmetric: if True, use ``U0^T M U0`` (for R_t / Q_t).

    Returns:
        ``‖off-diag(R)‖_F / ‖R‖_F`` where R is the rotated matrix.
        0.0 = no drift; large = U has moved.
    """
    M64 = M.to(torch.float64)
    if not M64.isfinite().all():
        return float("inf")
    if symmetric:
        assert V0 is None, "symmetric mode uses U0 on both sides"
        M64 = (M64 + M64.mT) / 2
        R = U0.mT @ M64 @ U0
    else:
        assert V0 is not None, "non-symmetric mode needs V0"
        R = U0.mT @ M64 @ V0
    denom = torch.linalg.matrix_norm(R, ord="fro").item()
    if denom == 0:
        return 0.0
    off = R.clone()
    off.diagonal().zero_()
    return (torch.linalg.matrix_norm(off, ord="fro") / denom).item()


def svd_polar(A: Tensor) -> Tensor:
    """Ground-truth polar factor via fp64 SVD.

    For A = U Σ V^T, polar factor is U V^T. Assumes A is m×n with m ≤ n
    (short-fat); for m > n the result is still the correct polar factor
    (closest orthogonal partial isometry).
    """
    A64 = A.to(torch.float64)
    U, S, Vh = torch.linalg.svd(A64, full_matrices=False)
    return (U @ Vh).to(A.dtype)
