"""Reference Newton-Schulz implementations with configurable compute dtype.

Mirrors ``dmuon/optim/newton_schulz.py`` logic but parameterizes the compute
dtype so we can measure the algorithmic floor (fp64) and compare to the
production hardcoded fp16 path.

The production functions in dmuon always do ``X = X.half()`` — these reference
functions leave dtype as a parameter.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from dmuon.optim.newton_schulz import (
    DEFAULT_COEFFICIENTS,
    DEFAULT_RESTART_ITERATIONS,
)


def direct_ns_ref(
    G: Tensor,
    *,
    compute_dtype: torch.dtype,
    coefficients: Optional[list] = None,
    eps: float = 1e-7,
) -> Tensor:
    """Direct-space NS with configurable compute dtype.

    Faithful to ``direct_newton_schulz`` (always uses cuBLAS mm, no SYRK).
    """
    if coefficients is None:
        coefficients = DEFAULT_COEFFICIENTS
    orig = G.dtype
    X = G.to(torch.float32)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    X = X.to(compute_dtype)
    for a, b, c in coefficients:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(orig)


def gram_ns_ref(
    G: Tensor,
    *,
    compute_dtype: torch.dtype,
    coefficients: Optional[list] = None,
    restart_iterations: Optional[list[int]] = None,
    eps: float = 1e-7,
) -> Tensor:
    """Gram-space NS with configurable compute dtype.

    Faithful to the non-SYRK branch of ``gram_newton_schulz_local``
    (so results are independent of the SYRK kernel). Runs in cuBLAS
    path with arbitrary compute_dtype.
    """
    if coefficients is None:
        coefficients = DEFAULT_COEFFICIENTS
    if restart_iterations is None:
        restart_iterations = DEFAULT_RESTART_ITERATIONS
    orig = G.dtype

    X = G.to(torch.float32)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    X = X.to(compute_dtype)

    m = X.shape[0]
    R = X @ X.T
    I = torch.eye(m, device=X.device, dtype=X.dtype)
    Q: Optional[Tensor] = None

    for i, (a, b, c) in enumerate(coefficients):
        if i in restart_iterations and i != 0:
            X = Q @ X
            R = X @ X.T
            Q = None

        Z = torch.addmm(R, R, R, alpha=c, beta=b)  # Z = b*R + c*R@R
        if Q is None:
            Q = Z + a * I
        else:
            Q = torch.addmm(Q, Z, Q, beta=a)  # Q = Z@Q + a*Q

        last = i == len(coefficients) - 1
        will_restart = (i + 1) in restart_iterations
        if not last and not will_restart:
            RZ = torch.addmm(R, R, Z, beta=a)  # RZ = R@Z + a*R
            R = torch.addmm(RZ, Z, RZ, beta=a)  # R = Z@RZ + a*RZ

    X = Q @ X
    if transposed:
        X = X.T
    return X.to(orig)
