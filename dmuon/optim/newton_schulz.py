"""Newton-Schulz orthogonalization with tiered hardware backends.

Backend selection (auto-detected at import time):
- **SM80+ & CuteDSL SYRK**: lower-triangle + mirror-write kernel (50% tile savings)
- **Fallback**: ``@torch.compile`` pure PyTorch

Two NS modes:
- :func:`newton_schulz`: Gram-space NS on a full matrix (default public entry).
- :func:`direct_newton_schulz`: classic parameter-space NS.

TP support lives entirely in the runtime layer (``dmuon._backends.fsdp2``):
for TP-sharded parameters the runtime does a TP gather so the TP owner
sees the full matrix, then calls one of the functions above — the
NS algorithms themselves are TP-agnostic.  See ``docs/guides/tp-support.md``.

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
    syrk_or_cublas as _syrk_or_cublas,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coefficients — per-step, from Dao-AILab/gram-newton-schulz
# ---------------------------------------------------------------------------

#: 5-step Newton-Schulz polynomial coefficients from @YouJiacheng
#: (https://x.com/YouJiacheng/status/1905861218138804534). Tuned for faster
#: convergence than the original Muon paper's coefficients at 5 iterations.
#: Each inner list is ``[a, b, c]`` for one NS step's polynomial
#: ``a*X + b*X*X^T*X + c*(X*X^T)^2*X``.
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

#: 5-step Newton-Schulz coefficients from the Polar Express paper
#: (https://arxiv.org/abs/2505.16932), rescaled by a 1.05 safety factor
#: for numerical stability. **This is DMuon's default backend** and is
#: what ``NewtonSchulz("gram")`` / ``NewtonSchulz("direct")`` use when
#: no explicit ``coefficients`` argument is provided.
POLAR_EXPRESS_COEFFICIENTS = [
    (a / _SAFETY_FACTOR, b / _SAFETY_FACTOR**3, c / _SAFETY_FACTOR**5)
    for (a, b, c) in _UNMODIFIED_POLAR_EXPRESS
]

# Default coefficients and restart positions
DEFAULT_COEFFICIENTS = POLAR_EXPRESS_COEFFICIENTS
DEFAULT_RESTART_ITERATIONS = [2]  # optimal for POLAR_EXPRESS per autotune


# ---------------------------------------------------------------------------
# NewtonSchulz — configurable NS backend object
# ---------------------------------------------------------------------------
class NewtonSchulz:
    """Configurable Newton-Schulz backend.

    Encapsulates the algorithm variant, coefficients, and SYRK kernel
    backend so they can be passed as a single object to :class:`~dmuon.Muon`.

    Args:
        backend: ``"gram"`` (default) for Gram-space NS with SYRK
            acceleration and restarts, or ``"direct"`` for classic
            parameter-space NS (Muon/Moonlight formulation).
        kernel: SYRK kernel backend to use inside Gram NS:

            * ``"auto"`` (default) — pick the best available on this GPU.
              SM80/87 → ``cute_sm80``, SM90+ with quack installed →
              ``quack``, otherwise ``cublas``.
            * ``"quack"`` — Tri Dao quack SYRK (SM90+, soft dep). Raises
              at construction if unavailable.
            * ``"cute_sm80"`` — DMuon-internal CuteDSL kernel (SM80/87).
            * ``"cublas"`` — universal fallback; bit-exact across runs.

            Env var override: ``DMUON_NS_KERNEL`` takes precedence only
            when this argument is left at ``"auto"``.
        coefficients: Per-step ``(a, b, c)`` coefficients.  ``None``
            uses :data:`POLAR_EXPRESS_COEFFICIENTS`.
        restart_iterations: Restart positions for Gram-space NS.
            ``None`` uses ``[2]``.  Ignored when *backend* is
            ``"direct"``.
        deterministic: Back-compat alias for ``kernel="cublas"``.  When
            ``True`` and ``kernel`` is still ``"auto"``, the kernel is
            forced to ``cublas`` (bit-exact reproducibility).  Explicit
            ``kernel=`` wins over ``deterministic`` — if both are given
            and they disagree, a warning is emitted and ``kernel`` wins.

    Example::

        import dmuon

        # Default (Gram-space, auto-selected kernel)
        ns = dmuon.NewtonSchulz()

        # Force cuBLAS (reproducible across runs)
        ns = dmuon.NewtonSchulz(kernel="cublas")
        ns = dmuon.NewtonSchulz(deterministic=True)  # equivalent

        # SM90+ explicit quack
        ns = dmuon.NewtonSchulz(kernel="quack")

        optimizer = dmuon.Muon(model, lr=0.02, ns_backend=ns)
    """

    def __init__(
        self,
        backend: str = "gram",
        kernel: str = "auto",
        coefficients: Optional[list[list[float]]] = None,
        restart_iterations: Optional[list[int]] = None,
        deterministic: bool = False,
    ):
        if backend not in ("gram", "direct"):
            raise ValueError(f"backend must be 'gram' or 'direct', got '{backend}'")
        # Lazy import to avoid a cycle with syrk_backends on cold start.
        from dmuon.kernels.syrk_backends import (
            SyrkBackend,
            resolve_backend,
            resolve_env_kernel,
        )

        # Resolve kernel choice in priority order:
        #   1. explicit kernel= kwarg (non-auto) ─ highest
        #   2. DMUON_NS_KERNEL env var ─ consulted only when kernel='auto'
        #   3. deterministic=True ─ legacy alias; maps to 'cublas' when
        #      kernel is still 'auto' after env resolution
        try:
            chosen = SyrkBackend(kernel)
        except ValueError:
            valid = ", ".join(b.value for b in SyrkBackend)
            raise ValueError(
                f"kernel={kernel!r} is not a valid SYRK backend; "
                f"choose one of: {valid}"
            )
        if chosen == SyrkBackend.AUTO:
            env_choice = resolve_env_kernel()
            if env_choice is not None:
                chosen = env_choice
            elif deterministic:
                chosen = SyrkBackend.CUBLAS
        elif deterministic and chosen != SyrkBackend.CUBLAS:
            logger.warning(
                "NewtonSchulz: kernel=%r conflicts with deterministic=True; "
                "honouring explicit kernel.  Pass kernel='cublas' for bit-exact.",
                chosen.value,
            )

        # Validate availability early so the failure is at construction
        # time, not buried inside the first SYRK call.
        chosen = resolve_backend(chosen)

        self.backend = backend
        self.kernel = chosen
        self.coefficients = coefficients
        self.restart_iterations = restart_iterations
        # deterministic becomes a derived flag: True iff the resolved
        # kernel is cublas.  Existing call sites that check
        # ``self.deterministic`` keep working unchanged.
        self.deterministic = chosen == SyrkBackend.CUBLAS

    def local(self, G: Tensor, steps: int) -> Tensor:
        """Run NS on a full (un-sharded) matrix.

        The runtime guarantees the matrix handed in here is the full
        logical gradient: for pure-DP params the owner already holds
        the full tensor; for TP-sharded params the TP gather
        step (``dmuon._backends.fsdp2.group.tp_gather_grads``) has
        reassembled the full matrix on the TP owner before this call.
        """
        if self.backend == "gram":
            return gram_newton_schulz(
                G, steps=steps,
                coefficients=self.coefficients,
                restart_iterations=self.restart_iterations,
                deterministic=self.deterministic,
            )
        return direct_newton_schulz(
            G, steps=steps, coefficients=self.coefficients,
        )

    def __repr__(self) -> str:
        coeff = "default" if self.coefficients is None else f"{len(self.coefficients)}-step custom"
        return (
            f"NewtonSchulz(backend={self.backend!r}, "
            f"kernel={self.kernel.value!r}, coefficients={coeff})"
        )


# ---------------------------------------------------------------------------
# Direct-space NS — standard Newton-Schulz in parameter space
# ---------------------------------------------------------------------------
def direct_newton_schulz(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: Optional[list[list[float]]] = None,
) -> Tensor:
    """Standard Newton-Schulz orthogonalization in direct (parameter) space.

    Iterates on the full (m, n) matrix:
    ``X_{k+1} = a_k X + b_k (X X^T) X + c_k (X X^T)^2 X``

    This is the classic formulation used by Muon/Moonlight. Compared to
    :func:`gram_newton_schulz` (Gram-space), direct NS is simpler but:

    - Does not benefit from SYRK symmetry acceleration
    - Does not support the restart mechanism
    - Intermediate ops are (m, n) instead of (m, m)

    Use this when you want the standard algorithm without Gram-space
    optimizations, e.g., for baseline comparison or small matrices where
    SYRK overhead is not justified.

    Args:
        G: Gradient matrix (m, n), any dtype.
        steps: Number of NS iterations (used only if coefficients is None).
        eps: Normalization epsilon.
        coefficients: Per-step ``(a, b, c)`` coefficients. Length determines
            number of iterations. Defaults to :data:`DEFAULT_COEFFICIENTS`.

    Returns:
        Orthogonalized update, same shape as G, in original dtype.
    """
    if coefficients is None:
        coefficients = DEFAULT_COEFFICIENTS

    original_dtype = G.dtype
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
    return X.to(original_dtype)


@torch.compile
def _compiled_direct_newton_schulz(
    G: Tensor, coefficients: list[list[float]], eps: float = 1e-7
) -> Tensor:
    """torch.compile'd variant of :func:`direct_newton_schulz`."""
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
# Default NS — public API (routes to Gram-space by default)
# ---------------------------------------------------------------------------
def newton_schulz(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: Optional[list[list[float]]] = None,
    restart_iterations: Optional[list[int]] = None,
) -> Tensor:
    """Newton-Schulz orthogonalization (default: Gram-space backend).

    Routes to :func:`gram_newton_schulz` by default for better precision
    (per-step coefficients, restart mechanism, SYRK acceleration).

    For the standard direct-space algorithm, use :func:`direct_newton_schulz`.

    Args:
        G: Gradient matrix (m, n), any dtype.
        steps: Ignored (determined by len(coefficients)).
        eps: Normalization epsilon.
        coefficients: Per-step coefficients. Defaults to POLAR_EXPRESS_COEFFICIENTS.
        restart_iterations: Restart positions. Defaults to [2].

    Returns:
        Orthogonalized update.
    """
    return gram_newton_schulz(
        G, steps=steps, eps=eps,
        coefficients=coefficients,
        restart_iterations=restart_iterations,
    )


# ---------------------------------------------------------------------------
# Gram NS — full-matrix Gram-space NS (TP-agnostic)
# ---------------------------------------------------------------------------
def gram_newton_schulz(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: Optional[list[list[float]]] = None,
    restart_iterations: Optional[list[int]] = None,
    deterministic: bool = False,
) -> Tensor:
    """Gram-space Newton-Schulz on a full (un-sharded) matrix.

    Adapted from Dao-AILab/gram-newton-schulz.  Iterates on the Gram matrix
    instead of the full gradient; uses per-step coefficients and restart
    mechanism for numerical stability.

    **TP handling**: this function is TP-agnostic.  For TP-sharded
    parameters the runtime gathers the full matrix to a designated TP
    owner via an All-to-All before calling this function (see
    ``tp_design.md`` and ``dmuon._backends.fsdp2.group.tp_gather_grads``).
    There is no in-function TP all-reduce.

    Args:
        G: Full gradient matrix (m, n), any dtype.
        steps: Ignored (determined by len(coefficients)).
        eps: Normalization epsilon.
        coefficients: Per-step coefficients. Defaults to POLAR_EXPRESS_COEFFICIENTS.
        restart_iterations: Iteration indices for restart. Defaults to [2].
        deterministic: If True, use cuBLAS for all ops (no SYRK kernel).

    Returns:
        Orthogonalized update, same shape as G.
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
    # Force row-major once so SYRK doesn't have to silently re-copy a
    # stride-swapped view on every call.
    X = X.contiguous()

    # Initial SYRK: R = X @ X^T
    m = X.shape[0]
    _use_syrk = _HAS_SYRK and X.is_cuda and not deterministic
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
            # Op 2: Z = c*R@R^T + b*R  (B==A, symmetric)
            _syrk_or_cublas(R, Z, C=R, alpha=c, beta=b)

            if Q is None:
                # Op 3: Q = Z + a*I  (first iter, B==A, symmetric)
                need_R_evolve = i < len(coefficients) - 1 and (i + 1) not in restart_iterations
                if not need_R_evolve:
                    _syrk_or_cublas(R, Q_bufs[q_idx], C=R, alpha=c, beta=b, diag_add=a)
                else:
                    Q_bufs[q_idx].copy_(Z)
                    Q_bufs[q_idx].diagonal().add_(a)
                Q = Q_bufs[q_idx]
            else:
                # Op 4: Q_new = Z@Q^T + a*Q  (B!=A, NOT symmetric → cuBLAS)
                q_next = 1 - q_idx
                torch.addmm(Q, Z, Q.T, alpha=1.0, beta=a, out=Q_bufs[q_next])
                Q = Q_bufs[q_next]
                q_idx = q_next

            if i < len(coefficients) - 1 and (i + 1) not in restart_iterations:
                # Ops 5,6: B!=A, symmetric (Z,R,RZ are polynomials of same
                # symmetric matrix → commute → result symmetric). SYRK OK.
                _syrk_or_cublas(R, RZ_buf, B=Z, C=R, alpha=1.0, beta=a)
                _syrk_or_cublas(RZ_buf, R_new, B=Z, C=RZ_buf, alpha=1.0, beta=a)
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


def gram_newton_schulz_factors(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: Optional[list[list[float]]] = None,
    restart_iterations: Optional[list[int]] = None,
    deterministic: bool = False,
) -> tuple[tuple[Tensor, ...], bool, Tensor]:
    """Return Gram-space NS left-factor segments for ``G``.

    ``gram_newton_schulz(G)`` applies one Gram multiplier segment per restart
    window to the normalized and possibly transposed view of ``G``.  Returning
    the segments instead of pre-composing them preserves the same fp16
    matrix-multiplication order as the original implementation.

    These factors are unrelated to attention Q projections.  The internal
    variable name ``Q`` follows the Gram-NS algebra, where the update is formed
    by left-multiplying the normalized matrix by one or more Gram factors.  The
    TP path can broadcast these smaller Gram factors and let every TP rank
    apply them to its local sequence/column shard instead of scattering the
    full update.

    Returns:
        ``(factor_segments, transposed, normalizer)`` where ``normalizer`` is
        ``G.norm()+eps`` under the same orientation rule as
        :func:`gram_newton_schulz`.
    """
    if coefficients is None:
        coefficients = DEFAULT_COEFFICIENTS
    if restart_iterations is None:
        restart_iterations = DEFAULT_RESTART_ITERATIONS

    X = G.float()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    normalizer = X.norm() + eps
    X = (X / normalizer).half().contiguous()

    m = X.shape[0]
    _use_syrk = _HAS_SYRK and X.is_cuda and not deterministic
    if _use_syrk:
        R = torch.empty(m, m, device=X.device, dtype=X.dtype)
        _syrk_or_cublas(X, R)
    else:
        R = X @ X.T

    I = torch.eye(m, device=X.device, dtype=X.dtype) if not _use_syrk else None
    factor: Optional[Tensor] = None
    factor_segments: list[Tensor] = []
    Z = torch.empty_like(R) if _use_syrk else None
    factor_bufs = [torch.empty_like(R), torch.empty_like(R)] if _use_syrk else [None, None]
    factor_idx = 0
    RZ_buf = torch.empty_like(R) if _use_syrk else None
    R_new = torch.empty_like(R) if _use_syrk else None

    for i, (a, b, c) in enumerate(coefficients):
        if i in restart_iterations and i != 0:
            assert factor is not None
            factor_segments.append(factor.clone())
            X = factor @ X
            if _use_syrk:
                _syrk_or_cublas(X, R)
            else:
                R = X @ X.T
            factor = None

        if _use_syrk:
            _syrk_or_cublas(R, Z, C=R, alpha=c, beta=b)

            if factor is None:
                need_R_evolve = (
                    i < len(coefficients) - 1 and (i + 1) not in restart_iterations
                )
                if not need_R_evolve:
                    _syrk_or_cublas(
                        R, factor_bufs[factor_idx], C=R, alpha=c, beta=b, diag_add=a
                    )
                else:
                    factor_bufs[factor_idx].copy_(Z)
                    factor_bufs[factor_idx].diagonal().add_(a)
                factor = factor_bufs[factor_idx]
            else:
                factor_next = 1 - factor_idx
                torch.addmm(
                    factor,
                    Z,
                    factor.T,
                    alpha=1.0,
                    beta=a,
                    out=factor_bufs[factor_next],
                )
                factor = factor_bufs[factor_next]
                factor_idx = factor_next

            if i < len(coefficients) - 1 and (i + 1) not in restart_iterations:
                _syrk_or_cublas(R, RZ_buf, B=Z, C=R, alpha=1.0, beta=a)
                _syrk_or_cublas(RZ_buf, R_new, B=Z, C=RZ_buf, alpha=1.0, beta=a)
                R = R_new
        else:
            Z_t = torch.addmm(R, R, R, alpha=c, beta=b)
            if factor is None:
                factor = Z_t + a * I
            else:
                factor = torch.addmm(factor, Z_t, factor, beta=a)
            if i < len(coefficients) - 1 and (i + 1) not in restart_iterations:
                RZ_t = torch.addmm(R, R, Z_t, beta=a)
                R = torch.addmm(RZ_t, Z_t, RZ_t, beta=a)

    assert factor is not None
    factor_segments.append(factor.clone())
    return tuple(factor_segments), transposed, normalizer


def gram_newton_schulz_q(*args, **kwargs):
    """Backward-compatible alias for :func:`gram_newton_schulz_factors`.

    The old name used ``Q`` from Gram-NS algebra.  Prefer
    ``gram_newton_schulz_factors`` in new code to avoid confusion with
    attention Q projections.
    """

    return gram_newton_schulz_factors(*args, **kwargs)


def gram_newton_schulz_distributed_local(
    G_local: Tensor,
    group: Optional[dist.ProcessGroup],
    *,
    transposed: bool,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: Optional[list[list[float]]] = None,
    restart_iterations: Optional[list[int]] = None,
) -> Tensor:
    """Gram-space NS for a local column shard of the oriented matrix.

    This is the math kernel for a Megatron-style TP shard-owner update path.
    The full logical matrix is first oriented exactly like
    :func:`gram_newton_schulz`: if ``transposed`` is true, ``G_local.T`` is the
    local shard.  The oriented local shard must be a column shard of the full
    oriented matrix.  The function reconstructs only the small Gram matrix via
    TP all-reduce and returns the local shard of the full NS output.

    It intentionally does not gather the full gradient, broadcast Q, or scatter
    a full update.  The caller remains responsible for momentum, scaling, and
    weight decay.
    """
    del steps
    if coefficients is None:
        coefficients = DEFAULT_COEFFICIENTS
    if restart_iterations is None:
        restart_iterations = DEFAULT_RESTART_ITERATIONS

    def _sum_across_group(tensor: Tensor) -> Tensor:
        if (
            group is None
            or not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size(group=group) == 1
        ):
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor

    original_dtype = G_local.dtype
    X = G_local.float()
    if transposed:
        X = X.T

    local_norm2 = X.square().sum()
    norm2 = _sum_across_group(local_norm2.reshape(()).clone())
    normalizer = norm2.sqrt() + eps
    X = (X / normalizer).half().contiguous()

    def _global_gram(local_x: Tensor) -> Tensor:
        gram = local_x @ local_x.T
        if (
            group is not None
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size(group=group) > 1
        ):
            # Gloo support for CPU fp16 reductions varies by build.  Reduce in
            # fp32, then continue with the same fp16 Gram dtype as the local
            # NS path.
            reduced = gram.float()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)
            gram = reduced.to(gram.dtype)
        return gram

    R = _global_gram(X)
    I = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
    Q: Optional[Tensor] = None

    for i, (a, b, c) in enumerate(coefficients):
        if i in restart_iterations and i != 0:
            assert Q is not None
            X = Q @ X
            R = _global_gram(X)
            Q = None

        Z_t = torch.addmm(R, R, R, alpha=c, beta=b)
        if Q is None:
            Q = Z_t + a * I
        else:
            Q = torch.addmm(Q, Z_t, Q, beta=a)
        if i < len(coefficients) - 1 and (i + 1) not in restart_iterations:
            RZ_t = torch.addmm(R, R, Z_t, beta=a)
            R = torch.addmm(RZ_t, Z_t, RZ_t, beta=a)

    assert Q is not None
    X = Q @ X
    if transposed:
        X = X.T
    return X.to(original_dtype)


def gram_newton_schulz_distributed_local_batched(
    G_local: Tensor,
    group: Optional[dist.ProcessGroup],
    *,
    transposed: bool,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: Optional[list[list[float]]] = None,
    restart_iterations: Optional[list[int]] = None,
) -> Tensor:
    """Batched variant of :func:`gram_newton_schulz_distributed_local`.

    ``G_local`` must be a 3D tensor ``[batch, rows, cols]`` containing local
    TP shards with identical logical orientation.  This helper is intentionally
    narrow: it is the math primitive needed to replace many same-shaped
    per-param Gram all-reduces with one packed all-reduce bucket.
    """
    del steps
    if G_local.dim() != 3:
        raise ValueError(
            "gram_newton_schulz_distributed_local_batched expects a 3D "
            f"[batch, rows, cols] tensor, got shape={tuple(G_local.shape)}"
        )
    if coefficients is None:
        coefficients = DEFAULT_COEFFICIENTS
    if restart_iterations is None:
        restart_iterations = DEFAULT_RESTART_ITERATIONS

    def _has_multi_rank_group() -> bool:
        return (
            group is not None
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size(group=group) > 1
        )

    original_dtype = G_local.dtype
    X = G_local.float()
    if transposed:
        X = X.transpose(-2, -1)

    local_norm2 = X.square().sum(dim=(-2, -1))
    norm2 = local_norm2.clone()
    if _has_multi_rank_group():
        dist.all_reduce(norm2, op=dist.ReduceOp.SUM, group=group)
    normalizer = norm2.sqrt().add_(eps).view(-1, 1, 1)
    X = (X / normalizer).half().contiguous()

    def _global_gram(local_x: Tensor) -> Tensor:
        gram = torch.stack([x @ x.T for x in local_x], dim=0)
        if _has_multi_rank_group():
            reduced = gram.float()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)
            gram = reduced.to(gram.dtype)
        return gram

    R = _global_gram(X)
    eye = torch.eye(R.shape[-1], device=R.device, dtype=R.dtype)
    Q: Optional[Tensor] = None

    for i, (a, b, c) in enumerate(coefficients):
        if i in restart_iterations and i != 0:
            assert Q is not None
            X = torch.stack([q_i @ x_i for q_i, x_i in zip(Q, X)], dim=0)
            R = _global_gram(X)
            Q = None

        Z = torch.stack(
            [torch.addmm(r_i, r_i, r_i, alpha=c, beta=b) for r_i in R],
            dim=0,
        )
        if Q is None:
            Q = Z + a * eye
        else:
            Q = torch.stack(
                [
                    torch.addmm(q_i, z_i, q_i, beta=a)
                    for q_i, z_i in zip(Q, Z)
                ],
                dim=0,
            )
        if i < len(coefficients) - 1 and (i + 1) not in restart_iterations:
            R = torch.stack(
                [
                    torch.addmm(
                        rz_i,
                        z_i,
                        rz_i,
                        beta=a,
                    )
                    for z_i, rz_i in (
                        (
                            z_i,
                            torch.addmm(r_i, r_i, z_i, beta=a),
                        )
                        for r_i, z_i in zip(R, Z)
                    )
                ],
                dim=0,
            )

    assert Q is not None
    X = torch.stack([q_i @ x_i for q_i, x_i in zip(Q, X)], dim=0)
    if transposed:
        X = X.transpose(-2, -1)
    return X.to(original_dtype)


# Backward-compat alias: pre-refactor code paths referenced the "_local"
# variant to distinguish it from the (now removed) TP-aware variant.
gram_newton_schulz_local = gram_newton_schulz
