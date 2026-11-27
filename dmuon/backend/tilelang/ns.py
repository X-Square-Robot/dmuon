"""TileLang implementation of the full Gram-space Newton–Schulz schedule.

Drop-in replacement for ``dmuon.optim.newton_schulz.gram_newton_schulz``
(NPU tensors only).  The op sequence mirrors the reference implementation
one-for-one so the numerical contract — operand order, per-storage
roundings, restart semantics — carries over unchanged:

    X  = rint16(G_f32 / (‖G‖+eps))                     (2-stage AIV)
    R  = rint16(X @ Xᵀ)                                (Gram fp32 + cast)
    for i, (a, b, c) in coefficients:
        restart i: X = Q@X; R = rint16(X@Xᵀ); Q = None
        Q is None: Q = rint16(c·(R@R) + b·R) + aI       (two roundings)
        else:       Q = rint16(Z@Q + a·Q) etc.          (single rounding)
        if not last and no restart next:
            RZ = rint16(R@Z + a·R) ;  R = rint16(RZ@Z + a·RZ)
    update = fp32(Q @ X)   (transposed store for tall inputs)

All cube ops produce fp32 in GM (the only L0C exit the ascend backend
lowers reliably) and every fp16 rounding happens in an AIV pass — the
``axpby32`` epilogue sees the raw fp32 accumulator, so each polynomial
step keeps the eager addmm single-rounding contract exactly.

The GEMM work-set lives in a *padded* space (multiples of 16) for
misaligned shapes: zero padding contributes exactly zero to every
contraction; the padded diagonal terms only affect sliced-away rows.
Mainstream training shapes are 256-multiples and skip padding entirely.
"""

import threading
import os
from collections import OrderedDict
from contextlib import contextmanager

import torch

from .kernels import elementwise as ew
from .kernels import gemm as g
from .kernels import deep_wrapper as dg
from .dispatch import capture_kernel_handles


class _Workspace:
    """Persistent per-shape buffers (fp16 iterates + fp32 scratch)."""

    def __init__(self, m: int, n: int, device):
        # (m, n) are the *padded* work dims (m' <= n').
        self.m, self.n = m, n
        e = torch.empty
        self.x16 = torch.zeros((m, n), device=device, dtype=torch.float16)
        self.x16_t = torch.zeros((n, m), device=device, dtype=torch.float16)
        # Second (m, n) iterate buffer so the MN-major restart never
        # reads and writes X in place.
        self.x16_alt = torch.zeros((m, n), device=device, dtype=torch.float16)
        self.r = e((m, m), device=device, dtype=torch.float16)
        self.r2 = e((m, m), device=device, dtype=torch.float16)
        self.z = e((m, m), device=device, dtype=torch.float16)
        self.q = e((m, m), device=device, dtype=torch.float16)
        self.q2 = e((m, m), device=device, dtype=torch.float16)
        self.rz = e((m, m), device=device, dtype=torch.float16)
        self.mirror_t = e((m, m), device=device, dtype=torch.float16)
        self.acc = e((m, m), device=device, dtype=torch.float32)  # gemm scratch (square)
        self.accx = e((m, n), device=device, dtype=torch.float32)  # gemm scratch (Q@X)
        self.partials = torch.zeros((64,), device=device, dtype=torch.float32)
        self.f32 = e((m, n), device=device, dtype=torch.float32)


_lock = threading.Lock()
_workspaces: dict = OrderedDict()


def _stream_key(device):
    stream = torch.npu.current_stream(device)
    return (str(device), int(stream.npu_stream))


def _tensor_bytes(obj):
    return sum(t.numel() * t.element_size() for t in vars(obj).values() if torch.is_tensor(t))


def _workspace(m: int, n: int, device, *, natural=False):
    cache_key = (m, n, _stream_key(device), natural)
    with _lock:
        ws = _workspaces.get(cache_key)
        if ws is None:
            factory = _NaturalWorkspace if natural else _Workspace
            ws = _workspaces[cache_key] = factory(m, n, device)
        _workspaces.move_to_end(cache_key)
        budget = int(os.environ.get("DMUON_TILELANG_WORKSPACE_BYTES", 2 * 1024**3))
        while len(_workspaces) > 1 and sum(_tensor_bytes(w) for w in _workspaces.values()) > budget:
            _workspaces.popitem(last=False)
        return ws


def _resolve_defaults(coefficients, restart_iterations):
    if coefficients is None or restart_iterations is None:
        from dmuon.optim.newton_schulz import (
            DEFAULT_COEFFICIENTS as dc,
            DEFAULT_RESTART_ITERATIONS as dr,
        )

        coefficients = dc if coefficients is None else coefficients
        restart_iterations = dr if restart_iterations is None else restart_iterations
    return coefficients, restart_iterations


class _NaturalWorkspace:
    """Buffers for the staging-free transposed path.

    The iterate is held as Xw (M, N) -- the input's own layout -- while the
    Gram and every polynomial are (N, N).  Sizing by M for the square buffers
    would be M^2, hence this separate workspace.
    """

    def __init__(self, M: int, N: int, device):
        self.M, self.N = M, N
        e = torch.empty
        self.x16 = e((M, N), device=device, dtype=torch.float16)
        self.x16_alt = e((M, N), device=device, dtype=torch.float16)
        self.r = e((N, N), device=device, dtype=torch.float16)
        self.r2 = e((N, N), device=device, dtype=torch.float16)
        self.z = e((N, N), device=device, dtype=torch.float16)
        self.q = e((N, N), device=device, dtype=torch.float16)
        self.q2 = e((N, N), device=device, dtype=torch.float16)
        self.rz = e((N, N), device=device, dtype=torch.float16)
        self.acc = e((N, N), device=device, dtype=torch.float32)
        self.partials = torch.zeros((64,), device=device, dtype=torch.float32)
        self.f32 = e((M, N), device=device, dtype=torch.float32)


def _momentum_norm_enabled() -> bool:
    """Fuse the squared-partial reduction into the momentum pass."""
    return os.environ.get("DMUON_TILELANG_MOMENTUM_NORM", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _mn_a_enabled() -> bool:
    """Staging-free transposed path (Gram via MN-major on the natural layout)."""
    return os.environ.get("DMUON_TILELANG_MN_A", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _use_natural_layout(rows, cols):
    return _mn_a_enabled() and rows > cols and rows % 16 == 0 and cols % 16 == 0


def _pad16(v: int) -> int:
    return (v + 15) // 16 * 16


def _symmetric_signature():
    return tuple(
        os.environ.get(name, default)
        for name, default in (
            ("DMUON_TILELANG_SYMMETRIC", "auto"),
            ("DMUON_TILELANG_SYMMETRIC_MIRROR", "separate"),
            ("DMUON_TILELANG_SYMMETRIC_CONFIG", ""),
            ("DMUON_TILELANG_MIRROR_COUNTER", "0"),
            ("DMUON_TILELANG_MIRROR_PADDING", "0"),
            ("DMUON_TILELANG_MIRROR_SIMD", "0"),
        )
    )


def _symmetric_beats_dense(m: int, k: int) -> bool:
    """Measured gate for choosing lower+mirror over the full-grid Gram/poly.

    Kernel-level wins are solid at m>=2048 with long contraction
    (2048x8192 1.2x blk160, 2560x8192+ 1.4x+, 5120^2 1.41x, 3584x18944
    1.43x) and marginal below (2560^2 1.05x, 2048^2 wash).  In-step each
    symmetric product adds one dependent mirror launch, so marginal kernel
    wins flip negative in the real optimizer step (shunfeng replay:
    auto-with-peak-TFLOPS-model 0.908x vs off 1.018x on 2026-11-05).
    The gate keeps only the clearly-winning region; a peak-rate model is
    wrong here because small-m dense runs at 47-100 TFLOPS in context.
    Boundary evidence (libero replay 20261105-08/09): m=2560 square polys
    are net-positive in context, m<=2048 square polys are net-negative
    (each symmetric product adds a dependent mirror launch).
    ``DMUON_TILELANG_SYMMETRIC_FORCE=1`` bypasses it for A/B measurements.
    """
    if os.environ.get("DMUON_TILELANG_SYMMETRIC_FORCE", "0") == "1":
        return True
    return m >= 2560 or m * k >= 12_000_000


def _mn_b_enabled() -> bool:
    """MN-major B for the final GEMM: drops transpose2d and x16_t.

    Verified bitwise against the NT path (4 shapes, rel 0.0); the GEMM
    itself costs the same (25us vs 26us at 1024x4096).
    """
    return os.environ.get("DMUON_TILELANG_MN_B", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _poly_pair_enabled() -> bool:
    # Grouped two-problem poly launch; DMUON_TILELANG_POLY_PAIR=0 disables.
    return os.environ.get("DMUON_TILELANG_POLY_PAIR", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _ns_ops():
    """Gram/poly op selection.

    ``auto`` (default) picks the lower+mirror profile exactly where the
    measured cost model says it wins, and the dense DeepGEMM path elsewhere;
    both are 100 % TileLang. ``gram``/``all`` force the symmetric profile for
    the Gram only or for every product, and ``off`` keeps the dense path.
    """
    mode, mirror, geometry, counter, padding, simd = _symmetric_signature()
    if mode in ("off", "0"):
        return dg.cast_fp16, dg.poly_fp16, dg.poly_fp16_diag
    if mode not in ("auto", "gram", "all") or mirror not in ("separate", "fused"):
        raise ValueError("SYMMETRIC must be off/auto/gram/all and mirror separate/fused")
    forced = tuple(map(int, geometry.split(","))) if geometry else None

    def _symmetric(a, bt, c, out, alpha, beta, q=None, diag=0.0):
        m, k = a.shape
        tuned_square = mode == "auto" and not forced and m == k == 2048
        if mode == "auto" and not tuned_square and not _symmetric_beats_dense(m, k):
            return False
        geometry = forced or (
            (256, 128, 128) if m >= 2560 else ((160, 160, 80) if m >= 2048 else (128, 128, 128))
        )
        block, mad_m, mad_n = (128, 128, 128) if tuned_square else geometry
        cfg = dg.triangular_config(
            m, k, block=block, mad_m=mad_m, mad_n=mad_n, epilogue_stages=1 if c is not None else 2
        )
        dg.symmetric_fp16(
            a,
            bt,
            c,
            out,
            alpha=alpha,
            beta=beta,
            config=cfg,
            mirror="fused" if tuned_square else mirror,
            diag_out=q,
            diag_add=diag,
            mirror_counter=counter == "1",
            mirror_padding=16 if tuned_square else int(padding),
            mirror_simd=tuned_square or simd == "1",
        )
        return True

    def _dense(a, bt, c, out, alpha, beta, q=None, diag=0.0):
        if c is None:
            dg.cast_fp16(a, bt, out, alpha)
        elif q is None:
            dg.poly_fp16(a, bt, c, out, alpha, beta)
        else:
            dg.poly_fp16_diag(a, bt, c, out, q, alpha, beta, diag)

    def gram(a, bt, out):
        if not _symmetric(a, bt, None, out, 1.0, 0.0):
            dg.cast_fp16(a, bt, out)

    def poly(a, bt, c, out, alpha, beta):
        if not _symmetric(a, bt, c, out, alpha, beta):
            dg.poly_fp16(a, bt, c, out, alpha, beta)

    def poly_diag(a, bt, c, z, q, alpha, beta, diag):
        if not _symmetric(a, bt, c, z, alpha, beta, q, diag):
            dg.poly_fp16_diag(a, bt, c, z, q, alpha, beta, diag)

    symmetric_poly = mode in ("auto", "all")
    return (
        gram,
        poly if symmetric_poly else dg.poly_fp16,
        poly_diag if symmetric_poly else dg.poly_fp16_diag,
    )


def _gram_newton_schulz_direct(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients=None,
    restart_iterations=None,
    deterministic: bool = False,
    reference=None,
    _workspace_override=None,
) -> torch.Tensor:
    if G.device.type != "npu":
        # CPU tensors use the reference implementation (supported path).
        if reference is None:
            raise RuntimeError("tilelang NS called without reference for CPU input")
        return reference(
            G,
            steps=steps,
            eps=eps,
            coefficients=coefficients,
            restart_iterations=restart_iterations,
            deterministic=deterministic,
        )

    coefficients, restart_iterations = _resolve_defaults(coefficients, restart_iterations)
    gram_op, poly_op, poly_diag_op = _ns_ops()
    original_dtype = G.dtype
    if G.ndim != 2:
        raise ValueError("Newton-Schulz expects a matrix")
    if G.numel() == 0:
        return torch.empty_like(G)

    M, N = G.shape
    transposed = M > N
    # Staging-free transposed path: keep Xw in the input's natural (M, N)
    # layout, compute the Gram as Xw^T Xw (both operands MN-major) and derive
    # the new iterate with a plain NT product.  This removes the (M,N)
    # transpose staging, the transposed fp16 copy and the restart transpose.
    natural = _use_natural_layout(M, N)
    if natural:
        X_f32 = ew.cast_to_fp32(G)
        m, n = N, M
    elif transposed:
        # Layout-only staging copy (no compute): work on the wide view.
        X_f32 = ew.cast_to_fp32(G).t().contiguous()
        m, n = N, M
    else:
        X_f32 = ew.cast_to_fp32(G)
        m, n = M, N

    aligned = m % 16 == 0 and n % 16 == 0
    mp, np_ = (m, n) if aligned else (_pad16(m), _pad16(n))

    if natural:
        ws = _workspace_override or _workspace(M, N, G.device, natural=True)
        x16 = ws.x16
        x16_t = None
        alt = ws.x16_alt
    else:
        ws = _workspace_override or _workspace(mp, np_, G.device)

    # Normalize: partial sums → total → X16 (compact (m, n) first).
    ctx_partials = (
        getattr(_OPT_CONTEXT, "partials", None)
        if getattr(_OPT_CONTEXT, "input", None) is G
        else None
    )
    if ctx_partials is None:
        ew.norm_stage1(X_f32, ws.partials)
        partials = ws.partials
    else:
        # The caller produced these while writing G; the sum of squares is
        # layout-invariant, so no second pass over the data is needed.
        partials = ctx_partials
    mn_b = _mn_b_enabled() and not transposed and min(M, N) % 16 == 0 and max(M, N) % 16 == 0
    if not natural:
        alt = ws.x16_alt if mn_b else None
    if natural:
        ew.norm_stage2(X_f32, partials, ws.x16, eps)
    elif aligned and mn_b:
        # The MN-major final GEMM reads X16 in place, so the transposed
        # copy that norm_stage2_t writes is never consumed.
        ew.norm_stage2(X_f32, partials, ws.x16, eps)
        x16, x16_t = ws.x16, None
    elif aligned:
        # Fused normalize + transposed store replaces the separate
        # transpose2d pass (one launch and one X16 read saved per NS).
        ew.norm_stage2_t(X_f32, partials, ws.x16, ws.x16_t, eps)
        x16, x16_t = ws.x16, ws.x16_t
    else:
        x16_compact = torch.empty((m, n), device=G.device, dtype=torch.float16)
        ew.norm_stage2(X_f32, partials, x16_compact, eps)
        # Zero-pad staging (layout only): one TileLang kernel writes the
        # compact region and zeroes the pad, replacing the eager fill + slice.
        from .kernels.padding import pad2d

        pad2d(x16_compact, ws.x16)
        x16 = ws.x16
        ew.transpose2d(ws.x16, ws.x16_t)
        x16_t = ws.x16_t

    # Initial Gram: R = rint16(X @ Xᵀ).  The adaptive profile picks either
    # the full-grid DeepGEMM MAD or the lower-triangle SYRK + mirror,
    # whichever the measured cost model says is faster for (m, k).
    r, r2 = ws.r, ws.r2
    if natural:
        dg.syrk_aa(x16, r)
    else:
        gram_op(x16, x16, r)

    z = ws.z
    q, q2 = ws.q, ws.q2
    rz = ws.rz
    acc = ws.acc
    q_valid = False
    n_steps = len(coefficients)

    for i, (a, b, c) in enumerate(coefficients):
        if i in restart_iterations and i != 0:
            # X = rint16(Q @ X), refresh X_T, rebuild Gram, all fused.
            if natural:
                # Xw <- rint16(Xw @ Q^T): the transposed iterate is exactly
                # the natural-layout product, so no transpose is needed.
                dg.cast_fp16(x16, q, alt)
                x16, alt = alt, x16
            elif mn_b:
                # Q @ X into the alternate buffer: every block reads all
                # rows of X while writing its own, so never in place.
                dg.cast_mn_b(q, x16, alt)
                x16, alt = alt, x16
            else:
                dg.cast_fp16(q, x16_t, x16)
                ew.transpose2d(x16, x16_t)
            if natural:
                dg.syrk_aa(x16, r)
            else:
                gram_op(x16, x16, r)
            q_valid = False

        # Dense raw-FP32 polynomial products need not remain bitwise
        # symmetric after rounding (independently observed for RZ/RZZ).
        # The opt-in symmetric profile explicitly chooses lower+mirror;
        # it is a separate numerical profile, not an assertion of exact
        # finite-precision commutation in the dense profile.
        # The pair kernel is validated on tail-free square geometries
        # (the block evenly divides the iterate); on tailed squares its
        # element-offset branch diverges (11.5% vs 2.3% at 1008x1008,
        # 15.9% vs 4.9% at 488x488, 2026-11-06).  Gate it to block-aligned
        # squares; sub-block squares use block == m and stay eligible.
        pair_ready = (
            q_valid
            and _poly_pair_enabled()
            and i < n_steps - 1
            and (i + 1) not in restart_iterations
            and r.shape == z.shape == q.shape
            and (r.shape[0] < 256 or r.shape[0] % 256 == 0)
        )
        if not q_valid:
            # Z and Q are written by the same fused epilogue; no separate
            # diag_add16 launch or extra GM pass is needed.
            poly_diag_op(r, r, r, z, q, c, b, a)
            q_valid = True
        else:
            poly_op(r, r, r, z, c, b)
            if pair_ready:
                # q_next = poly(Z, Q) and RZ = poly(R, Z) are independent
                # once Z is fresh: one grouped launch replaces two
                # (design + evidence in kernels/deepgemm/pair.py).
                dg.poly_pair_fp16(z, q, q, q2, 1.0, a, r, z, r, rz, 1.0, a)
            else:
                poly_op(z, q, q, q2, 1.0, a)
            q, q2 = q2, q

        if i < n_steps - 1 and (i + 1) not in restart_iterations:
            if not pair_ready:
                poly_op(r, z, r, rz, 1.0, a)
            poly_op(rz, z, rz, r2, 1.0, a)
            r, r2 = r2, r

    # Final: update = fp32(Q @ X).
    if aligned and original_dtype == torch.float32:
        # Write directly into an independent output allocation: public NS
        # results must not alias the reusable workspace.  For M > N inputs
        # the swapped form X^T @ Q^T == (Q @ X)^T lands the transposed result
        # in the GEMM epilogue itself, replacing the ws.f32 round trip plus
        # the separate cast_transposed pass (per-element dot products are
        # identical; only the output addressing changes).
        out = torch.empty(M, N, device=G.device, dtype=original_dtype)
        if natural:
            # out (M, N) = Xw @ Q^T, straight from the natural layout.
            dg.gemm_fp32(x16, q, out)
        elif transposed:
            # (Q @ X)^T == X^T @ Q^T lands the transposed result directly in
            # the epilogue, replacing the ws.f32 staging plus cast pass.
            g.gemm_nt(x16_t, q, out)
        elif mn_b:
            dg.gemm_fp32_mn_b(q, x16, out)
        else:
            g.gemm_nt(q, x16_t, out)
        return out
    if natural:
        dg.gemm_fp32(x16, q, ws.f32)
    else:
        g.gemm_nt(q, x16_t if x16_t is not None else x16.t().contiguous(), ws.f32)
    src = ws.f32[:m, :n]
    out = torch.empty(M, N, device=G.device, dtype=original_dtype)
    if transposed:
        # out (M, N) = transpose of the (m, n) work output.
        ew.cast_transposed(src.contiguous(), out)
    else:
        ew.cast_scale(src.contiguous().view(-1), out.view(-1), 1.0)
    return out


# Graphs own every captured workspace and are isolated by stream and numerical
# signature. Optimizer-only borrowing avoids both staging and result copies.
_GRAPH_CACHE = OrderedDict()
_GRAPH_LOCK = threading.RLock()
_OPT_INPUTS = OrderedDict()
_OPT_CONTEXT = threading.local()


def optimizer_input(shape, dtype, device):
    """Stable per-stream NS input, consumed before the next parameter reuses it."""
    key = (tuple(shape), str(dtype), _stream_key(device))
    with _GRAPH_LOCK:
        buf = _OPT_INPUTS.get(key)
        if buf is None:
            buf = _OPT_INPUTS[key] = torch.empty(shape, dtype=dtype, device=device)
        _OPT_INPUTS.move_to_end(key)
        budget = int(os.environ.get("DMUON_TILELANG_INPUT_BYTES", 512 * 1024**2))
        while (
            len(_OPT_INPUTS) > 1
            and sum(t.numel() * t.element_size() for t in _OPT_INPUTS.values()) > budget
        ):
            _OPT_INPUTS.popitem(last=False)
        return buf


@contextmanager
def optimizer_borrow(G, partials=None):
    """Internal synchronous enqueue scope; borrowed results never escape apply.

    ``partials`` optionally carries the squared-partial sums the caller already
    produced while writing G (see elementwise.momentum_norm), which lets the
    normalize step skip its own full re-read of G.
    """
    previous = getattr(_OPT_CONTEXT, "input", None)
    previous_p = getattr(_OPT_CONTEXT, "partials", None)
    _OPT_CONTEXT.input = G
    _OPT_CONTEXT.partials = partials
    try:
        yield
    finally:
        _OPT_CONTEXT.input = previous
        _OPT_CONTEXT.partials = previous_p


_OPT_PARTIALS: dict = {}


def optimizer_partials(numel, device):
    """Stable (64,) fp32 partial buffer for a given element count."""
    key = (int(numel), _stream_key(device))
    with _GRAPH_LOCK:
        buf = _OPT_PARTIALS.get(key)
        if buf is None:
            buf = _OPT_PARTIALS[key] = torch.zeros((64,), device=device, dtype=torch.float32)
        return buf


def clear_graph_cache():
    """Explicit safe teardown, including already-enqueued output consumers."""
    with _GRAPH_LOCK:
        for entry in _GRAPH_CACHE.values():
            entry["stream"].synchronize()
        _GRAPH_CACHE.clear()
        _OPT_INPUTS.clear()
        _OPT_PARTIALS.clear()


def graph_cache_status():
    with _GRAPH_LOCK:
        return {
            "entries": len(_GRAPH_CACHE),
            "bytes": sum(e["bytes"] for e in _GRAPH_CACHE.values()),
            "input_bytes": sum(t.numel() * t.element_size() for t in _OPT_INPUTS.values()),
        }


def _graph_room(required):
    budget = int(os.environ.get("DMUON_TILELANG_GRAPH_BYTES", 2 * 1024**3))
    limit = int(os.environ.get("DMUON_TILELANG_GRAPH_ENTRIES", 64))
    if required > budget or limit < 1:
        return False
    # Strict admission, no eviction: evicting a completed graph to admit a
    # new shape makes the evicted shape re-capture on its next step, which
    # turned the steady state into a capture storm (libero replay 2026-11-05,
    # 16M cap: 0.659x with constant re-capture and 17 GB peak).  Once the
    # budget is full, further shapes simply stay on the direct path.
    if len(_GRAPH_CACHE) >= limit:
        return False
    return sum(e["bytes"] for e in _GRAPH_CACHE.values()) + required <= budget


def _graph_enabled(G=None) -> bool:
    enabled = os.environ.get("DMUON_TILELANG_GRAPH", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    # Graph pays off for small NS calls.  Larger inputs make the required
    # stage copy more expensive than the host-launch savings it removes.
    # Private optimizer inputs skip the stage copy but must respect the same
    # size cap: admitting every private input graphed the whole population
    # and thrashed the graph-workspace budget (libero replay 2026-11-05,
    # 0.407x); capped private graphs stay small and launch-bound.
    if not enabled:
        return False
    if G is None:
        return True
    limit = int(os.environ.get("DMUON_TILELANG_GRAPH_MAX_NUMEL", 4 * 1024 * 1024))
    return G.numel() <= limit


def gram_newton_schulz_tilelang(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients=None,
    restart_iterations=None,
    deterministic: bool = False,
    reference=None,
) -> torch.Tensor:
    """Public NS entry; capture/replay the direct tilelang schedule per shape."""
    if G.device.type != "npu" or G.numel() == 0 or not _graph_enabled(G):
        return _gram_newton_schulz_direct(
            G,
            steps=steps,
            eps=eps,
            coefficients=coefficients,
            restart_iterations=restart_iterations,
            deterministic=deterministic,
            reference=reference,
        )

    coefficients, restart_iterations = _resolve_defaults(coefficients, restart_iterations)
    private = getattr(_OPT_CONTEXT, "input", None) is G
    # Strided public tensors use a compact persistent stage; private inputs
    # come from optimizer_input and must retain their captured address.
    key = (
        tuple(G.shape),
        str(G.dtype),
        _stream_key(G.device),
        float(eps),
        tuple(tuple(float(v) for v in row) for row in coefficients),
        tuple(int(v) for v in restart_iterations),
        bool(deterministic),
        G.data_ptr() if private else None,
        _symmetric_signature(),
    )
    with _GRAPH_LOCK:
        entry = _GRAPH_CACHE.get(key)
        if entry is None:
            import torch_npu

            m, n = sorted(G.shape)
            mp, np_ = _pad16(m), _pad16(n)
            # 7 square fp16, 1 square fp32; 2 rectangular fp16 + 2 fp32,
            # plus stage/output. Conservative admission before allocation.
            estimated = 18 * mp * mp + 12 * mp * np_ + 256 + 2 * G.numel() * G.element_size()
            estimated += (
                4 * G.numel() * (int(G.dtype != torch.float32) + int(G.shape[0] > G.shape[1]))
            )
            if (mp, np_) != (m, n):
                estimated += 6 * G.numel()  # compact normalized input and cropped FP32 result
            if not _graph_room(estimated):
                return _gram_newton_schulz_direct(
                    G,
                    steps=steps,
                    eps=eps,
                    coefficients=coefficients,
                    restart_iterations=restart_iterations,
                    deterministic=deterministic,
                    reference=reference,
                )
            stage = G if private else torch.empty(G.shape, dtype=G.dtype, device=G.device)
            if not private:
                stage.copy_(G)
            ws = (
                _NaturalWorkspace(*G.shape, G.device)
                if _use_natural_layout(*G.shape)
                else _Workspace(mp, np_, G.device)
            )
            kwargs = dict(
                steps=steps,
                eps=eps,
                coefficients=coefficients,
                restart_iterations=restart_iterations,
                deterministic=deterministic,
                reference=reference,
                _workspace_override=ws,
            )
            for _ in range(2):
                _gram_newton_schulz_direct(stage, **kwargs)
            stream = torch.npu.current_stream(G.device)
            stream.synchronize()
            graph = torch_npu.npu.NPUGraph()
            # Capture failures propagate. Retrying a potentially invalid
            # stream with eager compute would violate the backend contract.
            with capture_kernel_handles() as kernels, torch_npu.npu.graph(graph):
                graph_out = _gram_newton_schulz_direct(stage, **kwargs)
            entry = dict(
                graph=graph,
                stage=stage,
                output=graph_out,
                workspace=ws,
                kernels=kernels,
                stream=stream,
                bytes=estimated,
            )
            _GRAPH_CACHE[key] = entry
        _GRAPH_CACHE.move_to_end(key)
        entry.pop("retire_fence", None)
        if not private:
            entry["stage"].copy_(G)
        entry["graph"].replay()
        # Public calls always return independent storage, regardless of the
        # historical GRAPH_COPY_OUT escape hatch.
        return entry["output"] if private else entry["output"].clone()
