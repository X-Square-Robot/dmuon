"""AIV elementwise/reduction kernels for the DMuon step.

All kernels follow the official tilelang-ascend vector pattern
(``T.Kernel`` + ``T.Pipelined`` GM↔UB copies + ``T.SimtVF`` compute with
``T.alloc_reducer``/``T.reducer_update``/``T.finalize_reducer``
reductions), with concrete-N compilation and an LRU cache keyed by
(op, numel, dtypes).

Rounding contracts (must match the eager oracle at every storage boundary):

* momentum:  ``buf.mul_(mu); buf.add_(g)`` → two storage roundings in the
  state dtype; nesterov ``g.add(buf, alpha=mu)`` adds one more.
* normalize: X/(‖X‖+eps) in fp32 then a single fp32→fp16 RINT store; the
  transposed copy is produced in the same pass.
* apply:     ``owned.mul_(1-lr*wd)`` (store round), update ``.to(dtype)``
  (cast round), ``add_(update, alpha)`` (store round) — three low-dtype
  storage roundings, done in-register in one pass.
"""

import torch
import tilelang
import tilelang.ascend.language as T

from ..dispatch import cached_kernel, store_kernel

TL_F32 = torch.float32
TL_F16 = torch.float16

_NUM_BLOCKS = 64
_TILE = 8192  # elements per block per pipelined step (vecadd pattern)


def _dtype_name(dt: torch.dtype) -> str:
    return {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}[dt]


def _elements_per_iter(numel: int) -> int:
    return _NUM_BLOCKS * _TILE


def _iterations(numel: int) -> int:
    per = _elements_per_iter(numel)
    return max(1, (numel + per - 1) // per)


def _tail(numel: int) -> int:
    iters = _iterations(numel)
    return numel - (iters - 1) * _elements_per_iter(numel)


# ---------------------------------------------------------------------------
# Fused momentum + nesterov:  buf = mu*buf + g ;  out = g + mu*buf
# ---------------------------------------------------------------------------


def _build_momentum_fused(numel: int, g_dtype: str, buf_dtype: str, out_dtype: str, nesterov: bool):
    iters = _iterations(numel)
    tail = _tail(numel)

    @T.prim_func
    def main(
        G: T.Tensor((numel,), g_dtype),
        Buf: T.Tensor((numel,), buf_dtype),
        Out: T.Tensor((numel,), out_dtype),
        mu: T.float32,
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            g_ub = T.alloc_shared((_TILE,), g_dtype)
            buf_in_ub = T.alloc_shared((_TILE,), buf_dtype)
            buf_out_ub = T.alloc_shared((_TILE,), buf_dtype)
            out_ub = T.alloc_shared((_TILE,), out_dtype)
            T.annotate_buffer_versions({g_ub: 2, buf_in_ub: 2, buf_out_ub: 1, out_ub: 1})
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * _NUM_BLOCKS + bx) * _TILE
                rem = T.if_then_else(numel > begin, numel - begin, 0)
                n = T.if_then_else(rem >= _TILE, _TILE, rem)
                T.copy(G[begin : begin + n], g_ub[:n])
                T.copy(Buf[begin : begin + n], buf_in_ub[:n])
                with T.SimtVF(threads=256):
                    for i in T.Parallel(_TILE):
                        if i < n:
                            g = T.cast(g_ub[i], "float32")
                            t = T.cast(T.cast(buf_in_ub[i], "float32") * mu, buf_dtype)
                            new_buf = T.cast(T.cast(t, "float32") + g, buf_dtype)
                            buf_out_ub[i] = new_buf
                            if nesterov:
                                out_ub[i] = T.cast(g + mu * T.cast(new_buf, "float32"), out_dtype)
                            else:
                                out_ub[i] = T.cast(new_buf, out_dtype)
                T.copy(buf_out_ub[:n], Buf[begin : begin + n])
                T.copy(out_ub[:n], Out[begin : begin + n])

    return main


def momentum_fused(
    grad: torch.Tensor, buf: torch.Tensor, out: torch.Tensor, mu: float, nesterov: bool
):
    """buf ← μ·buf+g ; out ← g+μ·buf (nesterov) or buf (plain)."""
    numel = grad.numel()
    g_dt = _dtype_name(grad.dtype)
    b_dt = _dtype_name(buf.dtype)
    o_dt = _dtype_name(out.dtype)
    key = ("momentum_fused", numel, g_dt, b_dt, o_dt, nesterov)
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key,
            tilelang.compile(
                _build_momentum_fused(numel, g_dt, b_dt, o_dt, nesterov),
                target="ascend",
                out_idx=None,
            ),
        )
    k(grad.view(-1), buf.view(-1), out.view(-1), mu)


# ---------------------------------------------------------------------------
# First-step momentum init: buf = cast(g); out = g + mu*buf (nesterov)
# ---------------------------------------------------------------------------


def _build_momentum_init(numel: int, g_dtype: str, buf_dtype: str, out_dtype: str, nesterov: bool):
    iters = _iterations(numel)
    tail = _tail(numel)

    @T.prim_func
    def main(
        G: T.Tensor((numel,), g_dtype),
        Buf: T.Tensor((numel,), buf_dtype),
        Out: T.Tensor((numel,), out_dtype),
        mu: T.float32,
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            g_ub = T.alloc_shared((_TILE,), g_dtype)
            buf_ub = T.alloc_shared((_TILE,), buf_dtype)
            out_ub = T.alloc_shared((_TILE,), out_dtype)
            T.annotate_buffer_versions({g_ub: 2, buf_ub: 1, out_ub: 1})
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * _NUM_BLOCKS + bx) * _TILE
                rem = T.if_then_else(numel > begin, numel - begin, 0)
                n = T.if_then_else(rem >= _TILE, _TILE, rem)
                T.copy(G[begin : begin + n], g_ub[:n])
                with T.SimtVF(threads=256):
                    for i in T.Parallel(_TILE):
                        if i < n:
                            g = T.cast(g_ub[i], "float32")
                            stored = T.cast(g, buf_dtype)
                            buf_ub[i] = stored
                            if nesterov:
                                out_ub[i] = T.cast(g + mu * T.cast(stored, "float32"), out_dtype)
                            else:
                                out_ub[i] = T.cast(stored, out_dtype)
                T.copy(buf_ub[:n], Buf[begin : begin + n])
                T.copy(out_ub[:n], Out[begin : begin + n])

    return main


def momentum_init(
    grad: torch.Tensor, buf: torch.Tensor, out: torch.Tensor, mu: float, nesterov: bool
):
    numel = grad.numel()
    g_dt = _dtype_name(grad.dtype)
    b_dt = _dtype_name(buf.dtype)
    o_dt = _dtype_name(out.dtype)
    key = ("momentum_init", numel, g_dt, b_dt, o_dt, nesterov)
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key,
            tilelang.compile(
                _build_momentum_init(numel, g_dt, b_dt, o_dt, nesterov),
                target="ascend",
                out_idx=None,
            ),
        )
    k(grad.view(-1), buf.view(-1), out.view(-1), mu)


# ---------------------------------------------------------------------------
# Apply: owned = rint( rint(owned*wd) + alpha*rint(update) )  (3 roundings)
# ---------------------------------------------------------------------------


def _build_apply(numel: int, p_dtype: str, use_wd: bool, update_dtype: str = "float32"):
    iters = _iterations(numel)
    tail = _tail(numel)

    @T.prim_func
    def main(
        Owned: T.Tensor((numel,), p_dtype),
        Update: T.Tensor((numel,), update_dtype),
        Out: T.Tensor((numel,), p_dtype),
        wd_factor: T.float32,
        alpha: T.float32,
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            p_ub = T.alloc_shared((_TILE,), p_dtype)
            u_ub = T.alloc_shared((_TILE,), update_dtype)
            o_ub = T.alloc_shared((_TILE,), p_dtype)
            T.annotate_buffer_versions({p_ub: 2, u_ub: 2, o_ub: 1})
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * _NUM_BLOCKS + bx) * _TILE
                rem = T.if_then_else(numel > begin, numel - begin, 0)
                n = T.if_then_else(rem >= _TILE, _TILE, rem)
                T.copy(Owned[begin : begin + n], p_ub[:n])
                T.copy(Update[begin : begin + n], u_ub[:n])
                with T.SimtVF(threads=256):
                    for i in T.Parallel(_TILE):
                        if i < n:
                            if use_wd:
                                t1 = T.cast(T.cast(p_ub[i], "float32") * wd_factor, p_dtype)
                            else:
                                t1 = p_ub[i]
                            u16 = T.cast(u_ub[i], p_dtype)
                            o_ub[i] = T.cast(
                                T.cast(t1, "float32") + alpha * T.cast(u16, "float32"),
                                p_dtype,
                            )
                T.copy(o_ub[:n], Out[begin : begin + n])

    return main


def apply_update(owned: torch.Tensor, update: torch.Tensor, wd_factor, alpha: float):
    """owned ← wd·owned + alpha·cast(update), in place, matching the eager
    three-rounding sequence (mul_ store, .to() cast, add_ store)."""
    assert owned.shape == update.shape
    numel = owned.numel()
    if not numel:
        return
    p_dt = _dtype_name(owned.dtype)
    u_dt = _dtype_name(update.dtype)
    use_wd = wd_factor is not None
    key = ("apply", numel, p_dt, use_wd, u_dt)
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key,
            tilelang.compile(
                _build_apply(numel, p_dt, use_wd, u_dt), target="ascend", out_idx=None
            ),
        )
    k(owned.view(-1), update.view(-1), owned.view(-1), wd_factor if use_wd else 1.0, alpha)


# ---------------------------------------------------------------------------
# Cast+scale (TP pre-scaled delta): out = rint(update * scale)
# ---------------------------------------------------------------------------


def _build_cast_scale(numel: int, dst_dtype: str, src_dtype: str = "float32"):
    iters = _iterations(numel)
    tail = _tail(numel)

    @T.prim_func
    def main(
        Update: T.Tensor((numel,), src_dtype),
        Out: T.Tensor((numel,), dst_dtype),
        scale: T.float32,
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            u_ub = T.alloc_shared((_TILE,), src_dtype)
            o_ub = T.alloc_shared((_TILE,), dst_dtype)
            T.annotate_buffer_versions({u_ub: 2, o_ub: 1})
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * _NUM_BLOCKS + bx) * _TILE
                rem = T.if_then_else(numel > begin, numel - begin, 0)
                n = T.if_then_else(rem >= _TILE, _TILE, rem)
                T.copy(Update[begin : begin + n], u_ub[:n])
                with T.SimtVF(threads=256):
                    for i in T.Parallel(_TILE):
                        if i < n:
                            o_ub[i] = T.cast(u_ub[i] * scale, dst_dtype)
                T.copy(o_ub[:n], Out[begin : begin + n])

    return main


def cast_scale(update: torch.Tensor, out: torch.Tensor, scale: float):
    numel = out.numel()
    if not update.is_contiguous():
        update = update.contiguous()  # layout staging only
    dst_dt = _dtype_name(out.dtype)
    key = ("cast_scale", numel, dst_dt, _dtype_name(update.dtype))
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key,
            tilelang.compile(
                _build_cast_scale(numel, dst_dt, _dtype_name(update.dtype)),
                target="ascend",
                out_idx=None,
            ),
        )
    k(update.view(-1), out.view(-1), scale)


def cast_to_fp32(x: torch.Tensor) -> torch.Tensor:
    """TileLang equivalent of x.float(), with no torch compute fallback."""
    if x.dtype == torch.float32:
        return x if x.is_contiguous() else x.contiguous()
    out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    cast_scale(x, out, 1.0)
    return out


# ---------------------------------------------------------------------------
# Polynomial AXPBY epilogue (separate pass; see kernels/gemm.py docstring):
# out = rint16(alpha*Z + beta*C [+ a*delta_ij])  over an m x m fp16 matrix
# ---------------------------------------------------------------------------


def _build_axpby16(numel: int, dim: int, has_diag: bool):
    iters = _iterations(numel)
    tail = _tail(numel)

    @T.prim_func
    def main(
        Z: T.Tensor((numel,), "float16"),
        C: T.Tensor((numel,), "float16"),
        Out: T.Tensor((numel,), "float16"),
        alpha: T.float32,
        beta: T.float32,
        diag_add: T.float32,
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            z_ub = T.alloc_shared((_TILE,), "float16")
            c_ub = T.alloc_shared((_TILE,), "float16")
            o_ub = T.alloc_shared((_TILE,), "float16")
            T.annotate_buffer_versions({z_ub: 2, c_ub: 2, o_ub: 1})
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * _NUM_BLOCKS + bx) * _TILE
                rem = T.if_then_else(numel > begin, numel - begin, 0)
                n = T.if_then_else(rem >= _TILE, _TILE, rem)
                T.copy(Z[begin : begin + n], z_ub[:n])
                T.copy(C[begin : begin + n], c_ub[:n])
                with T.SimtVF(threads=256):
                    for i in T.Parallel(_TILE):
                        if i < n:
                            z16 = T.cast(
                                alpha * T.cast(z_ub[i], "float32")
                                + beta * T.cast(c_ub[i], "float32"),
                                "float16",
                            )
                            if has_diag:
                                idx = begin + i
                                r = idx // dim
                                c = idx % dim
                                off = T.float32(0.0) * T.if_then_else(
                                    diag_add >= T.float32(0.0), T.float32(1.0), T.float32(-1.0)
                                )
                                d = T.if_then_else(r == c, diag_add, off)
                                o_ub[i] = T.cast(T.cast(z16, "float32") + d, "float16")
                            else:
                                o_ub[i] = z16
                T.copy(o_ub[:n], Out[begin : begin + n])

    return main


def axpby16(
    z: torch.Tensor,
    c: torch.Tensor,
    out: torch.Tensor,
    alpha: float,
    beta: float,
    diag_add: float = 0.0,
    dim: int = 0,
):
    """out = rint16(α·Z + β·C [+ a·δij]) elementwise fp16 (flat m×m)."""
    numel = z.numel()
    has_diag = diag_add != 0.0
    key = ("axpby16", numel, dim if has_diag else 0, has_diag)
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key,
            tilelang.compile(_build_axpby16(numel, dim, has_diag), target="ascend", out_idx=None),
        )
    k(z.view(-1), c.view(-1), out.view(-1), alpha, beta, diag_add if has_diag else 0.0)


# ---------------------------------------------------------------------------
# Normalize stage 1: partial sum of squares per block (fp32)
# ---------------------------------------------------------------------------


def _build_momentum_norm(
    numel: int, g_dt: str, buf_dt: str, out_dt: str, nesterov: bool, init: bool
):
    """momentum + the squared-partial reduction over its own output.

    norm_stage1 used to re-read the whole ns_input right after momentum wrote
    it; both loops walk the same (numel) range with the same block/TILE
    decomposition, so the reduction rides along for free.
    """
    iters = _iterations(numel)

    @T.prim_func
    def main(
        G: T.Tensor((numel,), g_dt),
        Buf: T.Tensor((numel,), buf_dt),
        Out: T.Tensor((numel,), out_dt),
        Partials: T.Tensor((_NUM_BLOCKS,), "float32"),
        mu: T.float32,
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            g_ub = T.alloc_shared((_TILE,), g_dt)
            buf_in_ub = T.alloc_shared((_TILE,), buf_dt)
            buf_out_ub = T.alloc_shared((_TILE,), buf_dt)
            out_ub = T.alloc_shared((_TILE,), out_dt)
            step_sums = T.alloc_shared((iters,), "float32")
            T.annotate_buffer_versions({g_ub: 2, buf_in_ub: 2, buf_out_ub: 1, out_ub: 1})
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * _NUM_BLOCKS + bx) * _TILE
                rem = T.if_then_else(numel > begin, numel - begin, 0)
                n = T.if_then_else(rem >= _TILE, _TILE, rem)
                T.copy(G[begin : begin + n], g_ub[:n])
                if not init:
                    T.copy(Buf[begin : begin + n], buf_in_ub[:n])
                with T.SimtVF(threads=256):
                    for i in T.Parallel(_TILE):
                        if i < n:
                            g = T.cast(g_ub[i], "float32")
                            if init:
                                # First step: buf = cast(g); out = g + mu*buf.
                                new_buf = T.cast(g, buf_dt)
                            else:
                                t = T.cast(T.cast(buf_in_ub[i], "float32") * mu, buf_dt)
                                new_buf = T.cast(T.cast(t, "float32") + g, buf_dt)
                            buf_out_ub[i] = new_buf
                            if nesterov:
                                out_ub[i] = T.cast(g + mu * T.cast(new_buf, "float32"), out_dt)
                            else:
                                out_ub[i] = T.cast(new_buf, out_dt)
                T.copy(buf_out_ub[:n], Buf[begin : begin + n])
                T.copy(out_ub[:n], Out[begin : begin + n])
                with T.SimtVF(threads=256):
                    red = T.alloc_reducer((1,), "float32", op="sum")
                    T.reducer_init(red)
                    for i in T.Parallel(_TILE):
                        if i < n:
                            v = T.cast(out_ub[i], "float32")
                            T.reducer_update(red[0], v * v)
                    fres = T.alloc_fragment((1,), "float32")
                    T.finalize_reducer(red, fres)
                    step_sums[step] = fres[0]
            with T.SimtVF(threads=64):
                red2 = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(red2)
                for s_i in T.Parallel(iters):
                    T.reducer_update(red2[0], step_sums[s_i])
                fres2 = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(red2, fres2)
                Partials[bx] = fres2[0]

    return main


def momentum_norm(
    grad: torch.Tensor,
    buf: torch.Tensor,
    out: torch.Tensor,
    partials: torch.Tensor,
    mu: float,
    nesterov: bool,
    init: bool = False,
):
    """momentum (+ optional first-step init) fused with norm_stage1."""
    numel = grad.numel()
    g_dt = _dtype_name(grad.dtype)
    b_dt = _dtype_name(buf.dtype)
    o_dt = _dtype_name(out.dtype)
    key = ("momentum_norm", numel, g_dt, b_dt, o_dt, nesterov, init)
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key,
            tilelang.compile(
                _build_momentum_norm(numel, g_dt, b_dt, o_dt, nesterov, init),
                target="ascend",
                out_idx=None,
            ),
        )
    k(grad.view(-1), buf.view(-1), out.view(-1), partials, mu)


def _build_norm_stage1(numel: int):
    iters = _iterations(numel)
    tail = _tail(numel)

    @T.prim_func
    def main(
        X: T.Tensor((numel,), "float32"),
        Partials: T.Tensor((_NUM_BLOCKS,), "float32"),
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            x_ub = T.alloc_shared((_TILE,), "float32")
            step_sums = T.alloc_shared((iters,), "float32")
            T.annotate_buffer_versions({x_ub: 2})
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * _NUM_BLOCKS + bx) * _TILE
                rem = T.if_then_else(numel > begin, numel - begin, 0)
                n = T.if_then_else(rem >= _TILE, _TILE, rem)
                T.copy(X[begin : begin + n], x_ub[:n])
                with T.SimtVF(threads=256):
                    red = T.alloc_reducer((1,), "float32", op="sum")
                    T.reducer_init(red)
                    for i in T.Parallel(_TILE):
                        if i < n:
                            v = x_ub[i]
                            T.reducer_update(red[0], v * v)
                    fres = T.alloc_fragment((1,), "float32")
                    T.finalize_reducer(red, fres)
                    step_sums[step] = fres[0]
            with T.SimtVF(threads=64):
                red2 = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(red2)
                for s_i in T.Parallel(iters):
                    T.reducer_update(red2[0], step_sums[s_i])
                fres2 = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(red2, fres2)
                Partials[bx] = fres2[0]

    return main


def norm_stage1(x: torch.Tensor, partials: torch.Tensor):
    numel = x.numel()
    key = ("norm_stage1", numel, "float32")
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key, tilelang.compile(_build_norm_stage1(numel), target="ascend", out_idx=None)
        )
    # Every block writes Partials[bx] unconditionally after reducing >=1
    # initialized step_sums. Idle blocks reduce the sum identity (zero), so
    # there is no eager fill launch or dependency on prior buffer contents.
    k(x.view(-1), partials)


# ---------------------------------------------------------------------------
# Normalize stage 2: X16 = rint16(X / (sqrt(Σ partials) + eps))  (flat cast)
# ---------------------------------------------------------------------------


def _build_norm_stage2(numel: int):
    iters = _iterations(numel)
    tail = _tail(numel)

    @T.prim_func
    def main(
        X: T.Tensor((numel,), "float32"),
        Partials: T.Tensor((_NUM_BLOCKS,), "float32"),
        X16: T.Tensor((numel,), "float16"),
        eps: T.float32,
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            p_ub = T.alloc_shared((_NUM_BLOCKS,), "float32")
            s_ub = T.alloc_shared((1,), "float32")
            x_ub = T.alloc_shared((_TILE,), "float32")
            o_ub = T.alloc_shared((_TILE,), "float16")
            T.annotate_buffer_versions({x_ub: 2, o_ub: 1})
            T.copy(Partials, p_ub)
            with T.SimtVF(threads=64):
                red = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(red)
                for i in T.Parallel(_NUM_BLOCKS):
                    T.reducer_update(red[0], p_ub[i])
                fres = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(red, fres)
                s_ub[0] = T.sqrt(fres[0]) + eps
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * _NUM_BLOCKS + bx) * _TILE
                rem = T.if_then_else(numel > begin, numel - begin, 0)
                n = T.if_then_else(rem >= _TILE, _TILE, rem)
                T.copy(X[begin : begin + n], x_ub[:n])
                with T.SimtVF(threads=256):
                    for i in T.Parallel(_TILE):
                        if i < n:
                            o_ub[i] = T.cast(x_ub[i] / s_ub[0], "float16")
                T.copy(o_ub[:n], X16[begin : begin + n])

    return main


def norm_stage2(x: torch.Tensor, partials: torch.Tensor, x16: torch.Tensor, eps: float):
    numel = x.numel()
    key = ("norm_stage2", numel, "float32")
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key,
            tilelang.compile(_build_norm_stage2(numel), target="ascend", out_idx=None),
        )
    k(x.view(-1), partials, x16.view(-1), eps)


def _build_sym_select(numel: int, dim: int):
    iters = _iterations(numel)
    tail = _tail(numel)

    @T.prim_func
    def main(
        Lower: T.Tensor((numel,), "float16"),
        Transposed: T.Tensor((numel,), "float16"),
        Out: T.Tensor((numel,), "float16"),
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            l_ub = T.alloc_shared((_TILE,), "float16")
            t_ub = T.alloc_shared((_TILE,), "float16")
            T.annotate_buffer_versions({l_ub: 2, t_ub: 2})
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * _NUM_BLOCKS + bx) * _TILE
                rem = T.if_then_else(numel > begin, numel - begin, 0)
                n = T.if_then_else(rem >= _TILE, _TILE, rem)
                T.copy(Lower[begin : begin + n], l_ub[:n])
                T.copy(Transposed[begin : begin + n], t_ub[:n])
                with T.SimtVF(threads=256):
                    for i in T.Parallel(_TILE):
                        if i < n:
                            idx = begin + i
                            r = idx // dim
                            c = idx % dim
                            Out[idx] = T.if_then_else(r >= c, l_ub[i], t_ub[i])

    return main


def sym_select16(lower: torch.Tensor, transposed: torch.Tensor, out: torch.Tensor):
    """out[i,j] = lower[i,j] when i>=j else transposed[i,j]."""
    dim = lower.shape[0]
    numel = lower.numel()
    key = ("sym_select16", numel, dim)
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key, tilelang.compile(_build_sym_select(numel, dim), target="ascend", out_idx=None)
        )
    k(lower.view(-1), transposed.view(-1), out.view(-1))


def _build_copy_lower16(dim: int, tile: int):
    tiles = dim // tile

    @T.prim_func
    def main(
        Src: T.Tensor((dim, dim), "float16"),
        Dst: T.Tensor((dim, dim), "float16"),
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            in_ub = T.alloc_shared((tile, tile), "float16")
            for tile_idx in T.Persistent([tiles * tiles], _NUM_BLOCKS, bx):
                i_tile = tile_idx // tiles
                j_tile = tile_idx % tiles
                if i_tile >= j_tile:
                    T.copy(
                        Src[
                            i_tile * tile : (i_tile + 1) * tile, j_tile * tile : (j_tile + 1) * tile
                        ],
                        in_ub,
                    )
                    if i_tile == j_tile:
                        out_ub = T.alloc_shared((tile, tile), "float16")
                        T.copy(
                            Dst[
                                i_tile * tile : (i_tile + 1) * tile,
                                j_tile * tile : (j_tile + 1) * tile,
                            ],
                            out_ub,
                        )
                        with T.SimtVF(threads=256):
                            for i, j in T.Parallel(tile, tile):
                                if i >= j:
                                    out_ub[i, j] = in_ub[i, j]
                        T.copy(
                            out_ub,
                            Dst[
                                i_tile * tile : (i_tile + 1) * tile,
                                j_tile * tile : (j_tile + 1) * tile,
                            ],
                        )
                    else:
                        T.copy(
                            in_ub,
                            Dst[
                                i_tile * tile : (i_tile + 1) * tile,
                                j_tile * tile : (j_tile + 1) * tile,
                            ],
                        )

    return main


def copy_lower16(src: torch.Tensor, dst: torch.Tensor):
    """Overwrite only the lower triangle after transpose2d filled the upper."""
    dim = src.shape[0]
    tr, _ = _fit2d(dim, dim)
    tr = min(tr, 64)
    while tr > 16 and dim % tr:
        tr //= 2
    key = ("copy_lower16", dim, tr)
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key, tilelang.compile(_build_copy_lower16(dim, tr), target="ascend", out_idx=None)
        )
    k(src, dst)


# ---------------------------------------------------------------------------
# 2-D lower-triangle mirror for the optional SYRK split path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fused normalize + transpose for aligned matrices
# ---------------------------------------------------------------------------


def norm_stage2_t(
    x: torch.Tensor, partials: torch.Tensor, x16: torch.Tensor, x16_t: torch.Tensor, eps: float
):
    """Fused normalize + transposed store for aligned (rows, cols) matrices."""
    rows, cols = x.shape
    assert rows % 16 == 0 and cols % 16 == 0
    tr, tc = _fit2d(rows, cols)
    key = ("norm_stage2_t", rows, cols)
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key,
            tilelang.compile(
                _build_norm_stage2_t(rows, cols, tr, tc), target="ascend", out_idx=None
            ),
        )
    k(x, partials, x16, x16_t, eps)


def _build_norm_stage2_t(rows: int, cols: int, tr: int, tc: int):
    r_tiles, c_tiles = rows // tr, cols // tc

    @T.prim_func
    def main(
        X: T.Tensor((rows, cols), "float32"),
        Partials: T.Tensor((_NUM_BLOCKS,), "float32"),
        X16: T.Tensor((rows, cols), "float16"),
        X16T: T.Tensor((cols, rows), "float16"),
        eps: T.float32,
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            p_ub = T.alloc_shared((_NUM_BLOCKS,), "float32")
            s_ub = T.alloc_shared((1,), "float32")
            in_ub = T.alloc_shared((tr, tc), "float32")
            x_ub = T.alloc_shared((tr, tc), "float16")
            xt_ub = T.alloc_shared((tc, tr), "float16")
            T.copy(Partials, p_ub)
            with T.SimtVF(threads=64):
                red = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(red)
                for i in T.Parallel(_NUM_BLOCKS):
                    T.reducer_update(red[0], p_ub[i])
                fres = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(red, fres)
                s_ub[0] = T.sqrt(fres[0]) + eps
            for tile_idx in T.Persistent([r_tiles * c_tiles], _NUM_BLOCKS, bx):
                r0 = (tile_idx // c_tiles) * tr
                c0 = (tile_idx % c_tiles) * tc
                T.copy(X[r0 : r0 + tr, c0 : c0 + tc], in_ub)
                with T.SimtVF(threads=256):
                    x_frag = T.alloc_fragment((tr, tc), "float16")
                    for i, j in T.Parallel(tr, tc):
                        x_frag[i, j] = T.cast(in_ub[i, j] / s_ub[0], "float16")
                    for i, j in T.Parallel(tr, tc):
                        x_ub[i, j] = x_frag[i, j]
                        xt_ub[j, i] = x_frag[i, j]
                T.copy(x_ub, X16[r0 : r0 + tr, c0 : c0 + tc])
                T.copy(xt_ub, X16T[c0 : c0 + tc, r0 : r0 + tr])

    return main


# ---------------------------------------------------------------------------
# 2-D tiled transpose kernels (UB-internal transpose, block stores)
# ---------------------------------------------------------------------------


def _build_transpose2d(rows: int, cols: int, tr: int, tc: int, in_dt: str, out_dt: str):
    r_tiles = rows // tr
    c_tiles = cols // tc

    @T.prim_func
    def main(
        X: T.Tensor((rows, cols), in_dt),
        XT: T.Tensor((cols, rows), out_dt),
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            in_ub = T.alloc_shared((tr, tc), in_dt)
            out_ub = T.alloc_shared((tc, tr), out_dt)
            T.annotate_buffer_versions({in_ub: 2, out_ub: 1})
            for tile_idx in T.Persistent([r_tiles * c_tiles], _NUM_BLOCKS, bx):
                r0 = (tile_idx // c_tiles) * tr
                c0 = (tile_idx % c_tiles) * tc
                T.copy(X[r0 : r0 + tr, c0 : c0 + tc], in_ub)
                with T.SimtVF(threads=256):
                    # Fragment-mediated transpose: strided UB access in the
                    # naive form scalarizes (0.08 TB/s); staging through
                    # registers vectorizes it (1.6 TB/s, 256x16 tiles).
                    x_frag = T.alloc_fragment((tr, tc), in_dt)
                    for i, j in T.Parallel(tr, tc):
                        x_frag[i, j] = in_ub[i, j]
                    for i, j in T.Parallel(tr, tc):
                        out_ub[j, i] = T.cast(x_frag[i, j], out_dt)
                T.copy(out_ub, XT[c0 : c0 + tc, r0 : r0 + tr])

    return main


def _fit2d(rows: int, cols: int):
    def fit(v, cap):
        t = cap
        while t > 16 and v % t != 0:
            t //= 2
        return t

    return fit(rows, 256), fit(cols, 16)


def transpose2d(x: torch.Tensor, xt: torch.Tensor):
    """xt (cols,rows) ← transpose of x (rows,cols), same or cast dtype.
    Misaligned dims are zero-padded via staging (layout only)."""
    rows, cols = x.shape
    if rows % 16 or cols % 16:
        rp, cp = (rows + 15) // 16 * 16, (cols + 15) // 16 * 16
        xp = torch.zeros((rp, cp), device=x.device, dtype=x.dtype)
        xp[:rows, :cols] = x
        xtp = torch.zeros((cp, rp), device=x.device, dtype=xt.dtype)
        _transpose2d_aligned(xp, xtp)
        xt.copy_(xtp[:cols, :rows])
        return
    _transpose2d_aligned(x, xt)


def _transpose2d_aligned(x: torch.Tensor, xt: torch.Tensor):
    rows, cols = x.shape
    tr, tc = _fit2d(rows, cols)
    key = ("transpose2d", rows, cols, _dtype_name(x.dtype), _dtype_name(xt.dtype))
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key,
            tilelang.compile(
                _build_transpose2d(rows, cols, tr, tc, _dtype_name(x.dtype), _dtype_name(xt.dtype)),
                target="ascend",
                out_idx=None,
            ),
        )
    k(x, xt)


def cast_transposed(src: torch.Tensor, dst: torch.Tensor):
    """dst (N,M) any dtype ← cast(src (M,N) fp32), dst[i,j] = cast(src[j,i])."""
    transpose2d(src, dst)


# ---------------------------------------------------------------------------
# Polynomial AXPBY on the raw fp32 GEMM accumulator:
# out = rint16(alpha*Acc32 + beta*C16 [+ a*delta_ij])  — single rounding,
# exactly the eager addmm storage contract.
# ---------------------------------------------------------------------------


def _build_axpby32(numel: int, dim: int, has_diag: bool):
    iters = _iterations(numel)
    tail = _tail(numel)

    @T.prim_func
    def main(
        Acc: T.Tensor((numel,), "float32"),
        C: T.Tensor((numel,), "float16"),
        Out: T.Tensor((numel,), "float16"),
        alpha: T.float32,
        beta: T.float32,
        diag_add: T.float32,
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            a_ub = T.alloc_shared((_TILE,), "float32")
            c_ub = T.alloc_shared((_TILE,), "float16")
            o_ub = T.alloc_shared((_TILE,), "float16")
            T.annotate_buffer_versions({a_ub: 2, c_ub: 2, o_ub: 1})
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * _NUM_BLOCKS + bx) * _TILE
                rem = T.if_then_else(numel > begin, numel - begin, 0)
                n = T.if_then_else(rem >= _TILE, _TILE, rem)
                T.copy(Acc[begin : begin + n], a_ub[:n])
                T.copy(C[begin : begin + n], c_ub[:n])
                with T.SimtVF(threads=256):
                    for i in T.Parallel(_TILE):
                        if i < n:
                            z16 = T.cast(
                                alpha * a_ub[i] + beta * T.cast(c_ub[i], "float32"),
                                "float16",
                            )
                            if has_diag:
                                idx = begin + i
                                r = idx // dim
                                c = idx % dim
                                off = T.float32(0.0) * T.if_then_else(
                                    diag_add >= T.float32(0.0), T.float32(1.0), T.float32(-1.0)
                                )
                                d = T.if_then_else(r == c, diag_add, off)
                                o_ub[i] = T.cast(T.cast(z16, "float32") + d, "float16")
                            else:
                                o_ub[i] = z16
                T.copy(o_ub[:n], Out[begin : begin + n])

    return main


def axpby32(
    acc32: torch.Tensor,
    c: torch.Tensor,
    out: torch.Tensor,
    alpha: float,
    beta: float,
    diag_add: float = 0.0,
    dim: int = 0,
):
    """out = rint16(a*Acc32 + b*C16 [+ d*delta_ij]); acc32 is raw GEMM output."""
    numel = acc32.numel()
    assert c.dtype == torch.float16 and out.dtype == torch.float16
    has_diag = diag_add != 0.0
    key = ("axpby32", numel, dim if has_diag else 0, has_diag)
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key,
            tilelang.compile(_build_axpby32(numel, dim, has_diag), target="ascend", out_idx=None),
        )
    k(acc32.view(-1), c.view(-1), out.view(-1), alpha, beta, diag_add if has_diag else 0.0)


# ---------------------------------------------------------------------------
# Lower-triangle mirror: out[i,j] = in[i,j] if i>=j else in[j,i]
# ---------------------------------------------------------------------------


def _build_mirror16(numel: int, dim: int):
    iters = _iterations(numel)
    tail = _tail(numel)

    @T.prim_func
    def main(
        Src: T.Tensor((numel,), "float16"),
        Dst: T.Tensor((numel,), "float16"),
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            s_ub = T.alloc_shared((_TILE,), "float16")
            T.annotate_buffer_versions({s_ub: 2})
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * _NUM_BLOCKS + bx) * _TILE
                rem = T.if_then_else(numel > begin, numel - begin, 0)
                n = T.if_then_else(rem >= _TILE, _TILE, rem)
                T.copy(Src[begin : begin + n], s_ub[:n])
                with T.SimtVF(threads=256):
                    for i in T.Parallel(_TILE):
                        if i < n:
                            idx = begin + i
                            r = idx // dim
                            c = idx % dim
                            if c > r:
                                t = c * dim + r
                                tr = t // dim
                                tc = t % dim
                                Dst[c * dim + r] = s_ub[i]
                            else:
                                Dst[idx] = s_ub[i]

    return main


def mirror_lower16(src: torch.Tensor, dst: torch.Tensor):
    """dst = symmetrize(lower(src)): upper triangle filled from lower."""
    assert src.shape == dst.shape and src.dtype == torch.float16
    dim = src.shape[0]
    numel = src.numel()
    key = ("mirror16", numel, dim)
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key, tilelang.compile(_build_mirror16(numel, dim), target="ascend", out_idx=None)
        )
    k(src.view(-1), dst.view(-1))


# ---------------------------------------------------------------------------
# Q = Z + a*I in fp16 arithmetic (the eager second op after the addmm):
# out[i,j] = fp16_add(z[i,j], a) on the diagonal, +copysign(0,a) elsewhere.
# ---------------------------------------------------------------------------


def _build_diag_add16(numel: int, dim: int):
    iters = _iterations(numel)
    tail = _tail(numel)

    @T.prim_func
    def main(
        Z: T.Tensor((numel,), "float16"),
        Out: T.Tensor((numel,), "float16"),
        diag_add: T.float32,
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            z_ub = T.alloc_shared((_TILE,), "float16")
            o_ub = T.alloc_shared((_TILE,), "float16")
            T.annotate_buffer_versions({z_ub: 2, o_ub: 1})
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * _NUM_BLOCKS + bx) * _TILE
                rem = T.if_then_else(numel > begin, numel - begin, 0)
                n = T.if_then_else(rem >= _TILE, _TILE, rem)
                T.copy(Z[begin : begin + n], z_ub[:n])
                with T.SimtVF(threads=256):
                    for i in T.Parallel(_TILE):
                        if i < n:
                            idx = begin + i
                            r = idx // dim
                            c = idx % dim
                            off = T.float32(0.0) * T.if_then_else(
                                diag_add >= T.float32(0.0), T.float32(1.0), T.float32(-1.0)
                            )
                            d = T.if_then_else(r == c, diag_add, off)
                            o_ub[i] = T.cast(T.cast(z_ub[i], "float32") + d, "float16")
                T.copy(o_ub[:n], Out[begin : begin + n])

    return main


def diag_add16(z: torch.Tensor, out: torch.Tensor, diag_add: float):
    """out = Z + a·I (fp16 add semantics; the eager ``Z + a * eye`` op)."""
    numel = z.numel()
    dim = z.shape[0]
    key = ("diag_add16", numel, dim)
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key, tilelang.compile(_build_diag_add16(numel, dim), target="ascend", out_idx=None)
        )
    k(z.view(-1), out.view(-1), diag_add)
