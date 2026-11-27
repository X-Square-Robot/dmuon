"""Fused AdamW update with scalar coefficients passed at launch."""

import tilelang
import tilelang.ascend.language as T
from .dispatch import cached_kernel, store_kernel
from .kernels.elementwise import _NUM_BLOCKS, _dtype_name

_TILE = 2048


def _build_adamw_scalar(numel: int, p_dt: str, g_dt: str, ea_dt: str, eas_dt: str, use_wd: bool):
    iters = (numel + _NUM_BLOCKS * _TILE - 1) // (_NUM_BLOCKS * _TILE)
    tail = numel - (iters - 1) * _NUM_BLOCKS * _TILE

    @T.prim_func
    def main(
        P: T.Tensor((numel,), p_dt),
        G: T.Tensor((numel,), g_dt),
        EA: T.Tensor((numel,), ea_dt),
        EAS: T.Tensor((numel,), eas_dt),
        wd_factor: T.float32,
        b1: T.float32,
        one_m_b1: T.float32,
        b2: T.float32,
        one_m_b2: T.float32,
        bc2_sqrt: T.float32,
        eps: T.float32,
        step_size: T.float32,
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            p_ub = T.alloc_shared((_TILE,), p_dt)
            g_ub = T.alloc_shared((_TILE,), g_dt)
            ea_ub = T.alloc_shared((_TILE,), ea_dt)
            eas_ub = T.alloc_shared((_TILE,), eas_dt)
            T.annotate_buffer_versions({p_ub: 2, g_ub: 2, ea_ub: 2, eas_ub: 2})
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * _NUM_BLOCKS + bx) * _TILE
                rem = T.if_then_else(numel > begin, numel - begin, 0)
                n = T.if_then_else(rem >= _TILE, _TILE, rem)
                T.copy(P[begin : begin + n], p_ub[:n])
                T.copy(G[begin : begin + n], g_ub[:n])
                T.copy(EA[begin : begin + n], ea_ub[:n])
                T.copy(EAS[begin : begin + n], eas_ub[:n])
                with T.SimtVF(threads=256):
                    for i in T.Parallel(_TILE):
                        if i < n:
                            g = T.cast(g_ub[i], "float32")
                            p = T.cast(p_ub[i], "float32")
                            ea = T.cast(ea_ub[i], "float32")
                            eas = T.cast(eas_ub[i], "float32")
                            if use_wd:
                                p = T.cast(p * wd_factor, p_dt)
                                p = T.cast(p, "float32")
                            ea1 = T.cast(ea * b1, ea_dt)
                            ea2 = T.cast(T.cast(ea1, "float32") + one_m_b1 * g, ea_dt)
                            eas1 = T.cast(eas * b2, eas_dt)
                            # foreach_addcmul_ multiplies in the accumulation dtype.
                            g2 = g * g
                            eas2 = T.cast(
                                T.cast(eas1, "float32") + one_m_b2 * g2,
                                eas_dt,
                            )
                            d1 = T.cast(T.sqrt(T.cast(eas2, "float32")), eas_dt)
                            d2 = T.cast(T.cast(d1, "float32") / bc2_sqrt, eas_dt)
                            d3 = T.cast(T.cast(d2, "float32") + eps, eas_dt)
                            # addcdiv storage boundary: the quotient rounds
                            # to the state dtype before the scaled add.
                            ea2f = T.cast(ea2, "float32")
                            upd = T.cast(ea2f / T.cast(d3, "float32"), eas_dt)
                            p_new = T.cast(p - step_size * T.cast(upd, "float32"), p_dt)
                            p_ub[i] = p_new
                            ea_ub[i] = ea2
                            eas_ub[i] = eas2
                # Write state back (P written by second store below).
                with T.SimtVF(threads=256):
                    for i in T.Parallel(_TILE):
                        if i < n:
                            P[begin + i] = p_ub[i]
                T.copy(ea_ub[:n], EA[begin : begin + n])
                T.copy(eas_ub[:n], EAS[begin : begin + n])

    return main


def adamw_one_scalar(
    param, grad, exp_avg, exp_avg_sq, *, wd_factor, beta1, beta2, bc2_sqrt, eps, step_size
):
    numel = param.numel()
    if not numel:
        return
    dims = (
        numel,
        _dtype_name(param.dtype),
        _dtype_name(grad.dtype),
        _dtype_name(exp_avg.dtype),
        _dtype_name(exp_avg_sq.dtype),
        wd_factor is not None,
    )
    key = ("adamw_scalar_coeff_v1",) + dims
    kernel = cached_kernel(key)
    if kernel is None:
        kernel = store_kernel(
            key, tilelang.compile(_build_adamw_scalar(*dims), target="ascend", out_idx=None)
        )
    kernel(
        param.view(-1),
        grad.view(-1),
        exp_avg.view(-1),
        exp_avg_sq.view(-1),
        wd_factor if wd_factor is not None else 1.0,
        beta1,
        1.0 - beta1,
        beta2,
        1.0 - beta2,
        bc2_sqrt,
        eps,
        step_size,
    )
