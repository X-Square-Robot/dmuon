"""Lazy TileLang kernels for same-shape batched Gram Newton-Schulz."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch
from torch import Tensor

from dmuon.optim.newton_schulz import (
    DEFAULT_COEFFICIENTS,
    DEFAULT_RESTART_ITERATIONS,
)


_BEST_SYRK = {
    (1024, 1024): (64, 2, 128),
    (1024, 2560): (32, 2, 128),
    (1024, 4096): (32, 2, 128),
    (1024, 2048): (64, 2, 128),
    (1280, 1280): (64, 3, 128),
}


def _tilelang_modules():
    import tilelang
    import tilelang.language as T

    return tilelang, T


def _syrk_prim(B: int, M: int, K: int, bM: int, bK: int, stages: int, threads: int, poly: bool):
    _tilelang, T = _tilelang_modules()
    nb = (M + bM - 1) // bM
    nblk = nb * (nb + 1) // 2

    @T.prim_func
    def syrk(
        X: T.Tensor((B, M, K), "float16"),
        Out: T.Tensor((B, M, M), "float16"),
    ):
        with T.Kernel(nblk, B, threads=threads) as (blk, bz):
            by = T.alloc_local((1,), "int32")
            bx = T.alloc_local((1,), "int32")
            rem = T.alloc_local((1,), "int32")
            by[0] = 0
            rem[0] = blk
            for r in T.serial(nb):
                cond = rem[0] >= (r + 1)
                rem[0] = T.if_then_else(cond, rem[0] - (r + 1), rem[0])
                by[0] = T.if_then_else(cond, by[0] + 1, by[0])
            bx[0] = rem[0]
            As = T.alloc_shared((bM, bK), "float16")
            Bs = T.alloc_shared((bM, bK), "float16")
            Cl = T.alloc_fragment((bM, bM), "float")
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=stages):
                T.copy(X[bz, by[0] * bM, k * bK], As)
                T.copy(X[bz, bx[0] * bM, k * bK], Bs)
                T.gemm(As, Bs, Cl, transpose_B=True)
            Co = T.alloc_shared((bM, bM), "float16")
            T.copy(Cl, Co)
            T.copy(Co, Out[bz, by[0] * bM, bx[0] * bM])
            for i, j in T.Parallel(bM, bM):
                Out[bz, bx[0] * bM + i, by[0] * bM + j] = Co[j, i]

    @T.prim_func
    def syrk_poly(
        R: T.Tensor((B, M, K), "float16"),
        Out: T.Tensor((B, M, M), "float16"),
        alpha: T.float32,
        beta: T.float32,
    ):
        with T.Kernel(nblk, B, threads=threads) as (blk, bz):
            by = T.alloc_local((1,), "int32")
            bx = T.alloc_local((1,), "int32")
            rem = T.alloc_local((1,), "int32")
            by[0] = 0
            rem[0] = blk
            for r in T.serial(nb):
                cond = rem[0] >= (r + 1)
                rem[0] = T.if_then_else(cond, rem[0] - (r + 1), rem[0])
                by[0] = T.if_then_else(cond, by[0] + 1, by[0])
            bx[0] = rem[0]
            As = T.alloc_shared((bM, bK), "float16")
            Bs = T.alloc_shared((bM, bK), "float16")
            Cl = T.alloc_fragment((bM, bM), "float")
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=stages):
                T.copy(R[bz, by[0] * bM, k * bK], As)
                T.copy(R[bz, bx[0] * bM, k * bK], Bs)
                T.gemm(As, Bs, Cl, transpose_B=True)
            Db = T.alloc_shared((bM, bM), "float16")
            T.copy(R[bz, by[0] * bM, bx[0] * bM], Db)
            Co = T.alloc_shared((bM, bM), "float16")
            for i, j in T.Parallel(bM, bM):
                Co[i, j] = T.Cast(
                    "float16",
                    alpha * Cl[i, j] + beta * T.Cast("float", Db[i, j]),
                )
            T.copy(Co, Out[bz, by[0] * bM, bx[0] * bM])
            for i, j in T.Parallel(bM, bM):
                Out[bz, bx[0] * bM + i, by[0] * bM + j] = Co[j, i]

    return syrk_poly if poly else syrk


def _gemm_prim(
    B: int,
    M: int,
    N: int,
    K: int,
    bM: int,
    bN: int,
    bK: int,
    stages: int,
    threads: int,
    epi: bool,
):
    _tilelang, T = _tilelang_modules()

    @T.prim_func
    def gemm_epi(
        A: T.Tensor((B, M, K), "float16"),
        Bm: T.Tensor((B, K, N), "float16"),
        D: T.Tensor((B, M, N), "float16"),
        Out: T.Tensor((B, M, N), "float16"),
        alpha: T.float32,
        beta: T.float32,
    ):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), B, threads=threads) as (
            bx,
            by,
            bz,
        ):
            As = T.alloc_shared((bM, bK), "float16")
            Bs = T.alloc_shared((bK, bN), "float16")
            Cl = T.alloc_fragment((bM, bN), "float")
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=stages):
                T.copy(A[bz, by * bM, k * bK], As)
                T.copy(Bm[bz, k * bK, bx * bN], Bs)
                T.gemm(As, Bs, Cl)
            for i, j in T.Parallel(bM, bN):
                Out[bz, by * bM + i, bx * bN + j] = T.Cast(
                    "float16",
                    alpha * Cl[i, j] + beta * T.Cast("float", D[bz, by * bM + i, bx * bN + j]),
                )

    @T.prim_func
    def gemm_plain(
        A: T.Tensor((B, M, K), "float16"),
        Bm: T.Tensor((B, K, N), "float16"),
        Out: T.Tensor((B, M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), B, threads=threads) as (
            bx,
            by,
            bz,
        ):
            As = T.alloc_shared((bM, bK), "float16")
            Bs = T.alloc_shared((bK, bN), "float16")
            Cl = T.alloc_fragment((bM, bN), "float")
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=stages):
                T.copy(A[bz, by * bM, k * bK], As)
                T.copy(Bm[bz, k * bK, bx * bN], Bs)
                T.gemm(As, Bs, Cl)
            for i, j in T.Parallel(bM, bN):
                Out[bz, by * bM + i, bx * bN + j] = T.Cast("float16", Cl[i, j])

    return gemm_epi if epi else gemm_plain


@lru_cache(maxsize=None)
def _get_syrk_kernel(B: int, M: int, K: int, poly: bool):
    tilelang, _T = _tilelang_modules()
    bM = 128 if M % 128 == 0 else 64
    bK, stages, threads = _BEST_SYRK.get((M, K), (32, 2, 128))
    if K % bK:
        bK = 32
    return tilelang.compile(
        _syrk_prim(B, M, K, bM, bK, stages, threads, poly),
        out_idx=[1],
    )


@lru_cache(maxsize=None)
def _get_gemm_kernel(B: int, M: int, N: int, K: int, epi: bool):
    tilelang, _T = _tilelang_modules()
    bM = 128 if M % 128 == 0 else 64
    bN = 128 if N % 128 == 0 else 64
    out_idx = [3] if epi else [2]
    return tilelang.compile(
        _gemm_prim(B, M, N, K, bM, bN, 32, 3, 128, epi),
        out_idx=out_idx,
    )


def tl_syrk(x: Tensor) -> Tensor:
    batch, m, k = x.shape
    return _get_syrk_kernel(batch, m, k, False)(x.half().contiguous())


def tl_syrk_poly(r: Tensor, alpha: float, beta: float) -> Tensor:
    batch, m, k = r.shape
    return _get_syrk_kernel(batch, m, k, True)(
        r.half().contiguous(),
        float(alpha),
        float(beta),
    )


def tl_gemm_epi(a: Tensor, b: Tensor, d: Tensor, alpha: float, beta: float) -> Tensor:
    batch, m, k = a.shape
    n = b.shape[2]
    return _get_gemm_kernel(batch, m, n, k, True)(
        a.half().contiguous(),
        b.half().contiguous(),
        d.half().contiguous(),
        float(alpha),
        float(beta),
    )


def tl_gemm(a: Tensor, b: Tensor) -> Tensor:
    batch, m, k = a.shape
    n = b.shape[2]
    return _get_gemm_kernel(batch, m, n, k, False)(
        a.half().contiguous(),
        b.half().contiguous(),
    )


def tilelang_batched_gram_newton_schulz(
    g: Tensor,
    *,
    eps: float = 1e-7,
    coefficients: Optional[list[list[float]]] = None,
    restart_iterations: Optional[list[int]] = None,
) -> Tensor:
    """TileLang same-shape batched Gram Newton-Schulz."""

    if coefficients is None:
        coefficients = DEFAULT_COEFFICIENTS
    if restart_iterations is None:
        restart_iterations = DEFAULT_RESTART_ITERATIONS
    if coefficients != DEFAULT_COEFFICIENTS or restart_iterations != DEFAULT_RESTART_ITERATIONS:
        raise ValueError(
            "TileLang batched Gram NS currently supports DMuon's default "
            "coefficients/restart schedule only"
        )
    if g.dim() != 3:
        raise ValueError(f"expected [batch, rows, cols], got {tuple(g.shape)}")

    original_dtype = g.dtype
    x = g.float()
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    normalizer = x.flatten(1).norm(dim=1).add_(eps).view(-1, 1, 1)
    x = (x / normalizer).half().contiguous()

    r = tl_syrk(x)
    eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype).unsqueeze(0)
    eye = eye.expand(x.shape[0], x.shape[1], x.shape[1])
    q = None
    restart_set = set(restart_iterations)
    last = len(coefficients) - 1
    for idx, (a_coeff, b_coeff, c_coeff) in enumerate(coefficients):
        if idx in restart_set and idx != 0:
            if q is None:
                raise RuntimeError("restart reached before Q was initialized")
            x = tl_gemm_epi(q, x, x, 1.0, 0.0)
            r = tl_syrk(x)
            q = None

        z = tl_syrk_poly(r, c_coeff, b_coeff)
        if q is None:
            q = z + a_coeff * eye
        else:
            q = tl_gemm_epi(z, q, q, 1.0, a_coeff)
        if idx < last and (idx + 1) not in restart_set:
            rz = tl_gemm_epi(r, z, r, 1.0, a_coeff)
            r = tl_gemm_epi(z, rz, rz, 1.0, a_coeff)

    if q is None:
        raise RuntimeError("Q was not initialized")
    x = tl_gemm_epi(q, x, x, 1.0, 0.0)
    if transposed:
        x = x.transpose(-2, -1)
    return x.to(original_dtype)
