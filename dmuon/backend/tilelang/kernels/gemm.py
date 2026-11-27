"""Cube (MAD) kernels: Gram SYRK and NT-GEMM (ascend L1 MAD is NT-only).

Validated dav-950 facts this module is built on:

* ``T.gemm`` supports only trans_A=False, trans_B=True (both operands
  K-major): ``out = A @ Bt.T``.
* L0C (fp32) leaves through the *official* doors only: ``T.copy`` straight
  to an fp32 GM tile (this module), or the packed ``T.dual_copy`` pair.
  Custom L0C→UB hops + VF epilogues produced mis-scheduled code (Z3 shared
  -scope budget / buffer versioning), so every kernel here stores fp32 and
  all fp16 conversion / AXPBY happens in separate AIV passes
  (``elementwise.axpby32``) — which also gives the polynomial steps the
  eager single-rounding contract (α·acc32 + β·C16 → one RINT).
* The ascend copy lowering rejects symbolic tail extents, so tiles exactly
  divide the dims; misaligned shapes are zero-padded by the wrapper
  (layout staging; mainstream training shapes are 256-multiples).
"""

import torch
import tilelang
import tilelang.ascend.language as T

from ..dispatch import cached_kernel, fit_tiles, store_kernel


def _tiles(num: int, tile: int) -> int:
    return num // tile


# ---------------------------------------------------------------------------
# Gram: R32 = X @ X^T (fp32 out) via lower-triangle + mirror write
# ---------------------------------------------------------------------------


def _build_gram(m: int, k: int, tile_m: int, tile_k: int, num_blocks: int, num_stages: int):
    m_tiles = _tiles(m, tile_m)
    k_tiles = _tiles(k, tile_k)

    @T.prim_func
    def main(
        X: T.Tensor((m, k), "float16"),
        R: T.Tensor((m, m), "float32"),
    ):
        with T.Kernel(num_blocks) as bx:
            res = T.alloc_l0c((tile_m, tile_m), "float32")
            x1_l1 = T.alloc_l1((tile_m, tile_k), "float16")
            x2_l1 = T.alloc_l1((tile_m, tile_k), "float16")
            # Full grid, one store per tile (the mirror-store variant races
            # with accumulator reuse across Persistent work items and the
            # VF-scatter mirror cannot block-store; revisit both in the
            # generic-scheduling phase).  2x MAD on the Gram step only.
            for tile_idx in T.Persistent([m_tiles * m_tiles], num_blocks, bx):
                i_tile = tile_idx // m_tiles
                j_tile = tile_idx % m_tiles
                m0 = i_tile * tile_m
                n0 = j_tile * tile_m
                for kt in T.Pipelined(k_tiles, num_stages=num_stages):
                    T.copy(X[m0 : m0 + tile_m, kt * tile_k : (kt + 1) * tile_k], x1_l1)
                    T.copy(X[n0 : n0 + tile_m, kt * tile_k : (kt + 1) * tile_k], x2_l1)
                    T.gemm(x1_l1, x2_l1, res, transpose_B=True, clear_accum=kt == 0)
                T.copy(res, R[m0 : m0 + tile_m, n0 : n0 + tile_m])

    return main


def gram_nt(x: torch.Tensor, r32: torch.Tensor):
    """r32 (fp32) = X @ Xᵀ; DeepGEMM handles arbitrary extents natively."""
    from .deep_wrapper import gram_fp32

    gram_fp32(x, r32)


# ---------------------------------------------------------------------------
# NT GEMM: out = A @ Bt^T (fp32 out only)
# ---------------------------------------------------------------------------


def _build_gemm_nt(
    m: int, n: int, k: int, tile_m: int, tile_n: int, tile_k: int, num_blocks: int, num_stages: int
):
    m_tiles = _tiles(m, tile_m)
    n_tiles = _tiles(n, tile_n)
    k_tiles = _tiles(k, tile_k)
    out_tiles = m_tiles * n_tiles

    @T.prim_func
    def main(
        A: T.Tensor((m, k), "float16"),
        Bt: T.Tensor((n, k), "float16"),
        Out: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(num_blocks) as bx:
            res = T.alloc_l0c((tile_m, tile_n), "float32")
            a_l1 = T.alloc_l1((tile_m, tile_k), "float16")
            b_l1 = T.alloc_l1((tile_n, tile_k), "float16")
            for tile_idx in T.Persistent([out_tiles], num_blocks, bx):
                m_tile = tile_idx // n_tiles
                n_tile = tile_idx % n_tiles
                m0 = m_tile * tile_m
                n0 = n_tile * tile_n
                for kt in T.Pipelined(k_tiles, num_stages=num_stages):
                    T.copy(A[m0 : m0 + tile_m, kt * tile_k : (kt + 1) * tile_k], a_l1)
                    T.copy(Bt[n0 : n0 + tile_n, kt * tile_k : (kt + 1) * tile_k], b_l1)
                    T.gemm(a_l1, b_l1, res, transpose_B=True, clear_accum=kt == 0)
                T.copy(res, Out[m0 : m0 + tile_m, n0 : n0 + tile_n])

    return main


def _gemm_nt_aligned(a: torch.Tensor, bt: torch.Tensor, out: torch.Tensor):
    m, k = a.shape
    n = bt.shape[0]
    assert out.dtype == torch.float32
    cfg = fit_tiles(m, n, k)
    key = ("gemm_nt32", m, n, k, tuple(sorted(cfg.items())))
    kern = cached_kernel(key)
    if kern is None:
        kern = store_kernel(
            key,
            tilelang.compile(
                _build_gemm_nt(
                    m,
                    n,
                    k,
                    cfg["tile_m"],
                    cfg["tile_n"],
                    cfg["tile_k"],
                    cfg["num_blocks"],
                    cfg["num_stages"],
                ),
                target="ascend",
                out_idx=None,
            ),
        )
    kern(a, bt, out)


def gemm_nt(a: torch.Tensor, bt: torch.Tensor, out: torch.Tensor):
    """out (fp32) = A @ Bt^T, with native arbitrary-shape DeepGEMM tiles."""
    from .deep_wrapper import gemm_fp32

    gemm_fp32(a, bt, out)
