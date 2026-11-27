"""Ownership-safe, in-place lower-to-upper mirror using TileLang only."""

import torch
import tilelang
import tilelang.ascend.language as T
from ..dispatch import cached_kernel, store_kernel
from .deepgemm.triangular import triangular_coord


def _build_mirror(dim, block_size=0, tile=32, with_q=False):
    side = (dim + tile - 1) // tile
    local_side = block_size // tile if block_size else side
    per_block = local_side * (local_side + 1) // 2
    count = ((dim + block_size - 1) // block_size) * per_block if block_size else per_block
    cores = min(64, count)

    @T.prim_func
    def main(
        A: T.Tensor((dim, dim), "float16"), Q: T.Tensor((dim, dim), "float16"), diag_add: T.float32
    ):
        with T.Kernel(cores) as bx:
            src = T.alloc_shared((tile, tile), "float16")
            dst = T.alloc_shared((tile, tile), "float16")
            dst_q = T.alloc_shared((tile, tile), "float16") if with_q else None
            T.annotate_buffer_versions({src: 2, dst: 2, **({dst_q: 2} if with_q else {})})
            for index in T.Persistent([count], cores, bx):
                T.assume_no_conflict(A, cross=True)
                local_row, local_col = triangular_coord(index % per_block, local_side)
                group_offset = (index // per_block) * local_side if block_size else 0
                row, col = group_offset + local_row, group_offset + local_col
                r, c = row * tile, col * tile
                nr, nc = T.min(tile, dim - r), T.min(tile, dim - c)
                if r < dim and c < dim:
                    T.copy(A[r : r + nr, c : c + nc], src[:nr, :nc])
                    with T.SimtVF(threads=256):
                        fragment = T.alloc_fragment((tile, tile), "float16")
                        for i, j in T.Parallel(tile, tile):
                            if i < nr and j < nc:
                                fragment[i, j] = src[i, j]
                        for i, j in T.Parallel(tile, tile):
                            if i < nr and j < nc:
                                value = T.if_then_else(
                                    row != col or i >= j, fragment[i, j], src[j, i]
                                )
                                dst[j, i] = value
                                if with_q:
                                    if r + i == c + j:
                                        dst_q[j, i] = T.cast(
                                            T.cast(value, "float32") + diag_add, "float16"
                                        )
                                    else:
                                        dst_q[j, i] = value
                    T.copy(dst[:nc, :nr], A[c : c + nc, r : r + nr])
                    if with_q:
                        T.copy(dst_q[:nc, :nr], Q[c : c + nc, r : r + nr])

    return main


def mirror_lower_(
    matrix: torch.Tensor, *, diagonal_block: int = 0, diag_out=None, diag_add=0.0, tile: int = 32
):
    """Mirror all lower entries, or only diagonal square blocks.

    Every off-diagonal source tile is read-only and has one disjoint upper
    destination. A diagonal tile is loaded completely before its symmetric
    replacement is stored. No write/write overlap is asserted away.
    """
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    assert matrix.dtype == torch.float16 and matrix.is_contiguous()
    if diagonal_block and diagonal_block % tile:
        raise ValueError("diagonal_block must be a multiple of the mirror tile")
    if tile not in (16, 32, 64, 128):
        raise ValueError("mirror tile must be one of 16/32/64/128")
    key = ("mirror_lower_v3", matrix.shape[0], diagonal_block, diag_out is not None, tile)
    kernel = cached_kernel(key)
    if kernel is None:
        kernel = store_kernel(
            key,
            tilelang.compile(
                _build_mirror(
                    matrix.shape[0], diagonal_block, tile=tile, with_q=diag_out is not None
                ),
                target="ascend",
                out_idx=None,
            ),
        )
    kernel(matrix, diag_out if diag_out is not None else matrix, diag_add)
