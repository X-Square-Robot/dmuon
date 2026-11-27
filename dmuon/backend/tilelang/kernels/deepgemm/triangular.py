"""Balanced persistent traversal of a lower triangular square tile grid.

The folded rectangle enumerates every lower tile once without square roots,
dynamic searches, skipped upper tiles, or shape-specific coordinate tables.
"""

import tilelang.ascend.language as T
from tilelang.language.meta import meta_class
from tilelang.ascend.language.tile_schedule import AscendBaseTileScheduler


def triangular_coord(index, side):
    """Lower-triangle coordinate for an index in [0, side*(side+1)//2)."""
    width = (side + 1) // 2
    row, col = index // width, index % width
    if side % 2:
        return (
            T.if_then_else(col <= row, row, side - 1 - row),
            T.if_then_else(col <= row, col, side - col),
        )
    return (
        T.if_then_else(col < row, row - 1, side - 1 - row),
        T.if_then_else(col < row, col, side - 1 - col),
    )


@meta_class
class CompactTriangularScheduler(AscendBaseTileScheduler):
    def __init__(self, *, block_m, block_n, num_cores, shape_m, shape_n):
        assert block_m == block_n and shape_m == shape_n
        self._side = (shape_m + block_m - 1) // block_m
        super().__init__(
            block_m=block_m, block_n=block_n, num_cores=num_cores, shape_m=shape_m, shape_n=shape_n
        )
        self._num_blocks = self._side * (self._side + 1) // 2

    def _step(self):
        index = self._advance_block_idx()
        row, col = triangular_coord(index, self._side)
        self._m_idx[0], self._n_idx[0] = row, col
        self.valid_flag[0] = T.if_then_else(index < self._num_blocks, 1, 0)
