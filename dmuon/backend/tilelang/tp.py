"""Cast a tensor-parallel update to the owner dtype before scaling it."""

import tilelang
import tilelang.ascend.language as T

from .dispatch import cached_kernel, store_kernel
from .kernels import elementwise as ew


def _build_cast_then_scale(n: int, srcdt: str, dstdt: str):
    tile, blocks = 2048, 64
    iters = max(1, (n + blocks * tile - 1) // (blocks * tile))

    @T.prim_func
    def kernel(
        X: T.Tensor((n,), srcdt), O: T.Tensor((n,), dstdt), scale: T.float32, unity: T.float32
    ):
        with T.Kernel(blocks) as bx:
            xu = T.alloc_shared((tile,), srcdt)
            ou = T.alloc_shared((tile,), dstdt)
            for step in T.serial(iters):
                offset = (step * blocks + bx) * tile
                count = T.max(0, T.min(tile, n - offset))
                T.copy(X[offset : offset + count], xu[:count])
                with T.SimtVF(threads=256):
                    for i in T.Parallel(tile):
                        if i < count:
                            rounded = T.cast(T.cast(xu[i], "float32") * unity, dstdt)
                            ou[i] = T.cast(T.cast(rounded, "float32") * scale, dstdt)
                T.copy(ou[:count], O[offset : offset + count])

    return kernel


def cast_then_scale(update, out, scale):
    """Native TP delta semantics: cast to owner dtype, then in-place scale."""
    key = (
        "tp_cast_then_scale_f32bridge_v2",
        out.numel(),
        ew._dtype_name(update.dtype),
        ew._dtype_name(out.dtype),
    )
    kernel = cached_kernel(key)
    if kernel is None:
        kernel = store_kernel(
            key, tilelang.compile(_build_cast_then_scale(*key[1:]), target="ascend", out_idx=None)
        )
    kernel(update.contiguous().view(-1), out.view(-1), float(scale), 1.0)
