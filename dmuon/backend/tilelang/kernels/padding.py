"""Value-preserving rectangular padding without eager device fills."""

import torch
import torch_npu
import tilelang
import tilelang.ascend.language as T

from ..dispatch import cached_kernel, store_kernel

_DTYPES = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}


def _base_storage(tensor):
    return tensor.is_contiguous() and int(torch_npu.get_npu_format(tensor)) in (0, 2, 30)


def _build_pad(rows, cols, out_rows, out_cols, dtype):
    total = out_rows * out_cols
    blocks = min(64, (total + 255) // 256)
    span = (total + blocks - 1) // blocks

    @T.prim_func
    def main(X: T.Tensor((rows, cols), dtype), Out: T.Tensor((out_rows, out_cols), dtype)):
        with T.Kernel(blocks) as bx:
            with T.SimtVF(threads=256):
                for offset in T.Parallel(span):
                    index = bx * span + offset
                    if index < total:
                        row, col = index // out_cols, index % out_cols
                        if row < rows and col < cols:
                            Out[row, col] = X[row, col]
                        else:
                            Out[row, col] = T.cast(0, dtype)

    return main


def pad2d(source, out):
    """Write all output elements, preserving the input dtype and exact bits."""
    if (
        source.ndim != 2
        or out.ndim != 2
        or source.dtype != out.dtype
        or source.device != out.device
        or out.device.type != "npu"
        or source.shape[0] > out.shape[0]
        or source.shape[1] > out.shape[1]
        or not _base_storage(source)
        or not _base_storage(out)
    ):
        raise ValueError("Padding requires compatible contiguous matrices on one NPU")
    if not out.numel():
        return
    if source.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
        raise ValueError("Padding input and output storage must not overlap")
    dtype = _DTYPES[source.dtype]
    args = (*source.shape, *out.shape, dtype)
    key = ("pad2d_v1",) + args
    kernel = cached_kernel(key)
    if kernel is None:
        kernel = store_kernel(
            key, tilelang.compile(_build_pad(*args), target="ascend", out_idx=None)
        )
    kernel(source, out)
