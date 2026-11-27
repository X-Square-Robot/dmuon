"""Direct-space Newton--Schulz with the native storage-rounding order.

In Python, ``c * A @ A`` means ``(c * A) @ A``.  The scaled operand must
therefore be stored in FP16 before the contraction.  This backend deliberately
does not replace that expression with a raw-accumulator polynomial epilogue.
"""

from collections import OrderedDict
import os
import threading

import torch
import torch_npu
import tilelang
import tilelang.ascend.language as T

from .dispatch import cached_kernel, store_kernel
from .kernels import elementwise as ew, deep_wrapper as dg

_lock = threading.RLock()
_workspaces = OrderedDict()


class _Workspace:
    def __init__(self, m, n, device):
        def empty(shape):
            return torch.empty(shape, dtype=torch.float16, device=device)

        self.x = empty((m, n))
        self.next = empty((m, n))
        self.ax = empty((m, n))
        self.xt = empty((n, m))
        self.a = empty((m, m))
        self.at = empty((m, m))
        self.ca = empty((m, m))
        self.ba = empty((m, m))
        self.product = empty((m, m))
        self.b = empty((m, m))
        self.partials = torch.empty((64,), dtype=torch.float32, device=device)
        self.bytes = sum(
            t.numel() * t.element_size() for t in vars(self).values() if torch.is_tensor(t)
        )


def _workspace(m, n, device):
    stream = torch.npu.current_stream(device)
    key = ("direct", m, n, str(device), int(stream.npu_stream))
    if key in _workspaces:
        _workspaces.move_to_end(key)
        return _workspaces[key]
    ws = _Workspace(m, n, device)
    budget = int(os.environ.get("DMUON_TILELANG_WORKSPACE_BYTES", 2 * 1024**3))
    if ws.bytes <= budget:
        _workspaces[key] = ws
        while len(_workspaces) > 16 or sum(w.bytes for w in _workspaces.values()) > budget:
            _workspaces.popitem(last=False)
    return ws


def clear_workspace_cache():
    with _lock:
        _workspaces.clear()


def _build_pad(m, n, mp, np_):
    total = mp * np_
    blocks = min(64, (total + 255) // 256)
    span = (total + blocks - 1) // blocks

    @T.prim_func
    def main(X: T.Tensor((m, n), "float16"), Out: T.Tensor((mp, np_), "float16")):
        with T.Kernel(blocks) as bx:
            with T.SimtVF(threads=256):
                for offset in T.Parallel(span):
                    index = bx * span + offset
                    if index < total:
                        i, j = index // np_, index % np_
                        if i < m and j < n:
                            Out[i, j] = X[i, j]
                        else:
                            Out[i, j] = T.float16(0.0)

    return main


def _build_result(m, n, mp, np_, dtype, transposed):
    rows, cols = (n, m) if transposed else (m, n)
    total = rows * cols
    blocks = min(64, (total + 255) // 256)
    span = (total + blocks - 1) // blocks

    @T.prim_func
    def main(
        X: T.Tensor((mp, np_), "float16"), Out: T.Tensor((rows, cols), dtype), unity: T.float32
    ):
        with T.Kernel(blocks) as bx:
            with T.SimtVF(threads=256):
                for offset in T.Parallel(span):
                    index = bx * span + offset
                    if index < total:
                        i, j = index // cols, index % cols
                        if transposed:
                            Out[i, j] = T.cast(T.cast(X[j, i], "float32") * unity, dtype)
                        else:
                            Out[i, j] = T.cast(T.cast(X[i, j], "float32") * unity, dtype)

    return main


def _compiled(key, builder, *args):
    key = ("direct_ns_v1",) + key
    kernel = cached_kernel(key)
    if kernel is None:
        kernel = store_kernel(key, tilelang.compile(builder(*args), target="ascend", out_idx=None))
    return kernel


@torch.no_grad()
def direct_newton_schulz_tilelang(G, steps=5, eps=1e-7, coefficients=None, reference=None):
    if G.device.type != "npu":
        if reference is None:
            raise RuntimeError("CPU direct NS requires its native reference")
        return reference(G, steps=steps, eps=eps, coefficients=coefficients)
    if G.ndim != 2:
        raise ValueError("Newton-Schulz expects a matrix")
    if G.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("TileLang direct NS supports float32/float16/bfloat16")
    if G.numel() == 0:
        return torch.empty_like(G)
    if coefficients is None:
        from dmuon.optim.newton_schulz import DEFAULT_COEFFICIENTS

        coefficients = DEFAULT_COEFFICIENTS
    with torch.npu.device(G.device), _lock:
        source = G
        if torch_npu.get_npu_format(source) not in (0, 2, 30):
            source = torch_npu.npu_format_cast(source, 2)
        if not source.is_contiguous() or torch_npu.get_npu_format(source) != 2:
            source = torch.empty(tuple(G.shape), dtype=G.dtype, device=G.device)
            from .segmented_clip import _stage_read

            _stage_read(G, source)
        x32 = ew.cast_to_fp32(source)
        transposed = G.shape[0] > G.shape[1]
        if transposed:
            x32 = x32.t().contiguous()
        m, n = x32.shape
        mp, np_ = (m + 15) // 16 * 16, (n + 15) // 16 * 16
        ws = _workspace(mp, np_, G.device)
        ew.norm_stage1(x32, ws.partials)
        if (m, n) == (mp, np_):
            ew.norm_stage2(x32, ws.partials, ws.x, eps)
        else:
            compact = torch.empty((m, n), dtype=torch.float16, device=G.device)
            ew.norm_stage2(x32, ws.partials, compact, eps)
            args = (m, n, mp, np_)
            _compiled(("pad",) + args, _build_pad, *args)(compact, ws.x)
        x, next_x = ws.x, ws.next
        for a, b, c in coefficients:
            ew.transpose2d(x, ws.xt)
            dg.cast_fp16(x, x, ws.a)
            # Use an actual transpose: numerical symmetry is not assumed.
            ew.transpose2d(ws.a, ws.at)
            ew.cast_scale(ws.a, ws.ca, c)
            ew.cast_scale(ws.a, ws.ba, b)
            dg.cast_fp16(ws.ca, ws.at, ws.product)
            ew.axpby16(ws.ba, ws.product, ws.b, 1.0, 1.0)
            ew.cast_scale(x, ws.ax, a)
            dg.cast_fp16(ws.b, ws.xt, next_x)
            ew.axpby16(ws.ax, next_x, next_x, 1.0, 1.0)
            x, next_x = next_x, x
        out = torch.empty(tuple(G.shape), dtype=G.dtype, device=G.device)
        args = (m, n, mp, np_, ew._dtype_name(G.dtype), transposed)
        # Keep the FP32 bridge live through lowering. Direct FP16->BF16 SIMD
        # casts in this toolchain can reinterpret packed lanes instead of
        # converting them; an exhaustive 65,536-pattern gate covers this path.
        _compiled(("result",) + args, _build_result, *args)(x, out, 1.0)
        return out
