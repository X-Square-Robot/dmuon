"""Grad-clip compute replacement: local norms and per-tensor scaling.

Implements the Euclidean norm path of dmuon.grad_clip.

Eager semantics reproduced:
* per-tensor norm: fp32 accumulation, one rounding of the scalar to the
  tensor's *input* dtype, then exact widen to fp32 for the stack;
* final combine: sqrt(Σ norms²) in fp32;
* per-tensor scale: fp32 internal multiply by a *device-resident*
  coefficient (no host sync), one storage rounding per element.
"""

import torch
import torch_npu
import tilelang
import tilelang.ascend.language as T

from .dispatch import cached_kernel, store_kernel
from .kernels.elementwise import (
    _NUM_BLOCKS,
    _TILE,
    _dtype_name,
    _iterations,
    _tail,
)


# ---------------------------------------------------------------------------
# Per-tensor partial sum-of-squares (one launch per tensor)
# ---------------------------------------------------------------------------


def _build_tensor_norm_p1(numel: int, dt: str):
    iters = _iterations(numel)
    tail = _tail(numel)

    @T.prim_func
    def main(
        X: T.Tensor((numel,), dt),
        Partials: T.Tensor((_NUM_BLOCKS,), "float32"),
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            x_ub = T.alloc_shared((_TILE,), dt)
            step_sums = T.alloc_shared((iters,), "float32")
            T.annotate_buffer_versions({x_ub: 2})
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * _NUM_BLOCKS + bx) * _TILE
                rem = T.if_then_else(numel > begin, numel - begin, 0)
                n = T.if_then_else(
                    step == iters - 1,
                    T.if_then_else(rem >= tail, tail, rem),
                    T.if_then_else(rem >= _TILE, _TILE, rem),
                )
                T.copy(X[begin : begin + n], x_ub[:n])
                with T.SimtVF(threads=256):
                    red = T.alloc_reducer((1,), "float32", op="sum")
                    T.reducer_init(red)
                    for i in T.Parallel(_TILE):
                        if i < n:
                            v = T.cast(x_ub[i], "float32")
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


def _p1_kernel(t: torch.Tensor):
    key = ("tensor_norm_p1", t.numel(), _dtype_name(t.dtype))
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key,
            tilelang.compile(
                _build_tensor_norm_p1(t.numel(), _dtype_name(t.dtype)),
                target="ascend",
                out_idx=None,
            ),
        )
    return k


# ---------------------------------------------------------------------------
# Group finalize: (rows, 64) partials → norms rounded to input dtype, then
# widened to fp32 (exact), written into an fp32 slice of the norms vector
# ---------------------------------------------------------------------------


def _build_finalize_group(dt: str, total_rows: int):
    # One 64-row SimtVF block - the only validated shape class for the
    # (rows,) reducer/fragment under threads=64 ((64,1)->(1,64) is bijective
    # and single-block, so no cross-block lane corruption). Larger groups are
    # chunked into 64-row kernel launches by the caller.
    @T.prim_func
    def main(
        Partials: T.Tensor((64, _NUM_BLOCKS), "float32"),
        Indices: T.Tensor((64,), "int32"),
        Norms: T.Tensor((total_rows,), "float32"),
    ):
        with T.Kernel(1) as bx:
            p_ub = T.alloc_shared((64, _NUM_BLOCKS), "float32")
            T.copy(Partials, p_ub)
            with T.SimtVF(threads=64):
                red = T.alloc_reducer((64,), "float32", op="sum")
                T.reducer_init(red)
                for t, b in T.Parallel(64, _NUM_BLOCKS):
                    T.reducer_update(red[t], p_ub[t, b])
                fres = T.alloc_fragment((64,), "float32")
                T.finalize_reducer(red, fres)
                for t in T.Parallel(64):
                    # Round to the tensor dtype (eager norm output), then
                    # widen exactly to fp32 (eager .to(float32)).
                    Norms[Indices[t]] = T.cast(T.cast(T.sqrt(fres[t]), dt), "float32")

    return main


def _build_combine(n: int):
    @T.prim_func
    def main(Norms: T.Tensor((n,), "float32"), Out: T.Tensor((1,), "float32")):
        with T.Kernel(1) as bx:
            n_ub = T.alloc_shared((n,), "float32")
            T.copy(Norms, n_ub)
            with T.SimtVF(threads=64):
                red = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(red)
                for i in T.Parallel(n):
                    v = n_ub[i]
                    T.reducer_update(red[0], v * v)
                fres = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(red, fres)
                Out[0] = T.sqrt(fres[0])

    return main


# ---------------------------------------------------------------------------
# Gradient clipping operations
# ---------------------------------------------------------------------------


def local_total_norm_tilelang(entries, norm_type, optimizer, *, foreach=None, reference=None):
    if norm_type != 2.0:
        return reference(entries, norm_type, optimizer, foreach=foreach)

    tensors = [entry.grad.detach() for entry in entries]
    if len(tensors) == 0:
        from dmuon.grad_clip import _accumulator_device

        device = _accumulator_device(optimizer, entries)
        if device.type != "npu":
            return reference(entries, norm_type, optimizer, foreach=foreach)
        out = torch.empty((1,), device=device, dtype=torch.float32)
        _scalar_kernel("zero")(out, out)
        return out.reshape(())

    npu_idx = [i for i, t in enumerate(tensors) if t.device.type == "npu"]
    if not npu_idx:
        return reference(entries, norm_type, optimizer, foreach=foreach)
    if len(npu_idx) != len(tensors):
        return reference(entries, norm_type, optimizer, foreach=foreach)

    device = tensors[npu_idx[0]].device
    n = len(tensors)
    groups: dict = {}
    for i in npu_idx:
        groups.setdefault(_dtype_name(tensors[i].dtype), []).append(i)

    norms = torch.empty((n + 1,), device=device, dtype=torch.float32)
    for dt, idxs in groups.items():
        rows = len(idxs)
        padded = ((rows + 63) // 64) * 64
        partials = torch.zeros((padded, _NUM_BLOCKS), device=device, dtype=torch.float32)
        for row, i in enumerate(idxs):
            t = tensors[i]
            if not t.is_contiguous() or torch_npu.get_npu_format(t) not in (0, 2, 30):
                staged = torch.empty(tuple(t.shape), dtype=t.dtype, device=t.device)
                from .segmented_clip import _stage_read

                _stage_read(t, staged)
                t = staged
            _p1_kernel(t)(t.view(-1), partials[row])
        key = ("finalize_group_scatter64", dt, n)
        k = cached_kernel(key)
        if k is None:
            k = store_kernel(
                key,
                tilelang.compile(_build_finalize_group(dt, n + 1), target="ascend", out_idx=None),
            )
        # Scatter belongs to the TileLang finalizer; no eager index_put_.
        # Chunk into single-block 64-row launches; pad entries point at the
        # scratch slot (index n) and contribute sqrt(0)=0 there, never read.
        idx64 = torch.cat(
            (
                torch.tensor(idxs, device=device, dtype=torch.int32),
                torch.full((padded - rows,), n, device=device, dtype=torch.int32),
            )
        )
        for start in range(0, padded, 64):
            k(partials[start : start + 64], idx64[start : start + 64], norms)

    key = ("norm_combine", n)
    kc = cached_kernel(key)
    if kc is None:
        kc = store_kernel(key, tilelang.compile(_build_combine(n), target="ascend", out_idx=None))
    out = torch.empty((1,), device=device, dtype=torch.float32)
    kc(norms[:n], out)
    return out.reshape(())


def _build_mul_scale(numel: int, dt: str):
    iters = _iterations(numel)
    tail = _tail(numel)

    @T.prim_func
    def main(
        X: T.Tensor((numel,), dt),
        Out: T.Tensor((numel,), dt),
        Coef: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(_NUM_BLOCKS) as bx:
            x_ub = T.alloc_shared((_TILE,), dt)
            c_ub = T.alloc_shared((1,), "float32")
            T.copy(Coef, c_ub)
            T.annotate_buffer_versions({x_ub: 2})
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * _NUM_BLOCKS + bx) * _TILE
                rem = T.if_then_else(numel > begin, numel - begin, 0)
                n = T.if_then_else(
                    step == iters - 1,
                    T.if_then_else(rem >= tail, tail, rem),
                    T.if_then_else(rem >= _TILE, _TILE, rem),
                )
                T.copy(X[begin : begin + n], x_ub[:n])
                with T.SimtVF(threads=256):
                    for i in T.Parallel(_TILE):
                        if i < n:
                            x_ub[i] = T.cast(T.cast(x_ub[i], "float32") * c_ub[0], dt)
                T.copy(x_ub[:n], Out[begin : begin + n])

    return main


def clip_entries_with_norm_tilelang(entries, clip_coef_clamped, foreach=None, reference=None):
    tensors = [entry.grad for entry in entries]
    if len(tensors) == 0:
        return
    for t in tensors:
        if t.device.type != "npu":
            t.mul_(clip_coef_clamped.to(t.device))
            continue
        from .kernels.elementwise import cast_to_fp32

        coef = clip_coef_clamped.reshape(1)
        if coef.device != t.device:
            coef = coef.to(device=t.device)
        if coef.dtype != torch.float32:
            coef = cast_to_fp32(coef)
        key = ("mul_scale", t.numel(), _dtype_name(t.dtype))
        k = cached_kernel(key)
        if k is None:
            k = store_kernel(
                key,
                tilelang.compile(
                    _build_mul_scale(t.numel(), _dtype_name(t.dtype)), target="ascend", out_idx=None
                ),
            )
        if t.is_contiguous() and torch_npu.get_npu_format(t) in (0, 2, 30):
            k(t.view(-1), t.view(-1), coef)
        else:
            out = torch.empty(tuple(t.shape), dtype=t.dtype, device=t.device)
            from .segmented_clip import _stage_read, _stage_write

            _stage_read(t, out)  # layout staging only, explicitly contiguous ND
            k(out.view(-1), out.view(-1), coef)
            _stage_write(out, t)


def _build_scalar(mode):
    @T.prim_func
    def main(In: T.Tensor((1,), "float32"), Out: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            with T.SimtVF(threads=32):
                for i in T.Parallel(1):
                    if mode == "zero":
                        Out[0] = 0.0
                    elif mode == "square":
                        Out[0] = In[0] * In[0]
                    else:
                        Out[0] = T.sqrt(In[0])

    return main


def _scalar_kernel(mode):
    key = ("legacy_clip_scalar", mode)
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(key, tilelang.compile(_build_scalar(mode), target="ascend", out_idx=None))
    return k


def _build_legacy_stats(has_max):
    @T.prim_func
    def main(
        Norm: T.Tensor((1,), "float32"),
        Coef: T.Tensor((1,), "float32"),
        Clipped: T.Tensor((1,), "bool"),
        FoundInf: T.Tensor((1,), "bool"),
        max_norm: T.float32,
    ):
        with T.Kernel(1):
            with T.SimtVF(threads=32):
                for i in T.Parallel(1):
                    norm = Norm[0]
                    norm_abs_bits = T.reinterpret(norm, "uint32") & T.uint32(0x7FFFFFFF)
                    finite = norm_abs_bits < T.uint32(0x7F800000)
                    raw = T.float32(1.0)
                    if has_max:
                        raw = max_norm / (norm + 1.0e-6)
                        raw_abs_bits = T.reinterpret(raw, "uint32") & T.uint32(0x7FFFFFFF)
                        Coef[0] = T.if_then_else(
                            raw_abs_bits > T.uint32(0x7F800000), raw, T.min(raw, 1.0)
                        )
                    else:
                        Coef[0] = 1.0
                    Clipped[0] = finite and has_max and raw < 1.0
                    FoundInf[0] = not finite

    return main


def _legacy_stats(total_norm, max_norm):
    coef = torch.empty((1,), device=total_norm.device, dtype=torch.float32)
    clipped = torch.empty((1,), device=total_norm.device, dtype=torch.bool)
    found_inf = torch.empty_like(clipped)
    key = ("legacy_clip_stats", max_norm is not None)
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(
            key,
            tilelang.compile(
                _build_legacy_stats(max_norm is not None), target="ascend", out_idx=None
            ),
        )
    k(total_norm.reshape(1), coef, clipped, found_inf, 0.0 if max_norm is None else float(max_norm))
    return coef.reshape(()), clipped.reshape(()), found_inf.reshape(())


def compute_total_norm_tilelang(optimizer, entries, norm_type, *, error_if_nonfinite, foreach=None):
    """Legacy roundings and the scalar collective are preserved exactly."""
    if norm_type != 2.0:
        raise NotImplementedError("TileLang legacy clipping supports norm_type=2")
    from dmuon.grad_clip import _all_reduce_if_needed
    import torch.distributed as dist

    local = local_total_norm_tilelang(entries, norm_type, optimizer, foreach=foreach)
    power = torch.empty((1,), device=local.device, dtype=torch.float32)
    _scalar_kernel("square")(local.reshape(1), power)
    # Preserve scalar shape () used by the legacy distributed entry point.
    _all_reduce_if_needed(power.reshape(()), op=dist.ReduceOp.SUM)
    total = torch.empty_like(power)
    _scalar_kernel("sqrt")(power, total)
    result = total.reshape(())
    if error_if_nonfinite:
        _, _, found_inf = _legacy_stats(result, None)
        if bool(found_inf.item()):  # Explicitly requested error policy may synchronize.
            raise RuntimeError(
                "The total norm of order 2.0 for DMuon Muon gradients is non-finite, so it cannot be clipped."
            )
    return result


@torch.no_grad()
def global_norm_strategy_tilelang(context):
    """Legacy public strategy, including device-only coefficient/stats math."""
    from dmuon.grad_clip import MuonGradClipStats

    total = compute_total_norm_tilelang(
        context.optimizer,
        context.entries,
        context.norm_type,
        error_if_nonfinite=context.error_if_nonfinite,
        foreach=context.foreach,
    )
    if context.max_norm is not None and context.max_norm < 0:
        raise ValueError(f"max_norm must be non-negative, got {context.max_norm}")
    coef, clipped, found_inf = _legacy_stats(total, context.max_norm)
    if context.max_norm is not None:
        clip_entries_with_norm_tilelang(context.entries, coef, foreach=context.foreach)
    return MuonGradClipStats(
        total_norm=total,
        max_norm=context.max_norm,
        norm_type=context.norm_type,
        clip_coef=coef if context.max_norm is not None else 1.0,
        clipped=clipped if context.max_norm is not None else False,
        param_count=len(context.entries),
        found_inf=found_inf,
        strategy="global_norm",
    )
