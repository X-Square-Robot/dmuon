"""TileLang implementation of the public segmented gradient clip contract.

Norm, coefficient, nonfinite flags and scaling are device computations.  Torch
is used for allocation, metadata upload, layout copies and the single SUM
collective only.  No dtype/shape eligibility decision selects eager NPU math.

Plans are cached per stream and tensor identity, with weak input references and
a bounded byte budget.  Returned stats always own fresh storage, so a later
call cannot overwrite a result retained by the caller.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
import threading
import weakref

import torch
import torch.distributed as dist
import torch_npu
import tilelang

from .dispatch import cached_kernel, store_kernel
from .kernels import segmented_clip as kernels
from .kernels.elementwise import _dtype_name

_PLAN_LIMIT = 16
_PLAN_BYTES = 64 << 20
_plans = OrderedDict()
_lock = threading.RLock()


@dataclass
class _Group:
    dtype: str
    metadata: torch.Tensor
    segments: torch.Tensor
    count: int
    max_numel: int
    offset: int


@dataclass
class _Plan:
    refs: tuple
    stages: dict
    groups: tuple
    serial_scale_groups: tuple
    segment_ids: torch.Tensor
    reduce_flags: torch.Tensor
    partials: torch.Tensor
    local_sq: torch.Tensor
    reduced_sq: torch.Tensor
    jobs: int
    num_segments: int
    byte_size: int


def clear_plan_cache():
    """Release cached workspaces; does not invalidate returned clip stats."""
    with _lock:
        _plans.clear()


def plan_cache_info():
    with _lock:
        return {
            "entries": len(_plans),
            "bytes": sum(p.byte_size for p in _plans.values()),
            "budget": _PLAN_BYTES,
        }


def _compiled(key, builder, *args):
    key = ("segmented_clip_v2_dma",) + key
    k = cached_kernel(key)
    if k is None:
        k = store_kernel(key, tilelang.compile(builder(*args), target="ascend", out_idx=None))
    return k


def _format(tensor):
    return int(torch_npu.get_npu_format(tensor))


def _stage_read(source, stage):
    # Ascend 950 forbids ordinary copy_ from internal formats even when an
    # explicit format_cast created such a tensor. Canonicalize that transport.
    if _format(source) not in (0, 2, 30):
        source = torch_npu.npu_format_cast(source, 2)
    stage.copy_(source)


def _stage_write(stage, dest):
    fmt = _format(dest)
    if fmt in (0, 2, 30):
        dest.copy_(stage)
        return
    encoded = torch_npu.npu_format_cast(stage, fmt)
    nbytes = int(torch_npu.get_storage_size(encoded)) * encoded.element_size()
    dest_nbytes = int(torch_npu.get_storage_size(dest)) * dest.element_size()
    if dest.storage_offset() != 0 or dest_nbytes != nbytes or nbytes % 4:
        raise ValueError("Internal-format clip writeback requires a complete tensor storage")
    addresses = torch.tensor(
        (encoded.data_ptr(), dest.data_ptr()), dtype=torch.int64, device=dest.device
    )
    stream = torch.npu.current_stream(dest.device)
    encoded.record_stream(stream)
    dest.record_stream(stream)
    _compiled(("copy_storage", nbytes), kernels.copy_storage, nbytes)(addresses)


def _build_plan(tensors, segments, flags, chunk, device):
    # An empty public bucket list still needs a device scalar zero result.
    nsegments = max(1, len(flags))
    stages = {}
    by_dtype = {}
    for i, (grad, segment) in enumerate(zip(tensors, segments)):
        if not grad.is_contiguous() or _format(grad) not in (0, 2, 30):
            stages[i] = torch.empty(tuple(grad.shape), device=device, dtype=grad.dtype)
        data = stages.get(i, grad)
        dtype = _dtype_name(data.dtype)
        by_dtype.setdefault(dtype, []).append((data, segment))

    groups = []
    all_segments = []
    offset = 0
    for dtype, pairs in by_dtype.items():
        rows, ids = [], []
        max_numel = 1
        for tensor, segment in pairs:
            n = tensor.numel()
            max_numel = max(max_numel, n)
            for start in range(0, n, chunk):
                rows.append((tensor.data_ptr(), start, min(chunk, n - start)))
                ids.append(segment)
        if rows:
            meta = torch.tensor(rows, device=device, dtype=torch.int64)
            seg = torch.tensor(ids, device=device, dtype=torch.int32)
            groups.append(_Group(dtype, meta, seg, len(rows), max_numel, offset))
            offset += len(rows)
            all_segments.extend(ids)

    # The public API accepts tensor sequences. Repeated/overlapping views must
    # follow its sequential in-place scale semantics rather than race between
    # jobs. Conservative physical-span overlap detection is metadata-only; it
    # may serialize interleaved views, but never changes their arithmetic.
    spans = []
    for tensor in tensors:
        if tensor.numel():
            span = 1 + sum(
                (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
            )
            spans.append((tensor.data_ptr(), tensor.data_ptr() + span * tensor.element_size()))
    spans.sort()
    overlaps = False
    end = -1
    for begin, stop in spans:
        overlaps |= begin < end
        end = max(end, stop)
    serial_scale_groups = []
    if overlaps:
        for i, (grad, segment) in enumerate(zip(tensors, segments)):
            tensor = stages.get(i, grad)
            n = tensor.numel()
            rows = [
                (tensor.data_ptr(), start, min(chunk, n - start)) for start in range(0, n, chunk)
            ]
            if rows:
                group = _Group(
                    _dtype_name(tensor.dtype),
                    torch.tensor(rows, device=device, dtype=torch.int64),
                    torch.tensor([segment] * len(rows), device=device, dtype=torch.int32),
                    len(rows),
                    n,
                    0,
                )
                serial_scale_groups.append((i, group))

    segment_ids = torch.tensor(all_segments or [0], device=device, dtype=torch.int32)
    reduce_flags = torch.tensor(flags or [0], device=device, dtype=torch.int32)
    partials = torch.empty(max(1, offset), device=device, dtype=torch.float32)
    local_sq = torch.empty(nsegments, device=device, dtype=torch.float32)
    reduced_sq = torch.empty_like(local_sq)
    arrays = list(stages.values()) + [segment_ids, reduce_flags, partials, local_sq, reduced_sq]
    for group in groups:
        arrays.extend((group.metadata, group.segments))
    for _, group in serial_scale_groups:
        arrays.extend((group.metadata, group.segments))
    byte_size = sum(t.numel() * t.element_size() for t in arrays)
    return _Plan(
        tuple(weakref.ref(t) for t in tensors),
        stages,
        tuple(groups),
        tuple(serial_scale_groups),
        segment_ids,
        reduce_flags,
        partials,
        local_sq,
        reduced_sq,
        offset,
        nsegments,
        byte_size,
    )


def _get_plan(tensors, segments, flags, chunk, device, stream):
    signature = tuple(
        (id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, _format(t))
        for t in tensors
    )
    key = (str(device), int(stream.npu_stream), chunk, tuple(segments), tuple(flags), signature)
    # Indexed metadata plans key on template signatures (no data pointers);
    # the mode must never alias an absolute plan entry.
    # Indexed metadata binds fresh gradient addresses to a per-signature
    # template through one gather kernel; under per-step address churn it
    # saves the full host plan rebuild (measured 8.36 -> 6.31 ms/step on the
    # rank2-shaped population, 2026-11-06).  Ineligible inputs fall back to
    # the absolute plan inside build().
    metadata_mode = os.environ.get("DMUON_TILELANG_CLIP_METADATA", "indexed")
    if metadata_mode not in ("absolute", "indexed"):
        raise ValueError("DMUON_TILELANG_CLIP_METADATA must be absolute or indexed")
    if metadata_mode == "indexed":
        key += ("indexed_metadata_v1",)
    plan = _plans.get(key)
    if plan is not None and all(ref() is t for ref, t in zip(plan.refs, tensors)):
        _plans.move_to_end(key)
        return plan
    capturing = getattr(torch.npu, "is_current_stream_capturing", lambda: False)()
    if capturing:
        raise RuntimeError(
            "TileLang segmented clip plan must be warmed on the capture stream first"
        )
    if metadata_mode == "indexed":
        # Conservative eligibility inside build(): every rejected input
        # returns the absolute plan unchanged.
        from .indexed_clip import build

        plan = build(tensors, segments, flags, chunk, device)
    else:
        plan = _build_plan(tensors, segments, flags, chunk, device)
    if plan.byte_size <= _PLAN_BYTES:
        _plans[key] = plan
        while len(_plans) > _PLAN_LIMIT or sum(p.byte_size for p in _plans.values()) > _PLAN_BYTES:
            _plans.popitem(last=False)
    return plan


@torch.no_grad()
def clip_grad_norm_buckets_tilelang(buckets, max_norm, *, process_group=None, chunk_size=None):
    """Clip each bucket independently, preserving the NV collective protocol.

    The SUM collective has shape ``(len(buckets),)`` and is issued once if the
    process group is initialized and any bucket has ``reduce=True``.  Empty
    tensors and locally absent buckets do not alter this protocol.
    """
    from dmuon.fast_clip import (
        GradClipBucketStats,
        SegmentedGradClipResult,
        _resolve_chunk_size,
        _resolve_device,
    )

    buckets = tuple(buckets)
    device = _resolve_device(buckets)
    if device.type != "npu":
        raise ValueError("TileLang segmented clip requires NPU buckets")
    tensors, segments = [], []
    for index, bucket in enumerate(buckets):
        for grad in bucket.grads:
            if grad is None:
                continue
            if grad.device != device:
                raise ValueError("TileLang segmented clip buckets must share one NPU device")
            if grad.is_sparse or grad.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise TypeError(
                    "TileLang segmented clip supports dense float32/float16/bfloat16 gradients"
                )
            if max_norm is not None and torch._debug_has_internal_overlap(grad) == 1:
                raise ValueError("Cannot scale a gradient with overlapping logical elements")
            tensors.append(grad)
            segments.append(index)
    flags = [int(bool(bucket.reduce)) for bucket in buckets]
    # Bound the DMA reduction scratch independently of a caller's tuning hint.
    # Offsets/pointers stay int64; splitting a large hint only partitions sums.
    chunk = min(_resolve_chunk_size(chunk_size), 1 << 20)
    reducing = bool(dist.is_available() and dist.is_initialized() and any(flags))
    with torch.npu.device(device), _lock:
        stream = torch.npu.current_stream(device)
        plan = _get_plan(tensors, segments, flags, chunk, device, stream)
        # An indexed plan resolves its absolute metadata rows here, ahead of
        # every staging/norm consumer on the same stream.
        binding = getattr(plan, "binding", None)
        if binding is not None:
            binding.run(stream)
        for i, grad in enumerate(tensors):
            # Pointer-table accesses are invisible to the wrapper's argument
            # lifetime tracking, so register the source allocations explicitly.
            grad.record_stream(stream)
            if i in plan.stages:
                _stage_read(grad, plan.stages[i])
        for group in plan.groups:
            args = (group.count, group.max_numel, chunk, group.dtype)
            _compiled(("norm",) + args, kernels.norm_jobs, *args)(
                group.metadata, plan.partials[group.offset : group.offset + group.count]
            )
        args = (plan.jobs, plan.num_segments)
        _compiled(("combine",) + args, kernels.combine_jobs, *args)(
            plan.partials, plan.segment_ids, plan.reduce_flags, plan.local_sq, plan.reduced_sq
        )
        if reducing:
            dist.all_reduce(plan.reduced_sq, op=dist.ReduceOp.SUM, group=process_group)

        norms = torch.empty(plan.num_segments, device=device, dtype=torch.float32)
        coefs = torch.empty_like(norms)
        clipped = torch.empty(plan.num_segments, device=device, dtype=torch.bool)
        found_inf = torch.empty_like(clipped)
        total = torch.empty(1, device=device, dtype=torch.float32)
        args = (plan.num_segments, max_norm is not None, reducing)
        _compiled(("stats",) + args, kernels.finish_stats, *args)(
            plan.local_sq,
            plan.reduced_sq,
            plan.reduce_flags,
            norms,
            coefs,
            clipped,
            found_inf,
            total,
            0.0 if max_norm is None else float(max_norm),
        )
        if max_norm is not None:
            scale_groups = plan.serial_scale_groups or tuple((None, group) for group in plan.groups)
            for tensor_index, group in scale_groups:
                if tensor_index in plan.stages:
                    _stage_read(tensors[tensor_index], plan.stages[tensor_index])
                args = (group.count, group.max_numel, chunk, group.dtype, plan.num_segments)
                _compiled(("scale",) + args, kernels.scale_jobs, *args)(
                    group.metadata, group.segments, coefs
                )
                if tensor_index in plan.stages:
                    _stage_write(plan.stages[tensor_index], tensors[tensor_index])
            if not plan.serial_scale_groups:
                for i, staged in plan.stages.items():
                    _stage_write(staged, tensors[i])

        stats = tuple(
            GradClipBucketStats(
                name=bucket.name,
                total_norm=norms[i],
                max_norm=None if max_norm is None else float(max_norm),
                clip_coef=coefs[i],
                clipped=clipped[i],
                param_count=sum(grad is not None for grad in bucket.grads),
                found_inf=found_inf[i],
                fastpath=True,
            )
            for i, bucket in enumerate(buckets)
        )
        return SegmentedGradClipResult(
            total.reshape(()), stats, fastpath=True, fallback_reason=None
        )
