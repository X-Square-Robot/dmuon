"""Indexed clip metadata: template/address separation for segmented clip.

The absolute plan bakes each gradient's device pointer into its metadata
rows, so every new gradient allocation builds a fresh plan.  This module
keeps a per-signature *template* (row indices and segment ids only) and
binds concrete addresses at plan-build time through a small gather kernel,
so same-shape plan-cache misses reuse one compiled template.

Eligibility is conservative: overlapping storage, non-contiguous tensors
or internal NPU formats return the absolute plan unchanged.  The numerical
kernels are untouched; a bound plan only differs in how its metadata rows
acquire their pointers.
"""

from dataclasses import dataclass
import weakref

import torch
import tilelang
import tilelang.ascend.language as T

from . import segmented_clip as c
from .kernels.elementwise import _dtype_name
from .dispatch import (
    cached_kernel,
    capture_kernel_handles,
    store_kernel,
    use_pinned_kernel_handles,
)

_old = c._build_plan
_templates = weakref.WeakValueDictionary()


def bind_jobs(jobs, tensors):
    blocks = min(64, max(1, (jobs + 255) // 256))
    span = (jobs + blocks - 1) // blocks

    @T.prim_func
    def main(
        Addresses: T.Tensor((tensors,), "int64"),
        Index: T.Tensor((jobs, 3), "int64"),
        Out: T.Tensor((jobs, 3), "int64"),
    ):
        with T.Kernel(blocks) as bx:
            with T.SimtVF(threads=256):
                for offset in T.Parallel(span):
                    i = bx * span + offset
                    if i < jobs:
                        Out[i, 0] = Addresses[T.cast(Index[i, 0], "int32")]
                        Out[i, 1] = Index[i, 1]
                        Out[i, 2] = Index[i, 2]

    return main


@dataclass
class Template:
    groups: tuple
    segment_ids: object
    flags: object
    jobs: int
    nsegments: int
    handles: object


@dataclass
class Binding:
    template: object
    addresses: object
    groups: tuple

    def run(self, stream):
        self.addresses.record_stream(stream)
        with use_pinned_kernel_handles(self.template.handles):
            for key, index, output in self.groups:
                index.record_stream(stream)
                output.record_stream(stream)
                cached_kernel(key)(self.addresses, index, output)


def build(tensors, segments, flags, chunk, device):
    if not tensors:
        return _old(tensors, segments, flags, chunk, device)
    metadata = [
        (
            tuple(t.shape),
            tuple(t.stride()),
            t.dtype,
            c._format(t),
            t.numel(),
            t.data_ptr(),
            t.element_size(),
            t.is_contiguous(),
        )
        for t in tensors
    ]
    spans = sorted((m[5], m[5] + m[4] * m[6]) for m in metadata if m[4])
    end = -1
    overlap = False
    for begin, stop in spans:
        overlap |= begin < end
        end = max(end, stop)
    # This is a generic layout eligibility condition. All other inputs retain
    # the existing TileLang staging/serial alias implementation.
    if overlap or any(not m[7] or m[3] not in (0, 2, 30) for m in metadata):
        return _old(tensors, segments, flags, chunk, device)
    stream = int(torch.npu.current_stream(device).npu_stream)
    key = (
        str(device),
        stream,
        chunk,
        tuple(segments),
        tuple(flags),
        tuple(m[:5] for m in metadata),
    )
    template = _templates.get(key)
    if template is None:
        plan = _old(tensors, segments, flags, chunk, device)
        by_dtype = {}
        for i, (t, segment, m) in enumerate(zip(tensors, segments, metadata)):
            by_dtype.setdefault(_dtype_name(t.dtype), []).extend(
                (i, start, min(chunk, m[4] - start)) for start in range(0, m[4], chunk)
            )
        descriptors = []
        with capture_kernel_handles(max_entries=16) as handles:
            for group in plan.groups:
                rows = by_dtype[group.dtype]
                index = torch.tensor(rows, dtype=torch.int64, device=device)
                bind_key = ("experimental_clip_bind_v1", len(rows), len(tensors))
                kernel = cached_kernel(bind_key)
                if kernel is None:
                    store_kernel(
                        bind_key,
                        tilelang.compile(
                            bind_jobs(len(rows), len(tensors)), target="ascend", out_idx=None
                        ),
                    )
                descriptors.append(
                    (
                        group.dtype,
                        index,
                        group.segments,
                        group.count,
                        group.max_numel,
                        group.offset,
                        bind_key,
                    )
                )
        template = Template(
            tuple(descriptors),
            plan.segment_ids,
            plan.reduce_flags,
            plan.jobs,
            plan.num_segments,
            handles,
        )
        _templates[key] = template
        # The first ordinary plan is already correctly bound. It retains only
        # immutable descriptors, no source owner or live address in the cache.
        plan.index_template = template
        plan.byte_size += sum(
            index.numel() * index.element_size() for _, index, _, _, _, _, _ in descriptors
        )
        return plan
    addresses = torch.tensor([m[5] for m in metadata], dtype=torch.int64, device=device)
    groups = []
    bindings = []
    for dtype, index, ids, count, max_numel, offset, bind_key in template.groups:
        output = torch.empty((count, 3), dtype=torch.int64, device=device)
        groups.append(c._Group(dtype, output, ids, count, max_numel, offset))
        bindings.append((bind_key, index, output))
    partials = torch.empty(max(1, template.jobs), dtype=torch.float32, device=device)
    local = torch.empty(template.nsegments, dtype=torch.float32, device=device)
    reduced = torch.empty_like(local)
    arrays = [addresses, partials, local, reduced, template.segment_ids, template.flags]
    arrays.extend(value for group in groups for value in (group.metadata, group.segments))
    arrays.extend(index for _, index, _ in bindings)
    size = sum(value.numel() * value.element_size() for value in arrays)
    plan = c._Plan(
        tuple(weakref.ref(t) for t in tensors),
        {},
        tuple(groups),
        (),
        template.segment_ids,
        template.flags,
        partials,
        local,
        reduced,
        template.jobs,
        template.nsegments,
        size,
    )
    plan.binding = Binding(template, addresses, tuple(bindings))
    plan.index_template = template
    return plan
