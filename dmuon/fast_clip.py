"""Optional segmented CUDA fast path for gradient clipping.

The public training semantics stay segment-local: each bucket gets its own
gradient norm and clip coefficient.  The CUDA path only changes how those
segment-local norms are computed and applied.

Distributed safety: the CUDA kernels accelerate only the *local* norm/scale
arithmetic and are selected **per tensor**.  The collective that turns local
per-bucket squared norms into global ones is issued exactly once, with a fixed
shape derived from ``reduce`` flags that are identical on every rank.  A rank
whose tensors happen to be ineligible for the kernel (non-contiguous, unusual
dtype, CPU) transparently uses the torch reference math for *those tensors*
only -- it never changes how many collectives are issued, so ranks with
different local eligibility stay in lockstep.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Sequence

import torch
import torch.distributed as dist

from .grad_clip import MuonGradClipStats, _iter_muon_grad_entries


_DTYPE_CODES = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}
_DEFAULT_CHUNK_SIZE = 262144


@dataclass(frozen=True)
class GradClipBucket:
    """One independently clipped gradient segment."""

    name: str
    grads: Sequence[torch.Tensor]
    reduce: bool = True


@dataclass(frozen=True)
class GradClipBucketStats:
    """Pre-clip norm and coefficient for one gradient segment."""

    name: str
    total_norm: float
    max_norm: float | None
    clip_coef: float
    clipped: bool
    param_count: int
    found_inf: bool
    fastpath: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "total_norm": self.total_norm,
            "max_norm": self.max_norm,
            "clip_coef": self.clip_coef,
            "clipped": self.clipped,
            "param_count": self.param_count,
            "found_inf": self.found_inf,
            "fastpath": self.fastpath,
        }


@dataclass(frozen=True)
class SegmentedGradClipResult:
    """Result for a multi-segment clipping call."""

    total_norm: torch.Tensor
    bucket_stats: tuple[GradClipBucketStats, ...]
    fastpath: bool
    fallback_reason: str | None = None

    @property
    def stats_by_name(self) -> dict[str, GradClipBucketStats]:
        return {stats.name: stats for stats in self.bucket_stats}


@lru_cache(maxsize=1)
def _load_fast_clip_extension():
    """Load the compiled fast-clip CUDA extension, or ``None`` if unavailable.

    Distinguishes "not built" (expected when installed without a CUDA
    toolchain) from "built but failed to load" (an ABI mismatch, which is a
    real problem worth surfacing).  ``DMUON_FAST_CLIP=0`` disables the fast
    path; ``DMUON_FAST_CLIP_VERBOSE=1`` re-raises instead of degrading.
    """

    if os.environ.get("DMUON_FAST_CLIP", "1") == "0":
        return None
    verbose = os.environ.get("DMUON_FAST_CLIP_VERBOSE", "0") == "1"
    try:
        return import_module("dmuon._fast_clip_cuda")
    except ModuleNotFoundError:
        # Extension was never compiled (e.g. installed without CUDA_HOME, or
        # with --no-build-isolation missing).  Expected; stay quiet.
        if verbose:
            raise
        return None
    except Exception as exc:
        # Built but unloadable -- almost always a torch/CUDA ABI mismatch.
        # Warn once (lru_cache makes this a single call) so the silent
        # slow-path fallback is diagnosable.
        if verbose:
            raise
        warnings.warn(
            f"dmuon fast-clip extension failed to load ({exc!r}); "
            "falling back to the pure-Python clip. Rebuild with a CUDA "
            "toolchain matching your torch, or set DMUON_FAST_CLIP_VERBOSE=1.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _resolve_chunk_size(chunk_size: int | None) -> int:
    if chunk_size is not None:
        return max(1, int(chunk_size))
    raw_value = os.environ.get("DMUON_FAST_CLIP_CHUNK_SIZE")
    if raw_value is None:
        return _DEFAULT_CHUNK_SIZE
    try:
        return max(1, int(raw_value))
    except ValueError:
        return _DEFAULT_CHUNK_SIZE


def _resolve_device(buckets: Sequence[GradClipBucket]) -> torch.device:
    for bucket in buckets:
        for grad in bucket.grads:
            if grad is not None:
                return grad.device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _tensor_kernel_eligible(grad: torch.Tensor, device: torch.device) -> bool:
    """Whether the CUDA kernel can process this single tensor."""

    return (
        grad.device == device
        and device.type == "cuda"
        and not grad.is_sparse
        and grad.is_contiguous()
        and grad.dtype in _DTYPE_CODES
    )


def _build_metadata(
    tensors: Sequence[torch.Tensor],
    segments: Sequence[int],
    *,
    device: torch.device,
    chunk_size: int,
):
    ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in tensors],
        device=device,
        dtype=torch.int64,
    )
    numels = torch.tensor(
        [tensor.numel() for tensor in tensors],
        device=device,
        dtype=torch.int64,
    )
    dtypes = torch.tensor(
        [_DTYPE_CODES[tensor.dtype] for tensor in tensors],
        device=device,
        dtype=torch.int32,
    )
    tensor_segments = torch.tensor(segments, device=device, dtype=torch.int32)

    job_tensor_ids: list[int] = []
    job_offsets: list[int] = []
    for tensor_id, tensor in enumerate(tensors):
        numel = int(tensor.numel())
        for offset in range(0, numel, chunk_size):
            job_tensor_ids.append(tensor_id)
            job_offsets.append(offset)

    return {
        "ptrs": ptrs,
        "numels": numels,
        "dtypes": dtypes,
        "segments": tensor_segments,
        "job_tensor_ids": torch.tensor(
            job_tensor_ids, device=device, dtype=torch.int32
        ),
        "job_offsets": torch.tensor(job_offsets, device=device, dtype=torch.int64),
    }


def _split_by_kernel_eligibility(
    buckets: Sequence[GradClipBucket],
    device: torch.device,
    ext,
) -> tuple[list[torch.Tensor], list[int], list[tuple[torch.Tensor, int]]]:
    """Partition every live grad into kernel-eligible and torch-fallback sets.

    Returns ``(kernel_tensors, kernel_segments, torch_pairs)`` where
    ``torch_pairs`` is ``[(grad, segment_index), ...]``.  The decision is
    purely local and per-tensor; it never affects collective shape.
    """

    kernel_tensors: list[torch.Tensor] = []
    kernel_segments: list[int] = []
    torch_pairs: list[tuple[torch.Tensor, int]] = []
    for segment, bucket in enumerate(buckets):
        for grad in bucket.grads:
            if grad is None:
                continue
            if ext is not None and _tensor_kernel_eligible(grad, device):
                kernel_tensors.append(grad)
                kernel_segments.append(segment)
            else:
                torch_pairs.append((grad, segment))
    return kernel_tensors, kernel_segments, torch_pairs


def _segmented_local_sq(
    buckets: Sequence[GradClipBucket],
    *,
    device: torch.device,
    ext,
    chunk_size: int,
) -> torch.Tensor:
    """Per-bucket local sum of squared grads (no collective).

    Kernel-eligible tensors are summed by the CUDA kernel (``atomicAdd`` into
    ``local_sq``); the rest are summed with torch and accumulated into the same
    vector.  Both run on the current stream, so the torch adds and the kernel
    atomics compose correctly.
    """

    num_segments = len(buckets)
    local_sq = torch.zeros(num_segments, device=device, dtype=torch.float32)
    kernel_tensors, kernel_segments, torch_pairs = _split_by_kernel_eligibility(
        buckets, device, ext
    )

    for grad, segment in torch_pairs:
        local_sq[segment] += (
            grad.detach().float().pow(2).sum().to(device=device, dtype=torch.float32)
        )

    if kernel_tensors:
        metadata = _build_metadata(
            kernel_tensors, kernel_segments, device=device, chunk_size=chunk_size
        )
        ext.segmented_l2_norm_sq(
            metadata["ptrs"],
            metadata["numels"],
            metadata["dtypes"],
            metadata["segments"],
            metadata["job_tensor_ids"],
            metadata["job_offsets"],
            local_sq,
            chunk_size,
        )
    return local_sq


def _segmented_scale(
    buckets: Sequence[GradClipBucket],
    coefs: torch.Tensor,
    *,
    device: torch.device,
    ext,
    chunk_size: int,
) -> None:
    """Scale each grad in place by its bucket coefficient (no collective)."""

    kernel_tensors, kernel_segments, torch_pairs = _split_by_kernel_eligibility(
        buckets, device, ext
    )

    for grad, segment in torch_pairs:
        grad.mul_(coefs[segment].to(device=grad.device))

    if kernel_tensors:
        metadata = _build_metadata(
            kernel_tensors, kernel_segments, device=device, chunk_size=chunk_size
        )
        ext.segmented_scale(
            metadata["ptrs"],
            metadata["numels"],
            metadata["dtypes"],
            metadata["segments"],
            metadata["job_tensor_ids"],
            metadata["job_offsets"],
            coefs,
            chunk_size,
        )


def _apply_segment_reductions(
    local_sq: torch.Tensor,
    reduce_flags: Sequence[int],
    process_group=None,
) -> torch.Tensor:
    """The single collective: sum reduce=True segments across ranks.

    Issued unconditionally whenever the process group is live and any bucket
    reduces -- both conditions are derived identically on every rank, so the
    collective shape is rank-uniform.
    """

    if not (
        dist.is_available()
        and dist.is_initialized()
        and any(bool(flag) for flag in reduce_flags)
    ):
        return local_sq

    reduce_mask = torch.tensor(
        reduce_flags,
        device=local_sq.device,
        dtype=torch.bool,
    )
    reduced_sq = torch.where(reduce_mask, local_sq, torch.zeros_like(local_sq))
    dist.all_reduce(reduced_sq, op=dist.ReduceOp.SUM, group=process_group)
    return torch.where(reduce_mask, reduced_sq, local_sq)


def _build_stats(
    buckets: Sequence[GradClipBucket],
    total_norms: torch.Tensor,
    coefs: torch.Tensor,
    *,
    max_norm: float | None,
    fastpath: bool,
) -> tuple[GradClipBucketStats, ...]:
    """Assemble per-bucket stats with a single batched device-to-host copy."""

    if len(buckets) == 0:
        return tuple()

    stacked = torch.stack(
        [total_norms, coefs, torch.isfinite(total_norms).to(total_norms.dtype)]
    )
    host = stacked.detach().to("cpu")
    total_values = host[0].tolist()
    coef_values = host[1].tolist()
    finite_values = host[2].tolist()

    stats: list[GradClipBucketStats] = []
    for idx, bucket in enumerate(buckets):
        total = float(total_values[idx])
        coef = float(coef_values[idx])
        finite = bool(finite_values[idx])
        clipped = bool(max_norm is not None and finite and total > float(max_norm))
        param_count = sum(1 for grad in bucket.grads if grad is not None)
        stats.append(
            GradClipBucketStats(
                name=bucket.name,
                total_norm=total,
                max_norm=None if max_norm is None else float(max_norm),
                clip_coef=coef,
                clipped=clipped,
                param_count=param_count,
                found_inf=not finite,
                fastpath=fastpath,
            )
        )
    return tuple(stats)


@torch.no_grad()
def clip_grad_norm_buckets_(
    buckets: Sequence[GradClipBucket],
    max_norm: float | None,
    *,
    process_group=None,
    chunk_size: int | None = None,
) -> SegmentedGradClipResult:
    """Clip gradient buckets with segment-local, distributed-safe semantics.

    Each bucket keeps its own gradient norm and clip coefficient.  The CUDA
    kernels accelerate the eligible tensors; ineligible tensors (or all of them
    when the extension is unavailable) use the torch reference math.  The
    fast/slow choice is per tensor and purely local: it never changes the
    number or shape of collective operations, so this is safe to call from
    every rank in lockstep.
    """

    device = _resolve_device(buckets)
    ext = _load_fast_clip_extension()
    resolved_chunk_size = _resolve_chunk_size(chunk_size)
    reduce_flags = [1 if bucket.reduce else 0 for bucket in buckets]

    # Stage 1 -- local per-bucket squared norm (no collective).
    local_sq = _segmented_local_sq(
        buckets, device=device, ext=ext, chunk_size=resolved_chunk_size
    )

    # Stage 2 -- the single, rank-uniform collective.
    global_sq = _apply_segment_reductions(local_sq, reduce_flags, process_group)

    # Stage 3 -- total norm + coefficient (torch, matches reference semantics).
    total_norms = torch.sqrt(global_sq)
    if max_norm is None:
        coefs = torch.ones_like(total_norms)
    else:
        coefs = torch.clamp(float(max_norm) / (total_norms + 1e-6), max=1.0)

    # Stage 4 -- scale in place (no collective). found_inf never gates this.
    if max_norm is not None:
        _segmented_scale(
            buckets, coefs, device=device, ext=ext, chunk_size=resolved_chunk_size
        )

    # Stage 5 -- stats (single batched D2H).
    stats = _build_stats(
        buckets, total_norms, coefs, max_norm=max_norm, fastpath=ext is not None
    )
    combined_total = torch.linalg.vector_norm(total_norms, 2.0)
    return SegmentedGradClipResult(
        combined_total,
        stats,
        fastpath=ext is not None,
        fallback_reason=None if ext is not None else "cuda_extension_unavailable",
    )


@torch.no_grad()
def try_clip_optimizer_grad_norm_buckets_(
    optimizer,
    *,
    regular_grads: Sequence[torch.Tensor],
    adamw_grads: Sequence[torch.Tensor],
    max_norm: float,
    process_group=None,
) -> SegmentedGradClipResult | None:
    """Fast path for Wall-X style regular/Muon/dedicated-AdamW clipping.

    Returns ``None`` only when the fast-clip CUDA extension is entirely
    unavailable, letting the caller run its own path.  Extension availability
    is a property of the install and is therefore uniform across ranks, so this
    ``None`` decision is rank-consistent and cannot desync collectives.  Once a
    result is produced, clipping goes through the distributed-safe
    :func:`clip_grad_norm_buckets_`.
    """

    if _load_fast_clip_extension() is None:
        return None

    ensure = getattr(optimizer, "_ensure_grads_ready", None)
    if ensure is None:
        return None
    try:
        ensure(coalesce_wait=True)
    except TypeError:
        ensure()

    muon_entries = tuple(_iter_muon_grad_entries(optimizer))
    muon_grads = [entry.grad for entry in muon_entries]
    result = clip_grad_norm_buckets_(
        (
            GradClipBucket("regular", regular_grads, reduce=False),
            GradClipBucket("muon", muon_grads, reduce=True),
            GradClipBucket("adamw", adamw_grads, reduce=True),
        ),
        max_norm,
        process_group=process_group,
    )

    muon_stats = result.stats_by_name.get("muon")
    if muon_stats is not None:
        optimizer._last_muon_grad_clip_stats = MuonGradClipStats(
            total_norm=muon_stats.total_norm,
            max_norm=muon_stats.max_norm,
            norm_type=2.0,
            clip_coef=muon_stats.clip_coef,
            clipped=muon_stats.clipped,
            param_count=muon_stats.param_count,
            found_inf=muon_stats.found_inf,
            strategy="segmented_cuda_fastpath",
        )
    return result
