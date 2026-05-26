"""Gradient clipping for DMuon-owned Muon parameters.

PyTorch's ``torch.nn.utils.clip_grad_norm_`` only sees tensors in
``param.grad``.  DMuon dedicated parameters keep their reduced gradients on
``DedicatedParam`` objects instead, so Muon parameters need a small
DMuon-aware clipping entry point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.distributed as dist

try:
    from torch.utils._foreach_utils import (
        _device_has_foreach_support,
        _group_tensors_by_device_and_dtype,
        _has_foreach_support,
    )
except Exception:  # pragma: no cover - older torch fallback
    _device_has_foreach_support = None
    _group_tensors_by_device_and_dtype = None
    _has_foreach_support = None


@dataclass(frozen=True)
class MuonGradClipEntry:
    """One Muon gradient tensor eligible for DMuon clipping."""

    name: str
    grad: torch.Tensor
    shape: tuple[int, ...]
    is_tp_full_grad: bool = False


@dataclass(frozen=True)
class MuonGradClipStats:
    """Statistics returned by :func:`clip_grad_norm_`.

    ``total_norm`` is the norm before clipping.  ``max_norm=None`` means the
    call was stats-only and did not scale any gradient.
    """

    total_norm: float
    max_norm: float | None
    norm_type: float
    clip_coef: float
    clipped: bool
    param_count: int
    found_inf: bool
    strategy: str = "global_norm"

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_norm": self.total_norm,
            "max_norm": self.max_norm,
            "norm_type": self.norm_type,
            "clip_coef": self.clip_coef,
            "clipped": self.clipped,
            "param_count": self.param_count,
            "found_inf": self.found_inf,
            "strategy": self.strategy,
        }


@dataclass(frozen=True)
class MuonGradClipContext:
    """Context passed to Muon grad clip strategies.

    Future strategies such as MuonClip/QK-Clip can use ``entries`` to inspect
    parameter names, shapes, TP ownership, and the live gradient tensors.
    """

    optimizer: Any
    entries: tuple[MuonGradClipEntry, ...]
    max_norm: float | None
    norm_type: float
    error_if_nonfinite: bool
    foreach: bool | None = None


class MuonGradClipStrategy(Protocol):
    """Protocol for pluggable Muon gradient clipping strategies."""

    name: str

    def __call__(self, context: MuonGradClipContext) -> MuonGradClipStats:
        ...


class GlobalNormMuonClipStrategy:
    """Clip all DMuon-owned Muon gradients by one global p-norm coefficient."""

    name = "global_norm"

    @torch.no_grad()
    def __call__(self, context: MuonGradClipContext) -> MuonGradClipStats:
        total_norm_t = _compute_total_norm(
            context.optimizer,
            context.entries,
            context.norm_type,
            error_if_nonfinite=context.error_if_nonfinite,
            foreach=context.foreach,
        )

        max_norm = context.max_norm
        clip_coef = 1.0
        clipped = False
        if max_norm is not None:
            if max_norm < 0:
                raise ValueError(f"max_norm must be non-negative, got {max_norm}")
            clip_coef_t = float(max_norm) / (total_norm_t + 1e-6)
            # Match PyTorch clip_grad_norm_: always multiply by the clamped
            # coefficient to avoid a device-to-host sync branch.
            clip_coef_clamped = torch.clamp(clip_coef_t, max=1.0)
            _clip_entries_with_norm_(context.entries, clip_coef_clamped, context.foreach)
            raw_coef = float(clip_coef_t.detach().cpu().item())
            clip_coef = min(raw_coef, 1.0) if math.isfinite(raw_coef) else raw_coef
            clipped = math.isfinite(raw_coef) and raw_coef < 1.0
        total_norm = float(total_norm_t.detach().cpu().item())
        found_inf = not math.isfinite(total_norm)

        return MuonGradClipStats(
            total_norm=total_norm,
            max_norm=max_norm,
            norm_type=context.norm_type,
            clip_coef=clip_coef,
            clipped=clipped,
            param_count=len(context.entries),
            found_inf=found_inf,
            strategy=self.name,
        )


_STRATEGIES: dict[str, MuonGradClipStrategy] = {
    GlobalNormMuonClipStrategy.name: GlobalNormMuonClipStrategy(),
}


def register_muon_grad_clip_strategy(name: str, strategy: MuonGradClipStrategy) -> None:
    """Register a custom Muon grad clip strategy.

    The strategy receives live DMuon gradient tensors and may scale or inspect
    them in place.  This is intentionally small so specialized schemes can be
    added without changing the public training-loop API.
    """

    if not name:
        raise ValueError("strategy name must be non-empty")
    _STRATEGIES[name] = strategy


def clip_grad_norm_(
    optimizer: Any,
    max_norm: float | None,
    *,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    strategy: str | MuonGradClipStrategy = "global_norm",
) -> MuonGradClipStats:
    """Clip DMuon-owned Muon gradients only.

    This does **not** touch AdamW/non-dedicated parameters.  Training
    frameworks should keep using PyTorch's native clipping for ordinary
    ``param.grad`` tensors, and call this helper as the extra DMuon line.
    """

    ensure = getattr(optimizer, "_ensure_grads_ready", None)
    if ensure is None or not hasattr(optimizer, "_dedicated_params"):
        raise TypeError("dmuon.clip_grad_norm_ expects a dmuon.Muon optimizer")
    ensure()
    stats = _clip_ready_muon_grad_norm_(
        optimizer,
        max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite,
        foreach=foreach,
        strategy=strategy,
    )
    optimizer._last_muon_grad_clip_stats = stats
    return stats


def _clip_ready_muon_grad_norm_(
    optimizer: Any,
    max_norm: float | None,
    *,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    strategy: str | MuonGradClipStrategy = "global_norm",
) -> MuonGradClipStats:
    """Internal variant for callers that already ran ``_ensure_grads_ready``."""

    if norm_type <= 0 and norm_type != float("inf"):
        raise ValueError(f"norm_type must be positive or inf, got {norm_type}")
    entries = tuple(_iter_muon_grad_entries(optimizer))
    resolved = _resolve_strategy(strategy)
    context = MuonGradClipContext(
        optimizer=optimizer,
        entries=entries,
        max_norm=None if max_norm is None else float(max_norm),
        norm_type=float(norm_type),
        error_if_nonfinite=error_if_nonfinite,
        foreach=foreach,
    )
    stats = resolved(context)
    optimizer._last_muon_grad_clip_stats = stats
    return stats


def _resolve_strategy(strategy: str | MuonGradClipStrategy) -> MuonGradClipStrategy:
    if isinstance(strategy, str):
        try:
            return _STRATEGIES[strategy]
        except KeyError as exc:
            valid = ", ".join(sorted(_STRATEGIES))
            raise ValueError(
                f"unknown Muon grad clip strategy {strategy!r}; available: {valid}"
            ) from exc
    return strategy


def _iter_muon_grad_entries(optimizer: Any):
    muon_dp_ids = set(getattr(optimizer, "_dp_to_muon_group_idx", {}).keys())
    for idx, dp in enumerate(getattr(optimizer, "_dedicated_params", [])):
        if muon_dp_ids and id(dp) not in muon_dp_ids:
            continue
        is_tp = bool(getattr(dp, "is_dtensor", False) and getattr(dp, "tp_group", None) is not None)
        if is_tp:
            if not bool(getattr(dp, "is_tp_owner", False)):
                continue
            grad = getattr(dp, "_tp_full_grad", None)
            is_tp_full_grad = True
        else:
            grad = getattr(dp, "_reduced_grad", None)
            is_tp_full_grad = False
        if grad is None:
            continue
        name = str(getattr(dp, "param_name", f"dedicated_param_{idx}"))
        yield MuonGradClipEntry(
            name=name,
            grad=grad,
            shape=tuple(int(dim) for dim in grad.shape),
            is_tp_full_grad=is_tp_full_grad,
        )


def _compute_total_norm(
    optimizer: Any,
    entries: tuple[MuonGradClipEntry, ...],
    norm_type: float,
    *,
    error_if_nonfinite: bool,
    foreach: bool | None,
) -> torch.Tensor:
    if norm_type == float("inf"):
        local = _local_total_norm(entries, norm_type, optimizer, foreach=foreach)
        _all_reduce_if_needed(local, op=dist.ReduceOp.MAX)
        total_norm = local
    else:
        local = _local_total_norm(entries, norm_type, optimizer, foreach=foreach)
        local_power = local.pow(norm_type)
        _all_reduce_if_needed(local_power, op=dist.ReduceOp.SUM)
        total_norm = local_power.pow(1.0 / norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for DMuon Muon gradients is "
            "non-finite, so it cannot be clipped. To disable this error and "
            "scale the gradients by the non-finite norm anyway, set "
            "`error_if_nonfinite=False`."
        )
    return total_norm


def _local_total_norm(
    entries: tuple[MuonGradClipEntry, ...],
    norm_type: float,
    optimizer: Any,
    *,
    foreach: bool | None,
) -> torch.Tensor:
    tensors = [entry.grad.detach() for entry in entries]
    if len(tensors) == 0:
        return torch.zeros((), device=_accumulator_device(optimizer, entries), dtype=torch.float32)
    first_device = tensors[0].device

    norms: list[torch.Tensor] = []
    for (device, _dtype), device_tensors in _group_tensors(tensors).items():
        if _use_foreach(device_tensors, device, foreach):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend([torch.linalg.vector_norm(t, norm_type) for t in device_tensors])
    return torch.linalg.vector_norm(
        torch.stack([norm.to(device=first_device, dtype=torch.float32) for norm in norms]),
        norm_type,
    )


def _clip_entries_with_norm_(
    entries: tuple[MuonGradClipEntry, ...],
    clip_coef_clamped: torch.Tensor,
    foreach: bool | None,
) -> None:
    tensors = [entry.grad for entry in entries]
    if len(tensors) == 0:
        return
    for (device, _dtype), device_tensors in _group_tensors(tensors).items():
        if _use_foreach(device_tensors, device, foreach):
            torch._foreach_mul_(device_tensors, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            coef = clip_coef_clamped.to(device)
            for tensor in device_tensors:
                tensor.mul_(coef)


def _group_tensors(
    tensors: list[torch.Tensor],
) -> dict[tuple[torch.device, torch.dtype], list[torch.Tensor]]:
    if _group_tensors_by_device_and_dtype is None:
        grouped: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
        for tensor in tensors:
            grouped.setdefault((tensor.device, tensor.dtype), []).append(tensor)
        return grouped

    grouped_by_device_dtype = _group_tensors_by_device_and_dtype([tensors])
    grouped: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for (device, _dtype), ([device_tensors], _indices) in grouped_by_device_dtype.items():
        grouped.setdefault((device, _dtype), []).extend(device_tensors)
    return grouped


def _use_foreach(
    tensors: list[torch.Tensor],
    device: torch.device,
    foreach: bool | None,
) -> bool:
    if (
        _has_foreach_support is None
        or _device_has_foreach_support is None
        or not hasattr(torch, "_foreach_norm")
    ):
        return False
    if foreach is None:
        return bool(_has_foreach_support(tensors, device))
    return bool(foreach and _device_has_foreach_support(device))


def _accumulator_device(
    optimizer: Any,
    entries: tuple[MuonGradClipEntry, ...],
) -> torch.device:
    if entries:
        return entries[0].grad.device
    comm_ctx = getattr(optimizer, "_comm_ctx", None)
    device = getattr(comm_ctx, "device", None)
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _all_reduce_if_needed(tensor: torch.Tensor, *, op: dist.ReduceOp) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=op)
