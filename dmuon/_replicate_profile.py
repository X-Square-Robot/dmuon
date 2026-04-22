"""Phase C.7 profile infrastructure for the async replicate broadcast.

Gated entirely behind :envvar:`DMUON_REPLICATE_PROFILE`:

    unset / "0"   — fully disabled; zero overhead in the hot path
    "1"           — per-group wait-time sampled in ``_pre_forward_wait``
                   and reported from rank 0 via :func:`report`
    "2"           — also NSight range markers around dispatch / wait

The measurement itself lives in ``DedicatedParamGroup._pre_forward_wait``
(so the timing is consistent with the actual wait site); this module
collects the samples + renders the report.  Mirrors the structure of
``_balance_profile`` but specialised for replicate-stream telemetry.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


def _level() -> int:
    raw = os.environ.get("DMUON_REPLICATE_PROFILE", "0") or "0"
    try:
        return int(raw)
    except ValueError:
        return 0


def enabled() -> bool:
    return _level() >= 1


def nsight_markers_enabled() -> bool:
    return _level() >= 2


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


# ---- Sample collection ----------------------------------------------------


class ReplicateProfile:
    """Accumulates per-group wait samples until ``report()`` is called."""

    def __init__(self):
        # group_id (int) → list[float μs]
        self.wait_samples_us: dict[int, list[float]] = {}
        self.group_labels: dict[int, str] = {}
        self.fallback_events: list[tuple[int, str]] = []  # (step, group_label)

    def record_wait(self, group, wait_us: float, step: Optional[int] = None) -> None:
        gid = id(group)
        if gid not in self.wait_samples_us:
            self.wait_samples_us[gid] = []
            label = getattr(group, "_profile_label", None) or f"group_{gid:x}"
            self.group_labels[gid] = label
        self.wait_samples_us[gid].append(wait_us)

    def record_fallback(self, group, step: int) -> None:
        label = self.group_labels.get(id(group), f"group_{id(group):x}")
        self.fallback_events.append((step, label))

    # ------------------------------------------------------------------ API

    def report(self) -> None:
        """Render a per-group histogram from rank 0 only."""
        if not enabled() or _rank() != 0:
            return
        if not self.wait_samples_us:
            print("[DMUON_REPLICATE_PROFILE] no samples collected", flush=True)
            return
        print("\n" + "=" * 78)
        print("[DMUON_REPLICATE_PROFILE] per-group wait time summary (μs)")
        print("=" * 78)
        hdr = (
            f"  {'group':>28} {'n':>5} {'mean':>9} {'p50':>9} "
            f"{'p90':>9} {'p99':>9} {'max':>9}"
        )
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for gid, samples in sorted(
            self.wait_samples_us.items(),
            key=lambda kv: self.group_labels[kv[0]],
        ):
            label = self.group_labels[gid]
            if not samples:
                continue
            samples_sorted = sorted(samples)
            n = len(samples_sorted)
            mean = sum(samples_sorted) / n
            p50 = samples_sorted[n // 2]
            p90 = samples_sorted[min(n - 1, int(n * 0.9))]
            p99 = samples_sorted[min(n - 1, int(n * 0.99))]
            mx = samples_sorted[-1]
            print(
                f"  {label:>28} {n:>5d} {mean:>9.2f} {p50:>9.2f} "
                f"{p90:>9.2f} {p99:>9.2f} {mx:>9.2f}"
            )
        if self.fallback_events:
            print(f"\n  Fallback events: {len(self.fallback_events)}")
            for step, label in self.fallback_events[:10]:
                print(f"    step {step:>5d}  {label}")
            if len(self.fallback_events) > 10:
                print(f"    ... (+{len(self.fallback_events) - 10} more)")
        print("=" * 78 + "\n", flush=True)


_GLOBAL_PROFILE: Optional[ReplicateProfile] = None


def get_profile() -> ReplicateProfile:
    """Module-level singleton for scripts that want to call
    :func:`report` without plumbing the profile through their code."""
    global _GLOBAL_PROFILE
    if _GLOBAL_PROFILE is None:
        _GLOBAL_PROFILE = ReplicateProfile()
    return _GLOBAL_PROFILE


def reset() -> None:
    global _GLOBAL_PROFILE
    _GLOBAL_PROFILE = None


# ---- Collection hook called from DedicatedParamGroup ----------------------


def record_wait_from_group(group, wait_us: float) -> None:
    if not enabled() or wait_us <= 0.0:
        return
    get_profile().record_wait(group, wait_us)


def record_fallback_from_group(group, step: int) -> None:
    if not enabled():
        return
    get_profile().record_fallback(group, step)


# ---- Convenience: collect all samples from a model ------------------------


def replicate_profile_report(model: nn.Module) -> None:
    """Print the rank-0 per-group replicate-broadcast wait time report.

    Gated entirely by :envvar:`DMUON_REPLICATE_PROFILE`. When unset or ``"0"``
    this is a cheap no-op; set ``DMUON_REPLICATE_PROFILE=1`` at process
    start to enable sample collection in ``_pre_forward_wait``.

    The report shows n / mean / p50 / p90 / p99 / max wait times (μs) per
    DMuon group, plus fallback events if any were triggered. Intended to
    be called once at the end of training (or at checkpoint time) to
    diagnose async broadcast hiding quality.

    Args:
        model: The model with DMuon-attached groups. Only used here to
            match the API shape of other public utilities; the actual
            profile state is module-global.

    When distributed is initialised, only rank 0 prints; other ranks skip.
    """
    get_profile().report()


# Back-compat alias for existing imports (pre-rename 2026-04-22).
collect_and_report = replicate_profile_report
