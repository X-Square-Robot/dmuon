"""Analyze TP collectives and sparse regions in a torch profiler trace.

The script is intentionally independent from benchmark execution.  It is used
to verify whether a TP implementation actually reduces per-layer all-reduce
traffic, instead of relying on visual inspection in Chrome trace.

Example:
    python benchmarks/analyze_tp_trace.py rank0.trace.json --layers 32 \
        --json-out trace_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def _load_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return list(payload.get("traceEvents", []))
    return list(payload)


def _kernel_class(event: dict[str, Any]) -> str:
    name = str(event.get("name", "")).lower()
    cat = event.get("cat")
    if "nccl" in name:
        return "nccl"
    if any(x in name for x in ("gemm", "cutlass", "cublas", "sm90", "hgemm")):
        return "gemm"
    if any(x in name for x in ("fmha", "flash", "attention")):
        return "attention"
    if "softmax" in name or "nll_loss" in name:
        return "loss"
    if "vectorized_elementwise" in name:
        return "vectorized_elementwise"
    if "elementwise" in name:
        return "elementwise"
    if "reduce" in name:
        return "reduce"
    if cat == "gpu_memset" or "memset" in name:
        return "memset"
    if cat == "gpu_memcpy" or "memcpy" in name:
        return "memcpy"
    if "multi_tensor" in name:
        return "multi_tensor"
    return "other"


def _is_big_compute(event: dict[str, Any], threshold_us: float) -> bool:
    if float(event.get("dur", 0.0)) < threshold_us:
        return False
    klass = _kernel_class(event)
    return klass in {"gemm", "attention"}


def _merged_global_gaps(events: list[dict[str, Any]]) -> dict[str, Any]:
    intervals = sorted(
        (
            float(event["ts"]),
            float(event["ts"]) + float(event.get("dur", 0.0)),
        )
        for event in events
        if event.get("ph") == "X" and event.get("cat") in GPU_CATS
    )
    if not intervals:
        return {"max_gap_us": 0.0, "gap_count_gt_100us": 0, "gap_count_gt_1ms": 0}
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    gaps = [merged[i][0] - merged[i - 1][1] for i in range(1, len(merged))]
    return {
        "max_gap_us": max(gaps) if gaps else 0.0,
        "gap_count_gt_100us": sum(gap > 100.0 for gap in gaps),
        "gap_count_gt_1ms": sum(gap > 1000.0 for gap in gaps),
    }


def analyze_trace(
    trace_path: Path,
    *,
    layers: int | None,
    big_compute_threshold_us: float,
    big_compute_gap_threshold_us: float,
) -> dict[str, Any]:
    events = _load_events(trace_path)
    step_names = {
        str(event.get("name"))
        for event in events
        if event.get("cat") == "user_annotation"
        and str(event.get("name", "")).startswith("dmuon.profile.step_")
    }
    profile_step_count = max(1, len(step_names))
    gpu_events = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") in GPU_CATS
    ]
    if not gpu_events:
        return {
            "trace": str(trace_path),
            "gpu_event_count": 0,
            "error": "no GPU events found",
        }

    start_ts = min(float(event["ts"]) for event in gpu_events)
    end_ts = max(float(event["ts"]) + float(event.get("dur", 0.0)) for event in gpu_events)
    nccl_events = [
        event
        for event in gpu_events
        if "ncclDevKernel_AllReduce" in str(event.get("name", ""))
    ]
    klass_counts = Counter(_kernel_class(event) for event in gpu_events)
    klass_durations = defaultdict(float)
    for event in gpu_events:
        klass_durations[_kernel_class(event)] += float(event.get("dur", 0.0))

    big_compute_events = [
        event for event in gpu_events if _is_big_compute(event, big_compute_threshold_us)
    ]
    big_compute_events.sort(key=lambda event: float(event["ts"]))
    big_compute_gaps: list[dict[str, Any]] = []
    for prev, nxt in zip(big_compute_events, big_compute_events[1:]):
        gap_start = float(prev["ts"]) + float(prev.get("dur", 0.0))
        gap_end = float(nxt["ts"])
        gap = gap_end - gap_start
        if gap < big_compute_gap_threshold_us:
            continue
        in_window = [
            event
            for event in gpu_events
            if float(event["ts"]) < gap_end
            and float(event["ts"]) + float(event.get("dur", 0.0)) > gap_start
        ]
        counts = Counter(_kernel_class(event) for event in in_window)
        durations = defaultdict(float)
        for event in in_window:
            durations[_kernel_class(event)] += float(event.get("dur", 0.0))
        big_compute_gaps.append(
            {
                "gap_us": gap,
                "start_ms_from_first_gpu_event": (gap_start - start_ts) / 1000.0,
                "end_ms_from_first_gpu_event": (gap_end - start_ts) / 1000.0,
                "op_count": len(in_window),
                "kernel_counts": dict(counts.most_common()),
                "kernel_durations_ms": {
                    key: val / 1000.0
                    for key, val in sorted(
                        durations.items(), key=lambda item: item[1], reverse=True
                    )
                },
            }
        )
    big_compute_gaps.sort(key=lambda item: float(item["gap_us"]), reverse=True)

    allreduce_per_layer_step = None
    if layers:
        allreduce_per_layer_step = len(nccl_events) / float(layers * profile_step_count)
    return {
        "trace": str(trace_path),
        "layers": layers,
        "profile_step_count": profile_step_count,
        "gpu_event_count": len(gpu_events),
        "gpu_span_ms": (end_ts - start_ts) / 1000.0,
        "allreduce_count": len(nccl_events),
        "allreduce_per_layer_per_step": allreduce_per_layer_step,
        "nccl_duration_ms": sum(float(event.get("dur", 0.0)) for event in nccl_events)
        / 1000.0,
        "kernel_counts": dict(klass_counts.most_common()),
        "kernel_durations_ms": {
            key: val / 1000.0
            for key, val in sorted(
                klass_durations.items(), key=lambda item: item[1], reverse=True
            )
        },
        "global_idle": _merged_global_gaps(gpu_events),
        "big_compute_gap_threshold_us": big_compute_gap_threshold_us,
        "big_compute_gaps": big_compute_gaps[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--big-compute-threshold-us", type=float, default=80.0)
    parser.add_argument("--big-compute-gap-threshold-us", type=float, default=1000.0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    result = analyze_trace(
        args.trace,
        layers=args.layers,
        big_compute_threshold_us=args.big_compute_threshold_us,
        big_compute_gap_threshold_us=args.big_compute_gap_threshold_us,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
