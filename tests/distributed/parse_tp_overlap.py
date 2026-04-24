"""Parse a torch.profiler Chrome-trace JSON and compute per-stream NCCL
overlap metrics for the TP gather / scatter paths.

Primary question: when ``tp_gather_grads`` dispatches NCCL collectives on
``reduce_stream``, is ``compute_stream`` concurrently executing backward
compute kernels?  We answer with numerical ratios; a visual timeline can
be obtained by loading the JSON in ``chrome://tracing`` or Nsight.

Usage:
    python parse_tp_overlap.py /tmp/tp_overlap.pt.trace.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Iterable


def load_events(path: str) -> list[dict]:
    with open(path) as f:
        blob = json.load(f)
    return blob.get("traceEvents", [])


def kernel_events(events: Iterable[dict]) -> list[dict]:
    """Return ``ph=='X'`` (complete) CUDA kernel events with ts+dur set."""
    out: list[dict] = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        # torch.profiler uses ``cat='kernel'`` for device kernels and
        # ``cat='cpu_op'`` for host-side ops; accept also the newer
        # ``cat='gpu_op'`` / ``cat='cuda_runtime'`` NCCL hooks.
        if "kernel" not in cat.lower() and "gpu" not in cat.lower():
            continue
        if e.get("dur") is None or e.get("ts") is None:
            continue
        out.append(e)
    return out


def group_by_stream(kernels: list[dict]) -> dict[int, list[dict]]:
    """Group by the ``stream`` field in args (or fall back to ``tid``)."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for e in kernels:
        args = e.get("args", {})
        stream = args.get("stream")
        if stream is None:
            # Newer profiler puts the stream id in ``tid`` for kernel events
            stream = e.get("tid", -1)
        groups[int(stream)].append(e)
    return groups


def is_nccl(e: dict) -> bool:
    name = e.get("name", "").lower()
    return (
        "nccl" in name
        or "ncclallreduce" in name
        or "ncclreduce" in name
        or "ncclbroadcast" in name
        or "ncclscatter" in name
        or "ncclgather" in name
        or "ncclsend" in name
        or "ncclrecv" in name
        or "all_reduce" in name
        or "broadcast" in name
    )


def total_dur(events: list[dict]) -> float:
    return sum(e["dur"] for e in events)


def span(events: list[dict]) -> tuple[float, float]:
    if not events:
        return (0.0, 0.0)
    lo = min(e["ts"] for e in events)
    hi = max(e["ts"] + e["dur"] for e in events)
    return (lo, hi)


def merge_intervals(events: list[dict]) -> list[tuple[float, float]]:
    """Return sorted, merged (start, end) intervals covered by these events."""
    intervals = sorted(
        ((e["ts"], e["ts"] + e["dur"]) for e in events),
        key=lambda t: t[0],
    )
    merged: list[tuple[float, float]] = []
    for a, b in intervals:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def intersect_length(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """Total length of the intersection of two merged-intervals sets."""
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: parse_tp_overlap.py <trace.json>", file=sys.stderr)
        return 2
    events = load_events(sys.argv[1])
    kernels = kernel_events(events)
    print(f"total kernel events: {len(kernels)}")

    groups = group_by_stream(kernels)
    print(f"streams observed: {sorted(groups.keys())}")
    print()

    # Per-stream stats
    rows: list[tuple[int, int, int, float, float]] = []
    for s, evs in groups.items():
        nccl = [e for e in evs if is_nccl(e)]
        non_nccl = [e for e in evs if not is_nccl(e)]
        rows.append((s, len(evs), len(nccl), total_dur(nccl), total_dur(non_nccl)))
    print(f"{'stream':>8} {'#kernels':>10} {'#nccl':>8} {'nccl_us':>12} {'compute_us':>14}")
    for r in sorted(rows, key=lambda x: -(x[3] + x[4])):
        print(f"{r[0]:>8} {r[1]:>10} {r[2]:>8} {r[3]:>12.1f} {r[4]:>14.1f}")
    print()

    # Heuristic: the stream with the most NCCL time is NOT the compute
    # stream.  Among the non-NCCL-heaviest streams, pick the one with the
    # most non-nccl kernels as the compute stream.
    by_nccl_desc = sorted(rows, key=lambda x: -x[3])
    comm_streams = [r for r in by_nccl_desc if r[3] > 0]
    if not comm_streams:
        print("no NCCL activity detected; overlap metric N/A")
        return 0

    compute_row = max(
        (r for r in rows if r not in comm_streams[:2]),
        key=lambda r: r[4],
        default=None,
    )
    if compute_row is None:
        compute_row = max(rows, key=lambda r: r[4])

    print(f"assumed compute stream: {compute_row[0]} "
          f"(non-nccl dur {compute_row[4]:.0f} us)")
    print(f"top NCCL streams: {[c[0] for c in comm_streams[:3]]}")
    print()

    compute_evs = [e for e in groups[compute_row[0]] if not is_nccl(e)]
    compute_iv = merge_intervals(compute_evs)
    print(f"compute stream busy window total: {sum(b-a for a,b in compute_iv):.1f} us")
    print()

    # For each stream with any NCCL activity, compute overlap with compute.
    print(f"{'stream':>8} {'nccl_us':>12} {'overlap_us':>14} {'overlap_%':>12}")
    for r in sorted(comm_streams, key=lambda r: -r[3]):
        nccl_evs = [e for e in groups[r[0]] if is_nccl(e)]
        if not nccl_evs:
            continue
        nccl_iv = merge_intervals(nccl_evs)
        total = sum(b - a for a, b in nccl_iv)
        overlap = intersect_length(nccl_iv, compute_iv)
        pct = (overlap / total * 100.0) if total > 0 else float("nan")
        print(f"{r[0]:>8} {total:>12.1f} {overlap:>14.1f} {pct:>11.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
