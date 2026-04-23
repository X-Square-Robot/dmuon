"""Enhanced parser: per-stream NCCL op-name histogram + overlap.

Helps attribute which stream corresponds to DP reduce, TP gather,
replicate broadcast, etc.  Feeds the T2c overlap report.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def load(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f).get("traceEvents", [])


def kernels(evs):
    for e in evs:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "").lower()
        if "kernel" not in cat and "gpu" not in cat:
            continue
        if e.get("dur") is None:
            continue
        yield e


def main():
    path = sys.argv[1]
    evs = load(path)
    by_stream: dict[int, list[dict]] = defaultdict(list)
    for e in kernels(evs):
        sid = e.get("args", {}).get("stream", e.get("tid", -1))
        by_stream[int(sid)].append(e)

    def is_nccl(name: str) -> bool:
        n = name.lower()
        return any(k in n for k in (
            "nccl", "all_reduce", "broadcast", "reduce_scatter",
            "all_gather", "scatter", "gather", "send", "recv",
        ))

    print(f"{'stream':>8} {'#k':>6} {'#nccl':>6} {'nccl_us':>10} {'compute_us':>12}  top_nccl_ops")
    for sid in sorted(by_stream):
        evs_s = by_stream[sid]
        nccl = [e for e in evs_s if is_nccl(e.get("name", ""))]
        other = [e for e in evs_s if not is_nccl(e.get("name", ""))]
        if not evs_s:
            continue
        names = Counter()
        for e in nccl:
            # Truncate long mangled NCCL kernel names
            nm = e["name"][:40]
            names[nm] += 1
        top = ", ".join(f"{n}x{c}" for n, c in names.most_common(3))
        print(
            f"{sid:>8} {len(evs_s):>6} {len(nccl):>6} "
            f"{sum(e['dur'] for e in nccl):>10.1f} "
            f"{sum(e['dur'] for e in other):>12.1f}  {top}"
        )


if __name__ == "__main__":
    main()
