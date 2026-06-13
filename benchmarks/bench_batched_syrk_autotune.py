"""Autotune batched SYRK owner-work timings over the repo SM80 config space.

This is a thin driver around ``bench_batched_syrk_workload.py``.  It runs the
same owner-local Muon compute workload for every eligible
``SYRK_SM80_CONFIGS`` entry, picks the fastest config per (shape, batch), and
optionally reports batched owner-work assignment from those best timings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from benchmarks.batch_assignment_ilp import (  # noqa: E402
    ShapeTiming,
    solve_assignment_search,
)
from benchmarks import bench_batched_syrk_workload as workload_bench  # noqa: E402
from benchmarks.bench_batched_syrk_workload import (  # noqa: E402
    HAS_SYRK_SM80,
    WORKLOAD,
    benchmark_one,
    gram_shape,
    parse_batch_list,
)
from dmuon.kernels.syrk_sm80 import SYRK_SM80_CONFIGS  # noqa: E402


DETAIL_FIELDS = [
    "method",
    "name",
    "original_shape",
    "gram_shape",
    "count",
    "batch",
    "tile_m",
    "tile_k",
    "num_stages",
    "batch_total_ms",
    "single_task_ms",
    "median_ms",
    "mean_ms",
    "p20_ms",
    "p80_ms",
]

BEST_FIELDS = DETAIL_FIELDS + ["candidate_count"]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _best_rows(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in detail_rows:
        if not math.isfinite(float(row["batch_total_ms"])):
            continue
        by_key.setdefault((str(row["name"]), int(row["batch"])), []).append(row)

    best: list[dict[str, Any]] = []
    for _key, rows in sorted(by_key.items()):
        winner = min(
            rows,
            key=lambda item: (
                float(item["median_ms"]),
                int(item["tile_m"]),
                int(item["tile_k"]),
                int(item["num_stages"]),
            ),
        )
        item = dict(winner)
        item["candidate_count"] = len(rows)
        best.append(item)
    return best


def _timings_from_best(best: list[dict[str, Any]]) -> list[ShapeTiming]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in best:
        by_name.setdefault(str(row["name"]), []).append(row)
    timings: list[ShapeTiming] = []
    for name, rows in sorted(by_name.items()):
        rows.sort(key=lambda item: int(item["batch"]))
        times = {
            int(row["batch"]): float(row["batch_total_ms"])
            for row in rows
            if math.isfinite(float(row["batch_total_ms"]))
        }
        if not times:
            continue
        shape = rows[0]["original_shape"]
        timings.append(
            ShapeTiming(
                name=name,
                shape=(int(shape[0]), int(shape[1])),
                count=int(rows[0]["count"]),
                saturation_batch=max(times),
                times_ms=times,
            )
        )
    return timings


def _assignment_summary(best: list[dict[str, Any]], ranks: list[int]) -> list[dict[str, Any]]:
    timings = _timings_from_best(best)
    summaries: list[dict[str, Any]] = []
    for rank_count in ranks:
        solution = solve_assignment_search(timings, rank_count)
        summaries.append(
            {
                "ranks": rank_count,
                "max_load_ms": solution["optimal_max_load_ms"],
                "average_load_ms": solution["average_load_ms"],
                "total_work_ms": solution["total_work_ms"],
                "imbalance": solution["imbalance"],
                "search_iterations": solution.get("search_iterations", 0),
                "shape_plan": solution["shape_plan"],
                "rank_loads_ms": solution["rank_loads_ms"],
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--batches", type=str, default=None)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--timer", choices=("event", "wall"), default="event")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--no-nesterov", action="store_true")
    parser.add_argument("--cold-state", action="store_true")
    parser.add_argument("--no-update", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output-detail-csv", type=Path, default=Path("/tmp/dmuon_batched_syrk_autotune_detail.csv"))
    parser.add_argument("--output-best-csv", type=Path, default=Path("/tmp/dmuon_batched_syrk_autotune_best.csv"))
    parser.add_argument("--output-summary-json", type=Path, default=Path("/tmp/dmuon_batched_syrk_autotune_summary.json"))
    parser.add_argument("--ranks", type=int, nargs="*", default=[16, 32, 64])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this benchmark needs a CUDA GPU.")
    if not HAS_SYRK_SM80:
        import_error = getattr(workload_bench, "SYRK_IMPORT_ERROR", None)
        raise RuntimeError(f"syrk_sm80 is not importable: {import_error!r}")

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    selected = set(args.only.split(",")) if args.only else None
    detail_rows: list[dict[str, Any]] = []

    print(
        f"# device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}",
        flush=True,
    )
    print(
        f"# configs={SYRK_SM80_CONFIGS} dtype={args.dtype} "
        f"warmup={args.warmup} repeat={args.repeat}",
        flush=True,
    )

    for name, shape, count in WORKLOAD:
        if selected is not None and name not in selected:
            continue
        gshape = gram_shape(shape)
        m, _ = gshape
        batches = parse_batch_list(args.batches, count, args.max_batch)
        configs = [cfg for cfg in SYRK_SM80_CONFIGS if m % cfg[0] == 0]
        if not configs:
            print(f"# skip {name}: no config divides gram M={m}", flush=True)
            continue
        for batch in batches:
            candidates: list[dict[str, Any]] = []
            for tile_m, tile_k, num_stages in configs:
                print(
                    f"# running {name} shape={shape} gram={gshape} batch={batch} "
                    f"tile=({tile_m},{tile_k}) stages={num_stages}",
                    flush=True,
                )
                try:
                    row = benchmark_one(
                        name=name,
                        original_shape=shape,
                        count=count,
                        batch=batch,
                        dtype=dtype,
                        method="batched_syrk",
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        nesterov=not args.no_nesterov,
                        tile_m=tile_m,
                        tile_k=tile_k,
                        num_stages=num_stages,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        timer=args.timer,
                        include_update=not args.no_update,
                        steady_state=not args.cold_state,
                    )
                    row["tile_m"] = tile_m
                    row["tile_k"] = tile_k
                    row["num_stages"] = num_stages
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    if not args.continue_on_error:
                        raise
                    row = {
                        "method": "batched_syrk",
                        "name": name,
                        "original_shape": list(shape),
                        "gram_shape": list(gshape),
                        "count": count,
                        "batch": batch,
                        "tile_m": tile_m,
                        "tile_k": tile_k,
                        "num_stages": num_stages,
                        "batch_total_ms": float("nan"),
                        "single_task_ms": float("nan"),
                        "median_ms": float("nan"),
                        "mean_ms": float("nan"),
                        "p20_ms": float("nan"),
                        "p80_ms": float("nan"),
                        "error": f"OutOfMemoryError: {exc}",
                    }
                candidates.append(row)
                detail_rows.append(row)

            finite = [row for row in candidates if math.isfinite(float(row["batch_total_ms"]))]
            if finite:
                winner = min(finite, key=lambda row: float(row["batch_total_ms"]))
                print(
                    f"# best {name} batch={batch}: "
                    f"{winner['batch_total_ms']:.6f} ms "
                    f"tile=({winner['tile_m']},{winner['tile_k']}) "
                    f"stages={winner['num_stages']}",
                    flush=True,
                )

    best = _best_rows(detail_rows)
    summaries = _assignment_summary(best, args.ranks) if args.ranks else []

    _write_csv(args.output_detail_csv, detail_rows, DETAIL_FIELDS)
    _write_csv(args.output_best_csv, best, BEST_FIELDS)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(
        json.dumps({"best": best, "assignment": summaries}, indent=2),
        encoding="utf-8",
    )

    print(f"# wrote detail: {args.output_detail_csv}", flush=True)
    print(f"# wrote best: {args.output_best_csv}", flush=True)
    print(f"# wrote summary: {args.output_summary_json}", flush=True)
    for item in summaries:
        print(
            "assignment ranks={ranks} max_load_ms={max_load_ms:.6f} "
            "avg_load_ms={average_load_ms:.6f} imbalance={imbalance:.2%}".format(
                **item
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
