"""Simulate profiled_ilp load balance against the legacy LPT owner strategy.

The script uses the workload shape/count list discussed during profiled_ilp
design, generates deterministic measured batch timings for batch sizes
``1..min(max_batch, count)``, and prints max-rank load for ranks 16/32/64.

It does not require CUDA or TileLang.  It exercises the same ILP solver used by
``owner_strategy='profiled_ilp'`` and uses the existing LPT partitioner for the
legacy baseline.

Example:
    DMUON_CACHE_DIR=/tmp/dmuon_cache python benchmarks/profiled_ilp_simulation.py
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("DMUON_CACHE_DIR", "/tmp/dmuon_cache")
os.environ.setdefault("QUACK_CACHE_DIR", "/tmp/quack_cache")
CWD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CWD))

import torch
import torch.nn as nn

from dmuon._core.partition import compute_balanced_assignment
from dmuon._core.profiled_ilp import ProfiledShapeTiming, solve_profiled_assignment
from dmuon.optim.profiled_batch import ProfiledILPConfig


WORKLOAD: list[tuple[str, tuple[int, int], int]] = [
    ("VLM.qkv", (2560, 2048), 36),
    ("VLM.o", (2048, 2048), 36),
    ("VLM.gate_up", (22016, 2048), 36),
    ("VLM.down", (2048, 11008), 36),
    ("ACT.qkv", (2560, 1024), 36),
    ("ACT.o", (1024, 1024), 36),
    ("ACT.gate_up", (4096, 1024), 36),
    ("ACT.down", (1024, 2048), 36),
    ("VIS.qkv", (3840, 1280), 32),
    ("VIS.o", (1280, 1280), 32),
    ("VIS.gate_up", (6912, 1280), 32),
    ("VIS.down", (1280, 3456), 32),
    ("VIS.merger", (5120, 5120), 1),
]


class FakeDeviceMesh:
    mesh_dim_names = None

    def __init__(self, world_size: int):
        self._world_size = int(world_size)

    def size(self) -> int:
        return self._world_size


class WorkloadModel(nn.Module):
    def __init__(self, workload: Iterable[tuple[str, tuple[int, int], int]]):
        super().__init__()
        for idx, (name, shape, count) in enumerate(workload):
            block = nn.Module()
            for item_idx in range(count):
                block.register_parameter(
                    f"p{item_idx}",
                    nn.Parameter(torch.empty(shape, device="meta")),
                )
            safe_name = f"s{idx}_{name.replace('.', '_')}"
            self.add_module(safe_name, block)


@dataclass(frozen=True)
class SimulationRow:
    ranks: int
    solver_status: str
    lpt_max_ms: float
    lpt_avg_ms: float
    lpt_imbalance: float
    profiled_max_ms: float
    profiled_avg_ms: float
    profiled_imbalance: float
    lower_bound_ms: float
    improvement_pct: float
    profiled_loads_ms: list[float]


def _shape_cost_ms(shape: tuple[int, int]) -> float:
    rows, cols = shape
    small = min(rows, cols)
    big = max(rows, cols)
    numel = rows * cols
    ns_flops = 5 * small * (numel + 2 * big * small)
    byte_term = numel / 45_000_000
    return 0.04 + ns_flops / 1.55e12 + 0.08 * byte_term


def generate_synthetic_timings(
    workload: Iterable[tuple[str, tuple[int, int], int]],
    *,
    max_batch: int,
    seed: int,
) -> dict[tuple[int, int], dict[int, float]]:
    rng = random.Random(seed)
    timings: dict[tuple[int, int], dict[int, float]] = {}
    for _name, shape, count in workload:
        single_ms = _shape_cost_ms(shape) * rng.uniform(0.94, 1.06)
        limit = min(max_batch, count)
        overhead = single_ms * rng.uniform(0.35, 0.65)
        steady = single_ms * rng.uniform(0.42, 0.62)
        tau = rng.uniform(1.2, 2.6)
        shape_times: dict[int, float] = {}
        last = 0.0
        for batch in range(1, limit + 1):
            saturated_overhead = overhead * (1.0 - math.exp(-batch / tau))
            value = saturated_overhead + steady * batch
            value *= rng.uniform(0.985, 1.015)
            value = max(value, last + 1e-5)
            shape_times[batch] = round(value, 6)
            last = value
        # Pin batch=1 close to the single-work estimate so the LPT baseline
        # represents the legacy non-grouped owner-local compute path.
        shape_times[1] = round(single_ms, 6)
        timings[shape] = shape_times
    return timings


def load_timings(path: Path) -> dict[tuple[int, int], dict[int, float]]:
    data = json.loads(path.read_text())
    raw_shapes = data["shapes"] if isinstance(data, dict) and "shapes" in data else data
    timings: dict[tuple[int, int], dict[int, float]] = {}
    for item in raw_shapes:
        shape = tuple(int(v) for v in item["shape"])
        timings[shape] = {int(k): float(v) for k, v in item["times_ms"].items()}
    return timings


def _backend_choices(
    timings: dict[tuple[int, int], dict[int, float]]
) -> dict[tuple[int, int], dict[int, str]]:
    return {
        shape: {batch: "profiled" for batch in shape_timings}
        for shape, shape_timings in timings.items()
    }


def _timing_objects(
    workload: Iterable[tuple[str, tuple[int, int], int]],
    timings: dict[tuple[int, int], dict[int, float]],
    backend_choices: dict[tuple[int, int], dict[int, str]],
) -> list[ProfiledShapeTiming]:
    result = []
    for name, shape, count in workload:
        result.append(
            ProfiledShapeTiming(
                name=name,
                shape=shape,
                count=count,
                times_ms=dict(sorted(timings[shape].items())),
                backend_by_batch=backend_choices[shape],
            )
        )
    return result


def _min_total_batch_plan(timing: ProfiledShapeTiming) -> list[int]:
    dp = [float("inf")] * (timing.count + 1)
    prev = [0] * (timing.count + 1)
    dp[0] = 0.0
    for n in range(1, timing.count + 1):
        for batch_size, cost in timing.times_ms.items():
            if batch_size <= n and dp[n - batch_size] + cost < dp[n]:
                dp[n] = dp[n - batch_size] + cost
                prev[n] = batch_size
    batches = []
    n = timing.count
    while n > 0:
        batch_size = prev[n]
        if batch_size <= 0:
            raise RuntimeError(f"could not build heuristic plan for {timing.name}")
        batches.append(batch_size)
        n -= batch_size
    return batches


def _heuristic_profiled_loads(
    profiled_timings: list[ProfiledShapeTiming],
    ranks: int,
) -> list[float]:
    units: list[tuple[float, int, str]] = []
    for timing in profiled_timings:
        for batch_size in _min_total_batch_plan(timing):
            units.append((timing.times_ms[batch_size], batch_size, timing.name))
    heap = [(0.0, rank) for rank in range(ranks)]
    heapq.heapify(heap)
    loads = [0.0 for _ in range(ranks)]
    for cost, _batch_size, _name in sorted(units, reverse=True):
        load, rank = heapq.heappop(heap)
        load += cost
        loads[rank] = load
        heapq.heappush(heap, (load, rank))
    return loads


def lpt_loads_ms(
    workload: list[tuple[str, tuple[int, int], int]],
    timings: dict[tuple[int, int], dict[int, float]],
    ranks: int,
) -> list[float]:
    model = WorkloadModel(workload)
    assignment = compute_balanced_assignment(
        model,
        FakeDeviceMesh(ranks),
        predicate=lambda _name, _param: True,
        owner_strategy="lpt",
    ).dp_owners
    loads = [0.0 for _ in range(ranks)]
    for param, owner in assignment.items():
        shape = tuple(int(dim) for dim in param.shape)
        loads[int(owner)] += timings[shape][1]
    return loads


def _imbalance(loads: list[float]) -> float:
    avg = sum(loads) / len(loads) if loads else 0.0
    return max(loads) / avg - 1.0 if avg > 0 else 0.0


def run_simulation(
    *,
    ranks: int,
    workload: list[tuple[str, tuple[int, int], int]],
    timings: dict[tuple[int, int], dict[int, float]],
    max_batch: int,
    mip_rel_gap: float,
    time_limit_s: float | None,
) -> SimulationRow:
    backend_choices = _backend_choices(timings)
    profiled_timings = _timing_objects(workload, timings, backend_choices)
    solver_status = "ilp"
    try:
        solution = solve_profiled_assignment(
            profiled_timings,
            ranks,
            ProfiledILPConfig(
                max_batch=max_batch,
                ilp_mip_rel_gap=mip_rel_gap,
                ilp_time_limit_s=time_limit_s,
            ),
        )
        profiled_loads = [float(v) for v in solution["rank_loads_ms"]]
        profiled_avg = float(solution["average_load_ms"])
        profiled_imbalance = float(solution["imbalance"])
        total_work = float(solution["total_work_ms"])
    except RuntimeError:
        solver_status = "heuristic_after_ilp_timeout"
        profiled_loads = _heuristic_profiled_loads(profiled_timings, ranks)
        total_work = sum(profiled_loads)
        profiled_avg = total_work / ranks
        profiled_imbalance = _imbalance(profiled_loads)
    lpt_loads = lpt_loads_ms(workload, timings, ranks)

    max_single_batch = max(max(shape_times.values()) for shape_times in timings.values())
    lower_bound = max(total_work / ranks, max_single_batch)
    lpt_max = max(lpt_loads)
    profiled_max = max(profiled_loads)
    improvement = (lpt_max - profiled_max) / lpt_max if lpt_max > 0 else 0.0
    return SimulationRow(
        ranks=ranks,
        solver_status=solver_status,
        lpt_max_ms=lpt_max,
        lpt_avg_ms=sum(lpt_loads) / ranks,
        lpt_imbalance=_imbalance(lpt_loads),
        profiled_max_ms=profiled_max,
        profiled_avg_ms=profiled_avg,
        profiled_imbalance=profiled_imbalance,
        lower_bound_ms=lower_bound,
        improvement_pct=improvement * 100.0,
        profiled_loads_ms=profiled_loads,
    )


def print_rows(rows: list[SimulationRow], *, show_loads: bool) -> None:
    header = (
        "ranks  solver                    lpt_max_ms  lpt_avg_ms  profiled_max_ms  profiled_avg_ms  "
        "lower_bound_ms  improvement  profiled_imbalance"
    )
    print(header)
    for row in rows:
        print(
            f"{row.ranks:>5}  "
            f"{row.solver_status:<24}  "
            f"{row.lpt_max_ms:>10.6f}  "
            f"{row.lpt_avg_ms:>10.6f}  "
            f"{row.profiled_max_ms:>16.6f}  "
            f"{row.profiled_avg_ms:>16.6f}  "
            f"{row.lower_bound_ms:>14.6f}  "
            f"{row.improvement_pct:>10.2f}%  "
            f"{row.profiled_imbalance:>18.2%}"
        )
        if show_loads:
            formatted = ", ".join(f"{value:.4f}" for value in row.profiled_loads_ms)
            print(f"       profiled_loads=[{formatted}]")


def write_csv(path: Path, rows: list[SimulationRow]) -> None:
    fieldnames = [
        "ranks",
        "solver_status",
        "lpt_max_ms",
        "lpt_avg_ms",
        "lpt_imbalance",
        "profiled_max_ms",
        "profiled_avg_ms",
        "profiled_imbalance",
        "lower_bound_ms",
        "improvement_pct",
        "profiled_loads_ms",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = row.__dict__.copy()
            data["profiled_loads_ms"] = json.dumps(row.profiled_loads_ms)
            writer.writerow(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", default="16,32,64")
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--timings-json", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--mip-rel-gap", type=float, default=0.01)
    parser.add_argument(
        "--time-limit-s",
        type=float,
        default=5.0,
        help="MILP time limit per solve. Use <=0 to disable the limit.",
    )
    parser.add_argument("--show-loads", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranks_list = [int(item.strip()) for item in args.ranks.split(",") if item.strip()]
    timings = (
        load_timings(args.timings_json)
        if args.timings_json is not None
        else generate_synthetic_timings(
            WORKLOAD,
            max_batch=args.max_batch,
            seed=args.seed,
        )
    )
    rows = [
        run_simulation(
            ranks=ranks,
            workload=WORKLOAD,
            timings=timings,
            max_batch=args.max_batch,
            mip_rel_gap=args.mip_rel_gap,
            time_limit_s=args.time_limit_s if args.time_limit_s > 0 else None,
        )
        for ranks in ranks_list
    ]
    print_rows(rows, show_loads=args.show_loads)
    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv_output, rows)


if __name__ == "__main__":
    main()
