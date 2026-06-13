"""Solve batched owner-work assignment with a small integer program.

This script is a standalone prototype for profile-guided DMuon owner
assignment. It generates deterministic synthetic timing data for a workload,
then solves:

    minimize max_rank_load

where each shape can be split into batch tasks of any measured batch size.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class ShapeTiming:
    name: str
    shape: tuple[int, int]
    count: int
    saturation_batch: int
    times_ms: dict[int, float]


@dataclass(frozen=True)
class Variable:
    shape_idx: int
    batch_size: int
    rank: int


@dataclass(frozen=True)
class BatchUnit:
    shape_idx: int
    batch_size: int
    cost_ms: float


def _shape_cost_ms(shape: tuple[int, int]) -> float:
    rows, cols = shape
    small = min(rows, cols)
    big = max(rows, cols)
    numel = rows * cols
    ns_flops = 5 * small * (numel + 2 * big * small)
    byte_term = numel / 45_000_000
    return 0.04 + ns_flops / 1.55e12 + 0.08 * byte_term


def _saturation_batch(count: int, single_ms: float) -> int:
    if count <= 1:
        return 1
    if single_ms >= 1.15:
        return min(count, 4)
    if single_ms >= 0.55:
        return min(count, 6)
    if single_ms >= 0.28:
        return min(count, 8)
    if single_ms >= 0.15:
        return min(count, 12)
    return min(count, 16)


def _speedup_at_saturation(batch: int, rng: random.Random) -> float:
    if batch <= 1:
        return 1.0
    baseline = {
        4: 1.45,
        6: 1.75,
        8: 2.10,
        12: 2.75,
        16: 3.25,
    }.get(batch, 1.0 + 0.15 * batch)
    return baseline * rng.uniform(0.94, 1.06)


def generate_test_data(seed: int = 20260610) -> dict[str, Any]:
    rng = random.Random(seed)
    shapes: list[dict[str, Any]] = []
    for name, shape, count in WORKLOAD:
        t1 = _shape_cost_ms(shape) * rng.uniform(0.92, 1.08)
        sat = _saturation_batch(count, t1)
        speedup = _speedup_at_saturation(sat, rng)
        sat_per_item = t1 / speedup
        overhead = max(0.0, t1 - sat_per_item)
        tau = rng.uniform(1.2, 2.8)
        times: dict[int, float] = {}
        last = 0.0
        for b in range(1, sat + 1):
            # A fixed overhead that saturates plus a linear steady-state term.
            # This gives sublinear growth at small b and near-linear growth near
            # the saturation batch size.
            overhead_term = overhead * (1.0 - math.exp(-b / tau)) / (
                1.0 - math.exp(-1.0 / tau)
            )
            value = sat_per_item * b + overhead_term
            value *= rng.uniform(0.985, 1.015)
            value = max(value, last + 1e-4)
            times[b] = round(value, 6)
            last = value
        shapes.append(
            {
                "name": name,
                "shape": list(shape),
                "count": count,
                "saturation_batch": sat,
                "times_ms": {str(k): v for k, v in times.items()},
            }
        )
    return {
        "seed": seed,
        "rank_counts": [16, 32, 64],
        "description": (
            "Synthetic steady-state batch timing data. times_ms[b] is the "
            "measured total time for one batch task of size b."
        ),
        "shapes": shapes,
    }


def load_timings(path: Path) -> list[ShapeTiming]:
    data = json.loads(path.read_text())
    timings = []
    for item in data["shapes"]:
        timings.append(
            ShapeTiming(
                name=str(item["name"]),
                shape=(int(item["shape"][0]), int(item["shape"][1])),
                count=int(item["count"]),
                saturation_batch=int(item["saturation_batch"]),
                times_ms={int(k): float(v) for k, v in item["times_ms"].items()},
            )
        )
    return timings


def write_test_data(path: Path, *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_test_data(seed)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def _make_variables(timings: list[ShapeTiming], ranks: int) -> list[Variable]:
    variables: list[Variable] = []
    for shape_idx, timing in enumerate(timings):
        max_batch = min(timing.count, timing.saturation_batch)
        for batch_size in range(1, max_batch + 1):
            for rank in range(ranks):
                variables.append(Variable(shape_idx, batch_size, rank))
    return variables


def _solve_min_max_load(
    timings: list[ShapeTiming],
    ranks: int,
    *,
    time_limit_s: float | None,
    mip_rel_gap: float | None,
    allow_incumbent: bool,
) -> tuple[float, np.ndarray, list[Variable], dict[str, Any]]:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    variables = _make_variables(timings, ranks)
    n_x = len(variables)
    l_idx = n_x
    n_vars = n_x + 1

    rows = len(timings) + ranks + max(0, ranks - 1)
    matrix = lil_matrix((rows, n_vars), dtype=float)
    lower = np.full(rows, -np.inf, dtype=float)
    upper = np.full(rows, np.inf, dtype=float)

    for shape_idx, timing in enumerate(timings):
        lower[shape_idx] = timing.count
        upper[shape_idx] = timing.count

    for rank in range(ranks):
        row = len(timings) + rank
        upper[row] = 0.0
        matrix[row, l_idx] = -1.0

    # Symmetry break: ranks are interchangeable, so there is always an
    # equivalent optimum with non-increasing loads. This removes many
    # duplicate branch-and-bound states.
    for rank in range(ranks - 1):
        row = len(timings) + ranks + rank
        lower[row] = 0.0
        upper[row] = np.inf

    upper_bounds = np.full(n_vars, np.inf, dtype=float)
    for col, var in enumerate(variables):
        timing = timings[var.shape_idx]
        matrix[var.shape_idx, col] = var.batch_size
        load_row = len(timings) + var.rank
        matrix[load_row, col] = timing.times_ms[var.batch_size]
        if var.rank < ranks - 1:
            symmetry_row = len(timings) + ranks + var.rank
            matrix[symmetry_row, col] = timing.times_ms[var.batch_size]
        if var.rank > 0:
            symmetry_row = len(timings) + ranks + var.rank - 1
            matrix[symmetry_row, col] = -timing.times_ms[var.batch_size]
        upper_bounds[col] = timing.count // var.batch_size

    c = np.zeros(n_vars, dtype=float)
    c[l_idx] = 1.0
    integrality = np.ones(n_vars, dtype=int)
    integrality[l_idx] = 0

    options: dict[str, float] = {}
    if time_limit_s:
        options["time_limit"] = time_limit_s
    if mip_rel_gap is not None:
        options["mip_rel_gap"] = mip_rel_gap

    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(np.zeros(n_vars), upper_bounds),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options=options or None,
    )
    if not result.success and (not allow_incumbent or result.x is None):
        raise RuntimeError(f"stage 1 MILP failed: {result.message}")
    meta = {
        "success": bool(result.success),
        "message": str(result.message),
        "mip_gap": (
            None if getattr(result, "mip_gap", None) is None else float(result.mip_gap)
        ),
        "mip_dual_bound": (
            None
            if getattr(result, "mip_dual_bound", None) is None
            else float(result.mip_dual_bound)
        ),
        "mip_node_count": (
            None
            if getattr(result, "mip_node_count", None) is None
            else int(result.mip_node_count)
        ),
    }
    return float(result.fun), result.x[:n_x], variables, meta


def _solve_min_total_work_at_load(
    timings: list[ShapeTiming],
    ranks: int,
    variables: list[Variable],
    max_load_ms: float,
    *,
    time_limit_s: float | None,
    mip_rel_gap: float | None,
    allow_incumbent: bool,
) -> np.ndarray:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    n_vars = len(variables)
    rows = len(timings) + ranks + max(0, ranks - 1)
    matrix = lil_matrix((rows, n_vars), dtype=float)
    lower = np.full(rows, -np.inf, dtype=float)
    upper = np.full(rows, np.inf, dtype=float)

    for shape_idx, timing in enumerate(timings):
        lower[shape_idx] = timing.count
        upper[shape_idx] = timing.count

    for rank in range(ranks):
        row = len(timings) + rank
        upper[row] = max_load_ms + 1e-5

    for rank in range(ranks - 1):
        row = len(timings) + ranks + rank
        lower[row] = 0.0
        upper[row] = np.inf

    upper_bounds = np.full(n_vars, np.inf, dtype=float)
    c = np.zeros(n_vars, dtype=float)
    for col, var in enumerate(variables):
        timing = timings[var.shape_idx]
        cost = timing.times_ms[var.batch_size]
        c[col] = cost
        matrix[var.shape_idx, col] = var.batch_size
        matrix[len(timings) + var.rank, col] = cost
        if var.rank < ranks - 1:
            symmetry_row = len(timings) + ranks + var.rank
            matrix[symmetry_row, col] = cost
        if var.rank > 0:
            symmetry_row = len(timings) + ranks + var.rank - 1
            matrix[symmetry_row, col] = -cost
        upper_bounds[col] = timing.count // var.batch_size

    options: dict[str, float] = {}
    if time_limit_s:
        options["time_limit"] = time_limit_s
    if mip_rel_gap is not None:
        options["mip_rel_gap"] = mip_rel_gap

    result = milp(
        c,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(np.zeros(n_vars), upper_bounds),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options=options or None,
    )
    if not result.success and (not allow_incumbent or result.x is None):
        raise RuntimeError(f"stage 2 MILP failed: {result.message}")
    return result.x


def solve_assignment(
    timings: list[ShapeTiming],
    ranks: int,
    *,
    time_limit_s: float | None = None,
    mip_rel_gap: float | None = 0.01,
    allow_incumbent: bool = True,
) -> dict[str, Any]:
    import numpy as np

    best_l, stage1_x, variables, stage1_meta = _solve_min_max_load(
        timings,
        ranks,
        time_limit_s=time_limit_s,
        mip_rel_gap=mip_rel_gap,
        allow_incumbent=allow_incumbent,
    )
    stage2_used = True
    try:
        x = _solve_min_total_work_at_load(
            timings,
            ranks,
            variables,
            best_l,
            time_limit_s=time_limit_s,
            mip_rel_gap=mip_rel_gap,
            allow_incumbent=allow_incumbent,
        )
    except RuntimeError:
        x = stage1_x
        stage2_used = False
    counts = np.rint(x).astype(int)

    rank_loads = [0.0 for _ in range(ranks)]
    rank_tasks = [0 for _ in range(ranks)]
    rank_batches = [0 for _ in range(ranks)]
    rank_assignment: list[dict[str, dict[int, int]]] = [
        {} for _ in range(ranks)
    ]
    shape_plan: dict[str, dict[int, int]] = {}

    for value, var in zip(counts, variables):
        if value <= 0:
            continue
        timing = timings[var.shape_idx]
        cost = timing.times_ms[var.batch_size] * value
        rank_loads[var.rank] += cost
        rank_tasks[var.rank] += var.batch_size * value
        rank_batches[var.rank] += value
        by_shape = rank_assignment[var.rank].setdefault(timing.name, {})
        by_shape[var.batch_size] = by_shape.get(var.batch_size, 0) + value
        shape_counts = shape_plan.setdefault(timing.name, {})
        shape_counts[var.batch_size] = shape_counts.get(var.batch_size, 0) + value

    total_work = sum(rank_loads)
    avg_load = total_work / ranks
    max_load = max(rank_loads) if rank_loads else 0.0
    return {
        "ranks": ranks,
        "optimal_max_load_ms": max_load,
        "stage1_bound_ms": best_l,
        "total_work_ms": total_work,
        "average_load_ms": avg_load,
        "imbalance": (max_load / avg_load - 1.0) if avg_load > 0 else 0.0,
        "rank_loads_ms": rank_loads,
        "rank_tasks": rank_tasks,
        "rank_batches": rank_batches,
        "rank_assignment": rank_assignment,
        "shape_plan": shape_plan,
        "stage1": stage1_meta,
        "stage2_used": stage2_used,
    }


def _format_batches(batches: dict[int, int]) -> str:
    return "+".join(f"{size}x{count}" for size, count in sorted(batches.items()))


def _min_total_batch_plan(timing: ShapeTiming) -> list[int]:
    dp = [float("inf")] * (timing.count + 1)
    prev = [0] * (timing.count + 1)
    dp[0] = 0.0
    for n in range(1, timing.count + 1):
        for batch_size, cost in timing.times_ms.items():
            if batch_size <= n and dp[n - batch_size] + cost < dp[n]:
                dp[n] = dp[n - batch_size] + cost
                prev[n] = batch_size
    batches: list[int] = []
    n = timing.count
    while n > 0:
        batch_size = prev[n]
        if batch_size <= 0:
            raise RuntimeError(f"could not build batch plan for {timing.name}")
        batches.append(batch_size)
        n -= batch_size
    return batches


def _initial_units(timings: list[ShapeTiming]) -> list[BatchUnit]:
    units: list[BatchUnit] = []
    for shape_idx, timing in enumerate(timings):
        for batch_size in _min_total_batch_plan(timing):
            units.append(
                BatchUnit(
                    shape_idx=shape_idx,
                    batch_size=batch_size,
                    cost_ms=timing.times_ms[batch_size],
                )
            )
    return units


def _assign_units_lpt(
    units: list[BatchUnit], ranks: int
) -> tuple[list[float], list[list[BatchUnit]]]:
    import heapq

    heap = [(0.0, rank) for rank in range(ranks)]
    heapq.heapify(heap)
    assignments: list[list[BatchUnit]] = [[] for _ in range(ranks)]
    loads = [0.0 for _ in range(ranks)]
    for unit in sorted(units, key=lambda item: item.cost_ms, reverse=True):
        load, rank = heapq.heappop(heap)
        assignments[rank].append(unit)
        load += unit.cost_ms
        loads[rank] = load
        heapq.heappush(heap, (load, rank))
    return loads, assignments


def _solution_from_units(
    timings: list[ShapeTiming],
    ranks: int,
    units: list[BatchUnit],
    *,
    status: str,
    search_iterations: int,
) -> dict[str, Any]:
    rank_loads, assignments = _assign_units_lpt(units, ranks)
    rank_tasks = [0 for _ in range(ranks)]
    rank_batches = [0 for _ in range(ranks)]
    rank_assignment: list[dict[str, dict[int, int]]] = [
        {} for _ in range(ranks)
    ]
    shape_plan: dict[str, dict[int, int]] = {}

    for rank, rank_units in enumerate(assignments):
        for unit in rank_units:
            timing = timings[unit.shape_idx]
            rank_tasks[rank] += unit.batch_size
            rank_batches[rank] += 1
            by_shape = rank_assignment[rank].setdefault(timing.name, {})
            by_shape[unit.batch_size] = by_shape.get(unit.batch_size, 0) + 1
            shape_counts = shape_plan.setdefault(timing.name, {})
            shape_counts[unit.batch_size] = shape_counts.get(unit.batch_size, 0) + 1

    total_work = sum(rank_loads)
    avg_load = total_work / ranks
    max_load = max(rank_loads) if rank_loads else 0.0
    return {
        "ranks": ranks,
        "optimal_max_load_ms": max_load,
        "stage1_bound_ms": None,
        "total_work_ms": total_work,
        "average_load_ms": avg_load,
        "imbalance": (max_load / avg_load - 1.0) if avg_load > 0 else 0.0,
        "rank_loads_ms": rank_loads,
        "rank_tasks": rank_tasks,
        "rank_batches": rank_batches,
        "rank_assignment": rank_assignment,
        "shape_plan": shape_plan,
        "stage1": {
            "success": status == "proven",
            "message": status,
            "mip_gap": None,
            "mip_dual_bound": None,
            "mip_node_count": None,
        },
        "stage2_used": False,
        "search_iterations": search_iterations,
    }


def solve_assignment_search(
    timings: list[ShapeTiming],
    ranks: int,
) -> dict[str, Any]:
    """Fast global split search using measured batch times.

    This is not a proof-producing optimizer. It is useful when the exact MILP
    solver cannot prove optimality quickly with the local SciPy/HiGHS build.
    """
    units = _initial_units(timings)
    loads, _assignments = _assign_units_lpt(units, ranks)
    best_score = tuple(round(v, 9) for v in sorted(loads, reverse=True))
    iterations = 0

    while True:
        best_candidate: tuple[
            tuple[float, ...], float, int, list[BatchUnit]
        ] | None = None
        for idx, unit in enumerate(units):
            if unit.batch_size <= 1:
                continue
            timing = timings[unit.shape_idx]
            for left in range(1, unit.batch_size // 2 + 1):
                right = unit.batch_size - left
                if left not in timing.times_ms or right not in timing.times_ms:
                    continue
                candidate_units = list(units)
                candidate_units[idx : idx + 1] = [
                    BatchUnit(unit.shape_idx, left, timing.times_ms[left]),
                    BatchUnit(unit.shape_idx, right, timing.times_ms[right]),
                ]
                candidate_loads, _ = _assign_units_lpt(candidate_units, ranks)
                candidate_score = tuple(
                    round(v, 9) for v in sorted(candidate_loads, reverse=True)
                )
                candidate_total = sum(item.cost_ms for item in candidate_units)
                if candidate_score >= best_score:
                    continue
                score = (candidate_score, candidate_total, len(candidate_units))
                if best_candidate is None or score < best_candidate[:3]:
                    best_candidate = (
                        candidate_score,
                        candidate_total,
                        len(candidate_units),
                        candidate_units,
                    )
        if best_candidate is None:
            break
        best_score, _candidate_total, _candidate_len, units = best_candidate
        iterations += 1

    return _solution_from_units(
        timings,
        ranks,
        units,
        status="local_search",
        search_iterations=iterations,
    )


def print_solution(solution: dict[str, Any], *, show_assignments: bool) -> None:
    ranks = int(solution["ranks"])
    print(f"\n=== ranks={ranks} ===")
    load_label = (
        "max_load_ms"
        if solution["stage1"]["message"] == "local_search"
        else "optimal_max_load_ms"
    )
    print(
        "{}={:.6f} avg_load_ms={:.6f} total_work_ms={:.6f} "
        "imbalance={:.2%}".format(
            load_label,
            solution["optimal_max_load_ms"],
            solution["average_load_ms"],
            solution["total_work_ms"],
            solution["imbalance"],
        )
    )
    stage1 = solution["stage1"]
    if stage1["message"] == "local_search":
        print(
            "solver_status=local_search search_iterations={}".format(
                solution.get("search_iterations", 0)
            )
        )
    else:
        proven = bool(stage1["success"])
        print(
            "solver_status={} mip_gap={} stage2_used={}".format(
                "proven" if proven else "incumbent",
                "n/a" if stage1["mip_gap"] is None else f"{stage1['mip_gap']:.2%}",
                solution["stage2_used"],
            )
        )
    print("shape batch plan:")
    for name in sorted(solution["shape_plan"]):
        print(f"  {name:<12} {_format_batches(solution['shape_plan'][name])}")
    print("rank loads:")
    avg = float(solution["average_load_ms"])
    for rank, load in enumerate(solution["rank_loads_ms"]):
        pct = (load / avg - 1.0) if avg > 0 else 0.0
        print(
            f"  rank {rank:02d}: {load:9.6f} ms "
            f"({pct:+7.2%}), tasks={solution['rank_tasks'][rank]:3d}, "
            f"batches={solution['rank_batches'][rank]:3d}"
        )
        if show_assignments:
            parts = []
            for name, batches in sorted(solution["rank_assignment"][rank].items()):
                parts.append(f"{name}:{_format_batches(batches)}")
            print("    " + "; ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("batch_assignment_testdata.json"),
    )
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--write-testdata", action="store_true")
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--time-limit-s", type=float, default=60.0)
    parser.add_argument("--mip-rel-gap", type=float, default=0.01)
    parser.add_argument("--method", choices=("search", "milp"), default="search")
    parser.add_argument(
        "--strict-optimal",
        action="store_true",
        help="Fail if the MILP solver reaches the time limit before proving optimality.",
    )
    parser.add_argument("--show-assignments", action="store_true")
    args = parser.parse_args()

    if args.write_testdata or not args.data.exists():
        write_test_data(args.data, seed=args.seed)
        print(f"wrote test data: {args.data}")

    timings = load_timings(args.data)
    for ranks in args.ranks:
        if args.method == "milp":
            solution = solve_assignment(
                timings,
                ranks,
                time_limit_s=args.time_limit_s,
                mip_rel_gap=args.mip_rel_gap,
                allow_incumbent=not args.strict_optimal,
            )
        else:
            solution = solve_assignment_search(timings, ranks)
        print_solution(solution, show_assignments=args.show_assignments)


if __name__ == "__main__":
    main()
