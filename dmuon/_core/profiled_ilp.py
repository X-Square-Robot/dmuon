"""Profile-guided ILP owner assignment for DMuon."""

from __future__ import annotations

import logging
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import torch.nn as nn

from dmuon.optim.profiled_batch import (
    BackendMeasurement,
    ProfiledILPConfig,
    measure_owner_muon_backend,
    normalize_profiled_ilp_config,
    require_profiled_ilp_dependencies,
)

from .tp import is_tp_sharded

try:
    from torch.distributed import DeviceMesh
except ImportError:  # pragma: no cover
    from torch.distributed.device_mesh import DeviceMesh


logger = logging.getLogger(__name__)

OwnerCoord = tuple[int, int]
OwnerValue = Union[int, OwnerCoord]
AssignmentGroupKeyFn = Callable[[str, nn.Parameter], Optional[str]]
RouteHintFn = Callable[[str, nn.Parameter], Optional[str]]


@dataclass(frozen=True)
class ShapeWorkload:
    name: str
    shape: tuple[int, int]
    params: tuple[nn.Parameter, ...]

    @property
    def count(self) -> int:
        return len(self.params)


@dataclass(frozen=True)
class ProfiledShapeTiming:
    name: str
    shape: tuple[int, int]
    count: int
    times_ms: dict[int, float]
    backend_by_batch: dict[int, str]
    measurements: dict[int, tuple[BackendMeasurement, ...]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ILPVariable:
    shape_idx: int
    batch_size: int
    rank: int


@dataclass(frozen=True)
class ProfiledBatchParamMeta:
    group_key: tuple[object, ...]
    backend: str
    batch_size: int
    shape: tuple[int, int]
    measured_cost_ms: float
    owner: OwnerValue


def _dist_rank_world() -> tuple[int, int]:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
    except Exception:
        pass
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _progress(config: ProfiledILPConfig, message: str, *, force: bool = False) -> None:
    env = os.environ.get("DMUON_PROFILED_ILP_VERBOSE")
    env_enabled = env is not None and env.strip().lower() not in {"0", "false", "no"}
    if not force and not (bool(getattr(config, "verbose", False)) or env_enabled):
        return
    rank, _world = _dist_rank_world()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DMuon profiled_ilp][rank {rank}][{now}] {message}", flush=True)


def _config_signature(config: ProfiledILPConfig) -> dict[str, object]:
    dtype = config.dtype
    device = config.device
    return {
        "max_batch": int(config.max_batch),
        "warmup": int(config.warmup),
        "repeat": int(config.repeat),
        "dtype": None if dtype is None else str(dtype),
        "device_type": None if device is None else str(device).split(":")[0],
        "lr": float(config.lr),
        "momentum": float(config.momentum),
        "weight_decay": float(config.weight_decay),
        "nesterov": bool(config.nesterov),
        "correctness_rtol": float(config.correctness_rtol),
        "correctness_atol": float(config.correctness_atol),
        "backends": [str(item) for item in config.backends],
        "benchmark_timer": str(config.benchmark_timer),
    }


def _workload_signature(
    workloads: list[ShapeWorkload],
    config: ProfiledILPConfig,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile_config": _config_signature(config),
        "workloads": [
            {
                "shape": [int(workload.shape[0]), int(workload.shape[1])],
                "count": int(workload.count),
            }
            for workload in workloads
        ],
    }


def _signature_hash(signature: dict[str, object]) -> str:
    data = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def _resolve_profile_cache_path(
    workloads: list[ShapeWorkload],
    config: ProfiledILPConfig,
    signature: dict[str, object],
) -> Path:
    explicit = os.environ.get("DMUON_PROFILED_ILP_CACHE_PATH") or config.profile_cache_path
    if explicit:
        return Path(explicit)
    cache_dir = (
        os.environ.get("DMUON_PROFILED_ILP_CACHE_DIR")
        or os.environ.get("DMUON_CACHE_DIR")
        or tempfile.gettempdir()
    )
    del workloads
    return Path(cache_dir) / f"profiled_ilp_timings_{_signature_hash(signature)}.json"


def _serialize_timings(
    timings: list[ProfiledShapeTiming],
    signature: dict[str, object],
) -> dict[str, object]:
    return {
        **signature,
        "timings": [
            {
                "name": timing.name,
                "shape": [int(timing.shape[0]), int(timing.shape[1])],
                "count": int(timing.count),
                "times_ms": {
                    str(int(batch)): float(cost)
                    for batch, cost in sorted(timing.times_ms.items())
                },
                "backend_by_batch": {
                    str(int(batch)): str(backend)
                    for batch, backend in sorted(timing.backend_by_batch.items())
                },
            }
            for timing in timings
        ],
    }


def _deserialize_timings(payload: dict[str, object]) -> list[ProfiledShapeTiming]:
    timings = []
    for item in payload.get("timings", []):
        shape_values = item["shape"]
        shape = (int(shape_values[0]), int(shape_values[1]))
        times = {
            int(batch): float(cost)
            for batch, cost in dict(item["times_ms"]).items()
        }
        backend_by_batch = {
            int(batch): str(backend)
            for batch, backend in dict(item["backend_by_batch"]).items()
        }
        timings.append(
            ProfiledShapeTiming(
                name=str(item.get("name", f"shape={shape[0]}x{shape[1]}")),
                shape=shape,
                count=int(item["count"]),
                times_ms=times,
                backend_by_batch=backend_by_batch,
            )
        )
    return timings


def _load_cached_timings(
    path: Path,
    signature: dict[str, object],
) -> Optional[list[ProfiledShapeTiming]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("profiled_ilp: failed to read cache %s: %r", path, exc)
        return None

    expected = dict(signature)
    actual = {
        "schema_version": payload.get("schema_version"),
        "profile_config": payload.get("profile_config"),
        "workloads": payload.get("workloads"),
    }
    if actual != expected:
        return None
    return _deserialize_timings(payload)


def _write_cached_timings(
    path: Path,
    timings: list[ProfiledShapeTiming],
    signature: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rank, _world = _dist_rank_world()
    tmp_path = path.with_name(f"{path.name}.rank{rank}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_serialize_timings(timings, signature), f, indent=2)
    os.replace(tmp_path, path)


def _param_shape(param: nn.Parameter) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(param, "shape", ()))


def _prod(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def _route_for_param(
    name: str,
    param: nn.Parameter,
    route_hint_fn: Optional[RouteHintFn],
) -> str:
    if route_hint_fn is not None:
        route = route_hint_fn(name, param)
    else:
        route = getattr(param, "_dmuon_route_hint", None)
    if route is None:
        return "muon"
    route = str(route).strip().lower()
    aliases = {
        "matrix": "muon",
        "matrix_optimizer": "muon",
        "base": "adamw",
        "base_optimizer": "adamw",
        "base_adamw": "adamw",
        "dedicated_adamw": "adamw",
        "sharded": "sharded_adamw",
        "base_sharded": "sharded_adamw",
        "base_sharded_adamw": "sharded_adamw",
        "sharded_collective": "sharded_adamw",
    }
    return aliases.get(route, route)


def _matrix_shape(param: nn.Parameter) -> Optional[tuple[int, int]]:
    shape = _param_shape(param)
    if len(shape) < 2:
        return None
    rows = int(shape[0])
    cols = _prod(shape[1:])
    return rows, cols


def _shape_name(shape: tuple[int, int], param_names: list[str]) -> str:
    if len(param_names) == 1:
        return param_names[0]
    return f"shape={shape[0]}x{shape[1]}"


def _collect_workloads(
    model: nn.Module,
    predicate: Callable[[str, nn.Parameter], bool],
    route_hint_fn: Optional[RouteHintFn],
    dp_mesh_dim_names: frozenset[str],
) -> tuple[list[ShapeWorkload], list[tuple[nn.Parameter, str]]]:
    by_shape: dict[tuple[int, int], list[tuple[nn.Parameter, str]]] = defaultdict(list)
    fallback: list[tuple[nn.Parameter, str]] = []
    for name, param in model.named_parameters():
        if not predicate(name, param):
            continue
        if is_tp_sharded(param, dp_mesh_dim_names):
            raise NotImplementedError(
                "owner_strategy='profiled_ilp' currently supports non-TP "
                "owner-local Muon parameters only. Exclude TP-sharded params "
                "from the predicate or use owner_strategy='lpt'."
            )
        shape = _matrix_shape(param)
        if shape is None or _route_for_param(name, param, route_hint_fn) != "muon":
            fallback.append((param, name))
            continue
        by_shape[shape].append((param, name))

    workloads: list[ShapeWorkload] = []
    for shape in sorted(by_shape):
        items = by_shape[shape]
        params = tuple(param for param, _name in items)
        names = [name for _param, name in items]
        workloads.append(
            ShapeWorkload(
                name=_shape_name(shape, names),
                shape=shape,
                params=params,
            )
        )
    return workloads, fallback


def _profile_workload(
    workload: ShapeWorkload,
    config: ProfiledILPConfig,
    *,
    index: int,
    total: int,
) -> ProfiledShapeTiming:
    max_batch = max(1, min(int(config.max_batch), workload.count))
    measured_timings = config.measured_timings or {}
    measured_backend_choices = config.measured_backend_choices or {}
    if workload.shape in measured_timings:
        times = {
            int(batch): float(cost)
            for batch, cost in measured_timings[workload.shape].items()
            if 1 <= int(batch) <= max_batch
        }
        if not times:
            raise ValueError(
                f"profiled_ilp measured_timings for shape={workload.shape} "
                f"has no batch in [1, {max_batch}]"
            )
        backend_choices = {
            batch: measured_backend_choices.get(workload.shape, {}).get(
                batch, "cublas"
            )
            for batch in times
        }
        return ProfiledShapeTiming(
            name=workload.name,
            shape=workload.shape,
            count=workload.count,
            times_ms=times,
            backend_by_batch=backend_choices,
        )

    _progress(
        config,
        f"shape {index}/{total} start: name={workload.name}, "
        f"shape={workload.shape}, count={workload.count}, "
        f"batch=1..{max_batch}, backends={list(config.backends)}",
    )
    times_ms: dict[int, float] = {}
    backend_by_batch: dict[int, str] = {}
    measurements_by_batch: dict[int, tuple[BackendMeasurement, ...]] = {}
    for batch in range(1, max_batch + 1):
        measurements: list[BackendMeasurement] = []
        for backend in config.backends:
            t0 = time.perf_counter()
            _progress(
                config,
                f"shape {index}/{total} batch={batch}/{max_batch} "
                f"backend={backend}: start",
            )
            try:
                measurement = measure_owner_muon_backend(
                    shape=workload.shape,
                    batch=batch,
                    backend=backend,
                    config=config,
                )
            except Exception as exc:
                logger.info(
                    "profiled_ilp: backend=%s failed for shape=%s batch=%s: %r",
                    backend,
                    workload.shape,
                    batch,
                    exc,
                )
                _progress(
                    config,
                    f"shape {index}/{total} batch={batch}/{max_batch} "
                    f"backend={backend}: failed after "
                    f"{time.perf_counter() - t0:.1f}s: {exc!r}",
                    force=True,
                )
                continue
            measurements.append(measurement)
            _progress(
                config,
                f"shape {index}/{total} batch={batch}/{max_batch} "
                f"backend={backend}: median={measurement.median_ms:.4f}ms, "
                f"mean={measurement.mean_ms:.4f}ms, "
                f"correct={measurement.correct}, "
                f"max_rel_error={measurement.max_rel_error:.4g}, "
                f"elapsed={time.perf_counter() - t0:.1f}s",
            )
        valid = [item for item in measurements if item.correct]
        if not valid:
            detail = ", ".join(
                f"{m.backend}: correct={m.correct} rel={m.max_rel_error:.4g}"
                for m in measurements
            )
            raise RuntimeError(
                f"profiled_ilp: no correct backend for shape={workload.shape}, "
                f"batch={batch}. Measurements: {detail or '<none>'}"
            )
        best = min(valid, key=lambda item: item.median_ms)
        times_ms[batch] = float(best.median_ms)
        backend_by_batch[batch] = best.backend
        measurements_by_batch[batch] = tuple(measurements)
        _progress(
            config,
            f"shape {index}/{total} batch={batch}/{max_batch}: "
            f"best_backend={best.backend}, measured_cost={best.median_ms:.4f}ms",
        )
    _progress(
        config,
        f"shape {index}/{total} done: shape={workload.shape}, "
        f"timings_ms={times_ms}, backends={backend_by_batch}",
    )
    return ProfiledShapeTiming(
        name=workload.name,
        shape=workload.shape,
        count=workload.count,
        times_ms=times_ms,
        backend_by_batch=backend_by_batch,
        measurements=measurements_by_batch,
    )


def profile_shape_timings(
    workloads: list[ShapeWorkload],
    config: ProfiledILPConfig,
) -> list[ProfiledShapeTiming]:
    if config.measured_timings:
        return [
            _profile_workload(
                workload,
                config,
                index=index,
                total=len(workloads),
            )
            for index, workload in enumerate(workloads, start=1)
        ]

    rank, world = _dist_rank_world()
    signature = _workload_signature(workloads, config)
    cache_path = _resolve_profile_cache_path(workloads, config, signature)
    mode = str(config.profile_distributed_mode).strip().lower()
    profile_rank = int(config.profile_rank)

    cached = _load_cached_timings(cache_path, signature)
    if cached is not None:
        _progress(
            config,
            f"autotune cache hit: path={cache_path}, shapes={len(cached)}",
        )
        return cached

    total_measurements = sum(
        max(1, min(int(config.max_batch), workload.count)) * len(config.backends)
        for workload in workloads
    )

    if world > 1 and mode == "rank0_file" and rank != profile_rank:
        _progress(
            config,
            f"waiting for profiled_ilp timings from rank {profile_rank}: "
            f"path={cache_path}, timeout={config.profile_cache_wait_timeout_s}s",
            force=True,
        )
        deadline = time.monotonic() + float(config.profile_cache_wait_timeout_s)
        next_log = time.monotonic()
        while time.monotonic() < deadline:
            cached = _load_cached_timings(cache_path, signature)
            if cached is not None:
                _progress(
                    config,
                    f"loaded profiled_ilp timings: path={cache_path}, "
                    f"shapes={len(cached)}",
                    force=True,
                )
                return cached
            now = time.monotonic()
            if now >= next_log:
                remaining = max(0.0, deadline - now)
                _progress(
                    config,
                    f"still waiting for profiled_ilp timings at {cache_path}; "
                    f"remaining={remaining:.0f}s",
                    force=True,
                )
                next_log = now + max(5.0, float(config.profile_cache_poll_s))
            time.sleep(max(0.1, float(config.profile_cache_poll_s)))
        raise TimeoutError(
            "Timed out waiting for profiled_ilp timing cache from "
            f"rank {profile_rank}: {cache_path}"
        )

    _progress(
        config,
        f"autotune start: shapes={len(workloads)}, measurements={total_measurements}, "
        f"warmup={config.warmup}, repeat={config.repeat}, "
        f"backends={list(config.backends)}, cache_path={cache_path}",
        force=True,
    )
    t0 = time.perf_counter()
    timings = [
        _profile_workload(
            workload,
            config,
            index=index,
            total=len(workloads),
        )
        for index, workload in enumerate(workloads, start=1)
    ]
    elapsed = time.perf_counter() - t0
    _write_cached_timings(cache_path, timings, signature)
    _progress(
        config,
        f"autotune done: elapsed={elapsed:.1f}s, cache_written={cache_path}",
        force=True,
    )
    return timings


def _make_variables(timings: list[ProfiledShapeTiming], ranks: int) -> list[ILPVariable]:
    variables: list[ILPVariable] = []
    for shape_idx, timing in enumerate(timings):
        for batch_size in sorted(timing.times_ms):
            for rank in range(ranks):
                variables.append(ILPVariable(shape_idx, int(batch_size), rank))
    return variables


def _milp_options(
    *, time_limit_s: Optional[float], mip_rel_gap: Optional[float]
) -> Optional[dict[str, float]]:
    options: dict[str, float] = {}
    if time_limit_s is not None:
        options["time_limit"] = float(time_limit_s)
    if mip_rel_gap is not None:
        options["mip_rel_gap"] = float(mip_rel_gap)
    return options or None


def _solve_min_max_load(
    timings: list[ProfiledShapeTiming],
    ranks: int,
    config: ProfiledILPConfig,
) -> tuple[float, object, list[ILPVariable], dict[str, object]]:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    variables = _make_variables(timings, ranks)
    n_x = len(variables)
    max_load_idx = n_x
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
        matrix[row, max_load_idx] = -1.0

    for rank in range(ranks - 1):
        row = len(timings) + ranks + rank
        lower[row] = 0.0

    upper_bounds = np.full(n_vars, np.inf, dtype=float)
    for col, var in enumerate(variables):
        timing = timings[var.shape_idx]
        cost = timing.times_ms[var.batch_size]
        matrix[var.shape_idx, col] = var.batch_size
        matrix[len(timings) + var.rank, col] = cost
        if var.rank < ranks - 1:
            matrix[len(timings) + ranks + var.rank, col] = cost
        if var.rank > 0:
            matrix[len(timings) + ranks + var.rank - 1, col] = -cost
        upper_bounds[col] = timing.count // var.batch_size

    c = np.zeros(n_vars, dtype=float)
    c[max_load_idx] = 1.0
    integrality = np.ones(n_vars, dtype=int)
    integrality[max_load_idx] = 0
    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(np.zeros(n_vars), upper_bounds),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options=_milp_options(
            time_limit_s=config.ilp_time_limit_s,
            mip_rel_gap=config.ilp_mip_rel_gap,
        ),
    )
    if not result.success and (not config.ilp_allow_incumbent or result.x is None):
        raise RuntimeError(f"profiled_ilp stage 1 MILP failed: {result.message}")
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
    timings: list[ProfiledShapeTiming],
    ranks: int,
    variables: list[ILPVariable],
    max_load_ms: float,
    config: ProfiledILPConfig,
):
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
        upper[len(timings) + rank] = max_load_ms + 1e-5

    for rank in range(ranks - 1):
        lower[len(timings) + ranks + rank] = 0.0

    upper_bounds = np.full(n_vars, np.inf, dtype=float)
    c = np.zeros(n_vars, dtype=float)
    for col, var in enumerate(variables):
        timing = timings[var.shape_idx]
        cost = timing.times_ms[var.batch_size]
        c[col] = cost
        matrix[var.shape_idx, col] = var.batch_size
        matrix[len(timings) + var.rank, col] = cost
        if var.rank < ranks - 1:
            matrix[len(timings) + ranks + var.rank, col] = cost
        if var.rank > 0:
            matrix[len(timings) + ranks + var.rank - 1, col] = -cost
        upper_bounds[col] = timing.count // var.batch_size

    result = milp(
        c,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(np.zeros(n_vars), upper_bounds),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options=_milp_options(
            time_limit_s=config.ilp_time_limit_s,
            mip_rel_gap=config.ilp_mip_rel_gap,
        ),
    )
    if not result.success and (not config.ilp_allow_incumbent or result.x is None):
        raise RuntimeError(f"profiled_ilp stage 2 MILP failed: {result.message}")
    return result.x


def solve_profiled_assignment(
    timings: list[ProfiledShapeTiming],
    ranks: int,
    config: ProfiledILPConfig,
) -> dict[str, object]:
    import numpy as np

    best_load, stage1_x, variables, stage1_meta = _solve_min_max_load(
        timings, ranks, config
    )
    stage2_used = True
    try:
        x = _solve_min_total_work_at_load(timings, ranks, variables, best_load, config)
    except RuntimeError:
        x = stage1_x
        stage2_used = False
    counts = np.rint(x).astype(int)

    rank_loads = [0.0 for _ in range(ranks)]
    rank_assignment: list[dict[int, dict[int, int]]] = [defaultdict(dict) for _ in range(ranks)]
    shape_plan: dict[int, dict[int, int]] = defaultdict(dict)
    for value, var in zip(counts, variables):
        if int(value) <= 0:
            continue
        timing = timings[var.shape_idx]
        cost = timing.times_ms[var.batch_size] * int(value)
        rank_loads[var.rank] += cost
        by_shape = rank_assignment[var.rank].setdefault(var.shape_idx, {})
        by_shape[var.batch_size] = by_shape.get(var.batch_size, 0) + int(value)
        plan = shape_plan.setdefault(var.shape_idx, {})
        plan[var.batch_size] = plan.get(var.batch_size, 0) + int(value)

    total_work = sum(rank_loads)
    avg_load = total_work / ranks if ranks > 0 else 0.0
    max_load = max(rank_loads) if rank_loads else 0.0
    return {
        "rank_assignment": [dict(item) for item in rank_assignment],
        "shape_plan": dict(shape_plan),
        "rank_loads_ms": rank_loads,
        "total_work_ms": total_work,
        "average_load_ms": avg_load,
        "optimal_max_load_ms": max_load,
        "imbalance": (max_load / avg_load - 1.0) if avg_load > 0 else 0.0,
        "stage1": stage1_meta,
        "stage2_used": stage2_used,
    }


def _fallback_cost(param: nn.Parameter) -> float:
    return max(1.0, float(param.numel()) / 1_000_000.0)


def compute_profiled_ilp_assignment(
    model: nn.Module,
    mesh: DeviceMesh,
    predicate: Callable[[str, nn.Parameter], bool],
    *,
    replicate_mesh: Optional[DeviceMesh],
    route_hint_fn: Optional[RouteHintFn],
    profiled_ilp_config: object,
) -> tuple[
    dict[nn.Parameter, OwnerValue],
    dict[nn.Parameter, int],
    dict[nn.Parameter, ProfiledBatchParamMeta],
    dict[str, object],
]:
    """Compute DP/HSDP owners from profiled batch costs and a MILP solve."""

    require_profiled_ilp_dependencies()
    config = normalize_profiled_ilp_config(profiled_ilp_config)

    shard_size = mesh.size()
    replicate_size = replicate_mesh.size() if replicate_mesh is not None else 1
    slots: list[OwnerCoord] = [
        (s, r) for s in range(shard_size) for r in range(replicate_size)
    ]
    is_hsdp = replicate_mesh is not None

    dp_names: set[str] = set()
    if mesh.mesh_dim_names:
        dp_names |= set(mesh.mesh_dim_names)
    if replicate_mesh is not None and replicate_mesh.mesh_dim_names:
        dp_names |= set(replicate_mesh.mesh_dim_names)
    dp_mesh_dim_names = frozenset(dp_names)

    workloads, fallback_params = _collect_workloads(
        model,
        predicate,
        route_hint_fn,
        dp_mesh_dim_names,
    )
    timings = profile_shape_timings(workloads, config) if workloads else []
    solution = (
        solve_profiled_assignment(timings, len(slots), config)
        if timings
        else {
            "rank_assignment": [{} for _ in slots],
            "rank_loads_ms": [0.0 for _ in slots],
            "total_work_ms": 0.0,
            "average_load_ms": 0.0,
            "optimal_max_load_ms": 0.0,
            "imbalance": 0.0,
            "stage1": {},
            "stage2_used": False,
        }
    )

    assignment: dict[nn.Parameter, OwnerValue] = {}
    batch_meta: dict[nn.Parameter, ProfiledBatchParamMeta] = {}
    rank_loads = [float(v) for v in solution["rank_loads_ms"]]

    remaining_by_shape: dict[int, list[nn.Parameter]] = {
        idx: list(workload.params) for idx, workload in enumerate(workloads)
    }
    group_counters: dict[tuple[int, int, int], int] = defaultdict(int)
    rank_assignment = solution["rank_assignment"]
    for flat_rank, by_shape in enumerate(rank_assignment):
        owner_coord = slots[flat_rank]
        owner_value: OwnerValue = owner_coord if is_hsdp else owner_coord[0]
        for shape_idx, batches in by_shape.items():
            timing = timings[int(shape_idx)]
            for batch_size, batch_count in sorted(batches.items(), reverse=True):
                batch_size = int(batch_size)
                backend = timing.backend_by_batch[batch_size]
                measured_cost = timing.times_ms[batch_size]
                for _ in range(int(batch_count)):
                    params = remaining_by_shape[int(shape_idx)][:batch_size]
                    del remaining_by_shape[int(shape_idx)][:batch_size]
                    if len(params) != batch_size:
                        raise RuntimeError(
                            "profiled_ilp internal error: ILP consumed more "
                            f"params than available for shape={timing.shape}"
                        )
                    group_key_base = (owner_coord, timing.shape, backend, batch_size)
                    group_idx = group_counters[(flat_rank, int(shape_idx), batch_size)]
                    group_counters[(flat_rank, int(shape_idx), batch_size)] += 1
                    group_key = (*group_key_base, group_idx)
                    for param in params:
                        assignment[param] = owner_value
                        batch_meta[param] = ProfiledBatchParamMeta(
                            group_key=group_key,
                            backend=backend,
                            batch_size=batch_size,
                            shape=timing.shape,
                            measured_cost_ms=measured_cost,
                            owner=owner_value,
                        )

    for shape_idx, remaining in remaining_by_shape.items():
        if remaining:
            raise RuntimeError(
                "profiled_ilp internal error: ILP left unassigned params for "
                f"shape={timings[shape_idx].shape}: {len(remaining)}"
            )

    # Non-Muon/non-matrix dedicated params keep working under profiled_ilp, but
    # they are not batched.  Place them onto the current least-loaded owner.
    for param, _name in fallback_params:
        flat_rank = min(range(len(slots)), key=lambda idx: rank_loads[idx])
        owner_coord = slots[flat_rank]
        owner_value = owner_coord if is_hsdp else owner_coord[0]
        assignment[param] = owner_value
        rank_loads[flat_rank] += _fallback_cost(param)

    metadata = {
        "strategy": "profiled_ilp",
        "timings": timings,
        "solution": solution,
        "rank_loads_ms_with_fallback": rank_loads,
        "fallback_param_count": len(fallback_params),
        "max_batch": config.max_batch,
        "backends": config.backends,
        "profile_distributed_mode": config.profile_distributed_mode,
        "profile_cache_path": str(
            _resolve_profile_cache_path(
                workloads,
                config,
                _workload_signature(workloads, config),
            )
        )
        if workloads
        else None,
    }
    return assignment, {}, batch_meta, metadata
