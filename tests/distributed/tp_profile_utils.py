"""Utilities for TP-owner LPT diagnostics in torchrun tests.

The helpers in this file intentionally live under ``tests/distributed``:
they expose TP owner/load observability for validation and reports without
adding public DMuon API surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch.distributed as dist


def _shape_list(shape: Iterable[int]) -> list[int]:
    return [int(dim) for dim in shape]


def _numel_from_shape(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def iter_dedicated_params(model) -> list[Any]:
    """Return each DedicatedParam attached to ``model`` exactly once."""
    params: list[Any] = []
    seen: set[int] = set()
    for module in model.modules():
        state = getattr(module, "_dedicated_state", None)
        group = getattr(state, "group", None)
        for dp in getattr(group, "params", ()):
            key = id(dp)
            if key in seen:
                continue
            seen.add(key)
            params.append(dp)
    return params


def collect_tp_profile(
    model,
    *,
    scenario: str,
    replicate_async: bool,
    losses: list[float] | None = None,
    step_times_ms: list[float] | None = None,
) -> dict[str, Any]:
    """Collect rank-local TP owner/load diagnostics as JSON-safe data."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    all_params = iter_dedicated_params(model)
    tp_params = [
        dp for dp in all_params
        if getattr(dp, "is_dtensor", False) and getattr(dp, "tp_group", None) is not None
    ]

    tp_size = int(tp_params[0].tp_group.size()) if tp_params else 1
    tp_local_rank = int(tp_params[0].tp_group.rank()) if tp_params else None
    owner_load = {
        str(i): {
            "param_count": 0,
            "logical_numel": 0,
            "local_shard_numel": 0,
            "param_names": [],
        }
        for i in range(tp_size)
    }
    records: list[dict[str, Any]] = []
    local_participating_param_count = 0
    local_participating_shard_numel = 0

    for dp in tp_params:
        owner = int(dp._tp_owner_local_rank)
        local_shape = _shape_list(getattr(dp, "_orig_size", ()))
        full_shape = _shape_list(getattr(dp, "full_shape", local_shape))
        local_numel = _numel_from_shape(local_shape)
        logical_numel = _numel_from_shape(full_shape)
        name = str(getattr(dp, "param_name", "<unknown>"))

        bucket = owner_load[str(owner)]
        bucket["param_count"] += 1
        bucket["logical_numel"] += logical_numel
        bucket["local_shard_numel"] += local_numel
        bucket["param_names"].append(name)

        if getattr(dp, "_owned_data", None) is not None:
            local_participating_param_count += 1
            local_participating_shard_numel += local_numel

        records.append({
            "name": name,
            "full_shape": full_shape,
            "local_shape": local_shape,
            "logical_numel": logical_numel,
            "local_shard_numel": local_numel,
            "shard_dim": (
                None if getattr(dp, "shard_dim", None) is None
                else int(dp.shard_dim)
            ),
            "tp_owner_local_rank": owner,
            "is_tp_owner_on_this_rank": bool(getattr(dp, "is_tp_owner", False)),
        })

    owner_coverage = [
        int(rank_id) for rank_id, load in owner_load.items()
        if load["param_count"] > 0
    ]
    logical_values = [load["logical_numel"] for load in owner_load.values()]
    max_logical = max(logical_values) if logical_values else 0
    min_logical = min(logical_values) if logical_values else 0
    imbalance = (
        (max_logical - min_logical) / max(max_logical, 1)
        if logical_values else 0.0
    )

    return {
        "scenario": scenario,
        "rank": int(rank),
        "world_size": int(world_size),
        "replicate_async": bool(replicate_async),
        "tp_size": int(tp_size),
        "tp_local_rank": tp_local_rank,
        "dedicated_param_count": int(len(all_params)),
        "tp_param_count": int(len(tp_params)),
        "owner_coverage": owner_coverage,
        "owner_load_by_tp_rank": owner_load,
        "owner_logical_numel_imbalance": float(imbalance),
        "local_participating_param_count": int(local_participating_param_count),
        "local_participating_shard_numel": int(local_participating_shard_numel),
        "tp_collective_local_shard_numel_per_step": int(
            sum(r["local_shard_numel"] for r in records)
        ),
        "tp_collective_full_numel_per_step": int(
            sum(r["logical_numel"] for r in records)
        ),
        "losses": list(losses or []),
        "step_times_ms": [float(x) for x in (step_times_ms or [])],
        "params": records,
    }


def assert_tp_owner_spread(profile: dict[str, Any], *, min_owner_ranks: int = 2) -> None:
    """Assert LPT did not collapse all TP-sharded work onto one TP rank."""
    if profile["tp_size"] <= 1:
        return
    if profile["tp_param_count"] == 0:
        raise AssertionError("no TP-sharded dedicated params found")
    expected = min(int(min_owner_ranks), int(profile["tp_size"]))
    coverage = profile["owner_coverage"]
    if len(coverage) < expected:
        raise AssertionError(
            "TP-owner LPT should spread work across at least "
            f"{expected} TP ranks; got coverage={coverage}, "
            f"loads={profile['owner_load_by_tp_rank']}"
        )


def _aggregate_step_times(rank_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(p.get("step_times_ms", [])) for p in rank_profiles]
    if not lengths or min(lengths) == 0:
        return {}
    steps = min(lengths)
    per_step_max: list[float] = []
    per_step_mean: list[float] = []
    for i in range(steps):
        vals = [float(p["step_times_ms"][i]) for p in rank_profiles]
        per_step_max.append(max(vals))
        per_step_mean.append(sum(vals) / len(vals))
    return {
        "per_step_max_ms": per_step_max,
        "per_step_mean_ms": per_step_mean,
    }


def gather_tp_profiles(local_profile: dict[str, Any]) -> dict[str, Any]:
    """All-gather rank profiles and build a rank-0 summary."""
    if not dist.is_initialized():
        rank_profiles = [local_profile]
    else:
        rank_profiles = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(rank_profiles, local_profile)

    base = rank_profiles[0]
    return {
        "scenario": base["scenario"],
        "replicate_async": base["replicate_async"],
        "world_size": base["world_size"],
        "tp_size": base["tp_size"],
        "owner_coverage": base["owner_coverage"],
        "owner_load_by_tp_rank": base["owner_load_by_tp_rank"],
        "owner_logical_numel_imbalance": base["owner_logical_numel_imbalance"],
        "tp_param_count": base["tp_param_count"],
        "tp_collective_full_numel_per_step": base[
            "tp_collective_full_numel_per_step"
        ],
        "timing": _aggregate_step_times(rank_profiles),
        "ranks": rank_profiles,
    }


def write_tp_profile(path: str | Path, aggregate: dict[str, Any]) -> None:
    """Write a gathered TP profile on rank 0."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")


def maybe_write_tp_profile(path: str | None, local_profile: dict[str, Any]) -> None:
    """Gather and write a profile only when ``path`` is provided."""
    if not path:
        return
    aggregate = gather_tp_profiles(local_profile)
    write_tp_profile(path, aggregate)
