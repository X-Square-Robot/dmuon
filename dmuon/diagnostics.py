"""Lightweight runtime diagnostics for DMuon-managed models.

These helpers intentionally avoid CUDA synchronization and distributed
collectives.  They are used by benchmark entrypoints to make optimizer routing
and communication plans auditable in run summaries.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn


def _iter_dedicated_groups(model: nn.Module) -> list[Any]:
    groups: list[Any] = []
    seen: set[int] = set()
    for module in model.modules():
        states = getattr(module, "_dedicated_states", None)
        if states is None:
            state = getattr(module, "_dedicated_state", None)
            states = () if state is None else (state,)
        for state in states:
            group = getattr(state, "group", None)
            if group is None or id(group) in seen:
                continue
            seen.add(id(group))
            groups.append(group)
    return groups


def _param_name_by_id(model: nn.Module) -> dict[int, str]:
    names: dict[int, str] = {}
    for name, param in model.named_parameters():
        names[id(param)] = name
    return names


def _dp_param_ids(dp: Any) -> set[int]:
    ids: set[int] = set()
    for attr in ("_placeholder", "_orig_param"):
        param = getattr(dp, attr, None)
        if param is not None:
            ids.add(id(param))
    return ids


def _dp_display_name(dp: Any, name_by_id: dict[int, str]) -> str:
    for pid in _dp_param_ids(dp):
        if pid in name_by_id:
            return name_by_id[pid]
    return str(getattr(dp, "param_name", "<unknown>"))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_group_rank(group: Any) -> int | None:
    try:
        return int(group.rank())
    except Exception:
        return None


def _safe_group_size(group: Any) -> int | None:
    try:
        return int(group.size())
    except Exception:
        return None


def _owner_coord(value: Any) -> list[int] | str:
    if isinstance(value, tuple):
        return [int(v) for v in value]
    if isinstance(value, list):
        return [int(v) for v in value]
    try:
        return [int(value), 0]
    except Exception:
        return str(value)


def _dtype_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    return str(value)


def _param_element_size(dp: Any) -> int:
    for attr in ("_owned_data", "_sharded_adamw_data", "_orig_param", "_placeholder"):
        tensor = getattr(dp, attr, None)
        if tensor is not None and hasattr(tensor, "element_size"):
            try:
                return int(tensor.element_size())
            except Exception:
                pass
    dtype = (
        getattr(dp, "_param_dtype", None)
        or getattr(dp, "_compute_dtype", None)
        or getattr(dp, "_orig_dtype", None)
    )
    if isinstance(dtype, torch.dtype):
        return int(torch.empty((), dtype=dtype).element_size())
    return 2


def _param_bytes(dp: Any) -> int:
    return _safe_int(getattr(dp, "numel", 0)) * _param_element_size(dp)


def _iter_optimizer_params(group: dict[str, Any]) -> list[Any]:
    params = group.get("params", [])
    if isinstance(params, torch.Tensor):
        return [params]
    if isinstance(params, Iterable):
        return list(params)
    return []


def summarize_param_groups(
    model: nn.Module,
    optimizer: Any,
    *,
    max_rows: int = 128,
) -> dict[str, Any]:
    """Summarize how ``dmuon.Muon`` routed trainable parameters.

    The returned schema is intentionally JSON-friendly and stable enough for
    benchmark summaries.  Counts are local-rank facts unless the key explicitly
    says it includes all dedicated params.
    """

    param_groups = list(getattr(optimizer, "param_groups", []) or [])
    all_dps = list(getattr(optimizer, "_all_dedicated_params", []) or [])
    muon_map = dict(getattr(optimizer, "_dp_to_muon_group_idx", {}) or {})
    adamw_map = dict(getattr(optimizer, "_dp_to_adamw_group_idx", {}) or {})
    owned_muon = getattr(optimizer, "_muon_group_dps", {}) or {}
    owned_adamw = getattr(optimizer, "_adamw_group_dps", {}) or {}
    regular_adamw = getattr(optimizer, "_adamw_group_params", {}) or {}
    name_by_id = _param_name_by_id(model)

    dedicated_by_group: dict[int, list[Any]] = {}
    dedicated_adamw_by_group: dict[int, list[Any]] = {}
    dedicated_muon_by_group: dict[int, list[Any]] = {}
    for dp in all_dps:
        dp_id = id(dp)
        if dp_id in muon_map:
            idx = int(muon_map[dp_id])
            dedicated_by_group.setdefault(idx, []).append(dp)
            dedicated_muon_by_group.setdefault(idx, []).append(dp)
        if dp_id in adamw_map:
            idx = int(adamw_map[dp_id])
            dedicated_by_group.setdefault(idx, []).append(dp)
            dedicated_adamw_by_group.setdefault(idx, []).append(dp)

    groups: list[dict[str, Any]] = []
    for idx, group in enumerate(param_groups):
        dedicated = dedicated_by_group.get(idx, [])
        dedicated_muon = dedicated_muon_by_group.get(idx, [])
        dedicated_adamw = dedicated_adamw_by_group.get(idx, [])
        owned_dedicated = list(owned_muon.get(idx, []) or []) + list(
            owned_adamw.get(idx, []) or []
        )
        adamw_params = list(regular_adamw.get(idx, []) or [])
        route_counts: dict[str, int] = {}
        for dp in dedicated:
            route = str(getattr(dp, "_dmuon_route", "muon"))
            route_counts[route] = route_counts.get(route, 0) + 1
        groups.append(
            {
                "group_index": idx,
                "group_name": str(group.get("group_name", f"group_{idx}")),
                "semantic_group_name": str(
                    group.get("semantic_group_name", group.get("group_name", f"group_{idx}"))
                ),
                "subgroup_type": str(group.get("subgroup_type", "")),
                "use_muon": bool(group.get("use_muon", False)),
                "optimizer_param_count": len(_iter_optimizer_params(group)),
                "dedicated_param_count": len(dedicated),
                "owned_dedicated_param_count": len(owned_dedicated),
                "dedicated_muon_param_count": len(dedicated_muon),
                "dedicated_adamw_param_count": len(dedicated_adamw),
                "adamw_param_count": len(adamw_params),
                "route_param_counts": route_counts,
            }
        )

    rows: list[dict[str, Any]] = []
    total_rows = 0
    for dp in all_dps:
        total_rows += 1
        if len(rows) >= max_rows:
            continue
        group_index = muon_map.get(id(dp), adamw_map.get(id(dp)))
        group_name = None
        if group_index is not None and 0 <= int(group_index) < len(param_groups):
            group_name = param_groups[int(group_index)].get("group_name")
        rows.append(
            {
                "name": _dp_display_name(dp, name_by_id),
                "param_name": str(getattr(dp, "param_name", "<unknown>")),
                "route": str(getattr(dp, "_dmuon_route", "muon")),
                "param_dtype": _dtype_name(getattr(dp, "_param_dtype", None)),
                "grad_dtype": _dtype_name(getattr(dp, "_grad_dtype", None)),
                "output_dtype": _dtype_name(getattr(dp, "_output_dtype", None)),
                "master_dtype": _dtype_name(getattr(dp, "_master_dtype", None)),
                "optim_dtype": _dtype_name(getattr(dp, "_optim_dtype", None)),
                "cast_forward_inputs": bool(
                    getattr(dp, "_cast_forward_inputs", True)
                ),
                "matched_policy_overrides": list(
                    getattr(getattr(dp, "_dmuon_policy", None), "matched_overrides", ())
                ),
                "group_index": int(group_index) if group_index is not None else None,
                "group_name": str(group_name) if group_name is not None else None,
                "owner_rank": _owner_coord(getattr(dp, "owner_rank", None)),
                "is_owner": bool(getattr(dp, "is_owner", False)),
                "is_dtensor": bool(getattr(dp, "is_dtensor", False)),
                "uses_sharded_adamw": bool(
                    getattr(dp, "uses_sharded_adamw", lambda: False)()
                ),
                "numel": _safe_int(getattr(dp, "numel", 0)),
                "shape": [int(x) for x in getattr(dp, "full_shape", ())],
            }
        )

    return {
        "available": True,
        "num_groups": len(param_groups),
        "dedicated_param_count": len(all_dps),
        "owned_dedicated_param_count": sum(
            1 for dp in all_dps if bool(getattr(dp, "is_owner", False))
        ),
        "groups": groups,
        "parameters": rows,
        "parameters_truncated": max(0, total_rows - len(rows)),
    }


def summarize_comm_plan(
    model: nn.Module,
    *,
    max_groups: int = 64,
    max_params_per_group: int = 128,
) -> dict[str, Any]:
    """Summarize DMuon communication roots and payload estimates.

    This function mirrors the actual group ordering and owner buckets used by
    the FSDP2 backend.  It is an estimate only: it reports planned tensor sizes
    and roots, not measured NCCL latency.
    """

    groups = _iter_dedicated_groups(model)
    output_groups: list[dict[str, Any]] = []
    totals = {
        "dedicated_local_bytes": 0,
        "stage1_shard_reduce_tensor_bytes": 0,
        "stage1_shard_reduce_ring_bytes_per_rank": 0,
        "stage2_replicate_reduce_tensor_bytes_this_rank": 0,
        "post_step_replicate_broadcast_tensor_bytes_this_rank": 0,
        "owner_update_ring_bytes_per_rank": 0,
        "same_payload_allreduce_ring_bytes_per_rank": 0,
        "max_owner_bucket_bytes": 0,
    }
    truncated_groups = max(0, len(groups) - max_groups)

    for group_index, group in enumerate(groups[:max_groups]):
        params = list(getattr(group, "params", []) or [])
        by_owner = getattr(group, "_by_owner", {}) or {}
        global_owner_ranks = getattr(group, "_global_owner_ranks", {}) or {}
        dp_group = getattr(group, "_dp_group", None)
        shard_size = _safe_group_size(dp_group) or 1
        shard_rank = _safe_group_rank(dp_group)

        owner_buckets: list[dict[str, Any]] = []
        owner_bucket_bytes_in_order: list[int] = []
        for bucket_index, (owner, owner_params) in enumerate(by_owner.items()):
            bytes_for_owner = sum(
                _param_bytes(dp)
                for dp in owner_params
                if not bool(getattr(dp, "uses_sharded_adamw", lambda: False)())
            )
            if bytes_for_owner <= 0:
                continue
            owner_bucket_bytes_in_order.append(bytes_for_owner)
            owner_global = global_owner_ranks.get(owner)
            if owner_global is None:
                owner_global = getattr(owner_params[0], "_owner_global_rank", None)
            owner_buckets.append(
                {
                    "bucket_index": bucket_index,
                    "owner_coord": _owner_coord(owner),
                    "bytes": bytes_for_owner,
                    "param_count": len(owner_params),
                    "stage1_shard_reduce_root_global_rank": owner_global,
                }
            )
            totals["max_owner_bucket_bytes"] = max(
                totals["max_owner_bucket_bytes"], bytes_for_owner
            )

        param_collectives: list[dict[str, Any]] = []
        active_stage2_bytes = 0
        post_step_bytes = 0
        for param_index, dp in enumerate(params):
            param_bytes = _param_bytes(dp)
            owner_coord = getattr(dp, "owner_rank", None)
            replicate_group = getattr(dp, "replicate_group", None)
            stage2_active = (
                replicate_group is not None
                and shard_rank is not None
                and shard_rank == _safe_int(getattr(dp, "owner_shard", -1), -1)
            )
            if stage2_active:
                active_stage2_bytes += param_bytes
                post_step_bytes += param_bytes
            if len(param_collectives) < max_params_per_group:
                param_collectives.append(
                    {
                        "param_index": param_index,
                        "param_name": str(getattr(dp, "param_name", "<unknown>")),
                        "owner_coord": _owner_coord(owner_coord),
                        "bytes": param_bytes,
                        "route": str(getattr(dp, "_dmuon_route", "muon")),
                        "param_dtype": _dtype_name(getattr(dp, "_param_dtype", None)),
                        "grad_dtype": _dtype_name(getattr(dp, "_grad_dtype", None)),
                        "stage1_shard_reduce_root_global_rank": getattr(
                            dp, "_owner_global_rank", None
                        ),
                        "stage2_replicate_axis_active_on_this_rank": bool(
                            stage2_active
                        ),
                        "stage2_replicate_reduce_root_global_rank": getattr(
                            dp, "_owner_replicate_global_rank", None
                        ),
                        "post_step_replicate_broadcast_root_global_rank": getattr(
                            dp, "_owner_replicate_global_rank", None
                        ),
                    }
                )

        group_stage1_bytes = sum(_param_bytes(dp) for dp in params)
        group_owner_bytes = sum(owner_bucket_bytes_in_order)
        totals["dedicated_local_bytes"] += group_stage1_bytes
        totals["stage1_shard_reduce_tensor_bytes"] += group_stage1_bytes
        totals["stage1_shard_reduce_ring_bytes_per_rank"] += int(
            round(group_stage1_bytes * 2 * max(0, shard_size - 1) / max(1, shard_size))
        )
        totals["stage2_replicate_reduce_tensor_bytes_this_rank"] += active_stage2_bytes
        totals["post_step_replicate_broadcast_tensor_bytes_this_rank"] += post_step_bytes
        totals["owner_update_ring_bytes_per_rank"] += group_owner_bytes
        totals["same_payload_allreduce_ring_bytes_per_rank"] += int(
            round(group_owner_bytes * 2 * max(0, shard_size - 1) / max(1, shard_size))
        )
        output_groups.append(
            {
                "group_index": group_index,
                "debug_name": str(
                    getattr(group, "_debug_name", None) or f"group_{group_index}"
                ),
                "shard_rank": shard_rank,
                "replicate_rank": _safe_group_rank(
                    getattr(params[0], "replicate_group", None)
                )
                if params
                else None,
                "param_count": len(params),
                "owner_bucket_count": len(owner_buckets),
                "owner_bucket_bytes": owner_bucket_bytes_in_order,
                "owner_bucket_bytes_in_order": owner_bucket_bytes_in_order,
                "owner_buckets": owner_buckets,
                "param_collectives": param_collectives,
                "param_collectives_truncated": max(
                    0, len(params) - len(param_collectives)
                ),
            }
        )

    return {
        "available": True,
        "group_count": len(groups),
        "groups": output_groups,
        "groups_truncated": truncated_groups,
        "totals": totals,
    }
