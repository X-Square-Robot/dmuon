"""Diagnostics helpers for inspecting DMuon optimizer wiring."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _tensor_shape(tensor: Any) -> tuple[int, ...] | None:
    if tensor is None:
        return None
    local = getattr(tensor, "_local_tensor", None)
    if local is not None:
        return tuple(int(x) for x in local.shape)
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return tuple(int(x) for x in shape)


def _fqn_by_dedicated_param(model: nn.Module) -> dict[object, str]:
    module_to_fqn = {id(module): name for name, module in model.named_modules()}
    result = {}
    for module in model.modules():
        state = getattr(module, "_dedicated_state", None)
        if state is None:
            continue
        for dp in state.group.params:
            prefix = module_to_fqn.get(id(dp.module), "")
            result[dp] = f"{prefix}.{dp.param_name}" if prefix else dp.param_name
    return result


def summarize_param_groups(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    max_rows: int = 100,
) -> dict[str, Any]:
    """Return a local, read-only summary of DMuon optimizer param groups.

    The helper intentionally performs no distributed communication. Counts are
    local-rank observations, which makes it safe to call during training startup
    before emitting logs or dashboard metadata.
    """

    dummy_ids = {id(p) for p in getattr(optimizer, "_dummy_params", ())}
    all_dps = list(getattr(optimizer, "_all_dedicated_params", ()))
    owned_dps_by_group = {
        int(idx): list(params)
        for idx, params in getattr(optimizer, "_muon_group_dps", {}).items()
    }
    dp_to_group = getattr(optimizer, "_dp_to_muon_group_idx", {})
    adamw_params_by_group = {
        int(idx): list(params)
        for idx, params in getattr(optimizer, "_adamw_group_params", {}).items()
    }
    adamw_param_to_group = getattr(optimizer, "_adamw_param_to_group_idx", {})

    all_dps_by_group: dict[int, list] = {}
    for dp in all_dps:
        group_idx = dp_to_group.get(id(dp))
        if group_idx is not None:
            all_dps_by_group.setdefault(int(group_idx), []).append(dp)

    groups = []
    for idx, group in enumerate(optimizer.param_groups):
        params = list(group.get("params", ()))
        dps = all_dps_by_group.get(idx, [])
        owned_dps = owned_dps_by_group.get(idx, [])
        adamw_params = adamw_params_by_group.get(idx, [])
        groups.append(
            {
                "index": idx,
                "group_name": group.get("group_name", f"group_{idx}"),
                "semantic_group_name": group.get("semantic_group_name"),
                "subgroup_type": group.get("subgroup_type"),
                "use_muon": group.get("use_muon"),
                "lr": group.get("lr"),
                "weight_decay": group.get("weight_decay"),
                "momentum": group.get("momentum"),
                "betas": group.get("betas"),
                "eps": group.get("eps"),
                "param_count": len(params),
                "dummy_param_count": sum(1 for p in params if id(p) in dummy_ids),
                "dedicated_param_count": len(dps),
                "owned_dedicated_param_count": len(owned_dps),
                "tp_sharded_dedicated_param_count": sum(
                    1
                    for dp in dps
                    if getattr(dp, "is_dtensor", False)
                    and getattr(dp, "tp_group", None) is not None
                ),
                "adamw_param_count": len(adamw_params),
            }
        )

    model_param_names = {id(param): name for name, param in model.named_parameters()}
    dedicated_fqns = _fqn_by_dedicated_param(model)
    rows = []

    for dp in all_dps:
        group_idx = dp_to_group.get(id(dp))
        if group_idx is None:
            continue
        group = optimizer.param_groups[int(group_idx)]
        rows.append(
            {
                "fqn": dedicated_fqns.get(dp, getattr(dp, "param_name", "<unknown>")),
                "route": "muon",
                "group_index": int(group_idx),
                "group_name": group.get("group_name", f"group_{group_idx}"),
                "local_shape": _tensor_shape(getattr(dp, "_owned_data", None))
                or tuple(int(x) for x in getattr(dp, "_orig_size", ())),
                "full_shape": tuple(int(x) for x in getattr(dp, "full_shape", ())),
                "requires_grad": bool(getattr(dp, "_requires_grad", False)),
                "is_owner": bool(getattr(dp, "is_owner", False)),
                "owner_rank": getattr(dp, "owner_rank", None),
                "is_tp_sharded": bool(
                    getattr(dp, "is_dtensor", False)
                    and getattr(dp, "tp_group", None) is not None
                ),
                "is_tp_owner": bool(getattr(dp, "is_tp_owner", False)),
                "tp_owner_local_rank": getattr(dp, "_tp_owner_local_rank", None),
                "shard_dim": getattr(dp, "shard_dim", None),
            }
        )

    for param in getattr(optimizer, "_fsdp_params", ()):
        group_idx = adamw_param_to_group.get(id(param))
        if group_idx is None:
            continue
        group = optimizer.param_groups[int(group_idx)]
        rows.append(
            {
                "fqn": model_param_names.get(id(param), "<unnamed_adamw_param>"),
                "route": "adamw",
                "group_index": int(group_idx),
                "group_name": group.get("group_name", f"group_{group_idx}"),
                "local_shape": _tensor_shape(param),
                "full_shape": _tensor_shape(param),
                "requires_grad": bool(getattr(param, "requires_grad", False)),
                "is_owner": None,
                "owner_rank": None,
                "is_tp_sharded": False,
                "is_tp_owner": None,
                "tp_owner_local_rank": None,
                "shard_dim": None,
            }
        )

    rows.sort(key=lambda r: (int(r["group_index"]), str(r["fqn"])))
    truncated = max(0, len(rows) - max_rows)
    if max_rows >= 0:
        rows = rows[:max_rows]

    return {
        "num_groups": len(optimizer.param_groups),
        "groups": groups,
        "parameters": rows,
        "parameters_truncated": truncated,
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(_format_value(v) for v in value) + ")"
    return str(value)


def _format_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    str_rows = [[_format_value(v) for v in row] for row in rows]
    widths = [
        max(len(header), *(len(row[idx]) for row in str_rows))
        if str_rows
        else len(header)
        for idx, header in enumerate(headers)
    ]
    lines = [
        " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in str_rows:
        lines.append(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))
    return lines


def format_param_group_summary(summary: dict[str, Any]) -> str:
    """Format :func:`summarize_param_groups` output as a compact text report."""

    group_headers = [
        "idx",
        "group",
        "type",
        "lr",
        "wd",
        "params",
        "dummy",
        "dedicated",
        "owned",
        "tp",
        "adamw",
    ]
    group_rows = [
        [
            g["index"],
            g["group_name"],
            g["subgroup_type"],
            g["lr"],
            g["weight_decay"],
            g["param_count"],
            g["dummy_param_count"],
            g["dedicated_param_count"],
            g["owned_dedicated_param_count"],
            g["tp_sharded_dedicated_param_count"],
            g["adamw_param_count"],
        ]
        for g in summary.get("groups", [])
    ]

    param_headers = [
        "fqn",
        "route",
        "group",
        "local_shape",
        "full_shape",
        "owner",
        "tp",
    ]
    param_rows = [
        [
            row["fqn"],
            row["route"],
            row["group_name"],
            row["local_shape"],
            row["full_shape"],
            row["owner_rank"],
            row["is_tp_sharded"],
        ]
        for row in summary.get("parameters", [])
    ]

    lines = ["DMuon param group summary", "", "Groups:"]
    lines.extend(_format_table(group_headers, group_rows))
    lines.extend(["", "Parameters:"])
    lines.extend(_format_table(param_headers, param_rows))
    truncated = int(summary.get("parameters_truncated", 0) or 0)
    if truncated:
        lines.append(f"... truncated {truncated} parameter rows")
    return "\n".join(lines)
