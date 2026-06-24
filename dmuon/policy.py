"""Parameter policy resolution for DMuon-managed parameters."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Iterable, Mapping, Optional

import torch
import torch.nn as nn


PolicyLike = Mapping[str, Any] | "DMuonParamPolicy"
ParamPolicyFn = Callable[[str, nn.Parameter], Optional[PolicyLike]]
RouteHintFn = Callable[[str, nn.Parameter], Optional[str]]


_ROUTE_ALIASES = {
    "matrix": "muon",
    "matrix_optimizer": "muon",
    "base": "adamw",
    "base_adamw": "adamw",
    "dedicated_adamw": "adamw",
    "sharded": "sharded_adamw",
    "base_sharded": "sharded_adamw",
    "sharded_collective": "sharded_adamw",
    "base_sharded_adamw": "sharded_adamw",
}

_DTYPE_ALIASES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "torch.float32": torch.float32,
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "torch.float16": torch.float16,
    "half": torch.float16,
}

_POLICY_FIELDS = {
    "route",
    "param_dtype",
    "grad_dtype",
    "output_dtype",
    "cast_forward_inputs",
    "master_dtype",
    "optim_dtype",
    "muon_forward_unshard",
}

_DTYPE_FIELDS = {
    "param_dtype",
    "grad_dtype",
    "output_dtype",
    "master_dtype",
    "optim_dtype",
}


@dataclass(frozen=True)
class DMuonParamPolicy:
    """Final per-parameter DMuon routing and mixed-precision policy.

    ``param_dtype`` follows FSDP2's meaning: it is the dtype of the materialized
    forward/backward parameter view. ``grad_dtype`` is DMuon's gradient-buffer and
    reduction dtype override. ``master_dtype`` and ``optim_dtype`` describe the
    optimizer/storage precision policy and are intentionally separate from
    forward compute dtype.
    """

    route: Optional[str] = None
    param_dtype: Optional[torch.dtype] = None
    grad_dtype: Optional[torch.dtype] = None
    output_dtype: Optional[torch.dtype] = None
    cast_forward_inputs: bool = True
    master_dtype: Optional[torch.dtype] = torch.float32
    optim_dtype: Optional[torch.dtype] = torch.float32
    muon_forward_unshard: Optional[str] = None
    matched_overrides: tuple[int, ...] = ()


def normalize_route(route: Optional[str]) -> Optional[str]:
    if route is None:
        return None
    route = str(route).strip().lower()
    route = _ROUTE_ALIASES.get(route, route)
    if route not in {"muon", "adamw", "sharded_adamw"}:
        raise ValueError(
            "DMuon route must be one of 'muon', 'adamw', or "
            f"'sharded_adamw', got {route!r}"
        )
    return route


def normalize_dtype(value: Any) -> Optional[torch.dtype]:
    if value is None or isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"", "none", "null"}:
            return None
        if key in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[key]
    raise ValueError(f"Unsupported dtype policy value: {value!r}")


def _normalize_policy_values(values: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    unknown = set(values) - _POLICY_FIELDS
    if unknown:
        raise ValueError(f"{source}: unknown DMuon param policy field(s): {sorted(unknown)}")
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if key == "route":
            normalized[key] = normalize_route(value)
        elif key in _DTYPE_FIELDS:
            normalized[key] = normalize_dtype(value)
        elif key == "cast_forward_inputs":
            normalized[key] = bool(value)
        elif key == "muon_forward_unshard":
            normalized[key] = None if value is None else str(value).strip().lower()
        else:
            normalized[key] = value
    return normalized


def _coerce_policy(policy: Optional[PolicyLike], *, source: str) -> DMuonParamPolicy:
    if policy is None:
        return DMuonParamPolicy()
    if isinstance(policy, DMuonParamPolicy):
        values = {
            field.name: getattr(policy, field.name)
            for field in fields(DMuonParamPolicy)
            if field.name != "matched_overrides"
        }
        normalized = _normalize_policy_values(values, source=source)
        return replace(
            DMuonParamPolicy(),
            **normalized,
            matched_overrides=policy.matched_overrides,
        )
    if not isinstance(policy, Mapping):
        raise TypeError(f"{source}: expected mapping or DMuonParamPolicy, got {type(policy)!r}")
    values = _normalize_policy_values(policy, source=source)
    return DMuonParamPolicy(**values)


def _apply_policy(base: DMuonParamPolicy, override: Mapping[str, Any], *, source: str) -> DMuonParamPolicy:
    values = _normalize_policy_values(override, source=source)
    return replace(base, **values)


def _tokens(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    raise TypeError(f"name must be a string or iterable of strings, got {type(value)!r}")


def _override_matches(param_fqn: str, override: Mapping[str, Any]) -> bool:
    if "name" in override and "contains" in override:
        raise ValueError(
            "param_policy override accepts either 'name' or 'contains', not both"
        )
    contains = _tokens(override.get("name", override.get("contains")))
    if not contains:
        return True
    return any(token in param_fqn for token in contains)


def resolve_param_policies(
    *,
    params: Iterable[nn.Parameter],
    param_to_fqn: Mapping[nn.Parameter, str],
    compute_dtype: Optional[torch.dtype] = None,
    route_hint_fn: Optional[RouteHintFn] = None,
    param_policy: Optional[Mapping[str, Any]] = None,
    param_policy_fn: Optional[ParamPolicyFn] = None,
    default_muon_forward_unshard: Optional[str] = None,
) -> dict[nn.Parameter, DMuonParamPolicy]:
    """Resolve final per-parameter DMuon policies.

    ``route_hint_fn`` and ``compute_dtype`` are legacy inputs. New callers should
    prefer ``param_policy`` or ``param_policy_fn``. The function accepts the
    already-selected dedicated params to avoid re-running the user predicate.
    """

    if param_policy is not None and param_policy_fn is not None:
        raise ValueError("Pass at most one of param_policy and param_policy_fn")
    if (param_policy is not None or param_policy_fn is not None) and route_hint_fn is not None:
        raise ValueError(
            "route_hint_fn is legacy route-only API and cannot be combined with "
            "param_policy or param_policy_fn"
        )

    legacy_param_dtype = normalize_dtype(compute_dtype)
    base = DMuonParamPolicy(
        route=None,
        param_dtype=legacy_param_dtype,
        muon_forward_unshard=default_muon_forward_unshard,
    )

    overrides: list[Mapping[str, Any]] = []
    if param_policy is not None:
        defaults = param_policy.get("defaults", {})
        if not isinstance(defaults, Mapping):
            raise TypeError("param_policy['defaults'] must be a mapping")
        base = _apply_policy(base, defaults, source="param_policy.defaults")
        raw_overrides = param_policy.get("overrides", ())
        if raw_overrides is None:
            raw_overrides = ()
        if not isinstance(raw_overrides, Iterable) or isinstance(raw_overrides, (str, bytes)):
            raise TypeError("param_policy['overrides'] must be a sequence of mappings")
        for idx, override in enumerate(raw_overrides):
            if not isinstance(override, Mapping):
                raise TypeError(f"param_policy.overrides[{idx}] must be a mapping")
            unknown = set(override) - {"name", "contains", "set"}
            if unknown:
                raise ValueError(
                    f"param_policy.overrides[{idx}] has unknown selector field(s): "
                    f"{sorted(unknown)}"
                )
            if "set" not in override:
                raise ValueError(f"param_policy.overrides[{idx}] is missing required 'set'")
            if not isinstance(override["set"], Mapping):
                raise TypeError(f"param_policy.overrides[{idx}]['set'] must be a mapping")
            overrides.append(override)

    resolved: dict[nn.Parameter, DMuonParamPolicy] = {}
    for param in params:
        name = param_to_fqn[param]
        policy = base
        matched: list[int] = []

        if route_hint_fn is not None:
            policy = replace(policy, route=normalize_route(route_hint_fn(name, param)))
        else:
            attr_route = getattr(param, "_dmuon_route_hint", None)
            if attr_route is not None and param_policy is None and param_policy_fn is None:
                policy = replace(policy, route=normalize_route(attr_route))

        for idx, override in enumerate(overrides):
            if not _override_matches(name, override):
                continue
            policy = _apply_policy(
                policy,
                override["set"],
                source=f"param_policy.overrides[{idx}].set",
            )
            matched.append(idx)

        if param_policy_fn is not None:
            fn_policy = param_policy_fn(name, param)
            if fn_policy is not None:
                if isinstance(fn_policy, DMuonParamPolicy):
                    policy = replace(
                        _coerce_policy(
                            fn_policy,
                            source=f"param_policy_fn({name!r})",
                        ),
                        matched_overrides=policy.matched_overrides,
                    )
                else:
                    policy = _apply_policy(
                        policy,
                        fn_policy,
                        source=f"param_policy_fn({name!r})",
                    )

        resolved[param] = replace(policy, matched_overrides=tuple(matched))
    return resolved
