"""Ownership domains for multi-parallelism DMuon parameter management.

An :class:`OwnershipDomain` describes the replica topology (shard and optional
HSDP replicate axes) and gradient reduction scaling for one class of parameters
inside a single :func:`dmuon.dedicate_params` call.  Examples:

- Pure DP / FSDP2 / HSDP: a single implicit ``"default"`` domain built from
  ``mesh`` and optional ``replicate_mesh``.
- MoE with Expert Parallelism: a ``"dense"`` domain over ``dp * cp * ep`` and
  an ``"expert"`` domain over ``dp * cp`` with ``grad_scale=1.0 / ep_size``.
- Non-TP / non-EP parameters in a hybrid 5D mesh (e.g., LayerNorm or local
  action MoE): a ``"replicated"`` domain over ``dp * cp * ep * tp``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

from torch.distributed.device_mesh import DeviceMesh

DEFAULT_DOMAIN_NAME = "default"
_VALID_REDUCE_OPS = ("avg", "sum")


def _validate_1d_mesh(mesh: Any, *, field_name: str) -> None:
    if mesh is None:
        raise ValueError(f"OwnershipDomain.{field_name} cannot be None")
    ndim = getattr(mesh, "ndim", None)
    if ndim is not None and ndim != 1:
        raise ValueError(
            f"OwnershipDomain.{field_name} must be a 1D DeviceMesh, got ndim={ndim!r}"
        )
    if not callable(getattr(mesh, "size", None)) or not callable(getattr(mesh, "get_group", None)):
        raise TypeError(
            f"OwnershipDomain.{field_name} must provide DeviceMesh size() and get_group()"
        )
    if mesh.size() <= 0:
        raise ValueError(
            f"OwnershipDomain.{field_name} must have positive size, got {mesh.size()!r}"
        )


def _mesh_dim_names(mesh: Any) -> tuple[str, ...]:
    names = getattr(mesh, "mesh_dim_names", None)
    if not names:
        return ()
    return tuple(str(name) for name in names)


def normalize_domain_name(value: Any, *, allow_none: bool = False, source: str = "domain") -> Optional[str]:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{source} must be a non-empty string, got None")
    if not isinstance(value, str):
        raise TypeError(f"{source} must be a string, got {type(value)!r}")
    name = value.strip()
    if not name:
        raise ValueError(f"{source} must be a non-empty string")
    return name


@dataclass(frozen=True)
class OwnershipDomain:
    """Replica topology and gradient reduction semantics for a parameter subset.

    Args:
        name: Unique non-empty identifier referenced by ``DMuonParamPolicy.domain``.
        shard_mesh: 1D :class:`DeviceMesh` over the Stage-1 shard / DP axis where
            DMuon assigns owners, reduces gradients, and broadcasts updates.
        replicate_mesh: Optional 1D :class:`DeviceMesh` over the HSDP Stage-2
            replicate axis. When ``None``, the domain runs in 1D shard-only mode.
        grad_reduce_op: Collective reduction operator along the domain's replica
            axes (``"avg"`` or ``"sum"``). Defaults to ``"avg"``.
        grad_scale: Positive finite multiplier applied once to each owner's
            reduced gradient after cross-rank reduction (e.g., ``1.0 / ep_size``
            for EP-sharded experts when global loss is normalized over ``dp * ep``).
            Defaults to ``1.0`` (no extra kernel launched).
    """

    name: str
    shard_mesh: DeviceMesh
    replicate_mesh: Optional[DeviceMesh] = None
    grad_reduce_op: Literal["avg", "sum"] = "avg"
    grad_scale: float = 1.0

    def __post_init__(self) -> None:
        name = normalize_domain_name(self.name, allow_none=False, source="OwnershipDomain.name")
        object.__setattr__(self, "name", name)

        _validate_1d_mesh(self.shard_mesh, field_name="shard_mesh")
        if self.replicate_mesh is not None:
            _validate_1d_mesh(self.replicate_mesh, field_name="replicate_mesh")
            shard_dev = getattr(self.shard_mesh, "device_type", None)
            rep_dev = getattr(self.replicate_mesh, "device_type", None)
            if shard_dev is not None and rep_dev is not None and shard_dev != rep_dev:
                raise ValueError(
                    f"OwnershipDomain {name!r}: shard_mesh device_type {shard_dev!r} "
                    f"differs from replicate_mesh device_type {rep_dev!r}"
                )
            overlap = set(_mesh_dim_names(self.shard_mesh)) & set(
                _mesh_dim_names(self.replicate_mesh)
            )
            if overlap:
                raise ValueError(
                    f"OwnershipDomain {name!r}: shard_mesh and replicate_mesh share "
                    f"dimension name(s) {sorted(overlap)}"
                )

        if not isinstance(self.grad_reduce_op, str):
            raise TypeError(
                f"OwnershipDomain.grad_reduce_op must be a string, got {type(self.grad_reduce_op)!r}"
            )
        op = self.grad_reduce_op.strip().lower()
        if op not in _VALID_REDUCE_OPS:
            raise ValueError(
                f"OwnershipDomain.grad_reduce_op must be one of {_VALID_REDUCE_OPS}, got {self.grad_reduce_op!r}"
            )
        object.__setattr__(self, "grad_reduce_op", op)

        if isinstance(self.grad_scale, bool) or not isinstance(self.grad_scale, (int, float)):
            raise TypeError(
                f"OwnershipDomain.grad_scale must be a positive finite number, got {self.grad_scale!r}"
            )
        scale = float(self.grad_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"OwnershipDomain.grad_scale must be positive and finite, got {self.grad_scale!r}"
            )
        object.__setattr__(self, "grad_scale", scale)

    @property
    def is_hsdp(self) -> bool:
        return self.replicate_mesh is not None

    @property
    def device_type(self) -> str:
        return str(getattr(self.shard_mesh, "device_type", "cuda"))

    @property
    def dp_dim_names(self) -> set[str]:
        names = set(_mesh_dim_names(self.shard_mesh))
        if self.replicate_mesh is not None:
            names.update(_mesh_dim_names(self.replicate_mesh))
        return names


def resolve_ownership_domains(
    *,
    mesh: Optional[DeviceMesh] = None,
    replicate_mesh: Optional[DeviceMesh] = None,
    domains: Optional[Mapping[str, OwnershipDomain]] = None,
    default_domain: Optional[str] = None,
) -> tuple[dict[str, OwnershipDomain], Optional[str]]:
    """Normalize legacy ``(mesh, replicate_mesh)`` or explicit ``domains`` into a domain map.

    Returns ``(domain_map, resolved_default_domain_name)``. When ``domains`` has
    multiple entries and neither ``default_domain`` nor ``"default"`` is present,
    the returned default name is ``None`` and every parameter policy must select
    its domain explicitly.
    """
    if domains is None:
        if mesh is None:
            raise ValueError("dedicate_params requires either 'mesh' or 'domains'")
        name = (
            normalize_domain_name(default_domain, allow_none=False, source="default_domain")
            if default_domain is not None
            else DEFAULT_DOMAIN_NAME
        )
        assert name is not None
        return {name: OwnershipDomain(name=name, shard_mesh=mesh, replicate_mesh=replicate_mesh)}, name

    if mesh is not None or replicate_mesh is not None:
        raise ValueError("Pass either 'mesh'/'replicate_mesh' or 'domains', not both")
    if not isinstance(domains, Mapping):
        raise TypeError(f"domains must be a mapping of name -> OwnershipDomain, got {type(domains)!r}")
    if not domains:
        raise ValueError("domains mapping must be non-empty")

    resolved: dict[str, OwnershipDomain] = {}
    device_types: set[str] = set()
    for key, domain in domains.items():
        norm_key = normalize_domain_name(key, allow_none=False, source="domains key")
        assert norm_key is not None
        if not isinstance(domain, OwnershipDomain):
            raise TypeError(
                f"domains[{norm_key!r}] must be an OwnershipDomain, got {type(domain)!r}"
            )
        if domain.name != norm_key:
            raise ValueError(
                f"domains key {norm_key!r} does not match OwnershipDomain.name {domain.name!r}"
            )
        if norm_key in resolved:
            raise ValueError(f"Duplicate normalized domain name: {norm_key!r}")
        resolved[norm_key] = domain
        device_types.add(domain.device_type)

    if len(device_types) > 1:
        raise ValueError(
            f"All OwnershipDomains must share one device_type, got {sorted(device_types)}"
        )

    if default_domain is not None:
        norm_default = normalize_domain_name(
            default_domain, allow_none=False, source="default_domain"
        )
        if norm_default not in resolved:
            raise ValueError(
                f"default_domain {norm_default!r} is not present in domains {sorted(resolved)}"
            )
        return resolved, norm_default

    if len(resolved) == 1:
        return resolved, next(iter(resolved))
    if DEFAULT_DOMAIN_NAME in resolved:
        return resolved, DEFAULT_DOMAIN_NAME
    return resolved, None
