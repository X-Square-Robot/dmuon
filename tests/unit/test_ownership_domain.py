"""Unit tests for OwnershipDomain validation, domain resolution, and policy.domain."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

import dmuon
from dmuon.domain import DEFAULT_DOMAIN_NAME, OwnershipDomain, resolve_ownership_domains
from dmuon.policy import DMuonParamPolicy, resolve_param_policies


@dataclass
class _FakeMesh:
    mesh_size: int
    mesh_dim_names: tuple[str, ...] | None = None
    ndim: int = 1
    device_type: str = "cuda"

    def size(self) -> int:
        return self.mesh_size

    def get_group(self):
        return f"pg:{self.mesh_dim_names}:{self.mesh_size}"


def test_ownership_domain_top_level_export() -> None:
    assert dmuon.OwnershipDomain is OwnershipDomain
    assert "OwnershipDomain" in dmuon.__all__


def test_ownership_domain_valid_1d_and_2d() -> None:
    shard = _FakeMesh(4, ("shard",))
    rep = _FakeMesh(2, ("replicate",))

    d1 = OwnershipDomain(" dense ", shard_mesh=shard, grad_reduce_op="AVG", grad_scale=1)
    assert d1.name == "dense"
    assert not d1.is_hsdp
    assert d1.grad_reduce_op == "avg"
    assert d1.grad_scale == 1.0
    assert d1.dp_dim_names == {"shard"}
    assert d1.device_type == "cuda"

    d2 = OwnershipDomain(
        "expert",
        shard_mesh=shard,
        replicate_mesh=rep,
        grad_reduce_op="sum",
        grad_scale=0.25,
    )
    assert d2.is_hsdp
    assert d2.grad_reduce_op == "sum"
    assert d2.grad_scale == 0.25
    assert d2.dp_dim_names == {"shard", "replicate"}


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"name": "", "shard_mesh": _FakeMesh(2)}, ValueError),
        ({"name": "   ", "shard_mesh": _FakeMesh(2)}, ValueError),
        ({"name": 123, "shard_mesh": _FakeMesh(2)}, TypeError),
        ({"name": "x", "shard_mesh": None}, ValueError),
        ({"name": "x", "shard_mesh": _FakeMesh(2, ndim=2)}, ValueError),
        ({"name": "x", "shard_mesh": _FakeMesh(0)}, ValueError),
        ({"name": "x", "shard_mesh": _FakeMesh(2), "replicate_mesh": _FakeMesh(2, ndim=2)}, ValueError),
        (
            {
                "name": "x",
                "shard_mesh": _FakeMesh(2, ("dp",)),
                "replicate_mesh": _FakeMesh(2, ("dp",)),
            },
            ValueError,
        ),
        (
            {
                "name": "x",
                "shard_mesh": _FakeMesh(2, ("dp",), device_type="cuda"),
                "replicate_mesh": _FakeMesh(2, ("rep",), device_type="cpu"),
            },
            ValueError,
        ),
        ({"name": "x", "shard_mesh": _FakeMesh(2), "grad_reduce_op": "max"}, ValueError),
        ({"name": "x", "shard_mesh": _FakeMesh(2), "grad_reduce_op": 1}, TypeError),
        ({"name": "x", "shard_mesh": _FakeMesh(2), "grad_scale": 0.0}, ValueError),
        ({"name": "x", "shard_mesh": _FakeMesh(2), "grad_scale": -0.5}, ValueError),
        ({"name": "x", "shard_mesh": _FakeMesh(2), "grad_scale": float("nan")}, ValueError),
        ({"name": "x", "shard_mesh": _FakeMesh(2), "grad_scale": float("inf")}, ValueError),
        ({"name": "x", "shard_mesh": _FakeMesh(2), "grad_scale": True}, TypeError),
    ],
)
def test_ownership_domain_rejects_invalid_inputs(kwargs, exc) -> None:
    with pytest.raises(exc):
        OwnershipDomain(**kwargs)


def test_resolve_ownership_domains_legacy_and_multi() -> None:
    dense_mesh = _FakeMesh(4, ("dp_ep",))
    expert_mesh = _FakeMesh(2, ("dp",))
    rep_mesh = _FakeMesh(2, ("replicate",))

    # Legacy 1D / 2D single-domain fallback.
    doms, default = resolve_ownership_domains(mesh=dense_mesh, replicate_mesh=rep_mesh)
    assert default == DEFAULT_DOMAIN_NAME
    assert list(doms) == [DEFAULT_DOMAIN_NAME]
    assert doms[DEFAULT_DOMAIN_NAME].shard_mesh is dense_mesh
    assert doms[DEFAULT_DOMAIN_NAME].replicate_mesh is rep_mesh

    # Multi-domain with explicit default_domain.
    dense = OwnershipDomain("dense", shard_mesh=dense_mesh)
    expert = OwnershipDomain("expert", shard_mesh=expert_mesh, grad_scale=0.5)
    doms, default = resolve_ownership_domains(
        domains={"dense": dense, "expert": expert},
        default_domain="dense",
    )
    assert default == "dense"
    assert doms == {"dense": dense, "expert": expert}

    # Multi-domain without default_domain or "default" -> default is None.
    doms, default = resolve_ownership_domains(domains={"dense": dense, "expert": expert})
    assert default is None

    # Multi-domain with "default" key auto-selects "default".
    d_default = OwnershipDomain("default", shard_mesh=dense_mesh)
    doms, default = resolve_ownership_domains(domains={"default": d_default, "expert": expert})
    assert default == "default"

    # Single entry in domains auto-selects itself.
    doms, default = resolve_ownership_domains(domains={"expert": expert})
    assert default == "expert"


def test_resolve_ownership_domains_rejects_conflicts() -> None:
    m = _FakeMesh(2, ("dp",))
    d = OwnershipDomain("dense", shard_mesh=m)

    with pytest.raises(ValueError, match="either 'mesh' or 'domains'"):
        resolve_ownership_domains()
    with pytest.raises(ValueError, match="not both"):
        resolve_ownership_domains(mesh=m, domains={"dense": d})
    with pytest.raises(ValueError, match="not both"):
        resolve_ownership_domains(replicate_mesh=m, domains={"dense": d})
    with pytest.raises(ValueError, match="non-empty"):
        resolve_ownership_domains(domains={})
    with pytest.raises(ValueError, match="does not match"):
        resolve_ownership_domains(domains={"wrong": d})
    with pytest.raises(ValueError, match="not present"):
        resolve_ownership_domains(domains={"dense": d}, default_domain="missing")
    with pytest.raises(ValueError, match="device_type"):
        resolve_ownership_domains(
            domains={
                "dense": d,
                "cpu_dom": OwnershipDomain("cpu_dom", shard_mesh=_FakeMesh(2, device_type="cpu")),
            }
        )


def test_param_policy_resolves_domain_overrides_and_fn() -> None:
    model = nn.Module()
    model.attn = nn.Linear(4, 4, bias=False)
    model.experts = nn.Linear(4, 4, bias=False)
    params = dict(model.named_parameters())
    param_to_fqn = {p: n for n, p in params.items()}

    # Default policy keeps domain=None for full backward compatibility.
    plain = resolve_param_policies(params=params.values(), param_to_fqn=param_to_fqn)
    assert all(p.domain is None for p in plain.values())

    # Declarative param_policy with defaults + overrides (both set and inline).
    resolved = resolve_param_policies(
        params=params.values(),
        param_to_fqn=param_to_fqn,
        param_policy={
            "defaults": {"route": "muon", "domain": "dense"},
            "overrides": [
                {"name": "experts", "set": {"domain": "expert"}},
            ],
        },
    )
    assert resolved[params["attn.weight"]].domain == "dense"
    assert resolved[params["experts.weight"]].domain == "expert"

    # Inline domain override form.
    inline = resolve_param_policies(
        params=params.values(),
        param_to_fqn=param_to_fqn,
        param_policy={
            "overrides": [{"name": "experts", "domain": "expert"}],
        },
    )
    assert inline[params["attn.weight"]].domain is None
    assert inline[params["experts.weight"]].domain == "expert"

    # param_policy_fn returning dict or DMuonParamPolicy.
    fn_resolved = resolve_param_policies(
        params=params.values(),
        param_to_fqn=param_to_fqn,
        param_policy_fn=lambda name, _p: (
            DMuonParamPolicy(route="muon", domain="expert")
            if "experts" in name
            else {"route": "muon", "domain": "dense"}
        ),
    )
    assert fn_resolved[params["attn.weight"]].domain == "dense"
    assert fn_resolved[params["experts.weight"]].domain == "expert"

    # Reject invalid domain values.
    with pytest.raises(ValueError):
        resolve_param_policies(
            params=params.values(),
            param_to_fqn=param_to_fqn,
            param_policy={"defaults": {"domain": "   "}},
        )
    with pytest.raises(TypeError):
        resolve_param_policies(
            params=params.values(),
            param_to_fqn=param_to_fqn,
            param_policy_fn=lambda _n, _p: {"domain": 42},
        )
