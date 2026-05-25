"""Unit tests for TP owner-assignment strategies.

TP-owner LPT is stateful across parameters, so the public behaviour is
tested through ``compute_balanced_assignment``.
"""

from __future__ import annotations

import os
import sys
import inspect

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
try:
    from torch.distributed import init_device_mesh
except ImportError:
    from torch.distributed.device_mesh import init_device_mesh
try:
    from torch.distributed.tensor import distribute_tensor
except ImportError:
    pytest.skip("DTensor distribute_tensor is unavailable", allow_module_level=True)
from torch.distributed.tensor.placement_types import Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from dmuon import dedicate_params, dedicate_params_ddp
from dmuon._core.partition import compute_balanced_assignment


@pytest.fixture(scope="module", autouse=True)
def _fake_pg():
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(
        backend="fake",
        store=FakeStore(),
        rank=0,
        world_size=4,
    )
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _make_tp_model(tp_mesh, rows: list[int]) -> tuple[nn.Module, list[nn.Parameter]]:
    model = nn.Module()
    model.layers = nn.ModuleDict()
    params: list[nn.Parameter] = []
    for i, row in enumerate(rows):
        layer = nn.Module()
        layer.proj = nn.Module()
        layer.proj.weight = nn.Parameter(
            distribute_tensor(torch.randn(row, 16), tp_mesh, [Shard(0)])
        )
        model.layers[str(i)] = layer
        params.append(layer.proj.weight)
    return model, params


def test_tp_owner_lpt_balances_within_dp_owner():
    mesh = init_device_mesh("cpu", (1, 2), mesh_dim_names=("dp", "tp"))
    model, params = _make_tp_model(mesh["tp"], rows=[64, 48, 32, 16])

    result = compute_balanced_assignment(
        model,
        mesh["dp"],
        predicate=lambda name, _p: "proj" in name,
    )

    assert set(result.tp_owners[p] for p in params) == {0, 1}
    loads = [0, 0]
    for p in params:
        loads[result.tp_owners[p]] += p.numel()
    assert loads == [80 * 16, 80 * 16]


def test_tp_owner_lpt_bucketed_by_dp_owner():
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    model, params = _make_tp_model(mesh["tp"], rows=[100, 90, 80, 70])

    result = compute_balanced_assignment(
        model,
        mesh["dp"],
        predicate=lambda name, _p: "proj" in name,
    )

    owners_by_dp: dict[int, set[int]] = {}
    for p in params:
        dp_owner = result.dp_owners[p]
        assert isinstance(dp_owner, int)
        owners_by_dp.setdefault(dp_owner, set()).add(result.tp_owners[p])

    assert owners_by_dp == {0: {0, 1}, 1: {0, 1}}


def test_rank0_strategy_rejected_public_api():
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    model, _params = _make_tp_model(mesh["tp"], rows=[16])

    with pytest.raises(ValueError, match="supports only 'lpt'"):
        compute_balanced_assignment(
            model,
            mesh["dp"],
            predicate=lambda name, _p: "proj" in name,
            tp_owner_strategy="rank0",
        )


def test_dedicate_params_no_public_tp_owner_strategy():
    signature = inspect.signature(dedicate_params)
    assert "tp_owner_strategy" not in signature.parameters


def test_unknown_tp_owner_strategy_rejected():
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    model, _params = _make_tp_model(mesh["tp"], rows=[16])

    with pytest.raises(ValueError, match="Unsupported tp_owner_strategy"):
        compute_balanced_assignment(
            model,
            mesh["dp"],
            predicate=lambda name, _p: "proj" in name,
            tp_owner_strategy="bogus",
        )


def test_ddp_plus_tp_rejected_as_unsupported():
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    model, _params = _make_tp_model(mesh["tp"], rows=[16])

    with pytest.raises(NotImplementedError, match="does not support TP-sharded"):
        dedicate_params_ddp(
            model,
            mesh["dp"],
            predicate=lambda name, _p: "proj" in name,
        )
