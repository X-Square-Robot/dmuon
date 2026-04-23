"""Unit tests for TP owner-assignment strategies.

See ``docs/internal/research/tp_design.md`` §9.2.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import init_device_mesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from dmuon._core.tp import assign_tp_owner


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


@pytest.fixture
def tp_param():
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    W = distribute_tensor(torch.randn(16, 16), mesh["tp"], [Shard(0)])
    return nn.Parameter(W)


def test_assign_tp_owner_rank0(tp_param):
    assert assign_tp_owner(tp_param, frozenset({"dp"}), strategy="rank0") == 0


def test_assign_tp_owner_default_is_rank0(tp_param):
    assert assign_tp_owner(tp_param, frozenset({"dp"})) == 0


def test_assign_tp_owner_lpt_deferred(tp_param):
    with pytest.raises(NotImplementedError):
        assign_tp_owner(tp_param, frozenset({"dp"}), strategy="lpt")


def test_assign_tp_owner_unknown_strategy_raises(tp_param):
    with pytest.raises(ValueError, match="Unknown tp_owner_strategy"):
        assign_tp_owner(tp_param, frozenset({"dp"}), strategy="bogus")
