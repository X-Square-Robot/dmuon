"""Unit tests for TP auto-detection (``dmuon._core.tp``).

All tests use a fake ProcessGroup so they run on a single CPU without
NCCL or multi-rank orchestration.  The DTensor machinery still honours
``mesh_dim_names`` / placements even under the fake backend, which is
all we need to exercise the T1 detection logic.

See ``docs/internal/research/tp_design.md`` §9.1.
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
try:
    from torch.distributed import init_device_mesh
except ImportError:
    from torch.distributed.device_mesh import init_device_mesh
try:
    from torch.distributed.tensor import distribute_tensor
except ImportError:
    pytest.skip("DTensor distribute_tensor is unavailable", allow_module_level=True)
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from dmuon._core.tp import get_tp_mesh, get_tp_shard_dim, is_tp_sharded


@pytest.fixture(scope="module", autouse=True)
def _fake_pg():
    """Initialise a fake process group sized for the largest mesh in this module.

    The 3D HSDP+TP test uses a (2, 2, 2) mesh — 8 ranks — which sets the
    floor.  The fixture destroys any pre-existing group so each module gets
    the world_size it needs (pytest shares the process across modules).
    """
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(
        backend="fake",
        store=FakeStore(),
        rank=0,
        world_size=8,
    )
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def test_is_tp_sharded_colwise():
    """ColwiseParallel: DTensor on TP sub-mesh with Shard(0) placement."""
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    W = distribute_tensor(torch.randn(64, 32), mesh["tp"], [Shard(0)])
    p = nn.Parameter(W)
    assert is_tp_sharded(p, frozenset({"dp"})) is True
    assert get_tp_shard_dim(p, frozenset({"dp"})) == 0
    assert get_tp_mesh(p, frozenset({"dp"})).size() == 2


def test_is_tp_sharded_rowwise():
    """RowwiseParallel: Shard(1) placement on the TP axis."""
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    W = distribute_tensor(torch.randn(32, 64), mesh["tp"], [Shard(1)])
    p = nn.Parameter(W)
    assert is_tp_sharded(p, frozenset({"dp"})) is True
    assert get_tp_shard_dim(p, frozenset({"dp"})) == 1


def test_tp_replicated_returns_false():
    """TP-replicated params (Replicate on tp dim) are not TP-sharded."""
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    W = distribute_tensor(torch.randn(32, 32), mesh["tp"], [Replicate()])
    p = nn.Parameter(W)
    assert is_tp_sharded(p, frozenset({"dp"})) is False


def test_pure_dp_returns_false():
    """DTensor sharded only on a DP dim → not TP-sharded."""
    mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("dp",))
    W = distribute_tensor(torch.randn(32, 32), mesh, [Shard(0)])
    p = nn.Parameter(W)
    assert is_tp_sharded(p, frozenset({"dp"})) is False


def test_non_dtensor_returns_false():
    """Plain tensors (no DTensor wrapping) are not TP-sharded."""
    p = nn.Parameter(torch.randn(32, 32))
    assert is_tp_sharded(p, frozenset({"dp"})) is False


def test_unnamed_mesh_raises():
    """An unnamed DeviceMesh under a DTensor is a user error: raise ValueError."""
    mesh = init_device_mesh("cpu", (2, 2))  # no mesh_dim_names
    W = distribute_tensor(torch.randn(32, 32), mesh, [Shard(0), Shard(1)])
    p = nn.Parameter(W)
    with pytest.raises(ValueError, match="mesh_dim_names"):
        is_tp_sharded(p, frozenset({"dp"}))


def test_tp_size_1_degenerate_returns_false():
    """A TP dim with size 1 is semantically no-TP (§8.4); detection must
    NOT treat it as TP-sharded, otherwise T2 dispatches zero-communication
    collectives that still cost a handshake."""
    mesh = init_device_mesh("cpu", (4, 1), mesh_dim_names=("dp", "tp"))
    W = distribute_tensor(torch.randn(32, 32), mesh["tp"], [Shard(0)])
    p = nn.Parameter(W)
    assert is_tp_sharded(p, frozenset({"dp"})) is False


def test_hsdp_plus_tp_3d():
    """3D mesh (replicate, shard, tp): TP is the one dim name absent from DP set."""
    mesh = init_device_mesh(
        "cpu", (2, 2, 2), mesh_dim_names=("replicate", "shard", "tp")
    )
    W = distribute_tensor(torch.randn(64, 32), mesh["tp"], [Shard(0)])
    p = nn.Parameter(W)
    dp_names = frozenset({"replicate", "shard"})
    assert is_tp_sharded(p, dp_names) is True
    assert get_tp_mesh(p, dp_names).size() == 2
    assert get_tp_shard_dim(p, dp_names) == 0
