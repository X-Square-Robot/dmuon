"""End-to-end partition tests covering the TP auto-detection path.

Exercises ``compute_balanced_assignment`` returning ``AssignmentResult``
for models that contain a mix of TP-sharded and non-TP parameters, plus
the backward-compat case where no DTensor is present at all.

See ``docs/internal/research/tp_design.md`` §9.3.
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
from torch.distributed.tensor.placement_types import Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from dmuon._core.partition import AssignmentResult, compute_balanced_assignment


@pytest.fixture(scope="module", autouse=True)
def _fake_pg():
    """World size 16 covers the 3D HSDP+TP (2,4,2)=16 mesh plus smaller cases."""
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(
        backend="fake",
        store=FakeStore(),
        rank=0,
        world_size=16,
    )
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _make_mixed_model(
    tp_mesh,
    hidden: int = 256,
    intermediate: int = 1024,
    num_layers: int = 2,
) -> tuple[nn.Module, list[nn.Parameter], list[nn.Parameter]]:
    """Build a toy transformer where proj weights are TP-sharded DTensors and
    norm / embed weights are plain tensors.  Returns (model, tp_params, non_tp_params).
    """
    model = nn.Module()
    model.embed = nn.Embedding(100, hidden)  # non-TP
    model.layers = nn.ModuleDict()
    tp_params: list[nn.Parameter] = []
    non_tp_params: list[nn.Parameter] = []
    for i in range(num_layers):
        blk = nn.Module()
        # ColwiseParallel weights → Shard(0) on TP axis
        blk.q_proj = nn.Module()
        blk.q_proj.weight = nn.Parameter(
            distribute_tensor(torch.randn(hidden, hidden), tp_mesh, [Shard(0)])
        )
        blk.gate_proj = nn.Module()
        blk.gate_proj.weight = nn.Parameter(
            distribute_tensor(
                torch.randn(intermediate, hidden), tp_mesh, [Shard(0)]
            )
        )
        # RowwiseParallel → Shard(1) on TP axis
        blk.o_proj = nn.Module()
        blk.o_proj.weight = nn.Parameter(
            distribute_tensor(torch.randn(hidden, hidden), tp_mesh, [Shard(1)])
        )
        # Plain (non-TP) layernorm-style param
        blk.ln = nn.Module()
        blk.ln.weight = nn.Parameter(torch.randn(hidden))
        model.layers[str(i)] = blk

        tp_params += [blk.q_proj.weight, blk.gate_proj.weight, blk.o_proj.weight]
        non_tp_params += [blk.ln.weight]
    non_tp_params += [model.embed.weight]

    # Wire named_parameters to see nested modules (nn.Module without children
    # still picks up attribute-held Parameters automatically).
    return model, tp_params, non_tp_params


def test_partition_1d_dp_plus_tp_auto_detect():
    """1D DP shard × TP: every proj DTensor ends up in tp_owners with a
    legal LPT-selected rank; non-TP params stay out of tp_owners."""
    mesh = init_device_mesh("cpu", (4, 2), mesh_dim_names=("shard", "tp"))
    dp_mesh, tp_mesh = mesh["shard"], mesh["tp"]

    model, tp_params, non_tp_params = _make_mixed_model(tp_mesh)

    result = compute_balanced_assignment(
        model, dp_mesh, predicate=lambda n, p: "proj" in n or "ln" in n
    )
    assert isinstance(result, AssignmentResult)

    # Every TP-sharded proj param: in dp_owners (int) and tp_owners.
    for p in tp_params:
        assert p in result.dp_owners
        assert isinstance(result.dp_owners[p], int)
        assert 0 <= result.tp_owners[p] < 2
    assert set(result.tp_owners[p] for p in tp_params) == {0, 1}

    # Non-TP params (ln, embed): in dp_owners, NOT in tp_owners
    for p in non_tp_params:
        if p in result.dp_owners:
            assert p not in result.tp_owners


def test_partition_hsdp_plus_tp_auto_detect():
    """HSDP (2×4) + TP (2) on a 3D named mesh."""
    mesh = init_device_mesh(
        "cpu", (2, 4, 2), mesh_dim_names=("replicate", "shard", "tp")
    )
    shard_mesh = mesh["shard"]
    replicate_mesh = mesh["replicate"]
    tp_mesh = mesh["tp"]

    model, tp_params, non_tp_params = _make_mixed_model(tp_mesh)

    result = compute_balanced_assignment(
        model,
        shard_mesh,
        predicate=lambda n, p: "proj" in n or "ln" in n,
        replicate_mesh=replicate_mesh,
    )
    assert isinstance(result, AssignmentResult)

    # TP-sharded params: dp_owners value is a (shard, replicate) tuple and
    # TP owners are valid LPT-selected local TP ranks.
    for p in tp_params:
        coord = result.dp_owners[p]
        assert isinstance(coord, tuple) and len(coord) == 2
        s, r = coord
        assert 0 <= s < 4 and 0 <= r < 2
        assert 0 <= result.tp_owners[p] < 2

    # Non-TP params: dp_owners tuple-shaped, absent from tp_owners
    for p in non_tp_params:
        if p in result.dp_owners:
            assert isinstance(result.dp_owners[p], tuple)
            assert p not in result.tp_owners


def test_tp_sharded_small_params_do_not_merge():
    """TP-sharded params stay standalone even when below SMALL_PARAM_THRESHOLD.

    If they were merged into one allocation unit, all params in the same layer
    would receive the same DP owner.  Keeping them standalone lets the existing
    same-layer constraint spread them across owner slots.
    """
    mesh = init_device_mesh("cpu", (4, 2), mesh_dim_names=("shard", "tp"))
    model, tp_params, _non_tp_params = _make_mixed_model(
        mesh["tp"], hidden=64, intermediate=128, num_layers=1
    )

    result = compute_balanced_assignment(
        model,
        mesh["shard"],
        predicate=lambda name, _p: "proj" in name,
    )

    assert len(tp_params) == 3
    assert len({result.dp_owners[p] for p in tp_params}) == 3
    assert all(p in result.tp_owners for p in tp_params)


def test_partition_tp_owners_empty_when_no_dtensor():
    """Plain nn.Linear model → no DTensor → tp_owners is empty; dp_owners
    matches pre-TP dict-style usage exactly."""

    class MiniBlock(nn.Module):
        def __init__(self, h=128):
            super().__init__()
            self.q_proj = nn.Linear(h, h, bias=False)
            self.o_proj = nn.Linear(h, h, bias=False)

    class Mini(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 128)
            self.layers = nn.ModuleDict({"0": MiniBlock(), "1": MiniBlock()})

    model = Mini()

    class FakeDeviceMesh:
        mesh_dim_names = None

        def __init__(self, n):
            self._n = n

        def size(self):
            return self._n

    result = compute_balanced_assignment(
        model, FakeDeviceMesh(4), predicate=lambda n, p: "proj" in n
    )
    assert isinstance(result, AssignmentResult)
    assert result.tp_owners == {}
    # All proj params present in dp_owners; ints (1D mode)
    proj_count = sum(1 for n, _ in model.named_parameters() if "proj" in n)
    assert len(result.dp_owners) == proj_count
    for owner in result.dp_owners.values():
        assert isinstance(owner, int)


def test_partition_hsdp_no_tp_tp_owners_empty():
    """HSDP-only (no TP dim in the mesh): tp_owners should stay empty."""
    mesh = init_device_mesh(
        "cpu", (2, 4), mesh_dim_names=("replicate", "shard")
    )
    shard_mesh = mesh["shard"]
    replicate_mesh = mesh["replicate"]

    # Plain model — no DTensor anywhere
    class MiniBlock(nn.Module):
        def __init__(self, h=128):
            super().__init__()
            self.q_proj = nn.Linear(h, h, bias=False)

    class Mini(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict({"0": MiniBlock(), "1": MiniBlock()})

    model = Mini()

    result = compute_balanced_assignment(
        model,
        shard_mesh,
        predicate=lambda n, p: "proj" in n,
        replicate_mesh=replicate_mesh,
    )
    assert result.tp_owners == {}
    # HSDP dp_owners remain (shard, replicate) tuples
    for owner in result.dp_owners.values():
        assert isinstance(owner, tuple) and len(owner) == 2
