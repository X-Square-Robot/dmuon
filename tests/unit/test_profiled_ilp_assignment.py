"""Unit tests for profiled_ilp assignment and load modeling."""

import os
import sys
from collections import defaultdict

os.environ.setdefault("DMUON_CACHE_DIR", "/tmp/dmuon_test_cache")

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import pytest
import torch
import torch.nn as nn

pytest.importorskip("scipy")

from dmuon._core import profiled_ilp as profiled_ilp_mod
from dmuon._core.partition import compute_balanced_assignment
from dmuon._core.profiled_ilp import (
    ProfiledShapeTiming,
    solve_profiled_assignment,
)
from dmuon.optim.profiled_batch import ProfiledILPConfig


class FakeDeviceMesh:
    mesh_dim_names = None

    def __init__(self, world_size: int):
        self._world_size = world_size

    def size(self):
        return self._world_size


class ShapeBag(nn.Module):
    def __init__(self, specs: list[tuple[str, tuple[int, ...], int]]):
        super().__init__()
        for prefix, shape, count in specs:
            block = nn.Module()
            for idx in range(count):
                block.register_parameter(
                    f"p{idx}",
                    nn.Parameter(torch.empty(shape, device="meta")),
                )
            self.add_module(prefix, block)


def _patch_dependency_gate(monkeypatch):
    monkeypatch.setattr(
        profiled_ilp_mod,
        "require_profiled_ilp_dependencies",
        lambda: None,
    )


def _lpt_loads_ms(model: nn.Module, ranks: int, timings: dict[tuple[int, int], dict[int, float]]):
    assignment = compute_balanced_assignment(
        model,
        FakeDeviceMesh(ranks),
        predicate=lambda _name, _param: True,
        owner_strategy="lpt",
    ).dp_owners
    loads = [0.0 for _ in range(ranks)]
    for param, owner in assignment.items():
        shape = tuple(int(dim) for dim in param.shape)
        if len(shape) < 2:
            continue
        rows = int(shape[0])
        cols = 1
        for dim in shape[1:]:
            cols *= int(dim)
        loads[int(owner)] += timings[(rows, cols)][1]
    return loads


def test_profiled_ilp_assigns_batches_and_improves_max_rank_load(monkeypatch):
    _patch_dependency_gate(monkeypatch)
    model = ShapeBag(
        [
            ("a", (4, 4), 9),
            ("b", (8, 4), 6),
        ]
    )
    measured_timings = {
        (4, 4): {1: 1.0, 2: 1.45, 3: 2.0, 4: 2.75, 5: 3.55, 6: 4.3, 7: 5.05, 8: 5.8},
        (8, 4): {1: 2.0, 2: 3.1, 3: 4.2, 4: 5.6, 5: 7.2, 6: 8.8},
    }
    backend_choices = {
        shape: {batch: "cublas" for batch in times}
        for shape, times in measured_timings.items()
    }

    result = compute_balanced_assignment(
        model,
        FakeDeviceMesh(4),
        predicate=lambda _name, _param: True,
        owner_strategy="profiled_ilp",
        profiled_ilp_config={
            "max_batch": 8,
            "measured_timings": measured_timings,
            "measured_backend_choices": backend_choices,
            "backends": "cublas",
            "ilp_mip_rel_gap": 0.0,
        },
    )

    params = list(model.parameters())
    assert set(result.dp_owners) == set(params)
    assert set(result.batch_groups) == set(params)
    assert result.metadata["strategy"] == "profiled_ilp"

    grouped = defaultdict(list)
    for param, meta in result.batch_groups.items():
        grouped[meta.group_key].append((param, meta))
        assert meta.batch_size <= min(8, 9 if meta.shape == (4, 4) else 6)
        assert meta.backend == "cublas"

    for items in grouped.values():
        batch_size = items[0][1].batch_size
        assert len(items) == batch_size

    old_loads = _lpt_loads_ms(model, 4, measured_timings)
    new_loads = result.metadata["rank_loads_ms_with_fallback"]
    assert max(new_loads) < max(old_loads)


def test_profiled_ilp_clips_batch_candidates_to_shape_count(monkeypatch):
    _patch_dependency_gate(monkeypatch)
    model = ShapeBag([("tiny", (4, 4), 3)])
    result = compute_balanced_assignment(
        model,
        FakeDeviceMesh(2),
        predicate=lambda _name, _param: True,
        owner_strategy="profiled_ilp",
        profiled_ilp_config={
            "max_batch": 8,
            "measured_timings": {(4, 4): {1: 1.0, 2: 1.5, 3: 2.1, 8: 4.0}},
            "measured_backend_choices": {
                (4, 4): {1: "cublas", 2: "cublas", 3: "cublas", 8: "cublas"}
            },
            "ilp_mip_rel_gap": 0.0,
        },
    )

    assert result.batch_groups
    assert all(meta.batch_size <= 3 for meta in result.batch_groups.values())


def test_profiled_ilp_fallback_params_are_assigned_but_not_batched(monkeypatch):
    _patch_dependency_gate(monkeypatch)
    model = ShapeBag(
        [
            ("matrix", (4, 4), 4),
            ("vector", (4,), 2),
        ]
    )
    result = compute_balanced_assignment(
        model,
        FakeDeviceMesh(2),
        predicate=lambda _name, _param: True,
        owner_strategy="profiled_ilp",
        profiled_ilp_config={
            "max_batch": 4,
            "measured_timings": {(4, 4): {1: 1.0, 2: 1.5, 3: 2.2, 4: 2.9}},
            "measured_backend_choices": {
                (4, 4): {1: "cublas", 2: "cublas", 3: "cublas", 4: "cublas"}
            },
        },
    )

    assert len(result.dp_owners) == 6
    assert len(result.batch_groups) == 4
    assert result.metadata["fallback_param_count"] == 2


def test_profiled_ilp_stage2_failure_falls_back_to_stage1(monkeypatch):
    timing = ProfiledShapeTiming(
        "shape=4x4",
        (4, 4),
        5,
        {1: 1.0, 2: 1.6, 3: 2.4},
        {1: "cublas", 2: "cublas", 3: "cublas"},
    )

    def fail_stage2(*_args, **_kwargs):
        raise RuntimeError("synthetic stage2 failure")

    monkeypatch.setattr(
        profiled_ilp_mod,
        "_solve_min_total_work_at_load",
        fail_stage2,
    )
    solution = solve_profiled_assignment(
        [timing],
        3,
        ProfiledILPConfig(ilp_mip_rel_gap=0.0),
    )

    assigned = 0
    for batches in solution["shape_plan"][0].items():
        batch_size, count = batches
        assigned += int(batch_size) * int(count)
    assert assigned == timing.count
    assert solution["stage2_used"] is False
