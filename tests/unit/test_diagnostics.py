from types import SimpleNamespace

import torch
import torch.nn as nn

from dmuon.diagnostics import summarize_comm_plan, summarize_param_groups


class FakeProcessGroup:
    def __init__(self, rank: int = 0, size: int = 4) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class FakeDedicatedParam:
    def __init__(
        self,
        param: nn.Parameter,
        *,
        name: str,
        route: str,
        owner_rank: tuple[int, int],
        is_owner: bool,
        replicate_group: FakeProcessGroup | None = None,
    ) -> None:
        self._placeholder = param
        self._orig_param = param
        self.param_name = name
        self._dmuon_route = route
        self.owner_rank = owner_rank
        self.owner_shard = owner_rank[0]
        self.owner_replicate = owner_rank[1]
        self.is_owner = is_owner
        self.is_dtensor = False
        self.numel = param.numel()
        self.full_shape = tuple(param.shape)
        self._owned_data = torch.zeros_like(param)
        self._owner_global_rank = owner_rank[0]
        self._owner_replicate_global_rank = owner_rank[1]
        self.replicate_group = replicate_group

    def uses_sharded_adamw(self) -> bool:
        return self._dmuon_route == "sharded_adamw"


class FakeOptimizer:
    def __init__(
        self,
        matrix_param: nn.Parameter,
        base_param: nn.Parameter,
        regular_param: nn.Parameter,
        matrix_dp: FakeDedicatedParam,
        base_dp: FakeDedicatedParam,
    ) -> None:
        self.param_groups = [
            {
                "params": [matrix_param],
                "group_name": "matrix/muon",
                "semantic_group_name": "matrix",
                "subgroup_type": "muon",
                "use_muon": True,
            },
            {
                "params": [base_param, regular_param],
                "group_name": "base/adamw",
                "semantic_group_name": "base",
                "subgroup_type": "adamw",
                "use_muon": False,
            },
        ]
        self._all_dedicated_params = [matrix_dp, base_dp]
        self._dp_to_muon_group_idx = {id(matrix_dp): 0}
        self._dp_to_adamw_group_idx = {id(base_dp): 1}
        self._muon_group_dps = {0: [matrix_dp]}
        self._adamw_group_dps = {1: [base_dp]}
        self._adamw_group_params = {1: [regular_param]}


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = nn.Parameter(torch.zeros(4, 8))
        self.base = nn.Parameter(torch.zeros(8))
        self.regular = nn.Parameter(torch.zeros(2))
        self.layer = nn.Module()


def test_summarize_param_groups_counts_dmuon_type_split_routes() -> None:
    model = TinyModel()
    matrix_dp = FakeDedicatedParam(
        model.matrix, name="matrix", route="muon", owner_rank=(0, 0), is_owner=True
    )
    base_dp = FakeDedicatedParam(
        model.base,
        name="base",
        route="sharded_adamw",
        owner_rank=(1, 0),
        is_owner=True,
    )
    optimizer = FakeOptimizer(model.matrix, model.base, model.regular, matrix_dp, base_dp)

    summary = summarize_param_groups(model, optimizer)

    assert summary["available"] is True
    assert summary["num_groups"] == 2
    assert summary["dedicated_param_count"] == 2
    assert summary["owned_dedicated_param_count"] == 2
    assert summary["groups"][0]["dedicated_muon_param_count"] == 1
    assert summary["groups"][1]["dedicated_adamw_param_count"] == 1
    assert summary["groups"][1]["adamw_param_count"] == 1
    assert {row["route"] for row in summary["parameters"]} == {
        "muon",
        "sharded_adamw",
    }


def test_summarize_comm_plan_reports_forward_and_stage2_roots() -> None:
    model = TinyModel()
    shard_group = FakeProcessGroup(rank=1, size=4)
    replicate_group = FakeProcessGroup(rank=0, size=2)
    matrix_dp = FakeDedicatedParam(
        model.matrix,
        name="matrix",
        route="muon",
        owner_rank=(1, 0),
        is_owner=True,
        replicate_group=replicate_group,
    )
    base_dp = FakeDedicatedParam(
        model.base,
        name="base",
        route="sharded_adamw",
        owner_rank=(2, 0),
        is_owner=False,
        replicate_group=replicate_group,
    )
    group = SimpleNamespace(
        params=[matrix_dp, base_dp],
        _debug_name="layer0",
        _dp_group=shard_group,
        _by_owner={(1, 0): [matrix_dp], (2, 0): [base_dp]},
        _global_owner_ranks={(1, 0): 1, (2, 0): 2},
    )
    model.layer._dedicated_state = SimpleNamespace(group=group)

    plan = summarize_comm_plan(model)

    assert plan["available"] is True
    assert plan["group_count"] == 1
    assert plan["groups"][0]["debug_name"] == "layer0"
    assert plan["groups"][0]["owner_bucket_count"] == 1
    assert plan["totals"]["stage1_shard_reduce_tensor_bytes"] > 0
    assert plan["totals"]["stage2_replicate_reduce_tensor_bytes_this_rank"] > 0
    assert plan["groups"][0]["param_collectives"][0]["stage2_replicate_axis_active_on_this_rank"]
