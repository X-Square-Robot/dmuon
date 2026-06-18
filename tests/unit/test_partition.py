"""Unit tests for the partition algorithm (single GPU, no dist required)."""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pytest
import torch
import torch.nn as nn

from dmuon._core.partition import (
    SMALL_PARAM_THRESHOLD,
    _extract_layer_id,
    _matrix_optimizer_cost_units,
    compute_balanced_assignment,
)


class FakeDeviceMesh:
    """Minimal mock for DeviceMesh to test partition logic without distributed."""

    mesh_dim_names = None

    def __init__(self, world_size: int):
        self._world_size = world_size

    def size(self):
        return self._world_size


# ---- Test _extract_layer_id ----


def test_extract_layer_id_standard():
    assert _extract_layer_id("model.layers.3.mlp.gate_proj.weight") == "model.layers.3"
    assert (
        _extract_layer_id("model.layers.12.self_attn.q_proj.weight")
        == "model.layers.12"
    )


def test_extract_layer_id_no_layer():
    assert _extract_layer_id("model.embed_tokens.weight") is None
    assert _extract_layer_id("lm_head.weight") is None


def test_extract_layer_id_vit_blocks():
    """ViT uses `blocks.N` — extraction should distinguish from `layers.N`."""
    assert _extract_layer_id("visual.blocks.5.attn.qkv.weight") == "visual.blocks.5"
    assert _extract_layer_id("visual.blocks.0.mlp.fc1.weight") == "visual.blocks.0"


def test_extract_layer_id_prefix_disambiguates():
    """`blocks.3` and `layers.3` must not collide into the same bucket."""
    a = _extract_layer_id("visual.blocks.3.attn.qkv.weight")
    b = _extract_layer_id("model.layers.3.self_attn.q_proj.weight")
    assert a != b, f"collision: {a} == {b}"
    assert a == "visual.blocks.3"
    assert b == "model.layers.3"


def test_extract_layer_id_root_level():
    """Bare `layers.N` / `blocks.N` with no parent should not crash."""
    assert _extract_layer_id("layers.0.weight") == "_root.layers.0"
    assert _extract_layer_id("blocks.2.weight") == "_root.blocks.2"


# ---- Test balanced assignment ----


class MiniTransformerBlock(nn.Module):
    def __init__(self, hidden=512, intermediate=2048):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden // 4, bias=False)
        self.v_proj = nn.Linear(hidden, hidden // 4, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.ln = nn.LayerNorm(hidden)


class MiniModel(nn.Module):
    def __init__(self, num_layers=4, hidden=512, intermediate=2048):
        super().__init__()
        self.embed = nn.Embedding(1000, hidden)
        self.layers = nn.ModuleDict(
            {
                str(i): MiniTransformerBlock(hidden, intermediate)
                for i in range(num_layers)
            }
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers.values():
            x = layer.ln(x)  # simplified
        return self.norm(x)


@pytest.fixture
def model():
    return MiniModel(num_layers=4, hidden=512, intermediate=2048)


def test_assignment_covers_all_proj(model):
    """All proj params should be assigned."""
    mesh = FakeDeviceMesh(8)
    result = compute_balanced_assignment(
        model, mesh, predicate=lambda n, p: "proj" in n
    )
    assignment = result.dp_owners
    proj_params = [p for n, p in model.named_parameters() if "proj" in n]
    assert len(assignment) == len(proj_params)
    for p in proj_params:
        assert p in assignment


def test_assignment_excludes_non_proj(model):
    """Non-proj params (embed, layernorm) should NOT be assigned."""
    mesh = FakeDeviceMesh(8)
    result = compute_balanced_assignment(
        model, mesh, predicate=lambda n, p: "proj" in n
    )
    assignment = result.dp_owners
    for n, p in model.named_parameters():
        if "proj" not in n:
            assert p not in assignment


def test_balance(model):
    """Rank loads should be balanced within 5%."""
    mesh = FakeDeviceMesh(8)
    result = compute_balanced_assignment(
        model, mesh, predicate=lambda n, p: "proj" in n
    )
    assignment = result.dp_owners
    rank_loads = [0] * 8
    for param, rank in assignment.items():
        rank_loads[rank] += param.numel()

    max_load = max(rank_loads)
    min_load = min(r for r in rank_loads if r > 0)
    imbalance = (max_load - min_load) / max_load
    assert imbalance < 0.05, f"Imbalance too high: {imbalance:.1%}, loads={rank_loads}"


def test_rank0_owner_strategy_concentrates_work(model):
    mesh = FakeDeviceMesh(8)
    assignment = compute_balanced_assignment(
        model,
        mesh,
        predicate=lambda n, p: "proj" in n,
        owner_strategy="rank0",
    ).dp_owners

    assert assignment
    assert set(assignment.values()) == {0}


def test_round_robin_owner_strategy_uses_all_ranks():
    mesh = FakeDeviceMesh(8)
    model = MiniModel(num_layers=4, hidden=1024, intermediate=8192)
    assignment = compute_balanced_assignment(
        model,
        mesh,
        predicate=lambda n, p: "proj" in n,
        owner_strategy="round_robin",
    ).dp_owners

    assert assignment
    assert set(assignment.values()) == set(range(8))


def test_shape_aware_cost_weights_projection_compute_above_embedding_bytes():
    """LPT should not treat huge embeddings and projection matrices by numel only."""
    embed = nn.Parameter(torch.empty(65536, 1024, device="meta"))
    proj = nn.Parameter(torch.empty(4096, 4096, device="meta"))

    assert embed.numel() > proj.numel()
    assert _matrix_optimizer_cost_units(
        "model.layers.0.self_attn.q_proj.weight", proj
    ) > _matrix_optimizer_cost_units("model.embed_tokens.weight", embed)


def test_owner_cost_model_ablation_changes_lpt_order():
    mesh = FakeDeviceMesh(2)
    model = nn.Module()
    model.embed_tokens = nn.Embedding(65536, 1024, device="meta")
    model.proj = nn.Linear(4096, 4096, bias=False, device="meta")

    numel_assignment = compute_balanced_assignment(
        model,
        mesh,
        predicate=lambda _n, _p: True,
        owner_cost_model="numel",
    ).dp_owners
    optimizer_assignment = compute_balanced_assignment(
        model,
        mesh,
        predicate=lambda _n, _p: True,
        owner_cost_model="optimizer",
    ).dp_owners

    assert numel_assignment[model.embed_tokens.weight] == 0
    assert numel_assignment[model.proj.weight] == 1
    assert optimizer_assignment[model.proj.weight] == 0
    assert optimizer_assignment[model.embed_tokens.weight] == 1


def test_unknown_owner_strategy_rejected(model):
    mesh = FakeDeviceMesh(8)

    with pytest.raises(ValueError, match="Unsupported owner_strategy"):
        compute_balanced_assignment(
            model,
            mesh,
            predicate=lambda n, p: "proj" in n,
            owner_strategy="bogus",
        )


def test_unknown_owner_cost_model_rejected(model):
    mesh = FakeDeviceMesh(8)

    with pytest.raises(ValueError, match="Unsupported owner_cost_model"):
        compute_balanced_assignment(
            model,
            mesh,
            predicate=lambda n, p: "proj" in n,
            owner_cost_model="bogus",
        )


def test_same_layer_different_ranks(model):
    """Large params in the same layer should go to different ranks."""
    mesh = FakeDeviceMesh(8)
    result = compute_balanced_assignment(
        model, mesh, predicate=lambda n, p: "proj" in n
    )
    assignment = result.dp_owners
    # Check layer 0
    layer0_large = {}
    for n, p in model.named_parameters():
        if "layers.0" in n and "proj" in n and p.numel() >= SMALL_PARAM_THRESHOLD:
            layer0_large[n] = assignment[p]

    # All large params in the same layer should have different owner ranks
    ranks = list(layer0_large.values())
    assert len(ranks) == len(
        set(ranks)
    ), f"Same-layer large params assigned to same rank: {layer0_large}"


def test_max_owners_per_assignment_group_caps_owner_spread():
    """A custom group can deliberately trade load balance for fewer broadcasts."""
    mesh = FakeDeviceMesh(8)
    model = MiniModel(num_layers=1, hidden=1024, intermediate=8192)

    default = compute_balanced_assignment(
        model,
        mesh,
        predicate=lambda n, p: "proj" in n,
        assignment_group_key_fn=lambda _n, _p: "packed.layer",
    ).dp_owners
    capped = compute_balanced_assignment(
        model,
        mesh,
        predicate=lambda n, p: "proj" in n,
        assignment_group_key_fn=lambda _n, _p: "packed.layer",
        max_owners_per_group=2,
    ).dp_owners

    assert len(set(default.values())) > 2
    assert len(set(capped.values())) <= 2


def test_max_owners_per_assignment_group_rejects_non_positive():
    mesh = FakeDeviceMesh(8)
    model = MiniModel(num_layers=1, hidden=1024, intermediate=8192)

    with pytest.raises(ValueError, match="max_owners_per_group"):
        compute_balanced_assignment(
            model,
            mesh,
            predicate=lambda n, p: "proj" in n,
            max_owners_per_group=0,
        )


def test_small_params_merged(model):
    """Small params (k_proj, v_proj) in the same layer should share owner."""
    mesh = FakeDeviceMesh(8)
    result = compute_balanced_assignment(
        model, mesh, predicate=lambda n, p: "proj" in n
    )
    assignment = result.dp_owners
    for layer_idx in range(4):
        k_params = [
            (n, p)
            for n, p in model.named_parameters()
            if f"layers.{layer_idx}" in n and "k_proj" in n
        ]
        v_params = [
            (n, p)
            for n, p in model.named_parameters()
            if f"layers.{layer_idx}" in n and "v_proj" in n
        ]
        if k_params and v_params:
            k_rank = assignment[k_params[0][1]]
            v_rank = assignment[v_params[0][1]]
            # k_proj and v_proj are small and should be merged to same owner
            if k_params[0][1].numel() < SMALL_PARAM_THRESHOLD:
                assert k_rank == v_rank, (
                    f"Layer {layer_idx}: k_proj→rank{k_rank}, v_proj→rank{v_rank}, "
                    "expected same owner for small params"
                )


def test_pack_small_params_can_disable_same_layer_merge():
    """True RR baselines must assign original matrices, not packed units."""
    mesh = FakeDeviceMesh(8)
    model = MiniModel(num_layers=1, hidden=512, intermediate=2048)
    proj_params = [
        p
        for name, p in model.named_parameters()
        if "layers.0" in name and "proj" in name
    ]

    packed = compute_balanced_assignment(
        model,
        mesh,
        predicate=lambda n, p: "proj" in n,
        owner_strategy="round_robin",
    )
    no_pack = compute_balanced_assignment(
        model,
        mesh,
        predicate=lambda n, p: "proj" in n,
        owner_strategy="round_robin",
        pack_small_params=False,
    )

    assert packed.allocation_unit_count == 1
    assert packed.packed_allocation_unit_count == 1
    assert packed.pack_small_params is True
    assert len(set(packed.dp_owners[p] for p in proj_params)) == 1

    assert no_pack.allocation_unit_count == len(proj_params)
    assert no_pack.packed_allocation_unit_count == 0
    assert no_pack.pack_small_params is False
    assert len(set(no_pack.dp_owners[p] for p in proj_params)) > 1
