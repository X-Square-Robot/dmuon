"""Unit tests for the partition algorithm (single GPU, no dist required)."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch.nn as nn

from dmuon.partition import (
    SMALL_PARAM_THRESHOLD,
    _extract_layer_id,
    compute_balanced_assignment,
)


class FakeDeviceMesh:
    """Minimal mock for DeviceMesh to test partition logic without distributed."""

    def __init__(self, world_size: int):
        self._world_size = world_size

    def size(self):
        return self._world_size


# ---- Test _extract_layer_id ----


def test_extract_layer_id_standard():
    assert _extract_layer_id("model.layers.3.mlp.gate_proj.weight") == "layers.3"
    assert _extract_layer_id("model.layers.12.self_attn.q_proj.weight") == "layers.12"


def test_extract_layer_id_no_layer():
    assert _extract_layer_id("model.embed_tokens.weight") is None
    assert _extract_layer_id("lm_head.weight") is None


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
            {str(i): MiniTransformerBlock(hidden, intermediate) for i in range(num_layers)}
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
    assignment = compute_balanced_assignment(model, mesh, predicate=lambda n, p: "proj" in n)
    proj_params = [p for n, p in model.named_parameters() if "proj" in n]
    assert len(assignment) == len(proj_params)
    for p in proj_params:
        assert p in assignment


def test_assignment_excludes_non_proj(model):
    """Non-proj params (embed, layernorm) should NOT be assigned."""
    mesh = FakeDeviceMesh(8)
    assignment = compute_balanced_assignment(model, mesh, predicate=lambda n, p: "proj" in n)
    for n, p in model.named_parameters():
        if "proj" not in n:
            assert p not in assignment


def test_balance(model):
    """Rank loads should be balanced within 5%."""
    mesh = FakeDeviceMesh(8)
    assignment = compute_balanced_assignment(model, mesh, predicate=lambda n, p: "proj" in n)
    rank_loads = [0] * 8
    for param, rank in assignment.items():
        rank_loads[rank] += param.numel()

    max_load = max(rank_loads)
    min_load = min(r for r in rank_loads if r > 0)
    imbalance = (max_load - min_load) / max_load
    assert imbalance < 0.05, f"Imbalance too high: {imbalance:.1%}, loads={rank_loads}"


def test_same_layer_different_ranks(model):
    """Large params in the same layer should go to different ranks."""
    mesh = FakeDeviceMesh(8)
    assignment = compute_balanced_assignment(model, mesh, predicate=lambda n, p: "proj" in n)
    # Check layer 0
    layer0_large = {}
    for n, p in model.named_parameters():
        if "layers.0" in n and "proj" in n and p.numel() >= SMALL_PARAM_THRESHOLD:
            layer0_large[n] = assignment[p]

    # All large params in the same layer should have different owner ranks
    ranks = list(layer0_large.values())
    assert len(ranks) == len(set(ranks)), (
        f"Same-layer large params assigned to same rank: {layer0_large}"
    )


def test_small_params_merged(model):
    """Small params (k_proj, v_proj) in the same layer should share owner."""
    mesh = FakeDeviceMesh(8)
    assignment = compute_balanced_assignment(model, mesh, predicate=lambda n, p: "proj" in n)
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
