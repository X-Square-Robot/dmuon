"""Unit tests for hook-boundary module selection (single GPU, no dist required)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch.nn as nn

from dmuon.api import _find_hook_module, _resolve_hook_module


class TinyLinear(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, d, bias=False)


class VitBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = TinyLinear(d)
        self.mlp = TinyLinear(d)


class VisionTower(nn.Module):
    def __init__(self, d=64, n=4):
        super().__init__()
        self.blocks = nn.ModuleList([VitBlock(d) for _ in range(n)])


class DecoderLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = TinyLinear(d)
        self.mlp = TinyLinear(d)


class ToyVLA(nn.Module):
    def __init__(self, d=64, n_vit=4, n_dec=3):
        super().__init__()
        self.visual = VisionTower(d, n_vit)
        self.layers = nn.ModuleList([DecoderLayer(d) for _ in range(n_dec)])


def _boundary(mod):
    return isinstance(mod, (VisionTower, DecoderLayer))


def test_vit_params_collapse_to_vision_tower():
    """All ViT params should map to the same VisionTower hook module."""
    m = ToyVLA()
    vit_params = [p for n, p in m.named_parameters() if "visual.blocks" in n]
    assert len(vit_params) > 1
    hook_modules = [_find_hook_module(m, p, _boundary) for p in vit_params]
    assert all(h is m.visual for h in hook_modules), (
        f"expected all ViT params → m.visual, got {set(id(h) for h in hook_modules)} modules"
    )


def test_decoder_params_collapse_to_own_layer():
    """Each decoder layer's params should map to its own DecoderLayer module."""
    m = ToyVLA()
    for i, layer in enumerate(m.layers):
        layer_params = [p for n, p in m.named_parameters() if f"layers.{i}." in n]
        assert len(layer_params) > 1
        hook_modules = [_find_hook_module(m, p, _boundary) for p in layer_params]
        assert all(h is layer for h in hook_modules), (
            f"layer {i}: expected all params → m.layers[{i}], "
            f"got {[type(h).__name__ for h in hook_modules]}"
        )


def test_strict_raises_on_unmatched_param():
    """Strict mode should raise when a param has no ancestor matching the predicate."""
    m = ToyVLA()
    p = m.visual.blocks[0].attn.fc1.weight

    def never(mod):
        return False

    with pytest.raises(ValueError, match="no ancestor matched"):
        _find_hook_module(m, p, never, strict=True)


def test_lenient_falls_back_to_parent():
    """Lenient mode falls back to parent module when predicate matches nothing."""
    m = ToyVLA()
    p = m.visual.blocks[0].attn.fc1.weight

    def never(mod):
        return False

    mod = _find_hook_module(m, p, never, strict=False)
    assert mod is m.visual.blocks[0].attn.fc1, (
        f"expected fallback to direct parent (nn.Linear), got {type(mod).__name__}"
    )


def test_predicate_none_returns_none():
    """If predicate is None, caller should use the old path; helper returns None."""
    m = ToyVLA()
    p = m.visual.blocks[0].attn.fc1.weight
    mod = _find_hook_module(m, p, None)
    assert mod is None


def test_resolver_can_select_non_ancestor_boundary():
    """Resolver supports execution boundaries that are not parameter ancestors."""
    m = ToyVLA()
    p = m.visual.blocks[0].attn.fc1.weight

    mod = _resolve_hook_module(
        m,
        p,
        param_fqn="visual.blocks.0.attn.fc1.weight",
        hook_boundary_predicate=None,
        hook_boundary_resolver=lambda _name, _param: m.layers[1],
        strict=True,
    )

    assert mod is m.layers[1]


def test_resolver_strict_raises_on_unmatched_param():
    m = ToyVLA()
    p = m.visual.blocks[0].attn.fc1.weight

    with pytest.raises(ValueError, match="hook_boundary_resolver returned None"):
        _resolve_hook_module(
            m,
            p,
            param_fqn="visual.blocks.0.attn.fc1.weight",
            hook_boundary_predicate=None,
            hook_boundary_resolver=lambda _name, _param: None,
            strict=True,
        )
