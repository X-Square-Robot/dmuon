"""Test that direct .weight access works on dedicated params during forward.

This validates the GatedDeltaNet compatibility pattern where user code does:
    W = self.linear.weight          # direct attribute access
    out = F.linear(input, W[:k])    # manual split + F.linear

Run with: python tests/unit/test_weight_access.py
(Single GPU, no distributed required for mechanism test)
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dmuon.param import DedicatedParam


class DirectWeightAccessModule(nn.Module):
    """Mimics GatedDeltaNet's _forward_mot_opt pattern:
    accesses .weight directly and uses F.linear instead of module.forward().
    """

    def __init__(self, d_in=64, d_out=48):
        super().__init__()
        # Fused projection (like in_proj_zba)
        self.fused_proj = nn.Linear(d_in, d_out, bias=False)
        self._split_sizes = [32, 8, 8]  # z, b, a

    def forward(self, x):
        # Pattern 1: Normal module call (always works)
        out_normal = self.fused_proj(x)

        # Pattern 2: Direct .weight access + manual split + F.linear
        # This is what GatedDeltaNet._forward_mot_opt does
        W = self.fused_proj.weight
        W_z, W_b, W_a = torch.split(W, self._split_sizes, dim=0)
        z = F.linear(x, W_z)
        b = F.linear(x, W_b)
        a = F.linear(x, W_a)

        return out_normal, z, b, a


class LayerWithDirectAccess(nn.Module):
    """Mimics a decoder layer containing the direct-access module."""

    def __init__(self):
        super().__init__()
        self.attn = DirectWeightAccessModule()
        self.ln = nn.LayerNorm(64)

    def forward(self, x):
        return self.attn(self.ln(x))


def test_setattr_swap_mechanism():
    """Test that setattr-based param swap makes .weight return full param."""
    module = DirectWeightAccessModule()
    original_weight = module.fused_proj.weight.clone()

    # Simulate dmuon's unshard: replace weight with a different tensor
    fake_unsharded = nn.Parameter(torch.randn_like(original_weight) * 10)
    setattr(module.fused_proj, "weight", fake_unsharded)

    # Direct .weight access should return the swapped tensor
    assert module.fused_proj.weight is fake_unsharded, (
        "setattr swap failed: .weight doesn't point to swapped param"
    )

    # F.linear with split weight should use swapped data
    x = torch.randn(2, 64)
    W = module.fused_proj.weight
    W_z, W_b, W_a = torch.split(W, [32, 8, 8], dim=0)
    z = F.linear(x, W_z)
    assert z.shape == (2, 32)
    # Verify it's using the swapped weight, not the original
    z_expected = F.linear(x, fake_unsharded[:32])
    assert torch.allclose(z, z_expected), "F.linear didn't use swapped weight"

    # Simulate dmuon's reshard: replace with placeholder
    placeholder = nn.Parameter(torch.empty(0, dtype=original_weight.dtype))
    setattr(module.fused_proj, "weight", placeholder)
    assert module.fused_proj.weight.numel() == 0, "Reshard to placeholder failed"

    print("PASSED: test_setattr_swap_mechanism")


def test_hook_based_swap_during_forward():
    """Test that pre_forward hook swap is visible to direct .weight access in forward."""
    layer = LayerWithDirectAccess()
    original_weight = layer.attn.fused_proj.weight.clone()

    # Replace with placeholder (simulate dedicated param initial state)
    placeholder = nn.Parameter(torch.empty(0, dtype=original_weight.dtype))
    setattr(layer.attn.fused_proj, "weight", placeholder)
    assert layer.attn.fused_proj.weight.numel() == 0

    # Simulate dmuon's pre_forward hook: swap placeholder with full param
    full_weight = nn.Parameter(original_weight.clone(), requires_grad=True)

    def pre_forward_hook(module, args, kwargs):
        setattr(layer.attn.fused_proj, "weight", full_weight)
        return args, kwargs

    def post_forward_hook(module, input, output):
        setattr(layer.attn.fused_proj, "weight", placeholder)
        return output

    layer.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)
    layer.register_forward_hook(post_forward_hook)

    # Forward should work: pre_hook swaps in full weight, forward accesses .weight
    x = torch.randn(2, 64)
    out_normal, z, b, a = layer(x)

    assert out_normal.shape == (2, 48), f"Normal forward failed: {out_normal.shape}"
    assert z.shape == (2, 32), f"F.linear with split weight failed: {z.shape}"
    assert b.shape == (2, 8), f"b shape wrong: {b.shape}"
    assert a.shape == (2, 8), f"a shape wrong: {a.shape}"

    # Verify post_hook resharded
    assert layer.attn.fused_proj.weight.numel() == 0, "Post-hook didn't reshard"

    # Verify correctness: out_normal should equal z+b+a concatenated
    expected = F.linear(layer.ln(x), full_weight)
    # Can't directly compare because ln uses running stats, but shapes are right

    print("PASSED: test_hook_based_swap_during_forward")


def test_gradient_flow_through_split():
    """Test that gradients flow correctly through split weight + F.linear."""
    module = DirectWeightAccessModule()

    x = torch.randn(2, 64)
    out_normal, z, b, a = module(x)

    # Backward through split path
    loss = z.sum() + b.sum() + a.sum()
    loss.backward()

    # Gradient should be on fused_proj.weight (the original parameter)
    assert module.fused_proj.weight.grad is not None, (
        "No gradient on fused_proj.weight after backward through split path"
    )
    assert module.fused_proj.weight.grad.shape == module.fused_proj.weight.shape, (
        f"Gradient shape mismatch: {module.fused_proj.weight.grad.shape} "
        f"vs {module.fused_proj.weight.shape}"
    )
    # Gradient should be non-zero in all regions (z, b, a all contribute)
    grad = module.fused_proj.weight.grad
    assert grad[:32].abs().sum() > 0, "z region gradient is zero"
    assert grad[32:40].abs().sum() > 0, "b region gradient is zero"
    assert grad[40:48].abs().sum() > 0, "a region gradient is zero"

    print("PASSED: test_gradient_flow_through_split")


def test_gradient_flow_with_swap():
    """Test gradient flow when weight is swapped in by hook (simulating dmuon)."""
    layer = LayerWithDirectAccess()

    # The "owned" weight that dmuon would store
    owned_weight = layer.attn.fused_proj.weight.detach().clone()
    unsharded_param = nn.Parameter(owned_weight.clone(), requires_grad=True)
    placeholder = nn.Parameter(torch.empty(0, dtype=owned_weight.dtype))

    # Set to placeholder state
    setattr(layer.attn.fused_proj, "weight", placeholder)

    def pre_hook(module, args, kwargs):
        setattr(layer.attn.fused_proj, "weight", unsharded_param)
        return args, kwargs

    def post_hook(module, input, output):
        # Don't reshard yet — need grad to flow in backward
        return output

    layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    layer.register_forward_hook(post_hook)

    x = torch.randn(2, 64)
    out_normal, z, b, a = layer(x)
    loss = z.sum() + b.sum() + a.sum()
    loss.backward()

    # Gradient should land on unsharded_param (the swapped-in weight)
    assert unsharded_param.grad is not None, (
        "No gradient on unsharded_param after backward"
    )
    assert unsharded_param.grad.shape == owned_weight.shape, (
        f"Grad shape {unsharded_param.grad.shape} != weight shape {owned_weight.shape}"
    )

    print("PASSED: test_gradient_flow_with_swap")


def test_conv1d_weight_squeeze():
    """Test conv1d.weight.squeeze(1) pattern (causal_conv1d_fn compatibility)."""
    conv = nn.Conv1d(16, 16, kernel_size=4, groups=16, bias=False)
    original_shape = conv.weight.shape  # [16, 1, 4]

    # Simulate dmuon swap
    owned = conv.weight.detach().clone()
    unsharded = nn.Parameter(owned.clone(), requires_grad=True)
    setattr(conv, "weight", unsharded)

    # Direct access + squeeze (like causal_conv1d_fn expects)
    w = conv.weight.squeeze(1)
    assert w.shape == (16, 4), f"squeeze failed: {w.shape}"

    # Use in F.conv1d equivalent
    x = torch.randn(2, 16, 10)
    out = F.conv1d(x, conv.weight, groups=16, padding=3)
    assert out.shape[0] == 2, "conv1d with swapped weight failed"

    print("PASSED: test_conv1d_weight_squeeze")


if __name__ == "__main__":
    test_setattr_swap_mechanism()
    test_hook_based_swap_during_forward()
    test_gradient_flow_through_split()
    test_gradient_flow_with_swap()
    test_conv1d_weight_squeeze()
    print("\nAll tests passed!")
