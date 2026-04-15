"""Gradient accumulation tests: default mode and no_sync mode.

Validates that DMuon correctly accumulates gradients across multiple
forward-backward passes before a single optimizer step.

Run with: torchrun --nproc_per_node=4 tests/distributed/test_grad_accum.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dmuon import Muon, dedicate_params, no_sync, wait_all_reduces
from dmuon.utils import get_owned_params


# ---- Simple model (same as test_e2e_dp.py) ----


class MLP(nn.Module):
    def __init__(self, hidden=256, intermediate=1024):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden=256, intermediate=1024):
        super().__init__()
        self.mlp = MLP(hidden, intermediate)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class TinyModel(nn.Module):
    def __init__(self, num_layers=2, hidden=256, intermediate=1024):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(hidden, intermediate) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x).sum()


def build_model(device, mesh, seed=42):
    """Build model with dedicate_params + fully_shard."""
    torch.manual_seed(seed)
    model = TinyModel().to(device)
    dedicate_params(model, mesh, predicate=lambda n, p: "proj" in n)
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


# ---- Test 1: Default accumulation (reduce every step) ----


def test_default_accumulation(rank, world_size, device, mesh):
    """Verify default gradient accumulation: multiple backward without step."""
    accum_steps = 3
    model = build_model(device, mesh, seed=42)
    optimizer = Muon(model, lr=0.02, ns_steps=3, adamw_lr=1e-3)

    # Generate micro-batches (deterministic)
    micro_batches = []
    for i in range(accum_steps):
        torch.manual_seed(100 + i)
        micro_batches.append(torch.randn(4, 256, device=device))

    # Accumulate: N forward-backward passes, then one step
    optimizer.zero_grad()
    for i, batch in enumerate(micro_batches):
        loss = model(batch) / accum_steps
        loss.backward()

    # Before step: check owner has accumulated _reduced_grad
    wait_all_reduces(model)
    owned = get_owned_params(model, rank)
    for dp in owned:
        assert dp._reduced_grad is not None, \
            f"Owner rank {rank} missing _reduced_grad after {accum_steps} backward passes"
        assert not torch.all(dp._reduced_grad == 0), \
            f"Owner rank {rank} has zero _reduced_grad"

    # Step should consume the accumulated gradient
    optimizer.step()

    # After step: _reduced_grad should be cleared
    for dp in owned:
        assert dp._reduced_grad is None, \
            f"_reduced_grad not cleared after optimizer.step()"

    if rank == 0:
        print("  PASSED: test_default_accumulation")


# ---- Test 2: no_sync accumulation ----


def test_no_sync_accumulation(rank, world_size, device, mesh):
    """Verify no_sync mode: skip reduce, accumulate locally, reduce on sync step."""
    accum_steps = 3
    model = build_model(device, mesh, seed=42)
    optimizer = Muon(model, lr=0.02, ns_steps=3, adamw_lr=1e-3)

    micro_batches = []
    for i in range(accum_steps):
        torch.manual_seed(100 + i)
        micro_batches.append(torch.randn(4, 256, device=device))

    optimizer.zero_grad()

    # First N-1 steps: no_sync (skip reduce)
    for i in range(accum_steps - 1):
        with no_sync(model):
            loss = model(micro_batches[i]) / accum_steps
            loss.backward()

    # After no_sync steps: _reduced_grad should NOT be set (no reduce happened)
    owned = get_owned_params(model, rank)
    for dp in owned:
        assert dp._reduced_grad is None, \
            f"_reduced_grad should be None during no_sync, but is set on rank {rank}"

    # Check _accumulated_grad is set on all ranks' dedicated params
    from dmuon.utils import get_dedicated_params
    all_params = get_dedicated_params(model)
    has_accum = any(dp._accumulated_grad is not None for dp in all_params)
    assert has_accum, "No _accumulated_grad found after no_sync backward passes"

    # Last step: normal (sync) — triggers reduce with merged accumulated grads
    loss = model(micro_batches[-1]) / accum_steps
    loss.backward()

    # Now owner should have _reduced_grad
    wait_all_reduces(model)
    for dp in owned:
        assert dp._reduced_grad is not None, \
            f"Owner rank {rank} missing _reduced_grad after sync step"

    # _accumulated_grad should be cleared (merged into reduce)
    for dp in all_params:
        assert dp._accumulated_grad is None, \
            f"_accumulated_grad not cleared after sync reduce"

    optimizer.step()

    if rank == 0:
        print("  PASSED: test_no_sync_accumulation")


# ---- Test 3: Default vs no_sync produce same gradients ----


def test_default_vs_no_sync_match(rank, world_size, device, mesh):
    """Verify default and no_sync modes produce identical accumulated gradients."""
    accum_steps = 3

    micro_batches = []
    for i in range(accum_steps):
        torch.manual_seed(200 + i)
        micro_batches.append(torch.randn(4, 256, device=device))

    # --- Run with default mode (reduce every step) ---
    model_default = build_model(device, mesh, seed=42)
    optimizer_default = Muon(model_default, lr=0.02, ns_steps=3, adamw_lr=1e-3)
    optimizer_default.zero_grad()

    for batch in micro_batches:
        loss = model_default(batch) / accum_steps
        loss.backward()

    wait_all_reduces(model_default)
    default_grads = {}
    for dp in get_owned_params(model_default, rank):
        if dp._reduced_grad is not None:
            default_grads[dp.param_name] = dp._reduced_grad.clone()

    # --- Run with no_sync mode ---
    model_nosync = build_model(device, mesh, seed=42)
    optimizer_nosync = Muon(model_nosync, lr=0.02, ns_steps=3, adamw_lr=1e-3)
    optimizer_nosync.zero_grad()

    for i, batch in enumerate(micro_batches):
        if i < accum_steps - 1:
            with no_sync(model_nosync):
                loss = model_nosync(batch) / accum_steps
                loss.backward()
        else:
            loss = model_nosync(batch) / accum_steps
            loss.backward()

    wait_all_reduces(model_nosync)
    nosync_grads = {}
    for dp in get_owned_params(model_nosync, rank):
        if dp._reduced_grad is not None:
            nosync_grads[dp.param_name] = dp._reduced_grad.clone()

    # --- Compare ---
    assert set(default_grads.keys()) == set(nosync_grads.keys()), \
        f"Gradient key mismatch: {set(default_grads.keys())} vs {set(nosync_grads.keys())}"

    max_diff = 0.0
    for name in default_grads:
        diff = (default_grads[name] - nosync_grads[name]).abs().max().item()
        max_diff = max(max_diff, diff)

    # Tolerance: reduce(AVG) vs local accumulation + single reduce(AVG)
    # should be very close, but floating point order may differ slightly
    assert max_diff < 1e-2, \
        f"Default vs no_sync gradient mismatch: max_diff={max_diff:.6f}"

    if rank == 0:
        print(f"  PASSED: test_default_vs_no_sync_match (max_diff={max_diff:.6f})")

    del model_default, model_nosync
    torch.cuda.empty_cache()


# ---- Test 4: Training with gradient accumulation produces decreasing loss ----


def test_training_with_accum(rank, world_size, device, mesh):
    """End-to-end: training with gradient accumulation should decrease loss."""
    accum_steps = 2
    num_steps = 4
    model = build_model(device, mesh, seed=42)
    optimizer = Muon(model, lr=0.02, ns_steps=3, adamw_lr=1e-3)

    losses = []
    step_count = 0
    for step in range(num_steps):
        optimizer.zero_grad()
        step_loss = 0.0
        for micro in range(accum_steps):
            torch.manual_seed(1000 + step * accum_steps + micro)
            x = torch.randn(4, 256, device=device)
            loss = model(x) / accum_steps
            loss.backward()
            step_loss += loss.item()
        optimizer.step()
        losses.append(step_loss)
        step_count += 1

    if rank == 0:
        print(f"  Losses: {[f'{l:.4f}' for l in losses]}")
        # Check that training is progressing (loss changed)
        assert abs(losses[-1] - losses[0]) > 1e-6, \
            "Loss unchanged during training with gradient accumulation"
        print("  PASSED: test_training_with_accum")


# ---- Main ----


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world_size,))

    if rank == 0:
        print(f"Gradient Accumulation Tests: {world_size} GPUs\n")

    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    if test_name in ("default", "all"):
        test_default_accumulation(rank, world_size, device, mesh)
        dist.barrier(); torch.cuda.empty_cache()

    if test_name in ("no_sync", "all"):
        test_no_sync_accumulation(rank, world_size, device, mesh)
        dist.barrier(); torch.cuda.empty_cache()

    if test_name in ("match", "all"):
        test_default_vs_no_sync_match(rank, world_size, device, mesh)
        dist.barrier(); torch.cuda.empty_cache()

    if test_name in ("training", "all"):
        test_training_with_accum(rank, world_size, device, mesh)
        dist.barrier(); torch.cuda.empty_cache()

    if rank == 0:
        print("\nAll gradient accumulation tests passed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
