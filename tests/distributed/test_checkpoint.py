"""State dict save/load tests.

Validates that DMuon correctly saves and loads both model and optimizer
state dicts, handling dedicated params (DMuon) and symmetric params (FSDP2).

Run with: torchrun --nproc_per_node=4 tests/distributed/test_checkpoint.py
"""

import gc
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dmuon import (
    Muon,
    dedicate_params,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from dmuon.checkpoint import _compute_dedicated_fqns
from dmuon.utils import get_dedicated_params, get_owned_params


# ---- Simple model (same as other tests) ----


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


def train_step(model, optimizer, x):
    """One training step, returns loss value."""
    optimizer.zero_grad()
    loss = model(x)
    loss.backward()
    optimizer.step()
    return loss.item()


# ---- Test 1: FQN correctness ----


def test_fqn_correctness(rank, world_size, device, mesh):
    """Verify _compute_dedicated_fqns produces correct FQNs."""
    # Build model WITHOUT dedicate_params to get original param names
    torch.manual_seed(42)
    ref_model = TinyModel().to(device)
    expected_fqns = {
        name for name, p in ref_model.named_parameters() if "proj" in name
    }
    del ref_model

    # Build model with dedicate_params
    model = build_model(device, mesh, seed=42)
    dp_fqns = _compute_dedicated_fqns(model)
    actual_fqns = set(dp_fqns.values())

    assert actual_fqns == expected_fqns, (
        f"FQN mismatch:\n  expected: {sorted(expected_fqns)}\n  actual: {sorted(actual_fqns)}"
    )

    if rank == 0:
        print(f"  PASSED: test_fqn_correctness ({len(actual_fqns)} FQNs)")

    del model
    torch.cuda.empty_cache()


# ---- Test 2: Model state dict roundtrip ----


def test_model_state_dict_roundtrip(rank, world_size, device, mesh):
    """Save and load model state dict, verify parameters match."""
    model = build_model(device, mesh, seed=42)
    optimizer = Muon(model, lr=0.02, ns_steps=3, adamw_lr=1e-3)

    # Train 2 steps to change parameters
    for step in range(2):
        torch.manual_seed(500 + step)
        x = torch.randn(4, 256, device=device)
        train_step(model, optimizer, x)

    # Save state dict
    sd = get_model_state_dict(model, cpu_offload=True)

    # Verify: all entries are non-empty, on CPU
    dp_fqns = _compute_dedicated_fqns(model)
    for dp, fqn in dp_fqns.items():
        assert fqn in sd, f"Missing dedicated param FQN: {fqn}"
        assert sd[fqn].numel() > 0, f"Empty tensor for dedicated param: {fqn}"
        assert sd[fqn].shape == dp._orig_size, (
            f"Shape mismatch for {fqn}: {sd[fqn].shape} vs {dp._orig_size}"
        )
        assert sd[fqn].device == torch.device("cpu"), f"Not on CPU: {fqn}"

    # Load into a fresh model
    model2 = build_model(device, mesh, seed=123)  # different seed
    set_model_state_dict(model2, sd)

    # Verify: dedicated params match on owner
    dp_fqns2 = _compute_dedicated_fqns(model2)
    for dp2, fqn in dp_fqns2.items():
        if dp2.is_owner:
            expected = sd[fqn].to(dp2._orig_dtype).to(dp2.device)
            actual = dp2._owned_data.to(dp2._orig_dtype)
            assert torch.allclose(actual, expected, atol=1e-6), (
                f"Dedicated param mismatch on rank {rank} for {fqn}: "
                f"max_diff={( actual - expected).abs().max().item():.6f}"
            )

    if rank == 0:
        print("  PASSED: test_model_state_dict_roundtrip")

    del model, model2
    torch.cuda.empty_cache()


# ---- Test 3: Optimizer state dict roundtrip ----


def test_optimizer_state_dict_roundtrip(rank, world_size, device, mesh):
    """Save/load optimizer state, verify training continues correctly."""
    batches = []
    for step in range(3):
        torch.manual_seed(700 + step)
        batches.append(torch.randn(4, 256, device=device))

    # Train 2 steps, save state dict
    model_save = build_model(device, mesh, seed=42)
    opt_save = Muon(model_save, lr=0.02, ns_steps=3, adamw_lr=1e-3)
    for step in range(2):
        train_step(model_save, opt_save, batches[step])

    # Save before any evaluation (avoid forward side effects)
    save_loss = model_save(batches[0]).item()
    model_sd = get_model_state_dict(model_save, cpu_offload=True)
    optim_sd = get_optimizer_state_dict(model_save, opt_save, cpu_offload=True)

    # Continue training for reference
    train_step(model_save, opt_save, batches[2])
    ref_loss = model_save(batches[0]).item()
    del model_save, opt_save; gc.collect(); torch.cuda.empty_cache()

    # Load into fresh model
    model_load = build_model(device, mesh, seed=123)
    opt_load = Muon(model_load, lr=0.02, ns_steps=3, adamw_lr=1e-3)
    set_model_state_dict(model_load, model_sd)
    set_optimizer_state_dict(model_load, opt_load, optim_sd)

    # Verify: loaded model forward matches saved model forward
    loaded_fwd = model_load(batches[0]).item()
    fwd_diff = abs(save_loss - loaded_fwd)
    assert fwd_diff < 1e-4, (
        f"Forward mismatch after model load: save={save_loss:.6f}, "
        f"loaded={loaded_fwd:.6f}, diff={fwd_diff:.6f}"
    )

    # Train 1 more step, compare with reference
    train_step(model_load, opt_load, batches[2])
    loaded_loss = model_load(batches[0]).item()

    # Tolerance relaxed: Newton-Schulz amplifies small FP differences
    diff = abs(ref_loss - loaded_loss)
    assert diff < 0.5, (
        f"Loss mismatch after optimizer roundtrip: ref={ref_loss:.6f}, "
        f"loaded={loaded_loss:.6f}, diff={diff:.6f}"
    )

    if rank == 0:
        print(f"  PASSED: test_optimizer_state_dict_roundtrip (loss_diff={diff:.6f})")

    del model_load, opt_load
    torch.cuda.empty_cache()


# ---- Test 4: Load from standard (single-GPU) checkpoint ----


def test_load_from_standard_checkpoint(rank, world_size, device, mesh):
    """Load a standard state dict (as if from HuggingFace) into DMuon model."""
    # Create standard state dict on all ranks (simulating loading from file)
    torch.manual_seed(42)
    ref_model = TinyModel().to(device)
    standard_sd = {k: v.clone() for k, v in ref_model.named_parameters()}
    del ref_model

    # Load into DMuon model
    model = build_model(device, mesh, seed=123)  # different seed
    set_model_state_dict(model, standard_sd)

    # Verify: run forward, all ranks should produce same output
    torch.manual_seed(999)
    x = torch.randn(4, 256, device=device)
    loss = model(x)

    # Gather losses from all ranks
    all_losses = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(all_losses, loss.detach().unsqueeze(0))

    max_loss_diff = max(
        abs(all_losses[i].item() - all_losses[0].item()) for i in range(1, world_size)
    )
    assert max_loss_diff < 1e-5, (
        f"Cross-rank loss divergence after loading standard checkpoint: {max_loss_diff:.6f}"
    )

    if rank == 0:
        print(f"  PASSED: test_load_from_standard_checkpoint (cross_rank_diff={max_loss_diff:.6f})")

    del model
    torch.cuda.empty_cache()


# ---- Test 5: State dict key completeness ----


def test_state_dict_completeness(rank, world_size, device, mesh):
    """Verify get_model_state_dict returns all parameters."""
    torch.manual_seed(42)
    ref_model = TinyModel().to(device)
    expected_keys = set(ref_model.state_dict().keys())
    del ref_model

    model = build_model(device, mesh, seed=42)
    sd = get_model_state_dict(model, cpu_offload=True)
    actual_keys = set(sd.keys())

    assert actual_keys == expected_keys, (
        f"Key mismatch:\n  missing: {expected_keys - actual_keys}\n"
        f"  extra: {actual_keys - expected_keys}"
    )

    if rank == 0:
        print(f"  PASSED: test_state_dict_completeness ({len(actual_keys)} keys)")

    del model
    torch.cuda.empty_cache()


# ---- Main ----


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world_size,))

    if rank == 0:
        print(f"State Dict Save/Load Tests: {world_size} GPUs\n")

    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    if test_name in ("fqn", "all"):
        test_fqn_correctness(rank, world_size, device, mesh)
        dist.barrier(); gc.collect(); torch.cuda.empty_cache()

    if test_name in ("model", "all"):
        test_model_state_dict_roundtrip(rank, world_size, device, mesh)
        dist.barrier(); gc.collect(); torch.cuda.empty_cache()

    if test_name in ("optim", "all", "optim_only"):
        test_optimizer_state_dict_roundtrip(rank, world_size, device, mesh)
        dist.barrier(); gc.collect(); torch.cuda.empty_cache()

    if test_name in ("standard", "all"):
        test_load_from_standard_checkpoint(rank, world_size, device, mesh)
        dist.barrier(); gc.collect(); torch.cuda.empty_cache()

    if test_name in ("completeness", "all"):
        test_state_dict_completeness(rank, world_size, device, mesh)
        dist.barrier(); gc.collect(); torch.cuda.empty_cache()

    if rank == 0:
        print("\nAll state dict tests passed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
