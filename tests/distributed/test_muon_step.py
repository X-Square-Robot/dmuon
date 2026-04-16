"""Multi-GPU tests for Muon optimizer step correctness.

Run with: torchrun --nproc_per_node=4 tests/distributed/test_muon_step.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dmuon
from torch.distributed.fsdp import fully_shard


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# TinyModel
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, hidden=256, intermediate=1024):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, hidden=256, intermediate=1024):
        super().__init__()
        self.mlp = MLP(hidden, intermediate)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class TinyModel(nn.Module):
    def __init__(self, num_layers=4, hidden=256, intermediate=1024):
        super().__init__()
        self.layers = nn.ModuleList(
            [Block(hidden, intermediate) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x).sum()


# ---------------------------------------------------------------------------
# Test 1: reduced grad is set on all owned dedicated params after backward
# ---------------------------------------------------------------------------

def test_muon_reduced_grad_all_set(rank, world_size, device, mesh):

    torch.manual_seed(0)
    model = TinyModel().to(device)

    dmuon.dedicate_params(
        model, mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )

    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    optimizer = dmuon.Muon(model, lr=0.01, ns_steps=5, adamw_lr=0.01)

    optimizer.zero_grad()
    x = torch.randn(4, 256, device=device)
    loss = model(x)
    loss.backward()

    dmuon.wait_all_reduces(model)

    for dp in optimizer._dedicated_params:
        assert dp._reduced_grad is not None, (
            f"Rank {rank}: _reduced_grad is None for param with shape {dp._orig_size}"
        )
        assert dp._reduced_grad.shape == dp._orig_size, (
            f"Rank {rank}: shape mismatch {dp._reduced_grad.shape} vs {dp._orig_size}"
        )
        assert dp._reduced_grad.abs().max().item() > 0, (
            f"Rank {rank}: _reduced_grad is all zeros for param with shape {dp._orig_size}"
        )

    torch.cuda.synchronize()
    log(rank, "PASSED: test_muon_reduced_grad_all_set")


# ---------------------------------------------------------------------------
# Test 2: momentum buffer accumulates correctly
# ---------------------------------------------------------------------------

def test_muon_momentum_accumulation(rank, world_size, device, mesh):

    torch.manual_seed(0)
    model = TinyModel().to(device)

    dmuon.dedicate_params(
        model, mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )

    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    optimizer = dmuon.Muon(model, lr=0.01, momentum=0.95, ns_steps=5, adamw_lr=0.01)

    torch.manual_seed(42)
    x = torch.randn(4, 256, device=device)

    # Step 1: forward -> backward -> step, record momentum buffer
    optimizer.zero_grad()
    loss = model(x)
    loss.backward()
    optimizer.step()

    if len(optimizer._dedicated_params) == 0:
        log(rank, "PASSED: test_muon_momentum_accumulation (no owned params on this rank)")
        return

    dp = optimizer._dedicated_params[0]
    dp_id = id(dp)
    assert dp_id in optimizer.state, f"Rank {rank}: no state for first dedicated param"
    buf1 = optimizer.state[dp_id]["momentum_buffer"].clone()

    # Step 2: forward -> backward, capture grad before step, then step
    optimizer.zero_grad()
    loss = model(x)
    loss.backward()

    dmuon.wait_all_reduces(model)

    assert dp._reduced_grad is not None, (
        f"Rank {rank}: _reduced_grad is None before step 2"
    )
    grad2 = dp._reduced_grad.view(dp._reduced_grad.shape[0], -1).clone()

    optimizer.step()

    buf2 = optimizer.state[dp_id]["momentum_buffer"].clone()

    # Verify: buf2 ~= 0.95 * buf1 + grad2
    expected = 0.95 * buf1 + grad2
    max_abs_diff = (buf2 - expected).abs().max().item()
    buf2_scale = buf2.abs().max().item()
    tolerance = 0.01 * buf2_scale if buf2_scale > 0 else 0.01

    assert max_abs_diff < tolerance, (
        f"Rank {rank}: momentum mismatch. max_abs_diff={max_abs_diff:.6f}, "
        f"tolerance={tolerance:.6f} (1% of buf2 max {buf2_scale:.6f})"
    )

    # Verify shape: buf should be 2D matching param reshaped
    expected_rows = dp._orig_size[0]
    expected_cols = dp._orig_size.numel() // expected_rows
    assert buf2.shape == (expected_rows, expected_cols), (
        f"Rank {rank}: buf shape {buf2.shape} != expected ({expected_rows}, {expected_cols})"
    )

    torch.cuda.synchronize()
    log(rank, "PASSED: test_muon_momentum_accumulation")


# ---------------------------------------------------------------------------
# Test 3: Nesterov momentum — NS receives grad + μ*buf, not just buf
# ---------------------------------------------------------------------------

def test_muon_nesterov(rank, world_size, device, mesh):
    """Verify Nesterov vs non-Nesterov produce different weight updates.

    With nesterov=True (default), NS input is grad + μ*buf (lookahead).
    With nesterov=False, NS input is buf.
    The resulting weight updates should differ, proving Nesterov is active.
    """

    lr = 0.01
    mu = 0.95

    # --- Run with Nesterov=True ---
    torch.manual_seed(0)
    model1 = TinyModel().to(device)
    dmuon.dedicate_params(model1, mesh, predicate=lambda n, p: "proj" in n and p.ndim == 2)
    for layer in model1.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model1, mesh=mesh)
    opt1 = dmuon.Muon(model1, lr=lr, momentum=mu, nesterov=True, adamw_lr=lr)

    torch.manual_seed(42)
    x = torch.randn(4, 256, device=device)
    opt1.zero_grad()
    model1(x).backward()
    opt1.step()

    # --- Run with Nesterov=False ---
    torch.manual_seed(0)
    model2 = TinyModel().to(device)
    dmuon.dedicate_params(model2, mesh, predicate=lambda n, p: "proj" in n and p.ndim == 2)
    for layer in model2.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model2, mesh=mesh)
    opt2 = dmuon.Muon(model2, lr=lr, momentum=mu, nesterov=False, adamw_lr=lr)

    torch.manual_seed(42)
    x = torch.randn(4, 256, device=device)
    opt2.zero_grad()
    model2(x).backward()
    opt2.step()

    # --- Compare: weights should differ ---
    if len(opt1._dedicated_params) == 0:
        log(rank, "PASSED: test_muon_nesterov (no owned params)")
        return

    dp1 = opt1._dedicated_params[0]
    dp2 = opt2._dedicated_params[0]
    w1 = dp1._owned_data
    w2 = dp2._owned_data

    diff = (w1 - w2).abs().max().item()
    assert diff > 1e-6, (
        f"Rank {rank}: Nesterov=True and False produced identical weights "
        f"(diff={diff}). Nesterov is not active."
    )

    # Also verify momentum buffers are the SAME (Nesterov only changes NS input, not buf)
    buf1 = opt1.state[id(dp1)]["momentum_buffer"]
    buf2 = opt2.state[id(dp2)]["momentum_buffer"]
    buf_diff = (buf1 - buf2).abs().max().item()
    assert buf_diff < 1e-6, (
        f"Rank {rank}: momentum buffers differ (diff={buf_diff}). "
        f"Nesterov should not affect buf accumulation."
    )

    torch.cuda.synchronize()
    log(rank, f"PASSED: test_muon_nesterov (weight diff={diff:.6f}, buf diff={buf_diff:.2e})")


# ---------------------------------------------------------------------------
# Test 4: TP params get updated through gram_newton_schulz path
# ---------------------------------------------------------------------------

def test_muon_tp_path(rank, world_size, device, mesh):
    assert world_size >= 4, "test_muon_tp_path requires at least 4 GPUs"

    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    # 2D mesh: dp_size=2, tp_size=2 (separate from the 1D mesh passed in)
    mesh_2d = dist.init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
    dp_mesh = mesh_2d["dp"]
    tp_mesh = mesh_2d["tp"]

    torch.manual_seed(0)
    model = TinyModel().to(device)

    # Apply TP parallelism to MLP layers
    for layer in model.layers:
        parallelize_module(
            layer.mlp,
            tp_mesh,
            {
                "gate_proj": ColwiseParallel(),
                "up_proj": ColwiseParallel(),
                "down_proj": RowwiseParallel(),
            },
        )

    dmuon.dedicate_params(
        model, dp_mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )

    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)

    optimizer = dmuon.Muon(model, lr=0.01, ns_steps=5, adamw_lr=0.01)

    # Verify TP params are flagged correctly
    tp_params = [dp for dp in optimizer._dedicated_params if dp.is_dtensor and dp.tp_group is not None]
    tp_count = torch.tensor([len(tp_params)], device=device)
    dist.all_reduce(tp_count)
    assert tp_count.item() > 0, (
        f"No TP-aware dedicated params found (is_dtensor=True, tp_group≠None)"
    )

    if len(optimizer._dedicated_params) == 0:
        log(rank, "PASSED: test_muon_tp_path (no owned params on this rank)")
        return

    # Record initial weights for first owned param
    dp = optimizer._dedicated_params[0]
    w_before = dp._owned_data.clone()

    # forward -> backward -> step
    optimizer.zero_grad()
    x = torch.randn(4, 256, device=device)
    loss = model(x)
    loss.backward()
    optimizer.step()

    # Assert weights actually changed (NS update applied)
    max_diff = (dp._owned_data - w_before).abs().max().item()
    assert max_diff > 1e-6, (
        f"Rank {rank}: TP param weights did not change after optimizer step "
        f"(max_diff={max_diff}). Likely _reduced_grad was None."
    )

    torch.cuda.synchronize()
    log(rank, "PASSED: test_muon_tp_path")


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh("cuda", (world_size,))

    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    tests = {
        "reduced_grad": test_muon_reduced_grad_all_set,
        "momentum": test_muon_momentum_accumulation,
        "nesterov": test_muon_nesterov,
        "tp_path": test_muon_tp_path,
    }

    if test_name == "all":
        for name, fn in tests.items():
            log(rank, f"\n{'=' * 60}")
            log(rank, f"Running: {name}")
            log(rank, f"{'=' * 60}")
            dist.barrier()
            fn(rank, world_size, device, mesh)
            dist.barrier()
    elif test_name in tests:
        tests[test_name](rank, world_size, device, mesh)
    else:
        if rank == 0:
            print(f"Unknown test: {test_name}. Available: {list(tests.keys())}")
        sys.exit(1)

    dist.destroy_process_group()
