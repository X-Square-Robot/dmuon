"""End-to-end TP+DP test: dedicate_params with tensor parallelism.

Validates that dedicated ownership works correctly when parameters are
TP-sharded DTensors. The DP broadcast operates on _local_tensor (TP shard).

Run with: torchrun --nproc_per_node=8 tests/test_e2e_tp_dp.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmuon import dedicate_params
from dmuon.utils import get_owned_params

# ---- Model ----


class MLP(nn.Module):
    def __init__(self, hidden=256, intermediate=512):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, hidden=256, intermediate=512):
        super().__init__()
        self.mlp = MLP(hidden, intermediate)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class SmallModel(nn.Module):
    def __init__(self, num_layers=2, hidden=256, intermediate=512):
        super().__init__()
        self.layers = nn.ModuleList([Block(hidden, intermediate) for _ in range(num_layers)])
        self.head = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x).sum()


# ---- NS ----


def newton_schulz(G, steps=3):
    X = G / (G.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = 1.5 * A - 0.5 * A @ A
        X = B @ X
    return X


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    assert world_size == 8, f"This test requires 8 GPUs, got {world_size}"

    # 2D mesh: DP=4, TP=2
    mesh_2d = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
    dp_mesh = mesh_2d["dp"]
    tp_mesh = mesh_2d["tp"]

    dp_rank = dp_mesh.get_local_rank()
    tp_rank = tp_mesh.get_local_rank()

    if rank == 0:
        print(f"TP+DP test: {world_size} GPUs, DP=4 TP=2")
        print(f"  dp_mesh size={dp_mesh.size()}, tp_mesh size={tp_mesh.size()}")

    seed = 42
    num_steps = 5
    lr = 0.01

    # Build model
    torch.manual_seed(seed)
    model = SmallModel().to(device)

    # Step 1: Apply TP (before FSDP and dedicate_params)
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

    if rank == 0:
        print("TP applied. Parameter shapes (local):")
        for n, p in model.named_parameters():
            if hasattr(p, "_local_tensor"):
                print(f"  {n}: DTensor, local={p._local_tensor.shape}")
            else:
                print(f"  {n}: Tensor, shape={p.shape}")

    # Step 2: Dedicate proj params (operates on TP-sharded DTensors)
    assignment = dedicate_params(
        model,
        dp_mesh,
        predicate=lambda name, param: "proj" in name,
    )

    if rank == 0:
        print(f"\nDedicated {len(assignment)} params on dp_mesh")

    # Step 3: FSDP2 for remaining params
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)

    owned = get_owned_params(model, dp_rank)
    if rank == 0:
        print(f"Rank 0 (dp_rank={dp_rank}, tp_rank={tp_rank}) owns {len(owned)} params")

    # Training loop
    losses = []
    for step in range(num_steps):
        # Same data across all ranks (DP averages grads, but forward should match
        # when params are identical, which they are after broadcast)
        torch.manual_seed(seed + step + 1000)
        x = torch.randn(4, 256, device=device)

        loss = model(x)
        loss.backward()
        losses.append(loss.item())

        # Muon step on owned dedicated params
        for d_param in owned:
            if d_param._reduced_grad is not None:
                grad = d_param._reduced_grad
                param = d_param._owned_data
                G = grad.view(grad.shape[0], -1).float()
                update = newton_schulz(G)
                param.add_(update.view(param.shape).to(param.dtype), alpha=-lr)
                d_param._reduced_grad = None

        # SGD on symmetric params
        for module in [*model.layers, model]:
            state = module._get_fsdp_state()
            if state._fsdp_param_group is None:
                continue
            for fsdp_param in state._fsdp_param_group.fsdp_params:
                if fsdp_param.sharded_param.grad is not None:
                    fsdp_param.sharded_param._local_tensor.add_(
                        fsdp_param.sharded_param.grad._local_tensor, alpha=-lr
                    )
                    fsdp_param.sharded_param.grad = None

        if rank == 0:
            print(f"  Step {step}: loss = {loss.item():.6f}")

    # Verify: same TP position ranks should have same loss (they share params via DP broadcast)
    loss_tensor = torch.tensor(losses, device=device)

    # Gather within DP group (same TP position)
    dp_group = dp_mesh.get_group()
    dp_losses = [torch.zeros_like(loss_tensor) for _ in range(dp_mesh.size())]
    dist.all_gather(dp_losses, loss_tensor, group=dp_group)

    if rank == 0:
        print(f"\nLosses (rank 0): {[f'{l:.6f}' for l in losses]}")

        # Check: all DP peers (same TP position) have same loss
        all_match = True
        for step_idx in range(num_steps):
            step_losses = [dp_losses[r][step_idx].item() for r in range(dp_mesh.size())]
            max_diff = max(step_losses) - min(step_losses)
            if max_diff > 1e-3:
                print(f"FAILED: step {step_idx} DP loss mismatch: {step_losses}")
                all_match = False

        if all_match:
            print("PASSED: DP peers computed identical losses (DP communication correct)")

        if abs(losses[-1] - losses[0]) > 1e-6:
            print("PASSED: training produced parameter updates (loss changed)")
        else:
            print("WARNING: loss unchanged")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
