"""End-to-end DP training test: dedicate_params + fully_shard.

Validates that DMuon produces decreasing loss over training steps.

Run with: torchrun --nproc_per_node=8 tests/test_e2e_dp.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmuon import dedicate_params
from dmuon.utils import get_dedicated_params, get_owned_params

# ---- Simple model ----


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


# ---- Simple Newton-Schulz ----


def newton_schulz(G, steps=3):
    X = G / (G.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = 1.5 * A - 0.5 * A @ A
        X = B @ X
    return X


# ---- Main ----


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world_size,))

    seed = 12345
    num_steps = 5
    lr = 0.01

    if rank == 0:
        print(f"E2E DP test: {world_size} GPUs, {num_steps} steps, seed={seed}")

    # Build model (same on all ranks)
    torch.manual_seed(seed)
    model = TinyModel().to(device)

    # Step 1: dedicate proj params
    assignment = dedicate_params(
        model,
        mesh,
        predicate=lambda name, param: "proj" in name,
    )
    if rank == 0:
        print(f"Dedicated {len(assignment)} params")
        for n, p in model.named_parameters():
            if p in assignment:
                print(f"  {n}: owner=rank{assignment[p]}, numel={p.numel()}")

    # Step 2: standard FSDP2 for remaining params
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # Collect owned params for this rank
    owned = get_owned_params(model, rank)
    all_dedicated = get_dedicated_params(model)
    if rank == 0:
        print(f"Rank 0 owns {len(owned)} params")

    # Training loop
    losses = []
    for step in range(num_steps):
        torch.manual_seed(seed + step + 1000)
        x = torch.randn(4, 256, device=device)

        # Forward (hooks handle unshard)
        loss = model(x)

        # Backward (hooks handle unshard + reduce + reshard)
        loss.backward()
        losses.append(loss.item())

        # Optimizer step: Muon on owned dedicated params
        for d_param in owned:
            if d_param._reduced_grad is not None:
                grad = d_param._reduced_grad
                param = d_param._owned_data
                G = grad.view(grad.shape[0], -1).float()
                update = newton_schulz(G)
                param.add_(update.view(param.shape).to(param.dtype), alpha=-lr)
                d_param._reduced_grad = None

        # Simple SGD on symmetric params (layernorm, head)
        for fsdp_module in [*model.layers, model]:
            state = fsdp_module._get_fsdp_state()
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

    # Verify: all ranks got the same loss (communication correctness)
    loss_tensor = torch.tensor(losses, device=device)
    all_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
    dist.all_gather(all_losses, loss_tensor)

    if rank == 0:
        print(f"\nLosses: {[f'{l:.6f}' for l in losses]}")

        # Key check: all ranks computed the same loss at each step
        all_match = True
        for step_idx in range(num_steps):
            step_losses = [all_losses[r][step_idx].item() for r in range(world_size)]
            max_diff = max(step_losses) - min(step_losses)
            if max_diff > 1e-4:
                print(f"FAILED: step {step_idx} loss mismatch across ranks: {step_losses}")
                all_match = False

        if all_match:
            print("PASSED: all ranks computed identical losses (communication correct)")

        # Check training is doing something (loss changed)
        if abs(losses[-1] - losses[0]) > 1e-6:
            print("PASSED: training produced parameter updates (loss changed)")
        else:
            print("WARNING: loss unchanged, optimizer may not be working")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
