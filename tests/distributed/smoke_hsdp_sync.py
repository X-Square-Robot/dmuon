"""Phase B.7 smoke: 100-step HSDP-native DMuon on 4 GPUs (G=2, R=2).

Catches NaN / hang / leak without asserting a specific loss curve.  Prints
per-10-steps loss + final peak memory.

Run: ``torchrun --nproc_per_node=4 tests/distributed/smoke_hsdp_sync.py``
"""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

import dmuon
from dmuon import Muon, dedicate_params


class MLP(nn.Module):
    def __init__(self, hidden=512, intermediate=2048):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, hidden=512, intermediate=2048):
        super().__init__()
        self.mlp = MLP(hidden, intermediate)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class ToyTransformer(nn.Module):
    def __init__(self, num_layers=8, hidden=512, intermediate=2048):
        super().__init__()
        self.layers = nn.ModuleList(
            [Block(hidden, intermediate) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x).sum()


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        if rank == 0:
            print(f"SKIP: test requires 4 ranks (got {world_size})")
        dist.destroy_process_group()
        return
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    hsdp = init_device_mesh(
        "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
    )
    shard_mesh = hsdp["shard"]
    replicate_mesh = hsdp["replicate"]

    torch.manual_seed(42)
    model = ToyTransformer(num_layers=8, hidden=512, intermediate=2048).to(device)
    dedicate_params(
        model, shard_mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
        replicate_mesh=replicate_mesh,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=hsdp)
    fully_shard(model, mesh=hsdp)

    optimizer = Muon(
        model, lr=0.01, momentum=0.9, weight_decay=0.0,
        ns_steps=5,
        adamw_lr=1e-3,
    )

    num_steps = 100
    batch_size = 4
    hidden = 512

    if rank == 0:
        print(f"HSDP smoke: {world_size} GPUs, G=2, R=2, {num_steps} steps")
        print(f"  Model: 8 layers, hidden={hidden}")

    loss_history = []
    for step in range(num_steps):
        torch.manual_seed(1000 + step * world_size + rank)
        x = torch.randn(batch_size, hidden, device=device)
        optimizer.zero_grad()
        loss = model(x)
        loss.backward()
        optimizer.step()
        val = loss.item()
        loss_history.append(val)
        if math.isnan(val) or math.isinf(val):
            if rank == 0:
                print(f"FAIL: step {step} loss is NaN/Inf ({val})")
            dist.destroy_process_group()
            sys.exit(1)
        if rank == 0 and (step + 1) % 10 == 0:
            print(f"  step {step + 1:>3} loss={val:>+.6f}")

    dist.barrier()
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # Gather peak mem on rank 0
    peak_tensor = torch.tensor([peak_mb], device=device)
    all_peaks = [torch.zeros_like(peak_tensor) for _ in range(world_size)]
    dist.all_gather(all_peaks, peak_tensor)
    if rank == 0:
        peaks = [t.item() for t in all_peaks]
        print(f"\n  peak mem (MB) per rank: {['%.1f' % p for p in peaks]}")
        print(f"  loss[0]={loss_history[0]:+.4f}  loss[-1]={loss_history[-1]:+.4f}")
        print(f"\nPASS: 100 steps NaN-free")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
