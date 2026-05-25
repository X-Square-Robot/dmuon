"""Tensor Parallel + Data Parallel training with DMuon.

2D mesh (DP x TP) example.  DMuon detects TP-sharded parameters from
their ``DTensor`` structure automatically, so the required setup still
passes the same DP mesh slice used by FSDP2.
For each TP-sharded parameter DMuon runs a TP gather →
full-matrix Newton-Schulz on the TP owner → scatter back, preserving
Muon's exact math while every rank keeps only its TP shard in memory.

Run with: torchrun --nproc_per_node=4 examples/tp_dp.py
Requires at least 4 GPUs (2 DP x 2 TP).
"""

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

import dmuon


# ---------------------------------------------------------------------------
# Model with attention-like structure
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, d=256, n_heads=4, n_kv_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d // n_heads
        self.q_proj = nn.Linear(d, d, bias=False)                               # (256, 256)
        self.k_proj = nn.Linear(d, n_kv_heads * self.head_dim, bias=False)      # (128, 256) — GQA
        self.v_proj = nn.Linear(d, n_kv_heads * self.head_dim, bias=False)      # (128, 256) — GQA
        self.o_proj = nn.Linear(d, d, bias=False)                               # (256, 256)

    def forward(self, x):
        # Simplified: no actual attention, just projections for demo
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(q + k.repeat(1, 1, self.n_heads // self.n_kv_heads) + v.repeat(1, 1, self.n_heads // self.n_kv_heads))


class MLP(nn.Module):
    def __init__(self, d=256, ff=1024):
        super().__init__()
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, d=256, ff=1024, n_heads=4, n_kv_heads=2):
        super().__init__()
        self.attn = Attention(d, n_heads, n_kv_heads)
        self.mlp = MLP(d, ff)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, num_layers=4, d=256, ff=1024, n_heads=4, n_kv_heads=2):
        super().__init__()
        self.layers = nn.ModuleList([
            Block(d, ff, n_heads, n_kv_heads) for _ in range(num_layers)
        ])
        self.head = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x).sum()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size >= 4, "This example requires at least 4 GPUs"
    torch.cuda.set_device(rank)

    # 2D mesh: 2 DP x 2 TP
    dp_size = 2
    tp_size = world_size // dp_size
    mesh_2d = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
    dp_mesh = mesh_2d["dp"]
    tp_mesh = mesh_2d["tp"]

    if rank == 0:
        print(f"Mesh: {dp_size} DP x {tp_size} TP = {world_size} GPUs")

    # Build model
    torch.manual_seed(42)
    model = TinyTransformer().cuda()

    # Step 1: Apply TP
    for layer in model.layers:
        parallelize_module(
            layer.attn, tp_mesh,
            {
                "q_proj": ColwiseParallel(),    # Shard(0)
                "k_proj": ColwiseParallel(),    # Shard(0) — GQA, narrow
                "v_proj": ColwiseParallel(),    # Shard(0) — GQA, narrow
                "o_proj": RowwiseParallel(),    # Shard(1)
            },
        )
        parallelize_module(
            layer.mlp, tp_mesh,
            {
                "gate_proj": ColwiseParallel(),  # Shard(0)
                "up_proj": ColwiseParallel(),    # Shard(0)
                "down_proj": RowwiseParallel(),  # Shard(1)
            },
        )

    # Step 2: DMuon (uses dp_mesh)
    dmuon.dedicate_params(
        model, dp_mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )

    # Step 3: FSDP2 (uses dp_mesh)
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)

    # Optimizer — defaults are enough for TP.  DMuon's TP path (gather →
    # full-matrix NS on the TP owner → scatter back) activates automatically
    # for any DTensor parameter sharded on a non-DP mesh dim.  Advanced runs
    # can set tp_buffer_reuse= on dedicate_params() or tp_distributed_gram=
    # on Muon.
    optimizer = dmuon.Muon(
        model, lr=0.02, momentum=0.95,
        adamw_lr=1e-3,
    )

    # Inspect TP properties
    if rank == 0:
        print(f"\nNS backend: {dmuon.get_ns_backend()}")
        print(f"\nDedicated params owned by rank 0:")
        for dp in dmuon.get_owned_params(model, rank=0):
            tp_info = (
                f"tp_size={dp.tp_group.size()}, is_tp_owner={dp.is_tp_owner}"
                if dp.tp_group is not None else "no TP"
            )
            print(
                f"  {dp.param_name}: local={tuple(dp._orig_size)}, "
                f"full={tuple(dp.full_shape)}, "
                f"shard_dim={dp.shard_dim}, {tp_info}"
            )
        print()

    # Training loop
    for step in range(30):
        optimizer.zero_grad()
        x = torch.randn(4, 8, 256, device="cuda")  # (batch, seq, d)
        loss = model(x)
        loss.backward()
        optimizer.step()

        if rank == 0 and step % 10 == 0:
            print(f"step {step:3d}  loss={loss.item():.4f}")

    dist.destroy_process_group()
    if rank == 0:
        print("Done!")


if __name__ == "__main__":
    main()
