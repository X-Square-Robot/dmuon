"""Basic data parallel training with DMuon.

Run with: torchrun --nproc_per_node=4 examples/basic_dp.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

import dmuon


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, d=512, ff=2048):
        super().__init__()
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        h = self.ln(x)
        return x + self.down_proj(self.gate_proj(h) * self.up_proj(h))


class TinyModel(nn.Module):
    def __init__(self, num_layers=4, d=512, ff=2048):
        super().__init__()
        self.layers = nn.ModuleList([Block(d, ff) for _ in range(num_layers)])
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
    torch.cuda.set_device(rank)

    mesh = init_device_mesh("cuda", (world_size,))

    # Build model
    torch.manual_seed(42)
    model = TinyModel().cuda()

    # DMuon: dedicate 2D projection layers
    dmuon.dedicate_params(
        model, mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )

    # FSDP2: shard remaining parameters (LayerNorm, head)
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # Optimizer
    optimizer = dmuon.Muon(
        model, lr=0.02, momentum=0.95, ns_steps=5,
        adamw_lr=1e-3, adamw_weight_decay=0.01,
    )

    if rank == 0:
        owned = dmuon.get_owned_params(model, rank=rank)
        total = sum(dp.numel for dp in owned)
        print(f"NS backend: {dmuon.get_ns_backend()}")
        print(f"Rank 0 owns {len(owned)} dedicated params ({total:,} elements)")
        print()

    # Training loop
    for step in range(50):
        optimizer.zero_grad()
        x = torch.randn(4, 512, device="cuda")
        loss = model(x)
        loss.backward()
        optimizer.step()

        if rank == 0 and step % 10 == 0:
            print(f"step {step:3d}  loss={loss.item():.4f}")

    # Checkpoint save/load demo
    if rank == 0:
        print("\nSaving checkpoint...")
    model_sd = dmuon.get_model_state_dict(model)
    optim_sd = dmuon.get_optimizer_state_dict(model, optimizer)
    if rank == 0:
        torch.save({"model": model_sd, "optim": optim_sd, "step": 50}, "/tmp/dmuon_ckpt.pt")
    dist.barrier()

    # Load it back
    ckpt = torch.load("/tmp/dmuon_ckpt.pt", map_location="cpu")
    dmuon.set_model_state_dict(model, ckpt["model"])
    dmuon.set_optimizer_state_dict(model, optimizer, ckpt["optim"])
    if rank == 0:
        print("Checkpoint loaded successfully!")

    # Clean up
    dist.barrier()
    if rank == 0 and os.path.exists("/tmp/dmuon_ckpt.pt"):
        os.remove("/tmp/dmuon_ckpt.pt")

    dist.destroy_process_group()
    if rank == 0:
        print("Done!")


if __name__ == "__main__":
    main()
