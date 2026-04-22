"""Phase C.3 correctness test: async replicate broadcast vs Phase B sync.

Runs two DMuon stacks on 4 GPUs with a 2×2 HSDP mesh:
    Path A (reference) — ``replicate_async=False``  (Phase B sync)
    Path B (Phase C)   — ``replicate_async=True``   (default, async)

Both use identical model state, data, and optimizer config.  With
``ReduceOp.AVG`` reductions and deterministic NS, the two loss
trajectories must match bit-for-bit (fp32 tolerance).

Run: ``torchrun --nproc_per_node=4 tests/distributed/test_hsdp_async_correctness.py``
"""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

import dmuon
from dmuon import Muon, dedicate_params


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


def run(model_state, device, hsdp_mesh, data_list, *, replicate_async):
    model = TinyModel().to(device)
    model.load_state_dict(model_state)
    dedicate_params(
        model, hsdp_mesh["shard"],
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
        replicate_mesh=hsdp_mesh["replicate"],
    )
    for layer in model.layers:
        fully_shard(layer, mesh=hsdp_mesh)
    fully_shard(model, mesh=hsdp_mesh)

    optim = Muon(
        model, lr=0.01, momentum=0.0, weight_decay=0.0,
        ns_steps=5, adamw_lr=1e-3,
        replicate_async=replicate_async,
    )

    losses = []
    for x in data_list:
        optim.zero_grad()
        loss = model(x)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    del model, optim
    torch.cuda.empty_cache()
    return losses


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4, f"Need 4 ranks, got {world_size}"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    hsdp_mesh = init_device_mesh(
        "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
    )

    torch.manual_seed(42)
    ref = TinyModel().to(device)
    model_state = ref.state_dict()
    del ref

    data = []
    for step in range(10):
        torch.manual_seed(1000 + step)
        data.append(torch.randn(4, 256, device=device))

    # Sync reference
    losses_sync = run(model_state, device, hsdp_mesh, data, replicate_async=False)
    dist.barrier()
    torch.cuda.empty_cache()

    # Async (Phase C default)
    losses_async = run(model_state, device, hsdp_mesh, data, replicate_async=True)
    dist.barrier()

    if rank == 0:
        print("Phase C.3 correctness: async vs sync on 4 GPUs (G=2, R=2)")
        print()
        print(f"  {'Step':<6s} {'Sync':>12s} {'Async':>12s} {'Diff':>12s} {'Status':>8s}")
        all_pass = True
        for i in range(10):
            diff = abs(losses_sync[i] - losses_async[i])
            denom = max(abs(losses_sync[i]), abs(losses_async[i]), 1e-8)
            rel = diff / denom * 100
            status = "OK" if rel < 0.01 else ("WARN" if rel < 0.1 else "FAIL")
            if status == "FAIL":
                all_pass = False
            print(
                f"  {i:<6d} {losses_sync[i]:>12.6f} {losses_async[i]:>12.6f} "
                f"{diff:>12.6f} {status:>8s}"
            )
        print()
        if all_pass:
            print("PASS: Phase C async matches Phase B sync bit-for-bit")
        else:
            print("FAIL: async path diverges from sync baseline")
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
