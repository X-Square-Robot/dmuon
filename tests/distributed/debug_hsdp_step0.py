"""Debug: compare step-0 forward-pass loss for shard-only vs HSDP paths.

If step 0 losses differ, the bug is in how the initial forward is wired
(probably something about param broadcast in ``unshard()``), not in the
reduce/update pipeline.

Usage: ``torchrun --nproc_per_node=4 tests/distributed/debug_hsdp_step0.py``
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

from dmuon import dedicate_params


class SimpleProj(nn.Module):
    """Bare-bones model: single proj (dedicated) + identity skip.

    Forward does NOT go through FSDP2 — we test DMuon's own broadcast path
    in isolation.
    """

    def __init__(self, hidden=128):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.proj(x).sum()


def run(model_state, device, mesh_args, x):
    """Build a SimpleProj with the given mesh config, forward x, return loss."""
    model = SimpleProj().to(device)
    model.load_state_dict(model_state)

    shard_mesh = mesh_args["shard"]
    replicate_mesh = mesh_args.get("replicate")
    dedicate_params(
        model, shard_mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
        replicate_mesh=replicate_mesh,
    )
    # NOTE: NO fully_shard — isolates DMuon's own broadcast path.
    with torch.no_grad():
        loss = model(x)
    loss_val = loss.item()

    del model
    torch.cuda.empty_cache()
    return loss_val


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    torch.manual_seed(42)
    ref = SimpleProj().to(device)
    model_state = ref.state_dict()
    del ref

    torch.manual_seed(1000)
    x = torch.randn(4, 128, device=device)

    # Path A: 1D
    shard_only_mesh = init_device_mesh("cuda", (4,))
    loss_a = run(model_state, device, {"shard": shard_only_mesh}, x)
    dist.barrier()

    # Path B: HSDP 2x2
    hsdp = init_device_mesh("cuda", (2, 2), mesh_dim_names=("replicate", "shard"))
    loss_b = run(
        model_state, device,
        {"shard": hsdp["shard"], "replicate": hsdp["replicate"]},
        x,
    )
    dist.barrier()

    # Gather all rank losses on rank 0.
    loss_a_tensor = torch.tensor([loss_a], device=device)
    loss_b_tensor = torch.tensor([loss_b], device=device)
    all_a = [torch.zeros_like(loss_a_tensor) for _ in range(world_size)]
    all_b = [torch.zeros_like(loss_b_tensor) for _ in range(world_size)]
    dist.all_gather(all_a, loss_a_tensor)
    dist.all_gather(all_b, loss_b_tensor)

    if rank == 0:
        print(f"{'rank':>4} {'shard-only':>15} {'hsdp':>15} {'diff':>15}")
        for r in range(world_size):
            a = all_a[r].item()
            b = all_b[r].item()
            print(f"{r:>4} {a:>15.6f} {b:>15.6f} {abs(a-b):>15.6f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
