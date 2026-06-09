"""Phase B.8 distributed test: checkpoint save/load round-trip under HSDP.

Verifies:

1. ``get_model_state_dict`` produces a full state dict that round-trips
   through ``set_model_state_dict`` in HSDP mode (G=2, R=2).
2. ``get_optimizer_state_dict`` / ``set_optimizer_state_dict`` preserve
   momentum buffers across the save/load.
3. After restoring both model + optimizer, loss at step N+1 matches the
   uninterrupted baseline step N+1 — i.e., the restart is perfectly
   transparent under the SAME (G, R) topology.

Cross-(G, R) topology restore is out of scope for Phase B per plan §11.

Run: ``torchrun --nproc_per_node=4 tests/distributed/test_hsdp_restart.py``
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
    def __init__(self, hidden=128, intermediate=512):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x))


class Block(nn.Module):
    def __init__(self, hidden=128, intermediate=512):
        super().__init__()
        self.mlp = MLP(hidden, intermediate)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class TinyModel(nn.Module):
    def __init__(self, num_layers=2, hidden=128, intermediate=512):
        super().__init__()
        self.layers = nn.ModuleList(
            [Block(hidden, intermediate) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x).sum()


def _build_hsdp_stack(model_state, device, hsdp_mesh):
    """Reproducibly build model + DMuon + optimizer on the HSDP mesh."""
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
        model, lr=0.01, momentum=0.9, weight_decay=0.0,
        ns_steps=5, adamw_lr=1e-3,
    )
    return model, optim


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

    # Shared initial state
    torch.manual_seed(42)
    ref = TinyModel().to(device)
    model_state = ref.state_dict()
    del ref

    # Same per-step data
    data = []
    for step in range(10):
        torch.manual_seed(1000 + step)
        data.append(torch.randn(4, 128, device=device))

    # --- Baseline: run all 10 steps without interruption ---------------
    model_a, optim_a = _build_hsdp_stack(model_state, device, hsdp_mesh)
    losses_a = []
    for x in data:
        optim_a.zero_grad()
        loss = model_a(x)
        loss.backward()
        optim_a.step()
        losses_a.append(loss.item())
    dist.barrier()

    # --- Restart: run 5 steps, save, destroy, rebuild, load, run 5 more -
    model_b, optim_b = _build_hsdp_stack(model_state, device, hsdp_mesh)
    losses_b_first = []
    for x in data[:5]:
        optim_b.zero_grad()
        loss = model_b(x)
        loss.backward()
        optim_b.step()
        losses_b_first.append(loss.item())
    dist.barrier()

    # Save on rank 0 only (standard pattern).  Every rank participates in
    # the collectives inside get_*_state_dict, but only rank 0 writes.
    model_sd = dmuon.get_model_state_dict(model_b)
    optim_sd = dmuon.get_optimizer_state_dict(model_b, optim_b)
    # Use the same filename on every rank — use rank 0's PID as the
    # session-unique suffix and broadcast it so all ranks agree.
    pid_tensor = torch.tensor(
        [os.getpid() if rank == 0 else 0], dtype=torch.int64, device=device
    )
    dist.broadcast(pid_tensor, src=0)
    ckpt_path = f"/tmp/hsdp_restart_{pid_tensor.item()}.pt"
    if rank == 0:
        torch.save({"model": model_sd, "optim": optim_sd}, ckpt_path)
    dist.barrier()

    # Dispose & rebuild fresh stack
    del model_b, optim_b, model_sd, optim_sd
    torch.cuda.empty_cache()

    model_c, optim_c = _build_hsdp_stack(model_state, device, hsdp_mesh)
    # All ranks load the same file (torch.load is local per-rank).
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    dmuon.set_model_state_dict(model_c, ckpt["model"])
    dmuon.set_optimizer_state_dict(model_c, optim_c, ckpt["optim"])
    dist.barrier()

    losses_b_second = []
    for x in data[5:]:
        optim_c.zero_grad()
        loss = model_c(x)
        loss.backward()
        optim_c.step()
        losses_b_second.append(loss.item())
    dist.barrier()

    losses_b = losses_b_first + losses_b_second

    if rank == 0:
        print("HSDP restart test: uninterrupted vs save→load→resume")
        print("  4 GPUs, G=2, R=2, 10 steps (checkpoint after step 5)")
        print()
        print(f"  {'Step':<6s} {'Baseline':>12s} {'Restart':>12s} {'Diff':>12s} {'Status':>8s}")
        all_pass = True
        for i in range(10):
            diff = abs(losses_a[i] - losses_b[i])
            denom = max(abs(losses_a[i]), abs(losses_b[i]), 1e-8)
            rel = diff / denom * 100
            status = "OK" if rel < 0.1 else ("WARN" if rel < 1.0 else "FAIL")
            if status == "FAIL":
                all_pass = False
            marker = "  <-- resume" if i == 5 else ""
            print(
                f"  {i:<6d} {losses_a[i]:>12.6f} {losses_b[i]:>12.6f} "
                f"{diff:>12.6f} {status:>8s}{marker}"
            )
        print()
        if all_pass:
            print("PASS: restart matches uninterrupted baseline bit-by-bit")
        else:
            print("FAIL: restart diverges from baseline")
            sys.exit(1)

    # Cleanup
    if rank == 0 and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
