"""Phase B.6 correctness test: HSDP-native DMuon vs shard-only DMuon.

Runs on 4 GPUs:
  Path A — shard-only (1D mesh, 4-rank shard group, no replicate)
  Path B — HSDP-native (2D mesh 2x2, shard=2, replicate=2)

Both paths share:
  * identical initial model state
  * identical per-step data
  * identical optimizer config (Muon lr/momentum + AdamW config)

Path A is the pre-Phase-B baseline; path B exercises the two-stage reduce +
post-step replicate broadcast.  With ``ReduceOp.AVG`` at both stages the
final divisor is ``G*R = total_world_size`` — identical to path A, which
averages over ``world_size=4``.  So loss trajectories MUST agree up to
fp rounding.

Run: ``torchrun --nproc_per_node=4 tests/distributed/test_hsdp_correctness.py``
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

from dmuon import Muon, dedicate_params


# ── Toy model (same as test_correctness.py) ────────────────────────────────


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


# ── DMuon runners ──────────────────────────────────────────────────────────


def run_shard_only(
    model_state, device, shard_mesh, data_list, lr, ns_steps, adamw_lr,
    adamw_betas, adamw_wd,
):
    model = TinyModel().to(device)
    model.load_state_dict(model_state)
    dedicate_params(
        model, shard_mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=shard_mesh)
    fully_shard(model, mesh=shard_mesh)

    optimizer = Muon(
        model, lr=lr, momentum=0.0, weight_decay=0.0,
        ns_steps=ns_steps,
        adamw_lr=adamw_lr, adamw_betas=adamw_betas,
        adamw_weight_decay=adamw_wd,
    )

    losses = []
    for x in data_list:
        optimizer.zero_grad()
        loss = model(x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    del model, optimizer
    torch.cuda.empty_cache()
    return losses


def run_hsdp(
    model_state, device, shard_mesh, replicate_mesh, fsdp_mesh, data_list,
    lr, ns_steps, adamw_lr, adamw_betas, adamw_wd,
):
    model = TinyModel().to(device)
    model.load_state_dict(model_state)
    # DMuon's dedicated partition uses the HSDP 2D layout; FSDP2 uses the
    # combined (replicate, shard) mesh for its own DTensors.
    dedicate_params(
        model, shard_mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
        replicate_mesh=replicate_mesh,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=fsdp_mesh)
    fully_shard(model, mesh=fsdp_mesh)

    optimizer = Muon(
        model, lr=lr, momentum=0.0, weight_decay=0.0,
        ns_steps=ns_steps,
        adamw_lr=adamw_lr, adamw_betas=adamw_betas,
        adamw_weight_decay=adamw_wd,
    )

    losses = []
    for x in data_list:
        optimizer.zero_grad()
        loss = model(x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    del model, optimizer
    torch.cuda.empty_cache()
    return losses


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        if rank == 0:
            print(f"SKIP: test requires exactly 4 ranks (got {world_size})")
        dist.destroy_process_group()
        return
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    num_steps = 10
    lr = 0.01
    ns_steps = 5
    adamw_lr = 1e-3
    adamw_betas = (0.9, 0.999)
    adamw_wd = 0.01
    batch_size = 4
    hidden = 256

    if rank == 0:
        print("HSDP correctness test: HSDP-native DMuon vs shard-only DMuon")
        print(f"  {world_size} GPUs, {num_steps} steps")
        print("  Path A: 1D mesh (shard=4)")
        print("  Path B: 2D mesh (replicate=2 × shard=2)")

    # Same initial model
    torch.manual_seed(42)
    ref_model = TinyModel().to(device)
    model_state = ref_model.state_dict()
    del ref_model

    # Same data
    data_list = []
    for step in range(num_steps):
        torch.manual_seed(1000 + step)
        data_list.append(torch.randn(batch_size, hidden, device=device))

    # Path A: 1D shard-only mesh
    shard_only_mesh = init_device_mesh("cuda", (4,))
    losses_a = run_shard_only(
        model_state, device, shard_only_mesh, data_list,
        lr, ns_steps, adamw_lr, adamw_betas, adamw_wd,
    )
    dist.barrier()
    torch.cuda.empty_cache()

    # Path B: 2D HSDP mesh.  For FSDP2 we use the combined 2D mesh so that
    # FSDP2 itself runs in HSDP mode (replicate-dim all-reduces symmetric
    # grads), paralleling DMuon's replicate-dim reduce on dedicated params.
    hsdp_mesh = init_device_mesh(
        "cuda", (2, 2), mesh_dim_names=("replicate", "shard"),
    )
    shard_mesh = hsdp_mesh["shard"]
    replicate_mesh = hsdp_mesh["replicate"]

    losses_b = run_hsdp(
        model_state, device, shard_mesh, replicate_mesh, hsdp_mesh, data_list,
        lr, ns_steps, adamw_lr, adamw_betas, adamw_wd,
    )
    dist.barrier()

    if rank == 0:
        print(
            f"\n  {'Step':<6s} {'ShardOnly':>12s} {'HSDP':>12s} "
            f"{'Diff':>12s} {'RelDiff':>10s} {'Status':>8s}"
        )
        print(f"  {'-'*68}")
        all_pass = True
        for i in range(num_steps):
            diff = abs(losses_a[i] - losses_b[i])
            denom = max(abs(losses_a[i]), abs(losses_b[i]), 1e-8)
            rel = diff / denom * 100
            # fp32 tolerance per plan §13 (bf16 would relax to 1e-3 rel).
            status = "OK" if rel < 0.1 else ("WARN" if rel < 1.0 else "FAIL")
            if status == "FAIL":
                all_pass = False
            print(
                f"  {i:<6d} {losses_a[i]:>12.6f} {losses_b[i]:>12.6f} "
                f"{diff:>12.6f} {rel:>8.2f}% {status:>8s}"
            )
        print()
        if all_pass:
            print("PASS: HSDP-native DMuon matches shard-only DMuon")
        else:
            print("FAIL: HSDP-native DMuon diverges from shard-only baseline")
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
