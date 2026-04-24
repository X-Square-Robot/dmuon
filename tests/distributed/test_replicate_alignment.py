"""Audit replicate_broadcast sync/async alignment — HSDP-only, no TP.

Isolates the ReplicateBroadcastState.replicate_input pin from the TP
scatter path.  If sync_vs_async_gap == 0 here, the pin is benign (as
``_owned_data`` is a persistent tensor whose lifetime isn't governed by
the state tuple).  If gap > 0, the pin has the same bug as TP did and
needs the same fix.

Mesh: ``(replicate=2, shard=2)`` on 4 GPUs.  Model has pure-DP params
only (no ``parallelize_module``), so only replicate_broadcast fires.

Env:
  DMUON_RALIGN_MODE      ∈ {sync, async}
  DMUON_RALIGN_RUN       int (output-file name only)
  DMUON_RALIGN_OUT       output dir
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import dmuon
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import fully_shard


class MLP(nn.Module):
    def __init__(self, h=256, inter=1024):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, h=256, inter=1024):
        super().__init__()
        self.ln = nn.LayerNorm(h)
        self.mlp = MLP(h, inter)

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class Tiny(nn.Module):
    def __init__(self, num_layers=2, h=256, inter=1024):
        super().__init__()
        self.layers = nn.ModuleList([Block(h, inter) for _ in range(num_layers)])
        self.out = nn.Linear(h, h, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x).mean()


def main() -> int:
    mode = os.environ.get("DMUON_RALIGN_MODE", "sync")
    run_id = os.environ.get("DMUON_RALIGN_RUN", "0")
    out_dir = os.environ.get("DMUON_RALIGN_OUT", "/tmp/dmuon_ralign")
    assert mode in ("sync", "async"), f"bad mode {mode!r}"

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    # Auto-pick a 2D (replicate, shard) shape from the world size.
    if ws == 4:
        R, G = 2, 2
    elif ws == 8:
        R, G = 2, 4
    else:
        if rank == 0:
            print(f"SKIP: needs world 4 or 8, got {ws}")
        dist.destroy_process_group()
        return 0
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    mesh = init_device_mesh(
        "cuda", (R, G), mesh_dim_names=("replicate", "shard")
    )
    shard_mesh = mesh["shard"]
    replicate_mesh = mesh["replicate"]

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = Tiny(num_layers=2, h=256, inter=1024).to(device)

    # NO parallelize_module — pure HSDP, no TP at all
    dmuon.dedicate_params(
        model, shard_mesh,
        replicate_mesh=replicate_mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    replicate_async = (mode == "async")
    optimizer = dmuon.Muon(
        model, lr=0.02, momentum=0.95, weight_decay=0.01,
        adamw_lr=1e-3, replicate_async=replicate_async,
    )

    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    inputs = [torch.randn(4, 16, 256, device=device) for _ in range(3)]

    losses: list[float] = []
    digests: list[float] = []
    for it, x in enumerate(inputs):
        optimizer.zero_grad()
        loss = model(x)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        if optimizer._dedicated_params:
            d = optimizer._dedicated_params[0]._owned_data.float().mean().item()
        else:
            d = float("nan")
        losses.append(float(loss.item()))
        digests.append(d)
        if rank == 0:
            print(f"[{mode} r={run_id}] iter {it}: loss={loss.item():.10f} "
                  f"owned0_mean={d:.12e}", flush=True)

    torch.cuda.synchronize()
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{mode}_{run_id}.json")
        with open(out, "w") as f:
            json.dump({"mode": mode, "run_id": run_id,
                       "losses": losses, "owned0_mean": digests}, f)
        print(f"wrote {out}", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
