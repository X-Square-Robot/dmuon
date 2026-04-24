"""T2c — NSight-style overlap profile for the TP gather path.

Captures a ``torch.profiler`` trace on 8 GPUs (3D mesh `R=2, G=2, T=2`),
larger model (8 × h=2048 × inter=8192), with ``torch.compile`` on each
transformer block.  Runs 3 iters: iter 0 warmup, iter 1-2 recorded.

Outputs a Chrome-tracing JSON under the `--out` path.  Pair with
``parse_tp_overlap.py`` to compute the overlap metric CLI-side.

Usage:
    torchrun --nproc_per_node=8 profile_tp_overlap.py \\
        --out /tmp/tp_overlap.pt.trace.json
"""

from __future__ import annotations

import argparse
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
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


class MLP(nn.Module):
    def __init__(self, h: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, h: int, inter: int):
        super().__init__()
        self.ln = nn.LayerNorm(h)
        self.mlp = MLP(h, inter)

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class Tiny(nn.Module):
    def __init__(self, num_layers: int, h: int, inter: int):
        super().__init__()
        self.layers = nn.ModuleList([Block(h, inter) for _ in range(num_layers)])
        self.out = nn.Linear(h, h, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x).mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="trace JSON output path")
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--no-compile", action="store_true",
                    help="disable torch.compile wrap (debug)")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    if ws != 8:
        if rank == 0:
            print(f"SKIP: needs world=8, got {ws}")
        dist.destroy_process_group()
        return 0
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    mesh3d = init_device_mesh(
        "cuda", (2, 2, 2), mesh_dim_names=("replicate", "shard", "tp")
    )
    shard_mesh = mesh3d["shard"]
    replicate_mesh = mesh3d["replicate"]
    tp_mesh = mesh3d["tp"]

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = Tiny(args.num_layers, args.hidden, args.inter).to(device)

    plan = {
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }
    for layer in model.layers:
        parallelize_module(layer, tp_mesh, plan)

    dmuon.dedicate_params(
        model, shard_mesh,
        replicate_mesh=replicate_mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )
    dp_mesh_2d = mesh3d["replicate", "shard"]
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh_2d)
    fully_shard(model, mesh=dp_mesh_2d)

    # torch.compile each block in place (after fully_shard so FSDP2 wraps
    # the compiled callable as-is).
    if not args.no_compile:
        for i, layer in enumerate(model.layers):
            model.layers[i] = torch.compile(layer, dynamic=False)

    optimizer = dmuon.Muon(
        model, lr=0.02, momentum=0.95, weight_decay=0.01,
        adamw_lr=1e-3, replicate_async=True,  # hit T2d path
    )
    if rank == 0:
        print(f"profile: layers={args.num_layers} h={args.hidden} inter={args.inter} "
              f"batch={args.batch} seq={args.seqlen} compile={not args.no_compile} "
              f"dedicated={len(optimizer._dedicated_params)}", flush=True)

    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    inputs = [torch.randn(args.batch, args.seqlen, args.hidden, device=device)
              for _ in range(4)]

    # Warm up (iter 0) outside profiler
    optimizer.zero_grad()
    loss = model(inputs[0])
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    # Profiler window covers iter 1-2
    from torch.profiler import profile, ProfilerActivity, schedule

    sched = schedule(wait=0, warmup=0, active=2, repeat=1)

    out_path = args.out
    if rank == 0:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def handler(prof) -> None:
        if rank == 0:
            prof.export_chrome_trace(out_path)
            print(f"wrote {out_path}", flush=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        on_trace_ready=handler,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for it in range(1, 3):
            optimizer.zero_grad()
            loss = model(inputs[it])
            loss.backward()
            optimizer.step()
            if rank == 0:
                print(f"iter {it}: loss={loss.item():.6f}", flush=True)
            prof.step()
    torch.cuda.synchronize()

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
