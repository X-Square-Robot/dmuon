"""End-to-end benchmark: DMuon vs standard FSDP2.

Compares training step time breakdown:
  A. Standard FSDP2 + redundant NS (baseline)
  B. DMuon: dedicated ownership + owner-only NS

Run with: torchrun --nproc_per_node=8 benchmarks/bench_e2e.py
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---- Model (larger than test, smaller than Qwen) ----

class MLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = MLP(hidden, intermediate)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x):
        h = x + self.o_proj(self.q_proj(self.ln1(x)))
        return h + self.mlp(self.ln2(h))


class BenchModel(nn.Module):
    def __init__(self, num_layers=8, hidden=1024, intermediate=4096):
        super().__init__()
        self.layers = nn.ModuleList([Block(hidden, intermediate) for _ in range(num_layers)])
        self.head = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x).sum()


# ---- Newton-Schulz ----

def newton_schulz(G, steps=5):
    X = G / (G.norm() + 1e-7)
    X = X.half()
    for _ in range(steps):
        A = X @ X.T
        B = (1.5 * A - 0.5 * A @ A).to(X.dtype)
        X = B @ X
    return X


def bench_step(fn, warmup=3, repeat=10):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# ---- Baseline: standard FSDP2 + redundant NS ----

def run_baseline(model, mesh, device, hidden):
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    def step_fn():
        x = torch.randn(8, 32, hidden, device=device)
        loss = model(x)
        loss.backward()

        # Redundant NS: every rank all-gathers full grad and runs NS
        for module in [*model.layers, model]:
            state = module._get_fsdp_state()
            if state._fsdp_param_group is None:
                continue
            for fp in state._fsdp_param_group.fsdp_params:
                if fp.sharded_param.grad is None:
                    continue
                sharded_grad = fp.sharded_param.grad._local_tensor
                is_proj = len(fp._orig_size) == 2 and fp._orig_size.numel() >= 1024 * 1024
                if not is_proj:
                    # Non-proj params: simple SGD
                    fp.sharded_param._local_tensor.add_(sharded_grad, alpha=-0.01)
                    fp.sharded_param.grad = None
                    continue
                # Proj params: all-gather full grad + redundant NS
                grad_list = [torch.zeros_like(sharded_grad) for _ in range(mesh.size())]
                dist.all_gather(grad_list, sharded_grad)
                full_grad = torch.cat(grad_list, dim=0)[:fp._orig_size[0]]
                G = full_grad.view(fp._orig_size).float()
                G = G.view(G.shape[0], -1)
                update = newton_schulz(G)
                shard_size = sharded_grad.shape[0]
                shard_start = mesh.get_local_rank() * shard_size
                local_update = update[shard_start:shard_start + shard_size]
                fp.sharded_param._local_tensor.add_(
                    local_update.view(sharded_grad.shape).to(sharded_grad.dtype), alpha=-0.01
                )
                fp.sharded_param.grad = None

    return bench_step(step_fn)


# ---- DMuon ----

def run_dmuon(model, mesh, device, hidden):
    from dmuon import dedicate_params
    from dmuon.utils import get_owned_params

    dedicate_params(model, mesh, predicate=lambda n, p: "proj" in n)

    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    rank = mesh.get_local_rank()
    owned = get_owned_params(model, rank)

    def step_fn():
        x = torch.randn(8, 32, hidden, device=device)
        loss = model(x)
        loss.backward()

        # Owner-only NS (zero extra communication for optimizer)
        for dp in owned:
            if dp._reduced_grad is not None:
                G = dp._reduced_grad.view(dp._reduced_grad.shape[0], -1).float()
                update = newton_schulz(G)
                dp._owned_data.add_(
                    update.view(dp._owned_data.shape).to(dp._owned_data.dtype), alpha=-0.01
                )
                dp._reduced_grad = None

        # SGD for symmetric params
        for module in [*model.layers, model]:
            state = module._get_fsdp_state()
            if state._fsdp_param_group is None:
                continue
            for fp in state._fsdp_param_group.fsdp_params:
                if fp.sharded_param.grad is not None:
                    fp.sharded_param._local_tensor.add_(
                        fp.sharded_param.grad._local_tensor, alpha=-0.01
                    )
                    fp.sharded_param.grad = None

    return bench_step(step_fn)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world_size,))

    hidden = 1024
    intermediate = 4096
    num_layers = 8

    if rank == 0:
        print(f"E2E Benchmark: {world_size} GPUs")
        print(f"Model: {num_layers} layers, hidden={hidden}, intermediate={intermediate}")
        total_params = sum(p.numel() for p in BenchModel(num_layers, hidden, intermediate).parameters())
        print(f"Total params: {total_params / 1e6:.1f}M")
        print()

    # Baseline
    if rank == 0:
        print("Running baseline (FSDP2 + redundant NS)...")
    torch.manual_seed(42)
    model_baseline = BenchModel(num_layers, hidden, intermediate).to(device)
    baseline_ms = run_baseline(model_baseline, mesh, device, hidden)
    del model_baseline
    torch.cuda.empty_cache()

    dist.barrier()

    # DMuon
    if rank == 0:
        print("Running DMuon (dedicated ownership)...")
    torch.manual_seed(42)
    model_dmuon = BenchModel(num_layers, hidden, intermediate).to(device)
    dmuon_ms = run_dmuon(model_dmuon, mesh, device, hidden)
    del model_dmuon
    torch.cuda.empty_cache()

    if rank == 0:
        speedup = baseline_ms / dmuon_ms
        print()
        print(f"Results (median step time):")
        print(f"  Baseline (FSDP2 + redundant NS): {baseline_ms:.2f} ms")
        print(f"  DMuon (dedicated ownership):      {dmuon_ms:.2f} ms")
        print(f"  Speedup:                          {speedup:.2f}x")
        print()
        if speedup > 1:
            print(f"DMuon is {speedup:.2f}x faster ({(speedup-1)*100:.1f}% speedup)")
        else:
            print(f"DMuon is {1/speedup:.2f}x slower")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
