"""Benchmark: broadcast vs all-gather latency on different data sizes.

Measures the core communication primitive performance to validate
that per-parameter broadcast is competitive with packed all-gather.

Run with: torchrun --nproc_per_node=8 benchmarks/bench_comm.py
"""

import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def bench_fn(fn, warmup=5, repeat=20):
    """Benchmark a function: return median latency in ms."""
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
    return times[len(times) // 2]  # median


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    group = dist.group.WORLD

    if rank == 0:
        print(f"Communication benchmark: {world_size} GPUs")
        print(f"{'Size':>12s} {'Elements':>12s} {'AllGather':>12s} {'Broadcast':>12s} {'N×BC':>12s} {'Ratio':>8s}")
        print(f"{'(MB)':>12s} {'':>12s} {'(ms)':>12s} {'(ms)':>12s} {'(ms)':>12s} {'AG/N×BC':>8s}")
        print("-" * 80)

    # Qwen2.5-7B relevant sizes (bf16)
    test_cases = [
        ("k_proj",     1_840_000,  "small"),   # 1.84M = 3.68 MB
        ("kv_packed",   3_680_000,  "small"),   # k+v merged = 7.36 MB
        ("q_proj",     12_850_000, "medium"),   # 12.85M = 25.7 MB
        ("gate_proj",  67_890_000, "large"),    # 67.89M = 135.8 MB
        ("layer_all", 233_050_000, "full"),     # full layer = 466 MB
    ]

    for name, numel, category in test_cases:
        size_mb = numel * 2 / 1e6  # bf16

        # --- All-gather: standard FSDP2 pattern ---
        # Each rank has shard = numel / world_size
        shard_numel = (numel + world_size - 1) // world_size
        shard = torch.randn(shard_numel, dtype=torch.bfloat16, device=device)
        ag_output = torch.empty(shard_numel * world_size, dtype=torch.bfloat16, device=device)

        def all_gather_fn():
            dist.all_gather_into_tensor(ag_output, shard, group=group)

        ag_ms = bench_fn(all_gather_fn)

        # --- Single broadcast: ownership pattern ---
        full = torch.randn(numel, dtype=torch.bfloat16, device=device)

        def broadcast_fn():
            dist.broadcast(full, src=0, group=group)

        bc_ms = bench_fn(broadcast_fn)

        # --- N concurrent broadcasts: simulating per-layer ownership ---
        # For a layer: ~5 ownership params with different owners
        n_owners = min(5, world_size)
        bc_bufs = [torch.randn(numel // n_owners, dtype=torch.bfloat16, device=device)
                   for _ in range(n_owners)]

        def n_broadcast_fn():
            works = []
            for i in range(n_owners):
                w = dist.broadcast(bc_bufs[i], src=i, group=group, async_op=True)
                works.append(w)
            for w in works:
                w.wait()

        nbc_ms = bench_fn(n_broadcast_fn)

        ratio = ag_ms / nbc_ms if nbc_ms > 0 else float('inf')

        if rank == 0:
            print(f"{size_mb:>10.1f}MB  {numel:>12,d}  {ag_ms:>10.3f}ms  {bc_ms:>10.3f}ms  {nbc_ms:>10.3f}ms  {ratio:>7.2f}x")

    if rank == 0:
        print()
        print("Legend:")
        print("  AllGather = dist.all_gather_into_tensor (standard FSDP2)")
        print("  Broadcast = single dist.broadcast from rank 0")
        print("  N×BC      = 5 concurrent broadcasts from different ranks (DMuon pattern)")
        print("  Ratio     = AllGather / N×BC (>1 means DMuon pattern is faster)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
