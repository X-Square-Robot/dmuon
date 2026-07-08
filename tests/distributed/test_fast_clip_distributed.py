"""Distributed liveness + correctness check for segmented fast clip.

Reproduces the scenario that used to hang: a non-finite value in a
``reduce=False`` bucket on only one rank.  With the three-stage design the
collective count is identical on every rank, so the clip must complete and
produce the correct all-reduced per-bucket norms.

Usage (CPU, no GPU needed)::

    torchrun --nproc_per_node=2 tests/distributed/test_fast_clip_distributed.py

Exit code is non-zero on any assertion failure; a hang (the old bug) shows up
as a torchrun timeout.
"""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import math

import torch
import torch.distributed as dist

from dmuon.fast_clip import GradClipBucket, clip_grad_norm_buckets_


def run() -> None:
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # reduce=True buckets: each rank contributes a known local square; the
    # segment-local norm after reduction is sqrt(sum_r local_sq_r).
    muon = torch.tensor([float(rank + 1), 0.0])       # local sq = (rank+1)^2
    adamw = torch.tensor([0.0, float(rank + 2)])       # local sq = (rank+2)^2

    # reduce=False bucket stays rank-local; rank 0 injects a non-finite value.
    regular = torch.tensor([3.0, 4.0])
    if rank == 0:
        regular = torch.tensor([float("inf"), 4.0])

    result = clip_grad_norm_buckets_(
        (
            GradClipBucket("regular", [regular], reduce=False),
            GradClipBucket("muon", [muon], reduce=True),
            GradClipBucket("adamw", [adamw], reduce=True),
        ),
        max_norm=1.0,
    )
    stats = result.stats_by_name

    expected_muon = math.sqrt(sum((r + 1) ** 2 for r in range(world_size)))
    expected_adamw = math.sqrt(sum((r + 2) ** 2 for r in range(world_size)))

    assert abs(stats["muon"].total_norm - expected_muon) < 1e-4, (
        rank, stats["muon"].total_norm, expected_muon,
    )
    assert abs(stats["adamw"].total_norm - expected_adamw) < 1e-4, (
        rank, stats["adamw"].total_norm, expected_adamw,
    )
    # regular is local: only rank 0 saw the inf.
    assert stats["regular"].found_inf is (rank == 0), (rank, stats["regular"].found_inf)

    dist.barrier()
    if rank == 0:
        print(
            f"PASS: fast-clip distributed (world_size={world_size}) — "
            f"muon={expected_muon:.4f} adamw={expected_adamw:.4f}, no hang"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
