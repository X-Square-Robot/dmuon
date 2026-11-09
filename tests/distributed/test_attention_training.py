"""torchrun --standalone --nproc_per_node=2 ... [--cpu] [--hsdp]."""

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.helpers.attention_training import exercise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--hsdp", action="store_true")
    parser.add_argument("--output", default="attention-training.json")
    args = parser.parse_args()
    device = "cpu" if args.cpu else f"cuda:{os.environ['LOCAL_RANK']}"
    if args.cpu:
        from tests.distributed.test_matrix_layout_owner import synchronous_cpu_runtime

        synchronous_cpu_runtime()
    else:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("gloo" if args.cpu else "nccl")
    world, rank = dist.get_world_size(), dist.get_rank()
    if args.hsdp:
        mesh = init_device_mesh(
            "cpu" if args.cpu else "cuda", (2, world // 2), mesh_dim_names=("replicate", "shard")
        )
        shard, replicate = mesh["shard"], mesh["replicate"]
    else:
        shard = init_device_mesh("cpu" if args.cpu else "cuda", (world,), mesh_dim_names=("shard",))
        replicate = None
    results = []
    for kind in ("mla", "dsa", "csa2", "kda", "moba"):
        for mode in ("whole", "components", "per_head"):
            for reshard, recompute, dtype in (
                (True, False, torch.float32),
                (False, False, torch.float32),
                (True, True, torch.float32),
                (True, False, torch.bfloat16),
            ):
                result = exercise(
                    kind,
                    mode,
                    dtype,
                    reshard,
                    recompute,
                    shard,
                    device=device,
                    replicate=replicate,
                    world=world,
                    rank=rank,
                )
                results.append(result)
                if rank == 0:
                    print(json.dumps(result), flush=True)
                    Path(args.output).write_text(json.dumps(results, indent=2))
                dist.barrier()
    for kind in ("mla", "dsa", "csa2", "kda", "moba"):
        result = exercise(
            kind,
            "per_head",
            torch.float32,
            True,
            False,
            shard,
            device=device,
            replicate=replicate,
            world=world,
            rank=rank,
            native_ns=True,
        )
        results.append(result)
        if rank == 0:
            print(json.dumps(result), flush=True)
            Path(args.output).write_text(json.dumps(results, indent=2))
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
