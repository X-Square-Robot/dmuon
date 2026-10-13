"""Exercise transferred gradient storage with real streams and allocator pressure.

Run on four GPUs with no arguments, or four native NPUs with ``--device npu``.
This is a bounded communication/storage regression, not a training benchmark.
It covers shard-only and 2x2 HSDP, reduce->reshard->wait, repeated waits,
no_sync with a final missing raw gradient, padding and empty AdamW shards.
"""

import argparse
import gc
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from dmuon._backends.fsdp2.group import DedicatedParamGroup
from dmuon._backends.fsdp2.param import DedicatedParam
from dmuon._core.comm import DedicatedCommContext
from dmuon._core.device import get_device_module


def make_group(device, shard_group, replicate_group):
    module = nn.Module()
    params = []
    owner = (1, 1 if replicate_group is not None else 0)
    for name, shape, route in (
        ("weight", (16, 32), "muon"),
        ("padded", (5,), "sharded_adamw"),
        ("empty_shard", (1,), "sharded_adamw"),
    ):
        parameter = nn.Parameter(torch.ones(shape, dtype=torch.bfloat16, device=device))
        module.register_parameter(name, parameter)
        params.append(
            DedicatedParam(
                parameter,
                module,
                name,
                owner,
                shard_group,
                device,
                grad_dtype=torch.float32,
                replicate_group=replicate_group,
                route_hint=route,
            )
        )
    context = DedicatedCommContext(device, replicate_group=replicate_group)
    group = DedicatedParamGroup(params, context)
    return module, group


def allocator_pressure(runtime, device, stream):
    # Same-sized blocks are allocated/freed repeatedly while another stream
    # may still read a transferred gradient. Python holds no gradient refs.
    with runtime.stream(stream):
        for _ in range(8):
            blocks = [torch.empty((16, 32), device=device) for _ in range(8)]
            for block in blocks:
                block.fill_(-4096)
            del block, blocks


def consumer_delay(device):
    # Queue independent work before the read, making premature allocator reuse
    # visible without relying on an unavailable device-specific sleep kernel.
    scratch = torch.ones(262144, device=device)
    for _ in range(8):
        scratch.mul_(1.0001)


def consume_and_release(group, runtime, device, consumer, pressure, expected, *, release):
    observations = []
    with runtime.stream(consumer):
        consumer_delay(device)
        for param in group.params:
            grad = param._sharded_adamw_grad if param.uses_sharded_adamw() else param._reduced_grad
            if not param.uses_sharded_adamw() and not param.is_owner:
                assert grad is None
                continue
            assert grad is not None and grad.dtype == torch.float32
            # An independent output consumes the source on the non-default
            # stream. Release the source before allocator pressure is queued.
            observations.append((param, grad + 0.0))
            del grad
        if release:
            for param in group.params:
                param._reduced_grad = None
                param._sharded_adamw_grad = None
            group._muon_grad_ready_refs = []
    allocator_pressure(runtime, device, pressure)
    runtime.current_stream().wait_stream(consumer)
    for param, observed in observations:
        target = torch.full_like(observed, expected)
        if param.uses_sharded_adamw():
            target[param._sharded_adamw_valid_numel :] = 0
        torch.testing.assert_close(observed, target, rtol=0, atol=0)


def run_case(device, shard_group, replicate_group, *, no_sync_final_none, iterations):
    runtime = get_device_module(device)
    consumer, pressure = runtime.Stream(device=device), runtime.Stream(device=device)
    rank = dist.get_rank()
    average_rank_value = (dist.get_world_size() + 1) / 2
    for _ in range(iterations):
        module, group = make_group(device, shard_group, replicate_group)
        expected = 0.0
        for micro in range(3):
            group.unshard()
            group.wait_for_unshard()
            group.reduce_grads_enabled = not no_sync_final_none
            for param in group.params:
                param._unsharded_param.grad = torch.full_like(
                    param._unsharded_param, float(rank + 1 + micro)
                )
            group.reduce_grads()
            group.reshard()
            allocator_pressure(runtime, device, pressure)
            group.wait_for_stage1_reduce()
            group.wait_for_reduce(stream=consumer)
            assert group.wait_for_reduce(stream=consumer) is None
            expected += average_rank_value + micro
            if not no_sync_final_none:
                consume_and_release(
                    group,
                    runtime,
                    device,
                    consumer,
                    pressure,
                    expected,
                    release=micro == 2,
                )
        if no_sync_final_none:
            group.unshard()
            group.wait_for_unshard()
            owner = group.params[0]
            stash_ptr = owner._accumulated_grad.data_ptr()
            for param in group.params:
                assert param._unsharded_param.grad is None
            group.reduce_grads_enabled = True
            group.reduce_grads()
            assert all(param._accumulated_grad is None for param in group.params)
            assert group._pending_reduce[0][0].data_ptr() == stash_ptr
            group.reshard()
            allocator_pressure(runtime, device, pressure)
            group.wait_for_stage1_reduce()
            group.wait_for_reduce(stream=consumer)
            assert group.wait_for_reduce(stream=consumer) is None
            consume_and_release(
                group,
                runtime,
                device,
                consumer,
                pressure,
                expected,
                release=True,
            )
        runtime.synchronize()
        del module, group
        gc.collect()
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "npu"), default="cuda")
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 16:
        parser.error("--iterations must be between 1 and 16")
    runtime = get_device_module(args.device)
    if args.device == "npu":
        assert "torch_npu.contrib.transfer_to_npu" not in sys.modules, (
            "native test forbids transfer shim"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    runtime.set_device(local_rank)
    device = torch.device(args.device, local_rank)
    dist.init_process_group("nccl" if args.device == "cuda" else "hccl")
    assert dist.get_world_size() == 4, "requires four ranks"
    mesh = init_device_mesh(args.device, (2, 2), mesh_dim_names=("replicate", "shard"))
    for topology, shard_group, replicate_group in (
        ("shard", dist.group.WORLD, None),
        ("hsdp", mesh["shard"].get_group(), mesh["replicate"].get_group()),
    ):
        for no_sync_final_none in (False, True):
            run_case(
                device,
                shard_group,
                replicate_group,
                no_sync_final_none=no_sync_final_none,
                iterations=args.iterations,
            )
            if dist.get_rank() == 0:
                print(
                    f"PASS storage ownership: {topology}, no_sync-final-none={no_sync_final_none}",
                    flush=True,
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
