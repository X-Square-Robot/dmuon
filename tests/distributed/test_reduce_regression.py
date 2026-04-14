"""Multi-GPU regression tests for gradient reduce + reshard ordering bug.

The bug: after reduce_grads(), calling reshard() (which sets _unsharded_param=None)
before wait_for_reduce() would lose the gradient reference for single-param reduces.
The fix saves the grad tensor in _pending_reduce before reshard can clear it.

Run with: torchrun --nproc_per_node=4 tests/distributed/test_reduce_regression.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

# Add parent dir to path for import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dmuon.comm import DedicatedCommContext
from dmuon.group import DedicatedParamGroup
from dmuon.param import DedicatedParam


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def make_comm_ctx(device):
    return DedicatedCommContext(device)


def test_single_param_reduce_reshard_then_wait(rank, world_size, device, dp_group):
    """Regression test: reduce_grads() -> reshard() -> wait_for_reduce() must not lose grad.

    This is the exact sequence that triggered the bug. With a single param (no packing),
    the reduce op referenced _unsharded_param.grad directly. If reshard() cleared
    _unsharded_param before wait_for_reduce(), the grad reference was lost.
    """
    comm_ctx = make_comm_ctx(device)

    owner_rank = 1
    m = nn.Linear(32, 16, bias=False, device=device)
    m.weight.data.fill_(1.0)

    d_param = DedicatedParam(
        param=m.weight, module=m, param_name="weight",
        owner_rank=owner_rank, dp_group=dp_group, device=device,
    )

    param_group = DedicatedParamGroup([d_param], comm_ctx)

    param_group.unshard()
    param_group.wait_for_unshard()

    # Each rank sets grad to its rank + 1
    d_param._unsharded_param.grad = torch.full_like(
        d_param._unsharded_param.data, float(rank + 1)
    )

    # BUG TRIGGER ORDER: reduce -> reshard -> wait (reshard before wait)
    param_group.reduce_grads()
    param_group.reshard()
    param_group.wait_for_reduce()

    if rank == owner_rank:
        assert d_param._reduced_grad is not None, (
            f"Owner rank {rank}: _reduced_grad is None after reduce->reshard->wait "
            "(regression: grad reference lost when reshard cleared _unsharded_param)"
        )
        expected_avg = sum(range(1, world_size + 1)) / world_size  # (1+2+3+4)/4 = 2.5
        actual = d_param._reduced_grad.mean().item()
        assert abs(actual - expected_avg) < 1e-4, (
            f"Owner rank {rank}: expected avg grad {expected_avg}, got {actual}"
        )
    else:
        assert d_param._reduced_grad is None, (
            f"Non-owner rank {rank}: _reduced_grad should be None, "
            f"got tensor with mean {d_param._reduced_grad.mean().item()}"
        )

    torch.cuda.synchronize()
    log(rank, "PASSED: test_single_param_reduce_reshard_then_wait")


def test_packed_reduce_reshard_then_wait(rank, world_size, device, dp_group):
    """Test reduce->reshard->wait with packed (multi-param, same owner) reduce path.

    When multiple params share the same owner, they are packed into a single buffer
    for the reduce op. This test verifies that path also survives reshard before wait.
    """
    comm_ctx = make_comm_ctx(device)

    owner_rank = 2
    sizes = [(16, 32), (32, 16), (8, 64)]
    grad_base_values = [1.0, 10.0, 100.0]

    modules = []
    d_params = []
    for size in sizes:
        m = nn.Linear(size[1], size[0], bias=False, device=device)
        m.weight.data.fill_(1.0)
        d_param = DedicatedParam(
            param=m.weight, module=m, param_name="weight",
            owner_rank=owner_rank, dp_group=dp_group, device=device,
        )
        modules.append(m)
        d_params.append(d_param)

    param_group = DedicatedParamGroup(d_params, comm_ctx)

    param_group.unshard()
    param_group.wait_for_unshard()

    # Set different grad values per param: rank + base_value
    for i, d_param in enumerate(d_params):
        d_param._unsharded_param.grad = torch.full_like(
            d_param._unsharded_param.data, float(rank + grad_base_values[i])
        )

    # BUG TRIGGER ORDER: reduce -> reshard -> wait
    param_group.reduce_grads()
    param_group.reshard()
    param_group.wait_for_reduce()

    if rank == owner_rank:
        rank_avg = sum(range(world_size)) / world_size  # avg of rank values: (0+1+2+3)/4 = 1.5
        for i, d_param in enumerate(d_params):
            assert d_param._reduced_grad is not None, (
                f"Owner rank {rank}, param {i}: _reduced_grad is None after "
                "packed reduce->reshard->wait"
            )
            assert d_param._reduced_grad.shape == sizes[i], (
                f"Owner rank {rank}, param {i}: expected shape {sizes[i]}, "
                f"got {d_param._reduced_grad.shape}"
            )
            expected_avg = rank_avg + grad_base_values[i]
            actual = d_param._reduced_grad.mean().item()
            assert abs(actual - expected_avg) < 1e-3, (
                f"Owner rank {rank}, param {i}: expected avg grad {expected_avg}, "
                f"got {actual}"
            )
    else:
        for i, d_param in enumerate(d_params):
            assert d_param._reduced_grad is None, (
                f"Non-owner rank {rank}, param {i}: _reduced_grad should be None"
            )

    torch.cuda.synchronize()
    log(rank, "PASSED: test_packed_reduce_reshard_then_wait")


def test_stream_event_sync(rank, world_size, device, dp_group):
    """Test that CUDA event-based stream synchronization in wait_for_reduce works.

    Introduces heavy compute on the default stream before setting grads to create
    stream divergence, then verifies the reduced grad values are correct (not garbage
    from incomplete synchronization).
    """
    comm_ctx = make_comm_ctx(device)

    owner_rank = 0
    m = nn.Linear(32, 16, bias=False, device=device)
    m.weight.data.fill_(1.0)

    d_param = DedicatedParam(
        param=m.weight, module=m, param_name="weight",
        owner_rank=owner_rank, dp_group=dp_group, device=device,
    )

    param_group = DedicatedParamGroup([d_param], comm_ctx)

    param_group.unshard()
    param_group.wait_for_unshard()

    # Heavy compute on default stream to create stream divergence
    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)
    for _ in range(5):
        a = torch.matmul(a, b)

    # Set grad after heavy compute
    d_param._unsharded_param.grad = torch.full_like(
        d_param._unsharded_param.data, float(rank + 1)
    )

    # reduce -> reshard -> wait
    param_group.reduce_grads()
    param_group.reshard()
    param_group.wait_for_reduce()

    if rank == owner_rank:
        assert d_param._reduced_grad is not None, (
            f"Owner rank {rank}: _reduced_grad is None after stream-divergent "
            "reduce->reshard->wait (possible stream sync issue)"
        )
        expected_avg = sum(range(1, world_size + 1)) / world_size
        actual = d_param._reduced_grad.mean().item()
        assert abs(actual - expected_avg) < 1e-4, (
            f"Owner rank {rank}: expected avg grad {expected_avg}, got {actual} "
            "(possible stream sync issue: value may be garbage from incomplete sync)"
        )
    else:
        assert d_param._reduced_grad is None, (
            f"Non-owner rank {rank}: _reduced_grad should be None"
        )

    torch.cuda.synchronize()
    log(rank, "PASSED: test_stream_event_sync")


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dp_group = dist.group.WORLD

    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    tests = {
        "single_reduce_reshard": test_single_param_reduce_reshard_then_wait,
        "packed_reduce_reshard": test_packed_reduce_reshard_then_wait,
        "stream_sync": test_stream_event_sync,
    }

    if test_name == "all":
        for name, fn in tests.items():
            log(rank, f"\n{'=' * 60}")
            log(rank, f"Running: {name}")
            log(rank, f"{'=' * 60}")
            dist.barrier()
            fn(rank, world_size, device, dp_group)
            dist.barrier()
    elif test_name in tests:
        tests[test_name](rank, world_size, device, dp_group)
    else:
        if rank == 0:
            print(f"Unknown test: {test_name}. Available: {list(tests.keys())}")
        sys.exit(1)

    dist.destroy_process_group()
