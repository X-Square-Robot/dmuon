"""Multi-GPU tests for DedicatedParamGroup communication.

Run with: torchrun --nproc_per_node=8 tests/test_multiprocessing.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

# Add parent dir to path for import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmuon.group import DedicatedParamGroup
from dmuon.param import DedicatedParam


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def cleanup():
    dist.destroy_process_group()


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def test_single_param_broadcast():
    """Test: owner broadcasts full param, all ranks receive same data."""
    rank, world_size = setup()
    device = torch.device("cuda", rank)
    group = dist.group.WORLD

    # Create a parameter owned by rank 2
    owner_rank = 2
    param = nn.Parameter(torch.randn(64, 128, device=device))
    # All ranks should have the same initial value for verification
    # Set a known value on owner
    if rank == owner_rank:
        param.data.fill_(42.0)

    module = nn.Linear(128, 64, bias=False, device=device)
    module.weight = param

    d_param = DedicatedParam(
        param=param,
        module=module,
        param_name="weight",
        owner_rank=owner_rank,
        dp_group=group,
        device=device,
    )

    # After init: owner has data, non-owner has empty
    if rank == owner_rank:
        assert d_param._owned_data is not None
        assert d_param._owned_data.shape == (64, 128)
        assert d_param._owned_data.mean().item() == 42.0
    else:
        assert d_param._owned_data is None

    # Unshard (broadcast)
    work = d_param.alloc_and_broadcast(async_op=True)
    if work is not None:
        work.wait()
    d_param.finish_unshard()

    # All ranks should now have the same param value
    assert d_param._unsharded_param is not None
    assert d_param._unsharded_param.shape == (64, 128)
    assert d_param._unsharded_param.data.mean().item() == 42.0, (
        f"Rank {rank}: expected 42.0, got {d_param._unsharded_param.data.mean().item()}"
    )

    # Reshard
    d_param.reshard()
    assert d_param._unsharded_param is None

    log(rank, "PASSED: test_single_param_broadcast")
    cleanup()


def test_group_packed_broadcast():
    """Test: DedicatedParamGroup packs same-owner params and broadcasts correctly."""
    rank, world_size = setup()
    device = torch.device("cuda", rank)
    group = dist.group.WORLD

    # Create 3 params with different owners
    modules = []
    d_params = []
    expected_values = {0: 10.0, 1: 20.0, 2: 30.0}

    for i, (owner, val) in enumerate(expected_values.items()):
        m = nn.Linear(32, 16, bias=False, device=device)
        if rank == owner:
            m.weight.data.fill_(val)
        d_param = DedicatedParam(
            param=m.weight,
            module=m,
            param_name="weight",
            owner_rank=owner,
            dp_group=group,
            device=device,
        )
        modules.append(m)
        d_params.append(d_param)

    # Create group
    param_group = DedicatedParamGroup(d_params)

    # Unshard
    param_group.unshard()

    # Verify all ranks have correct values
    for i, (owner, val) in enumerate(expected_values.items()):
        actual = d_params[i]._unsharded_param.data.mean().item()
        assert abs(actual - val) < 1e-5, (
            f"Rank {rank}, param {i} (owner={owner}): expected {val}, got {actual}"
        )

    # Reshard
    param_group.reshard()

    log(rank, "PASSED: test_group_packed_broadcast")
    cleanup()


def test_gradient_reduce():
    """Test: gradients are correctly reduced to owner."""
    rank, world_size = setup()
    device = torch.device("cuda", rank)
    group = dist.group.WORLD

    owner_rank = 3
    m = nn.Linear(32, 16, bias=False, device=device)
    m.weight.data.fill_(1.0)

    d_param = DedicatedParam(
        param=m.weight,
        module=m,
        param_name="weight",
        owner_rank=owner_rank,
        dp_group=group,
        device=device,
    )

    param_group = DedicatedParamGroup([d_param])

    # Unshard
    param_group.unshard()

    # Simulate gradient: each rank has grad = rank + 1
    d_param._unsharded_param.grad = torch.full_like(d_param._unsharded_param.data, float(rank + 1))

    # Reduce grads (AVG)
    param_group.reduce_grads()

    # Owner should have avg gradient = (1+2+...+8)/8 = 4.5
    if rank == owner_rank:
        assert d_param._reduced_grad is not None
        expected_avg = sum(range(1, world_size + 1)) / world_size
        actual = d_param._reduced_grad.mean().item()
        assert abs(actual - expected_avg) < 1e-4, (
            f"Owner rank {rank}: expected avg grad {expected_avg}, got {actual}"
        )
    else:
        # Non-owner should not have reduced grad (or it's None)
        pass

    # Reshard
    param_group.reshard()

    log(rank, "PASSED: test_gradient_reduce")
    cleanup()


def test_forward_backward_e2e():
    """Test: full forward + backward with DedicatedParamGroup."""
    rank, world_size = setup()
    device = torch.device("cuda", rank)
    group = dist.group.WORLD

    # Simple model: 2 linears
    linear1 = nn.Linear(32, 64, bias=False, device=device)
    linear2 = nn.Linear(64, 16, bias=False, device=device)

    # Make all ranks start with same weights
    torch.manual_seed(42)
    linear1.weight.data = torch.randn(64, 32, device=device)
    linear2.weight.data = torch.randn(16, 64, device=device)

    # Dedicate linear1 to rank 0, linear2 to rank 1
    d_param1 = DedicatedParam(
        param=linear1.weight,
        module=linear1,
        param_name="weight",
        owner_rank=0,
        dp_group=group,
        device=device,
    )
    d_param2 = DedicatedParam(
        param=linear2.weight,
        module=linear2,
        param_name="weight",
        owner_rank=1,
        dp_group=group,
        device=device,
    )

    param_group = DedicatedParamGroup([d_param1, d_param2])

    # Forward
    param_group.unshard()
    x = torch.randn(4, 32, device=device)
    y = linear2(linear1(x))
    loss = y.sum()

    # Backward
    loss.backward()

    # Reduce grads
    param_group.reduce_grads()

    # Verify owner has grads
    if rank == 0:
        assert d_param1._reduced_grad is not None, "Rank 0 should have grad for linear1"
        assert d_param1._reduced_grad.shape == (64, 32)
    if rank == 1:
        assert d_param2._reduced_grad is not None, "Rank 1 should have grad for linear2"
        assert d_param2._reduced_grad.shape == (16, 64)

    param_group.reshard()

    log(rank, "PASSED: test_forward_backward_e2e")
    cleanup()


if __name__ == "__main__":
    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    tests = {
        "broadcast": test_single_param_broadcast,
        "packed": test_group_packed_broadcast,
        "reduce": test_gradient_reduce,
        "e2e": test_forward_backward_e2e,
    }

    if test_name == "all":
        for name, fn in tests.items():
            print(f"\n{'=' * 60}")
            print(f"Running: {name}")
            print(f"{'=' * 60}")
            fn()
    elif test_name in tests:
        tests[test_name]()
    else:
        print(f"Unknown test: {test_name}. Available: {list(tests.keys())}")
        sys.exit(1)
