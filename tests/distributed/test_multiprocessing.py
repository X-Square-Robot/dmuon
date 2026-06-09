"""Multi-GPU tests for DedicatedParamGroup communication.

Run with: torchrun --nproc_per_node=8 tests/test_multiprocessing.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

# Add parent dir to path for import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dmuon._core.comm import DedicatedCommContext
from dmuon._backends.fsdp2.group import DedicatedParamGroup
from dmuon._backends.fsdp2.param import DedicatedParam


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def make_comm_ctx(device):
    return DedicatedCommContext(device)


def test_single_param_broadcast(rank, world_size, device, dp_group):
    """Test: owner broadcasts full param, all ranks receive same data."""
    comm_ctx = make_comm_ctx(device)

    owner_rank = 2
    param = nn.Parameter(torch.randn(64, 128, device=device))
    if rank == owner_rank:
        param.data.fill_(42.0)

    module = nn.Linear(128, 64, bias=False, device=device)
    module.weight = param

    d_param = DedicatedParam(
        param=param, module=module, param_name="weight",
        owner_rank=owner_rank, dp_group=dp_group, device=device,
    )

    if rank == owner_rank:
        assert d_param._owned_data is not None
        assert d_param._owned_data.shape == (64, 128)
        assert d_param._owned_data.mean().item() == 42.0
    else:
        assert d_param._owned_data is None

    param_group = DedicatedParamGroup([d_param], comm_ctx)
    param_group.unshard()
    param_group.wait_for_unshard()

    assert d_param._unsharded_param is not None
    assert d_param._unsharded_param.shape == (64, 128)
    assert d_param._unsharded_param.data.mean().item() == 42.0, (
        f"Rank {rank}: expected 42.0, got {d_param._unsharded_param.data.mean().item()}"
    )

    d_param.reshard()
    assert d_param._unsharded_param is not None
    assert not d_param._is_unsharded
    assert module.weight is d_param._placeholder

    torch.cuda.synchronize()
    log(rank, "PASSED: test_single_param_broadcast")


def test_group_packed_broadcast(rank, world_size, device, dp_group):
    """Test: DedicatedParamGroup packs same-owner params and broadcasts correctly."""
    comm_ctx = make_comm_ctx(device)

    modules = []
    d_params = []
    expected_values = {0: 10.0, 1: 20.0, 2: 30.0}

    for i, (owner, val) in enumerate(expected_values.items()):
        m = nn.Linear(32, 16, bias=False, device=device)
        if rank == owner:
            m.weight.data.fill_(val)
        d_param = DedicatedParam(
            param=m.weight, module=m, param_name="weight",
            owner_rank=owner, dp_group=dp_group, device=device,
        )
        modules.append(m)
        d_params.append(d_param)

    param_group = DedicatedParamGroup(d_params, comm_ctx)

    param_group.unshard()
    param_group.wait_for_unshard()

    for i, (owner, val) in enumerate(expected_values.items()):
        actual = d_params[i]._unsharded_param.data.mean().item()
        assert abs(actual - val) < 1e-5, (
            f"Rank {rank}, param {i} (owner={owner}): expected {val}, got {actual}"
        )

    param_group.reshard()

    torch.cuda.synchronize()
    log(rank, "PASSED: test_group_packed_broadcast")


def test_gradient_reduce(rank, world_size, device, dp_group):
    """Test: gradients are correctly reduced to owner."""
    comm_ctx = make_comm_ctx(device)

    owner_rank = 3
    m = nn.Linear(32, 16, bias=False, device=device)
    m.weight.data.fill_(1.0)

    d_param = DedicatedParam(
        param=m.weight, module=m, param_name="weight",
        owner_rank=owner_rank, dp_group=dp_group, device=device,
    )

    param_group = DedicatedParamGroup([d_param], comm_ctx)

    param_group.unshard()
    param_group.wait_for_unshard()

    d_param._unsharded_param.grad = torch.full_like(d_param._unsharded_param.data, float(rank + 1))

    param_group.reduce_grads()
    param_group.wait_for_reduce()

    if rank == owner_rank:
        assert d_param._reduced_grad is not None
        expected_avg = sum(range(1, world_size + 1)) / world_size
        actual = d_param._reduced_grad.mean().item()
        assert abs(actual - expected_avg) < 1e-4, (
            f"Owner rank {rank}: expected avg grad {expected_avg}, got {actual}"
        )

    param_group.reshard()

    torch.cuda.synchronize()
    log(rank, "PASSED: test_gradient_reduce")


def test_forward_backward_e2e(rank, world_size, device, dp_group):
    """Test: full forward + backward with DedicatedParamGroup."""
    comm_ctx = make_comm_ctx(device)

    linear1 = nn.Linear(32, 64, bias=False, device=device)
    linear2 = nn.Linear(64, 16, bias=False, device=device)

    torch.manual_seed(42)
    linear1.weight.data = torch.randn(64, 32, device=device)
    linear2.weight.data = torch.randn(16, 64, device=device)

    d_param1 = DedicatedParam(
        param=linear1.weight, module=linear1, param_name="weight",
        owner_rank=0, dp_group=dp_group, device=device,
    )
    d_param2 = DedicatedParam(
        param=linear2.weight, module=linear2, param_name="weight",
        owner_rank=1, dp_group=dp_group, device=device,
    )

    param_group = DedicatedParamGroup([d_param1, d_param2], comm_ctx)

    param_group.unshard()
    param_group.wait_for_unshard()
    x = torch.randn(4, 32, device=device)
    y = linear2(linear1(x))
    loss = y.sum()

    loss.backward()

    param_group.reduce_grads()
    param_group.wait_for_reduce()

    if rank == 0:
        assert d_param1._reduced_grad is not None, "Rank 0 should have grad for linear1"
        assert d_param1._reduced_grad.shape == (64, 32)
    if rank == 1:
        assert d_param2._reduced_grad is not None, "Rank 1 should have grad for linear2"
        assert d_param2._reduced_grad.shape == (16, 64)

    param_group.reshard()

    torch.cuda.synchronize()
    log(rank, "PASSED: test_forward_backward_e2e")


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dp_group = dist.group.WORLD

    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    tests = {
        "broadcast": test_single_param_broadcast,
        "packed": test_group_packed_broadcast,
        "reduce": test_gradient_reduce,
        "e2e": test_forward_backward_e2e,
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
