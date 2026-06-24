"""Distributed smoke test for DMuon process_group_policy.

Run:

    torchrun --nproc_per_node=4 tests/distributed/test_process_group_policy.py
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dmuon._core.process_groups import (  # noqa: E402
    is_isolated_process_group,
    maybe_isolate_process_group,
    resolve_process_group_policy,
)


def _init() -> tuple[int, int]:
    if not dist.is_initialized():
        dist.init_process_group("gloo")
    return dist.get_rank(), dist.get_world_size()


def _destroy() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _subgroup_partitions(world_size: int) -> list[list[int]]:
    if world_size < 2 or world_size % 2 != 0:
        raise RuntimeError("test_process_group_policy requires an even world size >= 2")
    midpoint = world_size // 2
    return [list(range(0, midpoint)), list(range(midpoint, world_size))]


def main() -> None:
    rank, world_size = _init()
    partitions = _subgroup_partitions(world_size)
    source_group, _ = dist.new_subgroups_by_enumeration(partitions, backend="gloo")

    assert resolve_process_group_policy(None) == "isolated"
    assert resolve_process_group_policy("shared") == "shared"
    assert resolve_process_group_policy("reuse") == "shared"

    shared_group = maybe_isolate_process_group(
        source_group,
        policy="shared",
        role="test.dp",
    )
    assert shared_group is source_group

    isolated_group = maybe_isolate_process_group(
        source_group,
        policy="isolated",
        role="test.dp",
    )
    assert isolated_group is not source_group
    assert not is_isolated_process_group(source_group)
    assert is_isolated_process_group(isolated_group)
    assert tuple(dist.get_process_group_ranks(isolated_group)) == tuple(
        dist.get_process_group_ranks(source_group)
    )
    assert isolated_group.rank() == source_group.rank()
    assert isolated_group.size() == source_group.size()

    value = torch.tensor([float(rank + 1)])
    dist.all_reduce(value, op=dist.ReduceOp.SUM, group=isolated_group)
    expected = sum(r + 1 for r in dist.get_process_group_ranks(source_group))
    assert value.item() == expected

    if rank == 0:
        print("PASSED: test_process_group_policy", flush=True)
    _destroy()


if __name__ == "__main__":
    main()
