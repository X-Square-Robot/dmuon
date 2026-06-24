"""Process group ownership helpers for DMuon internals."""

from __future__ import annotations

import os
from typing import Optional

import torch.distributed as dist

_VALID_PROCESS_GROUP_POLICIES = {"isolated", "shared"}
_GROUP_CACHE: dict[tuple[str, tuple[int, ...]], dist.ProcessGroup] = {}
_ISOLATED_GROUP_IDS: set[int] = set()


def resolve_process_group_policy(policy: Optional[str] = None) -> str:
    """Return the normalized DMuon process-group policy.

    ``isolated`` makes DMuon clone its mesh process groups so trainer/logging
    collectives cannot interleave with DMuon's async post-step collectives on
    the same NCCL communicator. ``shared`` preserves the historical behavior
    of using the caller-provided DeviceMesh groups directly.
    """

    raw = policy
    if raw is None:
        raw = os.environ.get("DMUON_PROCESS_GROUP_POLICY", "isolated")
    normalized = str(raw).strip().lower()
    aliases = {
        "default": "isolated",
        "isolate": "isolated",
        "dmuon": "isolated",
        "reuse": "shared",
        "caller": "shared",
        "mesh": "shared",
        "none": "shared",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _VALID_PROCESS_GROUP_POLICIES:
        expected = ", ".join(sorted(_VALID_PROCESS_GROUP_POLICIES))
        raise ValueError(
            f"Unsupported DMuon process_group_policy={raw!r}; expected {expected}."
        )
    return normalized


def maybe_isolate_process_group(
    group: Optional[dist.ProcessGroup],
    *,
    policy: str,
    role: str,
) -> Optional[dist.ProcessGroup]:
    """Return a DMuon-owned clone of ``group`` when policy is ``isolated``.

    The current rank only knows its local DeviceMesh subgroup.  To create
    NCCL groups safely, every rank first exchanges its local subgroup ranks
    on WORLD, deduplicates the full partition for this role, then creates the
    whole non-overlapping partition in a deterministic order.  This keeps all
    ranks' ``new_group`` calls aligned while returning only the subgroup that
    contains the current rank.
    """

    if group is None or policy == "shared":
        return group
    if not (dist.is_available() and dist.is_initialized()):
        return group
    if dist.get_world_size() <= 1:
        return group

    local_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    cache_key = (str(role), local_ranks)
    cached = _GROUP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    gathered: list[object] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_ranks)
    rank_sets = sorted(
        {
            tuple(int(rank) for rank in ranks)
            for ranks in gathered
            if ranks is not None
        }
    )
    if local_ranks not in rank_sets:
        raise RuntimeError(
            f"DMuon isolated process group setup for role={role!r} did not "
            f"discover this rank's subgroup ranks={local_ranks}."
        )

    backend = dist.get_backend(group)
    current_group = None
    # Do not use new_subgroups_by_enumeration here: all ranks call new_group in
    # the same global order so DMuon owns the returned PG handles.  PyTorch/NCCL
    # may still derive the underlying communicator internally.  For diagnostics,
    # users can additionally enable the optional step-end isolated-PG fence in
    # dmuon.utils via DMUON_ISOLATED_PG_BARRIER=1.
    for ranks in rank_sets:
        created_group = dist.new_group(ranks=list(ranks), backend=backend)
        if ranks == local_ranks:
            current_group = created_group

    if current_group is None or current_group == dist.GroupMember.NON_GROUP_MEMBER:
        raise RuntimeError(
            f"DMuon isolated process group setup for role={role!r} returned "
            "a non-member group for the current rank."
        )

    _GROUP_CACHE[cache_key] = current_group
    _ISOLATED_GROUP_IDS.add(id(current_group))
    return current_group


def is_isolated_process_group(group: Optional[dist.ProcessGroup]) -> bool:
    """Return whether ``group`` is owned by DMuon's isolated PG policy."""

    if group is None or group == dist.GroupMember.NON_GROUP_MEMBER:
        return False
    return id(group) in _ISOLATED_GROUP_IDS
