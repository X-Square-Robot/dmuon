"""Phase B.3 + B.4 unit tests.

Verifies two invariants that Phase B depends on:

1. **B.3 — Global-owner gate**: ``DedicatedParam.is_owner`` is True only on
   the rank whose ``(shard, replicate)`` coord matches ``owner_rank``.  When
   ``replicate_group is None`` (1D shard-only), the check collapses to the
   Phase A shard-only semantics.
2. **B.4 — Owner-only resources**: ``_owned_data`` is populated only on the
   global owner; replicate peers in the same shard column get an
   (initially-empty) receive buffer; ranks on other shard columns get None.

No distributed runtime required — we drive ``DedicatedParam.__init__`` with
a stub ``ProcessGroup`` covering the various (shard, replicate) positions.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import pytest
import torch
import torch.nn as nn

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from dmuon._backends.fsdp2.param import DedicatedParam
import dmuon._backends.fsdp2.param as param_module


class _StubGroup:
    """Minimal ProcessGroup stub: just advertises a rank within the group."""

    def __init__(self, rank: int):
        self._rank = rank

    def rank(self) -> int:
        return self._rank


class _StubDist:
    @staticmethod
    def get_global_rank(group, rank):
        # The unit test only cares about relative ranks; returning the local
        # rank keeps stubs lightweight.
        return rank


def _make_dp(
    shard_rank: int,
    replicate_rank: Optional[int],
    owner_shard: int,
    owner_replicate: int,
    shape=(8, 4),
) -> DedicatedParam:
    """Build a DedicatedParam on CPU with stubbed groups.

    ``replicate_rank=None`` ⇒ 1D shard-only mode (no replicate_group).
    """
    param = nn.Parameter(torch.randn(*shape))
    module = nn.Linear(shape[1], shape[0], bias=False)
    module.weight = param

    dp_group = _StubGroup(shard_rank)
    replicate_group = (
        _StubGroup(replicate_rank) if replicate_rank is not None else None
    )

    orig_dist = param_module.dist
    param_module.dist = _StubDist()
    try:
        dp = DedicatedParam(
            param=param,
            module=module,
            param_name="weight",
            owner_rank=(owner_shard, owner_replicate),
            dp_group=dp_group,
            device=torch.device("cpu"),
            compute_dtype=None,
            replicate_group=replicate_group,
        )
    finally:
        param_module.dist = orig_dist
    return dp


# --- B.3: is_owner gate ----------------------------------------------------


def test_is_owner_true_on_global_owner_hsdp():
    dp = _make_dp(shard_rank=2, replicate_rank=1,
                  owner_shard=2, owner_replicate=1)
    assert dp.is_owner is True


def test_is_owner_false_on_shard_peer_of_owner():
    # Same shard column but different replicate row — a shard peer, NOT the
    # global owner.  This is the receive rank of the post-step broadcast.
    dp = _make_dp(shard_rank=2, replicate_rank=0,
                  owner_shard=2, owner_replicate=1)
    assert dp.is_owner is False


def test_is_owner_false_on_replicate_peer_of_owner():
    dp = _make_dp(shard_rank=3, replicate_rank=1,
                  owner_shard=2, owner_replicate=1)
    assert dp.is_owner is False


def test_is_owner_false_on_distant_rank():
    dp = _make_dp(shard_rank=0, replicate_rank=0,
                  owner_shard=2, owner_replicate=1)
    assert dp.is_owner is False


def test_is_owner_collapses_to_shard_only_when_replicate_group_none():
    """Phase A compatibility: int owner_rank + no replicate_group ⇒ 1D."""
    dp = _make_dp(shard_rank=3, replicate_rank=None,
                  owner_shard=3, owner_replicate=0)
    assert dp.is_owner is True
    dp2 = _make_dp(shard_rank=2, replicate_rank=None,
                   owner_shard=3, owner_replicate=0)
    assert dp2.is_owner is False


# --- B.4: _owned_data allocation rule --------------------------------------


def test_owned_data_populated_on_global_owner_hsdp():
    dp = _make_dp(shard_rank=2, replicate_rank=1,
                  owner_shard=2, owner_replicate=1, shape=(8, 4))
    assert dp._owned_data is not None
    assert dp._owned_data.shape == (8, 4)


def test_owned_data_populated_on_shard_peer_hsdp():
    """Every rank in the owner's shard column (including non-global-owner
    replicate peers) needs a POPULATED ``_owned_data``.  Each replicate row
    is a full model instance whose shard-dim broadcast fires within its own
    shard_group; that broadcast's sender is that row's shard-owner, which
    means every shard peer of the owner must carry the param value."""
    dp = _make_dp(shard_rank=2, replicate_rank=0,
                  owner_shard=2, owner_replicate=1, shape=(8, 4))
    assert dp._owned_data is not None
    assert dp._owned_data.shape == (8, 4)
    # Value comes from the local Parameter, which at construction time holds
    # the post-``load_state_dict`` values — identical across ranks.
    assert dp.is_owner is False


def test_owned_data_none_on_foreign_shard_column_hsdp():
    """Ranks outside the owner's shard column do not allocate anything."""
    dp = _make_dp(shard_rank=0, replicate_rank=0,
                  owner_shard=2, owner_replicate=1)
    assert dp._owned_data is None


def test_owned_data_none_on_non_owner_in_shard_only_mode():
    """1D mode (Phase A path): only the global owner has ``_owned_data``;
    every other rank gets None (no replicate dim to broadcast into)."""
    dp = _make_dp(shard_rank=2, replicate_rank=None,
                  owner_shard=3, owner_replicate=0)
    assert dp._owned_data is None
    dp_owner = _make_dp(shard_rank=3, replicate_rank=None,
                        owner_shard=3, owner_replicate=0)
    assert dp_owner._owned_data is not None


# --- Cached global rank for Stage-2 reduce ---------------------------------


def test_owner_replicate_global_rank_cached_when_hsdp():
    dp = _make_dp(shard_rank=2, replicate_rank=0,
                  owner_shard=2, owner_replicate=1)
    # Stubbed ``get_global_rank`` returns the passed-in rank as-is.
    assert dp._owner_replicate_global_rank == 1


def test_owner_replicate_global_rank_none_when_shard_only():
    dp = _make_dp(shard_rank=2, replicate_rank=None,
                  owner_shard=2, owner_replicate=0)
    assert dp._owner_replicate_global_rank is None


# --- Parametric sweep over (G, R, owner) -----------------------------------


@pytest.mark.parametrize("G,R", [(2, 2), (4, 2), (4, 4)])
def test_exactly_one_global_owner_across_grid(G, R):
    """For every (G, R) grid and every owner coord, exactly one of the G*R
    simulated ranks reports ``is_owner=True``.  Catches any accidental
    broadening of the gate (e.g. OR instead of AND)."""
    for owner_shard in range(G):
        for owner_replicate in range(R):
            owners_seen = 0
            for s in range(G):
                for r in range(R):
                    dp = _make_dp(
                        shard_rank=s, replicate_rank=r,
                        owner_shard=owner_shard,
                        owner_replicate=owner_replicate,
                    )
                    if dp.is_owner:
                        owners_seen += 1
            assert owners_seen == 1, (
                f"Expected exactly one global owner in {G}x{R} grid "
                f"for owner=({owner_shard},{owner_replicate}); got {owners_seen}"
            )


@pytest.mark.parametrize("G,R", [(2, 2), (4, 2), (4, 4)])
def test_owned_data_allocated_exactly_on_shard_column(G, R):
    """Exactly R ranks (the owner's shard column) should have ``_owned_data``
    allocated; all other G*R - R ranks should have None."""
    owner = (1, 0)
    allocated = 0
    for s in range(G):
        for r in range(R):
            dp = _make_dp(
                shard_rank=s, replicate_rank=r,
                owner_shard=owner[0], owner_replicate=owner[1],
            )
            if dp._owned_data is not None:
                allocated += 1
                assert s == owner[0], (
                    f"rank ({s},{r}) has _owned_data but is not on the "
                    f"owner's shard column {owner[0]}"
                )
    assert allocated == R, (
        f"Expected R={R} ranks with _owned_data (one shard column); got {allocated}"
    )
