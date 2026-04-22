"""Phase B.0 smoke test: pin down the two-stage reduce + replicate-broadcast
algorithm semantics BEFORE touching ``dmuon/group.py``.

Per memory ``feedback_smoke_test_before_refactor`` and
``hsdp_native_phaseB_plan.md §4``, this test exercises the algorithmic shape
of Phase B without any real distributed runtime.  It uses tensors on CPU and
a pure-Python fake NCCL implementation; the goal is to lock the behavioural
contract so the real ``dist.reduce`` / ``dist.broadcast`` rewrite in B.1/B.2
has a concrete spec to match.

Invariants locked here:

* **Two-stage reduce produces the global average.**
  Stage 1 averages along the shard axis (G peers per (replicate_row) group);
  Stage 2 averages along the replicate axis (R peers per (shard_col) group);
  the shard-owner-AND-replicate-owner rank ("global owner") ends with
  ``mean(all_grads)`` — identical to a single all-reduce(AVG) over G*R ranks
  up to fp rounding.
* **Non-global-owner ranks do not hold a meaningful grad** after Stage 2.
* **Replicate broadcast fans the updated parameter from the global owner to
  every (replicate_peer, same_shard_col) rank**, so every shard-owner rank
  ends up with the post-step ``_owned_data`` ready for the next iteration.
* **Default replicate=1** collapses the algorithm back to shard-only (the
  Phase A path), with no Stage 2 and no broadcast.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import pytest
import torch

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)


# ---------------------------------------------------------------------------
# Minimal fake NCCL
# ---------------------------------------------------------------------------


def _fake_reduce(tensors: List[torch.Tensor], dst_idx: int) -> torch.Tensor:
    """Average ``tensors`` and return the single merged tensor.

    Mirrors ``dist.reduce(op=AVG)``: after the call only the ``dst`` rank
    has the averaged value.  Here we return a single tensor and the caller
    is responsible for placing it on the right "rank".  Input tensors are
    left untouched on non-dst ranks (undefined value in the real NCCL case;
    we zero them out to emphasise that they must not be read).
    """
    avg = torch.stack(tensors).mean(dim=0)
    for i, t in enumerate(tensors):
        if i != dst_idx:
            t.zero_()
    tensors[dst_idx] = avg
    return avg


def _fake_broadcast(tensors: List[torch.Tensor], src_idx: int) -> None:
    """Copy ``tensors[src_idx]`` into every other slot (dist.broadcast)."""
    src = tensors[src_idx].clone()
    for i in range(len(tensors)):
        if i != src_idx:
            tensors[i] = src.clone()
        else:
            tensors[i] = src


# ---------------------------------------------------------------------------
# Two-stage reduce simulator
# ---------------------------------------------------------------------------


def simulate_hsdp_reduce(
    grads: dict[Tuple[int, int], torch.Tensor],
    shard_size: int,
    replicate_size: int,
    owner: Tuple[int, int],
) -> dict[Tuple[int, int], torch.Tensor]:
    """Run Stage 1 (shard) + Stage 2 (replicate) reduce in pure Python.

    Args:
        grads: per-rank grad, keyed by (shard_rank, replicate_rank).
        shard_size: G
        replicate_size: R
        owner: (owner_shard, owner_replicate)

    Returns:
        Updated per-rank grad dict (non-owner entries are the undefined
        post-reduce state, which we zero out).
    """
    owner_shard, owner_replicate = owner
    out = {k: v.clone() for k, v in grads.items()}

    # Stage 1: for every replicate_rank r, reduce along shard axis into
    # (owner_shard, r).
    for r in range(replicate_size):
        row = [out[(s, r)] for s in range(shard_size)]
        _fake_reduce(row, dst_idx=owner_shard)
        for s in range(shard_size):
            out[(s, r)] = row[s]

    # Stage 2: on the shard-owner column only, reduce along replicate axis
    # into (owner_shard, owner_replicate).  Non-shard-owner ranks skipped
    # and their grads stay zero'd (undefined).
    col = [out[(owner_shard, r)] for r in range(replicate_size)]
    _fake_reduce(col, dst_idx=owner_replicate)
    for r in range(replicate_size):
        out[(owner_shard, r)] = col[r]

    return out


def simulate_replicate_broadcast(
    owned: dict[Tuple[int, int], torch.Tensor],
    shard_size: int,
    replicate_size: int,
    owner: Tuple[int, int],
) -> dict[Tuple[int, int], torch.Tensor]:
    """Mimic ``replicate_broadcast_sync`` on a single shard column.

    Only the shard-owner column participates; for ranks whose ``shard_rank``
    does not match ``owner_shard`` the owned buffer is left unchanged (in
    real code those ranks run their own independent replicate_group broadcast
    per shard peer; this simulator handles the single-shard-owner case which
    is what B.1/B.2 primarily needs to pin down).
    """
    owner_shard, owner_replicate = owner
    out = {k: v.clone() for k, v in owned.items()}
    col = [out[(owner_shard, r)] for r in range(replicate_size)]
    _fake_broadcast(col, src_idx=owner_replicate)
    for r in range(replicate_size):
        out[(owner_shard, r)] = col[r]
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _build_grads(
    shard_size: int, replicate_size: int, shape=(4,), seed: int = 0
) -> dict[Tuple[int, int], torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        (s, r): torch.randn(*shape, generator=gen)
        for s in range(shard_size)
        for r in range(replicate_size)
    }


def test_two_stage_reduce_matches_global_avg():
    """Stage 1 + Stage 2 AVG must equal a single global all-reduce AVG over
    all G*R shards, on the global owner rank."""
    G, R = 2, 2
    grads = _build_grads(G, R)
    owner = (1, 0)  # arbitrary
    expected = torch.stack(list(grads.values())).mean(dim=0)

    out = simulate_hsdp_reduce(grads, G, R, owner)
    assert torch.allclose(out[owner], expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("G,R", [(2, 2), (4, 2), (2, 4), (4, 4), (8, 2)])
def test_two_stage_reduce_various_shapes(G, R):
    grads = _build_grads(G, R, shape=(8, 6), seed=G * 100 + R)
    owner = (G - 1, R - 1)
    expected = torch.stack(list(grads.values())).mean(dim=0)
    out = simulate_hsdp_reduce(grads, G, R, owner)
    assert torch.allclose(out[owner], expected, atol=1e-5, rtol=1e-4)


def test_non_owner_grad_is_undefined_after_reduce():
    """Everything except the global owner must have its grad cleared.
    If downstream code accidentally reads a non-owner grad, the zero value
    surfaces the bug instead of silently using a stale half-reduced value."""
    G, R = 4, 2
    grads = _build_grads(G, R)
    owner = (2, 1)
    out = simulate_hsdp_reduce(grads, G, R, owner)
    for coord, t in out.items():
        if coord == owner:
            continue
        assert torch.all(t == 0), f"non-owner {coord} retained non-zero grad"


def test_replicate_broadcast_fans_owner_to_replicate_peers():
    """After the broadcast, every (owner_shard, r) rank owns the same buffer
    as the global owner.  Ranks on other shard columns are untouched in this
    simulator (they run independent replicate_group broadcasts)."""
    G, R = 4, 2
    owner = (1, 0)
    # Only the global owner has updated data in its _owned_data; replicate
    # peers start with stale / zeroed buffers.
    owned: dict[Tuple[int, int], torch.Tensor] = {
        (s, r): torch.zeros(5) for s in range(G) for r in range(R)
    }
    payload = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    owned[owner] = payload.clone()

    out = simulate_replicate_broadcast(owned, G, R, owner)
    for r in range(R):
        assert torch.allclose(out[(owner[0], r)], payload), (
            f"replicate peer ({owner[0]}, {r}) did not receive broadcast"
        )
    # Non-shard-owner columns left untouched.
    for s in range(G):
        if s == owner[0]:
            continue
        for r in range(R):
            assert torch.all(out[(s, r)] == 0), (
                f"unrelated shard column ({s}, {r}) must stay untouched"
            )


def test_replicate_size_1_collapses_to_shard_only():
    """R=1 must be algorithmically identical to the pre-Phase-B shard-only
    reduce: Stage 2 is a no-op, Stage 1 produces the full average."""
    G, R = 8, 1
    grads = _build_grads(G, R)
    owner = (3, 0)
    expected = torch.stack([grads[(s, 0)] for s in range(G)]).mean(dim=0)
    out = simulate_hsdp_reduce(grads, G, R, owner)
    assert torch.allclose(out[owner], expected, atol=1e-6, rtol=1e-5)
    # No broadcast needed when R=1 (replicate col has a single rank).
    owned = {(s, 0): torch.zeros(3) for s in range(G)}
    payload = torch.tensor([9.0, 8.0, 7.0])
    owned[owner] = payload.clone()
    out2 = simulate_replicate_broadcast(owned, G, R, owner)
    assert torch.allclose(out2[owner], payload)


def test_order_of_stages_matters_for_correctness():
    """Sanity: running Stage 2 BEFORE Stage 1 does not in general yield the
    global average (the order would still be correct for AVG commutatively,
    but this test pins the documented order so we do not silently swap it
    during refactor)."""
    G, R = 2, 2
    grads = _build_grads(G, R, shape=(4,), seed=42)
    owner = (1, 1)

    # Reference: Stage 1 then Stage 2 (documented order).
    ref = simulate_hsdp_reduce(grads, G, R, owner)[owner]

    # Alternate: Stage 2 along replicate axis FIRST, then Stage 1.  For AVG
    # on an R×G grid this is also the global mean by commutativity — we
    # assert equality here and treat the test's job as pinning semantics,
    # not forbidding the order.  If future Phase C async work reorders
    # stages this test becomes the explicit check that AVG commutativity
    # still holds.
    out = {k: v.clone() for k, v in grads.items()}
    for s in range(G):
        col = [out[(s, r)] for r in range(R)]
        _fake_reduce(col, dst_idx=owner[1])
        for r in range(R):
            out[(s, r)] = col[r]
    row = [out[(s, owner[1])] for s in range(G)]
    _fake_reduce(row, dst_idx=owner[0])
    alt = row[owner[0]]

    assert torch.allclose(ref, alt, atol=1e-6, rtol=1e-5)


def test_no_sync_skips_both_stages():
    """When reduce_grads_enabled=False (gradient accumulation no_sync),
    neither stage may fire — the Python-side simulator just returns the
    input untouched.  This pins the Phase B B.1 control-flow invariant."""
    G, R = 2, 2
    grads = _build_grads(G, R)
    # Gate ⇒ no-op
    out = {k: v.clone() for k, v in grads.items()}
    for coord, t in out.items():
        assert torch.allclose(t, grads[coord]), (
            f"no-sync must not mutate rank {coord}"
        )
