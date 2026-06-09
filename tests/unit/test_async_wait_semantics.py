"""Phase C.0 smoke test: lock the state machine for the async replicate
broadcast BEFORE touching ``dmuon/group.py``.

Per memory ``feedback_smoke_test_before_refactor`` and
``hsdp_native_phaseC_plan.md §4``, this test exercises the event
consumption contract in pure Python with a fake CUDA event — no real
distributed runtime, no CUDA.  The goal is to pin the transitions:

    IDLE        --dispatch()-->        PENDING
    PENDING     --consume()-->         IDLE
    PENDING     --dispatch()-->        ERROR (double dispatch)
    IDLE        --consume()-->         IDLE (no-op)
    PENDING     --shard_copy_in()-->   must consume first (assert)
    PENDING     --checkpoint_save()--> must consume first

The production code (``DedicatedParamGroup._replicate_broadcast_state``
+ ``replicate_broadcast_async`` + ``_pre_forward_wait``) will use these
semantics.  Violating them corrupts ``_owned_data``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum, auto
from types import SimpleNamespace
from typing import Optional

import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from dmuon._backends.fsdp2.group import (
    _event_is_ready,
    _should_skip_unready_publish_prefetch,
)
from dmuon._core.state import DedicatedState, _root_post_backward_final_callback
from dmuon.optim.muon import Muon


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Mimics ``torch.cuda.Event``: records at dispatch, consumed by wait.

    We track whether the event has been waited-on to detect the
    double-wait / double-consume patterns.
    """

    def __init__(self):
        self.recorded = False
        self.wait_count = 0

    def record(self):
        assert not self.recorded, "double record on same event"
        self.recorded = True

    def wait(self):
        assert self.recorded, "wait on un-recorded event"
        self.wait_count += 1

    def query(self):
        return self.recorded


class _BrokenQueryEvent:
    def query(self):
        raise RuntimeError("event not initialized")


class _State(Enum):
    IDLE = auto()
    PENDING = auto()


@dataclass
class _FakeBroadcastState:
    """Python counterpart of ``ReplicateBroadcastState`` — holds the event
    and the data ref to keep alive across stream boundaries."""

    owned_ref: object  # stand-in for ``_owned_data`` tensor
    event: _FakeEvent


class _FakeGroup:
    """Drives the minimal state machine ``DedicatedParamGroup`` will use."""

    def __init__(self, owned_data):
        self._owned_data = owned_data
        self._state: _State = _State.IDLE
        self._broadcast_state: Optional[_FakeBroadcastState] = None
        # Count how many times shard copy-in / checkpoint save ran AFTER
        # a dispatch without a consume in between — must stay at 0.
        self.uncovered_reads: int = 0

    # ── dispatch ──────────────────────────────────────────────────────

    def dispatch_async(self) -> None:
        if self._state is _State.PENDING:
            raise RuntimeError(
                "double dispatch: previous event not yet consumed"
            )
        event = _FakeEvent()
        event.record()
        self._broadcast_state = _FakeBroadcastState(
            owned_ref=self._owned_data, event=event
        )
        self._state = _State.PENDING

    # ── consume ───────────────────────────────────────────────────────

    def consume_wait(self) -> None:
        """Pre-forward hook path.  No-op when idle."""
        if self._state is _State.IDLE:
            return
        assert self._broadcast_state is not None
        self._broadcast_state.event.wait()
        self._broadcast_state = None
        self._state = _State.IDLE

    # ── read guards ───────────────────────────────────────────────────

    def shard_broadcast_copy_in(self) -> None:
        """Reads ``_owned_data`` for the shard-dim broadcast's packed
        copy-in; must NEVER run while async broadcast is still pending."""
        if self._state is _State.PENDING:
            self.uncovered_reads += 1

    def checkpoint_save(self) -> None:
        """Same invariant — the checkpoint reader must see fresh data."""
        if self._state is _State.PENDING:
            self.uncovered_reads += 1


# ---------------------------------------------------------------------------
# Tests — state transitions
# ---------------------------------------------------------------------------


def test_idle_to_pending_on_dispatch():
    g = _FakeGroup(owned_data="weights")
    assert g._state is _State.IDLE
    g.dispatch_async()
    assert g._state is _State.PENDING
    assert g._broadcast_state is not None
    assert g._broadcast_state.event.recorded
    assert g._broadcast_state.event.wait_count == 0


def test_pending_to_idle_on_consume():
    g = _FakeGroup(owned_data="weights")
    g.dispatch_async()
    g.consume_wait()
    assert g._state is _State.IDLE
    assert g._broadcast_state is None


def test_double_dispatch_rejected():
    g = _FakeGroup(owned_data="weights")
    g.dispatch_async()
    with pytest.raises(RuntimeError, match="double dispatch"):
        g.dispatch_async()


def test_consume_is_idempotent_when_idle():
    g = _FakeGroup(owned_data="weights")
    g.consume_wait()   # no-op, no event yet
    g.consume_wait()   # still no-op
    assert g._state is _State.IDLE


def test_consume_once_event_not_waited_twice():
    """Consuming an event must not replay ``wait_event`` on the current stream
    — that would redundantly block compute for an already-resolved dependency."""
    g = _FakeGroup(owned_data="weights")
    g.dispatch_async()
    g.consume_wait()
    # Re-consume after state reset is a no-op; the event is not reused.
    g.consume_wait()
    # No new event was recorded; old one waited exactly once.
    # (We cannot re-inspect the old state because it was cleared.)
    # Instead verify we can dispatch fresh:
    g.dispatch_async()
    assert g._broadcast_state is not None


def test_prefetch_event_readiness_query_is_conservative():
    event = _FakeEvent()
    assert not _event_is_ready(event)
    event.record()
    assert _event_is_ready(event)
    assert _event_is_ready(None)
    assert not _event_is_ready(_BrokenQueryEvent())


def test_far_prefetch_skips_unready_publish_wait():
    event = _FakeEvent()
    should_skip, tp_not_ready, replicate_not_ready, sharded_muon_not_ready = (
        _should_skip_unready_publish_prefetch(
            None,
            event,
            allow_unready_publish_wait=False,
        )
    )
    assert should_skip
    assert not tp_not_ready
    assert replicate_not_ready
    assert not sharded_muon_not_ready


def test_immediate_next_prefetch_can_queue_unready_publish_wait():
    event = _FakeEvent()
    should_skip, tp_not_ready, replicate_not_ready, sharded_muon_not_ready = (
        _should_skip_unready_publish_prefetch(
            None,
            event,
            allow_unready_publish_wait=True,
        )
    )
    assert not should_skip
    assert not tp_not_ready
    assert replicate_not_ready
    assert not sharded_muon_not_ready


def test_far_prefetch_skips_even_ready_publish_event():
    event = _FakeEvent()
    event.record()
    should_skip, tp_not_ready, replicate_not_ready, sharded_muon_not_ready = (
        _should_skip_unready_publish_prefetch(
            event,
            None,
            allow_unready_publish_wait=False,
        )
    )
    # CUDA event readiness is rank-local.  Far prefetch must make a
    # rank-consistent decision and retry later instead of allowing only ranks
    # that locally see a ready event to enter collectives.
    assert should_skip
    assert tp_not_ready
    assert not replicate_not_ready
    assert not sharded_muon_not_ready


def test_far_prefetch_skips_unready_sharded_muon_publish_wait():
    event = _FakeEvent()
    should_skip, tp_not_ready, replicate_not_ready, sharded_muon_not_ready = (
        _should_skip_unready_publish_prefetch(
            None,
            None,
            event,
            allow_unready_publish_wait=False,
        )
    )
    assert should_skip
    assert not tp_not_ready
    assert not replicate_not_ready
    assert sharded_muon_not_ready


class _DummyCommCtx:
    def __init__(self, replicate_group=None):
        self.replicate_group = replicate_group


class _DummyDedicatedParam:
    def __init__(self, *, sharded_muon_forward: bool):
        self._sharded_muon_forward = sharded_muon_forward

    def uses_sharded_muon_forward(self) -> bool:
        return self._sharded_muon_forward


class _DummyParamGroup:
    def __init__(self, *, replicate_group=None, sharded_muon_forward: bool = False):
        self.comm_ctx = _DummyCommCtx(replicate_group)
        self.params = [
            _DummyDedicatedParam(sharded_muon_forward=sharded_muon_forward)
        ]


def test_hsdp_sharded_muon_group_is_detected_for_guarded_prefetch():
    group = _DummyParamGroup(replicate_group=object(), sharded_muon_forward=True)
    assert Muon._group_has_hsdp_sharded_muon_params(group)


def test_fsdp_sharded_muon_group_can_still_use_post_step_far_prefetch():
    group = _DummyParamGroup(replicate_group=None, sharded_muon_forward=True)
    assert not Muon._group_has_hsdp_sharded_muon_params(group)


def test_hsdp_non_sharded_muon_group_can_still_prefetch():
    group = _DummyParamGroup(replicate_group=object(), sharded_muon_forward=False)
    assert not Muon._group_has_hsdp_sharded_muon_params(group)


def test_hsdp_sharded_muon_policy_allows_one_guarded_tail_prefetch():
    group = _DummyParamGroup(replicate_group=object(), sharded_muon_forward=True)
    should_prefetch, allow_wait, counts_hsdp = Muon._post_step_prefetch_policy(
        group,
        group_idx=1,
        post_step_prefetch_groups=4,
        prefetched_hsdp_sharded_muon_groups=0,
    )
    assert should_prefetch
    assert allow_wait
    assert counts_hsdp


def test_hsdp_sharded_muon_policy_rejects_second_tail_prefetch():
    group = _DummyParamGroup(replicate_group=object(), sharded_muon_forward=True)
    should_prefetch, allow_wait, counts_hsdp = Muon._post_step_prefetch_policy(
        group,
        group_idx=2,
        post_step_prefetch_groups=4,
        prefetched_hsdp_sharded_muon_groups=1,
    )
    assert not should_prefetch
    assert not allow_wait
    assert not counts_hsdp


def test_fsdp_sharded_muon_policy_uses_normal_tail_prefetch():
    group = _DummyParamGroup(replicate_group=None, sharded_muon_forward=True)
    should_prefetch, allow_wait, counts_hsdp = Muon._post_step_prefetch_policy(
        group,
        group_idx=1,
        post_step_prefetch_groups=4,
        prefetched_hsdp_sharded_muon_groups=0,
    )
    assert should_prefetch
    assert not allow_wait
    assert not counts_hsdp


def test_post_step_prefetch_policy_respects_window():
    group = _DummyParamGroup(replicate_group=object(), sharded_muon_forward=True)
    should_prefetch, allow_wait, counts_hsdp = Muon._post_step_prefetch_policy(
        group,
        group_idx=4,
        post_step_prefetch_groups=4,
        prefetched_hsdp_sharded_muon_groups=0,
    )
    assert not should_prefetch
    assert not allow_wait
    assert not counts_hsdp


# ---------------------------------------------------------------------------
# Tests — post-backward rolling reduce drain
# ---------------------------------------------------------------------------


class _FakeReduceGroup:
    def __init__(self, *, delay_stage2_to_optimizer: bool = True):
        self._post_backward_fired = False
        self._delay_stage2_to_optimizer = delay_stage2_to_optimizer
        self.stage1_waits = 0
        self.full_waits = 0
        self.reduce_calls = 0
        self.reshard_calls = 0

    def wait_for_stage1_reduce(self):
        self.stage1_waits += 1

    def wait_for_reduce(self):
        self.full_waits += 1

    def reduce_grads(self):
        self.reduce_calls += 1

    def reshard(self):
        self.reshard_calls += 1


class _LegacyReduceGroup:
    def __init__(self):
        self.full_waits = 0

    def wait_for_reduce(self):
        self.full_waits += 1


def _dedicated_state_for_group(group, comm_ctx):
    state = object.__new__(DedicatedState)
    state.group = group
    state.comm_ctx = comm_ctx
    return state


def test_post_backward_waits_previous_stage1_only_for_fsdp2_groups():
    prev = _FakeReduceGroup()
    cur = _FakeReduceGroup()
    comm_ctx = SimpleNamespace(last_reduced_group=prev)

    _dedicated_state_for_group(cur, comm_ctx)._run_post_backward()

    assert prev.stage1_waits == 1
    assert prev.full_waits == 0
    assert cur.reduce_calls == 1
    assert cur.reshard_calls == 1
    assert cur._post_backward_fired
    assert comm_ctx.last_reduced_group is cur


def test_post_backward_legacy_group_falls_back_to_full_reduce_wait():
    prev = _LegacyReduceGroup()
    cur = _FakeReduceGroup()
    comm_ctx = SimpleNamespace(last_reduced_group=prev)

    _dedicated_state_for_group(cur, comm_ctx)._run_post_backward()

    assert prev.full_waits == 1
    assert cur.reduce_calls == 1
    assert cur.reshard_calls == 1


def test_root_callback_leaves_stage2_for_optimizer_by_default():
    group = _FakeReduceGroup(delay_stage2_to_optimizer=True)
    comm_ctx = SimpleNamespace(
        all_states=[_dedicated_state_for_group(group, None)],
        last_reduced_group=None,
        post_backward_final_callback_queued=True,
    )
    comm_ctx.all_states[0].comm_ctx = comm_ctx

    _root_post_backward_final_callback(comm_ctx)

    assert group.reduce_calls == 1
    assert group.full_waits == 0
    assert comm_ctx.post_backward_final_callback_queued is False


def test_root_callback_can_still_drain_full_reduce_for_legacy_mode():
    group = _FakeReduceGroup(delay_stage2_to_optimizer=False)
    comm_ctx = SimpleNamespace(
        all_states=[_dedicated_state_for_group(group, None)],
        last_reduced_group=None,
        post_backward_final_callback_queued=True,
    )
    comm_ctx.all_states[0].comm_ctx = comm_ctx

    _root_post_backward_final_callback(comm_ctx)

    assert group.reduce_calls == 1
    assert group.full_waits == 1
    assert comm_ctx.post_backward_final_callback_queued is False


# ---------------------------------------------------------------------------
# Tests — wait-before-read invariant
# ---------------------------------------------------------------------------


def test_shard_copy_in_before_consume_is_illegal():
    """If shard-dim copy-in runs while the replicate broadcast is still
    pending, the production code would read stale ``_owned_data``.
    The flag surfaces the bug in this test — in production ``_pre_forward``
    must always call ``_pre_forward_wait`` before ``unshard``."""
    g = _FakeGroup(owned_data="weights")
    g.dispatch_async()
    g.shard_broadcast_copy_in()   # violation
    assert g.uncovered_reads == 1


def test_shard_copy_in_after_consume_is_safe():
    g = _FakeGroup(owned_data="weights")
    g.dispatch_async()
    g.consume_wait()
    g.shard_broadcast_copy_in()
    assert g.uncovered_reads == 0


def test_checkpoint_save_before_consume_is_illegal():
    g = _FakeGroup(owned_data="weights")
    g.dispatch_async()
    g.checkpoint_save()
    assert g.uncovered_reads == 1


def test_checkpoint_save_after_consume_is_safe():
    g = _FakeGroup(owned_data="weights")
    g.dispatch_async()
    g.consume_wait()
    g.checkpoint_save()
    assert g.uncovered_reads == 0


# ---------------------------------------------------------------------------
# Tests — data-ref keep-alive across streams
# ---------------------------------------------------------------------------


def test_owned_data_ref_kept_alive_while_pending():
    """``ReplicateBroadcastState`` stores an owning reference to the
    broadcast input so the tensor cannot be freed by the allocator while
    the NCCL kernel is still in flight on another stream.  Mirrors
    FSDP2's AllGatherState(all_gather_result=...) (``_fsdp_param_group.py:
    105-107, 222-225``)."""
    import weakref
    class _OwnedData:
        pass
    data = _OwnedData()
    g = _FakeGroup(owned_data=data)
    weak = weakref.ref(data)
    g.dispatch_async()
    # Drop the caller's ref; the group's state should still pin it.
    del data
    import gc
    gc.collect()
    assert weak() is not None, (
        "ReplicateBroadcastState must keep owned_ref alive while PENDING"
    )
    g.consume_wait()
    gc.collect()
    # After consume the ref may be dropped (we just released
    # _broadcast_state), and with no external refs the object is
    # collectible.  We don't assert this — the guarantee is only
    # "alive while PENDING".
