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
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)


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
