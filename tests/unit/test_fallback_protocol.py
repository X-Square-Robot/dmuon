"""Phase C.4 unit tests: fallback state machine on ``DedicatedParamGroup``.

Exercises the counters + flip semantics without real CUDA / NCCL — we
populate ``_last_replicate_wait_us`` directly and step the monitor.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import dmuon._backends.fsdp2.group as group_mod
from dmuon._backends.fsdp2.group import (
    REPLICATE_FALLBACK_CONSECUTIVE_STEPS,
    REPLICATE_WAIT_THRESHOLD_US,
)


class _StubGroup:
    """Minimal stand-in: copies only the fallback-relevant attributes from
    ``DedicatedParamGroup``.  Methods are imported unbound from the real
    class, so any divergence in logic is caught automatically."""

    _update_replicate_fallback = group_mod.DedicatedParamGroup._update_replicate_fallback
    reset_replicate_fallback = group_mod.DedicatedParamGroup.reset_replicate_fallback

    def __init__(self):
        self._replicate_sync_fallback = False
        self._replicate_slow_wait_count = 0
        self._last_replicate_wait_us = 0.0


def test_single_slow_wait_does_not_trip():
    g = _StubGroup()
    g._last_replicate_wait_us = REPLICATE_WAIT_THRESHOLD_US * 5
    g._update_replicate_fallback()
    assert g._replicate_sync_fallback is False
    assert g._replicate_slow_wait_count == 1


def test_consecutive_slow_trips_flag():
    g = _StubGroup()
    for _ in range(REPLICATE_FALLBACK_CONSECUTIVE_STEPS):
        g._last_replicate_wait_us = REPLICATE_WAIT_THRESHOLD_US * 5
        g._update_replicate_fallback()
    assert g._replicate_sync_fallback is True


def test_fast_wait_resets_counter():
    g = _StubGroup()
    # Two slow then one fast — counter must reset.
    g._last_replicate_wait_us = REPLICATE_WAIT_THRESHOLD_US * 5
    g._update_replicate_fallback()
    g._last_replicate_wait_us = REPLICATE_WAIT_THRESHOLD_US * 5
    g._update_replicate_fallback()
    g._last_replicate_wait_us = REPLICATE_WAIT_THRESHOLD_US * 0.5  # fast
    g._update_replicate_fallback()
    assert g._replicate_slow_wait_count == 0
    assert g._replicate_sync_fallback is False


def test_no_sample_is_no_op():
    """When the profile env var is off, ``_last_replicate_wait_us`` stays
    at 0.0.  The monitor must then be a pure no-op — neither advancing
    the counter nor flipping the flag."""
    g = _StubGroup()
    for _ in range(REPLICATE_FALLBACK_CONSECUTIVE_STEPS * 3):
        g._last_replicate_wait_us = 0.0
        g._update_replicate_fallback()
    assert g._replicate_slow_wait_count == 0
    assert g._replicate_sync_fallback is False


def test_already_tripped_short_circuits():
    """Monitor must not keep advancing the counter (or doing work) after
    the flag has been tripped.  Prevents unnecessary CPU overhead in the
    degraded steady state."""
    g = _StubGroup()
    g._replicate_sync_fallback = True
    g._replicate_slow_wait_count = 99
    g._last_replicate_wait_us = REPLICATE_WAIT_THRESHOLD_US * 10
    g._update_replicate_fallback()
    # Counter untouched; wait-us untouched (no drain).
    assert g._replicate_slow_wait_count == 99
    assert g._last_replicate_wait_us == REPLICATE_WAIT_THRESHOLD_US * 10


def test_reset_clears_state():
    g = _StubGroup()
    g._replicate_sync_fallback = True
    g._replicate_slow_wait_count = 5
    g._last_replicate_wait_us = 250.0
    g.reset_replicate_fallback()
    assert g._replicate_sync_fallback is False
    assert g._replicate_slow_wait_count == 0
    assert g._last_replicate_wait_us == 0.0


def test_sample_drain_prevents_stale_replay():
    """Between iterations the sample field is zeroed after consumption,
    so a step without a fresh wait cannot replay the previous value."""
    g = _StubGroup()
    g._last_replicate_wait_us = REPLICATE_WAIT_THRESHOLD_US * 5
    g._update_replicate_fallback()   # first slow wait recorded
    assert g._last_replicate_wait_us == 0.0  # drained
    g._update_replicate_fallback()   # no sample — must NOT increment
    assert g._replicate_slow_wait_count == 1


def test_flip_never_reverses_automatically():
    """Single-direction degrade: once fallback is True, fast waits do NOT
    flip it back to False.  Only ``reset_replicate_fallback`` clears it.
    (Avoids oscillation if IB traffic is bursty.)"""
    g = _StubGroup()
    for _ in range(REPLICATE_FALLBACK_CONSECUTIVE_STEPS):
        g._last_replicate_wait_us = REPLICATE_WAIT_THRESHOLD_US * 5
        g._update_replicate_fallback()
    assert g._replicate_sync_fallback is True
    for _ in range(10):
        g._last_replicate_wait_us = 1.0  # fast
        g._update_replicate_fallback()
    assert g._replicate_sync_fallback is True


def test_exactly_threshold_does_not_trip():
    """A wait sample equal to the threshold is NOT slow (strict >).  This
    avoids oscillation when waits cluster right around the threshold."""
    g = _StubGroup()
    for _ in range(REPLICATE_FALLBACK_CONSECUTIVE_STEPS * 2):
        g._last_replicate_wait_us = REPLICATE_WAIT_THRESHOLD_US  # exactly
        g._update_replicate_fallback()
    assert g._replicate_sync_fallback is False
