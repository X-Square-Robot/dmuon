"""Phase C.5 unit test: ``broadcast_all_updates_async`` dispatches in the
order recorded by ``comm_ctx.post_forward_order`` (with model-walk
fallback for groups not yet seen).

Mirrors FSDP2's use of ``post_forward_order`` for backward prefetch
(``_fsdp_param_group.py:469-474``) — Phase C repurposes the same record
for **forward** dispatch priority, so earlier layers' replicate
broadcasts finish first and their ``_pre_forward_wait`` unblocks
sooner.
"""

from __future__ import annotations

import os
import sys
from typing import List

import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch.nn as nn

from dmuon.utils import broadcast_all_updates_async


class _RecordingGroup:
    """Just enough duck-typing for ``broadcast_all_updates_async``: it
    walks ``module._dedicated_state.group`` and calls
    ``replicate_broadcast_async``.  We record invocation order."""

    def __init__(self, tag: str, sink: List[str]):
        self.tag = tag
        self._sink = sink

    def replicate_broadcast_async(self):
        self._sink.append(self.tag)


class _State:
    def __init__(self, group):
        self.group = group


class _CommCtx:
    def __init__(self, post_forward_order):
        self.post_forward_order = post_forward_order


def _make_model(tags, sink, order_groups=None):
    """Build a small nn.Module whose submodules carry _dedicated_state."""
    model = nn.Module()
    groups = []
    for tag in tags:
        g = _RecordingGroup(tag, sink)
        groups.append(g)
        sub = nn.Module()
        sub._dedicated_state = _State(g)
        setattr(model, f"layer_{tag}", sub)
    model._dedicated_comm_ctx = _CommCtx(
        post_forward_order=(
            [groups[tags.index(t)] for t in order_groups]
            if order_groups is not None else []
        )
    )
    return model, groups


def test_first_epoch_uses_model_walk_order():
    """Empty ``post_forward_order`` → dispatch follows ``model.modules()``
    iteration order (insertion order for named submodules)."""
    sink: List[str] = []
    model, _ = _make_model(["A", "B", "C"], sink)
    broadcast_all_updates_async(model)
    assert sink == ["A", "B", "C"]


def test_post_forward_order_takes_priority():
    """After one forward the order should follow the recorded sequence
    — even if it differs from module insertion order."""
    sink: List[str] = []
    model, _ = _make_model(
        tags=["A", "B", "C"],
        sink=sink,
        order_groups=["C", "A", "B"],  # a non-insertion order
    )
    broadcast_all_updates_async(model)
    assert sink == ["C", "A", "B"]


def test_unseen_groups_appended_in_model_order():
    """When ``post_forward_order`` is partial (e.g. some layers skipped
    in the last forward), the missing groups must still fire — appended
    in model-walk order so nothing is silently dropped."""
    sink: List[str] = []
    model, _ = _make_model(
        tags=["A", "B", "C", "D"],
        sink=sink,
        order_groups=["B", "A"],  # C, D not seen
    )
    broadcast_all_updates_async(model)
    assert sink == ["B", "A", "C", "D"]


def test_duplicate_entries_collapsed_to_first_occurrence():
    """A group can appear in ``post_forward_order`` multiple times (e.g.
    activation checkpoint recompute).  We dedupe so the broadcast fires
    exactly once per step."""
    sink: List[str] = []
    model, groups = _make_model(["A", "B"], sink)
    # Insert ``A`` twice in the order.
    model._dedicated_comm_ctx.post_forward_order = [
        groups[0], groups[1], groups[0], groups[1],
    ]
    broadcast_all_updates_async(model)
    assert sink == ["A", "B"]


def test_no_comm_ctx_falls_through_to_model_walk():
    sink: List[str] = []
    model, _ = _make_model(["A", "B"], sink)
    # Simulate a model built without ``dedicate_params`` (no comm_ctx).
    del model._dedicated_comm_ctx
    broadcast_all_updates_async(model)
    assert sink == ["A", "B"]
