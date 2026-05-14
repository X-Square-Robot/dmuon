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

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch.nn as nn

from dmuon.utils import (
    broadcast_all_updates,
    broadcast_all_updates_async,
    prepare_group_muon_grads,
    prepare_muon_grads,
    wait_group_muon_grads,
)


class _RecordingGroup:
    """Just enough duck-typing for ``broadcast_all_updates_async``: it
    walks ``module._dedicated_state.group`` and calls
    ``replicate_broadcast_async``.  We record invocation order."""

    def __init__(self, tag: str, sink: List[str]):
        self.tag = tag
        self._debug_name = tag
        self._sink = sink
        self.comm_ctx = _CommCtx(post_forward_order=[])
        self._tp_gather_event = None
        self._muon_grad_ready_event = None

    def wait_for_reduce(self, stream=None):
        stream_tag = "prefetch" if stream is not None else "default"
        self._sink.append(f"reduce:{stream_tag}:{self.tag}")
        return None

    def tp_gather_grads(self, *, wait_current_stream=True):
        wait_tag = "wait" if wait_current_stream else "prefetch"
        self._sink.append(f"gather:{wait_tag}:{self.tag}")
        self._tp_gather_event = object()
        return self._tp_gather_event

    def wait_for_tp_gather(self):
        self._sink.append(f"ready:{self.tag}")
        self._tp_gather_event = None
        self._muon_grad_ready_event = None

    def replicate_broadcast_async(self):
        self._sink.append(self.tag)

    def replicate_broadcast_sync(self):
        self._sink.append(f"sync:{self.tag}")

    def wait_for_replicate_broadcast(self):
        self._sink.append(f"wait:{self.tag}")


class _State:
    def __init__(self, group):
        self.group = group


class _CommCtx:
    def __init__(self, post_forward_order):
        self.post_forward_order = post_forward_order
        self.reduce_stream = object()


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
            if order_groups is not None
            else []
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


def test_sync_post_step_uses_same_priority_order():
    """Sync mode should enter collective-bearing groups in the same order.

    The wait phase follows the same group list after all dispatches have been
    enqueued, matching production's dispatch-then-drain contract.
    """
    sink: List[str] = []
    model, _ = _make_model(
        tags=["A", "B", "C"],
        sink=sink,
        order_groups=["C", "A", "B"],
    )
    broadcast_all_updates(model)
    assert sink == [
        "sync:C",
        "sync:A",
        "sync:B",
        "wait:C",
        "wait:A",
        "wait:B",
    ]


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
        groups[0],
        groups[1],
        groups[0],
        groups[1],
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


def test_prepare_muon_grads_uses_forward_order_and_drains_events():
    sink: List[str] = []
    model, groups = _make_model(
        tags=["A", "B", "C"],
        sink=sink,
        order_groups=["C", "A", "B"],
    )
    prepare_muon_grads(model)
    assert sink == [
        "reduce:default:C",
        "gather:wait:C",
        "reduce:default:A",
        "gather:wait:A",
        "reduce:default:B",
        "gather:wait:B",
        "ready:C",
        "ready:A",
        "ready:B",
    ]
    assert all(g._tp_gather_event is None for g in groups)
    assert all(g._muon_grad_ready_event is None for g in groups)


def test_prepare_group_muon_grads_can_prefetch_on_reduce_stream():
    sink: List[str] = []
    model, groups = _make_model(["A"], sink)
    del model

    prepare_group_muon_grads(groups[0], use_reduce_stream=True)
    wait_group_muon_grads(groups[0])

    assert sink == ["reduce:prefetch:A", "gather:prefetch:A", "ready:A"]
