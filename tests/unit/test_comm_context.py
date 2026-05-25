from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from dmuon._core.comm import DedicatedCommContext


def test_comm_context_separates_replicate_reduce_and_publish_streams(monkeypatch):
    """HSDP Stage-2 reduce should not share the post-step publish stream."""

    created = []

    def fake_stream(*, device=None, priority=0):
        stream = SimpleNamespace(device=device, priority=priority, index=len(created))
        created.append(stream)
        return stream

    monkeypatch.setattr(torch.cuda, "Stream", fake_stream)

    ctx = DedicatedCommContext(torch.device("cuda", 0), replicate_group=object())

    assert ctx.replicate_reduce_stream is not ctx.replicate_broadcast_stream
    assert ctx.replicate_reduce_stream.priority == 0
    assert ctx.replicate_broadcast_stream.priority == 0
    assert ctx.broadcast_stream.priority == -1
    assert ctx.reduce_stream.priority == -1
