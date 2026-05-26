"""Forward-prefetch state linking follows module order."""

from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

from dmuon.api import _link_forward_prefetch_states


def test_forward_prefetch_links_states_by_model_module_order():
    model = nn.Module()
    model.first = nn.Linear(2, 2)
    model.second = nn.Linear(2, 2)
    model.third = nn.Linear(2, 2)

    first = SimpleNamespace(
        module=model.first, group="first", _next_group=None, _next_groups=[]
    )
    second = SimpleNamespace(
        module=model.second, group="second", _next_group=None, _next_groups=[]
    )
    third = SimpleNamespace(
        module=model.third, group="third", _next_group=None, _next_groups=[]
    )
    comm_ctx = SimpleNamespace(all_states=[third, first, second])

    _link_forward_prefetch_states(model, [third, first, second], comm_ctx)

    assert first._next_group == "second"
    assert first._next_groups == ["second", "third"]
    assert second._next_group == "third"
    assert second._next_groups == ["third"]
    assert third._next_group is None
    assert third._next_groups == []
    assert comm_ctx.all_states == [first, second, third]
