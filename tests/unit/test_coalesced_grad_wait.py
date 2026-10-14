"""Coalesced gradient waits use an explicit group's device runtime."""

from types import SimpleNamespace

import pytest

import dmuon.utils as utils


def group_with_stream(stream):
    return SimpleNamespace(
        comm_ctx=SimpleNamespace(reduce_stream=stream),
        _muon_grad_ready_event=object(),
        _muon_grad_ready_refs=[object()],
        _tp_gather_event=None,
    )


def test_coalesced_wait_uses_first_group_runtime_and_clears_all_groups(monkeypatch):
    recorded, waited, selected = [], [], []
    event = object()

    def record_event():
        recorded.append(event)
        return event

    stream = SimpleNamespace(record_event=record_event)
    groups = [group_with_stream(stream), group_with_stream(stream)]
    compute_stream = SimpleNamespace(wait_event=waited.append)

    def device_module(group):
        selected.append(group)
        return SimpleNamespace(current_stream=lambda: compute_stream)

    monkeypatch.setattr(utils, "_group_device_module", device_module)
    assert utils._coalesced_wait_prepared_muon_grads(groups)
    assert recorded == waited == [event]
    assert len(selected) == 1 and selected[0] is groups[0]
    for group in groups:
        assert group._muon_grad_ready_event is None
        assert group._muon_grad_ready_refs == []


def test_coalesced_wait_empty_groups_does_not_select_a_runtime(monkeypatch):
    def unexpected_runtime(group):
        raise AssertionError("Empty groups must not select a device runtime")

    monkeypatch.setattr(utils, "_group_device_module", unexpected_runtime)
    assert utils._coalesced_wait_prepared_muon_grads([])


@pytest.mark.parametrize("second_stream", [None, "different"])
def test_coalesced_wait_incompatible_streams_leave_groups_for_fallback(monkeypatch, second_stream):
    def unexpected_event():
        raise AssertionError("Incompatible streams must not record an aggregate event")

    first_stream = SimpleNamespace(record_event=unexpected_event)
    other_stream = SimpleNamespace(record_event=unexpected_event) if second_stream else None
    groups = [group_with_stream(first_stream), group_with_stream(other_stream)]
    ready_events = [group._muon_grad_ready_event for group in groups]

    def unexpected_runtime(group):
        raise AssertionError("Incompatible streams must defer the device wait to the caller")

    monkeypatch.setattr(utils, "_group_device_module", unexpected_runtime)
    assert not utils._coalesced_wait_prepared_muon_grads(groups)
    for group, ready_event in zip(groups, ready_events):
        assert group._muon_grad_ready_event is ready_event
        assert len(group._muon_grad_ready_refs) == 1
