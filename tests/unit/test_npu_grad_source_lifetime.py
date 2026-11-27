"""CPU metadata witnesses for the NPU reduction stream's tensor ownership."""

from types import SimpleNamespace

import pytest

from dmuon._backends.fsdp2 import group


class Gradient:
    device = SimpleNamespace(type="npu", index=0)
    dtype = None

    def __init__(self, name, events, *, contiguous=True):
        self.name = name
        self.events = events
        self.contiguous = contiguous

    @property
    def data(self):
        return self

    def is_contiguous(self):
        return self.contiguous

    def record_stream(self, stream):
        self.events.append((self.name, stream))


@pytest.mark.parametrize("kind", ["dtensor", "strided", "logical_matrix", "contiguous"])
def test_localization_records_original_and_result_on_npu_reduce_stream(monkeypatch, kind):
    events = []
    stream = object()
    monkeypatch.setattr(
        group, "get_device_module", lambda device: SimpleNamespace(current_stream=lambda _: stream)
    )
    source = Gradient("original", events, contiguous=kind in ("dtensor", "contiguous"))
    raw = Gradient("dtensor", events) if kind == "dtensor" else source
    if kind == "dtensor":
        raw._local_tensor = source
    result = source if kind == "contiguous" else Gradient("localized", events)

    def localize(tensor):
        assert tensor is raw
        events.append(("localize", stream))
        return result

    dp = SimpleNamespace(
        _unsharded_param=SimpleNamespace(grad=raw), _grad_dtype=None, local_grad_for_reduce=localize
    )
    if kind == "logical_matrix":
        dp._unsharded_param.grad = None
        dp._layout_grad = raw
    assert group._get_grad_for_reduce(dp) is result
    if kind == "contiguous":
        assert events == [("localize", stream), ("original", stream)]
    else:
        assert events == [("original", stream), ("localize", stream), ("localized", stream)]
