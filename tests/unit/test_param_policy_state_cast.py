from types import SimpleNamespace

import torch
import torch.nn as nn

from dmuon._core.state import DedicatedState
from dmuon.api import _attach_dedicated_state
from dmuon.utils import _isolated_pg_barrier_enabled


class CaptureDTypeModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_dtype = None

    def forward(self, x):
        self.input_dtype = x.dtype
        return x


class FakePolicyGroup:
    def __init__(self, *, name: str = "group", events: list[str] | None = None) -> None:
        self.name = name
        self.events = events
        self._param_dtype = torch.float32
        self._grad_dtype = torch.float32
        self._output_dtype = torch.bfloat16
        self._cast_forward_inputs = True
        self._post_backward_fired = False
        self.recorded_post_forward = False

    def _pre_forward_wait(self):
        if self.events is not None:
            self.events.append(f"{self.name}.pre_forward_wait")
        pass

    def unshard(self, **_kwargs):
        if self.events is not None:
            self.events.append(f"{self.name}.unshard")
        pass

    def wait_for_unshard(self):
        if self.events is not None:
            self.events.append(f"{self.name}.wait_for_unshard")
        pass

    def _backward_prefetch(self):
        pass

    def reshard(self):
        if self.events is not None:
            self.events.append(f"{self.name}.reshard")
        pass

    def _record_post_forward(self):
        if self.events is not None:
            self.events.append(f"{self.name}.record_post_forward")
        self.recorded_post_forward = True


def test_dedicated_state_casts_forward_inputs_and_outputs() -> None:
    module = CaptureDTypeModule()
    group = FakePolicyGroup()
    comm_ctx = SimpleNamespace(
        all_states=[],
        post_forward_order=[],
        reset_post_forward_order=lambda: None,
        forward_prefetch_depth=0,
    )
    DedicatedState(module, group, comm_ctx, reshard_after_forward=False)

    with torch.no_grad():
        out = module(torch.ones(2, dtype=torch.bfloat16))

    assert module.input_dtype is torch.float32
    assert out.dtype is torch.bfloat16
    assert group.recorded_post_forward is True


def test_parallel_dedicated_states_on_same_module_run_in_registration_order() -> None:
    events: list[str] = []
    module = CaptureDTypeModule()
    first = FakePolicyGroup(name="first", events=events)
    second = FakePolicyGroup(name="second", events=events)
    for group in (first, second):
        group._output_dtype = None
        group._cast_forward_inputs = False
    comm_ctx = SimpleNamespace(
        all_states=[],
        post_forward_order=[],
        reset_post_forward_order=lambda: None,
        forward_prefetch_depth=0,
    )

    first_state = DedicatedState(module, first, comm_ctx, reshard_after_forward=True)
    _attach_dedicated_state(module, first_state)
    second_state = DedicatedState(module, second, comm_ctx, reshard_after_forward=True)
    _attach_dedicated_state(module, second_state)

    with torch.no_grad():
        module(torch.ones(2, dtype=torch.bfloat16))

    assert module._dedicated_states == [first_state, second_state]
    assert module._dedicated_state is first_state
    assert events == [
        "first.pre_forward_wait",
        "first.unshard",
        "first.wait_for_unshard",
        "second.pre_forward_wait",
        "second.unshard",
        "second.wait_for_unshard",
        "first.reshard",
        "first.record_post_forward",
        "second.reshard",
        "second.record_post_forward",
    ]


def test_isolated_pg_barrier_default_disabled(monkeypatch) -> None:
    monkeypatch.delenv("DMUON_ISOLATED_PG_BARRIER", raising=False)
    assert not _isolated_pg_barrier_enabled()

    monkeypatch.setenv("DMUON_ISOLATED_PG_BARRIER", "1")
    assert _isolated_pg_barrier_enabled()

    monkeypatch.setenv("DMUON_ISOLATED_PG_BARRIER", "true")
    assert _isolated_pg_barrier_enabled()

    monkeypatch.setenv("DMUON_ISOLATED_PG_BARRIER", "0")
    assert not _isolated_pg_barrier_enabled()
