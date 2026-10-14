"""The device backend must preserve the existing parameter-policy casts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dmuon._core.state import DedicatedState


class CaptureNestedInputs(nn.Module):
    def forward(self, inputs, *, options):
        self.inputs = inputs
        self.options = options
        return {"result": inputs[0], "nested": options["nested"]}


class PolicyGroup:
    def __init__(self, cast):
        self._param_dtype = torch.float32
        self._output_dtype = torch.bfloat16
        self._cast_forward_inputs = cast
        self._post_backward_fired = False

    def _pre_forward_wait(self):
        pass

    def unshard(self, **kwargs):
        pass

    def wait_for_unshard(self):
        pass

    def reshard(self):
        pass

    def _record_post_forward(self):
        pass


@pytest.mark.parametrize("cast", [False, True])
def test_nested_policy_casts_only_floating_tensors(cast):
    module = CaptureNestedInputs()
    context = SimpleNamespace(
        all_states=[],
        post_forward_order=[],
        reset_post_forward_order=lambda: None,
        forward_prefetch_depth=0,
    )
    DedicatedState(module, PolicyGroup(cast), context, reshard_after_forward=False)
    value = torch.ones(2, dtype=torch.bfloat16)
    index = torch.tensor([1, 2], dtype=torch.int64)
    options = {"nested": [value, (index, "unchanged")], "flag": True}
    with torch.no_grad():
        output = module((value, index), options=options)
    expected_dtype = torch.float32 if cast else torch.bfloat16
    assert module.inputs[0].dtype == expected_dtype
    assert module.options["nested"][0].dtype == expected_dtype
    assert module.inputs[1] is index
    assert module.options["nested"][1][0] is index
    assert module.options["nested"][1][1] == "unchanged"
    assert module.options["flag"] is True
    assert output["result"].dtype == torch.bfloat16
