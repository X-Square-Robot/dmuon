"""Checkpoint-input lifetime checks; real FSDP master reload is tested on GPUs."""

import pytest
import torch
from torch import nn

import dmuon.checkpoint as checkpoint


class LiveComputeModule(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(8, dtype=dtype))
        self.master = self.weight.detach().float().clone()
        self.reshard_calls = 0

    def reshard(self):
        old_storage = self.weight.untyped_storage()
        self.weight = nn.Parameter(self.master)
        old_storage.resize_(0)
        self.reshard_calls += 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("view", [False, True])
def test_fsdp_reload_protects_checkpoint_views_before_release(monkeypatch, dtype, view):
    monkeypatch.setattr(checkpoint, "_FSDPModule", LiveComputeModule)
    model = LiveComputeModule(dtype)
    source = model.weight.detach()
    if view:
        source = source.view(2, 4).transpose(0, 1)
    expected = source.clone()
    state_dict = {"weight": source, "other.weight": source}
    protected = checkpoint._reshard_fsdp_for_model_load(model, state_dict, set())
    assert model.reshard_calls == 1
    assert state_dict["weight"] is source  # Caller mapping was not rewritten.
    for name in state_dict:
        torch.testing.assert_close(protected[name], expected, rtol=0, atol=0)
    assert protected["weight"].untyped_storage().data_ptr() != model.master.data_ptr()


@pytest.mark.parametrize("state_dict", [{}, {"dedicated.weight": torch.ones(8)}])
def test_dedicated_only_load_does_not_reshard_fsdp(monkeypatch, state_dict):
    monkeypatch.setattr(checkpoint, "_FSDPModule", LiveComputeModule)
    model = LiveComputeModule(torch.float32)
    checkpoint._reshard_fsdp_for_model_load(model, state_dict, {"dedicated.weight"})
    assert model.reshard_calls == 0


def test_independent_checkpoint_inputs_are_retained(monkeypatch):
    monkeypatch.setattr(checkpoint, "_FSDPModule", LiveComputeModule)
    model = LiveComputeModule(torch.bfloat16)
    state_dict = {"weight": torch.zeros(8)}
    protected = checkpoint._reshard_fsdp_for_model_load(model, state_dict, set())
    assert model.reshard_calls == 1
    assert protected["weight"] is state_dict["weight"]
