"""Checkpoint-input lifetime checks; real FSDP master reload is tested on GPUs."""

from types import SimpleNamespace

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
        self.group = SimpleNamespace(_all_gather_result=None)

    def _get_fsdp_state(self):
        return SimpleNamespace(_fsdp_param_group=self.group)

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


def test_pending_unshard_is_consumed_before_compute_storage_release(monkeypatch):
    class PendingModule(LiveComputeModule):
        def unshard(self):
            assert self.reshard_calls == 0
            self.group._all_gather_result = None

        def reshard(self):
            assert self.group._all_gather_result is None
            super().reshard()

    monkeypatch.setattr(checkpoint, "_FSDPModule", PendingModule)
    model = PendingModule(torch.bfloat16)
    model.group._all_gather_result = object()
    source = model.weight.detach()
    expected = source.clone()
    protected = checkpoint._reshard_fsdp_for_model_load(model, {"weight": source}, set())
    assert model.reshard_calls == 1
    torch.testing.assert_close(protected["weight"], expected, rtol=0, atol=0)


def test_export_after_eval_uses_master_precision(monkeypatch):
    monkeypatch.setattr(checkpoint, "_FSDPModule", LiveComputeModule)
    model = LiveComputeModule(torch.bfloat16)
    model.master.fill_(0.123456)
    state = checkpoint.get_model_state_dict(model, cpu_offload=False, rank0_only=False)
    assert state["weight"].dtype == torch.float32
    torch.testing.assert_close(state["weight"], model.master, rtol=0, atol=0)
