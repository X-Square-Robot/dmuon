"""Empty shard bookkeeping without initializing an NPU in CPU unit tests."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from dmuon.optim.muon import Muon


class EmptyNpuTensor(torch.Tensor):
    @property
    def device(self):
        return SimpleNamespace(type="npu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_empty_npu_adamw_shard_advances_state_without_foreach(monkeypatch, dtype):
    parameter = torch.empty(0, dtype=dtype).as_subclass(EmptyNpuTensor)
    gradient = torch.empty(0, dtype=dtype)
    optimizer = SimpleNamespace(state=defaultdict(dict))
    group = {"lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 0.01, "eps": 1e-8}

    def unexpected_foreach(*_args, **_kwargs):
        raise AssertionError("Empty NPU shards must not launch foreach kernels")

    for name in (
        "_foreach_mul_",
        "_foreach_add_",
        "_foreach_addcmul_",
        "_foreach_sqrt",
        "_foreach_div_",
        "_foreach_addcdiv_",
    ):
        monkeypatch.setattr(torch, name, unexpected_foreach)
    for step in (1, 2):
        Muon._adamw_update_batch(optimizer, [("empty", parameter, gradient)], group)
        state = optimizer.state["empty"]
        assert state["step"] == step
        for name in ("exp_avg", "exp_avg_sq"):
            assert state[name].shape == parameter.shape and state[name].numel() == 0
            assert state[name].dtype == dtype
