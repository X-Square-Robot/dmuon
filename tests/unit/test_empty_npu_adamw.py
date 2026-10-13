"""Empty shard bookkeeping on every device without launching foreach kernels."""

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
@pytest.mark.parametrize("tensor_class", [torch.Tensor, EmptyNpuTensor])
def test_empty_adamw_shard_advances_state_without_foreach(monkeypatch, dtype, tensor_class):
    parameter = torch.empty(0, dtype=dtype).as_subclass(tensor_class)
    gradient = torch.empty(0, dtype=dtype)
    optimizer = SimpleNamespace(state=defaultdict(dict))
    group = {"lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 0.01, "eps": 1e-8}

    def unexpected_foreach(*_args, **_kwargs):
        raise AssertionError("Empty shards must not launch foreach kernels")

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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_empty_shard_does_not_change_nonempty_adamw_update(monkeypatch, dtype):
    empty = torch.empty(0, dtype=dtype)
    parameter = torch.tensor([1.0, -2.0])
    reference = torch.nn.Parameter(parameter.clone())
    optimizer = SimpleNamespace(state=defaultdict(dict))
    group = {"lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 0.01, "eps": 1e-8}
    reference_optimizer = torch.optim.AdamW([reference], **group, foreach=False)

    def guard_foreach(original):
        def checked(tensors, *args, **kwargs):
            assert all(tensor.numel() > 0 for tensor in tensors)
            return original(tensors, *args, **kwargs)

        return checked

    for name in (
        "_foreach_mul_",
        "_foreach_add_",
        "_foreach_addcmul_",
        "_foreach_sqrt",
        "_foreach_div_",
        "_foreach_addcdiv_",
    ):
        monkeypatch.setattr(torch, name, guard_foreach(getattr(torch, name)))

    for step, values in enumerate(([0.2, -0.5], [-0.3, 0.4]), start=1):
        gradient = torch.tensor(values)
        reference.grad = gradient.clone()
        reference_optimizer.step()
        Muon._adamw_update_batch(
            optimizer,
            [("empty", empty, torch.empty_like(empty)), ("nonempty", parameter, gradient)],
            group,
        )
        assert optimizer.state["empty"]["step"] == step
        assert optimizer.state["nonempty"]["step"] == step
        torch.testing.assert_close(parameter, reference.detach(), rtol=1e-6, atol=1e-7)
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state["nonempty"][name],
                reference_optimizer.state[reference][name],
                rtol=1e-6,
                atol=1e-7,
            )
