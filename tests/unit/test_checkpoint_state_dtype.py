from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from dmuon.checkpoint import get_optimizer_state_dict, set_optimizer_state_dict


class FakeDedicatedParam:
    pass


@pytest.mark.parametrize("cpu_offload", [False, True])
@pytest.mark.parametrize("keep", [False, True])
def test_plain_adamw_export_honors_offload_and_rank_filter(monkeypatch, cpu_offload, keep):
    import dmuon.checkpoint as checkpoint

    model = nn.Linear(2, 2, bias=False)
    model.proj = nn.Linear(2, 2, bias=False)
    dp = FakeDedicatedParam()
    dp.module, dp.param_name, dp.device = model.proj, "weight", torch.device("cpu")
    model.proj._dedicated_state = SimpleNamespace(group=SimpleNamespace(params=[dp]))
    state = {
        "exp_avg": torch.ones(2, 2),
        "exp_avg_sq": torch.full((2, 2), 0.25),
        "step": 3,
        "metadata": "unchanged",
    }
    optimizer = SimpleNamespace(
        state={model.weight: state},
        _fsdp_params=[model.weight],
        param_groups=[{"params": [model.weight], "lr": 0.001}],
    )
    calls = []
    original_cpu = torch.Tensor.cpu

    def observed_cpu(tensor, *args, **kwargs):
        calls.append(id(tensor))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(checkpoint, "wait_all_replicate_broadcasts", lambda _model: None)
    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0 if keep else 1)
    monkeypatch.setattr(checkpoint.dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.Tensor, "cpu", observed_cpu)
    saved = get_optimizer_state_dict(model, optimizer, cpu_offload=cpu_offload)

    assert calls == (
        [id(state["exp_avg"]), id(state["exp_avg_sq"])] if keep and cpu_offload else []
    )
    if not keep:
        assert saved["fsdp"] == {} and saved["param_groups"] == []
        return
    entry = saved["fsdp"]["weight"]
    assert entry["step"] == 3 and entry["metadata"] == "unchanged"
    for name in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(entry[name], state[name], rtol=0, atol=0)


def test_set_optimizer_state_dict_restores_muon_momentum_to_compute_dtype():
    model = nn.Module()
    model.proj = nn.Linear(2, 2, bias=False)

    dp = FakeDedicatedParam()
    dp.module = model.proj
    dp.param_name = "weight"
    dp._compute_dtype = torch.bfloat16
    dp._orig_dtype = torch.float32
    dp._orig_size = torch.Size([2, 2])
    dp.device = torch.device("cpu")
    dp.is_owner = True
    dp.tp_group = None
    dp._owned_data = None
    model.proj._dedicated_state = SimpleNamespace(group=SimpleNamespace(params=[dp]))
    optimizer = SimpleNamespace(state={})
    state_dict = {
        "dedicated": {"proj.weight": {"momentum_buffer": torch.ones(2, 2, dtype=torch.float32)}}
    }

    set_optimizer_state_dict(model, optimizer, state_dict)

    momentum = optimizer.state[id(dp)]["momentum_buffer"]
    assert momentum.dtype == torch.bfloat16
    assert momentum.shape == (2, 2)


@pytest.mark.parametrize(
    "optim_dtype,grad_dtype,compute_dtype,orig_dtype,expected_dtype",
    [
        (torch.float32, torch.float16, torch.float16, torch.float32, torch.float32),
        (torch.float32, torch.bfloat16, torch.bfloat16, torch.float32, torch.float32),
        (torch.float16, torch.float32, torch.float32, torch.float32, torch.float16),
        (torch.bfloat16, torch.float32, torch.float32, torch.float32, torch.bfloat16),
        (None, torch.float32, torch.bfloat16, torch.float32, torch.bfloat16),
        (None, torch.float32, None, torch.float16, torch.float16),
    ],
)
def test_restored_muon_momentum_matches_fresh_optimizer_state(
    optim_dtype,
    grad_dtype,
    compute_dtype,
    orig_dtype,
    expected_dtype,
):
    from dmuon.optim.muon import _muon_state_dtype

    model = nn.Module()
    model.proj = nn.Linear(2, 2, bias=False)
    dp = FakeDedicatedParam()
    dp.module = model.proj
    dp.param_name = "weight"
    dp._optim_dtype = optim_dtype
    dp._grad_dtype = grad_dtype
    dp._compute_dtype = compute_dtype
    dp._orig_dtype = orig_dtype
    dp._orig_size = torch.Size([2, 2])
    dp.device = torch.device("cpu")
    dp.is_owner = True
    dp.tp_group = None
    dp._owned_data = None
    model.proj._dedicated_state = SimpleNamespace(group=SimpleNamespace(params=[dp]))
    optimizer = SimpleNamespace(state={})
    grad = torch.tensor([[2048.0, 1.0], [0.125, -4.0]], dtype=grad_dtype)
    fresh = grad.to(_muon_state_dtype(dp, grad.dtype)).clone()
    assert fresh.dtype == expected_dtype
    state_dict = {"dedicated": {"proj.weight": {"momentum_buffer": fresh.float()}}}

    set_optimizer_state_dict(model, optimizer, state_dict)

    restored = optimizer.state[id(dp)]["momentum_buffer"]
    assert restored.dtype == expected_dtype
    torch.testing.assert_close(restored, fresh, rtol=0, atol=0)
    next_grad = torch.full_like(grad, 0.25)
    fresh.mul_(0.95).add_(next_grad)
    restored.mul_(0.95).add_(next_grad)
    torch.testing.assert_close(restored, fresh, rtol=0, atol=0)
    torch.testing.assert_close(
        next_grad.add(restored, alpha=0.95),
        next_grad.add(fresh, alpha=0.95),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("grad_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fresh_muon_dtype_falls_back_to_actual_gradient(grad_dtype):
    from dmuon.optim.muon import _muon_state_dtype

    dp = SimpleNamespace(_optim_dtype=None, _compute_dtype=None, _orig_dtype=None)
    grad = torch.ones(2, 2, dtype=grad_dtype)
    assert grad.to(_muon_state_dtype(dp, grad.dtype)).dtype == grad_dtype


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_restored_plain_adamw_state_follows_parameter_device_and_dtype(device, dtype):
    model = nn.Linear(2, 2, bias=False, device=device, dtype=dtype)
    optimizer = SimpleNamespace(state={})
    saved = {
        "step": 7,
        "exp_avg": torch.full((2, 2), 0.25, dtype=torch.float32),
        "exp_avg_sq": torch.full((2, 2), 0.5, dtype=torch.float32),
        "metadata": "unchanged",
    }

    set_optimizer_state_dict(model, optimizer, {"fsdp": {"weight": saved}})

    restored = optimizer.state[model.weight]
    assert restored.keys() == saved.keys()
    assert restored["step"] == 7 and type(restored["step"]) is int
    assert restored["metadata"] == "unchanged"
    for name in ("exp_avg", "exp_avg_sq"):
        tensor = restored[name]
        assert tensor.device == model.weight.device
        assert tensor.dtype == model.weight.dtype
        assert tensor.shape == model.weight.shape
        if device == "cpu":
            torch.testing.assert_close(tensor, saved[name].to(dtype), rtol=0, atol=0)
        assert saved[name].device.type == "cpu" and saved[name].dtype == torch.float32
