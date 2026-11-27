from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

import dmuon.backend as backend
from dmuon.backend import torch as reference
from dmuon.optim.muon import Muon


def test_selector_defaults_to_tilelang_on_950dt_and_honors_torch_override(monkeypatch):
    tilelang = import_module("dmuon.backend.tilelang")
    fast = object()
    monkeypatch.setattr(tilelang, "compute", fast, raising=False)
    monkeypatch.setattr(backend, "_is_950dt", lambda index: index == 1)
    monkeypatch.setattr(backend, "_tilelang_available", lambda: True)
    target = SimpleNamespace(type="npu", index=1)
    other = SimpleNamespace(type="npu", index=0)
    assert backend.resolve_compute_backend(target) is fast
    assert backend.resolve_compute_backend(other) is reference
    with pytest.raises(ValueError, match="950DT"):
        backend.resolve_compute_backend(other, compute_backend="tilelang")
    assert backend.resolve_compute_backend(target, compute_backend="torch") is reference
    with pytest.raises(ValueError, match="compute_backend"):
        backend.resolve_compute_backend(target, compute_backend="typo")
    for device in ("cpu", "cuda:0"):
        assert backend.resolve_compute_backend(torch.device(device)) is reference


def test_public_newton_schulz_forwards_configuration(monkeypatch):
    ns = import_module("dmuon.optim.newton_schulz")
    mock = SimpleNamespace(
        gram_newton_schulz=lambda *args: args, direct_newton_schulz=lambda *args: args
    )
    monkeypatch.setattr(ns, "resolve_compute_backend", lambda device, compute_backend="auto": mock)
    matrix = torch.ones(3, 2)
    coeffs = [(1.0, 2.0, 3.0)]
    assert ns.gram_newton_schulz_local(matrix, 4, 1e-6, coeffs, [2], False) == (
        matrix,
        4,
        1e-6,
        coeffs,
        [2],
        False,
    )
    assert ns.direct_newton_schulz(matrix, 1, 1e-4, coeffs) == (matrix, 1, 1e-4, coeffs)


def test_missing_tilelang_auto_falls_back_and_explicit_request_is_actionable(monkeypatch):
    monkeypatch.setattr(backend, "_is_950dt", lambda _index: True)
    monkeypatch.setattr(backend, "_tilelang_available", lambda: False)
    target = SimpleNamespace(type="npu", index=1)
    assert backend.resolve_compute_backend(target) is reference
    with pytest.raises(RuntimeError, match="USE_ASCEND=ON"):
        backend.resolve_compute_backend(target, compute_backend="tilelang")


def test_explicit_cublas_uses_torch_while_npu_auto_can_use_tilelang(monkeypatch):
    ns = import_module("dmuon.optim.newton_schulz")
    mock = SimpleNamespace(gram_newton_schulz=lambda *args: ("selected", args))
    monkeypatch.setattr(ns, "resolve_compute_backend", lambda _device, compute_backend="auto": mock)
    monkeypatch.setattr(ns, "_torch_gram_newton_schulz", lambda *args: ("torch", args))
    matrix = torch.ones(3, 2)

    for kwargs in ({"kernel": "cublas"}, {"deterministic": True}):
        selector = ns.NewtonSchulz(**kwargs)
        assert selector.local(matrix, 4)[0] == "torch"

    automatic = ns.NewtonSchulz()
    npu_matrix = SimpleNamespace(device=SimpleNamespace(type="npu"))
    result = automatic.local(npu_matrix, 4, compute_backend="tilelang")
    assert result[0] == "selected"
    assert result[1][-1] is automatic.deterministic

    for kwargs in ({"kernel": "cublas"}, {"deterministic": True}):
        selector = ns.NewtonSchulz(**kwargs)
        with pytest.raises(ValueError, match="CUDA-only"):
            selector.local(npu_matrix, 4)


def test_public_deterministic_gram_ns_uses_torch_reference(monkeypatch):
    ns = import_module("dmuon.optim.newton_schulz")
    monkeypatch.setattr(
        ns,
        "resolve_compute_backend",
        lambda _device: (_ for _ in ()).throw(AssertionError("auto backend was selected")),
    )
    expected = object()
    monkeypatch.setattr(ns, "_torch_gram_newton_schulz", lambda *args: expected)
    assert ns.gram_newton_schulz(torch.ones(3, 2), deterministic=True) is expected
    npu_matrix = SimpleNamespace(device=SimpleNamespace(type="npu"))
    with pytest.raises(ValueError, match="CUDA-only"):
        ns.gram_newton_schulz(npu_matrix, deterministic=True)


def test_muon_ns_compute_forwards_optimizer_backend():
    opt = SimpleNamespace(
        _ns=SimpleNamespace(local=lambda *args, **kwargs: (args, kwargs)),
        _ns_steps=5,
        _compute_backend="tilelang",
    )
    matrix = SimpleNamespace(device=SimpleNamespace(type="npu"))
    args, kwargs = Muon._ns_compute(opt, matrix, is_tp=False)
    assert args == (matrix, 5)
    assert kwargs == {"compute_backend": "tilelang"}


@pytest.mark.parametrize("nesterov", [False, True])
def test_owner_update_keeps_state_dtype_and_group_decay(monkeypatch, nesterov):
    # CPU tensors exercise the same owner loop; there is no asynchronous CPU stream.
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    owned = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    expected = owned.clone()
    dp = SimpleNamespace(
        _owned_data=owned, tp_group=None, is_dtensor=False, _optim_dtype=torch.float32
    )
    opt = SimpleNamespace(
        state={},
        _device_module=SimpleNamespace(current_stream=lambda: None),
        _nesterov=nesterov,
        _tp_distributed_gram_enabled=lambda: False,
        _first_step_progress_shape=lambda *args, **kwargs: None,
        _profile_event_start=lambda name: None,
        _profile_event_end=lambda token: None,
        _profile_add=lambda *args: None,
        _ns_compute=lambda value, is_tp: value,
    )
    group = {"lr": 0.1, "weight_decay": 0.01, "momentum": 0.9}
    buffer = torch.zeros_like(owned)
    for step in range(2):
        grad = torch.tensor([[0.25, 0.5], [-0.5, 2.0]]) + step
        dp._reduced_grad = grad.to(torch.bfloat16)
        buffer.mul_(0.9).add_(grad)
        update = grad.add(buffer, alpha=0.9) if nesterov else buffer
        expected.mul_(0.999).add_(update, alpha=-0.1 * 0.2 * 2**0.5)
        Muon._step_muon_params(opt, [dp], group)
        torch.testing.assert_close(owned, expected)
        torch.testing.assert_close(opt.state[id(dp)]["momentum_buffer"], buffer)
        assert dp._reduced_grad is None
