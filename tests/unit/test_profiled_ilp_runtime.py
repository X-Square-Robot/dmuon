"""Unit tests for profiled_ilp runtime batch execution."""

import os
import sys
from types import SimpleNamespace

os.environ.setdefault("DMUON_CACHE_DIR", "/tmp/dmuon_test_cache")

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import torch

from dmuon.optim.muon import Muon


class FakeDedicatedParam:
    is_dtensor = False
    tp_group = None

    def __init__(self, *, group_key=None, backend="cublas"):
        self._profiled_ilp_group_key = group_key
        self._profiled_ilp_backend = backend
        self._reduced_grad = torch.ones(4, 4)
        self._owned_data = torch.zeros(4, 4)


class FakeMuon:
    def __init__(self, *, backend="gram", deterministic=False):
        self._ns = SimpleNamespace(
            backend=backend,
            deterministic=deterministic,
            coefficients=None,
            restart_iterations=None,
        )
        self._nesterov = True
        self.state = {}
        self.profile = {}

    def _first_step_progress_shape(self, *_args, **_kwargs):
        pass

    def _profile_event_start(self, _name):
        return None

    def _profile_event_end(self, _token):
        pass

    def _profile_add(self, key, value):
        self.profile[key] = self.profile.get(key, 0) + value


def test_profiled_ilp_runtime_batches_same_group(monkeypatch):
    calls = []

    def fake_batched_ns(ns_input_batch, *, backend, coefficients, restart_iterations):
        calls.append(
            {
                "shape": tuple(ns_input_batch.shape),
                "backend": backend,
                "coefficients": coefficients,
                "restart_iterations": restart_iterations,
            }
        )
        return torch.ones_like(ns_input_batch)

    import dmuon.optim.profiled_batch as profiled_batch

    monkeypatch.setattr(profiled_batch, "batched_gram_newton_schulz", fake_batched_ns)

    dps = [
        FakeDedicatedParam(group_key=("rank0", (4, 4), "cublas", 2, 0)),
        FakeDedicatedParam(group_key=("rank0", (4, 4), "cublas", 2, 0)),
    ]
    fake_muon = FakeMuon()

    processed = Muon._step_profiled_ilp_batch_params(
        fake_muon,
        dps,
        lr=0.1,
        mu=0.9,
        wd=0.0,
    )

    assert processed == {id(dp) for dp in dps}
    assert calls == [
        {
            "shape": (2, 4, 4),
            "backend": "cublas",
            "coefficients": None,
            "restart_iterations": None,
        }
    ]
    for dp in dps:
        assert dp._reduced_grad is None
        assert torch.allclose(dp._owned_data, torch.full((4, 4), -0.04))
        assert "momentum_buffer" in fake_muon.state[id(dp)]
    assert fake_muon.profile["profiled_ilp_batch_count"] == 1
    assert fake_muon.profile["ns_matrix_count"] == 2


def test_profiled_ilp_runtime_deterministic_forces_cublas(monkeypatch):
    backends = []

    def fake_batched_ns(ns_input_batch, *, backend, coefficients, restart_iterations):
        backends.append(backend)
        return torch.zeros_like(ns_input_batch)

    import dmuon.optim.profiled_batch as profiled_batch

    monkeypatch.setattr(profiled_batch, "batched_gram_newton_schulz", fake_batched_ns)

    dp = FakeDedicatedParam(group_key=("rank0", (4, 4), "tilelang", 1, 0), backend="tilelang")
    fake_muon = FakeMuon(deterministic=True)

    processed = Muon._step_profiled_ilp_batch_params(
        fake_muon,
        [dp],
        lr=0.1,
        mu=0.9,
        wd=0.0,
    )

    assert processed == {id(dp)}
    assert backends == ["cublas"]


def test_profiled_ilp_runtime_skips_direct_ns_backend(monkeypatch):
    called = False

    def fake_batched_ns(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("direct NS path must not call batched Gram NS")

    import dmuon.optim.profiled_batch as profiled_batch

    monkeypatch.setattr(profiled_batch, "batched_gram_newton_schulz", fake_batched_ns)

    dp = FakeDedicatedParam(group_key=("rank0", (4, 4), "cublas", 1, 0))
    fake_muon = FakeMuon(backend="direct")

    processed = Muon._step_profiled_ilp_batch_params(
        fake_muon,
        [dp],
        lr=0.1,
        mu=0.9,
        wd=0.0,
    )

    assert processed == set()
    assert called is False
    assert dp._reduced_grad is not None


def test_profiled_ilp_runtime_ignores_params_without_batch_key(monkeypatch):
    def fake_batched_ns(*_args, **_kwargs):
        raise AssertionError("param without profiled_ilp key must not be batched")

    import dmuon.optim.profiled_batch as profiled_batch

    monkeypatch.setattr(profiled_batch, "batched_gram_newton_schulz", fake_batched_ns)

    dp = FakeDedicatedParam(group_key=None)
    fake_muon = FakeMuon()

    processed = Muon._step_profiled_ilp_batch_params(
        fake_muon,
        [dp],
        lr=0.1,
        mu=0.9,
        wd=0.0,
    )

    assert processed == set()
    assert dp._reduced_grad is not None
