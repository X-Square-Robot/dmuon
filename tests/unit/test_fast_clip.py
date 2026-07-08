from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import dmuon.fast_clip as fast_clip
from dmuon.fast_clip import (
    GradClipBucket,
    _DEFAULT_CHUNK_SIZE,
    _resolve_chunk_size,
    clip_grad_norm_buckets_,
    try_clip_optimizer_grad_norm_buckets_,
)


def _bucket_stats(result):
    return {stats.name: stats for stats in result.bucket_stats}


def test_fast_clip_chunk_size_defaults_and_env_override(monkeypatch) -> None:
    monkeypatch.delenv("DMUON_FAST_CLIP_CHUNK_SIZE", raising=False)
    assert _resolve_chunk_size(None) == _DEFAULT_CHUNK_SIZE
    assert _DEFAULT_CHUNK_SIZE == 262144

    monkeypatch.setenv("DMUON_FAST_CLIP_CHUNK_SIZE", "131072")
    assert _resolve_chunk_size(None) == 131072

    monkeypatch.setenv("DMUON_FAST_CLIP_CHUNK_SIZE", "invalid")
    assert _resolve_chunk_size(None) == _DEFAULT_CHUNK_SIZE

    assert _resolve_chunk_size(4096) == 4096
    assert _resolve_chunk_size(0) == 1


def test_segmented_clip_cpu_preserves_per_bucket_semantics() -> None:
    """On CPU the kernel never applies; each bucket clips segment-locally."""
    regular = torch.tensor([3.0, 4.0])
    muon = torch.tensor([0.0, 12.0])
    adamw = torch.tensor([8.0, 15.0])

    result = clip_grad_norm_buckets_(
        (
            GradClipBucket("regular", [regular], reduce=False),
            GradClipBucket("muon", [muon], reduce=True),
            GradClipBucket("adamw", [adamw], reduce=True),
        ),
        6.0,
    )

    stats = _bucket_stats(result)
    assert stats["regular"].total_norm == pytest.approx(5.0)
    assert stats["muon"].total_norm == pytest.approx(12.0)
    assert stats["adamw"].total_norm == pytest.approx(17.0)
    assert stats["regular"].clip_coef == pytest.approx(1.0)
    assert stats["muon"].clip_coef == pytest.approx(6.0 / (12.0 + 1e-6))
    assert stats["adamw"].clip_coef == pytest.approx(6.0 / (17.0 + 1e-6))
    assert torch.allclose(regular, torch.tensor([3.0, 4.0]))
    assert torch.allclose(muon, torch.tensor([0.0, 12.0]) * stats["muon"].clip_coef)
    assert torch.allclose(adamw, torch.tensor([8.0, 15.0]) * stats["adamw"].clip_coef)
    assert result.fastpath is False


def test_collective_count_is_invariant_to_local_conditions(monkeypatch) -> None:
    """Regression for the distributed-hang bug.

    A non-finite value in a ``reduce=False`` bucket (a rank-local condition)
    must NOT change how many collectives are issued.  The clip must emit
    exactly one all_reduce, of shape ``[num_buckets]``, no matter what — that
    is what keeps ranks in lockstep.
    """
    calls: list[torch.Tensor] = []

    def fake_all_reduce(tensor, op=None, group=None):  # single-rank identity
        calls.append(tensor.clone())
        return tensor

    monkeypatch.setattr(fast_clip.dist, "is_available", lambda: True)
    monkeypatch.setattr(fast_clip.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(fast_clip.dist, "all_reduce", fake_all_reduce)

    regular = torch.tensor([float("inf"), 1.0])  # local non-finite, reduce=False
    muon = torch.tensor([3.0, 4.0])
    adamw = torch.tensor([0.0, 5.0])
    result = clip_grad_norm_buckets_(
        (
            GradClipBucket("regular", [regular], reduce=False),
            GradClipBucket("muon", [muon], reduce=True),
            GradClipBucket("adamw", [adamw], reduce=True),
        ),
        1.0,
    )

    assert len(calls) == 1  # exactly one fused collective
    assert calls[0].shape == (3,)  # one entry per bucket, not per-bucket scalars
    stats = _bucket_stats(result)
    assert stats["regular"].found_inf is True  # reported, not acted on via control flow
    assert stats["muon"].found_inf is False


def test_noncontiguous_tensor_is_clipped_not_skipped() -> None:
    """A non-contiguous grad used to abort the whole call; now it clips fine."""
    base = torch.randn(4, 4)
    grad = base.t()  # non-contiguous view
    assert not grad.is_contiguous()
    reference = grad.clone().contiguous()

    result = clip_grad_norm_buckets_(
        (GradClipBucket("muon", [grad], reduce=True),), 0.5
    )

    norm = reference.float().pow(2).sum().sqrt()
    coef = min(0.5 / (float(norm) + 1e-6), 1.0)
    assert torch.allclose(grad, reference * coef, atol=1e-5)
    assert result.stats_by_name["muon"].total_norm == pytest.approx(float(norm), rel=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_fast_clip_matches_python_reference(monkeypatch) -> None:
    device = torch.device("cuda")
    max_norm = 9.0
    source_buckets = (
        (
            "regular",
            [
                torch.linspace(-2.0, 2.0, 2048, device=device, dtype=torch.float32),
                torch.linspace(-1.0, 1.0, 513, device=device, dtype=torch.bfloat16),
            ],
            False,
        ),
        (
            "muon",
            [
                torch.randn(4097, device=device, dtype=torch.float32),
                torch.randn(123, device=device, dtype=torch.float16),
            ],
            True,
        ),
        (
            "adamw",
            [
                torch.randn(8193, device=device, dtype=torch.bfloat16),
                torch.randn(17, device=device, dtype=torch.float32),
            ],
            True,
        ),
    )

    def _clone(buckets):
        return tuple(
            GradClipBucket(name, [g.clone() for g in grads], reduce=reduce)
            for name, grads, reduce in buckets
        )

    fast_buckets = _clone(source_buckets)
    fast = clip_grad_norm_buckets_(fast_buckets, max_norm)
    if not fast.fastpath:
        pytest.skip("DMuon fast clip CUDA extension is unavailable")

    # Force the pure-torch path for the reference by disabling the extension.
    ref_buckets = _clone(source_buckets)
    monkeypatch.setattr(fast_clip, "_load_fast_clip_extension", lambda: None)
    ref = clip_grad_norm_buckets_(ref_buckets, max_norm)
    assert ref.fastpath is False

    fast_stats = _bucket_stats(fast)
    ref_stats = _bucket_stats(ref)
    for name in fast_stats:
        assert fast_stats[name].total_norm == pytest.approx(
            ref_stats[name].total_norm, rel=2e-3, abs=2e-3
        )
        assert fast_stats[name].clip_coef == pytest.approx(
            ref_stats[name].clip_coef, rel=2e-3, abs=2e-3
        )
    for fast_bucket, ref_bucket in zip(fast_buckets, ref_buckets):
        for fast_grad, ref_grad in zip(fast_bucket.grads, ref_bucket.grads):
            assert torch.allclose(fast_grad.float(), ref_grad.float(), rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_and_python_agree_on_nonfinite(monkeypatch) -> None:
    """NaN/inf must produce identical grads on the CUDA and torch paths."""
    device = torch.device("cuda")
    values = [1.0, float("nan"), 3.0, float("inf")]

    fast_grad = torch.tensor(values, device=device)
    fast = clip_grad_norm_buckets_((GradClipBucket("muon", [fast_grad], reduce=True),), 2.0)
    if not fast.fastpath:
        pytest.skip("DMuon fast clip CUDA extension is unavailable")

    ref_grad = torch.tensor(values, device=device)
    monkeypatch.setattr(fast_clip, "_load_fast_clip_extension", lambda: None)
    clip_grad_norm_buckets_((GradClipBucket("muon", [ref_grad], reduce=True),), 2.0)

    assert torch.equal(fast_grad.isnan(), ref_grad.isnan())
    finite = ~fast_grad.isnan() & ~fast_grad.isinf()
    assert torch.allclose(fast_grad[finite], ref_grad[finite], rtol=3e-3, atol=3e-3)


@dataclass
class _FakeDP:
    grad: torch.Tensor | None
    name: str = "proj"

    def __post_init__(self) -> None:
        self.param_name = self.name
        self.is_dtensor = False
        self.tp_group = None
        self.is_tp_owner = False
        self._reduced_grad = self.grad


class _FakeOptimizer:
    def __init__(self, grad: torch.Tensor) -> None:
        self._dedicated_params = [_FakeDP(grad)]
        self._dp_to_muon_group_idx = {id(self._dedicated_params[0]): 0}
        self._ensure_calls = 0
        self._last_muon_grad_clip_stats = None

    def _ensure_grads_ready(self, *, coalesce_wait: bool = False) -> None:
        self._ensure_calls += 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_optimizer_fast_clip_sets_muon_stats_when_available() -> None:
    muon_grad = torch.tensor([6.0, 8.0], device="cuda")
    regular_grad = torch.tensor([3.0, 4.0], device="cuda")
    adamw_grad = torch.tensor([0.0, 12.0], device="cuda")
    opt = _FakeOptimizer(muon_grad)

    result = try_clip_optimizer_grad_norm_buckets_(
        opt,
        regular_grads=[regular_grad],
        adamw_grads=[adamw_grad],
        max_norm=5.0,
    )
    if result is None:
        pytest.skip("DMuon fast clip CUDA extension is unavailable")

    assert opt._ensure_calls == 1
    assert opt._last_muon_grad_clip_stats is not None
    assert opt._last_muon_grad_clip_stats.total_norm == pytest.approx(10.0)
    assert opt._last_muon_grad_clip_stats.strategy == "segmented_cuda_fastpath"
