"""Clipping results preserve their scalar/tensor unions on every device."""

from types import SimpleNamespace

import pytest
import torch

from dmuon import GradClipBucket, clip_grad_norm_, clip_grad_norm_buckets_


def optimizer_for(grad):
    params = (
        []
        if grad is None
        else [
            SimpleNamespace(
                param_name="proj",
                _reduced_grad=grad,
                is_dtensor=False,
            )
        ]
    )
    return SimpleNamespace(
        _dedicated_params=params,
        _comm_ctx=SimpleNamespace(device=torch.device("cpu")),
        _ensure_grads_ready=lambda: None,
    )


@pytest.mark.parametrize("value", [[3.0, 4.0], [0.0, 0.0], [float("inf")], [float("nan")]])
@pytest.mark.parametrize("max_norm", [None, 2.5])
def test_muon_clip_result_contract(value, max_norm):
    grad = torch.tensor(value)
    original = grad.clone()
    stats = clip_grad_norm_(optimizer_for(grad), max_norm, foreach=False)
    assert stats.total_norm.dtype == torch.float32 and stats.total_norm.ndim == 0
    assert stats.found_inf.dtype == torch.bool and stats.found_inf.ndim == 0
    assert isinstance(stats.clip_coef, (float, torch.Tensor))
    assert isinstance(stats.clipped, (bool, torch.Tensor))
    assert bool(stats.found_inf) == (not bool(torch.isfinite(original).all()))
    if max_norm is None:
        assert type(stats.clip_coef) is float and type(stats.clipped) is bool
        assert stats.clip_coef == 1.0 and not stats.clipped
        torch.testing.assert_close(grad, original, rtol=0, atol=0, equal_nan=True)
    else:
        assert isinstance(stats.clip_coef, torch.Tensor) and stats.clip_coef.ndim == 0
        assert isinstance(stats.clipped, torch.Tensor) and stats.clipped.ndim == 0
        if bool(stats.found_inf):
            assert not bool(stats.clipped)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_nonfinite_muon_clip_raises_when_requested(value):
    with pytest.raises(RuntimeError, match="non-finite"):
        clip_grad_norm_(
            optimizer_for(torch.tensor([value])), 1.0, foreach=False, error_if_nonfinite=True
        )


def test_empty_muon_clip_stats_only():
    stats = clip_grad_norm_(optimizer_for(None), None, foreach=False)
    assert stats.total_norm.item() == 0 and stats.param_count == 0
    assert stats.clip_coef == 1.0 and stats.clipped is False
    assert not bool(stats.found_inf)


@pytest.mark.parametrize("max_norm", [None, 2.5])
def test_segmented_clip_keeps_tensor_fields_and_empty_bucket(max_norm):
    grad = torch.tensor([3.0, 4.0])
    result = clip_grad_norm_buckets_(
        [GradClipBucket("muon", [grad], reduce=False), GradClipBucket("empty", [], reduce=False)],
        max_norm,
    )
    assert not result.fastpath
    assert set(result.stats_by_name) == {"muon", "empty"}
    for stats in result.bucket_stats:
        for field in ("total_norm", "clip_coef", "clipped", "found_inf"):
            tensor = getattr(stats, field)
            assert isinstance(tensor, torch.Tensor) and tensor.ndim == 0
    assert result.stats_by_name["empty"].param_count == 0
    assert result.stats_by_name["empty"].total_norm.item() == 0
