from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from dmuon.grad_clip import (
    MuonGradClipStats,
    _iter_muon_grad_entries,
    _local_total_norm,
    clip_grad_norm_,
    register_muon_grad_clip_strategy,
)


@dataclass
class FakeCommCtx:
    device: torch.device = torch.device("cpu")


class FakeDP:
    def __init__(
        self,
        grad: torch.Tensor | None,
        *,
        name: str = "proj",
        is_tp: bool = False,
        is_tp_owner: bool = False,
        tp_full_grad: torch.Tensor | None = None,
    ) -> None:
        self.param_name = name
        self.is_dtensor = is_tp
        self.tp_group = object() if is_tp else None
        self.is_tp_owner = is_tp_owner
        self._reduced_grad = grad
        self._tp_full_grad = tp_full_grad


class FakeOptimizer:
    def __init__(self, dedicated_params: list[FakeDP]) -> None:
        self._dedicated_params = dedicated_params
        self._fsdp_params = []
        self._comm_ctx = FakeCommCtx()
        self._ensure_calls = 0
        self._last_muon_grad_clip_stats = None

    def _ensure_grads_ready(self) -> None:
        self._ensure_calls += 1


def test_clip_grad_norm_scales_only_dmuon_dedicated_grads() -> None:
    g1 = torch.tensor([3.0, 4.0])
    g2 = torch.tensor([0.0, 12.0])
    adamw_grad = torch.tensor([100.0])
    opt = FakeOptimizer([FakeDP(g1), FakeDP(g2)])
    opt._fsdp_params = [torch.nn.Parameter(torch.ones(1))]
    opt._fsdp_params[0].grad = adamw_grad.clone()

    stats = clip_grad_norm_(opt, 6.5, foreach=False)

    expected_coef = 6.5 / (13.0 + 1e-6)
    assert stats.param_count == 2
    assert stats.clipped
    assert stats.total_norm == pytest.approx(13.0)
    assert stats.clip_coef == pytest.approx(expected_coef)
    assert opt._ensure_calls == 1
    assert torch.allclose(g1, torch.tensor([3.0, 4.0]) * expected_coef)
    assert torch.allclose(g2, torch.tensor([0.0, 12.0]) * expected_coef)
    assert torch.equal(opt._fsdp_params[0].grad, adamw_grad)


def test_clip_grad_norm_stats_only_does_not_scale() -> None:
    grad = torch.tensor([6.0, 8.0])
    opt = FakeOptimizer([FakeDP(grad)])

    stats = clip_grad_norm_(opt, None, foreach=False)

    assert stats.total_norm == pytest.approx(10.0)
    assert stats.max_norm is None
    assert not stats.clipped
    assert torch.equal(grad, torch.tensor([6.0, 8.0]))


def test_local_total_norm_accumulates_in_float32_for_collectives() -> None:
    bf16_grad = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    opt = FakeOptimizer([FakeDP(bf16_grad)])

    total = _local_total_norm(tuple(), 2.0, opt, foreach=False)
    assert total.dtype is torch.float32

    entries = tuple(_iter_muon_grad_entries(opt))
    total = _local_total_norm(entries, 2.0, opt, foreach=False)
    assert total.dtype is torch.float32
    assert total.item() == pytest.approx(5.0)


def test_clip_grad_norm_uses_tp_full_grad_on_tp_owner_only() -> None:
    owner_full = torch.tensor([6.0, 8.0])
    non_owner_local = torch.tensor([100.0, 100.0])
    opt = FakeOptimizer(
        [
            FakeDP(
                torch.tensor([1.0, 1.0]),
                is_tp=True,
                is_tp_owner=True,
                tp_full_grad=owner_full,
            ),
            FakeDP(
                non_owner_local,
                is_tp=True,
                is_tp_owner=False,
                tp_full_grad=None,
            ),
        ]
    )

    stats = clip_grad_norm_(opt, 5.0, foreach=False)

    expected_coef = 5.0 / (10.0 + 1e-6)
    assert stats.param_count == 1
    assert stats.total_norm == pytest.approx(10.0)
    assert torch.allclose(owner_full, torch.tensor([6.0, 8.0]) * expected_coef)
    assert torch.equal(non_owner_local, torch.tensor([100.0, 100.0]))


def test_clip_grad_norm_error_if_nonfinite() -> None:
    opt = FakeOptimizer([FakeDP(torch.tensor([float("inf")]))])

    with pytest.raises(RuntimeError, match="non-finite"):
        clip_grad_norm_(opt, 1.0, error_if_nonfinite=True, foreach=False)


def test_custom_muon_clip_strategy_extension_point() -> None:
    class QuarterStrategy:
        name = "quarter_for_test"

        def __call__(self, context):
            for entry in context.entries:
                entry.grad.mul_(0.25)
            return MuonGradClipStats(
                total_norm=0.0,
                max_norm=context.max_norm,
                norm_type=context.norm_type,
                clip_coef=0.25,
                clipped=True,
                param_count=len(context.entries),
                found_inf=False,
                strategy=self.name,
            )

    register_muon_grad_clip_strategy("quarter_for_test", QuarterStrategy())
    grad = torch.tensor([4.0])
    opt = FakeOptimizer([FakeDP(grad)])

    stats = clip_grad_norm_(opt, 1.0, strategy="quarter_for_test")

    assert stats.strategy == "quarter_for_test"
    assert torch.equal(grad, torch.tensor([1.0]))
