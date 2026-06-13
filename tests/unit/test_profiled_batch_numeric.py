"""Numerical checks for profiled batched owner-local Muon compute."""

import os
import sys

os.environ.setdefault("DMUON_CACHE_DIR", "/tmp/dmuon_test_cache")

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import torch

from dmuon.optim.profiled_batch import owner_local_muon_batch_update


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.float().norm().clamp_min(1e-12)
    return float((a.float() - b.float()).norm().div(denom).item())


def test_batched_owner_muon_update_matches_per_item_loop_on_cublas_path():
    torch.manual_seed(20260611)
    grad = torch.randn(4, 4, 6, dtype=torch.float32)
    owned = torch.randn(4, 4, 6, dtype=torch.float32)
    momentum = torch.randn(4, 4, 6, dtype=torch.float32)

    batched_owned, batched_momentum = owner_local_muon_batch_update(
        grad.clone(),
        owned.clone(),
        momentum.clone(),
        backend="cublas",
        lr=0.02,
        momentum=0.95,
        weight_decay=0.01,
        nesterov=True,
    )

    loop_owned = []
    loop_momentum = []
    for idx in range(grad.shape[0]):
        item_owned, item_momentum = owner_local_muon_batch_update(
            grad[idx : idx + 1].clone(),
            owned[idx : idx + 1].clone(),
            momentum[idx : idx + 1].clone(),
            backend="cublas",
            lr=0.02,
            momentum=0.95,
            weight_decay=0.01,
            nesterov=True,
        )
        loop_owned.append(item_owned[0])
        loop_momentum.append(item_momentum[0])
    loop_owned = torch.stack(loop_owned, dim=0)
    loop_momentum = torch.stack(loop_momentum, dim=0)

    max_abs_owned = float((batched_owned - loop_owned).abs().max().item())
    max_abs_momentum = float((batched_momentum - loop_momentum).abs().max().item())
    assert max_abs_owned <= 1e-5
    assert max_abs_momentum <= 1e-6
    assert _relative_error(batched_owned, loop_owned) <= 1e-5
