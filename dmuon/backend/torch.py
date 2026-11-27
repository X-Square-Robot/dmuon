"""Torch implementation of the owner-local optimizer arithmetic."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from dmuon.optim.newton_schulz import (
    _torch_direct_newton_schulz as direct_newton_schulz,
    _torch_gram_newton_schulz as gram_newton_schulz,
)

__all__ = [
    "momentum_input",
    "apply_update_",
    "scaled_delta",
    "adamw_bucket_",
    "gram_newton_schulz",
    "direct_newton_schulz",
]


@contextmanager
def momentum_input(
    grad: torch.Tensor,
    state: dict[str, torch.Tensor],
    state_dtype: torch.dtype,
    momentum: float,
    nesterov: bool,
) -> Iterator[torch.Tensor]:
    """Update momentum state and yield the Newton-Schulz input."""
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = grad.to(state_dtype).clone()
    else:
        state["momentum_buffer"].mul_(momentum).add_(grad)
    buffer = state["momentum_buffer"]
    yield grad.add(buffer, alpha=momentum) if nesterov else buffer


def apply_update_(
    param: torch.Tensor, update: torch.Tensor, wd_factor: float | None, alpha: float
) -> None:
    """Apply decoupled weight decay and an owner-local parameter update in place."""
    if wd_factor is not None:
        param.mul_(wd_factor)
    param.add_(update.view_as(param).to(device=param.device, dtype=param.dtype), alpha=alpha)


def scaled_delta(
    update: torch.Tensor,
    shape: torch.Size | tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    alpha: float,
) -> torch.Tensor:
    """Materialize a scaled update for publication to tensor-parallel peers."""
    result = update.view(shape).to(device=device, dtype=dtype)
    result.mul_(alpha)
    return result


def adamw_bucket_(
    items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    step: int,
    lr: float,
    betas: tuple[float, float],
    weight_decay: float,
    eps: float,
) -> None:
    """Apply the foreach AdamW update to one owner-local bucket."""
    beta1, beta2 = betas
    params, grads, exp_avgs, exp_avg_sqs = map(list, zip(*items))
    if weight_decay > 0:
        torch._foreach_mul_(params, 1.0 - lr * weight_decay)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step
    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, bc2**0.5)
    torch._foreach_add_(denom, eps)
    torch._foreach_addcdiv_(params, exp_avgs, denom, value=-(lr / bc1))
