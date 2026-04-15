"""Utility functions."""

from contextlib import contextmanager
from typing import Optional

import torch.nn as nn

from .comm import DedicatedCommContext
from .param import DedicatedParam


def get_dedicated_params(model: nn.Module) -> list[DedicatedParam]:
    """Collect all DedicatedParam instances from a model."""
    result = []
    for module in model.modules():
        if hasattr(module, "_dedicated_state"):
            result.extend(module._dedicated_state.group.params)
    return result


def get_owned_params(model: nn.Module, rank: int) -> list[DedicatedParam]:
    """Collect DedicatedParam instances owned by a specific rank."""
    return [p for p in get_dedicated_params(model) if p.owner_rank == rank]


def get_comm_ctx(model: nn.Module) -> Optional[DedicatedCommContext]:
    """Get the DedicatedCommContext from a model, if it exists."""
    return getattr(model, "_dedicated_comm_ctx", None)


def wait_all_reduces(model: nn.Module) -> None:
    """Wait for all pending gradient reduces to complete and unpack results.

    Call this after loss.backward() and before the optimizer step.
    Gradient reduces are dispatched asynchronously during backward to overlap
    with backward computation. This function waits for them to finish and
    makes ``_reduced_grad`` available on each owner's DedicatedParam.
    """
    for module in model.modules():
        if hasattr(module, "_dedicated_state"):
            module._dedicated_state.group.wait_for_reduce()


@contextmanager
def no_sync(model: nn.Module):
    """Context manager to disable gradient reduction for gradient accumulation.

    Within this context, backward passes skip reduce communication and
    accumulate gradients locally. On the next backward outside this context,
    the accumulated gradients are merged and reduced normally.

    This also disables FSDP2's gradient sync for symmetric parameters.

    Usage::

        for i, batch in enumerate(dataloader):
            ctx = dmuon.no_sync(model) if (i + 1) % accum_steps != 0 else nullcontext()
            with ctx:
                loss = model(batch).loss / accum_steps
                loss.backward()
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    """
    # Disable reduce for dedicated params
    groups = []
    for module in model.modules():
        if hasattr(module, "_dedicated_state"):
            group = module._dedicated_state.group
            groups.append(group)
            group.reduce_grads_enabled = False
    # Disable reduce-scatter for FSDP2 symmetric params
    if hasattr(model, "set_requires_gradient_sync"):
        model.set_requires_gradient_sync(False)
    try:
        yield
    finally:
        for group in groups:
            group.reduce_grads_enabled = True
        if hasattr(model, "set_requires_gradient_sync"):
            model.set_requires_gradient_sync(True)
