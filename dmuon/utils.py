"""Utility functions."""

import torch.nn as nn

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
