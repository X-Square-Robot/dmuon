"""Public API: dedicate_params."""

import logging
from collections import defaultdict
from typing import Callable

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh

from .group import DedicatedParamGroup
from .param import DedicatedParam
from .partition import compute_balanced_assignment
from .state import DedicatedState

logger = logging.getLogger(__name__)


def _find_parent_module(model: nn.Module, target_param: nn.Parameter) -> tuple[nn.Module, str]:
    """Find the direct parent module and param name for a parameter."""
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param is target_param:
                return module, param_name
    raise ValueError(f"Parameter not found in model: {target_param.shape}")


def dedicate_params(
    model: nn.Module,
    mesh: DeviceMesh,
    predicate: Callable[[str, nn.Parameter], bool],
    compute_dtype: torch.dtype = None,
) -> dict[nn.Parameter, int]:
    """Mark parameters for dedicated ownership and register communication hooks.

    Parameters satisfying ``predicate`` are assigned to owner ranks via a
    balanced partition algorithm. Each marked parameter will be automatically
    ignored by subsequent ``fully_shard()`` calls (requires the monkey-patch
    from :mod:`dmuon.patch`).

    Args:
        model: The model whose parameters to partition.
        mesh: 1D DeviceMesh for the data-parallel dimension.
        predicate: Callable ``(param_name, param) -> bool``. Parameters
            returning True will use dedicated ownership.

    Returns:
        Assignment dict mapping each dedicated parameter to its owner rank.

    Example::

        from dmuon import dedicate_params
        from torch.distributed.fsdp import fully_shard

        dedicate_params(model, dp_mesh, predicate=lambda n, p: "proj" in n)

        for layer in model.layers:
            fully_shard(layer, mesh=dp_mesh)
        fully_shard(model, mesh=dp_mesh)
    """
    # 1. Compute balanced assignment
    assignment = compute_balanced_assignment(model, mesh, predicate)
    if not assignment:
        logger.warning("dedicate_params: no parameters matched the predicate")
        return assignment

    # Log assignment summary
    world_size = mesh.size()
    rank_loads = defaultdict(int)
    for param, rank in assignment.items():
        rank_loads[rank] += param.numel()
    max_load = max(rank_loads.values())
    min_load = min(rank_loads.values()) if len(rank_loads) == world_size else 0
    imbalance = (max_load - min_load) / max(max_load, 1)
    logger.info(
        f"dedicate_params: {len(assignment)} params assigned to {world_size} ranks, "
        f"imbalance={imbalance:.1%}, "
        f"loads={[rank_loads.get(r, 0) for r in range(world_size)]}"
    )

    # 2. Mark parameters
    for param, owner_rank in assignment.items():
        param._dedicated_owner_rank = owner_rank

    # 3. Group by parent module and create DedicatedParam + DedicatedState
    dp_group = mesh.get_group()
    device_type = mesh.device_type
    device = torch.device(device_type, torch.cuda.current_device())

    module_to_dparams: dict[nn.Module, list[DedicatedParam]] = defaultdict(list)

    for param, owner_rank in assignment.items():
        module, param_name = _find_parent_module(model, param)
        d_param = DedicatedParam(
            param=param,
            module=module,
            param_name=param_name,
            owner_rank=owner_rank,
            dp_group=dp_group,
            device=device,
            compute_dtype=compute_dtype,
        )
        module_to_dparams[module].append(d_param)

    for module, d_params in module_to_dparams.items():
        group = DedicatedParamGroup(d_params)
        state = DedicatedState(module, group)
        module._dedicated_state = state

    return assignment
