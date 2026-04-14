"""Public API: dedicate_params."""

import logging
from collections import defaultdict
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh

from .comm import DedicatedCommContext
from .group import DedicatedParamGroup
from .param import DedicatedParam
from .partition import _extract_layer_id, compute_balanced_assignment
from .state import DedicatedState

logger = logging.getLogger(__name__)


def _find_parent_module(model: nn.Module, target_param: nn.Parameter) -> tuple[nn.Module, str]:
    """Find the direct parent module and param name for a parameter."""
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param is target_param:
                return module, param_name
    raise ValueError(f"Parameter not found in model: {target_param.shape}")


def _find_layer_module(
    model: nn.Module, target_param: nn.Parameter
) -> tuple[nn.Module, nn.Module, str]:
    """Find (layer_module, parent_module, param_name) for a parameter.

    layer_module: the transformer layer (e.g., model.layers[i]) for hook registration.
    parent_module: the direct parent (e.g., q_proj Linear) for _set_module_param.
    param_name: name within parent_module.

    Falls back to parent_module as layer_module if no layer structure found.
    """
    parent_module, param_name = _find_parent_module(model, target_param)

    # Find the fully qualified name of parent_module
    parent_fqn: Optional[str] = None
    for name, mod in model.named_modules():
        if mod is parent_module:
            parent_fqn = name
            break

    if parent_fqn is None:
        return parent_module, parent_module, param_name

    # Extract layer prefix: "model.layers.3.self_attn.q_proj" → "layers.3"
    layer_id = _extract_layer_id(parent_fqn)
    if layer_id is None:
        return parent_module, parent_module, param_name

    # Build the full path to the layer module
    # e.g., parent_fqn = "model.layers.3.self_attn.q_proj", layer_id = "layers.3"
    idx = parent_fqn.find(layer_id)
    if idx < 0:
        return parent_module, parent_module, param_name
    layer_path = parent_fqn[: idx + len(layer_id)]
    try:
        layer_module = model.get_submodule(layer_path)
    except AttributeError:
        return parent_module, parent_module, param_name

    return layer_module, parent_module, param_name


def dedicate_params(
    model: nn.Module,
    mesh: DeviceMesh,
    predicate: Callable[[str, nn.Parameter], bool],
    compute_dtype: torch.dtype = None,
    reshard_after_forward: bool = True,
) -> dict[nn.Parameter, int]:
    """Mark parameters for dedicated ownership and register communication hooks.

    Parameters satisfying ``predicate`` are assigned to owner ranks via a
    balanced partition algorithm. Each marked parameter will be automatically
    ignored by subsequent ``fully_shard()`` calls (requires the monkey-patch
    from :mod:`dmuon.patch`).

    Communication hooks are registered at the **layer level** (e.g., on
    ``model.layers[i]``), not on individual sub-modules. This minimizes
    CPU launch overhead by batching all broadcasts/reduces per layer.

    Args:
        model: The model whose parameters to partition.
        mesh: 1D DeviceMesh for the data-parallel dimension.
        predicate: Callable ``(param_name, param) -> bool``. Parameters
            returning True will use dedicated ownership.
        compute_dtype: Optional dtype for communication (e.g., torch.bfloat16).
        reshard_after_forward: If True (default), reshard dedicated params after
            forward and re-broadcast in backward. If False (SHARD_GRAD_OP mode),
            keep params unsharded through forward+backward, eliminating backward
            broadcasts at the cost of higher memory.

    Returns:
        Assignment dict mapping each dedicated parameter to its owner rank.
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

    # 3. Create shared communication context
    dp_group = mesh.get_group()
    device_type = mesh.device_type
    device = torch.device(device_type, torch.cuda.current_device())
    comm_ctx = DedicatedCommContext(device)

    # 4. Group by layer module and create DedicatedParam + DedicatedState
    layer_to_dparams: dict[nn.Module, list[DedicatedParam]] = defaultdict(list)

    for param, owner_rank in assignment.items():
        layer_module, parent_module, param_name = _find_layer_module(model, param)
        d_param = DedicatedParam(
            param=param,
            module=parent_module,
            param_name=param_name,
            owner_rank=owner_rank,
            dp_group=dp_group,
            device=device,
            compute_dtype=compute_dtype,
        )
        layer_to_dparams[layer_module].append(d_param)

    all_states: list[DedicatedState] = []
    for layer_module, d_params in layer_to_dparams.items():
        group = DedicatedParamGroup(d_params, comm_ctx)
        state = DedicatedState(layer_module, group, comm_ctx, reshard_after_forward)
        layer_module._dedicated_state = state
        all_states.append(state)

    # Link states for forward prefetch: each state knows the next layer's group
    for i in range(len(all_states) - 1):
        all_states[i]._next_group = all_states[i + 1].group

    # Store comm_ctx on model for external access (e.g., reset in training loop)
    model._dedicated_comm_ctx = comm_ctx

    return assignment
