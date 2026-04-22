"""Public API: dedicate_params."""

import logging
from collections import defaultdict
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh

from ._owner_rank import normalize_owner_rank
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


def _find_hook_module(
    model: nn.Module,
    target_param: nn.Parameter,
    hook_boundary_predicate: Optional[Callable[[nn.Module], bool]],
    strict: bool = True,
) -> Optional[nn.Module]:
    """Return the lowest ancestor of ``target_param`` where the predicate is True.

    When ``hook_boundary_predicate`` is None, returns None — caller should fall
    back to the ``_find_layer_module`` path. This split keeps the two semantics
    (partition layer key vs. hook attachment module) decoupled.

    Args:
        model: Root module to search within.
        target_param: Parameter whose ancestor we want.
        hook_boundary_predicate: Callable ``(module) -> bool``. None means the
            caller opted out of explicit hook boundaries.
        strict: When True (default), raise if no ancestor matches the predicate.
            When False, fall back to the parameter's direct parent module.

    Returns:
        The chosen hook module, or None when predicate is None.
    """
    if hook_boundary_predicate is None:
        return None

    parent_module, _ = _find_parent_module(model, target_param)

    # Find parent's FQN (relative to `model`)
    parent_fqn: Optional[str] = None
    for name, mod in model.named_modules():
        if mod is parent_module:
            parent_fqn = name
            break

    # Walk from longest prefix down to root, checking the submodule at each
    # prefix against the predicate.
    parts = parent_fqn.split(".") if parent_fqn else []
    for depth in range(len(parts), -1, -1):
        path = ".".join(parts[:depth])
        try:
            mod = model.get_submodule(path) if path else model
        except AttributeError:
            continue
        if hook_boundary_predicate(mod):
            return mod

    if strict:
        raise ValueError(
            f"param of shape {tuple(target_param.shape)} "
            f"(parent_fqn={parent_fqn!r}): no ancestor matched "
            f"hook_boundary_predicate. Either extend the predicate to cover this "
            f"module, or exclude the param via the dedicate_params `predicate` arg."
        )
    return parent_module


def dedicate_params(
    model: nn.Module,
    mesh: DeviceMesh,
    predicate: Callable[[str, nn.Parameter], bool],
    compute_dtype: torch.dtype = None,
    reshard_after_forward: bool = True,
    replicate_mesh: Optional[DeviceMesh] = None,
    hook_boundary_predicate: Optional[Callable[[nn.Module], bool]] = None,
    hook_boundary_strict: bool = True,
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
        mesh: 1D DeviceMesh over the *shard* dimension (a.k.a. ``dp_group``).
            When ``replicate_mesh`` is provided, this becomes the shard axis
            of the HSDP 2D mesh.
        predicate: Callable ``(param_name, param) -> bool``. Parameters
            returning True will use dedicated ownership.
        compute_dtype: Optional dtype for communication (e.g., torch.bfloat16).
        reshard_after_forward: If True (default), reshard dedicated params after
            forward and re-broadcast in backward. If False (SHARD_GRAD_OP mode),
            keep params unsharded through forward+backward, eliminating backward
            broadcasts at the cost of higher memory.
        replicate_mesh: Optional 1D DeviceMesh over the *replicate* dimension.
            When provided, dedicate_params accepts a HSDP-style 2D layout; the
            LPT partition then balances globally over ``G·R`` owner slots
            (G = shard size, R = replicate size).  **Phase A** only plumbs the
            2D owner coord — all actual collectives still run on the shard
            group only; the replicate-dim reduce/broadcast lands in Phase B/C.
        hook_boundary_predicate: Optional ``(module) -> bool`` selector for the
            hook attachment module. When set, DMuon registers its pre/post
            forward hooks on the **lowest ancestor** of each dedicated param
            where the predicate is True. Use this to align hook boundaries
            with your FSDP2 ``fully_shard`` boundaries — e.g. treat the whole
            ViT as one hook site even though its parameters are distributed
            across ranks. Leave as None to use the built-in ``layers.N`` /
            ``blocks.N`` heuristic (:func:`_find_layer_module`).
        hook_boundary_strict: When True (default) and ``hook_boundary_predicate``
            is given, raise if a dedicated param has no ancestor matching the
            predicate. When False, fall back to the param's direct parent
            module. Strict is recommended to avoid silent per-sub-module hooks.

    Returns:
        Assignment dict mapping each dedicated parameter to its owner rank
        (``int`` in 1D shard-only mode, ``Tuple[int, int]`` in HSDP mode).
    """
    # 1. Compute balanced assignment (passes ``replicate_mesh`` through so
    #    Phase A.3 can switch to 2D LPT; in Phase A partition.py still returns
    #    plain shard ints, which are normalized below).
    assignment = compute_balanced_assignment(
        model, mesh, predicate, replicate_mesh=replicate_mesh
    )
    if not assignment:
        logger.warning("dedicate_params: no parameters matched the predicate")
        return assignment

    # Normalize every assignment value to a 2D ``(shard, replicate)`` coord.
    # In shard-only mode (``replicate_mesh is None``) partition.py returns
    # ints, which get promoted to ``(int, 0)`` — identical behaviour to
    # pre-Phase-A code paths.
    normalized: dict[nn.Parameter, tuple[int, int]] = {
        param: normalize_owner_rank(owner) for param, owner in assignment.items()
    }

    # Log assignment summary
    shard_size = mesh.size()
    replicate_size = replicate_mesh.size() if replicate_mesh is not None else 1
    total_slots = shard_size * replicate_size
    rank_loads: dict[tuple[int, int], int] = defaultdict(int)
    for param, coord in normalized.items():
        rank_loads[coord] += param.numel()
    loads_list = [
        rank_loads.get((s, r), 0)
        for s in range(shard_size)
        for r in range(replicate_size)
    ]
    max_load = max(loads_list) if loads_list else 0
    min_load = min(loads_list) if loads_list else 0
    imbalance = (max_load - min_load) / max(max_load, 1)
    mode = "HSDP" if replicate_mesh is not None else "shard-only"
    logger.info(
        f"dedicate_params[{mode}]: {len(normalized)} params over {total_slots} "
        f"owner slots (shard={shard_size}, replicate={replicate_size}), "
        f"imbalance={imbalance:.1%}, loads={loads_list}"
    )

    # 2. Mark parameters
    for param, coord in normalized.items():
        param._dedicated_owner_rank = coord

    # 3. Create shared communication context
    dp_group = mesh.get_group()
    replicate_group = replicate_mesh.get_group() if replicate_mesh is not None else None
    device_type = mesh.device_type
    device = torch.device(device_type, torch.cuda.current_device())
    comm_ctx = DedicatedCommContext(device, replicate_group=replicate_group)

    # 4. Group by layer module and create DedicatedParam + DedicatedState
    layer_to_dparams: dict[nn.Module, list[DedicatedParam]] = defaultdict(list)

    for param, coord in normalized.items():
        if hook_boundary_predicate is not None:
            # User-explicit hook boundary: lowest ancestor where predicate holds.
            # Mirrors FSDP2 `fully_shard(module, ...)` granularity — the user
            # tells DMuon where the per-layer unit is, independent of DMuon's
            # global LPT partition.
            layer_module = _find_hook_module(
                model, param, hook_boundary_predicate, strict=hook_boundary_strict
            )
            parent_module, param_name = _find_parent_module(model, param)
        else:
            # Default heuristic: infer layer module from `layers.N` / `blocks.N`
            # in the FQN; fall back to parent module if neither matches.
            layer_module, parent_module, param_name = _find_layer_module(model, param)
        d_param = DedicatedParam(
            param=param,
            module=parent_module,
            param_name=param_name,
            owner_rank=coord,
            dp_group=dp_group,
            device=device,
            compute_dtype=compute_dtype,
            replicate_group=replicate_group,
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
