"""Balanced parameter partition algorithm."""

from collections import defaultdict
from typing import Callable, Optional

import torch.nn as nn
from torch.distributed import DeviceMesh

from . import _balance_profile

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    DTensor = None

# Parameters smaller than this are merged with same-layer peers for one broadcast
SMALL_PARAM_THRESHOLD = 5_000_000


def _extract_layer_id(param_name: str) -> Optional[str]:
    """Extract layer identifier from parameter name.

    Matches both ``layers.N`` (transformer decoder) and ``blocks.N`` (ViT).
    The parent prefix is included so ``visual.blocks.3`` and ``model.layers.3``
    do not collide into the same LPT bucket.

    Examples:
        "model.layers.3.mlp.gate_proj.weight" → "model.layers.3"
        "visual.blocks.5.attn.qkv.weight"     → "visual.blocks.5"
        "model.embed_tokens.weight"            → None
    """
    parts = param_name.split(".")
    for i, part in enumerate(parts):
        if part in ("layers", "blocks") and i + 1 < len(parts):
            prefix = ".".join(parts[:i]) or "_root"
            return f"{prefix}.{part}.{parts[i + 1]}"
    return None


def _param_numel(param: nn.Parameter) -> int:
    """Get the local numel of a parameter (handles DTensor)."""
    if DTensor is not None and isinstance(param, DTensor):
        return param._local_tensor.numel()
    return param.numel()


def compute_balanced_assignment(
    model: nn.Module,
    mesh: DeviceMesh,
    predicate: Callable[[str, nn.Parameter], bool],
) -> dict[nn.Parameter, int]:
    """Compute a globally balanced dedicated ownership assignment.

    Algorithm: LPT (Longest Processing Time first) with two constraints:
    1. Same-layer parameters are assigned to different ranks (broadcast concurrency)
    2. Small parameters (< SMALL_PARAM_THRESHOLD) in the same layer are merged
       into one allocation unit so they share the same owner (packed broadcast)

    Args:
        model: The model to partition.
        mesh: Device mesh for the DP dimension.
        predicate: Function (param_name, param) → bool deciding which params to dedicate.

    Returns:
        Dict mapping each dedicated parameter to its owner rank.
    """
    world_size = mesh.size()

    # Collect candidates grouped by layer
    layer_params: dict[Optional[str], list[tuple[nn.Parameter, str, int]]] = defaultdict(list)
    for name, param in model.named_parameters():
        if predicate(name, param):
            layer_id = _extract_layer_id(name)
            numel = _param_numel(param)
            layer_params[layer_id].append((param, name, numel))

    # Build allocation units: large params standalone, small params merged per-layer
    alloc_units: list[tuple[list[nn.Parameter], Optional[str], int]] = []

    for layer_id, params in layer_params.items():
        small = [(p, n, s) for p, n, s in params if s < SMALL_PARAM_THRESHOLD]
        large = [(p, n, s) for p, n, s in params if s >= SMALL_PARAM_THRESHOLD]

        for p, _n, s in large:
            alloc_units.append(([p], layer_id, s))

        if small:
            merged_params = [p for p, _, _ in small]
            merged_numel = sum(s for _, _, s in small)
            alloc_units.append((merged_params, layer_id, merged_numel))

    # Sort by numel descending (LPT)
    alloc_units.sort(key=lambda x: x[2], reverse=True)

    # Greedy assignment with same-layer concurrency constraint
    rank_loads = [0] * world_size
    assignment: dict[nn.Parameter, int] = {}
    layer_usage: dict[Optional[str], set[int]] = defaultdict(set)

    for params_list, layer_id, total_numel in alloc_units:
        used_ranks = layer_usage[layer_id]
        # Prefer least-loaded rank that hasn't been used in this layer
        best_rank = min(
            range(world_size),
            key=lambda r: (r in used_ranks, rank_loads[r]),
        )
        for p in params_list:
            assignment[p] = best_rank
        rank_loads[best_rank] += total_numel
        layer_usage[layer_id].add(best_rank)

    _balance_profile.dump_assignment(
        alloc_units=alloc_units,
        assignment=assignment,
        rank_loads=rank_loads,
        world_size=world_size,
    )

    return assignment
