"""Balanced parameter partition algorithm.

Phase A of the HSDP-native refactor extends this to optionally partition
over a 2D ``(shard, replicate)`` mesh, with LPT now running across all
``G*R`` owner slots.  When ``replicate_mesh is None`` the behaviour is
identical to the previous 1D shard-only algorithm (including the int
return type, which keeps existing tests untouched).
"""

from collections import defaultdict
from typing import Callable, Optional, Tuple, Union

import torch.nn as nn
from torch.distributed import DeviceMesh

from .. import _balance_profile

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    DTensor = None

# Parameters smaller than this are merged with same-layer peers for one broadcast
SMALL_PARAM_THRESHOLD = 5_000_000

OwnerCoord = Tuple[int, int]
OwnerValue = Union[int, OwnerCoord]


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
    replicate_mesh: Optional[DeviceMesh] = None,
) -> dict[nn.Parameter, OwnerValue]:
    """Compute a globally balanced dedicated ownership assignment.

    Algorithm: LPT (Longest Processing Time first) with two constraints:
    1. Same-layer parameters are assigned to different owner slots (keeps
       shard-dim broadcast concurrency intact).  In HSDP mode the slot is a
       2D ``(shard, replicate)`` coord, so the constraint also naturally
       distributes work across replicate peers.
    2. Small parameters (< SMALL_PARAM_THRESHOLD) in the same layer are merged
       into one allocation unit so they share the same owner (packed broadcast).

    Args:
        model: The model to partition.
        mesh: 1D DeviceMesh for the shard dimension.
        predicate: Function (param_name, param) → bool deciding which params to dedicate.
        replicate_mesh: Optional 1D DeviceMesh for the HSDP replicate dimension.
            When given, LPT runs over ``G*R`` owner slots and the returned
            dict maps each param to a ``(shard, replicate)`` tuple.  When
            ``None``, behaviour and return type match the pre-HSDP 1D path.

    Returns:
        Dict mapping each dedicated parameter to its owner.  ``int`` in
        1D shard-only mode, ``Tuple[int, int]`` when ``replicate_mesh``
        is provided.
    """
    shard_size = mesh.size()
    replicate_size = replicate_mesh.size() if replicate_mesh is not None else 1
    is_hsdp = replicate_mesh is not None
    slots: list[OwnerCoord] = [
        (s, r) for s in range(shard_size) for r in range(replicate_size)
    ]

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

    # Greedy assignment with same-layer concurrency constraint.  Owner slots
    # are the full 2D grid; in shard-only mode every slot has replicate=0
    # so the grid collapses to ``shard_size`` entries and the search matches
    # the original 1D algorithm exactly.
    rank_loads: dict[OwnerCoord, int] = {slot: 0 for slot in slots}
    assignment: dict[nn.Parameter, OwnerCoord] = {}
    layer_usage: dict[Optional[str], set[OwnerCoord]] = defaultdict(set)

    for params_list, layer_id, total_numel in alloc_units:
        used_slots = layer_usage[layer_id]
        best_slot = min(
            slots,
            key=lambda s: (s in used_slots, rank_loads[s]),
        )
        for p in params_list:
            assignment[p] = best_slot
        rank_loads[best_slot] += total_numel
        layer_usage[layer_id].add(best_slot)

    _balance_profile.dump_assignment(
        alloc_units=alloc_units,
        assignment=assignment,
        rank_loads=rank_loads,
        shard_size=shard_size,
        replicate_size=replicate_size,
    )

    # Preserve 1D return type when no replicate mesh is configured, so
    # existing call sites, tests and checkpoints continue to see plain ints.
    if not is_hsdp:
        return {p: coord[0] for p, coord in assignment.items()}
    return assignment
