"""Balanced parameter partition algorithm.

Phase A of the HSDP-native refactor extends this to optionally partition
over a 2D ``(shard, replicate)`` mesh, with LPT now running across all
``G*R`` owner slots.  When ``replicate_mesh is None`` the behaviour is
identical to the previous 1D shard-only algorithm (including the int
return type, which keeps existing tests untouched).

T1 extends the return type to ``AssignmentResult``, which keeps the
existing ``dp_owners`` mapping (1D ``int`` or 2D ``(shard, replicate)``
tuple, unchanged in shape) and adds a sparse ``tp_owners`` dict populated
only for TP-sharded parameters.  See
``docs/internal/research/tp_design.md`` §6.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

import torch.nn as nn
from torch.distributed import DeviceMesh

from .. import _balance_profile
from .tp import assign_tp_owner, is_tp_sharded

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    DTensor = None

# Parameters smaller than this are merged with same-layer peers for one broadcast
SMALL_PARAM_THRESHOLD = 5_000_000

OwnerCoord = Tuple[int, int]
OwnerValue = Union[int, OwnerCoord]


@dataclass
class AssignmentResult:
    """Owner assignment output — DP and TP ownership stored separately.

    DP and TP are orthogonal concerns (mirroring FSDP2's own DP/TP
    separation).  TP metadata beyond the owner rank is NOT cached here;
    T2 reads it directly from ``param.device_mesh`` / ``param.placements``
    at hook registration time.

    For pre-refactor compatibility the object also behaves as a read-only
    dict over ``dp_owners`` — legacy callers that did
    ``assignment[p] / p in assignment / assignment.values()`` keep
    working without touching their sites.
    """

    dp_owners: dict[nn.Parameter, OwnerValue]
    """DP owner coord — ``int`` (1D DP) or ``(shard, replicate)`` (HSDP).
    Shape unchanged from pre-TP code paths."""

    tp_owners: dict[nn.Parameter, int] = field(default_factory=dict)
    """TP rank within the TP group, **only** populated for TP-sharded
    params.  Empty dict when no parameter is TP-sharded."""

    # ---- dict-like delegation over dp_owners (back-compat) ----

    def __getitem__(self, key):  # type: ignore[override]
        return self.dp_owners[key]

    def __contains__(self, key) -> bool:
        return key in self.dp_owners

    def __iter__(self):
        return iter(self.dp_owners)

    def __len__(self) -> int:
        return len(self.dp_owners)

    def __bool__(self) -> bool:
        return bool(self.dp_owners)

    def keys(self):
        return self.dp_owners.keys()

    def values(self):
        return self.dp_owners.values()

    def items(self):
        return self.dp_owners.items()

    def get(self, key, default=None):
        return self.dp_owners.get(key, default)


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
    tp_owner_strategy: str = "rank0",
) -> AssignmentResult:
    """Compute a globally balanced dedicated ownership assignment.

    Algorithm: LPT (Longest Processing Time first) with two constraints:
    1. Same-layer parameters are assigned to different owner slots (keeps
       shard-dim broadcast concurrency intact).  In HSDP mode the slot is a
       2D ``(shard, replicate)`` coord, so the constraint also naturally
       distributes work across replicate peers.
    2. Small parameters (< SMALL_PARAM_THRESHOLD) in the same layer are merged
       into one allocation unit so they share the same owner (packed broadcast).

    After DP LPT finishes, a second pass visits every assigned parameter
    and — for those whose DTensor is sharded on a non-DP mesh dim —
    picks a TP owner within the TP group.  TP auto-detection is
    FSDP2-aligned: the caller never passes a ``tp_mesh`` argument; the
    TP dimension is whatever mesh dim is not named in ``mesh`` /
    ``replicate_mesh``.

    Args:
        model: The model to partition.
        mesh: 1D DeviceMesh for the shard dimension.
        predicate: Function (param_name, param) → bool deciding which params to dedicate.
        replicate_mesh: Optional 1D DeviceMesh for the HSDP replicate dimension.
            When given, LPT runs over ``G*R`` owner slots and the returned
            dp_owners map each param to a ``(shard, replicate)`` tuple.
            When ``None``, dp_owners uses plain ``int`` as before.
        tp_owner_strategy: Strategy for picking a TP rank as owner of each
            TP-sharded parameter.  ``"rank0"`` (default, MVP) always picks
            TP rank 0; ``"lpt"`` is deferred to post-MVP.

    Returns:
        ``AssignmentResult`` with ``dp_owners`` (shape matches pre-TP
        behaviour) and ``tp_owners`` (only populated for TP-sharded
        parameters — empty dict when no TP is in use).
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

    # Preserve 1D dp_owners shape when no replicate mesh is configured, so
    # existing call sites, tests and checkpoints continue to see plain ints.
    dp_owners: dict[nn.Parameter, OwnerValue]
    if not is_hsdp:
        dp_owners = {p: coord[0] for p, coord in assignment.items()}
    else:
        dp_owners = dict(assignment)

    # Phase 3: TP owner assignment.  Build the set of DP mesh dim names
    # from the named DP mesh + replicate mesh; every DTensor mesh dim
    # outside this set is treated as a TP dim.  Purely-DP parameters and
    # non-DTensor parameters are skipped (is_tp_sharded returns False).
    dp_names: set[str] = set()
    if mesh.mesh_dim_names:
        dp_names |= set(mesh.mesh_dim_names)
    if replicate_mesh is not None and replicate_mesh.mesh_dim_names:
        dp_names |= set(replicate_mesh.mesh_dim_names)
    dp_mesh_dim_names = frozenset(dp_names)

    tp_owners: dict[nn.Parameter, int] = {}
    for p in dp_owners:
        if is_tp_sharded(p, dp_mesh_dim_names):
            tp_owners[p] = assign_tp_owner(
                p, dp_mesh_dim_names, strategy=tp_owner_strategy
            )

    return AssignmentResult(dp_owners=dp_owners, tp_owners=tp_owners)
