"""TP auto-detection and owner assignment for TP-sharded parameters.

DTensor-reuse design: we do NOT cache TP metadata.  All TP information
(mesh, shard_dim, sizes) is read directly from ``param.device_mesh`` and
``param.placements`` when needed (at T2 hook-registration time).  This
module provides only:

    * ``is_tp_sharded``    — boolean detection
    * ``get_tp_mesh``      — TP sub-mesh lookup
    * ``get_tp_shard_dim`` — tensor dim sharded on TP axis

FSDP2 alignment:
``dedicate_params`` never receives a ``tp_mesh`` argument.  The TP
dimension is inferred from each parameter's DTensor — the caller's DP
``mesh_dim_names`` are subtracted from the param's mesh_dim_names, and
whatever is left is TP.  This mirrors how ``fully_shard`` operates.

DTensor reuse: no ``TPShardingInfo`` /
``TPMeshSpec`` dataclasses.  T1 only produces ``(param → int tp_rank)``
owner mappings; T2 looks up mesh / shard_dim from DTensor at the moment
it registers hooks.  This keeps the module small and forward-compatible
with PyTorch DTensor evolution.
"""

from __future__ import annotations

import torch.nn as nn
try:
    from torch.distributed import DeviceMesh
except ImportError:  # Older PyTorch exposes DeviceMesh only from this module.
    from torch.distributed.device_mesh import DeviceMesh

try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Shard
except ImportError:
    DTensor = None
    Shard = None


def is_tp_sharded(
    param: nn.Parameter,
    dp_mesh_dim_names: frozenset[str],
) -> bool:
    """Return True iff ``param`` is a DTensor sharded on at least one mesh
    dim outside ``dp_mesh_dim_names``.

    FSDP2-aligned detection: the param's mesh_dim_names minus DMuon's
    DP dim names = TP dim names.  If at least one TP dim has a ``Shard``
    placement, the param is TP-sharded.  Purely ``Replicate`` on TP dims
    (TP-replicated) → False.

    Raises:
        ValueError: if ``param.device_mesh`` lacks ``mesh_dim_names``
            (same requirement as FSDP2's 2D/3D parallel API).
    """
    if DTensor is None or not isinstance(param, DTensor):
        return False
    names = param.device_mesh.mesh_dim_names
    if names is None:
        raise ValueError(
            "DMuon requires named DeviceMesh for TP detection; "
            "use init_device_mesh(..., mesh_dim_names=(...))."
        )
    tp_names = [n for n in names if n not in dp_mesh_dim_names]
    if not tp_names:
        return False
    for name in tp_names:
        idx = names.index(name)
        # TP dim of size 1 is semantically no-TP — skip so callers don't
        # dispatch zero-communication gather/scatter (wastes NCCL handshake
        # and forces T2 buffer allocation for a degenerate group).
        if param.device_mesh[name].size() <= 1:
            continue
        if isinstance(param.placements[idx], Shard):
            return True
    return False


def get_tp_mesh(
    param: nn.Parameter,
    dp_mesh_dim_names: frozenset[str],
) -> DeviceMesh:
    """Return the TP sub-mesh for a TP-sharded param.

    Assumes ``is_tp_sharded(param, dp_mesh_dim_names)`` is True.
    MVP supports 1D TP (exactly one non-DP mesh dim); 2D TP is deferred.
    """
    names = param.device_mesh.mesh_dim_names
    tp_names = [n for n in names if n not in dp_mesh_dim_names]
    assert len(tp_names) == 1, (
        f"MVP supports 1D TP only; got tp dim names {tp_names}"
    )
    return param.device_mesh[tp_names[0]]


def get_tp_shard_dim(
    param: nn.Parameter,
    dp_mesh_dim_names: frozenset[str],
) -> int:
    """Return the tensor dim sharded on the TP axis.

    ``0`` for ColwiseParallel (Shard(0)), ``1`` for RowwiseParallel
    (Shard(1)).  Assumes ``is_tp_sharded`` is True.
    """
    names = param.device_mesh.mesh_dim_names
    tp_name = next(n for n in names if n not in dp_mesh_dim_names)
    idx = names.index(tp_name)
    placement = param.placements[idx]
    assert isinstance(placement, Shard), (
        f"Expected Shard placement on TP dim, got {placement}"
    )
    return placement.dim
