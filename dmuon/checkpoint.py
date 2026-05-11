"""State dict save/load for DMuon models.

Handles both dedicated parameters (DMuon-managed) and symmetric parameters
(FSDP2-managed) in a single unified state dict.

Model state dicts are in standard format (compatible with single-GPU and
HuggingFace checkpoints). Optimizer state dicts use a DMuon-specific format
with separate sections for FSDP2 and dedicated parameter states.

Usage::

    import dmuon

    # Save
    model_sd = dmuon.get_model_state_dict(model)
    optim_sd = dmuon.get_optimizer_state_dict(model, optimizer)
    if dist.get_rank() == 0:
        torch.save({"model": model_sd, "optim": optim_sd}, "checkpoint.pt")

    # Load
    ckpt = torch.load("checkpoint.pt")
    dmuon.set_model_state_dict(model, ckpt["model"])
    dmuon.set_optimizer_state_dict(model, optimizer, ckpt["optim"])
"""

from typing import Any
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn

from ._backends.fsdp2 import DedicatedParam
from .utils import get_dedicated_params, wait_all_replicate_broadcasts

try:
    from torch.distributed.tensor import DTensor as _DTensor
except ImportError:
    _DTensor = None


# ---- FQN computation ----


def _compute_dedicated_fqns(model: nn.Module) -> dict[DedicatedParam, str]:
    """Map each DedicatedParam to its fully qualified name in the model.

    Walks ``model.named_modules()`` to build a reverse map from module id
    to FQN prefix, then appends ``dp.param_name`` for each dedicated param.
    """
    module_to_fqn = {id(mod): name for name, mod in model.named_modules()}
    result = {}
    for dp in get_dedicated_params(model):
        prefix = module_to_fqn.get(id(dp.module), "")
        fqn = f"{prefix}.{dp.param_name}" if prefix else dp.param_name
        result[dp] = fqn
    return result


# ---- Broadcast / all-gather helpers ----


def _broadcast_from_owner(
    dp: DedicatedParam, *, cpu_offload: bool, keep: bool = True
) -> torch.Tensor:
    """Broadcast ``_owned_data`` to every rank for a full-param reconstruction.

    Phase B: every rank in the owner's shard column already holds a populated
    ``_owned_data`` (see ``DedicatedParam.__init__`` allocation rule).  So a
    single broadcast within each row's shard group suffices — each row's
    shard-owner has the up-to-date value because either (a) it IS the global
    owner, or (b) the most recent ``broadcast_all_updates`` fanned the
    updated value into it along the replicate axis.

    If ``keep`` is False, every rank still participates in the NCCL broadcast
    (otherwise the collective would hang), but the result is discarded and
    ``None`` is returned. Use this for ``rank0_only`` state dicts so that
    non-rank0 ranks don't keep a full CPU copy per dedicated param.
    """
    buf = torch.empty(dp._orig_size, dtype=dp._orig_dtype, device=dp.device)
    if dp._owned_data is not None:
        buf.copy_(dp._owned_data.to(dp._orig_dtype))
    dist.broadcast(buf, src=dp._owner_global_rank, group=dp.dp_group)
    if not keep:
        del buf
        return None
    return buf.cpu() if cpu_offload else buf


def _broadcast_state_tensor(
    tensor, dp: DedicatedParam, *, cpu_offload: bool, keep: bool = True
) -> torch.Tensor:
    """Broadcast a momentum buffer from the global owner to every rank.

    Unlike ``_owned_data`` (held on every shard-peer of the owner), the
    momentum buffer lives ONLY on the global owner.  In HSDP we therefore
    need a two-stage fan-out:

    1. **Replicate axis** — inside the owner's shard column, broadcast from
       the global owner to its replicate-dim peers.  Only ranks on that
       column participate (skipping elsewhere is safe because each rank's
       replicate_group is unique to its shard column).
    2. **Shard axis** — within each row's shard group, broadcast from the
       row's shard-owner rank to its row peers.

    In 1D shard-only mode (``replicate_group is None``), Stage 1 is a no-op
    and the behaviour collapses to the original single-broadcast path.

    If ``keep`` is False, every rank still participates in both broadcast
    stages but the result is discarded and ``None`` is returned (for
    rank0_only).
    """
    m = dp._orig_size[0]
    n = dp._orig_size.numel() // m
    buf = torch.empty(m, n, dtype=torch.float32, device=dp.device)
    if dp.is_owner and tensor is not None:
        assert tensor.shape == (m, n), (
            f"Momentum buffer shape {tensor.shape} does not match expected "
            f"({m}, {n}) derived from param shape {dp._orig_size}. "
            f"If _step_muon's view convention changed, update _broadcast_state_tensor."
        )
        buf.copy_(tensor.float())
    # Stage 1 — replicate axis (HSDP only).  Fires only on ranks in the
    # owner's shard column; on other ranks the replicate_group is a
    # different process group (belonging to their own shard column) and
    # this param's data is distributed to them via Stage 2 instead.
    if (
        dp.replicate_group is not None
        and dp.dp_group.rank() == dp.owner_shard
    ):
        dist.broadcast(
            buf,
            src=dp._owner_replicate_global_rank,
            group=dp.replicate_group,
        )
    # Stage 2 — shard axis (always runs).  After Stage 1, every row's
    # shard-owner rank has the full buffer; now each row fans it to the
    # remaining shard peers.
    dist.broadcast(buf, src=dp._owner_global_rank, group=dp.dp_group)
    if not keep:
        del buf
        return None
    return buf.cpu() if cpu_offload else buf


def _all_gather_fsdp_tensor(
    local_tensor: torch.Tensor, dp_mesh, *, cpu_offload: bool, keep: bool = True
) -> torch.Tensor:
    """All-gather a FSDP2-sharded tensor to reconstruct the full tensor.

    Handles uneven shards (e.g., a [1, 64] tensor split across 4 ranks gives
    [1, 64] on rank 0 and [0, 64] on ranks 1-3).

    Phase B: ``dp_mesh`` may be 2D (HSDP).  Along replicate dims the param
    is fully replicated, so we only need the Shard-dim subgroup to
    reconstruct the full tensor.  The selection uses the dim whose
    placement is Shard(0) — FSDP2's convention for its sharded params.

    If ``keep`` is False, every rank still participates in the all_gather
    collective (required for NCCL correctness), but the gathered shards are
    freed without concatenation and ``None`` is returned. Use this for
    ``rank0_only`` state dicts.

    Args:
        local_tensor: Local shard (FSDP2 uses Shard(0) by default).
        dp_mesh: The DP device mesh.  May be 1D (shard-only) or 2D (HSDP).
        cpu_offload: Move result to CPU.
        keep: If False, discard the gathered result after the collective.

    Returns:
        Full (unsharded) tensor, or None if ``keep`` is False.
    """
    if dp_mesh.ndim == 1:
        world_size = dp_mesh.size()
        dp_group = dp_mesh.get_group()
    else:
        # HSDP: ``dp_mesh`` has names (typically ``replicate``, ``shard``).
        # Prefer the named lookup; fall back to the last dim (FSDP2's
        # convention) if names are missing.
        names = getattr(dp_mesh, "mesh_dim_names", None)
        if names is not None and "shard" in names:
            shard_dim = names.index("shard")
        else:
            shard_dim = dp_mesh.ndim - 1
        dp_group = dp_mesh.get_group(mesh_dim=shard_dim)
        world_size = dp_mesh.size(shard_dim)

    # Exchange shard sizes (handles uneven splits)
    local_size = torch.tensor([local_tensor.shape[0]], dtype=torch.int64,
                              device=local_tensor.device)
    all_sizes = [torch.zeros(1, dtype=torch.int64, device=local_tensor.device)
                 for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size, group=dp_group)
    sizes = [s.item() for s in all_sizes]

    if all(s == sizes[0] for s in sizes):
        # Even shards: simple all-gather
        gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_tensor.contiguous(), group=dp_group)
    else:
        # Uneven shards: pad to max size, all-gather, then trim
        max_size = max(sizes)
        if max_size == 0:
            # All shards empty — return empty tensor with correct tail dims
            if not keep:
                return None
            full_shape = list(local_tensor.shape)
            full_shape[0] = 0
            return torch.empty(full_shape, dtype=local_tensor.dtype,
                               device="cpu" if cpu_offload else local_tensor.device)
        padded_shape = list(local_tensor.shape)
        padded_shape[0] = max_size
        padded = torch.zeros(padded_shape, dtype=local_tensor.dtype,
                             device=local_tensor.device)
        if local_tensor.shape[0] > 0:
            padded[:local_tensor.shape[0]].copy_(local_tensor)
        gathered = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded.contiguous(), group=dp_group)
        # Trim each shard to its actual size
        gathered = [g[:s] for g, s in zip(gathered, sizes) if s > 0]

    if not keep:
        del gathered
        return None
    full = torch.cat(gathered, dim=0)
    return full.cpu() if cpu_offload else full


def _padded_local_shard(
    full: torch.Tensor,
    rank: int,
    world_size: int,
    local_ref: torch.Tensor,
) -> torch.Tensor:
    """Return ``full``'s rank-local chunk under fully_shard's even-padded
    sharding, matching the layout produced by ``_all_gather_fsdp_tensor``
    on the save side.

    ``torch.tensor_split(world_size, dim=0)`` produces *uneven* chunks for
    shapes not divisible by world_size (e.g. shape[0]=14 on ws=4 →
    [4,4,3,3]); ``fully_shard`` pads dim 0 so every rank holds the same
    ``ceil(shape0 / world_size)``. Pre-padding ``full`` with zeros along
    dim 0 to ``local_ref.shape[0] * world_size`` and narrowing to the
    rank's contiguous chunk reproduces the fully_shard layout.

    The pad value is irrelevant — fully_shard's padded slots are
    write-only (never read in compute), so any value works as long as
    the shard size matches ``local_ref``.
    """
    local_size_0 = local_ref.shape[0]
    padded_size = local_size_0 * world_size
    if full.shape[0] < padded_size:
        pad_shape = [padded_size - full.shape[0], *full.shape[1:]]
        pad = torch.zeros(pad_shape, dtype=full.dtype, device=full.device)
        full = torch.cat([full, pad], dim=0)
    return full.narrow(0, rank * local_size_0, local_size_0)


def _get_dp_mesh(model: nn.Module):
    """Extract the DP device mesh from FSDP2 params on the model."""
    for name, param in model.named_parameters():
        if _DTensor is not None and isinstance(param, _DTensor):
            return param._spec.mesh
    return None


# ---- Model state dict ----


def get_model_state_dict(
    model: nn.Module, *, cpu_offload: bool = True, rank0_only: bool = True
) -> dict[str, torch.Tensor]:
    """Get full model state dict with both dedicated and FSDP2 parameters.

    Produces a state dict identical to what a single-GPU model would produce,
    compatible with ``torch.save``/``torch.load`` and HuggingFace checkpoints.

    For dedicated params: broadcasts ``_owned_data`` from each owner.
    For FSDP2 symmetric params: all-gathers sharded DTensors.

    Args:
        model: Model with ``dedicate_params`` and ``fully_shard`` applied.
        cpu_offload: Move tensors to CPU (default True, recommended for saving).
        rank0_only: If True (default), only rank 0 returns a populated state
            dict; other ranks return ``{}``. All ranks still participate in
            the NCCL broadcast / all_gather collectives (required for
            correctness) but discard the gathered tensors so non-rank0 nodes
            don't accumulate ~tens of GB of CPU RAM per rank. Matches the
            semantics of FSDP1's ``FullStateDictConfig(rank0_only=True)``.

    Returns:
        Complete state dict with full (unsharded) tensors for all parameters
        on rank 0; empty dict on other ranks when ``rank0_only`` is True.
    """
    # Phase C.3: drain any pending async replicate broadcast so every
    # owner's ``_owned_data`` reflects the latest optimizer update.
    wait_all_replicate_broadcasts(model)

    dp_fqns = _compute_dedicated_fqns(model)
    dedicated_fqn_set = set(dp_fqns.values())
    dp_mesh = _get_dp_mesh(model)

    keep = (not rank0_only) or (dist.get_rank() == 0)
    sd: dict[str, torch.Tensor] = {}

    # 1. Dedicated params.
    #    FSDP2 path: broadcast from owner (every shard-peer holds _owned_data;
    #    non-shard-peers need the bytes).
    #    DDP path: every rank already has the live ``nn.Parameter`` in sync
    #    (post-step broadcast was drained above), so read .data directly.
    for dp, fqn in dp_fqns.items():
        mode = getattr(dp._orig_param if hasattr(dp, "_orig_param") else None,
                       "_dedicated_mode", None)
        if mode == "ddp":
            if keep:
                t = dp._orig_param.data
                sd[fqn] = t.cpu() if cpu_offload else t.clone()
        else:
            t = _broadcast_from_owner(dp, cpu_offload=cpu_offload, keep=keep)
            if keep:
                sd[fqn] = t

    # 2. Non-dedicated params. FSDP2 path → all-gather sharded DTensors.
    #    DDP-replicate path → param is already full on every rank; read .data.
    for name, param in model.named_parameters():
        if name in dedicated_fqn_set:
            continue
        if _DTensor is not None and isinstance(param, _DTensor) and dp_mesh is not None:
            full = _all_gather_fsdp_tensor(
                param._local_tensor, dp_mesh, cpu_offload=cpu_offload, keep=keep
            )
            if keep:
                sd[name] = full
        else:
            if keep:
                t = param.data
                sd[name] = t.cpu() if cpu_offload else t.clone()

    return sd


def set_model_state_dict(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> None:
    """Load a full state dict into a DMuon model.

    Handles both dedicated params (copy to owner's ``_owned_data``) and
    FSDP2 symmetric params (manual sharding into DTensors).

    The state dict should contain full (unsharded) tensors, as produced by
    :func:`get_model_state_dict` or a single-GPU ``model.state_dict()``.

    Args:
        model: Model with ``dedicate_params`` and ``fully_shard`` applied.
        state_dict: Full state dict mapping FQN to unsharded tensors.
    """
    # 1. Identify dedicated param FQNs.
    dp_fqns = _compute_dedicated_fqns(model)
    fqn_to_dp = {fqn: dp for dp, fqn in dp_fqns.items()}
    dedicated_fqn_set = set(fqn_to_dp.keys())

    # 2. Load dedicated params.
    #    FSDP2 path: copy from state_dict into every rank that holds
    #    ``_owned_data`` (every shard-peer across replicate rows).
    #    DDP path: every rank holds ``_owned_data`` AND the live
    #    ``nn.Parameter`` — copy the state into both so subsequent forwards
    #    see the loaded value and the next NS step reads from it.
    for fqn, dp in fqn_to_dp.items():
        if fqn not in state_dict:
            continue
        if dp._owned_data is not None:
            dp._owned_data.copy_(state_dict[fqn].to(dp._orig_dtype).to(dp.device))
        # DDP path: sync the live parameter too.
        orig_param = getattr(dp, "_orig_param", None)
        if orig_param is not None and getattr(orig_param, "_dedicated_mode", None) == "ddp":
            orig_param.data.copy_(state_dict[fqn].to(orig_param.dtype).to(orig_param.device))

    # 3. Load symmetric params: manually shard full tensors into FSDP2 DTensors.
    for name, param in model.named_parameters():
        if name in dedicated_fqn_set or name not in state_dict:
            continue
        full_tensor = state_dict[name]
        if _DTensor is not None and isinstance(param, _DTensor):
            local = param._local_tensor
            mesh = param._spec.mesh
            # HSDP: shard only along the Shard-dim axis; replicate axes take
            # the same slice on every peer.  1D mesh falls through unchanged.
            if mesh.ndim == 1:
                shard_rank = mesh.get_local_rank()
                shard_world = mesh.size()
            else:
                names = getattr(mesh, "mesh_dim_names", None)
                shard_dim = (
                    names.index("shard")
                    if names is not None and "shard" in names
                    else mesh.ndim - 1
                )
                shard_rank = mesh.get_local_rank(mesh_dim=shard_dim)
                shard_world = mesh.size(shard_dim)
            full_on_device = full_tensor.to(local.dtype).to(local.device)
            local.copy_(_padded_local_shard(full_on_device, shard_rank, shard_world, local))
        else:
            param.data.copy_(full_tensor.to(param.dtype).to(param.device))


# ---- Optimizer state dict ----


def get_optimizer_state_dict(
    model: nn.Module, optimizer: Any, *, cpu_offload: bool = True,
    rank0_only: bool = True,
) -> dict:
    """Get optimizer state dict for a DMuon Muon optimizer.

    Produces a dict with three sections:
    - ``"fsdp"``: FSDP2 AdamW state (FQN-keyed, full tensors)
    - ``"dedicated"``: Muon momentum buffers (FQN-keyed, broadcast from owners)
    - ``"param_groups"``: Hyperparameters for both groups

    Args:
        model: Model with ``dedicate_params`` and ``fully_shard`` applied.
        optimizer: :class:`dmuon.Muon` optimizer instance.
        cpu_offload: Move tensors to CPU (default True).
        rank0_only: If True (default), only rank 0 returns a populated dict;
            other ranks return an empty ``{"fsdp": {}, "dedicated": {},
            "param_groups": []}``. All ranks still participate in the NCCL
            broadcast / all_gather collectives.

    Returns:
        Optimizer state dict in DMuon format.
    """
    # Phase C.3: the momentum buffer is only on the global owner, which
    # is the same rank that writes ``_owned_data`` at step end; draining
    # pending async broadcast makes both consistent with the snapshot.
    wait_all_replicate_broadcasts(model)

    dp_fqns = _compute_dedicated_fqns(model)
    dedicated_fqn_set = set(dp_fqns.values())

    keep = (not rank0_only) or (dist.get_rank() == 0)

    # 1. Dedicated Muon state: broadcast momentum_buffer from owners.
    #    Must iterate ALL dedicated params (not just owned) so all ranks
    #    participate in each broadcast collective.
    all_dedicated = get_dedicated_params(model)
    dedicated_state: dict[str, dict[str, Any]] = {}

    # Check if any owner has momentum state (optimizer.step called at least once).
    # Use a flag broadcast to coordinate: if no owner has state, skip all broadcasts.
    any_has_state = any(
        id(dp) in optimizer.state for dp in all_dedicated if dp.is_owner
    )
    has_state_tensor = torch.tensor([int(any_has_state)], device=next(iter(dp_fqns)).device)
    dist.all_reduce(has_state_tensor, op=dist.ReduceOp.MAX)

    if has_state_tensor.item() > 0:
        for dp in all_dedicated:
            fqn = dp_fqns[dp]
            dp_id = id(dp)
            if dp.is_owner and dp_id in optimizer.state and "momentum_buffer" in optimizer.state[dp_id]:
                buf = optimizer.state[dp_id]["momentum_buffer"]
            else:
                buf = None
            full_buf = _broadcast_state_tensor(
                buf, dp, cpu_offload=cpu_offload, keep=keep
            )
            if keep:
                dedicated_state[fqn] = {"momentum_buffer": full_buf}

    # 2. FSDP2 AdamW state: all-gather sharded state tensors.
    dp_mesh = _get_dp_mesh(model)
    fsdp_state: dict[str, dict[str, Any]] = {}

    # Build FQN mapping for FSDP params.
    # optimizer._fsdp_params contains the same param objects as model.named_parameters().
    fsdp_param_to_fqn: dict[int, str] = {}
    for name, param in model.named_parameters():
        if name not in dedicated_fqn_set:
            fsdp_param_to_fqn[id(param)] = name

    for p in optimizer._fsdp_params:
        fqn = fsdp_param_to_fqn.get(id(p))
        if fqn is None or p not in optimizer.state:
            continue
        state = optimizer.state[p]
        fsdp_entry: dict[str, Any] = {}
        for key, val in state.items():
            if isinstance(val, torch.Tensor) and dp_mesh is not None:
                gathered = _all_gather_fsdp_tensor(
                    val, dp_mesh, cpu_offload=cpu_offload, keep=keep
                )
                if keep:
                    fsdp_entry[key] = gathered
            else:
                # Scalar (e.g., step count)
                if keep:
                    fsdp_entry[key] = val
        if keep:
            fsdp_state[fqn] = fsdp_entry

    # 3. Param group hyperparameters (without tensor refs).
    if keep:
        param_groups = [
            {k: v for k, v in g.items() if k != "params"}
            for g in optimizer.param_groups
        ]
    else:
        param_groups = []

    return {
        "fsdp": fsdp_state,
        "dedicated": dedicated_state,
        "param_groups": param_groups,
    }


def set_optimizer_state_dict(
    model: nn.Module, optimizer: Any, state_dict: dict
) -> None:
    """Load optimizer state dict into a DMuon Muon optimizer.

    Args:
        model: Model with ``dedicate_params`` and ``fully_shard`` applied.
        optimizer: :class:`dmuon.Muon` optimizer instance.
        state_dict: Optimizer state dict as produced by
            :func:`get_optimizer_state_dict`.
    """
    dp_fqns = _compute_dedicated_fqns(model)
    dedicated_fqn_set = set(dp_fqns.values())

    # 1. Dedicated Muon state: load momentum_buffer on owner.
    if "dedicated" in state_dict:
        fqn_to_dp = {fqn: dp for dp, fqn in dp_fqns.items()}
        for fqn, state in state_dict["dedicated"].items():
            if fqn not in fqn_to_dp:
                continue
            dp = fqn_to_dp[fqn]
            if dp.is_owner and "momentum_buffer" in state:
                buf = state["momentum_buffer"].to(dp.device)
                m = dp._owned_data.shape[0]
                buf = buf.view(m, -1)
                optimizer.state[id(dp)] = {"momentum_buffer": buf}

    # 2. FSDP2 AdamW state: shard full tensors back to local shards.
    if "fsdp" in state_dict:
        dp_mesh = _get_dp_mesh(model)
        # Build FQN -> FSDP param object mapping
        fqn_to_fsdp_param: dict[str, nn.Parameter] = {}
        for name, param in model.named_parameters():
            if name not in dedicated_fqn_set:
                fqn_to_fsdp_param[name] = param

        for fqn, saved_state in state_dict["fsdp"].items():
            p = fqn_to_fsdp_param.get(fqn)
            if p is None:
                continue

            opt_state: dict[str, Any] = {}
            for key, val in saved_state.items():
                if isinstance(val, torch.Tensor) and dp_mesh is not None:
                    if dp_mesh.ndim == 1:
                        shard_rank = dp_mesh.get_local_rank()
                        shard_world = dp_mesh.size()
                    else:
                        names = getattr(dp_mesh, "mesh_dim_names", None)
                        shard_dim = (
                            names.index("shard")
                            if names is not None and "shard" in names
                            else dp_mesh.ndim - 1
                        )
                        shard_rank = dp_mesh.get_local_rank(mesh_dim=shard_dim)
                        shard_world = dp_mesh.size(shard_dim)
                    local_ref = p._local_tensor if hasattr(p, "_local_tensor") else p.data
                    full_on_device = val.to(local_ref.dtype).to(local_ref.device)
                    opt_state[key] = _padded_local_shard(
                        full_on_device, shard_rank, shard_world, local_ref
                    ).contiguous()
                else:
                    opt_state[key] = val
            optimizer.state[p] = opt_state

    # 3. Load param group hyperparameters.
    if "param_groups" in state_dict:
        saved_groups = state_dict["param_groups"]
        current_groups = optimizer.param_groups
        if len(saved_groups) != len(current_groups):
            warnings.warn(
                "DMuon optimizer param group count mismatch while loading "
                f"checkpoint: saved={len(saved_groups)} current={len(current_groups)}. "
                "Only the matching prefix will be restored.",
                stacklevel=2,
            )
        for idx, (saved_pg, current_pg) in enumerate(zip(saved_groups, current_groups)):
            for structural_key in ("use_muon", "subgroup_type"):
                if (
                    structural_key in saved_pg
                    and structural_key in current_pg
                    and saved_pg[structural_key] != current_pg[structural_key]
                ):
                    raise RuntimeError(
                        "DMuon optimizer param group structure mismatch at "
                        f"index {idx}: {structural_key} saved="
                        f"{saved_pg[structural_key]!r} current="
                        f"{current_pg[structural_key]!r}"
                    )
            if (
                "group_name" in saved_pg
                and "group_name" in current_pg
                and saved_pg["group_name"] != current_pg["group_name"]
            ):
                warnings.warn(
                    "DMuon optimizer param group name mismatch while loading "
                    f"checkpoint at index {idx}: saved={saved_pg['group_name']!r} "
                    f"current={current_pg['group_name']!r}",
                    stacklevel=2,
                )
            for k, v in saved_pg.items():
                if k != "params":
                    current_pg[k] = v
