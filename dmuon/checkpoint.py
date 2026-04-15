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

import torch
import torch.distributed as dist
import torch.nn as nn

from .param import DedicatedParam
from .utils import get_dedicated_params

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


def _broadcast_from_owner(dp: DedicatedParam, *, cpu_offload: bool) -> torch.Tensor:
    """Broadcast ``_owned_data`` from owner to all ranks in dp_group."""
    buf = torch.empty(dp._orig_size, dtype=dp._orig_dtype, device=dp.device)
    if dp.is_owner:
        buf.copy_(dp._owned_data.to(dp._orig_dtype))
    dist.broadcast(buf, src=dp._owner_global_rank, group=dp.dp_group)
    return buf.cpu() if cpu_offload else buf


def _broadcast_state_tensor(
    tensor, dp: DedicatedParam, *, cpu_offload: bool
) -> torch.Tensor:
    """Broadcast a momentum buffer from owner to all ranks.

    The momentum buffer shape is derived from ``dp._orig_size``: it is always
    ``(orig_size[0], prod(orig_size[1:]))``, matching the 2D view used in
    ``_step_muon``.  All ranks know ``_orig_size``, so no shape metadata
    broadcast is needed — just one broadcast for the data.
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
    dist.broadcast(buf, src=dp._owner_global_rank, group=dp.dp_group)
    return buf.cpu() if cpu_offload else buf


def _all_gather_fsdp_tensor(
    local_tensor: torch.Tensor, dp_mesh, *, cpu_offload: bool
) -> torch.Tensor:
    """All-gather a FSDP2-sharded tensor to reconstruct the full tensor.

    Handles uneven shards (e.g., a [1, 64] tensor split across 4 ranks gives
    [1, 64] on rank 0 and [0, 64] on ranks 1-3).

    Args:
        local_tensor: Local shard (FSDP2 uses Shard(0) by default).
        dp_mesh: The DP device mesh.
        cpu_offload: Move result to CPU.

    Returns:
        Full (unsharded) tensor.
    """
    world_size = dp_mesh.size()
    dp_group = dp_mesh.get_group()

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

    full = torch.cat(gathered, dim=0)
    return full.cpu() if cpu_offload else full


def _get_dp_mesh(model: nn.Module):
    """Extract the DP device mesh from FSDP2 params on the model."""
    for name, param in model.named_parameters():
        if _DTensor is not None and isinstance(param, _DTensor):
            return param._spec.mesh
    return None


# ---- Model state dict ----


def get_model_state_dict(
    model: nn.Module, *, cpu_offload: bool = True
) -> dict[str, torch.Tensor]:
    """Get full model state dict with both dedicated and FSDP2 parameters.

    Produces a state dict identical to what a single-GPU model would produce,
    compatible with ``torch.save``/``torch.load`` and HuggingFace checkpoints.

    For dedicated params: broadcasts ``_owned_data`` from each owner.
    For FSDP2 symmetric params: all-gathers sharded DTensors.

    Args:
        model: Model with ``dedicate_params`` and ``fully_shard`` applied.
        cpu_offload: Move tensors to CPU (default True, recommended for saving).

    Returns:
        Complete state dict with full (unsharded) tensors for all parameters.
    """
    dp_fqns = _compute_dedicated_fqns(model)
    dedicated_fqn_set = set(dp_fqns.values())
    dp_mesh = _get_dp_mesh(model)

    sd: dict[str, torch.Tensor] = {}

    # 1. Dedicated params: broadcast from owner.
    for dp, fqn in dp_fqns.items():
        sd[fqn] = _broadcast_from_owner(dp, cpu_offload=cpu_offload)

    # 2. FSDP2 symmetric params: all-gather sharded DTensors.
    for name, param in model.named_parameters():
        if name in dedicated_fqn_set:
            continue
        if _DTensor is not None and isinstance(param, _DTensor) and dp_mesh is not None:
            full = _all_gather_fsdp_tensor(
                param._local_tensor, dp_mesh, cpu_offload=cpu_offload
            )
            sd[name] = full
        else:
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

    # 2. Load dedicated params: owner copies from state_dict to _owned_data.
    for fqn, dp in fqn_to_dp.items():
        if fqn not in state_dict:
            continue
        if dp.is_owner:
            dp._owned_data.copy_(state_dict[fqn].to(dp._orig_dtype).to(dp.device))

    # 3. Load symmetric params: manually shard full tensors into FSDP2 DTensors.
    for name, param in model.named_parameters():
        if name in dedicated_fqn_set or name not in state_dict:
            continue
        full_tensor = state_dict[name]
        if _DTensor is not None and isinstance(param, _DTensor):
            local = param._local_tensor
            mesh = param._spec.mesh
            rank = mesh.get_local_rank()
            world_size = mesh.size()
            full_on_device = full_tensor.to(local.dtype).to(local.device)
            chunks = full_on_device.tensor_split(world_size, dim=0)
            local.copy_(chunks[rank])
        else:
            param.data.copy_(full_tensor.to(param.dtype).to(param.device))


# ---- Optimizer state dict ----


def get_optimizer_state_dict(
    model: nn.Module, optimizer: Any, *, cpu_offload: bool = True
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

    Returns:
        Optimizer state dict in DMuon format.
    """
    dp_fqns = _compute_dedicated_fqns(model)
    dedicated_fqn_set = set(dp_fqns.values())

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
            full_buf = _broadcast_state_tensor(buf, dp, cpu_offload=cpu_offload)
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
                fsdp_entry[key] = _all_gather_fsdp_tensor(
                    val, dp_mesh, cpu_offload=cpu_offload
                )
            else:
                # Scalar (e.g., step count)
                fsdp_entry[key] = val
        fsdp_state[fqn] = fsdp_entry

    # 3. Param group hyperparameters (without tensor refs).
    param_groups = []
    for group in optimizer.param_groups:
        pg = {k: v for k, v in group.items() if k != "params"}
        param_groups.append(pg)

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
                    rank = dp_mesh.get_local_rank()
                    world_size = dp_mesh.size()
                    local_ref = p._local_tensor if hasattr(p, "_local_tensor") else p.data
                    full_on_device = val.to(local_ref.dtype).to(local_ref.device)
                    chunks = full_on_device.tensor_split(world_size, dim=0)
                    opt_state[key] = chunks[rank].contiguous()
                else:
                    opt_state[key] = val
            optimizer.state[p] = opt_state

    # 3. Load param group hyperparameters.
    if "param_groups" in state_dict:
        for saved_pg, current_pg in zip(state_dict["param_groups"], optimizer.param_groups):
            for k, v in saved_pg.items():
                if k != "params":
                    current_pg[k] = v
