"""Utility functions."""

from contextlib import contextmanager
from typing import Optional

import torch.nn as nn

from ._backends.ddp import DedicatedParamGroupDDP
from ._backends.fsdp2 import DedicatedParam
from ._core.comm import DedicatedCommContext
from ._core.owner_rank import OwnerRankLike, normalize_owner_rank


def get_dedicated_params(model: nn.Module) -> list[DedicatedParam]:
    """Collect all DedicatedParam instances from a model."""
    result = []
    for module in model.modules():
        if hasattr(module, "_dedicated_state"):
            result.extend(module._dedicated_state.group.params)
    return result


def get_owned_params(model: nn.Module, rank: OwnerRankLike) -> list[DedicatedParam]:
    """Collect DedicatedParam instances owned by a specific rank.

    Accepts either a plain shard ``int`` (1D legacy form, matched against
    ``owner_shard``) or a ``(shard, replicate)`` tuple (matched against the
    full ``owner_rank`` coord).
    """
    coord = normalize_owner_rank(rank)
    return [p for p in get_dedicated_params(model) if p.owner_rank == coord]


def get_comm_ctx(model: nn.Module) -> Optional[DedicatedCommContext]:
    """Get the DedicatedCommContext from a model, if it exists."""
    return getattr(model, "_dedicated_comm_ctx", None)


def wait_all_reduces(model: nn.Module) -> None:
    """Wait for all pending gradient reduces to complete and unpack results.

    Call this after loss.backward() and before the optimizer step.
    Gradient reduces are dispatched asynchronously during backward to overlap
    with backward computation. This function waits for them to finish and
    makes ``_reduced_grad`` available on each owner's DedicatedParam.

    T2: after the DP reduce pipeline resolves, kick off the TP gather (no-op
    for groups with no TP-sharded params) so every TP owner has its
    ``_tp_full_grad`` ready for the optimizer step.  The gather is
    dispatched on ``reduce_stream`` — see
    ``DedicatedParamGroup.tp_gather_grads``.
    """
    for module in model.modules():
        if hasattr(module, "_dedicated_state"):
            g = module._dedicated_state.group
            g.wait_for_reduce()
            gather = getattr(g, "tp_gather_grads", None)
            if gather is not None:
                gather()


def _iter_groups(model: nn.Module):
    """Yield every DedicatedParamGroup attached to ``model`` once."""
    for module in model.modules():
        if hasattr(module, "_dedicated_state"):
            yield module._dedicated_state.group


def _ordered_post_step_groups(model: nn.Module) -> list:
    """Return post-step groups in next-forward priority order.

    The order follows ``comm_ctx.post_forward_order`` when available and
    falls back to model-walk order for the first iteration or skipped modules.
    Every rank must use the same deterministic order because these groups
    enqueue NCCL collectives.
    """
    comm_ctx = getattr(model, "_dedicated_comm_ctx", None)
    order: list = []
    seen: set = set()
    if comm_ctx is not None:
        for g in comm_ctx.post_forward_order:
            gid = id(g)
            if gid not in seen:
                seen.add(gid)
                order.append(g)
    for g in _iter_groups(model):
        gid = id(g)
        if gid not in seen:
            seen.add(gid)
            order.append(g)
    return order


def _dispatch_post_step_sync(g) -> None:
    """Dispatch the post-step broadcast for one group (sync variant).

    FSDP2 path:
      1. (T2b) ``tp_scatter_delta`` — fan the full-matrix NS update back
         to each DP-owner TP shard (no-op when the group has no TP-sharded
         params).  Must run BEFORE the replicate broadcast so every DP
         owner's ``_owned_data`` carries the TP-correct update.
      2. ``replicate_broadcast_sync`` — fans ``_owned_data`` from the
         global owner to replicate peers along the HSDP replicate axis
         (no-op in 1D shard-only mode).

    DDP path: fans ``_owned_data`` from the owner across the DP group.
    """
    if isinstance(g, DedicatedParamGroupDDP):
        g.post_step_broadcast_sync()
    else:
        scatter = getattr(g, "tp_scatter_delta", None)
        if scatter is not None:
            scatter()
        g.replicate_broadcast_sync()


def _dispatch_post_step_async(g) -> None:
    """Dispatch the post-step publish path for one group (async variant).

    FSDP2/HSDP+TP groups run TP scatter before replicate broadcast on the
    same stream, so replicate peers only see TP-correct ``_owned_data``.
    DDP groups fan owner data across the DP group and leave the event for
    the next pre-forward wait.
    """
    if isinstance(g, DedicatedParamGroupDDP):
        g.post_step_broadcast_async()
    else:
        # T2d: async TP scatter (O2 overlap per tp_design.md §4.2) —
        # dispatch on ``replicate_broadcast_stream`` without waiting; the
        # cross-call event is drained on the next ``_pre_forward_wait``.
        scatter_async = getattr(g, "tp_scatter_delta_async", None)
        if scatter_async is not None:
            scatter_async()
        g.replicate_broadcast_async()


def _wait_post_step(g) -> None:
    if isinstance(g, DedicatedParamGroupDDP):
        g.wait_for_post_step_broadcast()
    else:
        g.wait_for_replicate_broadcast()


def broadcast_all_updates(model: nn.Module) -> None:
    """Sync post-step broadcast of updated ``_owned_data``.

    Dispatches on every dedicated group, then drains. FSDP2 path fans
    across the HSDP replicate axis (no-op for 1D); DDP path fans across
    the DP group.
    """
    groups = list(_iter_groups(model))
    for g in groups:
        _dispatch_post_step_sync(g)
    for g in groups:
        _wait_post_step(g)


def broadcast_all_updates_async(model: nn.Module) -> None:
    """Phase C.2 (async): dispatch the post-step replicate broadcasts on
    the dedicated replicate stream and return without waiting.

    The wait is consumed per-group by the next forward iteration's
    ``_pre_forward_wait`` hook (see ``DedicatedState._pre_forward``),
    letting each layer's replicate broadcast hide behind the compute of
    the prior layers.

    Group dispatch order follows the recorded forward order from the
    previous iteration — Phase C.5 priority scheduling that mirrors
    FSDP2's ``post_forward_order`` usage for backward prefetch
    (``_fsdp_param_group.py:469-474``).  First epoch falls through to
    the model-walk order.

    Fallback-to-sync is per-group and transparent: a group with
    ``_replicate_sync_fallback=True`` resolves inside
    ``replicate_broadcast_async`` (dispatch + wait inline); the caller
    sees no pending state for it.
    """
    for g in _ordered_post_step_groups(model):
        _dispatch_post_step_async(g)


def update_replicate_fallback(model: nn.Module) -> None:
    """Phase C.4 + T2e: advance BOTH the replicate-broadcast and TP-scatter
    async→sync fallback state machines on every group.  Cheap no-op when
    the profile env var is off (every group's wait-time sample stays 0.0
    and the state machine short-circuits)."""
    for g in _iter_groups(model):
        update_fn = getattr(g, "_update_replicate_fallback", None)
        if update_fn is not None:
            update_fn()
        tp_update = getattr(g, "_update_tp_scatter_fallback", None)
        if tp_update is not None:
            tp_update()


def reset_replicate_fallback(model: nn.Module) -> None:
    """Clear the Phase C.4 async→sync fallback flag on every group.

    Intended for users who want to re-enable async after fixing a slow-IB
    condition.  Safe to call from the training loop.
    """
    for g in _iter_groups(model):
        g.reset_replicate_fallback()


def wait_all_post_step_broadcasts(model: nn.Module) -> None:
    """Alias for :func:`wait_all_replicate_broadcasts`.

    Exposed under a more path-neutral name so DDP-path users do not have
    to think about HSDP's ``replicate`` terminology. Both FSDP2-path
    groups (HSDP replicate broadcast) and DDP-path groups (post-step
    broadcast across the DP group) are drained.
    """
    wait_all_replicate_broadcasts(model)


def wait_all_replicate_broadcasts(model: nn.Module) -> None:
    """Drain every group's pending async replicate broadcast.

    Phase C.3 safety net for any code path that needs consistent
    ``_owned_data`` without going through the forward hook — e.g.
    ``get_model_state_dict`` / ``get_optimizer_state_dict`` in
    :mod:`dmuon.checkpoint` call this before reading from global owners.

    In sync / 1D mode this is a cheap no-op: every group is already IDLE.
    """
    for g in _iter_groups(model):
        # Drain both the sync event (if any) and the async state.
        _wait_post_step(g)
        g._pre_forward_wait()


@contextmanager
def no_sync(model: nn.Module):
    """Context manager to disable gradient reduction for gradient accumulation.

    Within this context, backward passes skip reduce communication and
    accumulate gradients locally. On the next backward outside this context,
    the accumulated gradients are merged and reduced normally.

    This also disables FSDP2's gradient sync for symmetric parameters.

    Usage::

        for i, batch in enumerate(dataloader):
            ctx = dmuon.no_sync(model) if (i + 1) % accum_steps != 0 else nullcontext()
            with ctx:
                loss = model(batch).loss / accum_steps
                loss.backward()
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    """
    # Disable reduce for dedicated params (both FSDP2 and DDP paths)
    groups = []
    for module in model.modules():
        if hasattr(module, "_dedicated_state"):
            group = module._dedicated_state.group
            groups.append(group)
            group.reduce_grads_enabled = False
    # Disable reduce-scatter for FSDP2 symmetric params
    if hasattr(model, "set_requires_gradient_sync"):
        model.set_requires_gradient_sync(False)
    # Disable all-reduce on the DDP-path replicated group, if present.
    rep_group = getattr(model, "_replicated_group", None)
    rep_prev: Optional[bool] = None
    if rep_group is not None:
        rep_prev = rep_group._sync_enabled
        rep_group._sync_enabled = False
    try:
        yield
    finally:
        for group in groups:
            group.reduce_grads_enabled = True
        if hasattr(model, "set_requires_gradient_sync"):
            model.set_requires_gradient_sync(True)
        if rep_group is not None and rep_prev is not None:
            rep_group._sync_enabled = rep_prev
