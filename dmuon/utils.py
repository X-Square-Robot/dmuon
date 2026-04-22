"""Utility functions."""

from contextlib import contextmanager
from typing import Optional

import torch.nn as nn

from ._owner_rank import OwnerRankLike, normalize_owner_rank
from .comm import DedicatedCommContext
from .param import DedicatedParam


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
    """
    for module in model.modules():
        if hasattr(module, "_dedicated_state"):
            module._dedicated_state.group.wait_for_reduce()


def _iter_groups(model: nn.Module):
    """Yield every DedicatedParamGroup attached to ``model`` once."""
    for module in model.modules():
        if hasattr(module, "_dedicated_state"):
            yield module._dedicated_state.group


def broadcast_all_updates(model: nn.Module) -> None:
    """Phase B.2 (sync): fan the post-step ``_owned_data`` from each global
    owner to its replicate peers (HSDP only; no-op in 1D mode).

    The dispatch and wait phases are separated so the NCCL collectives can
    pipeline before we start blocking on events.  Phase C's async path
    uses :func:`broadcast_all_updates_async` + per-layer wait via
    ``_pre_forward_wait`` instead.
    """
    groups = list(_iter_groups(model))
    for g in groups:
        g.replicate_broadcast_sync()
    for g in groups:
        g.wait_for_replicate_broadcast()


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
    comm_ctx = getattr(model, "_dedicated_comm_ctx", None)
    order: list = []
    seen: set = set()
    if comm_ctx is not None:
        for g in comm_ctx.post_forward_order:
            gid = id(g)
            if gid not in seen:
                seen.add(gid)
                order.append(g)
    # Append groups not observed in post-forward order (first-epoch, or
    # groups whose layer did not run in the last forward).
    for g in _iter_groups(model):
        if id(g) not in seen:
            seen.add(id(g))
            order.append(g)
    for g in order:
        g.replicate_broadcast_async()


def update_replicate_fallback(model: nn.Module) -> None:
    """Phase C.4: advance the async→sync fallback state machine on every
    group.  Cheap no-op when the profile env var is off (every group's
    ``_last_replicate_wait_us`` stays at 0.0)."""
    for g in _iter_groups(model):
        g._update_replicate_fallback()


def reset_replicate_fallback(model: nn.Module) -> None:
    """Clear the Phase C.4 async→sync fallback flag on every group.

    Intended for users who want to re-enable async after fixing a slow-IB
    condition.  Safe to call from the training loop.
    """
    for g in _iter_groups(model):
        g.reset_replicate_fallback()


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
        g.wait_for_replicate_broadcast()
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
    # Disable reduce for dedicated params
    groups = []
    for module in model.modules():
        if hasattr(module, "_dedicated_state"):
            group = module._dedicated_state.group
            groups.append(group)
            group.reduce_grads_enabled = False
    # Disable reduce-scatter for FSDP2 symmetric params
    if hasattr(model, "set_requires_gradient_sync"):
        model.set_requires_gradient_sync(False)
    try:
        yield
    finally:
        for group in groups:
            group.reduce_grads_enabled = True
        if hasattr(model, "set_requires_gradient_sync"):
            model.set_requires_gradient_sync(True)
