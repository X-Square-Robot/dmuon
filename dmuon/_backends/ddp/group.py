"""DedicatedParamGroupDDP: DDP-path variant of DedicatedParamGroup.

Reuses the layer-level hook infrastructure in :class:`dmuon.state.DedicatedState`
via duck typing — the hooks call ``unshard`` / ``reshard`` / ``reduce_grads``
as they do on the FSDP2 path, but on a DDP group ``unshard`` and ``reshard``
are intentional no-ops because the parameter is already live on every rank
(no packed buffer, no placeholder).

Lifecycle per step on a DDP group:

* forward: ``_pre_forward_wait`` consumes the previous step's post-step
  async broadcast event. ``unshard`` / ``wait_for_unshard`` are no-op;
  ``_record_post_forward`` / ``_post_backward_fired`` reset run as normal.
* backward: ``reduce_grads`` dispatches one-stage ``dist.reduce`` (op=AVG)
  to each param's owner on ``dp_group``. Coalesced via
  ``dist._coalescing_manager``.
* optim.step: the owner runs Newton-Schulz on ``_owned_data``; non-owner
  ``_owned_data`` stays stale until the broadcast below.
* post-step: ``post_step_broadcast_sync`` broadcasts each owner's updated
  ``_owned_data`` back to all ranks in ``dp_group`` and copies the result
  into the live ``nn.Parameter.data``. Coalesced per group.

The async variant (``post_step_broadcast_async`` +
``_pre_forward_wait``) mirrors the FSDP2-path HSDP scheduler so the
broadcast can hide inside the next forward's compute.
"""

from collections import defaultdict
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist

from dmuon._core.comm import DedicatedCommContext
from dmuon._core.owner_rank import OwnerCoord

from .param import DedicatedParamDDP


class PostStepBroadcastState(NamedTuple):
    """State kept alive across an in-flight post-step broadcast.

    Mirrors :class:`dmuon.group.ReplicateBroadcastState` — holds an owning
    reference to one ``_owned_data`` tensor so the allocator arena the
    NCCL kernel reads is not freed while the kernel is in flight, plus
    the event that downstream ``_pre_forward_wait`` consumes.
    """

    broadcast_input: torch.Tensor
    event: torch.cuda.Event


class DedicatedParamGroupDDP:
    """Manages communication for DDP-path dedicated parameters in one layer.

    Interface-compatible with :class:`dmuon.group.DedicatedParamGroup` so
    ``DedicatedState`` can hook both via duck typing. Methods that do not
    apply to the DDP path (``unshard``, ``wait_for_unshard``, ``reshard``,
    ``_backward_prefetch``) are implemented as no-ops.
    """

    def __init__(
        self,
        params: list[DedicatedParamDDP],
        comm_ctx: DedicatedCommContext,
    ):
        self.params: list[DedicatedParamDDP] = params
        self.comm_ctx = comm_ctx
        self.device = params[0].device if params else torch.device("cuda")

        # Pre-group by owner so coalesced broadcasts/reduces can fuse
        # per-owner work into one NCCL kernel.
        self._by_owner: dict[OwnerCoord, list[DedicatedParamDDP]] = defaultdict(list)
        for p in params:
            self._by_owner[p.owner_rank].append(p)

        # no_sync gate — mirrors DedicatedParamGroup.reduce_grads_enabled.
        self.reduce_grads_enabled: bool = True

        # Event / pending-reduce machinery
        self._post_reduce_event: Optional[torch.cuda.Event] = None
        self._pending_reduce: list[tuple[Optional[torch.Tensor], list[DedicatedParamDDP]]] = []

        # Post-step broadcast state
        self._post_step_broadcast_event: Optional[torch.cuda.Event] = None
        self._post_step_broadcast_state: Optional[PostStepBroadcastState] = None

        # Forward-order / fast-path bookkeeping (used by DedicatedState)
        self._post_forward_indices: list[int] = []
        self._post_backward_fired: bool = False

        # Cached metadata
        self._dp_group: Optional[dist.ProcessGroup] = (
            params[0].dp_group if params else None
        )
        self._comm_dtype: Optional[torch.dtype] = (
            (params[0]._compute_dtype or params[0]._orig_dtype) if params else None
        )
        self._total_numel_by_owner: dict[OwnerCoord, int] = {
            owner: sum(p.numel for p in owner_params)
            for owner, owner_params in self._by_owner.items()
        }
        if self._dp_group is not None:
            self._global_owner_ranks: dict[OwnerCoord, int] = {
                owner: dist.get_global_rank(self._dp_group, owner[0])
                for owner in self._by_owner
            }
        else:
            self._global_owner_ranks = {}

    # ---- interface parity with DedicatedParamGroup (no-ops) -----------------

    def unshard(self) -> None:
        """DDP path: parameter is always live, nothing to allocate."""

    def wait_for_unshard(self) -> None:
        """DDP path: no pending unshard event."""

    def reshard(self) -> None:
        """DDP path: never resharded — no storage to free."""

    def _backward_prefetch(self) -> None:
        """DDP path: no unshard work to prefetch."""

    def _record_post_forward(self) -> None:
        """Record this group's position in the forward order for post-step
        broadcast scheduling, mirroring DedicatedParamGroup.
        """
        post_forward_index = len(self.comm_ctx.post_forward_order)
        self.comm_ctx.post_forward_order.append(self)
        self._post_forward_indices.append(post_forward_index)

    # ---- gradient reduction -------------------------------------------------

    def reduce_grads(self) -> None:
        """Dispatch one-stage reduce-to-owner on ``dp_group``.

        Reads ``.grad`` from each param's live ``nn.Parameter`` (via
        ``DedicatedParamDDP._orig_param``) and issues ``dist.reduce`` with
        ``op=AVG``. All reduces are coalesced into one NCCL kernel.
        Does NOT wait — call :meth:`wait_for_reduce` to synchronize.

        ``reduce_grads_enabled=False`` (the ``no_sync`` path): accumulate
        the full local gradient into ``_accumulated_grad`` without any
        collective, matching FSDP2 path semantics.
        """
        if not self.reduce_grads_enabled:
            # no_sync: accumulate full local gradients on every rank.
            for p in self.params:
                grad = p._orig_param.grad
                if grad is None:
                    continue
                if p._accumulated_grad is not None:
                    p._accumulated_grad.add_(grad.data)
                else:
                    p._accumulated_grad = grad.data.clone()
                p._orig_param.grad = None
            return

        # Flush any pending reduce from a previous backward (gradient
        # accumulation without optimizer.step).
        if self._post_reduce_event is not None:
            self.wait_for_reduce()

        # Merge any accumulated gradients from prior no_sync micro-batches.
        for p in self.params:
            if p._accumulated_grad is not None:
                cur = p._orig_param.grad
                if cur is not None:
                    cur.data.add_(p._accumulated_grad)
                else:
                    p._orig_param.grad = p._accumulated_grad.clone()
                p._accumulated_grad = None

        reduce_stream = self.comm_ctx.reduce_stream
        reduce_stream.wait_stream(torch.cuda.current_stream())

        self._pending_reduce = []
        dp_group = self._dp_group

        with torch.cuda.stream(reduce_stream):
            with dist._coalescing_manager(group=dp_group, device=self.device):
                for p in self.params:
                    grad = p._orig_param.grad
                    if grad is None:
                        continue
                    gdata = grad.data.contiguous()
                    dist.reduce(
                        gdata,
                        dst=self._global_owner_ranks[p.owner_rank],
                        op=dist.ReduceOp.AVG,
                        group=dp_group,
                    )
                    p._orig_param.grad = None
                    self._pending_reduce.append((gdata.view(-1), [p]))

        self._post_reduce_event = reduce_stream.record_event()

    def wait_for_reduce(self) -> None:
        """Wait for pending reduces and save owner-side gradient."""
        if self._post_reduce_event is None:
            return
        current_stream = torch.cuda.current_stream()
        current_stream.wait_event(self._post_reduce_event)
        self._post_reduce_event = None

        for grad_buf, plist in self._pending_reduce:
            if grad_buf is None:
                continue
            p = plist[0]
            if not p.is_owner:
                continue
            new_grad = grad_buf.view(p._orig_size)
            if p._reduced_grad is not None:
                p._reduced_grad.add_(new_grad)
            else:
                p._reduced_grad = new_grad.clone()

        self._pending_reduce = []

    # ---- post-step broadcast (owner → all ranks on dp_group) ----------------

    def post_step_broadcast_sync(self) -> None:
        """Broadcast each owner's updated ``_owned_data`` to every rank on
        ``dp_group``; every rank ``copy_`` s the result into its live
        ``nn.Parameter.data``.

        Coalesced into one NCCL kernel per group. Runs on the shared
        ``broadcast_stream`` so it does not block the default compute
        stream until :meth:`wait_for_post_step_broadcast` is called.
        """
        if self._dp_group is None:
            return

        bcast_stream = self.comm_ctx.broadcast_stream
        bcast_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(bcast_stream):
            with dist._coalescing_manager(group=self._dp_group, device=self.device):
                for p in self.params:
                    dist.broadcast(
                        p._owned_data,
                        src=p._owner_global_rank,
                        group=self._dp_group,
                    )
            # Copy-back into live Parameter must also run on bcast_stream
            # so it orders after the broadcast and before the event record.
            with torch.no_grad():
                for p in self.params:
                    p._orig_param.data.copy_(p._owned_data)

        self._post_step_broadcast_event = bcast_stream.record_event()

    def wait_for_post_step_broadcast(self) -> None:
        """Block current stream until the post-step broadcast is visible.

        Must be called before the next forward reads ``nn.Parameter.data``.
        No-op when no dispatch is pending.
        """
        if self._post_step_broadcast_event is None:
            return
        torch.cuda.current_stream().wait_event(self._post_step_broadcast_event)
        self._post_step_broadcast_event = None

    # ---- async post-step broadcast -----------------------------------------

    def post_step_broadcast_async(self) -> None:
        """Async variant used by group-pipelined post-step scheduling.

        The optimizer may call this immediately after the group's owner-side
        Muon update.  The event is consumed by this group's next
        ``_pre_forward_wait`` before the live parameter is read.
        """
        if self._post_step_broadcast_state is not None:
            raise RuntimeError(
                "post_step_broadcast_async: previous event still pending; "
                "pre_forward_wait was not consumed before the next dispatch"
            )
        self.post_step_broadcast_sync()
        # Move event ownership to the async-state slot so
        # _pre_forward_wait is the consumer (matches FSDP2-path contract).
        evt = self._post_step_broadcast_event
        self._post_step_broadcast_event = None
        if evt is None or not self.params:
            return
        # Pin any one ``_owned_data`` ref until the event is consumed —
        # mirrors ReplicateBroadcastState's allocator-pinning role.
        self._post_step_broadcast_state = PostStepBroadcastState(
            broadcast_input=self.params[0]._owned_data,
            event=evt,
        )

    def _pre_forward_wait(self) -> None:
        """Consume any pending post-step broadcast event before the next
        forward reads ``nn.Parameter.data``. No-op when IDLE.
        """
        state = self._post_step_broadcast_state
        if state is None:
            return
        torch.cuda.current_stream().wait_event(state.event)
        self._post_step_broadcast_state = None

    # ---- fallback / profile stubs (match DedicatedParamGroup interface) ----

    def _update_replicate_fallback(self) -> None:
        """No fallback state machine on the DDP path (P1). Stub kept so
        ``utils.update_replicate_fallback`` can iterate both group types
        uniformly.
        """

    def reset_replicate_fallback(self) -> None:
        """Stub — matches DedicatedParamGroup interface."""
