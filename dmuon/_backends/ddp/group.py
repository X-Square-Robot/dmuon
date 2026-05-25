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
from dmuon._core.dynamo import dynamo_disable
from dmuon._core.owner_rank import OwnerCoord

from dmuon._backends.fsdp2.group import (
    TPScatterState,
    _cached_tensor,
    _cached_tensor_list,
    _profile_range,
    _split_for_scatter,
    _tp_gather_buffer_reuse_enabled,
    _tp_scatter_buffer_reuse_enabled,
)

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
        self._muon_grad_ready_event: Optional[torch.cuda.Event] = None
        self._muon_grad_ready_refs: list[torch.Tensor] = []
        self._pending_reduce: list[
            tuple[Optional[torch.Tensor], list[DedicatedParamDDP]]
        ] = []

        # Post-step broadcast state
        self._post_step_broadcast_event: Optional[torch.cuda.Event] = None
        self._post_step_broadcast_state: Optional[PostStepBroadcastState] = None

        # TP scatter/gather state.  DDP+TP uses the same duck-typed surface as
        # the FSDP2 TP path so Muon can reuse its existing TP optimizer logic.
        self._tp_scatter_state: Optional[TPScatterState] = None
        self._tp_gather_event: Optional[torch.cuda.Event] = None
        self._tp_gather_refs: list[torch.Tensor] = []
        self._tp_gather_pending_full_grads: list[
            tuple[
                DedicatedParamDDP,
                list[torch.Tensor],
                torch.Tensor,
                int,
                Optional[torch.Tensor],
            ]
        ] = []
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

    @dynamo_disable
    def unshard(self, *, prefetch: bool = False) -> None:
        """DDP path: parameter is always live, nothing to allocate.

        ``DedicatedState`` shares the same forward-prefetch call path across
        DDP and FSDP groups.  DDP has no unshard work to prefetch, but accepts
        the keyword to keep the backend interface uniform.
        """

    @dynamo_disable
    def wait_for_unshard(self) -> None:
        """DDP path: no pending unshard event."""

    @dynamo_disable
    def reshard(self) -> None:
        """DDP path: never resharded — no storage to free."""

    @dynamo_disable
    def _backward_prefetch(self) -> None:
        """DDP path: no unshard work to prefetch."""

    @dynamo_disable
    def _record_post_forward(self) -> None:
        """Record this group's position in the forward order for post-step
        broadcast scheduling, mirroring DedicatedParamGroup.
        """
        post_forward_index = len(self.comm_ctx.post_forward_order)
        self.comm_ctx.post_forward_order.append(self)
        self._post_forward_indices.append(post_forward_index)

    # ---- gradient reduction -------------------------------------------------

    @dynamo_disable
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
                grad = p.local_grad_for_reduce(grad)
                if p._accumulated_grad is not None:
                    p._accumulated_grad.add_(grad.data)
                else:
                    p._accumulated_grad = grad.data.clone()
                p.clear_live_grad()
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
                    cur_local = p.local_grad_for_reduce(cur)
                    cur_local.data.add_(p._accumulated_grad)
                else:
                    p.set_live_grad_from_local(p._accumulated_grad.clone())
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
                    gdata = p.local_grad_for_reduce(grad).data.contiguous()
                    dist.reduce(
                        gdata,
                        dst=self._global_owner_ranks[p.owner_rank],
                        op=dist.ReduceOp.AVG,
                        group=dp_group,
                    )
                    p.clear_live_grad()
                    self._pending_reduce.append((gdata.view(-1), [p]))

        self._post_reduce_event = reduce_stream.record_event()

    @dynamo_disable
    def wait_for_reduce(
        self, stream: Optional[torch.cuda.Stream] = None
    ) -> Optional[torch.cuda.Event]:
        """Wait for pending reduces and save owner-side gradient."""
        if self._post_reduce_event is None:
            return None

        target_stream = stream if stream is not None else torch.cuda.current_stream()
        if stream is None:
            target_stream.wait_event(self._post_reduce_event)
        else:
            with torch.cuda.stream(stream):
                target_stream.wait_event(self._post_reduce_event)
        self._post_reduce_event = None

        stream_ctx = torch.cuda.stream(stream) if stream is not None else None
        if stream_ctx is None:
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
        else:
            with stream_ctx:
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

        self._muon_grad_ready_refs = [
            grad_buf
            for grad_buf, _plist in self._pending_reduce
            if grad_buf is not None
        ]
        self._pending_reduce = []
        self._muon_grad_ready_event = target_stream.record_event()
        return self._muon_grad_ready_event

    def wait_for_stage1_reduce(
        self, stream: Optional[torch.cuda.Stream] = None
    ) -> Optional[torch.cuda.Event]:
        """Compatibility hook for the FSDP2/HSDP Stage-1 drain protocol.

        DDP has a single reduce stage, so the buffer-safety wait and the full
        reduce wait are identical.
        """
        return self.wait_for_reduce(stream=stream)

    # ---- TP gather/scatter -------------------------------------------------

    def tp_gather_grads(
        self, *, wait_current_stream: bool = True
    ) -> Optional[torch.cuda.Event]:
        """Gather TP-local reduced grads to each parameter's TP owner.

        Only DP-owner ranks have ``_reduced_grad`` populated after the DP
        reduce.  Those ranks form exactly one TP group for the selected DP
        owner and therefore are the only ranks that enter the TP gather.
        """
        reduce_stream = self.comm_ctx.reduce_stream
        work: list[tuple[DedicatedParamDDP, torch.Tensor]] = []
        for p in self.params:
            if p.tp_group is None or p._reduced_grad is None:
                continue
            work.append((p, p._reduced_grad))

        if not work:
            self._tp_gather_event = None
            self._tp_gather_refs = []
            self._tp_gather_pending_full_grads = []
            return None

        if wait_current_stream:
            reduce_stream.wait_stream(torch.cuda.current_stream())

        grouped_work: list[
            tuple[dist.ProcessGroup, list[tuple[DedicatedParamDDP, torch.Tensor]]]
        ] = []
        for item in work:
            tp_group = item[0].tp_group
            for group, items in grouped_work:
                if group is tp_group:
                    items.append(item)
                    break
            else:
                grouped_work.append((tp_group, [item]))

        with torch.cuda.stream(reduce_stream):
            gather_refs: list[torch.Tensor] = []
            pending_full_grads: list[
                tuple[DedicatedParamDDP, list[torch.Tensor], torch.Tensor, int, Optional[torch.Tensor]]
            ] = []
            reuse_buffers = _tp_gather_buffer_reuse_enabled(self.comm_ctx)
            for tp_group, group_work in grouped_work:
                with dist._coalescing_manager(group=tp_group, device=self.device):
                    for p, local_grad in group_work:
                        tp_size = p.tp_group.size()
                        if p.is_tp_owner:
                            shard_dim = p.shard_dim if p.shard_dim is not None else 0
                            if reuse_buffers:
                                recv_bufs = _cached_tensor_list(
                                    p,
                                    "_tp_gather_recv_bufs",
                                    count=tp_size,
                                    shape=local_grad.shape,
                                    dtype=local_grad.dtype,
                                    device=local_grad.device,
                                )
                                full_grad_buf = _cached_tensor(
                                    p,
                                    "_tp_full_grad_buf",
                                    shape=p.full_shape,
                                    dtype=local_grad.dtype,
                                    device=local_grad.device,
                                )
                            else:
                                recv_bufs = [
                                    torch.empty_like(local_grad) for _ in range(tp_size)
                                ]
                                full_grad_buf = None
                            gather_refs.extend(recv_bufs)
                            gather_refs.append(local_grad)
                            dist.gather(
                                local_grad,
                                gather_list=recv_bufs,
                                dst=p._tp_owner_global_rank,
                                group=p.tp_group,
                            )
                            pending_full_grads.append(
                                (p, recv_bufs, local_grad, shard_dim, full_grad_buf)
                            )
                        else:
                            gather_refs.append(local_grad)
                            dist.gather(
                                local_grad,
                                gather_list=None,
                                dst=p._tp_owner_global_rank,
                                group=p.tp_group,
                            )
                            p._tp_full_grad = None
            self._tp_gather_pending_full_grads = pending_full_grads
            self._tp_gather_refs = gather_refs
        self._tp_gather_event = reduce_stream.record_event()
        self._muon_grad_ready_event = self._tp_gather_event
        return self._tp_gather_event

    def _materialize_tp_gathered_grads(self) -> None:
        pending = self._tp_gather_pending_full_grads
        if not pending:
            return
        with _profile_range("dmuon.ddp_tp_gather_materialize_full_grads"):
            for p, recv_bufs, local_grad, shard_dim, full_grad_buf in pending:
                recv_bufs[p.tp_group.rank()] = local_grad
                if full_grad_buf is None:
                    p._tp_full_grad = torch.cat(recv_bufs, dim=shard_dim)
                else:
                    torch.cat(recv_bufs, dim=shard_dim, out=full_grad_buf)
                    p._tp_full_grad = full_grad_buf
        self._tp_gather_pending_full_grads = []

    def wait_for_tp_gather(self) -> None:
        if self._tp_gather_event is None:
            return
        torch.cuda.current_stream().wait_event(self._tp_gather_event)
        self._materialize_tp_gathered_grads()
        self._tp_gather_event = None
        self._tp_gather_refs = []
        self._muon_grad_ready_refs = []
        self._muon_grad_ready_event = None

    def _tp_scatter_dispatch(self) -> Optional[list[torch.Tensor]]:
        work: list[DedicatedParamDDP] = [
            p
            for p in self.params
            if p.tp_group is not None and p._reduced_grad is not None
        ]
        if not work:
            return None

        bcast_stream = self.comm_ctx.replicate_broadcast_stream
        bcast_stream.wait_stream(torch.cuda.current_stream())
        bcast_stream.wait_stream(self.comm_ctx.reduce_stream)

        refs: list[torch.Tensor] = []
        updates: list[tuple[DedicatedParamDDP, torch.Tensor]] = []
        grouped_work: list[tuple[dist.ProcessGroup, list[DedicatedParamDDP]]] = []
        for p in work:
            for group, items in grouped_work:
                if group is p.tp_group:
                    items.append(p)
                    break
            else:
                grouped_work.append((p.tp_group, [p]))

        with torch.cuda.stream(bcast_stream):
            reuse_buffers = _tp_scatter_buffer_reuse_enabled(self.comm_ctx)
            for tp_group, group_work in grouped_work:
                with dist._coalescing_manager(group=tp_group, device=self.device):
                    for p in group_work:
                        shard_dim = p.shard_dim if p.shard_dim is not None else 0
                        if reuse_buffers:
                            recv_shard = _cached_tensor(
                                p,
                                "_tp_scatter_recv_buf",
                                shape=p._owned_data.shape,
                                dtype=p._owned_data.dtype,
                                device=p._owned_data.device,
                            )
                        else:
                            recv_shard = torch.empty_like(p._owned_data)
                        refs.append(recv_shard)
                        if p.is_tp_owner:
                            assert p._tp_full_delta is not None, (
                                f"{p.param_name}: TP owner has _tp_full_delta=None "
                                "— Muon._step_muon did not populate it."
                            )
                            refs.append(p._tp_full_delta)
                            split_buffers = None
                            if reuse_buffers and shard_dim != 0:
                                split_buffers = _cached_tensor_list(
                                    p,
                                    "_tp_scatter_split_bufs",
                                    count=p.tp_group.size(),
                                    shape=p._owned_data.shape,
                                    dtype=p._owned_data.dtype,
                                    device=p._owned_data.device,
                                )
                            splits = _split_for_scatter(
                                p._tp_full_delta,
                                p._orig_size[shard_dim],
                                dim=shard_dim,
                                out_buffers=split_buffers,
                            )
                            refs.extend(splits)
                            dist.scatter(
                                recv_shard,
                                scatter_list=splits,
                                src=p._tp_owner_global_rank,
                                group=p.tp_group,
                            )
                            update_shard = splits[p.tp_group.rank()]
                        else:
                            dist.scatter(
                                recv_shard,
                                scatter_list=None,
                                src=p._tp_owner_global_rank,
                                group=p.tp_group,
                            )
                            update_shard = recv_shard
                        refs.append(recv_shard)
                        updates.append((p, update_shard))

            for p, update_shard in updates:
                p._owned_data.mul_(p._tp_wd_factor).add_(update_shard)
                update_shard.record_stream(bcast_stream)
                p._tp_full_grad = None
                p._tp_full_delta = None
                p._reduced_grad = None
        return refs

    def tp_scatter_delta(self) -> None:
        refs = self._tp_scatter_dispatch()
        if refs is None:
            return
        event = self.comm_ctx.replicate_broadcast_stream.record_event()
        torch.cuda.current_stream().wait_event(event)
        self._tp_scatter_state = TPScatterState(refs=refs, event=event)

    def tp_scatter_delta_async(self) -> None:
        if self._tp_scatter_state is not None:
            group_name = getattr(self, "_debug_name", "<unknown>")
            raise RuntimeError(
                f"tp_scatter_delta_async[{group_name}]: previous event still pending; "
                "pre_forward_wait was not consumed before the next dispatch"
            )
        refs = self._tp_scatter_dispatch()
        if refs is None:
            return
        event = self.comm_ctx.replicate_broadcast_stream.record_event()
        self._tp_scatter_state = TPScatterState(refs=refs, event=event)

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
        tp_state = self._tp_scatter_state
        if tp_state is not None:
            # DDP+TP publishes in two stages: TP scatter first updates the
            # local TP shard, then DDP broadcasts that shard to peer replicas.
            # When TP scatter is truly async, the current stream no longer
            # carries this dependency, so order the DDP broadcast stream on
            # the TP scatter event explicitly.  HSDP gets this ordering by
            # using the same replicate_broadcast_stream for both stages.
            bcast_stream.wait_event(tp_state.event)
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
                    p.copy_owned_to_live()

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
            group_name = getattr(self, "_debug_name", "<unknown>")
            raise RuntimeError(
                f"post_step_broadcast_async[{group_name}]: previous event still pending; "
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

    @dynamo_disable
    def _pre_forward_wait(self) -> None:
        """Consume any pending post-step broadcast event before the next
        forward reads ``nn.Parameter.data``. No-op when IDLE.
        """
        tp_state = self._tp_scatter_state
        if tp_state is not None:
            torch.cuda.current_stream().wait_event(tp_state.event)
            self._tp_scatter_state = None

        state = self._post_step_broadcast_state
        if state is None:
            return
        torch.cuda.current_stream().wait_event(state.event)
        self._post_step_broadcast_state = None
