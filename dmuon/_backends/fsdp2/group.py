"""DedicatedParamGroup: manages communication for dedicated params in one layer.

Uses dedicated CUDA streams for broadcast/reduce (analogous to FSDP2's
all_gather_stream / reduce_scatter_stream) and CUDA events for GPU-side
synchronization instead of CPU-blocking work.wait().
"""

import os
from collections import defaultdict
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist

from dmuon._core.comm import DedicatedCommContext
from dmuon._core.owner_rank import OwnerCoord

from .param import DedicatedParam

try:
    from torch.distributed.tensor import DTensor as _DTensor
except ImportError:
    _DTensor = None


class ReplicateReduceState(NamedTuple):
    """State kept alive across the Stage-2 replicate reduce.

    Mirrors FSDP2's ``AllReduceState`` (``_fsdp_param_group.py:115-117``): we
    need to hold a reference to the input tensor + event until the end of
    backward so the reduce stream's allocation is not freed while the NCCL
    kernel is in flight.  ``wait_for_reduce`` consumes the event and
    releases the tuple.
    """

    replicate_input: torch.Tensor
    event: Optional[torch.cuda.Event]


# Phase C.4 fallback tuning — see ``hsdp_native_phaseC_plan.md §8``.
# Exposed at module scope so tests + users can tweak the thresholds without
# monkey-patching inside the group object.
REPLICATE_WAIT_THRESHOLD_US: float = 100.0
REPLICATE_FALLBACK_CONSECUTIVE_STEPS: int = 3


class TPScatterState(NamedTuple):
    """State kept alive across the async post-step TP scatter.

    T2d analogue of :class:`ReplicateBroadcastState`: the scatter is
    dispatched on ``replicate_broadcast_stream`` and the caller returns
    without waiting; the event is consumed on the next iteration's
    :meth:`_pre_forward_wait` hook (cross-call event-chain pattern, same
    as FSDP2's ``AllGatherState``).

    **Do NOT pin ``recv_shards`` Python-refs across iterations.**  The
    dispatch body already calls ``recv_shard.record_stream(bcast_stream)``
    which is the correct allocator-safety primitive — the caching
    allocator keeps the memory reserved until the scatter kernel
    completes on ``bcast_stream``.  Holding Python refs in a state tuple
    on top of that is NOT belt-and-suspenders: it changes the allocator's
    free-block pattern across iterations, which shifts the addresses of
    subsequent NCCL transport buffers and produces a different (but
    self-consistent) floating-point reduction trajectory than the sync
    path.  The ``recv_shards`` field therefore always carries an EMPTY
    list — we keep it for schema symmetry with ``ReplicateBroadcastState``.
    See ``docs/internal/research/tp_alignment_report.md`` (Phase B/C,
    2026-04-24) for the diagnostic trail.
    """

    recv_shards: list[torch.Tensor]
    event: torch.cuda.Event


class ReplicateBroadcastState(NamedTuple):
    """State kept alive across the async post-step replicate broadcast.

    Phase C analogue of FSDP2's ``AllGatherState`` (``_fsdp_param_group.py:
    105-107``): when the broadcast is dispatched on the default-priority
    replicate stream and the caller returns, the input tensor allocation
    could be freed before the NCCL kernel observes it.  Holding an owning
    reference here prevents that, and the event gets consumed by the next
    iteration's ``_pre_forward_wait`` hook — same cross-call event-chain
    pattern FSDP2 uses (``_fsdp_param_group.py:358-362``).

    The tuple carries just one ``_owned_data`` ref per group; all params in
    the group participate in the same ``dist._coalescing_manager`` batch,
    so a single ref suffices to pin the allocator arena the kernel reads.
    """

    replicate_input: torch.Tensor
    event: torch.cuda.Event


class DedicatedParamGroup:
    """Manages all dedicated parameters within one layer.

    Parameters with the same owner are packed into one broadcast/reduce call.
    All communication runs on dedicated CUDA streams from DedicatedCommContext.
    """

    def __init__(self, params: list[DedicatedParam], comm_ctx: DedicatedCommContext):
        self.params = params
        self.comm_ctx = comm_ctx
        self.device = params[0].device if params else torch.device("cuda")

        # Pre-group by owner so packed broadcasts / reduces can coalesce
        # all of an owner's params into a single NCCL call.  Phase A extends
        # the key from a 1D shard rank to a 2D ``(shard, replicate)`` coord;
        # in shard-only mode every entry has replicate=0 so the grouping
        # collapses back to the previous 1D buckets.
        self._by_owner: dict[OwnerCoord, list[DedicatedParam]] = defaultdict(list)
        for p in params:
            self._by_owner[p.owner_rank].append(p)

        # Two independent gradient-reduce gates (mirrors FSDP2's
        # ``reduce_grads`` + ``all_reduce_grads`` pair, see
        # ``_fsdp_param_group.py:185-189``).  ``reduce_grads_enabled`` skips
        # the entire pipeline (used by DMuon's ``no_sync`` ctx manager);
        # ``replicate_grads_enabled`` only skips the Stage-2 replicate reduce
        # while still performing the Stage-1 shard reduce — enabling HSDP
        # grad-accumulation semantics where a partial accumulator is held
        # across micro-batches.
        self.reduce_grads_enabled: bool = True
        self.replicate_grads_enabled: bool = True

        # Event-based synchronization (replaces work.wait()).  ``_post_reduce_event``
        # marks the end of the reduce pipeline — shard-only in Phase A, shard+replicate
        # in Phase B (mirrors FSDP2's ``_post_reduce_event``; see
        # ``_fsdp_param_group.py:213``).  ``_replicate_reduce_state`` keeps the
        # Stage-2 input + event alive until ``wait_for_reduce`` runs, mirroring
        # ``AllReduceState`` in ``_fsdp_param_group.py:115-117``.
        self._broadcast_event: Optional[torch.cuda.Event] = None
        self._post_reduce_event: Optional[torch.cuda.Event] = None
        self._replicate_reduce_state: Optional[ReplicateReduceState] = None
        # Phase B.2 / C.1: replicate-dim post-step broadcast state.
        #
        # Two event/state fields coexist — they correspond to two code paths:
        #   * ``_replicate_broadcast_event`` (bare event): used by the
        #     Phase B sync variant ``replicate_broadcast_sync`` /
        #     ``wait_for_replicate_broadcast``.  Caller dispatches and
        #     immediately waits, so a single event suffices.
        #   * ``_replicate_broadcast_state`` (NamedTuple): used by the
        #     Phase C async variant.  The event lives until the next
        #     iteration's ``_pre_forward_wait`` consumes it, and the
        #     tuple also pins the ``_owned_data`` ref — mirrors FSDP2's
        #     ``AllGatherState`` (``_fsdp_param_group.py:105-107``).
        self._replicate_broadcast_event: Optional[torch.cuda.Event] = None
        self._replicate_broadcast_state: Optional[ReplicateBroadcastState] = None
        # Per-group sync fallback flag.  Flipped by the Phase C.4 fallback
        # monitor after N consecutive slow waits; once set, all subsequent
        # async dispatches short-circuit to the Phase B sync path.  Reset
        # only by user code via ``reset_replicate_fallback()``.
        self._replicate_sync_fallback: bool = False
        # Last observed wait blocked-time on the default stream, set by
        # the Phase C.4/C.7 profile path.  0.0 means "no timing in the
        # last iteration" (profiler disabled or not yet consumed).
        self._last_replicate_wait_us: float = 0.0
        # Fallback state machine — count of consecutive slow waits.
        self._replicate_slow_wait_count: int = 0

        # T2d: async TP scatter state — mirrors ``_replicate_broadcast_state``
        # but covers the per-TP-group ``dist.scatter`` dispatched by
        # ``tp_scatter_delta_async``.  Consumed together with the replicate
        # broadcast state in ``_pre_forward_wait``.  See ``tp_design.md``
        # §4.2 (O2) + §5 fallback.
        self._tp_scatter_state: Optional[TPScatterState] = None
        self._tp_sync_fallback: bool = False
        self._last_tp_scatter_wait_us: float = 0.0
        self._tp_scatter_slow_wait_count: int = 0

        # Partial accumulator across ``no_sync`` micro-batches (per-param, on
        # the shard-owner rank).  Set during grad-accum when
        # ``replicate_grads_enabled`` is False; flushed into the next Stage-2
        # reduce when the gate flips back.  Mirrors FSDP2's
        # ``_partial_reduce_output`` (``_fsdp_param_group.py:220``).
        self._partial_reduce_by_param: dict[int, torch.Tensor] = {}

        # Deferred reduce unpack (fixes data race in old _packed_reduce)
        self._pending_reduce: list[tuple[Optional[torch.Tensor], list[DedicatedParam]]] = []

        # Prefetch tracking (mirrors FSDPParamGroup._post_forward_indices)
        self._post_forward_indices: list[int] = []

        # Unsharded state tracking (for reshard_after_forward=False)
        self._is_unsharded: bool = False

        # Post-backward fast-path tracking: reset in _pre_forward, set True when
        # reduce+reshard runs (either via _DedicatedPostBackward.backward fast path
        # or via the autograd-engine root callback). Used by the fallback to skip
        # groups that already ran.
        self._post_backward_fired: bool = False

        # NOTE: Phase 2 removed _forward_time_params. It used to snapshot the
        # forward-time _unsharded_param references because reshard + re-unshard
        # in pre_backward created a NEW Parameter and autograd's .grad went to
        # the old one. With Parameter reuse (persistent _unsharded_param whose
        # storage resizes 0↔full), the SAME Parameter object is live across
        # the forward/backward cycle, so autograd writes .grad directly onto
        # it. No snapshot, no grad-transfer step needed.

        # Cached per-group metadata (FSDP2 alignment phase 1 — previously
        # recomputed in every unshard / reduce_grads call).
        self._dp_group: Optional[dist.ProcessGroup] = (
            params[0].dp_group if params else None
        )
        self._comm_dtype: Optional[torch.dtype] = (
            (params[0]._compute_dtype or params[0]._orig_dtype) if params else None
        )
        # Map {owner_coord → global rank} for all owners represented in params.
        # ``dist.get_global_rank`` is always called with the shard coordinate:
        # ``dp_group`` is the shard group (in Phase B the replicate coord is
        # handled separately via ``replicate_group``).
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

        # Phase 2: Persistent per-owner packed broadcast buffer.
        # One buffer per owner, shared as storage across all params that owner
        # holds. Each DedicatedParam's _unsharded_param is installed as an
        # as_strided view into its owner's buffer (see bind_to_packed_buffer).
        # Storage is resized 0↔full via alloc_storage/free_storage on the
        # packed buffer itself — individual Parameter views automatically see
        # the resize since they share the underlying Storage object.
        #
        # Phase 3: precompute per-owner copy-in dst views into packed buf.
        # Each unshard's owner copy-in uses ``torch._foreach_copy_`` (one
        # Python dispatch + one fused kernel) instead of N separate ``.copy_``
        # calls. dst views survive ``free_storage`` → ``alloc_storage`` because
        # they share the packed buf's Storage object (resize is in-place).
        from dmuon._core.internal_utils import free_storage
        self._packed_buf_by_owner: dict[OwnerCoord, torch.Tensor] = {}
        self._copy_in_dsts_by_owner: dict[OwnerCoord, list[torch.Tensor]] = {}
        if self._comm_dtype is not None:
            for owner, total_numel in self._total_numel_by_owner.items():
                packed = torch.empty(total_numel, dtype=self._comm_dtype, device=self.device)
                self._packed_buf_by_owner[owner] = packed
                # Bind each param to a view of its owner's packed buf and
                # cache a 1-D dst slice for foreach copy-in.
                offset = 0
                dsts: list[torch.Tensor] = []
                for p in self._by_owner[owner]:
                    p.bind_to_packed_buffer(packed, offset)
                    dsts.append(packed[offset : offset + p.numel])
                    offset += p.numel
                self._copy_in_dsts_by_owner[owner] = dsts
                # Start in resharded state (storage freed)
                free_storage(packed)

    # ---- unshard (broadcast) — dispatch phase ----

    def unshard(self):
        """Dispatch broadcasts on broadcast_stream. Does NOT wait.

        Phase 2: each owner has one persistent packed buffer; params are
        as_strided views into it. We alloc the packed buf's storage, owner
        fills from its ``_owned_data``, one NCCL broadcast per owner (all
        coalesced into a single NCCL kernel) distributes the data. No
        scatter — views automatically see the storage.

        Phase C: even though ``_pre_forward`` already called
        ``_pre_forward_wait``, this method can also be entered through
        backward prefetch (``_backward_prefetch``), pre-backward, or
        forward-prefetch from an outer layer.  We therefore defensively
        consume any still-pending async replicate broadcast here too so
        the copy-in below always reads fresh ``_owned_data``.
        """
        self._pre_forward_wait()
        if self._is_unsharded:
            return  # still unsharded from forward (reshard_after_forward=False)
        if self._broadcast_event is not None:
            return  # already dispatched, pending wait_for_unshard

        broadcast_stream = self.comm_ctx.broadcast_stream
        broadcast_stream.wait_stream(torch.cuda.current_stream())

        from dmuon._core.internal_utils import alloc_storage
        dp_group = self._dp_group
        local_shard_rank = dp_group.rank()
        with torch.cuda.stream(broadcast_stream):
            # Alloc + owner copy-in BEFORE coalescing: these ops execute
            # immediately on broadcast_stream. Wrapped in no_grad +
            # preserve_version_counter so autograd doesn't see the resize /
            # copy_ as an inplace modification of tensors in the compute graph.
            for owner_coord, packed_buf in self._packed_buf_by_owner.items():
                with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(
                    packed_buf
                ):
                    alloc_storage(packed_buf)

            # Phase 3 / B: batch owner copy-in with torch._foreach_copy_.
            # The shard-dim broadcast runs independently inside each replicate
            # row's shard_group, so the sender for a given owner_coord is
            # whichever rank in this row matches ``owner_shard`` — regardless
            # of ``owner_replicate``.  We therefore copy-in every packed
            # buffer whose ``owner_shard`` matches this rank's shard index;
            # that includes the global owner (``owner_coord == my coord``) and
            # every replicate-row peer that shares the same shard column.
            for owner_coord, dsts in self._copy_in_dsts_by_owner.items():
                if owner_coord[0] != local_shard_rank:
                    continue
                srcs = [
                    p._owned_data.view(-1)
                    for p in self._by_owner[owner_coord]
                ]
                with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(
                    self._packed_buf_by_owner[owner_coord]
                ):
                    torch._foreach_copy_(dsts, srcs)

            with dist._coalescing_manager(group=dp_group, device=self.device):
                for owner_coord, packed_buf in self._packed_buf_by_owner.items():
                    dist.broadcast(
                        packed_buf,
                        src=self._global_owner_ranks[owner_coord],
                        group=dp_group,
                    )

        self._broadcast_event = broadcast_stream.record_event()

    def wait_for_unshard(self):
        """GPU-side wait for broadcasts to complete, then finalize params.

        After this call, all dedicated parameters are set on their modules
        and ready for forward/backward compute.
        """
        if self._broadcast_event is None:
            return

        torch.cuda.current_stream().wait_event(self._broadcast_event)

        # Finalize: set unsharded params on modules
        for p in self.params:
            p.finish_unshard()

        self._broadcast_event = None
        self._is_unsharded = True

    # ---- reshard ----

    def reshard(self):
        """Reshard all params: detach from modules, then free packed buffers.

        Detaching happens first (restores placeholders) so any forward after
        reshard sees a clear no-op tensor rather than a view into 0-sized
        storage.

        If a broadcast was dispatched (e.g. via forward prefetch) but never
        consumed by a ``wait_for_unshard``, we must drain that event *before*
        freeing the packed buffer storage — otherwise (1) freeing mid-broadcast
        is UB, and (2) the stale event persists to the next step, causing
        ``unshard()`` to short-circuit on the ``_broadcast_event is not None``
        guard and leaving ``_unsharded_param`` views pointing at freed storage.
        """
        if self._broadcast_event is not None:
            # Drain and discard — the prefetched unshard was not consumed.
            torch.cuda.current_stream().wait_event(self._broadcast_event)
            self._broadcast_event = None
        for p in self.params:
            p.reshard()
        from dmuon._core.internal_utils import free_storage
        for packed_buf in self._packed_buf_by_owner.values():
            with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(
                packed_buf
            ):
                free_storage(packed_buf)
        self._is_unsharded = False

    # ---- gradient reduction — dispatch phase ----

    def reduce_grads(self):
        """Dispatch gradient reduces. Stage-1 on the shard (``dp_group``) dim,
        and — when HSDP is enabled and ``replicate_grads_enabled`` — Stage-2
        on the replicate dim.  Does NOT wait; call :meth:`wait_for_reduce`
        to synchronize.

        Stream scheduling follows FSDP2's ``foreach_reduce`` pattern
        (``_fsdp_collectives.py:555-588``):

        1. Stage-1 reduce runs on ``reduce_stream``;
        2. When Stage-2 applies, ``replicate_broadcast_stream`` waits on
           ``reduce_stream`` via ``wait_stream``, then dispatches the
           replicate reduce on itself;
        3. ``_post_reduce_event`` is recorded on whichever stream is the
           pipeline's tail, so a single event wait covers both stages.

        Gate semantics (mirrors FSDP2's two bool gates at
        ``_fsdp_param_group.py:185-189``):

        * ``reduce_grads_enabled=False``: full ``no_sync`` — accumulate grads
          locally on every rank without any collective.
        * ``reduce_grads_enabled=True, replicate_grads_enabled=False``:
          Stage-1 runs to average across the shard peers and produce a
          per-shard-owner partial; this partial is stored in
          ``_partial_reduce_by_param[id(p)]`` until the gate flips back.
        """
        if not self.reduce_grads_enabled:
            # no_sync: accumulate full gradients locally (no communication)
            for p in self.params:
                if not p._is_unsharded or p._unsharded_param.grad is None:
                    continue
                grad = p._unsharded_param.grad.data
                if _DTensor is not None and isinstance(grad, _DTensor):
                    grad = grad._local_tensor
                if p._accumulated_grad is not None:
                    p._accumulated_grad.add_(grad)
                else:
                    p._accumulated_grad = grad.clone()
                p._unsharded_param.grad = None
            return

        # Flush any pending reduce from a previous backward (gradient accumulation
        # without optimizer.step). This ensures _reduced_grad is accumulated before
        # we dispatch a new reduce that would overwrite _pending_reduce/
        # _post_reduce_event.
        if self._post_reduce_event is not None or self._replicate_reduce_state is not None:
            self.wait_for_reduce()

        # Merge any accumulated gradients from prior no_sync steps
        for p in self.params:
            if p._accumulated_grad is not None and p._is_unsharded:
                if p._unsharded_param.grad is not None:
                    grad = p._unsharded_param.grad.data
                    if _DTensor is not None and isinstance(grad, _DTensor):
                        grad = grad._local_tensor
                    grad.add_(p._accumulated_grad)
                else:
                    p._unsharded_param.grad = p._accumulated_grad.clone()
                p._accumulated_grad = None

        reduce_stream = self.comm_ctx.reduce_stream
        replicate_stream = self.comm_ctx.replicate_broadcast_stream
        replicate_group = self.comm_ctx.replicate_group
        has_replicate = replicate_group is not None and self.replicate_grads_enabled

        # Ensure gradients are computed before reduce_stream reads them
        reduce_stream.wait_stream(torch.cuda.current_stream())

        self._pending_reduce = []
        # ``_stage2_pending``: per-(grad_buf, param) records; only shard-owner
        # ranks populate this list.  Consumed on the replicate stream below.
        stage2_pending: list[tuple[torch.Tensor, DedicatedParam]] = []
        dp_group = self._dp_group
        my_shard_rank = dp_group.rank()

        # Stage 1 — shard reduce.  Coalesce all reduces into a single fused
        # NCCL kernel.  Grad tensor refs are saved on ``_pending_reduce`` so
        # that ``reshard()`` freeing ``_unsharded_param``'s storage does not
        # dangle the post-reduce view.
        with torch.cuda.stream(reduce_stream):
            with dist._coalescing_manager(group=dp_group, device=self.device):
                for p in self.params:
                    if not p._is_unsharded or p._unsharded_param.grad is None:
                        continue
                    grad = p._unsharded_param.grad.data
                    if _DTensor is not None and isinstance(grad, _DTensor):
                        grad = grad._local_tensor
                    grad = grad.contiguous()
                    dist.reduce(
                        grad, dst=self._global_owner_ranks[p.owner_rank],
                        op=dist.ReduceOp.AVG, group=dp_group,
                    )
                    p._unsharded_param.grad = None
                    self._pending_reduce.append((grad.view(-1), [p]))
                    # Only the shard-owner rank holds the post-Stage-1 grad
                    # and therefore participates in Stage-2.
                    if has_replicate and my_shard_rank == p.owner_shard:
                        stage2_pending.append((grad, p))

        stage1_event = reduce_stream.record_event()

        # Stage 2 — replicate reduce.  Only shard-owner ranks dispatch; the
        # non-shard-owner grads are already "garbage" after Stage 1 (undefined
        # per NCCL spec).  The pipeline's tail event is recorded on whichever
        # stream ran last.
        if has_replicate and stage2_pending:
            replicate_stream.wait_stream(reduce_stream)
            with torch.cuda.stream(replicate_stream):
                # Accumulator flush: if a prior micro-batch ran with
                # ``replicate_grads_enabled=False``, this param's shard-only
                # partial is living in ``_partial_reduce_by_param``.  Fold it
                # into the Stage-1 output before the Stage-2 reduce.
                for grad, p in stage2_pending:
                    partial = self._partial_reduce_by_param.pop(id(p), None)
                    if partial is not None:
                        grad.add_(partial)
                with dist._coalescing_manager(
                    group=replicate_group, device=self.device
                ):
                    for grad, p in stage2_pending:
                        dist.reduce(
                            grad,
                            dst=p._owner_replicate_global_rank,
                            op=dist.ReduceOp.AVG,
                            group=replicate_group,
                        )
            stage2_event = replicate_stream.record_event()
            # Keep the Stage-1 tensor + Stage-2 event alive until
            # ``wait_for_reduce`` runs — mirrors FSDP2 AllReduceState.
            self._replicate_reduce_state = ReplicateReduceState(
                replicate_input=stage2_pending[0][0],
                event=stage2_event,
            )
            self._post_reduce_event = stage2_event
        elif has_replicate and not self.replicate_grads_enabled:
            # Replicate side is gated off: save Stage-1 output as partial
            # accumulator (shard-owner only; others drop their undefined grad).
            for grad, plist in self._pending_reduce:
                p = plist[0]
                if my_shard_rank != p.owner_shard:
                    continue
                existing = self._partial_reduce_by_param.get(id(p))
                if existing is not None:
                    existing.add_(grad.view(p._orig_size))
                else:
                    self._partial_reduce_by_param[id(p)] = grad.view(p._orig_size).clone()
            self._post_reduce_event = stage1_event
        else:
            self._post_reduce_event = stage1_event

    def wait_for_reduce(self):
        """GPU-side wait for reduces to complete, then save owner grad.

        Mirrors FSDP2's ``_wait_for_post_backward`` (``_fsdp_param_group.py:
        621-630``): first wait the tail event of the reduce pipeline, then
        wait the Stage-2 ``ReplicateReduceState`` event if present (both are
        the same event in Phase B, but the dual-wait keeps the control flow
        ready for the Phase C async-broadcast extension).

        After the wait, only the **global owner** has a meaningful grad —
        ``is_owner`` already encodes both shard and replicate dimensions.
        """
        if self._post_reduce_event is None and self._replicate_reduce_state is None:
            return

        current_stream = torch.cuda.current_stream()
        if self._post_reduce_event is not None:
            current_stream.wait_event(self._post_reduce_event)
            self._post_reduce_event = None
        if (
            self._replicate_reduce_state is not None
            and self._replicate_reduce_state.event is not None
        ):
            current_stream.wait_event(self._replicate_reduce_state.event)
        self._replicate_reduce_state = None

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

    # ---- TP gather (T2a) -------------------------------------------------

    def tp_gather_grads(self) -> None:
        """Reassemble the full (M, N) gradient on the TP owner for every
        TP-sharded parameter in this group.

        Pre-condition (enforced by caller ordering): ``wait_for_reduce`` has
        already populated ``p._reduced_grad`` on every DP-owner rank — one
        per TP coord — so every rank in the TP process group holds its
        TP-local shard of the averaged gradient (shape ``p._orig_size``).
        Ranks that are not DP owners have ``_reduced_grad is None`` and
        skip the collective entirely.

        Post-condition: on the TP owner (``p.is_tp_owner`` is True),
        ``p._tp_full_grad`` is a fresh ``(full_shape)`` tensor holding the
        reassembled gradient, ready for NS.  On non-owner TP ranks
        ``_tp_full_grad`` stays ``None`` (they still participate in the
        gather as senders).  Non-TP params are no-ops.

        Runs on ``reduce_stream`` so the gather overlaps with whatever
        subsequent work the user submits on the compute stream — the
        optimizer step consumes ``_tp_full_grad`` and will be serialised
        via a stream wait inserted by the caller right before NS.

        §2.3 阶段 ③ + §4.1 O1 overlap.
        """
        reduce_stream = self.comm_ctx.reduce_stream
        # Build per-param work list.  Only TP-sharded params with a
        # populated ``_reduced_grad`` participate (ranks that are not DP
        # owners hold no grad and must stay silent).
        work: list[tuple[DedicatedParam, torch.Tensor]] = []
        for p in self.params:
            if p.tp_group is None:
                continue
            if p._reduced_grad is None:
                # This rank is not a DP owner for `p` — skip entirely.
                continue
            work.append((p, p._reduced_grad))

        if not work:
            return

        # Ensure the gather sees the final reduced grad (wait_for_reduce
        # already ran on the current stream; reduce_stream may still be
        # mid-flight from an unrelated reduce).
        reduce_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(reduce_stream):
            with dist._coalescing_manager(group=None, device=self.device):
                for p, local_grad in work:
                    tp_size = p.tp_group.size()
                    if p.is_tp_owner:
                        # Allocate the full (M, N) buffer fresh every step
                        # (MVP — see tp_design.md §6.7 note on pre-alloc
                        # deferral).  gather_list entries must be separate
                        # tensors; we cat them post-gather along shard_dim.
                        shard_dim = p.shard_dim if p.shard_dim is not None else 0
                        recv_bufs = [
                            torch.empty_like(local_grad) for _ in range(tp_size)
                        ]
                        dist.gather(
                            local_grad,
                            gather_list=recv_bufs,
                            dst=p._tp_owner_global_rank,
                            group=p.tp_group,
                        )
                        p._tp_full_grad = torch.cat(recv_bufs, dim=shard_dim)
                    else:
                        dist.gather(
                            local_grad,
                            gather_list=None,
                            dst=p._tp_owner_global_rank,
                            group=p.tp_group,
                        )
                        p._tp_full_grad = None
        # No event record: the TP owner consumes ``_tp_full_grad`` on the
        # default (compute) stream in optimizer.step, which will
        # ``wait_stream(reduce_stream)`` just before reading.

    # ---- TP scatter (T2b) ------------------------------------------------

    def _tp_scatter_dispatch(self) -> Optional[list[torch.Tensor]]:
        """Shared dispatch body for sync + async TP scatter.

        Queues the scatter + ``_owned_data.mul_(wd).add_(shard)`` fuse on
        ``replicate_broadcast_stream`` and returns the list of transient
        ``recv_shard`` buffers.  Callers:
          * sync  (``tp_scatter_delta``) — ``wait_stream`` joins back.
          * async (``tp_scatter_delta_async``) — records event + stores
            the recv-shard list in :attr:`_tp_scatter_state`.

        Returns ``None`` when the group has no TP-sharded params with
        pending updates (no dispatch happened).
        """
        work: list[DedicatedParam] = [
            p for p in self.params
            if p.tp_group is not None and p._reduced_grad is not None
        ]
        if not work:
            return None

        bcast_stream = self.comm_ctx.replicate_broadcast_stream
        bcast_stream.wait_stream(torch.cuda.current_stream())

        recv_shards: list[torch.Tensor] = []
        with torch.cuda.stream(bcast_stream):
            with dist._coalescing_manager(group=None, device=self.device):
                for p in work:
                    shard_dim = p.shard_dim if p.shard_dim is not None else 0
                    recv_shard = torch.empty_like(p._owned_data)
                    if p.is_tp_owner:
                        assert p._tp_full_delta is not None, (
                            f"{p.param_name}: TP owner has _tp_full_delta=None "
                            "— Muon._step_muon did not populate it."
                        )
                        splits = [
                            s.contiguous()
                            for s in p._tp_full_delta.split(
                                p._orig_size[shard_dim], dim=shard_dim,
                            )
                        ]
                        dist.scatter(
                            recv_shard,
                            scatter_list=splits,
                            src=p._tp_owner_global_rank,
                            group=p.tp_group,
                        )
                    else:
                        dist.scatter(
                            recv_shard,
                            scatter_list=None,
                            src=p._tp_owner_global_rank,
                            group=p.tp_group,
                        )
                    p._owned_data.mul_(p._tp_wd_factor).add_(recv_shard)
                    recv_shard.record_stream(bcast_stream)
                    recv_shards.append(recv_shard)
                    p._tp_full_grad = None
                    p._tp_full_delta = None
                    p._reduced_grad = None
        return recv_shards

    def tp_scatter_delta(self) -> None:
        """Fan the full-matrix NS update back to each DP-owner TP shard.

        **Sync variant** (§2.3 阶段 ⑤): dispatch + join to compute stream
        before returning so the caller can immediately read ``_owned_data``.

        Pre-conditions (set by ``Muon._step_muon``):
          * TP owner: ``p._tp_full_delta`` is the pre-scaled update
            (``-lr * scale * NS_output``), shape ``p.full_shape``.
          * Every DP-owner TP rank: ``p._tp_wd_factor = 1 - lr * wd``.
          * ``p._owned_data`` still holds the old weight shard.

        Post: each DP-owner rank's ``p._owned_data`` is updated in place
        via ``owned.mul_(wd_factor).add_(shard)`` — Moonlight's
        ``new = (1 - lr*wd) * old - lr*scale * NS`` per shard.
        """
        if self._tp_scatter_dispatch() is None:
            return
        # Sync contract: compute stream waits for the scatter to land
        # before the caller can read ``_owned_data`` (the next unshard
        # broadcast reads it).
        torch.cuda.current_stream().wait_stream(
            self.comm_ctx.replicate_broadcast_stream
        )

    def tp_scatter_delta_async(self) -> None:
        """T2d async variant of :meth:`tp_scatter_delta`.

        Dispatches the scatter on ``replicate_broadcast_stream`` and
        returns immediately; records an event + pins the ``recv_shard``
        buffers in :attr:`_tp_scatter_state` so the caching allocator
        cannot reclaim them before the NCCL kernel observes the writes.
        The event is consumed by the NEXT iteration's
        :meth:`_pre_forward_wait` — mirrors the replicate-broadcast
        cross-call event chain.

        Gate semantics:
          * ``_tp_sync_fallback=True`` → degrade to sync; state stays IDLE.
          * Double-dispatch (state still PENDING) → ``RuntimeError``.
        """
        if self._tp_sync_fallback:
            self.tp_scatter_delta()
            return
        if self._tp_scatter_state is not None:
            raise RuntimeError(
                "tp_scatter_delta_async: previous event still pending — "
                "pre_forward_wait was not consumed before the next dispatch"
            )
        if self._tp_scatter_dispatch() is None:
            return
        event = self.comm_ctx.replicate_broadcast_stream.record_event()
        # Intentionally DO NOT pin recv_shards (see TPScatterState
        # docstring): the allocator is already kept honest by
        # ``record_stream`` inside ``_tp_scatter_dispatch``.  Pinning
        # across iterations causes sync/async loss divergence.
        self._tp_scatter_state = TPScatterState(
            recv_shards=[], event=event
        )

    # ---- replicate-dim post-step broadcast (Phase B.2) -------------------

    def replicate_broadcast_sync(self):
        """Dispatch the replicate-axis broadcast after ``optimizer.step()``.

        In HSDP mode the global owner (the rank whose ``(shard, replicate)``
        coord matches ``owner_rank``) has just written the updated parameter
        into its ``_owned_data``.  This method fans that buffer out to the
        other ranks in the same shard column via ``replicate_group``, so
        every shard-owner rank ends the iteration with a consistent
        ``_owned_data`` ready for the next shard-dim broadcast.

        **Sync, not async.**  Phase B keeps the broadcast synchronous
        (caller pairs each dispatch with ``wait_for_replicate_broadcast``
        before the next training iteration); Phase C will introduce the
        async variant that hides the IB transfer inside forward.

        No-op when ``replicate_group`` is None (pure 1D shard-only).
        """
        replicate_group = self.comm_ctx.replicate_group
        if replicate_group is None:
            return

        my_shard_rank = self._dp_group.rank()
        # Only ranks in some owner's shard column actually participate.  If
        # this rank is not in any owner's shard column for this group, skip
        # entirely — avoids dispatching an empty coalescing manager.
        if not any(my_shard_rank == p.owner_shard for p in self.params):
            return

        bcast_stream = self.comm_ctx.replicate_broadcast_stream
        bcast_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(bcast_stream):
            with dist._coalescing_manager(
                group=replicate_group, device=self.device
            ):
                for p in self.params:
                    if my_shard_rank != p.owner_shard:
                        continue
                    # ``_owned_data`` is allocated on every shard-peer of
                    # the owner (see ``DedicatedParam.__init__``); the
                    # global owner sends, the R-1 replicate peers receive
                    # into the same tensor.
                    dist.broadcast(
                        p._owned_data,
                        src=p._owner_replicate_global_rank,
                        group=replicate_group,
                    )

        self._replicate_broadcast_event = bcast_stream.record_event()

    def wait_for_replicate_broadcast(self):
        """Block current stream until the post-step replicate broadcast is
        visible.  Must be called before the next shard-dim ``unshard()`` so
        the updated ``_owned_data`` is safe to read.
        """
        if self._replicate_broadcast_event is None:
            return
        torch.cuda.current_stream().wait_event(self._replicate_broadcast_event)
        self._replicate_broadcast_event = None

    # ---- replicate-dim post-step broadcast (Phase C.1 async) -------------

    def replicate_broadcast_async(self) -> None:
        """Dispatch the replicate-axis broadcast asynchronously.

        The event is stored on :attr:`_replicate_broadcast_state` and
        consumed later by :meth:`_pre_forward_wait` (Phase C.3) in the next
        iteration.  Hiding target: the forward compute of *prior* layers
        that do not depend on this group's ``_owned_data``.

        Gate semantics:
            - ``replicate_group is None`` → no-op (1D shard-only mode).
            - ``_replicate_sync_fallback=True`` → degrade to the Phase B
              sync path; caller observes no pending state after return.
            - Double dispatch (calling while a state is still PENDING) is
              a programming error — detected by the
              ``test_async_wait_semantics`` smoke (C.0).

        The wait point MUST be before ``unshard()``'s copy-in reads
        ``_owned_data``; checkpoint save paths must also drain pending
        state via :func:`dmuon.utils.wait_all_replicate_broadcasts`.
        """
        replicate_group = self.comm_ctx.replicate_group
        if replicate_group is None:
            return

        # Fallback path: sync dispatch + wait inline.  State stays IDLE.
        if self._replicate_sync_fallback:
            self.replicate_broadcast_sync()
            self.wait_for_replicate_broadcast()
            return

        if self._replicate_broadcast_state is not None:
            raise RuntimeError(
                "replicate_broadcast_async: previous event still pending — "
                "pre_forward_wait was not consumed before the next dispatch"
            )

        my_shard_rank = self._dp_group.rank()
        if not any(my_shard_rank == p.owner_shard for p in self.params):
            return

        bcast_stream = self.comm_ctx.replicate_broadcast_stream
        bcast_stream.wait_stream(torch.cuda.current_stream())

        # Pick one ``_owned_data`` tensor to pin via the state tuple.  Any
        # rank-participating param works; allocator arena is shared within
        # the coalescing manager below.
        pin_ref: Optional[torch.Tensor] = None
        with torch.cuda.stream(bcast_stream):
            with dist._coalescing_manager(
                group=replicate_group, device=self.device
            ):
                for p in self.params:
                    if my_shard_rank != p.owner_shard:
                        continue
                    if pin_ref is None:
                        pin_ref = p._owned_data
                    dist.broadcast(
                        p._owned_data,
                        src=p._owner_replicate_global_rank,
                        group=replicate_group,
                    )

        event = bcast_stream.record_event()
        assert pin_ref is not None  # guaranteed by the "any(...)" check above
        self._replicate_broadcast_state = ReplicateBroadcastState(
            replicate_input=pin_ref,
            event=event,
        )

    def _pre_forward_wait(self) -> None:
        """Phase C.3: consume any pending async replicate broadcast before
        the shard-dim unshard's copy-in reads ``_owned_data``.

        No-op when the state is IDLE (either 1D shard-only mode, or sync
        fallback, or no prior async dispatch this iteration).

        When :envvar:`DMUON_REPLICATE_PROFILE` is set (Phase C.4 / C.7),
        the wait is bracketed by CUDA events with ``enable_timing=True``
        and ``_last_replicate_wait_us`` is populated.  The timing path
        forces a CPU sync on the current stream, so it costs one
        round-trip per group per step and is NOT safe to leave on in
        production; it is meant for fallback monitoring + profiling.
        """
        # T2d: drain the async TP scatter state first.  The scatter is
        # what writes ``_owned_data`` on every DP-owner TP rank; the
        # replicate broadcast fans that value out.  In pure-DP mode (no
        # replicate_group) the scatter state is the only thing to wait
        # on; in HSDP mode both fire on the same stream and waiting the
        # replicate broadcast event implicitly covers the scatter, but
        # we still wait the scatter event explicitly so the state tuple
        # (and its pinned recv_shards) can be released.
        tp_state = self._tp_scatter_state
        if tp_state is not None:
            profile_enabled = bool(
                int(os.environ.get("DMUON_REPLICATE_PROFILE", "0") or 0)
            )
            if profile_enabled:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                torch.cuda.current_stream().wait_event(tp_state.event)
                end.record()
                end.synchronize()
                self._last_tp_scatter_wait_us = (
                    start.elapsed_time(end) * 1000.0
                )
            else:
                torch.cuda.current_stream().wait_event(tp_state.event)
            self._tp_scatter_state = None

        state = self._replicate_broadcast_state
        if state is None:
            return

        profile_enabled = bool(int(os.environ.get("DMUON_REPLICATE_PROFILE", "0") or 0))
        if profile_enabled:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            torch.cuda.current_stream().wait_event(state.event)
            end.record()
            # ``elapsed_time`` requires both events to be complete; the
            # current-stream synchronize is the cheapest way to enforce it
            # (only waits on the two timing events on the default stream).
            end.synchronize()
            self._last_replicate_wait_us = start.elapsed_time(end) * 1000.0  # ms→μs
            from dmuon import _replicate_profile
            _replicate_profile.record_wait_from_group(
                self, self._last_replicate_wait_us
            )
        else:
            torch.cuda.current_stream().wait_event(state.event)

        self._replicate_broadcast_state = None

    # ---- Phase C.4 fallback monitor -------------------------------------

    def _update_replicate_fallback(self) -> None:
        """Read ``_last_replicate_wait_us`` (populated by the profile path
        in :meth:`_pre_forward_wait`) and advance the fallback state
        machine.

        Trips ``_replicate_sync_fallback`` after
        :data:`REPLICATE_FALLBACK_CONSECUTIVE_STEPS` consecutive waits
        above :data:`REPLICATE_WAIT_THRESHOLD_US`.  Single-direction:
        once tripped, stays tripped until
        :meth:`reset_replicate_fallback` is called.
        """
        if self._replicate_sync_fallback:
            return  # already degraded; nothing to do
        if self._last_replicate_wait_us <= 0.0:
            return  # no sample this step (profiler off or no dispatch)
        if self._last_replicate_wait_us > REPLICATE_WAIT_THRESHOLD_US:
            self._replicate_slow_wait_count += 1
        else:
            self._replicate_slow_wait_count = 0
        if self._replicate_slow_wait_count >= REPLICATE_FALLBACK_CONSECUTIVE_STEPS:
            self._replicate_sync_fallback = True
        # Drain the sample so a step without fresh data does not re-use
        # the last value.
        self._last_replicate_wait_us = 0.0

    def _update_tp_scatter_fallback(self) -> None:
        """T2e: tracks async TP scatter wait-time and flips
        ``_tp_sync_fallback`` after ``REPLICATE_FALLBACK_CONSECUTIVE_STEPS``
        consecutive slow waits (same thresholds as the replicate broadcast
        fallback; tp_design.md §5)."""
        if self._tp_sync_fallback:
            return
        if self._last_tp_scatter_wait_us <= 0.0:
            return
        if self._last_tp_scatter_wait_us > REPLICATE_WAIT_THRESHOLD_US:
            self._tp_scatter_slow_wait_count += 1
        else:
            self._tp_scatter_slow_wait_count = 0
        if self._tp_scatter_slow_wait_count >= REPLICATE_FALLBACK_CONSECUTIVE_STEPS:
            self._tp_sync_fallback = True
        self._last_tp_scatter_wait_us = 0.0

    def reset_replicate_fallback(self) -> None:
        """Manually clear the fallback flag.  Intended for user code that
        wants to re-enable async after fixing the slow-IB condition."""
        self._replicate_sync_fallback = False
        self._replicate_slow_wait_count = 0
        self._last_replicate_wait_us = 0.0

    def reset_tp_scatter_fallback(self) -> None:
        """Manually clear the async TP scatter fallback flag."""
        self._tp_sync_fallback = False
        self._tp_scatter_slow_wait_count = 0
        self._last_tp_scatter_wait_us = 0.0

    # ---- backward prefetch ----

    def _backward_prefetch(self) -> None:
        """Prefetch next layer's unshard during current layer's backward.

        Mirrors FSDP2's _backward_prefetch: uses reverse post-forward order.

        Skip when the target group has already completed its backward (i.e.,
        its ``_post_forward_indices`` is empty). Otherwise a prefetch would
        dispatch a broadcast of pre-optim ``_owned_data``; if optim.step then
        runs before the next forward consumes it, the subsequent forward
        reads a stale weight value.
        """
        if not self._post_forward_indices:
            return
        curr_index = self._post_forward_indices.pop()
        if (target_index := curr_index - 1) < 0:
            return
        target_group = self.comm_ctx.post_forward_order[target_index]
        if not target_group._post_forward_indices:
            return  # target already backward'd — prefetch would read stale data
        target_group.unshard()  # dispatch only — no wait

    def _record_post_forward(self) -> None:
        """Record this group's position in forward order for backward prefetch."""
        post_forward_index = len(self.comm_ctx.post_forward_order)
        self.comm_ctx.post_forward_order.append(self)
        self._post_forward_indices.append(post_forward_index)
