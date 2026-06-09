"""DedicatedParamGroup: manages communication for dedicated params in one layer.

Uses dedicated CUDA streams for broadcast/reduce (analogous to FSDP2's
all_gather_stream / reduce_scatter_stream) and CUDA events for GPU-side
synchronization instead of CPU-blocking work.wait().
"""

from collections import defaultdict
from contextlib import contextmanager
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist

from dmuon._core.comm import DedicatedCommContext
from dmuon._core.dynamo import dynamo_disable
from dmuon._core.owner_rank import OwnerCoord

from .param import DedicatedParam


@contextmanager
def _profile_range(name: str):
    with torch.profiler.record_function(name):
        yield


def _split_for_scatter(
    tensor: torch.Tensor,
    split_size: int,
    *,
    dim: int,
    out_buffers: Optional[list[torch.Tensor]] = None,
) -> list[torch.Tensor]:
    """Return contiguous scatter shards while avoiding unnecessary copies.

    TP owner scatter needs contiguous send tensors.  Splitting a contiguous
    matrix along dim 0 already produces contiguous views, while dim 1 splits
    generally do not.  The old path copied every shard unconditionally; for
    row-sharded parameters that turned the owner-side split into avoidable
    memory traffic inside post-step publish.
    """
    shards = list(tensor.split(split_size, dim=dim))
    if out_buffers is not None:
        if len(out_buffers) != len(shards):
            raise ValueError(
                f"out_buffers length {len(out_buffers)} does not match "
                f"scatter shard count {len(shards)}"
            )
        for out, shard in zip(out_buffers, shards):
            if tuple(out.shape) != tuple(shard.shape):
                raise ValueError(
                    f"out buffer shape {tuple(out.shape)} does not match "
                    f"scatter shard shape {tuple(shard.shape)}"
                )
            out.copy_(shard)
        return out_buffers
    return [shard if shard.is_contiguous() else shard.contiguous() for shard in shards]


def _tp_gather_buffer_reuse_enabled(comm_ctx: DedicatedCommContext) -> bool:
    return comm_ctx.tp_gather_buffer_reuse_enabled()


def _tp_scatter_buffer_reuse_enabled(comm_ctx: DedicatedCommContext) -> bool:
    return comm_ctx.tp_scatter_buffer_reuse_enabled()


def _tensor_payload_bytes(shape: torch.Size | tuple[int, ...], dtype: torch.dtype) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return int(numel * torch.empty((), dtype=dtype).element_size())


def _param_payload_bytes(p: DedicatedParam) -> int:
    return _tensor_payload_bytes(p._orig_size, p._orig_dtype)


def _group_profile_name(group: object) -> str:
    return str(getattr(group, "_debug_name", None) or f"group_{id(group):x}")


def _cached_tensor(
    owner: object,
    attr: str,
    *,
    shape: torch.Size | tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cached = getattr(owner, attr, None)
    if (
        cached is None
        or tuple(cached.shape) != tuple(shape)
        or cached.dtype != dtype
        or cached.device != device
    ):
        cached = torch.empty(tuple(shape), dtype=dtype, device=device)
        setattr(owner, attr, cached)
    return cached


def _cached_tensor_list(
    owner: object,
    attr: str,
    *,
    count: int,
    shape: torch.Size | tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    cached = getattr(owner, attr, None)
    valid = (
        isinstance(cached, list)
        and len(cached) == count
        and all(
            tuple(t.shape) == tuple(shape)
            and t.dtype == dtype
            and t.device == device
            for t in cached
        )
    )
    if not valid:
        cached = [
            torch.empty(tuple(shape), dtype=dtype, device=device)
            for _ in range(count)
        ]
        setattr(owner, attr, cached)
    return cached


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


def _adamw_replicate_allreduce_enabled(p: DedicatedParam) -> bool:
    return bool(getattr(p, "_dmuon_adamw_replicate_allreduce", False))


def _sharded_adamw_enabled(p: DedicatedParam) -> bool:
    return bool(getattr(p, "uses_sharded_adamw", lambda: False)())


def _sharded_muon_forward_enabled(p: DedicatedParam) -> bool:
    return bool(getattr(p, "uses_sharded_muon_forward", lambda: False)())


def _owner_broadcast_enabled(p: DedicatedParam) -> bool:
    return not _sharded_adamw_enabled(p) and not _sharded_muon_forward_enabled(p)


def _needs_post_step_replicate_broadcast(p: DedicatedParam) -> bool:
    # Production type-split AdamW uses DMuon owner update + post-step publish,
    # matching Muon tensors.  Replicate-all-reduce mode lets every shard-owner
    # replicate update locally, so there is no global-owner update to fan out.
    # Muon tensors using FSDP-style forward placement have a separate sharded
    # publish path: owner-row reduce-scatter followed by shard-column broadcast.
    # Broadcasting the full owner tensor across the replicate axis first would
    # duplicate that publish traffic and put a large transfer on the forward
    # critical path.
    return (
        not _adamw_replicate_allreduce_enabled(p)
        and not _sharded_adamw_enabled(p)
        and not _sharded_muon_forward_enabled(p)
    )


def _event_is_ready(event: Optional[torch.cuda.Event]) -> bool:
    if event is None:
        return True
    try:
        return bool(event.query())
    except RuntimeError:
        return False


def _should_skip_unready_publish_prefetch(
    tp_event: Optional[torch.cuda.Event],
    replicate_event: Optional[torch.cuda.Event],
    sharded_muon_event: Optional[torch.cuda.Event] = None,
    *,
    allow_unready_publish_wait: bool,
) -> tuple[bool, bool, bool, bool]:
    if not allow_unready_publish_wait:
        # Do not use per-rank CUDA event readiness to decide whether a far
        # prefetch may enter collectives.  Event readiness is local and can
        # diverge across ranks; letting only the "ready" ranks dispatch an
        # all-gather/broadcast creates collective order mismatches.  Far
        # prefetch retries from later hooks after the publish state is consumed.
        tp_pending = tp_event is not None
        replicate_pending = replicate_event is not None
        sharded_muon_pending = sharded_muon_event is not None
        should_skip = tp_pending or replicate_pending or sharded_muon_pending
        return should_skip, tp_pending, replicate_pending, sharded_muon_pending

    tp_not_ready = tp_event is not None and not _event_is_ready(tp_event)
    replicate_not_ready = (
        replicate_event is not None and not _event_is_ready(replicate_event)
    )
    sharded_muon_not_ready = (
        sharded_muon_event is not None and not _event_is_ready(sharded_muon_event)
    )
    should_skip = (
        tp_not_ready or replicate_not_ready or sharded_muon_not_ready
    ) and not allow_unready_publish_wait
    return should_skip, tp_not_ready, replicate_not_ready, sharded_muon_not_ready


class TPScatterState(NamedTuple):
    """State kept alive across the async post-step TP scatter.

    T2d analogue of :class:`ReplicateBroadcastState`: the scatter is
    dispatched on ``replicate_broadcast_stream`` and the caller returns
    without waiting; the event is consumed on the next iteration's
    :meth:`_pre_forward_wait` hook.

    The state pins TP-owner send shards and receiver scratch shards until the
    scatter/update event is visible.
    This is required because the owner builds transient contiguous split
    tensors for ``scatter_list`` and ``Muon._step_muon`` clears
    ``_tp_full_delta`` before async communication has necessarily consumed
    those tensors.  Receiver scratch shards are also used by an enqueued
    ``_owned_data.mul_(wd).add_(recv_shard)`` on the same stream, so keep a
    Python reference until the event is consumed instead of relying only on
    allocator stream recording.
    """

    refs: list[torch.Tensor]
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


class ShardedMuonPublishState(NamedTuple):
    """State kept alive across async Muon sharded-publish reduce-scatter."""

    refs: list[torch.Tensor]
    event: torch.cuda.Event


class DedicatedParamGroup:
    """Manages all dedicated parameters within one layer.

    Parameters with the same owner are packed into one broadcast/reduce call.
    All communication runs on dedicated CUDA streams from DedicatedCommContext.
    """

    def __init__(
        self,
        params: list[DedicatedParam],
        comm_ctx: DedicatedCommContext,
        *,
        delay_stage2_to_optimizer: bool = True,
    ):
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
        # Match FSDP2's split between the inter-group buffer-safety wait and
        # the optimizer dependency.  Backward only waits Stage-1 shard reduce
        # to bound temporary grad lifetime; the HSDP Stage-2 replicate tail is
        # consumed later by per-group optimizer preparation so it can overlap
        # with earlier groups' optimizer work and post-step publish.
        self._delay_stage2_to_optimizer: bool = bool(delay_stage2_to_optimizer)

        # Event-based synchronization (replaces work.wait()).  ``_post_reduce_event``
        # marks the end of the reduce pipeline — shard-only in Phase A, shard+replicate
        # in Phase B (mirrors FSDP2's ``_post_reduce_event``; see
        # ``_fsdp_param_group.py:213``).  ``_replicate_reduce_state`` keeps the
        # Stage-2 input + event alive until ``wait_for_reduce`` runs, mirroring
        # ``AllReduceState`` in ``_fsdp_param_group.py:115-117``.
        self._broadcast_event: Optional[torch.cuda.Event] = None
        self._sharded_adamw_unshard_event: Optional[torch.cuda.Event] = None
        # FSDP2 only waits the previous reduce-scatter event between backward
        # groups to keep its input buffer lifetime bounded; it does *not* wait
        # the HSDP all-reduce tail there.  DMuon mirrors that split with this
        # Stage-1 event: the rolling backward drain waits it to release
        # shard-reduce-only grad refs, while the full Stage-2 tail remains
        # pending until the optimizer/root-post-backward boundary.
        self._stage1_reduce_event: Optional[torch.cuda.Event] = None
        self._post_reduce_event: Optional[torch.cuda.Event] = None
        self._last_unshard_total_bytes: int = 0
        self._last_unshard_prefetch: bool = False
        self._replicate_reduce_state: Optional[ReplicateReduceState] = None
        self._muon_grad_ready_event: Optional[torch.cuda.Event] = None
        self._muon_grad_ready_refs: list[torch.Tensor] = []
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
        self._sharded_muon_publish_state: Optional[ShardedMuonPublishState] = None
        # T2d: TP scatter state — mirrors ``_replicate_broadcast_state``
        # and pins transient TP-owner send shards until the scatter event is
        # visible. Async uses it for lifetime + deferred waiting; sync uses it
        # for lifetime, since the current stream already waits on the event.
        # Consumed together with the replicate broadcast state in
        # ``_pre_forward_wait``. See ``tp_design.md`` §4.2 (O2).
        self._tp_scatter_state: Optional[TPScatterState] = None
        self._tp_gather_event: Optional[torch.cuda.Event] = None
        self._tp_gather_refs: list[torch.Tensor] = []
        self._tp_gather_pending_full_grads: list[
            tuple[
                DedicatedParam,
                list[torch.Tensor],
                torch.Tensor,
                int,
                Optional[torch.Tensor],
            ]
        ] = []
        # Partial accumulator across ``no_sync`` micro-batches (per-param, on
        # the shard-owner rank).  Set during grad-accum when
        # ``replicate_grads_enabled`` is False; flushed into the next Stage-2
        # reduce when the gate flips back.  Mirrors FSDP2's
        # ``_partial_reduce_output`` (``_fsdp_param_group.py:220``).
        self._partial_reduce_by_param: dict[int, torch.Tensor] = {}

        # Deferred reduce unpack (fixes data race in old _packed_reduce)
        self._pending_reduce: list[
            tuple[Optional[torch.Tensor], list[DedicatedParam]]
        ] = []

        # Prefetch tracking (mirrors FSDPParamGroup._post_forward_indices)
        self._post_forward_indices: list[int] = []

        # Unsharded state tracking (for reshard_after_forward=False)
        self._is_unsharded: bool = False

        # Post-backward fast-path tracking: reset in _pre_forward, set True when
        # reduce+reshard runs (either via _DedicatedPostBackward.backward fast path
        # or via the autograd-engine root callback).
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
            owner: sum(
                p.numel for p in owner_params if _owner_broadcast_enabled(p)
            )
            for owner, owner_params in self._by_owner.items()
            if any(_owner_broadcast_enabled(p) for p in owner_params)
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
                packed = torch.empty(
                    total_numel, dtype=self._comm_dtype, device=self.device
                )
                self._packed_buf_by_owner[owner] = packed
                # Bind each param to a view of its owner's packed buf and
                # cache a 1-D dst slice for foreach copy-in.
                offset = 0
                dsts: list[torch.Tensor] = []
                for p in self._by_owner[owner]:
                    if not _owner_broadcast_enabled(p):
                        continue
                    p.bind_to_packed_buffer(packed, offset)
                    dsts.append(packed[offset : offset + p.numel])
                    offset += p.numel
                self._copy_in_dsts_by_owner[owner] = dsts
                # Start in resharded state (storage freed)
                free_storage(packed)

    # ---- unshard (broadcast) — dispatch phase ----

    @dynamo_disable
    def unshard(
        self,
        *,
        prefetch: bool = False,
        allow_unready_publish_wait: bool = False,
    ):
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
        group_name = _group_profile_name(self)
        pending_tp_publish = self._tp_scatter_state is not None
        pending_replicate_publish = self._replicate_broadcast_state is not None
        pending_sharded_muon_publish = self._sharded_muon_publish_state is not None
        if prefetch and (
            pending_tp_publish
            or pending_replicate_publish
            or pending_sharded_muon_publish
        ):
            tp_event = (
                self._tp_scatter_state.event
                if self._tp_scatter_state is not None
                else None
            )
            replicate_event = (
                self._replicate_broadcast_state.event
                if self._replicate_broadcast_state is not None
                else None
            )
            sharded_muon_event = (
                self._sharded_muon_publish_state.event
                if self._sharded_muon_publish_state is not None
                else None
            )
            (
                should_skip,
                tp_not_ready,
                replicate_not_ready,
                sharded_muon_not_ready,
            ) = (
                _should_skip_unready_publish_prefetch(
                    tp_event,
                    replicate_event,
                    sharded_muon_event,
                    allow_unready_publish_wait=allow_unready_publish_wait,
                )
            )
            if should_skip:
                # A forward prefetch must not insert an unready publish wait into
                # the shard-dim broadcast stream for far-future groups: it can
                # stall stream FIFO progress and block nearer layers. The
                # immediate next forward group is allowed to queue the wait so
                # its unshard starts as soon as the publish event becomes ready.
                # Demand unshard still waits below; skipped prefetches retry
                # from later hooks.
                self.comm_ctx.record_forward_unshard_counter(
                    group_name,
                    prefetch_publish_not_ready_skips=1,
                    prefetch_tp_publish_not_ready_skips=1 if tp_not_ready else 0,
                    prefetch_replicate_publish_not_ready_skips=(
                        1 if replicate_not_ready else 0
                    ),
                    prefetch_sharded_muon_publish_not_ready_skips=(
                        1 if sharded_muon_not_ready else 0
                    ),
                )
                return
            if allow_unready_publish_wait and (
                tp_not_ready or replicate_not_ready or sharded_muon_not_ready
            ):
                self.comm_ctx.record_forward_unshard_counter(
                    group_name,
                    prefetch_unready_publish_waits_queued=1,
                    prefetch_unready_tp_publish_waits_queued=(
                        1 if tp_not_ready else 0
                    ),
                    prefetch_unready_replicate_publish_waits_queued=(
                        1 if replicate_not_ready else 0
                    ),
                    prefetch_unready_sharded_muon_publish_waits_queued=(
                        1 if sharded_muon_not_ready else 0
                    ),
                )
        publish_wait_stream = (
            self.comm_ctx.broadcast_stream if prefetch else torch.cuda.current_stream()
        )
        publish_wait_start = None
        publish_wait_end = None
        if self.comm_ctx.record_forward_profile:
            publish_wait_start = torch.cuda.Event(enable_timing=True)
            publish_wait_end = torch.cuda.Event(enable_timing=True)
            publish_wait_start.record(publish_wait_stream)
        if prefetch:
            with _profile_range(f"dmuon.forward_prefetch_publish.{group_name}"):
                self._pre_forward_prefetch_publish()
        else:
            with _profile_range(f"dmuon.pre_forward_wait.{group_name}"):
                self._pre_forward_wait()
        if publish_wait_start is not None and publish_wait_end is not None:
            publish_wait_end.record(publish_wait_stream)
            self.comm_ctx.record_forward_unshard_event(
                group_name=group_name,
                phase=(
                    "prefetch_publish_wait"
                    if prefetch
                    else "pre_forward_publish_wait"
                ),
                start=publish_wait_start,
                end=publish_wait_end,
                bytes=0,
                prefetch=prefetch,
            )
        if self._is_unsharded:
            self.comm_ctx.record_forward_unshard_counter(
                group_name, already_unsharded=1
            )
            return  # still unsharded from forward (reshard_after_forward=False)
        if (
            self._broadcast_event is not None
            or self._sharded_adamw_unshard_event is not None
        ):
            if not prefetch:
                self.comm_ctx.record_forward_unshard_counter(
                    group_name, prefetch_hits=1
                )
            return  # already dispatched, pending wait_for_unshard

        sharded_muon_params = [
            p for p in self.params if _sharded_muon_forward_enabled(p)
        ]
        if sharded_muon_params and any(
            not bool(getattr(p, "_sharded_muon_initialized", False))
            for p in sharded_muon_params
        ):
            if prefetch:
                # First-use sharded Muon publish is a collective over the shard
                # group.  Demand unshard has a consistent module order across
                # ranks; far prefetch may be skipped/retried differently across
                # ranks, so it must not be allowed to introduce the first
                # reduce-scatter sequence.
                self.comm_ctx.record_forward_unshard_counter(
                    group_name,
                    prefetch_sharded_muon_publish_not_ready_skips=1,
                )
                return
            # Meta-init only materializes the full logical matrix on owner
            # shard columns.  The all-gather forward placement needs every
            # rank's local shard before the first forward, so publish the
            # owner tensor once before the first all-gather. Subsequent
            # optimizer steps use the normal post-step sharded publish path.
            with _profile_range(f"dmuon.sharded_muon_initial_publish.{group_name}"):
                self.sharded_muon_publish_sync()

        broadcast_stream = self.comm_ctx.broadcast_stream
        sharded_adamw_separate_stream = bool(
            getattr(
                self.comm_ctx,
                "sharded_adamw_unshard_separate_stream_enabled",
                False,
            )
        )
        sharded_adamw_stream = (
            self.comm_ctx.sharded_adamw_unshard_stream
            if sharded_adamw_separate_stream
            else broadcast_stream
        )
        wait_current_start = None
        wait_current_end = None
        if self.comm_ctx.record_forward_profile:
            wait_current_start = torch.cuda.Event(enable_timing=True)
            wait_current_end = torch.cuda.Event(enable_timing=True)
            wait_current_start.record(broadcast_stream)
        broadcast_stream.wait_stream(torch.cuda.current_stream())
        if wait_current_start is not None and wait_current_end is not None:
            wait_current_end.record(broadcast_stream)
            self.comm_ctx.record_forward_unshard_event(
                group_name=group_name,
                phase="broadcast_stream_wait_current",
                start=wait_current_start,
                end=wait_current_end,
                bytes=0,
                prefetch=prefetch,
            )
        sharded_wait_current_start = None
        sharded_wait_current_end = None
        if sharded_adamw_separate_stream and self.comm_ctx.record_forward_profile:
            sharded_wait_current_start = torch.cuda.Event(enable_timing=True)
            sharded_wait_current_end = torch.cuda.Event(enable_timing=True)
            sharded_wait_current_start.record(sharded_adamw_stream)
        if sharded_adamw_separate_stream:
            sharded_adamw_stream.wait_stream(torch.cuda.current_stream())
        if (
            sharded_wait_current_start is not None
            and sharded_wait_current_end is not None
        ):
            sharded_wait_current_end.record(sharded_adamw_stream)
            self.comm_ctx.record_forward_unshard_event(
                group_name=group_name,
                phase="sharded_adamw_stream_wait_current",
                start=sharded_wait_current_start,
                end=sharded_wait_current_end,
                bytes=0,
                prefetch=prefetch,
            )

        from dmuon._core.internal_utils import alloc_storage

        dp_group = self._dp_group
        local_shard_rank = dp_group.rank()
        sharded_adamw_params = [
            p for p in self.params if _sharded_adamw_enabled(p)
        ]
        owner_broadcast_bytes = sum(
            int(packed_buf.numel() * packed_buf.element_size())
            for packed_buf in self._packed_buf_by_owner.values()
        )
        owner_broadcast_max_bucket_bytes = max(
            (
                int(packed_buf.numel() * packed_buf.element_size())
                for packed_buf in self._packed_buf_by_owner.values()
            ),
            default=0,
        )
        sharded_adamw_all_gather_bytes = sum(
            int(
                p._sharded_adamw_full_padded.numel()
                * p._sharded_adamw_full_padded.element_size()
            )
            for p in sharded_adamw_params
            if p._sharded_adamw_full_padded is not None
        )
        sharded_muon_all_gather_bytes = sum(
            int(
                p._sharded_muon_full_padded.numel()
                * p._sharded_muon_full_padded.element_size()
            )
            for p in sharded_muon_params
            if p._sharded_muon_full_padded is not None
        )
        total_unshard_bytes = (
            owner_broadcast_bytes
            + sharded_adamw_all_gather_bytes
            + sharded_muon_all_gather_bytes
        )
        self._last_unshard_total_bytes = total_unshard_bytes
        self._last_unshard_prefetch = bool(prefetch)
        self.comm_ctx.record_forward_unshard_counter(
            group_name,
            dispatch_calls=1,
            prefetch_dispatch_calls=1 if prefetch else 0,
            demand_dispatch_calls=0 if prefetch else 1,
            owner_broadcast_bytes=owner_broadcast_bytes,
            owner_broadcast_max_bucket_bytes=owner_broadcast_max_bucket_bytes,
            owner_broadcast_bucket_count=len(self._packed_buf_by_owner),
            sharded_adamw_all_gather_bytes=sharded_adamw_all_gather_bytes,
            sharded_adamw_param_count=len(sharded_adamw_params),
            sharded_muon_all_gather_bytes=sharded_muon_all_gather_bytes,
            sharded_muon_param_count=len(sharded_muon_params),
            tp_publish_waits=1 if pending_tp_publish else 0,
            replicate_publish_waits=1 if pending_replicate_publish else 0,
            sharded_muon_publish_waits=1 if pending_sharded_muon_publish else 0,
        )
        with torch.cuda.stream(broadcast_stream):
            # Alloc + owner copy-in BEFORE coalescing: these ops execute
            # immediately on broadcast_stream. Wrapped in no_grad +
            # preserve_version_counter so autograd doesn't see the resize /
            # copy_ as an inplace modification of tensors in the compute graph.
            for owner_coord, packed_buf in self._packed_buf_by_owner.items():
                with (
                    torch.no_grad(),
                    torch.autograd._unsafe_preserve_version_counter(packed_buf),
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
                    if _owner_broadcast_enabled(p)
                ]
                with (
                    torch.no_grad(),
                    torch.autograd._unsafe_preserve_version_counter(
                        self._packed_buf_by_owner[owner_coord]
                    ),
                ):
                    torch._foreach_copy_(dsts, srcs)

            if owner_broadcast_bytes:
                owner_start = None
                owner_end = None
                if self.comm_ctx.record_forward_profile:
                    owner_start = torch.cuda.Event(enable_timing=True)
                    owner_end = torch.cuda.Event(enable_timing=True)
                    owner_start.record(broadcast_stream)
                with _profile_range(f"dmuon.unshard_broadcast.dispatch.{group_name}"):
                    with dist._coalescing_manager(group=dp_group, device=self.device):
                        for owner_coord, packed_buf in self._packed_buf_by_owner.items():
                            profile_name = (
                                f"dmuon.unshard_broadcast.bucket."
                                f"owner{owner_coord[0]}_{owner_coord[1]}."
                                f"bytes{int(packed_buf.numel() * packed_buf.element_size())}."
                                f"params{len(self._by_owner[owner_coord])}.{group_name}"
                            )
                            with _profile_range(profile_name):
                                dist.broadcast(
                                    packed_buf,
                                    src=self._global_owner_ranks[owner_coord],
                                    group=dp_group,
                                )
                if owner_start is not None and owner_end is not None:
                    owner_end.record(broadcast_stream)
                    self.comm_ctx.record_forward_unshard_event(
                        group_name=group_name,
                        phase="owner_broadcast_dispatch",
                        start=owner_start,
                        end=owner_end,
                        bytes=owner_broadcast_bytes,
                        prefetch=prefetch,
                    )
        if sharded_adamw_params or sharded_muon_params:
            # DMuon-managed sharded AdamW base params: every shard rank owns a
            # flat shard and all-gathers the full forward view.  A separate
            # stream is optional because these large embedding/head gathers can
            # otherwise queue in front of the first decoder-layer owner
            # broadcasts on the regular forward unshard stream.
            with torch.cuda.stream(sharded_adamw_stream):
                for p in sharded_adamw_params:
                    assert p._sharded_adamw_full_padded is not None
                    assert p._sharded_adamw_comm_shard is not None
                    assert p._sharded_adamw_data is not None
                    with (
                        torch.no_grad(),
                        torch.autograd._unsafe_preserve_version_counter(
                            p._sharded_adamw_full_padded
                        ),
                    ):
                        alloc_storage(p._sharded_adamw_full_padded)
                        p._sharded_adamw_comm_shard.copy_(p._sharded_adamw_data)
                for p in sharded_muon_params:
                    assert p._sharded_muon_full_padded is not None
                    assert p._sharded_muon_comm_shard is not None
                    assert p._sharded_muon_data is not None
                    with (
                        torch.no_grad(),
                        torch.autograd._unsafe_preserve_version_counter(
                            p._sharded_muon_full_padded
                        ),
                    ):
                        alloc_storage(p._sharded_muon_full_padded)
                        p._sharded_muon_comm_shard.copy_(p._sharded_muon_data)

                gather_start = None
                gather_end = None
                if self.comm_ctx.record_forward_profile:
                    gather_start = torch.cuda.Event(enable_timing=True)
                    gather_end = torch.cuda.Event(enable_timing=True)
                    gather_start.record(sharded_adamw_stream)
                with _profile_range(
                    f"dmuon.unshard_all_gather.sharded_adamw.dispatch.{group_name}"
                ):
                    with dist._coalescing_manager(group=dp_group, device=self.device):
                        for p in sharded_adamw_params:
                            profile_name = (
                                f"dmuon.unshard_all_gather.sharded_adamw."
                                f"bytes{int(p._sharded_adamw_full_padded.numel() * p._sharded_adamw_full_padded.element_size())}."
                                f"{p.param_name}.{group_name}"
                            )
                            with _profile_range(profile_name):
                                dist.all_gather_into_tensor(
                                    p._sharded_adamw_full_padded,
                                    p._sharded_adamw_comm_shard,
                                    group=dp_group,
                                )
                        for p in sharded_muon_params:
                            profile_name = (
                                f"dmuon.unshard_all_gather.sharded_muon."
                                f"bytes{int(p._sharded_muon_full_padded.numel() * p._sharded_muon_full_padded.element_size())}."
                                f"{p.param_name}.{group_name}"
                            )
                            with _profile_range(profile_name):
                                dist.all_gather_into_tensor(
                                    p._sharded_muon_full_padded,
                                    p._sharded_muon_comm_shard,
                                    group=dp_group,
                                )
                if gather_start is not None and gather_end is not None:
                    gather_end.record(sharded_adamw_stream)
                    self.comm_ctx.record_forward_unshard_event(
                        group_name=group_name,
                        phase="sharded_param_all_gather_dispatch",
                        start=gather_start,
                        end=gather_end,
                        bytes=(
                            sharded_adamw_all_gather_bytes
                            + sharded_muon_all_gather_bytes
                        ),
                        prefetch=prefetch,
                    )
            self._sharded_adamw_unshard_event = sharded_adamw_stream.record_event()

        self._broadcast_event = broadcast_stream.record_event()

    @dynamo_disable
    def wait_for_unshard(self):
        """GPU-side wait for broadcasts to complete, then finalize params.

        After this call, all dedicated parameters are set on their modules
        and ready for forward/backward compute.
        """
        if self._broadcast_event is None and self._sharded_adamw_unshard_event is None:
            return

        group_name = _group_profile_name(self)
        current_stream = torch.cuda.current_stream()
        if self.comm_ctx.record_forward_profile:
            wait_start = torch.cuda.Event(enable_timing=True)
            wait_end = torch.cuda.Event(enable_timing=True)
            wait_start.record(current_stream)
            if self._broadcast_event is not None:
                current_stream.wait_event(self._broadcast_event)
            if self._sharded_adamw_unshard_event is not None:
                current_stream.wait_event(self._sharded_adamw_unshard_event)
            wait_end.record(current_stream)
            self.comm_ctx.record_forward_unshard_counter(group_name, wait_calls=1)
            self.comm_ctx.record_forward_unshard_event(
                group_name=group_name,
                phase="forward_wait",
                start=wait_start,
                end=wait_end,
                bytes=self._last_unshard_total_bytes,
                prefetch=self._last_unshard_prefetch,
            )
        else:
            if self._broadcast_event is not None:
                current_stream.wait_event(self._broadcast_event)
            if self._sharded_adamw_unshard_event is not None:
                current_stream.wait_event(self._sharded_adamw_unshard_event)

        # Finalize: set unsharded params on modules
        for p in self.params:
            p.finish_unshard()

        self._broadcast_event = None
        self._sharded_adamw_unshard_event = None
        self._is_unsharded = True

    # ---- reshard ----

    @dynamo_disable
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
        if self._sharded_adamw_unshard_event is not None:
            torch.cuda.current_stream().wait_event(self._sharded_adamw_unshard_event)
            self._sharded_adamw_unshard_event = None
        for p in self.params:
            p.reshard()
        from dmuon._core.internal_utils import free_storage

        for packed_buf in self._packed_buf_by_owner.values():
            with (
                torch.no_grad(),
                torch.autograd._unsafe_preserve_version_counter(packed_buf),
            ):
                free_storage(packed_buf)
        for p in self.params:
            if _sharded_adamw_enabled(p):
                full_padded = getattr(p, "_sharded_adamw_full_padded", None)
            elif _sharded_muon_forward_enabled(p):
                full_padded = getattr(p, "_sharded_muon_full_padded", None)
            else:
                continue
            if full_padded is None:
                continue
            with (
                torch.no_grad(),
                torch.autograd._unsafe_preserve_version_counter(full_padded),
            ):
                free_storage(full_padded)
        self._is_unsharded = False

    # ---- gradient reduction — dispatch phase ----

    @dynamo_disable
    def reduce_grads(self):
        """Dispatch gradient reduces. Stage-1 on the shard (``dp_group``) dim,
        and — when HSDP is enabled and ``replicate_grads_enabled`` — Stage-2
        on the replicate dim.  Does NOT wait; call :meth:`wait_for_reduce`
        to synchronize.

        Stream scheduling follows FSDP2's ``foreach_reduce`` pattern
        (``_fsdp_collectives.py:555-588``):

        1. Stage-1 reduce runs on ``reduce_stream``;
        2. When Stage-2 applies, ``replicate_reduce_stream`` waits on
           ``reduce_stream`` via event dependency, then dispatches the
           replicate reduce on itself;
        3. ``_stage1_reduce_event`` records the shard-reduce completion for
           FSDP2-style backward buffer-safety waits, while
           ``_post_reduce_event`` records the full pipeline tail for optimizer
           use.

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
                grad = p.local_grad_for_reduce(grad)
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
        if (
            self._post_reduce_event is not None
            or self._replicate_reduce_state is not None
        ):
            self.wait_for_reduce()

        # Merge any accumulated gradients from prior no_sync steps
        for p in self.params:
            if p._accumulated_grad is not None and p._is_unsharded:
                if p._unsharded_param.grad is not None:
                    grad = p._unsharded_param.grad.data
                    grad = p.local_grad_for_reduce(grad)
                    grad.add_(p._accumulated_grad)
                else:
                    p._unsharded_param.grad = p._accumulated_grad.clone()
                p._accumulated_grad = None

        reduce_stream = self.comm_ctx.reduce_stream
        replicate_stream = self.comm_ctx.replicate_reduce_stream
        replicate_group = self.comm_ctx.replicate_group
        has_replicate = replicate_group is not None and self.replicate_grads_enabled

        # Ensure gradients are computed before reduce_stream reads them
        reduce_stream.wait_stream(torch.cuda.current_stream())

        self._pending_reduce = []
        # ``stage2_pending``: per-(grad_buf, param) records; only shard-owner
        # ranks populate this list for owner-managed params.  Sharded AdamW
        # params populate ``sharded_stage2_pending`` on every shard rank after
        # reduce-scatter, then use replicate all-reduce for HSDP.
        stage2_pending: list[tuple[torch.Tensor, DedicatedParam]] = []
        sharded_stage2_pending: list[tuple[torch.Tensor, DedicatedParam]] = []
        dp_group = self._dp_group
        my_shard_rank = dp_group.rank()

        group_name = str(getattr(self, "_debug_name", None) or f"group_{id(self):x}")

        # Stage 1 — shard reduce.  Coalesce all reduces into a single fused
        # NCCL kernel.  Grad tensor refs are saved on ``_pending_reduce`` so
        # that ``reshard()`` freeing ``_unsharded_param``'s storage does not
        # dangle the post-reduce view.
        with _profile_range(f"dmuon.stage1_shard_reduce.{group_name}"):
            with torch.cuda.stream(reduce_stream):
                with dist._coalescing_manager(
                    group=dp_group, device=self.device
                ):
                    for p in self.params:
                        if _sharded_adamw_enabled(p):
                            continue
                        if not p._is_unsharded or p._unsharded_param.grad is None:
                            continue
                        grad = p._unsharded_param.grad.data
                        grad = p.local_grad_for_reduce(grad)
                        grad = grad.contiguous()
                        dist.reduce(
                            grad,
                            dst=self._global_owner_ranks[p.owner_rank],
                            op=dist.ReduceOp.AVG,
                            group=dp_group,
                        )
                        p._unsharded_param.grad = None
                        self._pending_reduce.append((grad.view(-1), [p]))
                        # Only the shard-owner rank holds the post-Stage-1 grad
                        # and therefore participates in Stage-2.
                        if has_replicate and my_shard_rank == p.owner_shard:
                            stage2_pending.append((grad, p))

                sharded_work = [
                    p
                    for p in self.params
                    if _sharded_adamw_enabled(p)
                    and p._is_unsharded
                    and p._unsharded_param.grad is not None
                ]
                if sharded_work:
                    with dist._coalescing_manager(
                        group=dp_group, device=self.device
                    ):
                        for p in sharded_work:
                            assert p._sharded_adamw_reduce_input is not None
                            assert p._sharded_adamw_comm_shard is not None
                            grad = p._unsharded_param.grad.data
                            grad = p.local_grad_for_reduce(grad)
                            flat_grad = grad.contiguous().view(-1)
                            reduce_input = p._sharded_adamw_reduce_input
                            reduce_input.zero_()
                            reduce_input[: flat_grad.numel()].copy_(flat_grad)
                            p._sharded_adamw_grad = torch.empty_like(
                                p._sharded_adamw_comm_shard
                            )
                            dist.reduce_scatter_tensor(
                                p._sharded_adamw_grad,
                                reduce_input,
                                op=dist.ReduceOp.AVG,
                                group=dp_group,
                            )
                            p._unsharded_param.grad = None
                            if has_replicate:
                                sharded_stage2_pending.append(
                                    (p._sharded_adamw_grad, p)
                                )

        stage1_event = reduce_stream.record_event()
        self._stage1_reduce_event = stage1_event

        # Stage 2 — replicate reduce.  Only shard-owner ranks dispatch; the
        # non-shard-owner grads are already "garbage" after Stage 1 (undefined
        # per NCCL spec).  The pipeline's tail event is recorded on whichever
        # stream ran last.
        combined_stage2_pending = stage2_pending + sharded_stage2_pending
        if has_replicate and combined_stage2_pending:
            stage2_event = self._launch_stage2_replicate_reduce(
                combined_stage2_pending,
                wait_event=stage1_event,
                group_name=group_name,
            )
            # Keep the Stage-1 tensor + Stage-2 event alive until
            # ``wait_for_reduce`` runs — mirrors FSDP2 AllReduceState.
            self._replicate_reduce_state = ReplicateReduceState(
                replicate_input=combined_stage2_pending[0][0],
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
                    self._partial_reduce_by_param[id(p)] = grad.view(
                        p._orig_size
                    ).clone()
            self._post_reduce_event = stage1_event
        else:
            self._post_reduce_event = stage1_event

    def _launch_stage2_replicate_reduce(
        self,
        stage2_pending: list[tuple[torch.Tensor, DedicatedParam]],
        *,
        wait_event: Optional[torch.cuda.Event],
        group_name: str,
    ) -> torch.cuda.Event:
        replicate_group = self.comm_ctx.replicate_group
        assert replicate_group is not None
        replicate_stream = self.comm_ctx.replicate_reduce_stream
        with _profile_range(f"dmuon.stage2_replicate_reduce.{group_name}"):
            if wait_event is not None:
                replicate_stream.wait_event(wait_event)
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
                        if (
                            _adamw_replicate_allreduce_enabled(p)
                            or _sharded_adamw_enabled(p)
                        ):
                            dist.all_reduce(
                                grad,
                                op=dist.ReduceOp.AVG,
                                group=replicate_group,
                            )
                        else:
                            dist.reduce(
                                grad,
                                dst=p._owner_replicate_global_rank,
                                op=dist.ReduceOp.AVG,
                                group=replicate_group,
                            )
        return replicate_stream.record_event()

    def _retain_pending_reduce_after_stage1(self, p: DedicatedParam) -> bool:
        """Whether this rank must keep ``p``'s Stage-1 grad ref alive.

        After Stage-1 completes, non-shard-owner ranks no longer need their
        reduce input tensors.  Shard-owner ranks must keep them until Stage-2
        finishes because the replicate reduce may still be reading them.  The
        final global owner (or the local AdamW all-reduce owner) also needs the
        tensor for ``wait_for_reduce()`` to materialize ``_reduced_grad``.
        """
        local_adamw_owner = (
            _adamw_replicate_allreduce_enabled(p)
            and p._owned_data is not None
        )
        if p.is_owner or local_adamw_owner:
            return True
        replicate_group = self.comm_ctx.replicate_group
        if replicate_group is None:
            return False
        return self._dp_group.rank() == p.owner_shard

    def _compact_pending_reduce_after_stage1(self) -> None:
        if not self._pending_reduce:
            return
        self._pending_reduce = [
            (grad_buf, plist)
            for grad_buf, plist in self._pending_reduce
            if grad_buf is not None
            and plist
            and self._retain_pending_reduce_after_stage1(plist[0])
        ]

    @dynamo_disable
    def wait_for_stage1_reduce(
        self, stream: Optional[torch.cuda.Stream] = None
    ) -> Optional[torch.cuda.Event]:
        """Wait only for the Stage-1 shard reduce and trim dead grad refs.

        This is the DMuon equivalent of FSDP2's inter-group
        ``reduce_scatter_state.event`` wait.  It bounds the lifetime of
        shard-reduce input buffers during backward, but deliberately avoids
        waiting the HSDP Stage-2 replicate reduce.  The Stage-2 tail is a true
        optimizer dependency and is drained by :meth:`wait_for_reduce` at the
        optimizer/root-post-backward boundary.
        """
        if self._stage1_reduce_event is None:
            return None

        target_stream = stream if stream is not None else torch.cuda.current_stream()

        def _wait_and_compact() -> None:
            target_stream.wait_event(self._stage1_reduce_event)
            self._compact_pending_reduce_after_stage1()

        if stream is None:
            _wait_and_compact()
        else:
            with torch.cuda.stream(stream):
                _wait_and_compact()

        self._stage1_reduce_event = None
        return None

    @dynamo_disable
    def wait_for_reduce(
        self, stream: Optional[torch.cuda.Stream] = None
    ) -> Optional[torch.cuda.Event]:
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
            return None

        target_stream = stream if stream is not None else torch.cuda.current_stream()
        replicate_state = self._replicate_reduce_state

        def _wait_and_unpack() -> None:
            if self._post_reduce_event is not None:
                target_stream.wait_event(self._post_reduce_event)
            if replicate_state is not None and replicate_state.event is not None:
                target_stream.wait_event(replicate_state.event)

            for grad_buf, plist in self._pending_reduce:
                if grad_buf is None:
                    continue
                p = plist[0]
                local_adamw_owner = (
                    _adamw_replicate_allreduce_enabled(p)
                    and p._owned_data is not None
                )
                if not (p.is_owner or local_adamw_owner):
                    continue
                new_grad = grad_buf.view(p._orig_size)
                if p._reduced_grad is not None:
                    p._reduced_grad.add_(new_grad)
                else:
                    p._reduced_grad = new_grad.clone()

        if stream is None:
            _wait_and_unpack()
        else:
            with torch.cuda.stream(stream):
                _wait_and_unpack()

        self._post_reduce_event = None
        self._stage1_reduce_event = None
        self._replicate_reduce_state = None
        self._muon_grad_ready_refs = [
            grad_buf
            for grad_buf, _plist in self._pending_reduce
            if grad_buf is not None
        ]
        if replicate_state is not None:
            self._muon_grad_ready_refs.append(replicate_state.replicate_input)
        self._pending_reduce = []
        self._muon_grad_ready_event = target_stream.record_event()
        return self._muon_grad_ready_event

    # ---- TP gather (T2a) -------------------------------------------------

    def tp_gather_grads(
        self, *, wait_current_stream: bool = True
    ) -> Optional[torch.cuda.Event]:
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
            self._tp_gather_event = None
            self._tp_gather_refs = []
            self._tp_gather_pending_full_grads = []
            return None

        # Compatibility path: wait_for_reduce may have unpacked _reduced_grad
        # on the current compute stream, so the gather must follow it.
        # Per-group prefetch unpacks on reduce_stream and disables this wait
        # so later groups' gathers can overlap current-group NS work.
        if wait_current_stream:
            reduce_stream.wait_stream(torch.cuda.current_stream())

        grouped_work = []
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
            self._tp_gather_pending_full_grads = []
            pending_full_grads: list[
                tuple[DedicatedParam, list[torch.Tensor], torch.Tensor, int, Optional[torch.Tensor]]
            ] = []
            reuse_buffers = _tp_gather_buffer_reuse_enabled(self.comm_ctx)
            for tp_group, group_work in grouped_work:
                with dist._coalescing_manager(group=tp_group, device=self.device):
                    for p, local_grad in group_work:
                        tp_size = p.tp_group.size()
                        if p.is_tp_owner:
                            # Allocate the full (M, N) buffer fresh every step
                            # (MVP — see tp_design.md §6.7 note on pre-alloc
                            # deferral).  gather_list entries must be separate
                            # tensors; we cat them post-gather along shard_dim.
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
        """Build TP-owner full grads after gather collectives have completed."""
        pending = self._tp_gather_pending_full_grads
        if not pending:
            return
        with _profile_range("dmuon.tp_gather_materialize_full_grads"):
            for p, recv_bufs, local_grad, shard_dim, full_grad_buf in pending:
                recv_bufs[p.tp_group.rank()] = local_grad
                if full_grad_buf is None:
                    p._tp_full_grad = torch.cat(recv_bufs, dim=shard_dim)
                else:
                    torch.cat(recv_bufs, dim=shard_dim, out=full_grad_buf)
                    p._tp_full_grad = full_grad_buf
        self._tp_gather_pending_full_grads = []

    def wait_for_tp_gather(self) -> None:
        """Wait until this group's TP gathered grads are visible to compute."""
        if self._tp_gather_event is None:
            return
        torch.cuda.current_stream().wait_event(self._tp_gather_event)
        self._materialize_tp_gathered_grads()
        self._tp_gather_event = None
        self._tp_gather_refs = []
        self._muon_grad_ready_refs = []
        self._muon_grad_ready_event = None

    # ---- TP scatter (T2b) ------------------------------------------------

    def _tp_scatter_dispatch(self) -> Optional[list[torch.Tensor]]:
        """Shared dispatch body for sync + async TP scatter.

        Queues the scatter + ``_owned_data.mul_(wd).add_(shard)`` fuse on
        ``replicate_broadcast_stream`` and returns transient TP-owner
        ``scatter_list`` refs.  Callers record an event and store those refs
        in :attr:`_tp_scatter_state` until the scatter stream has consumed
        them.

        Returns ``None`` when the group has no TP-sharded params with
        pending updates (no dispatch happened).
        """
        work: list[DedicatedParam] = [
            p
            for p in self.params
            if p.tp_group is not None and p._reduced_grad is not None
        ]
        if not work:
            return None

        bcast_stream = self.comm_ctx.replicate_broadcast_stream
        bcast_stream.wait_stream(torch.cuda.current_stream())
        # TP grad-gather prefetch uses ``reduce_stream`` while TP scatter uses
        # ``replicate_broadcast_stream``.  Both issue collectives on the same
        # TP process group, so preserve one global launch order across streams;
        # otherwise a prefetched gather for group N+1 can race/reorder with
        # group N's post-step scatter and corrupt the next forward.
        bcast_stream.wait_stream(self.comm_ctx.reduce_stream)

        refs: list[torch.Tensor] = []
        updates: list[tuple[DedicatedParam, torch.Tensor]] = []
        grouped_work = []
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
                            # The contiguous split copies are enqueued on
                            # bcast_stream below. Keep the source full-delta
                            # alive until the scatter event is consumed;
                            # otherwise async group pipelining can let later
                            # NS allocations reuse its storage before those
                            # copies have actually read it.
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
                            # ``dist.scatter`` is needed for the remote TP
                            # peers, but the source rank's output tensor is
                            # not a portable way to obtain its own shard.
                            # Use the already materialized split matching the
                            # local TP rank, otherwise deterministic debug
                            # fill can expose an uninitialized recv tensor.
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

            # Keep the local weight update explicitly after the coalesced
            # scatter dispatch.  This avoids relying on private
            # _coalescing_manager launch timing for recv_shard readiness.
            for p, update_shard in updates:
                p._owned_data.mul_(p._tp_wd_factor).add_(update_shard)
                update_shard.record_stream(bcast_stream)
                p._tp_full_grad = None
                p._tp_full_delta = None
                p._reduced_grad = None
        return refs

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
        refs = self._tp_scatter_dispatch()
        if refs is None:
            return
        event = self.comm_ctx.replicate_broadcast_stream.record_event()
        # Sync contract: compute stream waits for the scatter to land before
        # the caller can read ``_owned_data``. This is a GPU-side wait, not a
        # CPU block, so keep TP-owner split tensors alive until the event is
        # consumed by the next pre-forward/wait_all drain.
        torch.cuda.current_stream().wait_event(event)
        self._tp_scatter_state = TPScatterState(refs=refs, event=event)

    def tp_scatter_delta_async(self) -> None:
        """T2d async variant of :meth:`tp_scatter_delta`.

        Dispatches the scatter on ``replicate_broadcast_stream`` and
        returns immediately; records an event + pins transient TP-owner send
        split tensors in :attr:`_tp_scatter_state` so Python cannot release
        them before NCCL consumes ``scatter_list``. The event is consumed by
        the NEXT iteration's :meth:`_pre_forward_wait` — mirrors the
        replicate-broadcast cross-call event chain.

        Double-dispatch (state still PENDING) is a programming error.
        """
        if self._tp_scatter_state is not None:
            group_name = getattr(self, "_debug_name", "<unknown>")
            raise RuntimeError(
                f"tp_scatter_delta_async[{group_name}]: previous event still pending — "
                "pre_forward_wait was not consumed before the next dispatch"
            )
        refs = self._tp_scatter_dispatch()
        if refs is None:
            return
        event = self.comm_ctx.replicate_broadcast_stream.record_event()
        self._tp_scatter_state = TPScatterState(refs=refs, event=event)

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
        if not any(
            my_shard_rank == p.owner_shard
            and _needs_post_step_replicate_broadcast(p)
            for p in self.params
        ):
            return

        bcast_stream = self.comm_ctx.replicate_broadcast_stream
        bcast_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(bcast_stream):
            with dist._coalescing_manager(group=replicate_group, device=self.device):
                for p in self.params:
                    if my_shard_rank != p.owner_shard:
                        continue
                    if not _needs_post_step_replicate_broadcast(p):
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
            - Double dispatch (calling while a state is still PENDING) is
              a programming error.

        The wait point MUST be before ``unshard()``'s copy-in reads
        ``_owned_data``; checkpoint save paths must also drain pending
        state via :func:`dmuon.utils.wait_all_replicate_broadcasts`.
        """
        replicate_group = self.comm_ctx.replicate_group
        if replicate_group is None:
            return

        if self._replicate_broadcast_state is not None:
            group_name = getattr(self, "_debug_name", "<unknown>")
            raise RuntimeError(
                f"replicate_broadcast_async[{group_name}]: previous event still pending — "
                "pre_forward_wait was not consumed before the next dispatch"
            )

        my_shard_rank = self._dp_group.rank()
        eligible_params = [
            p
            for p in self.params
            if my_shard_rank == p.owner_shard
            and _needs_post_step_replicate_broadcast(p)
        ]
        if not eligible_params:
            return

        bcast_stream = self.comm_ctx.replicate_broadcast_stream
        bcast_stream.wait_stream(torch.cuda.current_stream())

        # Pick one ``_owned_data`` tensor to pin via the state tuple.  Any
        # rank-participating param works; allocator arena is shared within
        # the coalescing manager below.
        pin_ref: Optional[torch.Tensor] = eligible_params[0]._owned_data
        bucket_limit = self.comm_ctx.replicate_broadcast_bucket_bytes
        if bucket_limit <= 0:
            buckets = [eligible_params]
        else:
            buckets: list[list[DedicatedParam]] = []
            current: list[DedicatedParam] = []
            current_bytes = 0
            for p in eligible_params:
                payload_bytes = _param_payload_bytes(p)
                if current and current_bytes + payload_bytes > bucket_limit:
                    buckets.append(current)
                    current = []
                    current_bytes = 0
                current.append(p)
                current_bytes += payload_bytes
            if current:
                buckets.append(current)

        group_name = _group_profile_name(self)
        with torch.cuda.stream(bcast_stream):
            for bucket_idx, bucket in enumerate(buckets):
                bucket_bytes = sum(_param_payload_bytes(p) for p in bucket)
                with _profile_range(
                    f"dmuon.replicate_broadcast.bucket."
                    f"idx{bucket_idx}.bytes{bucket_bytes}."
                    f"params{len(bucket)}.{group_name}"
                ):
                    with dist._coalescing_manager(
                        group=replicate_group, device=self.device
                    ):
                        for p in bucket:
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

    def sharded_muon_publish_async(self) -> None:
        """Publish updated Muon owner tensors into rank-local forward shards.

        This opt-in path moves Muon-managed parameters from owner-broadcast
        forward placement to FSDP-style all-gather forward placement.  The
        owner still executes the full-matrix update into ``_owned_data``; this
        post-step publish uses reduce-scatter with zero inputs on non-owner
        ranks so the next forward can reconstruct the full view with all-gather.
        """
        work = [p for p in self.params if _sharded_muon_forward_enabled(p)]
        if not work:
            return
        if self._sharded_muon_publish_state is not None:
            group_name = getattr(self, "_debug_name", "<unknown>")
            raise RuntimeError(
                f"sharded_muon_publish_async[{group_name}]: previous event still "
                "pending — pre_forward_wait was not consumed before the next dispatch"
            )

        dp_group = self._dp_group
        replicate_group = self.comm_ctx.replicate_group
        local_shard_rank = dp_group.rank()
        local_replicate_rank = replicate_group.rank() if replicate_group else 0
        publish_stream = self.comm_ctx.replicate_broadcast_stream
        publish_stream.wait_stream(torch.cuda.current_stream())
        if self._tp_scatter_state is not None:
            publish_stream.wait_event(self._tp_scatter_state.event)
        if self._replicate_broadcast_state is not None:
            publish_stream.wait_event(self._replicate_broadcast_state.event)

        refs: list[torch.Tensor] = []
        group_name = _group_profile_name(self)
        with torch.cuda.stream(publish_stream):
            if replicate_group is None:
                with _profile_range(
                    f"dmuon.sharded_muon_publish.reduce_scatter.{group_name}"
                ):
                    with dist._coalescing_manager(group=dp_group, device=self.device):
                        for p in work:
                            assert p._sharded_muon_comm_shard is not None
                            assert p._sharded_muon_data is not None
                            assert p._sharded_muon_scatter_input is not None
                            scatter_input = p._sharded_muon_scatter_input
                            scatter_input.zero_()
                            if local_shard_rank == p.owner_shard:
                                assert p._owned_data is not None
                                scatter_input[: p.numel].copy_(p._owned_data.reshape(-1))
                            dist.reduce_scatter_tensor(
                                p._sharded_muon_comm_shard,
                                scatter_input,
                                op=dist.ReduceOp.SUM,
                                group=dp_group,
                            )
                            refs.extend(
                                (
                                    scatter_input,
                                    p._sharded_muon_comm_shard,
                                )
                            )
            else:
                # HSDP has a 2D mesh: shard groups reconstruct each row's
                # forward view, while replicate groups keep rows consistent.
                # For all-gather forward placement, avoid a full-matrix
                # replicate broadcast.  The global owner row first scatters
                # the updated matrix into shard-local chunks; then each shard
                # column broadcasts only its chunk across the replicate axis.
                with _profile_range(
                    f"dmuon.sharded_muon_publish.owner_row_reduce_scatter.{group_name}"
                ):
                    row_work = [
                        p
                        for p in work
                        if local_replicate_rank == p.owner_replicate
                    ]
                    if row_work:
                        with dist._coalescing_manager(
                            group=dp_group, device=self.device
                        ):
                            for p in row_work:
                                assert p._sharded_muon_comm_shard is not None
                                assert p._sharded_muon_data is not None
                                assert p._sharded_muon_scatter_input is not None
                                scatter_input = p._sharded_muon_scatter_input
                                scatter_input.zero_()
                                if local_shard_rank == p.owner_shard:
                                    assert p._owned_data is not None
                                    scatter_input[: p.numel].copy_(
                                        p._owned_data.reshape(-1)
                                    )
                                dist.reduce_scatter_tensor(
                                    p._sharded_muon_comm_shard,
                                    scatter_input,
                                    op=dist.ReduceOp.SUM,
                                    group=dp_group,
                                )
                                refs.extend(
                                    (
                                        scatter_input,
                                        p._sharded_muon_comm_shard,
                                    )
                                )
                with _profile_range(
                    f"dmuon.sharded_muon_publish.shard_column_broadcast.{group_name}"
                ):
                    with dist._coalescing_manager(
                        group=replicate_group, device=self.device
                    ):
                        for p in work:
                            assert p._sharded_muon_comm_shard is not None
                            assert p._sharded_muon_data is not None
                            src_rank = dist.get_global_rank(
                                replicate_group, p.owner_replicate
                            )
                            dist.broadcast(
                                p._sharded_muon_comm_shard,
                                src=src_rank,
                                group=replicate_group,
                            )

            for p in work:
                assert p._sharded_muon_comm_shard is not None
                assert p._sharded_muon_data is not None
                p._sharded_muon_data.copy_(
                    p._sharded_muon_comm_shard.to(dtype=p._sharded_muon_data.dtype)
                )
                p._sharded_muon_comm_shard.record_stream(publish_stream)
                p._sharded_muon_initialized = True

        self._sharded_muon_publish_state = ShardedMuonPublishState(
            refs=refs,
            event=publish_stream.record_event(),
        )

    def sharded_muon_publish_sync(self) -> None:
        """Synchronous variant of :meth:`sharded_muon_publish_async`."""
        self.sharded_muon_publish_async()
        self.wait_for_sharded_muon_publish()

    def wait_for_sharded_muon_publish(self) -> None:
        if self._sharded_muon_publish_state is None:
            return
        torch.cuda.current_stream().wait_event(self._sharded_muon_publish_state.event)
        self._sharded_muon_publish_state = None

    def _pre_forward_prefetch_publish(self) -> None:
        """Dispatch pending publish for a prefetched unshard without
        blocking the current compute stream.

        Actual forward entry uses :meth:`_pre_forward_wait` and clears the
        async state.  A prefetch orders every forward-unshard stream that may
        consume the published shard after the HSDP publish event, so the caller
        can continue running the current layer while the next group's
        publish/unshard pipeline progresses on communication streams.
        """

        streams = [self.comm_ctx.broadcast_stream]
        if bool(
            getattr(
                self.comm_ctx,
                "sharded_adamw_unshard_separate_stream_enabled",
                False,
            )
        ):
            streams.append(self.comm_ctx.sharded_adamw_unshard_stream)
        tp_state = self._tp_scatter_state
        if tp_state is not None:
            for stream in streams:
                stream.wait_event(tp_state.event)
        state = self._replicate_broadcast_state
        if state is not None:
            for stream in streams:
                stream.wait_event(state.event)
        sharded_state = self._sharded_muon_publish_state
        if sharded_state is not None:
            for stream in streams:
                stream.wait_event(sharded_state.event)

    @dynamo_disable
    def _pre_forward_wait(self) -> None:
        """Consume any pending async replicate broadcast before
        the shard-dim unshard's copy-in reads ``_owned_data``.

        No-op when the state is idle.
        """
        # T2d: drain the async TP scatter state first.  The scatter is
        # what writes ``_owned_data`` on every DP-owner TP rank; the
        # replicate broadcast fans that value out.  In pure-DP mode (no
        # replicate_group) the scatter state is the only thing to wait
        # on; in HSDP mode both fire on the same stream and waiting the
        # replicate broadcast event implicitly covers the scatter, but
        # we still wait the scatter event explicitly so the state tuple
        # (and its pinned TP-owner send refs) can be released.
        group_name = _group_profile_name(self)
        current_stream = torch.cuda.current_stream()
        tp_state = self._tp_scatter_state
        if tp_state is not None:
            start = end = None
            if self.comm_ctx.record_forward_profile:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(current_stream)
            current_stream.wait_event(tp_state.event)
            if start is not None and end is not None:
                end.record(current_stream)
                self.comm_ctx.record_forward_unshard_counter(
                    group_name, tp_publish_waits=1
                )
                self.comm_ctx.record_forward_unshard_event(
                    group_name=group_name,
                    phase="pre_forward_tp_publish_wait_direct",
                    start=start,
                    end=end,
                    bytes=0,
                    prefetch=False,
                )
            self._tp_scatter_state = None

        state = self._replicate_broadcast_state
        if state is not None:
            start = end = None
            if self.comm_ctx.record_forward_profile:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(current_stream)
            current_stream.wait_event(state.event)
            if start is not None and end is not None:
                end.record(current_stream)
                self.comm_ctx.record_forward_unshard_counter(
                    group_name, replicate_publish_waits=1
                )
                self.comm_ctx.record_forward_unshard_event(
                    group_name=group_name,
                    phase="pre_forward_replicate_publish_wait_direct",
                    start=start,
                    end=end,
                    bytes=0,
                    prefetch=False,
                )
            self._replicate_broadcast_state = None

        sharded_state = self._sharded_muon_publish_state
        if sharded_state is None:
            return

        start = end = None
        if self.comm_ctx.record_forward_profile:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(current_stream)
        current_stream.wait_event(sharded_state.event)
        if start is not None and end is not None:
            end.record(current_stream)
            self.comm_ctx.record_forward_unshard_counter(
                group_name, sharded_muon_publish_waits=1
            )
            self.comm_ctx.record_forward_unshard_event(
                group_name=group_name,
                phase="pre_forward_sharded_muon_publish_wait_direct",
                start=start,
                end=end,
                bytes=0,
                prefetch=False,
            )
        self._sharded_muon_publish_state = None

    # ---- backward prefetch ----

    @dynamo_disable
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
        target_group.unshard(prefetch=True)  # dispatch only — no wait

    @dynamo_disable
    def _record_post_forward(self) -> None:
        """Record this group's position in forward order for backward prefetch."""
        post_forward_index = len(self.comm_ctx.post_forward_order)
        self.comm_ctx.post_forward_order.append(self)
        self._last_post_forward_index = post_forward_index
        self._post_forward_indices.append(post_forward_index)
