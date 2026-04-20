"""DedicatedParamGroup: manages communication for dedicated params in one layer.

Uses dedicated CUDA streams for broadcast/reduce (analogous to FSDP2's
all_gather_stream / reduce_scatter_stream) and CUDA events for GPU-side
synchronization instead of CPU-blocking work.wait().
"""

from collections import defaultdict
from typing import Optional

import torch
import torch.distributed as dist

from .comm import DedicatedCommContext
from .param import DedicatedParam

try:
    from torch.distributed.tensor import DTensor as _DTensor
except ImportError:
    _DTensor = None


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
        # all of an owner's params into a single NCCL call.
        self._by_owner: dict[int, list[DedicatedParam]] = defaultdict(list)
        for p in params:
            self._by_owner[p.owner_rank].append(p)

        self.reduce_grads_enabled: bool = True

        # Event-based synchronization (replaces work.wait())
        self._broadcast_event: Optional[torch.cuda.Event] = None
        self._reduce_event: Optional[torch.cuda.Event] = None

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
        # Map {owner_rank → global rank} for all owners represented in params.
        # Phase 2 iterates params directly (no _by_owner grouping) but still
        # needs the global-rank lookup for dist.reduce(dst=...).
        self._total_numel_by_owner: dict[int, int] = {
            owner: sum(p.numel for p in owner_params)
            for owner, owner_params in self._by_owner.items()
        }
        if self._dp_group is not None:
            self._global_owner_ranks: dict[int, int] = {
                owner: dist.get_global_rank(self._dp_group, owner)
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
        from ._internal_utils import free_storage
        self._packed_buf_by_owner: dict[int, torch.Tensor] = {}
        self._copy_in_dsts_by_owner: dict[int, list[torch.Tensor]] = {}
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
        """
        if self._is_unsharded:
            return  # still unsharded from forward (reshard_after_forward=False)
        if self._broadcast_event is not None:
            return  # already dispatched, pending wait_for_unshard

        broadcast_stream = self.comm_ctx.broadcast_stream
        broadcast_stream.wait_stream(torch.cuda.current_stream())

        from ._internal_utils import alloc_storage
        dp_group = self._dp_group
        local_rank = dp_group.rank()
        with torch.cuda.stream(broadcast_stream):
            # Alloc + owner copy-in BEFORE coalescing: these ops execute
            # immediately on broadcast_stream. Wrapped in no_grad +
            # preserve_version_counter so autograd doesn't see the resize /
            # copy_ as an inplace modification of tensors in the compute graph.
            for owner_rank, packed_buf in self._packed_buf_by_owner.items():
                with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(
                    packed_buf
                ):
                    alloc_storage(packed_buf)

            # Phase 3: batch owner copy-in with torch._foreach_copy_.
            # Only the owner rank for a given owner_rank has non-empty
            # _owned_data; other ranks skip.
            if local_rank in self._copy_in_dsts_by_owner:
                dsts = self._copy_in_dsts_by_owner[local_rank]
                srcs = [
                    p._owned_data.view(-1) for p in self._by_owner[local_rank]
                ]
                with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(
                    self._packed_buf_by_owner[local_rank]
                ):
                    torch._foreach_copy_(dsts, srcs)

            with dist._coalescing_manager(group=dp_group, device=self.device):
                for owner_rank, packed_buf in self._packed_buf_by_owner.items():
                    dist.broadcast(
                        packed_buf,
                        src=self._global_owner_ranks[owner_rank],
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
        """
        for p in self.params:
            p.reshard()
        from ._internal_utils import free_storage
        for packed_buf in self._packed_buf_by_owner.values():
            with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(
                packed_buf
            ):
                free_storage(packed_buf)
        self._is_unsharded = False

    # ---- gradient reduction — dispatch phase ----

    def reduce_grads(self):
        """Dispatch gradient reduces on reduce_stream. Does NOT wait.

        Call wait_for_reduce() to synchronize and unpack gradients on owner.

        When ``reduce_grads_enabled`` is False (no_sync mode), gradients are
        accumulated locally on every rank without communication. The accumulated
        gradients are merged into the next sync reduce automatically.
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
        # we dispatch a new reduce that would overwrite _pending_reduce/_reduce_event.
        if self._reduce_event is not None:
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
        # Ensure gradients are computed before reduce_stream reads them
        reduce_stream.wait_stream(torch.cuda.current_stream())

        self._pending_reduce = []
        dp_group = self._dp_group
        with torch.cuda.stream(reduce_stream):
            # Coalesce all reduces into a single fused NCCL kernel.
            # Phase 2 removed _packed_reduce — each param reduces its own grad
            # in-place, coalescing_manager fuses them into one NCCL call.
            # We save the grad tensor ref here because reshard() will free
            # _unsharded_param's storage before wait_for_reduce() can unpack.
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

        self._reduce_event = reduce_stream.record_event()

    def wait_for_reduce(self):
        """GPU-side wait for reduces to complete, then save owner grad.

        Each ``_pending_reduce`` entry is ``(grad_tensor_ref, [param])``. The
        reduce was in-place on grad_tensor_ref, which was saved during
        ``reduce_grads`` before ``reshard()`` freed the storage. On owner
        rank, the grad_tensor_ref now holds the averaged grad — copy it
        into ``_reduced_grad`` for the optimizer step.
        """
        if self._reduce_event is None:
            return

        torch.cuda.current_stream().wait_event(self._reduce_event)

        for grad_buf, plist in self._pending_reduce:
            if grad_buf is None:
                continue
            p = plist[0]
            if self._dp_group.rank() != p.owner_rank:
                continue
            new_grad = grad_buf.view(p._orig_size)
            if p._reduced_grad is not None:
                p._reduced_grad.add_(new_grad)
            else:
                p._reduced_grad = new_grad.clone()

        self._reduce_event = None
        self._pending_reduce = []

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
