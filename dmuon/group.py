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

        # Pre-group by owner to avoid recomputing each time
        self._by_owner: dict[int, list[DedicatedParam]] = defaultdict(list)
        for p in params:
            self._by_owner[p.owner_rank].append(p)

        self._packed_bufs: list[torch.Tensor] = []  # keep packed buffers alive until finish
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

        # Saved forward-time unsharded params. Autograd writes .grad to the
        # tensor instances that were in the graph at forward time; after reshard
        # + re-unshard in pre_backward, dp._unsharded_param points to a NEW
        # tensor. post_backward transfers .grad from forward-time refs (saved
        # here) to the current _unsharded_param. Populated in _pre_forward,
        # cleared in _run_post_backward.
        self._forward_time_params: Optional[list[tuple]] = None

        # Cached per-owner metadata (FSDP2 alignment phase 1 — previously
        # recomputed in every _packed_broadcast / _packed_reduce call).
        self._dp_group: Optional[dist.ProcessGroup] = (
            params[0].dp_group if params else None
        )
        self._comm_dtype: Optional[torch.dtype] = (
            (params[0]._compute_dtype or params[0]._orig_dtype) if params else None
        )
        self._total_numel_by_owner: dict[int, int] = {
            owner: sum(p.numel for p in owner_params)
            for owner, owner_params in self._by_owner.items()
        }
        self._global_owner_ranks: dict[int, int] = (
            {
                owner: dist.get_global_rank(self._dp_group, owner)
                for owner in self._by_owner
            }
            if self._dp_group is not None
            else {}
        )

    # ---- unshard (broadcast) — dispatch phase ----

    def unshard(self):
        """Dispatch broadcasts on broadcast_stream. Does NOT wait.

        Call wait_for_unshard() to synchronize before using the parameters.
        No-op if already dispatched (pending wait) or already unsharded.
        """
        if self._is_unsharded:
            return  # still unsharded from forward (reshard_after_forward=False)
        if self._broadcast_event is not None:
            return  # already dispatched, pending wait_for_unshard

        broadcast_stream = self.comm_ctx.broadcast_stream
        # Ensure any prior compute (e.g., owner's data writes) is visible to broadcast_stream
        broadcast_stream.wait_stream(torch.cuda.current_stream())

        self._packed_bufs = []
        dp_group = self._dp_group
        with torch.cuda.stream(broadcast_stream):
            # Coalesce all broadcasts into a single fused NCCL kernel
            with dist._coalescing_manager(group=dp_group, device=self.device):
                for owner_rank, owner_params in self._by_owner.items():
                    if len(owner_params) == 1:
                        owner_params[0].alloc_and_broadcast(async_op=False)
                    else:
                        self._packed_broadcast(owner_rank, owner_params)

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
        self._packed_bufs = []
        self._is_unsharded = True

    def _packed_broadcast(self, owner_rank: int, params: list[DedicatedParam]) -> None:
        """Pack multiple params from the same owner into one broadcast.

        Caller must be on broadcast_stream.
        """
        total_numel = self._total_numel_by_owner[owner_rank]
        comm_dtype = self._comm_dtype
        buf = torch.empty(total_numel, dtype=comm_dtype, device=self.device)

        # Owner: copy data into packed buffer (copy_ handles dtype conversion inline)
        dp_group = self._dp_group
        if dp_group.rank() == owner_rank:
            offset = 0
            for p in params:
                buf[offset : offset + p.numel].copy_(p._owned_data.view(-1))
                offset += p.numel

        dist.broadcast(buf, src=self._global_owner_ranks[owner_rank], group=dp_group)

        # Store buffer ref and assign slices for finish_unshard
        self._packed_bufs.append(buf)
        offset = 0
        for p in params:
            p._broadcast_buf = buf[offset : offset + p.numel].view(p._orig_size)
            offset += p.numel

    # ---- reshard ----

    def reshard(self):
        """Reshard all params (free unsharded buffers, restore placeholders)."""
        for p in self.params:
            p.reshard()
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
                if p._unsharded_param is None or p._unsharded_param.grad is None:
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
            if p._accumulated_grad is not None and p._unsharded_param is not None:
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
            # Coalesce all reduces into a single fused NCCL kernel
            with dist._coalescing_manager(group=dp_group, device=self.device):
                for owner_rank, owner_params in self._by_owner.items():
                    params_with_grad = [
                        p
                        for p in owner_params
                        if p._unsharded_param is not None and p._unsharded_param.grad is not None
                    ]
                    if not params_with_grad:
                        continue

                    if len(params_with_grad) == 1:
                        # Single param: reduce in-place and save grad tensor reference.
                        # Must save ref here because reshard() will set _unsharded_param=None
                        # before wait_for_reduce() can read it.
                        p = params_with_grad[0]
                        grad = p._unsharded_param.grad.data
                        if _DTensor is not None and isinstance(grad, _DTensor):
                            grad = grad._local_tensor
                        grad = grad.contiguous()
                        dist.reduce(grad, dst=self._global_owner_ranks[owner_rank],
                                    op=dist.ReduceOp.AVG, group=dp_group)
                        p._unsharded_param.grad = None
                        self._pending_reduce.append((grad.view(-1), params_with_grad))
                    else:
                        packed_buf = self._packed_reduce(owner_rank, params_with_grad)
                        self._pending_reduce.append((packed_buf, params_with_grad))

        self._reduce_event = reduce_stream.record_event()

    def wait_for_reduce(self):
        """GPU-side wait for reduces to complete, then unpack gradients on owner.

        This fixes the data race in the old _packed_reduce which read the packed
        buffer before the NCCL reduce had completed.
        """
        if self._reduce_event is None:
            return

        torch.cuda.current_stream().wait_event(self._reduce_event)

        # Now safe to unpack — reduce is complete.
        # All paths (single-param and packed) store a buffer in _pending_reduce,
        # so _unpack_reduced_grads handles both uniformly.
        for buf, params in self._pending_reduce:
            if buf is not None:
                self._unpack_reduced_grads(buf, params)

        self._reduce_event = None
        self._pending_reduce = []

    def _packed_reduce(self, owner_rank: int, params: list[DedicatedParam]) -> Optional[torch.Tensor]:
        """Pack multiple grads to the same owner into one reduce.

        Caller must be on reduce_stream.
        Returns packed buffer for deferred unpacking in wait_for_reduce().
        """
        grad_list = []
        for p in params:
            if p._unsharded_param is None or p._unsharded_param.grad is None:
                continue
            g = p._unsharded_param.grad.data
            if _DTensor is not None and isinstance(g, _DTensor):
                g = g._local_tensor
            grad_list.append(g.contiguous().view(-1))

        if not grad_list:
            return None

        packed = torch.cat(grad_list)
        dist.reduce(packed, dst=self._global_owner_ranks[owner_rank],
                    op=dist.ReduceOp.AVG, group=self._dp_group)

        # Clear grads immediately (data is in packed buffer on reduce_stream)
        for p in params:
            if p._unsharded_param is not None:
                p._unsharded_param.grad = None

        return packed  # deferred unpack in wait_for_reduce

    def _unpack_reduced_grads(self, packed: torch.Tensor, params: list[DedicatedParam]) -> None:
        """Unpack reduced gradients from packed buffer to owner's _reduced_grad.

        Must be called after reduce is complete (after wait_for_reduce event sync).
        Accumulates onto existing _reduced_grad if present (gradient accumulation).
        """
        owner_rank = params[0].owner_rank
        if self._dp_group.rank() == owner_rank:
            offset = 0
            for p in params:
                n = p.numel
                new_grad = packed[offset : offset + n].view(p._orig_size)
                if p._reduced_grad is not None:
                    p._reduced_grad.add_(new_grad)
                else:
                    p._reduced_grad = new_grad.clone()
                offset += n

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
