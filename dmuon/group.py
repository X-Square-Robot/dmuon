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
        dp_group = self.params[0].dp_group
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
        total_numel = sum(p.numel for p in params)
        # Use compute_dtype (e.g., bf16) for communication if available
        comm_dtype = params[0]._compute_dtype or params[0]._orig_dtype
        buf = torch.empty(total_numel, dtype=comm_dtype, device=self.device)

        # Owner: copy data into packed buffer (with dtype conversion)
        dp_group = params[0].dp_group
        if dp_group.rank() == owner_rank:
            offset = 0
            for p in params:
                buf[offset : offset + p.numel].copy_(p._owned_data.to(comm_dtype).view(-1))
                offset += p.numel

        global_owner = dist.get_global_rank(dp_group, owner_rank)
        dist.broadcast(buf, src=global_owner, group=dp_group)

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
        """
        if not self.reduce_grads_enabled:
            return

        reduce_stream = self.comm_ctx.reduce_stream
        # Ensure gradients are computed before reduce_stream reads them
        reduce_stream.wait_stream(torch.cuda.current_stream())

        self._pending_reduce = []
        dp_group = self.params[0].dp_group
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
                        params_with_grad[0].reduce_grad(async_op=False)
                        self._pending_reduce.append((None, params_with_grad))
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

        # Now safe to unpack — reduce is complete
        for packed_buf, params in self._pending_reduce:
            if packed_buf is not None:
                self._unpack_reduced_grads(packed_buf, params)

        # Owner saves gradients, all ranks clear grad
        for p in self.params:
            p.save_grad_on_owner()

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
        dp_group = params[0].dp_group
        global_owner = dist.get_global_rank(dp_group, owner_rank)
        dist.reduce(packed, dst=global_owner, op=dist.ReduceOp.AVG, group=dp_group)

        # Clear grads immediately (data is in packed buffer on reduce_stream)
        for p in params:
            if p._unsharded_param is not None:
                p._unsharded_param.grad = None

        return packed  # deferred unpack in wait_for_reduce

    def _unpack_reduced_grads(self, packed: torch.Tensor, params: list[DedicatedParam]) -> None:
        """Unpack reduced gradients from packed buffer to owner's _reduced_grad.

        Must be called after reduce is complete (after wait_for_reduce event sync).
        """
        dp_group = params[0].dp_group
        owner_rank = params[0].owner_rank
        if dp_group.rank() == owner_rank:
            offset = 0
            for p in params:
                n = p.numel
                p._reduced_grad = packed[offset : offset + n].view(p._orig_size).clone()
                offset += n

    # ---- backward prefetch ----

    def _backward_prefetch(self) -> None:
        """Prefetch next layer's unshard during current layer's backward.

        Mirrors FSDP2's _backward_prefetch: uses reverse post-forward order.
        """
        if not self._post_forward_indices:
            return
        curr_index = self._post_forward_indices.pop()
        if (target_index := curr_index - 1) < 0:
            return
        target_group = self.comm_ctx.post_forward_order[target_index]
        target_group.unshard()  # dispatch only — no wait

    def _record_post_forward(self) -> None:
        """Record this group's position in forward order for backward prefetch."""
        post_forward_index = len(self.comm_ctx.post_forward_order)
        self.comm_ctx.post_forward_order.append(self)
        self._post_forward_indices.append(post_forward_index)
