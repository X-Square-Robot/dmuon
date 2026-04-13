"""DedicatedParamGroup: manages communication for dedicated params in one module."""

from collections import defaultdict
from typing import Optional

import torch
import torch.distributed as dist

from .param import DedicatedParam


class DedicatedParamGroup:
    """Manages all dedicated parameters within one module.

    Parameters with the same owner are packed into one broadcast/reduce call.
    Different owners' broadcasts run concurrently via async_op.
    """

    def __init__(self, params: list[DedicatedParam]):
        self.params = params
        self.device = params[0].device if params else torch.device("cuda")

        # Pre-group by owner to avoid recomputing each time
        self._by_owner: dict[int, list[DedicatedParam]] = defaultdict(list)
        for p in params:
            self._by_owner[p.owner_rank].append(p)

        self._unshard_works: list[tuple[Optional[dist.Work], int, list[DedicatedParam]]] = []
        self._packed_bufs: list[torch.Tensor] = []  # keep packed buffers alive until finish
        self.reduce_grads_enabled: bool = True

    # ---- unshard (broadcast) ----

    def unshard(self):
        """Broadcast all params. Same-owner params packed, different owners concurrent."""
        self._unshard_works = []
        self._packed_bufs = []
        for owner_rank, owner_params in self._by_owner.items():
            if len(owner_params) == 1:
                p = owner_params[0]
                work = p.alloc_and_broadcast(async_op=True)
                self._unshard_works.append((work, owner_rank, owner_params))
            else:
                work = self._packed_broadcast(owner_rank, owner_params)
                self._unshard_works.append((work, owner_rank, owner_params))

        # Wait for all broadcasts to complete
        for work, _, _ in self._unshard_works:
            if work is not None:
                work.wait()

        # Finalize: register unsharded params on modules
        for _, _, owner_params in self._unshard_works:
            for p in owner_params:
                p.finish_unshard()

        self._packed_bufs = []  # safe to release packed buffers now

        self._unshard_works = []

    def _packed_broadcast(
        self, owner_rank: int, params: list[DedicatedParam]
    ) -> Optional[dist.Work]:
        """Pack multiple params from the same owner into one broadcast."""
        total_numel = sum(p.numel for p in params)
        buf = torch.empty(total_numel, dtype=params[0]._orig_dtype, device=self.device)

        # Owner: copy data into packed buffer
        dp_group = params[0].dp_group
        if dp_group.rank() == owner_rank:
            offset = 0
            for p in params:
                buf[offset : offset + p.numel].copy_(p._owned_data.view(-1))
                offset += p.numel

        global_owner = dist.get_global_rank(dp_group, owner_rank)
        work = dist.broadcast(buf, src=global_owner, group=dp_group, async_op=True)

        # Store buffer ref for unpacking in finish_unshard
        # Keep buf alive in group._packed_bufs until finish_unshard completes
        self._packed_bufs.append(buf)
        offset = 0
        for p in params:
            p._broadcast_buf = buf[offset : offset + p.numel].view(p._orig_size)
            offset += p.numel

        return work

    # ---- reshard ----

    def reshard(self):
        """Reshard all params."""
        for p in self.params:
            p.reshard()

    # ---- gradient reduction ----

    def reduce_grads(self):
        """Reduce gradients to owners. Same-owner grads packed, different owners concurrent."""
        if not self.reduce_grads_enabled:
            return

        works: list[tuple[Optional[dist.Work], int, list[DedicatedParam]]] = []
        for owner_rank, owner_params in self._by_owner.items():
            # Collect params that have gradients
            params_with_grad = [
                p
                for p in owner_params
                if p._unsharded_param is not None and p._unsharded_param.grad is not None
            ]
            if not params_with_grad:
                continue

            if len(params_with_grad) == 1:
                w = params_with_grad[0].reduce_grad(async_op=True)
                works.append((w, owner_rank, params_with_grad))
            else:
                w = self._packed_reduce(owner_rank, params_with_grad)
                works.append((w, owner_rank, params_with_grad))

        # Wait for all reduces
        for work, _, _ in works:
            if work is not None:
                work.wait()

        # Owner saves gradients
        for p in self.params:
            p.save_grad_on_owner()

    def _packed_reduce(self, owner_rank: int, params: list[DedicatedParam]) -> Optional[dist.Work]:
        """Pack multiple grads to the same owner into one reduce."""
        try:
            from torch.distributed.tensor import DTensor as _DTensor
        except ImportError:
            _DTensor = None

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
        work = dist.reduce(
            packed, dst=global_owner, op=dist.ReduceOp.AVG, group=dp_group, async_op=True
        )

        # After wait, owner unpacks
        if dp_group.rank() == owner_rank:
            offset = 0
            for p in params:
                n = p.numel
                p._reduced_grad = packed[offset : offset + n].view(p._orig_size).clone()
                offset += n
            # Clear unsharded grads
            for p in params:
                p._unsharded_param.grad = None
        else:
            for p in params:
                if p._unsharded_param is not None:
                    p._unsharded_param.grad = None

        return work
