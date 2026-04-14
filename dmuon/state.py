"""DedicatedState: hook management for dedicated parameter groups.

Registers forward pre/post hooks and backward hooks on a layer module:
- pre_forward: dispatch broadcast on broadcast_stream, wait, finalize
- post_forward: reshard, record post-forward order, register backward hooks
- pre_backward: unshard (dispatch + wait), prefetch next layer
- post_backward: reduce grads (dispatch + wait), reshard
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from .comm import DedicatedCommContext
from .group import DedicatedParamGroup


def _is_backward_pass() -> bool:
    """Check if we are inside a backward pass (used to guard AC recompute)."""
    return torch._C._current_graph_task_id() != -1


class _DedicatedPreBackward(torch.autograd.Function):
    """Autograd function that triggers parameter unshard before backward."""

    @staticmethod
    def forward(ctx, group: DedicatedParamGroup, *tensors):
        ctx.group = group
        return tensors

    @staticmethod
    def backward(ctx, *grads):
        group = ctx.group
        # Unshard: dispatch (no-op if already prefetched) + wait
        group.unshard()
        group.wait_for_unshard()
        # Prefetch next layer's unshard (dispatch only, no wait)
        group._backward_prefetch()
        return (None,) + grads


class DedicatedState:
    """Manages hooks for dedicated parameters on a layer module.

    Registers forward pre/post hooks and backward hooks:
    - pre_forward: broadcast params from owners (unshard)
    - post_forward: reshard + record post-forward order + register backward hooks
    - pre_backward: unshard + prefetch next layer (via autograd Function)
    - post_backward: reduce grads + reshard (via param grad hooks)
    """

    def __init__(
        self,
        module: nn.Module,
        group: DedicatedParamGroup,
        comm_ctx: DedicatedCommContext,
        reshard_after_forward: bool = True,
    ):
        self.module = module
        self.group = group
        self.comm_ctx = comm_ctx
        self.reshard_after_forward = reshard_after_forward
        self._grad_hook_handles: list = []
        # Linked by api.py for forward prefetch (next layer's group)
        self._next_group: Optional[DedicatedParamGroup] = None

        # Register hooks (after FSDP2 hooks since FSDP2 uses prepend=True)
        self._pre_forward_handle = module.register_forward_pre_hook(
            self._pre_forward, with_kwargs=True
        )
        self._post_forward_handle = module.register_forward_hook(self._post_forward)

    def _pre_forward(self, module: nn.Module, args, kwargs):
        """Dispatch broadcasts on broadcast_stream, wait, finalize params."""
        self.group.unshard()            # no-op if already unsharded or prefetched
        self.group.wait_for_unshard()   # no-op if already unsharded
        # Forward prefetch: dispatch next layer's unshard (no wait)
        if self._next_group is not None:
            self._next_group.unshard()  # no-op if already unsharded
        return args, kwargs

    def _post_forward(self, module: nn.Module, input, output):
        """Reshard params (if enabled), record forward order, register backward hooks."""
        # Register post-backward gradient hooks on unsharded params
        # (must do before reshard since we need the unsharded param refs)
        if torch.is_grad_enabled():
            self._register_grad_hooks()

        if self.reshard_after_forward:
            self.group.reshard()

        # Record post-forward order for backward prefetch (skip during AC recompute)
        if not _is_backward_pass():
            self.group._record_post_forward()

        # Register pre-backward hook via autograd Function
        if torch.is_grad_enabled():
            output = self._register_pre_backward(output)
        return output

    def _register_pre_backward(self, output):
        """Wrap output through autograd Function to trigger unshard in backward."""
        flat, spec = tree_flatten(output)
        tensors = [t for t in flat if isinstance(t, torch.Tensor) and t.requires_grad]
        if not tensors:
            return output

        processed = _DedicatedPreBackward.apply(self.group, *tensors)

        tensor_idx = 0
        new_flat = []
        for item in flat:
            if isinstance(item, torch.Tensor) and item.requires_grad:
                new_flat.append(processed[tensor_idx])
                tensor_idx += 1
            else:
                new_flat.append(item)

        return tree_unflatten(new_flat, spec)

    def _register_grad_hooks(self):
        """Register hooks on parameters to trigger reduce after all grads computed."""
        # Remove previous hooks
        for handle in self._grad_hook_handles:
            handle.remove()
        self._grad_hook_handles.clear()

        # Track how many grads we're waiting for
        params_needing_grad = [
            p
            for p in self.group.params
            if p._unsharded_param is not None and p._unsharded_param.requires_grad
        ]
        if not params_needing_grad:
            return

        grad_count = [0]
        total = len(params_needing_grad)

        def make_hook(dedicated_param):
            def hook(grad):
                # Forward grad to current _unsharded_param so reduce_grads can find it.
                # Needed because reshard+re-unshard creates a NEW _unsharded_param,
                # but autograd computes grad on the OLD one (from forward).
                if dedicated_param._unsharded_param is not None:
                    dedicated_param._unsharded_param.grad = grad
                grad_count[0] += 1
                if grad_count[0] >= total:
                    # All grads computed — dispatch reduce (async) + reshard.
                    # Do NOT wait for reduce here — let it overlap with next layer's
                    # backward. wait_for_reduce is deferred to before the optimizer step
                    # via dmuon.wait_all_reduces(model).
                    self.group.reduce_grads()
                    self.group.reshard()
                return grad

            return hook

        for dp in params_needing_grad:
            handle = dp._unsharded_param.register_hook(make_hook(dp))
            self._grad_hook_handles.append(handle)
