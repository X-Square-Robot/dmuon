"""DedicatedState: hook management for dedicated parameter groups."""

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from .group import DedicatedParamGroup


class _DedicatedPreBackward(torch.autograd.Function):
    """Autograd function that triggers parameter unshard before backward."""

    @staticmethod
    def forward(ctx, group: DedicatedParamGroup, *tensors):
        ctx.group = group
        return tensors

    @staticmethod
    def backward(ctx, *grads):
        # Pre-backward: unshard params so backward can compute gradients
        ctx.group.unshard()
        return (None,) + grads


class DedicatedState:
    """Manages hooks for dedicated parameters on a module.

    Registers forward pre/post hooks and backward hooks:
    - pre_forward: broadcast params from owners (unshard)
    - post_forward: reshard + register pre-backward hook via autograd Function
    - post_backward: reduce grads to owners + reshard (via param grad hooks)
    """

    def __init__(self, module: nn.Module, group: DedicatedParamGroup):
        self.module = module
        self.group = group
        self._grad_hook_handles: list = []

        # Register hooks (after FSDP2 hooks since FSDP2 uses prepend=True)
        self._pre_forward_handle = module.register_forward_pre_hook(
            self._pre_forward, with_kwargs=True
        )
        self._post_forward_handle = module.register_forward_hook(self._post_forward)

    def _pre_forward(self, module: nn.Module, args, kwargs):
        """Broadcast dedicated params from owners."""
        self.group.unshard()
        return args, kwargs

    def _post_forward(self, module: nn.Module, input, output):
        """Reshard dedicated params and register backward hooks."""
        # Register post-backward gradient hooks on unsharded params
        # (must do before reshard since we need the unsharded param refs)
        if torch.is_grad_enabled():
            self._register_grad_hooks()

        self.group.reshard()

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
        """Register hooks on parameters to trigger reduce after grad is computed."""
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
                grad_count[0] += 1
                if grad_count[0] >= total:
                    # All grads computed — reduce to owners and reshard
                    self.group.reduce_grads()
                    self.group.reshard()
                return grad

            return hook

        for dp in params_needing_grad:
            handle = dp._unsharded_param.register_hook(make_hook(dp))
            self._grad_hook_handles.append(handle)
