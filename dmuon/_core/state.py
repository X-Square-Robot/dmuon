"""DedicatedState: hook management for dedicated parameter groups.

Registers forward pre/post hooks and backward hooks on a layer module:
- pre_forward: dispatch broadcast on broadcast_stream, wait, finalize,
               register post-backward hook (reduce + reshard) on inputs
- post_forward: reshard, record post-forward order, register pre-backward hook
- pre_backward: unshard (dispatch + wait), prefetch next layer, queue root callback
- post_backward: reduce grads, reshard

Note (Phase 2): the old forward-time-params snapshot + grad-transfer step is
gone. With persistent ``_unsharded_param`` (only storage is resized across
unshard/reshard), autograd writes ``.grad`` directly onto the same Parameter
object across both forward and the subsequent re-unshard, so nothing has to
be transferred.
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch.autograd import Variable

from .comm import DedicatedCommContext

if TYPE_CHECKING:
    # DedicatedState is backend-agnostic — it operates on the shared
    # duck-typed interface (``unshard`` / ``reshard`` / ``reduce_grads`` /
    # ``_pre_forward_wait`` / ``_post_backward_fired``). We alias the
    # FSDP2-variant name here purely for the type checker.
    from dmuon._backends.fsdp2.group import DedicatedParamGroup  # noqa: F401

try:
    from torch.distributed.tensor import DTensor as _DTensor
except ImportError:
    _DTensor = None


def _is_backward_pass() -> bool:
    """Check if we are inside a backward pass (used to guard AC recompute)."""
    return torch._C._current_graph_task_id() != -1


class _DedicatedPreBackward(torch.autograd.Function):
    """Autograd function that triggers parameter unshard before backward."""

    @staticmethod
    def forward(ctx, state: "DedicatedState", *tensors):
        ctx.state = state
        return tensors

    @staticmethod
    def backward(ctx, *grads):
        state = ctx.state
        # First pre_backward of this backward pass queues the autograd-engine
        # root callback once. Runs after the entire backward graph finishes
        # and force-fires post_backward on any group whose fast path did not.
        state._queue_root_post_backward_callback()
        group = state.group
        # Unshard: dispatch (no-op if already prefetched) + wait
        group.unshard()
        group.wait_for_unshard()
        # Prefetch next layer's unshard (dispatch only, no wait)
        group._backward_prefetch()
        return (None,) + grads


class _DedicatedPostBackward(torch.autograd.Function):
    """Autograd function that triggers gradient reduce + reshard after backward.

    Wraps INPUT tensors in pre_forward. In the autograd graph, this node sits
    "upstream" of the module computation, so its backward fires AFTER the
    module's backward has computed all parameter gradients.

    This replaces the old register_hook + counter approach, which had CUDA
    stream visibility issues causing sporadic NaN on multi-GPU.
    """

    @staticmethod
    def forward(ctx, state: "DedicatedState", *tensors):
        ctx.state = state
        return tensors

    @staticmethod
    def backward(ctx, *grads):
        ctx.state._run_post_backward()
        return (None,) + grads


class DedicatedState:
    """Manages hooks for dedicated parameters on a layer module.

    Registers forward pre/post hooks and backward hooks:
    - pre_forward: broadcast params (unshard) + register post-backward on inputs
    - post_forward: reshard + record post-forward order + register pre-backward on outputs
    - pre_backward: unshard + prefetch next layer (via _DedicatedPreBackward)
    - post_backward: reduce grads + reshard (via _DedicatedPostBackward)
    """

    def __init__(
        self,
        module: nn.Module,
        group: "DedicatedParamGroup",
        comm_ctx: DedicatedCommContext,
        reshard_after_forward: bool = True,
    ):
        self.module = module
        self.group = group
        self.comm_ctx = comm_ctx
        self.reshard_after_forward = reshard_after_forward
        # Linked by api.py for forward prefetch (next layer's group)
        self._next_group: Optional["DedicatedParamGroup"] = None

        # Register self so the autograd-engine root callback can iterate all
        # states and fire post_backward on any group whose fast path missed.
        comm_ctx.all_states.append(self)

        # Register hooks (after FSDP2 hooks since FSDP2 uses prepend=True)
        self._pre_forward_handle = module.register_forward_pre_hook(
            self._pre_forward, with_kwargs=True
        )
        self._post_forward_handle = module.register_forward_hook(self._post_forward)

    def _pre_forward(self, module: nn.Module, args, kwargs):
        """Dispatch broadcasts on broadcast_stream, wait, finalize params.

        Also wraps an input tensor through _DedicatedPostBackward so that
        gradient reduce + reshard fires after this module's backward.
        """
        # Phase C.3: consume any pending async replicate broadcast from
        # the previous step BEFORE ``unshard`` reads ``_owned_data``.
        # ``_pre_forward_wait`` is a no-op when the group is in IDLE
        # state (1D mode, sync fallback, or already-consumed event).
        self.group._pre_forward_wait()
        self.group.unshard()            # no-op if already unsharded or prefetched
        self.group.wait_for_unshard()   # no-op if already unsharded
        # Forward prefetch: dispatch next layer's unshard (no wait).
        # During activation-checkpoint recompute this hook runs inside backward;
        # the next forward layer is not the next layer consumed by backward.
        if self._next_group is not None and not _is_backward_pass():
            self._next_group.unshard()  # no-op if already unsharded
        # Reset fast-path flag for this forward — backward (fast path or root
        # callback fallback) will set it True.
        self.group._post_backward_fired = False
        if torch.is_grad_enabled():
            # Register post-backward hook on inputs (reduce + reshard after backward)
            args, kwargs = self._register_post_backward(args, kwargs)
        return args, kwargs

    def _post_forward(self, module: nn.Module, input, output):
        """Reshard params (if enabled), record forward order, register pre-backward."""
        if self.reshard_after_forward:
            self.group.reshard()

        # Record post-forward order for backward prefetch (skip during AC recompute)
        if not _is_backward_pass():
            self.group._record_post_forward()

        # Register pre-backward hook via autograd Function
        if torch.is_grad_enabled():
            output = self._register_pre_backward(output)
        return output

    # ---- post-backward fast path + fallback ------------------------------

    def _run_post_backward(self) -> None:
        """Execute reduce + reshard. Idempotent per forward.

        Called either from _DedicatedPostBackward.backward (fast path) or from
        the autograd-engine root callback (fallback, when no input required
        gradient or was reachable through backward).

        Phase 2: autograd writes .grad directly onto the persistent
        ``_unsharded_param`` object, so no forward-time snapshot / transfer
        step is needed — ``reduce_grads`` reads ``.grad`` in place.

        Rolling reduce drain (1-outstanding): before dispatching this group's
        reduce, wait on the previously-dispatched group's ``_pending_reduce``
        so per-rank backward memory caps at ~2 groups' full gradients instead
        of accumulating all N groups until ``wait_all_reduces`` /
        :func:`_root_post_backward_final_callback` at the end of backward.
        ``wait_for_reduce`` is idempotent — a no-op when the previous group
        has already been drained — so this is safe even when callers (e.g.
        gradient-accumulation paths or the inner-engine root callback under
        ``use_reentrant=True``) interleave drains and dispatches.
        """
        if self.group._post_backward_fired:
            return
        prev = self.comm_ctx.last_reduced_group
        if prev is not None and prev is not self.group:
            prev.wait_for_reduce()
        self.group.reduce_grads()
        self.group.reshard()
        self.comm_ctx.last_reduced_group = self.group
        self.group._post_backward_fired = True

    def _queue_root_post_backward_callback(self) -> None:
        """Queue (at most once per backward) a callback that fires after the
        entire backward graph finishes. Mirrors FSDP2's approach.
        """
        if self.comm_ctx.post_backward_final_callback_queued:
            return
        self.comm_ctx.post_backward_final_callback_queued = True
        Variable._execution_engine.queue_callback(
            lambda: _root_post_backward_final_callback(self.comm_ctx)
        )

    # ---- input/output wrapping (shallow scan, O(1) per call) -------------

    def _register_post_backward(self, args, kwargs):
        """Wrap one input tensor through _DedicatedPostBackward if possible.

        Only scans the top level of args and kwargs (no recursion into nested
        dict/list/tuple). For transformer layers this hits immediately on
        args[0] = hidden_states. For modules whose grad-requiring tensors are
        nested (e.g., a VLA batch dict), the fast path is skipped and the
        autograd-engine root callback (queued in _DedicatedPreBackward) runs
        post_backward at the end of the backward pass instead.

        Only ONE tensor needs wrapping — autograd topologically orders the
        Function's backward after all computation that produced it, which
        includes all param-grad computations of this module.
        """
        # Scan args top level
        for i, obj in enumerate(args):
            if isinstance(obj, torch.Tensor) and obj.requires_grad:
                wrapped = _DedicatedPostBackward.apply(self, obj)[0]
                new_args = args[:i] + (wrapped,) + args[i + 1:]
                return new_args, kwargs
        # Scan kwargs top level
        for k, obj in kwargs.items():
            if isinstance(obj, torch.Tensor) and obj.requires_grad:
                wrapped = _DedicatedPostBackward.apply(self, obj)[0]
                new_kwargs = dict(kwargs)
                new_kwargs[k] = wrapped
                return args, new_kwargs
        # Shallow scan found nothing — rely on root callback fallback.
        return args, kwargs

    def _register_pre_backward(self, output):
        """Wrap one output tensor through _DedicatedPreBackward (unshard trigger).

        Shallow scan on the output to avoid pytree recursion on complex
        returns. Transformer layers typically return a single tensor or a
        short tuple; both cases are hit in O(1).
        """
        if isinstance(output, torch.Tensor):
            if output.requires_grad:
                return _DedicatedPreBackward.apply(self, output)[0]
            return output
        if isinstance(output, tuple):
            for i, obj in enumerate(output):
                if isinstance(obj, torch.Tensor) and obj.requires_grad:
                    wrapped = _DedicatedPreBackward.apply(self, obj)[0]
                    return output[:i] + (wrapped,) + output[i + 1:]
        elif isinstance(output, list):
            for i, obj in enumerate(output):
                if isinstance(obj, torch.Tensor) and obj.requires_grad:
                    wrapped = _DedicatedPreBackward.apply(self, obj)[0]
                    new_out = list(output)
                    new_out[i] = wrapped
                    return new_out
        elif isinstance(output, dict):
            for k, obj in output.items():
                if isinstance(obj, torch.Tensor) and obj.requires_grad:
                    wrapped = _DedicatedPreBackward.apply(self, obj)[0]
                    # Mutate in place to preserve the concrete dict subclass
                    # (e.g. HuggingFace ModelOutput, which provides attribute
                    # access on top of dict). Building a new ``dict()`` loses
                    # that — callers that do ``output.loss`` would break.
                    output[k] = wrapped
                    return output
        return output


def _root_post_backward_final_callback(comm_ctx: DedicatedCommContext) -> None:
    """Run at the end of the backward pass (autograd engine callback).

    Two responsibilities:

    1. Force-fire post_backward on any group whose fast path did not run
       (e.g. when no input tensor required gradient, so
       ``_DedicatedPostBackward.backward`` never executed).
    2. Drain the rolling 1-outstanding reduce window: the last group whose
       reduce was dispatched still has a live ``_pending_reduce`` (held by
       the rolling-drain protocol — see :meth:`_run_post_backward`).  Walk
       every registered group and call ``wait_for_reduce`` to release the
       grad buffer back to the caching allocator before the optimizer step.
       ``wait_for_reduce`` is idempotent, so already-drained groups are
       cheap no-ops.

    The two passes are intentionally separated: step 1's ``_run_post_backward``
    calls may dispatch new reduces (updating ``last_reduced_group``), so the
    drain in step 2 has to run *after* all dispatches are in flight.
    """
    try:
        for state in comm_ctx.all_states:
            if state.group._post_backward_fired:
                continue
            state._run_post_backward()
        for state in comm_ctx.all_states:
            state.group.wait_for_reduce()
        comm_ctx.last_reduced_group = None
    finally:
        comm_ctx.post_backward_final_callback_queued = False
