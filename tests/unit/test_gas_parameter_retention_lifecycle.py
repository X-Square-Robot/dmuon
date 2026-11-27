"""CPU lifecycle outcomes for retention; stream/collective objects are fakes.

Real CPU autograd, production no-sync accumulation and post-backward/root
callbacks run here. These tests do not certify CUDA/NPU or real collectives.
"""

import os
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn

from dmuon import Muon, no_sync, retain_unsharded_params
from dmuon._backends.fsdp2.group import DedicatedParamGroup, _get_grad_for_reduce
from dmuon._backends.ddp import DedicatedParamGroupDDP
from dmuon._core.comm import DedicatedCommContext
from dmuon._core.internal_utils import alloc_storage
from dmuon._core.state import DedicatedState
from dmuon.checkpoint import get_model_state_dict, get_optimizer_state_dict
from dmuon.utils import broadcast_all_updates, broadcast_all_updates_async


class FakeStream:
    def __init__(self, log):
        self.log = log

    def wait_event(self, event):
        event.waited = True
        self.log.append(("wait-event", event.name))

    def wait_stream(self, stream):
        pass

    def record_event(self):
        return SimpleNamespace(name="reduced", waited=False)


class FakeParam:
    """Only placement/stream semantics are faked; grad tensors are real CPU."""

    def __init__(self, parameter, name, log):
        self._unsharded_param = parameter
        self._is_unsharded = True
        self._accumulated_grad = None
        self._reduced_grad = None
        self._grad_dtype = None
        self.name = name
        self.log = log
        self.fail_release = False

    def local_grad_for_reduce(self, gradient):
        return gradient

    def uses_sharded_adamw(self):
        return False

    def reshard(self):
        self.log.append(("param-release", self.name))
        if self.fail_release:
            raise RuntimeError("synthetic release failure")
        self._is_unsharded = False


class FakeGroup(DedicatedParamGroup):
    """Reuse real accumulation and event-safe reshard; fake final collective."""

    def __init__(self, module, context, name):
        self.comm_ctx = context
        self.log = context.log
        self.name = name
        self.params = [FakeParam(module.weight, name, self.log)]
        self.reduce_grads_enabled = True
        self.replicate_grads_enabled = True
        self._delay_stage2_to_optimizer = False
        self._post_backward_fired = False
        self._is_unsharded = True
        self._broadcast_event = None
        self._sharded_adamw_unshard_event = None
        self._packed_buf_by_owner = {0: torch.ones(1)}
        stream = FakeStream(self.log)
        self._device_module = SimpleNamespace(
            current_stream=lambda: stream, stream=lambda _: nullcontext()
        )
        self.pending = None

    def _pre_forward_wait(self):
        pass

    def unshard(self, **kwargs):
        for buffer in self._packed_buf_by_owner.values():
            alloc_storage(buffer)
        self._is_unsharded = True
        for param in self.params:
            param._is_unsharded = True

    def wait_for_unshard(self):
        pass

    def _record_post_forward(self):
        self.comm_ctx.post_forward_order.append(self)

    def _backward_prefetch(self):
        pass

    def reduce_grads(self):
        if not self.reduce_grads_enabled:
            self.log.append(("accumulate", self.name))
            DedicatedParamGroup.reduce_grads(self)
            return
        self.log.append(("reduce", self.name))
        param = self.params[0]
        self.pending = _get_grad_for_reduce(param, merge_accumulated=True)
        param._unsharded_param.grad = None

    def wait_for_stage1_reduce(self):
        self.log.append(("wait-stage1", self.name))

    def wait_for_reduce(self):
        self.log.append(("wait-reduce", self.name))
        if self.pending is not None:
            # A one-participant fake publication, not a real collective.
            self.params[0]._reduced_grad = self.pending
            self.pending = None


def fixture(count=1, tied=False):
    context = object.__new__(DedicatedCommContext)
    context._parameter_retention_depth = 0
    context._parameter_retention_cleanup_error = None
    context.all_states = []
    context.post_forward_order = []
    context.post_backward_final_callback_queued = False
    context.last_reduced_group = None
    context._in_root_post_backward = False
    context.log = []
    context.reduce_stream = FakeStream(context.log)
    context.broadcast_stream = FakeStream(context.log)
    context.reset_post_forward_order = context.post_forward_order.clear
    layers = [nn.Linear(1, 1, bias=False) for _ in range(count)]
    model = nn.ModuleList(layers)
    groups = []
    states = []
    for index, layer in enumerate(layers):
        group = FakeGroup(layer, context, str(index))
        state = DedicatedState(layer, group, context, reshard_after_forward=False)
        layer._dedicated_state = state
        groups.append(group)
        states.append(state)
    if tied:
        alias = DedicatedState(layers[0], groups[0], context, reshard_after_forward=False)
        layers[0]._dedicated_states = [states[0], alias]
        states.append(alias)
    model._dedicated_comm_ctx = context
    return model, context, groups, states


def backward(layer, scale):
    value = torch.ones(1, 1, requires_grad=True)
    (layer(value).sum() * scale).backward()


class RetentionLifecycleTests(unittest.TestCase):
    def test_real_cpu_grads_accumulate_once_and_final_sync_restores_release(self):
        model, context, groups, states = fixture()
        group, param = groups[0], groups[0].params[0]
        with retain_unsharded_params(model):
            for scale, expected in [(1, 1), (2, 3), (3, 6)]:
                with no_sync(model):
                    backward(model[0], scale)
                self.assertEqual(param._accumulated_grad.item(), expected)
                self.assertIsNone(param._unsharded_param.grad)
                self.assertTrue(group._is_unsharded)
                self.assertGreater(group._packed_buf_by_owner[0].untyped_storage().nbytes(), 0)
            backward(model[0], 4)
            self.assertEqual(param._reduced_grad.item(), 10)
            self.assertIsNone(param._accumulated_grad)
            self.assertIsNone(param._unsharded_param.grad)
            self.assertFalse(group._is_unsharded)
            final_reduce = context.log.index(("reduce", "0"))
            self.assertLess(final_reduce, context.log.index(("param-release", "0")))
            self.assertLess(
                context.log.index(("param-release", "0")),
                context.log.index(("wait-reduce", "0"), final_reduce),
            )
        self.assertFalse(states[0]._retain_unsharded_in_no_sync)
        self.assertEqual(context._parameter_retention_depth, 0)

    def test_default_no_sync_keeps_original_post_backward_release(self):
        model, context, groups, states = fixture()
        with no_sync(model):
            backward(model[0], 1)
        self.assertEqual(groups[0].params[0]._accumulated_grad.item(), 1)
        self.assertIsNone(model[0].weight.grad)
        self.assertFalse(groups[0]._is_unsharded)
        self.assertEqual(groups[0]._packed_buf_by_owner[0].untyped_storage().nbytes(), 0)
        self.assertFalse(states[0]._retain_unsharded_in_no_sync)
        self.assertEqual(context._parameter_retention_depth, 0)

    def test_nested_exception_releases_tied_and_unvisited_prefetch_once(self):
        model, context, groups, states = fixture(2, tied=True)
        event = SimpleNamespace(name="prefetched", waited=False)
        groups[1]._broadcast_event = event
        with self.assertRaisesRegex(ValueError, "synthetic incomplete GAS"):
            with retain_unsharded_params(model):
                with retain_unsharded_params(model):
                    self.assertEqual(context._parameter_retention_depth, 2)
                self.assertTrue(groups[0]._is_unsharded)
                self.assertTrue(all(state._retain_unsharded_in_no_sync for state in states))
                raise ValueError("synthetic incomplete GAS")
        self.assertTrue(event.waited)
        self.assertIsNone(groups[1]._broadcast_event)
        for group in groups:
            self.assertFalse(group._is_unsharded)
            self.assertEqual(group._packed_buf_by_owner[0].untyped_storage().nbytes(), 0)
            self.assertEqual(context.log.count(("param-release", group.name)), 1)
        self.assertFalse(any(state._retain_unsharded_in_no_sync for state in states))
        self.assertEqual(context._parameter_retention_depth, 0)
        self.assertIsNone(context._parameter_retention_cleanup_error)

    def test_active_step_zero_and_checkpoint_guards_preserve_accumulator(self):
        model, context, groups, _ = fixture()
        optimizer = object.__new__(Muon)
        torch.optim.Optimizer.__init__(optimizer, [model[0].weight], {})
        optimizer._comm_ctx = context
        optimizer._fsdp_params = []
        optimizer._all_dedicated_params = groups[0].params
        optimizer.zero_grad()
        self.assertIsNone(groups[0].params[0]._accumulated_grad)
        closures = []
        with retain_unsharded_params(model):
            with no_sync(model):
                backward(model[0], 2)
            for operation in [
                lambda: optimizer.step(lambda: closures.append(1)),
                optimizer.zero_grad,
                lambda: get_model_state_dict(model),
                lambda: get_optimizer_state_dict(model, optimizer),
                lambda: broadcast_all_updates(model),
                lambda: broadcast_all_updates_async(model),
            ]:
                with self.assertRaisesRegex(RuntimeError, "outside retain_unsharded_params"):
                    operation()
            self.assertEqual(groups[0].params[0]._accumulated_grad.item(), 2)
            self.assertEqual(closures, [])
        # A secondary context cannot hide behind an idle primary context.
        other = object.__new__(DedicatedCommContext)
        other._parameter_retention_depth = 1
        other._parameter_retention_cleanup_error = None
        optimizer._parameter_retention_contexts = (context, other)
        for operation in [lambda: optimizer.step(lambda: closures.append(1)), optimizer.zero_grad]:
            with self.assertRaisesRegex(RuntimeError, "outside retain_unsharded_params"):
                operation()
        self.assertEqual(groups[0].params[0]._accumulated_grad.item(), 2)
        self.assertEqual(closures, [])

    def test_unsupported_scope_rejects_before_flags_or_storage_change(self):
        for mode in ("forward-reshard", "mixed", "DDP"):
            with self.subTest(mode=mode):
                model, context, groups, states = fixture()
                if mode == "forward-reshard":
                    states[0].reshard_after_forward = True
                elif mode == "mixed":
                    model.set_reshard_after_backward = lambda _: None
                else:
                    states[0].group = object.__new__(DedicatedParamGroupDDP)
                with self.assertRaises(ValueError):
                    with retain_unsharded_params(model):
                        self.fail("unsupported scope entered")
                self.assertFalse(states[0]._retain_unsharded_in_no_sync)
                self.assertEqual(context._parameter_retention_depth, 0)
                self.assertGreater(groups[0]._packed_buf_by_owner[0].untyped_storage().nbytes(), 0)

    def test_cleanup_failure_attempts_other_groups_restores_flags_and_poison_blocks_forward(self):
        model, context, groups, states = fixture(2)
        groups[0].params[0].fail_release = True
        with self.assertRaisesRegex(RuntimeError, "synthetic release failure"):
            with retain_unsharded_params(model):
                pass
        self.assertFalse(groups[1]._is_unsharded)
        self.assertEqual(groups[1]._packed_buf_by_owner[0].untyped_storage().nbytes(), 0)
        self.assertFalse(any(state._retain_unsharded_in_no_sync for state in states))
        self.assertEqual(context._parameter_retention_depth, 0)
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            model[0](torch.ones(1, 1))
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            context.assert_no_parameter_retention("step")
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            with retain_unsharded_params(model):
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
