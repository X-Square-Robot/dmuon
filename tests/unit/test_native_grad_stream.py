"""Deterministic allocator model for deferred reads on native device streams."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import dmuon._backends.fsdp2.group as group_module
import dmuon._backends.fsdp2.matrix_group as matrix_module


class Storage:
    def __init__(self, values):
        self.values = values
        self.readers = set()


class Pool:
    def __init__(self):
        self.available = []
        self.pending = []

    def release(self, storage):
        (self.pending if storage.readers else self.available).append(storage)

    def overwrite_reusable_storage(self):
        # Other backward compute may allocate before the next post-backward hook.
        for storage in self.available:
            storage.values.fill_(-99)

    def complete(self, stream):
        for storage in self.pending:
            storage.readers.discard(stream)


class Stream:
    def __init__(self, pool):
        self.pool = pool
        self.work = []

    def wait_stream(self, other):
        pass

    def record_event(self):
        return object()

    def finish(self):
        for operation in self.work:
            operation()
        self.work.clear()
        self.pool.complete(self)


class Gradient:
    def __init__(self, values, runtime, device):
        self.storage = Storage(values)
        self.runtime, self.device = runtime, device
        self.dtype = values.dtype
        self.is_cuda = device.type == "cuda"

    @property
    def data(self):
        return self

    def record_stream(self, stream):
        self.storage.readers.add(stream)

    def _copy(self, dtype):
        copied = Gradient(
            torch.empty_like(self.storage.values, dtype=dtype), self.runtime, self.device
        )
        source, destination = self.storage, copied.storage
        self.runtime.active.work.append(lambda: destination.values.copy_(source.values))
        return copied

    def clone(self):
        return self._copy(self.dtype)

    def to(self, dtype):
        return self if dtype == self.dtype else self._copy(dtype)


class Parameter:
    def __init__(self, grad, pool):
        self._grad, self.pool = grad, pool

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        if self._grad is not None:
            self.pool.release(self._grad.storage)
        self._grad = value


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
@pytest.mark.parametrize("grad_dtype", [None, torch.float32], ids=["clone", "cast"])
@pytest.mark.parametrize("explicit_layout", [False, True], ids=["ordinary", "matrix"])
def test_no_sync_keeps_source_until_native_consumer_stream_finishes(
    monkeypatch, device_type, grad_dtype, explicit_layout
):
    pool = Pool()
    compute, reduce = Stream(pool), Stream(pool)
    runtime = SimpleNamespace(active=compute)
    device = SimpleNamespace(type=device_type, index=1)

    @contextmanager
    def stream_context(stream):
        previous, runtime.active = runtime.active, stream
        try:
            yield
        finally:
            runtime.active = previous

    def current_stream(requested_device=None):
        assert requested_device is None or requested_device is device
        return runtime.active

    def native_runtime(requested_device):
        assert requested_device is device
        return runtime

    runtime.stream = stream_context
    runtime.current_stream = current_stream
    monkeypatch.setattr(group_module, "get_device_module", native_runtime)
    monkeypatch.setattr(matrix_module, "get_device_module", native_runtime, raising=False)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    expected = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.bfloat16)
    param = SimpleNamespace(
        _is_unsharded=True,
        _unsharded_param=Parameter(Gradient(expected.clone(), runtime, device), pool),
        _accumulated_grad=None,
        _grad_dtype=grad_dtype,
        local_grad_for_reduce=lambda grad: grad,
    )
    group = object.__new__(group_module.DedicatedParamGroup)
    group.params = [param]
    group._device_module = runtime
    group._matrix_params = None
    group.reduce_grads_enabled = False
    group.comm_ctx = SimpleNamespace(reduce_stream=reduce)

    if explicit_layout:

        class Parent:
            pass

        parent = Parent()
        parent._unsharded_param = param._unsharded_param
        unit = SimpleNamespace(
            _unsharded_param=SimpleNamespace(requires_grad=True),
            matrix_spec=SimpleNamespace(shape=expected.shape),
            _grad_dtype=grad_dtype,
            _layout_grad=None,
        )
        mapping = SimpleNamespace(native_view=lambda gradient: None)
        parent.layout = SimpleNamespace(copy_maps=[mapping])
        original_empty = torch.empty

        def allocate_native(shape, *, dtype, device):
            assert device is parent._unsharded_param.grad.device
            return Gradient(original_empty(shape, dtype=dtype), runtime, device)

        def enqueue_copies(tasks, **kwargs):
            for task in tasks:
                source, destination = task.source.storage, task.destination.storage
                runtime.active.work.append(
                    lambda source=source, destination=destination: destination.values.copy_(
                        source.values
                    )
                )

        monkeypatch.setattr(matrix_module.torch, "empty", allocate_native)
        matrices = object.__new__(matrix_module.MatrixParamGroup)
        matrices.entries = {parent: [unit]}
        matrices.copy_executor = SimpleNamespace(run=enqueue_copies)
        group._matrix_params = matrices
        group.params = []

    group.reduce_grads()
    assert param._unsharded_param.grad is None
    assert runtime.active is compute
    # Recording the allocation stream instead of the consumer is insufficient.
    assert len(pool.pending) == 1
    assert pool.pending[0].readers == {reduce}
    pool.overwrite_reusable_storage()
    reduce.finish()
    torch.testing.assert_close(
        (unit._layout_grad if explicit_layout else param._accumulated_grad).storage.values,
        expected.to(grad_dtype or expected.dtype),
        rtol=0,
        atol=0,
    )


def test_cpu_gradient_does_not_request_accelerator_stream(monkeypatch):
    def unexpected_runtime(device):
        pytest.fail(f"CPU gradient requested an accelerator runtime: {device}")

    monkeypatch.setattr(group_module, "get_device_module", unexpected_runtime)
    grad = torch.arange(4, dtype=torch.float32)
    param = SimpleNamespace(
        _unsharded_param=SimpleNamespace(grad=grad),
        _accumulated_grad=None,
        _grad_dtype=None,
        local_grad_for_reduce=lambda value: value,
    )
    accumulated = group_module._get_grad_for_reduce(param, copy_borrowed=True)
    grad.fill_(-99)
    torch.testing.assert_close(accumulated, torch.arange(4, dtype=torch.float32))
