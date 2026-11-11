"""NS-only graph capture must coexist with independent pinned-memory work.

The background action is forced to happen inside capture, rather than relying
on a lucky DataLoader race. The NS input and output are persistent GPU buffers;
no CPU batch is an input to the graph. See NVIDIA's DataLoader pin-memory
capture-failure example. This does not exercise backward/training graph capture.
"""
from __future__ import annotations

import threading

import pytest
import torch

from dmuon.optim.newton_schulz import NewtonSchulz
from dmuon.optim.ns_graph import NSGraphCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
# Keep newly pinned blocks checked out, so the fresh-allocation test cannot
# silently pass by reusing blocks from an earlier parameterized case.
_PINNED_LIVE = []


@pytest.mark.parametrize("operation", ["new_pin", "cached_pin", "event_query"])
@pytest.mark.parametrize("shape", [(256, 512), (3072, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_ns_capture_with_background_pinning_and_event_query(operation, shape, dtype):
    device = torch.device("cuda", torch.cuda.current_device())
    backend = NewtonSchulz("gram")
    matrix = torch.randn(shape, device=device, dtype=dtype)
    backend.local(matrix, steps=5)  # finish lazy compilation before starting the timed handshake
    cache = NSGraphCache()
    requested, finished, ready = threading.Event(), threading.Event(), threading.Event()
    errors = []
    copies = []
    query_event = torch.cuda.Event()
    query_event.record()
    if operation == "cached_pin":
        # An asynchronous copy records reuse events against the pinned block.
        seed = torch.empty(16 * 1024 * 1024, dtype=torch.uint8).pin_memory()
        copies.append(seed.to(device, non_blocking=True))
        del seed
    host_stats = getattr(torch.cuda.memory, "host_memory_stats", None)
    if operation == "new_pin" and host_stats is None:
        pytest.skip("fresh pinned allocation counters require a newer PyTorch")
    worker_allocations = []

    def background():
        try:
            torch.cuda.set_device(device)
            ready.set()
            if not requested.wait(30):
                raise TimeoutError("capture never requested background operation")
            if operation == "event_query":
                query_event.query()
            else:
                size = (8 if operation == "new_pin" else 16) * 1024 * 1024
                before = host_stats()["num_host_alloc"] if host_stats is not None else None
                pinned = torch.empty(size, dtype=torch.uint8).pin_memory()
                after = host_stats()["num_host_alloc"] if host_stats is not None else None
                worker_allocations.append((before, after))
                assert pinned.is_pinned()
                _PINNED_LIVE.append(pinned)
        except BaseException as exc:
            errors.append(exc)
        finally:
            ready.set()
            finished.set()

    thread = threading.Thread(target=background, daemon=True)
    thread.start()
    assert ready.wait(30)
    assert not errors

    def ns_fn(value):
        output = backend.local(value, steps=5)
        if torch.cuda.is_current_stream_capturing():
            requested.set()
            assert finished.wait(30), "background operation did not finish during capture"
        return output

    try:
        cache.run(ns_fn, matrix)  # eager warmup; includes backend autotuning
        expected = backend.local(matrix, steps=5)
        actual = cache.run(ns_fn, matrix)  # second call captures and replays
        assert requested.is_set() and finished.is_set()
        assert not errors, repr(errors)
        state = next(iter(cache._graphs.values()))
        assert state.graph is not None and not state.dead, "capture fell back to eager"
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if operation == "new_pin":
            assert worker_allocations and worker_allocations[0][1] > worker_allocations[0][0]
        for _ in range(8):
            # A changing input excludes an accidentally replayed stale result.
            fresh = torch.randn_like(matrix)
            expected = backend.local(fresh, steps=5)
            actual = cache.run(ns_fn, fresh)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not state.dead
    finally:
        requested.set()
        thread.join(timeout=30)
    assert not thread.is_alive()


@torch.no_grad()
def test_ns_capture_still_rejects_sync_on_capture_thread():
    backend = NewtonSchulz("gram")
    matrix = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    cache = NSGraphCache()

    def invalid_ns(value):
        output = backend.local(value, steps=5)
        if torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()
        return output

    cache.run(invalid_ns, matrix)
    actual = cache.run(invalid_ns, matrix)
    state = next(iter(cache._graphs.values()))
    assert state.dead and state.graph is None
    expected = backend.local(matrix, steps=5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
