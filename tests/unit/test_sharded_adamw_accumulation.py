"""Exercise gradient accumulation through the real two-stage reduce scheduler."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dmuon._backends.fsdp2.group import DedicatedParamGroup
from dmuon.optim.muon import Muon


class Event:
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def finish(self):
        callbacks, self.callbacks = self.callbacks, []
        for callback in callbacks:
            callback()


class Stream:
    def __init__(self, name):
        self.name = name
        self.callbacks = []
        self.last_event = None

    def wait_stream(self, stream):
        self.wait_event(stream.record_event())

    def wait_event(self, event):
        event.finish()

    def record_event(self):
        event = Event(self.callbacks)
        self.callbacks = []
        self.last_event = event
        return event


class Runtime:
    def __init__(self):
        self.compute = Stream("compute")
        self.reduce = Stream("reduce")
        self.replicate = Stream("replicate")
        self.active = self.compute

    def current_stream(self):
        return self.compute

    @contextmanager
    def stream(self, stream):
        # Host tensor ops in this fixture execute immediately. Finish earlier
        # work on the same mock stream before a second stream block stashes it.
        if stream.last_event is not None:
            stream.last_event.finish()
        previous, self.active = self.active, stream
        try:
            yield
        finally:
            self.active = previous


def make_group(
    monkeypatch,
    *,
    dtype=torch.float32,
    grad_dtype=None,
    replicate=False,
    sharded=True,
    shard_rank=0,
    shard_size=1,
    numel=4,
):
    runtime = Runtime()
    trace, outputs, local_calls = [], [], []
    chunk = (numel + shard_size - 1) // shard_size
    parameter = nn.Parameter(torch.zeros(numel, dtype=dtype))
    param = SimpleNamespace(
        _unsharded_param=parameter,
        _is_unsharded=True,
        _accumulated_grad=None,
        _reduced_grad=None,
        _sharded_adamw_grad=None,
        _grad_dtype=grad_dtype,
        _orig_size=parameter.shape,
        _sharded_adamw_reduce_input=torch.zeros(chunk * shard_size, dtype=grad_dtype or dtype),
        _sharded_adamw_comm_shard=torch.zeros(chunk, dtype=grad_dtype or dtype),
        _sharded_adamw_chunk_numel=chunk,
        _owned_data=None,
        owner_rank=(0, 0),
        owner_shard=0,
        _owner_replicate_global_rank=0,
        is_owner=shard_rank == 0,
        uses_sharded_adamw=lambda: sharded,
        _dmuon_route="sharded_adamw" if sharded else "muon",
    )

    def local_grad(grad):
        local_calls.append(grad.dtype)
        return grad

    param.local_grad_for_reduce = local_grad
    dp_group = SimpleNamespace(rank=lambda: shard_rank)
    group = object.__new__(DedicatedParamGroup)
    group.params = [param]
    group.device = torch.device("cpu")
    group._device_module = runtime
    group._dp_group = dp_group
    group._global_owner_ranks = {(0, 0): 0}
    group._debug_name = "accumulation"
    group.comm_ctx = SimpleNamespace(
        reduce_stream=runtime.reduce,
        replicate_reduce_stream=runtime.replicate,
        replicate_group=object() if replicate else None,
        record_post_step_profile=False,
    )
    group.reduce_grads_enabled = True
    group.replicate_grads_enabled = True
    group._post_reduce_event = None
    group._stage1_reduce_event = None
    group._replicate_reduce_state = None
    group._pending_reduce = []
    group._partial_reduce_by_param = {}
    group._muon_grad_ready_event = None
    group._muon_grad_ready_refs = []

    def reduce_scatter(output, input, *, op, group):
        assert group is dp_group and op == torch.distributed.ReduceOp.AVG
        outputs.append(output)

        def finish():
            start = shard_rank * output.numel()
            output.copy_(input[start : start + output.numel()])
            trace.append(("stage1", output.data_ptr()))

        runtime.active.callbacks.append(finish)

    def all_reduce(grad, *, op, group):
        assert op == torch.distributed.ReduceOp.AVG

        def finish():
            # A second replicate peer contributes zero, so its AVG is half.
            grad.mul_(0.5)
            trace.append(("stage2", grad.data_ptr()))

        runtime.active.callbacks.append(finish)

    def reduce(grad, *, dst, op, group):
        assert dst == 0 and op == torch.distributed.ReduceOp.AVG
        if group is dp_group:
            runtime.active.callbacks.append(lambda: trace.append(("stage1", grad.data_ptr())))
        else:
            assert group is group_for_replicate
            all_reduce(grad, op=op, group=group)

    def record_stream(tensor, stream):
        trace.append(("record", tensor.data_ptr(), stream.name))

    monkeypatch.setattr(torch.distributed, "_coalescing_manager", lambda **kwargs: nullcontext())
    monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", reduce_scatter)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "reduce", reduce)
    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
    group_for_replicate = group.comm_ctx.replicate_group
    return group, param, runtime, trace, outputs, local_calls


def backward(group, param, values, *, synchronize=True):
    param._unsharded_param.grad = (
        None
        if values is None
        else torch.tensor(
            values,
            dtype=param._unsharded_param.dtype,
        )
    )
    group.reduce_grads_enabled = synchronize
    group.reduce_grads()


def result(param, sharded=True):
    return param._sharded_adamw_grad if sharded else param._reduced_grad


@pytest.mark.parametrize("replicate", [False, True])
def test_sharded_gas4_accumulates_complete_collectives(monkeypatch, replicate):
    group, param, runtime, trace, outputs, _ = make_group(monkeypatch, replicate=replicate)
    micros = ([1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 6, 8], [8, 6, 4, 2])
    for micro in micros:
        backward(group, param, micro)
        # Stage-1 compaction must retain every rank's sharded output, including
        # the buffer still needed by Stage-2 and the eventual accumulator.
        group.wait_for_stage1_reduce()
        assert group._pending_reduce[0][0] is outputs[-1]
    group.wait_for_reduce()
    expected = torch.tensor(micros, dtype=torch.float32).sum(0)
    if replicate:
        expected.mul_(0.5)
        for output in outputs:
            complete = trace.index(("stage2", output.data_ptr()))
            consumed = trace.index(("record", output.data_ptr(), "compute"))
            assert complete < consumed
    torch.testing.assert_close(result(param), expected, rtol=0, atol=0)
    assert group._pending_reduce == []
    assert outputs[-1] is group._muon_grad_ready_refs[0]
    assert group.wait_for_reduce() is None
    torch.testing.assert_close(result(param), expected, rtol=0, atol=0)


def test_gas1_reuses_collective_output_and_tracks_consumer_stream(monkeypatch):
    group, param, _, trace, outputs, _ = make_group(monkeypatch)
    backward(group, param, [1, 2, 3, 4])
    assert param._sharded_adamw_grad is None
    consumer = Stream("consumer")
    group.wait_for_reduce(stream=consumer)
    assert param._sharded_adamw_grad is outputs[0]
    assert ("record", outputs[0].data_ptr(), "consumer") in trace
    torch.testing.assert_close(result(param), torch.tensor([1.0, 2.0, 3.0, 4.0]), rtol=0, atol=0)


def test_nonowner_padded_shard_survives_stage1_compaction(monkeypatch):
    group, param, _, _, outputs, _ = make_group(monkeypatch, shard_rank=1, shard_size=2, numel=5)
    for micro in ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]):
        backward(group, param, micro)
        group.wait_for_stage1_reduce()
        assert group._pending_reduce[0][0] is outputs[-1]
    group.wait_for_reduce()
    torch.testing.assert_close(result(param), torch.tensor([6.0, 6.0, 0.0]), rtol=0, atol=0)


def test_missing_grad_and_zero_grad_do_not_reuse_previous_microbatches(monkeypatch):
    group, param, _, _, _, _ = make_group(monkeypatch)
    backward(group, param, [1, 2, 3, 4])
    backward(group, param, None)
    backward(group, param, [4, 3, 2, 1])
    group.wait_for_reduce()
    torch.testing.assert_close(result(param), torch.full((4,), 5.0), rtol=0, atol=0)
    optimizer = optimizer_fixture(group, param)
    Muon.zero_grad(optimizer)
    assert result(param) is None and param._accumulated_grad is None
    backward(group, param, [7, 8, 9, 10])
    group.wait_for_reduce()
    torch.testing.assert_close(result(param), torch.tensor([7.0, 8.0, 9.0, 10.0]), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sharded", [False, True])
def test_explicit_fp32_accumulation_matches_sync_without_low_precision_roundtrip(
    monkeypatch, dtype, sharded
):
    micros = ([2048] * 4, [1] * 4, [1] * 4, [-2048] * 4)
    outcomes = []
    for no_sync in (False, True):
        group, param, _, _, _, calls = make_group(
            monkeypatch,
            dtype=dtype,
            grad_dtype=torch.float32,
            sharded=sharded,
        )
        for index, micro in enumerate(micros):
            backward(group, param, micro, synchronize=not no_sync or index == 3)
            if no_sync and index < 3:
                assert param._accumulated_grad.dtype == torch.float32
        group.wait_for_reduce()
        assert result(param, sharded).dtype == torch.float32
        outcomes.append(result(param, sharded).clone())
        assert len(calls) == 4
    torch.testing.assert_close(outcomes[0], torch.full((4,), 2.0), rtol=0, atol=0)
    torch.testing.assert_close(outcomes[1], outcomes[0], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unspecified_grad_dtype_preserves_raw_no_sync_precision(monkeypatch, dtype):
    group, param, _, _, _, _ = make_group(monkeypatch, dtype=dtype)
    micros = ([2048] * 4, [1] * 4, [1] * 4, [-2048] * 4)
    expected = torch.tensor(micros[0], dtype=dtype)
    for micro in micros[1:-1]:
        expected.add_(torch.tensor(micro, dtype=dtype))
    expected = torch.tensor(micros[-1], dtype=dtype).add_(expected)
    for index, micro in enumerate(micros):
        backward(group, param, micro, synchronize=index == 3)
        if index < 3:
            assert param._accumulated_grad.dtype == dtype
    group.wait_for_reduce()
    assert result(param).dtype == dtype
    torch.testing.assert_close(result(param), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sharded", [False, True])
def test_fp32_stash_without_final_raw_grad_does_not_assign_parameter_grad(
    monkeypatch, dtype, sharded
):
    group, param, _, _, _, _ = make_group(
        monkeypatch,
        dtype=dtype,
        grad_dtype=torch.float32,
        sharded=sharded,
    )
    backward(group, param, [2048] * 4, synchronize=False)
    backward(group, param, [1] * 4, synchronize=False)
    backward(group, param, None)
    group.wait_for_reduce()
    assert param._unsharded_param.grad is None
    assert result(param, sharded).dtype == torch.float32
    torch.testing.assert_close(result(param, sharded), torch.full((4,), 2049.0), rtol=0, atol=0)


def optimizer_fixture(group, param):
    return SimpleNamespace(
        _fsdp_params=[],
        _all_dedicated_params=[param],
        _comm_ctx=SimpleNamespace(all_states=[SimpleNamespace(group=group)]),
    )


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("gates", [(False, True), (False, False, False, True), (True, False, True)])
def test_replicate_gated_partials_flush_once_and_preserve_completed_grads(
    monkeypatch, sharded, gates
):
    group, param, _, trace, _, _ = make_group(monkeypatch, replicate=True, sharded=sharded)
    expected = torch.zeros(4)
    for index, enabled in enumerate(gates):
        micro = torch.full((4,), float(index + 1))
        expected.add_(micro * 0.5)
        group.replicate_grads_enabled = enabled
        backward(group, param, micro.tolist())
        group.wait_for_stage1_reduce()
        if not enabled:
            assert group._pending_reduce == []
            assert id(param) in group._partial_reduce_by_param
        group.wait_for_reduce()
        before_repeat = result(param, sharded)
        before_repeat = before_repeat.clone() if before_repeat is not None else None
        assert group.wait_for_reduce() is None
        if before_repeat is not None:
            torch.testing.assert_close(result(param, sharded), before_repeat, rtol=0, atol=0)
    assert not group._partial_reduce_by_param
    torch.testing.assert_close(result(param, sharded), expected, rtol=0, atol=0)
    # Stashed buffers move to the replicate stream before being released.
    assert any(item[0] == "record" and item[-1] == "replicate" for item in trace)


@pytest.mark.parametrize("numel", [1, 5])
def test_replicate_gate_stashes_nonowner_shards_including_padding(monkeypatch, numel):
    group, param, _, _, _, _ = make_group(
        monkeypatch, replicate=True, shard_rank=1, shard_size=2, numel=numel
    )
    for enabled in (False, False, True):
        group.replicate_grads_enabled = enabled
        backward(group, param, [2.0] * numel)
        group.wait_for_stage1_reduce()
        group.wait_for_reduce()
        if not enabled:
            assert result(param) is None
            assert id(param) in group._partial_reduce_by_param
    assert not group._partial_reduce_by_param
    expected = torch.zeros(param._sharded_adamw_chunk_numel)
    start = param._sharded_adamw_chunk_numel
    expected[: max(0, numel - start)] = 3.0
    torch.testing.assert_close(result(param), expected, rtol=0, atol=0)


def test_owner_managed_nonowner_drops_undefined_stage1_output(monkeypatch):
    group, param, _, _, _, _ = make_group(
        monkeypatch, replicate=True, sharded=False, shard_rank=1, shard_size=2
    )
    group.replicate_grads_enabled = False
    backward(group, param, [1.0] * 4)
    group.wait_for_stage1_reduce()
    group.wait_for_reduce()
    assert not group._partial_reduce_by_param
    assert result(param, False) is None


@pytest.mark.parametrize("sharded", [False, True])
def test_zero_grad_discards_unflushed_replicate_partial(monkeypatch, sharded):
    group, param, _, _, _, _ = make_group(monkeypatch, replicate=True, sharded=sharded)
    group.replicate_grads_enabled = False
    backward(group, param, [12.0] * 4)
    group.wait_for_reduce()
    assert id(param) in group._partial_reduce_by_param
    assert result(param, sharded) is None
    Muon.zero_grad(optimizer_fixture(group, param))
    assert not group._partial_reduce_by_param
    group.replicate_grads_enabled = True
    backward(group, param, [2.0] * 4)
    group.wait_for_reduce()
    torch.testing.assert_close(result(param, sharded), torch.ones(4), rtol=0, atol=0)
