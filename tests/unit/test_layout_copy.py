"""Address maps and batched copies, with the same oracle for CPU and CUDA."""

import pytest
import torch

from dmuon import MatrixLayout
from dmuon._core.layout_copy import CopyExecutor, CopyTask, _extension


CASES = [
    ((12, 8), MatrixLayout.view(3, 4, 8)),
    ((24, 8), MatrixLayout.view(2, 3, 4, 8).permute(1, 0, 2, 3).reshape(3, 8, 8)),
    ((16, 8), MatrixLayout.view(2, 4, 2, 8).split((2, 1, 1), dim=1).reshape(-1, 8)),
    ((2, 9, 10), MatrixLayout.view(2, 9, 2, 5).permute(0, 2, 3, 1)),
    ((13, 21), MatrixLayout.view(13, 21).permute(1, 0)),
    ((2, 8, 12), MatrixLayout.rows(3, 5).permute(1, 0)),
    (
        (8, 12),
        MatrixLayout.view(2, 4, 3, 4)
        .permute(2, 0, 3, 1)
        .reshape(3, 8, 4)
        .split((1, 2), dim=0)
        .split((3, 5), dim=-2)
        .permute(0, 2, 1),
    ),
]


@pytest.fixture(params=["cpu", "cuda"])
def device(request, monkeypatch):
    if request.param == "cuda":
        if not torch.cuda.is_available() or _extension() is None:
            pytest.skip("CUDA device and the optional layout copy extension are required")
        monkeypatch.setenv("DMUON_COPY_BACKEND", "cuda")
    else:
        monkeypatch.setenv("DMUON_COPY_BACKEND", "torch")
    return request.param


@pytest.mark.parametrize("shape,layout", CASES)
def test_lowering_matches_the_original_tree(shape, layout):
    plan = layout.compile(shape)
    source = torch.arange(torch.tensor(shape).prod()).view(shape)
    expected = plan.pack(source)
    for spec, mapping in zip(plan.matrices, plan.copy_maps):
        indices = mapping.indices(torch.arange(spec.numel))
        actual = source.flatten()[indices].view(spec.shape)
        torch.testing.assert_close(
            actual, expected[spec.offset : spec.offset + spec.numel].view(spec.shape)
        )
        view = mapping.native_view(source)
        if view is not None:
            torch.testing.assert_close(view.reshape(spec.shape), actual)


@pytest.mark.parametrize("shape,layout", CASES)
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_gather_and_scatter_write_final_buffers(device, shape, layout, src_dtype, dst_dtype):
    plan = layout.compile(shape)
    torch.manual_seed(17)
    source = torch.randn(shape, device=device, dtype=src_dtype)
    expected = plan.pack(source).to(dst_dtype)
    outputs = [torch.empty(spec.shape, device=device, dtype=dst_dtype) for spec in plan.matrices]
    executor = CopyExecutor()
    executor.run([CopyTask(source, out, mapping) for out, mapping in zip(outputs, plan.copy_maps)])
    actual = torch.cat([out.flatten() for out in outputs])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    native = torch.empty(shape, device=device, dtype=dst_dtype)
    executor.run(
        [
            CopyTask(out, native, mapping, scatter=True)
            for out, mapping in zip(outputs, plan.copy_maps)
        ]
    )
    torch.testing.assert_close(native, source.to(dst_dtype), rtol=0, atol=0)
    if device == "cuda":
        assert executor.last_stats["launches"] == 1
        assert executor.last_stats["ctas"] <= 8


@pytest.mark.parametrize("scatter", [False, True])
def test_mixed_tasks_add_and_dtype_rounding(device, scatter):
    shape, layout = CASES[3]
    plan = layout.compile(shape)
    source = torch.randn(shape, device=device)
    initial = torch.randn(shape, device=device, dtype=torch.bfloat16)
    packed = plan.pack(source)
    executor = CopyExecutor()
    if scatter:
        actual = initial.clone()
        tasks = [
            CopyTask(packed[m.offset : m.offset + m.numel], actual, mapping, scatter=True, add=True)
            for m, mapping in zip(plan.matrices, plan.copy_maps)
        ]
        expected = initial + source.to(initial.dtype)
    else:
        initial = plan.pack(initial).clone()
        actual = [
            initial[m.offset : m.offset + m.numel].clone().view(m.shape) for m in plan.matrices
        ]
        tasks = [
            CopyTask(source, out, mapping, add=True) for out, mapping in zip(actual, plan.copy_maps)
        ]
        expected = initial + packed.to(initial.dtype)
    # Ordinary copy/cast and mapped add share one dispatch.
    plain = torch.randn(4099, device=device, dtype=torch.float16)
    plain_out = torch.empty_like(plain, dtype=torch.float32)
    tasks.append(CopyTask(plain, plain_out))
    executor.run(tasks)
    if not scatter:
        actual = torch.cat([out.flatten() for out in actual])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(plain_out, plain.float(), rtol=0, atol=0)


def test_identity_operations_are_recognized_without_changing_checkpoint_signature():
    original = MatrixLayout.view(2, 3, 4, 8)
    identity = original.permute(1, 0, 2, 3).permute(1, 0, 2, 3).reshape(6, 4, 8)
    assert identity.compile((24, 8)).is_contiguous
    assert identity.compile((24, 8)).state()["version"] == 2
    assert original.compile((24, 8)).signature != identity.compile((24, 8)).signature


def test_strided_native_gradient_has_no_full_contiguous_temporary(device):
    plan = CASES[1][1].compile((24, 8))
    source = torch.randn(8, 24, device=device).t()
    expected = plan.pack(source)
    outputs = [torch.empty(m.shape, device=device) for m in plan.matrices]
    CopyExecutor().run(
        [CopyTask(source, out, mapping) for out, mapping in zip(outputs, plan.copy_maps)]
    )
    torch.testing.assert_close(torch.cat([out.flatten() for out in outputs]), expected)


@pytest.mark.parametrize("budget", [1, 2, 4])
def test_finite_cta_budget_and_refreshed_addresses(device, monkeypatch, budget):
    monkeypatch.setenv("DMUON_COPY_MAX_CTAS", str(budget))
    executor = CopyExecutor()
    for step in range(3):
        source = torch.randn(77777, device=device)
        target = torch.empty_like(source)
        executor.run([CopyTask(source, target)])
        torch.testing.assert_close(target, source, rtol=0, atol=0)
        if device == "cuda":
            assert executor.last_stats["ctas"] <= budget
            assert len(executor._plans) == 1  # pointers never enter the static key


def test_invalid_backend_budget_and_strict_cuda(monkeypatch):
    executor = CopyExecutor()
    task = CopyTask(torch.ones(4), torch.empty(4))
    monkeypatch.setenv("DMUON_COPY_BACKEND", "invalid")
    with pytest.raises(ValueError, match="BACKEND"):
        executor.run([task])
    monkeypatch.setenv("DMUON_COPY_BACKEND", "torch")
    monkeypatch.setenv("DMUON_COPY_MAX_CTAS", "0")
    with pytest.raises(ValueError, match="positive"):
        executor.run([task])
    monkeypatch.setenv("DMUON_COPY_MAX_CTAS", "1")
    monkeypatch.setenv("DMUON_COPY_BACKEND", "cuda")
    with pytest.raises(RuntimeError, match="unavailable"):
        executor.run([task])
    with pytest.raises(RuntimeError, match="unavailable"):
        CopyExecutor(device="cpu").run([])
    monkeypatch.setenv("DMUON_COPY_BACKEND", "torch")
    monkeypatch.setenv("DMUON_COPY_MAX_CTAS", "0")
    with pytest.raises(ValueError, match="positive"):
        executor.run([])


def test_copy_completion_event_orders_a_consumer_stream(device):
    if device != "cuda":
        pytest.skip("CUDA stream ordering test")
    source = torch.randn(128, 192, device=device)
    plan = MatrixLayout.view(128, 192).permute(1, 0).compile(source.shape)
    target = torch.empty(192, 128, device=device)
    stream = torch.cuda.Stream(priority=-1)
    stream.wait_stream(torch.cuda.current_stream())
    executor = CopyExecutor()
    with torch.cuda.stream(stream):
        executor.run([CopyTask(source, target, plan.copy_maps[0])])
        ready = stream.record_event()
    torch.cuda.current_stream().wait_event(ready)
    actual = target.square().sum(1)
    torch.testing.assert_close(actual, source.t().square().sum(1))


def test_async_executor_does_not_retain_full_source_tensors(device):
    if device != "cuda":
        pytest.skip("CUDA allocator lifetime test")
    import gc
    import weakref

    source = torch.full((262144,), 42.0, device=device)
    target = torch.empty_like(source)
    weak = weakref.ref(source)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    executor = CopyExecutor()
    with torch.cuda.stream(stream):
        executor.run([CopyTask(source, target)])
        ready = stream.record_event()
    del source
    gc.collect()
    assert weak() is None
    # A subsequent allocation on the producer stream must not race the copy.
    temporary = torch.zeros_like(target)
    torch.cuda.current_stream().wait_event(ready)
    torch.testing.assert_close(target, torch.full_like(target, 42.0))
    assert temporary.numel() == target.numel()


@pytest.mark.parametrize("shape", [(), (0,), (1,), (7,), (8193,)])
def test_plain_copy_cast_including_scalars_and_empty_tensors(device, shape):
    source = torch.randn(shape, device=device)
    target = torch.empty(shape, device=device, dtype=torch.bfloat16)
    CopyExecutor().run([CopyTask(source, target)])
    torch.testing.assert_close(target, source.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_packed_transpose_even_rectangular_tails(device, dtype):
    plan = MatrixLayout.view(3, 258, 2, 514).permute(0, 2, 3, 1).compile((3, 258, 1028))
    native = torch.randn(plan.source_shape, device=device, dtype=dtype)
    expected = plan.pack(native)
    outputs = [torch.empty(m.shape, device=device, dtype=dtype) for m in plan.matrices]
    executor = CopyExecutor()
    executor.run([CopyTask(native, out, m) for out, m in zip(outputs, plan.copy_maps)])
    torch.testing.assert_close(torch.cat([x.flatten() for x in outputs]), expected, rtol=0, atol=0)
    restored = torch.empty_like(native)
    executor.run(
        [CopyTask(out, restored, m, scatter=True) for out, m in zip(outputs, plan.copy_maps)]
    )
    torch.testing.assert_close(restored, native, rtol=0, atol=0)
    if device == "cuda":
        assert executor.last_stats["path"] == "packed16"
        assert executor.last_stats["transpose_tile"] in (128, 256)
        assert executor.last_stats["launches"] == 1


@pytest.mark.parametrize("scatter", [False, True])
@pytest.mark.parametrize("widen", [False, True])
def test_one_batch_mixes_row_transpose_general_and_plain_copies(device, scatter, widen):
    layouts = [
        ((1536, 512), MatrixLayout.view(8, 3, 64, 512).permute(1, 0, 2, 3).reshape(3, 512, 512)),
        ((2, 258, 1028), MatrixLayout.view(2, 258, 2, 514).permute(0, 2, 3, 1)),
        CASES[-1],
    ]
    dtype = torch.float32 if widen else torch.bfloat16
    tasks, results = [], []
    for shape, layout in layouts:
        plan = layout.compile(shape)
        native = torch.randn(shape, device=device, dtype=torch.bfloat16)
        packed = plan.pack(native)
        if scatter:
            target = torch.empty(shape, device=device, dtype=dtype)
            tasks.extend(
                CopyTask(packed[s.offset : s.offset + s.numel], target, m, scatter=True)
                for s, m in zip(plan.matrices, plan.copy_maps)
            )
            results.append((target, native.to(dtype)))
        else:
            outputs = [torch.empty(s.shape, device=device, dtype=dtype) for s in plan.matrices]
            tasks.extend(CopyTask(native, out, m) for out, m in zip(outputs, plan.copy_maps))
            results.append((outputs, packed.to(dtype)))
    source = torch.randn(4099, device=device, dtype=torch.bfloat16)
    plain = torch.empty_like(source, dtype=dtype)
    tasks.append(CopyTask(source, plain, scatter=scatter))
    executor = CopyExecutor()
    executor.run(tasks)
    for actual, expected in results:
        if isinstance(actual, list):
            actual = torch.cat([x.flatten() for x in actual])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(plain, source.to(dtype), rtol=0, atol=0)
    if device == "cuda":
        assert executor.last_stats["path"] == ("packed16-widen" if widen else "packed16")
        assert executor.last_stats["launches"] == 1
        assert next(iter(executor._plans.values()))[-1] & 4


@pytest.mark.parametrize("scatter", [False, True])
@pytest.mark.parametrize("widen", [False, True])
def test_packed_transpose_refreshes_unaligned_addresses(device, scatter, widen):
    plan = MatrixLayout.view(66, 130).permute(1, 0).compile((66, 130))
    mapping = plan.copy_maps[0]
    dtype = torch.float32 if widen else torch.bfloat16
    executor = CopyExecutor()
    for offset in (1, 0, 1):
        native = torch.randn(66 * 130 + offset, device=device, dtype=torch.bfloat16)[offset:].view(
            66, 130
        )
        expected = plan.pack(native)
        if scatter:
            source = torch.empty(expected.numel() + offset, device=device, dtype=torch.bfloat16)[
                offset:
            ]
            source.copy_(expected)
            target = torch.empty(native.numel() + offset, device=device, dtype=dtype)[offset:].view(
                native.shape
            )
            task, oracle = CopyTask(source, target, mapping, scatter=True), native.to(dtype)
        else:
            target = torch.empty(native.numel() + offset, device=device, dtype=dtype)[offset:].view(
                mapping.shape
            )
            task, oracle = CopyTask(native, target, mapping), expected.to(dtype).view(mapping.shape)
        executor.run([task])
        torch.testing.assert_close(target, oracle, rtol=0, atol=0)
    if device == "cuda":
        assert len(executor._plans) == 1
        assert executor.last_stats["path"] == ("packed16-widen" if widen else "packed16")


def test_bfloat16_widening_preserves_all_encoding_bits(device):
    source = (
        torch.arange(65536, device=device, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .view(256, 256)
    )
    plan = MatrixLayout.view(256, 256).permute(1, 0).compile(source.shape)
    target = torch.empty(256, 256, device=device, dtype=torch.float32)
    executor = CopyExecutor()
    executor.run([CopyTask(source, target, plan.copy_maps[0])])
    expected = plan.pack(source).view(256, 256).float()
    torch.testing.assert_close(target.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)


def test_legacy_extension_uses_legacy_tile_plan(device, monkeypatch):
    if device != "cuda":
        pytest.skip("Extension compatibility needs CUDA")
    import dmuon._core.layout_copy as module
    from types import SimpleNamespace

    extension = _extension()
    adapter = SimpleNamespace(api_version=1, copy=extension.copy)
    monkeypatch.setattr(module, "_extension", lambda: adapter)
    source = torch.randn(514, 258, device=device, dtype=torch.bfloat16)
    plan = MatrixLayout.view(514, 258).permute(1, 0).compile(source.shape)
    target = torch.empty(258, 514, device=device, dtype=source.dtype)
    executor = CopyExecutor()
    executor.run([CopyTask(source, target, plan.copy_maps[0])])
    torch.testing.assert_close(target, source.t(), rtol=0, atol=0)
    assert executor.last_stats["path"] == "general"
    assert executor.last_stats["transpose_tile"] == 64
    adapter.api_version = 2
    target.zero_()
    executor.run([CopyTask(source, target, plan.copy_maps[0])])
    torch.testing.assert_close(target, source.t(), rtol=0, atol=0)
    assert executor.last_stats["path"] == "packed16"
    assert len(executor._plans) == 2
