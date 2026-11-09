"""FP32 master copy-in: conversion bits, bounds, pointers and stream lifetime."""

from types import SimpleNamespace

import pytest
import torch

from dmuon import MatrixLayout
from dmuon._core.layout_copy import CopyExecutor, CopyTask, _extension


@pytest.fixture(params=["cpu", "cuda"])
def device(request, monkeypatch):
    if request.param == "cuda" and (not torch.cuda.is_available() or _extension() is None):
        pytest.skip("requires CUDA and the optional copy extension")
    monkeypatch.setenv("DMUON_COPY_BACKEND", "cuda" if request.param == "cuda" else "torch")
    monkeypatch.setenv("DMUON_COPY_MAX_CTAS", "96")
    return request.param


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 7, 31, 4095, 4096, 4097, 65539])
@pytest.mark.parametrize(
    "source_offset,target_offset", [(0, 0), (1, 0), (0, 1), (1, 3), (0, 4), (2, 4), (3, 8)]
)
def test_copy_in_tails_alignment_and_guards(device, dtype, size, source_offset, target_offset):
    source = torch.randn(size + source_offset + 7, device=device)[
        source_offset : source_offset + size
    ]
    storage = torch.full((size + target_offset + 9,), 0.75, device=device, dtype=dtype)
    target = storage[target_offset : target_offset + size]
    before = storage.clone()
    executor = CopyExecutor()
    executor.run([CopyTask(source, target)])
    torch.testing.assert_close(
        target.view(torch.int16), source.to(dtype).view(torch.int16), rtol=0, atol=0
    )
    torch.testing.assert_close(storage[:target_offset], before[:target_offset], rtol=0, atol=0)
    torch.testing.assert_close(
        storage[target_offset + size :], before[target_offset + size :], rtol=0, atol=0
    )
    if device == "cuda" and size:
        assert executor.last_stats["path"] == (
            "fp32-bf16" if dtype == torch.bfloat16 else "fp32-fp16"
        )
        assert executor.last_stats["launches"] == 1
        assert executor.last_stats["ctas"] <= 96


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("offset", [0, 1])
def test_copy_in_all_exponents_rounding_and_special_values(device, dtype, offset):
    # Every sign/exponent/high mantissa, including NaNs/infinity/subnormals,
    # crossed with both BF16 and FP16 halfway-neighbor rounding patterns.
    high = torch.arange(65536, dtype=torch.int64, device=device)[:, None] << 16
    low = torch.tensor([0, 0xFFF, 0x1000, 0x1001, 0x7FFF, 0x8000, 0x8001, 0xFFFF], device=device)
    values = (high | low).to(torch.int32).flatten().view(torch.float32)
    source = torch.empty(values.numel() + offset, device=device)[offset:]
    source.copy_(values)
    target = torch.empty(values.numel() + offset, device=device, dtype=dtype)[offset:]
    CopyExecutor().run([CopyTask(source, target)])
    torch.testing.assert_close(
        target.view(torch.int16), source.to(dtype).view(torch.int16), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_copy_in_mapped_contiguous_intervals_and_pointer_refresh(device, dtype):
    plan = MatrixLayout.view(3, 7, 5).compile((3, 7, 5))
    executor = CopyExecutor()
    for offset in (0, 1, 3):
        native = torch.randn(105 + offset, device=device)[offset:].view(3, 7, 5)
        outputs = [torch.empty(7, 5, device=device, dtype=dtype) for _ in range(3)]
        executor.run(
            [CopyTask(native, out, mapping) for out, mapping in zip(outputs, plan.copy_maps)]
        )
        for index, output in enumerate(outputs):
            torch.testing.assert_close(
                output.view(torch.int16), native[index].to(dtype).view(torch.int16), rtol=0, atol=0
            )
        updated = [torch.randn(7, 5, device=device) for _ in range(3)]
        target = torch.full((105 + offset + 7,), 0.5, device=device, dtype=dtype)
        native_out = target[offset : offset + 105].view(3, 7, 5)
        executor.run(
            [
                CopyTask(value, native_out, mapping, scatter=True)
                for value, mapping in zip(updated, plan.copy_maps)
            ]
        )
        torch.testing.assert_close(
            native_out.view(torch.int16),
            torch.stack(updated).to(dtype).view(torch.int16),
            rtol=0,
            atol=0,
        )
        assert torch.all(target[offset + 105 :] == 0.5)


def test_copy_in_unsupported_batch_keeps_general_semantics(device):
    source = torch.randn(70003, device=device)
    tasks = [
        CopyTask(source, torch.empty_like(source, dtype=dtype))
        for dtype in (torch.bfloat16, torch.float16, torch.float32)
    ]
    executor = CopyExecutor()
    executor.run(tasks)
    for task in tasks:
        torch.testing.assert_close(
            task.destination, source.to(task.destination.dtype), rtol=0, atol=0
        )
    if device == "cuda":
        assert executor.last_stats["path"] == "general"
    target = torch.randn_like(source, dtype=torch.bfloat16)
    expected = target + source.to(target.dtype)
    executor.run([CopyTask(source, target, add=True)])
    torch.testing.assert_close(target, expected, rtol=0, atol=0)
    strided = source[:221].view(13, 17).t()
    plan = MatrixLayout.whole().compile(strided.shape)
    target = torch.empty(strided.shape, device=device, dtype=torch.bfloat16)
    executor.run([CopyTask(strided, target, plan.copy_maps[0])])
    torch.testing.assert_close(target, strided.to(target.dtype), rtol=0, atol=0)


def test_unmapped_scatter_rejects_strided_destination():
    source = torch.ones(4, 3)
    target = torch.empty(3, 4).t()
    with pytest.raises(ValueError, match="contiguous"):
        CopyTask(source, target, scatter=True).check()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_copy_in_api2_fallback(device, monkeypatch, dtype):
    if device != "cuda":
        pytest.skip("requires CUDA extension ABI")
    import dmuon._core.layout_copy as module

    extension = _extension()
    adapter = SimpleNamespace(api_version=2, copy=extension.copy)
    monkeypatch.setattr(module, "_extension", lambda: adapter)
    source = torch.randn(65539, device=device)
    target = torch.empty_like(source, dtype=dtype)
    executor = CopyExecutor()
    executor.run([CopyTask(source, target)])
    torch.testing.assert_close(target, source.to(dtype), rtol=0, atol=0)
    assert executor.last_stats["path"] == "general"
    adapter.api_version = 3
    executor.run([CopyTask(source, target)])
    assert executor.last_stats["path"].startswith("fp32-")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_copy_in_consumer_stream_and_allocator_lifetime(device, dtype):
    if device != "cuda":
        pytest.skip("requires CUDA streams and allocator")
    source = torch.randn(700003, device=device)
    expected = source.to(dtype)
    target = torch.empty_like(expected)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    executor = CopyExecutor()
    with torch.cuda.stream(producer):
        executor.run([CopyTask(source, target)])
        ready = producer.record_event()
    del source
    noise = [torch.full((700003,), 91.0, device=device) for _ in range(8)]
    with torch.cuda.stream(consumer):
        consumer.wait_event(ready)
        observed = target.clone()
    consumer.synchronize()
    torch.testing.assert_close(
        observed.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )
    assert noise
