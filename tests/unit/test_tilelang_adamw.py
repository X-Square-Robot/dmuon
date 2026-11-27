"""Mixed-precision AdamW regression on Ascend 950DT."""

import pytest
import torch


def test_bf16_gradient_square_accumulates_in_fp32():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available() or "950dt" not in torch.npu.get_device_name(0).lower():
        pytest.skip("requires Ascend 950DT")
    pytest.importorskip("tilelang.ascend.language")
    from dmuon.backend import torch as reference
    from dmuon.backend.tilelang import compute

    generator = torch.Generator().manual_seed(42)
    param = torch.randn(4096, generator=generator).to("npu", torch.float32)
    grad = torch.randn(4096, generator=generator).to("npu", torch.bfloat16)
    first = torch.zeros_like(param)
    second = torch.zeros_like(param)
    expected = (param.clone(), grad, first.clone(), second.clone())
    actual = (param, grad, first, second)
    for step in (1, 2):
        for impl, item in ((reference, expected), (compute, actual)):
            impl.adamw_bucket_([item], step, 0.01, (0.9, 0.95), 0.01, 1e-8)
    for index in (0, 2, 3):
        torch.testing.assert_close(actual[index], expected[index], rtol=2e-6, atol=1e-7)
