from types import SimpleNamespace

import pytest
import torch

from dmuon.grad_clip import _local_total_norm


@pytest.mark.parametrize("rows", [65, 103])
def test_tilelang_total_norm_chunks_finalize_groups(rows):
    torch_npu = pytest.importorskip("torch_npu")
    pytest.importorskip("tilelang")
    if not torch.npu.is_available():
        pytest.skip("native NPU is required")

    device = torch.device("npu", torch_npu.npu.current_device())
    generator = torch.Generator(device=device).manual_seed(10233)
    entries = tuple(
        SimpleNamespace(
            grad=torch.randn((257,), generator=generator, device=device, dtype=torch.bfloat16)
        )
        for _ in range(rows)
    )
    expected = _local_total_norm(entries, 2.0, None, foreach=False)

    from dmuon.backend.tilelang.clip import local_total_norm_tilelang

    actual = local_total_norm_tilelang(
        entries, 2.0, None, foreach=False, reference=_local_total_norm
    )

    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-6)
