"""Device selection must not replace PyTorch's CUDA implementation."""

from types import SimpleNamespace

import pytest
import torch

from dmuon._core.device import cuda_is_available, current_device, default_device, get_device_module
from dmuon._core import device as device_runtime


def test_cuda_runtime_is_the_original_module():
    assert get_device_module(torch.device("cuda", 0)) is torch.cuda


def test_npu_runtime_uses_the_registered_device_module(monkeypatch):
    runtime = SimpleNamespace(current_device=lambda: 2, is_available=lambda: True)
    monkeypatch.setattr(torch, "npu", runtime, raising=False)
    assert get_device_module("npu:2") is runtime


def test_npu_runtime_registers_the_optional_dependency_lazily(monkeypatch):
    runtime = SimpleNamespace(current_device=lambda: 0)
    monkeypatch.delattr(torch, "npu", raising=False)
    imports = []

    def import_native_device(name):
        imports.append(name)
        monkeypatch.setattr(torch, "npu", runtime, raising=False)

    monkeypatch.setattr(device_runtime, "import_module", import_native_device)
    assert get_device_module("npu:0") is runtime
    assert imports == ["torch_npu"]


def test_cuda_kernel_detection_rejects_a_compatibility_shim(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert not cuda_is_available()


def test_default_device_keeps_cuda_precedence(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12.4")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_available=lambda: True, current_device=lambda: 2),
        raising=False,
    )
    assert default_device() == torch.device("cuda", 3)


def test_default_device_can_select_npu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_available=lambda: True, current_device=lambda: 2),
        raising=False,
    )
    selected = torch.device("privateuseone", 2)
    monkeypatch.setattr(device_runtime, "current_device", lambda kind: selected if kind == "npu" else None)
    assert default_device() == selected


def test_cpu_current_device_does_not_load_an_accelerator(monkeypatch):
    def reject_accelerator(kind):
        raise AssertionError(f"unexpected accelerator access: {kind}")

    monkeypatch.setattr(device_runtime, "get_device_module", reject_accelerator)
    assert current_device("cpu") == torch.device("cpu")


def test_default_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_available=lambda: False), raising=False)
    assert default_device() == torch.device("cpu")


def test_unsupported_stream_device_is_rejected():
    with pytest.raises(ValueError, match="device|Device|support|runtime"):
        get_device_module("meta")
