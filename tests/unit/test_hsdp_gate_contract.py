"""Test the strict distributed oracle and dispatch without a process group."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


def load_driver(name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / "distributed" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def strict_driver():
    return load_driver("_strict_hsdp_gate", "ascend_hsdp_correctness.py")


def test_strict_oracle_accepts_equal_nested_state(strict_driver):
    value = {"model": torch.tensor([1.0, 0.25]), "step": 2, "metadata": ["unchanged"]}
    strict_driver.assert_exact(strict_driver.cpu_clone(value), value)


@pytest.mark.parametrize("case", ["ulp", "signed_zero", "nan", "schema"])
def test_strict_oracle_rejects_inexact_state(strict_driver, case):
    expected = torch.tensor([1.0])
    actual = expected.clone()
    if case == "ulp":
        actual = torch.nextafter(expected, torch.tensor([2.0]))
    elif case == "signed_zero":
        expected, actual = torch.tensor([0.0]), torch.tensor([-0.0])
    elif case == "nan":
        expected, actual = torch.tensor([float("nan")]), torch.tensor([float("nan")])
    else:
        expected, actual = {"step": 2}, {"step": 2, "extra": True}
    with pytest.raises(AssertionError):
        strict_driver.assert_exact(actual, expected)


def test_mandatory_entry_retains_cuda_default(monkeypatch):
    module = load_driver("_cuda_hsdp_entry", "test_hsdp_correctness.py")
    calls = []
    monkeypatch.setattr(module, "main", lambda: calls.append("cuda"))
    monkeypatch.setattr(sys, "argv", ["test_hsdp_correctness.py"])
    module.dispatch_backend()
    assert calls == ["cuda"]


def test_mandatory_entry_dispatches_native_npu(monkeypatch):
    module = load_driver("_npu_hsdp_entry", "test_hsdp_correctness.py")
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "ascend_hsdp_correctness",
        SimpleNamespace(main=lambda **kwargs: calls.append(kwargs)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_hsdp_correctness.py",
            "--device",
            "npu",
            "--steps",
            "4",
            "--resume-after",
            "2",
            "--output-dir",
            "/unused-cpu-test-proof",
        ],
    )
    module.dispatch_backend()
    assert calls == [
        {
            "steps": 4,
            "resume_after": 2,
            "output_dir": "/unused-cpu-test-proof",
            "model_cls": module.TinyModel,
        }
    ]
