"""Keep the native gate additive to the upstream CUDA entry point."""

import importlib.util
from pathlib import Path

import torch


def load_driver():
    path = Path(__file__).resolve().parents[1] / "distributed" / "test_hsdp_replicate_gate_accum.py"
    spec = importlib.util.spec_from_file_location("_replicate_gate_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upstream_entry_keeps_cuda_default_and_selects_native_npu():
    driver = load_driver()
    assert driver.parse_args([]).device == "cuda"
    assert driver.parse_args(["--device", "npu"]).device == "npu"


def test_recursive_oracle_rejects_missing_rank_shards_and_stale_state():
    driver = load_driver()
    expected = {"gradient": torch.tensor([1.0]), "state": {"step": 2}, "empty": torch.empty(0)}
    failures, worst = [], [0.0]
    driver.compare(expected, expected, failures, path="rank1", worst=worst)
    assert not failures
    actual = {**expected, "gradient": torch.tensor([2.0]), "state": {"step": 1}}
    driver.compare(actual, expected, failures, path="rank1", worst=worst)
    assert len(failures) == 2 and worst[0] == 1.0
    driver.compare({"state": expected["state"]}, expected, failures, path="rank1", worst=worst)
    assert "mapping keys differ" in failures[-1]
