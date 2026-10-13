"""Keep the native gate additive to the upstream CUDA entry point."""

import importlib.util
from pathlib import Path

import torch
import pytest
from types import SimpleNamespace


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


def live_gradient_fixture(driver, monkeypatch):
    class Dedicated:
        pass

    mapping = {}
    for name, route, values in (
        ("owner", "muon", [1.0, 2.0]),
        ("shared_adamw", "adamw", [3.0, 4.0]),
        ("padded", "sharded_adamw", [5.0, 6.0, 0.0]),
        ("empty", "sharded_adamw", []),
    ):
        param = Dedicated()
        param._dmuon_route = route
        param._reduced_grad = torch.tensor(values) if route != "sharded_adamw" else None
        param._sharded_adamw_grad = torch.tensor(values) if route == "sharded_adamw" else None
        mapping[param] = name
    local = torch.tensor([7.0, 8.0])
    plain = torch.tensor([9.0, 10.0])
    model = SimpleNamespace(
        named_parameters=lambda: [
            ("dtensor", SimpleNamespace(grad=SimpleNamespace(_local_tensor=local))),
            ("plain", SimpleNamespace(grad=plain)),
        ]
    )
    monkeypatch.setattr(driver, "_compute_dedicated_fqns", lambda _model: mapping)
    return model


def test_canonical_assignment_covers_local_routes_padding_and_empty(monkeypatch):
    driver = load_driver()
    model = live_gradient_fixture(driver, monkeypatch)
    actual = driver.live_gradients(model)
    canonical = {name: tensor + 0.25 for name, tensor in actual.items()}
    canonical["padded"][-1] = 0
    driver.assign_canonical_gradients(model, canonical)
    for name, tensor in actual.items():
        torch.testing.assert_close(tensor, canonical[name], rtol=0, atol=0)
        if tensor.numel():
            assert tensor.data_ptr() != canonical[name].data_ptr()
    canonical["owner"].zero_()
    torch.testing.assert_close(actual["owner"], torch.tensor([1.25, 2.25]), rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["names", "shape", "dtype", "nan"])
def test_invalid_canonical_schema_is_rejected_before_any_copy(monkeypatch, invalid):
    driver = load_driver()
    model = live_gradient_fixture(driver, monkeypatch)
    actual = driver.live_gradients(model)
    previous = {name: tensor.clone() for name, tensor in actual.items()}
    canonical = {name: tensor + 1 for name, tensor in actual.items()}
    if invalid == "names":
        del canonical["empty"]
    elif invalid == "shape":
        canonical["padded"] = torch.zeros(4)
    elif invalid == "dtype":
        canonical["dtensor"] = canonical["dtensor"].double()
    else:
        canonical["plain"][0] = float("nan")
    with pytest.raises(AssertionError):
        driver.assign_canonical_gradients(model, canonical)
    for name in previous:
        torch.testing.assert_close(actual[name], previous[name], rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["ulp", "signed_zero", "nan", "dtype", "step"])
def test_exact_state_oracle_rejects_ulp_bits_and_nonfinite_values(invalid):
    driver = load_driver()
    expected = {"weight": torch.tensor([1.0]), "empty": torch.empty(0), "step": 2}
    actual = {**expected, "weight": expected["weight"].clone()}
    if invalid == "ulp":
        actual["weight"] = torch.nextafter(actual["weight"], torch.tensor([2.0]))
    elif invalid == "signed_zero":
        expected["weight"], actual["weight"] = torch.tensor([0.0]), torch.tensor([-0.0])
    elif invalid == "nan":
        expected["weight"] = actual["weight"] = torch.tensor([float("nan")])
    elif invalid == "dtype":
        actual["weight"] = actual["weight"].double()
    else:
        actual["step"] = 2.0
    failures = []
    driver.compare_exact(actual, expected, failures, path="controlled")
    assert len(failures) == 1


def test_exact_state_oracle_accepts_identical_bytes_and_empty_shards():
    driver = load_driver()
    expected = {"weight": torch.tensor([1.0, -0.0]), "empty": torch.empty(0), "step": 2}
    failures = []
    driver.compare_exact(expected, expected, failures, path="controlled")
    assert not failures
