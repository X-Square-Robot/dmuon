"""Unit coverage for profiled_ilp owner assignment plumbing."""

import os
import sys

os.environ.setdefault("DMUON_CACHE_DIR", "/tmp/dmuon_test_cache")

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import importlib

import pytest
import torch
import torch.nn as nn

from dmuon._core.partition import compute_balanced_assignment
from dmuon.optim.profiled_batch import (
    normalize_profiled_ilp_config,
    require_profiled_ilp_dependencies,
)


class FakeDeviceMesh:
    mesh_dim_names = None

    def __init__(self, world_size: int):
        self._world_size = world_size

    def size(self):
        return self._world_size


def test_profiled_ilp_dependency_error_has_install_command(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "scipy":
            raise ImportError("synthetic missing scipy")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    with pytest.raises(ImportError) as exc_info:
        require_profiled_ilp_dependencies()

    message = str(exc_info.value)
    assert "owner_strategy='profiled_ilp' requires scipy" in message
    assert "pip install scipy" in message
    assert "dmuon[profiled_ilp]" in message


def test_profiled_ilp_tilelang_dependency_error_has_install_command(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "scipy":
            return object()
        if name == "tilelang":
            raise ImportError("synthetic missing tilelang")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    with pytest.raises(ImportError) as exc_info:
        require_profiled_ilp_dependencies()

    message = str(exc_info.value)
    assert "owner_strategy='profiled_ilp' requires tilelang" in message
    assert "pip install tilelang" in message
    assert "dmuon[profiled_ilp]" in message


def test_profiled_ilp_config_normalizes_user_facing_strings():
    config = normalize_profiled_ilp_config(
        {
            "dtype": "bf16",
            "device": "cuda:0",
            "backends": "tilelang,cute_sm80,cublas",
            "max_batch": 8,
        }
    )

    assert config.dtype is torch.bfloat16
    assert config.device == torch.device("cuda:0")
    assert config.backends == ("tilelang", "cute_sm80", "cublas")
    assert config.max_batch == 8


def test_profiled_ilp_config_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="unknown profiled_ilp dtype"):
        normalize_profiled_ilp_config({"dtype": "fp8"})


def test_legacy_owner_strategies_do_not_set_batch_metadata():
    model = nn.Sequential(
        nn.Linear(8, 8, bias=False),
        nn.Linear(8, 8, bias=False),
    )
    mesh = FakeDeviceMesh(2)

    for strategy in ("lpt", "round_robin", "rank0"):
        result = compute_balanced_assignment(
            model,
            mesh,
            predicate=lambda _name, _param: True,
            owner_strategy=strategy,
        )
        assert result.dp_owners
        assert result.batch_groups == {}
        assert result.metadata == {}
