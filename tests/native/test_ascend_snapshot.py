"""Exact pre/post migration comparison of separately executed NPU snapshots."""

import os

import pytest
import torch

from .ascend_cases import assert_tree_equal


def test_ascend_migration_snapshot():
    before = os.environ.get("DMUON_NPU_REFERENCE")
    after = os.environ.get("DMUON_NPU_SNAPSHOT")
    if not before or not after:
        pytest.skip("Set DMUON_NPU_REFERENCE and DMUON_NPU_SNAPSHOT to single-NPU snapshots")
    reference = torch.load(before, map_location="cpu", weights_only=False)
    actual = torch.load(after, map_location="cpu", weights_only=False)
    assert actual["metadata"]["legacy_bootstrap"] == "None"
    assert actual["metadata"]["restore_verified"] is True
    assert reference["metadata"]["restore_verified"] is False
    assert reference["metadata"]["legacy_restore_verified"] is False
    assert reference["metadata"]["pathway"] == actual["metadata"]["pathway"]
    assert reference["metadata"]["torch"] == actual["metadata"]["torch"]
    assert reference["metadata"]["torch_npu"] == actual["metadata"]["torch_npu"]
    assert reference["metadata"]["package"] != actual["metadata"]["package"]
    assert reference["metadata"]["momentum_policy"] == actual["metadata"]["momentum_policy"] == "optim_dtype"
    assert_tree_equal(actual["cases"], reference["cases"])
