"""Verify DedicatedParam caches numel / shard_dim / full_shape / tp_group
as plain attributes (not @property)."""

import torch
import torch.nn as nn
import pytest

from dmuon._backends.fsdp2.param import DedicatedParam


def _make_local_param(shape, requires_grad=True):
    return nn.Parameter(torch.randn(*shape), requires_grad=requires_grad)


def _make_dp(
    monkeypatch_dist=True,
    shape=(8, 4),
    is_owner=True,
    tp_group=None,
    tp_owner_local_rank=0,
    route_hint=None,
    muon_forward_unshard="broadcast",
):
    """Build a DedicatedParam on CPU with a minimal dp_group stub.

    We only need rank() + get_global_rank for __init__, so we patch with a
    trivial ProcessGroup-like object.
    """
    param = _make_local_param(shape)
    module = nn.Linear(shape[1], shape[0], bias=False)
    module.weight = param

    class _StubGroup:
        def size(self):
            return 2

        def rank(self):
            return 0 if is_owner else 1

    class _StubDist:
        @staticmethod
        def get_global_rank(group, rank):
            return rank

    # Patch dmuon._backends.fsdp2.param.dist.get_global_rank for this construction
    import dmuon._backends.fsdp2.param as param_module

    orig_dist = param_module.dist
    orig_compute_tp_group = DedicatedParam._compute_tp_group
    param_module.dist = _StubDist()
    if tp_group is not None:
        DedicatedParam._compute_tp_group = lambda self: tp_group
    try:
        dp = DedicatedParam(
            param=param,
            module=module,
            param_name="weight",
            owner_rank=0,
            dp_group=_StubGroup(),
            device=torch.device("cpu"),
            compute_dtype=None,
            tp_owner_local_rank=tp_owner_local_rank,
            route_hint=route_hint,
            muon_forward_unshard=muon_forward_unshard,
        )
    finally:
        param_module.dist = orig_dist
        DedicatedParam._compute_tp_group = orig_compute_tp_group
    return dp


def test_numel_is_cached_attr_not_property():
    dp = _make_dp(shape=(8, 4))
    # Must be plain int, not a property descriptor
    cls_attr = type(dp).__dict__.get("numel")
    assert not isinstance(cls_attr, property), (
        "numel must be a cached instance attr, not a @property"
    )
    assert isinstance(dp.numel, int)
    assert dp.numel == 32


def test_numel_matches_orig_size():
    for shape in [(8, 4), (1, 1), (3, 5, 7)]:
        dp = _make_dp(shape=shape)
        expected = 1
        for s in shape:
            expected *= s
        assert dp.numel == expected
        assert dp.numel == dp._orig_size.numel()


def test_shard_dim_is_cached_attr_not_property():
    dp = _make_dp(shape=(8, 4))
    cls_attr = type(dp).__dict__.get("shard_dim")
    assert not isinstance(cls_attr, property), "shard_dim must be cached attr"
    # Non-DTensor param: shard_dim is None
    assert dp.shard_dim is None


def test_full_shape_is_cached_attr_not_property():
    dp = _make_dp(shape=(8, 4))
    cls_attr = type(dp).__dict__.get("full_shape")
    assert not isinstance(cls_attr, property), "full_shape must be cached attr"
    # Non-DTensor param: full_shape == _orig_size
    assert dp.full_shape == dp._orig_size


def test_tp_group_is_cached_attr_not_property():
    dp = _make_dp(shape=(8, 4))
    cls_attr = type(dp).__dict__.get("tp_group")
    assert not isinstance(cls_attr, property), "tp_group must be cached attr"
    assert dp.tp_group is None


def test_tp_owner_local_rank_is_used_when_tp_group_exists():
    class _StubTPGroup:
        def size(self):
            return 2

        def rank(self):
            return 1

    dp = _make_dp(tp_group=_StubTPGroup(), tp_owner_local_rank=1)
    assert dp._tp_owner_local_rank == 1
    assert dp._tp_owner_global_rank == 1
    assert dp.is_tp_owner is True


def test_tp_owner_local_rank_bounds_checked():
    class _StubTPGroup:
        def size(self):
            return 2

        def rank(self):
            return 0

    with pytest.raises(ValueError, match="outside TP group size"):
        _make_dp(tp_group=_StubTPGroup(), tp_owner_local_rank=2)


def test_muon_all_gather_forward_uses_sharded_storage():
    dp = _make_dp(route_hint="muon", muon_forward_unshard="all_gather")
    assert dp.uses_sharded_muon_forward()
    assert dp._sharded_muon_chunk_numel == 16
    assert dp._sharded_muon_valid_numel == 16
    assert dp._sharded_muon_data is not None
    assert dp._sharded_muon_full_padded is not None
    assert dp._sharded_muon_comm_shard is not None
    assert dp._sharded_muon_scatter_input is not None
    assert dp._unsharded_param is not None
    assert dp._unsharded_param.shape == dp._orig_size


if __name__ == "__main__":
    test_numel_is_cached_attr_not_property()
    test_numel_matches_orig_size()
    test_shard_dim_is_cached_attr_not_property()
    test_full_shape_is_cached_attr_not_property()
    test_tp_group_is_cached_attr_not_property()
    test_tp_owner_local_rank_is_used_when_tp_group_exists()
    test_tp_owner_local_rank_bounds_checked()
    print("OK")
