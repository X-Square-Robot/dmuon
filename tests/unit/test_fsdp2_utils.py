"""Unit tests for dmuon._internal_utils (FSDP2-borrowed helpers)."""

import torch
import torch.nn as nn

from dmuon._core.internal_utils import (
    alloc_storage,
    free_storage,
    set_requires_grad_if_needed,
    unsafe_setattr_param,
)


def test_alloc_free_storage_roundtrip():
    t = torch.empty(16, dtype=torch.float32)
    assert t.untyped_storage().size() == 16 * 4
    free_storage(t)
    assert t.untyped_storage().size() == 0
    # idempotent
    free_storage(t)
    assert t.untyped_storage().size() == 0
    alloc_storage(t)
    assert t.untyped_storage().size() == 16 * 4
    # idempotent
    alloc_storage(t)
    assert t.untyped_storage().size() == 16 * 4


def test_alloc_storage_preserves_object_identity():
    """After alloc/free roundtrip, Tensor object stays the same — this is the
    invariant that lets FSDP2 reuse nn.Parameter across forwards."""
    t = torch.ones(4, dtype=torch.float32)
    original_id = id(t)
    free_storage(t)
    alloc_storage(t)
    assert id(t) == original_id


def test_unsafe_setattr_param_fast_path():
    """Module without custom __setattr__ — should write directly to
    ``_parameters`` dict and bypass ``register_parameter``."""
    m = nn.Linear(4, 4, bias=False)
    new_param = nn.Parameter(torch.zeros(4, 4))
    unsafe_setattr_param(m, "weight", new_param)
    assert m.weight is new_param
    assert m._parameters["weight"] is new_param
    # named_parameters still reports it
    names = dict(m.named_parameters())
    assert "weight" in names and names["weight"] is new_param


def test_unsafe_setattr_param_slow_path():
    """Module with overridden __setattr__ — should fall back to regular setattr."""

    class CustomModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2))
            self.log = []

        def __setattr__(self, name, value):
            self.__dict__.setdefault("log", []).append(name)
            super().__setattr__(name, value)

    m = CustomModule()
    m.log.clear()
    new_param = nn.Parameter(torch.ones(2))
    unsafe_setattr_param(m, "weight", new_param)
    assert m.weight is new_param
    # Custom __setattr__ DID get invoked (slow path)
    assert "weight" in m.log


def test_set_requires_grad_if_needed_noop():
    """When flags already match, dst is untouched."""
    src = torch.zeros(4, requires_grad=True)
    dst = torch.ones(4, requires_grad=True)
    set_requires_grad_if_needed(src, dst)
    assert dst.requires_grad is True


def test_set_requires_grad_if_needed_flip():
    src = torch.zeros(4, requires_grad=False)
    dst = torch.ones(4, requires_grad=True)
    set_requires_grad_if_needed(src, dst)
    assert dst.requires_grad is False


if __name__ == "__main__":
    test_alloc_free_storage_roundtrip()
    test_alloc_storage_preserves_object_identity()
    test_unsafe_setattr_param_fast_path()
    test_unsafe_setattr_param_slow_path()
    test_set_requires_grad_if_needed_noop()
    test_set_requires_grad_if_needed_flip()
    print("OK")
