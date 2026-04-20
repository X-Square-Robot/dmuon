"""Phase 3 unit tests: torch._foreach_copy_ behavior required for DMuon
``DedicatedParamGroup.unshard`` owner copy-in path.

These tests are single-process (no torchrun) — they verify the raw tensor
semantics we rely on, not the distributed unshard flow itself (covered by
``tests/distributed/test_e2e_dp.py``).
"""

import pytest
import torch

from dmuon._internal_utils import alloc_storage, free_storage


def _cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda", 0)


def test_foreach_copy_cross_dtype_bit_equal_to_sequential():
    """fp32 src → bf16 dst via ``_foreach_copy_`` must match per-slice
    ``.copy_`` exactly. This is the core invariant for Phase 3 mixed
    precision (``_owned_data`` fp32 → ``_packed_buf`` bf16)."""
    device = _cuda_or_skip()
    sizes = [7, 13, 29, 64, 127]
    total = sum(sizes)

    packed = torch.empty(total, dtype=torch.bfloat16, device=device)
    dsts = list(torch.split(packed, sizes))
    srcs = [torch.randn(s, dtype=torch.float32, device=device) for s in sizes]

    torch._foreach_copy_(dsts, srcs)

    ref = torch.empty(total, dtype=torch.bfloat16, device=device)
    off = 0
    for s, src in zip(sizes, srcs):
        ref[off : off + s].copy_(src)
        off += s

    torch.cuda.synchronize()
    assert torch.equal(packed, ref), "foreach_copy mismatch vs sequential"


def test_precomputed_dst_views_survive_resize_roundtrip():
    """dst views cached once before unshard loop must still produce correct
    results after ``free_storage`` → ``alloc_storage``. This is the key
    assumption behind caching ``_copy_in_dsts_by_owner`` in
    ``DedicatedParamGroup.__init__``.
    """
    device = _cuda_or_skip()
    sizes = [10, 20, 30]
    total = sum(sizes)

    packed = torch.empty(total, dtype=torch.bfloat16, device=device)

    # Cache dsts ONCE (mirrors group __init__)
    offset = 0
    cached_dsts = []
    for s in sizes:
        cached_dsts.append(packed[offset : offset + s])
        offset += s

    srcs = [torch.randn(s, dtype=torch.float32, device=device) for s in sizes]

    # Baseline write via cached dsts before any resize
    torch._foreach_copy_(cached_dsts, srcs)
    torch.cuda.synchronize()
    baseline = packed.clone()

    # Simulate several unshard/reshard cycles: free → alloc → foreach_copy
    for _ in range(3):
        free_storage(packed)
        assert packed.untyped_storage().size() == 0
        alloc_storage(packed)
        assert packed.untyped_storage().size() == total * 2  # bf16

        # Use the ORIGINAL cached dsts (not re-split)
        torch._foreach_copy_(cached_dsts, srcs)
        torch.cuda.synchronize()
        assert torch.equal(packed, baseline), (
            "cached dst views produced mismatched packed buffer after resize"
        )


def test_foreach_copy_is_equivalent_to_per_param_copy_in_packed_layout():
    """End-to-end equivalence check: given a packed buffer and N per-param
    source tensors, ``_foreach_copy_(dsts, srcs)`` must produce the same
    bytes as the current Phase 2 loop ``packed[off:off+n].copy_(src)``.
    """
    device = _cuda_or_skip()
    torch.manual_seed(0)
    shapes = [(4, 8), (16,), (3, 5, 7)]
    numels = [int(torch.prod(torch.tensor(s)).item()) for s in shapes]
    total = sum(numels)

    # Simulated owned_data tensors (fp32 master copies)
    owned = [torch.randn(*s, dtype=torch.float32, device=device) for s in shapes]

    # Method A: Phase 2 sequential copy_
    packed_a = torch.empty(total, dtype=torch.bfloat16, device=device)
    off = 0
    for src, n in zip(owned, numels):
        packed_a[off : off + n].copy_(src.view(-1))
        off += n

    # Method B: Phase 3 foreach_copy with precomputed dst views
    packed_b = torch.empty(total, dtype=torch.bfloat16, device=device)
    offset = 0
    dsts = []
    for n in numels:
        dsts.append(packed_b[offset : offset + n])
        offset += n
    srcs = [src.view(-1) for src in owned]
    torch._foreach_copy_(dsts, srcs)

    torch.cuda.synchronize()
    assert torch.equal(packed_a, packed_b), (
        "Phase 3 foreach_copy result differs from Phase 2 sequential copy_"
    )


def test_foreach_copy_single_element_list():
    """Owner might own just 1 param in a group. foreach_copy with a
    length-1 list should still work (degenerate case)."""
    device = _cuda_or_skip()
    packed = torch.empty(16, dtype=torch.bfloat16, device=device)
    src = torch.randn(16, dtype=torch.float32, device=device)
    torch._foreach_copy_([packed[:16]], [src])
    torch.cuda.synchronize()
    ref = torch.empty(16, dtype=torch.bfloat16, device=device)
    ref.copy_(src)
    assert torch.equal(packed, ref)
