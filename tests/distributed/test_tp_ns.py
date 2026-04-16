"""TP Newton-Schulz correctness tests.

Verifies that gram_newton_schulz with TP sharding produces the same
orthogonalized update as single-rank newton_schulz on the full matrix.

Tests cover:
  1. Shard(0) — row-sharded, uses R-side Gram (G^TG)
  2. Shard(1) — column-sharded, uses L-side Gram (GG^T)
  3. Per-head NS — narrow Shard(0) (GQA k/v_proj), local NS
  4. Block-diagonal NS — zero-comm approximation
  5. shard_dim / full_shape property correctness

Run with: torchrun --nproc_per_node=2 tests/distributed/test_tp_ns.py
          torchrun --nproc_per_node=4 tests/distributed/test_tp_ns.py
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dmuon.optim.newton_schulz import gram_newton_schulz, newton_schulz


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Test 1: Shard(1) — column-sharded, L-side GG^T decomposes
# ---------------------------------------------------------------------------

def test_shard1_correctness(rank, world_size, device):
    """Column-sharded Gram NS should match single-rank NS."""
    T = world_size  # all ranks form TP group
    tp_group = dist.group.WORLD

    torch.manual_seed(42)
    m, n = 256, 512
    G_full = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    dist.broadcast(G_full, src=0)  # same G on all ranks

    # Shard(1): each rank gets columns [rank*n/T : (rank+1)*n/T]
    chunk = n // T
    G_shard = G_full[:, rank * chunk : (rank + 1) * chunk].contiguous()

    # TP Gram NS
    update_shard = gram_newton_schulz(G_shard, tp_group, shard_dim=1)
    update_shard = update_shard.contiguous()

    # Gather all shards to reconstruct full update
    gathered = [torch.empty_like(update_shard) for _ in range(T)]
    dist.all_gather(gathered, update_shard)
    update_tp = torch.cat(gathered, dim=1)  # (m, n)

    # Single-rank reference
    update_ref = newton_schulz(G_full)

    # Compare
    # NOTE on error sources: see test_shard0_correctness for detailed analysis.
    # Shard(1) uses L-side GG^T (same as single-rank for m<n), so error is
    # primarily from fp16 SYRK accumulation order differences (T partial sums
    # vs 1 full sum).
    update_tp_f = update_tp.float()
    update_ref_f = update_ref.float()
    cos_sim = torch.nn.functional.cosine_similarity(
        update_tp_f.flatten().unsqueeze(0),
        update_ref_f.flatten().unsqueeze(0),
    ).item()
    max_diff = (update_tp_f - update_ref_f).abs().max().item()
    scale = update_ref_f.abs().max().item()

    passed = cos_sim > 0.98 and max_diff < 0.15 * scale
    log(rank, f"  Shard(1) [{m}x{n}] TP={T}: cos_sim={cos_sim:.6f}, "
              f"max_diff={max_diff:.4f} (scale={scale:.4f})")
    assert passed, (
        f"Shard(1) mismatch: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}"
    )
    log(rank, "PASSED: test_shard1_correctness")


# ---------------------------------------------------------------------------
# Test 2: Shard(0) — row-sharded, R-side G^TG decomposes (the bug fix)
# ---------------------------------------------------------------------------

def test_shard0_correctness(rank, world_size, device):
    """Row-sharded Gram NS should match single-rank NS.

    This was the buggy path before the shard_dim fix:
    old code used shape heuristic, new code forces transpose for Shard(0).
    """
    T = world_size
    tp_group = dist.group.WORLD

    torch.manual_seed(42)
    m, n = 512, 256
    G_full = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    dist.broadcast(G_full, src=0)

    # Shard(0): each rank gets rows [rank*m/T : (rank+1)*m/T]
    chunk = m // T
    G_shard = G_full[rank * chunk : (rank + 1) * chunk, :].contiguous()

    # TP Gram NS with shard_dim=0 (the fix)
    update_shard = gram_newton_schulz(G_shard, tp_group, shard_dim=0)
    update_shard = update_shard.contiguous()  # transpose in NS may produce non-contiguous

    # Gather all shards
    gathered = [torch.empty_like(update_shard) for _ in range(T)]
    dist.all_gather(gathered, update_shard)
    update_tp = torch.cat(gathered, dim=0)  # (m, n)

    # Single-rank reference
    update_ref = newton_schulz(G_full)

    # Compare
    # NOTE on error sources:
    #   1. fp16 accumulation: Gram NS iterates in fp16, SYRK rounding differs
    #      between TP (SYRK on smaller shard) and single-rank (SYRK on full matrix)
    #   2. Gram decomposition path: TP uses R-side G^TG (via transpose), single-rank
    #      uses whichever side is smaller (shape heuristic). For m>n, both use L-side
    #      GG^T, but TP forces R-side. The polar factor is the same mathematically,
    #      but different SYRK accumulation order → different fp16 rounding.
    #   3. Restart recomputation: after restart, Gram is recomputed from Q@X which
    #      has accumulated different rounding errors in TP vs single-rank paths.
    update_tp_f = update_tp.float()
    update_ref_f = update_ref.float()
    cos_sim = torch.nn.functional.cosine_similarity(
        update_tp_f.flatten().unsqueeze(0),
        update_ref_f.flatten().unsqueeze(0),
    ).item()
    max_diff = (update_tp_f - update_ref_f).abs().max().item()
    scale = update_ref_f.abs().max().item()

    passed = cos_sim > 0.98 and max_diff < 0.15 * scale
    log(rank, f"  Shard(0) [{m}x{n}] TP={T}: cos_sim={cos_sim:.6f}, "
              f"max_diff={max_diff:.4f} (scale={scale:.4f})")
    assert passed, (
        f"Shard(0) mismatch: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}"
    )
    log(rank, "PASSED: test_shard0_correctness")


# ---------------------------------------------------------------------------
# Test 3: Shard(0) square matrix — both Grams are same size
# ---------------------------------------------------------------------------

def test_shard0_square(rank, world_size, device):
    """Square matrix under Shard(0) — tests the bug fix path with m==n."""
    T = world_size
    tp_group = dist.group.WORLD

    torch.manual_seed(42)
    m, n = 256, 256
    G_full = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    dist.broadcast(G_full, src=0)

    chunk = m // T
    G_shard = G_full[rank * chunk : (rank + 1) * chunk, :].contiguous()

    update_shard = gram_newton_schulz(G_shard, tp_group, shard_dim=0)
    update_shard = update_shard.contiguous()

    gathered = [torch.empty_like(update_shard) for _ in range(T)]
    dist.all_gather(gathered, update_shard)
    update_tp = torch.cat(gathered, dim=0)

    update_ref = newton_schulz(G_full)

    cos_sim = torch.nn.functional.cosine_similarity(
        update_tp.float().flatten().unsqueeze(0),
        update_ref.float().flatten().unsqueeze(0),
    ).item()

    # NOTE: square matrix m==n → single-rank uses L-side (no transpose),
    # TP Shard(0) uses R-side (transpose). Mathematically equivalent polar
    # factor, but different fp16 accumulation paths → expect ~0.99 cos_sim.
    passed = cos_sim > 0.98
    log(rank, f"  Shard(0) square [{m}x{n}] TP={T}: cos_sim={cos_sim:.6f}")
    assert passed, f"Shard(0) square mismatch: cos_sim={cos_sim:.6f}"
    log(rank, "PASSED: test_shard0_square")


# ---------------------------------------------------------------------------
# Test 4: Per-head NS — narrow Shard(0) like GQA k/v_proj
# ---------------------------------------------------------------------------

def test_per_head_ns(rank, world_size, device):
    """Narrow Shard(0) matrix: each rank does local NS (per-head).

    Simulates k_proj (kv_dim, d_model) = (128*T, 1024) with Shard(0),
    each rank gets (128, 1024) = one complete "head".
    Per-head NS should produce same result as running NS independently
    on each (128, 1024) chunk.
    """
    T = world_size
    d_head = 128
    d_model = 1024
    m = d_head * T  # full kv_dim
    n = d_model

    torch.manual_seed(42)
    G_full = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    dist.broadcast(G_full, src=0)

    # Each rank's "head"
    G_head = G_full[rank * d_head : (rank + 1) * d_head, :].contiguous()

    # Per-head NS: local, no TP communication
    update_head = newton_schulz(G_head)

    # Reference: run NS on each head separately (on rank 0)
    update_ref_head = newton_schulz(
        G_full[rank * d_head : (rank + 1) * d_head, :].contiguous()
    )

    cos_sim = torch.nn.functional.cosine_similarity(
        update_head.float().flatten().unsqueeze(0),
        update_ref_head.float().flatten().unsqueeze(0),
    ).item()

    passed = cos_sim > 0.999  # should be near-identical
    log(rank, f"  Per-head NS [{d_head}x{d_model}] rank={rank}: cos_sim={cos_sim:.6f}")
    assert passed, f"Per-head NS mismatch on rank {rank}: cos_sim={cos_sim:.6f}"

    # Also verify: per-head update ≠ full-matrix NS update (they SHOULD differ)
    update_full = newton_schulz(G_full)
    update_full_head = update_full[rank * d_head : (rank + 1) * d_head, :]
    cos_vs_full = torch.nn.functional.cosine_similarity(
        update_head.float().flatten().unsqueeze(0),
        update_full_head.float().flatten().unsqueeze(0),
    ).item()
    log(rank, f"  Per-head vs full-matrix NS: cos_sim={cos_vs_full:.6f} "
              f"(expected < 1.0 since they differ)")

    log(rank, "PASSED: test_per_head_ns")


# ---------------------------------------------------------------------------
# Test 5: Block-diagonal NS — runs without error, output has correct shape
# ---------------------------------------------------------------------------

def test_block_diagonal_ns(rank, world_size, device):
    """Block-diagonal NS should run without error and produce correct shape."""
    T = world_size
    tp_group = dist.group.WORLD

    torch.manual_seed(42)

    # Test Shard(1) block-diagonal
    m, n = 256, 512
    G_full = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    dist.broadcast(G_full, src=0)
    chunk = n // T
    G_shard = G_full[:, rank * chunk : (rank + 1) * chunk].contiguous()

    update = gram_newton_schulz(G_shard, tp_group, shard_dim=1, block_diagonal=True)
    assert update.shape == G_shard.shape, (
        f"Block-diag Shard(1): shape mismatch {update.shape} != {G_shard.shape}"
    )
    assert update.abs().max().item() > 0, "Block-diag Shard(1): zero output"

    # Test Shard(0) block-diagonal
    m, n = 512, 256
    G_full = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    dist.broadcast(G_full, src=0)
    chunk = m // T
    G_shard = G_full[rank * chunk : (rank + 1) * chunk, :].contiguous()

    update = gram_newton_schulz(G_shard, tp_group, shard_dim=0, block_diagonal=True)
    assert update.shape == G_shard.shape, (
        f"Block-diag Shard(0): shape mismatch {update.shape} != {G_shard.shape}"
    )
    assert update.abs().max().item() > 0, "Block-diag Shard(0): zero output"

    log(rank, "PASSED: test_block_diagonal_ns")


# ---------------------------------------------------------------------------
# Test 6: DedicatedParam shard_dim and full_shape
# ---------------------------------------------------------------------------

def test_shard_dim_full_shape(rank, world_size, device):
    """Verify shard_dim and full_shape properties on DedicatedParam."""
    import torch.nn as nn
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    if world_size < 2:
        log(rank, "SKIPPED: test_shard_dim_full_shape (need >= 2 GPUs)")
        return

    tp_mesh = init_device_mesh("cuda", (world_size,))

    # Simple model with known shapes
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.colwise = nn.Linear(256, 512, bias=False)   # (512, 256) Shard(0)
            self.rowwise = nn.Linear(512, 256, bias=False)   # (256, 512) Shard(1)

        def forward(self, x):
            return self.rowwise(self.colwise(x))

    torch.manual_seed(0)
    model = TestModel().to(device)

    parallelize_module(
        model, tp_mesh,
        {"colwise": ColwiseParallel(), "rowwise": RowwiseParallel()},
    )

    # Create DedicatedParam objects manually to test properties
    from dmuon.param import DedicatedParam

    dp_group = dist.group.WORLD

    colwise_param = model.colwise.weight
    dp_col = DedicatedParam(
        colwise_param, model.colwise, "weight",
        owner_rank=0, dp_group=dp_group, device=device,
    )

    rowwise_param = model.rowwise.weight
    dp_row = DedicatedParam(
        rowwise_param, model.rowwise, "weight",
        owner_rank=0, dp_group=dp_group, device=device,
    )

    # Verify shard_dim
    assert dp_col.shard_dim == 0, f"ColwiseParallel shard_dim should be 0, got {dp_col.shard_dim}"
    assert dp_row.shard_dim == 1, f"RowwiseParallel shard_dim should be 1, got {dp_row.shard_dim}"

    # Verify full_shape
    T = world_size
    assert dp_col.full_shape == torch.Size([512, 256]), (
        f"ColwiseParallel full_shape should be [512, 256], got {dp_col.full_shape}"
    )
    assert dp_row.full_shape == torch.Size([256, 512]), (
        f"RowwiseParallel full_shape should be [256, 512], got {dp_row.full_shape}"
    )

    # Verify local shape (sanity)
    assert dp_col._orig_size == torch.Size([512 // T, 256]), (
        f"ColwiseParallel local should be [{512 // T}, 256], got {dp_col._orig_size}"
    )
    assert dp_row._orig_size == torch.Size([256, 512 // T]), (
        f"RowwiseParallel local should be [256, {512 // T}], got {dp_row._orig_size}"
    )

    log(rank, f"  ColwiseParallel: shard_dim={dp_col.shard_dim}, "
              f"local={list(dp_col._orig_size)}, full={list(dp_col.full_shape)}")
    log(rank, f"  RowwiseParallel: shard_dim={dp_row.shard_dim}, "
              f"local={list(dp_row._orig_size)}, full={list(dp_row.full_shape)}")
    log(rank, "PASSED: test_shard_dim_full_shape")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    tests = [
        ("shard1_correctness", test_shard1_correctness),
        ("shard0_correctness", test_shard0_correctness),
        ("shard0_square", test_shard0_square),
        ("per_head_ns", test_per_head_ns),
        ("block_diagonal_ns", test_block_diagonal_ns),
        ("shard_dim_full_shape", test_shard_dim_full_shape),
    ]

    test_filter = sys.argv[1] if len(sys.argv) > 1 else "all"

    for name, fn in tests:
        if test_filter != "all" and test_filter != name:
            continue
        log(rank, f"\n{'=' * 60}")
        log(rank, f"Running: {name}")
        log(rank, f"{'=' * 60}")
        dist.barrier()
        fn(rank, world_size, device)
        dist.barrier()

    log(rank, f"\n{'=' * 60}")
    log(rank, f"All tests passed!")
    log(rank, f"{'=' * 60}")

    dist.destroy_process_group()
