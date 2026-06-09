"""SYRK B!=A bug investigation: isolate root cause of NaN and non-determinism.

This test bypasses the B!=A guard (calls the kernel directly) to diagnose the
mirror-write epilogue bug. It is NOT a correctness test — it is a debugging tool.

Root cause hypothesis:
1. Mirror epilogue smem transpose has uninitialized reads (stale mainloop sA data)
2. Cooperative store pattern (if tidx < tile_m) misses some elements
3. Tile indexing is correct but MMA register layout doesn't densely cover smem

Test strategy:
- Isolate mirror vs mainloop by comparing single-tile (diagonal only) vs multi-tile
- Test symmetric B!=A (where mirror SHOULD be correct) vs arbitrary B!=A
- Compare lower triangle only (skip mirror) vs full matrix

Run with: python tests/unit/test_syrk_bneqa_debug.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from dmuon.optim.syrk_dispatch import HAS_SYRK

if not HAS_SYRK:
    print("SKIP: CuteDSL SYRK kernel not available")
    sys.exit(0)
if not torch.cuda.is_available():
    print("SKIP: CUDA not available")
    sys.exit(0)


# We need to call the kernel internals directly, bypassing the B!=A guard.
# Import the compilation function and call the compiled kernel directly.
from dmuon.kernels.syrk_sm80 import _compile_syrk_sm80
from dmuon.kernels.cute_dsl_utils import torch2cute_dtype_map
from cutlass import Float32

DEVICE = torch.device("cuda")


def _call_syrk_raw(A, D, B=None, C=None, alpha=1.0, beta=1.0, diag_add=0.0,
                    tile_m=128, tile_k=32, num_stages=4, skip_mirror=False):
    """Call SYRK kernel directly, bypassing the B!=A guard."""
    squeeze = A.ndim == 2
    if squeeze:
        A = A.unsqueeze(0)
        D = D.unsqueeze(0)
        if B is not None:
            B = B.unsqueeze(0)
        if C is not None:
            C = C.unsqueeze(0)

    if B is None:
        B = A

    if A.stride(-1) != 1:
        A = A.contiguous()
    if B.stride(-1) != 1:
        B = B.contiguous()

    A_p = A.permute(1, 2, 0)
    B_p = B.permute(1, 2, 0)
    D_p = D.permute(1, 2, 0)
    C_p = C.permute(1, 2, 0) if C is not None else None

    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype]
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None

    alpha_mode = 1 if alpha != 1.0 else 0
    beta_mode = 1 if beta != 1.0 else 0
    diag_add_mode = 1 if diag_add != 0.0 else 0

    compiled_fn = _compile_syrk_sm80(
        a_dtype, b_dtype, d_dtype, c_dtype,
        alpha_mode, beta_mode, diag_add_mode=diag_add_mode,
        tile_m=tile_m, tile_k=tile_k, num_stages=num_stages,
        skip_mirror=skip_mirror,
    )

    alpha_arg = Float32(alpha) if alpha_mode else None
    beta_arg = Float32(beta) if beta_mode else None
    diag_add_arg = Float32(diag_add) if diag_add_mode else None

    compiled_fn(A_p, B_p, D_p, C_p, alpha_arg, beta_arg, diag_add_arg)


def _tril_match(ref, out, atol=0.01):
    """Compare only lower triangle (excludes mirror write)."""
    mask = torch.tril(torch.ones_like(ref, dtype=torch.bool))
    ref_l = ref[mask].float()
    out_l = out[mask].float()
    return (ref_l - out_l).abs().max().item()


# ---------------------------------------------------------------------------
# Test 1: Single tile (M=tile_m) — only diagonal tile, no mirror
# ---------------------------------------------------------------------------

def test_single_tile():
    """With only 1 tile (M=tile_m), only diagonal tile is launched.
    Diagonal mirror is simple element-wise copy. If this fails, bug is in mainloop."""
    print("\n=== Test 1: Single tile (diagonal only, no off-diagonal mirror) ===")
    for dtype in [torch.float16, torch.bfloat16]:
        for tile_m in [64, 128]:
            M = tile_m  # exactly 1 tile
            K = 256
            torch.manual_seed(42)
            A = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.01
            B = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.01

            ref = torch.mm(A, B.T)
            out = torch.empty(M, M, device=DEVICE, dtype=dtype)
            _call_syrk_raw(A, out, B=B, tile_m=tile_m, tile_k=32, num_stages=4)

            has_nan = torch.isnan(out).any().item()
            max_diff = (ref.float() - out.float()).abs().max().item()
            tril_diff = _tril_match(ref, out)
            sym_diff = (out.float() - out.float().T).abs().max().item()
            print(f"  tile_m={tile_m} {dtype}: NaN={has_nan} "
                  f"max_diff={max_diff:.6f} tril_diff={tril_diff:.6f} "
                  f"sym_diff={sym_diff:.6f}")


# ---------------------------------------------------------------------------
# Test 2: Two tiles — introduces off-diagonal mirror
# ---------------------------------------------------------------------------

def test_two_tiles():
    """M = 2*tile_m → 3 tiles (2 diagonal + 1 off-diagonal).
    If single tile passes but this fails, bug is in off-diagonal mirror."""
    print("\n=== Test 2: Two tiles (with off-diagonal mirror) ===")
    for dtype in [torch.float16, torch.bfloat16]:
        for tile_m in [64, 128]:
            M = 2 * tile_m
            K = 256
            torch.manual_seed(42)
            A = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.01
            B = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.01

            ref = torch.mm(A, B.T)
            out = torch.empty(M, M, device=DEVICE, dtype=dtype)
            _call_syrk_raw(A, out, B=B, tile_m=tile_m, tile_k=32, num_stages=4)

            has_nan = torch.isnan(out).any().item()
            max_diff = (ref.float() - out.float()).abs().max().item()
            tril_diff = _tril_match(ref, out)

            # Check mirror correctness: upper = lower^T?
            lower = torch.tril(out.float(), diagonal=-1)
            upper = torch.triu(out.float(), diagonal=1)
            mirror_diff = (lower.T - upper).abs().max().item()

            print(f"  tile_m={tile_m} M={M} {dtype}: NaN={has_nan} "
                  f"max_diff={max_diff:.6f} tril_diff={tril_diff:.6f} "
                  f"mirror_diff={mirror_diff:.6f}")


# ---------------------------------------------------------------------------
# Test 3: Scaling M — find the threshold where NaN appears
# ---------------------------------------------------------------------------

def test_nan_threshold():
    """Sweep M to find where NaN first appears for B!=A."""
    print("\n=== Test 3: NaN threshold sweep ===")
    for dtype in [torch.float16, torch.bfloat16]:
        for tile_m in [64, 128]:
            for num_tiles in [1, 2, 3, 4, 8, 16]:
                M = tile_m * num_tiles
                K = 256
                torch.manual_seed(42)
                A = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.01
                B = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.01

                out = torch.empty(M, M, device=DEVICE, dtype=dtype)
                _call_syrk_raw(A, out, B=B, tile_m=tile_m, tile_k=32, num_stages=4)

                nan_count = torch.isnan(out).sum().item()
                total = M * M
                has_nan = nan_count > 0
                tag = "NaN!" if has_nan else "OK  "
                print(f"  [{tag}] tile_m={tile_m} M={M:5d} ({num_tiles} tiles) "
                      f"{dtype}: nan_count={nan_count}/{total}")


# ---------------------------------------------------------------------------
# Test 4: Symmetric B!=A — result SHOULD be symmetric
# ---------------------------------------------------------------------------

def test_symmetric_bneqa():
    """B!=A but A@B^T is symmetric (A,B are polynomials of same symmetric matrix).
    If the mirror write is correct for symmetric results, this should pass."""
    print("\n=== Test 4: Symmetric B!=A ===")
    for dtype in [torch.float16, torch.bfloat16]:
        for M in [128, 256, 512]:
            tile_m = 64 if M % 128 != 0 else 128
            K = M * 2
            torch.manual_seed(42)
            X = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.01
            R = torch.mm(X, X.T)  # symmetric
            Z = 0.5 * torch.mm(R, R) + 0.3 * R  # symmetric, commutes with R

            ref = torch.mm(R, Z.T)  # R@Z = R@Z^T since Z symmetric + commutes
            out = torch.empty(M, M, device=DEVICE, dtype=dtype)
            _call_syrk_raw(R, out, B=Z, tile_m=tile_m, tile_k=32, num_stages=4)

            has_nan = torch.isnan(out).any().item()
            max_diff = (ref.float() - out.float()).abs().max().item()
            tril_diff = _tril_match(ref, out)
            print(f"  M={M} {dtype}: NaN={has_nan} "
                  f"max_diff={max_diff:.6f} tril_diff={tril_diff:.6f}")


# ---------------------------------------------------------------------------
# Test 5: Lower triangle correctness isolation
# ---------------------------------------------------------------------------

def test_lower_triangle_only():
    """Check if the LOWER triangle of B!=A output is correct.
    If lower triangle is correct but full matrix is wrong → mirror bug.
    If lower triangle is also wrong → mainloop/tile-indexing bug."""
    print("\n=== Test 5: Lower triangle correctness ===")
    for dtype in [torch.float16]:
        for M in [128, 256, 512]:
            tile_m = 128 if M % 128 == 0 else 64
            K = 256
            torch.manual_seed(42)
            A = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.01
            B = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.01

            ref = torch.mm(A, B.T)
            out = torch.empty(M, M, device=DEVICE, dtype=dtype)
            _call_syrk_raw(A, out, B=B, tile_m=tile_m, tile_k=32, num_stages=4)

            has_nan_full = torch.isnan(out).any().item()

            # Check lower triangle only
            mask = torch.tril(torch.ones(M, M, dtype=torch.bool, device=DEVICE))
            out_lower = out[mask].float()
            ref_lower = ref[mask].float()
            has_nan_lower = torch.isnan(out_lower).any().item()
            lower_diff = (out_lower - ref_lower).abs().max().item()

            # Check upper triangle only (mirror)
            umask = torch.triu(torch.ones(M, M, dtype=torch.bool, device=DEVICE), diagonal=1)
            out_upper = out[umask].float()
            ref_upper = ref[umask].float()
            has_nan_upper = torch.isnan(out_upper).any().item()
            upper_diff = (out_upper - ref_upper).abs().max().item()

            # Diagonal
            diag_diff = (out.float().diag() - ref.float().diag()).abs().max().item()

            print(f"  M={M} {dtype}: "
                  f"lower(NaN={has_nan_lower}, diff={lower_diff:.6f}) "
                  f"upper(NaN={has_nan_upper}, diff={upper_diff:.6f}) "
                  f"diag(diff={diag_diff:.6f})")


# ---------------------------------------------------------------------------
# Test 6: B!=A determinism (10 runs)
# ---------------------------------------------------------------------------

def test_bneqa_determinism():
    """Run B!=A kernel 10 times, check if results are bit-exact."""
    print("\n=== Test 6: B!=A determinism ===")
    N_RUNS = 10
    for dtype in [torch.float16]:
        for M in [128, 256, 512]:
            tile_m = 128 if M % 128 == 0 else 64
            K = 256
            torch.manual_seed(42)
            A = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.01
            B = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.01

            results = []
            for _ in range(N_RUNS):
                out = torch.empty(M, M, device=DEVICE, dtype=dtype)
                _call_syrk_raw(A, out, B=B, tile_m=tile_m, tile_k=32, num_stages=4)
                results.append(out.clone())

            # Check pairwise
            max_diff = 0
            for i in range(1, N_RUNS):
                d = (results[0].float() - results[i].float()).abs().max().item()
                max_diff = max(max_diff, d)

            all_equal = all(torch.equal(results[0], r) for r in results[1:])
            tag = "EXACT" if all_equal else f"DIFF={max_diff:.6f}"
            print(f"  M={M} {dtype}: {tag}")


# ---------------------------------------------------------------------------
# Test 7: B!=A with C matrix
# ---------------------------------------------------------------------------

def test_bneqa_with_C():
    """B!=A + C matrix — the most problematic pattern."""
    print("\n=== Test 7: B!=A with C matrix ===")
    for dtype in [torch.float16, torch.bfloat16]:
        for M in [128, 256, 512]:
            tile_m = 128 if M % 128 == 0 else 64
            K = 256
            torch.manual_seed(42)
            A = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.01
            B = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.01
            C = torch.randn(M, M, device=DEVICE, dtype=dtype) * 0.01

            ref = torch.addmm(C, A, B.T, alpha=0.5, beta=0.3)
            out = torch.empty(M, M, device=DEVICE, dtype=dtype)
            _call_syrk_raw(A, out, B=B, C=C, alpha=0.5, beta=0.3,
                           tile_m=tile_m, tile_k=32, num_stages=4)

            has_nan = torch.isnan(out).any().item()
            max_diff = (ref.float() - out.float()).abs().max().item()
            tril_diff = _tril_match(ref, out)
            print(f"  M={M} {dtype}: NaN={has_nan} "
                  f"max_diff={max_diff:.6f} tril_diff={tril_diff:.6f}")


# ---------------------------------------------------------------------------
# Test 8: skip_mirror isolation — compare lower triangle with/without mirror
# ---------------------------------------------------------------------------

def test_skip_mirror_isolation():
    """Run B!=A with skip_mirror=True (only lower triangle written).
    If lower triangle matches cuBLAS, the mainloop is correct and the bug
    is purely in the mirror epilogue."""
    print("\n=== Test 8: skip_mirror isolation ===")
    for dtype in [torch.float16]:
        for M in [128, 256, 512]:
            tile_m = 128 if M % 128 == 0 else 64
            K = 256
            torch.manual_seed(42)
            A = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.01
            B = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.01

            ref = torch.mm(A, B.T)

            # With mirror (default)
            out_mirror = torch.full((M, M), float('nan'), device=DEVICE, dtype=dtype)
            _call_syrk_raw(A, out_mirror, B=B, tile_m=tile_m, tile_k=32, num_stages=4,
                           skip_mirror=False)

            # Without mirror (lower triangle only)
            out_nomirror = torch.full((M, M), float('nan'), device=DEVICE, dtype=dtype)
            _call_syrk_raw(A, out_nomirror, B=B, tile_m=tile_m, tile_k=32, num_stages=4,
                           skip_mirror=True)

            mask = torch.tril(torch.ones(M, M, dtype=torch.bool, device=DEVICE))

            # Lower triangle: both should match ref
            mirror_lower_diff = (out_mirror[mask].float() - ref[mask].float()).abs().max().item()
            nomirror_lower_diff = (out_nomirror[mask].float() - ref[mask].float()).abs().max().item()
            mirror_lower_nan = torch.isnan(out_mirror[mask]).any().item()
            nomirror_lower_nan = torch.isnan(out_nomirror[mask]).any().item()

            # Upper triangle: nomirror should be NaN (untouched), mirror should be filled
            umask = torch.triu(torch.ones(M, M, dtype=torch.bool, device=DEVICE), diagonal=1)
            nomirror_upper_nan_pct = torch.isnan(out_nomirror[umask]).float().mean().item()
            mirror_upper_nan = torch.isnan(out_mirror[umask]).any().item()

            print(f"  M={M} {dtype}:")
            print(f"    mirror:   lower(NaN={mirror_lower_nan}, diff={mirror_lower_diff:.6f}) "
                  f"upper(NaN={mirror_upper_nan})")
            print(f"    nomirror: lower(NaN={nomirror_lower_nan}, diff={nomirror_lower_diff:.6f}) "
                  f"upper(NaN%={nomirror_upper_nan_pct:.1%})")

            if not nomirror_lower_nan and nomirror_lower_diff < 0.01:
                print("    --> DIAGNOSIS: mainloop OK, bug is in mirror epilogue")
            elif not nomirror_lower_nan:
                print(f"    --> DIAGNOSIS: mainloop has precision issues (diff={nomirror_lower_diff:.6f})")
            else:
                print("    --> DIAGNOSIS: mainloop produces NaN, bug is in tile indexing/MMA")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

ALL_TESTS = {
    "single_tile": test_single_tile,
    "two_tiles": test_two_tiles,
    "nan_threshold": test_nan_threshold,
    "symmetric_bneqa": test_symmetric_bneqa,
    "lower_triangle": test_lower_triangle_only,
    "determinism": test_bneqa_determinism,
    "bneqa_with_C": test_bneqa_with_C,
    "skip_mirror": test_skip_mirror_isolation,
}

if __name__ == "__main__":
    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    if test_name == "all":
        tests = list(ALL_TESTS.items())
    elif test_name in ALL_TESTS:
        tests = [(test_name, ALL_TESTS[test_name])]
    else:
        print(f"Unknown test: {test_name}. Available: {', '.join(ALL_TESTS.keys())}, all")
        sys.exit(1)

    print(f"SYRK B!=A Debug Tests (GPU: {torch.cuda.get_device_name(0)})")
    print("=" * 70)
    for name, fn in tests:
        fn()

    print("\n" + "=" * 70)
    print("Debug tests complete. Analyze output to identify root cause.")
    print("Key diagnostics:")
    print("  - If single_tile fails: bug is in mainloop (tile indexing or MMA)")
    print("  - If single_tile OK but two_tiles fails: bug is in off-diagonal mirror")
    print("  - If lower_triangle OK but upper wrong: mirror-write corruption")
    print("  - If lower_triangle also wrong: tile partitioning or MMA bug")
