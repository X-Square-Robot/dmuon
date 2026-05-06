"""Unit tests for SYRK kernel correctness (single GPU, no dist required).

The SYRK kernel computes only the lower triangle and mirrors to upper triangle.
This means:
  - C must be symmetric (only lower triangle is read)
  - B!=A is only valid when A@B^T is symmetric (as in Gram NS intermediate products)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from dmuon.optim.syrk_dispatch import HAS_SYRK

if not HAS_SYRK:
    print("SKIP: CuteDSL SYRK kernel not available (missing dependencies or unsupported GPU)")
    sys.exit(0)

if not torch.cuda.is_available():
    print("SKIP: CUDA not available")
    sys.exit(0)

from dmuon.kernels.syrk_sm80 import syrk_sm80
from dmuon.optim.syrk_dispatch import _syrk_autotune_cache, _autotune_syrk

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def _make_symmetric(M_tensor):
    """Make a matrix symmetric: (M + M^T) / 2."""
    return (M_tensor + M_tensor.T) / 2


# ---------------------------------------------------------------------------
# Test 1: basic SYRK vs cuBLAS (A @ A^T)
# ---------------------------------------------------------------------------

def test_syrk_vs_cublas_basic():
    torch.manual_seed(42)
    tile_m, tile_k, num_stages = 128, 32, 4
    shapes = [(128, 256), (256, 256), (128, 1024), (256, 1024)]

    for M, K in shapes:
        if M % tile_m != 0:
            print(f"  [test_syrk_vs_cublas_basic] skipping ({M},{K}): M%tile_m!=0")
            continue

        A = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
        D_ref = torch.mm(A, A.T)
        D_syrk = torch.empty(M, M, device=DEVICE, dtype=DTYPE)
        syrk_sm80(A, D_syrk, tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)

        diff = (D_ref.float() - D_syrk.float()).abs().max().item()
        # bf16 accumulation can diverge; use relative tolerance
        scale = D_ref.float().abs().max().item()
        rel_tol = 0.01  # 1% relative
        abs_tol = max(0.5, scale * rel_tol)
        assert diff < abs_tol, (
            f"test_syrk_vs_cublas_basic FAILED for shape ({M},{K}): "
            f"max_abs_diff={diff:.4f} >= {abs_tol:.4f}"
        )
        print(f"  [test_syrk_vs_cublas_basic] shape ({M},{K}) max_abs_diff={diff:.4f} OK")


# ---------------------------------------------------------------------------
# Test 2: SYRK with C, alpha, beta (C must be symmetric)
# ---------------------------------------------------------------------------

def test_syrk_with_C_alpha_beta():
    torch.manual_seed(42)
    M, K = 256, 512
    alpha, beta = 0.5, 0.3

    A = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    C = _make_symmetric(torch.randn(M, M, device=DEVICE, dtype=DTYPE))

    D_ref = torch.addmm(C, A, A.T, alpha=alpha, beta=beta)

    D_syrk = torch.empty(M, M, device=DEVICE, dtype=DTYPE)
    syrk_sm80(A, D_syrk, C=C, alpha=alpha, beta=beta,
              tile_m=128, tile_k=32, num_stages=4)

    diff = (D_ref.float() - D_syrk.float()).abs().max().item()
    scale = D_ref.float().abs().max().item()
    abs_tol = max(0.5, scale * 0.01)
    assert diff < abs_tol, (
        f"test_syrk_with_C_alpha_beta FAILED: max_abs_diff={diff:.4f} >= {abs_tol:.4f}"
    )
    print(f"  [test_syrk_with_C_alpha_beta] max_abs_diff={diff:.4f} OK")


# ---------------------------------------------------------------------------
# Test 3: SYRK with diag_add (C must be symmetric)
# ---------------------------------------------------------------------------

def test_syrk_diag_add():
    torch.manual_seed(42)
    M, K = 256, 512
    alpha, beta = 1.0, 1.0

    A = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    C = _make_symmetric(torch.randn(M, M, device=DEVICE, dtype=DTYPE))

    for diag_val in [0.0, 1.0, -3.5]:
        D_ref = alpha * torch.mm(A, A.T) + beta * C
        D_ref.diagonal().add_(diag_val)

        D_syrk = torch.empty(M, M, device=DEVICE, dtype=DTYPE)
        syrk_sm80(A, D_syrk, C=C, alpha=alpha, beta=beta, diag_add=diag_val,
                  tile_m=128, tile_k=32, num_stages=4)

        diff = (D_ref.float() - D_syrk.float()).abs().max().item()
        scale = D_ref.float().abs().max().item()
        abs_tol = max(0.5, scale * 0.01)
        assert diff < abs_tol, (
            f"test_syrk_diag_add FAILED for diag_val={diag_val}: "
            f"max_abs_diff={diff:.4f} >= {abs_tol:.4f}"
        )
        print(f"  [test_syrk_diag_add] diag_val={diag_val} max_abs_diff={diff:.4f} OK")

    # Verify diag_add=0 gives same result as no-diag_add
    D_no_diag = torch.empty(M, M, device=DEVICE, dtype=DTYPE)
    syrk_sm80(A, D_no_diag, C=C, alpha=alpha, beta=beta,
              tile_m=128, tile_k=32, num_stages=4)
    D_with_zero = torch.empty(M, M, device=DEVICE, dtype=DTYPE)
    syrk_sm80(A, D_with_zero, C=C, alpha=alpha, beta=beta, diag_add=0.0,
              tile_m=128, tile_k=32, num_stages=4)
    diff_zero = (D_no_diag.float() - D_with_zero.float()).abs().max().item()
    assert diff_zero == 0.0, (
        f"test_syrk_diag_add FAILED: diag_add=0.0 differs from no-diag_add, "
        f"max_abs_diff={diff_zero:.4f}"
    )
    print(f"  [test_syrk_diag_add] diag_add=0.0 == no-diag: diff={diff_zero:.4f} OK")


# ---------------------------------------------------------------------------
# Test 4: SYRK with B != A (symmetric inputs: B@A^T must be symmetric)
# ---------------------------------------------------------------------------

def test_syrk_B_not_A():
    """SYRK with B!=A only valid when result is symmetric.

    In Gram NS, this is used for products of symmetric matrices (e.g., Q@Z, R@Z).
    We simulate this by constructing symmetric R and Z, then computing R@Z^T = R@Z.
    """
    torch.manual_seed(42)
    M = 256

    # Construct R and Z where Z is a polynomial of R (they commute, so R@Z is symmetric).
    # This mirrors Gram NS where Z = c*R^2 + b*R.
    # Use small values to keep absolute differences meaningful.
    X = torch.randn(M, 128, device=DEVICE, dtype=DTYPE) * 0.01
    R = torch.mm(X, X.T)  # symmetric Gram matrix, values ~O(0.01)
    Z = 0.5 * torch.mm(R, R) + 0.3 * R  # polynomial of R

    D_ref = torch.mm(R, Z.T)  # R @ Z^T = R @ Z (since Z is symmetric)
    D_syrk = torch.empty(M, M, device=DEVICE, dtype=DTYPE)
    syrk_sm80(R, D_syrk, B=Z, _symmetric=True,
              tile_m=128, tile_k=32, num_stages=4)

    diff = (D_ref.float() - D_syrk.float()).abs().max().item()
    scale = D_ref.float().abs().max().item()
    rel_diff = diff / (scale + 1e-8)
    # bf16 relative tolerance: 1%
    assert rel_diff < 0.01, (
        f"test_syrk_B_not_A FAILED: max_abs_diff={diff:.6f}, scale={scale:.6f}, "
        f"rel_diff={rel_diff:.4f} >= 0.01"
    )
    print(f"  [test_syrk_B_not_A] max_abs_diff={diff:.6f}, rel_diff={rel_diff:.4f} OK")


# ---------------------------------------------------------------------------
# Test 5: autotune cache behaviour
# ---------------------------------------------------------------------------

def test_autotune_cache():
    torch.manual_seed(42)
    device = DEVICE
    dtype = DTYPE

    # Clear cache
    _syrk_autotune_cache.clear()

    # First call for (128, 256)
    result1 = _autotune_syrk(128, 256, device, dtype, has_C=False)
    # Second call — must return same result (cache hit)
    result2 = _autotune_syrk(128, 256, device, dtype, has_C=False)
    assert result1 == result2, (
        f"test_autotune_cache FAILED: repeated call returned different result: "
        f"{result1} vs {result2}"
    )
    print(f"  [test_autotune_cache] repeated (128,256) gives same result: {result1} OK")

    # Different shape — new key
    _autotune_syrk(256, 256, device, dtype, has_C=False)
    assert len(_syrk_autotune_cache) == 2, (
        f"test_autotune_cache FAILED: expected 2 cache entries, got {len(_syrk_autotune_cache)}"
    )
    print(f"  [test_autotune_cache] cache has 2 entries after second shape OK")

    # Same shape (128,256) but has_C=True — should be a different key
    _autotune_syrk(128, 256, device, dtype, has_C=True)
    assert len(_syrk_autotune_cache) == 3, (
        f"test_autotune_cache FAILED: expected 3 cache entries, got {len(_syrk_autotune_cache)}"
    )
    print(f"  [test_autotune_cache] cache has 3 entries after has_C=True variant OK")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

ALL_TESTS = {
    "test_syrk_vs_cublas_basic": test_syrk_vs_cublas_basic,
    "test_syrk_with_C_alpha_beta": test_syrk_with_C_alpha_beta,
    "test_syrk_diag_add": test_syrk_diag_add,
    "test_syrk_B_not_A": test_syrk_B_not_A,
    "test_autotune_cache": test_autotune_cache,
}

if __name__ == "__main__":
    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    if test_name == "all":
        tests_to_run = list(ALL_TESTS.items())
    elif test_name in ALL_TESTS:
        tests_to_run = [(test_name, ALL_TESTS[test_name])]
    else:
        print(f"Unknown test: {test_name}")
        print(f"Available: {', '.join(ALL_TESTS.keys())}, all")
        sys.exit(1)

    passed, failed = 0, 0
    for name, fn in tests_to_run:
        try:
            fn()
            print(f"PASSED  {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAILED  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"FAILED  {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    sys.exit(1 if failed else 0)
