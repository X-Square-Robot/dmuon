"""SYRK kernel stress test: correctness across all shapes, modes, and dtypes.

cuBLAS is the ground truth. Tests verify SYRK produces matching results.

Run with: python tests/unit/test_syrk_stress.py
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

from dmuon.kernels.syrk_sm80 import syrk_sm80, SYRK_SM80_CONFIGS

DEVICE = torch.device("cuda")

# All M values that appear in real Transformers (must be divisible by tile_m)
# tile_m ∈ {64, 128}, so M must be multiple of 64
TRANSFORMER_M = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
# K values: d_model dimensions
TRANSFORMER_K = [128, 256, 512, 1024, 2048, 4096, 8192]
# Dtypes used in NS iteration
DTYPES = [torch.float16, torch.bfloat16]

# Relative tolerance per dtype
RTOL = {torch.float16: 0.02, torch.bfloat16: 0.02}


def _make_symmetric(T):
    return (T + T.T) / 2


def _check(name, ref, syrk_out, dtype, shape_str):
    """Compare SYRK output against cuBLAS reference."""
    diff = (ref.float() - syrk_out.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    if scale < 1e-6:
        scale = 1.0
    rdiff = diff / scale

    has_nan = torch.isnan(syrk_out).any().item()
    has_inf = torch.isinf(syrk_out).any().item()

    ok = not has_nan and not has_inf and rdiff < RTOL[dtype]
    if not ok:
        print(f"  FAIL {name} {shape_str} {dtype}: rdiff={rdiff:.6f} nan={has_nan} inf={has_inf}")
    return ok


def _find_config(M):
    """Find a valid tile config for this M."""
    for tile_m, tile_k, num_stages in SYRK_SM80_CONFIGS:
        if M % tile_m == 0:
            return tile_m, tile_k, num_stages
    return None


# ---------------------------------------------------------------------------
# Test 1: Basic SYRK (D = A @ A^T) across all shapes and dtypes
# ---------------------------------------------------------------------------

def test_basic_syrk():
    """D = A @ A^T for all Transformer shapes."""
    passed, failed = 0, 0
    for dtype in DTYPES:
        for M in TRANSFORMER_M:
            cfg = _find_config(M)
            if cfg is None:
                continue
            tile_m, tile_k, num_stages = cfg
            # Pick a few K values that are relevant for this M
            for K in [k for k in TRANSFORMER_K if k >= M // 4]:
                torch.manual_seed(42)
                A = torch.randn(M, K, device=DEVICE, dtype=dtype)
                ref = torch.mm(A, A.T)
                out = torch.empty(M, M, device=DEVICE, dtype=dtype)
                syrk_sm80(A, out, tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)

                if _check("basic", ref, out, dtype, f"({M},{K})"):
                    passed += 1
                else:
                    failed += 1

    print(f"test_basic_syrk: {passed} passed, {failed} failed")
    assert failed == 0, f"{failed} failures"


# ---------------------------------------------------------------------------
# Test 2: SYRK with C, alpha, beta (D = alpha * A @ A^T + beta * C)
# ---------------------------------------------------------------------------

def test_alpha_beta_C():
    """D = alpha * A @ A^T + beta * C for varied shapes."""
    passed, failed = 0, 0
    alphas_betas = [(0.5, 0.3), (1.0, 1.0), (2.0, -1.0), (0.1, 0.9)]

    for dtype in DTYPES:
        for M in [128, 256, 512, 1024]:
            cfg = _find_config(M)
            if cfg is None:
                continue
            tile_m, tile_k, num_stages = cfg
            K = max(M, 256)
            for alpha, beta in alphas_betas:
                torch.manual_seed(42)
                A = torch.randn(M, K, device=DEVICE, dtype=dtype)
                C = _make_symmetric(torch.randn(M, M, device=DEVICE, dtype=dtype))
                ref = torch.addmm(C, A, A.T, alpha=alpha, beta=beta)
                out = torch.empty(M, M, device=DEVICE, dtype=dtype)
                syrk_sm80(A, out, C=C, alpha=alpha, beta=beta,
                          tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)

                if _check(f"a={alpha},b={beta}", ref, out, dtype, f"({M},{K})"):
                    passed += 1
                else:
                    failed += 1

    print(f"test_alpha_beta_C: {passed} passed, {failed} failed")
    assert failed == 0, f"{failed} failures"


# ---------------------------------------------------------------------------
# Test 3: B != A guard — syrk_sm80 must reject B!=A with ValueError
# ---------------------------------------------------------------------------

def test_B_not_A_guard():
    """syrk_sm80 must reject B!=A without _symmetric flag, accept with it."""
    passed, failed = 0, 0
    dtype = torch.float16

    # B!=A without _symmetric → ValueError
    for M in [128, 256]:
        cfg = _find_config(M)
        if cfg is None:
            continue
        tile_m, tile_k, num_stages = cfg
        torch.manual_seed(42)
        A = torch.randn(M, M, device=DEVICE, dtype=dtype)
        B = torch.randn(M, M, device=DEVICE, dtype=dtype)
        out = torch.empty(M, M, device=DEVICE, dtype=dtype)

        try:
            syrk_sm80(A, out, B=B, tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)
            print(f"  FAIL B!=A guard ({M}): no exception raised")
            failed += 1
        except ValueError:
            passed += 1

    # B=None → true SYRK, always works
    for M in [128, 256]:
        cfg = _find_config(M)
        if cfg is None:
            continue
        tile_m, tile_k, num_stages = cfg
        torch.manual_seed(42)
        A = torch.randn(M, 512, device=DEVICE, dtype=dtype)
        out = torch.empty(M, M, device=DEVICE, dtype=dtype)
        try:
            syrk_sm80(A, out, tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)
            passed += 1
        except Exception as e:
            print(f"  FAIL B=None ({M}): unexpected {e}")
            failed += 1

    # B=A (same tensor) → works
    for M in [128]:
        cfg = _find_config(M)
        if cfg is None:
            continue
        tile_m, tile_k, num_stages = cfg
        torch.manual_seed(42)
        A = torch.randn(M, 512, device=DEVICE, dtype=dtype)
        out = torch.empty(M, M, device=DEVICE, dtype=dtype)
        try:
            syrk_sm80(A, out, B=A, tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)
            passed += 1
        except Exception as e:
            print(f"  FAIL B==A ({M}): unexpected {e}")
            failed += 1

    # B!=A with _symmetric=True → should work for symmetric results
    for M in [128, 256]:
        cfg = _find_config(M)
        if cfg is None:
            continue
        tile_m, tile_k, num_stages = cfg
        torch.manual_seed(42)
        X = torch.randn(M, M * 2, device=DEVICE, dtype=dtype) * 0.01
        R = torch.mm(X, X.T)
        Z = 0.5 * torch.mm(R, R) + 0.3 * R
        ref = torch.mm(R, Z.T)
        out = torch.empty(M, M, device=DEVICE, dtype=dtype)
        try:
            syrk_sm80(R, out, B=Z, _symmetric=True,
                      tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)
            if _check("_symmetric B!=A", ref, out, dtype, f"({M},{M})"):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAIL _symmetric B!=A ({M}): {e}")
            failed += 1

    print(f"test_B_not_A_guard: {passed} passed, {failed} failed")
    assert failed == 0, f"{failed} failures"


# ---------------------------------------------------------------------------
# Test 5: Determinism — same input N times, must be bit-exact
# ---------------------------------------------------------------------------

def test_determinism():
    """Run SYRK 10 times with identical input, verify bit-exact output."""
    passed, failed = 0, 0
    N_RUNS = 10

    for dtype in DTYPES:
        for M in [128, 256, 512, 1024]:
            cfg = _find_config(M)
            if cfg is None:
                continue
            tile_m, tile_k, num_stages = cfg
            K = M * 2
            torch.manual_seed(42)
            A = torch.randn(M, K, device=DEVICE, dtype=dtype)
            C = _make_symmetric(torch.randn(M, M, device=DEVICE, dtype=dtype))

            # Mode 1: basic SYRK
            baseline = torch.empty(M, M, device=DEVICE, dtype=dtype)
            syrk_sm80(A, baseline, tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)
            all_match = True
            for _ in range(N_RUNS - 1):
                out = torch.empty(M, M, device=DEVICE, dtype=dtype)
                syrk_sm80(A, out, tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)
                if not torch.equal(baseline, out):
                    all_match = False
                    diff = (baseline - out).abs().max().item()
                    print(f"  FAIL determinism basic ({M},{K}) {dtype}: max_diff={diff}")
                    break
            if all_match:
                passed += 1
            else:
                failed += 1

            # Mode 2: with C, alpha, beta
            baseline2 = torch.empty(M, M, device=DEVICE, dtype=dtype)
            syrk_sm80(A, baseline2, C=C, alpha=0.5, beta=0.3,
                      tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)
            all_match2 = True
            for _ in range(N_RUNS - 1):
                out2 = torch.empty(M, M, device=DEVICE, dtype=dtype)
                syrk_sm80(A, out2, C=C, alpha=0.5, beta=0.3,
                          tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)
                if not torch.equal(baseline2, out2):
                    all_match2 = False
                    diff = (baseline2 - out2).abs().max().item()
                    print(f"  FAIL determinism C+ab ({M},{K}) {dtype}: max_diff={diff}")
                    break
            if all_match2:
                passed += 1
            else:
                failed += 1

            # Mode 3: with diag_add
            baseline3 = torch.empty(M, M, device=DEVICE, dtype=dtype)
            syrk_sm80(A, baseline3, C=C, alpha=0.5, beta=0.3, diag_add=1.0,
                      tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)
            all_match3 = True
            for _ in range(N_RUNS - 1):
                out3 = torch.empty(M, M, device=DEVICE, dtype=dtype)
                syrk_sm80(A, out3, C=C, alpha=0.5, beta=0.3, diag_add=1.0,
                          tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)
                if not torch.equal(baseline3, out3):
                    all_match3 = False
                    diff = (baseline3 - out3).abs().max().item()
                    print(f"  FAIL determinism diag_add ({M},{K}) {dtype}: max_diff={diff}")
                    break
            if all_match3:
                passed += 1
            else:
                failed += 1

    print(f"test_determinism: {passed} passed, {failed} failed")
    assert failed == 0, f"{failed} failures"


# ---------------------------------------------------------------------------
# Test 6: NaN/Inf stability — extreme inputs should not produce NaN
# ---------------------------------------------------------------------------

def test_nan_stability():
    """SYRK should not produce NaN/Inf for various input magnitudes."""
    passed, failed = 0, 0
    dtype = torch.float16  # fp16 has smallest dynamic range

    for M in [128, 256]:
        cfg = _find_config(M)
        if cfg is None:
            continue
        tile_m, tile_k, num_stages = cfg
        K = 512

        for scale_name, scale in [
            ("tiny", 1e-4),
            ("small", 0.01),
            ("normal", 1.0),
            # scale=100 with K=512: A@A^T ~ 100^2*512 = 5.12M >> fp16 max 65504
            # Inf is expected (cuBLAS also overflows). Only test scales within fp16 range.
            ("moderate", 10.0),
        ]:
            torch.manual_seed(42)
            A = torch.randn(M, K, device=DEVICE, dtype=dtype) * scale
            out = torch.empty(M, M, device=DEVICE, dtype=dtype)
            syrk_sm80(A, out, tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)

            has_nan = torch.isnan(out).any().item()
            has_inf = torch.isinf(out).any().item()
            if has_nan or has_inf:
                print(f"  FAIL nan_stability ({M},{K}) scale={scale_name}: nan={has_nan} inf={has_inf}")
                failed += 1
            else:
                passed += 1

            # Also test with C + alpha + beta
            C = _make_symmetric(torch.randn(M, M, device=DEVICE, dtype=dtype) * scale)
            out2 = torch.empty(M, M, device=DEVICE, dtype=dtype)
            syrk_sm80(A, out2, C=C, alpha=0.5, beta=0.3,
                      tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)
            has_nan2 = torch.isnan(out2).any().item()
            has_inf2 = torch.isinf(out2).any().item()
            if has_nan2 or has_inf2:
                print(f"  FAIL nan_stability+C ({M},{K}) scale={scale_name}: nan={has_nan2} inf={has_inf2}")
                failed += 1
            else:
                passed += 1

    print(f"test_nan_stability: {passed} passed, {failed} failed")
    assert failed == 0, f"{failed} failures"


# ---------------------------------------------------------------------------
# Test 7: All tile configs — each config should produce correct results
# ---------------------------------------------------------------------------

def test_all_tile_configs():
    """Verify correctness for every tile config in SYRK_SM80_CONFIGS."""
    passed, failed = 0, 0
    dtype = torch.float16

    for tile_m, tile_k, num_stages in SYRK_SM80_CONFIGS:
        # Pick smallest M that works
        M = tile_m
        K = 256
        torch.manual_seed(42)
        A = torch.randn(M, K, device=DEVICE, dtype=dtype)
        C = _make_symmetric(torch.randn(M, M, device=DEVICE, dtype=dtype))

        # Basic
        ref = torch.mm(A, A.T)
        out = torch.empty(M, M, device=DEVICE, dtype=dtype)
        try:
            syrk_sm80(A, out, tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)
            if _check(f"cfg({tile_m},{tile_k},{num_stages})", ref, out, dtype, f"({M},{K})"):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAIL cfg({tile_m},{tile_k},{num_stages}) basic: {e}")
            failed += 1

        # With C
        ref2 = torch.addmm(C, A, A.T, alpha=0.5, beta=0.3)
        out2 = torch.empty(M, M, device=DEVICE, dtype=dtype)
        try:
            syrk_sm80(A, out2, C=C, alpha=0.5, beta=0.3,
                      tile_m=tile_m, tile_k=tile_k, num_stages=num_stages)
            if _check(f"cfg({tile_m},{tile_k},{num_stages})+C", ref2, out2, dtype, f"({M},{K})"):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAIL cfg({tile_m},{tile_k},{num_stages}) with C: {e}")
            failed += 1

    print(f"test_all_tile_configs: {passed} passed, {failed} failed")
    assert failed == 0, f"{failed} failures"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

ALL_TESTS = {
    "basic": test_basic_syrk,
    "alpha_beta_C": test_alpha_beta_C,
    "B_not_A_guard": test_B_not_A_guard,
    "determinism": test_determinism,
    "nan_stability": test_nan_stability,
    "all_configs": test_all_tile_configs,
}

if __name__ == "__main__":
    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    if test_name == "all":
        tests_to_run = list(ALL_TESTS.items())
    elif test_name in ALL_TESTS:
        tests_to_run = [(test_name, ALL_TESTS[test_name])]
    else:
        print(f"Unknown test: {test_name}. Available: {', '.join(ALL_TESTS.keys())}, all")
        sys.exit(1)

    total_passed, total_failed = 0, 0
    for name, fn in tests_to_run:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        try:
            fn()
            total_passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            total_failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            total_failed += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_passed} test groups passed, {total_failed} failed")
    if total_failed > 0:
        sys.exit(1)
