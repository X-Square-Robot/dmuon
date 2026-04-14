"""Unit tests for Newton-Schulz algorithm correctness (single GPU)."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from dmuon.optim.newton_schulz import (
    gram_newton_schulz_local,
    _compiled_newton_schulz,
    DEFAULT_COEFFICIENTS,
    YOU_COEFFICIENTS,
    POLAR_EXPRESS_COEFFICIENTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_cuda():
    if not torch.cuda.is_available():
        print("SKIPPED (no CUDA)")
        return False
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_gram_ns_vs_compiled_ns():
    """gram_newton_schulz_local and _compiled_newton_schulz produce similar results."""
    if not _require_cuda():
        return
    torch.manual_seed(42)

    shapes = [(128, 512), (256, 1024), (64, 256)]
    for shape in shapes:
        G = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

        out_gram = gram_newton_schulz_local(G, coefficients=DEFAULT_COEFFICIENTS)
        out_compiled = _compiled_newton_schulz(G, DEFAULT_COEFFICIENTS)

        # Shape must match
        assert out_gram.shape == out_compiled.shape, (
            f"Shape mismatch for {shape}: gram={out_gram.shape}, compiled={out_compiled.shape}"
        )

        # Numerical closeness (compare in float32; dtypes may differ: gram returns
        # original dtype bf16, compiled uses .half() internally so returns fp16)
        diff = (out_gram.float() - out_compiled.float()).abs().max().item()
        assert diff < 0.1, (
            f"Max abs diff too large for {shape}: {diff:.4f} (limit 0.1)"
        )

    print("test_gram_ns_vs_compiled_ns PASSED")


def test_ns_transposed_matrix():
    """NS handles tall matrices (m > n) via internal transpose."""
    if not _require_cuda():
        return
    torch.manual_seed(42)

    G_tall = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
    out_tall = gram_newton_schulz_local(G_tall)

    # Output shape must match input shape
    assert out_tall.shape == (512, 128), (
        f"Expected shape (512, 128), got {out_tall.shape}"
    )

    # Output must not be all zeros
    norm_tall = out_tall.float().norm().item()
    assert norm_tall > 0.01, (
        f"Output is near-zero: norm={norm_tall:.6f}"
    )

    # Relationship with wide matrix: NS(G_tall) should be close to NS(G_tall.T).T
    G_wide = G_tall.T.contiguous()
    out_wide = gram_newton_schulz_local(G_wide)

    diff = (out_tall.float() - out_wide.T.float()).abs().max().item()
    assert diff < 0.1, (
        f"NS(tall) vs NS(wide).T max abs diff too large: {diff:.4f} (limit 0.1)"
    )

    print("test_ns_transposed_matrix PASSED")


def test_ns_output_properties():
    """NS output should be approximately orthogonal with no NaN values."""
    if not _require_cuda():
        return
    torch.manual_seed(42)

    G = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    U = gram_newton_schulz_local(G)

    # No NaN values
    assert not torch.isnan(U).any(), "Output contains NaN values"

    # Approximate orthogonality: U @ U^T close to identity
    U_f = U.float()
    gram = U_f @ U_f.T
    I = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    orth_diff = (gram - I).abs().max().item()
    assert orth_diff < 0.15, (
        f"Orthogonality violation: max |U@U^T - I| = {orth_diff:.4f} (limit 0.15)"
    )

    print("test_ns_output_properties PASSED")


def test_ns_different_coefficients():
    """Different coefficient sets both produce valid orthogonal outputs."""
    if not _require_cuda():
        return
    torch.manual_seed(42)

    G = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)

    for name, coeffs in [("YOU", YOU_COEFFICIENTS), ("POLAR_EXPRESS", POLAR_EXPRESS_COEFFICIENTS)]:
        U = gram_newton_schulz_local(G, coefficients=coeffs)

        # No NaN
        assert not torch.isnan(U).any(), (
            f"{name} coefficients produced NaN values"
        )

        # Not all zeros
        norm = U.float().norm().item()
        assert norm > 0.01, (
            f"{name} coefficients produced near-zero output: norm={norm:.6f}"
        )

        # Approximate orthogonality
        U_f = U.float()
        gram = U_f @ U_f.T
        I = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        orth_diff = (gram - I).abs().max().item()
        assert orth_diff < 0.15, (
            f"{name} orthogonality violation: max |U@U^T - I| = {orth_diff:.4f} (limit 0.15)"
        )

    print("test_ns_different_coefficients PASSED")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

ALL_TESTS = {
    "test_gram_ns_vs_compiled_ns": test_gram_ns_vs_compiled_ns,
    "test_ns_transposed_matrix": test_ns_transposed_matrix,
    "test_ns_output_properties": test_ns_output_properties,
    "test_ns_different_coefficients": test_ns_different_coefficients,
}

if __name__ == "__main__":
    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    if test_name == "all":
        failed = []
        for name, fn in ALL_TESTS.items():
            try:
                fn()
            except Exception as e:
                print(f"{name} FAILED: {e}")
                failed.append(name)
        if failed:
            print(f"\n{len(failed)}/{len(ALL_TESTS)} tests FAILED: {failed}")
            sys.exit(1)
        else:
            print(f"\nAll {len(ALL_TESTS)} tests PASSED")
    else:
        if test_name not in ALL_TESTS:
            print(f"Unknown test: {test_name}")
            print(f"Available: {list(ALL_TESTS.keys())}")
            sys.exit(1)
        try:
            ALL_TESTS[test_name]()
        except Exception as e:
            print(f"{test_name} FAILED: {e}")
            sys.exit(1)
