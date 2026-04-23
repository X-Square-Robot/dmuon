"""Correctness + contract tests for the real quack→DMuon SYRK adapter.

These tests run only when ``dmuon.kernels.syrk_quack.is_supported()`` is
True — i.e. CUDA is available, a compatible quack version is installed,
and the device is SM90+.  Every other environment skips gracefully.

See ``docs/internal/benchmarks/quack_smoke_b300.md`` for the full B7/B8
probe; this file is the reproducible CI-friendly subset.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import pytest
import torch

from dmuon.kernels import syrk_quack


def _sm_version() -> int:
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


pytestmark = pytest.mark.skipif(
    not syrk_quack.is_supported(_sm_version()),
    reason=(
        "quack adapter requires SM90+, quack-kernels installed, and "
        "ADAPTER_READY=True"
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _reference(A, B_or_none, C, alpha, beta, diag_add):
    """CPU-precision reference for D = α · A @ Bᵀ + β · C + diag_add · I."""
    A32 = A.float()
    B32 = (A if B_or_none is None else B_or_none).float()
    D = alpha * (A32 @ B32.T)
    if C is not None:
        D = D + beta * C.float()
    if diag_add != 0.0:
        D = D.clone()
        D.diagonal().add_(diag_add)
    return D


def _random_sym(M, dtype, device):
    raw = torch.randn(M, M, device=device, dtype=dtype)
    return ((raw + raw.T) / 2).contiguous()


# ---------------------------------------------------------------------------
# Correctness matrix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,K", [(256, 128), (1024, 256), (2048, 512), (4096, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_true_syrk_matches_reference(M, K, dtype):
    """Core path: ``B is None`` (true SYRK), no C, no diag_add."""
    torch.manual_seed(42)
    A = torch.randn(M, K, device="cuda", dtype=dtype).contiguous()
    D = torch.empty(M, M, device="cuda", dtype=dtype)
    syrk_quack.syrk(A, D)
    torch.cuda.synchronize()

    ref = _reference(A, None, None, 1.0, 1.0, 0.0)
    rel = (D.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    tol = 5e-3 if dtype == torch.bfloat16 else 5e-4
    assert rel < tol, f"M={M} K={K} {dtype}: rel={rel:.3e} > tol={tol}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_with_C_alpha_beta(dtype):
    """addmm-style path: explicit C with alpha/beta scaling."""
    torch.manual_seed(7)
    M, K = 1024, 256
    A = torch.randn(M, K, device="cuda", dtype=dtype).contiguous()
    C = _random_sym(M, dtype, "cuda")
    D = torch.empty(M, M, device="cuda", dtype=dtype)
    syrk_quack.syrk(A, D, C=C, alpha=0.5, beta=0.3)
    torch.cuda.synchronize()
    ref = _reference(A, None, C, 0.5, 0.3, 0.0)
    rel = (D.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    tol = 5e-3 if dtype == torch.bfloat16 else 5e-4
    assert rel < tol


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_diag_add_post_process(dtype):
    """``diag_add`` must be applied on top of the quack output."""
    torch.manual_seed(3)
    M, K = 512, 128
    A = torch.randn(M, K, device="cuda", dtype=dtype).contiguous()
    D = torch.empty(M, M, device="cuda", dtype=dtype)
    syrk_quack.syrk(A, D, diag_add=0.25)
    torch.cuda.synchronize()
    ref = _reference(A, None, None, 1.0, 1.0, 0.25)
    rel = (D.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    tol = 5e-3 if dtype == torch.bfloat16 else 5e-4
    assert rel < tol


def test_explicit_B_routes_transpose():
    """When caller passes B != A, adapter still transposes correctly."""
    torch.manual_seed(11)
    M, K = 512, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).contiguous()
    # Construct B as a polynomial of A.@A.T so A @ B.T is symmetric
    X = A @ A.T
    B_coeff = (X + X.T) / 2  # symmetric; A @ B.T where B is derived
    # For the adapter test we need a B with shape (M, K) — derive from A
    B = (A * 1.1).contiguous()   # any (M, K) tensor; caller's contract
    # A @ B.T is generally non-symmetric; skip full assertion — we only
    # verify the adapter's transpose wiring produces A @ B.T, not A @ B.
    D = torch.empty(M, M, device="cuda", dtype=torch.bfloat16)
    syrk_quack.syrk(A, D, B=B)
    torch.cuda.synchronize()
    ref_AB = A.float() @ B.float().T
    # Adapter routes to quack's A @ B_quack with B_quack=B.T → A @ (B.T).T = A @ B
    # Wait — quack computes A @ B_quack = A @ B.T (correct by our mapping)
    # So D ≈ A @ B.T
    rel = (D.float() - ref_AB).abs().max().item() / max(ref_AB.abs().max().item(), 1e-9)
    # B != A → quack symmetric path reads only lower triangle; the
    # symmetric-output contract means upper triangle is mirrored from
    # lower.  For a non-symmetric A@B.T result, the lower triangle
    # should match the reference; assert on the lower triangle.
    lower_err = (D.float() - ref_AB).tril().abs().max().item()
    lower_rel = lower_err / max(ref_AB.abs().max().item(), 1e-9)
    assert lower_rel < 5e-3, (
        f"Transpose wiring wrong: lower-triangle rel={lower_rel:.3e}"
    )


# ---------------------------------------------------------------------------
# Edge cases from B8.0 open-question probes
# ---------------------------------------------------------------------------
def test_alpha_zero_yields_scaled_C():
    """O4 edge: alpha=0 → D = β · C (no matmul contribution)."""
    torch.manual_seed(5)
    M, K = 512, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).contiguous()
    C = _random_sym(M, torch.bfloat16, "cuda")
    D = torch.empty(M, M, device="cuda", dtype=torch.bfloat16)
    syrk_quack.syrk(A, D, C=C, alpha=0.0, beta=0.7)
    torch.cuda.synchronize()
    ref = 0.7 * C.float()
    rel = (D.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    assert rel < 5e-3


def test_beta_zero_with_C_ignores_C():
    """O4 edge: beta=0 with C present → D = α · A @ Aᵀ (C dropped)."""
    torch.manual_seed(13)
    M, K = 512, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).contiguous()
    C = _random_sym(M, torch.bfloat16, "cuda")
    D = torch.empty(M, M, device="cuda", dtype=torch.bfloat16)
    syrk_quack.syrk(A, D, C=C, alpha=1.0, beta=0.0)
    torch.cuda.synchronize()
    ref = A.float() @ A.float().T
    rel = (D.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    assert rel < 5e-3


def test_non_aligned_M_supported():
    """Non-power-of-two M must work (B7 observation)."""
    torch.manual_seed(17)
    for M in (1000, 1536, 3000):
        A = torch.randn(M, 256, device="cuda", dtype=torch.bfloat16).contiguous()
        D = torch.empty(M, M, device="cuda", dtype=torch.bfloat16)
        syrk_quack.syrk(A, D)
        torch.cuda.synchronize()
        ref = _reference(A, None, None, 1.0, 1.0, 0.0)
        rel = (D.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
        assert rel < 5e-3, f"non-aligned M={M}: rel={rel:.3e}"


# ---------------------------------------------------------------------------
# Output symmetry contract (B7 observation §3.3)
# ---------------------------------------------------------------------------
def test_output_is_fully_symmetric():
    """quack writes both triangles, unlike cute_sm80 which mirrors lower.
    Assert that the adapter preserves this property so downstream Gram
    iterations don't need to symmetrise the result."""
    torch.manual_seed(19)
    M, K = 2048, 512
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).contiguous()
    D = torch.empty(M, M, device="cuda", dtype=torch.bfloat16)
    syrk_quack.syrk(A, D)
    torch.cuda.synchronize()
    sym_err = (D.triu(1) - D.tril(-1).T).abs().max().item()
    assert sym_err == 0.0, f"quack output asymmetric: sym_err={sym_err}"


# ---------------------------------------------------------------------------
# Guard: unsupported env must raise clear error
# ---------------------------------------------------------------------------
def test_raises_when_quack_missing(monkeypatch):
    """If someone forces an import when HAS_QUACK=False, syrk must raise a
    clear install-hint error (not a silent ``None`` attribute crash)."""
    monkeypatch.setattr(syrk_quack, "HAS_QUACK", False)
    monkeypatch.setattr(syrk_quack, "_quack_gemm_symmetric", None)
    A = torch.zeros(4, 4, device="cuda", dtype=torch.bfloat16)
    D = torch.zeros(4, 4, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="pip install dmuon\\[quack\\]"):
        syrk_quack.syrk(A, D)
