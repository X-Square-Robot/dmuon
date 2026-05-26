"""Unit tests for the unified SYRK backend dispatch layer.

Exercises unified SYRK backend dispatch.
"""

from __future__ import annotations

import os
import sys
from unittest import mock

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import pytest
import torch

from dmuon.kernels import syrk_backends
from dmuon.kernels.syrk_backends import SyrkBackend


# ---------------------------------------------------------------------------
# detect_best_backend — per-SM-version decision table
# ---------------------------------------------------------------------------
def test_detect_sm80_with_cute_picks_cute():
    with mock.patch.object(syrk_backends, "_HAS_CUTE_SM80", True), \
         mock.patch.object(syrk_backends.syrk_quack, "HAS_QUACK", False):
        assert syrk_backends.detect_best_backend(80) == SyrkBackend.CUTE_SM80


def test_detect_sm80_no_cute_falls_back_to_cublas():
    with mock.patch.object(syrk_backends, "_HAS_CUTE_SM80", False):
        assert syrk_backends.detect_best_backend(80) == SyrkBackend.CUBLAS


def test_detect_sm90_with_quack_picks_quack():
    """Only when ADAPTER_READY is flipped (Phase B-H) does auto land on quack."""
    with mock.patch.object(syrk_backends.syrk_quack, "HAS_QUACK", True), \
         mock.patch.object(syrk_backends.syrk_quack, "ADAPTER_READY", True):
        assert syrk_backends.detect_best_backend(90) == SyrkBackend.QUACK


def test_detect_sm90_quack_installed_but_adapter_not_ready_falls_back():
    """Phase B-A reality: quack pip-installed on H100 but adapter stub — we
    MUST fall back to cublas, not route to the shell's NotImplementedError."""
    with mock.patch.object(syrk_backends.syrk_quack, "HAS_QUACK", True), \
         mock.patch.object(syrk_backends.syrk_quack, "ADAPTER_READY", False):
        assert syrk_backends.detect_best_backend(90) == SyrkBackend.CUBLAS


def test_detect_sm90_without_quack_falls_back_to_cublas():
    with mock.patch.object(syrk_backends.syrk_quack, "HAS_QUACK", False):
        assert syrk_backends.detect_best_backend(90) == SyrkBackend.CUBLAS
        assert syrk_backends.detect_best_backend(100) == SyrkBackend.CUBLAS


def test_detect_sm70_always_cublas():
    assert syrk_backends.detect_best_backend(70) == SyrkBackend.CUBLAS


# ---------------------------------------------------------------------------
# resolve_backend — turns AUTO into a concrete choice; errors on unsupported
# ---------------------------------------------------------------------------
def test_resolve_auto_returns_concrete():
    result = syrk_backends.resolve_backend(SyrkBackend.AUTO)
    assert result != SyrkBackend.AUTO
    assert result in SyrkBackend


def test_resolve_quack_raises_on_sm80_with_sm_hint():
    """On SM80 the hint should point the user to cute_sm80 / auto."""
    with mock.patch.object(syrk_backends, "_SM_VERSION", 80), \
         mock.patch.object(syrk_backends.syrk_quack, "HAS_QUACK", False):
        with pytest.raises(RuntimeError, match="SM90\\+"):
            syrk_backends.resolve_backend(SyrkBackend.QUACK)


def test_resolve_quack_raises_when_quack_not_installed():
    """On SM90 without quack, the hint should point at pip install."""
    with mock.patch.object(syrk_backends, "_SM_VERSION", 90), \
         mock.patch.object(syrk_backends.syrk_quack, "HAS_QUACK", False):
        with pytest.raises(RuntimeError, match="pip install dmuon\\[quack\\]"):
            syrk_backends.resolve_backend(SyrkBackend.QUACK)


def test_resolve_quack_raises_when_circuit_breaker_tripped():
    """When ``ADAPTER_READY=False`` (circuit breaker, e.g. emergency
    disable of a quack-kernels regression), the error message must
    point at the flag so operators know how to re-enable."""
    with mock.patch.object(syrk_backends, "_SM_VERSION", 90), \
         mock.patch.object(syrk_backends.syrk_quack, "HAS_QUACK", True), \
         mock.patch.object(syrk_backends.syrk_quack, "ADAPTER_READY", False):
        with pytest.raises(RuntimeError, match="ADAPTER_READY"):
            syrk_backends.resolve_backend(SyrkBackend.QUACK)


def test_resolve_cute_sm80_raises_when_missing():
    with mock.patch.object(syrk_backends, "_HAS_CUTE_SM80", False):
        with pytest.raises(RuntimeError, match="CuteDSL"):
            syrk_backends.resolve_backend(SyrkBackend.CUTE_SM80)


def test_resolve_compile_not_implemented():
    with pytest.raises(NotImplementedError, match="post-MVP"):
        syrk_backends.resolve_backend(SyrkBackend.COMPILE)


def test_resolve_cublas_always_works():
    assert syrk_backends.resolve_backend(SyrkBackend.CUBLAS) == SyrkBackend.CUBLAS


# ---------------------------------------------------------------------------
# get_backend_status — shape of the returned dict
# ---------------------------------------------------------------------------
def test_backend_status_keys():
    st = syrk_backends.get_backend_status()
    required = {
        "sm_version",
        "auto_choice",
        "quack_available",
        "cute_sm80_available",
        "cublas_always_available",
    }
    assert required <= set(st.keys())
    assert isinstance(st["sm_version"], int)
    assert isinstance(st["quack_available"], bool)
    assert st["cublas_always_available"] is True


# ---------------------------------------------------------------------------
# syrk_dispatch — cublas path produces the mathematical result
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dispatch_cublas_true_syrk_matches_manual():
    """cublas path, B=None (true SYRK): D = alpha · A @ Aᵀ."""
    torch.manual_seed(0)
    A = torch.randn(32, 16, device="cuda", dtype=torch.float32)
    D = torch.empty(32, 32, device="cuda", dtype=torch.float32)
    syrk_backends.syrk_dispatch(
        A, D, alpha=0.5, beta=0.0, backend=SyrkBackend.CUBLAS
    )
    torch.cuda.synchronize()
    expected = 0.5 * (A @ A.T)
    torch.testing.assert_close(D, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dispatch_cublas_with_C_and_diag_add():
    """cublas path with C and diag_add applied in-place."""
    torch.manual_seed(0)
    A = torch.randn(16, 8, device="cuda", dtype=torch.float32)
    C = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    D = torch.empty_like(C)
    syrk_backends.syrk_dispatch(
        A, D, C=C, alpha=1.0, beta=2.0, diag_add=3.0,
        backend=SyrkBackend.CUBLAS,
    )
    torch.cuda.synchronize()
    expected = A @ A.T + 2.0 * C
    expected.diagonal().add_(3.0)
    torch.testing.assert_close(D, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# resolve_env_kernel — env-var override parsing
# ---------------------------------------------------------------------------
def test_env_kernel_unset_returns_none():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DMUON_NS_KERNEL", None)
        assert syrk_backends.resolve_env_kernel() is None


def test_env_kernel_valid_value_parsed():
    with mock.patch.dict(os.environ, {"DMUON_NS_KERNEL": "cublas"}):
        assert syrk_backends.resolve_env_kernel() == SyrkBackend.CUBLAS


def test_env_kernel_invalid_raises():
    with mock.patch.dict(os.environ, {"DMUON_NS_KERNEL": "bogus"}):
        with pytest.raises(ValueError, match="not a valid backend"):
            syrk_backends.resolve_env_kernel()
