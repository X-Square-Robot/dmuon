"""Tests for ``NewtonSchulz(kernel=...)`` + env var override (B4).

Covers the resolution priority:
    explicit kernel=  >  DMUON_NS_KERNEL  >  deterministic=  >  'auto'

Exercises NewtonSchulz kernel argument handling.
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

import dmuon
from dmuon.kernels.syrk_backends import SyrkBackend


def _clear_env():
    os.environ.pop("DMUON_NS_KERNEL", None)


@pytest.fixture(autouse=True)
def _isolate_env():
    """Every test sees a clean DMUON_NS_KERNEL env state."""
    _clear_env()
    yield
    _clear_env()


# ---------------------------------------------------------------------------
# Explicit kernel= kwarg
# ---------------------------------------------------------------------------
def test_explicit_cublas_sets_cublas():
    ns = dmuon.NewtonSchulz(kernel="cublas")
    assert ns.kernel == SyrkBackend.CUBLAS
    assert ns.deterministic is True  # derived


def test_explicit_auto_resolves_to_concrete_backend():
    ns = dmuon.NewtonSchulz(kernel="auto")
    assert ns.kernel != SyrkBackend.AUTO
    assert ns.kernel in {SyrkBackend.CUTE_SM80, SyrkBackend.CUBLAS, SyrkBackend.QUACK}


def test_default_kernel_is_auto_resolved():
    """No arg → same as kernel='auto'."""
    ns_default = dmuon.NewtonSchulz()
    ns_auto = dmuon.NewtonSchulz(kernel="auto")
    assert ns_default.kernel == ns_auto.kernel


def test_invalid_kernel_string_raises():
    with pytest.raises(ValueError, match="is not a valid SYRK backend"):
        dmuon.NewtonSchulz(kernel="not_a_backend")


def test_explicit_quack_on_unsupported_hardware_raises():
    """On hardware that can't run quack (SM80, or SM90+ without the
    soft dep / with circuit breaker tripped), kernel='quack' must fail
    fast at construction.

    We don't skip on SM90+: even there, the test covers the ``HAS_QUACK=False``
    path by mocking it.  This keeps the fail-fast contract exercised on
    every platform.
    """
    from dmuon.kernels import syrk_quack
    with mock.patch.object(syrk_quack, "HAS_QUACK", False), \
         mock.patch.object(syrk_quack, "_quack_gemm_symmetric", None):
        with pytest.raises(RuntimeError, match="quack"):
            dmuon.NewtonSchulz(kernel="quack")


# ---------------------------------------------------------------------------
# DMUON_NS_KERNEL env var
# ---------------------------------------------------------------------------
def test_env_var_applies_when_kernel_is_auto():
    os.environ["DMUON_NS_KERNEL"] = "cublas"
    ns = dmuon.NewtonSchulz()
    assert ns.kernel == SyrkBackend.CUBLAS


def test_explicit_kernel_overrides_env_var():
    """Env var must NOT override an explicit kernel= kwarg (only 'auto').

    Mocks ``_HAS_CUTE_SM80=True`` so the test runs hardware-independently
    (on B-card we don't have CuteDSL; this test is purely about the
    precedence rule).
    """
    from dmuon.kernels import syrk_backends
    os.environ["DMUON_NS_KERNEL"] = "cublas"
    with mock.patch.object(syrk_backends, "_HAS_CUTE_SM80", True):
        ns = dmuon.NewtonSchulz(kernel="cute_sm80")
    assert ns.kernel == SyrkBackend.CUTE_SM80


# ---------------------------------------------------------------------------
# deterministic= back-compat
# ---------------------------------------------------------------------------
def test_deterministic_true_alone_maps_to_cublas():
    ns = dmuon.NewtonSchulz(deterministic=True)
    assert ns.kernel == SyrkBackend.CUBLAS
    assert ns.deterministic is True


def test_deterministic_false_keeps_auto():
    ns = dmuon.NewtonSchulz(deterministic=False)
    assert ns.kernel in {SyrkBackend.CUTE_SM80, SyrkBackend.CUBLAS, SyrkBackend.QUACK}


def test_explicit_kernel_overrides_deterministic_with_warning(caplog):
    """kernel='cute_sm80' + deterministic=True → warn, honor kernel.

    Mocks ``_HAS_CUTE_SM80=True`` so this test is hardware-independent.
    """
    import logging
    from dmuon.kernels import syrk_backends
    with caplog.at_level(logging.WARNING, logger="dmuon.optim.newton_schulz"), \
         mock.patch.object(syrk_backends, "_HAS_CUTE_SM80", True):
        ns = dmuon.NewtonSchulz(kernel="cute_sm80", deterministic=True)
    assert ns.kernel == SyrkBackend.CUTE_SM80
    assert any("honouring explicit kernel" in rec.message for rec in caplog.records)


def test_explicit_cublas_plus_deterministic_no_warning(caplog):
    """kernel='cublas' + deterministic=True are consistent → no warning."""
    import logging
    with caplog.at_level(logging.WARNING, logger="dmuon.optim.newton_schulz"):
        ns = dmuon.NewtonSchulz(kernel="cublas", deterministic=True)
    assert ns.kernel == SyrkBackend.CUBLAS
    assert not any("honouring" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# __repr__ includes kernel
# ---------------------------------------------------------------------------
def test_repr_shows_kernel():
    ns = dmuon.NewtonSchulz(kernel="cublas")
    rep = repr(ns)
    assert "kernel='cublas'" in rep
    assert "backend='gram'" in rep
