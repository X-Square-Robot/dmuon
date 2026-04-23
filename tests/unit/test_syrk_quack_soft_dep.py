"""Soft-dependency contract tests for :mod:`dmuon.kernels.syrk_quack`.

Verifies the *dispatch-level* behaviour only (SM gating, import
detection, error messages).  Correctness of the actual adapter code is
covered in ``test_syrk_quack_adapter.py`` and only runs when the
environment can execute quack kernels.  See
``docs/internal/research/ns_backend_dispatch_plan.md`` §3 (B1) and §4 (B8).
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

from dmuon.kernels import syrk_quack


def test_is_supported_sm80_returns_false():
    """SM80 (A100 / A800) must never select quack — even with quack
    installed, the SM version gate blocks it."""
    assert syrk_quack.is_supported(80) is False
    assert syrk_quack.is_supported(87) is False


def test_is_supported_sm90_gated_on_adapter_ready_and_has_quack():
    """On SM90+, supportability is ``ADAPTER_READY AND HAS_QUACK``.  Both
    flags are runtime-tunable so this test is hardware-independent."""
    expected = syrk_quack.ADAPTER_READY and syrk_quack.HAS_QUACK
    assert syrk_quack.is_supported(90) is expected
    assert syrk_quack.is_supported(100) is expected


def test_is_supported_sm70_returns_false():
    """Pre-Ampere devices always fall through to cublas."""
    assert syrk_quack.is_supported(70) is False


def test_adapter_gate_flags_are_bool():
    """Both phase-gate flags must be plain bools — the rest of the
    dispatch layer uses them in ``and`` chains."""
    assert isinstance(syrk_quack.ADAPTER_READY, bool)
    assert isinstance(syrk_quack.HAS_QUACK, bool)


def test_adapter_ready_flip_disables_supportability():
    """When ``ADAPTER_READY`` is set to False (circuit breaker), even a
    fully-equipped SM90+ with quack installed must report unsupported."""
    with mock.patch.object(syrk_quack, "ADAPTER_READY", False):
        assert syrk_quack.is_supported(90) is False
        assert syrk_quack.is_supported(100) is False


def test_syrk_raises_when_quack_missing():
    """If HAS_QUACK is False (soft-dep not installed), syrk must raise
    a clear install-hint error, not a silent None-attribute crash."""
    with mock.patch.object(syrk_quack, "HAS_QUACK", False), \
         mock.patch.object(syrk_quack, "_quack_gemm_symmetric", None):
        A = torch.zeros(4, 4)
        D = torch.zeros(4, 4)
        with pytest.raises(RuntimeError, match="pip install dmuon\\[quack\\]"):
            syrk_quack.syrk(A, D)
