"""Tests for the public ``get_ns_backend`` / ``get_backend_status`` APIs.

Exercises the public Newton-Schulz backend inspection API.
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

import dmuon


def _sm() -> int:
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def test_get_ns_backend_returns_expected_prefix():
    """The one-liner must start with 'Gram NS · kernel=' and embed the
    current SM version, so a grep of the startup log unambiguously
    identifies the active kernel."""
    s = dmuon.get_ns_backend()
    assert s.startswith("Gram NS · kernel=")
    assert "SM" in s


@pytest.mark.skipif(_sm() >= 90, reason="A-card only; SM90+ legitimately selects quack")
def test_get_ns_backend_on_a800_is_cute_or_cublas():
    """On an A800 (SM80) with cute_sm80 available the auto choice is
    cute_sm80; without it we fall back to cublas — never quack."""
    s = dmuon.get_ns_backend()
    assert "kernel=cute_sm80" in s or "kernel=cublas" in s
    assert "kernel=quack" not in s


@pytest.mark.skipif(_sm() < 90, reason="SM90+ only; A-card selects cute_sm80 / cublas")
def test_get_ns_backend_on_sm90_plus_is_quack_or_cublas():
    """On SM90+ (H100 / B200 / B300) auto should pick quack when the
    adapter is ready, or cublas when quack isn't installed / the
    circuit breaker is tripped.  cute_sm80 is never the answer on
    SM90+ because CuteDSL SM80 kernels are SM-gated."""
    s = dmuon.get_ns_backend()
    assert "kernel=quack" in s or "kernel=cublas" in s
    assert "kernel=cute_sm80" not in s


def test_get_backend_status_shape():
    st = dmuon.get_backend_status()
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
    assert isinstance(st["cute_sm80_available"], bool)
    assert st["cublas_always_available"] is True


def test_get_backend_status_auto_choice_is_concrete():
    """``auto_choice`` must be a concrete backend (never 'auto' itself)
    so callers can treat it as ground truth for what kernel will run."""
    st = dmuon.get_backend_status()
    assert st["auto_choice"] in {"quack", "cute_sm80", "cublas"}
