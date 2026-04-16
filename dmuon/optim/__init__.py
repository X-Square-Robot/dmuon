"""DMuon optimizer package."""

from .muon import Muon
from .newton_schulz import (
    POLAR_EXPRESS_COEFFICIENTS,
    YOU_COEFFICIENTS,
    NewtonSchulz,
    direct_newton_schulz,
    gram_newton_schulz,
    gram_newton_schulz_local,
    newton_schulz,
)
from .syrk_dispatch import get_ns_backend

__all__ = [
    "Muon",
    "NewtonSchulz",
    "newton_schulz",
    "direct_newton_schulz",
    "gram_newton_schulz",
    "gram_newton_schulz_local",
    "get_ns_backend",
    "YOU_COEFFICIENTS",
    "POLAR_EXPRESS_COEFFICIENTS",
]
