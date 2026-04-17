"""Test matrix fixtures with seed-driven spectrum sampling.

Key design: each ``seed`` produces a **different σ spectrum**, not just
different orthogonal U/V. Because NS is orthogonally invariant, varying
only U/V gives zero algorithmic variance. Varying the spectrum is what
actually stress-tests an NS implementation.

Fixture grid: ``spectrum_kind × m × aspect_ratio``, expanded parametrically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Seed-driven spectrum samplers
# ---------------------------------------------------------------------------
def sample_uniform(m: int, *, seed: int, device: str = "cuda") -> Tensor:
    """σ_i ~ Uniform[low, high], sorted decreasing. Base + random jitter.

    Models well-conditioned matrices (κ ≤ ~3).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    low = 0.25 + 0.1 * torch.rand(1, device=device, generator=gen).item()  # [0.25, 0.35]
    high = 0.9 + 0.1 * torch.rand(1, device=device, generator=gen).item()  # [0.9, 1.0]
    base = torch.linspace(high, low, m, device=device, dtype=torch.float64)
    noise = 0.1 * (high - low) * (
        torch.rand(m, device=device, generator=gen, dtype=torch.float64) - 0.5
    )
    s, _ = (base + noise).clamp(min=0.05).sort(descending=True)
    return s


def sample_gaussian_random(m: int, *, seed: int, device: str = "cuda") -> Tensor:
    """Spectrum of a random Gaussian m × (2m) matrix (Marchenko–Pastur-ish).

    Models natural random matrix spectra — realistic for randomly initialized
    weights and noisy gradients.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    n_aux = 2 * m
    A = torch.randn(m, n_aux, device=device, dtype=torch.float64, generator=gen)
    s = torch.linalg.svdvals(A).sort(descending=True).values
    return s / s.max()  # normalize σ_max = 1


def sample_power_law(m: int, *, seed: int, device: str = "cuda") -> Tensor:
    """σ_i = i^(-α), α ∈ [0.3, 0.7] random.

    Models gradients of trained models (mildly decaying spectra).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    alpha = 0.3 + 0.4 * torch.rand(1, device=device, generator=gen).item()
    i = torch.arange(1, m + 1, device=device, dtype=torch.float64)
    s = i.pow(-alpha)
    return s / s.max()


def sample_exponential(m: int, *, seed: int, device: str = "cuda") -> Tensor:
    """σ_i = exp(-β i / m), β ∈ [2, 8] random.

    Ill-conditioned. Stress test for numerical stability.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    beta = 2.0 + 6.0 * torch.rand(1, device=device, generator=gen).item()
    i = torch.arange(m, device=device, dtype=torch.float64)
    s = torch.exp(-beta * i / m)
    return s


def sample_heavy_tail(m: int, *, seed: int, device: str = "cuda") -> Tensor:
    """Gaussian bulk + 10% heavy tail (near-zero σ).

    Models rank-deficient gradients after LR warmup.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    bulk = torch.rand(m, device=device, generator=gen, dtype=torch.float64)
    k = max(1, int(0.1 * m))
    bulk[-k:] *= 1e-3  # heavy tail
    s, _ = bulk.sort(descending=True)
    return s / s.max()


SPECTRUM_SAMPLERS: dict[str, Callable] = {
    "uniform": sample_uniform,
    "gaussian_random": sample_gaussian_random,
    "power_law": sample_power_law,
    "exponential": sample_exponential,
    "heavy_tail": sample_heavy_tail,
}


# ---------------------------------------------------------------------------
# Matrix assembly: spectrum + random orthogonal bases
# ---------------------------------------------------------------------------
def _qr_orthogonal(rows: int, cols: int, *, gen: torch.Generator, device) -> Tensor:
    """Random orthogonal columns via QR of Gaussian. Returns (rows, cols) with rows ≥ cols.

    Much faster than full SVD for large matrices.
    """
    assert rows >= cols
    G = torch.randn(rows, cols, device=device, dtype=torch.float64, generator=gen)
    Q, _ = torch.linalg.qr(G)
    return Q


def assemble_matrix(spectrum: Tensor, n: int, *, seed: int, device) -> Tensor:
    """Build m × n matrix A = U @ diag(σ) @ V^T in fp64 with prescribed σ.

    U: (m, m) random orthogonal via QR of random Gaussian.
    V_partial: (n, m) random column-orthogonal via QR.
    A = U @ diag(σ) @ V_partial^T  has shape (m, n) and rank m.
    """
    m = spectrum.shape[0]
    assert m <= n, f"short-fat only: m={m}, n={n}"
    gen = torch.Generator(device=device).manual_seed(seed + 9_999_991)
    U = _qr_orthogonal(m, m, gen=gen, device=device)
    V_partial = _qr_orthogonal(n, m, gen=gen, device=device)  # (n, m)
    Sigma = torch.diag(spectrum.to(torch.float64))  # (m, m)
    return (U @ Sigma @ V_partial.mT).contiguous()


# ---------------------------------------------------------------------------
# Fixture container + parametric grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fixture:
    kind: str
    m: int
    n: int
    seed: int

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.m}x{self.n}_s{self.seed}"

    @property
    def aspect(self) -> float:
        return self.n / self.m

    def build(self, device: str = "cuda") -> Tensor:
        sampler = SPECTRUM_SAMPLERS[self.kind]
        spec = sampler(self.m, seed=self.seed, device=device)
        return assemble_matrix(spec, self.n, seed=self.seed, device=device)

    def spectrum(self, device: str = "cuda") -> Tensor:
        return SPECTRUM_SAMPLERS[self.kind](self.m, seed=self.seed, device=device)


def fixture_grid(
    *,
    kinds: list[str] | None = None,
    sizes: list[int] | None = None,
    aspects: list[int] | None = None,
    n_seeds: int = 20,
) -> list[Fixture]:
    """Cross-product of {kinds} × {sizes} × {aspects} × range(n_seeds).

    Defaults cover a reasonable grid. Override for ablations.
    """
    if kinds is None:
        kinds = ["uniform", "gaussian_random", "power_law", "exponential", "heavy_tail"]
    if sizes is None:
        sizes = [512, 1024, 2048]
    if aspects is None:
        aspects = [1, 2, 4, 8]

    fixtures = []
    for kind in kinds:
        for m in sizes:
            for aspect in aspects:
                n = m * aspect
                for seed in range(n_seeds):
                    fixtures.append(Fixture(kind=kind, m=m, n=n, seed=seed))
    return fixtures


# Backward-compat default (used by older callsites)
def default_fixtures() -> list[Fixture]:
    """Moderate grid, ~240 fixtures. Tune for speed vs coverage."""
    return fixture_grid(
        kinds=["uniform", "gaussian_random", "power_law", "exponential"],
        sizes=[512, 1024],
        aspects=[1, 4],
        n_seeds=15,
    )
