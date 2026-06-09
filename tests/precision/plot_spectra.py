"""Visualize the spectrum distributions used in fixtures.

Shows what each spectrum kind "looks like" so the report reader can connect
fixture names to matrix properties.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from tests.precision.matrix_zoo import SPECTRUM_SAMPLERS

ARTIFACTS = Path(__file__).parent / "artifacts"


def main():
    kinds = list(SPECTRUM_SAMPLERS.keys())
    fig, axes = plt.subplots(1, len(kinds), figsize=(3 * len(kinds), 3))

    m = 512
    for ax, kind in zip(axes, kinds):
        for seed in range(5):
            s = SPECTRUM_SAMPLERS[kind](m, seed=seed, device="cuda").cpu().numpy()
            ax.plot(s, alpha=0.6, label=f"seed={seed}")
        ax.set_title(f"{kind}")
        ax.set_yscale("log")
        ax.set_xlabel("index i")
        ax.set_ylabel("σ_i")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Fixture spectra (m=512, 5 seeds each)")
    fig.tight_layout()
    out = ARTIFACTS / "fixture_spectra.png"
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
