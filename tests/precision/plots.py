"""Plots for the precision report.

All plots read from ARTIFACTS/exp_*.csv. Writes PNG next to the CSVs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).parent / "artifacts"
PLOT_DIR = ARTIFACTS
PLOT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9, "figure.dpi": 120})


def plot_metric_by_dtype(csv_path: Path, metric: str, out_name: str):
    """Bar plot: metric per (kind, algo_base, dtype)."""
    df = pd.read_csv(csv_path)
    df[["base", "dtype"]] = df["algo"].str.rsplit("_", n=1, expand=True)

    kinds = sorted(df["kind"].unique())
    bases = ["direct", "gram_restart", "gram_norestart"]
    dtypes = ["fp64", "bf16", "fp16"]

    fig, axes = plt.subplots(1, len(kinds), figsize=(4 * len(kinds), 3.5), sharey=True)
    if len(kinds) == 1:
        axes = [axes]

    colors = {"fp64": "#1f77b4", "bf16": "#ff7f0e", "fp16": "#2ca02c"}
    width = 0.25

    for ax, kind in zip(axes, kinds):
        sub = df[df["kind"] == kind]
        x = np.arange(len(bases))
        for i, dt in enumerate(dtypes):
            vals = [
                sub[(sub["base"] == b) & (sub["dtype"] == dt)][metric].mean()
                for b in bases
            ]
            ax.bar(x + (i - 1) * width, vals, width, color=colors[dt], label=dt)
        ax.set_xticks(x)
        ax.set_xticklabels([b.replace("_", "\n") for b in bases], fontsize=8)
        ax.set_title(kind)
        ax.set_yscale("log")
        if ax is axes[0]:
            ax.set_ylabel(metric)
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = PLOT_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def plot_delta_boxplot(csv_path: Path, metric: str, out_name: str):
    """Box plot: per-matrix |low_precision - fp64| delta, grouped by algo."""
    df = pd.read_csv(csv_path)
    df[["base", "dtype"]] = df["algo"].str.rsplit("_", n=1, expand=True)

    # Pair each fp16/bf16 run with its fp64 counterpart on the same fixture
    keys = ["kind", "m", "aspect", "seed", "base"]
    p = df.pivot_table(index=keys, columns="dtype", values=metric, aggfunc="first")
    if "fp64" not in p.columns:
        return
    delta = (p.sub(p["fp64"], axis=0)).abs().reset_index()

    kinds = sorted(delta["kind"].unique())
    bases = ["direct", "gram_restart", "gram_norestart"]
    fig, axes = plt.subplots(1, len(kinds), figsize=(4 * len(kinds), 3.5), sharey=True)
    if len(kinds) == 1:
        axes = [axes]

    positions = []
    labels = []
    for i, b in enumerate(bases):
        for j, dt in enumerate(["bf16", "fp16"]):
            positions.append(i * 3 + j + 0.5)
            labels.append(f"{b[:4]}\n{dt}")

    for ax, kind in zip(axes, kinds):
        sub = delta[delta["kind"] == kind]
        data = []
        for b in bases:
            for dt in ["bf16", "fp16"]:
                col = dt
                if col in sub.columns:
                    vals = sub[sub["base"] == b][col].dropna().values
                else:
                    vals = np.array([])
                data.append(vals if len(vals) else [np.nan])
        ax.boxplot(data, positions=positions, widths=0.6, showfliers=False)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7, rotation=0)
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_title(kind)
        if ax is axes[0]:
            ax.set_ylabel(f"|{metric}_lowprec - fp64|")
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = PLOT_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def plot_diag_drift(csv_path: Path, out_name: str):
    """Exp D: per-step diagonalizability error trajectories."""
    df = pd.read_csv(csv_path)
    kinds = sorted(df["kind"].unique())
    dtypes = ["fp64", "bf16", "fp16"]

    fig, axes = plt.subplots(len(kinds), 3, figsize=(12, 3.2 * len(kinds)),
                             sharex=True, sharey="row")
    if len(kinds) == 1:
        axes = [axes]

    for row, kind in enumerate(kinds):
        for col, dt in enumerate(dtypes):
            ax = axes[row][col]
            sub = df[(df["kind"] == kind) & (df["dtype"] == dt)]
            for algo in ["direct", "gram_restart", "gram_norestart"]:
                g = sub[sub["algo"] == algo].groupby("step")["diag_err_X"].agg(["mean", "std"])
                if len(g):
                    ax.plot(g.index, g["mean"], marker="o", label=algo)
                    ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.15)
            ax.set_yscale("log")
            ax.set_title(f"{kind} / {dt}")
            if row == len(kinds) - 1:
                ax.set_xlabel("NS step")
            if col == 0:
                ax.set_ylabel("diag_err_X")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    out = PLOT_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def main():
    for exp, metric in [("B", "svd_rel_err"), ("B", "orth_error"), ("B", "psd_error"),
                        ("C", "svd_rel_err"), ("C", "bound_violation")]:
        csv = ARTIFACTS / f"exp_{exp}.csv"
        if csv.exists():
            plot_metric_by_dtype(csv, metric, f"exp_{exp}_{metric}_bars.png")
    for exp in ["B", "C"]:
        csv = ARTIFACTS / f"exp_{exp}.csv"
        if csv.exists():
            plot_delta_boxplot(csv, "svd_rel_err", f"exp_{exp}_delta_boxplot.png")
    csv_d = ARTIFACTS / "exp_D.csv"
    if csv_d.exists():
        plot_diag_drift(csv_d, "exp_D_diag_drift.png")


if __name__ == "__main__":
    main()
