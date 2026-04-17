"""CLI entry for NS precision experiments.

Usage:
    python -m tests.precision.run_experiment --exp A
    python -m tests.precision.run_experiment --exp B
    python -m tests.precision.run_experiment --exp all
    python -m tests.precision.run_experiment --exp B --n-seeds 30  # more data
"""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from tests.precision.matrix_zoo import Fixture, fixture_grid
from tests.precision.polar_metrics import (
    diagonalizability_error,
    direct_svd_error,
    polar_accuracy,
    svd_polar,
)
from tests.precision.ref_ns import direct_ns_ref, gram_ns_ref

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Algorithm configurations
# ---------------------------------------------------------------------------
@dataclass
class AlgoConfig:
    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]


def _dtag(dtype: torch.dtype) -> str:
    return {torch.float64: "fp64", torch.float32: "fp32",
            torch.bfloat16: "bf16", torch.float16: "fp16"}[dtype]


def make_configs(dtype: torch.dtype) -> list[AlgoConfig]:
    return [
        AlgoConfig(
            f"direct_{_dtag(dtype)}",
            lambda A, dt=dtype: direct_ns_ref(A, compute_dtype=dt),
        ),
        AlgoConfig(
            f"gram_restart_{_dtag(dtype)}",
            lambda A, dt=dtype: gram_ns_ref(A, compute_dtype=dt, restart_iterations=[2]),
        ),
        AlgoConfig(
            f"gram_norestart_{_dtag(dtype)}",
            lambda A, dt=dtype: gram_ns_ref(A, compute_dtype=dt, restart_iterations=[]),
        ),
    ]


# ---------------------------------------------------------------------------
# Per-(fixture, algo) runner
# ---------------------------------------------------------------------------
def _metrics_row(fixture: Fixture, algo_name: str, A64, X_ref, X_est, wall: float) -> dict:
    metrics = polar_accuracy(A64, X_est)
    metrics["svd_rel_err"] = direct_svd_error(X_ref, X_est)
    metrics["wall_s"] = wall
    metrics["fixture"] = fixture.name
    metrics["kind"] = fixture.kind
    metrics["m"] = fixture.m
    metrics["n"] = fixture.n
    metrics["aspect"] = fixture.aspect
    metrics["seed"] = fixture.seed
    metrics["algo"] = algo_name
    return metrics


def run_sweep(fixtures: list[Fixture], algos: list[AlgoConfig], *, tag: str) -> pd.DataFrame:
    """Outer loop over fixtures (build A + SVD once), inner loop over algos."""
    rows = []
    total = len(fixtures) * len(algos)
    idx = 0
    for fix in fixtures:
        A64 = fix.build(device="cuda")
        X_ref = svd_polar(A64)
        A_in = A64.to(torch.float32)
        for algo in algos:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            X_est = algo.fn(A_in)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            rows.append(_metrics_row(fix, algo.name, A64, X_ref, X_est, wall))
            idx += 1
            if idx % 50 == 0 or idx == total:
                latest = rows[-1]
                print(
                    f"[{tag}] {idx}/{total}  {fix.name:<42} {algo.name:<22} "
                    f"svd={latest['svd_rel_err']:.2e}  orth={latest['orth_error']:.2e}",
                    flush=True,
                )
        del A64, X_ref, A_in
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics: bootstrap CI + paired win-rate
# ---------------------------------------------------------------------------
METRIC_COLS = [
    "svd_rel_err", "orth_error", "residual_error",
    "psd_error", "dual_obj", "bound_violation",
]


def bootstrap_ci(values: np.ndarray, *, n_boot: int = 1000, q: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean."""
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [finite[rng.integers(0, len(finite), len(finite))].mean() for _ in range(n_boot)]
    alpha = (1 - q) / 2
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (kind, m, aspect, algo): mean, std, 95% CI (bootstrap)."""
    rows = []
    groups = df.groupby(["kind", "m", "aspect", "algo"])
    for (kind, m, aspect, algo), g in groups:
        row = dict(kind=kind, m=m, aspect=aspect, algo=algo, n=len(g))
        for col in METRIC_COLS:
            vals = g[col].to_numpy()
            row[f"{col}_mean"] = float(np.nanmean(vals))
            row[f"{col}_std"] = float(np.nanstd(vals))
            lo, hi = bootstrap_ci(vals)
            row[f"{col}_ci_lo"] = lo
            row[f"{col}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def win_rate(df: pd.DataFrame, metric: str, algo_a: str, algo_b: str) -> pd.DataFrame:
    """Paired comparison: per (kind, m, aspect, seed), which algo has lower metric?

    Returns fraction of matrices where algo_a < algo_b (lower is better for all
    error metrics).
    """
    pivot = df.pivot_table(
        index=["kind", "m", "aspect", "seed"],
        columns="algo",
        values=metric,
    )
    if algo_a not in pivot.columns or algo_b not in pivot.columns:
        return pd.DataFrame()
    diff = (pivot[algo_a] < pivot[algo_b]).astype(int)
    tied = (pivot[algo_a] == pivot[algo_b]).astype(int)
    return pd.DataFrame({
        f"{algo_a}_wins": diff.groupby(level=["kind", "m", "aspect"]).mean(),
        "ties": tied.groupby(level=["kind", "m", "aspect"]).mean(),
        "n_pairs": diff.groupby(level=["kind", "m", "aspect"]).count(),
    }).reset_index()


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def _default_grid(n_seeds: int = 15) -> list[Fixture]:
    return fixture_grid(
        kinds=["uniform", "gaussian_random", "power_law", "exponential"],
        sizes=[512, 1024],
        aspects=[1, 4],  # square + typical shortfat; aspect=2,8 dropped for speed
        n_seeds=n_seeds,
    )


def exp_A(n_seeds: int) -> pd.DataFrame:
    """fp64 algorithmic floor — sanity that Direct and Gram are algebraically equivalent."""
    fixtures = _default_grid(n_seeds)
    algos = make_configs(torch.float64)
    return run_sweep(fixtures, algos, tag="ExpA")


def exp_B(n_seeds: int) -> pd.DataFrame:
    """Precision sensitivity across dtypes. Core comparison."""
    fixtures = _default_grid(n_seeds)
    algos: list[AlgoConfig] = []
    for dt in [torch.float64, torch.bfloat16, torch.float16]:
        algos.extend(make_configs(dt))
    return run_sweep(fixtures, algos, tag="ExpB")


def exp_C(n_seeds: int) -> pd.DataFrame:
    """Stressed spectra only: exponential + heavy_tail with bf16/fp16."""
    fixtures = fixture_grid(
        kinds=["exponential", "heavy_tail"],
        sizes=[512, 1024],
        aspects=[1, 4],
        n_seeds=n_seeds,
    )
    algos: list[AlgoConfig] = []
    for dt in [torch.bfloat16, torch.float16]:
        algos.extend(make_configs(dt))
    return run_sweep(fixtures, algos, tag="ExpC")


def exp_D(n_seeds: int) -> pd.DataFrame:
    """Eigenvector drift: per-step diagonalizability of X_t / R_t / Q_t."""
    from dmuon.optim.newton_schulz import DEFAULT_COEFFICIENTS

    fixtures = fixture_grid(
        kinds=["uniform", "exponential"],
        sizes=[512, 1024],
        aspects=[1, 4],
        n_seeds=n_seeds,
    )
    rows = []
    total = len(fixtures)
    for fi, fix in enumerate(fixtures, 1):
        A64 = fix.build(device="cuda")
        A_in = A64.to(torch.float32)
        X0 = A_in / A_in.norm()
        transposed = X0.shape[0] > X0.shape[1]
        if transposed:
            X0 = X0.T
        X0_64 = X0.to(torch.float64)
        U0, _, Vh0 = torch.linalg.svd(X0_64, full_matrices=False)
        V0 = Vh0.mT

        for dt in [torch.float64, torch.bfloat16, torch.float16]:
            for algo_name, restart in [("gram_restart", [2]), ("gram_norestart", []), ("direct", None)]:
                # Inline iterations to capture per-step internals
                X = X0.to(dt)
                if algo_name == "direct":
                    for i, (a, b, c) in enumerate(DEFAULT_COEFFICIENTS):
                        A = X @ X.T
                        B = b * A + c * A @ A
                        X = a * X + B @ X
                        rows.append(dict(
                            fixture=fix.name, kind=fix.kind, m=fix.m, n=fix.n,
                            seed=fix.seed, algo=algo_name, dtype=_dtag(dt), step=i,
                            diag_err_R=diagonalizability_error(A, U0, symmetric=True),
                            diag_err_Q=float("nan"),
                            diag_err_X=diagonalizability_error(
                                X.to(torch.float64), U0, V0, symmetric=False
                            ),
                        ))
                else:
                    R = X @ X.T
                    m = X.shape[0]
                    I = torch.eye(m, device=X.device, dtype=X.dtype)
                    Q = None
                    for i, (a, b, c) in enumerate(DEFAULT_COEFFICIENTS):
                        if i in restart and i != 0:
                            X = Q @ X
                            R = X @ X.T
                            Q = None
                        Z = torch.addmm(R, R, R, alpha=c, beta=b)
                        if Q is None:
                            Q = Z + a * I
                        else:
                            Q = torch.addmm(Q, Z, Q, beta=a)
                        will_restart = (i + 1) in restart
                        if i < len(DEFAULT_COEFFICIENTS) - 1 and not will_restart:
                            RZ = torch.addmm(R, R, Z, beta=a)
                            R = torch.addmm(RZ, Z, RZ, beta=a)
                        X_proj = (Q @ X).to(torch.float64)
                        rows.append(dict(
                            fixture=fix.name, kind=fix.kind, m=fix.m, n=fix.n,
                            seed=fix.seed, algo=algo_name, dtype=_dtag(dt), step=i,
                            diag_err_R=diagonalizability_error(R, U0, symmetric=True),
                            diag_err_Q=diagonalizability_error(Q, U0, symmetric=True),
                            diag_err_X=diagonalizability_error(X_proj, U0, V0, symmetric=False),
                        ))
        if fi % 20 == 0 or fi == total:
            print(f"[ExpD] {fi}/{total}  {fix.name}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["A", "B", "C", "D", "all"], default="all")
    ap.add_argument("--n-seeds", type=int, default=15)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def _run(tag: str, fn):
        print(f"\n{'='*70}\n  Experiment {tag}  (n_seeds={args.n_seeds})\n{'='*70}")
        t0 = time.perf_counter()
        df = fn(args.n_seeds)
        wall = time.perf_counter() - t0
        out = ARTIFACTS / f"exp_{tag}.csv"
        df.to_csv(out, index=False)
        print(f"\n  wall={wall:.1f}s  rows={len(df)}  → {out}")
        if tag != "D":
            summary = summarize(df)
            summary_path = ARTIFACTS / f"exp_{tag}_summary.csv"
            summary.to_csv(summary_path, index=False)
            print(f"  → {summary_path}")
        return df

    exps = dict(A=exp_A, B=exp_B, C=exp_C, D=exp_D)
    if args.exp == "all":
        for tag, fn in exps.items():
            _run(tag, fn)
    else:
        _run(args.exp, exps[args.exp])


if __name__ == "__main__":
    main()
