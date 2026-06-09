"""Analysis helpers: build tables and plots from exp_*.csv artifacts.

Called after run_experiment.py finishes. Produces markdown-ready tables and
PNG plots that get embedded in the precision report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ARTIFACTS = Path(__file__).parent / "artifacts"


# ---------------------------------------------------------------------------
# Pretty-printed tables for the report
# ---------------------------------------------------------------------------
def summary_table(csv_path: Path, metric: str = "svd_rel_err") -> pd.DataFrame:
    """Mean ± std per (kind, algo), pivot for easy Markdown rendering."""
    df = pd.read_csv(csv_path)
    g = df.groupby(["kind", "algo"])[metric].agg(["mean", "std", "count"])
    g["display"] = g.apply(lambda r: f"{r['mean']:.3e} ± {r['std']:.1e}", axis=1)
    pivot = g["display"].unstack("algo")
    return pivot


def delta_from_fp64(csv_path: Path, metric: str = "svd_rel_err") -> pd.DataFrame:
    """Per-matrix delta: algo_at_lowprec - algo_at_fp64.

    Paired comparison — measures precision sensitivity.
    """
    df = pd.read_csv(csv_path)

    # Parse algo name: "direct_bf16" → base="direct", dtype="bf16"
    df[["base", "dtype"]] = df["algo"].str.rsplit("_", n=1, expand=True)

    # Pivot: one row per (fixture, base), columns per dtype
    keys = ["kind", "m", "aspect", "seed", "base"]
    p = df.pivot_table(index=keys, columns="dtype", values=metric, aggfunc="first")

    # Delta vs fp64
    if "fp64" not in p.columns:
        return pd.DataFrame()

    deltas = {}
    for dt in ["bf16", "fp16"]:
        if dt in p.columns:
            deltas[f"delta_{dt}"] = (p[dt] - p["fp64"]).abs()
    if not deltas:
        return pd.DataFrame()

    out = pd.concat(deltas, axis=1).reset_index()
    # Summarize by (kind, base)
    summary = out.groupby(["kind", "base"])[list(deltas.keys())].agg(["mean", "std", "max"])
    return summary


def winrate_pair(csv_path: Path, algo_a: str, algo_b: str,
                 metric: str = "svd_rel_err") -> pd.DataFrame:
    """Paired win rate — fraction of matrices where algo_a < algo_b on metric."""
    df = pd.read_csv(csv_path)
    p = df.pivot_table(
        index=["kind", "m", "aspect", "seed"],
        columns="algo",
        values=metric,
        aggfunc="first",
    )
    if algo_a not in p.columns or algo_b not in p.columns:
        return pd.DataFrame()
    wins_a = (p[algo_a] < p[algo_b]) & ~p[[algo_a, algo_b]].isna().any(axis=1)
    wins_b = (p[algo_b] < p[algo_a]) & ~p[[algo_a, algo_b]].isna().any(axis=1)
    ties = (p[algo_a] == p[algo_b]) & ~p[[algo_a, algo_b]].isna().any(axis=1)
    grp = p.groupby(level=["kind"]).size()
    out = pd.DataFrame({
        f"{algo_a}_wins": wins_a.groupby(level=["kind"]).sum(),
        f"{algo_b}_wins": wins_b.groupby(level=["kind"]).sum(),
        "ties": ties.groupby(level=["kind"]).sum(),
        "total": grp,
    })
    out[f"{algo_a}_pct"] = out[f"{algo_a}_wins"] / out["total"] * 100
    out[f"{algo_b}_pct"] = out[f"{algo_b}_wins"] / out["total"] * 100
    return out


def markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table."""
    return df.to_markdown(floatfmt=".3e")


# ---------------------------------------------------------------------------
# Main CLI: render all tables and save to a single .md fragment
# ---------------------------------------------------------------------------
def render_report_tables():
    out = []
    for exp in ["A", "B", "C"]:
        csv = ARTIFACTS / f"exp_{exp}.csv"
        if not csv.exists():
            out.append(f"## Experiment {exp}\n\n_Data file not found: {csv}_\n\n")
            continue
        out.append(f"## Experiment {exp}\n\n")

        for metric in ["svd_rel_err", "orth_error", "psd_error"]:
            out.append(f"### {metric}\n\n")
            tbl = summary_table(csv, metric)
            out.append(tbl.to_markdown() + "\n\n")

        out.append(f"### {exp}: delta from fp64 (precision sensitivity)\n\n")
        delta = delta_from_fp64(csv, "svd_rel_err")
        if not delta.empty:
            out.append(delta.to_markdown() + "\n\n")

        out.append(f"### {exp}: Direct vs Gram_restart win rate (bf16)\n\n")
        wr = winrate_pair(csv, "direct_bf16", "gram_restart_bf16", "svd_rel_err")
        if not wr.empty:
            out.append(wr.to_markdown() + "\n\n")

    content = "".join(out)
    out_path = ARTIFACTS / "report_tables.md"
    out_path.write_text(content)
    print(f"Wrote {out_path}")
    return content


if __name__ == "__main__":
    render_report_tables()
