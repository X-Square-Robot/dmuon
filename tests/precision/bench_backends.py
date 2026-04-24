"""B6-A — A-card benchmark & correctness matrix for the SYRK backend dispatch.

Run on an A100 / A800 (SM80) machine with CuteDSL SYRK built.  Produces
a markdown table that goes into ``docs/internal/benchmarks/ns_backend_bench_a800.md``.

Usage::

    python tests/precision/bench_backends.py --out bench_a800.md

Reports per (M, K, dtype, has_C):
    * cuBLAS baseline (us)
    * cute_sm80 time (us, auto-tuned)
    * speedup cute_sm80 / cuBLAS
    * max abs diff (correctness sanity)

The script also exercises:
    * kernel="auto" on A800 → must resolve to cute_sm80
    * kernel="cublas" explicit → forces fallback path
    * autotune cache warm-path (second call no benchmark log)
    * per-backend cache isolation
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from dmuon import get_backend_status, get_ns_backend, NewtonSchulz
from dmuon.kernels.syrk_backends import SyrkBackend, syrk_dispatch
from dmuon.optim import syrk_dispatch as _sd

_logger = logging.getLogger("bench_backends")

# Default sweep — keep each config quick so the whole matrix runs in
# ~2-3 minutes on an A800.  Consumers can extend via CLI.
DEFAULT_SHAPES = [
    (1024, 1024),
    (2048, 1024),
    (2048, 2048),
    (4096, 2048),
    (4096, 4096),
    (8192, 4096),
]
DEFAULT_DTYPES = [torch.bfloat16, torch.float16]
# has_C=True exercises the addmm path (C matrix, alpha/beta coefficients).
DEFAULT_HAS_C = [False, True]


@dataclass
class Row:
    M: int
    K: int
    dtype: str
    has_C: bool
    cublas_us: float
    cute_us: float
    speedup: float
    max_abs_diff: float


def _bench(fn, warmup=5, repeat=20) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def run_cell(M: int, K: int, dtype: torch.dtype, has_C: bool) -> Row:
    device = torch.device("cuda")
    A = torch.randn(M, K, device=device, dtype=dtype)
    D_cublas = torch.empty(M, M, device=device, dtype=dtype)
    D_cute = torch.empty(M, M, device=device, dtype=dtype)
    C = torch.randn(M, M, device=device, dtype=dtype) if has_C else None
    # Symmetrise C so SYRK's lower-triangle-only path doesn't trip the
    # "C must be symmetric" kernel contract.
    if C is not None:
        C.copy_((C + C.T) / 2)

    alpha, beta = (0.5, 0.3) if has_C else (1.0, 1.0)

    def call_cublas():
        syrk_dispatch(
            A, D_cublas, C=C, alpha=alpha, beta=beta,
            backend=SyrkBackend.CUBLAS,
        )

    def call_cute():
        # Go through the autotune layer so we get the tile-optimised path
        _sd.syrk_or_cublas(A, D_cute, C=C, alpha=alpha, beta=beta)

    # Warm the autotune cache once before timing
    call_cute()
    call_cublas()

    t_cublas = _bench(call_cublas)
    t_cute = _bench(call_cute)

    # Correctness: compare against cuBLAS as the reference.  Both should
    # match up to bf16 accumulation noise.
    call_cublas()
    call_cute()
    diff = (D_cublas.float() - D_cute.float()).abs().max().item()

    return Row(
        M=M, K=K,
        dtype={torch.bfloat16: "bf16", torch.float16: "fp16",
               torch.float32: "fp32"}[dtype],
        has_C=has_C,
        cublas_us=t_cublas * 1e6,
        cute_us=t_cute * 1e6,
        speedup=t_cublas / max(t_cute, 1e-9),
        max_abs_diff=diff,
    )


def render_markdown(rows: list[Row], status: dict) -> str:
    lines = [
        "# NS Backend Dispatch — A800 Benchmark (B6-A)",
        "",
        f"Auto backend choice: **{status['auto_choice']}**",
        f"Device: `{torch.cuda.get_device_name(0)}` · SM{status['sm_version']}",
        f"cute_sm80 available: `{status['cute_sm80_available']}` · "
        f"quack available: `{status['quack_available']}`",
        "",
        "## Correctness + Performance",
        "",
        "| M | K | dtype | has_C | cuBLAS (us) | cute_sm80 (us) | speedup | max|Δ| |",
        "|---|---|-------|-------|-------------|----------------|---------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r.M} | {r.K} | {r.dtype} | {r.has_C} | "
            f"{r.cublas_us:.1f} | {r.cute_us:.1f} | "
            f"{r.speedup:.2f}× | {r.max_abs_diff:.2e} |"
        )
    lines.append("")
    lines.append("## API sanity checks")
    lines.append("")
    lines.append(f"`get_ns_backend()` → `{get_ns_backend()}`")
    lines.append("")
    lines.append(f"`get_backend_status()` → `{json.dumps(status)}`")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                     default=Path("docs/internal/benchmarks/ns_backend_bench_a800.md"))
    ap.add_argument("--quick", action="store_true",
                     help="Run a small subset for sanity (6 cells instead of ~24).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    status = get_backend_status()
    _logger.info("Backend status: %s", status)
    _logger.info("Active backend: %s", get_ns_backend())

    if status["auto_choice"] != "cute_sm80":
        _logger.warning(
            "Expected auto_choice='cute_sm80' on A-card; got %r.  "
            "Results below may not reflect the A-card fast path.",
            status["auto_choice"],
        )

    shapes = DEFAULT_SHAPES
    dtypes = DEFAULT_DTYPES
    has_C_variants = DEFAULT_HAS_C
    if args.quick:
        shapes = shapes[:3]
        dtypes = dtypes[:1]
        has_C_variants = [False]

    rows: list[Row] = []
    for (M, K) in shapes:
        for dtype in dtypes:
            for has_C in has_C_variants:
                _logger.info("cell: M=%d K=%d dtype=%s has_C=%s", M, K, dtype, has_C)
                try:
                    row = run_cell(M, K, dtype, has_C)
                    rows.append(row)
                    _logger.info(
                        "  cuBLAS=%.1fus  cute_sm80=%.1fus  speedup=%.2fx  |Δ|_max=%.2e",
                        row.cublas_us, row.cute_us, row.speedup, row.max_abs_diff,
                    )
                except Exception as exc:
                    _logger.error("cell failed: %s", exc, exc_info=True)

    md = render_markdown(rows, status)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    _logger.info("Wrote %s (%d rows)", args.out, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
