"""Benchmark grouped-batch owner-side Muon compute for fixed workload shapes.

Despite the historical filename, this benchmark now measures the owner-local
Muon compute workload, not just one SYRK call.  The timed region covers:

    momentum_buffer = momentum * momentum_buffer + grad
    ns_input = grad + momentum * momentum_buffer
    update = Gram Newton-Schulz(ns_input)
    owned = owned * (1 - lr * weight_decay) - lr * scale * update

It intentionally excludes distributed communication:

    - gradient reduce / HSDP replicate reduce
    - TP gather / scatter
    - post-step owner publish / broadcast

The goal is to produce real per-shape, per-batch compute costs for owner-rank
workload assignment.

Useful runs:

    CUDA_VISIBLE_DEVICES=0 /usr/local/bin/python \
      benchmarks/bench_batched_syrk_workload.py --method loop --batches 1,2,4,8

    CUDA_VISIBLE_DEVICES=0 /usr/local/bin/python \
      benchmarks/bench_batched_syrk_workload.py --method batched_syrk --max-batch 16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional


# Keep JIT/cache writes out of /root, which may be read-only in managed jobs.
os.environ.setdefault("DMUON_HOME", "/tmp")
os.environ.setdefault("DMUON_CACHE_DIR", "/tmp/dmuon_cache")
os.environ.setdefault("QUACK_CACHE_DIR", "/tmp/quack_cache")
os.environ.setdefault("CUTE_DSL_CACHE_DIR", "/tmp/cute_dsl_cache")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from dmuon.optim.newton_schulz import (  # noqa: E402
    DEFAULT_COEFFICIENTS,
    DEFAULT_RESTART_ITERATIONS,
    gram_newton_schulz,
)

try:  # noqa: E402
    from dmuon.kernels.syrk_sm80 import syrk_sm80

    HAS_SYRK_SM80 = True
except Exception as exc:  # pragma: no cover - environment dependent
    syrk_sm80 = None
    HAS_SYRK_SM80 = False
    SYRK_IMPORT_ERROR = exc


WORKLOAD = [
    ("VLM.qkv", (2560, 2048), 36),
    ("VLM.o", (2048, 2048), 36),
    ("VLM.gate_up", (22016, 2048), 36),
    ("VLM.down", (2048, 11008), 36),
    ("ACT.qkv", (2560, 1024), 36),
    ("ACT.o", (1024, 1024), 36),
    ("ACT.gate_up", (4096, 1024), 36),
    ("ACT.down", (1024, 2048), 36),
    ("VIS.qkv", (3840, 1280), 32),
    ("VIS.o", (1280, 1280), 32),
    ("VIS.gate_up", (6912, 1280), 32),
    ("VIS.down", (1280, 3456), 32),
    ("VIS.merger", (5120, 5120), 1),
]


def gram_shape(shape: tuple[int, int]) -> tuple[int, int]:
    rows, cols = shape
    return (cols, rows) if rows > cols else (rows, cols)


def parse_batch_list(raw: Optional[str], count: int, max_batch: int) -> list[int]:
    limit = count if max_batch <= 0 else min(count, max_batch)
    if raw is None:
        return list(range(1, limit + 1))
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("--batches produced an empty batch list")
    bad = [v for v in values if v < 1 or v > count]
    if bad:
        raise ValueError(f"batch sizes must be in [1, {count}], got {bad}")
    return values


def percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def time_cuda_event(fn: Callable[[], None], repeat: int) -> list[float]:
    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def time_wall(fn: Callable[[], None], repeat: int) -> list[float]:
    times_ms: list[float] = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


def _syrk_batched(
    a: torch.Tensor,
    d: torch.Tensor,
    *,
    b: Optional[torch.Tensor] = None,
    c: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    diag_add: float = 0.0,
    tile_m: int = 128,
    tile_k: int = 32,
    num_stages: int = 4,
    symmetric: bool = False,
) -> None:
    if syrk_sm80 is None:
        raise RuntimeError(f"syrk_sm80 is not importable: {SYRK_IMPORT_ERROR!r}")
    syrk_sm80(
        a,
        d,
        B=b,
        C=c,
        alpha=alpha,
        beta=beta,
        diag_add=diag_add,
        tile_m=tile_m,
        tile_k=tile_k,
        num_stages=num_stages,
        _symmetric=symmetric,
    )


def batched_gram_newton_schulz(
    g: torch.Tensor,
    *,
    method: str,
    tile_m: int,
    tile_k: int,
    num_stages: int,
    eps: float = 1e-7,
    coefficients: Optional[list[list[float]]] = None,
    restart_iterations: Optional[list[int]] = None,
) -> torch.Tensor:
    """Batched Gram NS for same-shaped matrices.

    ``method="batched_torch"`` uses torch.bmm throughout.
    ``method="batched_syrk"`` uses syrk_sm80 for symmetric batched products
    and torch.bmm for the non-symmetric Q update.
    """
    if g.dim() != 3:
        raise ValueError(f"expected [batch, rows, cols], got {tuple(g.shape)}")
    if coefficients is None:
        coefficients = DEFAULT_COEFFICIENTS
    if restart_iterations is None:
        restart_iterations = DEFAULT_RESTART_ITERATIONS

    original_dtype = g.dtype
    x = g.float()
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)

    normalizer = x.square().sum(dim=(-2, -1)).sqrt().add_(eps).view(-1, 1, 1)
    x = (x / normalizer).half().contiguous()
    batch, m, _k = x.shape

    def gram(
        a: torch.Tensor,
        *,
        b: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        diag_add: float = 0.0,
        symmetric: bool = False,
    ) -> torch.Tensor:
        out = torch.empty(batch, m, m, device=a.device, dtype=a.dtype)
        if method == "batched_syrk":
            _syrk_batched(
                a,
                out,
                b=b,
                c=c,
                alpha=alpha,
                beta=beta,
                diag_add=diag_add,
                tile_m=tile_m,
                tile_k=tile_k,
                num_stages=num_stages,
                symmetric=symmetric,
            )
            return out

        bt = a.transpose(-2, -1) if b is None else b.transpose(-2, -1)
        if c is None:
            out = torch.bmm(a, bt)
            if alpha != 1.0:
                out.mul_(alpha)
        else:
            out = torch.baddbmm(c * beta, a, bt, beta=1.0, alpha=alpha)
        if diag_add != 0.0:
            diag = out.diagonal(dim1=-2, dim2=-1)
            diag.add_(diag_add)
        return out

    r = gram(x)
    q: Optional[torch.Tensor] = None

    for i, (a_coeff, b_coeff, c_coeff) in enumerate(coefficients):
        if i in restart_iterations and i != 0:
            if q is None:
                raise RuntimeError("restart reached before Q was initialized")
            x = torch.bmm(q, x)
            r = gram(x)
            q = None

        z = gram(r, c=r, alpha=c_coeff, beta=b_coeff)
        if q is None:
            need_r_evolve = (
                i < len(coefficients) - 1 and (i + 1) not in restart_iterations
            )
            if not need_r_evolve:
                q = gram(r, c=r, alpha=c_coeff, beta=b_coeff, diag_add=a_coeff)
            else:
                q = z.clone()
                q.diagonal(dim1=-2, dim2=-1).add_(a_coeff)
        else:
            q = torch.baddbmm(q * a_coeff, z, q.transpose(-2, -1))

        if i < len(coefficients) - 1 and (i + 1) not in restart_iterations:
            rz = gram(r, b=z, c=r, beta=a_coeff, symmetric=True)
            r = gram(rz, b=z, c=rz, beta=a_coeff, symmetric=True)

    if q is None:
        raise RuntimeError("Q was not initialized")
    x = torch.bmm(q, x)
    if transposed:
        x = x.transpose(-2, -1)
    return x.to(original_dtype)


class OwnerMuonWorkload:
    def __init__(
        self,
        *,
        shape: tuple[int, int],
        batch: int,
        dtype: torch.dtype,
        method: str,
        lr: float,
        momentum: float,
        weight_decay: float,
        nesterov: bool,
        tile_m: int,
        tile_k: int,
        num_stages: int,
        include_update: bool,
        steady_state: bool,
    ):
        self.shape = shape
        self.batch = batch
        self.method = method
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.tile_m = tile_m
        self.tile_k = tile_k
        self.num_stages = num_stages
        self.include_update = include_update

        device = torch.device("cuda")
        rows, cols = shape
        self.grad = torch.randn(batch, rows, cols, device=device, dtype=dtype)
        self.owned = torch.randn(batch, rows, cols, device=device, dtype=dtype)
        if steady_state:
            self.momentum_buffer: Optional[torch.Tensor] = torch.randn_like(self.grad)
        else:
            self.momentum_buffer = None

    def __call__(self) -> None:
        if self.method == "loop":
            self._call_loop()
        elif self.method in {"batched_torch", "batched_syrk"}:
            self._call_batched()
        else:
            raise ValueError(f"unknown method: {self.method}")

    def _call_loop(self) -> None:
        rows, cols = self.shape
        scale = 0.2 * math.sqrt(max(rows, cols))
        first_momentum_update = self.momentum_buffer is None
        if first_momentum_update:
            self.momentum_buffer = self.grad.clone()
        for idx in range(self.batch):
            grad = self.grad[idx]
            if not first_momentum_update:
                self.momentum_buffer[idx].mul_(self.momentum).add_(grad)
            buf = self.momentum_buffer[idx]
            ns_input = grad.add(buf, alpha=self.momentum) if self.nesterov else buf
            update = gram_newton_schulz(ns_input)
            if self.include_update:
                owned = self.owned[idx]
                if self.weight_decay > 0:
                    owned.mul_(1.0 - self.lr * self.weight_decay)
                owned.add_(update.to(device=owned.device, dtype=owned.dtype), alpha=-self.lr * scale)

    def _call_batched(self) -> None:
        rows, cols = self.shape
        scale = 0.2 * math.sqrt(max(rows, cols))
        if self.momentum_buffer is None:
            self.momentum_buffer = self.grad.clone()
        else:
            self.momentum_buffer.mul_(self.momentum).add_(self.grad)
        buf = self.momentum_buffer
        ns_input = self.grad.add(buf, alpha=self.momentum) if self.nesterov else buf
        update = batched_gram_newton_schulz(
            ns_input,
            method=self.method,
            tile_m=self.tile_m,
            tile_k=self.tile_k,
            num_stages=self.num_stages,
        )
        if self.include_update:
            if self.weight_decay > 0:
                self.owned.mul_(1.0 - self.lr * self.weight_decay)
            self.owned.add_(
                update.to(device=self.owned.device, dtype=self.owned.dtype),
                alpha=-self.lr * scale,
            )


def benchmark_one(
    *,
    name: str,
    original_shape: tuple[int, int],
    count: int,
    batch: int,
    dtype: torch.dtype,
    method: str,
    lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
    tile_m: int,
    tile_k: int,
    num_stages: int,
    warmup: int,
    repeat: int,
    timer: str,
    include_update: bool,
    steady_state: bool,
) -> dict:
    workload = OwnerMuonWorkload(
        shape=original_shape,
        batch=batch,
        dtype=dtype,
        method=method,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        tile_m=tile_m,
        tile_k=tile_k,
        num_stages=num_stages,
        include_update=include_update,
        steady_state=steady_state,
    )

    # First call may trigger JIT/load. Keep it out of steady-state timing.
    workload()
    torch.cuda.synchronize()
    for _ in range(warmup):
        workload()
    torch.cuda.synchronize()

    samples = time_cuda_event(workload, repeat) if timer == "event" else time_wall(workload, repeat)
    samples.sort()
    median_ms = statistics.median(samples)
    single_task_ms = median_ms / batch
    return {
        "name": name,
        "original_shape": list(original_shape),
        "gram_shape": list(gram_shape(original_shape)),
        "count": count,
        "batch": batch,
        "method": method,
        "batch_total_ms": median_ms,
        "single_task_ms": single_task_ms,
        "median_ms": median_ms,
        "mean_ms": statistics.fmean(samples),
        "p20_ms": percentile(samples, 0.20),
        "p80_ms": percentile(samples, 0.80),
        "samples_ms": samples,
    }


def print_table(rows: Iterable[dict]) -> None:
    print(
        "method,name,orig,gram,batch,batch_total_ms,single_task_ms,"
        "mean_ms,p20_ms,p80_ms"
    )
    for row in rows:
        print(
            f"{row['method']},"
            f"{row['name']},"
            f"{tuple(row['original_shape'])},"
            f"{tuple(row['gram_shape'])},"
            f"{row['batch']},"
            f"{row['batch_total_ms']:.6f},"
            f"{row['single_task_ms']:.6f},"
            f"{row['mean_ms']:.6f},"
            f"{row['p20_ms']:.6f},"
            f"{row['p80_ms']:.6f}"
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "method",
        "name",
        "original_shape",
        "gram_shape",
        "count",
        "batch",
        "batch_total_ms",
        "single_task_ms",
        "median_ms",
        "mean_ms",
        "p20_ms",
        "p80_ms",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("loop", "batched_torch", "batched_syrk"),
        default="batched_syrk",
        help=(
            "loop=current per-param owner path; batched_torch=full batched "
            "Muon compute with torch.bmm; batched_syrk=full batched Muon "
            "compute using syrk_sm80 for batched symmetric products."
        ),
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=16,
        help="Maximum batch size per shape; 0 means full count.",
    )
    parser.add_argument(
        "--batches",
        type=str,
        default=None,
        help="Comma-separated batch sizes, e.g. 1,2,4,8,16.",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-k", type=int, default=32)
    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--timer", choices=("event", "wall"), default="event")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--no-nesterov", action="store_true")
    parser.add_argument(
        "--cold-state",
        action="store_true",
        help="Start with an empty momentum buffer. Default measures steady state.",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Exclude final owned_data weight decay/add_ writeback from timing.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional comma-separated shape names to benchmark.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record errors such as OOM and continue with later shapes/batches.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available; this benchmark needs a CUDA GPU to produce "
            "real owner-side Muon compute timings."
        )
    if args.method == "batched_syrk" and not HAS_SYRK_SM80:
        raise RuntimeError(f"method=batched_syrk needs syrk_sm80: {SYRK_IMPORT_ERROR!r}")

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    selected = set(args.only.split(",")) if args.only else None

    rows: list[dict] = []
    print(
        f"# device={torch.cuda.get_device_name()} "
        f"capability={torch.cuda.get_device_capability()}"
    )
    print(
        f"# method={args.method} dtype={args.dtype} "
        f"tile=({args.tile_m},{args.tile_k}) stages={args.num_stages}"
    )
    print(
        f"# timed_scope=momentum+nesterov+GramNS+"
        f"{'update' if not args.no_update else 'no_update'}"
    )

    for name, shape, count in WORKLOAD:
        if selected is not None and name not in selected:
            continue
        m, _ = gram_shape(shape)
        if args.method == "batched_syrk" and m % args.tile_m != 0:
            print(
                f"# skip {name}: gram M={m} is not divisible by tile_m={args.tile_m}"
            )
            continue
        for batch in parse_batch_list(args.batches, count, args.max_batch):
            print(
                f"# running {name} shape={shape} gram={gram_shape(shape)} "
                f"batch={batch}",
                flush=True,
            )
            try:
                rows.append(
                    benchmark_one(
                        name=name,
                        original_shape=shape,
                        count=count,
                        batch=batch,
                        dtype=dtype,
                        method=args.method,
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        nesterov=not args.no_nesterov,
                        tile_m=args.tile_m,
                        tile_k=args.tile_k,
                        num_stages=args.num_stages,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        timer=args.timer,
                        include_update=not args.no_update,
                        steady_state=not args.cold_state,
                    )
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                if not args.continue_on_error:
                    raise
                rows.append(
                    {
                        "name": name,
                        "original_shape": list(shape),
                        "gram_shape": list(gram_shape(shape)),
                        "count": count,
                        "batch": batch,
                        "method": args.method,
                        "batch_total_ms": float("nan"),
                        "single_task_ms": float("nan"),
                        "median_ms": float("nan"),
                        "mean_ms": float("nan"),
                        "p20_ms": float("nan"),
                        "p80_ms": float("nan"),
                        "samples_ms": [],
                        "error": f"OutOfMemoryError: {exc}",
                    }
                )
                print(f"# OOM {name} batch={batch}: {exc}", flush=True)

    print_table(rows)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
