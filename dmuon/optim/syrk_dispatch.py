"""SYRK kernel dispatch with per-shape autotune.

Detects CuteDSL SYRK availability at import time, benchmarks all tile
configurations against cuBLAS for each (M, K, dtype) shape, and caches
the winner.  The main entry point is ``syrk_or_cublas()``.
"""

from __future__ import annotations

import json
import logging
import os
import time as _time
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
_SM_VERSION = 0
if torch.cuda.is_available():
    _cap = torch.cuda.get_device_capability()
    _SM_VERSION = _cap[0] * 10 + _cap[1]

# Try CuteDSL SYRK kernel (SM80+: A100/A800/H100)
HAS_SYRK = False
_syrk_sm80_fn = None
_SYRK_CONFIGS = []
try:
    from dmuon.kernels.syrk_sm80 import syrk_sm80 as _syrk_sm80_fn
    from dmuon.kernels.syrk_sm80 import SYRK_SM80_CONFIGS as _SYRK_CONFIGS
    HAS_SYRK = True
except ImportError:
    pass


def get_ns_backend() -> str:
    """Return a human-readable name of the active Newton-Schulz kernel backend.

    Returns one of:
      * ``"CuteDSL SYRK (SM80)"`` — CuteDSL SYRK kernel compiled for SM80
        (A100 / A800). 1.4–1.5× end-to-end speedup on Gram NS ops vs fallback.
      * ``"CuteDSL SYRK (SM90)"`` — CuteDSL SYRK kernel compiled for SM90+
        (H100 / H200), when supported by the installed CuteDSL version.
      * ``"torch.compile (fallback)"`` — generic ``@torch.compile``-compiled
        reference path, used when no CuteDSL SYRK is available for the
        current GPU (e.g. V100 / RTX 30xx / consumer Blackwell).

    Call once at startup to confirm you are on the fast path:

    >>> import dmuon
    >>> print(dmuon.get_ns_backend())
    CuteDSL SYRK (SM80)
    """
    if HAS_SYRK:
        return f"CuteDSL SYRK (SM{_SM_VERSION})"
    return "torch.compile (fallback)"


# ---------------------------------------------------------------------------
# SYRK autotune: benchmark all tile configs vs cuBLAS, cache per shape
# ---------------------------------------------------------------------------

# Cache: (M, K, device_idx, dtype, has_C) -> (tile_m, tile_k, num_stages) or None
_syrk_autotune_cache: dict[tuple, tuple | None] = {}


def _get_autotune_cache_path() -> Path:
    """Get path for persistent autotune cache file."""
    cache_dir = os.environ.get("DMUON_CACHE_DIR")
    if cache_dir:
        p = Path(cache_dir)
    else:
        p = Path.home() / ".cache" / "dmuon"
    p.mkdir(parents=True, exist_ok=True)
    gpu_name = torch.cuda.get_device_name(0).replace(" ", "_") if torch.cuda.is_available() else "cpu"
    return p / f"syrk_autotune_{gpu_name}.json"


_DTYPE_TO_STR = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def _key_to_json(key: tuple) -> str:
    """Serialize cache key to JSON-safe string."""
    M, K, dev_idx, dtype, has_C = key
    return json.dumps([M, K, dev_idx, _DTYPE_TO_STR.get(dtype, str(dtype)), has_C])


def _json_to_key(s: str) -> tuple:
    """Deserialize cache key from JSON string."""
    M, K, dev_idx, dtype_str, has_C = json.loads(s)
    return (M, K, dev_idx, _STR_TO_DTYPE.get(dtype_str, dtype_str), has_C)


def _load_autotune_cache() -> None:
    """Load persistent autotune cache from disk."""
    global _syrk_autotune_cache
    try:
        path = _get_autotune_cache_path()
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            for k, v in data.items():
                key = _json_to_key(k)
                _syrk_autotune_cache[key] = tuple(v) if v is not None else None
            logger.info(f"Loaded {len(data)} autotune entries from {path}")
    except Exception as e:
        logger.debug(f"Could not load autotune cache: {e}")


def _save_autotune_cache() -> None:
    """Save autotune cache to disk."""
    try:
        path = _get_autotune_cache_path()
        data = {}
        for k, v in _syrk_autotune_cache.items():
            data[_key_to_json(k)] = list(v) if v is not None else None
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.debug(f"Could not save autotune cache: {e}")


# Load persistent cache at import time
if HAS_SYRK:
    _load_autotune_cache()


def _bench_median(fn, warmup=5, repeat=20):
    """Quick benchmark returning median time in seconds."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(_time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def _autotune_syrk(M: int, K: int, device: torch.device, dtype: torch.dtype,
                    has_C: bool = False) -> tuple | None:
    """Find best SYRK config for shape (M, K). Returns (tile_m, tile_k, num_stages) or None if cuBLAS wins."""
    key = (M, K, device.index or 0, dtype, has_C)
    if key in _syrk_autotune_cache:
        return _syrk_autotune_cache[key]

    X = torch.randn(M, K, device=device, dtype=dtype)
    D = torch.empty(M, M, device=device, dtype=dtype)

    # cuBLAS baseline
    if has_C:
        C_mat = torch.randn(M, M, device=device, dtype=dtype)
        t_cublas = _bench_median(lambda: torch.addmm(C_mat, X, X.T, alpha=0.5, beta=0.3))
    else:
        t_cublas = _bench_median(lambda: torch.mm(X, X.T, out=D))

    best_time = t_cublas
    best_config = None

    for tile_m, tile_k, num_stages in _SYRK_CONFIGS:
        if M % tile_m != 0:
            continue
        try:
            if has_C:
                C_mat2 = torch.randn(M, M, device=device, dtype=dtype)
                def bench_syrk(tm=tile_m, tk=tile_k, ns=num_stages):
                    _syrk_sm80_fn(X, D, C=C_mat2, alpha=0.5, beta=0.3,
                                  tile_m=tm, tile_k=tk, num_stages=ns)
            else:
                def bench_syrk(tm=tile_m, tk=tile_k, ns=num_stages):
                    _syrk_sm80_fn(X, D, tile_m=tm, tile_k=tk, num_stages=ns)
            t = _bench_median(bench_syrk)
            if t < best_time:
                best_time = t
                best_config = (tile_m, tile_k, num_stages)
        except Exception:
            continue

    speedup = t_cublas / best_time if best_config else 1.0
    logger.info(
        f"SYRK autotune ({M},{K}) has_C={has_C}: "
        f"cuBLAS={t_cublas*1e6:.0f}us, best={best_time*1e6:.0f}us "
        f"config={best_config} speedup={speedup:.2f}x"
    )

    _syrk_autotune_cache[key] = best_config
    _save_autotune_cache()
    return best_config


def syrk_or_cublas(A: Tensor, D: Tensor, B: Tensor | None = None,
                   C: Tensor | None = None, alpha: float = 1.0,
                   beta: float = 1.0, diag_add: float = 0.0) -> None:
    """Symmetric GEMM with autotuned SYRK or cuBLAS fallback.

    Computes D = alpha * A @ B^T + beta * C + diag_add * I.
    When B is None, B = A (true SYRK).
    When B != A, the result MUST be symmetric (caller's responsibility).
    """
    M, K = A.shape[0], A.shape[1]
    has_C = C is not None
    BT = A.T if B is None else B.T
    is_true_syrk = B is None or B.data_ptr() == A.data_ptr()
    config = _autotune_syrk(M, K, A.device, A.dtype, has_C)
    if config is not None:
        tile_m, tile_k, num_stages = config
        _syrk_sm80_fn(A, D, B=B, C=C, alpha=alpha, beta=beta,
                       diag_add=diag_add,
                       tile_m=tile_m, tile_k=tile_k, num_stages=num_stages,
                       _symmetric=not is_true_syrk)
    else:
        # cuBLAS fallback
        if has_C:
            torch.addmm(C, A, BT, alpha=alpha, beta=beta, out=D)
        else:
            torch.mm(A, BT, out=D)
        if diag_add != 0.0:
            D.diagonal().add_(diag_add)
