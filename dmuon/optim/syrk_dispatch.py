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
import torch
import torch.distributed as dist
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
    """Return a human-readable one-liner describing the active NS kernel.

    Format: ``"Gram NS · kernel=<name> (SM<ver>, <detail>)"``.  Typical
    results:

      * ``"Gram NS · kernel=cute_sm80 (SM80, DMuon internal)"``
      * ``"Gram NS · kernel=quack (SM90, Tri Dao quack)"``
      * ``"Gram NS · kernel=cublas (SM80, universal fallback)"``

    This is the terse one-liner meant for startup log / user sanity-check.
    Use :func:`get_backend_status` for the full dict of availability
    flags.

    >>> import dmuon
    >>> print(dmuon.get_ns_backend())
    Gram NS · kernel=cute_sm80 (SM80, DMuon internal)
    """
    from dmuon.kernels.syrk_backends import SyrkBackend, detect_best_backend

    choice = detect_best_backend()
    if choice == SyrkBackend.QUACK:
        detail = "Tri Dao quack"
    elif choice == SyrkBackend.CUTE_SM80:
        detail = "DMuon internal"
    else:
        detail = "universal fallback"
    return f"Gram NS · kernel={choice.value} (SM{_SM_VERSION}, {detail})"


def get_backend_status() -> dict:
    """Return a full diagnostic snapshot of the NS kernel dispatch state.

    Returns a plain dict with:

      * ``sm_version``                  — int, detected compute capability (0 on CPU)
      * ``auto_choice``                  — which backend ``kernel="auto"`` resolves to
      * ``quack_available``              — bool, soft-dep flag
      * ``cute_sm80_available``          — bool, CuteDSL SYRK importable
      * ``cublas_always_available``      — bool, always ``True``

    Useful for programmatic checks and bug reports.
    """
    from dmuon.kernels.syrk_backends import get_backend_status as _status
    return _status()


# ---------------------------------------------------------------------------
# SYRK autotune: benchmark all tile configs vs cuBLAS, cache per shape
# ---------------------------------------------------------------------------
#
# B5 (ns_backend_dispatch_plan.md §3) split the persistent cache per
# (GPU, backend) so quack and cute_sm80 don't pollute each other's
# per-shape tile choices.  The in-memory cache key is extended with a
# ``backend`` string so a single process can autotune both if the user
# deliberately benchmarks multiple backends.
#
# Key:      (M, K, device_idx, dtype, has_C, backend_str)
# Filename: ~/.cache/dmuon/syrk_autotune_<GPU>_<backend>.json
# Legacy:   ~/.cache/dmuon/syrk_autotune_<GPU>.json  (pre-B5)
#           — migrated on first load, original kept as *.bak_preB5

# Default backend name tag used when callers don't override — the
# autotune layer still defaults to cute_sm80 on SM80, matching the
# pre-B5 behaviour.
_DEFAULT_AUTOTUNE_BACKEND = "cute_sm80"

# In-memory cache keyed by the full tuple including backend
_syrk_autotune_cache: dict[tuple, tuple | None] = {}


def _progress_log_enabled() -> bool:
    value = os.environ.get("DMUON_SYRK_AUTOTUNE_LOG", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def _rank_world_for_log() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        try:
            return dist.get_rank(), dist.get_world_size()
        except RuntimeError:
            pass
    return 0, 1


def _progress_log(message: str) -> None:
    if not _progress_log_enabled():
        return
    rank, world = _rank_world_for_log()
    print(f"[DMuon][rank={rank}/{world}] {message}", flush=True)


def _cache_dir() -> Path:
    cache_dir = os.environ.get("DMUON_CACHE_DIR")
    if cache_dir:
        p = Path(cache_dir)
    else:
        p = Path.home() / ".cache" / "dmuon"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _gpu_tag() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0).replace(" ", "_")
    return "cpu"


def _get_autotune_cache_path(backend: str = _DEFAULT_AUTOTUNE_BACKEND) -> Path:
    """Per-(GPU, backend) cache file path."""
    return _cache_dir() / f"syrk_autotune_{_gpu_tag()}_{backend}.json"


def _get_legacy_cache_path() -> Path:
    """Pre-B5 single-file path (no backend suffix)."""
    return _cache_dir() / f"syrk_autotune_{_gpu_tag()}.json"


_DTYPE_TO_STR = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def _key_to_json(key: tuple) -> str:
    """Serialize cache key (without backend, which is the file tag) to JSON."""
    M, K, dev_idx, dtype, has_C, _backend = key
    return json.dumps([M, K, dev_idx, _DTYPE_TO_STR.get(dtype, str(dtype)), has_C])


def _json_to_key(s: str, backend: str) -> tuple:
    """Deserialize a JSON cache row into the 6-tuple in-memory key."""
    M, K, dev_idx, dtype_str, has_C = json.loads(s)
    return (M, K, dev_idx, _STR_TO_DTYPE.get(dtype_str, dtype_str), has_C, backend)


def _migrate_legacy_cache_if_present() -> None:
    """One-shot migration of the pre-B5 single-file cache.

    The pre-B5 file stored only cute_sm80 data (cublas was never
    autotuned).  We copy it to the new cute_sm80 path and rename the
    original to ``*.bak_preB5`` so the user can recover if B5 introduces
    a regression.
    """
    legacy = _get_legacy_cache_path()
    if not legacy.exists():
        return
    new_path = _get_autotune_cache_path(_DEFAULT_AUTOTUNE_BACKEND)
    if new_path.exists():
        # Backend-specific file already written; the user has a fresh
        # cache; leave the legacy alone (don't overwrite newer data).
        return
    try:
        new_path.write_bytes(legacy.read_bytes())
        backup = legacy.with_suffix(legacy.suffix + ".bak_preB5")
        legacy.replace(backup)
        logger.info(
            "Migrated pre-B5 autotune cache %s → %s (backup: %s)",
            legacy, new_path, backup,
        )
    except OSError as e:
        logger.debug("Could not migrate legacy cache: %s", e)


def _load_autotune_cache(backend: str = _DEFAULT_AUTOTUNE_BACKEND) -> None:
    """Load persistent autotune cache from disk for ``backend``."""
    global _syrk_autotune_cache
    try:
        path = _get_autotune_cache_path(backend)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            for k, v in data.items():
                key = _json_to_key(k, backend)
                _syrk_autotune_cache[key] = tuple(v) if v is not None else None
            logger.info(
                "Loaded %d autotune entries for backend=%s from %s",
                len(data), backend, path,
            )
    except Exception as e:
        logger.debug("Could not load autotune cache for backend=%s: %s", backend, e)


def _save_autotune_cache(backend: str = _DEFAULT_AUTOTUNE_BACKEND) -> None:
    """Save entries for ``backend`` to its per-backend JSON file."""
    try:
        path = _get_autotune_cache_path(backend)
        data = {}
        for k, v in _syrk_autotune_cache.items():
            if k[-1] != backend:
                continue
            data[_key_to_json(k)] = list(v) if v is not None else None
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.debug("Could not save autotune cache for backend=%s: %s", backend, e)


# Migrate legacy cache file once, then load per-backend caches at import.
if HAS_SYRK:
    _migrate_legacy_cache_if_present()
    _load_autotune_cache(_DEFAULT_AUTOTUNE_BACKEND)


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
                    has_C: bool = False,
                    backend: str = _DEFAULT_AUTOTUNE_BACKEND) -> tuple | None:
    """Find best SYRK config for shape (M, K).  Returns ``(tile_m, tile_k,
    num_stages)`` or ``None`` if cuBLAS wins.

    ``backend`` tags the cache entry so quack / cute_sm80 autotune results
    never pollute each other's choices (B5)."""
    key = (M, K, device.index or 0, dtype, has_C, backend)
    if key in _syrk_autotune_cache:
        return _syrk_autotune_cache[key]

    _progress_log(
        "SYRK autotune cache miss; benchmarking per-shape backend "
        f"for shape=({M}, {K}), dtype={dtype}, device={device}, "
        f"has_C={has_C}, backend={backend}. This is expected on the first "
        "optimizer step and can take noticeably longer than steady state."
    )
    started_at = _time.perf_counter()

    X = torch.randn(M, K, device=device, dtype=dtype)
    D = torch.empty(M, M, device=device, dtype=dtype)

    # cuBLAS baseline
    _progress_log(
        "SYRK autotune baseline started "
        f"for shape=({M}, {K}), dtype={dtype}, backend={backend}, has_C={has_C}"
    )
    if has_C:
        C_mat = torch.randn(M, M, device=device, dtype=dtype)
        t_cublas = _bench_median(lambda: torch.addmm(C_mat, X, X.T, alpha=0.5, beta=0.3))
    else:
        t_cublas = _bench_median(lambda: torch.mm(X, X.T, out=D))
    _progress_log(
        "SYRK autotune baseline finished "
        f"for shape=({M}, {K}), dtype={dtype}, backend={backend}, "
        f"cuBLAS={t_cublas*1e6:.0f}us"
    )

    best_time = t_cublas
    best_config = None

    eligible_configs = [
        (tile_m, tile_k, num_stages)
        for tile_m, tile_k, num_stages in _SYRK_CONFIGS
        if M % tile_m == 0
    ]
    if not eligible_configs:
        _progress_log(
            "SYRK autotune has no eligible tile configs "
            f"for shape=({M}, {K}), dtype={dtype}, backend={backend}; "
            "using cuBLAS fallback"
        )

    for idx, (tile_m, tile_k, num_stages) in enumerate(eligible_configs, start=1):
        _progress_log(
            "SYRK autotune candidate started "
            f"{idx}/{len(eligible_configs)} for shape=({M}, {K}), "
            f"dtype={dtype}, backend={backend}, "
            f"tile_m={tile_m}, tile_k={tile_k}, num_stages={num_stages}"
        )
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
            _progress_log(
                "SYRK autotune candidate finished "
                f"{idx}/{len(eligible_configs)} for shape=({M}, {K}), "
                f"dtype={dtype}, backend={backend}, "
                f"tile_m={tile_m}, tile_k={tile_k}, num_stages={num_stages}, "
                f"time={t*1e6:.0f}us"
            )
            if t < best_time:
                best_time = t
                best_config = (tile_m, tile_k, num_stages)
        except Exception as exc:
            _progress_log(
                "SYRK autotune candidate failed "
                f"{idx}/{len(eligible_configs)} for shape=({M}, {K}), "
                f"dtype={dtype}, backend={backend}, "
                f"tile_m={tile_m}, tile_k={tile_k}, num_stages={num_stages}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

    speedup = t_cublas / best_time if best_config else 1.0
    elapsed = _time.perf_counter() - started_at
    _progress_log(
        "SYRK autotune finished "
        f"for shape=({M}, {K}), dtype={dtype}, backend={backend}: "
        f"cuBLAS={t_cublas*1e6:.0f}us, best={best_time*1e6:.0f}us, "
        f"config={best_config}, speedup={speedup:.2f}x, "
        f"elapsed={elapsed:.3f}s"
    )
    logger.info(
        f"SYRK autotune ({M},{K}) has_C={has_C}: "
        f"cuBLAS={t_cublas*1e6:.0f}us, best={best_time*1e6:.0f}us "
        f"config={best_config} speedup={speedup:.2f}x"
    )

    _syrk_autotune_cache[key] = best_config
    _save_autotune_cache(backend)
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
