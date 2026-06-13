"""Profiled batched owner-local Muon compute helpers.

This module is intentionally internal.  The public ``NewtonSchulz`` object
keeps its existing single-matrix contract; ``owner_strategy='profiled_ilp'``
uses these helpers to tune and execute same-shape batches.
"""

from __future__ import annotations

import importlib
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from dmuon.optim.newton_schulz import (
    DEFAULT_COEFFICIENTS,
    DEFAULT_RESTART_ITERATIONS,
)


_SCIPY_INSTALL = "pip install scipy"
_TILELANG_INSTALL = "pip install tilelang"


def _import_or_install_hint(module_name: str, install_command: str, dep_label: str) -> None:
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"owner_strategy='profiled_ilp' requires {dep_label}.\n"
            "Install it with:\n"
            f"  {install_command}\n"
            "or install the full profiled_ilp extra with:\n"
            "  pip install 'dmuon[profiled_ilp]'"
        ) from exc


def require_profiled_ilp_dependencies() -> None:
    """Validate hard dependencies for ``owner_strategy='profiled_ilp'``.

    The check is deliberately lazy so the default owner strategies do not
    require SciPy or TileLang to be installed.
    """

    _import_or_install_hint("scipy", _SCIPY_INSTALL, "scipy")
    _import_or_install_hint("tilelang", _TILELANG_INSTALL, "tilelang")


@dataclass
class ProfiledILPConfig:
    """Configuration for profiled owner assignment and batched Muon runtime."""

    max_batch: int = 8
    warmup: int = 5
    repeat: int = 20
    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None
    lr: float = 0.02
    momentum: float = 0.95
    weight_decay: float = 0.0
    nesterov: bool = True
    correctness_rtol: float = 5e-2
    correctness_atol: float = 5e-3
    ilp_time_limit_s: Optional[float] = None
    ilp_mip_rel_gap: Optional[float] = 0.01
    ilp_allow_incumbent: bool = True
    backends: tuple[str, ...] = ("tilelang", "cute_sm80", "cublas")
    measured_timings: Optional[dict[tuple[int, int], dict[int, float]]] = None
    measured_backend_choices: Optional[dict[tuple[int, int], dict[int, str]]] = None
    benchmark_timer: str = "event"
    verbose: bool = True
    profile_distributed_mode: str = "rank0_file"
    profile_rank: int = 0
    profile_cache_path: Optional[str] = None
    profile_cache_wait_timeout_s: float = 7200.0
    profile_cache_poll_s: float = 5.0
    # Additional metadata is carried through to logs/tests without changing the
    # dataclass constructor each time profiling grows a new diagnostic.
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendMeasurement:
    backend: str
    median_ms: float
    mean_ms: float
    samples_ms: tuple[float, ...]
    correct: bool
    max_rel_error: float


def normalize_profiled_ilp_config(config: object = None) -> ProfiledILPConfig:
    if config is None:
        return ProfiledILPConfig()
    if isinstance(config, ProfiledILPConfig):
        return config
    if isinstance(config, dict):
        values = dict(config)
        dtype = values.get("dtype")
        if isinstance(dtype, str):
            aliases = {
                "fp16": torch.float16,
                "float16": torch.float16,
                "half": torch.float16,
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp32": torch.float32,
                "float32": torch.float32,
            }
            key = dtype.strip().lower()
            if key not in aliases:
                raise ValueError(f"unknown profiled_ilp dtype string: {dtype!r}")
            values["dtype"] = aliases[key]
        device = values.get("device")
        if isinstance(device, str):
            values["device"] = torch.device(device)
        backends = values.get("backends")
        if isinstance(backends, str):
            values["backends"] = tuple(
                item.strip() for item in backends.split(",") if item.strip()
            )
        return ProfiledILPConfig(**values)
    raise TypeError(
        "profiled_ilp_config must be None, a dict, or ProfiledILPConfig; "
        f"got {type(config).__name__}"
    )


def _syrk_sm80_fn():
    try:
        from dmuon.kernels.syrk_sm80 import syrk_sm80
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"cute_sm80 backend is not importable: {exc!r}") from exc
    return syrk_sm80


def _cute_syrk_batched(
    a: Tensor,
    d: Tensor,
    *,
    b: Optional[Tensor] = None,
    c: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    diag_add: float = 0.0,
    tile_m: int = 128,
    tile_k: int = 32,
    num_stages: int = 4,
    symmetric: bool = False,
) -> None:
    syrk_sm80 = _syrk_sm80_fn()
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


def _cublas_gram(
    a: Tensor,
    *,
    b: Optional[Tensor] = None,
    c: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    diag_add: float = 0.0,
) -> Tensor:
    bt = a.transpose(-2, -1) if b is None else b.transpose(-2, -1)
    if c is None:
        out = torch.bmm(a, bt)
        if alpha != 1.0:
            out.mul_(alpha)
    else:
        out = torch.baddbmm(c, a, bt, beta=beta, alpha=alpha)
    if diag_add != 0.0:
        out.diagonal(dim1=-2, dim2=-1).add_(diag_add)
    return out


def _cute_gram(
    a: Tensor,
    *,
    b: Optional[Tensor] = None,
    c: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    diag_add: float = 0.0,
    symmetric: bool = False,
    tile_m: int = 128,
    tile_k: int = 32,
    num_stages: int = 4,
) -> Tensor:
    batch, m, _k = a.shape
    out = torch.empty(batch, m, m, device=a.device, dtype=a.dtype)
    _cute_syrk_batched(
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


def _tilelang_module():
    candidates = []
    env_module = os.environ.get("DMUON_TILELANG_NS_MODULE")
    if env_module:
        candidates.append(env_module)
    candidates.extend(["dmuon_tilelang_ns", "solution.ns_solution"])

    errors: list[str] = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc!r}")
            continue
        if hasattr(module, "_batched_gram_ns") or hasattr(module, "batched_ns"):
            return module
        errors.append(f"{module_name}: missing _batched_gram_ns/batched_ns")
    joined = "; ".join(errors) if errors else "no candidates tried"
    raise RuntimeError(
        "TileLang is installed, but DMuon could not find a TileLang batched "
        "Newton-Schulz adapter. Set DMUON_TILELANG_NS_MODULE to a module that "
        f"exports _batched_gram_ns(tensor) or batched_ns(list). Details: {joined}"
    )


def _tilelang_batched_gram_newton_schulz(g: Tensor) -> Tensor:
    # Keep the hard dependency error consistent with profiled_ilp's entry gate.
    _import_or_install_hint("tilelang", _TILELANG_INSTALL, "tilelang")
    if os.environ.get("DMUON_TILELANG_NS_MODULE"):
        module = _tilelang_module()
        if hasattr(module, "_batched_gram_ns"):
            return module._batched_gram_ns(g)
        outs = module.batched_ns([g[idx] for idx in range(g.shape[0])])
        return torch.stack(list(outs), dim=0)

    from dmuon.kernels.tilelang_batched import tilelang_batched_gram_newton_schulz

    return tilelang_batched_gram_newton_schulz(g)


def batched_gram_newton_schulz(
    g: Tensor,
    *,
    backend: str,
    eps: float = 1e-7,
    coefficients: Optional[list[list[float]]] = None,
    restart_iterations: Optional[list[int]] = None,
    tile_m: int = 128,
    tile_k: int = 32,
    num_stages: int = 4,
) -> Tensor:
    """Run Gram Newton-Schulz on a same-shape matrix batch."""

    if g.dim() != 3:
        raise ValueError(f"expected [batch, rows, cols], got {tuple(g.shape)}")
    backend = str(backend).lower()
    if backend == "tilelang":
        if coefficients is not None or restart_iterations is not None:
            raise ValueError(
                "tilelang batched Gram NS currently supports DMuon's default "
                "coefficients/restart schedule only"
            )
        return _tilelang_batched_gram_newton_schulz(g)
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

    def gram(
        a: Tensor,
        *,
        b: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        diag_add: float = 0.0,
        symmetric: bool = False,
    ) -> Tensor:
        if backend == "cublas":
            del symmetric
            return _cublas_gram(
                a,
                b=b,
                c=c,
                alpha=alpha,
                beta=beta,
                diag_add=diag_add,
            )
        if backend == "cute_sm80":
            return _cute_gram(
                a,
                b=b,
                c=c,
                alpha=alpha,
                beta=beta,
                diag_add=diag_add,
                symmetric=symmetric,
                tile_m=tile_m,
                tile_k=tile_k,
                num_stages=num_stages,
            )
        raise ValueError(
            f"unsupported profiled_ilp batch backend {backend!r}; "
            "expected 'tilelang', 'cute_sm80', or 'cublas'"
        )

    r = gram(x)
    q: Optional[Tensor] = None

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


def owner_local_muon_batch_update(
    grad: Tensor,
    owned: Tensor,
    momentum_buffer: Optional[Tensor],
    *,
    backend: str,
    lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
) -> tuple[Tensor, Tensor]:
    """Run the complete owner-local Muon compute for one same-shape batch."""

    if momentum_buffer is None:
        momentum_buffer = grad.clone()
    else:
        momentum_buffer.mul_(momentum).add_(grad)
    ns_input = grad.add(momentum_buffer, alpha=momentum) if nesterov else momentum_buffer
    update = batched_gram_newton_schulz(ns_input, backend=backend)

    rows = int(owned.shape[-2])
    cols = int(owned.shape[-1])
    scale = 0.2 * math.sqrt(max(rows, cols))
    if weight_decay > 0:
        owned.mul_(1.0 - lr * weight_decay)
    owned.add_(update.to(device=owned.device, dtype=owned.dtype), alpha=-lr * scale)
    return owned, momentum_buffer


def _cuda_event_samples(fn: Callable[[], None], repeat: int) -> list[float]:
    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _wall_samples(fn: Callable[[], None], repeat: int) -> list[float]:
    samples: list[float] = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _max_relative_error(a: Tensor, b: Tensor) -> float:
    diff = (a.float() - b.float()).flatten(1)
    denom = b.float().flatten(1).norm(dim=1).clamp_min(1e-12)
    rel = diff.norm(dim=1) / denom
    return float(rel.max().item())


def measure_owner_muon_backend(
    *,
    shape: tuple[int, int],
    batch: int,
    backend: str,
    config: ProfiledILPConfig,
) -> BackendMeasurement:
    """Measure one backend for one shape/batch complete owner-local workload."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "owner_strategy='profiled_ilp' profiling requires CUDA because it "
            "measures owner-local GPU Muon compute."
        )
    device = config.device or torch.device("cuda", torch.cuda.current_device())
    dtype = config.dtype or torch.float16
    rows, cols = shape
    grad = torch.randn(batch, rows, cols, device=device, dtype=dtype)
    owned = torch.randn(batch, rows, cols, device=device, dtype=dtype)
    buf = torch.randn_like(grad)

    ref_owned, _ = owner_local_muon_batch_update(
        grad.clone(),
        owned.clone(),
        buf.clone(),
        backend="cublas",
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )
    cand_owned, _ = owner_local_muon_batch_update(
        grad.clone(),
        owned.clone(),
        buf.clone(),
        backend=backend,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )
    torch.cuda.synchronize()
    max_rel_error = _max_relative_error(cand_owned, ref_owned)
    correct = (
        max_rel_error <= config.correctness_rtol
        or torch.allclose(
            cand_owned.float(),
            ref_owned.float(),
            rtol=config.correctness_rtol,
            atol=config.correctness_atol,
        )
    )

    def workload() -> None:
        owner_local_muon_batch_update(
            grad,
            owned,
            buf,
            backend=backend,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov,
        )

    workload()
    torch.cuda.synchronize()
    for _ in range(max(0, int(config.warmup))):
        workload()
    torch.cuda.synchronize()
    if config.benchmark_timer == "wall":
        samples = _wall_samples(workload, int(config.repeat))
    else:
        samples = _cuda_event_samples(workload, int(config.repeat))
    samples.sort()
    return BackendMeasurement(
        backend=backend,
        median_ms=float(statistics.median(samples)),
        mean_ms=float(statistics.fmean(samples)),
        samples_ms=tuple(float(v) for v in samples),
        correct=bool(correct),
        max_rel_error=float(max_rel_error),
    )
