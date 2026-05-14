"""Paired in-process TP overlap loss alignment check.

This harness isolates true sync-vs-async overlap differences from the
same-mode cross-``torchrun`` noise that can appear in the full LLM path.
It builds two identical models in one process group, feeds them identical
batches, and compares loss plus owned-data digests after every step.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dmuon
from dmuon.utils import wait_all_replicate_broadcasts
from test_tp_alignment import (  # noqa: E402
    _build_model,
    _compute_loss,
    _make_inputs,
)
from tp_profile_utils import iter_dedicated_params  # noqa: E402


class _IdentityNewtonSchulz(dmuon.NewtonSchulz):
    """Deterministic communication-only backend for paired overlap tests."""

    def __init__(self) -> None:
        super().__init__(backend="direct", coefficients=[])

    def local(self, G: torch.Tensor, steps: int) -> torch.Tensor:
        return G


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


def _make_optimizer(model: torch.nn.Module, mode: str) -> dmuon.Muon:
    if mode not in {"sync", "async", "async_drain"}:
        raise ValueError(f"unsupported mode: {mode!r}")
    ns_backend_name = os.environ.get("DMUON_PAIRED_NS_BACKEND", "gram")
    ns_kernel = os.environ.get("DMUON_PAIRED_NS_KERNEL")
    if ns_backend_name == "identity":
        ns_backend = _IdentityNewtonSchulz()
    elif ns_backend_name == "direct_norm":
        ns_backend = dmuon.NewtonSchulz(backend="direct", coefficients=[])
    elif ns_kernel:
        ns_backend = dmuon.NewtonSchulz(backend=ns_backend_name, kernel=ns_kernel)
    else:
        ns_backend = ns_backend_name
    return dmuon.Muon(
        model,
        lr=_env_float("DMUON_PAIRED_LR", 0.02),
        momentum=_env_float("DMUON_PAIRED_MOMENTUM", 0.95),
        weight_decay=_env_float("DMUON_PAIRED_WEIGHT_DECAY", 0.01),
        adamw_lr=_env_float("DMUON_PAIRED_ADAMW_LR", 1e-3),
        ns_backend=ns_backend,
        replicate_async=mode in {"async", "async_drain"},
    )


def _owned_digest(optimizer: dmuon.Muon, device: torch.device) -> float:
    torch.cuda.synchronize()
    total = torch.zeros((), device=device, dtype=torch.float64)
    for dp in optimizer._dedicated_params:
        data = getattr(dp, "_owned_data", None)
        if data is not None:
            total += data.float().sum().double()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return float(total.item())


def _param_digest(model: torch.nn.Module, device: torch.device) -> dict[str, float]:
    torch.cuda.synchronize()
    out: dict[str, float] = {}
    for idx, dp in enumerate(iter_dedicated_params(model)):
        val = torch.zeros((), device=device, dtype=torch.float64)
        data = getattr(dp, "_owned_data", None)
        if data is not None:
            val += data.float().sum().double()
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        full_shape = tuple(int(x) for x in getattr(dp, "full_shape", ()))
        shard_dim = getattr(dp, "shard_dim", None)
        owner = getattr(dp, "_tp_owner_local_rank", None)
        name = str(getattr(dp, "param_name", "<unknown>"))
        key = f"{idx:04d}:{name}:shape={full_shape}:shard={shard_dim}:owner={owner}"
        out[key] = float(val.item())
    return out


def _grad_digest(optimizer: dmuon.Muon, device: torch.device) -> dict[str, float]:
    """Digest the Muon input gradients after reduce/TP-gather, before update."""
    torch.cuda.synchronize()
    out: dict[str, float] = {}
    for idx, dp in enumerate(optimizer._dedicated_params):
        is_tp = bool(getattr(dp, "tp_group", None) is not None)
        if is_tp and getattr(dp, "is_tp_owner", False):
            data = getattr(dp, "_tp_full_grad", None)
        else:
            data = getattr(dp, "_reduced_grad", None)
        val = torch.zeros((), device=device, dtype=torch.float64)
        if data is not None:
            val += data.float().sum().double()
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        full_shape = tuple(int(x) for x in getattr(dp, "full_shape", ()))
        shard_dim = getattr(dp, "shard_dim", None)
        owner = getattr(dp, "_tp_owner_local_rank", None)
        name = str(getattr(dp, "param_name", "<unknown>"))
        key = f"{idx:04d}:{name}:shape={full_shape}:shard={shard_dim}:owner={owner}"
        out[key] = float(val.item())
    return out


def _max_param_gap(
    left: dict[str, float],
    right: dict[str, float],
) -> tuple[float, str, float, float]:
    max_item = (0.0, "", 0.0, 0.0)
    for key, lval in left.items():
        rval = right.get(key)
        if rval is None:
            return float("inf"), key, lval, float("nan")
        if not math.isfinite(float(lval)) or not math.isfinite(float(rval)):
            return float("inf"), key, float(lval), float(rval)
        gap = abs(float(lval) - float(rval))
        if gap > max_item[0]:
            max_item = (gap, key, float(lval), float(rval))
    return max_item


def _run_step(
    model: torch.nn.Module,
    optimizer: dmuon.Muon,
    batch: Any,
    model_kind: str,
    mode: str,
) -> tuple[torch.Tensor, dict[str, float] | None]:
    optimizer.zero_grad()
    loss = _compute_loss(model, batch, model_kind)
    loss.backward()
    grad_digest = None
    if _env_bool("DMUON_PAIRED_RECORD_GRADS", False):
        optimizer._ensure_grads_ready()
        grad_digest = _grad_digest(optimizer, next(model.parameters()).device)
    optimizer.step()
    if mode == "async_drain":
        torch.cuda.synchronize()
    return loss.detach(), grad_digest


def main() -> int:
    topology = os.environ.get("DMUON_PAIRED_TOPOLOGY", "tp4")
    model_kind = os.environ.get("DMUON_PAIRED_MODEL", "llama")
    default_scope = "mlp" if model_kind == "tiny" else "full"
    tp_scope = os.environ.get("DMUON_PAIRED_TP_SCOPE", default_scope)
    left_mode = os.environ.get("DMUON_PAIRED_LEFT_MODE", "sync")
    right_mode = os.environ.get("DMUON_PAIRED_RIGHT_MODE", "async")
    steps = _env_int("DMUON_PAIRED_STEPS", 4)
    loss_tol = _env_float("DMUON_PAIRED_LOSS_TOL", 1e-7)
    digest_tol = _env_float("DMUON_PAIRED_DIGEST_TOL", 1e-5)
    param_tol = _env_float("DMUON_PAIRED_PARAM_TOL", 1e-5)
    out_path = os.environ.get("DMUON_PAIRED_OUT")

    if _env_bool("DMUON_PAIRED_DETERMINISTIC", False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    left_model, model_cfg = _build_model(
        topology,
        world_size=world_size,
        device=device,
        model_kind=model_kind,
        tp_scope=tp_scope,
    )
    right_model, right_cfg = _build_model(
        topology,
        world_size=world_size,
        device=device,
        model_kind=model_kind,
        tp_scope=tp_scope,
    )
    if model_cfg != right_cfg:
        raise RuntimeError(f"paired model configs differ: {model_cfg} vs {right_cfg}")

    left_opt = _make_optimizer(left_model, left_mode)
    right_opt = _make_optimizer(right_model, right_mode)
    batches = _make_inputs(
        model_kind=model_kind,
        model_cfg=model_cfg,
        steps=steps,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    failed = False
    for step, batch in enumerate(batches):
        left_loss, left_grad = _run_step(
            left_model, left_opt, batch, model_kind, left_mode
        )
        right_loss, right_grad = _run_step(
            right_model, right_opt, batch, model_kind, right_mode
        )

        left_digest = _owned_digest(left_opt, device)
        right_digest = _owned_digest(right_opt, device)
        left_params = _param_digest(left_model, device)
        right_params = _param_digest(right_model, device)
        param_gap, param_key, param_left, param_right = _max_param_gap(
            left_params, right_params
        )
        grad_gap = 0.0
        grad_key = ""
        grad_left = 0.0
        grad_right = 0.0
        if left_grad is not None and right_grad is not None:
            grad_gap, grad_key, grad_left, grad_right = _max_param_gap(
                left_grad, right_grad
            )

        loss_gap_tensor = (left_loss.float() - right_loss.float()).abs()
        dist.all_reduce(loss_gap_tensor, op=dist.ReduceOp.MAX)
        loss_gap = float(loss_gap_tensor.item())
        digest_gap = abs(left_digest - right_digest)
        finite = (
            math.isfinite(float(left_loss.item()))
            and math.isfinite(float(right_loss.item()))
            and math.isfinite(loss_gap)
            and math.isfinite(left_digest)
            and math.isfinite(right_digest)
            and math.isfinite(digest_gap)
            and math.isfinite(param_gap)
        )
        step_failed = (
            not finite
            or loss_gap > loss_tol
            or digest_gap > digest_tol
            or param_gap > param_tol
        )
        failed = failed or step_failed

        row = {
            "step": step,
            "left_loss": float(left_loss.item()),
            "right_loss": float(right_loss.item()),
            "loss_gap": loss_gap,
            "left_digest": left_digest,
            "right_digest": right_digest,
            "digest_gap": digest_gap,
            "param_gap": param_gap,
            "param_key": param_key,
            "param_left": param_left,
            "param_right": param_right,
            "grad_gap": grad_gap,
            "grad_key": grad_key,
            "grad_left": grad_left,
            "grad_right": grad_right,
            "failed": step_failed,
        }
        rows.append(row)
        if rank == 0:
            status = "FAIL" if step_failed else "PASS"
            print(
                f"[paired {left_mode} vs {right_mode}] step={step} "
                f"loss_gap={loss_gap:.6e} digest_gap={digest_gap:.6e} "
                f"param_gap={param_gap:.6e} {status}",
                flush=True,
            )

    wait_all_replicate_broadcasts(left_model)
    wait_all_replicate_broadcasts(right_model)
    torch.cuda.synchronize()

    if rank == 0:
        payload = {
            "topology": topology,
            "model": model_kind,
            "model_config": model_cfg,
            "tp_scope": tp_scope,
            "left_mode": left_mode,
            "right_mode": right_mode,
            "world_size": world_size,
            "steps": steps,
            "loss_tol": loss_tol,
            "digest_tol": digest_tol,
            "param_tol": param_tol,
            "tp_scatter_async": os.environ.get("DMUON_TP_SCATTER_ASYNC", "0"),
            "ns_backend": os.environ.get("DMUON_PAIRED_NS_BACKEND", "gram"),
            "ns_kernel": os.environ.get("DMUON_PAIRED_NS_KERNEL", ""),
            "rows": rows,
            "passed": not failed,
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        if out_path:
            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n")
            print(f"wrote {path}", flush=True)
        else:
            print(text, flush=True)

    dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
