"""Single-model checkpoint/replay oracle for TP overlap correctness.

The paired two-model LLM oracle can drift even in ``sync`` vs ``sync`` because
two FSDP2/DTensor graphs coexist in one process group.  This harness instead
uses one model, snapshots it, runs one trajectory, restores the snapshot, and
replays a second trajectory under another post-step mode.

Run examples:

    torchrun --nproc_per_node=4 tests/distributed/test_tp_overlap_replay.py

    DMUON_REPLAY_MODEL=llama \
    DMUON_REPLAY_TOPOLOGY=tp4 \
    DMUON_REPLAY_LEFT_MODE=sync \
    DMUON_REPLAY_RIGHT_MODE=async_scatter \
    torchrun --nproc_per_node=4 tests/distributed/test_tp_overlap_replay.py
"""

from __future__ import annotations

import contextlib
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
from dmuon.utils import (
    broadcast_all_updates,
    broadcast_all_updates_async,
    wait_all_replicate_broadcasts,
)
from test_tp_alignment import (  # noqa: E402
    _build_model,
    _compute_loss,
    _make_inputs,
)
from tp_profile_utils import iter_dedicated_params  # noqa: E402


class _IdentityNewtonSchulz(dmuon.NewtonSchulz):
    """Deterministic communication-only backend for replay diagnostics."""

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


def _trace(message: str) -> None:
    if not _env_bool("DMUON_REPLAY_TRACE", False):
        return
    rank = dist.get_rank() if dist.is_initialized() else -1
    line = f"[replay-trace rank={rank}] {message}"
    trace_dir = os.environ.get("DMUON_REPLAY_TRACE_DIR")
    if trace_dir:
        path = Path(trace_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"rank{rank}.log").open("a") as f:
            f.write(line + "\n")
            f.flush()
    if _env_bool("DMUON_REPLAY_TRACE_STDOUT", True):
        print(line, flush=True)


def _make_ns_backend() -> str | dmuon.NewtonSchulz:
    name = os.environ.get("DMUON_REPLAY_NS_BACKEND", "identity")
    kernel = os.environ.get("DMUON_REPLAY_NS_KERNEL")
    if name == "identity":
        return _IdentityNewtonSchulz()
    if name == "direct_norm":
        return dmuon.NewtonSchulz(backend="direct", coefficients=[])
    if kernel:
        return dmuon.NewtonSchulz(backend=name, kernel=kernel)
    return name


def _replicate_async(mode: str) -> bool:
    if mode == "sync":
        return False
    if mode in {"async", "async_fallback", "async_scatter", "async_drain"}:
        return True
    raise ValueError(f"unsupported replay mode: {mode!r}")


def _tp_scatter_async(mode: str) -> bool:
    return mode in {"async_scatter", "async_drain"}


def _drain_after_step(mode: str) -> bool:
    return mode == "async_drain"


def _make_optimizer(model: torch.nn.Module, mode: str) -> dmuon.Muon:
    return dmuon.Muon(
        model,
        lr=_env_float("DMUON_REPLAY_LR", 0.02),
        momentum=_env_float("DMUON_REPLAY_MOMENTUM", 0.95),
        weight_decay=_env_float("DMUON_REPLAY_WEIGHT_DECAY", 0.01),
        adamw_lr=_env_float("DMUON_REPLAY_ADAMW_LR", 1e-3),
        ns_backend=_make_ns_backend(),
        replicate_async=_replicate_async(mode),
    )


@contextlib.contextmanager
def _tp_scatter_env(enabled: bool):
    yield


def _param_key(idx: int, dp: Any) -> str:
    full_shape = tuple(int(x) for x in getattr(dp, "full_shape", ()))
    shard_dim = getattr(dp, "shard_dim", None)
    owner = getattr(dp, "_tp_owner_local_rank", None)
    name = str(getattr(dp, "param_name", "<unknown>"))
    return f"{idx:04d}:{name}:shape={full_shape}:shard={shard_dim}:owner={owner}"


def _digest_tensor(data: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    val = torch.zeros((), device=device, dtype=torch.float64)
    if data is not None:
        val += data.detach().float().sum().double()
    return val


def _allreduce_digest_values(
    entries: list[tuple[str, torch.Tensor | None]],
    device: torch.device,
) -> dict[str, float]:
    """Reduce per-param digests with one world collective.

    The replay oracle intentionally mixes global digest checks with DP/HSDP
    subgroup checkpoint collectives.  Launching one WORLD all-reduce per
    parameter lets faster ranks enter later subgroup collectives while slower
    ranks are still enqueueing WORLD collectives, which can deadlock NCCL
    across overlapping process groups.  Use a single vector all-reduce and a
    barrier to make the oracle's bookkeeping phase a clean boundary.
    """

    if entries:
        vals = torch.stack([
            _digest_tensor(data, device) for _key, data in entries
        ])
    else:
        vals = torch.empty(0, device=device, dtype=torch.float64)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    barrier_kwargs = (
        {"device_ids": [device.index]}
        if device.type == "cuda" and device.index is not None
        else {}
    )
    dist.barrier(**barrier_kwargs)
    return {
        key: float(vals[idx].item())
        for idx, (key, _data) in enumerate(entries)
    }


def _iter_groups(model: torch.nn.Module) -> list[Any]:
    groups: list[Any] = []
    seen: set[int] = set()
    for module in model.modules():
        state = getattr(module, "_dedicated_state", None)
        group = getattr(state, "group", None)
        if group is None or id(group) in seen:
            continue
        seen.add(id(group))
        groups.append(group)
    return groups


def _reset_runtime_state(model: torch.nn.Module) -> None:
    """Return DMuon communication/runtime state to a clean replay boundary."""

    wait_all_replicate_broadcasts(model)
    torch.cuda.synchronize()

    comm_ctx = getattr(model, "_dedicated_comm_ctx", None)
    if comm_ctx is not None:
        comm_ctx.post_backward_final_callback_queued = False
        comm_ctx.reset_post_forward_order()

    for group in _iter_groups(model):
        wait_reduce = getattr(group, "wait_for_reduce", None)
        if wait_reduce is not None:
            wait_reduce()
        pre_wait = getattr(group, "_pre_forward_wait", None)
        if pre_wait is not None:
            pre_wait()
        reshard = getattr(group, "reshard", None)
        if reshard is not None:
            reshard()

        for attr, value in (
            ("_broadcast_event", None),
            ("_post_reduce_event", None),
            ("_replicate_reduce_state", None),
            ("_replicate_broadcast_event", None),
            ("_replicate_broadcast_state", None),
            ("_tp_scatter_state", None),
            ("_pending_reduce", []),
            ("_partial_reduce_by_param", {}),
            ("_post_forward_indices", []),
            ("_post_backward_fired", False),
            ("_last_replicate_wait_us", 0.0),
            ("_last_tp_scatter_wait_us", 0.0),
        ):
            if hasattr(group, attr):
                setattr(group, attr, value.copy() if isinstance(value, (list, dict)) else value)

        for dp in getattr(group, "params", ()):
            for attr in (
                "_reduced_grad",
                "_accumulated_grad",
                "_tp_full_grad",
                "_tp_full_delta",
            ):
                if hasattr(dp, attr):
                    setattr(dp, attr, None)
            orig_param = getattr(dp, "_orig_param", None)
            if orig_param is not None:
                orig_param.grad = None
            unsharded = getattr(dp, "_unsharded_param", None)
            if unsharded is not None:
                unsharded.grad = None

    torch.cuda.synchronize()


def _owned_digest(model: torch.nn.Module, device: torch.device) -> dict[str, float]:
    _trace("owned_digest synchronize begin")
    torch.cuda.synchronize()
    _trace("owned_digest synchronize done")
    params = iter_dedicated_params(model)
    _trace(f"owned_digest param_count={len(params)}")
    entries = [
        (_param_key(idx, dp), getattr(dp, "_owned_data", None))
        for idx, dp in enumerate(params)
    ]
    _trace("owned_digest vector_allreduce begin")
    out = _allreduce_digest_values(entries, device)
    _trace("owned_digest vector_allreduce done")
    return out


def _grad_digest(optimizer: dmuon.Muon, device: torch.device) -> dict[str, float]:
    torch.cuda.synchronize()
    entries: list[tuple[str, torch.Tensor | None]] = []
    for idx, dp in enumerate(iter_dedicated_params(optimizer.model)):
        is_tp = bool(getattr(dp, "tp_group", None) is not None)
        if is_tp and getattr(dp, "is_tp_owner", False):
            data = getattr(dp, "_tp_full_grad", None)
        else:
            data = getattr(dp, "_reduced_grad", None)
        entries.append((_param_key(idx, dp), data))
    return _allreduce_digest_values(entries, device)


def _delta_digest(optimizer: dmuon.Muon, device: torch.device) -> dict[str, float]:
    torch.cuda.synchronize()
    entries: list[tuple[str, torch.Tensor | None]] = []
    for idx, dp in enumerate(iter_dedicated_params(optimizer.model)):
        data = getattr(dp, "_tp_full_delta", None)
        entries.append((_param_key(idx, dp), data))
    return _allreduce_digest_values(entries, device)


def _max_digest_gap(
    left: dict[str, float],
    right: dict[str, float],
) -> tuple[float, str, float, float]:
    max_item = (0.0, "", 0.0, 0.0)
    for key, lval in left.items():
        rval = right.get(key)
        if rval is None:
            return float("inf"), key, float(lval), float("nan")
        if not math.isfinite(float(lval)) or not math.isfinite(float(rval)):
            return float("inf"), key, float(lval), float(rval)
        gap = abs(float(lval) - float(rval))
        if gap > max_item[0]:
            max_item = (gap, key, float(lval), float(rval))
    return max_item


def _snapshot(
    model: torch.nn.Module,
    optimizer: dmuon.Muon,
    *,
    reset_runtime: bool = True,
) -> dict[str, Any]:
    _trace(f"snapshot begin reset_runtime={reset_runtime}")
    if reset_runtime:
        _trace("snapshot reset_runtime begin")
        _reset_runtime_state(model)
        _trace("snapshot reset_runtime done")
    else:
        _trace("snapshot cuda synchronize begin")
        torch.cuda.synchronize()
        _trace("snapshot cuda synchronize done")
    _trace("snapshot model_state begin")
    model_state = dmuon.get_model_state_dict(
        model, cpu_offload=True, rank0_only=False
    )
    _trace("snapshot model_state done")
    _trace("snapshot optimizer_state begin")
    optimizer_state = dmuon.get_optimizer_state_dict(
        model, optimizer, cpu_offload=True, rank0_only=False
    )
    _trace("snapshot optimizer_state done")
    snap = {
        "model": model_state,
        "optimizer": optimizer_state,
        "cpu_rng": torch.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state().clone(),
    }
    _trace("snapshot done")
    return snap


def _restore(
    model: torch.nn.Module,
    optimizer: dmuon.Muon,
    snapshot: dict[str, Any],
) -> None:
    _trace("restore begin")
    _trace("restore reset_runtime begin")
    _reset_runtime_state(model)
    _trace("restore reset_runtime done")
    _trace("restore model_state begin")
    dmuon.set_model_state_dict(model, snapshot["model"])
    _trace("restore model_state done")
    optimizer.state.clear()
    _trace("restore optimizer_state begin")
    dmuon.set_optimizer_state_dict(model, optimizer, snapshot["optimizer"])
    _trace("restore optimizer_state done")
    torch.set_rng_state(snapshot["cpu_rng"])
    torch.cuda.set_rng_state(snapshot["cuda_rng"])
    optimizer.zero_grad()
    _trace("restore final reset_runtime begin")
    _reset_runtime_state(model)
    _trace("restore final reset_runtime done")
    torch.cuda.synchronize()
    _trace("restore done")


def _clone_dp_state(model: torch.nn.Module) -> dict[str, dict[str, torch.Tensor | None]]:
    state: dict[str, dict[str, torch.Tensor | None]] = {}
    for idx, dp in enumerate(iter_dedicated_params(model)):
        entry: dict[str, torch.Tensor | None] = {}
        for attr in ("_reduced_grad", "_tp_full_grad", "_tp_full_delta"):
            val = getattr(dp, attr, None)
            entry[attr] = None if val is None else val.detach().clone()
        state[_param_key(idx, dp)] = entry
    return state


def _restore_dp_state(
    model: torch.nn.Module,
    state: dict[str, dict[str, torch.Tensor | None]],
) -> None:
    for idx, dp in enumerate(iter_dedicated_params(model)):
        entry = state.get(_param_key(idx, dp), {})
        for attr in ("_reduced_grad", "_tp_full_grad", "_tp_full_delta"):
            val = entry.get(attr)
            setattr(dp, attr, None if val is None else val.to(dp.device).clone())


def _restore_pre_step_state(
    model: torch.nn.Module,
    optimizer: dmuon.Muon,
    snapshot: dict[str, Any],
    dp_state: dict[str, dict[str, torch.Tensor | None]],
) -> None:
    _restore(model, optimizer, snapshot)
    _restore_dp_state(model, dp_state)
    optimizer._grads_ready = True
    torch.cuda.synchronize()


@torch.no_grad()
def _manual_update_ready(
    model: torch.nn.Module,
    optimizer: dmuon.Muon,
    mode: str,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    _trace(f"manual_update mode={mode} grad_digest begin")
    grad = _grad_digest(optimizer, device)
    _trace(f"manual_update mode={mode} grad_digest done")

    _trace(f"manual_update mode={mode} step_muon begin")
    optimizer._step_muon()
    _trace(f"manual_update mode={mode} step_muon done")
    _trace(f"manual_update mode={mode} delta_digest begin")
    delta = _delta_digest(optimizer, device)
    _trace(f"manual_update mode={mode} delta_digest done")

    _trace(f"manual_update mode={mode} adamw begin")
    optimizer._step_adamw()
    _trace(f"manual_update mode={mode} adamw done")
    if _replicate_async(mode):
        _trace(f"manual_update mode={mode} broadcast_async begin")
        broadcast_all_updates_async(model)
        _trace(f"manual_update mode={mode} broadcast_async done")
    else:
        _trace(f"manual_update mode={mode} broadcast_sync begin")
        broadcast_all_updates(model)
        _trace(f"manual_update mode={mode} broadcast_sync done")
    if _drain_after_step(mode):
        _trace(f"manual_update mode={mode} drain begin")
        wait_all_replicate_broadcasts(model)
        _trace(f"manual_update mode={mode} drain done")
    optimizer._grads_ready = False

    _trace(f"manual_update mode={mode} owned_digest begin")
    post_scatter_owned = _owned_digest(model, device)
    _trace(f"manual_update mode={mode} owned_digest done")
    return {
        "pre_step_grad": grad,
        "post_muon_delta": delta,
        "post_scatter_owned": post_scatter_owned,
    }


def _manual_optimizer_step(
    model: torch.nn.Module,
    optimizer: dmuon.Muon,
    mode: str,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    optimizer._ensure_grads_ready()
    return _manual_update_ready(model, optimizer, mode, device)


def _run_trajectory(
    model: torch.nn.Module,
    optimizer: dmuon.Muon,
    batches: list[Any],
    *,
    model_kind: str,
    mode: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _tp_scatter_env(_tp_scatter_async(mode)):
        for step, batch in enumerate(batches):
            optimizer.zero_grad()
            loss = _compute_loss(model, batch, model_kind)
            loss.backward()
            stage_digests = _manual_optimizer_step(model, optimizer, mode, device)
            rows.append({
                "step": step,
                "loss": float(loss.detach().float().item()),
                "stages": stage_digests,
            })
    wait_all_replicate_broadcasts(model)
    torch.cuda.synchronize()
    return rows


def _run_pre_step_replay(
    model: torch.nn.Module,
    optimizer: dmuon.Muon,
    batches: list[Any],
    *,
    model_kind: str,
    left_mode: str,
    right_mode: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left_rows: list[dict[str, Any]] = []
    right_rows: list[dict[str, Any]] = []
    for step, batch in enumerate(batches):
        _trace(f"pre_step_replay step={step} begin")
        optimizer.zero_grad()
        _trace(f"pre_step_replay step={step} forward begin")
        loss = _compute_loss(model, batch, model_kind)
        _trace(f"pre_step_replay step={step} forward done")
        _trace(f"pre_step_replay step={step} backward begin")
        loss.backward()
        _trace(f"pre_step_replay step={step} backward done")
        _trace(f"pre_step_replay step={step} ensure_grads begin")
        optimizer._ensure_grads_ready()
        _trace(f"pre_step_replay step={step} ensure_grads done")

        _trace(f"pre_step_replay step={step} pre_step_snapshot begin")
        pre_step_snapshot = _snapshot(model, optimizer, reset_runtime=False)
        _trace(f"pre_step_replay step={step} pre_step_snapshot done")
        pre_step_dp_state = _clone_dp_state(model)

        with _tp_scatter_env(_tp_scatter_async(left_mode)):
            _trace(f"pre_step_replay step={step} left update begin")
            left_stages = _manual_update_ready(model, optimizer, left_mode, device)
            _trace(f"pre_step_replay step={step} left update done")
        left_rows.append({
            "step": step,
            "loss": float(loss.detach().float().item()),
            "stages": left_stages,
        })
        _trace(f"pre_step_replay step={step} left_final_snapshot begin")
        left_final_snapshot = _snapshot(model, optimizer)
        _trace(f"pre_step_replay step={step} left_final_snapshot done")

        _trace(f"pre_step_replay step={step} restore_pre_step begin")
        _restore_pre_step_state(
            model, optimizer, pre_step_snapshot, pre_step_dp_state
        )
        _trace(f"pre_step_replay step={step} restore_pre_step done")
        with _tp_scatter_env(_tp_scatter_async(right_mode)):
            _trace(f"pre_step_replay step={step} right update begin")
            right_stages = _manual_update_ready(model, optimizer, right_mode, device)
            _trace(f"pre_step_replay step={step} right update done")
        right_rows.append({
            "step": step,
            "loss": float(loss.detach().float().item()),
            "stages": right_stages,
        })

        _trace(f"pre_step_replay step={step} restore_left_final begin")
        _restore(model, optimizer, left_final_snapshot)
        _trace(f"pre_step_replay step={step} restore_left_final done")
    return left_rows, right_rows


def _compare_rows(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    loss_tol: float,
    digest_tol: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    failed = False
    stage_names = ("pre_step_grad", "post_muon_delta", "post_scatter_owned")
    for left, right in zip(left_rows, right_rows):
        loss_gap_tensor = torch.tensor(
            abs(float(left["loss"]) - float(right["loss"])),
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(loss_gap_tensor, op=dist.ReduceOp.MAX)
        loss_gap = float(loss_gap_tensor.item())

        stage_gaps: dict[str, dict[str, Any]] = {}
        max_stage_gap = 0.0
        first_stage = ""
        for stage in stage_names:
            gap, key, lval, rval = _max_digest_gap(
                left["stages"][stage],
                right["stages"][stage],
            )
            stage_gaps[stage] = {
                "gap": gap,
                "key": key,
                "left": lval,
                "right": rval,
            }
            if gap > max_stage_gap:
                max_stage_gap = gap
                first_stage = stage

        finite = math.isfinite(loss_gap) and math.isfinite(max_stage_gap)
        row_failed = (
            not finite
            or loss_gap > loss_tol
            or max_stage_gap > digest_tol
        )
        failed = failed or row_failed
        rows.append({
            "step": int(left["step"]),
            "left_loss": left["loss"],
            "right_loss": right["loss"],
            "loss_gap": loss_gap,
            "max_stage_gap": max_stage_gap,
            "first_stage": first_stage,
            "stage_gaps": stage_gaps,
            "failed": row_failed,
        })
    return rows, failed


def _compare_digest_stage(
    left: dict[str, float],
    right: dict[str, float],
) -> dict[str, Any]:
    gap, key, lval, rval = _max_digest_gap(left, right)
    return {
        "gap": gap,
        "key": key,
        "left": lval,
        "right": rval,
    }


def main() -> int:
    topology = os.environ.get("DMUON_REPLAY_TOPOLOGY", "tp4")
    model_kind = os.environ.get("DMUON_REPLAY_MODEL", "tiny")
    default_scope = "mlp" if model_kind == "tiny" else "full"
    tp_scope = os.environ.get("DMUON_REPLAY_TP_SCOPE", default_scope)
    left_mode = os.environ.get("DMUON_REPLAY_LEFT_MODE", "sync")
    right_mode = os.environ.get("DMUON_REPLAY_RIGHT_MODE", "sync")
    snapshot_stage = os.environ.get("DMUON_REPLAY_SNAPSHOT_STAGE", "pre_step")
    steps = _env_int("DMUON_REPLAY_STEPS", 3)
    loss_tol = _env_float("DMUON_REPLAY_LOSS_TOL", 1e-9)
    digest_tol = _env_float("DMUON_REPLAY_DIGEST_TOL", 1e-9)
    out_path = os.environ.get("DMUON_REPLAY_OUT")

    if (
        _env_bool("DMUON_REPLAY_DETERMINISTIC", False)
        or _env_bool("DMUON_REPLAY_DISABLE_TF32", False)
    ):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if _env_bool("DMUON_REPLAY_DETERMINISTIC", False):
        torch.use_deterministic_algorithms(True, warn_only=True)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    _trace(
        f"main init topology={topology} model={model_kind} "
        f"left={left_mode} right={right_mode} world={world_size}"
    )

    _trace("main build_model begin")
    model, model_cfg = _build_model(
        topology,
        world_size=world_size,
        device=device,
        model_kind=model_kind,
        tp_scope=tp_scope,
    )
    _trace("main build_model done")
    snapshot_opt = _make_optimizer(model, left_mode)
    _trace("main initial_snapshot begin")
    snap = _snapshot(model, snapshot_opt)
    _trace("main initial_snapshot done")
    del snapshot_opt

    _trace("main make_inputs begin")
    batches = _make_inputs(
        model_kind=model_kind,
        model_cfg=model_cfg,
        steps=steps,
        device=device,
    )
    _trace("main make_inputs done")

    if snapshot_stage == "pre_step":
        optimizer = _make_optimizer(model, left_mode)
        _trace("main restore_initial begin")
        _restore(model, optimizer, snap)
        _trace("main restore_initial done")
        left_initial_owned = _owned_digest(model, device)
        right_initial_owned = dict(left_initial_owned)
        left_rows, right_rows = _run_pre_step_replay(
            model,
            optimizer,
            batches,
            model_kind=model_kind,
            left_mode=left_mode,
            right_mode=right_mode,
            device=device,
        )
    elif snapshot_stage == "initial":
        left_opt = _make_optimizer(model, left_mode)
        _restore(model, left_opt, snap)
        left_initial_owned = _owned_digest(model, device)
        left_rows = _run_trajectory(
            model, left_opt, batches, model_kind=model_kind, mode=left_mode, device=device
        )
        del left_opt

        right_opt = _make_optimizer(model, right_mode)
        _restore(model, right_opt, snap)
        right_initial_owned = _owned_digest(model, device)
        right_rows = _run_trajectory(
            model, right_opt, batches, model_kind=model_kind, mode=right_mode, device=device
        )
    else:
        raise ValueError(
            "DMUON_REPLAY_SNAPSHOT_STAGE must be 'pre_step' or 'initial', "
            f"got {snapshot_stage!r}"
        )

    comparison, failed = _compare_rows(
        left_rows,
        right_rows,
        loss_tol=loss_tol,
        digest_tol=digest_tol,
        device=device,
    )
    initial_owned_gap = _compare_digest_stage(left_initial_owned, right_initial_owned)
    if (
        not math.isfinite(float(initial_owned_gap["gap"]))
        or float(initial_owned_gap["gap"]) > digest_tol
    ):
        failed = True

    if rank == 0:
        print(
            f"[replay {left_mode} vs {right_mode}] initial_owned_gap="
            f"{initial_owned_gap['gap']:.6e} key={initial_owned_gap['key']}",
            flush=True,
        )
        for row in comparison:
            status = "FAIL" if row["failed"] else "PASS"
            print(
                f"[replay {left_mode} vs {right_mode}] step={row['step']} "
                f"loss_gap={row['loss_gap']:.6e} "
                f"max_stage_gap={row['max_stage_gap']:.6e} "
                f"first_stage={row['first_stage']} {status}",
                flush=True,
            )
        payload = {
            "topology": topology,
            "model": model_kind,
            "model_config": model_cfg,
            "tp_scope": tp_scope,
            "left_mode": left_mode,
            "right_mode": right_mode,
            "snapshot_stage": snapshot_stage,
            "world_size": world_size,
            "steps": steps,
            "loss_tol": loss_tol,
            "digest_tol": digest_tol,
            "tp_scatter_async": {
                "left": _tp_scatter_async(left_mode),
                "right": _tp_scatter_async(right_mode),
            },
            "ns_backend": os.environ.get("DMUON_REPLAY_NS_BACKEND", "identity"),
            "ns_kernel": os.environ.get("DMUON_REPLAY_NS_KERNEL", ""),
            "initial_owned_gap": initial_owned_gap,
            "rows": comparison,
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

    wait_all_replicate_broadcasts(model)
    torch.cuda.synchronize()
    dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
