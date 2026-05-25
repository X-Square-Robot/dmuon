"""Instrumentation test for TP communication ordering.

This is a focused state-machine check, not a performance benchmark.  It
drives one small CausalLM/Tiny training loop while splitting
``Muon.step()`` into its internal phases so we can assert the intended
communication dependencies directly:

1. ``prepare_muon_grads`` waits reduce tails and TP-gathers full gradients
   on TP owners;
2. Muon produces full deltas on TP owners while non-owners wait for scatter;
3. post-step TP scatter pins owner send refs until the scatter event is
   consumed;
4. HSDP replicate broadcast, when present, is pending in async mode;
5. an explicit wait/drain leaves no pending scatter/broadcast state before
   the next forward can read ``_owned_data``.

Environment mirrors ``test_tp_alignment.py``:

* ``DMUON_COMM_TOPOLOGY``: ``tp2``, ``tp4``, ``dp_tp2``, ``dp_tp4``,
  or ``hsdp_tp2``.
* ``DMUON_COMM_MODE``: ``sync`` or ``async``.
* ``DMUON_COMM_MODEL``: ``tiny`` or ``llama``.
* ``DMUON_COMM_STEPS``: optimizer steps, default 2.
* ``DMUON_COMM_OUT``: optional JSON output path.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[2]
TEST_DIST = REPO / "tests" / "distributed"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TEST_DIST))

from test_tp_alignment import (  # noqa: E402
    _build_model,
    _compute_loss,
    _make_inputs,
)

import dmuon  # noqa: E402
from dmuon.utils import (  # noqa: E402
    _ordered_post_step_groups,
    prepare_muon_grads,
    wait_all_replicate_broadcasts,
)


def _groups(model) -> list[Any]:
    seen: set[int] = set()
    out: list[Any] = []
    for module in model.modules():
        state = getattr(module, "_dedicated_state", None)
        group = getattr(state, "group", None)
        if group is None or id(group) in seen:
            continue
        seen.add(id(group))
        out.append(group)
    return out


def _local_stats(model) -> dict[str, int]:
    stats = {
        "tp_params": 0,
        "reduced_grads": 0,
        "tp_full_grads": 0,
        "tp_full_deltas": 0,
        "pending_tp_states": 0,
        "pending_tp_gather_events": 0,
        "pinned_tp_send_refs": 0,
        "pending_muon_grad_ready_events": 0,
        "pending_replicate_states": 0,
        "pending_replicate_events": 0,
    }
    for group in _groups(model):
        if getattr(group, "_muon_grad_ready_event", None) is not None:
            stats["pending_muon_grad_ready_events"] += 1
        if getattr(group, "_tp_gather_event", None) is not None:
            stats["pending_tp_gather_events"] += 1
        tp_state = getattr(group, "_tp_scatter_state", None)
        if tp_state is not None:
            stats["pending_tp_states"] += 1
            stats["pinned_tp_send_refs"] += len(getattr(tp_state, "refs", ()))
        if getattr(group, "_replicate_broadcast_state", None) is not None:
            stats["pending_replicate_states"] += 1
        if getattr(group, "_replicate_broadcast_event", None) is not None:
            stats["pending_replicate_events"] += 1
        for dp in getattr(group, "params", ()):
            if getattr(dp, "tp_group", None) is None:
                continue
            stats["tp_params"] += 1
            if getattr(dp, "_reduced_grad", None) is not None:
                stats["reduced_grads"] += 1
            if getattr(dp, "_tp_full_grad", None) is not None:
                stats["tp_full_grads"] += 1
            if getattr(dp, "_tp_full_delta", None) is not None:
                stats["tp_full_deltas"] += 1
    return stats


def _sum_stats(local: dict[str, int]) -> dict[str, int]:
    keys = sorted(local)
    values = torch.tensor([local[k] for k in keys], device="cuda", dtype=torch.long)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {k: int(v.item()) for k, v in zip(keys, values)}


def _dispatch_post_step_for_observation(model, mode: str) -> None:
    """Dispatch post-step TP/replicate communication without draining it.

    The public sync helper intentionally drains immediately.  This test
    needs the intermediate state, so it mirrors the group-level dispatch
    order and leaves the final wait to ``wait_all_replicate_broadcasts``.
    """
    for group in _ordered_post_step_groups(model):
        if mode == "async":
            scatter_async = getattr(group, "tp_scatter_delta_async", None)
            if scatter_async is not None:
                scatter_async()
            group.replicate_broadcast_async()
        else:
            scatter = getattr(group, "tp_scatter_delta", None)
            if scatter is not None:
                scatter()
            group.replicate_broadcast_sync()


def _assert_no_pending(model) -> None:
    local = _local_stats(model)
    total = _sum_stats(local)
    for key in (
        "reduced_grads",
        "tp_full_grads",
        "tp_full_deltas",
        "pending_tp_gather_events",
        "pending_tp_states",
        "pinned_tp_send_refs",
        "pending_muon_grad_ready_events",
        "pending_replicate_states",
        "pending_replicate_events",
    ):
        if total[key] != 0:
            raise AssertionError(f"expected no pending {key}, got {total}")


def _write_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path or dist.get_rank() != 0:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    topology = os.environ.get("DMUON_COMM_TOPOLOGY", "tp4")
    mode = os.environ.get("DMUON_COMM_MODE", "async")
    model_kind = os.environ.get("DMUON_COMM_MODEL", "llama")
    tp_scope = os.environ.get("DMUON_COMM_TP_SCOPE", "full")
    steps = int(os.environ.get("DMUON_COMM_STEPS", "2"))
    out_path = os.environ.get("DMUON_COMM_OUT")
    if mode not in ("sync", "async"):
        raise ValueError("DMUON_COMM_MODE must be 'sync' or 'async'")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    try:
        model, model_cfg = _build_model(
            topology,
            world_size=world_size,
            device=device,
            model_kind=model_kind,
            tp_scope=tp_scope,
        )
        optimizer = dmuon.Muon(
            model,
            lr=0.02,
            momentum=0.95,
            weight_decay=0.01,
            adamw_lr=1e-3,
            replicate_async=(mode == "async"),
        )
        inputs = _make_inputs(
            model_kind=model_kind,
            model_cfg=model_cfg,
            steps=steps,
            device=device,
        )
        records: list[dict[str, Any]] = []

        _assert_no_pending(model)
        for step, batch in enumerate(inputs):
            optimizer.zero_grad()
            loss = _compute_loss(model, batch, model_kind)
            loss.backward()

            prepare_muon_grads(model)
            after_prepare = _sum_stats(_local_stats(model))
            if after_prepare["tp_params"] == 0:
                raise AssertionError("comm-order test requires TP params")
            if after_prepare["reduced_grads"] == 0:
                raise AssertionError(f"no reduced grads after prepare: {after_prepare}")
            if after_prepare["tp_full_grads"] == 0:
                raise AssertionError(f"no TP full grads after gather: {after_prepare}")
            if after_prepare["pending_tp_gather_events"] != 0:
                raise AssertionError(
                    "prepare_muon_grads should drain TP gather events: "
                    f"{after_prepare}"
                )
            if after_prepare["pending_muon_grad_ready_events"] != 0:
                raise AssertionError(
                    "prepare_muon_grads should drain readiness events: "
                    f"{after_prepare}"
                )
            if after_prepare["tp_full_deltas"] != 0:
                raise AssertionError(
                    f"TP full deltas should not exist before Muon: {after_prepare}"
                )

            optimizer._step_muon()
            after_muon = _sum_stats(_local_stats(model))
            if after_muon["tp_full_deltas"] == 0:
                raise AssertionError(f"Muon did not create TP deltas: {after_muon}")
            if after_muon["reduced_grads"] == 0:
                raise AssertionError(
                    "TP reduced grads must stay until scatter builds its work list: "
                    f"{after_muon}"
                )

            optimizer._step_adamw()
            _dispatch_post_step_for_observation(model, mode)

            after_dispatch = _sum_stats(_local_stats(model))
            if after_dispatch["pending_tp_states"] == 0:
                raise AssertionError(
                    f"TP scatter state missing after post-step dispatch: {after_dispatch}"
                )
            if not _ordered_post_step_groups(model):
                raise AssertionError("post-step group order is empty")
            if after_dispatch["pinned_tp_send_refs"] == 0:
                raise AssertionError(
                    "TP owner send split refs were not pinned through scatter event: "
                    f"{after_dispatch}"
                )
            if mode == "async" and topology == "hsdp_tp2":
                if after_dispatch["pending_replicate_states"] == 0:
                    raise AssertionError(
                        "HSDP async replicate broadcast state missing: "
                        f"{after_dispatch}"
                    )

            wait_all_replicate_broadcasts(model)
            after_drain = _sum_stats(_local_stats(model))
            _assert_no_pending(model)

            records.append(
                {
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "after_prepare": after_prepare,
                    "after_muon": after_muon,
                    "after_dispatch": after_dispatch,
                    "after_drain": after_drain,
                    "post_step_groups": group_summary,
                }
            )
            if rank == 0:
                print(
                    f"[{topology}/{mode}/{model_kind}] step={step} "
                    f"loss={float(loss.detach().item()):.10f} "
                    f"refs={after_dispatch['pinned_tp_send_refs']} "
                    f"replicate_states={after_dispatch['pending_replicate_states']}",
                    flush=True,
                )

        _write_json(
            out_path,
            {
                "topology": topology,
                "mode": mode,
                "model": model_kind,
                "tp_scope": tp_scope,
                "world_size": world_size,
                "records": records,
            },
        )
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
