"""Phase A alignment instrument — records the loss trajectory of one run.

Reads ``DMUON_ALIGN_MODE ∈ {sync, async, async_drain}``,
``DMUON_ALIGN_RUN`` (integer run id; used only in the output-file name),
and ``DMUON_ALIGN_OUT`` (path for the per-rank JSON dump).

Builds the 3D ``(R=2, G=2, T=2)`` HSDP × TP mesh exactly as in
``test_3d_mesh.py``, runs 3 Muon.step iterations with fixed seeds, and
dumps ``{rank, mode, run_id, losses: [...]}`` from rank 0.

The parent bash wrapper runs this 4-6 times (2 modes × 2 runs, + optional
async_drain variant), then diffs the JSONs to compute:

  * sync_self       = max_i |loss_sync_r1[i] - loss_sync_r2[i]|
  * async_self      = max_i |loss_async_r1[i] - loss_async_r2[i]|
  * sync_async_gap  = max_i |loss_sync_r1[i] - loss_async_r1[i]|

Decision rule (per ``docs/internal/research/tp_alignment_plan.md``):

  * sync_self ≈ async_self ≈ sync_async_gap  → all noise, doc L3 and stop
  * sync_async_gap >> sync_self              → genuine async bug, Phase B
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import dmuon
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


class MLP(nn.Module):
    def __init__(self, h=256, inter=1024):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, h=256, inter=1024):
        super().__init__()
        self.ln = nn.LayerNorm(h)
        self.mlp = MLP(h, inter)

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class Tiny(nn.Module):
    def __init__(self, num_layers=2, h=256, inter=1024):
        super().__init__()
        self.layers = nn.ModuleList([Block(h, inter) for _ in range(num_layers)])
        self.out = nn.Linear(h, h, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x).mean()


def main() -> int:
    mode = os.environ.get("DMUON_ALIGN_MODE", "sync")
    run_id = os.environ.get("DMUON_ALIGN_RUN", "0")
    out_dir = os.environ.get("DMUON_ALIGN_OUT", "/tmp/dmuon_align")

    assert mode in (
        "sync", "async", "async_drain", "sync_nowait",
        "async_tp_only", "sync_tp_only", "async_nopin",
    ), f"unknown DMUON_ALIGN_MODE={mode!r}"

    if mode == "async_nopin":
        # Hypothesis test: make async NOT pin recv_shards in the
        # TPScatterState (rely solely on recv_shard.record_stream to
        # keep the allocator from reclaiming them).  If this matches
        # sync, the pin itself was the divergence cause (allocator
        # churn moves subsequent NCCL transport buffers).
        from dmuon._backends.fsdp2 import group as _group_mod

        def _tp_scatter_delta_async_nopin(self) -> None:
            if self._tp_sync_fallback:
                self.tp_scatter_delta()
                return
            if self._tp_scatter_state is not None:
                raise RuntimeError(
                    "async_nopin: previous event still pending"
                )
            recv_shards = self._tp_scatter_dispatch()
            if recv_shards is None:
                return
            event = self.comm_ctx.replicate_broadcast_stream.record_event()
            # Pin an EMPTY recv_shards list instead of the real tensors.
            # record_stream on each recv_shard should already be enough
            # for allocator safety.
            self._tp_scatter_state = _group_mod.TPScatterState(
                recv_shards=[], event=event,
            )
        _group_mod.DedicatedParamGroup.tp_scatter_delta_async = (
            _tp_scatter_delta_async_nopin
        )

    if mode == "async_tp_only":
        # async TP scatter but SYNC replicate broadcast + wait.  Isolates
        # whether replicate_broadcast_async is the divergence source.
        from dmuon._backends.fsdp2 import group as _group_mod
        from dmuon import utils as _utils_mod

        def _force_sync_replicate(g) -> None:
            scatter_async = getattr(g, "tp_scatter_delta_async", None)
            if scatter_async is not None:
                scatter_async()
            g.replicate_broadcast_sync()
            g.wait_for_replicate_broadcast()
        _utils_mod._dispatch_post_step_async = _force_sync_replicate
    elif mode == "sync_tp_only":
        # Conversely: sync TP scatter + ASYNC replicate broadcast.
        from dmuon import utils as _utils_mod

        def _async_replicate_only(g) -> None:
            scatter = getattr(g, "tp_scatter_delta", None)
            if scatter is not None:
                scatter()
            g.replicate_broadcast_async()
        _utils_mod._dispatch_post_step_async = _async_replicate_only
    if mode == "sync_nowait":
        # Diagnostic: run sync path but strip the wait_stream at the end
        # of tp_scatter_delta.  If the resulting trajectory matches async,
        # that specific wait_stream is the sole cause of sync-vs-async.
        from dmuon._backends.fsdp2 import group as _group_mod

        _orig = _group_mod.DedicatedParamGroup.tp_scatter_delta

        def _tp_scatter_delta_nowait(self) -> None:
            if self._tp_scatter_dispatch() is None:
                return
            # intentionally skip ``current_stream().wait_stream(bcast)``

        _group_mod.DedicatedParamGroup.tp_scatter_delta = (
            _tp_scatter_delta_nowait
        )

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        if rank == 0:
            print(f"SKIP: alignment test needs world=8, got {world_size}")
        dist.destroy_process_group()
        return 0
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    mesh3d = init_device_mesh(
        "cuda", (2, 2, 2), mesh_dim_names=("replicate", "shard", "tp")
    )
    shard_mesh = mesh3d["shard"]
    replicate_mesh = mesh3d["replicate"]
    tp_mesh = mesh3d["tp"]

    # --- Fixed seed for model init so weights are identical across runs ---
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = Tiny(num_layers=2, h=256, inter=1024).to(device)

    plan = {
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }
    for layer in model.layers:
        parallelize_module(layer, tp_mesh, plan)

    dmuon.dedicate_params(
        model, shard_mesh,
        replicate_mesh=replicate_mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )
    dp_mesh_2d = mesh3d["replicate", "shard"]
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh_2d)
    fully_shard(model, mesh=dp_mesh_2d)

    replicate_async = (
        mode in ("async", "async_drain", "async_tp_only", "sync_tp_only",
                 "async_nopin")
    )
    optimizer = dmuon.Muon(
        model, lr=0.02, momentum=0.95, weight_decay=0.01,
        adamw_lr=1e-3, replicate_async=replicate_async,
    )

    # --- Fixed seed for inputs — every rank sees the SAME input per iter
    # (so rank 0's loss truly reflects one trajectory, independent of
    # pseudo-random input generation). ---
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    inputs = [torch.randn(4, 16, 256, device=device) for _ in range(3)]

    losses: list[float] = []
    weight_digests: list[float] = []

    def _digest() -> float:
        """Rank 0 owned-data digest: mean() of the first owned DTensor
        DedicatedParam's ``_owned_data``.  Must be called with the compute
        stream fully synchronised so any pending async NCCL has landed.
        Returns NaN on ranks that own nothing (they dump no signal)."""
        torch.cuda.synchronize()
        if not optimizer._dedicated_params:
            return float("nan")
        return optimizer._dedicated_params[0]._owned_data.float().mean().item()

    for it, x in enumerate(inputs):
        optimizer.zero_grad()
        loss = model(x)
        loss.backward()
        optimizer.step()
        if mode == "async_drain":
            torch.cuda.synchronize()
        # Diagnostic: owned-data digest AFTER step completes (for async,
        # includes a full sync to force the scatter/broadcast to land).
        digest = _digest()
        losses.append(float(loss.item()))
        weight_digests.append(digest)
        if rank == 0:
            print(f"[{mode} run={run_id}] iter {it}: loss={loss.item():.10f} "
                  f"owned0_mean={digest:.12e}", flush=True)
    torch.cuda.synchronize()

    # Drain any pending async state before teardown, otherwise
    # destroy_process_group leaks warning.
    from dmuon.utils import wait_all_replicate_broadcasts  # noqa: F401
    try:
        from dmuon.utils import wait_all_replicate_broadcasts as _drain
        _drain(model)
    except Exception:
        pass

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{mode}_{run_id}.json")
        with open(out_path, "w") as f:
            json.dump({
                "mode": mode,
                "run_id": run_id,
                "losses": losses,
                "owned0_mean": weight_digests,
            }, f)
        print(f"wrote {out_path}", flush=True)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
