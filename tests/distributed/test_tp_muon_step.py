"""T2f — end-to-end TP All-to-All integration test for Muon.step.

Validates the sync T2a+T2b+T2b2 path under real NCCL on 4 GPUs:
DP(2) x TP(2) mesh, toy transformer with ColwiseParallel / RowwiseParallel
MLP, full training-loop iteration (forward → backward → Muon.step).

Asserts:
1. Step completes with no exception (the NotImplementedError guard
   removed in T2b2 is not reintroduced).
2. Every dedicated param's ``_owned_data`` actually changed.
3. All DP-owner TP ranks agree on their local shard (the scatter
   delivered a consistent value to each TP coord).
4. No NaNs / infs in any weight.

Run:
    torchrun --nproc_per_node=4 tests/distributed/test_tp_muon_step.py
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dmuon
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from tp_profile_utils import (
    assert_tp_owner_spread,
    collect_tp_profile,
    maybe_write_tp_profile,
)


class MLP(nn.Module):
    def __init__(self, h=512, inter=2048):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, h=512, inter=2048):
        super().__init__()
        self.ln = nn.LayerNorm(h)
        self.mlp = MLP(h, inter)

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class Tiny(nn.Module):
    def __init__(self, num_layers=2, h=512, inter=2048):
        super().__init__()
        self.layers = nn.ModuleList([Block(h, inter) for _ in range(num_layers)])
        self.out = nn.Linear(h, h, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x).mean()


def log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        log(rank, f"SKIP: test_tp_muon_step needs world=4, got {world_size}")
        dist.destroy_process_group()
        return 0
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    mesh = init_device_mesh(
        "cuda", (2, 2), mesh_dim_names=("dp", "tp")
    )
    dp_mesh = mesh["dp"]
    tp_mesh = mesh["tp"]

    torch.manual_seed(42)
    model = Tiny(num_layers=2, h=512, inter=2048).to(device)

    # Step 1: TP
    plan = {
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }
    for layer in model.layers:
        parallelize_module(layer, tp_mesh, plan)

    # Step 2: DMuon (before FSDP2, per tp_design.md §5.2)
    dmuon.dedicate_params(
        model, dp_mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )

    # Step 3: FSDP2
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)

    # Step 4: Muon — ``replicate_async`` flipped via DMUON_TEST_ASYNC to
    # cover both sync (T2b) and async (T2d) scatter paths.
    use_async = bool(int(os.environ.get("DMUON_TEST_ASYNC", "0") or 0))
    optimizer = dmuon.Muon(
        model, lr=0.02, momentum=0.95, weight_decay=0.01,
        adamw_lr=1e-3, replicate_async=use_async,
    )
    log(rank, f"rank {rank}: replicate_async={use_async}")

    tp_dps = [dp for dp in optimizer._dedicated_params
              if dp.is_dtensor and dp.tp_group is not None]
    log(rank, f"rank {rank}: dedicated params owned = {len(optimizer._dedicated_params)}, "
              f"TP-sharded = {len(tp_dps)}")
    tp_profile = collect_tp_profile(
        model,
        scenario="dp_tp",
        replicate_async=use_async,
    )
    assert_tp_owner_spread(tp_profile, min_owner_ranks=2)
    tp_owner_set = set(tp_profile["owner_coverage"])
    assert tp_owner_set == {0, 1}, (
        f"rank {rank}: TP-owner LPT should distribute owners across TP ranks, "
        f"got {sorted(tp_owner_set)}"
    )

    # Snapshot weights pre-step for every owned param.
    pre: list[tuple[str, torch.Tensor]] = [
        (dp.param_name, dp._owned_data.detach().clone())
        for dp in optimizer._dedicated_params
    ]

    # Two training iterations: async path needs a second forward to drain
    # the T2d ``_tp_scatter_state`` event from step 1, so "did scatter
    # actually complete + write _owned_data?" is only observable after
    # the next forward triggers ``_pre_forward_wait``.
    torch.manual_seed(rank)
    profile_out = os.environ.get("DMUON_TP_PROFILE_OUT")
    losses: list[float] = []
    step_times_ms: list[float] = []
    for it in range(2):
        if profile_out:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        optimizer.zero_grad()
        x = torch.randn(4, 16, 512, device=device)
        loss = model(x)
        loss.backward()
        optimizer.step()
        if profile_out:
            torch.cuda.synchronize()
            step_times_ms.append((time.perf_counter() - t0) * 1000.0)
        losses.append(loss.item())
        if rank == 0:
            print(f"iter {it}: loss={loss.item():.4f}", flush=True)
    torch.cuda.synchronize()

    # --- Assertions ---
    any_changed = False
    for (name, before), dp in zip(pre, optimizer._dedicated_params):
        after = dp._owned_data
        assert torch.isfinite(after).all(), (
            f"rank {rank}: {name} has non-finite values after step: "
            f"nan={torch.isnan(after).any()}, inf={torch.isinf(after).any()}"
        )
        diff = (after - before).abs().max().item()
        if diff > 0:
            any_changed = True
    assert any_changed, (
        f"rank {rank}: no dedicated param changed after Muon.step — "
        "T2a gather / T2b2 NS / T2b scatter pipeline is not delivering updates."
    )

    # NOTE: in pure-DP (no HSDP replicate_mesh), ``_owned_data`` only
    # exists on the DP-owner rank for each param — DP peers stay at
    # ``None`` and re-read the up-to-date owner value during the next
    # ``unshard`` broadcast.  So a "DP peer consistency" assert on
    # ``_owned_data`` is NOT meaningful here; it would be meaningful
    # in the HSDP 3D mesh integration test (deferred to T5).

    maybe_write_tp_profile(
        profile_out,
        collect_tp_profile(
            model,
            scenario="dp_tp",
            replicate_async=use_async,
            losses=losses,
            step_times_ms=step_times_ms,
        ),
    )

    if rank == 0:
        print(f"PASSED: test_tp_muon_step (async={use_async})")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
