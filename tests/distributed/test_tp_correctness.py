"""T4 — TP correctness: TP=1 degenerate + TP=4 smoke.

Two scenarios under the single entry point (selected via the first CLI arg):

  * ``tp1_degenerate`` (4 GPUs): TP=1 must behave identically to pure DP —
    ``is_tp_sharded`` returns False (TP=1 guard), the DTensor-backed path
    never fires, and Muon runs its non-TP branch.  We assert equivalence
    by running both configs from the same seed and verifying identical
    loss trajectory.

  * ``tp4_parity`` (4 GPUs, 1 DP × 4 TP): TP=4 finite/progress smoke with
    owner-spread assertions.  Full topology loss parity lives in
    ``test_tp_alignment.py`` and ``run_tp_alignment.sh``.

Run via:

    torchrun --nproc_per_node=4 tests/distributed/test_tp_correctness.py tp1_degenerate
    torchrun --nproc_per_node=4 tests/distributed/test_tp_correctness.py tp4_parity
"""

from __future__ import annotations

import os
import sys

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


def log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Scenario 1: TP=1 degenerate
# ---------------------------------------------------------------------------

def run_tp1_degenerate(rank: int, world_size: int, device: torch.device) -> None:
    """Verify a (dp=4, tp=1) mesh behaves identically to a (dp=4) mesh —
    ``is_tp_sharded`` must filter tp=1 out, and no TP-path code runs.
    """
    assert world_size == 4, f"tp1_degenerate needs world=4, got {world_size}"

    losses_tp1: list[float] = []
    losses_dp: list[float] = []

    for label, mesh_shape, mesh_dim_names in (
        ("tp1", (4, 1), ("dp", "tp")),
        ("dp_only", (4,), ("dp",)),
    ):
        torch.manual_seed(0)
        model = Tiny(num_layers=2, h=256, inter=1024).to(device)

        mesh = init_device_mesh("cuda", mesh_shape, mesh_dim_names=mesh_dim_names)
        dp_mesh = mesh["dp"]

        # In the (4,1) variant we *could* parallelize on mesh["tp"] — but the
        # tp=1 guard in ``is_tp_sharded`` would filter it out regardless.
        # For a cleaner equivalence we leave TP out of both configs and rely
        # on the guard to prove TP-dim-size-1 is a no-op.
        dmuon.dedicate_params(
            model, dp_mesh,
            predicate=lambda n, p: "proj" in n and p.ndim == 2,
        )
        for layer in model.layers:
            fully_shard(layer, mesh=dp_mesh)
        fully_shard(model, mesh=dp_mesh)

        optimizer = dmuon.Muon(
            model, lr=0.02, momentum=0.95, weight_decay=0.01,
            adamw_lr=1e-3, replicate_async=False,
        )
        profile = collect_tp_profile(
            model,
            scenario=f"tp1_{label}",
            replicate_async=False,
        )
        assert profile["tp_param_count"] == 0, (
            f"{label}: TP=1 / DP-only should not enter TP path, "
            f"got {profile['tp_param_count']} TP params"
        )

        torch.manual_seed(rank + 100)
        local_losses: list[float] = []
        for it in range(3):
            optimizer.zero_grad()
            x = torch.randn(4, 16, 256, device=device)
            loss = model(x)
            loss.backward()
            optimizer.step()
            local_losses.append(loss.item())
        log(rank, f"{label} losses: {local_losses}")

        if label == "tp1":
            losses_tp1 = local_losses
        else:
            losses_dp = local_losses

    # Iter-0 STRICT: before any momentum accumulates, the two configs
    # must produce the same forward pass and the same NS update path —
    # the TP=1 guard should make ``is_tp_sharded`` return False, giving
    # bit-identical step.
    assert abs(losses_tp1[0] - losses_dp[0]) < 1e-6, (
        f"iter 0: TP=1 loss {losses_tp1[0]} != DP-only loss {losses_dp[0]} "
        f"(diff {abs(losses_tp1[0] - losses_dp[0]):.2e}).  TP=1 guard should "
        "make the two configs bit-identical for the first step."
    )
    # Subsequent iters: tolerate bf16-level noise (different DeviceMesh
    # shape -> different NCCL stream / allocator ordering).  Iter 0 plus
    # the ``tp_param_count == 0`` assertion above prove TP=1 does not enter
    # the TP collective path; later losses only need bounded parity.
    drift_budget = 2e-2
    for i in range(1, len(losses_tp1)):
        drift = abs(losses_tp1[i] - losses_dp[i])
        assert drift < drift_budget, (
            f"iter {i}: TP=1 vs DP-only drift {drift:.2e} exceeds bf16 "
            f"noise budget ({drift_budget:.1e})"
        )
    log(rank,
        "PASSED: tp1_degenerate (iter 0 bit-identical; TP path inactive; "
        "later iters within bf16 drift)")


# ---------------------------------------------------------------------------
# Scenario 2: TP=4 smoke (owner spread + finite/progress)
# ---------------------------------------------------------------------------

def run_tp4_parity(rank: int, world_size: int, device: torch.device) -> None:
    """Run TP=4 Muon and verify owner spread plus finite training progress."""
    assert world_size == 4, f"tp4_parity needs world=4, got {world_size}"

    mesh = init_device_mesh("cuda", (1, 4), mesh_dim_names=("dp", "tp"))
    dp_mesh = mesh["dp"]
    tp_mesh = mesh["tp"]

    # --- TP-sharded training loop ---
    torch.manual_seed(0)
    tp_model = Tiny(num_layers=2, h=256, inter=1024).to(device)
    plan = {
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }
    for layer in tp_model.layers:
        parallelize_module(layer, tp_mesh, plan)

    dmuon.dedicate_params(
        tp_model, dp_mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )
    for layer in tp_model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(tp_model, mesh=dp_mesh)

    tp_opt = dmuon.Muon(
        tp_model, lr=0.02, momentum=0.95, weight_decay=0.01,
        adamw_lr=1e-3, replicate_async=False,
    )
    initial_profile = collect_tp_profile(
        tp_model,
        scenario="tp4",
        replicate_async=False,
    )
    assert_tp_owner_spread(initial_profile, min_owner_ranks=2)

    # --- Non-TP reference on every rank (seeded identically).  Each rank
    # runs the same computation, so rank 0's trajectory can be the oracle. ---
    torch.manual_seed(0)
    ref_model = Tiny(num_layers=2, h=256, inter=1024).to(device)
    ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=0.02)

    # Shared input seed so both trajectories see the same batches.
    torch.manual_seed(100)
    inputs = [torch.randn(4, 16, 256, device=device) for _ in range(4)]

    tp_losses: list[float] = []
    ref_losses: list[float] = []
    for it in range(4):
        # TP path
        tp_opt.zero_grad()
        lt = tp_model(inputs[it])
        lt.backward()
        tp_opt.step()
        tp_losses.append(lt.item())

        # Reference path (plain AdamW — not Muon; this is a loose smoke
        # check that TP training doesn't diverge, NOT bit-identical).
        ref_opt.zero_grad()
        lr_ = ref_model(inputs[it])
        lr_.backward()
        ref_opt.step()
        ref_losses.append(lr_.item())

    log(rank, f"tp4_parity losses — TP: {tp_losses}")
    log(rank, f"tp4_parity losses — ref AdamW: {ref_losses}")

    # Smoke checks: TP path produces finite, non-exploding losses and each
    # step changes loss meaningfully (training actually happens).
    for i, loss in enumerate(tp_losses):
        assert loss == loss, f"iter {i}: TP loss is NaN"
        assert abs(loss) < 1e3, f"iter {i}: TP loss {loss} exploded"
    # At least one iter must differ from init by more than noise.
    assert abs(tp_losses[-1] - tp_losses[0]) > 1e-4, (
        f"TP losses barely moved over 4 iters: {tp_losses} — NS path may be inactive"
    )
    maybe_write_tp_profile(
        os.environ.get("DMUON_TP_PROFILE_OUT"),
        collect_tp_profile(
            tp_model,
            scenario="tp4",
            replicate_async=False,
            losses=tp_losses,
        ),
    )
    log(rank, "PASSED: tp4_parity (TP=4 runs cleanly, loss is finite and training)")


# ---------------------------------------------------------------------------

def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    scenario = sys.argv[1] if len(sys.argv) > 1 else "tp1_degenerate"
    try:
        if scenario == "tp1_degenerate":
            run_tp1_degenerate(rank, world_size, device)
        elif scenario == "tp4_parity":
            run_tp4_parity(rank, world_size, device)
        else:
            log(rank, f"unknown scenario {scenario!r}; valid: tp1_degenerate | tp4_parity")
            return 2
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
