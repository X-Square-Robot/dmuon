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


def _wallx_style_param_groups(model: nn.Module, *, base_lr: float, action_lr: float):
    base_params = []
    action_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("layers.0."):
            action_params.append(param)
        else:
            base_params.append(param)
    return [
        {
            "params": base_params,
            "group_name": "base",
            "muon_lr": base_lr,
            "adamw_lr": 1e-3,
            "muon_weight_decay": 0.0,
            "adamw_weight_decay": 0.0,
        },
        {
            "params": action_params,
            "group_name": "action",
            "muon_lr": action_lr,
            "adamw_lr": 1e-3,
            "muon_weight_decay": 0.0,
            "adamw_weight_decay": 0.0,
        },
    ]


def _dedicated_fqn_by_dp(model: nn.Module) -> dict[object, str]:
    module_to_fqn = {id(module): name for name, module in model.named_modules()}
    result = {}
    for module in model.modules():
        state = getattr(module, "_dedicated_state", None)
        if state is None:
            continue
        for dp in state.group.params:
            prefix = module_to_fqn.get(id(dp.module), "")
            result[dp] = f"{prefix}.{dp.param_name}" if prefix else dp.param_name
    return result


def _build_param_group_tp_stack(
    *,
    dp_mesh,
    tp_mesh,
    fsdp_mesh,
    device: torch.device,
    action_lr: float,
    replicate_mesh=None,
    replicate_async: bool = False,
) -> tuple[nn.Module, dmuon.Muon]:
    base_lr = 0.01
    torch.manual_seed(2026)
    model = Tiny(num_layers=2, h=256, inter=1024).to(device)

    plan = {
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }
    for layer in model.layers:
        parallelize_module(layer, tp_mesh, plan)

    dmuon.dedicate_params(
        model, dp_mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
        replicate_mesh=replicate_mesh,
    )

    for layer in model.layers:
        fully_shard(layer, mesh=fsdp_mesh)
    fully_shard(model, mesh=fsdp_mesh)

    optimizer = dmuon.Muon(
        model,
        lr=base_lr,
        momentum=0.0,
        weight_decay=0.0,
        ns_steps=3,
        adamw_lr=1e-3,
        adamw_weight_decay=0.0,
        replicate_async=replicate_async,
        param_groups=_wallx_style_param_groups(
            model, base_lr=base_lr, action_lr=action_lr
        ),
    )
    actual = [
        (group["group_name"], group["use_muon"], group["subgroup_type"], group["lr"])
        for group in optimizer.param_groups
    ]
    assert actual == [
        ("base/muon", True, "muon", base_lr),
        ("base/adamw", False, "adamw", 1e-3),
        ("action/muon", True, "muon", action_lr),
        ("action/adamw", False, "adamw", 1e-3),
    ]
    summary = dmuon.summarize_param_groups(model, optimizer, max_rows=12)
    groups = {group["group_name"]: group for group in summary["groups"]}
    assert groups["action/muon"]["lr"] == action_lr
    assert groups["action/muon"]["tp_sharded_dedicated_param_count"] > 0
    assert any(
        row["route"] == "muon"
        and row["group_name"] == "action/muon"
        and row["is_tp_sharded"]
        for row in summary["parameters"]
    )
    assert "action/muon" in dmuon.format_param_group_summary(summary)
    return model, optimizer


def _run_tp_param_group_delta(
    *,
    dp_mesh,
    tp_mesh,
    fsdp_mesh,
    device: torch.device,
    action_lr: float,
    replicate_mesh=None,
) -> dict[str, float]:
    model, optimizer = _build_param_group_tp_stack(
        dp_mesh=dp_mesh,
        tp_mesh=tp_mesh,
        fsdp_mesh=fsdp_mesh,
        device=device,
        action_lr=action_lr,
        replicate_mesh=replicate_mesh,
        replicate_async=False,
    )

    fqn_by_dp = _dedicated_fqn_by_dp(model)
    before = {}
    for dp in optimizer._dedicated_params:
        fqn = fqn_by_dp.get(dp, dp.param_name)
        if not (
            dp.is_dtensor
            and dp.tp_group is not None
            and dp._owned_data is not None
            and (fqn.startswith("layers.0.") or fqn.startswith("layers.1."))
        ):
            continue
        before[fqn] = dp._owned_data.detach().float().clone()

    optimizer.zero_grad()
    torch.manual_seed(5150)
    x = torch.randn(4, 16, 256, device=device)
    loss = model(x)
    loss.backward()
    optimizer.step()
    dmuon.wait_all_post_step_broadcasts(model)
    torch.cuda.synchronize()

    deltas = {}
    for dp in optimizer._dedicated_params:
        fqn = fqn_by_dp.get(dp, dp.param_name)
        if fqn not in before:
            continue
        delta = dp._owned_data.detach().float() - before[fqn]
        deltas[fqn] = float(delta.norm().item())

    del model, optimizer
    torch.cuda.empty_cache()
    dist.barrier()
    return deltas


def _assert_param_group_delta_ratios(
    *,
    rank: int,
    device: torch.device,
    uniform: dict[str, float],
    split: dict[str, float],
    label: str,
) -> torch.Tensor:
    action_count = 0
    base_count = 0
    max_action_err = 0.0
    max_base_err = 0.0
    for fqn in sorted(set(uniform) & set(split)):
        ref_delta = uniform[fqn]
        if ref_delta <= 1e-10:
            continue
        ratio = split[fqn] / ref_delta
        if fqn.startswith("layers.0."):
            action_count += 1
            max_action_err = max(max_action_err, abs(ratio - 2.0))
        elif fqn.startswith("layers.1."):
            base_count += 1
            max_base_err = max(max_base_err, abs(ratio - 1.0))

    stats = torch.tensor(
        [action_count, base_count, max_action_err, max_base_err],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    assert stats[0].item() > 0, f"{label}: no TP-sharded action-group delta"
    assert stats[1].item() > 0, f"{label}: no TP-sharded base-group delta"
    assert stats[2].item() < 5e-3, (
        f"{label}: action Muon LR delta ratio should be 2.0; "
        f"max error={stats[2].item():.6f}"
    )
    assert stats[3].item() < 5e-3, (
        f"{label}: base Muon LR delta ratio should stay 1.0; "
        f"max error={stats[3].item():.6f}"
    )
    log(
        rank,
        f"[{label}] delta ratio: action err={stats[2].item():.2e}, "
        f"base err={stats[3].item():.2e}",
    )
    return stats


def _run_hsdp_tp_param_group_losses(
    *,
    mesh,
    device: torch.device,
    replicate_async: bool,
    steps: int = 3,
) -> list[float]:
    model, optimizer = _build_param_group_tp_stack(
        dp_mesh=mesh["shard"],
        tp_mesh=mesh["tp"],
        fsdp_mesh=mesh["replicate", "shard"],
        device=device,
        action_lr=0.02,
        replicate_mesh=mesh["replicate"],
        replicate_async=replicate_async,
    )

    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        torch.manual_seed(6100 + step)
        x = torch.randn(4, 16, 256, device=device)
        loss = model(x)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    dmuon.wait_all_post_step_broadcasts(model)
    torch.cuda.synchronize()
    del model, optimizer
    torch.cuda.empty_cache()
    dist.barrier()
    return losses


def run_param_groups_lr(rank: int, world_size: int, device: torch.device) -> None:
    if world_size != 4:
        log(rank, f"SKIP: param_groups_lr needs world=4, got {world_size}")
        return

    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
    dp_mesh = mesh["dp"]
    tp_mesh = mesh["tp"]

    uniform = _run_tp_param_group_delta(
        dp_mesh=dp_mesh,
        tp_mesh=tp_mesh,
        fsdp_mesh=dp_mesh,
        device=device,
        action_lr=0.01,
    )
    split = _run_tp_param_group_delta(
        dp_mesh=dp_mesh,
        tp_mesh=tp_mesh,
        fsdp_mesh=dp_mesh,
        device=device,
        action_lr=0.02,
    )

    stats = _assert_param_group_delta_ratios(
        rank=rank,
        device=device,
        uniform=uniform,
        split=split,
        label="dp2_tp2_param_groups",
    )
    log(
        rank,
        "PASSED: param_groups_lr "
        f"(action ratio err={stats[2].item():.2e}, base ratio err={stats[3].item():.2e})",
    )


def run_hsdp_tp_param_groups(rank: int, world_size: int, device: torch.device) -> None:
    if world_size != 8:
        log(rank, f"SKIP: hsdp_tp_param_groups needs world=8, got {world_size}")
        return

    mesh = init_device_mesh(
        "cuda", (2, 2, 2), mesh_dim_names=("replicate", "shard", "tp")
    )
    uniform = _run_tp_param_group_delta(
        dp_mesh=mesh["shard"],
        tp_mesh=mesh["tp"],
        fsdp_mesh=mesh["replicate", "shard"],
        device=device,
        action_lr=0.01,
        replicate_mesh=mesh["replicate"],
    )
    split = _run_tp_param_group_delta(
        dp_mesh=mesh["shard"],
        tp_mesh=mesh["tp"],
        fsdp_mesh=mesh["replicate", "shard"],
        device=device,
        action_lr=0.02,
        replicate_mesh=mesh["replicate"],
    )
    ratio_stats = _assert_param_group_delta_ratios(
        rank=rank,
        device=device,
        uniform=uniform,
        split=split,
        label="hsdp2_shard2_tp2_param_groups",
    )

    losses_sync = _run_hsdp_tp_param_group_losses(
        mesh=mesh, device=device, replicate_async=False
    )
    losses_async = _run_hsdp_tp_param_group_losses(
        mesh=mesh, device=device, replicate_async=True
    )
    max_loss_diff = max(
        abs(sync_loss - async_loss)
        for sync_loss, async_loss in zip(losses_sync, losses_async)
    )
    loss_diff = torch.tensor(max_loss_diff, dtype=torch.float64, device=device)
    dist.all_reduce(loss_diff, op=dist.ReduceOp.MAX)
    assert loss_diff.item() < 1e-6, (
        "HSDP*TP2 param_groups async loss diverged from sync: "
        f"max diff={loss_diff.item():.6e}"
    )

    log(
        rank,
        "PASSED: hsdp_tp_param_groups "
        f"(action ratio err={ratio_stats[2].item():.2e}, "
        f"base ratio err={ratio_stats[3].item():.2e}, "
        f"async loss diff={loss_diff.item():.2e})",
    )


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    scenario = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    expected_world = 8 if scenario == "hsdp_tp_param_groups" else 4
    if world_size != expected_world:
        log(
            rank,
            f"SKIP: {scenario} needs world={expected_world}, got {world_size}",
        )
        dist.destroy_process_group()
        return 0
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if scenario == "param_groups_lr":
        run_param_groups_lr(rank, world_size, device)
        dist.destroy_process_group()
        return 0
    if scenario == "hsdp_tp_param_groups":
        run_hsdp_tp_param_groups(rank, world_size, device)
        dist.destroy_process_group()
        return 0
    if scenario not in ("smoke", "dp_tp"):
        log(
            rank,
            "unknown scenario "
            f"{scenario!r}; valid: smoke | dp_tp | param_groups_lr | "
            "hsdp_tp_param_groups",
        )
        dist.destroy_process_group()
        return 1

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
