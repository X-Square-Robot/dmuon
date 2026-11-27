"""Distributed checkpoint save/resume test across heterogeneous PP x EP domains.

Validates that public ``get_model_state_dict``, ``get_optimizer_state_dict``,
``set_model_state_dict``, and ``set_optimizer_state_dict`` perform zero WORLD
collectives and handle:
  - PP stages with different parameter counts and domain membership (Stage 0
    has dense params only; Stage 1 has dense + MoE experts),
  - Cold Muon experts and cold dedicated-AdamW params with absent optimizer state,
  - Opposite hot/cold expert patterns across EP lanes in DP=2 x EP=2,
  - Exact step-3 continuation parity after checkpoint reload.

Run:
    torchrun --standalone --nproc_per_node=4 tests/distributed/test_multi_domain_checkpoint.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dmuon
from dmuon import MatrixLayout, OwnershipDomain


class Stage0Model(nn.Module):
    """Pipeline stage 0: dense-only (zero expert parameters)."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.embed = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.Parameter(torch.ones(hidden))
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(x) * self.norm)


class LocalExperts(nn.Module):
    """Local expert submodule with real forward() so DMuon hooks fire on it."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.w_muon_packed = nn.Parameter(torch.randn(2, hidden, hidden) * 0.05)
        self.w_muon_e0 = nn.Parameter(torch.randn(hidden, hidden) * 0.05)
        self.w_muon_e1 = nn.Parameter(torch.randn(hidden, hidden) * 0.05)
        self.w_adamw_warm = nn.Parameter(torch.randn(hidden, hidden) * 0.05)
        self.w_adamw_cold = nn.Parameter(torch.randn(hidden, hidden) * 0.05)
        self.w_sharded = nn.Parameter(torch.randn(hidden, hidden) * 0.05)

    def forward(
        self,
        h: torch.Tensor,
        *,
        active_experts: tuple[int, ...] = (0,),
        use_cold_adamw: bool = False,
    ) -> torch.Tensor:
        out = (
            F.silu(F.linear(h, self.w_muon_packed[0]))
            + F.silu(F.linear(h, self.w_muon_packed[1]))
            + F.linear(h, self.w_adamw_warm)
            + F.linear(h, self.w_sharded)
        )
        if 0 in active_experts:
            out = out + F.silu(F.linear(h, self.w_muon_e0))
        if 1 in active_experts:
            out = out + F.silu(F.linear(h, self.w_muon_e1))
        if use_cold_adamw:
            out = out + F.linear(h, self.w_adamw_cold)
        return out


class Stage1MoEModel(nn.Module):
    """Pipeline stage 1: dense + local experts with controllable cold experts."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.attn = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.Parameter(torch.ones(hidden))
        self.experts = LocalExperts(hidden)
        self.head = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        active_experts: tuple[int, ...] = (0,),
        use_cold_adamw: bool = False,
    ) -> torch.Tensor:
        h = self.attn(x) * self.norm
        out = self.experts(h, active_experts=active_experts, use_cold_adamw=use_cold_adamw)
        return self.head(h + 0.1 * out)


def _policy_fn(name: str, param: nn.Parameter) -> dict[str, object]:
    domain = "expert" if name.startswith("experts.") else "dense"
    if name.endswith("w_muon_packed"):
        return {
            "domain": domain,
            "route": "muon",
            "matrix_layout": MatrixLayout.view(*param.shape),
        }
    if name.endswith(("norm", "w_adamw_warm", "w_adamw_cold")):
        return {"domain": domain, "route": "adamw"}
    if name.endswith(("embed.weight", "w_sharded")):
        return {"domain": domain, "route": "sharded_adamw"}
    return {"domain": domain, "route": "muon"}


def _build_stage_model_and_opt(
    *,
    pp_rank: int,
    ep_rank: int,
    domains: dict[str, OwnershipDomain],
    device: torch.device,
    seed: int,
):
    torch.manual_seed(seed + pp_rank * 100 + ep_rank * 10)
    model = Stage0Model().to(device) if pp_rank == 0 else Stage1MoEModel().to(device)
    dmuon.dedicate_params(
        model,
        domains=domains,
        default_domain="dense",
        predicate=lambda _n, _p: True,
        param_policy_fn=_policy_fn,
        reshard_after_forward=True,
        process_group_policy="isolated",
    )
    opt = dmuon.Muon(
        model,
        lr=1e-2,
        adamw_lr=1e-2,
        weight_decay=0.01,
        adamw_weight_decay=0.01,
        ns_backend=dmuon.NewtonSchulz("gram", kernel="cublas"),
        first_step_progress_log=False,
        replicate_async=False,
        group_prepare_ahead=False,
        forward_prefetch_depth=0,
    )
    return model, opt


def _assert_state_dicts_equal(actual: dict, expected: dict, *, label: str) -> None:
    assert set(actual.keys()) == set(expected.keys()), (
        f"{label}: key mismatch {sorted(actual.keys())} vs {sorted(expected.keys())}"
    )
    for key in sorted(actual.keys()):
        a, b = actual[key], expected[key]
        if isinstance(a, dict):
            _assert_state_dicts_equal(a, b, label=f"{label}.{key}")
        elif isinstance(a, torch.Tensor):
            torch.testing.assert_close(
                a, b, atol=0.0, rtol=0.0, msg=lambda m, _k=key: f"{label}.{_k}\n{m}"
            )
        else:
            assert a == b, f"{label}.{key}: {a!r} != {b!r}"


def _run_pp2_ep2_checkpoint_test() -> None:
    rank = dist.get_rank()
    device = torch.device("cuda", rank)

    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("pp", "ep"))
    pp_rank = mesh["pp"].get_local_rank()
    ep_rank = mesh["ep"].get_local_rank()
    singleton_mesh = init_device_mesh("cuda", (4, 1), mesh_dim_names=("w", "one"))["one"]

    domains = {
        "dense": OwnershipDomain("dense", shard_mesh=mesh["ep"], grad_scale=1.0),
        "expert": OwnershipDomain("expert", shard_mesh=singleton_mesh, grad_scale=0.5),
    }

    model_a, opt_a = _build_stage_model_and_opt(
        pp_rank=pp_rank, ep_rank=ep_rank, domains=domains, device=device, seed=42
    )

    def _step(model, opt, step_idx: int, *, active_experts=(0,), use_cold_adamw=False):
        opt.zero_grad()
        gen = torch.Generator(device="cpu").manual_seed(500 + step_idx * 17 + rank)
        x = torch.randn(4, 32, generator=gen).to(device)
        t = torch.randn(4, 32, generator=gen).to(device)
        if pp_rank == 0:
            out = model(x)
        else:
            out = model(x, active_experts=active_experts, use_cold_adamw=use_cold_adamw)
        loss = F.mse_loss(out, t)
        loss.backward()
        opt.step()

    # Run 2 steps: on Stage 1, w_muon_e1 and w_adamw_cold stay completely cold.
    for s in range(2):
        _step(model_a, opt_a, s, active_experts=(0,), use_cold_adamw=False)

    # Export via public APIs only (rank0_only=False) — ZERO WORLD collectives!
    model_sd = dmuon.get_model_state_dict(model_a, cpu_offload=True, rank0_only=False)
    optim_sd = dmuon.get_optimizer_state_dict(model_a, opt_a, cpu_offload=True, rank0_only=False)

    if pp_rank == 0:
        assert "embed.weight" in model_sd and "proj.weight" in model_sd
        assert not any(k.startswith("experts.") for k in model_sd)
        assert "proj.weight" in optim_sd["dedicated"]
        assert optim_sd["dedicated_adamw"]["embed.weight"]["step"] == 2
        assert optim_sd["dedicated_adamw"]["norm"]["step"] == 2
    else:
        assert "experts.w_muon_packed" in model_sd
        assert "experts.w_muon_e0" in optim_sd["dedicated"]
        assert "experts.w_muon_e1" not in optim_sd["dedicated"]
        packed_keys = sorted(
            k for k in optim_sd["dedicated"] if k.startswith("experts.w_muon_packed::")
        )
        assert len(packed_keys) == 2, f"Expected 2 packed matrices, got {packed_keys}"
        assert optim_sd["dedicated_adamw"]["experts.w_adamw_warm"]["step"] == 2
        assert optim_sd["dedicated_adamw"]["experts.w_sharded"]["step"] == 2
        assert "experts.w_adamw_cold" not in optim_sd["dedicated_adamw"]

    # Build fresh model_b with different seed and restore from checkpoint.
    model_b, opt_b = _build_stage_model_and_opt(
        pp_rank=pp_rank, ep_rank=ep_rank, domains=domains, device=device, seed=999
    )
    dmuon.set_model_state_dict(model_b, model_sd)
    dmuon.set_optimizer_state_dict(model_b, opt_b, optim_sd)

    # Run step 3 on both (activating cold expert 1 and w_adamw_cold too) -> bitwise match!
    _step(model_a, opt_a, 2, active_experts=(0, 1), use_cold_adamw=True)
    _step(model_b, opt_b, 2, active_experts=(0, 1), use_cold_adamw=True)

    sd_a = dmuon.get_model_state_dict(model_a, cpu_offload=True, rank0_only=False)
    sd_b = dmuon.get_model_state_dict(model_b, cpu_offload=True, rank0_only=False)
    _assert_state_dicts_equal(sd_b, sd_a, label=f"pp2_ep2_model_rank{rank}")

    opt_sd_a = dmuon.get_optimizer_state_dict(model_a, opt_a, cpu_offload=True, rank0_only=False)
    opt_sd_b = dmuon.get_optimizer_state_dict(model_b, opt_b, cpu_offload=True, rank0_only=False)
    _assert_state_dicts_equal(opt_sd_b, opt_sd_a, label=f"pp2_ep2_optim_rank{rank}")

    dist.barrier()
    if rank == 0:
        print("PASS 3A_pp2_ep2_unequal_stages_cold_experts_resume", flush=True)


def _run_dp2_ep2_asymmetric_cold_checkpoint_test() -> None:
    """Verify DP=2 x EP=2 where EP lane 0 and EP lane 1 have opposite cold experts."""
    rank = dist.get_rank()
    device = torch.device("cuda", rank)

    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "ep"))
    ep_rank = mesh["ep"].get_local_rank()
    domains = {
        "dense": OwnershipDomain("dense", shard_mesh=mesh["dp", "ep"]._flatten(), grad_scale=1.0),
        "expert": OwnershipDomain("expert", shard_mesh=mesh["dp"], grad_scale=0.5),
    }

    model_a, opt_a = _build_stage_model_and_opt(
        pp_rank=1, ep_rank=ep_rank, domains=domains, device=device, seed=77
    )

    def _step(model, opt, step_idx: int, active_experts):
        opt.zero_grad()
        gen = torch.Generator(device="cpu").manual_seed(900 + step_idx * 13 + rank)
        x = torch.randn(4, 32, generator=gen).to(device)
        t = torch.randn(4, 32, generator=gen).to(device)
        loss = F.mse_loss(model(x, active_experts=active_experts, use_cold_adamw=(ep_rank == 1)), t)
        loss.backward()
        opt.step()

    # EP lane 0 activates w_muon_e0 (w_muon_e1 cold, w_adamw_cold cold).
    # EP lane 1 activates w_muon_e1 (w_muon_e0 cold, w_adamw_cold warm).
    active = (0,) if ep_rank == 0 else (1,)
    for s in range(2):
        _step(model_a, opt_a, s, active)

    model_sd = dmuon.get_model_state_dict(model_a, cpu_offload=True, rank0_only=False)
    optim_sd = dmuon.get_optimizer_state_dict(model_a, opt_a, cpu_offload=True, rank0_only=False)

    if ep_rank == 0:
        assert "experts.w_muon_e0" in optim_sd["dedicated"]
        assert "experts.w_muon_e1" not in optim_sd["dedicated"]
        assert "experts.w_adamw_cold" not in optim_sd["dedicated_adamw"]
    else:
        assert "experts.w_muon_e1" in optim_sd["dedicated"]
        assert "experts.w_muon_e0" not in optim_sd["dedicated"]
        assert optim_sd["dedicated_adamw"]["experts.w_adamw_cold"]["step"] == 2

    model_b, opt_b = _build_stage_model_and_opt(
        pp_rank=1, ep_rank=ep_rank, domains=domains, device=device, seed=1234
    )
    dmuon.set_model_state_dict(model_b, model_sd)
    dmuon.set_optimizer_state_dict(model_b, opt_b, optim_sd)

    _step(model_a, opt_a, 2, (0, 1))
    _step(model_b, opt_b, 2, (0, 1))

    sd_a = dmuon.get_model_state_dict(model_a, cpu_offload=True, rank0_only=False)
    sd_b = dmuon.get_model_state_dict(model_b, cpu_offload=True, rank0_only=False)
    _assert_state_dicts_equal(sd_b, sd_a, label=f"dp2_ep2_model_rank{rank}")

    opt_sd_a = dmuon.get_optimizer_state_dict(model_a, opt_a, cpu_offload=True, rank0_only=False)
    opt_sd_b = dmuon.get_optimizer_state_dict(model_b, opt_b, cpu_offload=True, rank0_only=False)
    _assert_state_dicts_equal(opt_sd_b, opt_sd_a, label=f"dp2_ep2_optim_rank{rank}")

    dist.barrier()
    if rank == 0:
        print("PASS 3B_dp2_ep2_opposite_cold_experts_resume", flush=True)


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    try:
        assert dist.get_world_size() == 4, "test_multi_domain_checkpoint requires 4 GPUs"
        _run_pp2_ep2_checkpoint_test()
        _run_dp2_ep2_asymmetric_cold_checkpoint_test()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
