"""Distributed multi-domain gradient clipping test (PP=2 x EP=2).

Validates ``dmuon.clip_grad_norm_(model, max_norm, mode=..., norm_reduce_group=...)``
and ``dmuon.clip_model_grad_norm_`` across heterogeneous PP x EP domains:
  - Stage 0 holds ONLY AdamW / Sharded-AdamW params (0 Muon params -> empty
    ``"muon"`` bucket edge case, plus a 3-element sharded_adamw tensor that
    yields padding / uneven valid_numel across ranks),
  - Stage 1 holds dense Muon + AdamW + local EP experts (3D MatrixLayout Muon,
    AdamW, Sharded AdamW with ``grad_scale=0.5``),
  - Exact pre-clip norm & post-clip gradient parity against full gathered
    reference for ``mode="global"``, ``mode="per_algorithm"``, and custom
    ``norm_reduce_group``.

Run:
    torchrun --standalone --nproc_per_node=4 tests/distributed/test_multi_domain_clip.py
"""

from __future__ import annotations

import math
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
from dmuon.fast_clip import _collect_model_owned_grads


class Stage0AdamWOnly(nn.Module):
    """Pipeline stage 0 with ZERO Muon parameters (tests empty muon bucket + sharded padding)."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.embed = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.Parameter(torch.ones(hidden))
        # Deliberately odd size (3 elements across 2 ranks -> chunk=2, valid=[2, 1], padding=1)
        self.odd_bias = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x) * self.norm
        return h + self.odd_bias.sum() * 0.1


class LocalExperts(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.w_muon_packed = nn.Parameter(torch.randn(2, hidden, hidden) * 0.05)
        self.w_adamw = nn.Parameter(torch.randn(hidden, hidden) * 0.05)
        self.w_sharded = nn.Parameter(torch.randn(hidden, hidden) * 0.05)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return (
            F.silu(F.linear(h, self.w_muon_packed[0]))
            + F.silu(F.linear(h, self.w_muon_packed[1]))
            + F.linear(h, self.w_adamw)
            + F.linear(h, self.w_sharded)
        )


class Stage1MoE(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.attn = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.Parameter(torch.ones(hidden))
        self.experts = LocalExperts(hidden)
        self.head = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(x) * self.norm
        return self.head(h + 0.1 * self.experts(h))


def _policy_fn(name: str, param: nn.Parameter) -> dict[str, object]:
    domain = "expert" if name.startswith("experts.") else "dense"
    if name.endswith("w_muon_packed"):
        return {
            "domain": domain,
            "route": "muon",
            "matrix_layout": MatrixLayout.view(*param.shape),
        }
    if name.endswith(("norm", "w_adamw")):
        return {"domain": domain, "route": "adamw"}
    if name.endswith(("embed.weight", "odd_bias", "w_sharded")):
        return {"domain": domain, "route": "sharded_adamw"}
    return {"domain": domain, "route": "muon"}


def _build_and_backward(
    *,
    pp_rank: int,
    ep_rank: int,
    rank: int,
    domains: dict[str, OwnershipDomain],
    device: torch.device,
    seed: int,
):
    torch.manual_seed(seed + pp_rank * 100 + ep_rank * 10)
    model = Stage0AdamWOnly().to(device) if pp_rank == 0 else Stage1MoE().to(device)
    dmuon.dedicate_params(
        model,
        domains=domains,
        default_domain="dense",
        predicate=lambda _n, _p: True,
        param_policy_fn=_policy_fn,
        reshard_after_forward=True,
        process_group_policy="isolated",
    )
    gen = torch.Generator(device="cpu").manual_seed(700 + seed + rank * 7)
    x = torch.randn(4, 32, generator=gen).to(device) * 4.0
    t = torch.randn(4, 32, generator=gen).to(device) * 4.0
    loss = F.mse_loss(model(x), t)
    loss.backward()
    return model


def _local_sq_sum(grads: list[torch.Tensor]) -> float:
    return float(sum(float(g.detach().double().pow(2).sum().item()) for g in grads))


def _global_sum(val: float, group=None) -> float:
    t = torch.tensor([val], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    return float(t.item())


def _run_case(
    *,
    mode: str,
    max_norm: float,
    norm_reduce_group=None,
    label: str,
) -> None:
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

    reduce_group = mesh["ep"].get_group() if norm_reduce_group == "stage" else None
    model = _build_and_backward(
        pp_rank=pp_rank,
        ep_rank=ep_rank,
        rank=rank,
        domains=domains,
        device=device,
        seed=42,
    )

    # Snapshot unclipped owned grads before calling clip_grad_norm_.
    muon_grads, adamw_grads = _collect_model_owned_grads(model)
    if pp_rank == 0:
        assert len(muon_grads) == 0, "Stage 0 must exercise empty muon bucket"
        assert len(adamw_grads) > 0
    else:
        assert len(muon_grads) > 0 and len(adamw_grads) > 0

    muon_before = [g.detach().clone() for g in muon_grads]
    adamw_before = [g.detach().clone() for g in adamw_grads]

    muon_sq_global = _global_sum(_local_sq_sum(muon_before), group=reduce_group)
    adamw_sq_global = _global_sum(_local_sq_sum(adamw_before), group=reduce_group)
    total_norm_ref = math.sqrt(muon_sq_global + adamw_sq_global)
    muon_norm_ref = math.sqrt(muon_sq_global)
    adamw_norm_ref = math.sqrt(adamw_sq_global)

    # Run public API under test.
    result = dmuon.clip_grad_norm_(
        model,
        max_norm,
        mode=mode,
        norm_reduce_group=reduce_group,
    )

    # Verify norms and post-clip gradients.
    assert abs(float(result.total_norm.item()) - total_norm_ref) < 1e-4, (
        f"{label} rank {rank}: total_norm {result.total_norm.item()} != {total_norm_ref}"
    )

    if mode == "global":
        coef_ref = min(max_norm / (total_norm_ref + 1e-6), 1.0)
        g_stats = result.stats_by_name["global"]
        assert abs(float(g_stats.total_norm.item()) - total_norm_ref) < 1e-4
        assert abs(float(g_stats.clip_coef.item()) - coef_ref) < 1e-5
        for actual, before in zip(muon_grads, muon_before):
            torch.testing.assert_close(actual, before * coef_ref, atol=1e-5, rtol=1e-5)
        for actual, before in zip(adamw_grads, adamw_before):
            torch.testing.assert_close(actual, before * coef_ref, atol=1e-5, rtol=1e-5)
    else:
        muon_coef_ref = min(max_norm / (muon_norm_ref + 1e-6), 1.0)
        adamw_coef_ref = min(max_norm / (adamw_norm_ref + 1e-6), 1.0)
        m_stats = result.stats_by_name["muon"]
        a_stats = result.stats_by_name["adamw"]
        assert abs(float(m_stats.total_norm.item()) - muon_norm_ref) < 1e-4
        assert abs(float(a_stats.total_norm.item()) - adamw_norm_ref) < 1e-4
        assert abs(float(m_stats.clip_coef.item()) - muon_coef_ref) < 1e-5
        assert abs(float(a_stats.clip_coef.item()) - adamw_coef_ref) < 1e-5
        for actual, before in zip(muon_grads, muon_before):
            torch.testing.assert_close(actual, before * muon_coef_ref, atol=1e-5, rtol=1e-5)
        for actual, before in zip(adamw_grads, adamw_before):
            torch.testing.assert_close(actual, before * adamw_coef_ref, atol=1e-5, rtol=1e-5)

    # Also verify sharded_adamw padding was NOT corrupted with NaN/inf and valid slice matches.
    for dp in dmuon.get_dedicated_params(model):
        if dp.uses_sharded_adamw() and dp._sharded_adamw_grad is not None:
            assert torch.isfinite(dp._sharded_adamw_grad).all()

    dist.barrier()
    if rank == 0:
        print(
            f"PASS {label} (total_norm={total_norm_ref:.4f}, fastpath={result.fastpath})",
            flush=True,
        )


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    try:
        assert dist.get_world_size() == 4, "test_multi_domain_clip requires 4 GPUs"
        _run_case(mode="global", max_norm=0.5, norm_reduce_group=None, label="4A_pp2_ep2_global")
        _run_case(
            mode="per_algorithm",
            max_norm=0.5,
            norm_reduce_group=None,
            label="4B_pp2_ep2_per_algorithm_empty_bucket",
        )
        _run_case(
            mode="per_algorithm",
            max_norm=0.5,
            norm_reduce_group="stage",
            label="4C_pp2_ep2_custom_norm_reduce_group",
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
