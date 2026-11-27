"""Distributed correctness tests for DMuon OwnershipDomain (EP, DP, HSDP, async).

Run:
    torchrun --standalone --nproc_per_node=4 tests/distributed/test_multi_domain_ep.py
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
from dmuon.utils import iter_dedicated_states


class _AllToAll(torch.autograd.Function):
    """Differentiable equal-split all-to-all across an EP process group."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        out = torch.empty_like(tensor)
        dist.all_to_all_single(out, tensor.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_in = torch.empty_like(grad_output)
        dist.all_to_all_single(grad_in, grad_output.contiguous(), group=ctx.group)
        return grad_in, None


class ExpertBlock(nn.Module):
    """Local expert slice holding Muon (3D), AdamW (2D), and Sharded AdamW (2D) weights."""

    def __init__(self, hidden: int, num_local_experts: int = 2):
        super().__init__()
        self.w_muon = nn.Parameter(torch.randn(num_local_experts, hidden, hidden) * 0.05)
        self.w_adamw = nn.Parameter(torch.randn(hidden, hidden) * 0.05)
        self.w_sharded = nn.Parameter(torch.randn(hidden, hidden) * 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, hidden)
        out = torch.zeros_like(x)
        for e in range(self.w_muon.shape[0]):
            out = out + F.silu(F.linear(x, self.w_muon[e]))
        out = out + F.linear(x, self.w_adamw)
        out = out + F.linear(x, self.w_sharded)
        return out


class MoELayer(nn.Module):
    def __init__(self, hidden: int, ep_group: dist.ProcessGroup, num_local_experts: int = 2):
        super().__init__()
        self.ep_group = ep_group
        self.ep_size = ep_group.size()
        self.attn_proj = nn.Linear(hidden, hidden, bias=False)
        self.norm_scale = nn.Parameter(torch.ones(hidden))
        self.experts = ExpertBlock(hidden, num_local_experts=num_local_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (ep_size * tokens_per_ep, hidden)
        h = self.attn_proj(x) * self.norm_scale
        if self.ep_size > 1:
            dispatched = _AllToAll.apply(h, self.ep_group)
            expert_out = self.experts(dispatched)
            combined = _AllToAll.apply(expert_out, self.ep_group)
        else:
            combined = self.experts(h)
        return x + 0.1 * combined


class DistributedMoEModel(nn.Module):
    def __init__(self, hidden: int, ep_group: dist.ProcessGroup, num_layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(hidden, hidden, bias=False)
        self.layers = nn.ModuleList([MoELayer(hidden, ep_group) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.out_proj(h)


class ReferenceMoELayer(nn.Module):
    """Single-GPU exact mathematical reference holding all EP lanes' experts."""

    def __init__(self, hidden: int, ep_size: int, num_local_experts: int = 2):
        super().__init__()
        self.ep_size = ep_size
        self.attn_proj = nn.Linear(hidden, hidden, bias=False)
        self.norm_scale = nn.Parameter(torch.ones(hidden))
        self.experts_by_ep = nn.ModuleList(
            [ExpertBlock(hidden, num_local_experts=num_local_experts) for _ in range(ep_size)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (dp_size, ep_size, ep_size * tokens_per_ep, hidden)
        dp_size, ep_size, total_t, hidden = x.shape
        t_chunk = total_t // ep_size
        h = self.attn_proj(x) * self.norm_scale
        # All-to-all within each DP lane along the ep_size axis:
        # rank e sends chunk dst to rank dst -> transpose(1, 2)
        h_chunks = h.view(dp_size, ep_size, ep_size, t_chunk, hidden)
        dispatched = h_chunks.transpose(1, 2).contiguous().view(dp_size, ep_size, total_t, hidden)
        out_lanes = []
        for e in range(ep_size):
            out_lanes.append(self.experts_by_ep[e](dispatched[:, e]))
        expert_out = torch.stack(out_lanes, dim=1)
        out_chunks = expert_out.view(dp_size, ep_size, ep_size, t_chunk, hidden)
        combined = out_chunks.transpose(1, 2).contiguous().view(dp_size, ep_size, total_t, hidden)
        return x + 0.1 * combined


class ReferenceMoEModel(nn.Module):
    def __init__(self, hidden: int, ep_size: int, num_layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(hidden, hidden, bias=False)
        self.layers = nn.ModuleList(
            [ReferenceMoELayer(hidden, ep_size) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.out_proj(h)


def _copy_ref_to_dist(ref: ReferenceMoEModel, dist_model: DistributedMoEModel, ep_rank: int) -> None:
    with torch.no_grad():
        dist_model.in_proj.weight.copy_(ref.in_proj.weight)
        dist_model.out_proj.weight.copy_(ref.out_proj.weight)
        for ref_layer, dist_layer in zip(ref.layers, dist_model.layers):
            dist_layer.attn_proj.weight.copy_(ref_layer.attn_proj.weight)
            dist_layer.norm_scale.copy_(ref_layer.norm_scale)
            ref_exp = ref_layer.experts_by_ep[ep_rank]
            dist_layer.experts.w_muon.copy_(ref_exp.w_muon)
            dist_layer.experts.w_adamw.copy_(ref_exp.w_adamw)
            dist_layer.experts.w_sharded.copy_(ref_exp.w_sharded)


def _policy_fn(name: str, param: nn.Parameter) -> dict[str, object]:
    is_expert = ".experts." in name
    domain = "expert" if is_expert else "dense"
    if name.endswith("w_muon"):
        return {
            "domain": domain,
            "route": "muon",
            "matrix_layout": MatrixLayout.view(*param.shape),
        }
    if name.endswith("w_adamw") or name.endswith("norm_scale"):
        return {"domain": domain, "route": "adamw"}
    if name.endswith("w_sharded") or name.endswith("in_proj.weight"):
        return {"domain": domain, "route": "sharded_adamw"}
    return {"domain": domain, "route": "muon"}


def _ref_policy_fn(name: str, param: nn.Parameter) -> dict[str, object]:
    if name.endswith("w_muon"):
        return {
            "route": "muon",
            "matrix_layout": MatrixLayout.view(*param.shape),
        }
    if name.endswith("w_adamw") or name.endswith("norm_scale"):
        return {"route": "adamw"}
    if name.endswith("w_sharded") or name.endswith("in_proj.weight"):
        return {"route": "sharded_adamw"}
    return {"route": "muon"}


def _make_inputs(step: int, dp_size: int, ep_size: int, hidden: int, device: torch.device):
    gen = torch.Generator(device="cpu").manual_seed(1000 + step)
    total_t = ep_size * 2
    x = torch.randn(dp_size, ep_size, total_t, hidden, generator=gen, dtype=torch.float32).to(
        device
    )
    target = torch.randn(dp_size, ep_size, total_t, hidden, generator=gen, dtype=torch.float32).to(
        device
    )
    return x, target


def _assert_dist_matches_ref(
    dist_model: DistributedMoEModel,
    ref_model: ReferenceMoEModel,
    ep_rank: int,
    *,
    muon_atol: float = 5e-4,
    adamw_atol: float = 5e-4,
    label: str = "",
) -> None:
    dist_sd = dmuon.get_model_state_dict(dist_model, cpu_offload=True, rank0_only=False)
    ref_sd = dmuon.get_model_state_dict(ref_model, cpu_offload=True, rank0_only=False)

    pairs: list[tuple[str, torch.Tensor, torch.Tensor, float]] = [
        ("in_proj.weight", dist_sd["in_proj.weight"], ref_sd["in_proj.weight"], adamw_atol),
        ("out_proj.weight", dist_sd["out_proj.weight"], ref_sd["out_proj.weight"], muon_atol),
    ]
    for i in range(len(dist_model.layers)):
        pairs.append(
            (
                f"layers.{i}.attn_proj.weight",
                dist_sd[f"layers.{i}.attn_proj.weight"],
                ref_sd[f"layers.{i}.attn_proj.weight"],
                muon_atol,
            )
        )
        pairs.append(
            (
                f"layers.{i}.norm_scale",
                dist_sd[f"layers.{i}.norm_scale"],
                ref_sd[f"layers.{i}.norm_scale"],
                adamw_atol,
            )
        )
        for sub, tol in (("w_muon", muon_atol), ("w_adamw", adamw_atol), ("w_sharded", adamw_atol)):
            dk = f"layers.{i}.experts.{sub}"
            rk = f"layers.{i}.experts_by_ep.{ep_rank}.{sub}"
            pairs.append((dk, dist_sd[dk], ref_sd[rk], tol))

    for key, actual, expected, tol in pairs:
        torch.testing.assert_close(
            actual,
            expected,
            atol=tol,
            rtol=tol,
            msg=lambda m, _k=key: f"{label}:{_k}\n{m}",
        )


def _run_ep_parity_case(
    *,
    dp_size: int,
    ep_size: int,
    label: str,
    group_prepare_ahead: bool = False,
    forward_prefetch_depth: int = 0,
    replicate_async: bool = False,
    reshard_after_forward: bool = True,
    grad_accum_steps: int = 1,
) -> None:
    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    hidden = 32

    mesh = init_device_mesh("cuda", (dp_size, ep_size), mesh_dim_names=("dp", "ep"))
    dp_rank = mesh["dp"].get_local_rank()
    ep_rank = mesh["ep"].get_local_rank()
    dense_mesh = mesh["dp", "ep"]._flatten()
    expert_mesh = mesh["dp"]
    ep_group = mesh["ep"].get_group()

    # Build deterministic reference model on this device (1-rank DMuon for exact optimizer math)
    torch.manual_seed(42)
    ref_model = ReferenceMoEModel(hidden, ep_size=ep_size, num_layers=2).to(device)
    dist_model = DistributedMoEModel(hidden, ep_group=ep_group, num_layers=2).to(device)
    _copy_ref_to_dist(ref_model, dist_model, ep_rank)

    # Wrap reference model with a 1-rank local mesh so Muon NS + AdamW math is identical.
    # Reuse a per-rank 1-GPU mesh.
    self_mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("self",)) if dist.get_world_size() == 1 else None
    # Actually construct single-rank submesh via mesh["dp"] when dp_size==1, else build a 1-rank ref mesh
    if dp_size == 1:
        ref_mesh = mesh["dp"]
    else:
        # 4-rank world -> (4,) 1-rank slices via (4, 1)["one"]
        ref_mesh = init_device_mesh("cuda", (dist.get_world_size(), 1), mesh_dim_names=("w", "one"))[
            "one"
        ]
    _ = self_mesh

    dmuon.dedicate_params(
        ref_model,
        mesh=ref_mesh,
        predicate=lambda _n, _p: True,
        param_policy_fn=_ref_policy_fn,
        process_group_policy="shared",
    )
    ref_opt = dmuon.Muon(
        ref_model,
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

    domains = {
        "dense": OwnershipDomain(
            name="dense",
            shard_mesh=dense_mesh,
            grad_scale=1.0,
        ),
        "expert": OwnershipDomain(
            name="expert",
            shard_mesh=expert_mesh,
            grad_scale=1.0 / ep_size,
        ),
    }
    dmuon.dedicate_params(
        dist_model,
        domains=domains,
        default_domain="dense",
        predicate=lambda _n, _p: True,
        param_policy_fn=_policy_fn,
        reshard_after_forward=reshard_after_forward,
        process_group_policy="isolated",
    )

    # Structural invariants: single DedicatedCommContext & exact per-domain topology
    comm_ctx = dist_model._dedicated_comm_ctx
    states = list(iter_dedicated_states(dist_model))
    assert states, f"{label}: expected dedicated states"
    assert len(comm_ctx.all_states) == len(states), (
        f"{label}: comm_ctx.all_states mismatch ({len(comm_ctx.all_states)} vs {len(states)})"
    )
    seen_domains = set()
    for state in states:
        assert state.comm_ctx is comm_ctx
        group = state.group
        seen_domains.add(group._domain_name)
        if group._domain_name == "dense":
            assert group._dp_group.size() == dp_size * ep_size
            assert group._grad_scale == 1.0
        elif group._domain_name == "expert":
            assert group._dp_group.size() == dp_size
            assert abs(group._grad_scale - (1.0 / ep_size)) < 1e-12
            expected_peers = tuple(dist.get_process_group_ranks(expert_mesh.get_group()))
            actual_peers = tuple(dist.get_process_group_ranks(group._dp_group))
            assert actual_peers == expected_peers
        else:
            raise AssertionError(f"Unexpected domain {group._domain_name!r}")
    assert seen_domains == {"dense", "expert"}, f"{label}: missing domain in {seen_domains}"

    dist_opt = dmuon.Muon(
        dist_model,
        lr=1e-2,
        adamw_lr=1e-2,
        weight_decay=0.01,
        adamw_weight_decay=0.01,
        ns_backend=dmuon.NewtonSchulz("gram", kernel="cublas"),
        first_step_progress_log=False,
        replicate_async=replicate_async,
        group_prepare_ahead=group_prepare_ahead,
        forward_prefetch_depth=forward_prefetch_depth,
    )

    for step in range(3):
        ref_opt.zero_grad()
        dist_opt.zero_grad()

        for mb in range(grad_accum_steps):
            x_all, target_all = _make_inputs(
                step * grad_accum_steps + mb, dp_size, ep_size, hidden, device
            )
            # Reference: global mean MSE across all dp * ep ranks, divided by grad_accum_steps
            ref_out = ref_model(x_all)
            ref_loss = F.mse_loss(ref_out, target_all) / grad_accum_steps
            ref_loss.backward()

            # Distributed: local shard loss on (dp_rank, ep_rank)
            x_local = x_all[dp_rank, ep_rank]
            t_local = target_all[dp_rank, ep_rank]
            use_no_sync = mb < grad_accum_steps - 1
            if use_no_sync:
                with dmuon.no_sync(dist_model):
                    dist_out = dist_model(x_local)
                    dist_loss = F.mse_loss(dist_out, t_local) / grad_accum_steps
                    dist_loss.backward()
            else:
                dist_out = dist_model(x_local)
                dist_loss = F.mse_loss(dist_out, t_local) / grad_accum_steps
                dist_loss.backward()

        ref_opt.step()
        dist_opt.step()

    _assert_dist_matches_ref(dist_model, ref_model, ep_rank, label=label)
    dist.barrier()
    if rank == 0:
        print(f"PASS {label}", flush=True)


def _run_mixed_hsdp_1d_case() -> None:
    """Verify mixed 2D HSDP dense domain + 1D expert domain in a single dedicate_params call."""
    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    hidden = 32
    dp_size = 2
    ep_size = 2

    mesh = init_device_mesh("cuda", (dp_size, ep_size), mesh_dim_names=("dp", "ep"))
    dp_rank = mesh["dp"].get_local_rank()
    ep_rank = mesh["ep"].get_local_rank()
    ep_group = mesh["ep"].get_group()

    torch.manual_seed(77)
    ref_model = ReferenceMoEModel(hidden, ep_size=ep_size, num_layers=2).to(device)
    dist_model = DistributedMoEModel(hidden, ep_group=ep_group, num_layers=2).to(device)
    _copy_ref_to_dist(ref_model, dist_model, ep_rank)

    ref_mesh = init_device_mesh("cuda", (dist.get_world_size(), 1), mesh_dim_names=("w", "one"))[
        "one"
    ]
    dmuon.dedicate_params(
        ref_model,
        mesh=ref_mesh,
        predicate=lambda _n, _p: True,
        param_policy_fn=_ref_policy_fn,
        process_group_policy="shared",
    )
    ref_opt = dmuon.Muon(
        ref_model,
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

    # Dense uses 2D HSDP (shard=ep, replicate=dp); Expert uses 1D (shard=dp, replicate=None)
    domains = {
        "dense": OwnershipDomain(
            name="dense",
            shard_mesh=mesh["ep"],
            replicate_mesh=mesh["dp"],
            grad_scale=1.0,
        ),
        "expert": OwnershipDomain(
            name="expert",
            shard_mesh=mesh["dp"],
            replicate_mesh=None,
            grad_scale=1.0 / ep_size,
        ),
    }
    dmuon.dedicate_params(
        dist_model,
        domains=domains,
        default_domain="dense",
        predicate=lambda _n, _p: True,
        param_policy_fn=_policy_fn,
        reshard_after_forward=True,
        process_group_policy="isolated",
    )

    # comm_ctx.replicate_group is None (mixed 2D + 1D), while each group holds its own _replicate_group
    assert dist_model._dedicated_comm_ctx.replicate_group is None
    for state in iter_dedicated_states(dist_model):
        if state.group._domain_name == "dense":
            assert state.group._replicate_group is not None
            assert state.group._replicate_group.size() == dp_size
        else:
            assert state.group._replicate_group is None

    dist_opt = dmuon.Muon(
        dist_model,
        lr=1e-2,
        adamw_lr=1e-2,
        weight_decay=0.01,
        adamw_weight_decay=0.01,
        ns_backend=dmuon.NewtonSchulz("gram", kernel="cublas"),
        first_step_progress_log=False,
        replicate_async=True,
        group_prepare_ahead=True,
        forward_prefetch_depth=1,
    )

    for step in range(3):
        ref_opt.zero_grad()
        dist_opt.zero_grad()
        x_all, target_all = _make_inputs(200 + step, dp_size, ep_size, hidden, device)
        ref_loss = F.mse_loss(ref_model(x_all), target_all)
        ref_loss.backward()
        dist_loss = F.mse_loss(dist_model(x_all[dp_rank, ep_rank]), target_all[dp_rank, ep_rank])
        dist_loss.backward()
        ref_opt.step()
        dist_opt.step()

    _assert_dist_matches_ref(dist_model, ref_model, ep_rank, label="mixed_2d_hsdp_1d_expert")
    dist.barrier()
    if rank == 0:
        print("PASS mixed_2d_hsdp_1d_expert", flush=True)


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    try:
        world_size = dist.get_world_size()
        assert world_size == 4, f"test_multi_domain_ep requires 4 GPUs, got {world_size}"

        # 2A: EP=4, DP=1 (1-rank expert shard_mesh + grad_scale=0.25)
        _run_ep_parity_case(
            dp_size=1,
            ep_size=4,
            label="2A_ep4_dp1_sync",
        )
        # 2B: EP=2, DP=2 (2-rank expert shard_mesh + 4-rank dense shard_mesh)
        _run_ep_parity_case(
            dp_size=2,
            ep_size=2,
            label="2B_ep2_dp2_sync",
        )
        # 2C: EP=2, DP=2 with async publish, group_prepare_ahead, forward prefetch, and no_sync grad accum
        _run_ep_parity_case(
            dp_size=2,
            ep_size=2,
            label="2C_ep2_dp2_async_prefetch_nosync",
            group_prepare_ahead=True,
            forward_prefetch_depth=1,
            replicate_async=True,
            reshard_after_forward=True,
            grad_accum_steps=2,
        )
        # 2D: Mixed 2D HSDP dense + 1D expert in one dedicate_params call
        _run_mixed_hsdp_1d_case()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
