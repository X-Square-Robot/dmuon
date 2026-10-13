"""HSDP gradient accumulation with the replicate-reduce gate turned off.

``DedicatedParamGroup.replicate_grads_enabled = False`` mirrors FSDP2's
``set_requires_all_reduce(False)``: each micro-batch still runs the Stage-1
shard reduce, but the Stage-2 replicate reduce is deferred and the shard-only
partial is held in ``_partial_reduce_by_param`` until the gate flips back.

The reduced gradients handed to the optimizer must match plain ``no_sync``
accumulation, which defers both stages.  This covers owner-managed Muon params
and sharded AdamW params; FSDP2 symmetric params use FSDP2's own gate.

Run: ``torchrun --nproc_per_node=4 tests/distributed/test_hsdp_replicate_gate_accum.py``
"""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from dmuon import (
    Muon,
    dedicate_params,
    no_sync,
    set_model_state_dict,
    wait_all_post_step_broadcasts,
    wait_all_reduces,
)
from dmuon.checkpoint import _compute_dedicated_fqns
from dmuon.utils import iter_dedicated_states


class MLP(nn.Module):
    def __init__(self, hidden=128, intermediate=512):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, hidden=128, intermediate=512):
        super().__init__()
        self.mlp = MLP(hidden, intermediate)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class TinyModel(nn.Module):
    def __init__(self, num_layers=2, hidden=128, intermediate=512):
        super().__init__()
        self.layers = nn.ModuleList(
            [Block(hidden, intermediate) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x).sum()


SHARDED_ADAMW_NAMES = {"head.weight"}


def _route(name, _param):
    if name in SHARDED_ADAMW_NAMES:
        return "sharded_adamw"
    if "proj" in name:
        return "muon"
    return "adamw"


def build(model_state, device, hsdp_mesh):
    model = TinyModel().to(device)
    model.load_state_dict(model_state)
    dedicate_params(
        model,
        hsdp_mesh["shard"],
        replicate_mesh=hsdp_mesh["replicate"],
        predicate=lambda n, p: "proj" in n or n in SHARDED_ADAMW_NAMES,
        route_hint_fn=_route,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=hsdp_mesh)
    fully_shard(model, mesh=hsdp_mesh)

    by_route = {"muon": [], "sharded_adamw": [], "adamw": []}
    for name, param in model.named_parameters():
        by_route[_route(name, param)].append(param)
    optimizer = Muon(
        model,
        lr=0.02,
        momentum=0.0,
        weight_decay=0.0,
        ns_steps=3,
        adamw_lr=1e-3,
        param_groups=[
            {"params": by_route["muon"], "dmuon_route": "muon"},
            {"params": by_route["sharded_adamw"], "dmuon_route": "sharded_adamw"},
            {"params": by_route["adamw"], "dmuon_route": "adamw"},
        ],
    )
    sharded = [
        dp
        for dp in optimizer._all_dedicated_params
        if getattr(dp, "_dmuon_route", None) == "sharded_adamw"
    ]
    assert sharded, "expected sharded AdamW-route dedicated params"
    return model, optimizer


def set_replicate_gate(model, enabled):
    for state in iter_dedicated_states(model):
        state.group.replicate_grads_enabled = enabled
    model.set_requires_all_reduce(enabled)


def capture_grads(model, optimizer):
    """Gradients exactly as the optimizer is about to consume them."""
    wait_all_reduces(model)
    torch.cuda.synchronize()
    grads = {}
    for dp, fqn in _compute_dedicated_fqns(model).items():
        if getattr(dp, "_dmuon_route", None) == "sharded_adamw":
            grad = dp._sharded_adamw_grad
        else:
            grad = dp._reduced_grad
        if grad is not None:
            grads[fqn] = grad.detach().float().cpu().clone()
    for name, param in model.named_parameters():
        if param.grad is not None:
            local = getattr(param.grad, "_local_tensor", param.grad)
            grads[name] = local.detach().float().cpu().clone()
    return grads


def train(model_state, device, hsdp_mesh, batches, *, gated):
    model, optimizer = build(model_state, device, hsdp_mesh)
    grads_per_step = []
    for first, last in batches:
        if gated:
            set_replicate_gate(model, False)
            (model(first) / 2).backward()
            set_replicate_gate(model, True)
        else:
            with no_sync(model):
                (model(first) / 2).backward()
        (model(last) / 2).backward()
        grads_per_step.append(capture_grads(model, optimizer))
        optimizer.step()
        optimizer.zero_grad()
        for state in iter_dedicated_states(model):
            assert not state.group._partial_reduce_by_param, (
                "replicate-gated partial survived the optimizer step"
            )
        # Reset the weights so later steps compare gradients rather than
        # accumulated bf16 Newton-Schulz rounding; optimizer and gate state
        # carry over, so stale partials would still show up.
        wait_all_post_step_broadcasts(model)
        torch.cuda.synchronize()
        set_model_state_dict(model, model_state)
    del optimizer, model
    torch.cuda.empty_cache()
    return grads_per_step


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4, "requires 4 ranks for a 2x2 HSDP mesh"
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    hsdp_mesh = init_device_mesh(
        "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
    )

    torch.manual_seed(0)
    model_state = {k: v.to(device) for k, v in TinyModel().state_dict().items()}
    num_steps = 3
    # Per-rank data so that both the shard and the replicate average matter.
    gen = torch.Generator(device="cpu").manual_seed(1234 + rank)
    batches = [
        [torch.randn(4, 128, generator=gen).to(device) for _ in range(2)]
        for _ in range(num_steps)
    ]

    reference = train(model_state, device, hsdp_mesh, batches, gated=False)
    dist.barrier()
    gated = train(model_state, device, hsdp_mesh, batches, gated=True)
    dist.barrier()

    failures = []
    worst = 0.0
    for step, (ref_step, gated_step) in enumerate(zip(reference, gated)):
        if ref_step.keys() != gated_step.keys():
            failures.append(
                f"step {step}: grad sets differ: "
                f"{sorted(ref_step.keys() ^ gated_step.keys())}"
            )
            continue
        for name, ref in ref_step.items():
            got = gated_step[name]
            diff = (got - ref).abs().max().item()
            worst = max(worst, diff / max(ref.abs().max().item(), 1e-12))
            if not torch.allclose(got, ref, rtol=1e-5, atol=1e-7):
                failures.append(f"step {step} {name}: max_abs_diff={diff:.3e}")
    flag = torch.tensor([len(failures)], device=device)
    dist.all_reduce(flag)
    if failures:
        print(f"[rank {rank}] FAILED:\n  " + "\n  ".join(failures), flush=True)
    worst_t = torch.tensor([worst], device=device)
    dist.all_reduce(worst_t, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"replicate-gate vs no_sync: worst rel grad diff = {worst_t.item():.3e}")
        if flag.item() == 0:
            print("PASSED: replicate-gated accumulation matches no_sync")
    dist.barrier()
    dist.destroy_process_group()
    if flag.item():
        sys.exit(1)


if __name__ == "__main__":
    main()
