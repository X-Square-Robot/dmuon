"""Distributed smoke test for param_policy mixed precision.

Run:

    torchrun --nproc_per_node=2 tests/distributed/test_param_policy_mixed_precision.py fsdp2
    torchrun --nproc_per_node=2 tests/distributed/test_param_policy_mixed_precision.py ddp
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dmuon


class BackboneBlock(nn.Module):
    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, d, bias=False)
        self.input_dtype = None
        self.weight_dtype = None
        self.output_dtype = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_dtype = x.dtype
        x = self.norm(x)
        self.weight_dtype = self.proj.weight.dtype
        out = self.proj(x)
        self.output_dtype = out.dtype
        return out


class ActionHead(nn.Module):
    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, d, bias=False)
        self.input_dtype = None
        self.weight_dtype = None
        self.output_dtype = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_dtype = x.dtype
        x = self.norm(x)
        self.weight_dtype = self.proj.weight.dtype
        out = self.proj(x)
        self.output_dtype = out.dtype
        return out


class TinyPolicyModel(nn.Module):
    def __init__(self, vocab: int = 16, d: int = 32) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.backbone = BackboneBlock(d)
        self.action_head = ActionHead(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(tokens)
        h = self.backbone(x)
        logits = self.lm_head(h)
        action = self.action_head(h)
        return logits.float().pow(2).mean() + action.float().pow(2).mean()


def _param_policy(include_sharded_adamw: bool) -> dict:
    overrides = [
        {
            "name": ["backbone.proj", "action_head.proj"],
            "set": {"route": "muon"},
        },
        {
            "name": ["action_head"],
            "set": {
                "param_dtype": torch.float32,
                "grad_dtype": torch.float32,
                "output_dtype": torch.float32,
                "cast_forward_inputs": True,
            },
        },
    ]
    if include_sharded_adamw:
        overrides.insert(
            1,
            {
                "name": ["embed_tokens", "lm_head"],
                "set": {"route": "sharded_adamw"},
            },
        )
    return {
        "defaults": {
            "route": "adamw",
            "param_dtype": torch.bfloat16,
            "master_dtype": torch.float32,
            "optim_dtype": torch.float32,
        },
        "overrides": overrides,
    }


def _hook_boundary_factory(model: TinyPolicyModel):
    boundary_ids = {
        id(model.embed_tokens),
        id(model.backbone),
        id(model.action_head),
        id(model.lm_head),
    }
    return lambda module: id(module) in boundary_ids


def _assert_policy_summary(model: nn.Module, optimizer: dmuon.Muon) -> None:
    summary = dmuon.summarize_param_groups(model, optimizer, max_rows=64)
    rows = {row["name"]: row for row in summary["parameters"]}
    assert rows["backbone.proj.weight"]["route"] == "muon"
    assert rows["backbone.proj.weight"]["param_dtype"] == "bfloat16"
    assert rows["action_head.proj.weight"]["route"] == "muon"
    assert rows["action_head.proj.weight"]["param_dtype"] == "float32"
    assert rows["action_head.proj.weight"]["grad_dtype"] == "float32"
    assert rows["action_head.proj.weight"]["output_dtype"] == "float32"
    assert rows["action_head.norm.weight"]["route"] == "adamw"
    assert rows["action_head.norm.weight"]["param_dtype"] == "float32"


def _assert_runtime_dtypes(model: TinyPolicyModel) -> None:
    assert model.backbone.input_dtype is torch.bfloat16
    assert model.backbone.weight_dtype is torch.bfloat16
    assert model.action_head.input_dtype is torch.float32
    assert model.action_head.weight_dtype is torch.float32
    assert model.action_head.output_dtype is torch.float32


def run_fsdp2(rank: int, world_size: int) -> None:
    torch.manual_seed(1234)
    model = TinyPolicyModel().cuda()
    mesh = init_device_mesh("cuda", (world_size,))

    dmuon.dedicate_params(
        model,
        mesh,
        predicate=lambda _name, param: param.requires_grad,
        hook_boundary_predicate=_hook_boundary_factory(model),
        param_policy=_param_policy(include_sharded_adamw=True),
    )
    fully_shard(model.backbone, mesh=mesh)
    fully_shard(model.action_head, mesh=mesh)
    fully_shard(model, mesh=mesh)

    optimizer = dmuon.Muon(model, lr=0.01, ns_steps=1, adamw_lr=0.001)
    _assert_policy_summary(model, optimizer)

    tokens = torch.randint(0, 16, (4, 3), device="cuda")
    optimizer.zero_grad()
    loss = model(tokens)
    assert torch.isfinite(loss.detach())
    _assert_runtime_dtypes(model)
    loss.backward()
    optimizer.step()

    model_sd = dmuon.get_model_state_dict(model, cpu_offload=False, rank0_only=False)
    assert model_sd["action_head.proj.weight"].dtype is torch.float32
    assert model_sd["backbone.proj.weight"].dtype is torch.float32

    if rank == 0:
        print("PASSED: fsdp2 param_policy mixed precision smoke", flush=True)


def run_ddp(rank: int, world_size: int) -> None:
    torch.manual_seed(1234)
    model = TinyPolicyModel().cuda()
    mesh = init_device_mesh("cuda", (world_size,))

    dmuon.dedicate_params_ddp(
        model,
        mesh,
        predicate=lambda _name, param: param.requires_grad,
        hook_boundary_predicate=_hook_boundary_factory(model),
        param_policy=_param_policy(include_sharded_adamw=False),
    )
    dmuon.replicate(model, mesh=mesh)

    assert model.backbone.proj.weight.dtype is torch.bfloat16
    assert model.action_head.proj.weight.dtype is torch.float32

    optimizer = dmuon.Muon(model, lr=0.01, ns_steps=1, adamw_lr=0.001)
    _assert_policy_summary(model, optimizer)

    tokens = torch.randint(0, 16, (4, 3), device="cuda")
    optimizer.zero_grad()
    loss = model(tokens)
    assert torch.isfinite(loss.detach())
    _assert_runtime_dtypes(model)
    loss.backward()
    optimizer.step()
    dmuon.wait_all_post_step_broadcasts(model)

    assert model.backbone.proj.weight.dtype is torch.bfloat16
    assert model.action_head.proj.weight.dtype is torch.float32
    model_sd = dmuon.get_model_state_dict(model, cpu_offload=False, rank0_only=False)
    assert model_sd["action_head.proj.weight"].dtype is torch.float32
    assert model_sd["backbone.proj.weight"].dtype is torch.float32

    if rank == 0:
        print("PASSED: ddp param_policy mixed precision smoke", flush=True)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "fsdp2"
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    if world_size != 2:
        raise RuntimeError(f"param_policy smoke expects 2 ranks, got {world_size}")

    if mode == "fsdp2":
        run_fsdp2(rank, world_size)
    elif mode == "ddp":
        run_ddp(rank, world_size)
    else:
        raise RuntimeError(f"unknown mode {mode!r}; expected 'fsdp2' or 'ddp'")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
