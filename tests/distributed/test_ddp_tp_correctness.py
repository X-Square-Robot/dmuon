"""Correctness smoke for the supported DDP+TP path.

Run with:

    torchrun --nproc_per_node=4 tests/distributed/test_ddp_tp_correctness.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import dmuon


class Block(nn.Module):
    def __init__(self, d: int = 64, ff: int = 128):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        return x + self.down_proj(self.gate_proj(h) * self.up_proj(h))


class TinyModel(nn.Module):
    def __init__(self, layers: int = 2, d: int = 64, ff: int = 128):
        super().__init__()
        self.layers = nn.ModuleList([Block(d, ff) for _ in range(layers)])
        self.head = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.head(x).sum()


def _model_digest(model: nn.Module) -> torch.Tensor:
    parts = []
    for p in model.parameters():
        local = p._local_tensor if hasattr(p, "_local_tensor") else p.data
        parts.append(local.detach().float().sum())
    return torch.stack(parts).sum()


def _run(replicate_async: bool) -> tuple[list[float], float]:
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))

    torch.manual_seed(1234)
    model = TinyModel().to(device).to(torch.bfloat16)
    for layer in model.layers:
        parallelize_module(
            layer,
            mesh["tp"],
            {
                "gate_proj": ColwiseParallel(),
                "up_proj": ColwiseParallel(),
                "down_proj": RowwiseParallel(),
            },
        )

    dmuon.dedicate_params_ddp_tp(
        model,
        mesh["dp"],
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )
    dmuon.replicate_tp(model, mesh=mesh["dp"])
    optimizer = dmuon.Muon(
        model,
        lr=0.01,
        momentum=0.9,
        ns_steps=3,
        adamw_lr=1e-3,
        adamw_weight_decay=0.0,
        replicate_async=replicate_async,
    )

    losses: list[float] = []
    for step in range(3):
        optimizer.zero_grad()
        torch.manual_seed(9000 + step)
        x = torch.randn(4, 8, 64, device=device, dtype=torch.bfloat16)
        loss = model(x)
        assert torch.isfinite(loss).all().item()
        loss.backward()
        optimizer.step()
        dmuon.wait_all_post_step_broadcasts(model)
        losses.append(float(loss.detach().cpu()))

    digest = _model_digest(model)
    # Same TP-coordinate DP peers should hold identical local shards.
    gathered = [torch.zeros_like(digest) for _ in range(mesh["dp"].size())]
    dist.all_gather(gathered, digest, group=mesh["dp"].get_group())
    for other in gathered[1:]:
        assert torch.allclose(gathered[0], other, rtol=0, atol=0)
    return losses, float(digest.detach().cpu())


def main() -> None:
    dist.init_process_group("nccl")
    try:
        assert dist.get_world_size() == 4, "DDP+TP smoke requires 4 GPUs"
        sync_losses, sync_digest = _run(replicate_async=False)
        async_losses, async_digest = _run(replicate_async=True)
        if dist.get_rank() == 0:
            print(f"sync_losses={sync_losses}")
            print(f"async_losses={async_losses}")
            assert sync_losses == async_losses
            assert sync_digest == async_digest
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
