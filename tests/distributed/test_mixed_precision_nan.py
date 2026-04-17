"""Test: mixed precision + dedicated params NaN reproduction.

Runs multiple trials to detect sporadic NaN from async reduce race conditions.

Run with: torchrun --nproc_per_node=2 tests/distributed/test_mixed_precision_nan.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dmuon
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


class FusedAttn(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.in_proj_qkv = nn.Linear(d, d * 3, bias=False)
        self.in_proj_zba = nn.Linear(d, d + 32, bias=False)
        self._zba_split = [d, 16, 16]
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        qkv = self.in_proj_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        W = self.in_proj_zba.weight
        Wz, Wb, Wa = torch.split(W, self._zba_split, dim=0)
        z = F.linear(x, Wz)
        return self.out_proj(v * torch.sigmoid(z))


class Layer(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.attn = FusedAttn(d)
        self.mlp_gate = nn.Linear(d, d * 2, bias=False)
        self.mlp_down = nn.Linear(d * 2, d, bias=False)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        h = F.silu(self.mlp_gate(self.norm2(x)))
        return x + self.mlp_down(h)


class Model(nn.Module):
    def __init__(self, n_layers=2, d=128):
        super().__init__()
        self.layers = nn.ModuleList([Layer(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x).sum()


def run_trial(run_id, mesh, mp_policy, n_steps=10):
    """Run one trial, return (got_nan, nan_step, nan_param_name)."""
    rank = dist.get_rank()

    torch.manual_seed(0)
    model = Model().cuda().float()

    dmuon.dedicate_params(
        model, mesh,
        predicate=lambda n, p: p.ndim == 2 and "norm" not in n and "head" not in n,
        compute_dtype=torch.bfloat16,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    ns = dmuon.NewtonSchulz("direct")
    opt = dmuon.Muon(model, lr=0.02, momentum=0.95, ns_backend=ns, adamw_lr=1e-4)

    torch.manual_seed(42 + run_id)
    for step in range(n_steps):
        opt.zero_grad()
        x = torch.randn(4, 128, device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x)

        if torch.isnan(loss):
            # Find which param NaN'd
            nan_param = "unknown"
            for dp in dmuon.get_owned_params(model, rank):
                if dp._owned_data is not None and torch.isnan(dp._owned_data).any():
                    nan_param = f"{dp.param_name} {tuple(dp._owned_data.shape)}"
                    break
            del model, opt
            torch.cuda.empty_cache()
            return True, step, nan_param

        loss.backward()
        opt.step()

    final_loss = loss.item()
    del model, opt
    torch.cuda.empty_cache()
    return False, n_steps, f"OK loss={final_loss:.4f}"


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    mesh = dist.device_mesh.init_device_mesh("cuda", (dist.get_world_size(),))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)

    n_trials = 10
    nan_count = 0
    results = []

    for i in range(n_trials):
        dist.barrier()
        got_nan, step, info = run_trial(i, mesh, mp)
        if got_nan:
            nan_count += 1
        results.append((got_nan, step, info))
        if rank == 0:
            status = f"NaN at step {step} ({info})" if got_nan else info
            print(f"Trial {i:2d}: {status}", flush=True)

    if rank == 0:
        print(f"\nNaN rate: {nan_count}/{n_trials}")
        if nan_count > 0:
            print("ISSUE: Sporadic NaN detected — likely async reduce race condition")
        else:
            print("PASSED: No NaN in any trial")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
