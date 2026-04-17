"""Multi-GPU test: direct .weight access on dedicated params.

Tests that the GatedDeltaNet pattern (self.module.weight + F.linear) works
correctly with dmuon dedicated params in a distributed setting.

Compares two identical models:
  A) Fused Linear + direct .weight access + F.linear (GatedDeltaNet pattern)
  B) Separate Linears + module.forward() (standard pattern)

Both should produce identical forward results and non-NaN optimizer steps.

Run with: torchrun --nproc_per_node=2 tests/distributed/test_direct_weight_access.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dmuon
from torch.distributed.fsdp import fully_shard


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Model A: Fused proj + direct .weight access (GatedDeltaNet pattern)
# ---------------------------------------------------------------------------

class FusedAttn(nn.Module):
    """One fused linear, split weight manually."""

    def __init__(self, d=128, z_dim=128, ba_dim=16):
        super().__init__()
        self.in_proj_qkv = nn.Linear(d, d * 3, bias=False)
        self.in_proj_zba = nn.Linear(d, z_dim + ba_dim * 2, bias=False)
        self._zba_split = [z_dim, ba_dim, ba_dim]
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        qkv = self.in_proj_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Direct .weight access pattern
        W_zba = self.in_proj_zba.weight
        W_z, W_b, W_a = torch.split(W_zba, self._zba_split, dim=0)
        z = F.linear(x, W_z)
        b = F.linear(x, W_b)
        a = F.linear(x, W_a)

        return self.out_proj(v * torch.sigmoid(z))


# ---------------------------------------------------------------------------
# Model B: Separate linears + module.forward() (standard pattern)
# ---------------------------------------------------------------------------

class SeparateAttn(nn.Module):
    """Separate linears, use module.forward()."""

    def __init__(self, d=128, z_dim=128, ba_dim=16):
        super().__init__()
        self.in_proj_qkv = nn.Linear(d, d * 3, bias=False)
        self.in_proj_z = nn.Linear(d, z_dim, bias=False)
        self.in_proj_b = nn.Linear(d, ba_dim, bias=False)
        self.in_proj_a = nn.Linear(d, ba_dim, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        qkv = self.in_proj_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        z = self.in_proj_z(x)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        return self.out_proj(v * torch.sigmoid(z))


class TestLayer(nn.Module):
    def __init__(self, attn_module, d=128):
        super().__init__()
        self.attn = attn_module
        self.mlp_gate = nn.Linear(d, d * 2, bias=False)
        self.mlp_down = nn.Linear(d * 2, d, bias=False)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        h = F.silu(self.mlp_gate(self.norm2(x)))
        x = x + self.mlp_down(h)
        return x


class TestModel(nn.Module):
    def __init__(self, attn_cls, n_layers=2, d=128):
        super().__init__()
        self.layers = nn.ModuleList([
            TestLayer(attn_cls(d), d) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x).sum()


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_model(name, model, mesh, rank, x, n_steps=5):
    """Run n_steps of training, return (losses, has_nan)."""
    dmuon.dedicate_params(
        model, mesh,
        predicate=lambda n, p: p.ndim == 2 and "norm" not in n and "head" not in n,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    ns = dmuon.NewtonSchulz("direct")
    optimizer = dmuon.Muon(model, lr=0.02, momentum=0.95, ns_backend=ns, adamw_lr=1e-3)

    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = model(x)
        losses.append(loss.item())

        if torch.isnan(loss):
            log(rank, f"  [{name}] step {step}: NaN loss!")
            # Dump diagnostics
            for dp in dmuon.get_owned_params(model, rank):
                if dp._owned_data is not None and torch.isnan(dp._owned_data).any():
                    log(rank, f"    NaN in owned: {dp.param_name} {tuple(dp._owned_data.shape)}")
            return losses, True

        loss.backward()

        # Diagnostic: check _reduced_grad before step
        dmuon.wait_all_reduces(model)
        for dp in optimizer._dedicated_params:
            if dp._reduced_grad is not None and torch.isnan(dp._reduced_grad).any():
                log(rank, f"  [{name}] step {step}: NaN in _reduced_grad "
                    f"{dp.param_name} {tuple(dp._reduced_grad.shape)}")

        optimizer.step()

        # Diagnostic: check owned_data after step
        for dp in dmuon.get_owned_params(model, rank):
            if dp._owned_data is not None and torch.isnan(dp._owned_data).any():
                log(rank, f"  [{name}] step {step}: NaN after step in "
                    f"{dp.param_name} {tuple(dp._owned_data.shape)}")
                return losses, True

        log(rank, f"  [{name}] step {step}: loss={loss.item():.6f}")

    return losses, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    mesh = dist.device_mesh.init_device_mesh("cuda", (world_size,))

    # Same seed, same weights
    torch.manual_seed(42)
    x = torch.randn(4, 128, device="cuda")

    # --- Model A: Fused + direct weight access ---
    log(rank, "\n=== Model A: Fused + direct .weight access ===")
    torch.manual_seed(0)
    model_a = TestModel(FusedAttn).to("cuda")
    losses_a, nan_a = run_model("Fused", model_a, mesh, rank, x)
    del model_a
    torch.cuda.empty_cache()
    dist.barrier()

    # --- Model B: Separate + module.forward() ---
    log(rank, "\n=== Model B: Separate + module.forward() ===")
    torch.manual_seed(0)
    model_b = TestModel(SeparateAttn).to("cuda")
    losses_b, nan_b = run_model("Separate", model_b, mesh, rank, x)
    del model_b
    torch.cuda.empty_cache()
    dist.barrier()

    # --- Summary ---
    log(rank, "\n=== SUMMARY ===")
    log(rank, f"  Fused (direct .weight):  nan={nan_a}  losses={[f'{l:.4f}' for l in losses_a]}")
    log(rank, f"  Separate (module.forward): nan={nan_b}  losses={[f'{l:.4f}' for l in losses_b]}")

    if nan_a and not nan_b:
        log(rank, "  ISSUE: Direct .weight access causes NaN but module.forward() does not")
    elif nan_a and nan_b:
        log(rank, "  ISSUE: Both NaN — not a direct access issue, likely NS/optimizer issue")
    elif not nan_a and not nan_b:
        log(rank, "  PASSED: Both models train without NaN")
    else:
        log(rank, "  UNEXPECTED: Separate NaN but Fused OK")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
