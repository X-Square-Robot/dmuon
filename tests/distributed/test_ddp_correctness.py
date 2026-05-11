"""P1 correctness tests for DMuon's DDP path.

Run with::

    torchrun --nproc_per_node=2 tests/distributed/test_ddp_correctness.py

Coverage:
  * forward_backward + step converges on 2 ranks;
  * dedicated params (``_owned_data`` AND live ``nn.Parameter``) are
    bit-identical across ranks after ``optimizer.step``;
  * non-dedicated params (ln/head) stay in sync via ``replicate``;
  * only owner's ``_reduced_grad`` is populated before step;
  * checkpoint roundtrip loads identically on all ranks.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dmuon


class Block(nn.Module):
    def __init__(self, d=128, ff=512):
        super().__init__()
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return x + self.down_proj(self.gate_proj(self.ln(x)) * self.up_proj(self.ln(x)))


class TinyModel(nn.Module):
    def __init__(self, num_layers=3, d=128, ff=512):
        super().__init__()
        self.layers = nn.ModuleList([Block(d, ff) for _ in range(num_layers)])
        self.head = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x).sum()


def _all_ranks_equal(t: torch.Tensor) -> bool:
    """Return True iff every rank holds the same tensor (within fp tol)."""
    world = dist.get_world_size()
    gathered = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(gathered, t.contiguous())
    for other in gathered[1:]:
        if not torch.allclose(gathered[0], other, rtol=0, atol=0):
            return False
    return True


def _setup_model_and_optim(seed=42, *, replicate_async=False):
    torch.manual_seed(seed)
    model = TinyModel().cuda()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = init_device_mesh("cuda", (world_size,))

    dmuon.dedicate_params_ddp(
        model, mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )
    dmuon.replicate(model, mesh=mesh)
    optimizer = dmuon.Muon(
        model, lr=0.02, momentum=0.95, ns_steps=3,
        adamw_lr=1e-3, adamw_weight_decay=0.0,
        replicate_async=replicate_async,
    )
    return model, optimizer, mesh, rank


def _pending_post_step_states(model) -> int:
    pending = 0
    for module in model.modules():
        state = getattr(module, "_dedicated_state", None)
        if state is None:
            continue
        group = state.group
        if getattr(group, "_post_step_broadcast_state", None) is not None:
            pending += 1
    return pending


def test_forward_backward_numerics():
    """Train 5 steps; loss should decrease monotonically on rank 0."""
    model, optimizer, _mesh, rank = _setup_model_and_optim()
    losses = []
    torch.manual_seed(123)  # same input on every rank; DDP averages same grad
    for step in range(5):
        optimizer.zero_grad()
        x = torch.randn(4, 128, device="cuda")
        loss = model(x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if rank == 0:
        print(f"[numerics] losses = {[f'{l:.4f}' for l in losses]}")
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


def test_post_step_broadcast_syncs_all_ranks():
    """After ``optimizer.step``, every rank's dedicated param must be identical."""
    model, optimizer, _mesh, rank = _setup_model_and_optim()
    optimizer.zero_grad()
    x = torch.randn(4, 128, device="cuda")
    loss = model(x)
    loss.backward()
    optimizer.step()

    # Drain any pending post-step broadcast (sync path already drained).
    dmuon.wait_all_post_step_broadcasts(model)

    errors = []
    for name, p in model.named_parameters():
        if not hasattr(p, "_dedicated_owner_rank"):
            continue
        if not _all_ranks_equal(p.data):
            errors.append(name)
    if rank == 0:
        assert not errors, f"Dedicated params diverged across ranks: {errors}"
        dedicated_count = sum(
            1 for _name, p in model.named_parameters()
            if hasattr(p, "_dedicated_owner_rank")
        )
        print(
            f"[post_step_broadcast] all {dedicated_count} dedicated params "
            "bit-identical across ranks"
        )


def test_reduce_to_owner_only():
    """After backward + wait_all_reduces, only the owner's _reduced_grad
    is populated for each dedicated param."""
    model, optimizer, mesh, rank = _setup_model_and_optim()

    # Hack the internals: run a backward but DON'T call optimizer.step so we
    # can inspect _reduced_grad directly.
    x = torch.randn(4, 128, device="cuda")
    loss = model(x)
    loss.backward()
    dmuon.wait_all_reduces(model)

    errors = []
    for module in model.modules():
        if not hasattr(module, "_dedicated_state"):
            continue
        for dp in module._dedicated_state.group.params:
            has_grad = dp._reduced_grad is not None
            if dp.is_owner and not has_grad:
                errors.append(f"{dp.param_name}: owner rank {rank} missing _reduced_grad")
            if (not dp.is_owner) and has_grad:
                errors.append(
                    f"{dp.param_name}: non-owner rank {rank} unexpectedly has _reduced_grad"
                )
    assert not errors, f"reduce_to_owner violations on rank {rank}: {errors}"
    if rank == 0:
        print("[reduce_to_owner] owners have _reduced_grad, non-owners cleared")


def test_replicate_avgs_non_dedicated_grads():
    """ln/head grads are all-reduced across ranks, so every rank has the
    same grad after backward."""
    model, optimizer, mesh, rank = _setup_model_and_optim()
    torch.manual_seed(rank + 1000)  # different input per rank!
    x = torch.randn(4, 128, device="cuda")
    loss = model(x)
    loss.backward()

    errors = []
    for name, p in model.named_parameters():
        if hasattr(p, "_dedicated_owner_rank"):
            continue
        if p.grad is None:
            continue
        if not _all_ranks_equal(p.grad.data):
            errors.append(name)
    if rank == 0:
        assert not errors, f"Non-dedicated grads diverged: {errors}"
        print("[replicate] all non-dedicated grads bit-identical across ranks")


def test_muon_grad_clip_scales_dedicated_owner_grads():
    """DMuon clip only scales dedicated/Muon gradients and step still works."""

    model, optimizer, _mesh, rank = _setup_model_and_optim()
    optimizer.zero_grad()
    x = torch.randn(4, 128, device="cuda")
    loss = model(x)
    loss.backward()

    stats = dmuon.clip_grad_norm_(optimizer, max_norm=1e-3)
    local_sq = torch.zeros((), device="cuda")
    for dp in optimizer._dedicated_params:
        if dp._reduced_grad is not None:
            local_sq += dp._reduced_grad.float().pow(2).sum()
    local_norm = local_sq.sqrt()
    assert local_norm.item() <= 1e-3 + 1e-7
    assert stats.param_count >= 0
    assert optimizer._grads_ready

    optimizer.step()
    for dp in optimizer._dedicated_params:
        assert dp._reduced_grad is None
    if rank == 0:
        print("[muon_grad_clip] dedicated grads clipped and optimizer.step completed")


def test_async_post_step_broadcast_leaves_pending_state_until_drain():
    """DDP async path should dispatch group states and drain them explicitly."""

    model, optimizer, _mesh, rank = _setup_model_and_optim(replicate_async=True)
    optimizer.zero_grad()
    x = torch.randn(4, 128, device="cuda")
    loss = model(x)
    loss.backward()
    optimizer.step()

    pending = _pending_post_step_states(model)
    assert pending > 0, f"rank {rank}: expected pending async post-step state"

    dmuon.wait_all_post_step_broadcasts(model)
    assert _pending_post_step_states(model) == 0

    errors = []
    for name, p in model.named_parameters():
        if hasattr(p, "_dedicated_owner_rank") and not _all_ranks_equal(p.data):
            errors.append(name)
    if rank == 0:
        assert not errors, f"Async DDP dedicated params diverged: {errors}"
        print("[async_post_step] pending states drain and params sync")


def test_async_matches_sync_loss_trajectory():
    """Group-pipelined async DDP should match the sync post-step path."""

    batches = []
    for step in range(3):
        torch.manual_seed(7000 + step)
        batches.append(torch.randn(4, 128, device="cuda"))

    def run(*, replicate_async: bool) -> torch.Tensor:
        model, optimizer, _mesh, _rank = _setup_model_and_optim(
            seed=777, replicate_async=replicate_async
        )
        losses = []
        for x in batches:
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        dmuon.wait_all_post_step_broadcasts(model)
        return torch.tensor(losses, device="cuda")

    sync_losses = run(replicate_async=False)
    dist.barrier()
    async_losses = run(replicate_async=True)
    dist.barrier()

    if not torch.allclose(sync_losses, async_losses, rtol=0, atol=1e-6):
        raise AssertionError(
            f"async losses diverged from sync: sync={sync_losses.tolist()} "
            f"async={async_losses.tolist()}"
        )
    if dist.get_rank() == 0:
        print("[async_vs_sync] loss trajectories match")


def test_checkpoint_roundtrip():
    """Save → load → verify every rank holds identical parameters."""
    model, optimizer, _, rank = _setup_model_and_optim()

    # Take one step so optimizer state exists.
    optimizer.zero_grad()
    x = torch.randn(4, 128, device="cuda")
    loss = model(x)
    loss.backward()
    optimizer.step()

    dmuon.wait_all_post_step_broadcasts(model)

    model_sd = dmuon.get_model_state_dict(model, cpu_offload=True)
    optim_sd = dmuon.get_optimizer_state_dict(model, optimizer, cpu_offload=True)

    tmp = None
    if rank == 0:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt").name
        torch.save({"model": model_sd, "optim": optim_sd}, tmp)
    objs = [tmp]
    dist.broadcast_object_list(objs, src=0)
    tmp = objs[0]
    dist.barrier()

    # Reset model, then load back.
    torch.manual_seed(999)
    # Just perturb params and then restore via ckpt.
    with torch.no_grad():
        for p in model.parameters():
            p.data.add_(torch.randn_like(p.data) * 0.01)

    ckpt = torch.load(tmp, map_location="cpu", weights_only=False)
    dmuon.set_model_state_dict(model, ckpt["model"])
    dmuon.set_optimizer_state_dict(model, optimizer, ckpt["optim"])
    dist.barrier()

    errors = []
    for name, p in model.named_parameters():
        if not _all_ranks_equal(p.data):
            errors.append(name)
    if rank == 0:
        assert not errors, f"Checkpoint reload diverged across ranks: {errors}"
        print("[checkpoint] roundtrip identical on all ranks")
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    dist.barrier()


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    tests = [
        test_forward_backward_numerics,
        test_post_step_broadcast_syncs_all_ranks,
        test_reduce_to_owner_only,
        test_replicate_avgs_non_dedicated_grads,
        test_muon_grad_clip_scales_dedicated_owner_grads,
        test_async_post_step_broadcast_leaves_pending_state_until_drain,
        test_async_matches_sync_loss_trajectory,
        test_checkpoint_roundtrip,
    ]

    for t in tests:
        if rank == 0:
            print(f"\n==== {t.__name__} ====")
        t()
        dist.barrier()

    if rank == 0:
        print("\nALL DDP P1 TESTS PASSED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
