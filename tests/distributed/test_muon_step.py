"""Multi-GPU tests for Muon optimizer step correctness.

Run with: torchrun --nproc_per_node=4 tests/distributed/test_muon_step.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dmuon
from torch.distributed.fsdp import fully_shard


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# TinyModel
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, hidden=256, intermediate=1024):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, hidden=256, intermediate=1024):
        super().__init__()
        self.mlp = MLP(hidden, intermediate)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class TinyModel(nn.Module):
    def __init__(self, num_layers=4, hidden=256, intermediate=1024):
        super().__init__()
        self.layers = nn.ModuleList(
            [Block(hidden, intermediate) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x).sum()


# ---------------------------------------------------------------------------
# Test 1: reduced grad is set on all owned dedicated params after backward
# ---------------------------------------------------------------------------

def test_muon_reduced_grad_all_set(rank, world_size, device, mesh):

    torch.manual_seed(0)
    model = TinyModel().to(device)

    dmuon.dedicate_params(
        model, mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )

    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    optimizer = dmuon.Muon(model, lr=0.01, ns_steps=5, adamw_lr=0.01)

    optimizer.zero_grad()
    x = torch.randn(4, 256, device=device)
    loss = model(x)
    loss.backward()

    dmuon.wait_all_reduces(model)

    for dp in optimizer._dedicated_params:
        assert dp._reduced_grad is not None, (
            f"Rank {rank}: _reduced_grad is None for param with shape {dp._orig_size}"
        )
        assert dp._reduced_grad.shape == dp._orig_size, (
            f"Rank {rank}: shape mismatch {dp._reduced_grad.shape} vs {dp._orig_size}"
        )
        assert dp._reduced_grad.abs().max().item() > 0, (
            f"Rank {rank}: _reduced_grad is all zeros for param with shape {dp._orig_size}"
        )

    torch.cuda.synchronize()
    log(rank, "PASSED: test_muon_reduced_grad_all_set")


# ---------------------------------------------------------------------------
# Test 2: momentum buffer accumulates correctly
# ---------------------------------------------------------------------------

def test_muon_momentum_accumulation(rank, world_size, device, mesh):

    torch.manual_seed(0)
    model = TinyModel().to(device)

    dmuon.dedicate_params(
        model, mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )

    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    optimizer = dmuon.Muon(model, lr=0.01, momentum=0.95, ns_steps=5, adamw_lr=0.01)

    torch.manual_seed(42)
    x = torch.randn(4, 256, device=device)

    # Step 1: forward -> backward -> step, record momentum buffer
    optimizer.zero_grad()
    loss = model(x)
    loss.backward()
    optimizer.step()

    if len(optimizer._dedicated_params) == 0:
        log(rank, "PASSED: test_muon_momentum_accumulation (no owned params on this rank)")
        return

    dp = optimizer._dedicated_params[0]
    dp_id = id(dp)
    assert dp_id in optimizer.state, f"Rank {rank}: no state for first dedicated param"
    buf1 = optimizer.state[dp_id]["momentum_buffer"].clone()

    # Step 2: forward -> backward, capture grad before step, then step
    optimizer.zero_grad()
    loss = model(x)
    loss.backward()

    dmuon.wait_all_reduces(model)

    assert dp._reduced_grad is not None, (
        f"Rank {rank}: _reduced_grad is None before step 2"
    )
    grad2 = dp._reduced_grad.view(dp._reduced_grad.shape[0], -1).clone()

    optimizer.step()

    buf2 = optimizer.state[dp_id]["momentum_buffer"].clone()

    # Verify: buf2 ~= 0.95 * buf1 + grad2
    expected = 0.95 * buf1 + grad2
    max_abs_diff = (buf2 - expected).abs().max().item()
    buf2_scale = buf2.abs().max().item()
    tolerance = 0.01 * buf2_scale if buf2_scale > 0 else 0.01

    assert max_abs_diff < tolerance, (
        f"Rank {rank}: momentum mismatch. max_abs_diff={max_abs_diff:.6f}, "
        f"tolerance={tolerance:.6f} (1% of buf2 max {buf2_scale:.6f})"
    )

    # Verify shape: buf should be 2D matching param reshaped
    expected_rows = dp._orig_size[0]
    expected_cols = dp._orig_size.numel() // expected_rows
    assert buf2.shape == (expected_rows, expected_cols), (
        f"Rank {rank}: buf shape {buf2.shape} != expected ({expected_rows}, {expected_cols})"
    )

    torch.cuda.synchronize()
    log(rank, "PASSED: test_muon_momentum_accumulation")


# ---------------------------------------------------------------------------
# Test 3: Nesterov momentum — NS receives grad + μ*buf, not just buf
# ---------------------------------------------------------------------------

def test_muon_nesterov(rank, world_size, device, mesh):
    """Verify Nesterov vs non-Nesterov produce different weight updates.

    With nesterov=True (default), NS input is grad + μ*buf (lookahead).
    With nesterov=False, NS input is buf.
    The resulting weight updates should differ, proving Nesterov is active.
    """

    lr = 0.01
    mu = 0.95

    # --- Run with Nesterov=True ---
    torch.manual_seed(0)
    model1 = TinyModel().to(device)
    dmuon.dedicate_params(model1, mesh, predicate=lambda n, p: "proj" in n and p.ndim == 2)
    for layer in model1.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model1, mesh=mesh)
    opt1 = dmuon.Muon(model1, lr=lr, momentum=mu, nesterov=True, adamw_lr=lr)

    torch.manual_seed(42)
    x = torch.randn(4, 256, device=device)
    opt1.zero_grad()
    model1(x).backward()
    opt1.step()

    # --- Run with Nesterov=False ---
    torch.manual_seed(0)
    model2 = TinyModel().to(device)
    dmuon.dedicate_params(model2, mesh, predicate=lambda n, p: "proj" in n and p.ndim == 2)
    for layer in model2.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model2, mesh=mesh)
    opt2 = dmuon.Muon(model2, lr=lr, momentum=mu, nesterov=False, adamw_lr=lr)

    torch.manual_seed(42)
    x = torch.randn(4, 256, device=device)
    opt2.zero_grad()
    model2(x).backward()
    opt2.step()

    # --- Compare: weights should differ ---
    if len(opt1._dedicated_params) == 0:
        log(rank, "PASSED: test_muon_nesterov (no owned params)")
        return

    dp1 = opt1._dedicated_params[0]
    dp2 = opt2._dedicated_params[0]
    w1 = dp1._owned_data
    w2 = dp2._owned_data

    diff = (w1 - w2).abs().max().item()
    assert diff > 1e-6, (
        f"Rank {rank}: Nesterov=True and False produced identical weights "
        f"(diff={diff}). Nesterov is not active."
    )

    # Also verify momentum buffers are the SAME (Nesterov only changes NS input, not buf)
    buf1 = opt1.state[id(dp1)]["momentum_buffer"]
    buf2 = opt2.state[id(dp2)]["momentum_buffer"]
    buf_diff = (buf1 - buf2).abs().max().item()
    assert buf_diff < 1e-6, (
        f"Rank {rank}: momentum buffers differ (diff={buf_diff}). "
        f"Nesterov should not affect buf accumulation."
    )

    torch.cuda.synchronize()
    log(rank, f"PASSED: test_muon_nesterov (weight diff={diff:.6f}, buf diff={buf_diff:.2e})")


# ---------------------------------------------------------------------------
# Test 4: PyTorch-style param_groups on FSDP2-wrapped model
# ---------------------------------------------------------------------------

def test_muon_param_groups_fsdp2(rank, world_size, device, mesh):
    torch.manual_seed(0)
    model = TinyModel().to(device)

    dmuon.dedicate_params(
        model, mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )

    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    base_params = []
    action_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("layers.0."):
            action_params.append(param)
        else:
            base_params.append(param)

    optimizer = dmuon.Muon(
        model,
        lr=0.01,
        ns_steps=5,
        adamw_lr=0.001,
        param_groups=[
            {
                "params": base_params,
                "group_name": "base",
                "muon_lr": 0.01,
                "adamw_lr": 0.001,
                "adamw_weight_decay": 0.0,
            },
            {
                "params": action_params,
                "group_name": "action",
                "muon_lr": 0.02,
                "adamw_lr": 0.002,
                "adamw_weight_decay": 0.0,
            },
        ],
    )

    expected = [
        ("base/muon", True, "muon", 0.01),
        ("base/adamw", False, "adamw", 0.001),
        ("action/muon", True, "muon", 0.02),
        ("action/adamw", False, "adamw", 0.002),
    ]
    actual = [
        (
            group["group_name"],
            group["use_muon"],
            group["subgroup_type"],
            group["lr"],
        )
        for group in optimizer.param_groups
    ]
    assert actual == expected
    group_idx = {
        group["group_name"]: idx for idx, group in enumerate(optimizer.param_groups)
    }
    assert group_idx["action/muon"] in optimizer._muon_group_dps
    assert optimizer._muon_group_dps[group_idx["action/muon"]]
    assert optimizer._adamw_group_params[group_idx["action/adamw"]]

    optimizer.zero_grad()
    x = torch.randn(4, 256, device=device)
    loss = model(x)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    log(rank, "PASSED: test_muon_param_groups_fsdp2")


def test_muon_all_trainable_type_split_fsdp2(rank, world_size, device, mesh):
    def build_type_split_model_and_optimizer(seed):
        torch.manual_seed(seed)
        model = TinyModel().to(device)
        matrix_names = set()
        base_names = set()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "proj" in name and param.ndim == 2:
                matrix_names.add(name)
            else:
                base_names.add(name)

        dmuon.dedicate_params(
            model,
            mesh,
            predicate=lambda _n, p: p.requires_grad,
        )
        matrix_params = []
        base_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in matrix_names:
                matrix_params.append(param)
            elif name in base_names:
                base_params.append(param)

        for layer in model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)

        optimizer = dmuon.Muon(
            model,
            lr=0.01,
            ns_steps=2,
            adamw_lr=0.001,
            param_groups=[
                {
                    "params": matrix_params,
                    "group_name": "matrix",
                    "dmuon_route": "muon",
                    "muon_lr": 0.01,
                },
                {
                    "params": base_params,
                    "group_name": "base",
                    "dmuon_route": "adamw",
                    "adamw_lr": 0.001,
                    "adamw_weight_decay": 0.0,
                },
            ],
        )
        return model, optimizer

    model, optimizer = build_type_split_model_and_optimizer(seed=0)

    group_idx = {
        group["group_name"]: idx for idx, group in enumerate(optimizer.param_groups)
    }
    matrix_dps = optimizer._muon_group_dps[group_idx["matrix/muon"]]
    dedicated_adamw_dps = [
        dp
        for dp in optimizer._all_dedicated_params
        if id(dp) in optimizer._dp_to_adamw_group_idx
    ]
    assert matrix_dps
    assert dedicated_adamw_dps

    optimizer.zero_grad()
    x = torch.randn(4, 256, device=device)
    loss = model(x)
    loss.backward()
    optimizer.step()

    dedicated_adamw_state_count = sum(
        1
        for key, state in optimizer.state.items()
        if isinstance(key, int) and "exp_avg" in state
    )
    assert dedicated_adamw_state_count > 0

    model_sd = dmuon.get_model_state_dict(model, cpu_offload=True, rank0_only=False)
    optim_sd = dmuon.get_optimizer_state_dict(
        model,
        optimizer,
        cpu_offload=True,
        rank0_only=False,
    )
    assert optim_sd["dedicated_adamw"], "checkpoint missing dedicated AdamW state"
    for fqn, state in optim_sd["dedicated_adamw"].items():
        assert "step" in state, f"{fqn}: missing AdamW step"
        assert state["step"] > 0, f"{fqn}: AdamW step was not saved"
        assert "exp_avg" in state, f"{fqn}: missing AdamW exp_avg"
        assert "exp_avg_sq" in state, f"{fqn}: missing AdamW exp_avg_sq"

    reloaded_model, reloaded_optimizer = build_type_split_model_and_optimizer(seed=1234)
    dmuon.set_model_state_dict(reloaded_model, model_sd)
    dmuon.set_optimizer_state_dict(reloaded_model, reloaded_optimizer, optim_sd)
    restored_adamw_state_count = sum(
        1
        for key, state in reloaded_optimizer.state.items()
        if isinstance(key, int) and "exp_avg" in state and "exp_avg_sq" in state
    )
    assert restored_adamw_state_count == dedicated_adamw_state_count, (
        f"dedicated AdamW state count mismatch after checkpoint load: "
        f"{restored_adamw_state_count} != {dedicated_adamw_state_count}"
    )

    torch.cuda.synchronize()
    log(rank, "PASSED: test_muon_all_trainable_type_split_fsdp2")


def test_muon_all_trainable_type_split_hsdp(rank, world_size, device, mesh):
    assert world_size >= 4, "HSDP type-split test requires at least 4 GPUs"

    from torch.distributed.device_mesh import init_device_mesh

    hsdp_mesh = init_device_mesh(
        "cuda", (2, world_size // 2), mesh_dim_names=("replicate", "shard")
    )
    replicate_mesh = hsdp_mesh["replicate"]
    shard_mesh = hsdp_mesh["shard"]

    def build_model_and_optimizer(seed):
        torch.manual_seed(seed)
        model = TinyModel(num_layers=2, hidden=128, intermediate=512).to(device)
        matrix_names = set()
        base_names = set()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "proj" in name and param.ndim == 2:
                matrix_names.add(name)
            else:
                base_names.add(name)

        dmuon.dedicate_params(
            model,
            shard_mesh,
            replicate_mesh=replicate_mesh,
            predicate=lambda _n, p: p.requires_grad,
        )
        matrix_params = []
        base_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in matrix_names:
                matrix_params.append(param)
            elif name in base_names:
                base_params.append(param)

        for layer in model.layers:
            fully_shard(layer, mesh=hsdp_mesh)
        fully_shard(model, mesh=hsdp_mesh)

        optimizer = dmuon.Muon(
            model,
            lr=0.01,
            ns_steps=2,
            adamw_lr=0.001,
            param_groups=[
                {
                    "params": matrix_params,
                    "group_name": "matrix",
                    "dmuon_route": "muon",
                    "muon_lr": 0.01,
                },
                {
                    "params": base_params,
                    "group_name": "base",
                    "dmuon_route": "adamw",
                    "adamw_lr": 0.001,
                    "adamw_weight_decay": 0.0,
                },
            ],
        )
        return model, optimizer

    model, optimizer = build_model_and_optimizer(0)

    adamw_dps = [
        dp
        for dp in optimizer._all_dedicated_params
        if id(dp) in optimizer._dp_to_adamw_group_idx
    ]
    assert adamw_dps, "expected dedicated AdamW-route params"
    assert all(
        not getattr(dp, "_dmuon_adamw_replicate_allreduce", False)
        for dp in adamw_dps
    )

    torch.manual_seed(42)
    x = torch.randn(4, 128, device=device)
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(x)
        assert torch.isfinite(loss.detach())
        loss.backward()
        optimizer.step()

    local_adamw_state_count = sum(
        1
        for dp in adamw_dps
        if dp.is_owner
        and id(dp) in optimizer.state
        and "exp_avg" in optimizer.state[id(dp)]
    )
    expected_local_count = sum(1 for dp in adamw_dps if dp.is_owner)
    assert local_adamw_state_count == expected_local_count, (
        f"local dedicated AdamW state count mismatch: "
        f"{local_adamw_state_count} != {expected_local_count}"
    )

    model_sd = dmuon.get_model_state_dict(model, cpu_offload=True, rank0_only=False)
    optim_sd = dmuon.get_optimizer_state_dict(
        model,
        optimizer,
        cpu_offload=True,
        rank0_only=False,
    )
    assert optim_sd["dedicated_adamw"], "checkpoint missing dedicated AdamW state"

    reloaded_model, reloaded_optimizer = build_model_and_optimizer(1234)
    dmuon.set_model_state_dict(reloaded_model, model_sd)
    dmuon.set_optimizer_state_dict(reloaded_model, reloaded_optimizer, optim_sd)
    reloaded_adamw_dps = [
        dp
        for dp in reloaded_optimizer._all_dedicated_params
        if id(dp) in reloaded_optimizer._dp_to_adamw_group_idx
    ]
    restored_local_count = sum(
        1
        for dp in reloaded_adamw_dps
        if dp.is_owner
        and id(dp) in reloaded_optimizer.state
        and "exp_avg" in reloaded_optimizer.state[id(dp)]
    )
    expected_reloaded_count = sum(
        1 for dp in reloaded_adamw_dps if dp.is_owner
    )
    assert restored_local_count == expected_reloaded_count, (
        f"restored local dedicated AdamW state count mismatch: "
        f"{restored_local_count} != {expected_reloaded_count}"
    )

    torch.cuda.synchronize()
    log(rank, "PASSED: test_muon_all_trainable_type_split_hsdp")


def test_muon_sharded_base_adamw_fsdp2(rank, world_size, device, mesh):
    def build_model_and_optimizer(seed):
        torch.manual_seed(seed)
        model = TinyModel(num_layers=2, hidden=128, intermediate=512).to(device)
        matrix_names = set()
        sharded_base_names = {"head.weight"}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "proj" in name and param.ndim == 2:
                matrix_names.add(name)

        dmuon.dedicate_params(
            model,
            mesh,
            predicate=lambda _n, p: p.requires_grad,
            route_hint_fn=lambda n, _p: (
                "sharded_adamw"
                if n in sharded_base_names
                else ("muon" if n in matrix_names else "adamw")
            ),
        )
        for layer in model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)

        optimizer = dmuon.Muon(
            model,
            lr=0.01,
            ns_steps=2,
            adamw_lr=0.001,
            adamw_weight_decay=0.0,
        )
        return model, optimizer

    model, optimizer = build_model_and_optimizer(0)
    sharded_dps = [
        dp
        for dp in optimizer._all_dedicated_params
        if getattr(dp, "_dmuon_route", None) == "sharded_adamw"
    ]
    assert sharded_dps, "expected sharded AdamW-route dedicated params"
    assert all(getattr(dp, "_sharded_adamw_data", None) is not None for dp in sharded_dps)
    assert all(id(dp) in optimizer._dp_to_adamw_group_idx for dp in sharded_dps)

    torch.manual_seed(42)
    x = torch.randn(4, 128, device=device)
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(x)
        assert torch.isfinite(loss.detach())
        loss.backward()
        optimizer.step()

    state_count = sum(
        1
        for dp in sharded_dps
        if id(dp) in optimizer.state and "exp_avg" in optimizer.state[id(dp)]
    )
    assert state_count == len(sharded_dps)

    model_sd = dmuon.get_model_state_dict(model, cpu_offload=True, rank0_only=False)
    optim_sd = dmuon.get_optimizer_state_dict(
        model,
        optimizer,
        cpu_offload=True,
        rank0_only=False,
    )
    assert optim_sd["dedicated_adamw"], "checkpoint missing sharded AdamW state"

    reloaded_model, reloaded_optimizer = build_model_and_optimizer(1234)
    dmuon.set_model_state_dict(reloaded_model, model_sd)
    dmuon.set_optimizer_state_dict(reloaded_model, reloaded_optimizer, optim_sd)
    reloaded_sharded_dps = [
        dp
        for dp in reloaded_optimizer._all_dedicated_params
        if getattr(dp, "_dmuon_route", None) == "sharded_adamw"
    ]
    restored_count = sum(
        1
        for dp in reloaded_sharded_dps
        if id(dp) in reloaded_optimizer.state
        and "exp_avg" in reloaded_optimizer.state[id(dp)]
        and tuple(reloaded_optimizer.state[id(dp)]["exp_avg"].shape)
        == tuple(dp._sharded_adamw_data.shape)
    )
    assert restored_count == len(reloaded_sharded_dps)

    torch.cuda.synchronize()
    log(rank, "PASSED: test_muon_sharded_base_adamw_fsdp2")


def test_muon_sharded_base_adamw_hsdp(rank, world_size, device, mesh):
    assert world_size >= 4, "HSDP sharded AdamW test requires at least 4 GPUs"

    from torch.distributed.device_mesh import init_device_mesh

    hsdp_mesh = init_device_mesh(
        "cuda", (2, world_size // 2), mesh_dim_names=("replicate", "shard")
    )
    replicate_mesh = hsdp_mesh["replicate"]
    shard_mesh = hsdp_mesh["shard"]

    torch.manual_seed(0)
    model = TinyModel(num_layers=2, hidden=128, intermediate=512).to(device)
    matrix_names = set()
    sharded_base_names = {"head.weight"}
    base_names = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "proj" in name and param.ndim == 2:
            matrix_names.add(name)
        elif name not in sharded_base_names:
            base_names.add(name)

    dmuon.dedicate_params(
        model,
        shard_mesh,
        replicate_mesh=replicate_mesh,
        predicate=lambda _n, p: p.requires_grad,
        route_hint_fn=lambda n, _p: (
            "sharded_adamw"
            if n in sharded_base_names
            else ("muon" if n in matrix_names else "adamw")
        ),
    )

    matrix_params = []
    sharded_base_params = []
    base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in matrix_names:
            matrix_params.append(param)
        elif name in sharded_base_names:
            sharded_base_params.append(param)
        elif name in base_names:
            base_params.append(param)

    for layer in model.layers:
        fully_shard(layer, mesh=hsdp_mesh)
    fully_shard(model, mesh=hsdp_mesh)

    optimizer = dmuon.Muon(
        model,
        lr=0.01,
        ns_steps=2,
        adamw_lr=0.001,
        param_groups=[
            {
                "params": matrix_params,
                "group_name": "matrix",
                "dmuon_route": "muon",
                "muon_lr": 0.01,
            },
            {
                "params": sharded_base_params,
                "group_name": "base_sharded",
                "dmuon_route": "sharded_adamw",
                "adamw_lr": 0.001,
                "adamw_weight_decay": 0.0,
            },
            {
                "params": base_params,
                "group_name": "base",
                "dmuon_route": "adamw",
                "adamw_lr": 0.001,
                "adamw_weight_decay": 0.0,
            },
        ],
    )

    sharded_dps = [
        dp
        for dp in optimizer._all_dedicated_params
        if getattr(dp, "_dmuon_route", None) == "sharded_adamw"
    ]
    assert sharded_dps, "expected sharded AdamW-route dedicated params"
    assert all(getattr(dp, "_sharded_adamw_data", None) is not None for dp in sharded_dps)

    torch.manual_seed(42)
    x = torch.randn(4, 128, device=device)
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(x)
        assert torch.isfinite(loss.detach())
        loss.backward()
        optimizer.step()

    assert all(
        id(dp) in optimizer.state and "exp_avg" in optimizer.state[id(dp)]
        for dp in sharded_dps
    )

    torch.cuda.synchronize()
    log(rank, "PASSED: test_muon_sharded_base_adamw_hsdp")


# ---------------------------------------------------------------------------
# Test 5: TP params — step smoke.
#
# The TP path is live. This smoke keeps the older distributed test in sync with
# the current contract: TP-sharded dedicated params must complete a DMuon step
# without falling back to the removed pre-T2 NotImplementedError guard.
# ---------------------------------------------------------------------------

def test_muon_tp_path(rank, world_size, device, mesh):
    assert world_size >= 4, "test_muon_tp_path requires at least 4 GPUs"

    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    # 2D mesh: dp_size=2, tp_size=2 (separate from the 1D mesh passed in)
    mesh_2d = dist.init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
    dp_mesh = mesh_2d["dp"]
    tp_mesh = mesh_2d["tp"]

    torch.manual_seed(0)
    model = TinyModel().to(device)

    # Apply TP parallelism to MLP layers
    for layer in model.layers:
        parallelize_module(
            layer.mlp,
            tp_mesh,
            {
                "gate_proj": ColwiseParallel(),
                "up_proj": ColwiseParallel(),
                "down_proj": RowwiseParallel(),
            },
        )

    dmuon.dedicate_params(
        model, dp_mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )

    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)

    optimizer = dmuon.Muon(model, lr=0.01, ns_steps=5, adamw_lr=0.01)

    # Verify TP params are flagged correctly
    tp_params = [dp for dp in optimizer._dedicated_params if dp.is_dtensor and dp.tp_group is not None]
    tp_count = torch.tensor([len(tp_params)], device=device)
    dist.all_reduce(tp_count)
    assert tp_count.item() > 0, (
        f"No TP-aware dedicated params found (is_dtensor=True, tp_group≠None)"
    )

    losses = []
    for step in range(2):
        optimizer.zero_grad()
        torch.manual_seed(2026 + step)
        x = torch.randn(4, 256, device=device)
        loss = model(x)
        assert torch.isfinite(loss.detach())
        loss.backward()
        optimizer.step()
        dmuon.wait_all_post_step_broadcasts(model)
        losses.append(loss.detach().float())

    loss_tensor = torch.stack(losses)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    assert torch.isfinite(loss_tensor).all().item()
    torch.cuda.synchronize()
    log(rank, "PASSED: test_muon_tp_path")


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh("cuda", (world_size,))

    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    tests = {
        "reduced_grad": test_muon_reduced_grad_all_set,
        "momentum": test_muon_momentum_accumulation,
        "nesterov": test_muon_nesterov,
        "param_groups_fsdp2": test_muon_param_groups_fsdp2,
        "all_trainable_type_split_fsdp2": test_muon_all_trainable_type_split_fsdp2,
        "all_trainable_type_split_hsdp": test_muon_all_trainable_type_split_hsdp,
        "sharded_base_adamw_fsdp2": test_muon_sharded_base_adamw_fsdp2,
        "sharded_base_adamw_hsdp": test_muon_sharded_base_adamw_hsdp,
        "tp_path": test_muon_tp_path,
    }

    if test_name == "all":
        for name, fn in tests.items():
            log(rank, f"\n{'=' * 60}")
            log(rank, f"Running: {name}")
            log(rank, f"{'=' * 60}")
            dist.barrier()
            fn(rank, world_size, device, mesh)
            dist.barrier()
    elif test_name in tests:
        tests[test_name](rank, world_size, device, mesh)
    else:
        if rank == 0:
            print(f"Unknown test: {test_name}. Available: {list(tests.keys())}")
        sys.exit(1)

    dist.destroy_process_group()
