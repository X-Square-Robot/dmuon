"""Finite four-rank native-NPU restart regression; run only after admission.

Example (select four idle cards in the launcher environment first)::

    torchrun --standalone --nproc_per_node=4 tests/native/ascend_distributed.py \
        --package-root /absolute/path/to/dmuon --topology all \
        --output-dir /absolute/path/to/new-proof-directory

Each topology has two GAS=4 paths: sync with the ordinary BF16/default
gradient policy, and no_sync with an explicit FP32 gradient policy. Each path
performs three updates, writes a real tiny checkpoint, runs two uninterrupted
updates, restores into a fresh stack, and repeats those two updates. The
restart comparison is exact within each path. The two paths are not numerical
oracles for one another. The total is 28 global optimizer calls, not 28 per
rank. This is a bounded functional regression, not training-quality evidence.

Only main() imports torch/torch_npu. Syntax compilation is CPU-only and does
not initialize a process group, touch an NPU, install a package, or launch work.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
import traceback
from types import SimpleNamespace


WORLD_SIZE = 4
GAS = 4
UPDATES_BEFORE_SAVE = 3
CONTINUATION_INDICES = (3, 4)  # Inactive Muon/optional bias, then active.
CALLS_PER_PATH = 7
MAX_GLOBAL_OPTIMIZER_CALLS = 28
MAX_CPU_PAYLOAD_BYTES = 2 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 1024 * 1024


def source_hash(root):
    """Hash the package source bytes actually made available to this process."""
    files = {}
    aggregate = hashlib.sha256()
    for path in sorted((root / "dmuon").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[relative] = digest
        aggregate.update(relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    if not files:
        raise ValueError(f"No dmuon Python source under {root}")
    return {"sha256": aggregate.hexdigest(), "files": files}


def tensor_digest(tensor, torch):
    # CPU uint8 bytes work for BF16 without NumPy's BF16 conversion support.
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(bytes(raw.tolist())).hexdigest()


def json_tree(value, torch):
    """Keep JSON evidence small while retaining dtype/shape/exact-byte hashes."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        entry = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "sha256": tensor_digest(tensor, torch),
        }
        if tensor.numel():
            numbers = tensor.to(torch.float64)
            entry.update(
                min=float(numbers.min()),
                max=float(numbers.max()),
                l2=float(numbers.norm()),
                finite=bool(torch.isfinite(numbers).all()),
            )
            if tensor.numel() <= 4:
                entry["values"] = tensor.reshape(-1).tolist()
        return entry
    if isinstance(value, dict):
        return {str(key): json_tree(item, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_tree(item, torch) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _serialized(value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > MAX_CPU_PAYLOAD_BYTES:
        raise ValueError(f"Tiny-test CPU payload exceeded {MAX_CPU_PAYLOAD_BYTES} bytes")
    return payload


def all_gather_cpu(value, runtime):
    """HCCL sees NPU byte tensors, never CPU tensors or backend object buffers."""
    torch, dist, device = runtime.torch, runtime.dist, runtime.device
    payload = _serialized(value)
    length = torch.tensor([len(payload)], dtype=torch.int64, device=device)
    lengths = [torch.empty_like(length) for _ in range(WORLD_SIZE)]
    dist.all_gather(lengths, length)
    sizes = [int(item.cpu().item()) for item in lengths]
    maximum = max(sizes)
    assert maximum <= MAX_CPU_PAYLOAD_BYTES
    buffer = torch.zeros(maximum, dtype=torch.uint8, device=device)
    buffer[: len(payload)].copy_(torch.tensor(list(payload), dtype=torch.uint8, device=device))
    buffers = [torch.empty_like(buffer) for _ in range(WORLD_SIZE)]
    dist.all_gather(buffers, buffer)
    return [pickle.loads(bytes(item[:size].cpu().tolist())) for item, size in zip(buffers, sizes)]


def broadcast_cpu(value, runtime):
    """Broadcast a rank-0 CPU checkpoint using an explicitly NPU HCCL wire."""
    torch, dist, device = runtime.torch, runtime.dist, runtime.device
    payload = _serialized(value) if runtime.rank == 0 else b""
    length = torch.tensor([len(payload)], dtype=torch.int64, device=device)
    dist.broadcast(length, src=0)
    size = int(length.cpu().item())
    assert 0 < size <= MAX_CPU_PAYLOAD_BYTES
    if runtime.rank == 0:
        buffer = torch.tensor(list(payload), dtype=torch.uint8, device=device)
    else:
        buffer = torch.empty(size, dtype=torch.uint8, device=device)
    dist.broadcast(buffer, src=0)
    return pickle.loads(bytes(buffer.cpu().tolist()))


def collective_check(stage, check, runtime):
    """Join local assertion outcomes before the next collective-heavy stage."""
    try:
        check()
        error = None
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    outcomes = all_gather_cpu({"rank": runtime.rank, "error": error}, runtime)
    failed = [item for item in outcomes if item["error"] is not None]
    if failed:
        raise AssertionError(f"{stage}: {failed}")


def capture_rng(runtime):
    torch = runtime.torch
    return {
        "python": random.getstate(),
        "cpu": torch.get_rng_state().clone(),
        "npu": torch.npu.get_rng_state(runtime.device).cpu().clone(),
    }


def restore_rng(state, runtime):
    random.setstate(state["python"])
    runtime.torch.set_rng_state(state["cpu"])
    runtime.torch.npu.set_rng_state(state["npu"], runtime.device)


def dedicated_by_name(model, runtime):
    names = {id(module): name for name, module in model.named_modules()}
    return {
        f"{names[id(dp.module)]}.{dp.param_name}": dp
        for dp in runtime.dmuon.get_dedicated_params(model)
    }


def build_stack(mesh, topology, grad_dtype, runtime):
    """Framework grouping/policy stays outside optimizer math and runtime."""
    torch, dmuon, cases = runtime.torch, runtime.dmuon, runtime.cases

    class TinyFourRankModel(cases.TinyModel):
        def __init__(self):
            super().__init__(torch.bfloat16)
            self.action_input = cases.Projection(torch.float32)

        def forward(self, inputs, *, options):
            loss = super().forward(inputs, options=options)
            fp32 = inputs[0].float()
            fp32_options = {**options, "nested": [fp32, options["nested"][1]]}
            action = self.action_input((fp32, inputs[1]), options=fp32_options)
            assert action.dtype == torch.float32
            return loss + action.square().mean() * (1.0 / 1024.0)

    torch.manual_seed(10233)
    model = TinyFourRankModel().to(runtime.device)
    original_shapes = {name: list(param.shape) for name, param in model.named_parameters()}
    assert all(param.dtype == torch.float32 for param in model.parameters())
    assert all(param.numel() <= 128 * 128 for param in model.parameters())
    assert max(max(shape, default=0) for shape in original_shapes.values()) <= 128
    routes = {
        "proj.weight": "muon",
        "owner.weight": "adamw",
        "sharded.weight": "sharded_adamw",
        "action_input.weight": "adamw",
    }

    def policy(name, _parameter):
        is_muon = name == "proj.weight"
        return {
            "route": routes[name],
            "param_dtype": torch.bfloat16 if is_muon else torch.float32,
            "grad_dtype": grad_dtype if is_muon else torch.float32,
            "master_dtype": torch.float32,
            "optim_dtype": torch.float32,
            "cast_forward_inputs": True,
        }

    shard_mesh = mesh if topology == "dp4" else mesh["shard"]
    replicate_mesh = None if topology == "dp4" else mesh["replicate"]
    dmuon.dedicate_params(
        model,
        shard_mesh,
        predicate=lambda name, param: name in routes,
        replicate_mesh=replicate_mesh,
        param_policy_fn=policy,
    )
    if topology == "dp4":
        # DMuon's dedicated sharded-AdamW path needs no fully_shard NN module.
        dmuon.replicate(model, mesh=mesh)
    else:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        precision = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=grad_dtype)
        fully_shard(model.norm, mesh=mesh, mp_policy=precision)
        fully_shard(model.head, mesh=mesh, mp_policy=precision)
        fully_shard(model, mesh=mesh)

    # Build groups from the wrapped model, exactly as a framework integration
    # does. Route hints construct storage; param_groups select update policy.
    wrapped = dict(model.named_parameters())
    groups = []
    for name, names, route in (
        ("matrix", ["proj.weight"], "muon"),
        ("owner_base", ["owner.weight"], "adamw"),
        ("sharded_base", ["sharded.weight"], "sharded_adamw"),
        ("action_fp32", ["action_input.weight"], "adamw"),
        ("ordinary_base", [name for name in wrapped if name not in routes], "adamw"),
    ):
        groups.append(
            {
                "params": [wrapped[name] for name in names],
                "group_name": name,
                "dmuon_route": route,
                "muon_lr": 0.02,
                "adamw_lr": 0.001,
                "weight_decay": 0.01,
            }
        )
    optimizer = dmuon.Muon(
        model,
        param_groups=groups,
        ns_steps=5,
        ns_backend=dmuon.NewtonSchulz(backend="direct", kernel="cublas"),
        replicate_async=True,
        first_step_progress_log=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda index: 0.97**index)
    actual = dedicated_by_name(model, runtime)
    assert {name: dp._dmuon_route for name, dp in actual.items()} == routes
    return SimpleNamespace(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cursor={"next_update": 0, "microbatches_seen": 0},
        original_shapes=original_shapes,
        grad_dtype=grad_dtype,
    )


def local_state(stack, runtime, *, expect_momentum):
    """Inspect live buffers BEFORE export can cast/gather and mask dtype bugs."""
    torch, dmuon, cases = runtime.torch, runtime.dmuon, runtime.cases
    names = dedicated_by_name(stack.model, runtime)
    momentum, adamw, policy = {}, {}, {}
    for name, dp in names.items():
        assert dp._orig_dtype == torch.float32, name
        assert dp._master_dtype == torch.float32 and dp._optim_dtype == torch.float32, name
        expected_compute = torch.bfloat16 if name == "proj.weight" else torch.float32
        assert dp._param_dtype == expected_compute, name
        if name == "proj.weight":
            assert dp._grad_dtype == stack.grad_dtype, name
        policy[name] = {
            "route": dp._dmuon_route,
            "owner_coord": tuple(dp.owner_rank),
            "is_owner": bool(dp.is_owner),
            "param_dtype": str(dp._param_dtype),
            "grad_dtype": str(dp._grad_dtype),
            "master_dtype": str(dp._master_dtype),
            "optim_dtype": str(dp._optim_dtype),
            "orig_dtype": str(dp._orig_dtype),
            "local_shard_rank": dp.dp_group.rank(),
            "local_replicate_rank": (
                dp.replicate_group.rank() if dp.replicate_group is not None else 0
            ),
        }
        for attr in ("_owned_data", "_sharded_adamw_data"):
            tensor = getattr(dp, attr, None)
            if tensor is not None:
                assert tensor.dtype == torch.float32, (name, attr, tensor.dtype)
                assert bool(torch.isfinite(tensor).all()), (name, attr)
        state = stack.optimizer.state.get(id(dp), {})
        if dp._dmuon_route == "muon":
            if dp.is_owner and expect_momentum:
                assert "momentum_buffer" in state, name
            for key in ("momentum_buffer", "tp_local_momentum_buffer"):
                if key in state:
                    assert state[key].dtype == dp._optim_dtype == torch.float32, (name, key)
                    assert bool(torch.isfinite(state[key]).all()), (name, key)
                    momentum[f"{name}:{key}"] = cases.cpu_tree(state[key])
        elif state:
            for key in ("exp_avg", "exp_avg_sq"):
                assert state[key].dtype == torch.float32, (name, key)
                assert bool(torch.isfinite(state[key]).all()), (name, key)
            assert state["step"] == stack.cursor["next_update"], (name, state["step"])
            adamw[name] = cases.cpu_tree(state)

    # Public rank lookup uses the actual shard/replicate coordinate, not owner 0.
    sample = next(iter(names.values()))
    coordinate = (
        sample.dp_group.rank(),
        sample.replicate_group.rank() if sample.replicate_group is not None else 0,
    )
    public_owned = set(map(id, dmuon.get_owned_params(stack.model, coordinate)))
    assert public_owned == {id(dp) for dp in names.values() if dp.is_owner}
    ordinary = {}
    for name, parameter in stack.model.named_parameters():
        if name in names or parameter not in stack.optimizer.state:
            continue
        state = stack.optimizer.state[parameter]
        local_parameter = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        for key in ("exp_avg", "exp_avg_sq"):
            assert state[key].dtype == local_parameter.dtype == torch.float32, (name, key)
            assert bool(torch.isfinite(state[key]).all()), (name, key)
        expected_step = (
            (stack.cursor["next_update"] + 1) // 2
            if name == "optional_bias"
            else stack.cursor["next_update"]
        )
        assert state["step"] == expected_step, (name, state["step"], expected_step)
        ordinary[name] = cases.cpu_tree(state)
    assert "unused" not in ordinary
    if expect_momentum:
        assert set(ordinary) == {
            "norm.weight",
            "norm.bias",
            "head.weight",
            "head.bias",
            "optional_bias",
        }, set(ordinary)
    assert stack.model.unused.grad is None
    return {
        "master": cases.master_snapshot(dmuon, stack.model),
        "live_momentum": momentum,
        "live_dedicated_adamw": adamw,
        "live_ordinary_adamw": ordinary,
        "policy": policy,
    }


def snapshot(stack, runtime, *, expect_momentum):
    # Collectives remain symmetric if a local dtype check fails.
    inspected = {}

    def inspect():
        inspected.update(local_state(stack, runtime, expect_momentum=expect_momentum))

    collective_check("live state before export", inspect, runtime)
    model = runtime.dmuon.get_model_state_dict(stack.model, cpu_offload=True, rank0_only=False)
    optimizer = runtime.dmuon.get_optimizer_state_dict(
        stack.model, stack.optimizer, cpu_offload=True, rank0_only=False
    )

    def inspect_export():
        assert model.keys() == stack.original_shapes.keys()
        for name, tensor in model.items():
            assert list(tensor.shape) == stack.original_shapes[name], name
            assert tensor.dtype == runtime.torch.float32, (name, tensor.dtype)
            assert bool(runtime.torch.isfinite(tensor).all()), name
        assert "unused" not in optimizer["fsdp"]
        for state in optimizer["fsdp"].values():
            for value in state.values():
                if isinstance(value, runtime.torch.Tensor):
                    assert value.device.type == "cpu", value.device
        if expect_momentum:
            assert set(optimizer["dedicated"]) == {"proj.weight"}
            assert set(optimizer["dedicated_adamw"]) == {
                "owner.weight",
                "sharded.weight",
                "action_input.weight",
            }
            assert (
                optimizer["dedicated"]["proj.weight"]["momentum_buffer"].dtype
                == runtime.torch.float32
            )
        assert stack.scheduler.last_epoch == stack.cursor["next_update"]
        assert stack.cursor["microbatches_seen"] == stack.cursor["next_update"] * GAS

    collective_check("exported model and counters", inspect_export, runtime)
    return {
        "model": runtime.cases.cpu_tree(model),
        "optimizer": runtime.cases.cpu_tree(optimizer),
        **inspected,
        "scheduler": runtime.cases.cpu_tree(stack.scheduler.state_dict()),
        "rng": capture_rng(runtime),
        "cursor": dict(stack.cursor),
    }


def update(stack, runtime, *, index, no_sync):
    torch, dmuon, cases = runtime.torch, runtime.dmuon, runtime.cases
    assert stack.cursor["next_update"] == index
    runtime.optimizer_calls += 1
    assert runtime.optimizer_calls <= MAX_GLOBAL_OPTIMIZER_CALLS < 100
    stack.optimizer.zero_grad(set_to_none=True)
    losses, inputs = [], []
    active = index % 2 == 0
    for micro in range(GAS):
        # Both restored RNG streams affect the real input and loss. Rank seeds
        # give distinct inputs; there is no rank-identical synthetic shortcut.
        cpu_input = torch.randn(2, 8, dtype=torch.float32)
        x = (
            cpu_input.to(runtime.device)
            + torch.rand(2, 8, dtype=torch.float32, device=runtime.device) / 128.0
            + random.random() / 256.0
            + runtime.rank / 8.0
        )
        indices = torch.tensor([0, 1], dtype=torch.int64, device=runtime.device)
        options = {
            "nested": [x.clone(), (indices, "unchanged")],
            "use_extra": active,
            "use_muon": active,
        }
        context = dmuon.no_sync(stack.model) if no_sync and micro < GAS - 1 else nullcontext()
        with context:
            loss = stack.model((x, indices), options=options) / GAS
            loss.backward()
        losses.append(cases.cpu_tree(loss))
        inputs.append(cases.cpu_tree(x))
    dmuon.wait_all_reduces(stack.model)
    dps = dedicated_by_name(stack.model, runtime)
    stats = dmuon.clip_grad_norm_(stack.optimizer, None)
    expected_local = sum(
        int(dp.is_owner and active) for dp in dps.values() if dp._dmuon_route == "muon"
    )
    local_count = torch.tensor([stats.param_count], dtype=torch.int64, device=runtime.device)
    runtime.dist.all_reduce(local_count)
    global_count = int(local_count.cpu().item())
    gradients = {}

    def inspect_gradients():
        assert stats.param_count == expected_local, (stats.param_count, expected_local)
        assert global_count == int(active), (global_count, active)
        assert not bool(stats.found_inf)
        for name, dp in dps.items():
            grad = (
                getattr(dp, "_sharded_adamw_grad", None)
                if dp._dmuon_route == "sharded_adamw"
                else dp._reduced_grad
            )
            if grad is not None:
                assert bool(torch.isfinite(grad).all()), name
                expected_dtype = (
                    (stack.grad_dtype or torch.bfloat16) if name == "proj.weight" else torch.float32
                )
                assert grad.dtype == expected_dtype, (name, grad.dtype, expected_dtype)
                gradients[name] = cases.cpu_tree(grad)
            if name == "proj.weight" and dp.is_owner:
                assert (grad is not None) == active, name
        if not active:
            assert stack.model.optional_bias.grad is None
        assert stack.model.unused.grad is None
        assert all(bool(torch.isfinite(loss)) for loss in losses)
        assert stack.model.action_input.seen_dtypes == (
            "torch.float32",
            "torch.float32",
            "torch.int64",
        )
        if active:
            assert stack.model.proj.seen_dtypes == (
                "torch.bfloat16",
                "torch.bfloat16",
                "torch.int64",
            )

    collective_check("owner-aware gradients and exceptions", inspect_gradients, runtime)
    stack.optimizer.step()
    stack.scheduler.step()
    stack.cursor["next_update"] += 1
    stack.cursor["microbatches_seen"] += GAS
    dmuon.wait_all_post_step_broadcasts(stack.model)
    result = snapshot(stack, runtime, expect_momentum=True)
    result.update(
        losses=losses,
        inputs=inputs,
        gradients=gradients,
        clip=cases.cpu_tree(stats.as_dict()),
        global_muon_grad_count=global_count,
        input_dtypes=stack.model.proj.seen_dtypes if active else None,
        action_dtypes=stack.model.action_input.seen_dtypes,
    )
    return result


def save_checkpoint(stack, saved_snapshot, path, runtime):
    # The checkpoint getters run on ALL ranks even though rank 0 alone keeps
    # their full tensors and writes the disk artifact.
    model = runtime.dmuon.get_model_state_dict(stack.model, cpu_offload=True, rank0_only=True)
    optimizer = runtime.dmuon.get_optimizer_state_dict(
        stack.model, stack.optimizer, cpu_offload=True, rank0_only=True
    )
    # A rank-0-only snapshot cannot preserve replica-local collective rounding.
    # Retain each rank's public export for an exact same-rank continuation test.
    per_rank = all_gather_cpu(
        {
            "rng": capture_rng(runtime),
            "cursor": dict(stack.cursor),
            "model": saved_snapshot["model"],
            "optimizer": saved_snapshot["optimizer"],
        },
        runtime,
    )
    status = None
    if runtime.rank == 0:
        try:
            checkpoint = {
                "format": "dmuon-tiny4rank-v1",
                "model": model,
                "optimizer": optimizer,
                "scheduler": runtime.cases.cpu_tree(stack.scheduler.state_dict()),
                "per_rank": per_rank,
                "source_sha256": runtime.source["sha256"],
            }
            runtime.cases.assert_tree_equal(model, saved_snapshot["model"], path="disk.model")
            runtime.cases.assert_tree_equal(
                optimizer, saved_snapshot["optimizer"], path="disk.optimizer"
            )
            with path.open("xb") as handle:
                runtime.torch.save(checkpoint, handle)
            size = path.stat().st_size
            assert size <= MAX_CHECKPOINT_BYTES, size
            status = {
                "ok": True,
                "path": str(path),
                "bytes": size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        except Exception as exception:
            status = {"ok": False, "error": f"{type(exception).__name__}: {exception}"}
    status = broadcast_cpu(status, runtime)
    if not status["ok"]:
        raise RuntimeError(f"Checkpoint write failed: {status}")
    return status


def load_checkpoint(path, runtime):
    result = None
    if runtime.rank == 0:
        try:
            # map_location=CPU prevents deserialization from initializing a
            # wrong-rank device; native setters move each tensor explicitly.
            checkpoint = runtime.torch.load(path, map_location="cpu", weights_only=False)
            result = {"ok": True, "checkpoint": checkpoint}
        except Exception as exception:
            result = {"ok": False, "error": f"{type(exception).__name__}: {exception}"}
    result = broadcast_cpu(result, runtime)
    if not result["ok"]:
        raise RuntimeError(f"Checkpoint read failed: {result}")
    return result["checkpoint"]


def run_path(mesh, topology, no_sync, output_dir, runtime):
    grad_dtype = runtime.torch.float32 if no_sync else None
    path_name = "gas4_no_sync_grad_fp32" if no_sync else "gas4_sync_grad_default"
    started_calls = runtime.optimizer_calls
    stack = build_stack(mesh, topology, grad_dtype, runtime)
    seed = 32000 + runtime.rank
    random.seed(seed)
    runtime.torch.manual_seed(seed)
    runtime.torch.npu.manual_seed(seed)
    initial = snapshot(stack, runtime, expect_momentum=False)
    pre_save = [
        update(stack, runtime, index=index, no_sync=no_sync) for index in range(UPDATES_BEFORE_SAVE)
    ]
    checkpoint_path = output_dir / f"checkpoint-{topology}-{path_name}.pt"
    checkpoint_proof = save_checkpoint(stack, pre_save[-1], checkpoint_path, runtime)
    uninterrupted = [
        update(stack, runtime, index=index, no_sync=no_sync) for index in CONTINUATION_INDICES
    ]
    original_shapes = stack.original_shapes
    del stack
    gc.collect()
    restored = build_stack(mesh, topology, grad_dtype, runtime)
    checkpoint = load_checkpoint(checkpoint_path, runtime)
    assert checkpoint["format"] == "dmuon-tiny4rank-v1"
    assert checkpoint["source_sha256"] == runtime.source["sha256"]
    rank_state = checkpoint["per_rank"][runtime.rank]
    runtime.dmuon.set_model_state_dict(restored.model, rank_state["model"])
    runtime.dmuon.set_optimizer_state_dict(
        restored.model, restored.optimizer, rank_state["optimizer"]
    )
    restored.scheduler.load_state_dict(checkpoint["scheduler"])
    restored.cursor = dict(checkpoint["per_rank"][runtime.rank]["cursor"])
    restore_rng(checkpoint["per_rank"][runtime.rank]["rng"], runtime)
    restored_initial = snapshot(restored, runtime, expect_momentum=True)
    saved_state = {key: pre_save[-1][key] for key in restored_initial}
    collective_check(
        "fresh stack restored state",
        lambda: runtime.cases.assert_tree_equal(
            restored_initial, saved_state, path=f"{topology}.{path_name}.restore_initial"
        ),
        runtime,
    )
    resumed = [
        update(restored, runtime, index=index, no_sync=no_sync) for index in CONTINUATION_INDICES
    ]

    def check_restart(
        uninterrupted=uninterrupted, resumed=resumed, pre_save=pre_save, cursor=restored.cursor
    ):
        for index, expected, actual in zip(CONTINUATION_INDICES, uninterrupted, resumed):
            runtime.cases.assert_tree_equal(
                actual, expected, path=f"{topology}.{path_name}.update_{index}"
            )
        # An inactive Muon step must not mutate its master value or momentum;
        # inactive optional_bias must retain its AdamW counter.
        for previous, inactive in ((pre_save[0], pre_save[1]), (pre_save[2], uninterrupted[0])):
            for section in ("model", "master"):
                runtime.cases.assert_tree_equal(
                    inactive[section]["proj.weight"],
                    previous[section]["proj.weight"],
                    path=f"inactive.{section}",
                )
            runtime.cases.assert_tree_equal(
                inactive["live_momentum"], previous["live_momentum"], path="inactive.live_momentum"
            )
            assert (
                inactive["optimizer"]["fsdp"]["optional_bias"]["step"]
                == previous["optimizer"]["fsdp"]["optional_bias"]["step"]
            )
        assert runtime.optimizer_calls - started_calls == CALLS_PER_PATH
        assert cursor == {"next_update": 5, "microbatches_seen": 20}

    collective_check("exact uninterrupted versus disk resume", check_restart, runtime)
    record = {
        "path": path_name,
        "grad_dtype": str(grad_dtype),
        "no_sync": no_sync,
        "gas": GAS,
        "global_optimizer_calls": CALLS_PER_PATH,
        "rank": runtime.rank,
        "checkpoint": checkpoint_proof,
        "original_shapes": original_shapes,
        "initial": json_tree(initial, runtime.torch),
        "pre_save": json_tree(pre_save, runtime.torch),
        "restored_initial": json_tree(restored_initial, runtime.torch),
        "uninterrupted": json_tree(uninterrupted, runtime.torch),
        "restored": json_tree(resumed, runtime.torch),
        "exact_restart": True,
        "rtol": 0,
        "atol": 0,
        "checkpoint_layout": "rank-local full snapshots and canonical rank-0 export",
        "cross_path_comparison": "not asserted",
    }
    del restored, checkpoint, initial, pre_save, uninterrupted, resumed
    gc.collect()
    return record


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def rank0_write(stage, write, runtime):
    status = None
    if runtime.rank == 0:
        try:
            write()
            status = {"ok": True}
        except Exception as exception:
            status = {"ok": False, "error": f"{type(exception).__name__}: {exception}"}
    status = broadcast_cpu(status, runtime)
    if not status["ok"]:
        raise RuntimeError(f"{stage}: {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh shared directory; only rank 0 writes any artifact",
    )
    parser.add_argument("--topology", choices=("dp4", "hsdp2x2", "all"), default="all")
    args = parser.parse_args()
    package_root = args.package_root.resolve(strict=True)
    if not (package_root / "dmuon" / "__init__.py").is_file():
        parser.error("--package-root must contain dmuon/__init__.py")
    sys.path.insert(0, str(package_root))
    # These are this launch's local environment only. No pip, .pth, package,
    # hardware capability, or global torch API edits are performed.
    os.environ["DMUON_NS_CUDA_GRAPH"] = "0"

    import torch
    import torch_npu  # Native registration only; no transfer_to_npu shim.
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    import dmuon

    assert "torch_npu.contrib.transfer_to_npu" not in sys.modules
    assert torch.version.cuda is None and not torch.cuda.is_available()
    package_file = Path(dmuon.__file__).resolve()
    assert package_file == (package_root / "dmuon" / "__init__.py").resolve()
    case_file = package_root / "tests" / "native" / "ascend_cases.py"
    spec = importlib.util.spec_from_file_location("_native_four_rank_cases", case_file)
    assert spec is not None and spec.loader is not None
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    assert Path(cases.__file__).resolve() == case_file.resolve()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == WORLD_SIZE
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    source = source_hash(package_root)
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    runtime = SimpleNamespace(
        torch=torch,
        dist=dist,
        dmuon=dmuon,
        cases=cases,
        device=device,
        rank=rank,
        source=source,
        optimizer_calls=0,
    )
    output_dir = args.output_dir.resolve()
    started = time.monotonic()
    reports = []
    owns_output_directory = False
    metadata = {
        "scope": "finite native NPU functional/restart regression",
        "package_root": str(package_root),
        "dmuon_file": str(package_file),
        "script_file": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "case_file": str(case_file.resolve()),
        "case_sha256": hashlib.sha256(case_file.read_bytes()).hexdigest(),
        "source": source,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "backend": dist.get_backend(),
        "world_size": dist.get_world_size(),
        "rank": rank,
        "local_rank": local_rank,
        "device": str(device),
        "ns_backend": "direct",
        "ns_cuda_graph": os.environ["DMUON_NS_CUDA_GRAPH"],
        "transfer_shim_loaded": False,
        "max_global_optimizer_calls": MAX_GLOBAL_OPTIMIZER_CALLS,
    }
    try:
        assert dist.get_rank() == rank and dist.get_world_size() == WORLD_SIZE
        assert dist.get_backend() == "hccl"
        package_proofs = all_gather_cpu(
            {
                "source": source["sha256"],
                "script": metadata["script_sha256"],
                "cases": metadata["case_sha256"],
            },
            runtime,
        )
        assert all(item == package_proofs[0] for item in package_proofs)
        directory_status = None
        if rank == 0:
            try:
                output_dir.mkdir(parents=True, exist_ok=False)
                directory_status = {"ok": True}
            except Exception as exception:
                directory_status = {"ok": False, "error": str(exception)}
        directory_status = broadcast_cpu(directory_status, runtime)
        if not directory_status["ok"]:
            raise RuntimeError(f"Fresh output directory required: {directory_status}")
        owns_output_directory = True
        topologies = ("dp4", "hsdp2x2") if args.topology == "all" else (args.topology,)
        for topology in topologies:
            topology_start = time.monotonic()
            if topology == "dp4":
                mesh = init_device_mesh("npu", (4,), mesh_dim_names=("dp",))
            else:
                mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("replicate", "shard"))
            paths = [
                run_path(mesh, topology, no_sync, output_dir, runtime) for no_sync in (False, True)
            ]
            rank_report = {
                "metadata": metadata,
                "topology": topology,
                "mesh_shape": list(mesh.shape),
                "mesh_coordinate": mesh.get_coordinate(),
                "paths": paths,
                "passed": True,
                "elapsed_seconds": time.monotonic() - topology_start,
            }
            all_reports = all_gather_cpu(rank_report, runtime)
            # Rank-varied data is independently visible in the per-rank output.
            first_input_hashes = [
                item["paths"][0]["pre_save"][0]["inputs"][0]["sha256"] for item in all_reports
            ]
            assert len(set(first_input_hashes)) == WORLD_SIZE
            summary = {
                "topology": topology,
                "passed": True,
                "world_size": WORLD_SIZE,
                "global_optimizer_calls": 2 * CALLS_PER_PATH,
                "gas": GAS,
                "exact_restart": True,
                "rtol": 0,
                "atol": 0,
                "rank_varied_first_input_sha256": first_input_hashes,
                "path_names": [item["path"] for item in paths],
                "claim_boundary": "functional same-topology restart only; no long training or quality claim",
            }

            def write_topology(reports=all_reports):
                for item in reports:
                    write_json(output_dir / f"{topology}-rank{item['metadata']['rank']}.json", item)
                write_json(output_dir / f"{topology}-summary.json", summary)
                print(
                    f"PASS {topology}: GAS4 sync/default and no_sync/FP32; exact disk restart",
                    flush=True,
                )

            rank0_write(f"write {topology} evidence", write_topology, runtime)
            reports.append(summary)
            del paths, all_reports, mesh
            gc.collect()
        expected_calls = 2 * CALLS_PER_PATH * len(topologies)
        assert runtime.optimizer_calls == expected_calls <= MAX_GLOBAL_OPTIMIZER_CALLS < 100

        def write_overall():
            write_json(
                output_dir / "summary.json",
                {
                    "passed": True,
                    "topologies": reports,
                    "global_optimizer_calls": runtime.optimizer_calls,
                    "elapsed_seconds": time.monotonic() - started,
                    "source_sha256": source["sha256"],
                },
            )
            print(
                f"PASS: {runtime.optimizer_calls} global optimizer calls; {output_dir}", flush=True
            )

        rank0_write("write overall evidence", write_overall, runtime)
    except Exception as exception:
        # Collective assertion failures reach every rank. A backend failure can
        # interrupt a rank at any point, so do not add new failure collectives.
        if rank == 0 and owns_output_directory:
            failure = {
                "metadata": metadata,
                "passed": False,
                "global_optimizer_calls": runtime.optimizer_calls,
                "error": f"{type(exception).__name__}: {exception}",
                "traceback": traceback.format_exc(),
                "completed_topologies": reports,
            }
            failure_path = output_dir / "failure-rank0.json"
            if not failure_path.exists():
                write_json(failure_path, failure)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
