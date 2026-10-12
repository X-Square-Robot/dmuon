"""Strict native-NPU shard-only/HSDP comparison using the upstream TinyModel.

All four ranks use seed 42 for model initialization and CPU inputs seeded
1000 + step. Both meshes receive the same tensors. HSDP alone saves a canonical
rank-0 checkpoint and repeats two updates after a fresh-stack CPU restore.
There are 2 * steps + 2 distributed optimizer calls. No tolerance, gradient
override, per-rank checkpoint override, or CUDA compatibility shim is used.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import timedelta
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

import dmuon


def cpu_clone(value: Any) -> Any:
    """Detach and copy every tensor so later updates cannot mutate evidence."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_clone(item) for item in value)
    return value


def summarize(value: Any) -> Any:
    """Record tensor shape, dtype and exact-byte hash in compact JSON."""
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        assert bool(torch.isfinite(value).all()), "Non-finite captured state"
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): summarize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [summarize(item) for item in value]
    return value if value is None or isinstance(value, (str, int, float, bool)) else repr(value)


def assert_exact(actual: Any, expected: Any, path: str = "root") -> None:
    """Compare every tensor and schema field without a numerical tolerance."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
            equal_nan=False,
            msg=lambda message: f"{path}: {message}",
        )
        actual_bytes = actual.detach().contiguous().reshape(-1).view(torch.uint8)
        expected_bytes = expected.detach().contiguous().reshape(-1).view(torch.uint8)
        assert torch.equal(actual_bytes, expected_bytes), f"{path}: tensor bits differ"
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            assert_exact(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_exact(left, right, f"{path}[{index}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def collective_check(stage: str, check: Callable[[], None], device: torch.device) -> None:
    """Join local assertions before any rank enters the next state collective."""
    error = None
    try:
        check()
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    flag = torch.tensor([int(error is not None)], dtype=torch.int32, device=device)
    flags = [torch.empty_like(flag) for _ in range(4)]
    dist.all_gather(flags, flag)
    failed = [rank for rank, item in enumerate(flags) if int(item.cpu().item())]
    if failed:
        raise AssertionError(f"{stage}: failed ranks {failed}; local error: {error}")


def same_on_all_ranks(value: Any, device: torch.device, stage: str) -> list[str]:
    """Verify a rank-0 canonical state is valid using an NPU byte-tensor wire."""
    encoded = json.dumps(summarize(value), sort_keys=True, allow_nan=False).encode()
    digest = hashlib.sha256(encoded).digest()
    wire = torch.tensor(list(digest), dtype=torch.uint8, device=device)
    received = [torch.empty_like(wire) for _ in range(4)]
    dist.all_gather(received, wire)
    hashes = [bytes(item.cpu().tolist()).hex() for item in received]
    assert len(set(hashes)) == 1, f"{stage}: cross-rank hashes {hashes}"
    return hashes


def build_stack(
    model_cls: type[nn.Module], initial: dict[str, Any], device: torch.device, mesh: Any, hsdp: bool
) -> tuple[nn.Module, dmuon.Muon]:
    """Use the original model, parameter predicate and optimizer configuration."""
    model = model_cls().to(device)
    model.load_state_dict(initial)
    dmuon.dedicate_params(
        model,
        mesh["shard"] if hsdp else mesh,
        predicate=lambda name, parameter: "proj" in name and parameter.ndim == 2,
        replicate_mesh=mesh["replicate"] if hsdp else None,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    optimizer = dmuon.Muon(
        model,
        lr=0.01,
        momentum=0.0,
        weight_decay=0.0,
        ns_steps=5,
        adamw_lr=1e-3,
        adamw_betas=(0.9, 0.999),
        adamw_weight_decay=0.01,
    )
    return model, optimizer


def snapshot(
    model: nn.Module, optimizer: dmuon.Muon, loss: torch.Tensor | None = None
) -> dict[str, Any]:
    """Drain public communication waits and gather independent full CPU states."""
    dmuon.wait_all_reduces(model)
    dmuon.wait_all_post_step_broadcasts(model)
    dmuon.wait_all_replicate_broadcasts(model)
    torch.npu.synchronize()
    return {
        "loss": cpu_clone(loss),
        "model": cpu_clone(dmuon.get_model_state_dict(model, rank0_only=False)),
        "optimizer": cpu_clone(dmuon.get_optimizer_state_dict(model, optimizer, rank0_only=False)),
    }


def update(
    model: nn.Module, optimizer: dmuon.Muon, data: torch.Tensor, report: dict[str, Any]
) -> dict[str, Any]:
    """Perform one real forward, backward and optimizer update."""
    optimizer.zero_grad()
    loss = model(data)
    loss.backward()
    dmuon.wait_all_reduces(model)
    report["global_optimizer_calls"] += 1
    optimizer.step()
    return snapshot(model, optimizer, loss)


def save_checkpoint(
    model: nn.Module,
    optimizer: dmuon.Muon,
    saved: dict[str, Any],
    path: Path,
    save_step: int,
    device: torch.device,
) -> None:
    """All ranks gather; rank 0 alone writes the standard full-state checkpoint."""
    dmuon.wait_all_reduces(model)
    dmuon.wait_all_post_step_broadcasts(model)
    dmuon.wait_all_replicate_broadcasts(model)
    model_state = cpu_clone(dmuon.get_model_state_dict(model, rank0_only=True))
    optimizer_state = cpu_clone(dmuon.get_optimizer_state_dict(model, optimizer, rank0_only=True))

    def write() -> None:
        if dist.get_rank() == 0:
            assert_exact(model_state, saved["model"], "disk.model")
            assert_exact(optimizer_state, saved["optimizer"], "disk.optimizer")
            with path.open("xb") as handle:
                torch.save(
                    {"model": model_state, "optimizer": optimizer_state, "next_step": save_step},
                    handle,
                )

    collective_check("write canonical HSDP checkpoint", write, device)


def main(
    steps: int = 10,
    resume_after: int | None = None,
    output_dir: str | Path | None = None,
    model_cls: type[nn.Module] | None = None,
) -> None:
    """Run strict cross-mesh comparison and two HSDP disk-restart updates.

    ``steps=4, resume_after=2`` costs ten distributed optimizer calls. The
    default retains the upstream ten-step trajectories and costs 22 calls.
    """
    import torch_npu  # Native registration, without transfer_to_npu.

    if model_cls is None:
        from test_hsdp_correctness import TinyModel

        model_cls = TinyModel
    resume_after = min(5, steps - 2) if resume_after is None else resume_after
    if not (steps >= 4 and 1 <= resume_after <= steps - 2 and 2 * steps + 2 <= 100):
        raise ValueError("Need at least four steps, two continuation steps, and <=100 calls")
    assert "torch_npu.contrib.transfer_to_npu" not in sys.modules
    assert torch.version.cuda is None, "Use native NPU PyTorch"
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    rank = dist.get_rank()
    report: dict[str, Any] = {
        "passed": False,
        "rank": rank,
        "steps": steps,
        "resume_after": resume_after,
        "global_optimizer_calls": 0,
        "rtol": 0,
        "atol": 0,
        "records": [],
        "restart": [],
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": str(device),
        "driver": str(Path(__file__).resolve()),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dmuon_file": str(Path(dmuon.__file__).resolve()),
        "hccl_deterministic": os.environ.get("HCCL_DETERMINISTIC", "unset"),
        "claim": "functional cross-mesh and canonical HSDP restart only",
    }
    directory_ready = False
    started = time.monotonic()
    try:
        assert dist.get_world_size() == 4, "Exactly four ranks are required"
        if output_dir is None:
            identifier = torch.tensor([os.getpid() if rank == 0 else 0], device=device)
            dist.broadcast(identifier, src=0)
            output_dir = Path("/tmp") / f"dmuon-native-hsdp-{identifier.cpu().item()}"
        output_dir = Path(output_dir).resolve()
        collective_check(
            "fresh output directory",
            lambda: output_dir.mkdir(parents=True, exist_ok=False) if rank == 0 else None,
            device,
        )
        directory_ready = True
        torch.manual_seed(42)
        reference = model_cls()
        initial = cpu_clone(reference.state_dict())
        assert all(tensor.dtype == torch.float32 for tensor in initial.values())
        hidden = reference.layers[0].mlp.gate_proj.in_features
        del reference
        data = [
            torch.randn(4, hidden, generator=torch.Generator().manual_seed(1000 + index)).to(device)
            for index in range(steps)
        ]
        report["initial_hashes"] = same_on_all_ranks(initial, device, "initial model")
        report["input_hashes"] = [
            same_on_all_ranks(item.cpu(), device, f"input {index}")
            for index, item in enumerate(data)
        ]
        mesh_a = init_device_mesh("npu", (4,))
        mesh_b = init_device_mesh("npu", (2, 2), mesh_dim_names=("replicate", "shard"))
        model_a, optimizer_a = build_stack(model_cls, initial, device, mesh_a, False)
        model_b, optimizer_b = build_stack(model_cls, initial, device, mesh_b, True)
        initial_a, initial_b = snapshot(model_a, optimizer_a), snapshot(model_b, optimizer_b)
        collective_check(
            "initial cross-mesh state",
            lambda actual=initial_b, expected=initial_a: assert_exact(actual, expected),
            device,
        )
        expected = []
        checkpoint_path = output_dir / "hsdp-checkpoint.pt"
        for index, item in enumerate(data):
            state_a = update(model_a, optimizer_a, item, report)
            state_b = update(model_b, optimizer_b, item, report)
            report["records"].append(
                {
                    "step": index,
                    "a": summarize(state_a),
                    "b": summarize(state_b),
                    "loss_a": float(state_a["loss"]),
                    "loss_b": float(state_b["loss"]),
                }
            )
            collective_check(
                f"cross-mesh update {index}",
                lambda actual=state_b, expected=state_a: assert_exact(actual, expected),
                device,
            )
            same_on_all_ranks(state_b, device, f"canonical HSDP state {index}")
            if index + 1 == resume_after:
                saved = {"loss": None, "model": state_b["model"], "optimizer": state_b["optimizer"]}
                save_checkpoint(model_b, optimizer_b, saved, checkpoint_path, resume_after, device)
            if resume_after <= index < resume_after + 2:
                expected.append(state_b)
        del model_a, optimizer_a, model_b, optimizer_b, initial_a, initial_b, state_a, state_b
        gc.collect()
        torch.npu.empty_cache()
        restored, restored_optimizer = build_stack(model_cls, initial, device, mesh_b, True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["next_step"] == resume_after
        dmuon.set_model_state_dict(restored, checkpoint["model"])
        dmuon.set_optimizer_state_dict(restored, restored_optimizer, checkpoint["optimizer"])
        restored_state = snapshot(restored, restored_optimizer)
        collective_check(
            "canonical disk restore", lambda: assert_exact(restored_state, saved), device
        )
        for offset, baseline in enumerate(expected):
            index = resume_after + offset
            actual = update(restored, restored_optimizer, data[index], report)
            report["restart"].append(
                {
                    "step": index,
                    "actual": summarize(actual),
                    "expected": summarize(baseline),
                    "loss": float(actual["loss"]),
                }
            )
            collective_check(
                f"disk restart update {index}", lambda: assert_exact(actual, baseline), device
            )
        assert report["global_optimizer_calls"] == 2 * steps + 2
        report.update(
            passed=True,
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        )
        if rank == 0:
            print(
                f"PASS: native HSDP exact cross-mesh/restart; {2 * steps + 2} optimizer calls",
                flush=True,
            )
    except Exception as exception:
        report["error"] = f"{type(exception).__name__}: {exception}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        if directory_ready:
            with (output_dir / f"rank{rank}.json").open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
                handle.write("\n")
        dist.destroy_process_group()
