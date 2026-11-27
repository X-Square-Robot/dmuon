"""Compare synchronized complete owner updates on one Ascend NPU.

The layer cases preserve the shapes, routes and counts of one rank's captured
owner parameters in one transformer layer. The per-shape cases pair a single
Muon matrix with an AdamW vector. Parameters and changing gradients are seeded; no
training checkpoints are required. A synthetic backward supplies BF16 gradients
through DMuon's normal ownership hooks before timing starts. The timed region
includes optional segmented clipping, Muon.step() and device completion.
"""

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dmuon

# Muon and AdamW shapes in their owner-local stored orientation.
CASES = {
    "libero-4b-linear-attn-b": ([(32, 2560)], [(2560,)]),
    "libero-4b-action-down": ([(1280, 2560)], [(2560,)]),
    "libero-4b-mlp-down": ([(2560, 9216)], [(2560,)]),
    "shunfeng-2b-vision-fc1": ([(4096, 1024)], [(4096,)]),
    "shunfeng-2b-linear-attn-out": ([(2048, 2048)], [(2048,)]),
    "shunfeng-2b-mlp-down": ([(2048, 6144)], [(2048,)]),
    # All parameters owned on rank 0 for language_model.layers.1 (4 ranks).
    "libero-4b-owner-layer": (
        [(4096, 2560), (32, 2560), (32, 2560), (1280, 2560)],
        [(32,), (32,), (8192, 1, 4), (128,), (2560,), (2560,)] + [(3840, 1280), (3840,)] * 2,
    ),
    # All parameters owned on rank 0 for language_model.layers.7 (4 ranks).
    "shunfeng-2b-owner-layer": (
        [(512, 2048)] * 2
        + [(2048, 2048), (4096, 1024)]
        + [(512, 1024)] * 2
        + [(1024, 2048), (4096, 1024), (1024, 2048)],
        [(256,)] * 4 + [(2048,)] * 2 + [(3072, 1024), (3072,)] * 2,
    ),
}

EMBEDDING_SHARD = {"libero-4b-owner-layer": 158770560, "shunfeng-2b-owner-layer": 131213824}


class UpdateCase(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.names = tuple(initial)
        for name, tensor in initial.items():
            self.register_parameter(name, nn.Parameter(tensor.clone()))

    def forward(self, anchor, *grads):
        loss = anchor
        for name, grad in zip(self.names, grads):
            loss = loss + (getattr(self, name) * grad).sum()
        return loss


def make_arm(backend, mesh, initial, case, decay):
    model = UpdateCase(initial)

    def policy(name, _param):
        return {
            "route": name.split("_")[0],
            "param_dtype": torch.bfloat16,
            "grad_dtype": torch.bfloat16,
            "master_dtype": torch.float32,
            "optim_dtype": torch.float32,
        }

    dmuon.dedicate_params(
        model,
        mesh,
        predicate=lambda _n, _p: True,
        param_policy_fn=policy,
        reshard_after_forward=False,
        process_group_policy="shared",
    )
    libero = case.startswith("libero-")
    optimizer = dmuon.Muon(
        model,
        lr=1e-4 if libero else 0.02,
        adamw_lr=5e-5 if libero else 1e-3,
        adamw_betas=(0.95, 0.95) if libero else (0.9, 0.999),
        momentum=0.95,
        weight_decay=decay,
        adamw_weight_decay=decay,
        nesterov=True,
        ns_steps=5,
        ns_backend="gram",
        compute_backend=backend,
        replicate_async=False,
        first_step_progress_log=False,
    )
    owners = {dp.param_name: dp for dp in dmuon.get_owned_params(model, 0)}
    return model, optimizer, owners


def update(backend, arm, grads, clip_norm):
    model, opt, owners = arm
    opt.zero_grad()
    anchor = torch.zeros((), device=grads[0].device, requires_grad=True)
    model(anchor, *grads).backward()
    dmuon.prepare_muon_grads(model)
    buckets = [
        dmuon.GradClipBucket(
            route,
            [p._reduced_grad for p in owners.values() if p._dmuon_route == route],
            reduce=False,
        )
        for route in ("muon", "adamw")
    ]
    torch.npu.synchronize()
    start = time.perf_counter()
    if clip_norm:
        dmuon.clip_grad_norm_buckets_(buckets, clip_norm, compute_backend=backend)
    opt.step()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000


def relative_l2(actual, expected, *, require_nonzero=False):
    error = torch.linalg.vector_norm(actual.float() - expected.float()).item()
    norm = torch.linalg.vector_norm(expected.float()).item()
    if not math.isfinite(error) or not math.isfinite(norm):
        raise ValueError("non-finite parameter or optimizer state")
    if require_nonzero and norm == 0:
        raise ValueError("reference optimizer did not update a parameter")
    return error / max(norm, 1e-30)


def compare_step(arms, before, step):
    _, torch_opt, ref = arms["torch"]
    _, tile_opt, got = arms["tilelang"]
    errors = {}
    for name in ref:
        route = ref[name]._dmuon_route
        expected, actual = ref[name]._owned_data, got[name]._owned_data
        field = f"{route}_parameter"
        errors[field] = max(errors.get(field, 0.0), relative_l2(actual, expected))
        field = f"{route}_update"
        update_error = relative_l2(
            actual - before["tilelang"][name],
            expected - before["torch"][name],
            require_nonzero=True,
        )
        errors[field] = max(errors.get(field, 0.0), update_error)
        ref_state = torch_opt.state[id(ref[name])]
        tile_state = tile_opt.state[id(got[name])]
        fields = ("momentum_buffer",) if route == "muon" else ("exp_avg", "exp_avg_sq")
        for field in fields:
            errors[field] = max(
                errors.get(field, 0.0), relative_l2(tile_state[field], ref_state[field])
            )
        if route == "adamw" and not (tile_state["step"] == ref_state["step"] == step):
            raise ValueError("AdamW step counters differ")
    return errors


@contextmanager
def trace_compute_calls():
    """Count backend arithmetic entry points in a separate, untimed update."""
    calls = Counter()
    stages = {
        "momentum_fused",
        "momentum_norm",
        "gram_newton_schulz_tilelang",
        "apply_update",
        "adamw_one_scalar",
        "clip_grad_norm_buckets_tilelang",
    }

    def observe(frame, event, _arg):
        if event != "call" or frame.f_code.co_name not in stages:
            return
        module = frame.f_globals.get("__name__", "")
        if module.startswith("dmuon.backend.tilelang."):
            calls[frame.f_code.co_name] += 1

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def run_case(name, args, mesh, device):
    matrices, adamw = CASES[name]
    if args.include_embedding_shard and name in EMBEDDING_SHARD:
        adamw = [*adamw, (EMBEDDING_SHARD[name],)]
    shapes = {
        f"{route}_{i}": shape
        for route, group in (("muon", matrices), ("adamw", adamw))
        for i, shape in enumerate(group)
    }
    rng = torch.Generator().manual_seed(args.seed)
    initial = {
        key: (torch.randn(shape, generator=rng) * 0.02).to(device) for key, shape in shapes.items()
    }
    arms = {
        backend: make_arm(backend, mesh, initial, name, args.weight_decay)
        for backend in ("torch", "tilelang")
    }
    del initial
    times = {backend: [] for backend in arms}
    worst = {}
    calls = {}
    timed_end = args.warmup + args.steps
    for i in range(timed_end + 1):
        grads = [
            torch.randn(size, generator=rng).to(device=device, dtype=torch.bfloat16)
            for size in shapes.values()
        ]
        before = {
            backend: {k: p._owned_data.clone() for k, p in arm[2].items()}
            for backend, arm in arms.items()
        }
        order = ("torch", "tilelang") if (i - args.warmup) % 2 == 0 else ("tilelang", "torch")
        for backend in order:
            if i == timed_end:
                with trace_compute_calls() as observed:
                    update(backend, arms[backend], grads, args.clip_norm)
                calls[backend] = dict(observed)
            else:
                elapsed = update(backend, arms[backend], grads, args.clip_norm)
                if i >= args.warmup:
                    times[backend].append(elapsed)
        for field, value in compare_step(arms, before, i + 1).items():
            worst[field] = max(worst.get(field, 0.0), value)
    medians = {backend: statistics.median(samples) for backend, samples in times.items()}
    if calls["torch"] or not calls["tilelang"]:
        raise RuntimeError(f"backend selection was not observed: {calls}")
    return {
        "case": name,
        "muon_shapes": matrices,
        "adamw_shapes": adamw,
        "compute_calls": calls,
        "samples_ms": times,
        "median_ms": medians,
        "speedup": medians["torch"] / medians["tilelang"],
        "worst_relative_l2": worst,
        "precision_pass": all(
            value
            <= (0.08 if field == "muon_update" else 0.03 if field == "muon_parameter" else 1e-4)
            for field, value in worst.items()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=CASES, action="append", help="repeat to select cases; default: all"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0, help="visible NPU index")
    parser.add_argument(
        "--clip-norm", type=float, default=1.0, help="per-segment max norm; 0 disables clipping"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="nonzero by default to exercise decoupled weight decay",
    )
    parser.add_argument(
        "--include-embedding-shard",
        action="store_true",
        help="add the captured rank-local flat AdamW embedding shard to layer cases",
    )
    parser.add_argument("--output", type=Path, help="write timings and precision metrics as JSON")
    args = parser.parse_args()
    if args.warmup < 1 or args.steps < 2 or args.clip_norm < 0 or args.weight_decay < 0:
        parser.error("need warmup >= 1, steps >= 2, and nonnegative clipping and weight decay")

    import torch_npu

    torch.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    results = []
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group(
            "hccl", init_method=(Path(directory) / "store").as_uri(), rank=0, world_size=1
        )
        try:
            mesh = init_device_mesh("npu", (1,))
            for name in args.case or CASES:
                result = run_case(name, args, mesh, device)
                results.append(result)
                print(json.dumps(result), flush=True)
        finally:
            dist.destroy_process_group()
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "device": torch.npu.get_device_name(args.device),
                    "torch": torch.__version__,
                    "torch_npu": torch_npu.__version__,
                    "seed": args.seed,
                    "warmup": args.warmup,
                    "steps": args.steps,
                    "clip_norm": args.clip_norm,
                    "weight_decay": args.weight_decay,
                    "cases": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0 if all(row["precision_pass"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
