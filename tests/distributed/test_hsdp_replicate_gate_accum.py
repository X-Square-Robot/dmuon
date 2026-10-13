"""HSDP gradient accumulation with the replicate-reduce gate turned off.

``DedicatedParamGroup.replicate_grads_enabled = False`` mirrors FSDP2's
``set_requires_all_reduce(False)``: each micro-batch still runs the Stage-1
shard reduce, but the Stage-2 replicate reduce is deferred and the shard-only
partial is held in ``_partial_reduce_by_param`` until the gate flips back.

The reduced gradients handed to the optimizer must match plain ``no_sync``
accumulation, which defers both stages.  This covers owner-managed Muon params
and sharded AdamW params; FSDP2 symmetric params use FSDP2's own gate.

The continuous optimizer/state check uses controlled equal-gradient inputs:
each native gated gradient first passes the unchanged gradient oracle, then
receives the reference's frozen gradient before the optimizer step. Model and
optimizer state must match byte-for-byte over three naturally continuing
updates. This does not assert equal raw training trajectories for two different
floating-point accumulation orders through low-precision Newton-Schulz.

Run: ``torchrun --nproc_per_node=4 tests/distributed/test_hsdp_replicate_gate_accum.py``
For native HCCL, append ``--device npu``. Both require four physical devices.
All ranks participate in every parameter on the final micro-batch: this test
does not introduce a partial-only collective protocol for unused parameters.
"""

import argparse
from contextlib import nullcontext
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from dmuon import (
    Muon,
    NewtonSchulz,
    dedicate_params,
    get_model_state_dict,
    get_optimizer_state_dict,
    no_sync,
    set_model_state_dict,
    wait_all_post_step_broadcasts,
    wait_all_reduces,
)
from dmuon.checkpoint import _compute_dedicated_fqns
from dmuon._core.device import get_device_module
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
        self.layers = nn.ModuleList([Block(hidden, intermediate) for _ in range(num_layers)])
        self.head = nn.Linear(hidden, 1, bias=False)
        # Odd length tests padding; length one leaves a rank with no valid
        # AdamW elements in each shard group.
        self.padded_weight = nn.Parameter(torch.randn(5))
        self.tiny_weight = nn.Parameter(torch.randn(1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return (
            self.head(x).sum()
            + x.mean()
            * (self.padded_weight.square().sum() + self.tiny_weight.square().sum())
            * 0.03125
        )


SHARDED_ADAMW_NAMES = {"head.weight", "padded_weight", "tiny_weight"}


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
    get_device_module(next(model.parameters()).device).current_stream().synchronize()
    for state in iter_dedicated_states(model):
        assert not state.group._partial_reduce_by_param, (
            "replicate partial was not flushed before the optimizer step"
        )
    return {name: grad.detach().cpu().clone() for name, grad in live_gradients(model).items()}


def live_gradients(model):
    """Return the actual local tensors about to be consumed, including padding."""
    grads = {}
    for dp, fqn in _compute_dedicated_fqns(model).items():
        if getattr(dp, "_dmuon_route", None) == "sharded_adamw":
            grad = dp._sharded_adamw_grad
        else:
            grad = dp._reduced_grad
        if grad is not None:
            grads[fqn] = grad
    for name, param in model.named_parameters():
        if param.grad is not None:
            local = getattr(param.grad, "_local_tensor", param.grad)
            assert name not in grads, f"duplicate gradient entry: {name}"
            grads[name] = local
    return grads


def assign_canonical_gradients(model, canonical):
    """Copy frozen equal-gradient inputs after validating the entire schema."""
    actual = live_gradients(model)
    assert actual.keys() == canonical.keys(), "canonical gradient names differ"
    for name, grad in actual.items():
        source = canonical[name]
        assert isinstance(source, torch.Tensor), f"{name}: canonical gradient is not a tensor"
        assert source.shape == grad.shape and source.dtype == grad.dtype, (
            f"{name}: canonical gradient shape/dtype differs"
        )
        assert torch.isfinite(source).all(), f"{name}: non-finite canonical gradient"
    with torch.no_grad():
        for name, grad in actual.items():
            grad.copy_(canonical[name])


def train(
    model_state,
    device,
    hsdp_mesh,
    batches,
    *,
    gates,
    gated,
    reset_weights,
    canonical_observations=None,
    gradient_failures=None,
    gradient_worst=None,
):
    runtime = get_device_module(device)
    model, optimizer = build(model_state, device, hsdp_mesh)
    observations = []
    for step, micros in enumerate(batches):
        for index, (batch, enabled) in enumerate(zip(micros, gates)):
            if gated:
                set_replicate_gate(model, enabled)
            context = no_sync(model) if not gated and index < len(micros) - 1 else nullcontext()
            with context:
                (model(batch) / len(micros)).backward()
            # Exercise the early buffer-safety drain, including non-owner
            # shards. Repeated final waits must not unpack a partial twice.
            for state in iter_dedicated_states(model):
                state.group.wait_for_stage1_reduce()
        gradients = capture_grads(model, optimizer)
        wait_all_reduces(model)
        repeated = capture_grads(model, optimizer)
        for name in gradients:
            torch.testing.assert_close(repeated[name], gradients[name], rtol=0, atol=0)
        if canonical_observations is not None:
            assert gradient_failures is not None
            canonical = canonical_observations[step]["gradients"]
            compare(
                gradients,
                canonical,
                gradient_failures,
                path=f"controlled-native-gradient[{step}]",
                worst=gradient_worst if gradient_worst is not None else [0.0],
            )
            # Validate the native result first. The explicit copy isolates the
            # optimizer/state test from accumulation-order roundoff in NS.
            assign_canonical_gradients(model, canonical)
        optimizer.step()
        wait_all_post_step_broadcasts(model)
        observation = {"gradients": gradients}
        if not reset_weights:
            observation["model"] = get_model_state_dict(model, rank0_only=False)
            observation["optimizer"] = get_optimizer_state_dict(model, optimizer, rank0_only=False)
        observations.append(observation)
        for state in iter_dedicated_states(model):
            assert not state.group._partial_reduce_by_param, (
                "replicate-gated partial survived the optimizer step"
            )
        optimizer.zero_grad()
        # Reset the weights so later steps compare gradients rather than
        # accumulated bf16 Newton-Schulz rounding; optimizer and gate state
        # carry over, so stale partials would still show up.
        if reset_weights:
            runtime.current_stream().synchronize()
            set_model_state_dict(model, model_state)
    # An abandoned gated accumulation must not pollute the next step.
    set_replicate_gate(model, False)
    model(batches[0][0]).backward()
    wait_all_reduces(model)
    assert any(state.group._partial_reduce_by_param for state in iter_dedicated_states(model))
    optimizer.zero_grad()
    assert all(not state.group._partial_reduce_by_param for state in iter_dedicated_states(model))
    set_replicate_gate(model, True)
    del optimizer, model
    gc.collect()
    runtime.synchronize()
    runtime.empty_cache()
    return observations


def compare(actual, expected, failures, *, path, worst):
    if isinstance(expected, torch.Tensor):
        if (
            not isinstance(actual, torch.Tensor)
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
        ):
            failures.append(f"{path}: tensor schema differs")
            return
        if expected.numel() == 0:
            return
        if expected.is_floating_point():
            diff = (actual.float() - expected.float()).abs().max().item()
            worst[0] = max(worst[0], diff / max(expected.float().abs().max().item(), 1e-12))
            if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-7):
                failures.append(f"{path}: max_abs_diff={diff:.3e}")
        elif not torch.equal(actual, expected):
            failures.append(f"{path}: tensor values differ")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            failures.append(f"{path}: mapping keys differ")
            return
        for name in expected:
            compare(actual[name], expected[name], failures, path=f"{path}.{name}", worst=worst)
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            failures.append(f"{path}: sequence schema differs")
            return
        for index, item in enumerate(expected):
            compare(actual[index], item, failures, path=f"{path}[{index}]", worst=worst)
    elif actual != expected:
        failures.append(f"{path}: values differ")


def compare_exact(actual, expected, failures, *, path):
    """Strict state oracle, including dtype, empty shards and signed-zero bits."""
    if isinstance(expected, torch.Tensor):
        if (
            not isinstance(actual, torch.Tensor)
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
        ):
            failures.append(f"{path}: tensor schema differs")
            return
        if expected.is_floating_point() and (
            not torch.isfinite(actual).all() or not torch.isfinite(expected).all()
        ):
            failures.append(f"{path}: non-finite state")
            return
        left = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        right = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(left, right):
            failures.append(f"{path}: tensor bytes differ")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            failures.append(f"{path}: mapping keys differ")
            return
        for name in expected:
            compare_exact(actual[name], expected[name], failures, path=f"{path}.{name}")
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            failures.append(f"{path}: sequence schema differs")
            return
        for index, item in enumerate(expected):
            compare_exact(actual[index], item, failures, path=f"{path}[{index}]")
    elif type(actual) is not type(expected) or actual != expected:
        failures.append(f"{path}: values differ")


def diagnose_ns_rounding(reference, gated, device):
    """Report the raw first-step rounding boundary; never an acceptance gate."""
    ns = NewtonSchulz()
    for name, ref_cpu in reference[0]["gradients"].items():
        if "proj" not in name:
            continue
        ref = ref_cpu.to(device)
        got = gated[0]["gradients"][name].to(device)

        def normalized_half(grad):
            value = grad.float()
            if value.shape[0] > value.shape[1]:
                value = value.T
            return (value / (value.norm() + 1e-7)).half().contiguous()

        ref_half, got_half = normalized_half(ref), normalized_half(got)
        ref_update, got_update = ns.local(ref, 3), ns.local(got, 3)
        repeated_update = ns.local(ref, 3)
        assert torch.equal(ref_update, repeated_update), "same-input NS replay differed"
        grad_diff = (got - ref).abs().max().item()
        half_mismatches = int((got_half != ref_half).sum().item())
        ns_diff = (got_update - ref_update).abs().max().item()
        print(
            f"DIAGNOSTIC {name}: raw_grad_max_abs={grad_diff:.9e}, "
            f"normalized_fp16_changed={half_mismatches}, ns_max_abs={ns_diff:.9e}, "
            "same_input_ns_replay=exact",
            flush=True,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "npu"), default="cuda")
    parser.add_argument("--diagnose-ns-rounding", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    runtime = get_device_module(args.device)
    if args.device == "npu":
        assert "torch_npu.contrib.transfer_to_npu" not in sys.modules, (
            "native NPU gate forbids the transfer shim"
        )
    dist.init_process_group("nccl" if args.device == "cuda" else "hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4, "requires 4 ranks for a 2x2 HSDP mesh"
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    runtime.set_device(local_rank)
    device = torch.device(args.device, local_rank)
    hsdp_mesh = init_device_mesh(args.device, (2, 2), mesh_dim_names=("replicate", "shard"))

    torch.manual_seed(0)
    model_state = {k: v.to(device) for k, v in TinyModel().state_dict().items()}
    num_steps = 3
    # Per-rank data so that both the shard and the replicate average matter.
    gen = torch.Generator(device="cpu").manual_seed(1234 + rank)
    failures = []
    worst = [0.0]
    cases = (
        ((False, True), True),
        ((False, False, False, True), True),
        ((True, False, True), True),
        ((False, True), False),
    )
    for gates, reset_weights in cases:
        batches = [
            [torch.randn(4, 128, generator=gen).to(device) * 0.125 for _ in gates]
            for _ in range(num_steps)
        ]
        if args.diagnose_ns_rounding and reset_weights:
            continue  # Advance the generator to the original continuous case.
        if args.diagnose_ns_rounding:
            batches = batches[:1]
        reference = train(
            model_state,
            device,
            hsdp_mesh,
            batches,
            gates=gates,
            gated=False,
            reset_weights=reset_weights,
        )
        dist.barrier()
        gated = train(
            model_state,
            device,
            hsdp_mesh,
            batches,
            gates=gates,
            gated=True,
            reset_weights=reset_weights,
            canonical_observations=reference
            if not reset_weights and not args.diagnose_ns_rounding
            else None,
            gradient_failures=failures,
            gradient_worst=worst,
        )
        dist.barrier()
        if args.diagnose_ns_rounding:
            diagnose_ns_rounding(reference, gated, device)
            continue
        if reset_weights:
            compare(gated, reference, failures, path=f"gates={gates},reset=True", worst=worst)
        else:
            for step, (actual, expected) in enumerate(zip(gated, reference)):
                for section in ("model", "optimizer"):
                    compare_exact(
                        actual[section],
                        expected[section],
                        failures,
                        path=f"controlled-same-gradient[{step}].{section}",
                    )
            if rank == 0:
                print(
                    "Controlled same-gradient trajectory: three continuing updates; model/state byte-exact oracle",
                    flush=True,
                )
    flag = torch.tensor([len(failures)], device=device)
    dist.all_reduce(flag)
    if failures:
        print(f"[rank {rank}] FAILED:\n  " + "\n  ".join(failures), flush=True)
    worst_t = torch.tensor(worst, device=device)
    dist.all_reduce(worst_t, op=dist.ReduceOp.MAX)
    if rank == 0:
        if not args.diagnose_ns_rounding:
            print(f"replicate-gate vs no_sync: worst rel consumed-grad diff = {worst_t.item():.3e}")
        if flag.item() == 0:
            if args.diagnose_ns_rounding:
                print(
                    "DIAGNOSTIC ONLY: raw accumulation/FP16/NS differences reported; no trajectory equivalence claim"
                )
            else:
                print(
                    "PASSED: native consumed-gradient oracle and controlled same-gradient optimizer trajectory"
                )
    dist.barrier()
    dist.destroy_process_group()
    if flag.item():
        sys.exit(1)


if __name__ == "__main__":
    main()
