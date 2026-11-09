"""Original-model matrix policy integration under real multi-process collectives.

CUDA: torchrun --standalone --nproc_per_node=2 tests/distributed/test_matrix_layout_owner.py
CPU: add --cpu (Gloo collectives, simulated synchronous device streams/events).
Four ranks: add --hsdp. Add --recompute for activation checkpointing.
"""

import argparse
from contextlib import nullcontext
from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dmuon
from dmuon import MatrixLayout


class Event:
    def __init__(self, *args, **kwargs):
        pass

    def record(self, *args):
        pass

    def query(self):
        return True


class Stream:
    def __init__(self, *args, **kwargs):
        pass

    def wait_event(self, *args):
        pass

    def wait_stream(self, *args):
        pass

    def record_event(self):
        return Event()


def synchronous_cpu_runtime():
    from importlib import import_module

    runtime = SimpleNamespace(
        Stream=Stream,
        Event=Event,
        current_stream=lambda: Stream(),
        stream=lambda _stream: nullcontext(),
    )
    for name in [
        "dmuon._core.comm",
        "dmuon._backends.fsdp2.group",
        "dmuon.optim.muon",
        "dmuon.utils",
    ]:
        import_module(name).get_device_module = lambda _device: runtime
    torch.Tensor.record_stream = lambda _tensor, _stream: None
    dist._coalescing_manager = lambda **_kwargs: nullcontext()
    native_reduce = dist.reduce

    def reduce(tensor, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if op == dist.ReduceOp.AVG:
            if async_op:
                raise ValueError("The CPU harness implements synchronous reductions only")
            native_reduce(tensor, dst, op=dist.ReduceOp.SUM, group=group)
            tensor.div_(dist.get_world_size(group))
            return None
        return native_reduce(tensor, dst, op=op, group=group, async_op=async_op)

    dist.reduce = reduce


class Model(nn.Module):
    def __init__(self, weight, expert=None, column=False, compute_dtype=None):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.expert = expert
        self.column = column
        self.compute_dtype = compute_dtype

    def forward(self, hidden):
        weight = (
            self.weight.to(self.compute_dtype) if self.compute_dtype is not None else self.weight
        )
        if self.column:
            if weight.ndim == 3:
                if self.expert is not None:
                    return nn.functional.linear(hidden, weight[self.expert].t())
                return torch.einsum("bh,ehi->bei", hidden, weight).sum(1)
            return hidden @ weight
        if weight.ndim == 3:
            if self.expert is not None:
                return nn.functional.linear(hidden, weight[self.expert])
            return torch.einsum("bh,eih->bei", hidden, weight).sum(1)
        return nn.functional.linear(hidden, weight)


class WrappedLayer(nn.Module):
    """Several native parameters with mixed layouts at one wrapping boundary."""

    def __init__(self, device, dtype):
        super().__init__()
        self.qkv = nn.Parameter(torch.randn(24, 8, device=device) * 0.05)
        self.gate_up = nn.Parameter(torch.randn(2, 8, 8, device=device) * 0.05)
        self.down = nn.Parameter(torch.randn(2, 4, 8, device=device) * 0.05)
        self.shared = nn.Parameter(torch.randn(8, 8, device=device) * 0.05)
        self.dtype = dtype

    def forward(self, hidden):
        qkv = nn.functional.linear(hidden, self.qkv.to(self.dtype))
        mixed = qkv.view(-1, 2, 3, 4).sum(2).reshape(-1, 8)
        gate, up = torch.einsum("bh,ehi->bei", mixed, self.gate_up.to(self.dtype)).chunk(2, -1)
        experts = torch.einsum(
            "bei,eih->beh", torch.nn.functional.silu(gate) * up, self.down.to(self.dtype)
        ).sum(1)
        return hidden + experts + nn.functional.linear(mixed, self.shared.to(self.dtype))


class WrappedStack(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.layers = nn.ModuleList([WrappedLayer(device, dtype) for _ in range(3)])

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def run_wrapped_prefetch(device, shard, replicate, args):
    torch.manual_seed(91)
    dtype = torch.bfloat16 if args.mixed else torch.float32
    model = WrappedStack(device, dtype)
    reference = deepcopy(model)
    layouts = {
        "qkv": MatrixLayout.view(2, 3, 4, 8).permute(1, 0, 2, 3).reshape(3, 8, 8),
        "gate_up": MatrixLayout.view(2, 8, 2, 4).permute(0, 2, 3, 1),
        "down": MatrixLayout.view(2, 4, 8).permute(0, 2, 1),
    }
    dmuon.dedicate_params(
        model,
        shard,
        replicate_mesh=replicate,
        predicate=lambda _n, _p: True,
        hook_boundary_predicate=lambda module: isinstance(module, WrappedLayer),
        process_group_policy="shared",
        compute_dtype=dtype,
        param_policy_fn=lambda name, _p: {
            "matrix_layout": layouts.get(name.rsplit(".", 1)[-1]),
            "grad_dtype": torch.float32,
        },
    )
    model._dedicated_comm_ctx.forward_prefetch_depth = 2
    optimizer = dmuon.Muon(
        model, lr=0.01, momentum=0, weight_decay=0, first_step_progress_log=False
    )
    for step in range(2):
        optimizer.zero_grad()
        reference.zero_grad()
        torch.manual_seed(101 + step + dist.get_rank())
        hidden = torch.randn(3, 8, device=device, dtype=dtype, requires_grad=True)
        ref_hidden = hidden.detach().clone().requires_grad_()
        actual = checkpoint(model, hidden, use_reentrant=False) if args.recompute else model(hidden)
        expected = reference(ref_hidden)
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(hidden.grad, ref_hidden.grad)
        dmuon.prepare_muon_grads(model)
        with torch.no_grad():
            for name, param in reference.named_parameters():
                dist.all_reduce(param.grad)
                param.grad.div_(dist.get_world_size())
                layout = layouts.get(name.rsplit(".", 1)[-1], MatrixLayout.whole())
                plan = layout.compile(tuple(param.shape))
                value, gradient = plan.pack(param).clone(), plan.pack(param.grad)
                for m in plan.matrices:
                    block = value[m.offset : m.offset + m.numel].view(m.shape)
                    grad = gradient[m.offset : m.offset + m.numel].view(m.shape)
                    block.add_(dmuon.newton_schulz(grad), alpha=-0.01 * 0.2 * max(m.shape) ** 0.5)
                param.copy_(plan.unpack(value))
        optimizer.step()
        saved = dmuon.get_model_state_dict(model, rank0_only=False)
        for name, param in reference.named_parameters():
            torch.testing.assert_close(saved[name], param.cpu(), rtol=1e-3, atol=1e-4)
    if dist.get_rank() == 0:
        print(
            "passed three wrapping groups, depth-2 prefetch, two steps, mixed layouts/dtypes",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--hsdp", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--replicate-gated", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--stack", action="store_true")
    args = parser.parse_args()
    if args.cpu:
        device = "cpu"
        synchronous_cpu_runtime()
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required unless --cpu is specified")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = "cuda"
    dist.init_process_group("gloo" if args.cpu else "nccl")
    world = dist.get_world_size()
    if args.hsdp:
        if world < 4 or world % 2:
            raise ValueError("--hsdp requires an even world size of at least four")
        mesh = init_device_mesh(device, (2, world // 2), mesh_dim_names=("replicate", "shard"))
        shard, replicate = mesh["shard"], mesh["replicate"]
    else:
        shard = init_device_mesh(device, (world,), mesh_dim_names=("shard",))
        replicate = None
    cases = [
        ((12, 8), MatrixLayout.whole(), False),
        ((12, 8), MatrixLayout.view(3, 4, 8), False),
        ((12, 8), MatrixLayout.view(3, 2, 2, 8), False),
        ((12, 8), MatrixLayout.rows(8, 2, 2), False),
        ((3, 8, 8), MatrixLayout.view(3, 8, 8), False),
        ((3, 8, 8), MatrixLayout.view(3, 2, 4, 8), False),
        ((24, 8), MatrixLayout.view(2, 3, 4, 8).permute(1, 0, 2, 3).reshape(3, 8, 8), False),
        ((24, 8), MatrixLayout.view(2, 3, 4, 8).permute(1, 0, 2, 3), False),
        ((16, 8), MatrixLayout.view(2, 4, 2, 8).split((2, 1, 1), dim=1).reshape(-1, 8), False),
        ((8, 24), MatrixLayout.view(8, 3, 8).permute(1, 2, 0), True),
        ((3, 8, 8), MatrixLayout.view(3, 8, 2, 4).permute(0, 2, 3, 1), True),
    ]
    compute_dtype = torch.bfloat16 if args.mixed else torch.float32
    for shape, layout, column in cases:
        torch.manual_seed(31)
        expert = dist.get_rank() % shape[0] if args.sparse and len(shape) == 3 else None
        model = Model(
            torch.randn(shape, device=device),
            expert=expert,
            column=column,
            compute_dtype=compute_dtype,
        )
        reference = deepcopy(model)
        dmuon.dedicate_params(
            model,
            shard,
            replicate_mesh=replicate,
            predicate=lambda _name, p: p.requires_grad,
            compute_dtype=compute_dtype,
            process_group_policy="shared",
            param_policy_fn=lambda _n, _p: {"matrix_layout": layout, "grad_dtype": torch.float32},
        )
        optimizer = dmuon.Muon(model, lr=0.01, momentum=0, first_step_progress_log=False)
        for microbatch in range(2):
            torch.manual_seed(71 + dist.get_rank() + microbatch)
            hidden = torch.randn(4, 8, device=device, dtype=compute_dtype, requires_grad=True)
            if args.replicate_gated and replicate is not None:
                for state in model._dedicated_comm_ctx.all_states:
                    state.group.replicate_grads_enabled = microbatch != 0
                context = nullcontext()
            else:
                context = dmuon.no_sync(model) if microbatch == 0 else nullcontext()
            with context:
                actual = (
                    checkpoint(model, hidden, use_reentrant=False)
                    if args.recompute
                    else model(hidden)
                )
                expected = reference(hidden)
                torch.testing.assert_close(actual, expected)
                actual.square().mean().backward()
                expected.square().mean().backward()
        dmuon.prepare_muon_grads(model)
        dist.all_reduce(reference.weight.grad)
        reference.weight.grad.div_(world)
        plan = layout.compile(shape)
        packed_gradient = plan.pack(reference.weight.grad)
        for unit in dmuon.get_dedicated_params(model):
            if unit.is_owner:
                spec = unit.matrix_spec
                expected = packed_gradient[spec.offset : spec.offset + spec.numel].view(spec.shape)
                torch.testing.assert_close(unit._reduced_grad, expected, rtol=1e-4, atol=1e-5)
        optimizer.step()
        with torch.no_grad():
            packed_weight = plan.pack(reference.weight).clone()
            for spec in plan.matrices:
                value = packed_weight[spec.offset : spec.offset + spec.numel].view(spec.shape)
                gradient = packed_gradient[spec.offset : spec.offset + spec.numel].view(spec.shape)
                value.add_(
                    dmuon.newton_schulz(gradient), alpha=-0.01 * 0.2 * max(spec.shape) ** 0.5
                )
            reference.weight.copy_(plan.unpack(packed_weight))
        torch.testing.assert_close(
            model(hidden),
            reference(hidden),
            rtol=0.02 if args.mixed else 1e-3,
            atol=0.01 if args.mixed else 1e-4,
        )
        saved = dmuon.get_model_state_dict(model, rank0_only=False)
        assert set(saved) == {"weight"}
        torch.testing.assert_close(saved["weight"], reference.weight.cpu(), rtol=1e-3, atol=1e-4)
        state = dmuon.get_optimizer_state_dict(model, optimizer, rank0_only=False)
        assert len(state["dedicated"]) == len(layout.compile(shape).matrices)
        # Each call must use a new original-shape slot after loading, even if a forward was live.
        dmuon.set_model_state_dict(model, saved)
        dmuon.set_optimizer_state_dict(model, optimizer, state)
        torch.testing.assert_close(
            model(hidden),
            reference(hidden),
            rtol=0.02 if args.mixed else 1e-3,
            atol=0.01 if args.mixed else 1e-4,
        )
        if dist.get_rank() == 0:
            print(
                "passed", shape, layout, "hsdp", args.hsdp, "recompute", args.recompute, flush=True
            )
        dist.barrier()
    if args.stack:
        run_wrapped_prefetch(device, shard, replicate, args)
    # Different but locally valid layouts must be rejected on every rank.
    invalid = Model(torch.randn(12, 8, device=device))
    rank_layouts = [
        MatrixLayout.whole() if dist.get_rank() == 0 else MatrixLayout.view(3, 4, 8),
        MatrixLayout.view(3, 4, 8)
        if dist.get_rank() == 0
        else (MatrixLayout.view(2, 3, 2, 8).permute(1, 0, 2, 3).reshape(3, 4, 8)),
    ]
    for rank_layout in rank_layouts:
        rejected = False
        try:
            dmuon.dedicate_params(
                invalid,
                shard,
                replicate_mesh=replicate,
                predicate=lambda _n, _p: True,
                process_group_policy="shared",
                param_policy_fn=lambda _n, _p: {"matrix_layout": rank_layout},
            )
        except RuntimeError as error:
            if "differs across ranks" not in str(error):
                raise
            rejected = True
        assert rejected and tuple(invalid.weight.shape) == (12, 8)
    for invalid_geometry in [
        MatrixLayout.view(2, 8, 8) if dist.get_rank() == 0 else MatrixLayout.whole(),
        MatrixLayout.view(12, 8).split((5, 6), dim=0)
        if dist.get_rank() == 0
        else MatrixLayout.whole(),
    ]:
        rejected = False
        try:
            dmuon.dedicate_params(
                invalid,
                shard,
                replicate_mesh=replicate,
                predicate=lambda _n, _p: True,
                process_group_policy="shared",
                param_policy_fn=lambda _n, _p: {"matrix_layout": invalid_geometry},
            )
        except ValueError as error:
            if "exactly cover" not in str(error):
                raise
            rejected = True
        assert rejected and tuple(invalid.weight.shape) == (12, 8)
    dist.barrier()
    if dist.get_rank() == 0:
        print("passed inconsistent/invalid-layout rejection on all ranks", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
