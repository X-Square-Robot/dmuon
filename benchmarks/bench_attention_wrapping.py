"""Three real wrapping groups: forward/backward, prefetch waits and memory.

This is a synthetic projection/MoE workload, not model throughput. It deliberately
excludes optimizer.step/NS. Use torchrun with 1 or 2 CUDA ranks.
"""

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dmuon
from dmuon import MatrixLayout


class Layer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.qkv = nn.Parameter(torch.randn(3 * hidden, hidden) * 0.01)
        self.gate_up = nn.Parameter(torch.randn(2, hidden, 2 * hidden) * 0.01)
        self.down = nn.Parameter(torch.randn(2, hidden, hidden) * 0.01)

    def forward(self, x):
        h = self.hidden
        qkv = F.linear(x.to(torch.bfloat16), self.qkv.to(torch.bfloat16))
        mixed = qkv.reshape(-1, 8, 3, h // 8).sum(2).reshape(-1, h)
        gate, up = (mixed @ self.gate_up[0].to(torch.bfloat16)).chunk(2, -1)
        return x + (F.silu(gate) * up @ self.down[0].to(torch.bfloat16)).float()


class Stack(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.layers = nn.ModuleList([Layer(hidden) for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("shard",))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    results = []
    for hidden in (256, 1024):
        for backend, budget in (("torch", 8), ("cuda", 8), ("cuda", 32), ("cuda", 132)):
            os.environ["DMUON_COPY_BACKEND"] = backend
            os.environ["DMUON_COPY_MAX_CTAS"] = str(budget)
            torch.manual_seed(19)
            model = Stack(hidden).to(device)
            reference = deepcopy(model)
            layouts = {
                "qkv": MatrixLayout.view(8, 3, hidden // 8, hidden)
                .permute(1, 0, 2, 3)
                .reshape(3, hidden, hidden),
                "gate_up": MatrixLayout.view(2, hidden, 2, hidden).permute(0, 2, 3, 1),
                "down": MatrixLayout.view(2, hidden, hidden),
            }

            def policy(name, param):
                return {
                    "route": "muon",
                    "matrix_layout": layouts[name.rsplit(".", 1)[-1]],
                    "param_dtype": torch.bfloat16,
                    "grad_dtype": torch.float32,
                    "master_dtype": torch.float32,
                }

            dmuon.dedicate_params(
                model,
                mesh,
                predicate=lambda n, p: True,
                hook_boundary_predicate=lambda m: isinstance(m, Layer),
                process_group_policy="shared",
                param_policy_fn=policy,
            )
            optimizer = dmuon.Muon(model, first_step_progress_log=False)
            ctx = model._dedicated_comm_ctx
            ctx.forward_prefetch_depth = 2
            ctx.record_forward_profile = False
            x = torch.randn(32, hidden, device=device, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            actual, expected = model(x), reference(y)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.square().mean().backward()
            expected.square().mean().backward()
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
            dmuon.prepare_muon_grads(model)
            optimizer.zero_grad()
            del actual, expected, reference, y

            def iteration(optimizer=optimizer, x=x, model=model):
                optimizer.zero_grad()
                x.grad = None
                model(x).square().mean().backward()
                dmuon.prepare_muon_grads(model)

            for _ in range(3):
                iteration()
            torch.cuda.synchronize()
            times, gpu_times = [], []
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            for _ in range(5):
                dist.barrier()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                wall_start = time.perf_counter()
                for _ in range(10):
                    iteration()
                end.record()
                end.synchronize()
                times.append((time.perf_counter() - wall_start) * 1000 / 10)
                gpu_times.append(start.elapsed_time(end) / 10)
            peak = torch.cuda.max_memory_allocated()
            ctx.record_forward_profile = True
            ctx.consume_forward_unshard_profile()
            for _ in range(5):
                iteration()
            torch.cuda.synchronize()
            profile = ctx.consume_forward_unshard_profile()
            result = {
                "hidden": hidden,
                "backend": backend,
                "cta_budget": budget,
                "world": dist.get_world_size(),
                "groups": 3,
                "prefetch_depth": 2,
                "wall_ms": times,
                "gpu_interval_ms": gpu_times,
                "wall_median_ms": statistics.median(times),
                "peak_allocated_bytes": peak,
                "baseline_allocated_bytes": before,
                "profile": profile,
                "profile_iterations": 5,
                "forward_input_grad_exact": True,
            }
            results.append(result)
            if dist.get_rank() == 0:
                print(json.dumps(result), flush=True)
                Path(args.output).write_text(json.dumps(results, indent=2))
            dist.barrier()
            del model, optimizer, ctx, x, iteration
            gc.collect()
            torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
