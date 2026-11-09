"""Paired same-layout copy timings and controlled GEMM overlap, JSON evidence.

Device-work graphs exclude pointer uploads/Python; dispatch measurements include
them. No result here represents full-model or distributed training throughput.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.bench_layout_copy import capture, frozen_kernel, graph_overlap, measure
from dmuon._core.layout_copy import CopyExecutor, CopyTask
from tests.helpers.attention_layouts import attention_layout_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = []
    stream = torch.cuda.Stream(priority=-1)
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    b, c = torch.randn_like(a), torch.empty_like(a)
    gemm = lambda: torch.mm(a, b, out=c)
    selected = (
        "k3_q_b_whole",
        "k3_q_b_per_head",
        "k3_kv_b_whole",
        "column_latent_moe",
        "mla_fused_q_kv_a",
        "v41_grouped_output",
    )
    for case in attention_layout_cases("native"):
        if case.name not in selected:
            continue
        source = torch.randn(case.shape, device="cuda", dtype=torch.bfloat16)
        plan = case.layout.compile(case.shape)
        for scatter in (False, True):
            dtype = torch.bfloat16 if scatter else torch.float32
            matrices = [x.to(dtype).contiguous() for x in case.blocks(source)]
            target = torch.empty_like(source)
            tasks = [
                CopyTask(value, target, m, scatter=True) if scatter else CopyTask(source, value, m)
                for value, m in zip(matrices, plan.copy_maps)
            ]
            stream.wait_stream(torch.cuda.current_stream())
            os.environ["DMUON_COPY_BACKEND"] = "torch"
            reference = CopyExecutor()
            previous = lambda reference=reference, tasks=tasks: reference.run(tasks, validate=False)
            torch_dispatch, _ = measure(previous, stream, 20, 5)
            torch_graph = capture(previous, stream, 4)
            torch_graph_us, _ = measure(torch_graph.replay, stream, 20, 5)
            torch_overlap = graph_overlap(previous, gemm, stream, 20, 5, 4)
            for budget in (8, 32, 132):
                os.environ["DMUON_COPY_BACKEND"] = "cuda"
                os.environ["DMUON_COPY_MAX_CTAS"] = str(budget)
                executor = CopyExecutor()
                operation = lambda executor=executor, tasks=tasks: executor.run(
                    tasks, validate=False
                )
                operation()
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                cuda_dispatch, _ = measure(operation, stream, 20, 5)
                torch.cuda.synchronize()
                peak_extra = torch.cuda.max_memory_allocated() - allocated
                fixed = frozen_kernel(executor, tasks, stream)
                graph = capture(fixed, stream, 4)
                cuda_graph, _ = measure(graph.replay, stream, 20, 5)
                overlap = graph_overlap(fixed, gemm, stream, 20, 5, 4)
                if scatter:
                    torch.testing.assert_close(target, source, rtol=0, atol=0)
                else:
                    for value, expected in zip(matrices, case.blocks(source)):
                        torch.testing.assert_close(value, expected.float(), rtol=0, atol=0)
                # Same bytes/tasks and CTA budget, with address remapping
                # removed. This measures the attainable copy rate at this SM
                # budget; it is more useful than the full-chip peak alone.
                identity_tasks = []
                for value, spec in zip(matrices, plan.matrices):
                    interval = (target if scatter else source).flatten()[
                        spec.offset : spec.offset + spec.numel
                    ]
                    identity_tasks.append(
                        CopyTask(value.flatten(), interval)
                        if scatter
                        else CopyTask(interval, value.flatten())
                    )
                identity = CopyExecutor()
                identity.run(identity_tasks, validate=False)
                identity_fixed = frozen_kernel(identity, identity_tasks, stream)
                identity_graph = capture(identity_fixed, stream, 4)
                identity_us, _ = measure(identity_graph.replay, stream, 20, 5)
                result = {
                    "name": case.name,
                    "shape": case.shape,
                    "phase": "restore" if scatter else "grad_gather_cast",
                    "tasks": len(tasks),
                    "cta_budget": budget,
                    "dispatch": executor.last_stats,
                    "torch_dispatch": torch_dispatch,
                    "cuda_dispatch": cuda_dispatch,
                    "torch_device_us": torch_graph_us["median"] / 4,
                    "cuda_device_us": cuda_graph["median"] / 4,
                    "torch_device_rounds_us": [x / 4 for x in torch_graph_us["trials"]],
                    "cuda_device_rounds_us": [x / 4 for x in cuda_graph["trials"]],
                    "torch_overlap": torch_overlap,
                    "cuda_overlap": overlap,
                    "warm_peak_extra_bytes": peak_extra,
                    "same_budget_contiguous_us": identity_us["median"] / 4,
                    "contiguous_dispatch": identity.last_stats,
                    "logical_traffic_bytes": source.numel() * (4 if scatter else 6),
                    "exact": True,
                }
                results.append(result)
                print(json.dumps(result), flush=True)
                Path(args.output).write_text(json.dumps(results, indent=2))
                del (
                    executor,
                    graph,
                    fixed,
                    operation,
                    identity,
                    identity_graph,
                    identity_fixed,
                    identity_tasks,
                )
            del matrices, tasks, target, reference, torch_graph, previous
        del source, plan
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
