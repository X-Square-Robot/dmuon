"""Measure layout copies and compute overlap on the selected CUDA device.

Example: PYTHONPATH=. python benchmarks/bench_layout_copy.py --hidden 1024 --batch 4
Reports local copy/overlap timings, not distributed training throughput.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dmuon import MatrixLayout
from dmuon._core.layout_copy import CopyExecutor, CopyTask, _extension


def summarize(values, cpu_values=None):
    result = {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "trials": values,
    }
    if cpu_values is not None:
        result.update(cpu_submit_median=statistics.median(cpu_values), cpu_submit_trials=cpu_values)
    return result


def measure(operation, stream, iterations, repeats, timing=None):
    """Median of per-round means; retain every profiled kernel pair, not only last."""
    trials, kernel_trials, cpu_trials = [], [], []
    for _ in range(repeats):
        pairs = []
        with torch.cuda.stream(stream):
            for _ in range(3):
                operation()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            cpu_start = time.perf_counter()
            start.record()
            for _ in range(iterations):
                operation()
                if timing is not None:
                    pairs.append(timing())
            end.record()
            cpu_trials.append((time.perf_counter() - cpu_start) * 1e6 / iterations)
        end.synchronize()
        trials.append(1000 * start.elapsed_time(end) / iterations)
        if pairs:
            kernel_trials.append(statistics.mean(a.elapsed_time(b) * 1000 for a, b in pairs))
    return summarize(trials, cpu_trials), summarize(kernel_trials) if kernel_trials else None


def capture(operation, stream, unroll):
    # Capture fixed-address GPU work only. No pointer uploads/CPU dispatch are
    # being claimed eliminated from the real runtime; this isolates device work.
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(unroll):
            operation()
    return graph


def frozen_kernel(executor, tasks, stream):
    desc, prog, tiles, ready, _, _, transpose = next(iter(executor._plans.values()))
    stream.wait_event(ready)
    with torch.cuda.stream(stream):
        pointers = torch.tensor(
            [(t.source.data_ptr(), t.destination.data_ptr()) for t in tasks],
            dtype=torch.int64,
            device="cuda",
        )
    ctas = executor.last_stats["ctas"]
    ext = _extension()
    return lambda: ext.copy(desc, prog, pointers, tiles, ctas, transpose)


def graph_overlap(operation, gemm, stream, iterations, repeats, unroll):
    """GPU fork/join graph: both branches are prequeued, with no Python gap.

    This is a controlled copy/GEMM overlap experiment, not a wrapping benchmark.
    External events keep timing nodes when captured; every unrolled pair has
    its own events, so all pairs contribute to each round's mean.
    """
    # Initialize lazy cuBLAS handles outside capture, including standalone callers.
    with torch.cuda.stream(stream):
        for _ in range(3):
            gemm()
            operation()
    stream.synchronize()
    worker = torch.cuda.Stream(priority=-1)
    events = []
    for _ in range(unroll):
        events.append(tuple(torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)))
    graph = torch.cuda.CUDAGraph()
    stream.synchronize()
    worker.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        for copy_start, copy_end, gemm_start, gemm_end in events:
            worker.wait_stream(stream)
            with torch.cuda.stream(worker):
                copy_start.record()
                operation()
                copy_end.record()
            gemm_start.record(stream)
            gemm()
            gemm_end.record(stream)
            stream.wait_stream(worker)
    for _ in range(3):
        with torch.cuda.stream(stream):
            graph.replay()
    stream.synchronize()
    joined, compute_times, copy_times = [], [], []
    for _ in range(repeats):
        with torch.cuda.stream(stream):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
        end.synchronize()
        joined.append(start.elapsed_time(end) * 1000 / unroll)
        copy_times.append(statistics.mean(a.elapsed_time(b) * 1000 for a, b, _, _ in events))
        compute_times.append(statistics.mean(a.elapsed_time(b) * 1000 for _, _, a, b in events))
    return {
        "joined_us": summarize(joined),
        "gemm_us": summarize(compute_times),
        "copy_us": summarize(copy_times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ctas", default="1,2,4,8,16,24,32,48,64,96,132")
    parser.add_argument("--graph-unroll", type=int, default=8)
    parser.add_argument("--gemm-size", type=int, default=2048)
    parser.add_argument("--overlap", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or _extension() is None:
        raise RuntimeError("Build the optional CUDA layout copy extension and select a CUDA device")
    h, heads = args.hidden, args.heads
    budgets = [int(v) for v in args.ctas.split(",")]
    if (
        h % heads
        or min(
            h,
            heads,
            args.batch,
            args.iterations,
            args.repeats,
            args.graph_unroll,
            args.gemm_size,
            *budgets,
        )
        <= 0
    ):
        raise ValueError("Positive shapes and hidden divisible by heads are required")
    props = torch.cuda.get_device_properties(0)
    print(
        json.dumps(
            {
                "device": props.name,
                "sms": props.multi_processor_count,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "hidden": h,
                "heads": heads,
                "batch": args.batch,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "graph_unroll": args.graph_unroll,
                "overlap_pairs_per_round": args.graph_unroll,
                "gemm_size": args.gemm_size,
                "method": "extended-v2: unprofiled dispatch; mean-of-all kernel events; frozen CUDA graph; fork/join overlap",
            }
        ),
        flush=True,
    )
    stream = torch.cuda.Stream(priority=-1)
    gemm = None
    gemm_stats = None
    overlap_control = None
    if args.overlap:
        a = torch.randn(args.gemm_size, args.gemm_size, device="cuda", dtype=torch.bfloat16)
        b, c = torch.randn_like(a), torch.empty_like(a)
        gemm = lambda: torch.mm(a, b, out=c)
        stream.wait_stream(torch.cuda.current_stream())
        gemm_graph = capture(gemm, stream, args.graph_unroll)
        gemm_stats, _ = measure(gemm_graph.replay, stream, args.iterations, args.repeats)
        gemm_stats = {
            k: (
                [v / args.graph_unroll for v in value]
                if k.endswith("trials")
                else value / args.graph_unroll
            )
            for k, value in gemm_stats.items()
        }
        overlap_control = graph_overlap(
            lambda: None, gemm, stream, args.iterations, args.repeats, args.graph_unroll
        )
    cases = [
        (
            "interleaved_qkv",
            (3 * h, h),
            MatrixLayout.view(heads, 3, h // heads, h).permute(1, 0, 2, 3).reshape(3, h, h),
        ),
        ("column_fused_moe", (2, h, 2 * h), MatrixLayout.view(2, h, 2, h).permute(0, 2, 3, 1)),
    ]
    for name, shape, layout in cases:
        plan = layout.compile(shape)
        originals = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(args.batch)
        ]
        packed = [plan.pack(source) for source in originals]
        matrices = [
            [torch.empty(m.shape, device="cuda", dtype=torch.bfloat16) for m in plan.matrices]
            for _ in originals
        ]
        native_outs = [torch.empty_like(source) for source in originals]
        stream.wait_stream(torch.cuda.current_stream())
        for phase in ("gather", "scatter"):
            tasks = []
            for source, packed_source, outputs, native in zip(
                originals, packed, matrices, native_outs
            ):
                for spec, mapping, output in zip(plan.matrices, plan.copy_maps, outputs):
                    tasks.append(
                        CopyTask(source, output, mapping)
                        if phase == "gather"
                        else CopyTask(
                            packed_source[spec.offset : spec.offset + spec.numel].view(spec.shape),
                            native,
                            mapping,
                            scatter=True,
                        )
                    )

            def previous():
                if phase == "gather":
                    for source, outputs in zip(originals, matrices):
                        temp = plan.pack(source)
                        torch._foreach_copy_(
                            outputs,
                            [
                                temp[m.offset : m.offset + m.numel].view(m.shape)
                                for m in plan.matrices
                            ],
                        )
                else:
                    for source, target in zip(packed, native_outs):
                        plan.unpack(source, out=target)

            reference, _ = measure(previous, stream, args.iterations, args.repeats)
            reference_graph = capture(previous, stream, args.graph_unroll)
            reference_gpu, _ = measure(
                reference_graph.replay, stream, args.iterations, args.repeats
            )
            reference_gpu = {
                k: (
                    [v / args.graph_unroll for v in value]
                    if k.endswith("trials")
                    else value / args.graph_unroll
                )
                for k, value in reference_gpu.items()
            }
            reference_overlap = (
                graph_overlap(
                    previous, gemm, stream, args.iterations, args.repeats, args.graph_unroll
                )
                if gemm
                else None
            )
            for budget in budgets:
                os.environ["DMUON_COPY_BACKEND"] = "cuda"
                os.environ["DMUON_COPY_MAX_CTAS"] = str(budget)
                os.environ["DMUON_COPY_PROFILE"] = "0"
                executor = CopyExecutor()
                operation = lambda: executor.run(tasks, validate=False)
                actual, _ = measure(operation, stream, args.iterations, args.repeats)
                os.environ["DMUON_COPY_PROFILE"] = "1"
                _, kernel = measure(
                    operation, stream, args.iterations, args.repeats, lambda: executor.last_timing
                )
                os.environ["DMUON_COPY_PROFILE"] = "0"
                fixed = frozen_kernel(executor, tasks, stream)
                graph = capture(fixed, stream, args.graph_unroll)
                gpu, _ = measure(graph.replay, stream, args.iterations, args.repeats)
                gpu = {
                    k: (
                        [v / args.graph_unroll for v in value]
                        if k.endswith("trials")
                        else value / args.graph_unroll
                    )
                    for k, value in gpu.items()
                }
                # Keep correctness outside timing, including larger grids and
                # graph replay. Each task overwrites its disjoint output.
                if phase == "gather":
                    for outputs, expected in zip(matrices, packed):
                        torch.testing.assert_close(
                            torch.cat([m.flatten() for m in outputs]), expected, rtol=0, atol=0
                        )
                else:
                    for actual_native, expected in zip(native_outs, originals):
                        torch.testing.assert_close(actual_native, expected, rtol=0, atol=0)
                result = {
                    "layout": name,
                    "phase": phase,
                    "batch": args.batch,
                    "tensor_bytes": sum(t.numel * t.source.element_size() for t in tasks),
                    "cta_budget": budget,
                    "torch_previous_us": reference["median"],
                    "fused_us": actual["median"],
                    "kernel_us": kernel["median"],
                    "kernel_graph_us": gpu["median"],
                    "torch_graph_us": reference_gpu["median"],
                    "statistics": {
                        "torch_dispatch": reference,
                        "fused_dispatch": actual,
                        "kernel_event": kernel,
                        "kernel_graph": gpu,
                        "torch_graph": reference_gpu,
                    },
                    "dispatch": executor.last_stats,
                    "correctness": "exact",
                }
                if args.overlap:
                    result["overlap"] = graph_overlap(
                        fixed, gemm, stream, args.iterations, args.repeats, args.graph_unroll
                    )
                    result["torch_overlap"] = reference_overlap
                    result["gemm_alone"] = gemm_stats
                    result["overlap_control"] = overlap_control
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
