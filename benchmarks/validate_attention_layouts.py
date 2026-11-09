"""Optional real-shape and pinned upstream checks; no checkpoint downloads.

Run with --native or --upstream --moba-source /pinned/moba_naive.py.
Upstream requires fla-core==0.6.0, einops==0.8.2, CUDA and Triton.
"""

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dmuon._core.layout_copy import CopyExecutor, CopyTask
from tests.helpers.attention_layouts import attention_layout_cases, oracle_scatter
from tests.helpers.attention_models import attention, delta_reference, moba_mask


def native():
    os.environ["DMUON_COPY_BACKEND"] = "cuda"
    results = []
    for case in attention_layout_cases("native"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        source = torch.randn(case.shape, device="cuda", dtype=torch.bfloat16)
        plan = case.layout.compile(case.shape)
        for dtype in (torch.bfloat16, torch.float32):
            expected = [x.to(dtype) for x in case.blocks(source)]
            values = [torch.empty(m.shape, dtype=dtype, device="cuda") for m in plan.matrices]
            executor = CopyExecutor()
            executor.run([CopyTask(source, out, m) for out, m in zip(values, plan.copy_maps)])
            for actual, want in zip(values, expected):
                torch.testing.assert_close(actual, want, rtol=0, atol=0)
            changed = [(x + (i % 7 + 1) * 0.125).contiguous() for i, x in enumerate(expected)]
            target = torch.full(case.shape, float("nan"), device="cuda", dtype=dtype)
            executor.run(
                [
                    CopyTask(value, target, m, scatter=True)
                    for value, m in zip(changed, plan.copy_maps)
                ]
            )
            wanted = oracle_scatter(case, changed)
            torch.testing.assert_close(target, wanted, rtol=0, atol=0)
            results.append(
                {
                    "name": case.name,
                    "shape": case.shape,
                    "dtype": str(dtype),
                    "logical_matrices": len(values),
                    "stress_variant": case.stress_variant,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "copy": executor.last_stats,
                    "exact": True,
                }
            )
            print(json.dumps(results[-1]), flush=True)
            del expected, values, changed, target, wanted, executor, actual, want
            gc.collect()
        del source, plan
    return results


def upstream(moba_source):
    spec = importlib.util.spec_from_file_location("pinned_moba_naive", moba_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = []
    for lengths in ((6,), (7, 11), (2, 9, 17)):
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(71)
            values = [
                torch.randn(sum(lengths), 2, 16, device="cuda", dtype=dtype, requires_grad=True)
                for _ in range(3)
            ]
            other = [x.detach().clone().requires_grad_() for x in values]
            cumulative = torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
            )
            output = module.moba_attn_varlen_naive(*values, cumulative, max(lengths), 4, 2)
            expected = []
            offset = 0
            for length in lengths:
                q, k, v = [x[offset : offset + length].unsqueeze(0) for x in other]
                expected.append(attention(q, k, v, moba_mask(q, k, block=4)).squeeze(0).to(dtype))
                offset += length
            expected = torch.cat(expected)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            gradient = torch.randn_like(output)
            output.backward(gradient)
            expected.backward(gradient)
            errors = []
            for a, b in zip(values, other):
                errors.append((a.grad.float() - b.grad.float()).abs().max().item())
                torch.testing.assert_close(
                    a.grad, b.grad, rtol=0.02, atol=0.02 if dtype == torch.bfloat16 else 2e-6
                )
            results.append(
                {
                    "upstream": "MoBA pinned naive",
                    "lengths": lengths,
                    "dtype": str(dtype),
                    "forward_exact": True,
                    "grad_max_errors": errors,
                }
            )
    from fla.ops.kda import chunk_kda
    import fla

    assert fla.__version__ == "0.6.0"
    for length in (32, 64, 129):
        torch.manual_seed(92)
        q, k = [
            F.normalize(torch.randn(1, length, 2, 128, device="cuda"), dim=-1).to(torch.bfloat16)
            for _ in range(2)
        ]
        v = torch.randn_like(q)
        g = -torch.rand_like(q).float() * 0.08
        beta = torch.rand(1, length, 2, device="cuda", dtype=torch.bfloat16)
        inputs = [x.detach().requires_grad_() for x in (q, k, v, g, beta)]
        others = [x.detach().float().requires_grad_() for x in inputs]
        output, state = chunk_kda(*inputs, scale=1.0, output_final_state=True)
        expected, expected_state = delta_reference(*others)
        errors = [
            (output.float() - expected).abs().max().item(),
            (state.float() - expected_state).abs().max().item(),
        ]
        torch.testing.assert_close(output.float(), expected, rtol=0.03, atol=0.012)
        torch.testing.assert_close(state.float(), expected_state, rtol=0.03, atol=0.012)
        gradient = torch.randn_like(output)
        (output.float() * gradient.float()).sum().backward()
        (expected * gradient.float()).sum().backward()
        grad_errors, grad_relative_l2 = [], []
        for a, b in zip(inputs, others):
            grad_errors.append((a.grad.float() - b.grad).abs().max().item())
            relative = ((a.grad.float() - b.grad).norm() / b.grad.norm().clamp_min(1e-20)).item()
            grad_relative_l2.append(relative)
            assert relative < 0.02
            torch.testing.assert_close(a.grad.float(), b.grad, rtol=0.05, atol=0.035)
        results.append(
            {
                "upstream": "fla-core 0.6.0 chunk_kda",
                "length": length,
                "head_dim": 128,
                "dtype": "BF16, FP32 gates",
                "max_errors": errors,
                "grad_max_errors": grad_errors,
                "grad_relative_l2": grad_relative_l2,
                "scale": 1.0,
            }
        )
        print(json.dumps(results[-1]), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--upstream", action="store_true")
    parser.add_argument("--moba-source")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = native() if args.native else upstream(args.moba_source)
    Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
