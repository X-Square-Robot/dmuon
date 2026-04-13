"""LLM benchmark: DMuon on Qwen2.5-7B and Llama-3.1-8B.

Measures:
  1. DMuon step time (fwd + bwd + owner NS)
  2. Loss consistency across ranks (correctness)
  3. Baseline FSDP2 step time (fwd + bwd + SGD, no NS) for reference

Run: torchrun --nproc_per_node=8 benchmarks/bench_llm.py [qwen|llama|both]
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmuon import dedicate_params
from dmuon.utils import get_owned_params


def newton_schulz(G, steps=5):
    X = G / (G.norm() + 1e-7)
    X = X.to(torch.bfloat16)
    for _ in range(steps):
        A = X @ X.T
        B = (1.5 * A - 0.5 * A @ A).to(X.dtype)
        X = B @ X
    return X


def build_qwen25_7b(device):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    config = Qwen2Config(
        hidden_size=3584, intermediate_size=18944, num_hidden_layers=28,
        num_attention_heads=28, num_key_value_heads=4, vocab_size=152064,
        max_position_embeddings=4096, use_cache=False, tie_word_embeddings=False,
    )
    with device:
        model = Qwen2ForCausalLM(config)
    return model, "Qwen2.5-7B"


def build_llama_8b(device):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(
        hidden_size=4096, intermediate_size=14336, num_hidden_layers=32,
        num_attention_heads=32, num_key_value_heads=8, vocab_size=128256,
        max_position_embeddings=4096, use_cache=False, tie_word_embeddings=False,
    )
    with device:
        model = LlamaForCausalLM(config)
    return model, "Llama-3.1-8B"


def bench_fn(fn, warmup=2, repeat=5):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def run_dmuon_bench(build_fn, device, mesh, rank, world_size):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    seq_len = 64
    batch_size = 1

    torch.manual_seed(42)
    model, model_name = build_fn(device)
    total_params = sum(p.numel() for p in model.parameters())

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  {model_name} — {total_params/1e9:.2f}B params")
        print(f"{'='*60}")

    # Apply DMuon + FSDP2
    dedicate_params(
        model, mesh,
        predicate=lambda name, param: "proj" in name and param.ndim == 2,
        compute_dtype=torch.bfloat16,
    )
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    owned = get_owned_params(model, mesh.get_local_rank())
    n_dedicated = sum(1 for m in model.modules() if hasattr(m, '_dedicated_state')
                      for _ in m._dedicated_state.group.params)

    if rank == 0:
        print(f"  Dedicated params: {n_dedicated}")
        print(f"  Rank 0 owns: {len(owned)} params")
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  GPU memory after init: {mem:.1f} GB")

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    def step():
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        # Owner-only NS
        for dp in owned:
            if dp._reduced_grad is not None:
                G = dp._reduced_grad.view(dp._reduced_grad.shape[0], -1).to(torch.bfloat16)
                update = newton_schulz(G)
                dp._owned_data.add_(update.view(dp._owned_data.shape).to(dp._owned_data.dtype), alpha=-0.01)
                dp._reduced_grad = None
        # SGD for symmetric
        for module in model.modules():
            state = getattr(module, '_get_fsdp_state', lambda: None)()
            if state is None or state._fsdp_param_group is None:
                continue
            for fp in state._fsdp_param_group.fsdp_params:
                if fp.sharded_param.grad is not None:
                    fp.sharded_param._local_tensor.add_(
                        fp.sharded_param.grad._local_tensor, alpha=-0.01)
                    fp.sharded_param.grad = None
        return out.loss.item()

    # Warmup
    for _ in range(2):
        step()

    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    dmuon_ms = bench_fn(step)

    # Correctness: collect losses from all ranks
    losses = [step() for _ in range(3)]
    loss_t = torch.tensor(losses, device=device)
    all_losses = [torch.zeros_like(loss_t) for _ in range(world_size)]
    dist.all_gather(all_losses, loss_t)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    if rank == 0:
        max_diff = 0
        for s in range(3):
            vals = [all_losses[r][s].item() for r in range(world_size)]
            max_diff = max(max_diff, max(vals) - min(vals))

        print(f"\n  DMuon Results:")
        print(f"    Step time:        {dmuon_ms:.2f} ms")
        print(f"    Peak GPU memory:  {peak_mem:.1f} GB")
        print(f"    Loss (last 3):    {[f'{l:.4f}' for l in losses]}")
        print(f"    Max rank diff:    {max_diff:.6f}")
        if max_diff < 0.01:
            print(f"    ✅ Loss consistent across {world_size} ranks")
        else:
            print(f"    ⚠️  Loss mismatch: {max_diff:.6f}")

    del model
    torch.cuda.empty_cache()
    return dmuon_ms


def run_baseline_bench(build_fn, device, mesh, rank, world_size):
    """Baseline: standard FSDP2 + redundant NS.

    Every rank all-gathers full gradient for proj params, runs NS redundantly,
    then applies update to its local shard.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    seq_len = 64
    batch_size = 1
    local_rank = mesh.get_local_rank()

    torch.manual_seed(42)
    model, model_name = build_fn(device)

    # Identify proj param FQNs for NS (same as DMuon predicate)
    proj_params = set()
    for name, param in model.named_parameters():
        if "proj" in name and param.ndim == 2:
            proj_params.add(param)

    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Build param identity map (sharded param → original param)
    fsdp_param_is_proj = {}
    for module in model.modules():
        state = getattr(module, '_get_fsdp_state', lambda: None)()
        if state is None or state._fsdp_param_group is None:
            continue
        for fp in state._fsdp_param_group.fsdp_params:
            # Check if this param was originally a proj param by size/shape
            is_proj = (len(fp._orig_size) == 2
                       and fp._orig_size.numel() >= 1024 * 1024
                       and fp._orig_size[0] <= 20000)  # exclude lm_head/embed
            fsdp_param_is_proj[id(fp)] = is_proj

    def step():
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()

        for module in model.modules():
            state = getattr(module, '_get_fsdp_state', lambda: None)()
            if state is None or state._fsdp_param_group is None:
                continue
            for fp in state._fsdp_param_group.fsdp_params:
                if fp.sharded_param.grad is None:
                    continue
                sharded_grad = fp.sharded_param.grad._local_tensor

                if fsdp_param_is_proj.get(id(fp), False):
                    # Proj param: all-gather full grad + redundant NS
                    grad_list = [torch.zeros_like(sharded_grad) for _ in range(world_size)]
                    dist.all_gather(grad_list, sharded_grad)
                    full_grad = torch.cat(grad_list, dim=0)[:fp._orig_size[0]]
                    G = full_grad.view(fp._orig_size[0], -1).to(torch.bfloat16)
                    update = newton_schulz(G)
                    shard_size = sharded_grad.shape[0]
                    shard_start = local_rank * shard_size
                    local_update = update[shard_start:shard_start + shard_size]
                    fp.sharded_param._local_tensor.add_(
                        local_update.view(sharded_grad.shape).to(sharded_grad.dtype), alpha=-0.01)
                    del grad_list, full_grad, G, update, local_update
                else:
                    # Non-proj: SGD
                    fp.sharded_param._local_tensor.add_(sharded_grad, alpha=-0.01)
                fp.sharded_param.grad = None

    baseline_ms = bench_fn(step)

    if rank == 0:
        print(f"\n  Baseline (FSDP2 + redundant NS on all ranks):")
        print(f"    Step time:  {baseline_ms:.2f} ms")

    del model
    torch.cuda.empty_cache()
    return baseline_ms


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world_size,))

    target = sys.argv[1] if len(sys.argv) > 1 else "both"

    if rank == 0:
        print(f"LLM Benchmark: {world_size} × A800-80GB, bf16, seq=64, bs=1")

    if target in ("qwen", "both"):
        baseline_ms = run_baseline_bench(build_qwen25_7b, device, mesh, rank, world_size)
        dist.barrier(); torch.cuda.empty_cache()
        dmuon_ms = run_dmuon_bench(build_qwen25_7b, device, mesh, rank, world_size)
        if rank == 0:
            print(f"\n  Summary (Qwen2.5-7B):")
            speedup = baseline_ms / dmuon_ms
            print(f"    Baseline (redundant NS): {baseline_ms:.2f} ms")
            print(f"    DMuon (owner-only NS):   {dmuon_ms:.2f} ms")
            print(f"    Speedup:                 {speedup:.2f}x")

    dist.barrier(); torch.cuda.empty_cache()

    if target in ("llama", "both"):
        baseline_ms = run_baseline_bench(build_llama_8b, device, mesh, rank, world_size)
        dist.barrier(); torch.cuda.empty_cache()
        dmuon_ms = run_dmuon_bench(build_llama_8b, device, mesh, rank, world_size)
        if rank == 0:
            print(f"\n  Summary (Llama-3.1-8B):")
            speedup = baseline_ms / dmuon_ms
            print(f"    Baseline (redundant NS): {baseline_ms:.2f} ms")
            print(f"    DMuon (owner-only NS):   {dmuon_ms:.2f} ms")
            print(f"    Speedup:                 {speedup:.2f}x")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
