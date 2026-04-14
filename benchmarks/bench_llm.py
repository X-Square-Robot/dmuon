"""Unified LLM benchmark: DMuon vs baselines with phase breakdown.

Baselines:
  - FSDP2 + AdamW (standard training, no NS)
  - DDP + Muon (small models only, redundant NS on every rank)
  - FSDP2 + Muon (all-gather grad + redundant NS on every rank)
  - DMuon (dedicated ownership, owner-only NS)

Output: Forward / Backward / Optimizer / Total for each method.

Run:
  torchrun --nproc_per_node=8 benchmarks/bench_llm.py             # all models
  torchrun --nproc_per_node=8 benchmarks/bench_llm.py 1b          # Qwen-1.5B only
  torchrun --nproc_per_node=8 benchmarks/bench_llm.py 3b          # Llama-3B only
  torchrun --nproc_per_node=8 benchmarks/bench_llm.py 7b          # Qwen-7B only
  torchrun --nproc_per_node=8 benchmarks/bench_llm.py 8b          # Llama-8B only
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.nn.parallel import DistributedDataParallel as DDP

from dmuon import Muon, dedicate_params, wait_all_reduces, get_ns_backend
from dmuon.utils import get_owned_params

SEQ_LEN = 2048
BATCH_SIZE = 2
MP_POLICY = None  # set in main


# ── Model builders ────────────────────────────────────────────────────────────

def build_qwen_1b(device):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    config = Qwen2Config(
        hidden_size=1536, intermediate_size=8960, num_hidden_layers=28,
        num_attention_heads=12, num_key_value_heads=2, vocab_size=151936,
        max_position_embeddings=4096, use_cache=False, tie_word_embeddings=True,
    )
    with device:
        model = Qwen2ForCausalLM(config)
    return model, "Qwen2.5-1.5B"


def build_llama_3b(device):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(
        hidden_size=3072, intermediate_size=8192, num_hidden_layers=28,
        num_attention_heads=24, num_key_value_heads=8, vocab_size=128256,
        max_position_embeddings=4096, use_cache=False, tie_word_embeddings=False,
    )
    with device:
        model = LlamaForCausalLM(config)
    return model, "Llama-3.2-3B"


def build_qwen_7b(device):
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


# ── Baseline NS ───────────────────────────────────────────────────────────────

def newton_schulz_baseline(G, steps=5):
    """Baseline NS used in DDP+Muon / FSDP2+Muon (redundant on every rank)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    X = X.to(torch.bfloat16)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X


# ── Benchmark utility ─────────────────────────────────────────────────────────

def bench_phases(step_fn, warmup=3, repeat=8):
    """Benchmark a step function that returns (fwd_ms, bwd_ms, opt_ms).
    Returns median of each phase."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()

    fwd_list, bwd_list, opt_list = [], [], []
    for _ in range(repeat):
        fwd, bwd, opt = step_fn()
        fwd_list.append(fwd)
        bwd_list.append(bwd)
        opt_list.append(opt)

    fwd_list.sort(); bwd_list.sort(); opt_list.sort()
    mid = len(fwd_list) // 2
    return fwd_list[mid], bwd_list[mid], opt_list[mid]


# ── Method: FSDP2 + AdamW ────────────────────────────────────────────────────

def run_fsdp_adamw(build_fn, device, mesh, rank, world_size):
    torch.manual_seed(42)
    model, name = build_fn(device)

    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=MP_POLICY)
    fully_shard(model, mesh=mesh, mp_policy=MP_POLICY)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)

    def step():
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, labels=labels)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out.loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000

    fwd, bwd, opt = bench_phases(step)
    peak = torch.cuda.max_memory_allocated() / 1e9

    del model, optimizer
    torch.cuda.empty_cache()
    return fwd, bwd, opt, peak


# ── Method: DDP + Muon ────────────────────────────────────────────────────────

def run_ddp_muon(build_fn, device, mesh, rank, world_size):
    torch.manual_seed(42)
    model, name = build_fn(device)
    model = model.to(torch.bfloat16)
    proj_params = {p for n, p in model.named_parameters() if "proj" in n and p.ndim == 2}
    model = DDP(model, device_ids=[rank])

    input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)

    def step():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, labels=labels)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out.loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        for p in model.parameters():
            if p.grad is None:
                continue
            if p in proj_params:
                G = p.grad.view(p.grad.shape[0], -1).to(torch.bfloat16)
                update = newton_schulz_baseline(G)
                p.data.add_(update.view(p.shape).to(p.dtype), alpha=-0.01)
            else:
                p.data.add_(p.grad, alpha=-0.01)
            p.grad = None
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000

    fwd, bwd, opt = bench_phases(step)
    peak = torch.cuda.max_memory_allocated() / 1e9

    del model
    torch.cuda.empty_cache()
    return fwd, bwd, opt, peak


# ── Method: FSDP2 + Muon (all-gather grad + redundant NS) ────────────────────

def run_fsdp_muon(build_fn, device, mesh, rank, world_size):
    local_rank = mesh.get_local_rank()
    torch.manual_seed(42)
    model, name = build_fn(device)

    # Identify proj params before FSDP
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=MP_POLICY)
    fully_shard(model, mesh=mesh, mp_policy=MP_POLICY)

    # Build proj param map
    fsdp_param_is_proj = {}
    for module in model.modules():
        state = getattr(module, '_get_fsdp_state', lambda: None)()
        if state is None or state._fsdp_param_group is None:
            continue
        for fp in state._fsdp_param_group.fsdp_params:
            is_proj = (len(fp._orig_size) == 2
                       and fp._orig_size.numel() >= 1024 * 1024
                       and fp._orig_size[0] <= 20000)
            fsdp_param_is_proj[id(fp)] = is_proj

    input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)

    def step():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, labels=labels)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out.loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        for module in model.modules():
            state = getattr(module, '_get_fsdp_state', lambda: None)()
            if state is None or state._fsdp_param_group is None:
                continue
            for fp in state._fsdp_param_group.fsdp_params:
                if fp.sharded_param.grad is None:
                    continue
                sg = fp.sharded_param.grad._local_tensor

                if fsdp_param_is_proj.get(id(fp), False):
                    gl = [torch.zeros_like(sg) for _ in range(world_size)]
                    dist.all_gather(gl, sg)
                    fg = torch.cat(gl, dim=0)[:fp._orig_size[0]]
                    G = fg.view(fp._orig_size[0], -1).to(torch.bfloat16)
                    upd = newton_schulz_baseline(G)
                    ss = sg.shape[0]
                    lu = upd[local_rank * ss:(local_rank + 1) * ss]
                    fp.sharded_param._local_tensor.add_(
                        lu.view(sg.shape).to(sg.dtype), alpha=-0.01)
                    del gl, fg, G, upd, lu
                else:
                    fp.sharded_param._local_tensor.add_(sg, alpha=-0.01)
                fp.sharded_param.grad = None

        torch.cuda.synchronize()
        t3 = time.perf_counter()
        return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000

    fwd, bwd, opt = bench_phases(step)
    peak = torch.cuda.max_memory_allocated() / 1e9

    del model
    torch.cuda.empty_cache()
    return fwd, bwd, opt, peak


# ── Method: DMuon ─────────────────────────────────────────────────────────────

def run_dmuon(build_fn, device, mesh, rank, world_size):
    torch.manual_seed(42)
    model, name = build_fn(device)

    dedicate_params(
        model, mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
        compute_dtype=torch.bfloat16,
        reshard_after_forward=False,
    )
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=MP_POLICY)
    fully_shard(model, mesh=mesh, mp_policy=MP_POLICY)

    optimizer = Muon(model, lr=0.01, ns_steps=5, adamw_lr=0.01)

    input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)

    def step():
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, labels=labels)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out.loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000

    fwd, bwd, opt = bench_phases(step)
    peak = torch.cuda.max_memory_allocated() / 1e9

    # Correctness check
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.item())
    loss_t = torch.tensor(losses, device=device)
    all_losses = [torch.zeros_like(loss_t) for _ in range(world_size)]
    dist.all_gather(all_losses, loss_t)

    max_diff = 0
    if rank == 0:
        for s in range(3):
            vals = [all_losses[r][s].item() for r in range(world_size)]
            max_diff = max(max_diff, max(vals) - min(vals))

    del model, optimizer
    torch.cuda.empty_cache()
    return fwd, bwd, opt, peak, max_diff


# ── Benchmark driver ──────────────────────────────────────────────────────────

def bench_model(build_fn, device, mesh, rank, world_size, include_ddp=False):
    torch.manual_seed(42)
    m, name = build_fn(device)
    total = sum(p.numel() for p in m.parameters())
    del m; torch.cuda.empty_cache()

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"  {name} ({total/1e9:.2f}B params) — {world_size} GPUs, seq={SEQ_LEN}, bs={BATCH_SIZE}")
        print(f"{'='*80}")

    results = {}

    # 1. FSDP2 + AdamW
    dist.barrier(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fwd, bwd, opt, peak = run_fsdp_adamw(build_fn, device, mesh, rank, world_size)
    results["FSDP2+AdamW"] = (fwd, bwd, opt, peak)
    dist.barrier()

    # 2. DDP + Muon (small models only)
    if include_ddp:
        dist.barrier(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        fwd, bwd, opt, peak = run_ddp_muon(build_fn, device, mesh, rank, world_size)
        results["DDP+Muon"] = (fwd, bwd, opt, peak)
        dist.barrier()

    # 3. FSDP2 + Muon
    dist.barrier(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fwd, bwd, opt, peak = run_fsdp_muon(build_fn, device, mesh, rank, world_size)
    results["FSDP2+Muon"] = (fwd, bwd, opt, peak)
    dist.barrier()

    # 4. DMuon
    dist.barrier(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fwd, bwd, opt, peak, max_diff = run_dmuon(build_fn, device, mesh, rank, world_size)
    results["DMuon"] = (fwd, bwd, opt, peak)
    dist.barrier()

    # Print results
    if rank == 0:
        print(f"\n  {'Method':<18s} {'Forward':>10s} {'Backward':>10s} {'Optimizer':>10s} {'Total':>10s} {'Mem':>8s}")
        print(f"  {'-'*68}")
        for method, (f, b, o, p) in results.items():
            total_ms = f + b + o
            print(f"  {method:<18s} {f:>8.1f}ms {b:>8.1f}ms {o:>8.1f}ms {total_ms:>8.1f}ms {p:>6.1f}GB")

        # Speedup summary
        adamw_total = sum(results["FSDP2+AdamW"][:3])
        dmuon_total = sum(results["DMuon"][:3])
        print(f"\n  Speedup vs FSDP2+AdamW:")
        for method, (f, b, o, p) in results.items():
            total_ms = f + b + o
            spd = adamw_total / total_ms if total_ms > 0 else 0
            print(f"    {method:<18s} {spd:.2f}x")

        # Optimizer-only speedup
        adamw_opt = results["FSDP2+AdamW"][2]
        print(f"\n  Optimizer-only time:")
        for method, (f, b, o, p) in results.items():
            spd = adamw_opt / o if o > 0 else 0
            print(f"    {method:<18s} {o:>8.1f}ms  ({spd:.2f}x vs AdamW)")

        if max_diff < 0.01:
            print(f"\n  DMuon loss consistent across ranks (diff={max_diff:.6f})")
        else:
            print(f"\n  DMuon loss mismatch: {max_diff:.6f}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

MODELS = {
    "1b": (build_qwen_1b, True),   # (builder, include_ddp)
    "3b": (build_llama_3b, True),
    "7b": (build_qwen_7b, False),
    "8b": (build_llama_8b, False),
}


def main():
    global MP_POLICY
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world_size,))
    MP_POLICY = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)

    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if rank == 0:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"LLM Benchmark: {world_size} x {gpu_name}, bf16")
        print(f"NS backend: {get_ns_backend()}")

    all_results = {}

    if target == "all":
        targets = ["1b", "3b", "7b", "8b"]
    else:
        targets = [target]

    for t in targets:
        if t not in MODELS:
            if rank == 0:
                print(f"Unknown model: {t}, choices: {list(MODELS.keys())}")
            continue
        build_fn, include_ddp = MODELS[t]
        results = bench_model(build_fn, device, mesh, rank, world_size, include_ddp)
        all_results[t] = results
        dist.barrier(); torch.cuda.empty_cache()

    # Final summary table
    if rank == 0 and len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"  SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Model':<16s} {'Method':<18s} {'Fwd':>8s} {'Bwd':>8s} {'Opt':>8s} {'Total':>8s} {'Mem':>7s}")
        print(f"  {'-'*75}")
        for model_key, results in all_results.items():
            build_fn, _ = MODELS[model_key]
            for method, (f, b, o, p) in results.items():
                total_ms = f + b + o
                print(f"  {model_key.upper():<16s} {method:<18s} {f:>6.1f}ms {b:>6.1f}ms {o:>6.1f}ms {total_ms:>6.1f}ms {p:>5.1f}GB")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
