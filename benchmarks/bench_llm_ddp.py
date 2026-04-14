"""LLM benchmark: DMuon (FSDP2) vs DDP + Muon.

DDP keeps full model on each GPU (no sharding), so no all-gather needed
for NS — each rank has the full gradient. This is the "ideal" baseline
for optimizer computation but uses more memory.

We compare:
  A. DDP + Muon: every rank has full params+grads, runs NS redundantly
  B. DMuon (FSDP2): dedicated ownership, owner-only NS, broadcast/reduce

Uses smaller models (1B, 3B) that fit in single GPU memory for DDP.

Run: torchrun --nproc_per_node=8 benchmarks/bench_llm_ddp.py [1b|3b|both]
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmuon import Muon, dedicate_params, wait_all_reduces
from dmuon.utils import get_owned_params


def newton_schulz(G, steps=5):
    X = G / (G.norm() + 1e-7)
    X = X.to(torch.bfloat16)
    for _ in range(steps):
        A = X @ X.T
        B = (1.5 * A - 0.5 * A @ A).to(X.dtype)
        X = B @ X
    return X


# ---- Model builders ----

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


def bench_fn(fn, warmup=3, repeat=8):
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


# ---- DDP + Muon baseline ----

def run_ddp_muon(build_fn, device, mesh, rank, world_size):
    """DDP: every rank has full model, all-reduce grads, then NS locally."""
    seq_len = 2048
    batch_size = 2

    torch.manual_seed(42)
    model, model_name = build_fn(device)
    model = model.to(torch.bfloat16)
    total_params = sum(p.numel() for p in model.parameters())

    if rank == 0:
        print(f"\n  [DDP + Muon] {model_name} — {total_params/1e9:.2f}B params")

    # Identify proj params
    proj_params = {p for n, p in model.named_parameters() if "proj" in n and p.ndim == 2}

    model = DDP(model, device_ids=[rank])

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    def step():
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        # DDP already all-reduced gradients. Each rank has full averaged grad.
        # Run NS on proj params (redundant but no extra communication)
        for p in model.parameters():
            if p.grad is None:
                continue
            if p in proj_params:
                G = p.grad.view(p.grad.shape[0], -1).to(torch.bfloat16)
                update = newton_schulz(G)
                p.data.add_(update.view(p.shape).to(p.dtype), alpha=-0.01)
            else:
                p.data.add_(p.grad, alpha=-0.01)
            p.grad = None
        return out.loss.item()

    ddp_ms = bench_fn(step)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    if rank == 0:
        print(f"    Step time:        {ddp_ms:.2f} ms")
        print(f"    Peak GPU memory:  {peak_mem:.1f} GB")

    del model
    torch.cuda.empty_cache()
    return ddp_ms


# ---- DMuon (FSDP2) ----

def run_dmuon(build_fn, device, mesh, rank, world_size):
    """DMuon: dedicated ownership, owner-only NS."""
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    seq_len = 2048
    batch_size = 2

    torch.manual_seed(42)
    model, model_name = build_fn(device)
    total_params = sum(p.numel() for p in model.parameters())

    if rank == 0:
        print(f"\n  [DMuon] {model_name} — {total_params/1e9:.2f}B params")

    dedicate_params(
        model, mesh,
        predicate=lambda name, param: "proj" in name and param.ndim == 2,
        compute_dtype=torch.bfloat16,
        reshard_after_forward=False,
    )
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    optimizer = Muon(model, lr=0.01, ns_steps=5, adamw_lr=0.01)

    if rank == 0:
        n_dedicated = sum(1 for m in model.modules() if hasattr(m, '_dedicated_state')
                          for _ in m._dedicated_state.group.params)
        print(f"    Dedicated params: {n_dedicated}, rank 0 owns: {len(optimizer._dedicated_params)}")

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    def step():
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()
        return out.loss.item()

    dmuon_ms = bench_fn(step)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Correctness check
    losses = [step() for _ in range(3)]
    loss_t = torch.tensor(losses, device=device)
    all_losses = [torch.zeros_like(loss_t) for _ in range(world_size)]
    dist.all_gather(all_losses, loss_t)

    if rank == 0:
        max_diff = max(
            max(all_losses[r][s].item() for r in range(world_size)) -
            min(all_losses[r][s].item() for r in range(world_size))
            for s in range(3)
        )
        print(f"    Step time:        {dmuon_ms:.2f} ms")
        print(f"    Peak GPU memory:  {peak_mem:.1f} GB")
        if max_diff < 0.01:
            print(f"    ✅ Loss consistent across ranks (diff={max_diff:.6f})")
        else:
            print(f"    ⚠️  Loss mismatch: {max_diff:.6f}")

    del model
    torch.cuda.empty_cache()
    return dmuon_ms


# ---- FSDP2 + redundant NS baseline ----

def run_fsdp_redundant(build_fn, device, mesh, rank, world_size):
    """FSDP2 + redundant NS: all-gather grad per param, every rank runs NS."""
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    seq_len = 2048
    batch_size = 2
    local_rank = mesh.get_local_rank()

    torch.manual_seed(42)
    model, model_name = build_fn(device)
    total_params = sum(p.numel() for p in model.parameters())

    if rank == 0:
        print(f"\n  [FSDP2 + redundant NS] {model_name} — {total_params/1e9:.2f}B params")

    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Identify proj params by shape (skip vocab-sized params)
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
                sg = fp.sharded_param.grad._local_tensor
                if fsdp_param_is_proj.get(id(fp), False):
                    gl = [torch.zeros_like(sg) for _ in range(world_size)]
                    dist.all_gather(gl, sg)
                    fg = torch.cat(gl, dim=0)[:fp._orig_size[0]]
                    G = fg.view(fp._orig_size[0], -1).to(torch.bfloat16)
                    upd = newton_schulz(G)
                    ss = sg.shape[0]
                    lu = upd[local_rank * ss:(local_rank + 1) * ss]
                    fp.sharded_param._local_tensor.add_(lu.view(sg.shape).to(sg.dtype), alpha=-0.01)
                    del gl, fg, G, upd, lu
                else:
                    fp.sharded_param._local_tensor.add_(sg, alpha=-0.01)
                fp.sharded_param.grad = None

    fsdp_ms = bench_fn(step)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    if rank == 0:
        print(f"    Step time:        {fsdp_ms:.2f} ms")
        print(f"    Peak GPU memory:  {peak_mem:.1f} GB")

    del model
    torch.cuda.empty_cache()
    return fsdp_ms


def bench_model(build_fn, device, mesh, rank, world_size):
    torch.manual_seed(42)
    m, name = build_fn(device)
    total = sum(p.numel() for p in m.parameters())
    del m; torch.cuda.empty_cache()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  {name} ({total/1e9:.2f}B params)")
        print(f"{'='*60}")

    # 1. DDP + Muon
    ddp_ms = run_ddp_muon(build_fn, device, mesh, rank, world_size)
    dist.barrier(); torch.cuda.empty_cache()

    # 2. FSDP2 + redundant NS
    fsdp_ms = run_fsdp_redundant(build_fn, device, mesh, rank, world_size)
    dist.barrier(); torch.cuda.empty_cache()

    # 3. DMuon
    dmuon_ms = run_dmuon(build_fn, device, mesh, rank, world_size)
    dist.barrier(); torch.cuda.empty_cache()

    if rank == 0:
        print(f"\n  {'─'*50}")
        print(f"  Summary:")
        print(f"    DDP + redundant Muon:    {ddp_ms:>8.2f} ms")
        print(f"    FSDP2 + redundant NS:    {fsdp_ms:>8.2f} ms")
        print(f"    DMuon (owner-only NS):   {dmuon_ms:>8.2f} ms")
        print(f"    DMuon vs DDP:            {ddp_ms/dmuon_ms:.2f}x")
        print(f"    DMuon vs FSDP2+NS:       {fsdp_ms/dmuon_ms:.2f}x")


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world_size,))

    target = sys.argv[1] if len(sys.argv) > 1 else "both"

    if rank == 0:
        print(f"LLM DDP vs DMuon Benchmark: {world_size} × A800-80GB, bf16")

    if target in ("1b", "both"):
        bench_model(build_qwen_1b, device, mesh, rank, world_size)

    dist.barrier(); torch.cuda.empty_cache()

    if target in ("3b", "both"):
        bench_model(build_llama_3b, device, mesh, rank, world_size)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
