"""Optimizer benchmark: phase timing and detailed step breakdown.

Subcommands:
  phase   — DDP+Muon vs DMuon forward/backward/optimizer phase comparison
  detail  — Detailed DMuon optimizer step breakdown (wait/momentum/NS/update/adamw)

Run:
  torchrun --nproc_per_node=N benchmarks/bench_optimizer.py phase
  torchrun --nproc_per_node=N benchmarks/bench_optimizer.py detail
  torchrun --nproc_per_node=N benchmarks/bench_optimizer.py           # both
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP

from dmuon import Muon, dedicate_params, get_ns_backend, prepare_muon_grads
from dmuon.utils import get_owned_params

# ── Shared config ─────────────────────────────────────────────────────────────

SEQ_LEN, BATCH_SIZE = 2048, 2


def make_llama_config():
    from transformers import LlamaConfig
    return LlamaConfig(
        hidden_size=3072, intermediate_size=8192, num_hidden_layers=28,
        num_attention_heads=24, num_key_value_heads=8, vocab_size=128256,
        max_position_embeddings=4096, use_cache=False, tie_word_embeddings=False,
    )


def newton_schulz_baseline(G, steps=5):
    """Baseline NS used in DDP+Muon (every rank, redundant).

    Transposes when m > n so NS operates on the smaller (n, n) Gram space,
    matching standard Muon practice.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    X = X.to(torch.bfloat16)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


# ── Subcommand: phase ────────────────────────────────────────────────────────

def bench_phases(name, model_fn, rank, warmup=3, repeat=5):
    fwd_times, bwd_times, opt_times = [], [], []

    for step_i in range(warmup + repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fwd_out = model_fn["forward"]()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        model_fn["backward"](fwd_out)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        model_fn["optimizer"]()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        if step_i >= warmup:
            fwd_times.append((t1 - t0) * 1000)
            bwd_times.append((t2 - t1) * 1000)
            opt_times.append((t3 - t2) * 1000)

    fwd_times.sort()
    bwd_times.sort()
    opt_times.sort()
    mid = len(fwd_times) // 2
    fwd, bwd, opt = fwd_times[mid], bwd_times[mid], opt_times[mid]
    total = fwd + bwd + opt

    if rank == 0:
        print(f"  [{name}]")
        print(f"    Forward:   {fwd:>8.1f} ms")
        print(f"    Backward:  {bwd:>8.1f} ms")
        print(f"    Optimizer: {opt:>8.1f} ms")
        print(f"    Total:     {total:>8.1f} ms")

    return total


def run_phase(rank, world_size, device, mesh, mp_policy):
    from transformers import LlamaForCausalLM
    config = make_llama_config()
    input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)

    if rank == 0:
        print("=" * 60)
        print("PHASE BENCHMARK: DDP+Muon vs DMuon")
        print("=" * 60)

    # DDP + Muon baseline
    torch.manual_seed(42)
    with device:
        model_ddp = LlamaForCausalLM(config).to(torch.bfloat16)
    proj_params = {p for n, p in model_ddp.named_parameters() if "proj" in n and p.ndim == 2}
    model_ddp_wrapped = DDP(model_ddp, device_ids=[rank])

    def ddp_fwd(model=model_ddp_wrapped):
        return model(input_ids=input_ids, labels=labels)

    def ddp_bwd(out):
        out.loss.backward()

    def ddp_opt(model=model_ddp_wrapped):
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

    ddp_total = bench_phases(
        "DDP + Muon (baseline)",
        {"forward": ddp_fwd, "backward": ddp_bwd, "optimizer": ddp_opt},
        rank,
    )
    del model_ddp_wrapped
    del model_ddp
    torch.cuda.empty_cache()
    dist.barrier()

    # DMuon
    torch.manual_seed(42)
    with device:
        model_dmuon = LlamaForCausalLM(config)
    dedicate_params(model_dmuon, mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
        compute_dtype=torch.bfloat16, reshard_after_forward=False)
    for layer in model_dmuon.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model_dmuon, mesh=mesh, mp_policy=mp_policy)
    optimizer = Muon(model_dmuon, lr=0.01, ns_steps=5, adamw_lr=0.01)

    def dmuon_fwd():
        return model_dmuon(input_ids=input_ids, labels=labels)

    def dmuon_bwd(out):
        out.loss.backward()

    def dmuon_opt():
        optimizer.step()
        optimizer.zero_grad()

    dmuon_total = bench_phases(
        "DMuon (CuteDSL SYRK)",
        {"forward": dmuon_fwd, "backward": dmuon_bwd, "optimizer": dmuon_opt},
        rank,
    )

    if rank == 0:
        print(f"\n  Speedup: {ddp_total / dmuon_total:.2f}x")


# ── Subcommand: detail ───────────────────────────────────────────────────────

def run_detail(rank, world_size, device, mesh, mp_policy):
    from transformers import LlamaForCausalLM
    config = make_llama_config()

    torch.manual_seed(42)
    with device:
        model = LlamaForCausalLM(config)

    dedicate_params(model, mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
        compute_dtype=torch.bfloat16, reshard_after_forward=False)
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    owned = get_owned_params(model, mesh.get_local_rank())
    input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)

    ns = sys.modules["dmuon.optim.newton_schulz"]

    if rank == 0:
        print("=" * 60)
        print("DETAIL BENCHMARK: DMuon Optimizer Step Breakdown")
        print("=" * 60)
        print(f"Owned params: {len(owned)}")
        shapes = {}
        for dp in owned:
            s = tuple(dp._owned_data.shape)
            shapes[s] = shapes.get(s, 0) + 1
        for s, cnt in sorted(shapes.items()):
            print(f"  {s}: {cnt} params")

    # Warmup
    optimizer = Muon(model, lr=0.01, ns_steps=5, adamw_lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()

    # Detailed timing of one optimizer step
    optimizer.zero_grad()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()

    torch.cuda.synchronize()

    # Phase 1: prepare_muon_grads (wait reduce tails + TP gather when present)
    t0 = time.perf_counter()
    prepare_muon_grads(model)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Phase 2: Muon NS per-param
    ns_times = []
    momentum_times = []
    update_times = []

    for dp in owned:
        if dp._reduced_grad is None:
            continue

        grad = dp._reduced_grad.view(dp._reduced_grad.shape[0], -1)

        # Momentum
        torch.cuda.synchronize()
        tm0 = time.perf_counter()
        dp_id = id(dp)
        state = optimizer.state.get(dp_id, {})
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = grad.clone()
            optimizer.state[dp_id] = state
        else:
            state["momentum_buffer"].mul_(0.95).add_(grad)
        buf = state["momentum_buffer"]
        torch.cuda.synchronize()
        tm1 = time.perf_counter()
        momentum_times.append((tm1 - tm0) * 1e6)

        # Newton-Schulz
        torch.cuda.synchronize()
        tn0 = time.perf_counter()
        update = ns.gram_newton_schulz_local(buf)
        torch.cuda.synchronize()
        tn1 = time.perf_counter()
        ns_times.append((tn1 - tn0) * 1e6)

        # Param update
        torch.cuda.synchronize()
        tu0 = time.perf_counter()
        owned_data = dp._owned_data
        m = owned_data.shape[0]
        n = owned_data.view(m, -1).shape[1]
        scale = 0.2 * (max(m, n) ** 0.5)
        owned_data.add_(update.view(owned_data.shape).to(owned_data.dtype), alpha=-0.01 * scale)
        dp._reduced_grad = None
        torch.cuda.synchronize()
        tu1 = time.perf_counter()
        update_times.append((tu1 - tu0) * 1e6)

    # Phase 3: AdamW
    torch.cuda.synchronize()
    ta0 = time.perf_counter()
    optimizer._step_adamw()
    torch.cuda.synchronize()
    ta1 = time.perf_counter()

    if rank == 0:
        wait_ms = (t1 - t0) * 1000
        ns_total = sum(ns_times) / 1000
        mom_total = sum(momentum_times) / 1000
        upd_total = sum(update_times) / 1000
        adamw_ms = (ta1 - ta0) * 1000
        total = wait_ms + ns_total + mom_total + upd_total + adamw_ms

        print("\n=== Optimizer Step Breakdown ===")
        print(f"  prepare_muon_grads:{wait_ms:>8.1f} ms")
        print(f"  momentum:          {mom_total:>8.1f} ms  ({len(momentum_times)} params)")
        print(f"  newton_schulz:     {ns_total:>8.1f} ms  ({len(ns_times)} params)")
        print(f"  param_update:      {upd_total:>8.1f} ms")
        print(f"  adamw:             {adamw_ms:>8.1f} ms")
        print(f"  TOTAL:             {total:>8.1f} ms")

        print("\n=== NS Time by Shape ===")
        shape_ns = {}
        for dp, t in zip(owned, ns_times):
            s = tuple(dp._owned_data.shape)
            if s not in shape_ns:
                shape_ns[s] = []
            shape_ns[s].append(t)
        for s in sorted(shape_ns.keys()):
            times = shape_ns[s]
            avg = sum(times) / len(times)
            print(f"  {s}: {avg:.0f} us avg x {len(times)} = {sum(times)/1000:.1f} ms")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world_size,))
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)

    if rank == 0:
        print(f"Optimizer Benchmark: {world_size} GPUs, seq={SEQ_LEN}, bs={BATCH_SIZE}")
        print(f"NS backend: {get_ns_backend()}\n")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"

    if cmd in ("phase", "all"):
        run_phase(rank, world_size, device, mesh, mp_policy)
        if cmd == "all":
            dist.barrier()
            if rank == 0:
                print()

    if cmd in ("detail", "all"):
        run_detail(rank, world_size, device, mesh, mp_policy)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
