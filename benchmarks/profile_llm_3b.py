"""Profile Llama-3.2-3B (Qwen 3B class) with torch profiler.

Generates chrome trace JSON files for three approaches:
  1. DDP + Muon
  2. FSDP2 + redundant NS
  3. DMuon (dedicated ownership)

Run: torchrun --nproc_per_node=8 benchmarks/profile_llm_3b.py [ddp|fsdp|dmuon|all]

Output: profile_3b_{approach}_rank{rank}.json in the benchmarks/ directory.
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, schedule

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmuon import dedicate_params, wait_all_reduces
from dmuon.utils import get_owned_params

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")


def newton_schulz(G, steps=5):
    X = G / (G.norm() + 1e-7)
    X = X.to(torch.bfloat16)
    for _ in range(steps):
        A = X @ X.T
        B = (1.5 * A - 0.5 * A @ A).to(X.dtype)
        X = B @ X
    return X


def build_llama_3b(device):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(
        hidden_size=3072, intermediate_size=8192, num_hidden_layers=28,
        num_attention_heads=24, num_key_value_heads=8, vocab_size=128256,
        max_position_embeddings=4096, use_cache=False, tie_word_embeddings=False,
    )
    with device:
        model = LlamaForCausalLM(config)
    return model


# ---- DDP + Muon ----

def profile_ddp_muon(device, mesh, rank, world_size):
    seq_len, batch_size = 2048, 2

    torch.manual_seed(42)
    model = build_llama_3b(device).to(torch.bfloat16)
    proj_params = {p for n, p in model.named_parameters() if "proj" in n and p.ndim == 2}
    model = DDP(model, device_ids=[rank])

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    def step():
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
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

    # Warmup
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    trace_path = os.path.join(OUTDIR, f"profile_3b_ddp_rank{rank}.json")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        step()
        torch.cuda.synchronize()

    if rank == 0:
        prof.export_chrome_trace(trace_path)
        print(f"  [DDP + Muon] trace saved: {trace_path}")

    del model
    torch.cuda.empty_cache()


# ---- FSDP2 + redundant NS ----

def profile_fsdp_redundant(device, mesh, rank, world_size):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    seq_len, batch_size = 2048, 2
    local_rank = mesh.get_local_rank()

    torch.manual_seed(42)
    model = build_llama_3b(device)

    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

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

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)

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

    # Warmup
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    trace_path = os.path.join(OUTDIR, f"profile_3b_fsdp_rank{rank}.json")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        step()
        torch.cuda.synchronize()

    if rank == 0:
        prof.export_chrome_trace(trace_path)
        print(f"  [FSDP2 + redundant NS] trace saved: {trace_path}")

    del model
    torch.cuda.empty_cache()


# ---- DMuon ----

def profile_dmuon(device, mesh, rank, world_size):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    seq_len, batch_size = 2048, 2

    torch.manual_seed(42)
    model = build_llama_3b(device)

    dedicate_params(
        model, mesh,
        predicate=lambda name, param: "proj" in name and param.ndim == 2,
        compute_dtype=torch.bfloat16,
        reshard_after_forward=False,
    )
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    owned = get_owned_params(model, mesh.get_local_rank())

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    def step():
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        wait_all_reduces(model)
        for dp in owned:
            if dp._reduced_grad is not None:
                G = dp._reduced_grad.view(dp._reduced_grad.shape[0], -1).to(torch.bfloat16)
                update = newton_schulz(G)
                dp._owned_data.add_(update.view(dp._owned_data.shape).to(dp._owned_data.dtype), alpha=-0.01)
                dp._reduced_grad = None
        for module in model.modules():
            state = getattr(module, '_get_fsdp_state', lambda: None)()
            if state is None or state._fsdp_param_group is None:
                continue
            for fp in state._fsdp_param_group.fsdp_params:
                if fp.sharded_param.grad is not None:
                    fp.sharded_param._local_tensor.add_(
                        fp.sharded_param.grad._local_tensor, alpha=-0.01)
                    fp.sharded_param.grad = None

    # Warmup
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    trace_path = os.path.join(OUTDIR, f"profile_3b_dmuon_rank{rank}.json")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        step()
        torch.cuda.synchronize()

    if rank == 0:
        prof.export_chrome_trace(trace_path)
        print(f"  [DMuon] trace saved: {trace_path}")

    del model
    torch.cuda.empty_cache()


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world_size,))

    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    os.makedirs(OUTDIR, exist_ok=True)

    if rank == 0:
        print(f"Profiling Llama-3.2-3B: {world_size} GPUs, target={target}")
        print(f"Output dir: {OUTDIR}")

    if target in ("ddp", "all"):
        if rank == 0:
            print("\n--- Profiling DDP + Muon ---")
        profile_ddp_muon(device, mesh, rank, world_size)
        dist.barrier()
        torch.cuda.empty_cache()

    if target in ("fsdp", "all"):
        if rank == 0:
            print("\n--- Profiling FSDP2 + redundant NS ---")
        profile_fsdp_redundant(device, mesh, rank, world_size)
        dist.barrier()
        torch.cuda.empty_cache()

    if target in ("dmuon", "all"):
        if rank == 0:
            print("\n--- Profiling DMuon ---")
        profile_dmuon(device, mesh, rank, world_size)
        dist.barrier()
        torch.cuda.empty_cache()

    if rank == 0:
        print("\nDone! Trace files:")
        for f in sorted(os.listdir(OUTDIR)):
            if f.startswith("profile_3b_"):
                print(f"  {os.path.join(OUTDIR, f)}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
