"""Benchmark DMuon TP on random-init HuggingFace Llama/Qwen models.

This harness reuses the model shapes from ``benchmarks/bench_llm.py`` but
adds a real TP setup:

    TP parallelize -> dmuon.dedicate_params(dp mesh) -> FSDP2(dp mesh)

By default all standard transformer projections are tensor-parallelized:
attention q/k/v/o plus MLP gate/up/down.  Set
``DMUON_TP_LLM_PARALLELIZE=mlp`` only for diagnostics or for model/topology
pairs whose attention head layout cannot be split by the requested TP size.

Examples:

    torchrun --nproc_per_node=4 benchmarks/bench_tp_llm.py llama3b tp4
    DMUON_TP_LLM_ASYNC=1 torchrun --nproc_per_node=2 benchmarks/bench_tp_llm.py qwen1b tp2
    DMUON_TP_LLM_LAYERS=2 DMUON_TP_LLM_SEQ=64 torchrun --nproc_per_node=2 benchmarks/bench_tp_llm.py llama3b tp2
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

REPO = Path(__file__).resolve().parents[1]
TEST_DIST = REPO / "tests" / "distributed"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TEST_DIST))

import dmuon  # noqa: E402
from dmuon.utils import wait_all_replicate_broadcasts  # noqa: E402
from tp_profile_utils import (  # noqa: E402
    collect_tp_profile,
    gather_tp_profiles,
    iter_dedicated_params,
)


@dataclass(frozen=True)
class ModelSpec:
    family: str
    label: str
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    vocab: int
    max_positions: int = 4096
    tie_word_embeddings: bool = False


MODEL_SPECS = {
    "qwen1b": ModelSpec(
        family="qwen2",
        label="Qwen2.5-1.5B",
        hidden=1536,
        intermediate=8960,
        layers=28,
        heads=12,
        kv_heads=2,
        vocab=151936,
        tie_word_embeddings=True,
    ),
    "llama3b": ModelSpec(
        family="llama",
        label="Llama-3.2-3B",
        hidden=3072,
        intermediate=8192,
        layers=28,
        heads=24,
        kv_heads=8,
        vocab=128256,
    ),
    "qwen7b": ModelSpec(
        family="qwen2",
        label="Qwen2.5-7B",
        hidden=3584,
        intermediate=18944,
        layers=28,
        heads=28,
        kv_heads=4,
        vocab=152064,
    ),
    "llama8b": ModelSpec(
        family="llama",
        label="Llama-3.1-8B",
        hidden=4096,
        intermediate=14336,
        layers=32,
        heads=32,
        kv_heads=8,
        vocab=128256,
    ),
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _resolved_spec(model_key: str) -> tuple[ModelSpec, dict[str, Any]]:
    if model_key not in MODEL_SPECS:
        raise ValueError(f"unknown model={model_key!r}; choices={sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[model_key]
    config = {
        "model_key": model_key,
        "model_label": spec.label,
        "family": spec.family,
        "hidden": _env_int("DMUON_TP_LLM_HIDDEN", spec.hidden),
        "intermediate": _env_int("DMUON_TP_LLM_INTER", spec.intermediate),
        "layers": _env_int("DMUON_TP_LLM_LAYERS", spec.layers),
        "heads": _env_int("DMUON_TP_LLM_HEADS", spec.heads),
        "kv_heads": _env_int("DMUON_TP_LLM_KV_HEADS", spec.kv_heads),
        "vocab": _env_int("DMUON_TP_LLM_VOCAB", spec.vocab),
        "max_positions": _env_int(
            "DMUON_TP_LLM_MAX_POSITIONS", spec.max_positions
        ),
        "tie_word_embeddings": bool(spec.tie_word_embeddings),
    }
    config["max_full_tp_size"] = math.gcd(
        math.gcd(config["hidden"], config["intermediate"]),
        math.gcd(config["heads"], config["kv_heads"]),
    )
    return spec, config


def build_model(model_key: str, device: torch.device):
    _spec, cfg = _resolved_spec(model_key)
    if cfg["family"] == "qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM

        hf_config = Qwen2Config(
            hidden_size=cfg["hidden"],
            intermediate_size=cfg["intermediate"],
            num_hidden_layers=cfg["layers"],
            num_attention_heads=cfg["heads"],
            num_key_value_heads=cfg["kv_heads"],
            vocab_size=cfg["vocab"],
            max_position_embeddings=cfg["max_positions"],
            use_cache=False,
            tie_word_embeddings=cfg["tie_word_embeddings"],
        )
        with device:
            return Qwen2ForCausalLM(hf_config), cfg

    if cfg["family"] == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        hf_config = LlamaConfig(
            hidden_size=cfg["hidden"],
            intermediate_size=cfg["intermediate"],
            num_hidden_layers=cfg["layers"],
            num_attention_heads=cfg["heads"],
            num_key_value_heads=cfg["kv_heads"],
            vocab_size=cfg["vocab"],
            max_position_embeddings=cfg["max_positions"],
            use_cache=False,
            tie_word_embeddings=cfg["tie_word_embeddings"],
        )
        with device:
            return LlamaForCausalLM(hf_config), cfg

    raise ValueError(f"unsupported family={cfg['family']!r}")


def _make_mesh(topology: str, world_size: int):
    if topology == "tp2":
        if world_size != 2:
            raise RuntimeError(f"tp2 needs world=2, got {world_size}")
        mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("dp", "tp"))
        return mesh, mesh["dp"], mesh["tp"], mesh["dp"], None
    if topology == "tp4":
        if world_size != 4:
            raise RuntimeError(f"tp4 needs world=4, got {world_size}")
        mesh = init_device_mesh("cuda", (1, 4), mesh_dim_names=("dp", "tp"))
        return mesh, mesh["dp"], mesh["tp"], mesh["dp"], None
    if topology == "dp_tp2":
        if world_size != 4:
            raise RuntimeError(f"dp_tp2 needs world=4, got {world_size}")
        mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
        return mesh, mesh["dp"], mesh["tp"], mesh["dp"], None
    if topology == "dp_tp4":
        if world_size != 8:
            raise RuntimeError(f"dp_tp4 needs world=8, got {world_size}")
        mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
        return mesh, mesh["dp"], mesh["tp"], mesh["dp"], None
    if topology == "hsdp_tp2":
        if world_size != 8:
            raise RuntimeError(f"hsdp_tp2 needs world=8, got {world_size}")
        mesh = init_device_mesh(
            "cuda", (2, 2, 2), mesh_dim_names=("replicate", "shard", "tp")
        )
        return mesh, mesh["shard"], mesh["tp"], mesh["replicate", "shard"], mesh["replicate"]
    raise RuntimeError(
        "topology must be one of: tp2 | tp4 | dp_tp2 | dp_tp4 | hsdp_tp2"
    )


def _validate_tp_config(cfg: dict[str, Any], tp_size: int, mode: str) -> None:
    if cfg["intermediate"] % tp_size != 0:
        raise ValueError(
            f"intermediate={cfg['intermediate']} must be divisible by tp_size={tp_size}"
        )
    if cfg["hidden"] % tp_size != 0:
        raise ValueError(f"hidden={cfg['hidden']} must be divisible by tp_size={tp_size}")
    if mode == "full":
        if cfg["heads"] % tp_size != 0:
            raise ValueError(f"heads={cfg['heads']} must be divisible by tp_size={tp_size}")
        if cfg["kv_heads"] % tp_size != 0:
            raise ValueError(
                f"kv_heads={cfg['kv_heads']} must be divisible by tp_size={tp_size}"
            )


def apply_tp(model, tp_mesh, mode: str) -> None:
    if mode not in ("mlp", "full"):
        raise ValueError("DMUON_TP_LLM_PARALLELIZE must be 'mlp' or 'full'")

    mlp_plan = {
        "gate_proj": ColwiseParallel(),
        "up_proj": ColwiseParallel(),
        "down_proj": RowwiseParallel(),
    }
    attn_plan = {
        "q_proj": ColwiseParallel(),
        "k_proj": ColwiseParallel(),
        "v_proj": ColwiseParallel(),
        "o_proj": RowwiseParallel(),
    }
    for layer in model.model.layers:
        parallelize_module(layer.mlp, tp_mesh, mlp_plan)
        if mode == "full":
            parallelize_module(layer.self_attn, tp_mesh, attn_plan)


def _force_tp_owner_rank0(model) -> None:
    for dp in iter_dedicated_params(model):
        tp_group = getattr(dp, "tp_group", None)
        if tp_group is None:
            continue
        dp._tp_owner_local_rank = 0
        dp._tp_owner_global_rank = dist.get_global_rank(tp_group, 0)
        dp.is_tp_owner = tp_group.rank() == 0


def _drain_async(model) -> None:
    wait_all_replicate_broadcasts(model)


def _step(model, optimizer, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
    optimizer.zero_grad()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    optimizer.step()
    return float(out.loss.detach().item())


def _run_benchmark(
    model,
    optimizer,
    *,
    device: torch.device,
    batch: int,
    seq: int,
    vocab: int,
    warmup_steps: int,
    measure_steps: int,
    trials: int,
) -> tuple[list[float], list[float]]:
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    input_ids = torch.randint(0, vocab, (batch, seq), device=device)
    labels = torch.randint(0, vocab, (batch, seq), device=device)

    losses: list[float] = []
    for _ in range(warmup_steps):
        losses.append(_step(model, optimizer, input_ids, labels))
    _drain_async(model)
    torch.cuda.synchronize()

    trial_step_ms: list[float] = []
    for _ in range(trials):
        _drain_async(model)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(measure_steps):
            losses.append(_step(model, optimizer, input_ids, labels))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        trial_step_ms.append((t1 - t0) * 1000.0 / measure_steps)

    _drain_async(model)
    torch.cuda.synchronize()
    return trial_step_ms, losses


def _summarize_rank_payloads(
    rank_payloads: list[dict[str, Any]],
    *,
    batch: int,
    seq: int,
    data_parallel_factor: int,
) -> dict[str, Any]:
    trial_count = min(len(p["trial_step_ms"]) for p in rank_payloads)
    per_trial_max = [
        max(float(p["trial_step_ms"][i]) for p in rank_payloads)
        for i in range(trial_count)
    ]
    p50 = _percentile(per_trial_max, 0.50)
    p90 = _percentile(per_trial_max, 0.90)
    avg = statistics.mean(per_trial_max) if per_trial_max else 0.0
    local_tokens_per_s = (batch * seq / (p50 / 1000.0)) if p50 > 0 else 0.0
    local_samples_per_s = (batch / (p50 / 1000.0)) if p50 > 0 else 0.0
    global_tokens_per_s = local_tokens_per_s * data_parallel_factor
    global_samples_per_s = local_samples_per_s * data_parallel_factor
    return {
        "data_parallel_factor": int(data_parallel_factor),
        "global_batch": int(batch * data_parallel_factor),
        "step_ms_per_trial_max_rank": per_trial_max,
        "step_ms_avg": float(avg),
        "step_ms_p50": float(p50),
        "step_ms_p90": float(p90),
        "samples_per_s_p50": float(local_samples_per_s),
        "tokens_per_s_p50": float(local_tokens_per_s),
        "global_samples_per_s_p50": float(global_samples_per_s),
        "global_tokens_per_s_p50": float(global_tokens_per_s),
        "peak_memory_allocated_gb_max_rank": max(
            p["peak_memory_allocated_gb"] for p in rank_payloads
        ),
        "peak_memory_reserved_gb_max_rank": max(
            p["peak_memory_reserved_gb"] for p in rank_payloads
        ),
    }


def _write_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    model_key = sys.argv[1] if len(sys.argv) > 1 else "llama3b"
    topology = sys.argv[2] if len(sys.argv) > 2 else "tp4"
    owner = os.environ.get("DMUON_TP_LLM_OWNER", "lpt")
    if owner not in ("lpt", "rank0"):
        raise ValueError("DMUON_TP_LLM_OWNER must be 'lpt' or 'rank0'")

    parallelize_mode = os.environ.get("DMUON_TP_LLM_PARALLELIZE", "full")
    replicate_async = bool(int(os.environ.get("DMUON_TP_LLM_ASYNC", "0") or 0))
    warmup_steps = _env_int("DMUON_TP_LLM_WARMUP", 3)
    measure_steps = _env_int("DMUON_TP_LLM_STEPS", 6)
    trials = _env_int("DMUON_TP_LLM_TRIALS", 3)
    batch = _env_int("DMUON_TP_LLM_BATCH", 1)
    seq = _env_int("DMUON_TP_LLM_SEQ", 256)
    out_path = os.environ.get("DMUON_TP_LLM_OUT")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    try:
        torch.cuda.reset_peak_memory_stats(device)
        mesh, dp_mesh, tp_mesh, fsdp_mesh, replicate_mesh = _make_mesh(
            topology, world_size
        )
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model, model_cfg = build_model(model_key, device)
        logical_param_count = sum(int(p.numel()) for p in model.parameters())
        _validate_tp_config(model_cfg, int(tp_mesh.size()), parallelize_mode)
        apply_tp(model, tp_mesh, parallelize_mode)

        dmuon.dedicate_params(
            model,
            dp_mesh,
            replicate_mesh=replicate_mesh,
            predicate=lambda n, p: "proj" in n and p.ndim == 2,
            compute_dtype=torch.bfloat16,
            reshard_after_forward=False,
        )
        if owner == "rank0":
            _force_tp_owner_rank0(model)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        )
        for layer in model.model.layers:
            fully_shard(layer, mesh=fsdp_mesh, mp_policy=mp_policy)
        fully_shard(model, mesh=fsdp_mesh, mp_policy=mp_policy)

        optimizer = dmuon.Muon(
            model,
            lr=0.01,
            ns_steps=5,
            adamw_lr=0.01,
            replicate_async=replicate_async,
        )

        profile = collect_tp_profile(
            model,
            scenario=f"{model_key}_{topology}_{owner}_{parallelize_mode}",
            replicate_async=replicate_async,
        )
        tp_profile = gather_tp_profiles(profile)
        if rank == 0:
            print(
                f"bench model={model_cfg['model_label']} topology={topology} "
                f"owner={owner} async={replicate_async} tp={tp_mesh.size()} "
                f"mode={parallelize_mode} params={logical_param_count / 1e9:.2f}B "
                f"coverage={tp_profile['owner_coverage']}",
                flush=True,
            )

        trial_step_ms, losses = _run_benchmark(
            model,
            optimizer,
            device=device,
            batch=batch,
            seq=seq,
            vocab=model_cfg["vocab"],
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            trials=trials,
        )

        local_payload = {
            "rank": rank,
            "trial_step_ms": trial_step_ms,
            "losses": losses[-min(len(losses), 5):],
            "peak_memory_allocated_gb": torch.cuda.max_memory_allocated(device)
            / 1024**3,
            "peak_memory_reserved_gb": torch.cuda.max_memory_reserved(device)
            / 1024**3,
        }
        rank_payloads = [None for _ in range(world_size)]
        dist.all_gather_object(rank_payloads, local_payload)

        if rank == 0:
            summary = _summarize_rank_payloads(
                rank_payloads,
                batch=batch,
                seq=seq,
                data_parallel_factor=int(fsdp_mesh.size()),
            )
            result = {
                "model": model_key,
                "model_config": model_cfg,
                "logical_param_count": logical_param_count,
                "topology": topology,
                "owner": owner,
                "parallelize_mode": parallelize_mode,
                "replicate_async": replicate_async,
                "world_size": world_size,
                "config": {
                    "warmup_steps": warmup_steps,
                    "measure_steps": measure_steps,
                    "trials": trials,
                    "batch": batch,
                    "seq": seq,
                },
                "summary": summary,
                "tp_profile": tp_profile,
                "ranks": rank_payloads,
            }
            _write_json(out_path, result)
            print(
                "result "
                f"p50={summary['step_ms_p50']:.3f}ms "
                f"p90={summary['step_ms_p90']:.3f}ms "
                f"tokens/s={summary['tokens_per_s_p50']:.1f} "
                f"global_tokens/s={summary['global_tokens_per_s_p50']:.1f} "
                f"peak_mem={summary['peak_memory_allocated_gb_max_rank']:.2f}GB",
                flush=True,
            )
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
