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
    DMUON_TP_LLM_LAYERS=2 DMUON_TP_LLM_SEQ=64 \
        torchrun --nproc_per_node=2 benchmarks/bench_tp_llm.py llama3b tp2
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from contextlib import nullcontext
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
from torch.profiler import ProfilerActivity, profile, record_function

REPO = Path(__file__).resolve().parents[1]
TEST_DIST = REPO / "tests" / "distributed"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TEST_DIST))

from tp_fused_manual import apply_fused_manual_tp  # noqa: E402
from tp_profile_utils import (  # noqa: E402
    collect_tp_profile,
    gather_tp_profiles,
    iter_dedicated_params,
)

import dmuon  # noqa: E402
from dmuon.utils import prepare_muon_grads, wait_all_replicate_broadcasts  # noqa: E402


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
    "qwen14b": ModelSpec(
        family="qwen2",
        label="Qwen2.5-14B",
        hidden=5120,
        intermediate=13824,
        layers=48,
        heads=40,
        kv_heads=8,
        vocab=152064,
        max_positions=131072,
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
        "max_positions": _env_int("DMUON_TP_LLM_MAX_POSITIONS", spec.max_positions),
        "tie_word_embeddings": bool(spec.tie_word_embeddings),
        "attn_implementation": os.environ.get("DMUON_TP_LLM_ATTN_IMPL") or None,
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
        if cfg["attn_implementation"] is not None:
            hf_config._attn_implementation = cfg["attn_implementation"]
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
        if cfg["attn_implementation"] is not None:
            hf_config._attn_implementation = cfg["attn_implementation"]
        with device:
            return LlamaForCausalLM(hf_config), cfg

    raise ValueError(f"unsupported family={cfg['family']!r}")


def _make_mesh(topology: str, world_size: int):
    if topology == "dp_only":
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
        return mesh, mesh["dp"], None, mesh["dp"], None
    if topology == "hsdp":
        if world_size != 8:
            raise RuntimeError(f"hsdp needs world=8, got {world_size}")
        mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("replicate", "shard"))
        return mesh, mesh["shard"], None, mesh["replicate", "shard"], mesh["replicate"]
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
    if topology == "tp8":
        if world_size != 8:
            raise RuntimeError(f"tp8 needs world=8, got {world_size}")
        mesh = init_device_mesh("cuda", (1, 8), mesh_dim_names=("dp", "tp"))
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
    if topology == "dp_tp8":
        if world_size != 16:
            raise RuntimeError(f"dp_tp8 needs world=16, got {world_size}")
        mesh = init_device_mesh("cuda", (2, 8), mesh_dim_names=("dp", "tp"))
        return mesh, mesh["dp"], mesh["tp"], mesh["dp"], None
    if topology == "hsdp_tp2":
        if world_size != 8:
            raise RuntimeError(f"hsdp_tp2 needs world=8, got {world_size}")
        mesh = init_device_mesh(
            "cuda", (2, 2, 2), mesh_dim_names=("replicate", "shard", "tp")
        )
        return (
            mesh,
            mesh["shard"],
            mesh["tp"],
            mesh["replicate", "shard"],
            mesh["replicate"],
        )
    raise RuntimeError(
        "topology must be one of: dp_only | hsdp | tp2 | tp4 | tp8 | dp_tp2 | "
        "dp_tp4 | dp_tp8 | hsdp_tp2"
    )


def _validate_tp_config(cfg: dict[str, Any], tp_size: int, mode: str) -> None:
    if cfg["intermediate"] % tp_size != 0:
        raise ValueError(
            f"intermediate={cfg['intermediate']} must be divisible by tp_size={tp_size}"
        )
    if cfg["hidden"] % tp_size != 0:
        raise ValueError(
            f"hidden={cfg['hidden']} must be divisible by tp_size={tp_size}"
        )
    if mode == "full":
        if cfg["heads"] % tp_size != 0:
            raise ValueError(
                f"heads={cfg['heads']} must be divisible by tp_size={tp_size}"
            )
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


def apply_tp_backend(model, tp_mesh, mode: str, backend: str) -> dict[str, Any]:
    if backend == "dtensor":
        apply_tp(model, tp_mesh, mode)
        return {
            "tp_backend": backend,
            "expected_allreduces_per_layer": 7 if mode == "full" else 3,
        }
    if backend == "fused_manual":
        stats = apply_fused_manual_tp(model, tp_mesh, mode)
        return {
            "tp_backend": backend,
            "expected_allreduces_per_layer": stats.expected_allreduces_per_layer,
        }
    raise ValueError("DMUON_TP_LLM_TP_IMPL must be 'dtensor' or 'fused_manual'")


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return bool(int(raw))


def _loss_scalar(loss: torch.Tensor) -> float:
    return float(loss.detach().item())


def _step_impl(
    forward_model,
    state_model,
    optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    step_scope: str,
    zero_set_to_none: bool,
    prepare_grads: bool,
    no_sync_fwd_bwd: bool,
    drain_after_step: bool,
) -> torch.Tensor:
    with record_function("dmuon.bench.zero_grad"):
        optimizer.zero_grad(set_to_none=zero_set_to_none)
    sync_ctx = (
        dmuon.no_sync(state_model)
        if step_scope == "fwd_bwd" and no_sync_fwd_bwd
        else nullcontext()
    )
    with sync_ctx:
        with record_function("dmuon.bench.forward"):
            out = forward_model(input_ids=input_ids, labels=labels)
            loss = out.loss
        with record_function("dmuon.bench.backward"):
            loss.backward()
    if step_scope == "train":
        with record_function("dmuon.bench.optimizer_step"):
            optimizer.step()
        if drain_after_step:
            with record_function("dmuon.bench.drain_post_step"):
                wait_all_replicate_broadcasts(state_model)
    elif step_scope == "fwd_bwd":
        if prepare_grads:
            with record_function("dmuon.bench.prepare_muon_grads"):
                prepare_muon_grads(state_model)
    else:
        raise ValueError("DMUON_TP_LLM_STEP_SCOPE must be 'train' or 'fwd_bwd'")
    return loss.detach()


class _EagerRunner:
    def __init__(
        self,
        *,
        forward_model,
        state_model,
        optimizer,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        step_scope: str,
        zero_set_to_none: bool,
        prepare_grads: bool,
        no_sync_fwd_bwd: bool,
        drain_after_step: bool,
    ) -> None:
        self.forward_model = forward_model
        self.state_model = state_model
        self.optimizer = optimizer
        self.input_ids = input_ids
        self.labels = labels
        self.step_scope = step_scope
        self.zero_set_to_none = zero_set_to_none
        self.prepare_grads = prepare_grads
        self.no_sync_fwd_bwd = no_sync_fwd_bwd
        self.drain_after_step = drain_after_step

    def step(self) -> torch.Tensor:
        return _step_impl(
            self.forward_model,
            self.state_model,
            self.optimizer,
            self.input_ids,
            self.labels,
            step_scope=self.step_scope,
            zero_set_to_none=self.zero_set_to_none,
            prepare_grads=self.prepare_grads,
            no_sync_fwd_bwd=self.no_sync_fwd_bwd,
            drain_after_step=self.drain_after_step,
        )


class _FwdBwdGradManager:
    """Minimal grad clearer for fwd+bwd-only benchmark modes."""

    def __init__(self, model) -> None:
        self.model = model

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.model.parameters():
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()
        for dp in iter_dedicated_params(self.model):
            dp._reduced_grad = None
            dp._accumulated_grad = None
            if hasattr(dp, "_tp_full_grad"):
                dp._tp_full_grad = None
            if hasattr(dp, "_tp_full_delta"):
                dp._tp_full_delta = None

    def consume_last_step_profile(self) -> dict[str, object]:
        return {}


class _CudaGraphFwdBwdRunner:
    """Replay a captured forward+backward graph on static synthetic tensors."""

    def __init__(
        self,
        *,
        forward_model,
        state_model,
        optimizer,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
        capture_warmup_steps: int,
        zero_set_to_none: bool,
        prepare_grads: bool,
        no_sync_fwd_bwd: bool,
        drain_after_step: bool,
    ) -> None:
        if zero_set_to_none:
            raise ValueError("cuda_graph fwd_bwd requires in-place grad zeroing")
        if drain_after_step:
            raise ValueError("cuda_graph fwd_bwd does not support drain_after_step")
        self.graph = torch.cuda.CUDAGraph()
        self.loss: torch.Tensor | None = None

        # Warm up on a side stream so allocator state is stable before capture.
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(max(1, capture_warmup_steps)):
                _step_impl(
                    forward_model,
                    state_model,
                    optimizer,
                    input_ids,
                    labels,
                    step_scope="fwd_bwd",
                    zero_set_to_none=False,
                    prepare_grads=prepare_grads,
                    no_sync_fwd_bwd=no_sync_fwd_bwd,
                    drain_after_step=False,
                )
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        torch.cuda.synchronize(device)
        dist.barrier(device_ids=[device.index])

        with torch.cuda.graph(self.graph):
            self.loss = _step_impl(
                forward_model,
                state_model,
                optimizer,
                input_ids,
                labels,
                step_scope="fwd_bwd",
                zero_set_to_none=False,
                prepare_grads=prepare_grads,
                no_sync_fwd_bwd=no_sync_fwd_bwd,
                drain_after_step=False,
            )
        torch.cuda.synchronize(device)
        dist.barrier(device_ids=[device.index])

    def step(self) -> torch.Tensor:
        self.graph.replay()
        assert self.loss is not None
        return self.loss.detach()


def _compile_forward_model(model):
    mode = os.environ.get("DMUON_TP_LLM_COMPILE_MODE", "default")
    backend = os.environ.get("DMUON_TP_LLM_COMPILE_BACKEND", "inductor")
    fullgraph = _env_bool("DMUON_TP_LLM_COMPILE_FULLGRAPH", False)
    dynamic = _env_bool("DMUON_TP_LLM_COMPILE_DYNAMIC", False)
    disable_cudagraph_trees = _env_bool(
        "DMUON_TP_LLM_COMPILE_DISABLE_CUDAGRAPH_TREES", True
    )
    if disable_cudagraph_trees:
        try:
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraph_trees = False
        except Exception:
            pass
    return torch.compile(
        model,
        backend=backend,
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )


def _make_step_runner(
    *,
    exec_mode: str,
    forward_model,
    state_model,
    optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    step_scope: str,
    capture_warmup_steps: int,
    zero_set_to_none: bool,
    prepare_grads: bool,
    no_sync_fwd_bwd: bool,
    drain_after_step: bool,
):
    if exec_mode in ("eager", "compile"):
        return _EagerRunner(
            forward_model=forward_model,
            state_model=state_model,
            optimizer=optimizer,
            input_ids=input_ids,
            labels=labels,
            step_scope=step_scope,
            zero_set_to_none=zero_set_to_none,
            prepare_grads=prepare_grads,
            no_sync_fwd_bwd=no_sync_fwd_bwd,
            drain_after_step=drain_after_step,
        )
    if exec_mode == "cuda_graph":
        if step_scope != "fwd_bwd":
            raise ValueError("cuda_graph currently supports only fwd_bwd scope")
        return _CudaGraphFwdBwdRunner(
            forward_model=forward_model,
            state_model=state_model,
            optimizer=optimizer,
            input_ids=input_ids,
            labels=labels,
            device=device,
            capture_warmup_steps=capture_warmup_steps,
            zero_set_to_none=zero_set_to_none,
            prepare_grads=prepare_grads,
            no_sync_fwd_bwd=no_sync_fwd_bwd,
            drain_after_step=drain_after_step,
        )
    raise ValueError("DMUON_TP_LLM_EXEC must be 'eager', 'compile', or 'cuda_graph'")


def _run_benchmark(
    forward_model,
    state_model,
    optimizer,
    *,
    device: torch.device,
    batch: int,
    seq: int,
    vocab: int,
    warmup_steps: int,
    measure_steps: int,
    trials: int,
    exec_mode: str,
    step_scope: str,
    capture_warmup_steps: int,
    zero_set_to_none: bool,
    prepare_grads: bool,
    no_sync_fwd_bwd: bool,
    drain_after_step: bool,
) -> tuple[list[float], list[float], list[float]]:
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    input_ids = torch.randint(0, vocab, (batch, seq), device=device)
    labels = torch.randint(0, vocab, (batch, seq), device=device)
    runner = _make_step_runner(
        exec_mode=exec_mode,
        forward_model=forward_model,
        state_model=state_model,
        optimizer=optimizer,
        input_ids=input_ids,
        labels=labels,
        device=device,
        step_scope=step_scope,
        capture_warmup_steps=capture_warmup_steps,
        zero_set_to_none=zero_set_to_none,
        prepare_grads=prepare_grads,
        no_sync_fwd_bwd=no_sync_fwd_bwd,
        drain_after_step=drain_after_step,
    )

    reference_losses: list[float] = []
    if step_scope == "fwd_bwd" and exec_mode != "eager":
        ref_runner = _EagerRunner(
            forward_model=state_model,
            state_model=state_model,
            optimizer=optimizer,
            input_ids=input_ids,
            labels=labels,
            step_scope=step_scope,
            zero_set_to_none=zero_set_to_none,
            prepare_grads=prepare_grads,
            no_sync_fwd_bwd=no_sync_fwd_bwd,
            drain_after_step=drain_after_step,
        )
        ref_loss = ref_runner.step()
        torch.cuda.synchronize(device)
        reference_losses.append(_loss_scalar(ref_loss))
        optimizer.zero_grad(set_to_none=zero_set_to_none)
        torch.cuda.synchronize(device)

    losses: list[float] = []
    for _ in range(warmup_steps):
        loss = runner.step()
        torch.cuda.synchronize(device)
        losses.append(_loss_scalar(loss))
    _drain_async(state_model)
    torch.cuda.synchronize(device)

    trial_step_ms: list[float] = []
    for _ in range(trials):
        _drain_async(state_model)
        torch.cuda.synchronize(device)
        trial_losses: list[torch.Tensor] = []
        t0 = time.perf_counter()
        for _ in range(measure_steps):
            trial_losses.append(runner.step())
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        trial_step_ms.append((t1 - t0) * 1000.0 / measure_steps)
        losses.extend(_loss_scalar(loss) for loss in trial_losses)

    _drain_async(state_model)
    torch.cuda.synchronize(device)
    return trial_step_ms, losses, reference_losses


def _run_torch_profile(
    forward_model,
    state_model,
    optimizer,
    *,
    device: torch.device,
    batch: int,
    seq: int,
    vocab: int,
    steps: int,
    trace_path: str,
    rank: int,
    exec_mode: str,
    step_scope: str,
    capture_warmup_steps: int,
    zero_set_to_none: bool,
    prepare_grads: bool,
    no_sync_fwd_bwd: bool,
    drain_after_step: bool,
) -> list[dict[str, object]]:
    """Run a short synchronized profile window and export rank-0 chrome trace."""
    torch.manual_seed(200)
    torch.cuda.manual_seed_all(200)
    input_ids = torch.randint(0, vocab, (batch, seq), device=device)
    labels = torch.randint(0, vocab, (batch, seq), device=device)
    runner = _make_step_runner(
        exec_mode=exec_mode,
        forward_model=forward_model,
        state_model=state_model,
        optimizer=optimizer,
        input_ids=input_ids,
        labels=labels,
        device=device,
        step_scope=step_scope,
        capture_warmup_steps=capture_warmup_steps,
        zero_set_to_none=zero_set_to_none,
        prepare_grads=prepare_grads,
        no_sync_fwd_bwd=no_sync_fwd_bwd,
        drain_after_step=drain_after_step,
    )

    optimizer_profiles: list[dict[str, object]] = []
    _drain_async(state_model)
    torch.cuda.synchronize(device)
    dist.barrier(device_ids=[device.index])

    if rank == 0:
        trace_out = Path(trace_path)
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(
            activities=activities,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for step_idx in range(steps):
                with record_function(f"dmuon.profile.step_{step_idx}"):
                    runner.step()
                torch.cuda.synchronize(device)
                if step_scope == "train":
                    optimizer_profiles.append(optimizer.consume_last_step_profile())
                prof.step()
        prof.export_chrome_trace(str(trace_out))
    else:
        for _ in range(steps):
            runner.step()
        torch.cuda.synchronize(device)

    _drain_async(state_model)
    torch.cuda.synchronize(device)
    dist.barrier(device_ids=[device.index])
    return optimizer_profiles


def _summarize_rank_payloads(
    rank_payloads: list[dict[str, Any]],
    *,
    batch: int,
    seq: int,
    data_parallel_factor: int,
    logical_param_count: int,
    world_size: int,
    peak_tflops_per_gpu: float,
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
    approx_train_tflops = 6.0 * logical_param_count * global_tokens_per_s / 1e12
    approx_mfu = (
        approx_train_tflops / (world_size * peak_tflops_per_gpu)
        if peak_tflops_per_gpu > 0
        else 0.0
    )
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
        "approx_train_tflops": float(approx_train_tflops),
        "approx_mfu": float(approx_mfu),
        "peak_tflops_per_gpu_assumption": float(peak_tflops_per_gpu),
        "peak_memory_allocated_gb_max_rank": max(
            p["peak_memory_allocated_gb"] for p in rank_payloads
        ),
        "peak_memory_reserved_gb_max_rank": max(
            p["peak_memory_reserved_gb"] for p in rank_payloads
        ),
    }


def _summarize_loss_reference(rank_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    diffs: list[float] = []
    comparisons: list[dict[str, Any]] = []
    for payload in rank_payloads:
        refs = payload.get("reference_losses") or []
        losses = payload.get("losses") or []
        if not refs or not losses:
            continue
        ref = float(refs[-1])
        actual = float(losses[0])
        diff = abs(actual - ref)
        diffs.append(diff)
        comparisons.append(
            {
                "rank": int(payload["rank"]),
                "reference": ref,
                "actual": actual,
                "abs_diff": diff,
                "bit_equal": actual == ref,
            }
        )
    return {
        "available": bool(comparisons),
        "bit_equal": bool(comparisons) and all(c["bit_equal"] for c in comparisons),
        "max_abs_diff": max(diffs) if diffs else None,
        "comparisons": comparisons,
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
    if owner not in ("lpt", "round_robin", "rank0"):
        raise ValueError("DMUON_TP_LLM_OWNER must be 'lpt', 'round_robin', or 'rank0'")
    owner_cost_model = os.environ.get("DMUON_TP_LLM_OWNER_COST_MODEL", "optimizer")
    if owner_cost_model not in ("optimizer", "numel"):
        raise ValueError("DMUON_TP_LLM_OWNER_COST_MODEL must be 'optimizer' or 'numel'")
    hsdp_column_balance = _env_bool("DMUON_TP_LLM_HSDP_COLUMN_BALANCE", True)

    parallelize_mode = os.environ.get("DMUON_TP_LLM_PARALLELIZE", "full")
    replicate_async = bool(int(os.environ.get("DMUON_TP_LLM_ASYNC", "0") or 0))
    warmup_steps = _env_int("DMUON_TP_LLM_WARMUP", 3)
    measure_steps = _env_int("DMUON_TP_LLM_STEPS", 6)
    trials = _env_int("DMUON_TP_LLM_TRIALS", 3)
    batch = _env_int("DMUON_TP_LLM_BATCH", 1)
    seq = _env_int("DMUON_TP_LLM_SEQ", 256)
    out_path = os.environ.get("DMUON_TP_LLM_OUT")
    profile_enabled = bool(int(os.environ.get("DMUON_TP_LLM_PROFILE", "0") or 0))
    profile_dir = os.environ.get("DMUON_TP_LLM_PROFILE_DIR")
    profile_steps = _env_int("DMUON_TP_LLM_PROFILE_STEPS", 2)
    peak_tflops_per_gpu = float(os.environ.get("DMUON_TP_LLM_PEAK_TFLOPS", "312"))
    exec_mode = os.environ.get("DMUON_TP_LLM_EXEC", "eager")
    if exec_mode not in ("eager", "compile", "cuda_graph"):
        raise ValueError(
            "DMUON_TP_LLM_EXEC must be 'eager', 'compile', or 'cuda_graph'"
        )
    tp_impl = os.environ.get("DMUON_TP_LLM_TP_IMPL", "dtensor")
    if tp_impl not in ("dtensor", "fused_manual"):
        raise ValueError("DMUON_TP_LLM_TP_IMPL must be 'dtensor' or 'fused_manual'")
    step_scope = os.environ.get("DMUON_TP_LLM_STEP_SCOPE", "train")
    if step_scope not in ("train", "fwd_bwd"):
        raise ValueError("DMUON_TP_LLM_STEP_SCOPE must be 'train' or 'fwd_bwd'")
    if tp_impl == "fused_manual" and step_scope != "fwd_bwd":
        raise ValueError("fused_manual TP is benchmark-only and requires fwd_bwd scope")
    capture_warmup_steps = _env_int("DMUON_TP_LLM_CUDA_GRAPH_WARMUP", 3)
    zero_set_to_none = _env_bool(
        "DMUON_TP_LLM_ZERO_SET_TO_NONE", default=(step_scope == "train")
    )
    prepare_grads = _env_bool(
        "DMUON_TP_LLM_PREPARE_GRADS",
        default=(step_scope == "fwd_bwd" and exec_mode != "cuda_graph"),
    )
    no_sync_fwd_bwd = _env_bool(
        "DMUON_TP_LLM_NO_SYNC",
        default=(step_scope == "fwd_bwd" and exec_mode == "cuda_graph"),
    )
    drain_after_step = _env_bool("DMUON_TP_LLM_DRAIN_AFTER_STEP", False)
    deterministic = _env_bool("DMUON_TP_LLM_DETERMINISTIC", False)
    skip_size1_fsdp_env = os.environ.get("DMUON_TP_LLM_SKIP_SIZE1_FSDP")
    if exec_mode == "cuda_graph":
        if step_scope != "fwd_bwd":
            raise ValueError(
                "cuda_graph currently requires DMUON_TP_LLM_STEP_SCOPE=fwd_bwd"
            )
        zero_set_to_none = False

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    try:
        if deterministic:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats(device)
        mesh, dp_mesh, tp_mesh, fsdp_mesh, replicate_mesh = _make_mesh(
            topology, world_size
        )
        skip_size1_fsdp = (
            (step_scope == "fwd_bwd")
            if skip_size1_fsdp_env is None or skip_size1_fsdp_env == ""
            else bool(int(skip_size1_fsdp_env))
        ) and int(fsdp_mesh.size()) == 1
        if skip_size1_fsdp and step_scope == "train":
            raise ValueError("DMUON_TP_LLM_SKIP_SIZE1_FSDP is only valid for fwd_bwd")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model, model_cfg = build_model(model_key, device)
        logical_param_count = sum(int(p.numel()) for p in model.parameters())
        effective_parallelize_mode = parallelize_mode
        tp_backend_info: dict[str, Any] = {
            "tp_backend": "none",
            "expected_allreduces_per_layer": 0,
        }
        if tp_mesh is not None:
            _validate_tp_config(model_cfg, int(tp_mesh.size()), parallelize_mode)
            tp_backend_info = apply_tp_backend(
                model, tp_mesh, parallelize_mode, tp_impl
            )
        else:
            effective_parallelize_mode = "none"
        if skip_size1_fsdp:
            model.to(torch.bfloat16)

        dmuon.dedicate_params(
            model,
            dp_mesh,
            replicate_mesh=replicate_mesh,
            predicate=lambda n, p: "proj" in n and p.ndim == 2,
            compute_dtype=torch.bfloat16,
            reshard_after_forward=False,
            owner_strategy=owner,
            owner_cost_model=owner_cost_model,
            hsdp_column_balance=hsdp_column_balance,
        )
        if owner == "rank0":
            _force_tp_owner_rank0(model)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        )
        fsdp_wrapped = not skip_size1_fsdp
        if fsdp_wrapped:
            for layer in model.model.layers:
                fully_shard(layer, mesh=fsdp_mesh, mp_policy=mp_policy)
            fully_shard(model, mesh=fsdp_mesh, mp_policy=mp_policy)

        if step_scope == "train":
            optimizer = dmuon.Muon(
                model,
                lr=0.01,
                ns_steps=5,
                adamw_lr=0.01,
                replicate_async=replicate_async,
            )
        else:
            optimizer = _FwdBwdGradManager(model)
        forward_model = (
            _compile_forward_model(model) if exec_mode == "compile" else model
        )

        profile = collect_tp_profile(
            model,
            scenario=f"{model_key}_{topology}_{owner}_{effective_parallelize_mode}",
            replicate_async=replicate_async,
        )
        tp_profile = gather_tp_profiles(profile)
        if rank == 0:
            print(
                f"bench model={model_cfg['model_label']} topology={topology} "
                f"owner={owner} async={replicate_async} "
                f"owner_cost_model={owner_cost_model} "
                f"hsdp_column_balance={hsdp_column_balance} "
                f"tp={1 if tp_mesh is None else tp_mesh.size()} "
                f"mode={effective_parallelize_mode} tp_impl={tp_impl} "
                f"exec={exec_mode} scope={step_scope} "
                f"fsdp_wrapped={fsdp_wrapped} "
                f"params={logical_param_count / 1e9:.2f}B "
                f"coverage={tp_profile['owner_coverage']}",
                flush=True,
            )

        trial_step_ms, losses, reference_losses = _run_benchmark(
            forward_model,
            model,
            optimizer,
            device=device,
            batch=batch,
            seq=seq,
            vocab=model_cfg["vocab"],
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            trials=trials,
            exec_mode=exec_mode,
            step_scope=step_scope,
            capture_warmup_steps=capture_warmup_steps,
            zero_set_to_none=zero_set_to_none,
            prepare_grads=prepare_grads,
            no_sync_fwd_bwd=no_sync_fwd_bwd,
            drain_after_step=drain_after_step,
        )

        trace_path = None
        optimizer_profiles: list[dict[str, object]] = []
        if profile_enabled:
            if not profile_dir:
                raise ValueError(
                    "DMUON_TP_LLM_PROFILE_DIR is required when "
                    "DMUON_TP_LLM_PROFILE=1"
                )
            trace_path = str(
                Path(profile_dir) / f"{model_key}_{topology}_{owner}_"
                f"{'async' if replicate_async else 'sync'}_"
                f"{step_scope}_{exec_mode}_rank0.trace.json"
            )
            optimizer_profiles = _run_torch_profile(
                forward_model,
                model,
                optimizer,
                device=device,
                batch=batch,
                seq=seq,
                vocab=model_cfg["vocab"],
                steps=profile_steps,
                trace_path=trace_path,
                rank=rank,
                exec_mode=exec_mode,
                step_scope=step_scope,
                capture_warmup_steps=capture_warmup_steps,
                zero_set_to_none=zero_set_to_none,
                prepare_grads=prepare_grads,
                no_sync_fwd_bwd=no_sync_fwd_bwd,
                drain_after_step=drain_after_step,
            )

        local_payload = {
            "rank": rank,
            "trial_step_ms": trial_step_ms,
            "losses": losses[-min(len(losses), 5) :],
            "reference_losses": reference_losses,
            "peak_memory_allocated_gb": torch.cuda.max_memory_allocated(device)
            / 1024**3,
            "peak_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
        }
        rank_payloads = [None for _ in range(world_size)]
        dist.all_gather_object(rank_payloads, local_payload)

        if rank == 0:
            summary = _summarize_rank_payloads(
                rank_payloads,
                batch=batch,
                seq=seq,
                data_parallel_factor=int(fsdp_mesh.size()),
                logical_param_count=logical_param_count,
                world_size=world_size,
                peak_tflops_per_gpu=peak_tflops_per_gpu,
            )
            result = {
                "model": model_key,
                "model_config": model_cfg,
                "logical_param_count": logical_param_count,
                "topology": topology,
                "owner": owner,
                "owner_cost_model": owner_cost_model,
                "hsdp_column_balance": hsdp_column_balance,
                "parallelize_mode": effective_parallelize_mode,
                "tp_backend": tp_impl,
                "tp_backend_info": tp_backend_info,
                "replicate_async": replicate_async,
                "world_size": world_size,
                "config": {
                    "warmup_steps": warmup_steps,
                    "measure_steps": measure_steps,
                    "trials": trials,
                    "batch": batch,
                    "seq": seq,
                    "profile_enabled": profile_enabled,
                    "profile_steps": profile_steps if profile_enabled else 0,
                    "exec_mode": exec_mode,
                    "tp_impl": tp_impl,
                    "step_scope": step_scope,
                    "cuda_graph_warmup_steps": capture_warmup_steps,
                    "zero_set_to_none": zero_set_to_none,
                    "prepare_grads": prepare_grads,
                    "no_sync_fwd_bwd": no_sync_fwd_bwd,
                    "drain_after_step": drain_after_step,
                    "deterministic": deterministic,
                    "owner_cost_model": owner_cost_model,
                    "hsdp_column_balance": hsdp_column_balance,
                    "fsdp_wrapped": fsdp_wrapped,
                    "skip_size1_fsdp": skip_size1_fsdp,
                    "compile_backend": os.environ.get(
                        "DMUON_TP_LLM_COMPILE_BACKEND", "inductor"
                    ),
                    "compile_mode": os.environ.get(
                        "DMUON_TP_LLM_COMPILE_MODE", "default"
                    ),
                    "compile_fullgraph": _env_bool(
                        "DMUON_TP_LLM_COMPILE_FULLGRAPH", False
                    ),
                    "compile_dynamic": _env_bool("DMUON_TP_LLM_COMPILE_DYNAMIC", False),
                    "compile_disable_cudagraph_trees": _env_bool(
                        "DMUON_TP_LLM_COMPILE_DISABLE_CUDAGRAPH_TREES", True
                    ),
                },
                "summary": summary,
                "loss_reference": _summarize_loss_reference(rank_payloads),
                "torch_profile_trace": trace_path,
                "optimizer_step_profiles": optimizer_profiles,
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
