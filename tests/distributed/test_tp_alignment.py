"""TP topology alignment instrument.

Records deterministic loss trajectories for one topology/mode/baseline run.
The parent ``run_tp_alignment.sh`` drives the matrix and compares JSON
artifacts.  This file intentionally stays as a single reusable harness so
new TP topology parity checks do not duplicate model/setup code.

Environment:

* ``DMUON_ALIGN_TOPOLOGY``:
  ``tp2`` (2 GPU), ``tp4`` (4 GPU), ``dp_tp`` / ``dp_tp2`` (4 GPU),
  ``dp_tp4`` (8 GPU), ``hsdp_tp`` / ``hsdp_tp2`` (8 GPU), ``tp1`` (4 GPU),
  or ``dp_only`` (4 GPU).
* ``DMUON_ALIGN_MODE``:
  ``sync``, ``async``, or ``async_drain``.
* ``DMUON_ALIGN_OWNER``:
  ``lpt`` (default) or ``rank0``.  ``rank0`` is a private benchmark-only
  baseline for proving LPT preserves the loss trajectory while changing
  TP-owner placement; it is not public API.
* ``DMUON_ALIGN_MODEL``:
  ``tiny`` (default), ``llama``, or ``qwen``.  ``tiny`` keeps the original
  synthetic MLP-block workload.  ``llama``/``qwen`` run a small random-init
  HuggingFace CausalLM while reusing this same topology harness.
* ``DMUON_ALIGN_TP_SCOPE``:
  ``mlp`` or ``full``.  ``full`` tensor-parallelizes attention q/k/v/o plus
  MLP gate/up/down for CausalLM workloads.
* ``DMUON_ALIGN_STEPS``:
  number of optimizer steps, default 3.
* ``DMUON_ALIGN_OUT`` / ``DMUON_ALIGN_RUN``:
  JSON output directory and run identifier.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch.distributed import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from tp_profile_utils import collect_tp_profile, iter_dedicated_params

import dmuon


class MLP(nn.Module):
    def __init__(self, h=256, inter=1024):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, h=256, inter=1024):
        super().__init__()
        self.ln = nn.LayerNorm(h)
        self.mlp = MLP(h, inter)

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class Tiny(nn.Module):
    def __init__(self, num_layers=2, h=256, inter=1024):
        super().__init__()
        self.layers = nn.ModuleList([Block(h, inter) for _ in range(num_layers)])
        self.out = nn.Linear(h, h, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x).mean()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _parallelize_tiny(model: nn.Module, tp_mesh) -> None:
    plan = {
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }
    for layer in model.layers:
        parallelize_module(layer, tp_mesh, plan)


def _llm_config(model_kind: str) -> dict[str, Any]:
    if model_kind == "llama":
        cfg = {
            "family": "llama",
            "hidden": _env_int("DMUON_ALIGN_HIDDEN", 256),
            "intermediate": _env_int("DMUON_ALIGN_INTER", 1024),
            "layers": _env_int("DMUON_ALIGN_LAYERS", 2),
            "heads": _env_int("DMUON_ALIGN_HEADS", 8),
            "kv_heads": _env_int("DMUON_ALIGN_KV_HEADS", 4),
            "vocab": _env_int("DMUON_ALIGN_VOCAB", 1024),
            "max_positions": _env_int("DMUON_ALIGN_MAX_POSITIONS", 128),
        }
    elif model_kind == "qwen":
        cfg = {
            "family": "qwen2",
            "hidden": _env_int("DMUON_ALIGN_HIDDEN", 256),
            "intermediate": _env_int("DMUON_ALIGN_INTER", 1024),
            "layers": _env_int("DMUON_ALIGN_LAYERS", 2),
            "heads": _env_int("DMUON_ALIGN_HEADS", 8),
            "kv_heads": _env_int("DMUON_ALIGN_KV_HEADS", 2),
            "vocab": _env_int("DMUON_ALIGN_VOCAB", 1024),
            "max_positions": _env_int("DMUON_ALIGN_MAX_POSITIONS", 128),
        }
    else:
        raise ValueError(f"unknown CausalLM align model={model_kind!r}")
    cfg["max_full_tp_size"] = math.gcd(
        math.gcd(cfg["hidden"], cfg["intermediate"]),
        math.gcd(cfg["heads"], cfg["kv_heads"]),
    )
    return cfg


def _build_llm(model_kind: str, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    cfg = _llm_config(model_kind)
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
            tie_word_embeddings=False,
            attention_dropout=0.0,
        )
        hf_config._attn_implementation = "eager"
        return LlamaForCausalLM(hf_config).to(device), cfg
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
            tie_word_embeddings=False,
            attention_dropout=0.0,
        )
        hf_config._attn_implementation = "eager"
        return Qwen2ForCausalLM(hf_config).to(device), cfg
    raise ValueError(f"unsupported CausalLM family={cfg['family']!r}")


def _parallelize_llm(model: nn.Module, tp_mesh, tp_scope: str) -> None:
    if tp_scope not in ("mlp", "full"):
        raise ValueError("DMUON_ALIGN_TP_SCOPE must be 'mlp' or 'full'")

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
        if tp_scope == "full":
            parallelize_module(layer.self_attn, tp_mesh, attn_plan)


def _validate_tp_config(
    model_kind: str,
    model_cfg: dict[str, Any],
    tp_size: int,
    tp_scope: str,
) -> None:
    if tp_size <= 1:
        return
    if model_cfg["intermediate"] % tp_size != 0:
        raise ValueError(
            f"intermediate={model_cfg['intermediate']} must be divisible by "
            f"tp_size={tp_size}"
        )
    if model_cfg["hidden"] % tp_size != 0:
        raise ValueError(
            f"hidden={model_cfg['hidden']} must be divisible by tp_size={tp_size}"
        )
    if model_kind in ("llama", "qwen") and tp_scope == "full":
        if model_cfg["heads"] % tp_size != 0:
            raise ValueError(
                f"heads={model_cfg['heads']} must be divisible by "
                f"tp_size={tp_size}"
            )
        if model_cfg["kv_heads"] % tp_size != 0:
            raise ValueError(
                f"kv_heads={model_cfg['kv_heads']} must be divisible by "
                f"tp_size={tp_size}"
            )


def _parallelize_model(
    model: nn.Module,
    tp_mesh,
    *,
    model_kind: str,
    model_cfg: dict[str, Any],
    tp_scope: str,
) -> None:
    tp_size = int(tp_mesh.size())
    _validate_tp_config(model_kind, model_cfg, tp_size, tp_scope)
    if tp_size <= 1:
        return
    if model_kind == "tiny":
        _parallelize_tiny(model, tp_mesh)
    else:
        _parallelize_llm(model, tp_mesh, tp_scope)


def _fsdp_units(model: nn.Module, model_kind: str):
    if model_kind == "tiny":
        return model.layers
    return model.model.layers


def _force_tp_owner_rank0(model: nn.Module) -> None:
    """Private alignment-only baseline: collapse TP ownership to rank 0."""
    for dp in iter_dedicated_params(model):
        tp_group = getattr(dp, "tp_group", None)
        if tp_group is None:
            continue
        dp._tp_owner_local_rank = 0
        dp._tp_owner_global_rank = dist.get_global_rank(tp_group, 0)
        dp.is_tp_owner = tp_group.rank() == 0


def _build_model(
    topology: str,
    *,
    world_size: int,
    device: torch.device,
    model_kind: str,
    tp_scope: str,
    h: int = 256,
    inter: int = 1024,
    num_layers: int = 2,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build one deterministic topology variant."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if model_kind == "tiny":
        model_cfg = {
            "family": "tiny",
            "hidden": h,
            "intermediate": inter,
            "layers": num_layers,
        }
        model = Tiny(num_layers=num_layers, h=h, inter=inter).to(device)
    elif model_kind in ("llama", "qwen"):
        model, model_cfg = _build_llm(model_kind, device)
    else:
        raise ValueError(
            "DMUON_ALIGN_MODEL must be one of: tiny | llama | qwen"
        )

    if topology in ("hsdp_tp", "hsdp_tp2"):
        if world_size != 8:
            raise RuntimeError(f"hsdp_tp needs world=8, got {world_size}")
        mesh = init_device_mesh(
            "cuda", (2, 2, 2), mesh_dim_names=("replicate", "shard", "tp")
        )
        _parallelize_model(
            model,
            mesh["tp"],
            model_kind=model_kind,
            model_cfg=model_cfg,
            tp_scope=tp_scope,
        )
        dmuon.dedicate_params(
            model,
            mesh["shard"],
            replicate_mesh=mesh["replicate"],
            predicate=lambda n, p: "proj" in n and p.ndim == 2,
        )
        fsdp_mesh = mesh["replicate", "shard"]
    elif topology in ("dp_tp", "dp_tp2"):
        if world_size != 4:
            raise RuntimeError(f"dp_tp needs world=4, got {world_size}")
        mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
        _parallelize_model(
            model,
            mesh["tp"],
            model_kind=model_kind,
            model_cfg=model_cfg,
            tp_scope=tp_scope,
        )
        dmuon.dedicate_params(
            model,
            mesh["dp"],
            predicate=lambda n, p: "proj" in n and p.ndim == 2,
        )
        fsdp_mesh = mesh["dp"]
    elif topology == "dp_tp4":
        if world_size != 8:
            raise RuntimeError(f"dp_tp4 needs world=8, got {world_size}")
        mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
        _parallelize_model(
            model,
            mesh["tp"],
            model_kind=model_kind,
            model_cfg=model_cfg,
            tp_scope=tp_scope,
        )
        dmuon.dedicate_params(
            model,
            mesh["dp"],
            predicate=lambda n, p: "proj" in n and p.ndim == 2,
        )
        fsdp_mesh = mesh["dp"]
    elif topology == "tp2":
        if world_size != 2:
            raise RuntimeError(f"tp2 needs world=2, got {world_size}")
        mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("dp", "tp"))
        _parallelize_model(
            model,
            mesh["tp"],
            model_kind=model_kind,
            model_cfg=model_cfg,
            tp_scope=tp_scope,
        )
        dmuon.dedicate_params(
            model,
            mesh["dp"],
            predicate=lambda n, p: "proj" in n and p.ndim == 2,
        )
        fsdp_mesh = mesh["dp"]
    elif topology == "tp4":
        if world_size != 4:
            raise RuntimeError(f"tp4 needs world=4, got {world_size}")
        mesh = init_device_mesh("cuda", (1, 4), mesh_dim_names=("dp", "tp"))
        _parallelize_model(
            model,
            mesh["tp"],
            model_kind=model_kind,
            model_cfg=model_cfg,
            tp_scope=tp_scope,
        )
        dmuon.dedicate_params(
            model,
            mesh["dp"],
            predicate=lambda n, p: "proj" in n and p.ndim == 2,
        )
        fsdp_mesh = mesh["dp"]
    elif topology == "tp1":
        if world_size != 4:
            raise RuntimeError(f"tp1 needs world=4, got {world_size}")
        mesh = init_device_mesh("cuda", (4, 1), mesh_dim_names=("dp", "tp"))
        # Keep TP size-1 as an inactive axis, matching test_tp_correctness.py.
        dmuon.dedicate_params(
            model,
            mesh["dp"],
            predicate=lambda n, p: "proj" in n and p.ndim == 2,
        )
        fsdp_mesh = mesh["dp"]
    elif topology == "dp_only":
        if world_size != 4:
            raise RuntimeError(f"dp_only needs world=4, got {world_size}")
        mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp",))
        dmuon.dedicate_params(
            model,
            mesh["dp"],
            predicate=lambda n, p: "proj" in n and p.ndim == 2,
        )
        fsdp_mesh = mesh["dp"]
    else:
        raise RuntimeError(
            "DMUON_ALIGN_TOPOLOGY must be one of: "
            "tp2 | tp4 | dp_tp | dp_tp2 | dp_tp4 | "
            "hsdp_tp | hsdp_tp2 | tp1 | dp_only"
        )

    for layer in _fsdp_units(model, model_kind):
        fully_shard(layer, mesh=fsdp_mesh)
    fully_shard(model, mesh=fsdp_mesh)
    return model, model_cfg


def _make_inputs(
    *,
    model_kind: str,
    model_cfg: dict[str, Any],
    steps: int,
    device: torch.device,
) -> list[Any]:
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    if model_kind == "tiny":
        hidden = int(model_cfg["hidden"])
        return [torch.randn(4, 16, hidden, device=device) for _ in range(steps)]

    batch = _env_int("DMUON_ALIGN_BATCH", 2)
    seq = _env_int("DMUON_ALIGN_SEQ", 32)
    vocab = int(model_cfg["vocab"])
    batches: list[dict[str, torch.Tensor]] = []
    for _ in range(steps):
        input_ids = torch.randint(0, vocab, (batch, seq), device=device)
        labels = torch.randint(0, vocab, (batch, seq), device=device)
        batches.append({"input_ids": input_ids, "labels": labels})
    return batches


def _compute_loss(model: nn.Module, batch: Any, model_kind: str) -> torch.Tensor:
    if model_kind == "tiny":
        return model(batch)
    return model(**batch).loss


def main() -> int:
    topology = os.environ.get("DMUON_ALIGN_TOPOLOGY", "hsdp_tp")
    mode = os.environ.get("DMUON_ALIGN_MODE", "sync")
    owner_mode = os.environ.get("DMUON_ALIGN_OWNER", "lpt")
    model_kind = os.environ.get("DMUON_ALIGN_MODEL", "tiny")
    default_tp_scope = "mlp" if model_kind == "tiny" else "full"
    tp_scope = os.environ.get("DMUON_ALIGN_TP_SCOPE", default_tp_scope)
    run_id = os.environ.get("DMUON_ALIGN_RUN", "0")
    out_dir = os.environ.get("DMUON_ALIGN_OUT", "/tmp/dmuon_align")
    steps = int(os.environ.get("DMUON_ALIGN_STEPS", "3"))

    assert mode in ("sync", "async", "async_drain"), (
        f"unknown DMUON_ALIGN_MODE={mode!r}"
    )
    assert owner_mode in ("lpt", "rank0"), (
        f"unknown DMUON_ALIGN_OWNER={owner_mode!r}"
    )

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    os.environ.setdefault("DMUON_NS_KERNEL", "cublas")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    model, model_cfg = _build_model(
        topology,
        world_size=world_size,
        device=device,
        model_kind=model_kind,
        tp_scope=tp_scope,
    )
    if owner_mode == "rank0":
        _force_tp_owner_rank0(model)

    replicate_async = mode in ("async", "async_drain")
    optimizer = dmuon.Muon(
        model, lr=0.02, momentum=0.95, weight_decay=0.01,
        adamw_lr=1e-3, replicate_async=replicate_async,
    )
    profile = collect_tp_profile(
        model,
        scenario=f"{model_kind}_{topology}_{owner_mode}_{tp_scope}",
        replicate_async=replicate_async,
    )

    # Fixed seed for inputs: every rank sees the same batch per iteration.
    inputs = _make_inputs(
        model_kind=model_kind,
        model_cfg=model_cfg,
        steps=steps,
        device=device,
    )

    losses: list[float] = []
    weight_digests: list[float] = []
    digest_each_step = not (
        mode == "async"
        and not bool(int(os.environ.get("DMUON_ALIGN_ASYNC_DIGEST_EACH_STEP", "0") or 0))
    )
    param_digest_enabled = bool(
        int(os.environ.get("DMUON_ALIGN_PARAM_DIGEST", "0") or 0)
    )
    final_param_digest_enabled = bool(
        int(os.environ.get("DMUON_ALIGN_FINAL_PARAM_DIGEST", "0") or 0)
    )
    param_digests: list[dict[str, float]] = []

    def _digest() -> float:
        """Global owned-data digest after pending async work is visible."""
        torch.cuda.synchronize()
        dist.barrier(device_ids=[device.index])
        total = torch.zeros((), device=device, dtype=torch.float64)
        for dp in optimizer._dedicated_params:
            data = getattr(dp, "_owned_data", None)
            if data is not None:
                total += data.float().sum().double()
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        return float(total.item())

    def _param_digest() -> dict[str, float]:
        """Per-parameter global owned-data digest for alignment diagnostics."""
        torch.cuda.synchronize()
        dist.barrier(device_ids=[device.index])
        out: dict[str, float] = {}
        module_names = {id(module): name for name, module in model.named_modules()}
        for idx, dp in enumerate(iter_dedicated_params(model)):
            val = torch.zeros((), device=device, dtype=torch.float64)
            data = getattr(dp, "_owned_data", None)
            if data is not None:
                val += data.float().sum().double()
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            full_shape = tuple(int(x) for x in getattr(dp, "full_shape", ()))
            shard_dim = getattr(dp, "shard_dim", None)
            owner = getattr(dp, "_tp_owner_local_rank", None)
            prefix = module_names.get(id(getattr(dp, "module", None)), "")
            param_name = str(getattr(dp, "param_name", "<unknown>"))
            name = f"{prefix}.{param_name}" if prefix else param_name
            key = f"{idx:04d}:{name}:shape={full_shape}:shard={shard_dim}:owner={owner}"
            out[key] = float(val.item())
        return out

    from dmuon.utils import wait_all_replicate_broadcasts

    for it, x in enumerate(inputs):
        optimizer.zero_grad()
        loss = _compute_loss(model, x, model_kind)
        loss.backward()
        optimizer.step()
        if mode == "async_drain":
            wait_all_replicate_broadcasts(model)
        losses.append(float(loss.item()))
        if digest_each_step:
            # This inserts a WORLD all-reduce. Raw async mode deliberately
            # skips it: post-step TP/HSDP work is a next-forward dependency,
            # so out-of-band collectives must drain first or they can perturb
            # cross-process-group ordering.
            digest = _digest()
            weight_digests.append(digest)
            digest_text = f"{digest:.12e}"
        else:
            digest_text = "deferred"
        if param_digest_enabled:
            param_digests.append(_param_digest())
        if rank == 0:
            print(
                f"[{topology}/{owner_mode}/{mode} run={run_id}] iter {it}: "
                f"loss={loss.item():.10f} digest={digest_text}",
                flush=True,
            )
    torch.cuda.synchronize()

    # Drain any pending async state before teardown.
    wait_all_replicate_broadcasts(model)
    final_weight_digest = _digest()
    final_param_digest = (
        _param_digest() if final_param_digest_enabled else {}
    )

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"{topology}_{owner_mode}_{mode}_{run_id}.json"
        )
        with open(out_path, "w") as f:
            json.dump({
                "model": model_kind,
                "model_config": model_cfg,
                "tp_scope": tp_scope,
                "topology": topology,
                "mode": mode,
                "owner_mode": owner_mode,
                "run_id": run_id,
                "world_size": world_size,
                "tp_param_count": profile["tp_param_count"],
                "owner_coverage": profile["owner_coverage"],
                "owner_load_by_tp_rank": profile["owner_load_by_tp_rank"],
                "losses": losses,
                "weight_digest": weight_digests,
                "weight_digest_each_step": digest_each_step,
                "final_weight_digest": final_weight_digest,
                "param_digest": param_digests,
                "final_param_digest": final_param_digest,
            }, f)
        print(f"wrote {out_path}", flush=True)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
