"""Correctness smoke for the benchmark-only fused manual TP backend.

Run with:
    torchrun --nproc_per_node=2 tests/distributed/test_tp_fused_manual.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

from bench_tp_llm import build_model  # noqa: E402
from tp_fused_manual import apply_fused_manual_tp  # noqa: E402


def _configure_tiny_llm() -> None:
    os.environ["DMUON_TP_LLM_LAYERS"] = "2"
    os.environ["DMUON_TP_LLM_HIDDEN"] = "128"
    os.environ["DMUON_TP_LLM_INTER"] = "256"
    os.environ["DMUON_TP_LLM_HEADS"] = "4"
    os.environ["DMUON_TP_LLM_KV_HEADS"] = "2"
    os.environ["DMUON_TP_LLM_VOCAB"] = "512"
    os.environ["DMUON_TP_LLM_MAX_POSITIONS"] = "64"
    os.environ["DMUON_TP_LLM_ATTN_IMPL"] = "sdpa"


def _gather_shards(tensor: torch.Tensor, *, dim: int, group) -> torch.Tensor:
    shards = [torch.empty_like(tensor) for _ in range(group.size())]
    dist.all_gather(shards, tensor.contiguous(), group=group)
    return torch.cat(shards, dim=dim)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max().item())


def _check_model(model_key: str, device: torch.device, tp_mesh) -> dict[str, float]:
    _configure_tiny_llm()
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    ref, cfg = build_model(model_key, device)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    fused, _ = build_model(model_key, device)
    apply_fused_manual_tp(fused, tp_mesh, "full")
    ref.eval()
    fused.eval()

    torch.manual_seed(4321)
    input_ids = torch.randint(0, cfg["vocab"], (2, 16), device=device)
    labels = torch.randint(0, cfg["vocab"], (2, 16), device=device)

    ref.zero_grad(set_to_none=True)
    fused.zero_grad(set_to_none=True)
    ref_loss = ref(input_ids=input_ids, labels=labels).loss
    fused_loss = fused(input_ids=input_ids, labels=labels).loss
    ref_loss.backward()
    fused_loss.backward()

    loss = float(fused_loss.detach().item())
    expected = math.log(float(cfg["vocab"]))
    if not math.isfinite(loss) or not (0.25 * expected <= loss <= 4.0 * expected):
        raise AssertionError(
            f"{model_key} fused loss {loss:.6f} outside reasonable range "
            f"around log(vocab)={expected:.6f}"
        )

    tp_group = tp_mesh.get_group()
    ref_layer = ref.model.layers[0]
    fused_layer = fused.model.layers[0]
    checks = {
        "loss_abs": abs(float(ref_loss.detach().item()) - loss),
        "q_weight_grad_abs": _max_abs(
            ref_layer.self_attn.q_proj.weight.grad,
            _gather_shards(fused_layer.self_attn.q_weight.grad, dim=0, group=tp_group),
        ),
        "k_weight_grad_abs": _max_abs(
            ref_layer.self_attn.k_proj.weight.grad,
            _gather_shards(fused_layer.self_attn.k_weight.grad, dim=0, group=tp_group),
        ),
        "v_weight_grad_abs": _max_abs(
            ref_layer.self_attn.v_proj.weight.grad,
            _gather_shards(fused_layer.self_attn.v_weight.grad, dim=0, group=tp_group),
        ),
        "o_weight_grad_abs": _max_abs(
            ref_layer.self_attn.o_proj.weight.grad,
            _gather_shards(fused_layer.self_attn.o_proj.weight.grad, dim=1, group=tp_group),
        ),
        "gate_weight_grad_abs": _max_abs(
            ref_layer.mlp.gate_proj.weight.grad,
            _gather_shards(fused_layer.mlp.gate_weight.grad, dim=0, group=tp_group),
        ),
        "up_weight_grad_abs": _max_abs(
            ref_layer.mlp.up_proj.weight.grad,
            _gather_shards(fused_layer.mlp.up_weight.grad, dim=0, group=tp_group),
        ),
        "down_weight_grad_abs": _max_abs(
            ref_layer.mlp.down_proj.weight.grad,
            _gather_shards(fused_layer.mlp.down_proj.weight.grad, dim=1, group=tp_group),
        ),
    }
    if fused_layer.self_attn.q_bias is not None:
        checks["q_bias_grad_abs"] = _max_abs(
            ref_layer.self_attn.q_proj.bias.grad,
            _gather_shards(fused_layer.self_attn.q_bias.grad, dim=0, group=tp_group),
        )
    max_abs = max(checks.values())
    if max_abs > 5e-4:
        raise AssertionError(f"{model_key} fused_manual mismatch: {checks}")
    checks["loss"] = loss
    checks["expected_loss_center"] = expected
    return checks


def main() -> int:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world = dist.get_world_size()
    if world not in (2, 4):
        raise RuntimeError(f"expected TP world size 2 or 4, got {world}")
    mesh = init_device_mesh("cuda", (1, world), mesh_dim_names=("dp", "tp"))
    try:
        results = {
            "qwen": _check_model("qwen1b", device, mesh["tp"]),
            "llama": _check_model("llama3b", device, mesh["tp"]),
        }
        if dist.get_rank() == 0:
            print(json.dumps(results, indent=2, sort_keys=True))
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
