"""Benchmark-only fused manual TP modules for HF Llama/Qwen models.

This backend is deliberately isolated from the DMuon core API.  It validates
whether grouping the column-parallel input-gradient all-reduces for
``q/k/v`` and ``gate/up`` improves the TP benchmark before we decide whether
to productionize the path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _slice_range(size: int, tp_rank: int, tp_size: int) -> tuple[int, int]:
    if size % tp_size != 0:
        raise ValueError(f"cannot shard size={size} across tp_size={tp_size}")
    part = size // tp_size
    start = tp_rank * part
    return start, start + part


def _local_parameter(
    tensor: torch.Tensor,
    *,
    dim: int,
    tp_rank: int,
    tp_size: int,
) -> nn.Parameter:
    start, end = _slice_range(int(tensor.shape[dim]), tp_rank, tp_size)
    shard = tensor.detach().narrow(dim, start, end - start).clone()
    return nn.Parameter(shard)


def _all_reduce_inplace(tensor: torch.Tensor, group) -> torch.Tensor:
    if group is not None and group.size() > 1:
        dist.all_reduce(tensor, group=group)
    return tensor


class _FusedQKVFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias, group):
        x2 = x.reshape(-1, x.shape[-1])
        q = x2.matmul(q_weight.t())
        k = x2.matmul(k_weight.t())
        v = x2.matmul(v_weight.t())
        if q_bias is not None:
            q = q + q_bias
        if k_bias is not None:
            k = k + k_bias
        if v_bias is not None:
            v = v + v_bias
        q = q.reshape(*x.shape[:-1], q_weight.shape[0])
        k = k.reshape(*x.shape[:-1], k_weight.shape[0])
        v = v.reshape(*x.shape[:-1], v_weight.shape[0])
        ctx.save_for_backward(x, q_weight, k_weight, v_weight)
        ctx.has_bias = (q_bias is not None, k_bias is not None, v_bias is not None)
        ctx.group = group
        return q, k, v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        x, q_weight, k_weight, v_weight = ctx.saved_tensors
        x2 = x.reshape(-1, x.shape[-1])
        grad_q2 = grad_q.contiguous().reshape(-1, q_weight.shape[0])
        grad_k2 = grad_k.contiguous().reshape(-1, k_weight.shape[0])
        grad_v2 = grad_v.contiguous().reshape(-1, v_weight.shape[0])

        grad_x2 = (
            grad_q2.matmul(q_weight)
            + grad_k2.matmul(k_weight)
            + grad_v2.matmul(v_weight)
        )
        _all_reduce_inplace(grad_x2, ctx.group)

        grad_q_weight = grad_q2.t().matmul(x2)
        grad_k_weight = grad_k2.t().matmul(x2)
        grad_v_weight = grad_v2.t().matmul(x2)
        grad_q_bias = grad_q2.sum(dim=0) if ctx.has_bias[0] else None
        grad_k_bias = grad_k2.sum(dim=0) if ctx.has_bias[1] else None
        grad_v_bias = grad_v2.sum(dim=0) if ctx.has_bias[2] else None
        return (
            grad_x2.reshape_as(x),
            grad_q_weight,
            grad_k_weight,
            grad_v_weight,
            grad_q_bias,
            grad_k_bias,
            grad_v_bias,
            None,
        )


class _FusedGateUpFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, gate_bias, up_bias, group):
        x2 = x.reshape(-1, x.shape[-1])
        gate = x2.matmul(gate_weight.t())
        up = x2.matmul(up_weight.t())
        if gate_bias is not None:
            gate = gate + gate_bias
        if up_bias is not None:
            up = up + up_bias
        gate = gate.reshape(*x.shape[:-1], gate_weight.shape[0])
        up = up.reshape(*x.shape[:-1], up_weight.shape[0])
        ctx.save_for_backward(x, gate_weight, up_weight)
        ctx.has_bias = (gate_bias is not None, up_bias is not None)
        ctx.group = group
        return gate, up

    @staticmethod
    def backward(ctx, grad_gate, grad_up):
        x, gate_weight, up_weight = ctx.saved_tensors
        x2 = x.reshape(-1, x.shape[-1])
        grad_gate2 = grad_gate.contiguous().reshape(-1, gate_weight.shape[0])
        grad_up2 = grad_up.contiguous().reshape(-1, up_weight.shape[0])

        grad_x2 = grad_gate2.matmul(gate_weight) + grad_up2.matmul(up_weight)
        _all_reduce_inplace(grad_x2, ctx.group)

        grad_gate_weight = grad_gate2.t().matmul(x2)
        grad_up_weight = grad_up2.t().matmul(x2)
        grad_gate_bias = grad_gate2.sum(dim=0) if ctx.has_bias[0] else None
        grad_up_bias = grad_up2.sum(dim=0) if ctx.has_bias[1] else None
        return (
            grad_x2.reshape_as(x),
            grad_gate_weight,
            grad_up_weight,
            grad_gate_bias,
            grad_up_bias,
            None,
        )


class _RowParallelLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, group):
        out = F.linear(x, weight)
        _all_reduce_inplace(out, group)
        if bias is not None:
            out = out + bias
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_out2 = grad_out.contiguous().reshape(-1, grad_out.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])
        grad_x = grad_out2.matmul(weight).reshape_as(x)
        grad_weight = grad_out2.t().matmul(x2)
        grad_bias = grad_out2.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_weight, grad_bias, None


class RowParallelLinear(nn.Module):
    """Linear whose input dimension is sharded and whose output is replicated."""

    def __init__(
        self,
        source: nn.Linear,
        *,
        tp_rank: int,
        tp_size: int,
        group,
    ) -> None:
        super().__init__()
        self.weight = _local_parameter(
            source.weight,
            dim=1,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.bias = (
            nn.Parameter(source.bias.detach().clone())
            if source.bias is not None
            else None
        )
        self.group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RowParallelLinearFn.apply(x, self.weight, self.bias, self.group)


@dataclass(frozen=True)
class FusedManualStats:
    layers: int
    tp_size: int
    expected_allreduces_per_layer: int = 4


class FusedManualMLP(nn.Module):
    def __init__(self, source: nn.Module, *, tp_rank: int, tp_size: int, group) -> None:
        super().__init__()
        self.gate_weight = _local_parameter(
            source.gate_proj.weight,
            dim=0,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.up_weight = _local_parameter(
            source.up_proj.weight,
            dim=0,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.gate_bias = (
            _local_parameter(
                source.gate_proj.bias,
                dim=0,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
            if source.gate_proj.bias is not None
            else None
        )
        self.up_bias = (
            _local_parameter(
                source.up_proj.bias,
                dim=0,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
            if source.up_proj.bias is not None
            else None
        )
        self.down_proj = RowParallelLinear(
            source.down_proj,
            tp_rank=tp_rank,
            tp_size=tp_size,
            group=group,
        )
        self.act_fn = source.act_fn
        self.group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = _FusedGateUpFn.apply(
            x,
            self.gate_weight,
            self.up_weight,
            self.gate_bias,
            self.up_bias,
            self.group,
        )
        return self.down_proj(self.act_fn(gate) * up)


class FusedManualAttention(nn.Module):
    def __init__(
        self,
        source: nn.Module,
        *,
        tp_rank: int,
        tp_size: int,
        group,
    ) -> None:
        super().__init__()
        self.config = source.config
        self.layer_idx = source.layer_idx
        self.head_dim = source.head_dim
        global_heads = int(getattr(source, "num_heads", source.config.num_attention_heads))
        global_kv_heads = int(
            getattr(source, "num_key_value_heads", source.config.num_key_value_heads)
        )
        self.num_heads = global_heads // tp_size
        self.num_key_value_heads = global_kv_heads // tp_size
        self.num_key_value_groups = global_heads // global_kv_heads
        self.scaling = source.scaling
        self.attention_dropout = source.attention_dropout
        self.sliding_window = getattr(source, "sliding_window", None)
        self.is_causal = getattr(source, "is_causal", True)
        self.group = group

        self.q_weight = _local_parameter(
            source.q_proj.weight,
            dim=0,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.k_weight = _local_parameter(
            source.k_proj.weight,
            dim=0,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.v_weight = _local_parameter(
            source.v_proj.weight,
            dim=0,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.q_bias = (
            _local_parameter(source.q_proj.bias, dim=0, tp_rank=tp_rank, tp_size=tp_size)
            if source.q_proj.bias is not None
            else None
        )
        self.k_bias = (
            _local_parameter(source.k_proj.bias, dim=0, tp_rank=tp_rank, tp_size=tp_size)
            if source.k_proj.bias is not None
            else None
        )
        self.v_bias = (
            _local_parameter(source.v_proj.bias, dim=0, tp_rank=tp_rank, tp_size=tp_size)
            if source.v_proj.bias is not None
            else None
        )
        self.o_proj = RowParallelLinear(
            source.o_proj,
            tp_rank=tp_rank,
            tp_size=tp_size,
            group=group,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Any = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        if past_key_value is not None:
            raise ValueError("fused_manual TP benchmark does not support KV cache")

        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb,
            repeat_kv,
        )

        input_shape = hidden_states.shape[:-1]
        q, k, v = _FusedQKVFn.apply(
            hidden_states,
            self.q_weight,
            self.k_weight,
            self.v_weight,
            self.q_bias,
            self.k_bias,
            self.v_bias,
            self.group,
        )
        query_states = q.view(*input_shape, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = k.view(
            *input_shape, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = v.view(
            *input_shape, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            scale=self.scaling,
            is_causal=query_states.shape[2] > 1
            and attention_mask is None
            and self.is_causal,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None


def apply_fused_manual_tp(model: nn.Module, tp_mesh, mode: str) -> FusedManualStats:
    if mode not in ("mlp", "full"):
        raise ValueError("fused_manual TP mode must be 'mlp' or 'full'")
    group = tp_mesh.get_group()
    tp_rank = int(tp_mesh.get_local_rank())
    tp_size = int(tp_mesh.size())
    layer_count = 0
    for layer in model.model.layers:
        layer.mlp = FusedManualMLP(
            layer.mlp,
            tp_rank=tp_rank,
            tp_size=tp_size,
            group=group,
        )
        if mode == "full":
            layer.self_attn = FusedManualAttention(
                layer.self_attn,
                tp_rank=tp_rank,
                tp_size=tp_size,
                group=group,
            )
        layer_count += 1
    expected = 4 if mode == "full" else 2
    return FusedManualStats(
        layers=layer_count,
        tp_size=tp_size,
        expected_allreduces_per_layer=expected,
    )
