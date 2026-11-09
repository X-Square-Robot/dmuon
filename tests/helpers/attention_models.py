"""Small differentiable attention references for layout/runtime integration.

These preserve the researched projection orders, not the complete released
models. All math is ordinary PyTorch. Optional official-kernel comparisons live
in a separate experiment, so passing these tests is not upstream compatibility.
"""

from dataclasses import dataclass
import io
import math

import torch
from torch import nn
from torch.nn import functional as F

import dmuon
from dmuon import MatrixLayout
from tests.helpers.attention_layouts import LayoutCase, head_blocks, row_blocks


class Float32NS(dmuon.NewtonSchulz):
    """Keep tiny integration gradients away from FP16 rounding boundaries.

    This tests layout/runtime bookkeeping, not the production NS implementation;
    that implementation has its own tests and separate GPU experiments.
    """

    def __init__(self):
        from dmuon.optim.newton_schulz import DEFAULT_COEFFICIENTS

        super().__init__("direct", kernel="cublas", coefficients=DEFAULT_COEFFICIENTS)

    def local(self, gradient, steps):
        x = gradient.float().contiguous()
        transposed = x.shape[0] > x.shape[1]
        if transposed:
            x = x.t()
        x = x / (x.norm() + 1e-7)
        for a, b, c in self.coefficients[:steps]:
            gram = x @ x.t()
            x = a * x + (b * gram + c * (gram @ gram)) @ x
        return x.t() if transposed else x


def linear(x, weight, dtype):
    return F.linear(x.to(dtype), weight.to(dtype)).float()


def attention(q, k, v, mask):
    scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, -torch.inf)
    return torch.einsum("bhts,bshv->bthv", scores.softmax(-1), v.float())


def causal_mask(t, s, device):
    return torch.arange(s, device=device)[None, :] <= torch.arange(t, device=device)[:, None]


def moba_mask(q, k, block=2, topk=2):
    """Parameter-free causal block selection, matching pinned MoBA naive logic."""
    t = q.shape[1]
    keys = torch.stack([part.mean(1) for part in k.split(block, dim=1)], dim=1)
    gate = torch.einsum("bthd,bnhd->bhtn", q.float(), keys.float())
    current = torch.arange(t, device=q.device) // block
    chunks = torch.arange(keys.shape[1], device=q.device)
    gate = gate.masked_fill(chunks[None, :] > current[:, None], -torch.inf)
    gate = gate.masked_fill(chunks[None, :] == current[:, None], torch.inf)
    selected = gate.topk(min(topk, keys.shape[1]), dim=-1).indices
    mask = torch.zeros_like(gate, dtype=torch.bool).scatter_(-1, selected, True)
    mask = mask.repeat_interleave(block, dim=-1)[..., :t]
    return mask & causal_mask(t, t, q.device)


def delta_reference(q, k, v, log_decay, beta, initial=None):
    """Channel-wise gated delta recurrence, inputs already normalized/gated."""
    b, t, h, d = q.shape
    state = (
        q.new_zeros(b, h, d, v.shape[-1], dtype=torch.float32)
        if initial is None
        else initial.float()
    )
    outputs = []
    for index in range(t):
        state = state * log_decay[:, index].float().exp().unsqueeze(-1)
        key = k[:, index].float()
        prediction = torch.einsum("bhd,bhdv->bhv", key, state)
        error = (v[:, index].float() - prediction) * beta[:, index].float().unsqueeze(-1)
        state = state + key.unsqueeze(-1) * error.unsqueeze(-2)
        outputs.append(torch.einsum("bhd,bhdv->bhv", q[:, index].float(), state))
    return torch.stack(outputs, dim=1), state


class TinyAttention(nn.Module):
    hidden = 16
    heads = 2
    nope = 4
    rope = 2
    value = 4
    qrank = 5
    krank = 4

    def __init__(self, kind, *, dtype=torch.float32, layout_mode="components", layer=0):
        super().__init__()
        self.kind, self.dtype, self.layout_mode = kind, dtype, layout_mode
        self.specs = {}
        H, N, Q, K = self.hidden, self.heads, self.qrank, self.krank
        if kind == "csa2":
            # Three layers exercise Full -> Reindex -> Reuse with shared,
            # differentiable compressed KV (ratio 2) and a local raw window.
            self.compresses, self.has_indexer = layer == 0, layer < 2
            D, O = self.nope + self.rope, 3
            self._parameter("q_a", (Q, H))
            self._head_parameter("q_b", self.nope, self.rope, Q)
            self._parameter("wkv", (D, H))
            self._parameter(
                "wo_a",
                (N * O, D),
                MatrixLayout.view(N, O, D),
                lambda w: list(w.reshape(N, O, D).unbind(0)),
            )
            self._parameter("wo_b", (H, N * O))
            if self.has_indexer:
                self._parameter(
                    "index_q",
                    (N * 4, Q),
                    MatrixLayout.view(N, 4, Q),
                    lambda w: list(w.reshape(N, 4, Q).unbind(0)),
                )
                self._parameter("index_weights", (N, H))
            if self.compresses:
                self._parameter(
                    "compress_fused", (2 * D, H), MatrixLayout.rows(D, D), row_blocks(D, D)
                )
                self._parameter("index_key", (4, D))
            self._parameter("unused", (3, H))
            return
        if kind in ("mla", "dsa"):
            self._parameter(
                "fused_a",
                (Q + K + self.rope, H),
                MatrixLayout.rows(Q, K + self.rope),
                row_blocks(Q, K + self.rope),
            )
            self._head_parameter("q_b", self.nope, self.rope, Q)
            self._head_parameter("kv_b", self.nope, self.value, K)
            self._parameter("o_proj", (H, N * self.value))
            if kind == "dsa":
                self._parameter(
                    "index_fused", (4 + N, H), MatrixLayout.rows(4, N), row_blocks(4, N)
                )
                self._parameter(
                    "index_q",
                    (N * 4, Q),
                    MatrixLayout.view(N, 4, Q),
                    lambda x: list(x.reshape(N, 4, Q).unbind(0)),
                )
        else:
            d = 4
            self.dim = d
            if kind == "kda":
                # Official K3 KDA has separate Q/K/V projection parameters.
                for name in ("q_proj", "k_proj", "v_proj"):
                    layout = (
                        MatrixLayout.view(N, d, H)
                        if layout_mode == "per_head"
                        else MatrixLayout.whole()
                    )
                    selector = (
                        (lambda x: list(x.reshape(N, d, H).unbind(0)))
                        if layout_mode == "per_head"
                        else lambda x: [x]
                    )
                    self._parameter(name, (N * d, H), layout, selector)
            elif layout_mode == "per_head":
                self._parameter(
                    "qkv",
                    (3 * N * d, H),
                    MatrixLayout.view(3, N, d, H),
                    lambda x: [x.reshape(3, N, d, H)[i, j] for i in range(3) for j in range(N)],
                )
            else:
                self._parameter(
                    "qkv",
                    (3 * N * d, H),
                    MatrixLayout.view(3, N * d, H),
                    lambda x: list(x.reshape(3, N * d, H).unbind(0)),
                )
            self._parameter("o_proj", (H, N * d))
            if kind == "kda":
                # Deliberately left on AdamW: 2D gates, vectors and 3D depthwise convolution.
                self.f_a = nn.Parameter(torch.randn(d, H) * 0.03)
                self.f_b = nn.Parameter(torch.randn(N * d, d) * 0.03)
                self.beta_proj = nn.Parameter(torch.randn(N, H) * 0.03)
                self.output_gate = nn.Parameter(torch.randn(N * d, H) * 0.03)
                self.A_log = nn.Parameter(torch.full((N,), -2.0))
                self.dt_bias = nn.Parameter(torch.zeros(N * d))
                for name in ("q_conv", "k_conv", "v_conv"):
                    setattr(self, name, nn.Parameter(torch.randn(N * d, 1, 3) * 0.03))
        # A selected but unused projection must retain grad=None and no optimizer state.
        self._parameter("unused", (3, H))

    def _parameter(self, name, shape, layout=None, selector=None):
        setattr(self, name, nn.Parameter(torch.randn(shape) * 0.06))
        if self.layout_mode == "whole" or layout is None:
            layout = MatrixLayout.whole()
            selector = lambda t: [t.view(t.shape[0], -1)]
        self.specs[name] = LayoutCase(name, shape, layout, selector, "tiny_trainable_reference")

    def _head_parameter(self, name, left, right, rank):
        layout = MatrixLayout.view(self.heads, left + right, rank).split((left, right), dim=1)
        whole = self.layout_mode != "per_head"
        if whole:
            layout = layout.reshape(-1, rank)
        self._parameter(
            name,
            (self.heads * (left + right), rank),
            layout,
            head_blocks(self.heads, left, right, rank, whole=whole),
        )

    def forward(self, x, memory=None):
        b, t, _ = x.shape
        N, H, dtype = self.heads, self.hidden, self.dtype
        if self.kind == "csa2":
            return self._csa2(x, memory)
        if self.kind in ("mla", "dsa"):
            source = x
            query_a = linear(x, self.fused_a[: self.qrank], dtype)
            latent_rope = linear(source, self.fused_a[self.qrank :], dtype)
            latent, rope_key = latent_rope.split((self.krank, self.rope), dim=-1)
            q = linear(query_a, self.q_b, dtype).reshape(b, t, N, self.nope + self.rope)
            kv = linear(latent, self.kv_b, dtype).reshape(
                b, source.shape[1], N, self.nope + self.value
            )
            key, value = kv.split((self.nope, self.value), dim=-1)
            key = torch.cat((key, rope_key.unsqueeze(2).expand(-1, -1, N, -1)), dim=-1)
            mask = causal_mask(t, source.shape[1], x.device)
            aux = x.new_zeros(())
            if self.kind != "mla":
                query_idx = linear(query_a, self.index_q, dtype).reshape(b, t, N, 4)
                key_idx, weights = linear(x, self.index_fused, dtype).split((4, N), -1)
                scores = (
                    torch.einsum("bthd,bsd->bhts", query_idx, key_idx).relu()
                    * weights.sigmoid().permute(0, 2, 1).unsqueeze(-1)
                ).sum(1)
                # Auxiliary differentiable loss trains the selector; topk.indices alone does not.
                aux = scores.square().mean() * 0.03 + query_idx.square().mean() * 0.001
                scores = scores.masked_fill(~causal_mask(t, t, x.device), -torch.inf)
                selected = scores.topk(min(2, t), dim=-1).indices
                sparse = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, selected, True)
                sparse |= torch.eye(t, device=x.device, dtype=torch.bool)
                # Memory in this tiny shared-source test has the same length as x.
                mask = (sparse & causal_mask(t, t, x.device)).unsqueeze(1)
            o = attention(q, key, value, mask).flatten(2)
            output = linear(o, self.o_proj, dtype)
            return output, memory, aux
        if self.kind == "kda":
            q, k, v = [
                linear(x, getattr(self, name), dtype).reshape(b, t, N, self.dim)
                for name in ("q_proj", "k_proj", "v_proj")
            ]
        else:
            q, k, v = linear(x, self.qkv, dtype).reshape(b, t, 3, N, self.dim).unbind(2)
        if self.kind == "moba":
            o = attention(q, k, v, moba_mask(q, k)).flatten(2)
            return linear(o, self.o_proj, dtype), memory, x.new_zeros(())
        if self.kind != "kda":
            raise ValueError(self.kind)
        projected = []
        for name, part in zip(("q_conv", "k_conv", "v_conv"), (q, k, v)):
            inp = part.flatten(2).transpose(1, 2).to(dtype)
            weight = getattr(self, name).to(dtype)
            projected.append(
                F.silu(F.conv1d(inp, weight, padding=2, groups=N * self.dim)[..., :t])
                .transpose(1, 2)
                .reshape(b, t, N, self.dim)
                .float()
            )
        q, k, v = projected
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        gate = linear(linear(x, self.f_a, dtype), self.f_b, dtype).reshape(b, t, N, self.dim)
        dt = self.dt_bias.to(dtype).float().reshape(N, self.dim)
        rate = self.A_log.to(dtype).float().exp().view(N, 1)
        log_decay = -F.softplus(gate + dt) * rate
        beta = linear(x, self.beta_proj, dtype).sigmoid()
        o, _ = delta_reference(q, k, v, log_decay, beta)
        o = o * linear(x, self.output_gate, dtype).reshape_as(o).sigmoid()
        return linear(o.flatten(2), self.o_proj, dtype), memory, x.new_zeros(())

    def _csa2(self, x, memory):
        b, t, _ = x.shape
        N, D, dtype = self.heads, self.nope + self.rope, self.dtype
        query_a = linear(x, self.q_a, dtype)
        q = linear(query_a, self.q_b, dtype).reshape(b, t, N, D)
        local = linear(x, self.wkv, dtype)
        if self.compresses:
            assert t % 2 == 0
            kv, scores = linear(x, self.compress_fused, dtype).chunk(2, -1)
            compressed = (
                kv.reshape(b, t // 2, 2, D) * scores.reshape(b, t // 2, 2, D).softmax(2)
            ).sum(2)
            keys = linear(compressed, self.index_key, dtype)
            indices = None
        else:
            assert memory is not None
            compressed, keys, indices = memory
        query_pos = torch.arange(t, device=x.device)
        compressed_pos = torch.arange(compressed.shape[1], device=x.device) * 2 + 1
        valid = compressed_pos[None, :] <= query_pos[:, None]
        aux = x.new_zeros(())
        if self.has_indexer:
            iq = linear(query_a, self.index_q, dtype).reshape(b, t, N, 4)
            weights = linear(x, self.index_weights, dtype).sigmoid()
            scores = (
                torch.einsum("bthd,bsd->bhts", iq, keys).relu()
                * weights.permute(0, 2, 1).unsqueeze(-1)
            ).sum(1)
            aux = scores.square().mean() * 0.03 + iq.square().mean() * 0.001
            indices = (
                scores.masked_fill(~valid, -torch.inf).topk(min(2, compressed.shape[1]), -1).indices
            )
        selected = torch.zeros(b, t, compressed.shape[1], device=x.device, dtype=torch.bool)
        selected.scatter_(-1, indices, True)
        selected &= valid
        raw_mask = causal_mask(t, t, x.device) & (query_pos[None, :] >= query_pos[:, None] - 1)
        mask = torch.cat((raw_mask.expand(b, -1, -1), selected), -1).unsqueeze(1)
        # KV is shared by heads; raw and compressed history share one attention.
        kv = torch.cat((local, compressed), dim=1).unsqueeze(2).expand(-1, -1, N, -1)
        o = attention(q, kv, kv, mask)
        groups = o.reshape(b, t, N, D)
        projected = torch.einsum(
            "btgd,grd->btgr", groups.to(dtype), self.wo_a.to(dtype).reshape(N, 3, D)
        ).float()
        return linear(projected.flatten(2), self.wo_b, dtype), (compressed, keys, indices), aux


class TinyLayer(nn.Module):
    def __init__(self, kind, *, dtype, layout_mode, layer):
        super().__init__()
        self.attn = TinyAttention(kind, dtype=dtype, layout_mode=layout_mode, layer=layer)
        self.dtype = dtype
        self.gate_up = nn.Parameter(torch.randn(3, 16, 12) * 0.04)
        self.down = nn.Parameter(torch.randn(3, 6, 16) * 0.04)
        self.norm_scale = nn.Parameter(torch.ones(16))
        if layout_mode == "whole":
            gu, down = MatrixLayout.whole(), MatrixLayout.whole()
            gu_ref = lambda t: [t.view(3, -1)]
            down_ref = lambda t: [t.view(3, -1)]
        else:
            gu = MatrixLayout.view(3, 16, 2, 6).permute(0, 2, 3, 1)
            down = MatrixLayout.view(3, 6, 16)
            gu_ref = lambda t: [
                t[e, :, b * 6 : (b + 1) * 6].t() for e in range(3) for b in range(2)
            ]
            down_ref = lambda t: list(t.unbind(0))
        self.specs = {
            "gate_up": LayoutCase("gate_up", (3, 16, 12), gu, gu_ref, "stacked_variant", True),
            "down": LayoutCase("down", (3, 6, 16), down, down_ref, "stacked_variant", True),
        }

    def forward(self, x, memory=None, expert=0):
        o, memory, aux = self.attn(x, memory)
        gate, up = (x.to(self.dtype) @ self.gate_up[expert].to(self.dtype)).float().chunk(2, -1)
        ff = (F.silu(gate) * up).to(self.dtype) @ self.down[expert].to(self.dtype)
        x = (x.float() + o + ff.float()) * self.norm_scale.to(self.dtype).float()
        return x, memory, aux


class TinyAttentionStack(nn.Module):
    def __init__(self, kind, *, dtype=torch.float32, layout_mode="components", depth=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [TinyLayer(kind, dtype=dtype, layout_mode=layout_mode, layer=i) for i in range(depth)]
        )

    def forward(self, x, expert=0, recompute=False):
        memory, auxiliary = None, x.new_zeros(())
        for layer in self.layers:
            if recompute:
                from torch.utils.checkpoint import checkpoint

                x, memory, aux = checkpoint(layer, x, memory, expert, use_reentrant=False)
            else:
                x, memory, aux = layer(x, memory, expert)
            auxiliary = auxiliary + aux
        return x, auxiliary


def matrix_specs(model):
    return {
        f"{prefix}.{name}" if prefix else name: case
        for prefix, module in model.named_modules()
        for name, case in getattr(module, "specs", {}).items()
    }


def install(model, mesh, *, dtype, reshard=True, replicate=None, native_ns=False):
    specs = matrix_specs(model)

    def policy(name, _parameter):
        common = {
            "param_dtype": dtype,
            "grad_dtype": torch.float32,
            "master_dtype": torch.float32,
            "optim_dtype": torch.float32,
        }
        if name in specs:
            return {**common, "route": "muon", "matrix_layout": specs[name].layout}
        return {**common, "route": "adamw"}

    dmuon.dedicate_params(
        model,
        mesh,
        replicate_mesh=replicate,
        predicate=lambda _n, p: p.requires_grad,
        hook_boundary_predicate=lambda m: isinstance(m, TinyLayer),
        reshard_after_forward=reshard,
        process_group_policy="shared",
        param_policy_fn=policy,
    )
    model._dedicated_comm_ctx.forward_prefetch_depth = 2
    optimizer = dmuon.Muon(
        model,
        lr=0.004,
        momentum=0.7,
        weight_decay=0.01,
        adamw_lr=0.002,
        adamw_weight_decay=0.01,
        ns_backend=(dmuon.NewtonSchulz("direct", kernel="cublas") if native_ns else Float32NS()),
        first_step_progress_log=False,
    )
    return optimizer


@dataclass
class ReferenceOptimizer:
    model: nn.Module
    native_ns: bool = False

    def __post_init__(self):
        self.specs = matrix_specs(self.model)
        self.params = dict(self.model.named_parameters())
        self.indices = {name: case.indices() for name, case in self.specs.items()}
        self.momentum = {}
        self.ns = dmuon.NewtonSchulz("direct", kernel="cublas") if self.native_ns else Float32NS()
        self.adam = torch.optim.AdamW(
            [p for n, p in self.params.items() if n not in self.specs],
            lr=0.002,
            weight_decay=0.01,
            foreach=False,
        )

    @torch.no_grad()
    def step(self):
        for name, case in self.specs.items():
            parameter = self.params[name]
            if parameter.grad is None:
                continue
            # NS consumes canonical contiguous logical matrices. The native
            # selectors may return strided views; normalize those before NS
            # so reduction order does not depend on native storage strides.
            gradient = [g.contiguous() for g in case.blocks(parameter.grad)]
            weights = case.blocks(parameter)
            if name not in self.momentum:
                self.momentum[name] = [g.float().clone() for g in gradient]
            else:
                for m, g in zip(self.momentum[name], gradient):
                    m.mul_(0.7).add_(g)
            for pos, weight, grad, momentum in zip(
                self.indices[name], weights, gradient, self.momentum[name]
            ):
                update = self.ns.local(grad.add(momentum, alpha=0.7), 5)
                value = weight * (1 - 0.004 * 0.01)
                value = value.add(
                    update.to(value.dtype), alpha=-0.004 * 0.2 * max(value.shape) ** 0.5
                )
                parameter.flatten()[pos.to(parameter.device).flatten()] = value.flatten()
        self.adam.step()


def checkpoint_bytes(model, optimizer):
    out = io.BytesIO()
    torch.save(
        {
            "model": dmuon.get_model_state_dict(model, rank0_only=False),
            "optimizer": dmuon.get_optimizer_state_dict(model, optimizer, rank0_only=False),
        },
        out,
    )
    return out.getvalue()


def restore_bytes(model, optimizer, data):
    saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    dmuon.set_model_state_dict(model, saved["model"])
    dmuon.set_optimizer_state_dict(model, optimizer, saved["optimizer"])


def unit_name(model, unit):
    if getattr(unit, "matrix_parent", None) is not None:
        return unit.matrix_parent.fqn, unit.matrix_spec.index
    modules = {id(m): name for name, m in model.named_modules()}
    prefix = modules[id(unit.module)]
    return f"{prefix}.{unit.param_name}" if prefix else unit.param_name, None


def assert_owner_gradients(model, reference, *, rtol, atol):
    params = dict(reference.named_parameters())
    specs = matrix_specs(reference)
    for unit in dmuon.get_dedicated_params(model):
        if not unit.is_owner:
            continue
        name, index = unit_name(model, unit)
        expected = params[name].grad
        if expected is None:
            assert unit._reduced_grad is None, name
            continue
        if index is not None:
            expected = specs[name].blocks(expected)[index]
        torch.testing.assert_close(
            unit._reduced_grad, expected, rtol=rtol, atol=atol, msg=lambda error: f"{name}: {error}"
        )


def assert_owner_states(model, optimizer, reference, *, rtol, atol):
    for unit in dmuon.get_dedicated_params(model):
        if not unit.is_owner:
            continue
        name, index = unit_name(model, unit)
        state = optimizer.state.get(id(unit), {})
        if name.endswith("unused"):
            assert not state, (name, state)
        elif index is not None:
            torch.testing.assert_close(
                state["momentum_buffer"],
                reference.momentum[name][index],
                rtol=rtol,
                atol=atol,
                msg=lambda error: f"{name}: {error}",
            )
        else:
            expected = reference.adam.state[reference.params[name]]
            assert int(state["step"]) == int(expected["step"]), name
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    state[key],
                    expected[key],
                    rtol=rtol,
                    atol=atol,
                    msg=lambda error: f"{name}: {error}",
                )


def assert_model_weights(model, reference, *, rtol, atol):
    actual = dmuon.get_model_state_dict(model, rank0_only=False)
    expected = reference.state_dict()
    assert set(actual) == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(
            actual[name].cpu(),
            value.detach().cpu(),
            rtol=rtol,
            atol=atol,
            msg=lambda error: f"{name}: {error}",
        )
