# Custom Hook Boundaries

!!! tip "TL;DR"
    The default `layers.N` / `blocks.N` heuristic works for standard LLMs. For VLA,
    MoE, or nested multi-modal models, pass `hook_boundary_predicate` to control which
    module is the hook-registration site. Keep `hook_boundary_strict=True` (the default)
    to catch mismatches at setup time.

---

## Why you might need this

DMuon registers forward/backward broadcast and reduce hooks at the *layer* level,
not on individual sub-modules. Batching all hooks onto a single module per layer
reduces CPU launch overhead and enables packed broadcasts.

For a standard transformer, the built-in heuristic reads the fully-qualified parameter
name and extracts the first `layers.N` or `blocks.N` segment via
`partition.py::_extract_layer_id`. This works well for paths like
`model.layers.3.self_attn.q_proj.weight`.

**Situations where the heuristic silently fails:**

- **VLA / multi-modal models** — a vision tower uses `blocks.N` while the decoder uses
  `layers.N`; parameters may collapse onto the wrong boundary or fall through to
  per-`Linear` hooks.
- **MoE models** — experts are often named `layers.N.mlp.experts.K`; the heuristic
  merges the router and every expert into the same `layers.N` site.
- **Nested model structures** — Qwen-VL-style paths such as
  `model.vlm.llm.model.layers.3.mlp.gate_proj` may confuse the heuristic if another
  `layers` level is introduced above the LLM.
- **Per-Linear fallback** — if no match is found, the heuristic falls back to the
  parameter's direct parent `nn.Linear`, giving each projection its own hook.

All of these degrade performance without producing errors.

---

## The API

```python
dmuon.dedicate_params(
    model,
    mesh,
    predicate=lambda n, p: p.ndim == 2 and "proj" in n,
    hook_boundary_predicate=lambda m: isinstance(m, MyLayerClass),
    hook_boundary_strict=True,   # default — recommended
)
```

**`hook_boundary_predicate`** is a callable `(module) -> bool`. DMuon calls
`_find_hook_module`, which walks each dedicated parameter's ancestors from bottom to
top and returns the first ancestor where the predicate is `True`.

**`hook_boundary_strict`** (default `True`) raises `ValueError` at `dedicate_params`
time if any dedicated parameter has no matching ancestor. Set to `False` only for
exploratory prototyping.

The predicate affects only hook registration; it is independent of the LPT balanced
partition, which uses the separate `predicate` argument.

---

## Worked examples

### Example 1: VLA model (vision tower + decoder layers + action head)

Based on the `ToyVLA` structure in `tests/unit/test_hook_boundary.py`. All 24 ViT
blocks collapse onto a single `model.visual` hook site; each decoder layer and the
action head get their own site.

```python title="vla_setup.py"
import torch.nn as nn
import dmuon
from torch.distributed.fsdp import fully_shard

class VisionTower(nn.Module):
    def __init__(self, d=1024, n=24):
        super().__init__()
        self.blocks = nn.ModuleList([VitBlock(d) for _ in range(n)])

class DecoderLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, 4 * d, bias=False)
        self.down_proj = nn.Linear(4 * d, d, bias=False)

class ActionHead(nn.Module):
    def __init__(self, d, n_actions=7):
        super().__init__()
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, n_actions, bias=False)

class ToyVLA(nn.Module):
    def __init__(self, d=1024, n_vit=24, n_dec=28):
        super().__init__()
        self.visual = VisionTower(d, n_vit)
        self.layers = nn.ModuleList([DecoderLayer(d) for _ in range(n_dec)])
        self.action_head = ActionHead(d)

model = ToyVLA().cuda()

def boundary(m):
    return isinstance(m, (VisionTower, DecoderLayer, ActionHead))

dmuon.dedicate_params(
    model, mesh,
    predicate=lambda n, p: p.ndim == 2,
    hook_boundary_predicate=boundary,
    hook_boundary_strict=True,
)
for layer in model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model.visual, mesh=mesh)
fully_shard(model.action_head, mesh=mesh)
fully_shard(model, mesh=mesh)
optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)
```

---

### Example 2: MoE model

Each expert is a separate hook site; the router stays with the outer `MoELayer`.
The router weight is excluded from dedicated ownership via `predicate`.

```python title="moe_setup.py"
import torch.nn as nn
import dmuon
from torch.distributed.fsdp import fully_shard

class Expert(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

class MoELayer(nn.Module):
    def __init__(self, d, d_ff, n_experts=8):
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d, d_ff) for _ in range(n_experts)])
        self.o_proj = nn.Linear(d, d, bias=False)

model = MoEModel().cuda()  # MoEModel wraps MoELayer × n_layers

dmuon.dedicate_params(
    model, mesh,
    predicate=lambda n, p: p.ndim == 2 and "router" not in n,
    hook_boundary_predicate=lambda m: isinstance(m, (Expert, MoELayer)),
    hook_boundary_strict=True,
)
for layer in model.layers:
    for expert in layer.experts:
        fully_shard(expert, mesh=mesh)
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)
optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)
```

---

### Example 3: Nested Qwen-VL-style model

```python title="qwenvl_setup.py"
import torch
import dmuon
from torch.distributed.fsdp import fully_shard
from transformers import Qwen2VLForConditionalGeneration

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16,
).cuda()

def boundary(m):
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VLDecoderLayer, Qwen2VLVisionBlock,
    )
    return isinstance(m, (Qwen2VLDecoderLayer, Qwen2VLVisionBlock))

dmuon.dedicate_params(
    model, mesh,
    predicate=lambda n, p: p.ndim == 2 and any(
        k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj")
    ),
    hook_boundary_predicate=boundary,
    hook_boundary_strict=True,
)
for layer in model.model.layers:
    fully_shard(layer, mesh=mesh)
for block in model.visual.blocks:
    fully_shard(block, mesh=mesh)
fully_shard(model, mesh=mesh)
optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)
```

---

### Example 4: nn.Sequential — when NOT to use hook_boundary_predicate

A flat `nn.Sequential` with no logical grouping already gets per-`Linear` hooks from
the default heuristic — that is correct. Do not add an artificial predicate.

```python title="sequential_ok.py"
import torch.nn as nn
import dmuon
from torch.distributed.fsdp import fully_shard

model = nn.Sequential(
    nn.Linear(1024, 4096, bias=False),
    nn.GELU(),
    nn.Linear(4096, 1024, bias=False),
).cuda()

# No hook_boundary_predicate needed; per-Linear hooks are correct here.
dmuon.dedicate_params(model, mesh, predicate=lambda n, p: p.ndim == 2)
fully_shard(model, mesh=mesh)
optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)
```

Forcing `lambda m: isinstance(m, nn.Sequential)` would collapse all parameters onto
the root, eliminating layer-level pipelining.

---

## Aligning with `fully_shard` boundaries

Define the predicate once and reuse it for both `hook_boundary_predicate` and
`fully_shard`. Aligned boundaries make the forward ordering predictable and the
prefetch pipeline most effective.

```python title="aligned_boundaries.py"
import dmuon
from torch.distributed.fsdp import fully_shard

def is_transformer_layer(m):
    return isinstance(m, TransformerLayer)

dmuon.dedicate_params(
    model, mesh, predicate=lambda n, p: p.ndim == 2,
    hook_boundary_predicate=is_transformer_layer, hook_boundary_strict=True,
)
for module in model.modules():
    if is_transformer_layer(module):
        fully_shard(module, mesh=mesh)
fully_shard(model, mesh=mesh)
```

---

## `strict=True` vs `strict=False`

`strict=True` raises immediately if any dedicated parameter has no matching ancestor —
catches typos, missing imports, and partially-covered model structures. Use
`strict=False` only for exploratory prototyping. In production, always use `strict=True`.

---

## Pitfalls

**Too-narrow predicate** — strict mode raises; lenient mode silently falls back to
per-`Linear` hooks. Extend the predicate to cover all module types that contain
dedicated parameters.

**Too-broad predicate** — `lambda m: isinstance(m, nn.Module)` matches every module
including the root, collapsing all parameters into one site and losing layer-level
pipelining.

**Predicate with side effects** — `_find_hook_module` calls the predicate once per
ancestor per parameter. Avoid stateful or expensive predicates.

**Shared expert modules in MoE** — two layers referencing the same expert object map
both parameter sets to the same hook site. Verify partition assignments before training.

---

## See also

- [HSDP Guide](hsdp.md) — HSDP-specific hook boundary notes
- [Z2 vs Z3 Modes](z2-z3-modes.md) — `reshard_after_forward` interacts with hook granularity
- [Architecture](../design/architecture.md) — how hooks compose with FSDP2
- [API Reference](../reference/api.md) — full `dedicate_params` signature
