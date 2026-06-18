# Training Guide

!!! tip "TL;DR"
    1. Call `dmuon.dedicate_params(model, mesh, predicate=...)` **before** `fully_shard()` to assign matrix parameters to dedicated owners.
    2. Wrap with standard FSDP2 `fully_shard()` — DMuon auto-skips dedicated params.
    3. Use `dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)` as the optimizer — it handles both Muon (dedicated) and AdamW (symmetric) params in one call.

---

## Overview

A DMuon training setup has four steps:

1. **Build model** — standard PyTorch model
2. **`dedicate_params()`** — mark matrix parameters for dedicated ownership
3. **`fully_shard()`** — apply FSDP2 to the remaining parameters
4. **Training loop** — same as standard PyTorch

## Step 1: Model Preparation

DMuon works with any `nn.Module`. No special base class or wrapper needed.

```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

dist.init_process_group("nccl")
torch.cuda.set_device(dist.get_rank())

mesh = init_device_mesh("cuda", (dist.get_world_size(),))

model = MyModel().cuda()
```

!!! tip "HuggingFace models"
    DMuon works with HuggingFace models. Use `AutoModelForCausalLM.from_pretrained(...)` as usual, then apply DMuon + FSDP2.

## Step 2: Dedicate Parameters

```python
import dmuon

assignment = dmuon.dedicate_params(
    model,
    mesh,
    predicate=lambda n, p: "proj" in n and p.ndim == 2,
)
```

### Writing the Predicate

The `predicate` function decides which parameters enter DMuon's dedicated
ownership runtime.  In the default setup, selected parameters use Muon and
unselected parameters remain on the normal FSDP2/AdamW path.  The predicate
receives the fully-qualified parameter name and the parameter tensor:

```python
def predicate(name: str, param: nn.Parameter) -> bool:
    return ...  # True = DMuon-managed Muon, False = FSDP2/AdamW
```

**Common patterns:**

=== "Standard Transformer (Llama, Qwen, Mistral)"

    ```python
    # All 2D projection layers → Muon
    predicate = lambda n, p: "proj" in n and p.ndim == 2
    ```

=== "Selective (exclude embeddings)"

    ```python
    # Projection layers only, exclude embed/head
    def predicate(n, p):
        if p.ndim != 2:
            return False
        if "embed" in n or "head" in n or "lm_head" in n:
            return False
        return "proj" in n
    ```

=== "GateDelta / Hybrid Attention"

    ```python
    # Exclude very small projections (a_proj, b_proj in GateDelta)
    def predicate(n, p):
        if p.ndim != 2:
            return False
        if p.numel() < 100_000:  # too small for NS to be worthwhile
            return False
        return "proj" in n
    ```

**Guidelines:**

- **2D matrices only** — 1D parameters (LayerNorm, bias) should use AdamW
- **Large enough for NS** — Very small matrices don't benefit from Newton-Schulz. A rough threshold: `numel > 100k`
- **Embedding/head layers** — Usually kept under AdamW (they don't fit the NS optimization geometry well)

### Advanced: Type-Split Routing

For large scaling runs where DMuon should own communication for all trainable
parameters, pass a broader `predicate` and a `route_hint_fn`.  Route
`"muon"` keeps large matrix parameters on the matrix-optimizer path; route
`"adamw"` keeps small AdamW parameters on DMuon's owner broadcast/reduce path;
route `"sharded_adamw"` is reserved for very large AdamW tensors such as
embeddings and `lm_head`, where all ranks should share the communication.

```python
SHARDED_ADAMW_NAME_PARTS = ("embed_tokens", "lm_head")


def route_hint(name, param):
    if param.ndim == 2 and "proj" in name:
        return "muon"
    if param.ndim == 2 and any(part in name for part in SHARDED_ADAMW_NAME_PARTS):
        return "sharded_adamw"
    return "adamw"

dmuon.dedicate_params(
    model,
    mesh,
    predicate=lambda n, p: p.requires_grad,
    route_hint_fn=route_hint,
)
```

The default `predicate=lambda n, p: "proj" in n and p.ndim == 2` remains the
simpler integration path.  Use type-split routing only when you want DMuon to
own placement and collectives for the non-Muon trainable parameters as well.
See [Pure DMuon Routing](pure-dmuon-routing.md) for the full route policy.

### Inspecting the Assignment

```python
# What did each rank get?
owned = dmuon.get_owned_params(model, rank=dist.get_rank())
total_owned = sum(dp.numel for dp in owned)
print(f"Rank {dist.get_rank()}: owns {len(owned)} params, {total_owned:,} elements")
```

## Step 3: Apply FSDP2

```python
from torch.distributed.fsdp import fully_shard

for layer in model.layers:  # or model.model.layers for HuggingFace
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)
```

This is standard FSDP2 usage. DMuon's monkey-patch ensures `fully_shard()` skips dedicated parameters automatically.

!!! warning "Order: dedicate first, then shard"
    `dedicate_params()` must be called **before** `fully_shard()`. The monkey-patch needs `_dedicated_owner_rank` markers to be present when FSDP2 processes the parameters.

## Step 4: Create Optimizer

```python
optimizer = dmuon.Muon(
    model,
    lr=0.02,              # Muon learning rate (dedicated params)
    momentum=0.95,        # Momentum coefficient
    ns_steps=5,           # Newton-Schulz iterations
    nesterov=True,        # Nesterov momentum (recommended)
    weight_decay=0.0,     # Weight decay for dedicated params
    adamw_lr=1e-3,        # AdamW learning rate (symmetric params)
    adamw_betas=(0.9, 0.999),
    adamw_weight_decay=0.01,
    adamw_eps=1e-8,
)
```

`dmuon.Muon` manages both parameter types in a single optimizer:

- **Group 0** (dedicated params): Muon — momentum + NS + update, owner only
- **Group 1** (symmetric params): AdamW — standard, all ranks

### Semantic Param Groups

Use `param_groups` when a training framework needs business-level learning
rate groups, such as a VLA action expert with a higher LR. Build the groups
from the same wrapped model object that you pass to `dmuon.Muon`; with FSDP2,
this means after `dedicate_params()` and after wrapping. DMuon lowers each user
group into two optimizer subgroups: `<name>/muon` for dedicated parameters and
`<name>/adamw` for symmetric parameters and AdamW-routed dedicated parameters.

```python
base_params = []
action_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "action_transformer" in name:
        action_params.append(param)
    else:
        base_params.append(param)

optimizer = dmuon.Muon(
    model,
    lr=5e-5,
    adamw_lr=5e-5,
    param_groups=[
        {"params": base_params, "lr": 5e-5, "group_name": "base"},
        {"params": action_params, "lr": 1e-4, "group_name": "action"},
    ],
)
```

`lr` applies to both Muon and AdamW subgroups for that semantic group. Advanced
callers can override route-specific values with `muon_lr`, `adamw_lr`,
`muon_weight_decay`, `adamw_weight_decay`, `momentum`, `adamw_betas`, and
`adamw_eps`. Every trainable parameter must appear in exactly one user group;
stale pre-wrapping parameters, duplicate parameters, and missing parameters
raise during optimizer construction.

Semantic `param_groups` are a hyperparameter grouping surface by default; they
do not choose DMuon routes. For DMuon-managed parameters, the per-parameter
route written by `dedicate_params(route_hint_fn=...)` is preserved even when a
user group contains a mix of `"muon"`, `"adamw"`, and `"sharded_adamw"`
parameters. A group-level route is applied only when the user group explicitly
sets `dmuon_route`, `dmuon_optimizer`, or `matrix_optimizer`; use those keys only
when every DMuon-managed parameter in that semantic group should be forced onto
the same route.

Schedulers and checkpoints work through `optimizer.param_groups` as usual. The
visible group names become `base/muon`, `base/adamw`, `action/muon`, and
`action/adamw`, which is the public surface for auditing the route split.

### Hyperparameter Guide

| Parameter | Default | Notes |
|-----------|---------|-------|
| `lr` | 0.02 | Muon learning rate. Scaled internally by `0.2 * sqrt(max(m,n))` per param. |
| `momentum` | 0.95 | Higher = more smoothing. 0.95 is the standard Muon/Moonlight value. |
| `ns_steps` | 5 | Number of NS iterations. 5 is sufficient for convergence. |
| `ns_backend` | `"gram"` | `"gram"` or `"direct"` string, or a `dmuon.NewtonSchulz(...)` object for custom coefficients. |
| `nesterov` | True | Nesterov lookahead: `ns_input = grad + mu * buf`. Recommended. |
| `adamw_lr` | 1e-3 | Separate learning rate for non-matrix parameters. |
| `param_groups` | None | Optional semantic PyTorch-style parameter groups, lowered into Muon and AdamW subgroups. |

## Step 5: Training Loop

```python
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()
    optimizer.step()  # (1)!

    if dist.get_rank() == 0:
        print(f"step {step}: loss={loss.item():.4f}")
```

1. `optimizer.step()` internally: (a) waits for async gradient reduces to complete, (b) runs Muon on routed matrix params, and (c) runs AdamW on either FSDP2-managed params or DMuon-managed sharded AdamW params, depending on the route setup.

The training loop is identical to standard PyTorch. No special hooks or context managers needed.

### Gradient Clipping

Use PyTorch's native `clip_grad_norm_` for ordinary `param.grad` tensors, and
add DMuon's Muon-only clip when you want dedicated parameters covered too:

```python
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()

    # Non-dedicated / AdamW parameters: handled by the training framework.
    torch.nn.utils.clip_grad_norm_(adamw_params, max_norm=1.0)

    # DMuon dedicated / Muon parameters: gradients live on DedicatedParam.
    dmuon.clip_grad_norm_(optimizer, max_norm=1.0)

    optimizer.step()
```

!!! info "What is clipped?"
    `dmuon.clip_grad_norm_` only clips DMuon dedicated parameters. It does not
    touch AdamW parameters, so existing training frameworks can keep their
    standard PyTorch clipping path unchanged.

    Muon clipping happens after DMuon's async reduce / TP gather and before
    momentum + Newton-Schulz. Newton-Schulz bounds the final matrix update
    scale, so this clip is mainly a safety guard for anomalous gradients,
    momentum-buffer contamination, and non-finite checks rather than the main
    learning-rate control mechanism.

The default strategy is global p-norm clipping over Muon gradients. Custom
strategies can be registered with `dmuon.register_muon_grad_clip_strategy(...)`
for future schemes such as MuonClip or projection-specific clipping.

## Logging and Debugging

### Check NS Backend

```python
print(f"NS backend: {dmuon.get_ns_backend()}")
# "Gram NS · kernel=cute_sm80 (SM80, DMuon internal)"    — A100/A800 fast path
# "Gram NS · kernel=quack    (SM90, Tri Dao quack)"      — H100/B200/B300 fast path
# "Gram NS · kernel=cublas   (SM80, universal fallback)" — cuBLAS everywhere else
```

Use `dmuon.get_backend_status()` for the full dict of per-backend
availability flags.  See [Backend dispatch](../reference/newton-schulz.md#backend-dispatch)
for the auto-detection ladder and the `kernel=` / `DMUON_NS_KERNEL`
overrides.

### Verify Parameter Assignment

```python
import logging
logging.basicConfig(level=logging.INFO)

# dedicate_params() logs assignment summary:
# INFO: dedicate_params: 56 params assigned to 8 ranks, imbalance=0.2%, loads=[...]
```

### Check Dedicated vs Symmetric Counts

```python
all_dp = dmuon.get_dedicated_params(model)
owned_dp = dmuon.get_owned_params(model, rank=dist.get_rank())
fsdp_count = len(list(model.parameters())) - len(all_dp)

print(f"Dedicated: {len(all_dp)} total, {len(owned_dp)} owned by this rank")
print(f"Symmetric (FSDP2): {fsdp_count}")
```

## Scaling Out

When you cross from single-node multi-GPU to **multi-node** training, switch from a 1D `init_device_mesh("cuda", (world_size,))` to a 2D HSDP mesh and pass the replicate dimension to `dedicate_params`. DMuon handles the two-stage grad reduce (shard → replicate) + async post-step broadcast automatically; everything else in the training loop is unchanged.

```python
hsdp = init_device_mesh(
    "cuda", (replicate_size, shard_size),
    mesh_dim_names=("replicate", "shard"),
)
dmuon.dedicate_params(
    model, hsdp["shard"],
    predicate=lambda n, p: "proj" in n and p.ndim == 2,
    replicate_mesh=hsdp["replicate"],   # ← the HSDP knob
)
for layer in model.layers:
    fully_shard(layer, mesh=hsdp)
fully_shard(model, mesh=hsdp)
```

See the dedicated [HSDP guide](hsdp.md) for the full API and sync vs async mode.

For the DMuon-Z2 vs DMuon-Z3 packed-buffer lifecycle choice applicable under both FSDP and HSDP, see [Z2 vs Z3 Modes](z2-z3-modes.md).

## See also

- [HSDP (Multi-Node)](hsdp.md) — 2D mesh training with async broadcast
- [Custom Hook Boundaries](custom-hook-boundaries.md) — Control which module receives DMuon's forward/backward hooks
- [Z2 vs Z3 Modes](z2-z3-modes.md) — Packed-buffer lifecycle and memory/comm tradeoff
- [Tensor Parallelism](tp-support.md) — Using DMuon with TP
- [Checkpointing](checkpoint.md) — Save and load training state
- [Gradient Accumulation](grad-accumulation.md) — Effective batch size scaling
