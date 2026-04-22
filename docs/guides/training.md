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

The `predicate` function decides which parameters use Muon (dedicated) vs AdamW (symmetric). It receives the fully-qualified parameter name and the parameter tensor:

```python
def predicate(name: str, param: nn.Parameter) -> bool:
    return ...  # True = dedicated (Muon), False = symmetric (AdamW)
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

### Hyperparameter Guide

| Parameter | Default | Notes |
|-----------|---------|-------|
| `lr` | 0.02 | Muon learning rate. Scaled internally by `0.2 * sqrt(max(m,n))` per param. |
| `momentum` | 0.95 | Higher = more smoothing. 0.95 is the standard Muon/Moonlight value. |
| `ns_steps` | 5 | Number of NS iterations. 5 is sufficient for convergence. |
| `ns_backend` | `"gram"` | `"gram"` or `"direct"` string, or a `dmuon.NewtonSchulz(...)` object for custom coefficients. |
| `nesterov` | True | Nesterov lookahead: `ns_input = grad + mu * buf`. Recommended. |
| `adamw_lr` | 1e-3 | Separate learning rate for non-matrix parameters. |

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

1. `optimizer.step()` internally: (a) waits for all async gradient reduces to complete, (b) runs Muon on dedicated params, (c) runs AdamW on FSDP2 params.

The training loop is identical to standard PyTorch. No special hooks or context managers needed.

### Gradient Clipping

Use PyTorch's native `clip_grad_norm_` — it works with DMuon out of the box:

```python
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

!!! info "Why does this work?"
    Dedicated params store their gradients in `_reduced_grad`, not `param.grad`. So `clip_grad_norm_` naturally sees only symmetric params (LayerNorm, embeddings) — which are the ones that actually need clipping.

    Dedicated params don't need clipping because Newton-Schulz orthogonalization projects the gradient onto an orthogonal matrix with bounded spectral norm, regardless of the input gradient magnitude.

## Logging and Debugging

### Check NS Backend

```python
print(f"NS backend: {dmuon.get_ns_backend()}")
# "syrk_sm80" = CuteDSL SYRK kernel (fastest)
# "compiled"  = @torch.compile fallback
```

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

See the dedicated [HSDP guide](hsdp.md) for the full API, sync vs async mode, the fallback protocol, and profiling.

For the DMuon-Z2 vs DMuon-Z3 packed-buffer lifecycle choice applicable under both FSDP and HSDP, see [Z2 vs Z3 Modes](z2-z3-modes.md).

## See also

- [HSDP (Multi-Node)](hsdp.md) — 2D mesh training with async broadcast
- [Custom Hook Boundaries](custom-hook-boundaries.md) — Control which module receives DMuon's forward/backward hooks
- [Z2 vs Z3 Modes](z2-z3-modes.md) — Packed-buffer lifecycle and memory/comm tradeoff
- [Profiling & Fallback](profiling-and-fallback.md) — Measure broadcast latency and tune the async fallback protocol
- [Tensor Parallelism](tp-support.md) — Using DMuon with TP
- [Checkpointing](checkpoint.md) — Save and load training state
- [Gradient Accumulation](grad-accumulation.md) — Effective batch size scaling
