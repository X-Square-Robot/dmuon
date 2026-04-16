# Tensor Parallelism

This guide covers how DMuon works with Tensor Parallelism (TP) and the different Newton-Schulz modes available for TP-sharded parameters.

---

## The Challenge

With TP, each rank holds a **shard** of the parameter, not the full matrix. The owner rank's gradient is also a TP shard. Standard Newton-Schulz requires the full (m, n) matrix — but we only have (m/T, n) or (m, n/T) on each TP rank.

**Naive solution**: all-gather the full gradient across TP ranks → O(mn) communication. This defeats the purpose of DMuon.

**DMuon's solution**: **Gram Newton-Schulz** — iterate on the Gram matrix instead of the full parameter. The Gram matrix can be reconstructed from TP shards via a single all-reduce of a much smaller (d, d) matrix.

## Gram NS: How It Works

Standard NS iterates on the full (m, n) matrix X. Gram NS rewrites the iteration to work on the Gram matrix R = X @ X^T (size m x m) or R = X^T @ X (size n x n).

The key insight is that Gram matrices **decompose** under TP sharding:

| TP Sharding | Example | Local Shape | Decomposable Gram | All-reduce Size |
|---|---|---|---|---|
| **Shard(0)** (row) | q_proj, gate_proj | (m/T, n) | R-side: $G^TG = \sum_i G_i^T G_i$ | n x n |
| **Shard(1)** (col) | o_proj, down_proj | (m, n/T) | L-side: $GG^T = \sum_i G_i G_i^T$ | m x m |

Each TP rank computes its local Gram $G_i^T G_i$ or $G_i G_i^T$, then a single **all-reduce** across TP ranks gives the exact global Gram. The NS iteration then proceeds locally on this (d, d) matrix.

!!! success "Communication reduction"
    For a standard Transformer, the all-reduce size is always **d_model x d_model** regardless of the parameter shape. This is O(d^2) vs O(mn) — a significant reduction for FFN layers where intermediate_size >> d_model.

## Setup: TP + DMuon + FSDP2

The setup order is: **TP first, then DMuon, then FSDP2**.

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, parallelize_module,
)
from torch.distributed.fsdp import fully_shard
import dmuon

# 2D mesh: dp_size x tp_size
mesh_2d = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
dp_mesh = mesh_2d["dp"]
tp_mesh = mesh_2d["tp"]

model = MyModel().cuda()

# Step 1: Apply TP
for layer in model.layers:
    parallelize_module(
        layer.self_attn, tp_mesh,
        {
            "q_proj": ColwiseParallel(),   # Shard(0)
            "k_proj": ColwiseParallel(),   # Shard(0)
            "v_proj": ColwiseParallel(),   # Shard(0)
            "o_proj": RowwiseParallel(),   # Shard(1)
        },
    )
    parallelize_module(
        layer.mlp, tp_mesh,
        {
            "gate_proj": ColwiseParallel(),  # Shard(0)
            "up_proj": ColwiseParallel(),    # Shard(0)
            "down_proj": RowwiseParallel(),  # Shard(1)
        },
    )

# Step 2: DMuon (use dp_mesh, not full mesh)
dmuon.dedicate_params(
    model, dp_mesh,
    predicate=lambda n, p: "proj" in n and p.ndim == 2,
)

# Step 3: FSDP2 (also dp_mesh)
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)

# Optimizer
optimizer = dmuon.Muon(model, lr=0.02, ns_steps=5, adamw_lr=1e-3)
```

!!! info "Why dp_mesh?"
    `dedicate_params` and `fully_shard` use the **DP mesh** — they distribute parameters across data-parallel ranks. TP sharding is already applied; DMuon operates on the TP-local shard.

## Three NS Modes

DMuon provides three NS modes for TP-sharded parameters, selected by optimizer flags:

### 1. Exact Gram NS (Default)

All-reduces the Gram matrix across TP ranks to get the exact global Gram.

```python
optimizer = dmuon.Muon(model, lr=0.02)
# per_head_ns=True (default), block_diagonal_ns=False (default)
```

**Behavior per parameter:**

| Parameter | Shard | Gram Side | All-reduce Size | Exact? |
|-----------|-------|-----------|-----------------|--------|
| q_proj (8192, 8192) | Shard(0) | R-side G^TG | 8192 x 8192 | Yes |
| k_proj (1024, 8192) | Shard(0) | Per-head NS | *none* | Yes |
| gate_proj (28672, 8192) | Shard(0) | R-side G^TG | 8192 x 8192 | Yes |
| o_proj (8192, 8192) | Shard(1) | L-side GG^T | 8192 x 8192 | Yes |
| down_proj (8192, 28672) | Shard(1) | L-side GG^T | 8192 x 8192 | Yes |

### 2. Per-Head NS (Default for GQA k/v_proj)

For GQA models, k_proj and v_proj have fewer heads than q_proj (e.g., 8 KV heads vs 32 Q heads in Llama-3). When TP size <= num_kv_heads, each TP rank holds **complete KV heads**.

This means local NS on each rank is **exact** — no TP communication needed.

```python
optimizer = dmuon.Muon(model, lr=0.02, per_head_ns=True)  # default
```

**Detection logic**: A parameter uses per-head NS when all three conditions are met:

1. `per_head_ns=True` (default)
2. `shard_dim == 0` (row-sharded, ColwiseParallel)
3. `full_m < full_n` (narrow matrix — the full row dimension is smaller than the column dimension)

This correctly identifies GQA k/v_proj while excluding q_proj, gate_proj, etc.

!!! example "Llama-3 8B with TP=8"
    - k_proj: full (1024, 8192) → 1024 < 8192 → **per-head NS** (zero TP comm)
    - q_proj: full (8192, 8192) → 8192 = 8192 → **exact Gram NS**
    - gate_proj: full (28672, 8192) → 28672 > 8192 → **exact Gram NS**

### 3. Block-Diagonal NS (Experimental)

Skips the Gram all-reduce entirely, using only the local partial Gram. This eliminates **all** TP optimizer communication at the cost of an approximation.

```python
optimizer = dmuon.Muon(model, lr=0.02, block_diagonal_ns=True)
```

!!! warning "Experimental"
    Block-diagonal NS is an approximation. It extends the block-diagonal preconditioning principle from Shampoo to Newton-Schulz. Convergence validation is still in progress — use with caution and monitor loss curves.

## Attention Variant Reference

Different attention architectures produce different TP sharding patterns. DMuon handles all of them through its generic routing logic (shard_dim + full_shape), without needing to know the attention type.

| Variant | Key Difference | k/v_proj Shape | Per-Head NS? | Special Notes |
|---------|---------------|----------------|--------------|---------------|
| **MHA** | n_heads = n_kv_heads | (d, d) | No (square) | All exact Gram NS |
| **GQA** | n_kv_heads < n_heads | (kv_dim, d) | Yes | Zero TP comm for k/v |
| **MQA** | n_kv_heads = 1 | (head_dim, d) | Yes | Even narrower than GQA |
| **GateDelta** | V heads > QK heads | (d, d) for v | No (not narrow) | a/b_proj too small — use AdamW |
| **GLA** | Similar to GQA | varies | Depends on shape | Check full_m vs full_n |
| **RetNet** | Similar to MHA | (d, d) | No | All exact Gram NS |

**Predicate advice for GateDelta:**

```python
def predicate(n, p):
    if p.ndim != 2:
        return False
    # Exclude very small projections (a_proj, b_proj in GateDelta)
    if p.numel() < 100_000:
        return False
    return "proj" in n
```

## Inspecting TP Properties

```python
for dp in dmuon.get_owned_params(model, rank=dist.get_rank()):
    print(
        f"{dp.param_name}: "
        f"local={tuple(dp._orig_size)}, "
        f"full={tuple(dp.full_shape)}, "
        f"shard_dim={dp.shard_dim}, "
        f"tp_group={'yes' if dp.tp_group else 'no'}"
    )
```

Example output (Llama-3 8B, TP=8):
```
q_proj:    local=(1024, 8192), full=(8192, 8192), shard_dim=0, tp_group=yes
k_proj:    local=(128, 8192),  full=(1024, 8192), shard_dim=0, tp_group=yes
v_proj:    local=(128, 8192),  full=(1024, 8192), shard_dim=0, tp_group=yes
o_proj:    local=(8192, 1024), full=(8192, 8192), shard_dim=1, tp_group=yes
gate_proj: local=(3584, 8192), full=(28672, 8192), shard_dim=0, tp_group=yes
up_proj:   local=(3584, 8192), full=(28672, 8192), shard_dim=0, tp_group=yes
down_proj: local=(8192, 3584), full=(8192, 28672), shard_dim=1, tp_group=yes
```
