# Example: TP + DP Training

A complete example using Tensor Parallelism (TP) combined with Data Parallelism (DP) on a 2D device mesh.

---

## The Script

::: details Full source: `examples/tp_dp.py`

```python
--8<-- "examples/tp_dp.py"
```

:::

## Walkthrough

### 2D Mesh Setup

```python
# 4 GPUs: 2 DP ranks x 2 TP ranks
mesh_2d = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
dp_mesh = mesh_2d["dp"]
tp_mesh = mesh_2d["tp"]
```

The mesh is 2D: DP dimension for data parallelism, TP dimension for tensor parallelism.

### Apply TP First

```python
for layer in model.layers:
    parallelize_module(
        layer.attn, tp_mesh,
        {
            "q_proj": ColwiseParallel(),   # Shard(0) — row-sharded
            "k_proj": ColwiseParallel(),   # Shard(0)
            "v_proj": ColwiseParallel(),   # Shard(0)
            "o_proj": RowwiseParallel(),   # Shard(1) — col-sharded
        },
    )
    parallelize_module(
        layer.mlp, tp_mesh,
        {
            "gate_proj": ColwiseParallel(),
            "up_proj": ColwiseParallel(),
            "down_proj": RowwiseParallel(),
        },
    )
```

TP must be applied **before** DMuon and FSDP2.

### Then DMuon + FSDP2

```python
# DMuon uses dp_mesh (data parallel dimension)
dmuon.dedicate_params(model, dp_mesh, predicate=lambda n, p: "proj" in n and p.ndim == 2)

# FSDP2 also uses dp_mesh
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)
```

### Optimizer with TP Options

```python
optimizer = dmuon.Muon(
    model, lr=0.02,
    per_head_ns=True,         # Zero TP comm for GQA k/v_proj (default)
    block_diagonal_ns=False,  # Set True for zero TP comm everywhere (experimental)
    adamw_lr=1e-3,
)
```

The optimizer automatically detects TP-sharded parameters and routes to the appropriate NS variant.

## What Happens Under the Hood

For each dedicated parameter, the optimizer checks:

1. **Is it a DTensor with a TP group?** If no → local `newton_schulz()`
2. **Is it narrow Shard(0)?** (e.g., GQA k/v_proj where full_m < full_n) → per-head `newton_schulz()`, zero TP communication
3. **Is `block_diagonal_ns=True`?** → `gram_newton_schulz(..., block_diagonal=True)`, zero TP communication
4. **Otherwise** → `gram_newton_schulz(..., shard_dim=dp.shard_dim)`, exact Gram with TP all-reduce

## Run

```bash
# Requires 4 GPUs (2 DP x 2 TP)
torchrun --nproc_per_node=4 examples/tp_dp.py
```
