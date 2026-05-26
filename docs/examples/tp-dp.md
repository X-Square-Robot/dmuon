# Example: TP + DP Training

Complete example using Tensor Parallelism (TP) combined with Data
Parallelism (DP) on a 2D device mesh.  DMuon detects TP-sharded
parameters automatically via `DTensor`, so the required setup still uses
the same DP mesh slice you pass to FSDP2.

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
# 4 GPUs: 2 DP x 2 TP
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
```

`mesh_dim_names` is required — DMuon infers the TP axis by name-set
subtraction (`DTensor.mesh_dim_names − dp_mesh_dim_names`).

### Call Order: TP → DMuon → FSDP2

```python
# 1. TP
for layer in model.layers:
    parallelize_module(
        layer.attn, mesh["tp"],
        {
            "q_proj": ColwiseParallel(),
            "k_proj": ColwiseParallel(),
            "v_proj": ColwiseParallel(),
            "o_proj": RowwiseParallel(),
        },
    )
    parallelize_module(
        layer.mlp, mesh["tp"],
        {
            "gate_proj": ColwiseParallel(),
            "up_proj":   ColwiseParallel(),
            "down_proj": RowwiseParallel(),
        },
    )

# 2. DMuon — pass only the DP slice
dmuon.dedicate_params(
    model, mesh["dp"],
    predicate=lambda n, p: "proj" in n and p.ndim == 2,
)

# 3. FSDP2 — also the DP slice
for layer in model.layers:
    fully_shard(layer, mesh=mesh["dp"])
fully_shard(model, mesh=mesh["dp"])
```

DMuon must precede `fully_shard` so its parameters can opt out of
FSDP2's sharding contract.

### Optimizer

```python
optimizer = dmuon.Muon(model, lr=0.02, momentum=0.95, adamw_lr=1e-3)
```

Most TP runs need no extra optimizer knobs.  Advanced runs can set
`tp_buffer_reuse=` on `dedicate_params()` to reuse TP gather/scatter
scratch buffers, and `tp_distributed_gram=` on `Muon` to use the
TP-aware distributed Gram path when its factor payload is smaller than
the full update scatter.  `replicate_async` is the DP/HSDP post-step
publish overlap switch; in the current release DMuon falls back to
synchronous publish when TP-sharded dedicated parameters are present.

## What Happens Under the Hood

For each dedicated parameter, DMuon's optimizer step:

1. **DP reduce** → gathers the gradient to the DP owner rank (standard
   DMuon path, unchanged by TP).
2. **TP gather** (only for TP-sharded `DTensor` params, on `reduce_stream`
   so it overlaps with backward compute) → reassembles the full `(m, n)`
   gradient at a designated TP owner inside the TP group.
3. **Newton-Schulz** — runs on the full matrix at the TP owner,
   identical kernel path to the non-TP case.
4. **TP scatter** (on `replicate_broadcast_stream`) → fans the update
   back to each DP-owner rank as a TP-local shard.
5. **Replicate broadcast** — standard HSDP fan-out to replicate peers.

Plain (non-TP) `DTensor` and `torch.Tensor` parameters skip steps 2/4
entirely.

## Run

```bash
# Requires 4 GPUs (2 DP x 2 TP)
torchrun --nproc_per_node=4 examples/tp_dp.py
```

See also: [TP Support guide](../guides/tp-support.md) — deeper treatment
of the All-to-All pipeline, 3D HSDP×TP mesh setup, sync/async semantics,
and inspection APIs.
