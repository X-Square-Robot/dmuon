# Tensor Parallelism

DMuon composes with PyTorch's native tensor parallelism (TP) via
`DTensor`.  You apply TP the way you always would — DMuon detects
TP-sharded parameters automatically and routes them through a
TP gather → full-matrix Newton-Schulz → TP scatter pipeline, so
Muon's exact mathematical definition is preserved even when each rank
holds only a slice of each weight matrix.

**Key property**: the TP path is completely transparent.
`dedicate_params` takes no `tp_mesh` argument — you pass the same DP
slice of your mesh that you hand to `fully_shard`, and DMuon infers TP
from each parameter's `DTensor` structure.  Mirrors FSDP2's own
TP-oblivious setup.

---

## How It Works

For each parameter the user marked for dedicated ownership:

* **Plain `torch.Tensor`** — DMuon's standard DP path (reduce-to-owner,
  broadcast).  No change from the non-TP case.
* **`DTensor` sharded only on DP mesh dim(s)** — same as above.
* **`DTensor` sharded on a non-DP mesh dim (TP)** — DMuon appoints one
  rank in the TP group as "TP owner" for each parameter.  TP owners are
  chosen by per-DP-owner-bucket LPT so full-matrix Newton-Schulz work is
  spread across TP ranks.  At each optimizer step:
  1. Every DP-owner rank in the TP group runs a `dist.gather` on
     `reduce_stream`, reassembling the full `(m, n)` gradient at the TP
     owner.  (This piggybacks on the DP reduce stream, so gather
     executes in parallel with the backward compute — verified ~100%
     overlap on 8-GPU 3D HSDP×TP.)
  2. The TP owner runs Newton-Schulz on the **full matrix** (same kernel
     path as non-TP).
  3. `dist.scatter` on `replicate_broadcast_stream` sends each DP-owner
     rank its shard of the update.
  4. For HSDP, the standard replicate broadcast fans the TP-correct
     update out to replicate peers.  For 2D DP×TP there is no replicate
     axis; the next forward's shard broadcast reads the updated owner
     shard.

In the current release, TP-sharded dedicated parameters use the synchronous
post-step publish path even if `replicate_async=True` is requested.  This keeps
TP training on the checked numerical path while the async TP scatter path
remains a diagnostic/performance development target.

---

## Setup

Call order: **TP first, then DMuon, then FSDP2.**  The DMuon call must
precede `fully_shard` so its parameters can opt out of FSDP2's sharding
contract.

```python
import dmuon
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, parallelize_module,
)

# 2D mesh (dp × tp) — most common setup
mesh = init_device_mesh(
    "cuda", (dp_size, tp_size),
    mesh_dim_names=("dp", "tp"),        # names are required
)

model = MyModel().cuda()

# Step 1 — TP
for layer in model.layers:
    parallelize_module(
        layer.self_attn, mesh["tp"],
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

# Step 2 — DMuon (takes the DP slice of the mesh, not the TP slice)
dmuon.dedicate_params(
    model, mesh["dp"],
    predicate=lambda n, p: "proj" in n and p.ndim == 2,
)

# Step 3 — FSDP2 (also the DP slice)
for layer in model.layers:
    fully_shard(layer, mesh=mesh["dp"])
fully_shard(model, mesh=mesh["dp"])

# Optimizer — TP works with the default settings
optimizer = dmuon.Muon(model, lr=0.02, momentum=0.95, adamw_lr=1e-3)
```

!!! info "Why `mesh["dp"]`, not the full mesh?"
    Both `dedicate_params` and `fully_shard` operate on the
    data-parallel dimension only — they're TP-oblivious.  TP sharding
    has already been applied by `parallelize_module` and is visible to
    DMuon through each parameter's `DTensor.device_mesh`.  This matches
    the FSDP2 convention.

### 3D mesh: HSDP × TP

For multi-node training add a replicate axis.  DMuon supports the
three-axis mesh out of the box:

```python
mesh3d = init_device_mesh(
    "cuda", (R, G, T),
    mesh_dim_names=("replicate", "shard", "tp"),
)

# Step 1 — TP
parallelize_module(model, mesh3d["tp"], plan)

# Step 2 — DMuon (DP = replicate × shard)
dmuon.dedicate_params(
    model,
    mesh=mesh3d["shard"],
    replicate_mesh=mesh3d["replicate"],
    predicate=...,
)

# Step 3 — FSDP2 (same DP 2D slice)
fully_shard(model, mesh=mesh3d["replicate", "shard"])

optimizer = dmuon.Muon(model, lr=0.02)
```

---

## Requirements

1. **`mesh_dim_names` is required** whenever TP is present.  DMuon
   identifies the TP axis by subtracting its DP dim names from each
   parameter's `DTensor.device_mesh.mesh_dim_names`; an unnamed mesh
   under a `DTensor` raises `ValueError`.
2. **TP size of 1 is a no-op.**  A `(dp=N, tp=1)` mesh behaves
   bit-identically to a `(dp=N,)` mesh — DMuon's detection guard treats
   size-1 TP as "no TP".
3. **Call order** — `parallelize_module` → `dmuon.dedicate_params` →
   `fully_shard`.  DMuon must see TP-wrapped parameters *before* FSDP2
   registers its own sharding contract over them.

---

## DDP + TP

When the data-parallel dimension should stay fully replicated while TP stays
inside each replica, use the TP-aware DDP entry points instead of the FSDP2
path:

```python
parallelize_module(model, mesh["tp"], plan)                 # TP first
dmuon.dedicate_params_ddp_tp(model, mesh["dp"], predicate=...)
dmuon.replicate_tp(model, mesh["dp"])                       # non-dedicated params
optimizer = dmuon.Muon(model, lr=0.02)
```

`dedicate_params_ddp_tp()` installs the TP gather → owner update → TP scatter
path for dedicated matrices.  `replicate_tp()` handles non-dedicated TP
parameters by broadcasting their TP-local shards across the DP mesh.  Plain
`dedicate_params_ddp()` still rejects TP-sharded dedicated parameters because
it does not install the TP-aware replicated-gradient path.

---

## Runtime knobs

Most TP runs use the defaults.  The advanced knobs are explicit constructor
arguments rather than environment variables:

* `dedicate_params(..., tp_buffer_reuse=...)` controls whether TP gather and/or
  scatter scratch buffers are reused.  Accepted values are `False`, `True`,
  `"gather"`, `"scatter"`, and `"all"`.
* `Muon(..., tp_distributed_gram=True)` enables the TP-aware distributed Gram
  path for TP-sharded matrices.  With the default
  `tp_distributed_gram_policy="beneficial"`, DMuon only uses it when the Gram
  factor payload is expected to be smaller than scattering the full update.
* `Muon(..., replicate_async=...)` controls DP/HSDP post-step publish overlap.
  When TP-sharded dedicated parameters are present, DMuon currently falls back
  to synchronous post-step publish for correctness.

---

## Sync vs async post-step

`Muon` exposes `replicate_async` for post-step publish timing:

```python
# Current TP-safe default — scatter + broadcast complete before step() returns.
optimizer = dmuon.Muon(model, lr=0.02)

# Sync — scatter + broadcast complete before step() returns.  Useful
# for profiling or for pipelines where the next iter's forward starts
# close enough that overlap doesn't help.
optimizer = dmuon.Muon(model, lr=0.02, replicate_async=False)
```

TP async publish is not enabled by default in this release.  The TP diagnostic
tests still exercise the underlying scatter/publish state machine, but user
training stays on the synchronous path until sync-vs-async parity is covered
across the public TP matrix.

---

## Inspecting TP properties

```python
import dmuon
import torch.distributed as dist

for dp in dmuon.get_owned_params(model, rank=dist.get_rank()):
    print(
        f"{dp.param_name}: "
        f"local={tuple(dp._orig_size)}, "
        f"full={tuple(dp.full_shape)}, "
        f"shard_dim={dp.shard_dim}, "
        f"is_tp_owner={dp.is_tp_owner}, "
        f"tp_group_size={dp.tp_group.size() if dp.tp_group else 1}"
    )
```

A TP-sharded parameter reports `tp_group_size > 1`, a populated
`shard_dim`, and `is_tp_owner=True` on exactly one rank per TP group
for that parameter.  Different TP-sharded parameters may have different
TP owners; this is expected and is how DMuon balances NS work across the
TP group.

---

## Limitations

* **1D TP only** for the MVP.  A multi-dim TP axis (e.g. 2D tensor
  parallel) raises in the detection helper; extend `get_tp_mesh` when
  it's needed.
* **Single-owner NS per parameter.**  Each TP-sharded parameter has one
  TP owner for its full-matrix Newton-Schulz call, but owners vary across
  parameters via LPT.  Canzona-style fused All-to-All + micro-group
  batching that parallelises a single group of NS calls more tightly is
  listed as future work.
* **Small TP-sharded params do NOT participate in the DMuon small-param
  merge** (`SMALL_PARAM_THRESHOLD`).  Each TP-sharded parameter makes
  its own gather/scatter round-trip even when < 5M numel; in practice
  these are rare.
## See also

* [HSDP guide](hsdp.md) — replicate × shard setup; composes with TP
* [`dedicate_params` API](../reference/api.md) — full signature
* [Checkpointing](checkpoint.md) — state-dict behavior for dedicated params
* [Communication Cost Analysis](../reference/communication-cost.md) —
  broadcast/reduce cost model
