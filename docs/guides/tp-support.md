# Tensor Parallelism

DMuon composes with PyTorch's native tensor parallelism (TP) via
`DTensor`.  You apply TP the way you always would — DMuon detects
TP-sharded parameters automatically and routes them through an
All-to-All gather → full-matrix Newton-Schulz → scatter pipeline, so
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
  rank in the TP group as "TP owner".  At each optimizer step:
  1. Every DP-owner rank in the TP group runs a `dist.gather` on
     `reduce_stream`, reassembling the full `(m, n)` gradient at the TP
     owner.  (This piggybacks on the DP reduce stream, so gather
     executes in parallel with the backward compute — verified ~100%
     overlap on 8-GPU 3D HSDP×TP.)
  2. The TP owner runs Newton-Schulz on the **full matrix** (same kernel
     path as non-TP).
  3. `dist.scatter` on `replicate_broadcast_stream` sends each DP-owner
     rank its shard of the update.
  4. The standard HSDP replicate broadcast fans the update out to
     replicate peers.

Sync (`replicate_async=False`) and async (default `replicate_async=True`)
post-step paths produce **bit-identical** weight trajectories on 3D
HSDP×TP — async only changes WHEN the final scatter completes, not the
numerical result.

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

# Optimizer — nothing TP-specific to pass
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

## Sync vs async post-step

`Muon` exposes one TP-relevant flag:

```python
# Default — async scatter + replicate broadcast; each group's post-step
# comm is consumed by the next iteration's forward (overlap).
optimizer = dmuon.Muon(model, lr=0.02)                        # async
optimizer = dmuon.Muon(model, lr=0.02, replicate_async=True)  # explicit

# Sync — scatter + broadcast complete before step() returns.  Useful
# for profiling or for pipelines where the next iter's forward starts
# close enough that overlap doesn't help.
optimizer = dmuon.Muon(model, lr=0.02, replicate_async=False)
```

Both modes produce **bit-identical loss trajectories** on 3D HSDP×TP
(verified 2026-04-24).  Async's benefit is purely that the scatter
NCCL kernels run concurrently with the next iteration's forward
compute — step time is up to ~3% lower on toy 3D meshes, larger on
slow inter-node links.

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
(currently TP rank 0 — the MVP policy; LPT-balanced TP ownership is a
post-MVP follow-up).

---

## Limitations

* **1D TP only** for the MVP.  A multi-dim TP axis (e.g. 2D tensor
  parallel) raises in the detection helper; extend `get_tp_mesh` when
  it's needed.
* **Single-owner NS per TP group.**  The TP owner runs Newton-Schulz
  alone; the other `T-1` ranks wait in `dist.gather` / `dist.scatter`.
  Canzona-style fused All-to-All + micro-group batching that parallelises
  NS across the TP group is listed as future work.
* **Small TP-sharded params do NOT participate in the DMuon small-param
  merge** (`SMALL_PARAM_THRESHOLD`).  Each TP-sharded parameter makes
  its own gather/scatter round-trip even when < 5M numel; in practice
  these are rare.

---

## See also

* [HSDP guide](hsdp.md) — replicate × shard setup; composes with TP
* [`dedicate_params` API](../reference/api.md) — full signature
* `docs/internal/research/tp_design.md` — full design with
  lifecycle diagrams, overlap / fallback semantics, and change log
* `docs/internal/research/tp_overlap_profile.md` — NSight-style
  overlap measurement (100% on 8-GPU 3D mesh toy)
* `docs/internal/research/tp_alignment_report.md` — sync/async
  bit-identical alignment verification
