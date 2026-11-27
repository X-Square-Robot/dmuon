# Ownership Domains (Multi-Parallelism) {#ownership-domains}

<p class="dmuon-lead">Combine DP, HSDP, CP, EP, PP, and TP in a single <code>dedicate_params()</code> call with one shared communication context, per-domain gradient scaling, domain-local checkpointing, and global/per-algorithm gradient clipping.</p>

When every parameter in a model shares the same replica topology (pure DP, 1D FSDP2, 2D HSDP, or DP+TP), passing `mesh` (and optional `replicate_mesh`) to `dmuon.dedicate_params()` is all you need.

In hybrid parallel architectures such as **Mixture-of-Experts with Expert Parallelism (EP)** or **Pipeline Parallelism (PP) + EP + HSDP**, different parameter subsets live on different replica axes:

- **Dense parameters** (attention, norms, routers, embeddings, shared experts) are replicated across `dp × cp × ep`.
- **EP-sharded experts** are unique across `ep` ranks and replicated only across `dp × cp`, with gradients scaled by `1.0 / ep_size` to match global loss normalization.
- **Mixed 1D + 2D HSDP** models can keep dense parameters on a 2D `(replicate, shard)` HSDP domain while experts live on a 1D `dp` domain.

`OwnershipDomain` makes those topologies first-class inside DMuon — without calling `dedicate_params()` twice, without splitting `DedicatedCommContext`, and without disabling async publish or prefetch.

---

## Pure DP / FSDP2 / HSDP remains unchanged {#single-domain-compat}

Existing single-domain callers need **zero code changes**:

```python
# Unchanged — internally wrapped as a single OwnershipDomain("default", shard_mesh=mesh, ...)
dmuon.dedicate_params(model, mesh, replicate_mesh=rep_mesh, predicate=...)
```

When `domains` is not passed:

- `grad_scale == 1.0` skips any scaling kernel (zero overhead),
- LPT owner assignment, async publish, and checkpoint schemas stay bit-for-bit identical.

---

## Defining domains for MoE + EP {#moe-ep-example}

Pass `domains` (a dict of `name -> OwnershipDomain`) and `default_domain` to `dmuon.dedicate_params()`, and assign per-parameter domains via `"domain"` in `param_policy` or `param_policy_fn`:

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
import dmuon
from dmuon import MatrixLayout, OwnershipDomain

# Example: 4 GPUs with DP=2, EP=2 (also works for DP=1, EP=4)
mesh = init_device_mesh("cuda", (dp_size, ep_size), mesh_dim_names=("dp", "ep"))
dense_mesh = mesh["dp", "ep"]._flatten()
expert_mesh = mesh["dp"]

domains = {
    "dense": OwnershipDomain(
        name="dense",
        shard_mesh=dense_mesh,
        grad_scale=1.0,
    ),
    "expert": OwnershipDomain(
        name="expert",
        shard_mesh=expert_mesh,
        grad_scale=1.0 / ep_size,
    ),
}

def policy_fn(name: str, param: torch.nn.Parameter) -> dict[str, object]:
    is_expert = ".mlp.experts." in name
    domain = "expert" if is_expert else "dense"
    if is_expert and param.ndim == 3:
        return {
            "domain": domain,
            "route": "muon",
            "matrix_layout": MatrixLayout.view(*param.shape),
            "param_dtype": torch.bfloat16,
            "grad_dtype": torch.bfloat16,
        }
    if name.endswith(("embed_tokens.weight", "lm_head.weight")):
        return {"domain": domain, "route": "sharded_adamw"}
    if param.ndim >= 2:
        return {"domain": domain, "route": "muon"}
    return {"domain": domain, "route": "adamw"}

dmuon.dedicate_params(
    model,
    domains=domains,
    default_domain="dense",
    predicate=lambda _n, p: p.requires_grad,
    param_policy_fn=policy_fn,
)

optimizer = dmuon.Muon(
    model,
    lr=0.02,
    adamw_lr=3e-4,
    replicate_async=True,
    group_prepare_ahead=True,
    forward_prefetch_depth=1,
)
```

### What happens internally

1. **Per-domain planning**: each domain runs its own LPT owner assignment over its own `shard_mesh` (and optional `replicate_mesh`), so expert owners are chosen strictly inside `expert_mesh` while dense owners spread across the full `dense_mesh`.
2. **One shared `DedicatedCommContext`**: all domains share the same CUDA streams, `all_states`, `post_forward_order`, and forward/backward prefetch chain.
3. **Per-group self-contained execution**: each `DedicatedParamGroup` holds parameters from a single domain along with its own `_dp_group`, `_replicate_group`, `_grad_reduce_op`, and `_grad_scale`.
4. **Correct `grad_scale` with `no_sync`**: `grad_scale` is applied once when cross-rank reduction completes (`wait_for_reduce`), covering Muon, `MatrixLayout`, AdamW, and `sharded_adamw`. Local accumulation inside `dmuon.no_sync(model)` never scales gradients multiple times.

---

## Multi-parallelism topology recipes {#topology-recipes}

| Parallelism Setup | Parameter Class | `shard_mesh` | `replicate_mesh` | `grad_scale` |
|---|---|---|---|---|
| Pure DP / FSDP2 | All | `mesh["dp"]` | `None` | `1.0` |
| 2D HSDP | All | `mesh["shard"]` | `mesh["replicate"]` | `1.0` |
| DP × EP (incl. `DP=1, EP=4`) | Dense | `mesh["dp", "ep"]._flatten()` | `None` | `1.0` |
| DP × EP | Local experts | `mesh["dp"]` | `None` | `1.0 / ep_size` |
| DP × CP × EP | Dense | `mesh["dp", "cp", "ep"]._flatten()` | `None` | `1.0` |
| DP × CP × EP | Local experts | `mesh["dp", "cp"]._flatten()` | `None` | `1.0 / ep_size` |
| PP × DP × EP | Dense (per PP stage) | `mesh["dp", "ep"]._flatten()` | `None` | `1.0` |
| PP × DP × EP | Experts (per PP stage) | `mesh["dp"]` | `None` | `1.0 / ep_size` |
| Mixed 2D HSDP + 1D EP | Dense | `mesh["shard"]` | `mesh["replicate"]` | `1.0` |
| Mixed 2D HSDP + 1D EP | Local experts | `mesh["replicate"]` (or `mesh["dp"]`) | `None` | `1.0 / ep_size` |
| 5D (PP × DP × CP × EP × TP) | TP-sharded dense | `mesh["dp", "cp", "ep"]._flatten()` | `None` | `1.0` |
| 5D (PP × DP × CP × EP × TP) | Non-TP replicated | `mesh["dp", "cp", "ep", "tp"]._flatten()` | `None` | `1.0` |

!!! note "Tensor Parallelism stays automatic"
    For TP-sharded `DTensor` parameters, do not include the `"tp"` axis in `shard_mesh` or `replicate_mesh`. DMuon detects TP per domain by comparing the parameter's `DTensor` mesh dim names against that domain's `dp_dim_names`.

---

## Multi-domain gradient clipping {#multi-domain-clipping}

Because every logical parameter element across all domains (DP/HSDP/CP/EP/PP/TP) has a unique owner (or disjoint `sharded_adamw` slice), global or per-algorithm gradient clipping needs only **one `all_reduce(SUM)`** over `norm_reduce_group` (default: `WORLD`):

```python
# Global L2 clipping across all owned Muon + AdamW + sharded_adamw gradients
result = dmuon.clip_grad_norm_(model, max_norm=1.0, mode="global")

# Independent L2 clipping for Muon vs AdamW (including sharded_adamw)
result = dmuon.clip_grad_norm_(model, max_norm=1.0, mode="per_algorithm")
muon_norm = result.stats_by_name["muon"].total_norm
adamw_norm = result.stats_by_name["adamw"].total_norm
```

Key properties:

- Automatically waits pending reduces (`wait_all_reduces(model)`) before reading gradients.
- Slices `sharded_adamw` gradients to `[:_sharded_adamw_valid_numel]` so padding never skews norms.
- Uses fixed bucket layouts on every rank, so pipeline stages or ranks with zero local gradients in one bucket (e.g., an embedding-only stage with 0 Muon params) stay strictly in lockstep.
- Passing a `dmuon.Muon` optimizer with `mode=None` preserves the legacy Muon-only `clip_grad_norm_(optimizer, max_norm)` behavior.

---

## Multi-domain & PP × EP checkpointing {#multi-domain-checkpoint}

`dmuon.get_model_state_dict`, `dmuon.get_optimizer_state_dict`, `dmuon.set_model_state_dict`, and `dmuon.set_optimizer_state_dict` synchronize optimizer state presence strictly within each parameter's `(dp_group, replicate_group[, tp_group])` — **zero `WORLD` collectives**:

```python
# Each PP stage / EP lane exports its own full local module + optimizer state
model_sd = dmuon.get_model_state_dict(model, cpu_offload=True, rank0_only=False)
optim_sd = dmuon.get_optimizer_state_dict(model, optimizer, cpu_offload=True, rank0_only=False)

# Restore on the same topology
dmuon.set_model_state_dict(model, model_sd)
dmuon.set_optimizer_state_dict(model, optimizer, optim_sd)
```

- Pipeline stages with different parameter counts (e.g., Stage 0 dense-only vs Stage 1 dense + MoE) export and reload independently without hanging.
- Cold experts that received no gradients remain absent from `optim_sd["dedicated"]` / `optim_sd["dedicated_adamw"]` and do not broadcast uninitialized placeholders.

---

## Gotchas {#gotchas}

1. **Pass the same `domains` keys on every PP stage** when using `process_group_policy="isolated"` (the default). Even if Stage 0 matches zero parameters in `"expert"`, passing `domains={"dense": ..., "expert": ...}` on every stage keeps isolated process-group creation lockstep across `WORLD`.
2. **Do not pass both `mesh` and `domains`**: `dedicate_params()` raises `ValueError` if both are specified.
3. **Never manually divide expert gradients by `ep_size`**: set `grad_scale=1.0 / ep_size` on the `"expert"` `OwnershipDomain` instead so scaling works identically across Muon, `MatrixLayout`, AdamW, `sharded_adamw`, and `no_sync` gradient accumulation.
