# Newton-Schulz Variants

!!! tip "TL;DR"
    DMuon ships two NS backends: **Gram-space** (`"gram"`, default) and
    **direct-space** (`"direct"`).  Gram-space is faster (SYRK kernel, restart
    mechanism, smaller intermediates) and is the right choice for all production
    use.  Direct-space is the classic Muon/Moonlight formulation — useful for
    baselines and small matrices.  Both accept custom `(a, b, c)` coefficient
    sets (`POLAR_EXPRESS_COEFFICIENTS` default, `YOU_COEFFICIENTS` alternative).

---

## Overview

| Function | Space | TP support | SYRK accel | Restarts | Use case |
|---|---|---|---|---|---|
| `newton_schulz()` | Gram | No (local) | Yes | Yes | **Default** — single-rank or DP-only |
| `gram_newton_schulz()` | Gram | Yes | Yes | Yes | **TP params** — exact or block-diagonal |
| `NewtonSchulz("gram")` | Gram | Routing | Yes | Yes | Pass to `Muon(ns_backend=...)` |
| `NewtonSchulz("direct")` | Direct | No | No | No | Baseline / ablation |
| `direct_newton_schulz()` | Direct | No | No | No | Direct function call |

---

## Direct-space NS (classic)

The standard formulation from Muon (Jordan et al., 2024) and Moonlight.
Iterates on the full (m, n) matrix:

$$
X_{k+1} = a_k X_k + b_k (X_k X_k^T) X_k + c_k (X_k X_k^T)^2 X_k
$$

Properties:

- Intermediate matrices are (m, n) — same size as the gradient
- No symmetry exploitation; general GEMM cost per step
- No restart mechanism
- Simple, well-understood, good for baseline comparison

Use `NewtonSchulz("direct")` or call `direct_newton_schulz()` directly.

---

## Gram-space NS (Dao-AILab)

Reformulates NS to iterate on the Gram matrix $R = X X^T$ of size (m, m),
adapted from [Dao-AILab/gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz):

$$
Z_k = b_k R_k + c_k R_k^2
$$
$$
Q_{k+1} = Z_k Q_k + a_k Q_k \quad (\text{accumulated product})
$$

$R$ is evolved from $R_k$ and $Z_k$ using the recurrence; at restart steps
$Q$ is applied to $X$ and $R$ is recomputed from scratch.  Final output:
$X_{\text{out}} = Q \cdot X$.

**Advantages over direct-space:**

- Intermediate matrices are (m, m); significantly smaller when m < n (typical
  for wide projection layers)
- $R$ is symmetric — the CuteDSL SYRK kernel saves ~50 % of tiles
- Restart mechanism prevents numerical drift
- $R$ decomposes as a sum of local Gram matrices — enables exact TP via all-reduce

---

## Precision pipeline

All variants use the same two-stage precision strategy:

1. **fp32 normalization**: `X = G.float() / (G.norm() + eps)` — stabilizes the
   spectral norm before iteration
2. **fp16 iteration**: `X = X.half()` — 10-bit mantissa gives lower per-step
   rounding error than bf16's 7-bit mantissa for values bounded near [0, 1]

!!! info "Why fp16 over bf16?"
    After normalization the values sit near [0, 1].  fp16's wider mantissa (10
    bits) provides better precision in this range.  The reduced dynamic range of
    fp16 is not a concern because the normalization step already bounds the
    values.

---

## Coefficient sets

DMuon ships two coefficient sets, both providing 5 Newton-Schulz iterations.

### POLAR_EXPRESS_COEFFICIENTS (default)

From the Polar Express paper (arXiv:2505.16932), with a safety factor of 1.05
applied to the raw coefficients:

```python
# Approximate values after safety scaling
POLAR_EXPRESS_COEFFICIENTS = [
    (7.893, -20.381, 14.939),
    (3.912, -2.544,  0.470),
    (3.761, -2.512,  0.476),
    (3.160, -2.148,  0.440),
    (2.191, -1.441,  0.361),
]
```

### YOU_COEFFICIENTS

From [@YouJiacheng](https://x.com/YouJiacheng/status/1905861218138804534):

```python
YOU_COEFFICIENTS = [
    [4.0848, -6.8946, 2.9270],
    [3.9505, -6.3029, 2.6377],
    [3.7418, -5.5913, 2.3037],
    [2.8769, -3.1427, 1.2046],
    [2.8366, -3.0525, 1.2012],
]
```

### Using a custom coefficient set

```python
import dmuon

# Override via NewtonSchulz object
ns = dmuon.NewtonSchulz("gram", coefficients=dmuon.YOU_COEFFICIENTS)
optimizer = dmuon.Muon(model, lr=0.02, ns_backend=ns)

# Or pass directly to standalone NS functions
update = dmuon.newton_schulz(G, coefficients=dmuon.YOU_COEFFICIENTS)
```

!!! note
    `Muon` uses `POLAR_EXPRESS_COEFFICIENTS` by default.  The You coefficients
    are available for experiments where the original Muon formulation is desired.

---

## Restart mechanism

Gram-space NS includes a **restart** mechanism adapted from
Dao-AILab/gram-newton-schulz.  At specified iteration indices the accumulated
product $Q$ is applied back to $X$ and the Gram matrix $R$ is recomputed from
scratch, preventing numerical drift from the Gram evolution recurrence.

Default restart position: `[2]` (restart after iterations 0 and 1, before
iteration 2).

```python
import dmuon

# Default restarts
update = dmuon.newton_schulz(G, restart_iterations=[2])

# More aggressive restarts
ns = dmuon.NewtonSchulz("gram", restart_iterations=[1, 3])
```

---

## SYRK acceleration

The Gram matrix $R = X X^T$ is symmetric.  DMuon's CuteDSL SYRK kernel
(adapted from [Dao-AILab/quack](https://github.com/Dao-AILab/quack)) exploits
this:

- Computes only the lower-triangular half of $R$
- Mirrors the result to the upper triangle
- Saves approximately 50 % of tiles vs. a general GEMM

This applies to all Gram-space variants (`newton_schulz`, `gram_newton_schulz`).
The direct-space variant does not use SYRK.

Check the active backend:

```python
import dmuon

print(dmuon.get_ns_backend())
# "syrk_sm80"  — CuteDSL SYRK kernel (SM80+, e.g. A100/H100)
# "compiled"   — @torch.compile fallback (any CUDA GPU)
```

The SYRK kernel activates automatically on SM80+ GPUs when CuteDSL is
available.  It falls back to `@torch.compile` PyTorch on other hardware.

### Deterministic mode

The SYRK kernel may produce non-deterministic results across runs due to
float accumulation order.  Force cuBLAS for exact reproducibility:

```python
ns = dmuon.NewtonSchulz(deterministic=True)
optimizer = dmuon.Muon(model, ns_backend=ns)
```

!!! warning "SYRK B != A bug"
    The CuteDSL SYRK kernel has a known non-determinism issue when `B != A`
    (certain intermediate computations in the Gram recurrence).  The workaround
    is `deterministic=True`, which routes all ops through cuBLAS at a ~1.5x
    performance cost.  This is being tracked for a future kernel fix.

---

## TP routing summary

When `Muon` encounters a TP-sharded parameter, it selects the NS path
according to the following decision tree:

```
Is the param a DTensor with a TP group?
├── No  → NewtonSchulz.local()  (standard Gram NS or direct, no comm)
└── Yes
    ├── per_head_ns=True AND Shard(0) AND full_m < full_n
    │   → NewtonSchulz.local()   (per-head, zero TP comm)
    ├── block_diagonal_ns=True
    │   → NewtonSchulz.tp(..., block_diagonal=True)   (zero TP comm)
    └── Otherwise (default)
        → NewtonSchulz.tp(..., shard_dim=dp.shard_dim)  (exact Gram, TP all-reduce)
```

For **Shard(0)** (row-sharded): the iteration transposes to use $G^T G$, which
decomposes exactly as $\sum_i G_i^T G_i$ — one all-reduce gives the exact Gram.
For **Shard(1)** (col-sharded): uses $G G^T$, which decomposes as
$\sum_i G_i G_i^T$.

---

## References and acknowledgments

- **Gram Newton-Schulz** — Dao et al., 2026.  Blog post:
  [dao-ailab.github.io/blog/2026/gram-newton-schulz/](https://dao-ailab.github.io/blog/2026/gram-newton-schulz/).
  Source: [Dao-AILab/gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz).
  DMuon's Gram NS logic, per-step coefficients, restart mechanism, and SYRK
  symmetry optimization are adapted directly from this work.
- **SYRK kernel** — adapted from [Dao-AILab/quack](https://github.com/Dao-AILab/quack)
  by Tri Dao et al.
- **Muon optimizer** — Jordan et al., arXiv:2502.16982, 2024.  Introduced the
  momentum + Newton-Schulz orthogonalization formulation that DMuon extends.
- **Polar Express coefficients** — arXiv:2505.16932.
- **You coefficients** — [@YouJiacheng](https://x.com/YouJiacheng/status/1905861218138804534).

---

## See also

- [API Reference](api.md)
- [Communication Cost Analysis](communication-cost.md)
- [Training guide](../guides/training.md)
- [Tensor Parallelism](../guides/tp-support.md)
