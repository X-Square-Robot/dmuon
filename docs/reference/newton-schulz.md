# Newton-Schulz Variants

DMuon implements four Newton-Schulz variants, each optimized for different scenarios. This page explains the differences and when to use each.

---

## Overview

| Function | Space | TP Support | SYRK Accel | Restarts | Use Case |
|----------|-------|-----------|------------|----------|----------|
| `newton_schulz()` | Gram | No (local) | Yes | Yes | **Default** — single-rank or DP-only |
| `gram_newton_schulz()` | Gram | Yes | Yes | Yes | **TP params** — exact or block-diagonal |
| `gram_newton_schulz_local()` | Gram | No (local) | Yes | Yes | Internal — same as `newton_schulz` |
| `direct_newton_schulz()` | Direct | No | No | No | **Baseline** — classic Muon/Moonlight algorithm |

## Gram-Space vs Direct-Space

### Direct-Space NS (Classic)

The standard formulation from Muon/Moonlight. Iterates on the full (m, n) matrix:

$$
X_{k+1} = a_k X_k + b_k (X_k X_k^T) X_k + c_k (X_k X_k^T)^2 X_k
$$

- Intermediate matrices are (m, n) — same size as the gradient
- Simple, well-understood
- Cannot exploit Gram matrix symmetry

### Gram-Space NS (Dao-AILab)

Reformulated to iterate on the Gram matrix R = X @ X^T (size m x m when m < n):

$$
Z_k = b_k R_k + c_k R_k^2
$$
$$
Q_{k+1} = Q_k Z_k + a_k Q_k \quad \text{(accumulate product)}
$$
$$
R_{k+1} \text{ evolved from } R_k \text{ and } Z_k
$$

Final output: $X_{\text{out}} = Q \cdot X$

**Advantages:**

- Intermediate matrices are (m, m) when m < n — can be significantly smaller
- R is symmetric → SYRK kernel saves 50% of tiles
- Supports restarts for numerical stability
- TP-compatible: R decomposes as sum of local Gram matrices

## Precision Pipeline

All variants use the same precision strategy:

1. **fp32 normalization**: `X = G.float() / (G.norm() + eps)` — ensures accurate spectral norm
2. **fp16 iteration**: `X = X.half()` — 10-bit mantissa for lower rounding error per step

!!! info "Why fp16 over bf16?"
    After normalization, values are bounded near [0, 1]. fp16's 10-bit mantissa gives better precision than bf16's 7-bit mantissa in this range. The reduced dynamic range of fp16 is not a concern since values are already normalized.

## Coefficient Sets

DMuon ships two coefficient sets, both providing 5 NS iterations:

### POLAR_EXPRESS_COEFFICIENTS (Default)

From the [Polar Express paper](https://arxiv.org/pdf/2505.16932), with a safety factor of 1.05:

```python
POLAR_EXPRESS_COEFFICIENTS = [
    (7.8926, -20.3805, 14.9388),
    (3.9115, -2.5444, 0.4704),
    (3.7607, -2.5120, 0.4762),
    (3.1604, -2.1476, 0.4402),
    (2.1911, -1.4409, 0.3614),
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

To use a different coefficient set:

```python
import dmuon

# Pass to individual NS calls
update = dmuon.newton_schulz(G, coefficients=dmuon.YOU_COEFFICIENTS)
```

!!! note
    The Muon optimizer always uses `POLAR_EXPRESS_COEFFICIENTS` (default). Custom coefficients can be passed directly to NS functions for experimentation.

## Restart Mechanism

Gram-space NS includes a **restart** mechanism (from Dao-AILab): at specified iterations, the accumulated product Q is applied to X, and the Gram matrix R is recomputed from scratch.

This prevents numerical drift from accumulating through the Gram evolution equations.

Default restart position: iteration 2 (i.e., after steps 0 and 1, restart before step 2).

```python
# Custom restarts
update = dmuon.newton_schulz(G, restart_iterations=[2])  # default
update = dmuon.newton_schulz(G, restart_iterations=[1, 3])  # more restarts
```

## SYRK Acceleration

The Gram matrix R = X @ X^T is symmetric. DMuon's CuteDSL SYRK kernel exploits this:

- Only computes the lower triangle of R
- Mirrors the lower triangle to the upper triangle
- Saves ~50% of tiles compared to a general GEMM

This applies to all Gram-space variants (`newton_schulz`, `gram_newton_schulz`, `gram_newton_schulz_local`). The direct-space variant does not benefit from SYRK.

Check the active backend:

```python
print(dmuon.get_ns_backend())
# "syrk_sm80"  — CuteDSL SYRK kernel (SM80+)
# "compiled"   — @torch.compile fallback (any GPU)
```

## TP Routing Summary

When the Muon optimizer encounters a TP-sharded parameter, it routes to the appropriate NS variant:

```
TP param?
├── No → newton_schulz() (local Gram NS)
└── Yes
    ├── per_head_ns=True AND Shard(0) AND full_m < full_n
    │   → newton_schulz() (per-head, zero TP comm)
    ├── block_diagonal_ns=True
    │   → gram_newton_schulz(..., block_diagonal=True) (zero TP comm)
    └── Otherwise
        → gram_newton_schulz(..., shard_dim=dp.shard_dim) (exact, TP all-reduce)
```
