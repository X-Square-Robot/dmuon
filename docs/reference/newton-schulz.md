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

## Backend dispatch

Newton-Schulz has two independent axes: the **algorithm** (Gram vs. direct)
and the underlying **SYRK kernel** implementation.  DMuon dispatches both
automatically and exposes each as an override knob.

### Two-axis architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User API:  dmuon.NewtonSchulz(                             │
│                 backend="gram",     ← Axis 1: algorithm     │
│                 kernel="auto",       ← Axis 2: SYRK kernel  │
│             )                                                │
├─────────────────────────────────────────────────────────────┤
│  Axis 1 — Algorithm                                         │
│     "gram"    → Gram-space NS + SYRK ops + restarts (default)│
│     "direct"  → classic parameter-space NS                  │
├─────────────────────────────────────────────────────────────┤
│  Axis 2 — SYRK kernel backend                               │
│     "auto"       → pick best for current GPU (default)      │
│     "quack"      → Tri Dao quack (SM90+, opt-in soft dep)   │
│     "cute_sm80"  → DMuon-internal CuteDSL (SM80/87 only)    │
│     "cublas"     → torch.mm / torch.addmm (universal)       │
└─────────────────────────────────────────────────────────────┘
```

The two axes are orthogonal — any `backend` × `kernel` combination is
valid.  Direct-space NS does not use SYRK, so the `kernel` argument is
a no-op when `backend="direct"`.

### Auto-detection ladder

With `kernel="auto"` (the default), DMuon picks the fastest available
backend for the current device:

```
SM version detected at import ─►
    ┌── SM ≥ 90  ─── quack installed?  ── yes ──► quack
    │                                │
    │                                └── no  ──► cublas  + warn
    │
    ├── SM 80/87 ─── cute_sm80 built? ── yes ──► cute_sm80
    │                                │
    │                                └── no  ──► cublas
    │
    └── SM < 80  ─────────────────────────────► cublas
```

Graceful degradation is the rule: `kernel="auto"` always picks something
that works, logging the chosen path at startup.  Explicit `kernel="quack"`
on an SM80 device fails fast with an install hint.

### Resolution priority

When multiple knobs are set, precedence is:

```
explicit NewtonSchulz(kernel=...)      ← highest (always wins)
          │
          ▼ only if kernel left at "auto"
DMUON_NS_KERNEL env var
          │
          ▼ only if env unset
deterministic=True                     ← legacy alias, maps to "cublas"
          │
          ▼
auto-detected default
```

Setting `deterministic=True` and `kernel="cute_sm80"` simultaneously
emits a warning and honours the explicit kernel.

### Inspecting the active backend

```python
import dmuon

# Human-readable one-liner — good for startup logs
print(dmuon.get_ns_backend())
# "Gram NS · kernel=cute_sm80 (SM80, DMuon internal)"
# "Gram NS · kernel=quack (SM90, Tri Dao quack)"
# "Gram NS · kernel=cublas (SM80, universal fallback)"

# Full diagnostic dict — good for bug reports / programmatic checks
print(dmuon.get_backend_status())
# {
#   "sm_version": 80,
#   "auto_choice": "cute_sm80",
#   "quack_available": False,
#   "cute_sm80_available": True,
#   "cublas_always_available": True,
# }
```

### Forcing a specific kernel

```python
# Force cuBLAS for bit-exact reproducibility across runs
ns = dmuon.NewtonSchulz(kernel="cublas")
ns = dmuon.NewtonSchulz(deterministic=True)   # legacy equivalent

# Force the SM80 CuteDSL kernel (raises if cute_sm80 wasn't built)
ns = dmuon.NewtonSchulz(kernel="cute_sm80")

# Cluster-wide override via env var (takes effect only when code uses "auto")
# export DMUON_NS_KERNEL=cublas
```

!!! info "quack backend"
    The `quack` SYRK backend is enabled on SM90+ devices when the
    `quack-kernels` soft dependency is installed (`pip install dmuon[quack]`).
    It is validated by the optional backend tests and is expected to be most
    useful on large matrices where SM90+ symmetric GEMM kernels have enough
    work to amortize dispatch overhead.

    A runtime circuit-breaker
    `dmuon.kernels.syrk_quack.ADAPTER_READY` can be flipped to `False`
    to emergency-disable the quack path without uninstalling the
    package; `kernel="auto"` then falls back to `cublas`.

    `get_backend_status()["auto_choice"]` always reports the kernel
    that will actually run, so you can see ground truth at a glance.

---

## TP handling

The NS kernels (`newton_schulz`, `gram_newton_schulz`,
`direct_newton_schulz`) are **TP-agnostic**: they operate on a full
(un-sharded) matrix and have no `tp_group` argument.  For TP-sharded
parameters the DMuon runtime reassembles the full matrix at a
designated TP owner via TP gather before invoking NS, then
scatters the update back to each DP-owner rank:

```
DP reduce  →  TP gather (dist.gather on reduce_stream)  →
    Newton-Schulz on full (m, n) matrix at TP owner  →
TP scatter (dist.scatter on replicate_broadcast_stream)  →
    replicate broadcast
```

This is automatic for any `DTensor` parameter whose `device_mesh`
contains a mesh dim outside the DP dim names — no explicit TP flag on
`dmuon.Muon`.  TP ownership is picked by the deterministic LPT assignment
inside `compute_balanced_assignment`, so TP-sharded full-matrix work is
spread across local TP ranks while preserving the same loss trajectory.

Practical consequences:

* Same NS precision regardless of TP — the kernel always sees the full
  matrix.
* Extra comm cost per TP-sharded param: one `dist.gather` + one
  `dist.scatter`, both sized `(T − 1)/T · |p|`.  Both run on DMuon's
  dedicated comm streams and empirically achieve ~100% overlap with
  backward compute on 8-GPU 3D HSDP×TP toy.
* No change to non-TP param behaviour.

See the [TP support guide](../guides/tp-support.md) for setup, the
full lifecycle, and the sync / async semantics.

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
