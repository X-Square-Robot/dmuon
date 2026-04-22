# API Reference

!!! tip "TL;DR"
    DMuon exposes four surface areas: **setup** (`dedicate_params`, `install_patch`),
    **optimizer** (`Muon`, `NewtonSchulz`, NS functions and constants), **state
    management** (`no_sync`, `wait_all_reduces`, replicate-broadcast helpers,
    `DedicatedCommContext`), and **checkpointing** (`get/set_model/optimizer_state_dict`).
    Start with `dedicate_params` + `Muon`; reach for the rest when you need fine-grained
    control.

---

## Module constants

Two module-level knobs in `dmuon.group` let you tune the async→sync fallback
protocol without touching the optimizer constructor.  Import and mutate before
training starts:

```python
import dmuon.group as g

g.REPLICATE_WAIT_THRESHOLD_US = 250   # default: 100 μs; raise on fast IB networks
g.REPLICATE_FALLBACK_CONSECUTIVE_STEPS = 5  # default: 3; steps before flipping to sync
```

| Name | Default | Description |
|---|---|---|
| `REPLICATE_WAIT_THRESHOLD_US` | `100.0` | Per-layer replicate-broadcast wait above which a step is counted as "slow". |
| `REPLICATE_FALLBACK_CONSECUTIVE_STEPS` | `3` | Consecutive slow steps required before a group permanently switches to sync broadcast. |

Reset a group that has fallen back: `dmuon.reset_replicate_fallback(model)`.

---

## Setup

### dedicate_params

Called once before `fully_shard()`.  Assigns each Muon-target parameter to
a single owner rank and registers the per-layer forward/backward hooks.  See
[Custom Hook Boundaries](../guides/custom-hook-boundaries.md) and
[Z2 vs Z3 Modes](../guides/z2-z3-modes.md) for the two most common
customization points.

::: dmuon.dedicate_params

---

### install_patch

`import dmuon` calls this automatically.  You should never need to call it
directly unless you are constructing a DMuon environment without the normal
import path.

::: dmuon.install_patch

---

## Optimizer

### Muon

The primary optimizer class.  Manages Muon (Newton-Schulz + momentum) on
dedicated parameters and AdamW on FSDP2-managed symmetric parameters in a
single object.  Compatible with `torch.optim.lr_scheduler`.

::: dmuon.Muon

---

### NewtonSchulz

Configurable NS backend object.  Pass to `Muon(ns_backend=...)` to select the
algorithm variant, override coefficients, or enable deterministic mode.  See
[Newton-Schulz Variants](newton-schulz.md) for a full comparison.

::: dmuon.NewtonSchulz

---

### newton_schulz

Standalone NS function (Gram-space by default).  Use directly for NS outside
the optimizer loop.

::: dmuon.newton_schulz

---

### gram_newton_schulz

TP-aware Gram NS with SYRK decomposition.  See
[Tensor Parallelism](../guides/tp-support.md).

::: dmuon.gram_newton_schulz

---

### get_ns_backend

Inspect which hardware backend is active (`"syrk_sm80"` or `"compiled"`).

::: dmuon.get_ns_backend

---

### YOU_COEFFICIENTS

5-step `(a, b, c)` coefficients from
[@YouJiacheng](https://x.com/YouJiacheng/status/1905861218138804534).

::: dmuon.YOU_COEFFICIENTS

---

### POLAR_EXPRESS_COEFFICIENTS

Default 5-step coefficients from Polar Express (arXiv:2505.16932, safety
factor 1.05).  Used when no `coefficients` argument is provided.

::: dmuon.POLAR_EXPRESS_COEFFICIENTS

---

## Utilities — DMuon state management

### no_sync

Context manager for gradient accumulation; suppresses DMuon reduce and
FSDP2's reduce-scatter within the block.  See
[Gradient Accumulation](../guides/grad-accumulation.md).

::: dmuon.no_sync

---

### wait_all_reduces

Drain pending async gradient reduces.  Called automatically by `Muon.step()`.

::: dmuon.wait_all_reduces

---

### broadcast_all_updates

Synchronous post-step replicate broadcast (HSDP Phase B).  No-op in 1D mode.
Prefer the async variant unless debugging.

::: dmuon.broadcast_all_updates

---

### broadcast_all_updates_async

Async post-step replicate broadcast (default in `Muon`).  Each layer's event
is consumed at the start of the next forward pass.  See
[Profiling & Fallback](../guides/profiling-and-fallback.md).

::: dmuon.broadcast_all_updates_async

---

### wait_all_replicate_broadcasts

Drain every group's pending async replicate broadcast.  Call before reading
`_owned_data` outside the normal forward/step cycle.

::: dmuon.wait_all_replicate_broadcasts

---

### reset_replicate_fallback

Re-enable async broadcast on groups that permanently switched to sync.  Safe
to call from the training loop after fixing a slow-IB condition.

::: dmuon.reset_replicate_fallback

---

### replicate_profile_report

Print per-group wait-time summary to stdout (rank 0 only).  Requires
`DMUON_REPLICATE_PROFILE=1`.  Call at the end of training.

::: dmuon.replicate_profile_report

---

### get_dedicated_params

Enumerate all `DedicatedParam` objects across the model.

::: dmuon.get_dedicated_params

---

### get_owned_params

Filter `DedicatedParam` objects owned by a rank.  Accepts `int` (1D) or
`(shard, replicate)` tuple (HSDP).

::: dmuon.get_owned_params

---

### get_comm_ctx

::: dmuon.get_comm_ctx

---

### DedicatedCommContext

Shared CUDA streams (broadcast, reduce, replicate-broadcast) and prefetch
ordering state.  Analogous to FSDP2's `FSDPCommContext`.

::: dmuon.DedicatedCommContext

---

## Checkpointing

All four are **collective** — every rank must call.  Drains async state
before reading/writing.  Standard format: compatible with single-GPU and
HuggingFace checkpoints.  See [Checkpointing](../guides/checkpoint.md).

### get_model_state_dict

::: dmuon.get_model_state_dict

---

### set_model_state_dict

::: dmuon.set_model_state_dict

---

### get_optimizer_state_dict

::: dmuon.get_optimizer_state_dict

---

### set_optimizer_state_dict

::: dmuon.set_optimizer_state_dict

---

## See also

- [Getting Started — Core Concepts](../getting-started/concepts.md)
- [Newton-Schulz Variants](newton-schulz.md)
- [Communication Cost Analysis](communication-cost.md)
- [Checkpointing guide](../guides/checkpoint.md)
