# API Reference

!!! tip "TL;DR"
    DMuon exposes four surface areas: **setup** (`dedicate_params`, `install_patch`),
    **optimizer** (`Muon`, `NewtonSchulz`, NS functions and constants), **state
    management** (`no_sync`, `wait_all_reduces`, replicate-broadcast helpers,
    `DedicatedCommContext`), and **checkpointing** (`get/set_model/optimizer_state_dict`).
    Start with `dedicate_params` + `Muon`; reach for the rest when you need fine-grained
    control.

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

### dedicate_params_ddp

DDP-path setup for dedicated parameters when the data-parallel model is
replicated instead of FSDP2-sharded.

::: dmuon.dedicate_params_ddp

---

### dedicate_params_ddp_tp

DDP-path setup for dedicated parameters when tensor parallelism is active
inside each replicated data-parallel group.

::: dmuon.dedicate_params_ddp_tp

---

### replicate

DDP-style replication helper for non-dedicated parameters.

::: dmuon.replicate

---

### replicate_tp

TP-aware companion to `dedicate_params_ddp_tp()` for non-dedicated
`DTensor` parameters.

::: dmuon.replicate_tp

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
matrix-routed dedicated parameters and AdamW on the base path in a single
object.  The base path can be ordinary FSDP2-managed parameters or
DMuon-managed sharded AdamW parameters selected via `route_hint_fn`.
Compatible with `torch.optim.lr_scheduler`.

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

Inspect which NS kernel is active.  Returns a one-line summary string
(e.g. `"Gram NS · kernel=cute_sm80 (SM80, DMuon internal)"`, `"Gram NS · kernel=quack (SM90, Tri Dao quack)"`, or `"Gram NS · kernel=cublas (SM70, universal fallback)"`).  See
[Backend dispatch](newton-schulz.md#backend-dispatch).

::: dmuon.get_ns_backend

---

### get_backend_status

Full diagnostic dict of the NS kernel dispatch layer — `sm_version`,
`auto_choice`, and per-backend availability flags.  Useful for
programmatic checks and bug reports.

::: dmuon.get_backend_status

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

### prepare_muon_grads

Prepare all pending Muon gradients after backward.  This is broader than a
plain reduce wait because TP-sharded parameters may also need a TP gather before
Muon can run.

::: dmuon.prepare_muon_grads

---

### wait_all_reduces

Backward-compatible alias for `prepare_muon_grads()`.  Called automatically by
`Muon.step()`.

::: dmuon.wait_all_reduces

---

### broadcast_all_updates

Synchronous post-step replicate broadcast (HSDP Phase B).  No-op in 1D mode.
Prefer the async variant unless debugging.

::: dmuon.broadcast_all_updates

---

### broadcast_all_updates_async

Async post-step replicate broadcast (default in `Muon`).  Each layer's event
is consumed at the start of the next forward pass.

::: dmuon.broadcast_all_updates_async

---

### wait_all_replicate_broadcasts

Drain every group's pending async replicate broadcast.  Call before reading
`_owned_data` outside the normal forward/step cycle.

::: dmuon.wait_all_replicate_broadcasts

---

### wait_all_post_step_broadcasts

Compatibility alias for `wait_all_replicate_broadcasts()`.

::: dmuon.wait_all_post_step_broadcasts

---

### clip_grad_norm_

Clip gradients for DMuon-owned Muon parameters.

::: dmuon.clip_grad_norm_

---

### register_muon_grad_clip_strategy

Register a custom strategy for `clip_grad_norm_()`.

::: dmuon.register_muon_grad_clip_strategy

---

### MuonGradClipStats

Return type for DMuon gradient clipping.

::: dmuon.MuonGradClipStats

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
