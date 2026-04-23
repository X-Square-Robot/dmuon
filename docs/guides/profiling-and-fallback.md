# Profiling and Fallback

!!! tip "TL;DR"
    Set `DMUON_REPLICATE_PROFILE=1` to collect per-group wait-time histograms for the
    async replicate broadcast. The fallback protocol automatically degrades groups with
    sustained blocked waits (> 100 μs for 3 consecutive steps) to synchronous mode.
    Reset with `dmuon.reset_replicate_fallback(model)`.

---

## The fallback protocol

The async replicate broadcast (Phase C) hides post-step IB traffic inside the next
iteration's forward compute. This works well when the replicate broadcast completes
before the layer that issued it is needed in the next forward. If the IB link is slow
or heavily contended, the broadcast may not finish in time and the forward hook blocks —
hurting throughput more than a straightforward synchronous broadcast would.

DMuon's fallback protocol monitors this per group and automatically degrades to
synchronous mode when a group is consistently slow.

The governing constants (defined in `dmuon/group.py`, accessible at module scope):

| Constant | Default | Meaning |
|---|---|---|
| `REPLICATE_WAIT_THRESHOLD_US` | `100.0` μs | Blocked wait above this triggers a slow-step count |
| `REPLICATE_FALLBACK_CONSECUTIVE_STEPS` | `3` | After this many consecutive slow steps, the group is degraded to sync |

The degradation is **single-direction**: once a group falls back to sync, it stays
there. To re-enable async (after fixing the underlying IB condition):

```python
import dmuon
dmuon.reset_replicate_fallback(model)
```

The fallback monitor only activates when `DMUON_REPLICATE_PROFILE=1` is set, because
timing each wait requires CUDA event synchronisation. In production without the env var,
no timing overhead is incurred and the fallback never trips (groups stay async).

---

## Environment variables

| Variable | Value | Effect |
|---|---|---|
| `DMUON_REPLICATE_PROFILE` | unset / `0` | fully disabled; zero hot-path overhead |
| `DMUON_REPLICATE_PROFILE` | `1` | per-group wait-time samples collected; `replicate_profile_report()` renders a table |
| `DMUON_REPLICATE_PROFILE` | `2` | level 1 plus NSight range markers around dispatch and wait phases |

Basic invocation:

```bash
DMUON_REPLICATE_PROFILE=1 torchrun --nproc_per_node=4 train.py
```

In your training script, call the report from rank 0 after the training loop (or after
a fixed number of steps):

```python title="train.py"
import dmuon

# ... setup and training loop ...

# Print the per-group wait histogram (rank 0 only; no-op on other ranks).
dmuon.replicate_profile_report()
```

---

## Reading the report

Sample output:

```
==============================================================================
[DMUON_REPLICATE_PROFILE] per-group wait time summary (μs)
==============================================================================
                         group     n      mean       p50       p90       p99       max
                 ----------------------------------------
                layers.0.mlp    100     14.22     13.80     18.40     22.30     25.10
                layers.1.mlp    100     18.70     17.90     24.10     28.40     31.80
               layers.10.mlp    100     82.10     79.30    118.40    145.20    312.00
               layers.11.mlp    100     91.50     88.60    132.10    178.90    520.40
                layers.2.mlp    100     15.50     14.90     19.20     23.10     28.80
                     ...
==============================================================================
```

**Columns**: `n` = number of samples; `mean` / `p50` / `p90` / `p99` / `max` = wait
time in microseconds.

**Rules of thumb**:

- `p90 < 100 μs` across all groups: async hiding is working well. No action needed.
- `p99` significantly wider than `p90` (e.g. p90 = 25 μs, p99 = 150 μs): occasional
  IB spikes or brief compute imbalances. Acceptable if infrequent.
- `p90 > 100 μs` on multiple groups: the replicate broadcast is consistently not
  finishing in time. Consider switching to DMuon-Z3 (if on Z2) to reduce IB traffic,
  or widening the threshold and accepting sync fallback.
- `max > 1 ms` on any group: investigate IB saturation or host-side scheduling
  interference (e.g. a large all-gather elsewhere in the forward).

When the fallback protocol trips, a `Fallback events` section appears at the bottom
of the report showing which groups degraded and at which step.

---

## Tuning thresholds

Change the thresholds from Python before training starts. The constants are
module-global in `dmuon.group`; changes take effect immediately for all subsequent
`_update_replicate_fallback` calls:

```python title="tune_thresholds.py"
import dmuon._backends.fsdp2.group as g

# Raise the threshold to 250 μs before declaring a step "slow".
g.REPLICATE_WAIT_THRESHOLD_US = 250.0      # default: 100.0

# Require 5 consecutive slow steps before degrading to sync.
g.REPLICATE_FALLBACK_CONSECUTIVE_STEPS = 5 # default: 3
```

!!! warning "Module-global state"
    These constants affect all `DedicatedParamGroup` instances in the current process.
    If you run multiple experiments in the same Python process, reset the constants
    between runs or use separate processes.

Typical tuning workflow:

1. Run with `DMUON_REPLICATE_PROFILE=1` at default thresholds for 100 steps.
2. Inspect the report: if `p90` on the slowest groups is around 150 μs, raise
   `REPLICATE_WAIT_THRESHOLD_US` to 200 before the full training run.
3. If groups are frequently falling back (many fallback events in the report), either
   raise the threshold further or accept sync mode by setting
   `replicate_async=False` on the `Muon` constructor.

---

## NSight workflow

Level 2 (`DMUON_REPLICATE_PROFILE=2`) inserts NSight range markers around the dispatch
and wait phases of the replicate broadcast. This lets you correlate the CUDA timeline
with the wait histograms.

```bash
DMUON_REPLICATE_PROFILE=2 nsys profile \
    --trace=cuda,nvtx \
    --output=dmuon_trace \
    torchrun --nproc_per_node=4 train.py
```

Open `dmuon_trace.nsys-rep` in NSight Systems and filter on the `DMUON::` NVTX
namespace. Three marker types appear:

| Marker | Meaning |
|---|---|
| `DMUON::replicate_dispatch` | The broadcast was enqueued on the replicate stream |
| `DMUON::replicate_wait` | The current stream blocked waiting for the broadcast |
| `DMUON::replicate_effective` | Gap between dispatch end and wait start — time the broadcast was hidden |

A healthy async trace shows `replicate_effective` intervals covering most of the
preceding forward-layer kernels, with `replicate_wait` appearing as a thin bar at
the layer boundary.

---

## API summary

| Function | Description |
|---|---|
| `dmuon.replicate_profile_report()` | Print the per-group wait histogram from rank 0. No-op on other ranks and when profiling is disabled. |
| `dmuon.reset_replicate_fallback(model)` | Clear the sync-fallback flag on every group, re-enabling async mode. |
| `dmuon.wait_all_replicate_broadcasts(model)` | Drain all pending async replicate broadcasts immediately. Call before checkpointing or any code that reads `_owned_data` outside the forward hook. |

---

## Common diagnosis recipes

### "Training seems slower than expected"

Enable `DMUON_REPLICATE_PROFILE=1` for 200 steps and check `p90` per group. If
several groups show `p90 > 100 μs`, the async broadcast is not hiding:

- If you are on DMuon-Z2: switch to DMuon-Z3 (`reshard_after_forward=True`) to
  reduce total IB bytes per step.
- If already on DMuon-Z3: check whether non-DMuon all-gathers (FSDP2 forward) are
  saturating IB bandwidth. Use NSight level 2 to find overlapping large collectives.
- As a last resort, set `replicate_async=False` on `Muon` to remove async overhead
  entirely and baseline from sync mode.

### "Loss diverges or behaves strangely mid-training"

Extremely unlikely to be caused by the fallback protocol (async and sync paths are
bit-identical). Confirm by checking `replicate_profile_report()` for fallback event
count. If fallback has triggered on many groups, the optimizer state is still correct —
the only difference is that those groups no longer hide their IB traffic.

If you suspect a correctness issue, run with `replicate_async=False` for 50 steps and
compare loss curves. If they match, the issue is unrelated to the async path.

### "Replicate broadcast never overlaps with forward"

If `replicate_effective` markers in NSight are near-zero, the forward pass is too
short to hide the IB transfer. Options:

- Increase batch size to lengthen forward compute.
- Use `replicate_async=False` explicitly — sync mode removes the wasted
  "start async, immediately block" overhead.
- Use HSDP with a larger `shard_size` so the forward pass of each layer is longer.

### "Fallback keeps tripping on one group"

Identify the group name from the fallback events section of the report. If it is always
the same group (e.g. a large embedding or a wide MLP), that group's broadcast may
always take longer than the others. Raise `REPLICATE_WAIT_THRESHOLD_US` to a value just
above that group's `p90`, or exclude that parameter from dedicated ownership via the
`predicate` argument to `dedicate_params`.

---

## See also

- [HSDP Guide](hsdp.md) — the async broadcast in the HSDP context
- [Z2 vs Z3 Modes](z2-z3-modes.md) — reducing IB bytes to help the async path
- [Troubleshooting](../troubleshooting.md) — general training issues
- [API Reference](../reference/api.md) — full function signatures
