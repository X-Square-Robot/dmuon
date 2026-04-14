# TODO

## High Priority

### ~~Gram Newton-Schulz with TP SYRK Decomposition~~ ✅ Done

Implemented in `gram_newton_schulz()` with TP all-reduce + CuteDSL SYRK kernel.
5/7 Gram NS operations accelerated with SYRK (lower-triangle + mirror write).
See `docs/syrk_benchmark.md` for detailed performance data.

### ~~Prefetch Pipeline~~ ✅ Done

Implemented in `DedicatedState`:
- Forward prefetch: `_next_group.unshard()` in `_pre_forward` (linked via `api.py`)
- Backward prefetch: `_backward_prefetch()` in `_DedicatedPreBackward.backward`

### Gradient Accumulation (no_sync)

```python
class DedicatedParamGroup:
    reduce_grads_enabled: bool = True  # set False during no_sync
```

When disabled, non-owner accumulates full gradient locally (higher memory).

## Medium Priority

### State Dict Save/Load

- Owner saves complete parameter
- Load: distribute to owner based on partition
- Conversion: DMuon checkpoint ↔ standard FSDP2 checkpoint ↔ single-GPU

### Owner Zero-Copy Optimization

Owner currently copies `_owned_data` into broadcast buffer. Can avoid this by using `_owned_data` directly as the broadcast source.

### HSDP Support

Hybrid Sharded Data Parallel: `(Replicate, Shard)` mesh. DMuon needs to handle the replicate dimension.

## Low Priority

### torch.compile Support

FSDP2's traceable mode expects specific communication patterns. DMuon's broadcast/reduce hooks need compiler compatibility.

### Activation Checkpointing Validation

AC recomputation triggers DMuon hooks (broadcast from owner). Should work since owner always has data, but needs thorough testing.

### Online Switching

Switch between AdamW (symmetric) and Muon (dedicated) mid-training. Requires re-partitioning parameters without restart.
