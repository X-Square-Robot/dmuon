# TODO

## High Priority

### Gram Newton-Schulz with TP SYRK Decomposition

Currently, when TP is enabled, the owner runs standard NS on its TP shard. This is **mathematically incorrect** — NS involves `G @ G^T` which couples columns, so `NS(G_shard) ≠ NS(G_full)_shard`.

The correct approach uses Gram NS with SYRK column decomposition:

```python
# Owner has G_i (m, n/T) — one TP shard

# 1. Local SYRK (column-decomposable)
gram_local = G_i @ G_i.T              # (m, m) — local

# 2. TP all-reduce (the ONLY TP communication)
dist.all_reduce(gram_local, group=tp_group)  # (m, m) — exact G @ G^T

# 3. NS iterations on Gram matrix (m×m, all TP ranks identical)
Q = gram_ns(gram_local)               # (m, m) — local

# 4. Project back (column-parallel, zero communication)
update_i = Q @ G_i                     # (m, n/T) — local
```

**Why this works**: `G @ G^T = Σ G_i @ G_i^T` — SYRK sums over the column (inner) dimension, which is exactly what TP splits.

**Communication reduction**: O(m²) instead of O(m×n). For gate_proj (3584, 18944) with TP=8: 25.6M elements vs 135.8M — **5.3× less**.

**Implementation needed**:
- DedicatedParam needs `tp_group` reference
- New `gram_newton_schulz_tp()` function
- Benchmark: Gram NS TP comm vs standard NS TP comm

### Prefetch Pipeline

Overlap next-layer broadcast with current-layer forward compute:

```python
class DedicatedState:
    _next_state: Optional['DedicatedState'] = None

    def _post_forward(self, module, input, output):
        self.group.reshard()
        if self._next_state:
            self._next_state.group.prefetch_unshard()  # async broadcast
        ...
```

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
