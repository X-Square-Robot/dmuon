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

### ~~Gradient Accumulation~~ ✅ Done

Two modes supported:
- **Default**: every backward reduces to owner, `_reduced_grad` accumulates automatically. Zero API changes.
- **no_sync**: `dmuon.no_sync(model)` context manager skips reduce, accumulates locally, merges on next sync step. Also controls FSDP2 symmetric params.

## Medium Priority

### ~~State Dict Save/Load~~ ✅ Done

Implemented in `dmuon/checkpoint.py`:
- `get_model_state_dict()` / `set_model_state_dict()`: full state dict for model params
- `get_optimizer_state_dict()` / `set_optimizer_state_dict()`: Muon + AdamW states
- Dedicated params: broadcast from owner. FSDP2 params: manual all-gather/shard.
- Compatible with single-GPU `torch.save`/`torch.load` and HuggingFace checkpoints.

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
