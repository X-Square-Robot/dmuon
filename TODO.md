# TODO

## Completed

- [x] Core dedicated ownership model (broadcast/reduce/owner-only NS)
- [x] Balanced partition with concurrency constraints (LPT + same-layer spreading)
- [x] FSDP2 composition via monkey-patch (zero modification to FSDP2 internals)
- [x] TP compatibility (DTensor-aware broadcast/reduce + Gram NS)
- [x] Gram NS with TP SYRK decomposition (O(m^2) TP comm instead of O(mn))
- [x] CuteDSL SYRK kernel (5/7 Gram NS ops accelerated, 1.4-1.5x E2E speedup)
- [x] Prefetch pipeline (forward + backward)
- [x] Gradient accumulation (default + `no_sync` context manager)
- [x] State dict save/load (compatible with single-GPU and HuggingFace checkpoints)
- [x] LLM step time benchmarks (Qwen2.5-1.5B/7B, Llama-3.2-3B/3.1-8B, 8xA800)

---

## In Progress

### Convergence Validation

Training loss curves and downstream evaluation to verify DMuon preserves Muon's convergence quality:
- From-scratch pretraining on Qwen2.5-1.5B and Llama-3.2-3B
- Dataset: FineWeb-Edu, 10-15B tokens
- Baselines: FSDP2+AdamW, DDP+Muon, FSDP2+Muon (naive), DMuon
- Metrics: loss vs steps, loss vs wall-clock, perplexity, downstream accuracy

### HSDP Support

Hybrid Sharded Data Parallel: `(Replicate, Shard)` mesh for multi-node training. DMuon needs to handle the replicate dimension — prerequisite for multi-node scaling experiments.

### Training Examples

Populate `examples/` directory:
- `train_llm.py`: complete LLM training script (single-node multi-GPU)
- `train_tp_dp.py`: TP + DP training example
- `resume_training.py`: checkpoint save/load with resume

---

## Planned

### Multi-Node Scaling

Strong and weak scaling experiments on 16-64 GPUs across multiple nodes. Depends on HSDP support.

### Larger Model Benchmarks

Extend step time benchmarks to 14B+ models (e.g., Qwen2.5-14B).

### Optimizer Generalization

Extend DMuon's ownership model beyond Muon:
- **L-only Shampoo**: left preconditioner with SYRK acceleration, fully reusing TP infrastructure
- **SOAP-Muon hybrid**: left-side SOAP eigenbasis + right-side Muon NS

### Communication & Memory Profiling

- torch.profiler traces showing broadcast/reduce overlap with compute
- Per-rank memory breakdown across model sizes

### Ablation Studies

- Gram NS vs standard NS (communication savings)
- SYRK kernel vs cuBLAS fallback
- Prefetch pipeline on/off
- Partition quality: LPT vs round-robin

### Activation Checkpointing Validation

AC recomputation triggers DMuon hooks (broadcast from owner). Should work since owner always has data, but needs thorough testing.

### CI/CD

GitHub Actions for automated linting + unit tests on push/PR.

---

## Future Directions

- Owner zero-copy broadcast optimization
- Optimizer state quantization (FP16/INT8 momentum)
- torch.compile support
- Pipeline Parallelism support
- HuggingFace Trainer / torchtitan integration
- Online switching between AdamW and Muon mid-training
