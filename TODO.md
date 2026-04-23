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

## Completed (2026-04-22 Phase A/B/C sweep)

- [x] **Phase A** — 2D mesh infrastructure (`owner_rank: Tuple[int, int]`, `dedicate_params(replicate_mesh=...)`, LPT over G·R slots)
- [x] **Phase B** — Two-stage reduce (shard AVG + replicate AVG) + sync replicate broadcast; bit-identical vs shard-only baseline on 4 GPU G=2 R=2
- [x] **Phase C** — Async forward-hidden replicate broadcast + per-layer priority + fallback protocol + profile infra; bit-identical vs Phase B sync on 4 GPU
- [x] **HSDP checkpoint** — save/load bit-identical restart on 4 GPU HSDP
- [x] **Docs sweep** — `paper_outline_dmuon.md v2.1` + `paper_strategy_dmuon.md v3` + `canzona_overlap_audit.md` + `landscape_post_canzona.md v2`

---

## Completed (2026-04-23 → 2026-04-24 NS backend dispatch sweep — B1–B8)

Canonical docs: `docs/internal/research/backend_dispatch_design.md`,
`docs/internal/research/ns_backend_dispatch_plan.md`,
`docs/internal/benchmarks/ns_backend_bench_a800.md`,
`docs/internal/benchmarks/quack_smoke_b300.md`.

- [x] **B1** — `syrk_quack.py` soft-dep shell + `ADAPTER_READY` circuit breaker
- [x] **B2** — `syrk_backends.py` unified dispatch (`SyrkBackend` enum, auto-detection ladder, `syrk_dispatch` entry)
- [x] **B3** — `get_ns_backend()` rewrite + new `get_backend_status()` public API
- [x] **B4** — `NewtonSchulz(kernel=…)` + `DMUON_NS_KERNEL` env var + `deterministic` backward-compat alias
- [x] **B5** — Autotune cache split per (GPU × backend) + legacy `.bak_preB5` migration
- [x] **B6-A** — A800 benchmark matrix (24 cells), 1.3–1.7× cute_sm80 speedup at M ≥ 4096
- [x] **B7** — quack API smoke on B300 (SM103): signature, dtype/shape matrix, perf crossover at M ≈ 4096
- [x] **B8** — real `syrk_quack.syrk` adapter + 9 correctness tests; B300 `kernel=auto → quack` end-to-end, Gram NS parity with cuBLAS
- [x] User-facing docs — `reference/newton-schulz.md` (EN+中文) backend dispatch section with 3 ASCII diagrams + quickstart `get_ns_backend()` self-check

---

## NS Backend Dispatch — Phase B-H remainder (hardware-blocked)

Waiting on H100 / B200 access; B300 work is covered in the "Completed" block above.

- [ ] **B9** — H100 + B200 benchmark matrix (re-run `tests/precision/bench_backends.py` on each), verify `kernel="auto"` selects quack, export to `docs/internal/benchmarks/ns_backend_bench_h100.md` + `…_b200.md`
- [ ] **B10** — Cross-backend loss parity (single training step, `kernel="quack"` vs `kernel="cublas"`, bf16 atol=1e-4) + pin `quack-kernels` version range in `pyproject.toml` + changelog entry for the new opt-in backend

---

## Pre-Paper Submission P0 (2026-05 engineering, ~10 days total)

### Naive Baseline Scripts (for E1 per-byte traces + E2 speedup A/B)

- [ ] **Naive Muon-on-DDP baseline** — DDP wrapper + per-rank NS on full grad, ~1d
- [ ] **Naive Muon-on-FSDP-ZeRO2 baseline** — `fully_shard(reshard_after_forward=False)` + manual grad AG + per-rank NS, ~1-2d
- [ ] **Naive Muon-on-FSDP-ZeRO3 baseline** — `fully_shard` default + manual grad AG + per-rank NS, ~1d

### Matrix Optimizer Extension

- [ ] **Shampoo/SOAP DMuon adaptation** — reuse `_owned_data` + owner-step pattern, ~5d. For A4 3×3 matrix (DP × optimizer)

### Experiment Harness

- [ ] **3×3 matrix harness** — 3 DP settings × 3 optimizers benchmark infrastructure, ~3d
- [ ] **E1 per-byte NCCL trace script** — 16 GPU bytes-level validation of Theorems 1/2a/2b/3
- [ ] **E2 speedup benchmark script** — multi-scale (16/32/64/128/256 GPU) × multi-model (Qwen2.5-1.5B/7B, Llama-3)

---

## Phase D (2026-07, cluster-dependent)

### Convergence Validation

Training loss curves and downstream evaluation to verify DMuon preserves Muon's convergence quality:
- From-scratch pretraining on Qwen2.5-1.5B and Llama-3.2-3B
- Dataset: FineWeb-Edu, 10-15B tokens
- Baselines: FSDP2+AdamW, naive DDP+Muon, naive FSDP-Z2+Muon, naive FSDP-Z3+Muon, **DMuon (all 4 DP settings)**
- Metrics: loss vs steps, loss vs wall-clock, perplexity, downstream accuracy

### Multi-Domain Empirical (E3/E4/B4)

- LLM (Qwen2.5, Llama-3)
- VLM (Qwen-VL fine-tune)
- Video diffusion (Wan pretrain)
- VLA robotics (internal)

### Training Examples

Populate `examples/` directory:
- `train_llm.py`: complete LLM training script (single-node multi-GPU)
- `train_tp_dp.py`: TP + DP training example
- `resume_training.py`: checkpoint save/load with resume
- `train_hsdp.py`: 2D mesh HSDP training example

### NSight Profile (E5)

- 32 GPU Qwen2.5-7B HSDP async profile
- Forward IB utilization curve; blocked-wait histogram
- Validate async hiding ≥10% vs Phase B sync

---

## Planned (Paper Submission Window 2026-10/11)

### Larger Model Benchmarks

Extend step time benchmarks to 14B+ and 32B models (Qwen2.5-14B, 32B, Llama-3.1-8B at 256 GPU).

### Communication & Memory Profiling

- torch.profiler traces showing broadcast/reduce overlap with compute
- Per-rank memory breakdown across model sizes, all 4 DP settings
- Bytes-level NCCL traces for Theorems 1/2a/2b/3 empirical validation

### DeepSpeed ZeRO Integration (future work in paper §Discussion)

Described in paper but not implemented for submission. The core dedicated ownership primitive + cross-step async scheduling is framework-portable — only needs two hooks: (1) tell ZeRO to skip dedicated params, (2) register forward/backward hooks. Validates runtime-portable engineering claim in future work.

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
