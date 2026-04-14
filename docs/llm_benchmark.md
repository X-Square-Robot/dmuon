# LLM Training Step Benchmark

**Hardware:** 8 x NVIDIA A800-SXM4-80GB
**Config:** bf16, seq_len=2048, batch_size=2
**NS backend:** CuteDSL SYRK (SM80), 5 steps, POLAR_EXPRESS coefficients
**Date:** 2026-04-14

## Methods

| Method | Description |
|--------|-------------|
| FSDP2+AdamW | Standard FSDP2 training with AdamW optimizer (no NS) |
| DDP+Muon | DDP (full model per GPU) + redundant NS on every rank |
| FSDP2+Muon | FSDP2 + all-gather grad per proj param + redundant NS on every rank |
| DMuon | Dedicated ownership, owner-only NS with CuteDSL SYRK, broadcast/reduce |

---

## Qwen2.5-1.5B (1.54B params)

| Method | Forward | Backward | Optimizer | Total | Peak Mem |
|--------|--------:|---------:|----------:|------:|---------:|
| FSDP2+AdamW | 111.1 ms | 200.4 ms | 16.9 ms | 328.5 ms | 23.5 GB |
| DDP+Muon | 94.7 ms | 240.5 ms | 2,371.3 ms | 2,706.6 ms | 28.4 GB |
| FSDP2+Muon | 110.7 ms | 203.0 ms | 2,465.8 ms | 2,779.5 ms | 23.5 GB |
| DMuon | 113.3 ms | 193.7 ms | 32.4 ms | 339.4 ms | 26.8 GB |

**DMuon vs FSDP2+AdamW:** +3% overhead (339 vs 328 ms)
**DMuon vs DDP+Muon:** 8.0x faster
**DMuon vs FSDP2+Muon:** 8.2x faster

---

## Llama-3.2-3B (3.61B params)

| Method | Forward | Backward | Optimizer | Total | Peak Mem |
|--------|--------:|---------:|----------:|------:|---------:|
| FSDP2+AdamW | 197.2 ms | 376.0 ms | 26.4 ms | 599.6 ms | 29.7 GB |
| DDP+Muon | 166.9 ms | 437.1 ms | 2,771.7 ms | 3,375.7 ms | 40.6 GB |
| FSDP2+Muon | 199.8 ms | 382.2 ms | 2,937.7 ms | 3,519.7 ms | 29.7 GB |
| DMuon | 196.3 ms | 364.8 ms | 99.4 ms | 660.4 ms | 38.4 GB |

**DMuon vs FSDP2+AdamW:** +10% overhead (660 vs 600 ms)
**DMuon vs DDP+Muon:** 5.1x faster
**DMuon vs FSDP2+Muon:** 5.3x faster

---

## Qwen2.5-7B (7.62B params)

DDP cannot fit 7B model in single GPU memory.

| Method | Forward | Backward | Optimizer | Total | Peak Mem |
|--------|--------:|---------:|----------:|------:|---------:|
| FSDP2+AdamW | 358.4 ms | 701.3 ms | 53.9 ms | 1,113.6 ms | 48.8 GB |
| FSDP2+Muon | 365.7 ms | 710.8 ms | 22,390.7 ms | 23,467.1 ms | 48.8 GB |
| DMuon | 368.1 ms | 669.9 ms | 189.2 ms | 1,227.2 ms | 65.2 GB |

**DMuon vs FSDP2+AdamW:** +10% overhead (1,227 vs 1,114 ms)
**DMuon vs FSDP2+Muon:** 19.1x faster

---

## Llama-3.1-8B (8.03B params)

DDP cannot fit 8B model in single GPU memory.

| Method | Forward | Backward | Optimizer | Total | Peak Mem |
|--------|--------:|---------:|----------:|------:|---------:|
| FSDP2+AdamW | 380.4 ms | 756.6 ms | 55.0 ms | 1,192.0 ms | 48.4 GB |
| FSDP2+Muon | 388.6 ms | 766.2 ms | 13,715.1 ms | 14,869.9 ms | 48.4 GB |
| DMuon | 371.3 ms | 718.1 ms | 261.7 ms | 1,351.1 ms | 68.7 GB |

**DMuon vs FSDP2+AdamW:** +13% overhead (1,351 vs 1,192 ms)
**DMuon vs FSDP2+Muon:** 11.0x faster

---

## Summary

### Total Step Time (ms)

| Model | FSDP2+AdamW | DDP+Muon | FSDP2+Muon | DMuon | vs AdamW |
|-------|----------:|--------:|-----------:|------:|------:|
| Qwen2.5-1.5B | 328 | 2,707 | 2,780 | 339 | +3% |
| Llama-3.2-3B | 600 | 3,376 | 3,520 | 660 | +10% |
| Qwen2.5-7B | 1,114 | — | 23,467 | 1,227 | +10% |
| Llama-3.1-8B | 1,192 | — | 14,870 | 1,351 | +13% |

### Optimizer-Only Time (ms)

| Model | AdamW | DDP+Muon | FSDP2+Muon | DMuon |
|-------|------:|--------:|-----------:|------:|
| Qwen2.5-1.5B | 16.9 | 2,371 | 2,466 | 32.4 |
| Llama-3.2-3B | 26.4 | 2,772 | 2,938 | 99.4 |
| Qwen2.5-7B | 53.9 | — | 22,391 | 189.2 |
| Llama-3.1-8B | 55.0 | — | 13,715 | 261.7 |

### Key Observations

1. **DMuon adds 3-13% overhead** vs FSDP2+AdamW. The overhead comes from the Muon optimizer step (Newton-Schulz orthogonalization on proj layers), which is the cost of using a matrix optimizer instead of AdamW.

2. **DMuon is 5-19x faster** than naive FSDP2+Muon. Without DMuon, using Muon with FSDP2 requires per-parameter all-gather + redundant NS on every rank, making the optimizer step 100-400x slower than AdamW. DMuon reduces this to 2-5x slower than AdamW by running NS only on the owner rank.

3. **Forward/backward times are similar** across all methods (~same model, same compute). The difference is entirely in the optimizer step.

4. **DMuon uses more memory** than FSDP2 (26-69 GB vs 23-48 GB) because the owner stores full parameters. This is the memory-compute tradeoff of dedicated ownership.

5. **DMuon optimizer scales with model size**: 32ms (1.5B) → 99ms (3B) → 189ms (7B) → 262ms (8B). The owner runs NS on ~1/8 of all proj params, with CuteDSL SYRK accelerating 5/7 symmetric operations.

---

## How to Reproduce

```bash
# All models
torchrun --nproc_per_node=8 benchmarks/bench_llm.py

# Single model
torchrun --nproc_per_node=8 benchmarks/bench_llm.py 1b   # Qwen-1.5B
torchrun --nproc_per_node=8 benchmarks/bench_llm.py 3b   # Llama-3B
torchrun --nproc_per_node=8 benchmarks/bench_llm.py 7b   # Qwen-7B
torchrun --nproc_per_node=8 benchmarks/bench_llm.py 8b   # Llama-8B
```
