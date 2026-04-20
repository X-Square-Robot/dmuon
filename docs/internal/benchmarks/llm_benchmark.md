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
| FSDP2+AdamW | 109.9 ms | 200.5 ms | 17.3 ms | 327.6 ms | 23.5 GB |
| DDP+Muon | 91.7 ms | 212.6 ms | 320.7 ms | 625.0 ms | 28.4 GB |
| FSDP2+Muon | 109.0 ms | 201.8 ms | 373.0 ms | 683.8 ms | 23.5 GB |
| DMuon | 114.9 ms | 193.8 ms | 31.1 ms | 339.9 ms | 26.8 GB |

**DMuon vs FSDP2+AdamW:** +4% overhead (340 vs 328 ms)
**DMuon vs DDP+Muon:** 1.8x faster (optimizer: 10.3x)
**DMuon vs FSDP2+Muon:** 2.0x faster (optimizer: 12.0x)

---

## Llama-3.2-3B (3.61B params)

| Method | Forward | Backward | Optimizer | Total | Peak Mem |
|--------|--------:|---------:|----------:|------:|---------:|
| FSDP2+AdamW | 197.5 ms | 374.9 ms | 26.5 ms | 598.9 ms | 29.7 GB |
| DDP+Muon | 164.4 ms | 397.7 ms | 1,108.6 ms | 1,670.7 ms | 40.6 GB |
| FSDP2+Muon | 198.1 ms | 379.3 ms | 1,232.2 ms | 1,809.5 ms | 29.7 GB |
| DMuon | 197.1 ms | 363.7 ms | 98.8 ms | 659.6 ms | 38.4 GB |

**DMuon vs FSDP2+AdamW:** +10% overhead (660 vs 599 ms)
**DMuon vs DDP+Muon:** 2.5x faster (optimizer: 11.2x)
**DMuon vs FSDP2+Muon:** 2.7x faster (optimizer: 12.5x)

---

## Qwen2.5-7B (7.62B params)

DDP cannot fit 7B model in single GPU memory.

| Method | Forward | Backward | Optimizer | Total | Peak Mem |
|--------|--------:|---------:|----------:|------:|---------:|
| FSDP2+AdamW | 356.0 ms | 698.8 ms | 53.1 ms | 1,107.9 ms | 48.8 GB |
| FSDP2+Muon | 362.6 ms | 705.5 ms | 2,917.2 ms | 3,985.3 ms | 48.8 GB |
| DMuon | 367.0 ms | 666.3 ms | 188.8 ms | 1,222.1 ms | 65.2 GB |

**DMuon vs FSDP2+AdamW:** +10% overhead (1,222 vs 1,108 ms)
**DMuon vs FSDP2+Muon:** 3.3x faster (optimizer: 15.5x)

---

## Llama-3.1-8B (8.03B params)

DDP cannot fit 8B model in single GPU memory.

| Method | Forward | Backward | Optimizer | Total | Peak Mem |
|--------|--------:|---------:|----------:|------:|---------:|
| FSDP2+AdamW | 378.8 ms | 753.4 ms | 55.5 ms | 1,187.7 ms | 48.4 GB |
| FSDP2+Muon | 386.5 ms | 762.2 ms | 3,467.9 ms | 4,616.7 ms | 48.4 GB |
| DMuon | 369.5 ms | 718.9 ms | 260.4 ms | 1,348.7 ms | 68.7 GB |

**DMuon vs FSDP2+AdamW:** +13% overhead (1,349 vs 1,188 ms)
**DMuon vs FSDP2+Muon:** 3.4x faster (optimizer: 13.3x)

---

## Summary

### Total Step Time (ms)

| Model | FSDP2+AdamW | DDP+Muon | FSDP2+Muon | DMuon | vs AdamW |
|-------|----------:|--------:|-----------:|------:|------:|
| Qwen2.5-1.5B | 328 | 625 | 684 | 340 | +4% |
| Llama-3.2-3B | 599 | 1,671 | 1,810 | 660 | +10% |
| Qwen2.5-7B | 1,108 | — | 3,985 | 1,222 | +10% |
| Llama-3.1-8B | 1,188 | — | 4,617 | 1,349 | +13% |

### Optimizer-Only Time (ms)

| Model | AdamW | DDP+Muon | FSDP2+Muon | DMuon | Speedup vs FSDP2+Muon |
|-------|------:|--------:|-----------:|------:|------:|
| Qwen2.5-1.5B | 17.3 | 320.7 | 373.0 | 31.1 | 12.0x |
| Llama-3.2-3B | 26.5 | 1,108.6 | 1,232.2 | 98.8 | 12.5x |
| Qwen2.5-7B | 53.1 | — | 2,917.2 | 188.8 | 15.5x |
| Llama-3.1-8B | 55.5 | — | 3,467.9 | 260.4 | 13.3x |

### Key Observations

1. **DMuon adds 4-13% overhead** vs FSDP2+AdamW. The overhead comes from the Muon optimizer step (Newton-Schulz orthogonalization on proj layers), which is the cost of using a matrix optimizer instead of AdamW.

2. **DMuon is 12-15x faster** (optimizer-only) than naive FSDP2+Muon. Without DMuon, using Muon with FSDP2 requires per-parameter all-gather + redundant NS on every rank, making the optimizer step 20-60x slower than AdamW. DMuon reduces this to 2-5x slower than AdamW by running NS only on the owner rank with 1/8 parameters.

3. **Forward/backward times are similar** across all methods (~same model, same compute). The difference is entirely in the optimizer step.

4. **DMuon uses more memory** than FSDP2 (27-69 GB vs 24-48 GB) because the owner stores full parameters. This is the memory-compute tradeoff of dedicated ownership.

5. **DMuon optimizer scales with model size**: 31ms (1.5B) -> 99ms (3B) -> 189ms (7B) -> 260ms (8B). The owner runs NS on ~1/8 of all proj params, with CuteDSL SYRK accelerating 5/7 symmetric operations.

### Where Does the 12-15x Speedup Come From?

DMuon optimizer speedup over FSDP2+Muon decomposes into two factors:

- **1/8 sharding**: each rank owns ~1/8 of proj params -> ~8x reduction
- **Gram NS + SYRK**: operates on (min(m,n), min(m,n)) Gram space with SYRK kernel -> ~1.6x additional speedup

Combined: ~8 x 1.6 = ~12.8x, consistent with measured 12-15x.

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
