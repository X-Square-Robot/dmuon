# SYRK Kernel Benchmark — CuteDSL vs cuBLAS

**Hardware:** NVIDIA A800-SXM4-80GB (SM80)
**Kernel:** CuteDSL SYRK `tile_m=128, tile_k=64, num_stages=3` (autotuned)
**Date:** 2026-04-14

## Overview

Gram Newton-Schulz (NS) uses symmetric matrix operations throughout its iteration.
All intermediate matrices (R, Z, Q, RZ) are polynomials of the symmetric Gram matrix R,
so all products between them produce symmetric results. This allows **5 out of 7** operation
types per NS step to use the SYRK lower-triangle + mirror-write optimization (50% tile savings).

**Operations per step:**

| Operation | Type | SYRK? |
|-----------|------|-------|
| R = X @ X^T | True SYRK | Yes |
| Z = c*R^2 + b*R | SYRK + epilogue | Yes |
| Q = Z + a*I | Fused into Z (diag_add) | Yes (fused) |
| Q = Q@Z + a*Q | Symmetric GEMM (B != A) | Yes |
| RZ = R@Z + a*R | Symmetric GEMM (B != A) | Yes |
| R = Z@RZ + a*RZ | Symmetric GEMM (B != A) | Yes |
| Q @ X | Rectangular GEMM | No (cuBLAS) |

---

## Table 1: Per-Operation SYRK Speedup

CuteDSL SYRK (tile 128x64, 3 stages) vs cuBLAS for the initial SYRK (`R = X @ X^T`)
and the Z computation (`Z = c*R^2 + b*R`). "NS shape" is `(min(m,n), max(m,n))` after transpose.

| Model | Projection | Original Shape | NS Shape | R=X@X^T cuBLAS | R=X@X^T SYRK | Speedup | Z cuBLAS | Z SYRK | Z Speedup |
|-------|-----------|----------------|----------|----------------|--------------|---------|----------|--------|-----------|
| Llama-1B | q/o_proj | (2048, 2048) | (2048, 2048) | 153 us | 114 us | **1.34x** | 159 us | 122 us | **1.30x** |
| Llama-1B | gate/up/down | (2048, 8192) | (2048, 8192) | 413 us | 323 us | **1.28x** | 159 us | 122 us | **1.30x** |
| Llama-1B | k/v_proj | (2048, 512) | (512, 2048) | 34 us | 69 us | 0.50x | 32 us | 49 us | 0.65x |
| Llama-3B | q/o_proj | (3072, 3072) | (3072, 3072) | 256 us | 179 us | **1.43x** | 278 us | 188 us | **1.47x** |
| Llama-3B | gate/up/down | (3072, 8192) | (3072, 8192) | 638 us | 405 us | **1.58x** | 278 us | 188 us | **1.47x** |
| Llama-3B | k/v_proj | (3072, 1024) | (1024, 3072) | 60 us | 73 us | 0.83x | 42 us | 52 us | 0.81x |
| Llama-8B | q/o_proj | (4096, 4096) | (4096, 4096) | 548 us | 365 us | **1.50x** | 601 us | 381 us | **1.58x** |
| Llama-8B | gate/up/down | (4096, 14336) | (4096, 14336) | 1856 us | 1100 us | **1.69x** | 601 us | 381 us | **1.58x** |
| Llama-8B | k/v_proj | (4096, 1024) | (1024, 4096) | 75 us | 91 us | 0.82x | 42 us | 52 us | 0.81x |
| Qwen-3B | q/o_proj | (2048, 2048) | (2048, 2048) | 139 us | 103 us | **1.35x** | 159 us | 122 us | **1.30x** |
| Qwen-3B | gate/up/down | (2048, 11008) | (2048, 11008) | 399 us | 336 us | **1.19x** | 159 us | 122 us | **1.30x** |
| Qwen-7B | q/o_proj | (3584, 3584) | (3584, 3584) | 386 us | 259 us | **1.49x** | 420 us | 275 us | **1.53x** |
| Qwen-7B | gate/up/down | (3584, 18944) | (3584, 18944) | 1939 us | 1148 us | **1.69x** | 420 us | 275 us | **1.53x** |
| Qwen-14B | q/o_proj | (5120, 5120) | (5120, 5120) | 1089 us | 689 us | **1.58x** | 1164 us | 706 us | **1.65x** |
| Qwen-14B | gate/up/down | (5120, 13824) | (5120, 13824) | 2853 us | 1747 us | **1.63x** | 1164 us | 706 us | **1.65x** |
| Mistral-7B | q/o_proj | (4096, 4096) | (4096, 4096) | 571 us | 372 us | **1.54x** | 601 us | 381 us | **1.58x** |
| Mistral-7B | gate/up/down | (4096, 14336) | (4096, 14336) | 1863 us | 1118 us | **1.67x** | 601 us | 381 us | **1.58x** |
| DeepSeek-V3 | q_proj | (7168, 7168) | (7168, 7168) | 2856 us | 1787 us | **1.60x** | 2981 us | 1809 us | **1.65x** |
| DeepSeek-V3 | gate/down | (7168, 18432) | (7168, 18432) | 7171 us | 4423 us | **1.62x** | 2981 us | 1809 us | **1.65x** |

**Summary:** For m >= 2048, SYRK provides **1.19x - 1.69x** per-operation speedup. For m <= 1024 (GQA k/v_proj), cuBLAS is faster and autotune falls back automatically.

---

## Table 2: Gram NS End-to-End Speedup

Full Gram NS iteration (5 steps, POLAR_EXPRESS coefficients, restart at step 2).
SYRK accelerates 5/7 operation types; only rectangular Q@X uses cuBLAS.

| NS Shape (m, k) | Models Using This Shape | cuBLAS NS | SYRK NS | Speedup |
|-----------------|------------------------|-----------|---------|---------|
| (2048, 2048) | Llama-1B q/o, Qwen-3B q/o | 1647 us | 1867 us | 0.88x |
| (2048, 8192) | Llama-1B gate/up/down | 2787 us | 2531 us | **1.10x** |
| (2048, 11008) | Qwen-3B gate/up/down | 3357 us | 3011 us | **1.11x** |
| (3072, 3072) | Llama-3B q/o | 5220 us | 3476 us | **1.50x** |
| (3072, 8192) | Llama-3B gate/up/down | 7080 us | 5014 us | **1.41x** |
| (3584, 3584) | Qwen-7B q/o | 7660 us | 5280 us | **1.45x** |
| (3584, 18944) | Qwen-7B gate/up/down | 15028 us | 11216 us | **1.34x** |
| (4096, 4096) | Llama-8B q/o, Mistral-7B q/o | 11138 us | 7453 us | **1.49x** |
| (4096, 14336) | Llama-8B gate/up/down, Mistral-7B | 17431 us | 12730 us | **1.37x** |
| (5120, 5120) | Qwen-14B q/o | 21022 us | 14127 us | **1.49x** |
| (5120, 13824) | Qwen-14B gate/up/down | 28949 us | 20508 us | **1.41x** |
| (7168, 7168) | DeepSeek-V3 q | 56049 us | 36723 us | **1.53x** |
| (7168, 18432) | DeepSeek-V3 gate/down | 75304 us | 52255 us | **1.44x** |

**Summary by model size:**

| Model Size | Square Shape Speedup | Rect Shape Speedup |
|-----------|---------------------|-------------------|
| 1B (m=2048) | 0.88x (cuBLAS wins) | 1.10x |
| 3B (m=3072) | **1.50x** | **1.41x** |
| 7B (m=3584-4096) | **1.45-1.49x** | **1.34-1.37x** |
| 14B (m=5120) | **1.49x** | **1.41x** |
| DeepSeek-V3 (m=7168) | **1.53x** | **1.44x** |

---

## Autotune Behavior

The SYRK autotune selects the best tile configuration per (M, K, dtype) shape.
For shapes where cuBLAS is faster, it automatically falls back.

| M range | Best Config | E2E Speedup | Notes |
|---------|-------------|-------------|-------|
| m <= 1024 | cuBLAS fallback | ~1.0x | Kernel launch overhead dominates |
| m = 2048 | (128, 64, 3) | 0.88-1.11x | Marginal; depends on K |
| m = 3072 | (128, 64, 3) | **1.41-1.50x** | Consistent gains |
| m = 4096 | (128, 64, 3) | **1.37-1.49x** | Consistent gains |
| m = 5120+ | (128, 64, 3) | **1.41-1.53x** | Best gains at large M |

---

## How to Reproduce

```bash
# Per-operation + E2E benchmark across all LLM shapes
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_syrk.py summary

# Detailed per-sub-operation breakdown
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_syrk.py detail

# Both summary + detail
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_syrk.py
```
