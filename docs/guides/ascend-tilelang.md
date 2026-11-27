# Ascend 950DT optimizer benchmark

On Ascend 950DT, `dmuon.Muon` uses TileLang by default when its Ascend modules are installed. Automatic selection falls back to PyTorch if TileLang is unavailable. Set `dmuon.Muon(..., compute_backend="torch")` to force PyTorch or `dmuon.Muon(..., compute_backend="tilelang")` to require TileLang and receive an actionable error when it is missing. Standalone Newton–Schulz and bucket-clipping APIs accept the same optional `compute_backend` argument. Explicit `kernel="cublas"` and `deterministic=True` are CUDA-only and raise on Ascend; CUDA uses its existing backend selection. [Installation](../getting-started/installation.md)

## Complete optimizer step

The table reports milliseconds per owner-local update on one Ascend 950DT with CANN 9.3.0-beta.1, PyTorch 2.10.0+cpu and torch-npu 2.10.0.post4. The compiler was built from the unmodified upstream TileLang v0.1.15 tag. The measurements enabled Newton–Schulz graph replay (`DMUON_TILELANG_GRAPH=1`), which admits inputs of up to 4,194,304 elements and launches larger inputs directly. [Upstream compiler and build instructions](https://github.com/tile-ai/tilelang/blob/v0.1.15/tilelang/ascend/README.md)

| Training workload | Owner-local parameters | Torch (ms) | TileLang (ms) | Speedup |
|---|---|---:|---:|---:|
| WALL-4B LIBERO | 32 × 2560, linear attention projection | 1.427 | 1.178 | 1.21× |
| WALL-4B LIBERO | 1280 × 2560, action MLP down projection | 1.729 | 1.468 | 1.18× |
| WALL-4B LIBERO | 2560 × 9216, text MLP down projection | 4.060 | 3.569 | 1.14× |
| WALL-4B LIBERO | `layers.1`, 4 Muon + 10 AdamW tensors | 5.191 | 3.974 | 1.31× |
| WALL-4B LIBERO | `layers.1` + 158,770,560-element AdamW embedding shard | 12.647 | 6.592 | 1.92× |
| WALL-2B Shunfeng | 4096 × 1024, vision MLP first projection | 1.604 | 1.288 | 1.25× |
| WALL-2B Shunfeng | 2048 × 2048, linear attention output projection | 2.031 | 1.896 | 1.07× |
| WALL-2B Shunfeng | 2048 × 6144, text MLP down projection | 2.416 | 2.227 | 1.08× |
| WALL-2B Shunfeng | `layers.7`, 9 Muon + 10 AdamW tensors | 6.105 | 4.627 | 1.32× |
| WALL-2B Shunfeng | `layers.7` + 131,213,824-element AdamW embedding shard | 12.104 | 6.761 | 1.79× |

Shapes and routes come from four-rank training captures. Each matrix row pairs one full Muon weight with an AdamW vector. Each layer row includes every parameter owned by rank 0 in that layer, with its original shape and multiplicity. The embedding rows also include rank 0's flat local shard. The benchmark generates identical initial weights and changing gradients for the two backends using seed 42.

Both backends use BF16 gradients and FP32 master weights and optimizer states. Timing starts after backward and gradient reduction, includes per-group clipping at 1.0 and `Muon.step()`, and ends at device synchronization. The step applies Nesterov momentum of 0.95, five Gram Newton–Schulz iterations, decoupled weight decay of 0.01 and AdamW. Each backend receives three warmup steps and twelve timed steps, alternating which backend runs first. Speedup is Torch median time divided by TileLang median time.

## Precision

All ten cases passed. The benchmark checks relative L2 error against PyTorch after each update, including one extra untimed update that records which TileLang operations ran. It also checks that the AdamW step counts match.

| Compared quantity | Limit | Worst observed relative L2 |
|---|---:|---:|
| Muon weight update | 8e-2 | 2.146e-2 |
| Muon master weight | 3e-2 | 8.301e-3 |
| AdamW weight update | 1e-4 | 8.442e-5 |
| AdamW master weight | 1e-4 | 4.218e-7 |
| Muon momentum and AdamW moments | 1e-4 | 1.354e-7 |

## Reproduce

Run from the repository root with one available NPU. Each table row compares both backends in a fresh process. Use `--help` to list all cases.

```bash
export DMUON_TILELANG_GRAPH=1
python benchmarks/bench_ascend_step.py --case libero-4b-owner-layer --output layer.json
python benchmarks/bench_ascend_step.py --case libero-4b-owner-layer \
  --include-embedding-shard --output layer-shard.json
```
