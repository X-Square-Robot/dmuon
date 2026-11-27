# 昇腾 950DT 优化器基准测试

在昇腾 950DT 上，安装 TileLang 昇腾模块后，`dmuon.Muon` 默认使用 TileLang；未安装时自动回退到 PyTorch。使用 `dmuon.Muon(..., compute_backend="torch")` 可强制使用 PyTorch，使用 `dmuon.Muon(..., compute_backend="tilelang")` 则要求 TileLang 可用，缺失时会给出可操作的错误。独立的 Newton–Schulz 和分桶裁剪 API 也接受同名可选参数 `compute_backend`。显式 `kernel="cublas"` 和 `deterministic=True` 仅支持 CUDA，在昇腾上会报错；CUDA 沿用已有的后端选择方式。[安装指南](../getting-started/installation.md)

## 完整优化器更新

下表是在单张昇腾 950DT 上执行一次本地归属参数更新的耗时，单位为毫秒。测试使用 CANN 9.3.0-beta.1、PyTorch 2.10.0+cpu 和 torch-npu 2.10.0.post4。编译器由未经修改的上游 TileLang v0.1.15 源码构建。本次测试启用了 Newton–Schulz 图回放（`DMUON_TILELANG_GRAPH=1`），元素数不超过 4,194,304 的输入可使用图回放，更大的输入直接启动内核。[上游编译器及构建说明](https://github.com/tile-ai/tilelang/blob/v0.1.15/tilelang/ascend/README.md)

| 训练任务 | 本地归属参数 | Torch（ms） | TileLang（ms） | 加速比 |
|---|---|---:|---:|---:|
| WALL-4B LIBERO | 32 × 2560，线性注意力投影 | 1.427 | 1.178 | 1.21× |
| WALL-4B LIBERO | 1280 × 2560，动作 MLP 下投影 | 1.729 | 1.468 | 1.18× |
| WALL-4B LIBERO | 2560 × 9216，文本 MLP 下投影 | 4.060 | 3.569 | 1.14× |
| WALL-4B LIBERO | `layers.1`，4 个 Muon + 10 个 AdamW 张量 | 5.191 | 3.974 | 1.31× |
| WALL-4B LIBERO | `layers.1` + 含 158,770,560 个元素的 AdamW 词嵌入分片 | 12.647 | 6.592 | 1.92× |
| WALL-2B 顺丰 | 4096 × 1024，视觉 MLP 第一层投影 | 1.604 | 1.288 | 1.25× |
| WALL-2B 顺丰 | 2048 × 2048，线性注意力输出投影 | 2.031 | 1.896 | 1.07× |
| WALL-2B 顺丰 | 2048 × 6144，文本 MLP 下投影 | 2.416 | 2.227 | 1.08× |
| WALL-2B 顺丰 | `layers.7`，9 个 Muon + 10 个 AdamW 张量 | 6.105 | 4.627 | 1.32× |
| WALL-2B 顺丰 | `layers.7` + 含 131,213,824 个元素的 AdamW 词嵌入分片 | 12.104 | 6.761 | 1.79× |

张量形状和优化器分组取自四卡训练记录。每个单矩阵用例将一个完整的 Muon 权重与一个 AdamW 向量配对。按层统计的用例保留该层分配给 rank 0 的全部参数及其实际形状和数量。词嵌入用例额外加入 rank 0 的本地扁平分片。基准使用种子 42 为两个后端生成相同的初始权重和逐步变化的梯度。

两个后端均使用 BF16 梯度、FP32 主权重和 FP32 优化器状态。计时从反向传播和梯度归约完成后开始，包含阈值为 1.0 的分组梯度裁剪和 `Muon.step()`，到设备同步完成为止。更新使用 0.95 的 Nesterov 动量、五步 Gram Newton–Schulz、0.01 的解耦权重衰减以及 AdamW。每个后端预热 3 步、计时 12 步，逐步交替先后顺序。加速比取 Torch 耗时中位数除以 TileLang 耗时中位数。

## 精度

十个用例均通过检查。每次更新后，基准都会计算相对于 PyTorch 的相对 L2 误差，并检查 AdamW 步数。额外执行的一次未计时更新会记录实际调用的 TileLang 操作。

| 对比项 | 门槛 | 最大实测相对 L2 |
|---|---:|---:|
| Muon 权重更新量 | 8e-2 | 2.146e-2 |
| Muon 主权重 | 3e-2 | 8.301e-3 |
| AdamW 权重更新量 | 1e-4 | 8.442e-5 |
| AdamW 主权重 | 1e-4 | 4.218e-7 |
| Muon 动量和 AdamW 一阶、二阶矩 | 1e-4 | 1.354e-7 |

## 复现

在仓库根目录用一张空闲 NPU 运行以下命令。表中每一行均在独立进程内对照两个后端。使用 `--help` 查看全部用例。

```bash
export DMUON_TILELANG_GRAPH=1
python benchmarks/bench_ascend_step.py --case libero-4b-owner-layer --output layer.json
python benchmarks/bench_ascend_step.py --case libero-4b-owner-layer \
  --include-embedding-shard --output layer-shard.json
```
