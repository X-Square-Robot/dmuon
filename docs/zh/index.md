# DMuon

**基于 PyTorch FSDP2 的 Muon 专属所有权框架。**

*一个所有者。一次 Newton-Schulz。零优化器通信。*

---

## 问题

矩阵优化器（如 [Muon](https://arxiv.org/abs/2502.16982)）需要**完整的梯度矩阵**来执行 Newton-Schulz 正交化。但 FSDP2 会把一切都分片——每个 rank 只持有 1/R 的梯度。

朴素方案代价高昂：

1. **All-gather** 完整梯度到每个 rank — O(mn) 额外通信
2. **每个 rank** 重复运行相同的 Newton-Schulz — R 倍冗余计算

对于 7B 模型在 8 卡上，这会带来 **3-4 倍**的额外开销。

## 解决方案

DMuon 为每个矩阵参数指定一个**所有者 rank**。只有所有者存储完整参数并执行 Newton-Schulz——无需 all-gather，无冗余计算。

```
标准 FSDP2 + Muon                DMuon
========================        ========================
all-gather 完整梯度               reduce 梯度到所有者
  O(mn) 通信                       O(mn/R) 通信
每个 rank 运行 NS                 仅所有者运行 NS
  R 倍冗余                          1 次
```

| | 标准 FSDP2 + Muon | DMuon |
|---|---|---|
| 优化器通信 | all-gather 完整梯度 | **零** |
| NS 计算 | R 次（每个 rank） | **1 次**（仅所有者） |
| 相比 AdamW 总开销 | 200-400% | **4-13%** |

## 快速预览

```python
import dmuon  # 自动 patch FSDP2

# 1. 标记参数为专属所有权
dmuon.dedicate_params(model, dp_mesh, predicate=lambda n, p: "proj" in n)

# 2. 照常使用 FSDP2——专属参数自动处理
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)

# 3. 使用 Muon 训练
optimizer = dmuon.Muon(model, lr=0.02, momentum=0.95)
```

就这么简单。前向广播、反向 reduce 和所有者独占 Newton-Schulz 全部自动完成。

## 特性

- **零优化器通信** — 所有者在 reduce 后已持有完整梯度
- **1/R NS 计算** — 仅所有者运行 Newton-Schulz，不是每个 rank
- **FSDP2 原生** — 与 `fully_shard()` 并行工作，无需修改 FSDP2 内部
- **TP 兼容** — Gram Newton-Schulz 配合 TP SYRK 分解，O(d_model^2) TP 通信
- **检查点兼容** — 标准 state dict，兼容 HuggingFace 和单卡加载
- **梯度累积** — `no_sync()` 上下文管理器，与 FSDP2 相同模式

## 性能测试

**8 x A800-SXM4-80GB, bf16, seq=2048**

| 模型 | FSDP2+AdamW | DMuon | 额外开销 |
|:------|----------:|------:|------:|
| Qwen2.5-1.5B | 328 ms | 340 ms | +4% |
| Llama-3.2-3B | 599 ms | 660 ms | +10% |
| Qwen2.5-7B | 1,108 ms | 1,222 ms | +10% |
| Llama-3.1-8B | 1,188 ms | 1,349 ms | +13% |

优化器步骤比朴素 FSDP2+Muon **快 12-15 倍**。

## 开始使用

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **安装**

    安装 DMuon 并验证环境。

    [:octicons-arrow-right-24: 安装指南](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **快速开始**

    5 分钟内运行第一次分布式训练。

    [:octicons-arrow-right-24: 快速开始](getting-started/quickstart.md)

-   :material-head-lightbulb:{ .lg .middle } **核心概念**

    理解专属所有权的工作原理及其与 FSDP2 的组合。

    [:octicons-arrow-right-24: 核心概念](getting-started/concepts.md)

-   :material-book-open-variant:{ .lg .middle } **API 文档**

    完整的函数签名和参数说明。

    [:octicons-arrow-right-24: API 文档](reference/api.md)

</div>
