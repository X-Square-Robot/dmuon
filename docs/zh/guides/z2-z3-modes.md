# Z2 与 Z3 模式

!!! tip "TL;DR"
    DMuon-Z3（默认）在每次 forward 后释放 packed buffer，显存更低，与 FSDP2 ZeRO-3
    的习惯一致。DMuon-Z2 让 buffer 在 forward 和 backward 之间保持常驻，少一次
    broadcast，通信更优。3B 参数以上的模型一般用 Z3 就好。

---

## 两种 packed buffer 生命周期

同一个 owner 下的所有 Muon-target 参数会被打包进一个连续 buffer，用于 shard 维度的
broadcast。两种模式下这个 buffer 的生命周期不同：

| | **DMuon-Z3**（默认）| **DMuon-Z2** |
|---|---|---|
| `reshard_after_forward` | `True` | `False` |
| forward 结束后 | buffer storage 释放，安装占位张量 | buffer 保持常驻 |
| backward 开始前 | owner 从 `_owned_data` 重新 broadcast | backward 直接复用常驻 buffer |
| 每步 shard broadcast 次数 | 2（fwd 一次，bwd 一次）| 1（仅 fwd）|
| 每 shard rank 的稳态显存 | 单层 packed buffer transient | 全量 `P_M` 常驻 |

两种模式结果 bit-identical：到达 owner 的梯度值完全一样。区别只在通信次数和显存占用。

---

## 每步通信字节数

以下数据统计 shard 进程组（N 个 rank）上的通信量。`P_M` 为所有 Muon-target 参数的总元素数。

| 配置 | 通信字节/步 |
|---|---|
| 无 DMuon、Naive Muon on FSDP2-Z2 | `3(N-1)/N · P_M` |
| **DMuon-Z2** | `2(N-1)/N · P_M` |
| 无 DMuon、Naive Muon on FSDP2-Z3 | `4(N-1)/N · P_M` |
| **DMuon-Z3** | `3(N-1)/N · P_M` |

相比 naive 基准，DMuon 在两种模式下都省掉了一次梯度 all-gather（owner 在 backward reduce
后就已经有完整梯度，optimizer step 不需要再 all-gather）。DMuon-Z2 在此基础上再省掉
backward re-broadcast，达到通信最优的 `2(N-1)/N · P_M`。

两种模式还消除了 naive FSDP2+Muon 中 `(N-1)` 次冗余的 Newton-Schulz 计算（naive 方案下
每个 rank 都对完整梯度跑 NS；dedicated ownership 下只有 owner 跑一次）。

---

## 显存开销

**DMuon-Z3**：训练过程中，任意时刻只有一个层的 packed buffer 在 broadcast stream 上分配。
Muon-target 参数的稳态显存：

```
显存 ≈ max_layer_P_M（transient，每次只有一层）
```

其中 `max_layer_P_M` 是该 rank 上所有 owned group 中单层最大的 packed buffer。

**DMuon-Z2**：从第一次 forward 开始，所有 packed buffer 同时常驻，直到 optimizer step 结束。
稳态显存：

```
显存 ≈ P_M / N（所有 owned packed buffer，按 owner shard 列）
```

以一个 7B 模型为例，若 50% 参数属于 Muon-target projection，则 `P_M ≈ 3.5B` 参数
（bf16 约 7 GB）。8 路 shard 下，DMuon-Z2 每个 rank 多占约 875 MB——不算小，但对 80 GB 显
卡来说可以接受。

---

## 决策树

这些判断只是经验起点，不是硬规则。真实选择还要看 Muon-target 参数占比、GPU 显存余量、
节点间网络、activation 显存、梯度累积设置，以及实际 profiling 的吞吐结果。

- **模型 > 10B 参数？** → 优先从 DMuon-Z3（默认）开始。多数情况下，额外的 forward
  broadcast 相比节省的显存更容易接受。
- **模型 < 3B 且 8+ GPU？** → DMuon-Z2 可以考虑。通信占主导时，省掉 backward broadcast
  有明显收益。
- **DMuon-Z3 下 OOM？** → 先不要急着切 Z2；packed buffer 在 Z3 下是 transient 的。优先排查
  activation 显存和梯度累积 buffer 大小。
- **和 `fully_shard(..., reshard_after_forward=X)` 配合使用？** → 让 `dedicate_params`
  的 `reshard_after_forward` 与之保持一致。对称配置让 Muon-target 和非 Muon 参数的显存模型
  保持统一，易于推理。
- **使用 HSDP 多机？** → 选择同样适用。上面的通信量统计针对 shard 维；replicate 维的
  post-step broadcast 是独立的，不受 Z2/Z3 影响。

---

## 非对称配置

DMuon-Z2 + FSDP2-Z3（或反过来）合法，偶尔是最优选择：

- **DMuon-Z2 + FSDP2-Z3**：Muon-target 参数数量少但单个很大（如某个巨型 projection），
  非 Muon 参数（embedding、norm）数量多且单个小。大的 Muon buffer 常驻省一次 broadcast；
  FSDP2-Z3 避免大量小参数撑爆显存。
- **DMuon-Z3 + FSDP2-Z2**：不太常见。非 Muon 参数主导显存时合理，且 Muon-target 参数
  小到 backward re-broadcast 几乎无成本。

非对称配置增加心智负担。推荐从对称配置出发，有 profiling 数据支撑再做调整。

---

## 与 ZeRO-2 / ZeRO-3 的关系

DMuon 的 Z2/Z3 命名沿用 PyTorch FSDP2 `reshard_after_forward` 的习惯：

- **ZeRO-2 风格**（`reshard_after_forward=False`）：参数在 forward 和 backward 之间保持
  unsharded 状态，也就是已经收集好的完整形态。
- **ZeRO-3 风格**（`reshard_after_forward=True`）：参数在 forward 结束后 reshard 并释放，
  backward 需要时再重新收集。

DMuon 管理的 Muon-target 参数不再走 FSDP2 的 sharded-state 路径，也就不是由 FSDP2
按普通参数的方式分片。每个 Muon-target 参数会有一个单一 owner，owner 持有权威的
`_owned_data`；训练时需要完整参数时，再从 owner broadcast 出 packed buffer。DMuon 的
Z2/Z3 flag 控制的正是**这个 packed buffer 的生命周期**，语义上对应 FSDP2 对普通参数
storage 的处理方式。

因此，DMuon-Z2/Z3 和 FSDP2-Z2/Z3 是两个独立旋钮：前者控制 Muon-target packed buffer，
后者控制普通 FSDP2 参数。两者可以独立设置，只是非对称配置会增加分析成本，通常需要
profiling 数据支撑。

---

## 相关文档

- [HSDP 训练](hsdp.md) —— HSDP 多机场景下的 Z2/Z3 选择
- [自定义 Hook 边界](custom-hook-boundaries.md) —— hook 粒度影响同时存在的 packed buffer 数量
- [训练流程](training.md) —— 完整单机工作流
- [通信成本分析](../reference/communication-cost.md) —— 详细字节开销推导
