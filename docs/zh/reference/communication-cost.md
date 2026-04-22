# 通信成本分析

!!! tip "TL;DR"
    DMuon 在 **DMuon-Z2** 模式下达到 PyTorch-DP 每步通信的理论下界——
    `2(N-1)/N · P_M` 字节，等同于一次 ring all-reduce。在 **DMuon-Z3**
    模式（默认）下，额外消耗 `(N-1)/N · P_M` 字节，与 ZeRO-3 惯例的
    显存-通信权衡一致。两种模式均消除了朴素 FSDP2+Muon 所需的优化步
    all-gather，并将 NS 计算从 R 份副本降至 1 份。

---

## 符号说明

| 符号 | 含义 |
|---|---|
| N | 分片组大小（1D 时的 DP 并行度；HSDP 时的 shard 维大小） |
| R | 复制组大小（1D 时为 1；HSDP 时的 replicate 维大小） |
| P_M | Muon 目标（专属）参数的总元素数 |
| P_p | 单个参数的元素数 |
| Ring all-reduce 代价 | `2(N-1)/N · P`（reduce-scatter + all-gather） |
| Broadcast / reduce 代价 | `(N-1)/N · P`（单方向） |

所有字节计数以每参数元素为单位；乘以 `sizeof(dtype)` 即得实际传输字节数。

---

## 四个定理——DP 系列全覆盖

### 定理 1：DDP

| | 朴素 Muon（DDP） | DMuon（DDP） |
|---|---|---|
| 反向 | `2(N-1)/N · P_M` all-reduce | `(N-1)/N · P_M` reduce 到 owner |
| 前向广播 | — | `(N-1)/N · P_M` 从 owner 广播 |
| **合计** | `2(N-1)/N · P_M` | `2(N-1)/N · P_M` |
| NS 计算 | N 份副本 | **1 份** |

**结论：** 通信字节数相同；消除 N-1 次冗余 NS 计算。对任意 N > 1 均有收益。

---

### 定理 2a：DMuon-Z2（FSDP，reshard_after_forward=False）

朴素 FSDP2 + Muon 需要三次通信：

1. 前向 all-gather：`(N-1)/N · P_M`
2. 反向 reduce-scatter：`(N-1)/N · P_M`
3. 优化步 all-gather（为 NS 重建完整梯度）：`(N-1)/N · P_M`

**朴素合计：** `3(N-1)/N · P_M`

DMuon-Z2 以两次通信替代以上三次：

1. 前向从 owner 广播：`(N-1)/N · P_M`
2. 反向 reduce 到 owner：`(N-1)/N · P_M`

**DMuon-Z2 合计：** `2(N-1)/N · P_M`

这等于 N 个 rank 交换 P_M 元素的 ring all-reduce 理论下界。DMuon-Z2
达到理论最优通信量。

**显存代价：** 每个 rank 常驻存储 P_M 元素（owner 存完整参数；非 owner
在前向+反向期间保留广播副本）。

---

### 定理 2b：DMuon-Z3（FSDP，reshard_after_forward=True——默认）

朴素 FSDP2 + Muon 需要四次通信：

1. 前向 all-gather：`(N-1)/N · P_M`
2. 反向 all-gather（重物化用于梯度计算）：`(N-1)/N · P_M`
3. 反向 reduce-scatter：`(N-1)/N · P_M`
4. 优化步 all-gather：`(N-1)/N · P_M`

**朴素合计：** `4(N-1)/N · P_M`

DMuon-Z3 以三次通信替代以上四次：

1. 前向从 owner 广播：`(N-1)/N · P_M`
2. 反向重广播（前向后参数已重分片）：`(N-1)/N · P_M`
3. 反向 reduce 到 owner：`(N-1)/N · P_M`

**DMuon-Z3 合计：** `3(N-1)/N · P_M`

相比朴素 FSDP2+Muon 节省一次完整 all-gather，同时消除冗余 NS 计算。

**显存代价：** 非 owner rank 在每次前向后释放广播缓冲区；只有 owner
常驻 P_M 元素。逐层打包缓冲区为临时分配。

---

### 定理 3：HSDP（2D mesh，分片大小 N，复制大小 R）

HSDP 引入复制维度。DMuon 的两阶段协议：

**反向：** 在分片组内 reduce 梯度（`(N-1)/N · P_M`），再在复制组上做
AVG reduce（`(R-1)/R · P_M`）。总除数 = N·R，等效于在全局做一次 all-reduce。

**优化步后：** 将 `_owned_data` 从全局 owner 异步广播到 R-1 个复制对等节点，
该广播隐藏在下一次前向传播中。

**每步总字节数：**

| 阶段 | 字节 |
|---|---|
| 分片维 reduce（反向） | `(N-1)/N · P_M` |
| 复制维 reduce（反向） | `(R-1)/R · P_M` |
| 复制广播（异步，优化步后/前向前） | `(R-1)/R · P_M` |
| 分片广播（前向） | `(N-1)/N · P_M` |

这与原生 HSDP（AG + RS + AR）的通信模式相符，同时将 NS 计算从 N·R 份
副本降至 1 份。

---

## 理论下界

N 个 rank 交换 P 元素的 ring all-reduce 理论下界为 `2(N-1)/N · P`。
这是紧致的——任何要求每个 rank 在步骤结束时持有更新参数的算法，至少需要
传输这么多元素。

**DMuon-Z2** 达到 `2(N-1)/N · P_M`，命中 Muon 目标参数的理论下界。

**DMuon-Z3** 使用 `3(N-1)/N · P_M`，超过下界 `(N-1)/N` 项。这与 FSDP
ZeRO-3 对非优化器参数接受的额外通信相同：额外的通信换取了每次前向后
重分片参数所带来的峰值显存降低。

---

## 显存代价

| 模式 | Muon 目标参数在每个 rank 的显存 |
|---|---|
| DMuon-Z2（`reshard_after_forward=False`） | 每 rank P_M（完整副本，常驻） |
| DMuon-Z3（`reshard_after_forward=True`，默认） | owner P_M；非 owner 前向期间一个逐层打包缓冲区（临时） |

对于显存紧张的大模型，推荐 Z3。若以通信效率优先且显存充裕，Z2 消除了
一个广播方向。

通过 `dedicate_params(..., reshard_after_forward=False)` 选择 Z2。

---

## 与 Canzona 的关系

Canzona（Wang et al., arXiv:2602.06079）将专属所有权原语扩展到
Megatron 张量并行 + ZeRO-1，引入了 Micro-Group Scheduling 和 All-to-All
通信。DMuon 和 Canzona 是对同一原语的平行扩展，该原语最早由
Distributed Shampoo（Shi et al., 2023）和 ZeRO-1（Rajbhandari et al., 2020）
提出。

核心区别在于目标技术栈：**Canzona 面向 Megatron-LM**（TP+PP+ZeRO1 组合）；
**DMuon 面向 PyTorch DDP/FSDP2/HSDP**，无 Megatron 依赖。目前两者之间
没有直接的 head-to-head 基准对比。在讨论专属所有权原语时，可同时引用
两者。

---

## 复现这些数字

HSDP 通信的逐比特正确性在 `tests/distributed/test_hsdp_correctness.py`
中验证：在 4-GPU（G=2, R=2）测试环境上，DMuon-HSDP 在 10 步训练中与
shard-only DMuon 逐比特匹配。

逐字节 NCCL 追踪验证（Phase D）已规划；详见路线图中的 `[TBD Phase D]`。

---

## 参见

- [HSDP 指南](../../guides/hsdp.md)
- [Z2 与 Z3 模式](../../guides/z2-z3-modes.md)
- [设计/架构](../../design/architecture.md)
