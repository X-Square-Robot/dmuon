# API 文档

!!! tip "TL;DR"
    DMuon 公开四个功能区：**初始化**（`dedicate_params`、`install_patch`）、
    **优化器**（`Muon`、`NewtonSchulz`、NS 函数与常量）、**状态管理**
    （`no_sync`、`wait_all_reduces`、replicate-broadcast 工具函数、
    `DedicatedCommContext`）以及**检查点**（`get/set_model/optimizer_state_dict`）。
    从 `dedicate_params` + `Muon` 开始；需要精细控制时再使用其余接口。

---

## 模块常量

`dmuon.group` 中的两个模块级变量可在不修改优化器构造函数的情况下调整
异步→同步降级协议。在训练开始前导入并修改：

```python
import dmuon.group as g

g.REPLICATE_WAIT_THRESHOLD_US = 250   # 默认：100 μs；快速 IB 网络可适当调高
g.REPLICATE_FALLBACK_CONSECUTIVE_STEPS = 5  # 默认：3；触发降级所需的连续慢步数
```

| 名称 | 默认值 | 说明 |
|---|---|---|
| `REPLICATE_WAIT_THRESHOLD_US` | `100.0` | 单层 replicate broadcast 等待时长超过此阈值则记为"慢步"。 |
| `REPLICATE_FALLBACK_CONSECUTIVE_STEPS` | `3` | 连续慢步数达到此值后，该 group 永久切换为同步广播。 |

重置已降级的 group：`dmuon.reset_replicate_fallback(model)`。

---

## 初始化

### dedicate_params

在 `fully_shard()` 之前调用一次。将每个 Muon 目标参数分配给单一 owner rank，
并注册逐层的前向/反向 hook。常见自定义点见
[自定义 Hook 边界](../../guides/custom-hook-boundaries.md) 和
[Z2 与 Z3 模式](../../guides/z2-z3-modes.md)。

::: dmuon.dedicate_params

---

### install_patch

`import dmuon` 会自动调用此函数。除非在不经过正常 import 路径的情况下
构建 DMuon 环境，否则无需手动调用。

::: dmuon.install_patch

---

## 优化器

### Muon

主优化器类。在同一对象中管理专属参数上的 Muon（Newton-Schulz + 动量）
和 FSDP2 托管的对称参数上的 AdamW，兼容 `torch.optim.lr_scheduler`。

::: dmuon.Muon

---

### NewtonSchulz

可配置的 NS 后端对象。传入 `Muon(ns_backend=...)` 以选择算法变体、覆盖系数
或启用确定性模式。完整对比见 [Newton-Schulz 变体](newton-schulz.md)。

::: dmuon.NewtonSchulz

---

### newton_schulz

独立 NS 函数，默认路由至 Gram 空间后端。在优化器循环外需要 NS 时直接使用，
例如自定义训练循环或实验代码。

::: dmuon.newton_schulz

---

### gram_newton_schulz

具备 TP 感知的 Gram NS（带 SYRK 分解）。`Muon` 内部为 Tensor-Parallel
参数调用此函数；此处暴露供构建自定义 TP 优化器的用户使用。
见 [张量并行](../../guides/tp-support.md)。

::: dmuon.gram_newton_schulz

---

### get_ns_backend

查询当前活跃的硬件后端。在不支持 SM80+ 的机器上调试时有用。

::: dmuon.get_ns_backend

---

### YOU_COEFFICIENTS

来自 [@YouJiacheng](https://x.com/YouJiacheng/status/1905861218138804534)
的 5 步逐迭代 `(a, b, c)` 系数。传入 `NewtonSchulz(coefficients=...)` 或
直接传入 NS 函数的 `coefficients` 参数。

::: dmuon.YOU_COEFFICIENTS

---

### POLAR_EXPRESS_COEFFICIENTS

默认 5 步系数，来自 Polar Express 论文（arXiv:2505.16932），应用了 1.05
安全因子。不传 `coefficients` 参数时默认使用。

::: dmuon.POLAR_EXPRESS_COEFFICIENTS

---

## 工具函数 — DMuon 状态管理

### no_sync

梯度累积的上下文管理器。在上下文内抑制 DMuon reduce 和 FSDP2 的
reduce-scatter；最后一个 micro-batch 在上下文外调用 backward 以触发
合并 reduce。见 [梯度累积](../../guides/grad-accumulation.md)。

::: dmuon.no_sync

---

### wait_all_reduces

等待所有异步梯度 reduce 完成。`Muon.step()` 会自动调用；
仅当需要在 backward 和 step 之间手动访问 `_reduced_grad` 时才需要
单独调用。

::: dmuon.wait_all_reduces

---

### broadcast_all_updates

同步的后置 replicate broadcast（HSDP Phase B 路径）。将更新后的
`_owned_data` 从全局 owner 广播到每个 replicate 对等节点。在 1D
shard-only 模式下为空操作。除非调试，优先使用异步变体。

::: dmuon.broadcast_all_updates

---

### broadcast_all_updates_async

异步的后置 replicate broadcast（`Muon` 默认值）。立即返回；每一层的
event 在下一次前向传播开始时被消费。已触发降级协议的 group 在此调用内
同步完成。见 [性能分析与 Fallback](../../guides/profiling-and-fallback.md)。

::: dmuon.broadcast_all_updates_async

---

### wait_all_replicate_broadcasts

等待所有 group 的异步 replicate broadcast 完成。在正常前向/step 周期
之外需要读取 `_owned_data` 的代码（如自定义检查点或评估逻辑）中调用。

::: dmuon.wait_all_replicate_broadcasts

---

### reset_replicate_fallback

重新启用所有因降级协议而切换为同步的 group 的异步广播。修复慢 IB
状况后可从训练循环中安全调用。

::: dmuon.reset_replicate_fallback

---

### replicate_profile_report

向标准输出打印逐 group 的等待时间汇总（仅 rank 0）。需要在进程启动前
设置 `DMUON_REPLICATE_PROFILE=1`。在训练或分析窗口结束时调用。

::: dmuon.replicate_profile_report

---

### get_dedicated_params

枚举模型中所有 `DedicatedParam` 对象。用于检查 ownership 分配、参数数量
和负载均衡情况。

::: dmuon.get_dedicated_params

---

### get_owned_params

筛选属于指定 rank 坐标的 `DedicatedParam` 对象。接受整数（1D）或
`(shard, replicate)` 元组（HSDP）。

::: dmuon.get_owned_params

---

### get_comm_ctx

获取存储在模型上的 `DedicatedCommContext`。若未调用 `dedicate_params`
则返回 `None`。

::: dmuon.get_comm_ctx

---

### DedicatedCommContext

持有专属 CUDA stream（broadcast、reduce、replicate-broadcast）和
预取顺序状态的共享通信上下文。类比 FSDP2 的 `FSDPCommContext`。
大多数用户无需直接构造。

::: dmuon.DedicatedCommContext

---

## 检查点

以下四个函数均为**集体操作** — 每个 rank 都必须调用。它们在读写张量前
会排空待处理的异步状态。状态字典为标准格式，兼容单 GPU 的
`torch.save`/`torch.load` 和 HuggingFace 检查点。
见 [检查点指南](../../guides/checkpoint.md)。

### get_model_state_dict

::: dmuon.get_model_state_dict

---

### set_model_state_dict

::: dmuon.set_model_state_dict

---

### get_optimizer_state_dict

::: dmuon.get_optimizer_state_dict

---

### set_optimizer_state_dict

::: dmuon.set_optimizer_state_dict

---

## 参见

- [核心概念](../../getting-started/concepts.md)
- [Newton-Schulz 变体](newton-schulz.md)
- [通信成本分析](communication-cost.md)
- [检查点指南](../../guides/checkpoint.md)
