# API 文档

!!! tip "TL;DR"
    DMuon 公开四个功能区：**初始化**（`dedicate_params`、`install_patch`）、
    **优化器**（`Muon`、`NewtonSchulz`、NS 函数与常量）、**状态管理**
    （`no_sync`、`wait_all_reduces`、replicate-broadcast 工具函数、
    `DedicatedCommContext`）以及**检查点**（`get/set_model/optimizer_state_dict`）。
    从 `dedicate_params` + `Muon` 开始；需要精细控制时再使用其余接口。

---

## 初始化

### dedicate_params

在 `fully_shard()` 之前调用一次。将每个 Muon 目标参数分配给单一 owner rank，
并注册逐层的前向/反向 hook。常见自定义点见
[自定义 Hook 边界](../../guides/custom-hook-boundaries.md) 和
[Z2 与 Z3 模式](../../guides/z2-z3-modes.md)。

::: dmuon.dedicate_params

---

### dedicate_params_ddp

DDP 路径下的 dedicated parameter 初始化入口，用于 data-parallel 模型保持
replicated、而不是由 FSDP2 分片的场景。

::: dmuon.dedicate_params_ddp

---

### dedicate_params_ddp_tp

DDP 路径下启用 TP 的 dedicated parameter 初始化入口，适用于每个 replicated
data-parallel group 内部还有 Tensor Parallelism 的场景。

::: dmuon.dedicate_params_ddp_tp

---

### replicate

非专属参数的 DDP-style replication helper。

::: dmuon.replicate

---

### replicate_tp

`dedicate_params_ddp_tp()` 的 TP-aware companion，用于非专属 `DTensor`
参数。

::: dmuon.replicate_tp

---

### install_patch

`import dmuon` 会自动调用此函数。除非在不经过正常 import 路径的情况下
构建 DMuon 环境，否则无需手动调用。

::: dmuon.install_patch

---

## 优化器

### Muon

主优化器类。在同一对象中管理 matrix-routed 专属参数上的 Muon
（Newton-Schulz + 动量）和 base path 上的 AdamW。Base path 可以是普通
FSDP2-managed 参数，也可以是通过 `route_hint_fn` 选择的 DMuon-managed
sharded AdamW 参数。兼容 `torch.optim.lr_scheduler`。

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

查询当前活跃的 NS 内核。返回形如
`"Gram NS · kernel=cute_sm80 (SM80, DMuon internal)"`、
`"Gram NS · kernel=quack (SM90, Tri Dao quack)"` 或
`"Gram NS · kernel=cublas (SM70, universal fallback)"` 的单行摘要。详见
[后端分发](newton-schulz.md#backend-dispatch)。

::: dmuon.get_ns_backend

---

### get_backend_status

NS 内核分发层的完整诊断 dict —— `sm_version`、`auto_choice`，以及各
后端的可用性标志。适合程序化检查和 bug report。

::: dmuon.get_backend_status

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

### prepare_muon_grads

在 backward 之后准备所有 pending 的 Muon 梯度。它不只是等待 reduce；对
TP-sharded 参数，还可能需要在 Muon 运行前触发 TP gather。

::: dmuon.prepare_muon_grads

---

### wait_all_reduces

`prepare_muon_grads()` 的向后兼容 alias。`Muon.step()` 会自动调用；仅当需要
在 backward 和 step 之间手动访问 prepared gradient 时才需要单独调用。

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
event 在下一次前向传播开始时被消费。

::: dmuon.broadcast_all_updates_async

---

### wait_all_replicate_broadcasts

等待所有 group 的异步 replicate broadcast 完成。在正常前向/step 周期
之外需要读取 `_owned_data` 的代码（如自定义检查点或评估逻辑）中调用。

::: dmuon.wait_all_replicate_broadcasts

---

### wait_all_post_step_broadcasts

`wait_all_replicate_broadcasts()` 的兼容 alias。

::: dmuon.wait_all_post_step_broadcasts

---

### clip_grad_norm_

裁剪 DMuon-owned Muon 参数的梯度。

::: dmuon.clip_grad_norm_

---

### register_muon_grad_clip_strategy

为 `clip_grad_norm_()` 注册自定义策略。

::: dmuon.register_muon_grad_clip_strategy

---

### MuonGradClipStats

DMuon 梯度裁剪的返回类型。

::: dmuon.MuonGradClipStats

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
