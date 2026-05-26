# 张量并行

DMuon 通过 `DTensor` 原生支持 PyTorch 张量并行（TP）。你按照惯常方式应用
TP——DMuon 会**自动检测** TP 分片的参数，并将其导向 TP gather →
完整矩阵 Newton-Schulz → TP scatter 流水线；在每个 rank 只持有权重分片的
情形下仍然保持 Muon 的数学定义不变。

**关键特性**：TP 路径对用户完全透明。`dedicate_params` **不接受** `tp_mesh`
参数——你传进去的是和 `fully_shard` 一样的 DP 切片，DMuon 通过每个参数的
`DTensor` 结构自动推断 TP 维度。这与 FSDP2 的 TP-oblivious 设计同构。

---

## 工作原理

对每个被 `dedicate_params` 选中的参数：

* **普通 `torch.Tensor`** — DMuon 标准 DP 路径（reduce→owner、broadcast）。
  与非 TP 场景完全一致。
* **仅在 DP 维度上分片的 `DTensor`** — 同上。
* **在非 DP mesh 维度（即 TP 轴）上分片的 `DTensor`** — DMuon 为每个参数在
  TP group 内选定一个 rank 作为 "TP owner"。TP owner 由每个 DP owner
  bucket 内的 LPT 选择，从而把完整矩阵 Newton-Schulz 计算分散到多个 TP
  rank。每个 optimizer step：
  1. 该 TP group 内所有 DP-owner rank 走 `dist.gather` on
     `reduce_stream`，把完整 `(m, n)` 梯度汇聚到 TP owner。
     （gather 借用 DP reduce 的 comm stream，所以和 backward compute 天然
     并行——8-GPU 3D HSDP×TP toy 下测到 **~100% overlap**。）
  2. TP owner 在**完整矩阵**上跑 Newton-Schulz（和非 TP 路径走同一个 NS
     kernel）。
  3. `dist.scatter` on `replicate_broadcast_stream`，把 update 的每一片
     发回各 DP-owner rank。
  4. HSDP 下，标准 replicate broadcast 会把 TP-correct update 扩散到
     replicate peer；2D DP×TP 没有 replicate 轴，下一次 forward 的 shard
     broadcast 直接读取已经更新好的 owner shard。

Sync（`replicate_async=False`）和 async（默认 `replicate_async=True`）两路
的 post-step 通信设计目标是保持相同数值轨迹；async 只改变 scatter 完成的
**时间**，不改变数学结果。下面的分布式 loss matrix 会在已支持 TP 拓扑上
检查这一点。

---

## 配置

调用顺序：**先 TP，再 DMuon，最后 FSDP2。** DMuon 必须在 `fully_shard`
**之前**被调用，这样它的参数才能 opt out 于 FSDP2 的分片契约。

```python
import dmuon
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, parallelize_module,
)

# 2D mesh（dp × tp）— 最常见布局
mesh = init_device_mesh(
    "cuda", (dp_size, tp_size),
    mesh_dim_names=("dp", "tp"),        # 必须传 dim 名称
)

model = MyModel().cuda()

# Step 1 — TP
for layer in model.layers:
    parallelize_module(
        layer.self_attn, mesh["tp"],
        {
            "q_proj": ColwiseParallel(),
            "k_proj": ColwiseParallel(),
            "v_proj": ColwiseParallel(),
            "o_proj": RowwiseParallel(),
        },
    )
    parallelize_module(
        layer.mlp, mesh["tp"],
        {
            "gate_proj": ColwiseParallel(),
            "up_proj":   ColwiseParallel(),
            "down_proj": RowwiseParallel(),
        },
    )

# Step 2 — DMuon（只传 DP 切片，不传 TP 切片）
dmuon.dedicate_params(
    model, mesh["dp"],
    predicate=lambda n, p: "proj" in n and p.ndim == 2,
)

# Step 3 — FSDP2（同样是 DP 切片）
for layer in model.layers:
    fully_shard(layer, mesh=mesh["dp"])
fully_shard(model, mesh=mesh["dp"])

# Optimizer — 默认设置即可支持 TP
optimizer = dmuon.Muon(model, lr=0.02, momentum=0.95, adamw_lr=1e-3)
```

!!! info "为什么是 `mesh["dp"]` 而不是完整 mesh?"
    `dedicate_params` 和 `fully_shard` 都只处理 DP 维度——它们是
    TP-oblivious 的。TP 分片已经通过 `parallelize_module` 应用到参数上，
    DMuon 通过 `DTensor.device_mesh` 看到这个信息。这是 FSDP2 的标准约定。

### 3D mesh：HSDP × TP

多机训练加上 replicate 轴。DMuon 原生支持三轴 mesh：

```python
mesh3d = init_device_mesh(
    "cuda", (R, G, T),
    mesh_dim_names=("replicate", "shard", "tp"),
)

# Step 1 — TP
parallelize_module(model, mesh3d["tp"], plan)

# Step 2 — DMuon（DP = replicate × shard）
dmuon.dedicate_params(
    model,
    mesh=mesh3d["shard"],
    replicate_mesh=mesh3d["replicate"],
    predicate=...,
)

# Step 3 — FSDP2（同一个 DP 2D 切片）
fully_shard(model, mesh=mesh3d["replicate", "shard"])

optimizer = dmuon.Muon(model, lr=0.02)
```

---

## 要求

1. **有 TP 时必须用带 `mesh_dim_names` 的 DeviceMesh。** DMuon 通过名称
   集合差识别 TP 轴（`parameter.DTensor.mesh_dim_names − dp_mesh_dim_names
   = TP dim names`）。未命名 mesh 下的 `DTensor` 会 raise `ValueError`。
2. **TP size = 1 自动视为无 TP。** `(dp=N, tp=1)` mesh 行为和 `(dp=N,)`
   **bit-identical**——检测 guard 会短路到纯 DP 路径。
3. **调用顺序**：`parallelize_module` → `dmuon.dedicate_params` →
   `fully_shard`。DMuon 必须在 FSDP2 注册分片契约之前看到 TP-wrapped
   的参数。

---

## DDP + TP

当 data-parallel 维度保持完全 replicated、TP 保持在每个 replica 内部时，
使用 TP-aware DDP 入口，而不是 FSDP2 路径：

```python
parallelize_module(model, mesh["tp"], plan)                 # 先应用 TP
dmuon.dedicate_params_ddp_tp(model, mesh["dp"], predicate=...)
dmuon.replicate_tp(model, mesh["dp"])                       # 非专属参数
optimizer = dmuon.Muon(model, lr=0.02)
```

`dedicate_params_ddp_tp()` 为专属矩阵安装 TP gather → owner update → TP
scatter 路径。`replicate_tp()` 负责把非专属 TP 参数的 TP-local shard 沿 DP
mesh 广播。普通 `dedicate_params_ddp()` 仍会拒绝 TP-sharded dedicated
parameters，因为它不会安装 TP-aware replicated-gradient path。

---

## Runtime 参数

大多数 TP 训练使用默认值即可。高级参数现在都是显式构造参数，而不是环境变量：

* `dedicate_params(..., tp_buffer_reuse=...)` 控制是否复用 TP gather 和/或
  scatter scratch buffer。可选值是 `False`、`True`、`"gather"`、
  `"scatter"` 和 `"all"`。
* `Muon(..., tp_distributed_gram=True)` 为 TP-sharded 矩阵开启
  TP-aware distributed Gram 路径。默认
  `tp_distributed_gram_policy="beneficial"` 下，只有当 Gram factor payload
  预计小于完整 update scatter 时才使用。
* `Muon(..., replicate_async=...)` 控制 DP/HSDP post-step publish overlap。
  它不是 TP 专属开关，但 TP scatter 会参与同一条 post-step publish 调度。

---

## Sync vs async post-step

`Muon` 使用 `replicate_async` 控制 post-step publish 的时序：

```python
# 默认 — scatter + broadcast 异步（留到下一次 forward 的 pre_forward_wait 消费）
optimizer = dmuon.Muon(model, lr=0.02)                        # async
optimizer = dmuon.Muon(model, lr=0.02, replicate_async=True)  # 显式

# 同步 — scatter + broadcast 在 step() 返回前完成
optimizer = dmuon.Muon(model, lr=0.02, replicate_async=False)
```

两种模式在已支持的 TP 拓扑上产生一致 loss 轨迹（2026-04-30 已覆盖
TP2、TP4、DP×TP2、DP×TP4、HSDP×TP2）。async 的好处纯粹是 scatter
NCCL kernel 能和下一个 iter 的 forward compute 并发。seq512 Llama3B
benchmark 中，async 相对 sync 的 p50 step time 提升约 `1.05x` 到
`1.14x`，具体取决于拓扑。

---

## 查看 TP 属性

```python
import dmuon
import torch.distributed as dist

for dp in dmuon.get_owned_params(model, rank=dist.get_rank()):
    print(
        f"{dp.param_name}: "
        f"local={tuple(dp._orig_size)}, "
        f"full={tuple(dp.full_shape)}, "
        f"shard_dim={dp.shard_dim}, "
        f"is_tp_owner={dp.is_tp_owner}, "
        f"tp_group_size={dp.tp_group.size() if dp.tp_group else 1}"
    )
```

一个 TP-sharded 参数会显示 `tp_group_size > 1`、非 None 的 `shard_dim`，
并且 `is_tp_owner=True` 仅出现在该参数 TP group 内的一个 rank 上。不同
TP-sharded 参数可以有不同 TP owner；这是 DMuon 用 LPT 均衡 NS 计算负载的
预期行为。

---

## 限制

* **MVP 只支持 1D TP 轴。** 2D TP 会在 `get_tp_mesh` 抛 assert；有需求
  时再扩。
* **每个参数单 owner 跑 NS。** 每个 TP-sharded 参数有一个 TP owner 执行
  完整矩阵 Newton-Schulz，但 owner 会通过 LPT 在不同参数之间变化。
  Canzona 风格的 fused All-to-All + micro-group batching（更紧密地并行化
  一组 NS 调用）列为 future work。
* **TP-sharded 小参数不参与 DMuon 的 small-param 合并**
  （`SMALL_PARAM_THRESHOLD`）。每个 TP-sharded 参数独立跑一次 gather
  / scatter，即使 numel < 5M。实际训练里这种小 TP-sharded 参数很少见。
## 参考

* [HSDP 指南](hsdp.md) — replicate × shard 配置；和 TP 叠加使用
* [`dedicate_params` API](../reference/api.md) — 完整签名
* `docs/internal/research/tp_design.md` — 最终实现设计、生命周期顺序、
  correctness gates 和 benchmark 摘要
* `docs/internal/research/tp_overlap_profile.md` — NSight 风格
  overlap 测量（8-GPU 3D mesh toy 下 100%）
* `docs/internal/research/tp_alignment_report.md` — sync / async
  bit-identical 对齐验证
