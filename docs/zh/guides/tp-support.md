# 张量并行

!!! tip "TL;DR"
    先应用 TP（`parallelize_module`），再 DMuon（在 DP mesh 上 `dedicate_params`），最后 FSDP2（在 DP mesh 上 `fully_shard`）。DMuon 使用 Gram Newton-Schulz 在 TP 本地分片上运行，仅需一次小型 all-reduce，而无需收集完整梯度矩阵。

---

!!! warning "TP + HSDP 组合状态"
    DMuon 的 TP 支持（Gram Newton-Schulz + TP SYRK 分解）仅在 **1D DP mesh**（纯 FSDP，无 HSDP replicate 维）上经过验证。2D HSDP × TP 组合尚未纳入测试矩阵。如需同时使用，请先用 FSDP+TP（单 replicate 行），并提交 issue。

---

## 挑战

使用 TP 时，每个 rank 持有的是参数的**分片**，而非完整矩阵。所有者 rank 的梯度也是 TP 分片。标准 Newton-Schulz 需要完整的 (m, n) 矩阵——但我们在每个 TP rank 上只有 (m/T, n) 或 (m, n/T)。

**朴素方案**：跨 TP rank all-gather 完整梯度 → O(mn) 通信，违背了 DMuon 的初衷。

**DMuon 方案**：**Gram Newton-Schulz** — 在 Gram 矩阵上迭代，而非完整参数。Gram 矩阵可以通过对更小的 (d, d) 矩阵做一次 all-reduce 从 TP 分片重构。

## Gram NS 原理

标准 NS 在完整的 (m, n) 矩阵 X 上迭代。Gram NS 将迭代改写为在 Gram 矩阵 R = X @ X^T（m x m）或 R = X^T @ X（n x n）上进行。

关键洞察是 Gram 矩阵在 TP 分片下**可分解**：

| TP 分片方式 | 示例 | 本地形状 | 可分解的 Gram | All-reduce 大小 |
|---|---|---|---|---|
| **Shard(0)**（行分片） | q_proj, gate_proj | (m/T, n) | R 侧: $G^TG = \sum_i G_i^T G_i$ | n x n |
| **Shard(1)**（列分片） | o_proj, down_proj | (m, n/T) | L 侧: $GG^T = \sum_i G_i G_i^T$ | m x m |

每个 TP rank 计算本地 Gram $G_i^T G_i$ 或 $G_i G_i^T$，然后跨 TP rank 做一次 **all-reduce** 即可得到精确的全局 Gram。NS 迭代在这个 (d, d) 矩阵上本地进行。

!!! success "通信量降低"
    对于标准 Transformer，all-reduce 大小始终是 **d_model x d_model**，与参数形状无关。这是 O(d^2) vs O(mn)——对于 FFN 层（intermediate_size >> d_model）显著降低。

## 设置：TP + DMuon + FSDP2

设置顺序是：**先 TP，再 DMuon，最后 FSDP2**。

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, parallelize_module,
)
from torch.distributed.fsdp import fully_shard
import dmuon

# 2D mesh: dp_size x tp_size
mesh_2d = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
dp_mesh = mesh_2d["dp"]
tp_mesh = mesh_2d["tp"]

model = MyModel().cuda()

# 第 1 步：应用 TP
for layer in model.layers:
    parallelize_module(
        layer.self_attn, tp_mesh,
        {
            "q_proj": ColwiseParallel(),   # Shard(0)
            "k_proj": ColwiseParallel(),   # Shard(0)
            "v_proj": ColwiseParallel(),   # Shard(0)
            "o_proj": RowwiseParallel(),   # Shard(1)
        },
    )
    parallelize_module(
        layer.mlp, tp_mesh,
        {
            "gate_proj": ColwiseParallel(),  # Shard(0)
            "up_proj": ColwiseParallel(),    # Shard(0)
            "down_proj": RowwiseParallel(),  # Shard(1)
        },
    )

# 第 2 步：DMuon（使用 dp_mesh）
dmuon.dedicate_params(
    model, dp_mesh,
    predicate=lambda n, p: "proj" in n and p.ndim == 2,
)

# 第 3 步：FSDP2（也使用 dp_mesh）
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)

# 优化器
optimizer = dmuon.Muon(model, lr=0.02, ns_steps=5, adamw_lr=1e-3)
```

!!! info "为什么用 dp_mesh？"
    `dedicate_params` 和 `fully_shard` 使用 **DP mesh**——它们在数据并行 rank 之间分发参数。TP 分片已经应用完毕；DMuon 操作的是 TP 本地分片。

## 三种 NS 模式

DMuon 为 TP 分片参数提供三种 NS 模式，通过优化器参数选择：

### 1. 精确 Gram NS（默认）

跨 TP rank all-reduce Gram 矩阵，得到精确的全局 Gram。

```python
optimizer = dmuon.Muon(model, lr=0.02)
# per_head_ns=True（默认），block_diagonal_ns=False（默认）
```

**各参数行为：**

| 参数 | 分片方式 | Gram 侧 | All-reduce 大小 | 精确？ |
|------|----------|---------|-----------------|--------|
| q_proj (8192, 8192) | Shard(0) | R 侧 G^TG | 8192 x 8192 | 是 |
| k_proj (1024, 8192) | Shard(0) | 逐头 NS | *无* | 是 |
| gate_proj (28672, 8192) | Shard(0) | R 侧 G^TG | 8192 x 8192 | 是 |
| o_proj (8192, 8192) | Shard(1) | L 侧 GG^T | 8192 x 8192 | 是 |
| down_proj (8192, 28672) | Shard(1) | L 侧 GG^T | 8192 x 8192 | 是 |

### 2. 逐头 NS（GQA k/v_proj 默认使用）

对于 GQA 模型，k_proj 和 v_proj 的头数少于 q_proj（例如 Llama-3 中 8 个 KV 头 vs 32 个 Q 头）。当 TP 大小 <= KV 头数时，每个 TP rank 持有**完整的 KV 头**。

这意味着每个 rank 上的本地 NS 是**精确的**——无需 TP 通信。

```python
optimizer = dmuon.Muon(model, lr=0.02, per_head_ns=True)  # 默认
```

**检测逻辑**：参数在以下三个条件同时满足时使用逐头 NS：

1. `per_head_ns=True`（默认）
2. `shard_dim == 0`（行分片，ColwiseParallel）
3. `full_m < full_n`（窄矩阵——完整行维度小于列维度）

这能正确识别 GQA k/v_proj，同时排除 q_proj、gate_proj 等。

!!! example "Llama-3 8B，TP=8"
    - k_proj：完整 (1024, 8192) → 1024 < 8192 → **逐头 NS**（零 TP 通信）
    - q_proj：完整 (8192, 8192) → 8192 = 8192 → **精确 Gram NS**
    - gate_proj：完整 (28672, 8192) → 28672 > 8192 → **精确 Gram NS**

### 3. 块对角 NS（实验性）

完全跳过 Gram all-reduce，仅使用本地部分 Gram。以近似为代价消除**所有** TP 优化器通信。

```python
optimizer = dmuon.Muon(model, lr=0.02, block_diagonal_ns=True)
```

!!! warning "实验性"
    块对角 NS 是一种近似。它将 Shampoo 的块对角预条件化原理扩展到 Newton-Schulz。收敛验证仍在进行中——请谨慎使用并监控 loss 曲线。

## 注意力变体参考

不同注意力架构产生不同的 TP 分片模式。DMuon 通过通用路由逻辑（shard_dim + full_shape）处理所有变体，无需知道注意力类型。

| 变体 | 关键差异 | k/v_proj 形状 | 逐头 NS？ | 特殊说明 |
|------|---------|---------------|----------|---------|
| **MHA** | n_heads = n_kv_heads | (d, d) | 否（方阵） | 全部使用精确 Gram NS |
| **GQA** | n_kv_heads < n_heads | (kv_dim, d) | 是 | k/v 零 TP 通信 |
| **MQA** | n_kv_heads = 1 | (head_dim, d) | 是 | 比 GQA 更窄 |
| **GateDelta** | V 头 > QK 头 | (d, d)（v） | 否（不窄） | a/b_proj 太小——用 AdamW |
| **GLA** | 类似 GQA | 视具体而定 | 取决于形状 | 检查 full_m vs full_n |
| **RetNet** | 类似 MHA | (d, d) | 否 | 全部使用精确 Gram NS |

**GateDelta 的 predicate 建议：**

```python
def predicate(n, p):
    if p.ndim != 2:
        return False
    # 排除非常小的投影（GateDelta 中的 a_proj, b_proj）
    if p.numel() < 100_000:
        return False
    return "proj" in n
```

## 查看 TP 属性

```python
for dp in dmuon.get_owned_params(model, rank=dist.get_rank()):
    print(
        f"{dp.param_name}: "
        f"local={tuple(dp._orig_size)}, "
        f"full={tuple(dp.full_shape)}, "
        f"shard_dim={dp.shard_dim}, "
        f"tp_group={'是' if dp.tp_group else '否'}"
    )
```

示例输出（Llama-3 8B, TP=8）：
```
q_proj:    local=(1024, 8192), full=(8192, 8192), shard_dim=0, tp_group=是
k_proj:    local=(128, 8192),  full=(1024, 8192), shard_dim=0, tp_group=是
v_proj:    local=(128, 8192),  full=(1024, 8192), shard_dim=0, tp_group=是
o_proj:    local=(8192, 1024), full=(8192, 8192), shard_dim=1, tp_group=是
gate_proj: local=(3584, 8192), full=(28672, 8192), shard_dim=0, tp_group=是
up_proj:   local=(3584, 8192), full=(28672, 8192), shard_dim=0, tp_group=是
down_proj: local=(8192, 3584), full=(8192, 28672), shard_dim=1, tp_group=是
```

## 相关文档

- [HSDP 训练（多机）](hsdp.md) —— 2D mesh 配置；注意 HSDP × TP 组合尚未验证（见上方警告）
- [自定义 Hook 边界](custom-hook-boundaries.md) —— `hook_boundary_predicate` 可使用 `isinstance(m, TPWrappedModule)` 与 TP 包装模块对齐
- [训练流程](training.md) —— 完整 1D 训练流程（在此基础上加 TP）
- [API 文档](../reference/api.md) —— 含 `hook_boundary_predicate` 的 `dedicate_params` 签名
