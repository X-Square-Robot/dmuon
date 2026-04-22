# 训练指南

!!! tip "TL;DR"
    1. 在 `fully_shard()` **之前**调用 `dmuon.dedicate_params(model, mesh, predicate=...)` 将矩阵参数分配给专属所有者。
    2. 使用标准 FSDP2 的 `fully_shard()` 包装模型——DMuon 会自动跳过专属参数。
    3. 使用 `dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)` 作为优化器——在一次调用中同时处理 Muon（专属）和 AdamW（对称）参数。

---

## 概述

DMuon 训练设置分四步：

1. **构建模型** — 标准 PyTorch 模型
2. **`dedicate_params()`** — 标记矩阵参数为专属所有权
3. **`fully_shard()`** — 对剩余参数应用 FSDP2
4. **训练循环** — 与标准 PyTorch 相同

## 第 1 步：模型准备

DMuon 适用于任何 `nn.Module`。无需特殊基类或包装器。

```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

dist.init_process_group("nccl")
torch.cuda.set_device(dist.get_rank())

mesh = init_device_mesh("cuda", (dist.get_world_size(),))

model = MyModel().cuda()
```

!!! tip "HuggingFace 模型"
    DMuon 兼容 HuggingFace 模型。照常使用 `AutoModelForCausalLM.from_pretrained(...)`，然后应用 DMuon + FSDP2。

## 第 2 步：标记专属参数

```python
import dmuon

assignment = dmuon.dedicate_params(
    model,
    mesh,
    predicate=lambda n, p: "proj" in n and p.ndim == 2,
)
```

### 编写 Predicate

`predicate` 函数决定哪些参数使用 Muon（专属）vs AdamW（对称）。它接收参数的全限定名和参数张量：

```python
def predicate(name: str, param: nn.Parameter) -> bool:
    return ...  # True = 专属（Muon），False = 对称（AdamW）
```

**常见模式：**

=== "标准 Transformer（Llama, Qwen, Mistral）"

    ```python
    # 所有 2D 投影层 → Muon
    predicate = lambda n, p: "proj" in n and p.ndim == 2
    ```

=== "选择性（排除 embedding）"

    ```python
    # 仅投影层，排除 embed/head
    def predicate(n, p):
        if p.ndim != 2:
            return False
        if "embed" in n or "head" in n or "lm_head" in n:
            return False
        return "proj" in n
    ```

=== "GateDelta / 混合注意力"

    ```python
    # 排除非常小的投影（GateDelta 中的 a_proj, b_proj）
    def predicate(n, p):
        if p.ndim != 2:
            return False
        if p.numel() < 100_000:  # 太小，NS 没有收益
            return False
        return "proj" in n
    ```

**指导原则：**

- **仅 2D 矩阵** — 1D 参数（LayerNorm、bias）应使用 AdamW
- **足够大以受益于 NS** — 非常小的矩阵不会从 Newton-Schulz 中获益。粗略阈值：`numel > 100k`
- **Embedding/head 层** — 通常保留在 AdamW 下（它们不太适合 NS 优化几何）

### 查看分配结果

```python
# 当前 rank 分到了什么？
owned = dmuon.get_owned_params(model, rank=dist.get_rank())
total_owned = sum(dp.numel for dp in owned)
print(f"Rank {dist.get_rank()}: 拥有 {len(owned)} 个专属参数，{total_owned:,} 个元素")
```

## 第 3 步：应用 FSDP2

```python
from torch.distributed.fsdp import fully_shard

for layer in model.layers:  # 或 HuggingFace 的 model.model.layers
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)
```

这是标准 FSDP2 用法。DMuon 的 monkey-patch 确保 `fully_shard()` 自动跳过专属参数。

!!! warning "顺序：先 dedicate，再 shard"
    `dedicate_params()` 必须在 `fully_shard()` **之前**调用。Monkey-patch 需要在 FSDP2 处理参数时 `_dedicated_owner_rank` 标记已经存在。

## 第 4 步：创建优化器

```python
optimizer = dmuon.Muon(
    model,
    lr=0.02,              # Muon 学习率（专属参数）
    momentum=0.95,        # 动量系数
    ns_steps=5,           # Newton-Schulz 迭代次数
    nesterov=True,        # Nesterov 动量（推荐）
    weight_decay=0.0,     # 专属参数的权重衰减
    adamw_lr=1e-3,        # AdamW 学习率（对称参数）
    adamw_betas=(0.9, 0.999),
    adamw_weight_decay=0.01,
    adamw_eps=1e-8,
)
```

`dmuon.Muon` 在单一优化器中管理两类参数：

- **组 0**（专属参数）：Muon — 动量 + NS + 更新，仅所有者
- **组 1**（对称参数）：AdamW — 标准，所有 rank

### 超参数指南

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 0.02 | Muon 学习率。内部按 `0.2 * sqrt(max(m,n))` 逐参数缩放。 |
| `momentum` | 0.95 | 越大越平滑。0.95 是标准 Muon/Moonlight 值。 |
| `ns_steps` | 5 | NS 迭代次数。5 次足以收敛。 |
| `ns_backend` | `"gram"` | `"gram"` 或 `"direct"` 字符串，或 `dmuon.NewtonSchulz(...)` 对象以自定义系数。 |
| `nesterov` | True | Nesterov 前瞻：`ns_input = grad + mu * buf`。推荐开启。 |
| `adamw_lr` | 1e-3 | 非矩阵参数的独立学习率。 |

## 第 5 步：训练循环

```python
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()
    optimizer.step()  # (1)!

    if dist.get_rank() == 0:
        print(f"step {step}: loss={loss.item():.4f}")
```

1. `optimizer.step()` 内部：(a) 等待所有异步梯度 reduce 完成，(b) 对专属参数运行 Muon，(c) 对 FSDP2 参数运行 AdamW。

训练循环与标准 PyTorch 完全相同。无需特殊 hook 或上下文管理器。

### 梯度裁剪

直接使用 PyTorch 原生的 `clip_grad_norm_`——与 DMuon 天然兼容：

```python
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

!!! info "为什么直接可用？"
    专属参数的梯度存储在 `_reduced_grad` 中，不在 `param.grad` 上。因此 `clip_grad_norm_` 天然只看到对称参数（LayerNorm、embedding）——恰好是真正需要裁剪的那些。

    专属参数不需要裁剪，因为 Newton-Schulz 正交化会将梯度投影到正交矩阵上，输出的谱范数有界，与输入梯度大小无关。

## 日志与调试

### 检查 NS 后端

```python
print(f"NS 后端: {dmuon.get_ns_backend()}")
# "syrk_sm80" = CuteDSL SYRK 内核（最快）
# "compiled"  = @torch.compile 后备方案
```

### 验证参数分配

```python
import logging
logging.basicConfig(level=logging.INFO)

# dedicate_params() 会输出分配摘要：
# INFO: dedicate_params: 56 params assigned to 8 ranks, imbalance=0.2%, loads=[...]
```

### 检查专属与对称参数数量

```python
all_dp = dmuon.get_dedicated_params(model)
owned_dp = dmuon.get_owned_params(model, rank=dist.get_rank())
fsdp_count = len(list(model.parameters())) - len(all_dp)

print(f"专属参数: 共 {len(all_dp)} 个，当前 rank 拥有 {len(owned_dp)} 个")
print(f"对称参数（FSDP2）: {fsdp_count} 个")
```

## 扩展到多机

从单机多卡走到**多机训练**时，把 1D `init_device_mesh("cuda", (world_size,))` 换成 2D HSDP mesh，并把 replicate 维度传给 `dedicate_params`。DMuon 自动处理两阶段 grad reduce（shard → replicate）+ 异步 post-step broadcast；训练循环其他部分无改动。

```python
hsdp = init_device_mesh(
    "cuda", (replicate_size, shard_size),
    mesh_dim_names=("replicate", "shard"),
)
dmuon.dedicate_params(
    model, hsdp["shard"],
    predicate=lambda n, p: "proj" in n and p.ndim == 2,
    replicate_mesh=hsdp["replicate"],   # ← HSDP 开关
)
for layer in model.layers:
    fully_shard(layer, mesh=hsdp)
fully_shard(model, mesh=hsdp)
```

完整 API、同步/异步模式、fallback 协议、profile 说明见 [HSDP 训练指南](hsdp.md)。

FSDP 和 HSDP 下都适用的 DMuon-Z2 vs DMuon-Z3 packed buffer 生命周期选择，详见 [Z2 与 Z3 模式](z2-z3-modes.md)。

## 相关文档

- [HSDP 训练（多机）](hsdp.md) — 2D mesh + 异步 broadcast
- [自定义 Hook 边界](custom-hook-boundaries.md) — 控制 DMuon 的前向/反向 hook 绑定到哪个模块
- [Z2 与 Z3 模式](z2-z3-modes.md) — Packed buffer 生命周期与显存/通信权衡
- [性能分析与 Fallback](profiling-and-fallback.md) — 广播延迟测量与异步 fallback 调优
- [张量并行](tp-support.md) — 使用 DMuon + TP
- [检查点](checkpoint.md) — 保存和加载训练状态
- [梯度累积](grad-accumulation.md) — 等效批量大小扩展
