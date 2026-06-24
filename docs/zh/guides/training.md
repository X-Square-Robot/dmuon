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

`predicate` 函数决定哪些参数进入 DMuon 的 dedicated ownership runtime。
默认设置下，被选中的参数走 Muon，未选中的参数继续走普通 FSDP2/AdamW 路径。
它接收参数的全限定名和参数张量：

```python
def predicate(name: str, param: nn.Parameter) -> bool:
    return ...  # True = DMuon-managed Muon，False = FSDP2/AdamW
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

### 高级模式：Type-Split Routing

在大规模 scaling run 中，如果希望所有可训练参数的通信都由 DMuon 管理，可以传入
更宽的 `predicate` 和一个 `param_policy`。`"muon"` route 让大矩阵参数走
矩阵优化器路径；`"adamw"` route 让小 AdamW 参数走 DMuon 的 owner
broadcast/reduce 路径；`"sharded_adamw"` 只留给 embedding、`lm_head` 这类
需要所有 rank 分担通信的大 AdamW 张量。

```python
dmuon.dedicate_params(
    model,
    mesh,
    predicate=lambda n, p: p.requires_grad,
    param_policy={
        "defaults": {"route": "adamw", "param_dtype": torch.bfloat16},
        "overrides": [
            {
                "name": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "set": {"route": "muon"},
            },
            {
                "name": ["embed_tokens", "lm_head"],
                "set": {"route": "sharded_adamw"},
            },
        ],
    },
)
```

默认的 `predicate=lambda n, p: "proj" in n and p.ndim == 2` 仍然是更简单的
集成路径。只有当你希望 DMuon 同时接管非 Muon 可训练参数的 placement 和
collectives 时，才需要 type-split routing。`route_hint_fn` 仍然支持旧的
route-only 集成，但不能表达 per-module dtype policy。完整路由策略见
[Pure DMuon 路由](pure-dmuon-routing.md)。

### Process Group Policy

DMuon 默认使用 `process_group_policy="isolated"`。这个模式会根据调用方 mesh
里的 rank 复制一套 DMuon 自己的 DP/HSDP/TP process groups，而不是复用
trainer 的 group handle。这样可以把 DMuon 的异步通信序列和 trainer 的
logging、metrics、checkpoint collectives 分开。

`isolated` 不等于 step 末尾同步。DMuon 默认保留 cross-step overlap。只有在
排查疑似 process group 顺序问题时，才设置 `DMUON_ISOLATED_PG_BARRIER=1`
作为诊断 fence。这个 fence 会在 `optimizer.step()` 末尾 drain DMuon publish
work，并对 DMuon-owned process groups 做 barrier，所以正常吞吐和 MFU 测试
都应该保持关闭。

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

### 语义参数组

当训练框架需要业务级学习率分组时使用 `param_groups`，例如 VLA 的
action expert 使用更高学习率。参数组必须从传给 `dmuon.Muon` 的同一个
wrapped model 上构建；在 FSDP2 下，也就是先完成 `dedicate_params()` 和
FSDP2 wrapping，再用当前模型的 `named_parameters()` 分组。DMuon 会把每个
用户组降解为两个优化器子组：`<name>/muon` 管理专属参数，
`<name>/adamw` 管理对称参数和 AdamW-route 的专属参数。

```python
base_params = []
action_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "action_transformer" in name:
        action_params.append(param)
    else:
        base_params.append(param)

optimizer = dmuon.Muon(
    model,
    lr=5e-5,
    adamw_lr=5e-5,
    param_groups=[
        {"params": base_params, "lr": 5e-5, "group_name": "base"},
        {"params": action_params, "lr": 1e-4, "group_name": "action"},
    ],
)
```

普通场景下，`lr` 同时作用于该语义组的 Muon 和 AdamW 子组。高级场景可以用
`muon_lr`、`adamw_lr`、`muon_weight_decay`、`adamw_weight_decay`、
`momentum`、`adamw_betas`、`adamw_eps` 分别覆盖两条路径。每个
trainable 参数必须且只能出现在一个用户组中；如果传入 wrapping 前的旧参数、
重复参数或漏掉参数，优化器构造阶段会直接报错。

默认情况下，语义 `param_groups` 只表达超参数分组，不负责选择 DMuon route。
对于 DMuon-managed 参数，`dedicate_params(param_policy=...)` 写入的逐参数
route 会被保留；即使同一个用户组里同时包含 `"muon"`、`"adamw"` 和
`"sharded_adamw"` 参数，也不会被整组改路由。只有当用户组显式设置
`dmuon_route`、`dmuon_optimizer` 或 `matrix_optimizer` 时，DMuon 才会把这个
语义组里的 DMuon-managed 参数整体强制到指定 route；这些 key 只应在该语义组
内所有 DMuon-managed 参数确实都要走同一路径时使用。

Scheduler 和 checkpoint 仍然通过 `optimizer.param_groups` 工作。可见组名会
变为 `base/muon`、`base/adamw`、`action/muon`、`action/adamw`，这也是
检查 route split 的公开入口。

### 超参数指南

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 0.02 | Muon 学习率。内部按 `0.2 * sqrt(max(m,n))` 逐参数缩放。 |
| `momentum` | 0.95 | 越大越平滑。0.95 是标准 Muon/Moonlight 值。 |
| `ns_steps` | 5 | NS 迭代次数。5 次足以收敛。 |
| `ns_backend` | `"gram"` | `"gram"` 或 `"direct"` 字符串，或 `dmuon.NewtonSchulz(...)` 对象以自定义系数。 |
| `nesterov` | True | Nesterov 前瞻：`ns_input = grad + mu * buf`。推荐开启。 |
| `adamw_lr` | 1e-3 | 非矩阵参数的独立学习率。 |
| `param_groups` | None | 可选的 PyTorch 风格语义参数组，会被降解成 Muon 和 AdamW 子组。 |

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

1. `optimizer.step()` 内部：(a) 等待异步梯度 reduce 完成，(b) 对 routed matrix params 运行 Muon，(c) 根据 route 设置，对 FSDP2-managed 参数或 DMuon-managed sharded AdamW 参数运行 AdamW。

训练循环与标准 PyTorch 完全相同。无需特殊 hook 或上下文管理器。

### 梯度裁剪

普通 `param.grad` 仍然使用 PyTorch 原生 `clip_grad_norm_`；如果希望
DMuon 专属参数也被覆盖，再额外补一行 DMuon 的 Muon-only clip：

```python
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()

    # 非专属 / AdamW 参数：继续交给训练框架原有逻辑。
    torch.nn.utils.clip_grad_norm_(adamw_params, max_norm=1.0)

    # DMuon 专属 / Muon 参数：梯度不在 param.grad 上，需要 DMuon 入口。
    dmuon.clip_grad_norm_(optimizer, max_norm=1.0)

    optimizer.step()
```

!!! info "这里裁剪了什么？"
    `dmuon.clip_grad_norm_` 只裁剪 DMuon 专属参数，不会触碰 AdamW 参数。
    因此现有训练框架可以继续使用标准 PyTorch clip 处理普通
    `param.grad`，DMuon 只补齐 Muon 参数这一部分。

    Muon clip 发生在 DMuon 异步 reduce / TP gather 之后、momentum +
    Newton-Schulz 之前。Newton-Schulz 会约束最终矩阵 update 的尺度，
    所以这里的 clip 更像异常梯度、momentum buffer 污染和 non-finite
    检查的保护，而不是主要的学习率控制机制。

默认策略是对 Muon 梯度做 global p-norm clipping。后续如果需要接入
MuonClip、QK/投影层专用 clip 等方案，可以通过
`dmuon.register_muon_grad_clip_strategy(...)` 注册自定义策略。

## 日志与调试

### 检查 NS 后端

```python
print(f"NS 后端: {dmuon.get_ns_backend()}")
# "Gram NS · kernel=cute_sm80 (SM80, DMuon internal)"    — A100/A800 快路径
# "Gram NS · kernel=quack    (SM90, Tri Dao quack)"      — H100/B200/B300 快路径
# "Gram NS · kernel=cublas   (SM80, universal fallback)" — 通用 cuBLAS 后备
```

`dmuon.get_backend_status()` 返回完整的各后端可用性 dict。完整的
自动检测阶梯与 `kernel=` / `DMUON_NS_KERNEL` 覆盖方式详见
[后端分发](../reference/newton-schulz.md#backend-dispatch)。

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

完整 API 与同步/异步模式见 [HSDP 训练指南](hsdp.md)。

FSDP 和 HSDP 下都适用的 DMuon-Z2 vs DMuon-Z3 packed buffer 生命周期选择，详见 [Z2 与 Z3 模式](z2-z3-modes.md)。

## 相关文档

- [HSDP 训练（多机）](hsdp.md) — 2D mesh + 异步 broadcast
- [自定义 Hook 边界](custom-hook-boundaries.md) — 控制 DMuon 的前向/反向 hook 绑定到哪个模块
- [Z2 与 Z3 模式](z2-z3-modes.md) — Packed buffer 生命周期与显存/通信权衡
- [张量并行](tp-support.md) — 使用 DMuon + TP
- [检查点](checkpoint.md) — 保存和加载训练状态
- [梯度累积](grad-accumulation.md) — 等效批量大小扩展
