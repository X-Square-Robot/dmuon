# API 文档

DMuon 公开 API 的完整参考。

---

## 核心 API

### `dmuon.dedicate_params`

```python
dmuon.dedicate_params(
    model: nn.Module,
    mesh: DeviceMesh,
    predicate: Callable[[str, nn.Parameter], bool],
    compute_dtype: torch.dtype = None,
    reshard_after_forward: bool = True,
) -> dict[nn.Parameter, int]
```

标记参数为专属所有权并注册通信 hook。

满足 `predicate` 的参数通过均衡分配算法分配给所有者 rank。每个标记的参数将在后续 `fully_shard()` 调用中被自动忽略。

**参数：**

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `nn.Module` | *必需* | 要分配参数的模型。 |
| `mesh` | `DeviceMesh` | *必需* | 数据并行维度的 1D DeviceMesh。 |
| `predicate` | `Callable` | *必需* | `(param_name, param) -> bool`。返回 True 的参数使用专属所有权。 |
| `compute_dtype` | `torch.dtype` | `None` | 可选的通信数据类型（如 `torch.bfloat16`）。 |
| `reshard_after_forward` | `bool` | `True` | True 时前向后重分片（类似 `FULL_SHARD`）。False 时保持完整（类似 `SHARD_GRAD_OP`）。 |

**返回：** 将每个专属参数映射到其所有者 rank（int）的字典。

---

### `dmuon.wait_all_reduces`

```python
dmuon.wait_all_reduces(model: nn.Module) -> None
```

等待所有待处理的梯度 reduce 完成。

由 `optimizer.step()` 自动调用。仅在需要在 step 前手动访问 `_reduced_grad` 时才需要。

---

### `dmuon.no_sync`

```python
@contextmanager
dmuon.no_sync(model: nn.Module)
```

禁用梯度归约的上下文管理器，用于梯度累积。

在此上下文内，反向传播跳过 reduce 通信并在本地累积梯度。同时禁用 FSDP2 对对称参数的梯度同步。

```python
with dmuon.no_sync(model):
    loss = model(batch).loss / accum_steps
    loss.backward()
```

---

## 优化器

### `dmuon.Muon`

```python
dmuon.Muon(
    model: nn.Module,
    lr: float = 0.02,
    momentum: float = 0.95,
    weight_decay: float = 0.0,
    ns_steps: int = 5,
    adamw_lr: float = 1e-3,
    adamw_betas: tuple[float, float] = (0.9, 0.999),
    adamw_weight_decay: float = 0.01,
    adamw_eps: float = 1e-8,
    nesterov: bool = True,
    per_head_ns: bool = True,
    block_diagonal_ns: bool = False,
)
```

DMuon 分布式训练的组合优化器。

管理两个参数组：

- **组 0**（专属参数）：Muon — 动量 + Newton-Schulz 正交化，仅所有者运行
- **组 1**（对称参数）：AdamW，所有 rank 在 FSDP2 分片上运行

**参数：**

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `nn.Module` | *必需* | 已应用 `dedicate_params` 和 `fully_shard` 的模型。 |
| `lr` | `float` | `0.02` | 专属参数的 Muon 学习率。内部按 `0.2 * sqrt(max(m,n))` 缩放。 |
| `momentum` | `float` | `0.95` | 专属参数的动量系数。 |
| `weight_decay` | `float` | `0.0` | 专属参数的权重衰减（解耦式，类似 AdamW）。 |
| `ns_steps` | `int` | `5` | Newton-Schulz 迭代次数。 |
| `ns_backend` | `str` 或 `NewtonSchulz` | `"gram"` | NS 后端配置。传入字符串简写（`"gram"` 或 `"direct"`）使用默认系数，或传入 `NewtonSchulz` 对象完全自定义（见下方）。 |
| `nesterov` | `bool` | `True` | 使用 Nesterov 动量：`ns_input = grad + mu * buf`。 |
| `per_head_ns` | `bool` | `True` | 对窄 Shard(0) 参数（GQA k/v_proj）使用逐头本地 NS。 |
| `block_diagonal_ns` | `bool` | `False` | 跳过所有 TP 参数的 Gram all-reduce（实验性）。 |
| `adamw_lr` | `float` | `1e-3` | 对称参数的 AdamW 学习率。 |
| `adamw_betas` | `tuple` | `(0.9, 0.999)` | AdamW beta 系数。 |
| `adamw_weight_decay` | `float` | `0.01` | AdamW 权重衰减。 |
| `adamw_eps` | `float` | `1e-8` | AdamW epsilon。 |

**方法：**

- `step(closure=None)` — 执行一步优化。内部：(1) 等待 reduce，(2) 对专属参数运行 Muon，(3) 对 FSDP2 参数运行 AdamW。
- `zero_grad(set_to_none=True)` — 清除两类参数的梯度。

---

### `dmuon.NewtonSchulz`

```python
dmuon.NewtonSchulz(
    backend: str = "gram",
    coefficients: list[list[float]] = None,
    restart_iterations: list[int] = None,
)
```

可配置的 Newton-Schulz 后端对象。传入 `Muon(ns_backend=...)` 以自定义系数或算法选择。

**参数：**

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend` | `str` | `"gram"` | `"gram"`：Gram 空间 NS（SYRK + 重启）。`"direct"`：经典参数空间 NS。 |
| `coefficients` | `list` | `None` | 逐步 `(a, b, c)` 系数。`None` 使用 `POLAR_EXPRESS_COEFFICIENTS`。 |
| `restart_iterations` | `list[int]` | `None` | Gram 空间 NS 的重启位置。`None` 使用 `[2]`。`"direct"` 时忽略。 |

**用法：**

```python
import dmuon

# 默认 Gram 空间 NS
optimizer = dmuon.Muon(model, lr=0.02, ns_backend="gram")

# 经典 Muon/Moonlight + You 系数
ns = dmuon.NewtonSchulz("direct", coefficients=dmuon.YOU_COEFFICIENTS)
optimizer = dmuon.Muon(model, lr=0.02, ns_backend=ns)

# Gram 空间 + You 系数
ns = dmuon.NewtonSchulz("gram", coefficients=dmuon.YOU_COEFFICIENTS)
optimizer = dmuon.Muon(model, lr=0.02, ns_backend=ns)
```

!!! note "说明"
    需要 Gram 分解的 TP 参数（精确或块对角）内部始终使用 `gram_newton_schulz`——`backend` 设置仅影响本地（非 TP）参数和 TP 逐头参数。自定义 `coefficients` 应用于所有路径。

---

## Newton-Schulz 函数

### `dmuon.newton_schulz`

```python
dmuon.newton_schulz(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: list[list[float]] = None,
    restart_iterations: list[int] = None,
) -> Tensor
```

Newton-Schulz 正交化（默认 Gram 空间后端）。

路由到 `gram_newton_schulz_local()`，使用 Gram 空间迭代，支持逐步系数、重启机制和 SYRK 加速。

**参数：**

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `G` | `Tensor` | *必需* | 梯度矩阵 (m, n)，任意 dtype。 |
| `steps` | `int` | `5` | 忽略（由 `len(coefficients)` 决定）。 |
| `eps` | `float` | `1e-7` | 归一化 epsilon。 |
| `coefficients` | `list` | `POLAR_EXPRESS_COEFFICIENTS` | 逐步 `(a, b, c)` 系数。 |
| `restart_iterations` | `list[int]` | `[2]` | 重启位置，提升数值稳定性。 |

**返回：** 正交化后的更新，与 G 形状相同。

---

### `dmuon.gram_newton_schulz`

```python
dmuon.gram_newton_schulz(
    G_shard: Tensor,
    tp_group: dist.ProcessGroup,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: list[list[float]] = None,
    restart_iterations: list[int] = None,
    shard_dim: int = None,
    block_diagonal: bool = False,
) -> Tensor
```

带 TP SYRK 分解的 Gram Newton-Schulz。

在 Gram 矩阵上迭代。`shard_dim` 控制使用哪侧 Gram：

- **Shard(0)**（行分片）：转置使用 R 侧 G^TG（可分解为本地项之和）
- **Shard(1)**（列分片）：使用 L 侧 GG^T（可分解为本地项之和）

**参数：**

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `G_shard` | `Tensor` | *必需* | 当前 rank 的 TP 分片梯度。 |
| `tp_group` | `ProcessGroup` | *必需* | 用于 all-reduce 的 TP 进程组。 |
| `shard_dim` | `int` | `None` | TP 分片维度（0 或 1）。None 时回退到形状启发式。 |
| `block_diagonal` | `bool` | `False` | True 时跳过 TP all-reduce（块对角近似）。 |

**返回：** 正交化后的更新分片，与输入形状相同。

---

### `dmuon.direct_newton_schulz`

```python
dmuon.direct_newton_schulz(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: list[list[float]] = None,
) -> Tensor
```

标准的直接（参数）空间 Newton-Schulz。

在完整 (m, n) 矩阵上迭代：$X_{k+1} = a_k X + b_k (XX^T)X + c_k (XX^T)^2 X$

这是 Muon/Moonlight 的经典算法。用于基线对比或 Gram 空间开销不值得的小矩阵。

**返回：** 正交化后的更新，与 G 形状相同。

---

## 检查工具

### `dmuon.get_dedicated_params`

```python
dmuon.get_dedicated_params(model: nn.Module) -> list[DedicatedParam]
```

从模型中收集所有 `DedicatedParam` 实例（跨所有 rank）。

---

### `dmuon.get_owned_params`

```python
dmuon.get_owned_params(model: nn.Module, rank: int) -> list[DedicatedParam]
```

收集指定 rank 拥有的 `DedicatedParam` 实例。

---

### `dmuon.get_comm_ctx`

```python
dmuon.get_comm_ctx(model: nn.Module) -> Optional[DedicatedCommContext]
```

获取模型的 `DedicatedCommContext`（如果存在）。

---

### `dmuon.get_ns_backend`

```python
dmuon.get_ns_backend() -> str
```

返回当前 Newton-Schulz 后端：`"syrk_sm80"` 或 `"compiled"`。

---

## 检查点函数

### `dmuon.get_model_state_dict`

```python
dmuon.get_model_state_dict(
    model: nn.Module,
    *,
    cpu_offload: bool = True,
) -> dict[str, torch.Tensor]
```

获取包含专属参数和 FSDP2 参数的完整模型 state dict。

生成的 state dict 与单卡模型输出相同。**集合操作**——所有 rank 必须调用。

---

### `dmuon.set_model_state_dict`

```python
dmuon.set_model_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None
```

将完整 state dict 加载到 DMuon 模型。处理专属参数和 FSDP2 参数。

---

### `dmuon.get_optimizer_state_dict`

```python
dmuon.get_optimizer_state_dict(
    model: nn.Module,
    optimizer: Muon,
    *,
    cpu_offload: bool = True,
) -> dict
```

获取 DMuon Muon 优化器的 state dict。**集合操作**——所有 rank 必须调用。

返回包含以下段的字典：`"fsdp"`（AdamW 状态）、`"dedicated"`（Muon 动量缓冲区）、`"param_groups"`（超参数）。

---

### `dmuon.set_optimizer_state_dict`

```python
dmuon.set_optimizer_state_dict(
    model: nn.Module,
    optimizer: Muon,
    state_dict: dict,
) -> None
```

将优化器 state dict 加载到 DMuon Muon 优化器。

---

## DedicatedParam 属性

`DedicatedParam` 实例（由 `get_dedicated_params` / `get_owned_params` 返回）暴露以下属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `is_owner` | `bool` | 当前 rank 是否拥有该参数。 |
| `owner_rank` | `int` | 拥有该参数的 DP 本地 rank。 |
| `numel` | `int` | （本地）参数的元素数。 |
| `param_name` | `str` | 参数在父模块中的名称（如 `"weight"`）。 |
| `is_dtensor` | `bool` | 是否为 TP 分片的 DTensor 参数。 |
| `tp_group` | `ProcessGroup` | TP 进程组，非 DTensor 时为 None。 |
| `shard_dim` | `int` | TP 分片维度（0 或 1），非 DTensor 时为 None。 |
| `full_shape` | `torch.Size` | 完整（未分片）参数形状。 |
| `_orig_size` | `torch.Size` | 本地（分片后）参数形状。 |
| `_owned_data` | `Tensor` | 完整参数数据（仅所有者 rank）。 |
| `_reduced_grad` | `Tensor` | 归约后的梯度（仅所有者，backward + wait 后）。 |

---

## 常量

### 系数集

```python
dmuon.YOU_COEFFICIENTS          # @YouJiacheng 的 5 步系数
dmuon.POLAR_EXPRESS_COEFFICIENTS  # Polar Express 论文的 5 步系数（默认）
```

两者都是 `(a, b, c)` 元组的列表。传递给 NS 函数的 `coefficients` 参数。
