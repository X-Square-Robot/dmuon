# 所有权域（多并行支持） {#ownership-domains}

<p class="dmuon-lead">在单次 <code>dedicate_params()</code> 调用中原生组合 DP、HSDP、CP、EP、PP 和 TP：共享唯一通信上下文、内置每域梯度缩放、域内检查点同步以及全局/分算法快速梯度裁剪。</p>

当模型中所有参数共享同一套副本拓扑（纯 DP、1D FSDP2、2D HSDP 或 DP+TP）时，只需向 `dmuon.dedicate_params()` 传入 `mesh`（及可选 `replicate_mesh`）。

但在 **MoE + 专家并行（EP）** 或 **流水线并行（PP）+ EP + HSDP** 等混合并行架构中，不同参数子集落在不同的副本维度上：

- **Dense 参数**（Attention、Norm、Router、Embedding、共享专家）跨 `dp × cp × ep` 复制；
- **EP 本地专家参数**在 `ep` 维度上互不相同，只跨 `dp × cp` 复制，且归约后的梯度需要乘 `1.0 / ep_size` 以对齐全局 loss 归一化；
- **2D HSDP + 1D 混合场景**中，Dense 参数可运行在 2D `(replicate, shard)` HSDP 域上，而本地专家运行在 1D `dp` 域上。

`OwnershipDomain` 将上述多并行拓扑作为 DMuon 的一等公民概念原生支持——无需重复调用 `dedicate_params()`、无需分裂 `DedicatedCommContext`，也无需关闭异步 publish 与前向预取。

---

## 纯 DP / FSDP2 / HSDP 零改动兼容 {#single-domain-compat}

现有单域调用方**完全不需要修改任何代码**：

```python
# 保持不变 —— 内部自动包装为唯一 OwnershipDomain("default", shard_mesh=mesh, ...)
dmuon.dedicate_params(model, mesh, replicate_mesh=rep_mesh, predicate=...)
```

未传 `domains` 时：

- `grad_scale == 1.0` 直接跳过缩放 kernel（零额外开销）；
- LPT 分区、异步通信行为以及 checkpoint 格式与原有单域路径逐位一致。

---

## MoE + EP 完整实战示例 {#moe-ep-example}

向 `dmuon.dedicate_params()` 传入 `domains`（`name -> OwnershipDomain` 映射）和 `default_domain`，并在 `param_policy` 或 `param_policy_fn` 中通过 `"domain"` 字段指定每个参数所属的域：

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
import dmuon
from dmuon import MatrixLayout, OwnershipDomain

# 示例：4 卡 DP=2, EP=2（同样适用于 DP=1, EP=4）
mesh = init_device_mesh("cuda", (dp_size, ep_size), mesh_dim_names=("dp", "ep"))
dense_mesh = mesh["dp", "ep"]._flatten()
expert_mesh = mesh["dp"]

domains = {
    "dense": OwnershipDomain(
        name="dense",
        shard_mesh=dense_mesh,
        grad_scale=1.0,
    ),
    "expert": OwnershipDomain(
        name="expert",
        shard_mesh=expert_mesh,
        grad_scale=1.0 / ep_size,
    ),
}

def policy_fn(name: str, param: torch.nn.Parameter) -> dict[str, object]:
    is_expert = ".mlp.experts." in name
    domain = "expert" if is_expert else "dense"
    if is_expert and param.ndim == 3:
        return {
            "domain": domain,
            "route": "muon",
            "matrix_layout": MatrixLayout.view(*param.shape),
            "param_dtype": torch.bfloat16,
            "grad_dtype": torch.bfloat16,
        }
    if name.endswith(("embed_tokens.weight", "lm_head.weight")):
        return {"domain": domain, "route": "sharded_adamw"}
    if param.ndim >= 2:
        return {"domain": domain, "route": "muon"}
    return {"domain": domain, "route": "adamw"}

dmuon.dedicate_params(
    model,
    domains=domains,
    default_domain="dense",
    predicate=lambda _n, p: p.requires_grad,
    param_policy_fn=policy_fn,
)

optimizer = dmuon.Muon(
    model,
    lr=0.02,
    adamw_lr=3e-4,
    replicate_async=True,
    group_prepare_ahead=True,
    forward_prefetch_depth=1,
)
```

### 内部执行机制

1. **按域独立规划**：每个域在自身的 `shard_mesh`（及可选 `replicate_mesh`）上独立运行 LPT 均衡分配，保证本地专家的 owner 严格落在 `expert_mesh` 内部，同时 dense 参数跨全量 `dense_mesh` 均衡分布。
2. **唯一 `DedicatedCommContext`**：全模型所有域共享同一套 CUDA stream、`all_states`、`post_forward_order` 与跨层预取链。
3. **每组自治（Per-Group Self-Contained）**：每个 `DedicatedParamGroup` 仅包含同一域的参数，独立持有自身的 `_dp_group`、`_replicate_group`、`_grad_reduce_op` 与 `_grad_scale`。
4. **`grad_scale` 与 `no_sync` 安全配合**：`grad_scale` 仅在真正完成跨卡归约后的 `wait_for_reduce` 阶段乘一次，同时覆盖 Muon、`MatrixLayout`、AdamW 和 `sharded_adamw`；在 `dmuon.no_sync(model)` 局部累加期间绝不重复缩放。

---

## 多并行拓扑配置速查表 {#topology-recipes}

| 并行组合 | 参数类别 | `shard_mesh` | `replicate_mesh` | `grad_scale` |
|---|---|---|---|---|
| 纯 DP / FSDP2 | 全部 | `mesh["dp"]` | `None` | `1.0` |
| 2D HSDP | 全部 | `mesh["shard"]` | `mesh["replicate"]` | `1.0` |
| DP × EP（含 `DP=1, EP=4`） | Dense | `mesh["dp", "ep"]._flatten()` | `None` | `1.0` |
| DP × EP | 本地专家 | `mesh["dp"]` | `None` | `1.0 / ep_size` |
| DP × CP × EP | Dense | `mesh["dp", "cp", "ep"]._flatten()` | `None` | `1.0` |
| DP × CP × EP | 本地专家 | `mesh["dp", "cp"]._flatten()` | `None` | `1.0 / ep_size` |
| PP × DP × EP | Dense（各 PP stage） | `mesh["dp", "ep"]._flatten()` | `None` | `1.0` |
| PP × DP × EP | 本地专家（各 PP stage） | `mesh["dp"]` | `None` | `1.0 / ep_size` |
| 混合 2D HSDP + 1D EP | Dense | `mesh["shard"]` | `mesh["replicate"]` | `1.0` |
| 混合 2D HSDP + 1D EP | 本地专家 | `mesh["replicate"]`（或 `mesh["dp"]`） | `None` | `1.0 / ep_size` |
| 5D（PP × DP × CP × EP × TP） | TP 切分 Dense | `mesh["dp", "cp", "ep"]._flatten()` | `None` | `1.0` |
| 5D（PP × DP × CP × EP × TP） | 非 TP 复制参数 | `mesh["dp", "cp", "ep", "tp"]._flatten()` | `None` | `1.0` |

!!! note "张量并行（TP）仍保持自动识别"
    对 TP 切分的 `DTensor` 参数，不要把 `"tp"` 轴放入 `shard_mesh` 或 `replicate_mesh`。DMuon 会按每个域自身的 `dp_dim_names` 自动与 `DTensor` 的维度名做差集识别 TP 轴。

---

## 多域梯度裁剪 {#multi-domain-clipping}

由于每个逻辑参数元素在全拓扑（DP/HSDP/CP/EP/PP/TP）内都满足唯一 owner（或互不相交的 `sharded_adamw` 分片）不变量，全局或分算法裁剪只需跨 `norm_reduce_group`（默认 `WORLD`）做**一次 `all_reduce(SUM)`**：

```python
# 全局 L2 裁剪（覆盖所有域的 Muon + AdamW + sharded_adamw 梯度）
result = dmuon.clip_grad_norm_(model, max_norm=1.0, mode="global")

# 分算法独立 L2 裁剪（Muon 与 AdamW/sharded_adamw 各自独立计算范数与裁剪系数）
result = dmuon.clip_grad_norm_(model, max_norm=1.0, mode="per_algorithm")
muon_norm = result.stats_by_name["muon"].total_norm
adamw_norm = result.stats_by_name["adamw"].total_norm
```

核心特性：

- 读取梯度前自动调用 `wait_all_reduces(model)`，确保归约与 `grad_scale` 已就绪；
- 对 `sharded_adamw` 仅切片 `[:_sharded_adamw_valid_numel]`，绝不把 padding 计入范数；
- 全卡保持固定 bucket 结构，即使某些 rank / PP stage 某一算法下有 0 个本地梯度（例如仅含 Embedding 的 Stage 0 无 Muon 参数）也能严格 lockstep 无死锁；
- 传入 `dmuon.Muon` 优化器且 `mode=None` 时，完全保留原有仅裁剪 Muon 梯度的行为。

---

## 多域与 PP × EP 检查点 {#multi-domain-checkpoint}

`dmuon.get_model_state_dict`、`dmuon.get_optimizer_state_dict`、`dmuon.set_model_state_dict` 与 `dmuon.set_optimizer_state_dict` 仅在每个参数所属的 `(dp_group, replicate_group[, tp_group])` 内通信，**不包含任何隐式 `WORLD` collective**：

```python
# 每个 PP stage / EP lane 导出本卡持有的完整模块参数与优化器状态（适合外部 DCP 聚合写盘）
model_sd = dmuon.get_model_state_dict(model, cpu_offload=True, rank0_only=False)
optim_sd = dmuon.get_optimizer_state_dict(model, optimizer, cpu_offload=True, rank0_only=False)

# 恢复
dmuon.set_model_state_dict(model, model_sd)
dmuon.set_optimizer_state_dict(model, optimizer, optim_sd)
```

- 参数数量不同的 PP stage（如 Stage 0 仅含 Dense 层，Stage 1 含 Dense + MoE 层）可安全导出与恢复，不会发生跨 stage 死锁；
- 尚未收到梯度的“冷专家”不会出现在 `optim_sd["dedicated"]` / `optim_sd["dedicated_adamw"]` 中，也不会广播未初始化的占位张量。

---

## 注意事项 {#gotchas}

1. **使用默认 `process_group_policy="isolated"` 时，所有 PP stage 需传入一致的 `domains` key 集合**：即使 Stage 0 在 `"expert"` 域匹配到 0 个参数，所有 stage 仍按相同字典序建隔离 PG，防止 `WORLD` 上建组错序。
2. **不要同时传 `mesh` 和 `domains`**：两者互斥，同时传入会立即抛 `ValueError`。
3. **不要再在外部手动对专家梯度做 `div_(ep_size)`**：统一在 `"expert"` 的 `OwnershipDomain` 上声明 `grad_scale=1.0 / ep_size`，即可自动覆盖 Muon、`MatrixLayout`、AdamW、`sharded_adamw` 以及 `no_sync` 梯度累积。
