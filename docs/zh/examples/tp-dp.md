# 示例：TP + DP 训练

一个在 2D 设备网格上使用张量并行（TP）和数据并行（DP）的完整示例。

---

## 脚本

::: details 完整源码：`examples/tp_dp.py`

```python
--8<-- "examples/tp_dp.py"
```

:::

## 逐步讲解

### 2D 网格设置

```python
# 4 张 GPU：2 个 DP rank x 2 个 TP rank
mesh_2d = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
dp_mesh = mesh_2d["dp"]
tp_mesh = mesh_2d["tp"]
```

网格是 2D 的：DP 维度用于数据并行，TP 维度用于张量并行。

### 先应用 TP

```python
for layer in model.layers:
    parallelize_module(
        layer.attn, tp_mesh,
        {
            "q_proj": ColwiseParallel(),   # Shard(0) — 行分片
            "k_proj": ColwiseParallel(),   # Shard(0)
            "v_proj": ColwiseParallel(),   # Shard(0)
            "o_proj": RowwiseParallel(),   # Shard(1) — 列分片
        },
    )
    parallelize_module(
        layer.mlp, tp_mesh,
        {
            "gate_proj": ColwiseParallel(),
            "up_proj": ColwiseParallel(),
            "down_proj": RowwiseParallel(),
        },
    )
```

TP 必须在 DMuon 和 FSDP2 **之前**应用。

### 然后 DMuon + FSDP2

```python
# DMuon 使用 dp_mesh（数据并行维度）
dmuon.dedicate_params(model, dp_mesh, predicate=lambda n, p: "proj" in n and p.ndim == 2)

# FSDP2 同样使用 dp_mesh
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)
```

### 带 TP 选项的优化器

```python
optimizer = dmuon.Muon(
    model, lr=0.02,
    per_head_ns=True,         # GQA k/v_proj 零 TP 通信（默认）
    block_diagonal_ns=False,  # 设为 True 则所有参数零 TP 通信（实验性）
    adamw_lr=1e-3,
)
```

优化器自动检测 TP 分片参数并路由到对应的 NS 变体。

## 内部机制

对于每个专属参数，优化器检查：

1. **是否为带 TP 组的 DTensor？** 否 → 本地 `newton_schulz()`
2. **是否为窄 Shard(0)？**（如 GQA k/v_proj，full_m < full_n） → 逐头 `newton_schulz()`，零 TP 通信
3. **`block_diagonal_ns=True`？** → `gram_newton_schulz(..., block_diagonal=True)`，零 TP 通信
4. **其他** → `gram_newton_schulz(..., shard_dim=dp.shard_dim)`，精确 Gram + TP all-reduce

## 运行

```bash
# 需要 4 张 GPU（2 DP x 2 TP）
torchrun --nproc_per_node=4 examples/tp_dp.py
```
