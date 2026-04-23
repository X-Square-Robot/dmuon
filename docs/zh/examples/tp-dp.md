# 示例：TP + DP 训练

一个在 2D 设备网格上把张量并行（TP）和数据并行（DP）组合起来的完整
示例。DMuon 通过 `DTensor` **自动检测** TP-sharded 参数——用户无需
传任何 TP-specific 参数。

---

## 脚本

::: details 完整源码：`examples/tp_dp.py`

```python
--8<-- "examples/tp_dp.py"
```

:::

## 逐步讲解

### 2D mesh 设置

```python
# 4 张 GPU：2 DP × 2 TP
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
```

**必须传 `mesh_dim_names`**——DMuon 通过名称集合差识别 TP 轴
（`DTensor.mesh_dim_names − dp_mesh_dim_names`）。

### 调用顺序：TP → DMuon → FSDP2

```python
# 1. TP
for layer in model.layers:
    parallelize_module(
        layer.attn, mesh["tp"],
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

# 2. DMuon — 只传 DP 切片
dmuon.dedicate_params(
    model, mesh["dp"],
    predicate=lambda n, p: "proj" in n and p.ndim == 2,
)

# 3. FSDP2 — 同样是 DP 切片
for layer in model.layers:
    fully_shard(layer, mesh=mesh["dp"])
fully_shard(model, mesh=mesh["dp"])
```

DMuon 必须在 `fully_shard` **之前**调用，这样它的参数才能 opt out 于
FSDP2 的分片契约。

### 优化器

```python
optimizer = dmuon.Muon(model, lr=0.02, momentum=0.95, adamw_lr=1e-3)
```

**没有 TP-specific 参数。** 唯一和 TP 通信相关的是 `replicate_async`
（默认 `True`）：post-step 的 scatter + replicate broadcast 是否要和
下一 iter 的 forward compute 并发。sync 与 async 两个路径在 3D HSDP×TP
下产生 **bit-identical** 的 loss 轨迹。

## 内部流程

对每个被 `dedicate_params` 选中的参数，DMuon 的 step 执行：

1. **DP reduce** → 把梯度归约到 DP-owner rank（和非 TP 路径完全一致）。
2. **TP gather**（仅对 TP-sharded `DTensor` 参数，在 `reduce_stream`
   上运行，和 backward compute 天然并行）→ 把 `(m, n)` 完整梯度汇聚
   到 TP owner。
3. **Newton-Schulz** → 在 TP owner 上对完整矩阵跑 NS，和非 TP 路径
   走同一个 NS kernel。
4. **TP scatter**（on `replicate_broadcast_stream`）→ 把更新的每个分片
   发回各 DP-owner rank。
5. **Replicate broadcast** → 标准 HSDP 扩散到 replicate peer。

普通 `DTensor`（只在 DP 轴分片）和 `torch.Tensor` 参数**跳过步骤 2/4**。

## 运行

```bash
# 需要 4 张 GPU（2 DP × 2 TP）
torchrun --nproc_per_node=4 examples/tp_dp.py
```

参考：[TP 支持指南](../guides/tp-support.md) — 更深入地讲解 All-to-All
流水线、3D HSDP×TP mesh 配置、sync / async 语义、以及检查 API。
