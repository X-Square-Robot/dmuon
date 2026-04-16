# 示例：数据并行训练

一个完整、可运行的 DMuon 纯数据并行（无 TP）示例。

---

## 脚本

::: details 完整源码：`examples/basic_dp.py`

```python
--8<-- "examples/basic_dp.py"
```

:::

## 逐步讲解

### 模型定义

```python
class Block(nn.Module):
    def __init__(self, d=512, ff=2048):
        super().__init__()
        self.gate_proj = nn.Linear(d, ff, bias=False)   # 2D → Muon
        self.up_proj = nn.Linear(d, ff, bias=False)      # 2D → Muon
        self.down_proj = nn.Linear(ff, d, bias=False)    # 2D → Muon
        self.ln = nn.LayerNorm(d)                        # 1D → AdamW

    def forward(self, x):
        h = self.ln(x)
        return x + self.down_proj(self.gate_proj(h) * self.up_proj(h))
```

一个简单的 SwiGLU MLP 块。投影层是 2D 矩阵（适合 Muon），LayerNorm 是 1D（使用 AdamW）。

### DMuon 设置

```python
# Predicate：2D 投影层 → 专属（Muon）
dmuon.dedicate_params(
    model, mesh,
    predicate=lambda n, p: "proj" in n and p.ndim == 2,
)

# FSDP2：在各 rank 间分片 LayerNorm 和 head
for layer in model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

# 单一优化器同时处理两类参数
optimizer = dmuon.Muon(model, lr=0.02, momentum=0.95, adamw_lr=1e-3)
```

### 训练循环

```python
for step in range(100):
    optimizer.zero_grad()
    x = torch.randn(batch_size, d_model, device="cuda")
    loss = model(x)
    loss.backward()
    optimizer.step()
```

与标准 PyTorch 训练循环完全相同。

### 检查点

```python
# 保存
model_sd = dmuon.get_model_state_dict(model)
optim_sd = dmuon.get_optimizer_state_dict(model, optimizer)
if dist.get_rank() == 0:
    torch.save({"model": model_sd, "optim": optim_sd, "step": step}, path)

# 加载
ckpt = torch.load(path, map_location="cpu")
dmuon.set_model_state_dict(model, ckpt["model"])
dmuon.set_optimizer_state_dict(model, optimizer, ckpt["optim"])
```

## 运行

```bash
torchrun --nproc_per_node=4 examples/basic_dp.py
```
