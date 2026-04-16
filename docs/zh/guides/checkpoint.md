# 检查点

DMuon 提供检查点函数，在单一统一的 state dict 中处理专属参数（DMuon 管理）和对称参数（FSDP2 管理）。

---

## 为什么需要特殊处理？

专属参数仅存储在所有者 rank 上——`model.state_dict()` 在非所有者 rank 上只能看到空占位符。DMuon 的检查点函数会收集所有参数，生成一个**标准 state dict**，兼容单卡加载和 HuggingFace。

## 保存

```python
import torch
import torch.distributed as dist
import dmuon

# 收集 state dict（所有 rank 必须调用）
model_sd = dmuon.get_model_state_dict(model)
optim_sd = dmuon.get_optimizer_state_dict(model, optimizer)

# 仅 rank 0 写入磁盘
if dist.get_rank() == 0:
    torch.save({"model": model_sd, "optim": optim_sd}, "checkpoint.pt")
dist.barrier()
```

!!! info "所有 rank 必须调用"
    `get_model_state_dict()` 和 `get_optimizer_state_dict()` 是集合操作——所有 rank 必须调用，即使只有 rank 0 保存结果。

## 加载（恢复训练）

```python
# 所有 rank 加载检查点
ckpt = torch.load("checkpoint.pt", map_location="cpu")

dmuon.set_model_state_dict(model, ckpt["model"])
dmuon.set_optimizer_state_dict(model, optimizer, ckpt["optim"])
```

恢复内容包括：

- **模型权重** — 专属参数和 FSDP2 管理的参数
- **优化器状态** — 动量缓冲区（Muon）和 Adam 矩（AdamW）
- **步数计数器** — 用于正确的 AdamW 偏差校正

## 加载预训练模型（无优化器状态）

加载预训练模型（无优化器状态）方法相同：

```python
pretrained_sd = torch.load("pretrained_model.pt", map_location="cpu")
dmuon.set_model_state_dict(model, pretrained_sd)
```

兼容：

- 单卡 `torch.save(model.state_dict(), ...)` 检查点
- HuggingFace `model.save_pretrained()` 检查点（safetensors 或 bin 格式）

## 完整示例

```python
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
import dmuon

def setup_model(mesh):
    """构建并包装模型。"""
    model = MyModel().cuda()
    dmuon.dedicate_params(model, mesh, predicate=lambda n, p: "proj" in n and p.ndim == 2)
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)
    return model, optimizer

def save_checkpoint(model, optimizer, step, path="checkpoint.pt"):
    model_sd = dmuon.get_model_state_dict(model)
    optim_sd = dmuon.get_optimizer_state_dict(model, optimizer)
    if dist.get_rank() == 0:
        torch.save({"model": model_sd, "optim": optim_sd, "step": step}, path)
    dist.barrier()

def load_checkpoint(model, optimizer, path="checkpoint.pt"):
    ckpt = torch.load(path, map_location="cpu")
    dmuon.set_model_state_dict(model, ckpt["model"])
    dmuon.set_optimizer_state_dict(model, optimizer, ckpt["optim"])
    return ckpt.get("step", 0)

# --- 主程序 ---
dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (dist.get_world_size(),))
model, optimizer = setup_model(mesh)

# 若检查点存在则恢复
start_step = 0
if os.path.exists("checkpoint.pt"):
    start_step = load_checkpoint(model, optimizer)

for step in range(start_step, total_steps):
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()
    optimizer.step()

    if (step + 1) % save_interval == 0:
        save_checkpoint(model, optimizer, step + 1)
```

## State Dict 格式

### 模型 State Dict

模型 state dict 使用**标准 PyTorch 格式**——将全限定参数名映射到张量的扁平字典：

```python
{
    "layers.0.self_attn.q_proj.weight": tensor(...),
    "layers.0.self_attn.k_proj.weight": tensor(...),
    "layers.0.ln.weight": tensor(...),
    ...
}
```

### 优化器 State Dict

优化器 state dict 使用 DMuon 特有格式，包含分离的段：

```python
{
    "fsdp2": { ... },          # FSDP2 参数状态（Adam 矩）
    "dedicated": {              # 专属参数状态（动量缓冲区）
        "layers.0.self_attn.q_proj.weight": {
            "momentum_buffer": tensor(...)
        },
        ...
    }
}
```
