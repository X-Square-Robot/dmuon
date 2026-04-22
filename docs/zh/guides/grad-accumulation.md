# 梯度累积

!!! tip "TL;DR"
    用 `dmuon.no_sync(model)` 包裹累积的微批次，同时抑制专属参数和对称参数的梯度通信。仅在最后一个微批次调用 `optimizer.step()` + `optimizer.zero_grad()`。在 `.backward()` 前将 loss 除以 `accum_steps`，使累积梯度等价于单次大批次梯度。

---

## 基本用法

```python
from contextlib import nullcontext
import dmuon

accum_steps = 4

for i, batch in enumerate(dataloader):
    # 累积步骤跳过 reduce
    is_accumulating = (i + 1) % accum_steps != 0
    ctx = dmuon.no_sync(model) if is_accumulating else nullcontext()

    with ctx:
        loss = model(batch).loss / accum_steps
        loss.backward()

    # 累积完所有微批次后再 step
    if not is_accumulating:
        optimizer.step()
        optimizer.zero_grad()
```

## 工作原理

`dmuon.no_sync(model)` 同时禁用**两类**参数的梯度通信：

- **专属参数**：跳过 reduce 到所有者。梯度在每个 rank 本地累积到 `_accumulated_grad`。
- **对称参数**：调用 `model.set_requires_gradient_sync(False)` 跳过 FSDP2 的 reduce-scatter。

在 `no_sync()` **外部**的下一次 backward 时：

- **专属参数**：累积的梯度与新梯度合并后再 reduce。
- **对称参数**：FSDP2 自动处理累积梯度。

`optimizer.step()` 后调用 `optimizer.zero_grad()` 会清除 `_reduced_grad` 和 `_accumulated_grad`。

## 完整训练循环示例

```python
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
import dmuon

dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (dist.get_world_size(),))
model = MyModel().cuda()

dmuon.dedicate_params(model, mesh, predicate=lambda n, p: "proj" in n and p.ndim == 2)
for layer in model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)

accum_steps = 4
global_step = 0

for i, batch in enumerate(dataloader):
    is_accumulating = (i + 1) % accum_steps != 0
    ctx = dmuon.no_sync(model) if is_accumulating else nullcontext()

    with ctx:
        loss = model(batch).loss / accum_steps
        loss.backward()

    if not is_accumulating:
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

        if dist.get_rank() == 0:
            print(f"step {global_step}: loss={loss.item() * accum_steps:.4f}")
```

!!! tip "Loss 缩放"
    在 `.backward()` 前将 loss 除以 `accum_steps`，这样累积后的梯度等价于单次大批次的梯度。

!!! note "DMuon-Z2 与梯度累积"
    注意 `reshard_after_forward=False`（DMuon-Z2）会在微批次之间保持 packed buffer 常驻，与梯度累积有交互——显存影响详见 [Z2 与 Z3 模式](z2-z3-modes.md)。

## 相关文档

- [训练流程](training.md) —— 完整 DMuon 训练流程
- [Z2 与 Z3 模式](z2-z3-modes.md) —— packed buffer 生命周期及其对累积时显存的影响
