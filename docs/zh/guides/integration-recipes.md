# 集成方案

!!! tip "TL;DR"
    DMuon 可以与 HuggingFace Trainer、torchtitan 以及自定义训练循环配合使用——在
    标准 FSDP2 包装后只需 3 行额外设置。在 `fully_shard` 之前调用
    `dmuon.dedicate_params`，然后用 `dmuon.Muon` 作为优化器，训练循环本身不需要修改。

---

## 设计原则

DMuon 通过两种机制集成：

1. **import 时 monkey-patch** —— `import dmuon` 会 patch `fully_shard`，让它自动跳过
   携带 `_dedicated_owner_rank` 属性的参数，无需修改 FSDP2 内部代码。
2. **forward/backward hook** —— `dedicate_params` 在选定的层级模块上注册 pre/post
   forward hook，在专用 CUDA stream 上发起 shard broadcast 和梯度 reduce。

只要训练循环按 `loss.backward()` + `optimizer.step()` 的顺序执行，DMuon 就能无缝接入。

---

## HuggingFace Transformers 与 Accelerate

```python title="hf_dmuon.py"
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM
import dmuon

dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
mesh = init_device_mesh("cuda", (dist.get_world_size(),))

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B", torch_dtype=torch.bfloat16,
).cuda()

# 第一步：在 fully_shard 之前标记 dedicated 参数
dmuon.dedicate_params(
    model, mesh,
    predicate=lambda n, p: p.ndim == 2 and any(
        k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj")
    ),
)
# 第二步：正常应用 FSDP2，dedicated 参数自动跳过
for layer in model.model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

# 第三步：用 dmuon.Muon 作为优化器
optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    outputs = model(**batch)
    outputs.loss.backward()
    optimizer.step()
```

### HuggingFace Trainer

通过 `optimizers` 参数传入自定义优化器：

```python title="hf_trainer_dmuon.py"
from transformers import Trainer, TrainingArguments
import dmuon

# ... 模型初始化、dedicate_params、fully_shard 同上 ...

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    fsdp="",           # 禁用 Trainer 自带的 FSDP 包装
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    optimizers=(optimizer, None),   # (optimizer, lr_scheduler)
)
trainer.train()
```

!!! warning "禁用 Trainer 内置的 FSDP 包装"
    已经手动调用过 `fully_shard` 时，在 `TrainingArguments` 里传 `fsdp=""` 来禁用
    Trainer 自带的 FSDP 包装。重复 `fully_shard` 会报错或产生错误行为。

对于 Qwen-VL 等嵌套多模态模型，在 `dedicate_params` 时传入 `hook_boundary_predicate`
（详见[自定义 Hook 边界](custom-hook-boundaries.md)），再调用 `fully_shard`。

---

## torchtitan

在 torchtitan 的 `parallelize_model` 之前调用 `dedicate_params`，DMuon 的 monkey-patch
会确保 `fully_shard` 自动跳过 dedicated 参数：

```python title="torchtitan_dmuon.py"
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import dmuon

mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
model = build_model(config)

# 在 torchtitan 包装模型之前应用 DMuon
dmuon.dedicate_params(
    model, mesh["dp"],
    predicate=lambda n, p: p.ndim == 2 and "proj" in n,
)
parallelize_model(model, mesh, config)   # torchtitan 在这里调用 fully_shard + TP

optimizer = dmuon.Muon(model, lr=config.lr, adamw_lr=config.adamw_lr)

for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

使用 HSDP 异步模式时，在 torchtitan 的 DCP checkpoint 保存前先调用
`dmuon.wait_all_replicate_broadcasts(model)`，确保 `_owned_data` 是最新的。
第一方集成（DMuon 作为 torchtitan 配置中的具名优化器）在路线图中，手动方式是目前
已支持的路径。

---

## DeepSpeed ZeRO

DMuon 当前实现针对 FSDP2。dedicated ownership 原语设计上对运行时可移植：被
`dedicate_params` 标记的参数携带 `_dedicated_owner_rank`，DeepSpeed adapter 可在
ZeRO 分区时跳过这些参数。

当前状态：

- **ZeRO-0 / ZeRO-1 / ZeRO-2**：原理上兼容；adapter 尚未实现。
- **ZeRO-3**：DeepSpeed ZeRO-3 的 bucket-based 存储与 DMuon packed-buffer broadcast
  集成需要 adapter 层面的工作。

**此集成在路线图中，尚未发布。** 今天请使用 FSDP2。

---

## 自定义训练循环

最简调用序列：

```python title="custom_loop.py"
import dmuon
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (world_size,))
model = MyModel().cuda()

dmuon.dedicate_params(
    model, mesh, predicate=lambda n, p: p.ndim == 2 and "proj" in n
)
for layer in model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch).loss
    loss.backward()
    # optimizer.step() 内部处理 wait_all_reduces + NS + AdamW +
    # broadcast_all_updates，无需手动调用
    optimizer.step()
```

特定场景下的手动控制：

```python title="manual_overrides.py"
import dmuon
from contextlib import nullcontext

# HSDP 异步模式下保存 checkpoint 前：drain pending broadcast
dmuon.wait_all_replicate_broadcasts(model)
model_sd = dmuon.get_model_state_dict(model)

# 梯度累积
for i, batch in enumerate(dataloader):
    ctx = dmuon.no_sync(model) if (i + 1) % accum_steps != 0 else nullcontext()
    with ctx:
        loss = model(batch).loss / accum_steps
        loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 跨框架的 checkpoint

DMuon 的 checkpoint API 与框架无关，详见[检查点](checkpoint.md)：

```python title="checkpoint_save.py"
import torch
import torch.distributed as dist
import dmuon

model_sd = dmuon.get_model_state_dict(model)      # 先 drain async broadcast
optim_sd = dmuon.get_optimizer_state_dict(model, optimizer)
if dist.get_rank() == 0:
    torch.save({"model": model_sd, "optim": optim_sd}, "checkpoint.pt")
dist.barrier()

ckpt = torch.load("checkpoint.pt", map_location="cpu")
dmuon.set_model_state_dict(model, ckpt["model"])
dmuon.set_optimizer_state_dict(model, optimizer, ckpt["optim"])
```

HuggingFace Trainer 可以添加 training callback，在每次保存前调用
`dmuon.wait_all_replicate_broadcasts(model)`，或重写 `_save_checkpoint` 改用
`dmuon.get_model_state_dict`。

---

## 相关文档

- [训练流程](training.md) —— 完整单机工作流
- [检查点](checkpoint.md) —— state-dict 语义与 HSDP 断点续训
- [HSDP 训练](hsdp.md) —— 多机 replicate mesh 配置
- [自定义 Hook 边界](custom-hook-boundaries.md) —— 嵌套模型结构的处理
