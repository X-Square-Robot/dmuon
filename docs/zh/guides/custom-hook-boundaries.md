# 自定义 Hook 边界

!!! tip "TL;DR"
    标准 LLM 用默认的 `layers.N` / `blocks.N` 启发式就够了。对于 VLA、MoE、或嵌套
    多模态模型，通过 `hook_boundary_predicate` 告诉 DMuon 把 hook 注册在哪个模块上。
    保持 `hook_boundary_strict=True`（默认），让配置错误在启动时就暴露出来。

---

## 为什么需要自定义

DMuon 把 forward/backward 的 broadcast 和 reduce hook 注册在**层级**上，而不是每个子模块。
把同一层的所有 hook 合并到单个模块，可以降低 CPU launch 开销，并启用 packed broadcast。

对于标准 transformer，内置启发式读取参数的 fully-qualified name，通过
`partition.py::_extract_layer_id` 提取第一个 `layers.N` 或 `blocks.N` 段。对于
`model.layers.3.self_attn.q_proj.weight` 这样的路径，效果很好。

**以下情况会静默失败：**

- **VLA / 多模态模型** —— 视觉塔用 `blocks.N`，解码器用 `layers.N`，可能混淆边界或退化
  为每个 `Linear` 各自一个 hook。
- **MoE 模型** —— expert 常命名为 `layers.N.mlp.experts.K`，启发式把 router 和所有
  expert 归到同一个 `layers.N` 边界。
- **嵌套结构** —— Qwen-VL 风格路径如 `model.vlm.llm.model.layers.3.mlp.gate_proj`，
  外层如果再加 `layers` 级，启发式可能找到错误的层。
- **per-Linear 兜底** —— 找不到匹配时，退到直接父 `nn.Linear`，每个 projection 各自一个 hook。

以上问题都不报错，只悄悄降低性能。

---

## API

```python
dmuon.dedicate_params(
    model,
    mesh,
    predicate=lambda n, p: p.ndim == 2 and "proj" in n,
    hook_boundary_predicate=lambda m: isinstance(m, MyLayerClass),
    hook_boundary_strict=True,   # 默认 — 推荐
)
```

**`hook_boundary_predicate`** 是一个 `(module) -> bool` 的 callable。DMuon 调用内部的
`_find_hook_module`，从每个 dedicated 参数的祖先中**从下往上**走，返回第一个满足谓词的祖先。

**`hook_boundary_strict`**（默认 `True`）在任何 dedicated 参数没有满足谓词的祖先时在
`dedicate_params` 里抛出 `ValueError`，让配置错误立刻暴露。仅原型探索时考虑设为 `False`。

`hook_boundary_predicate` 只影响 hook 注册位置，与 LPT 均衡分配（由 `predicate` 控制）
完全独立。

---

## 示例

### 示例 1：VLA 模型（视觉塔 + 解码器层 + 动作头）

基于 `tests/unit/test_hook_boundary.py` 中的 `ToyVLA` 结构。24 个 ViT block 收敛到
单个 `model.visual` 站点；每个解码层和动作头各自成为独立站点。

```python title="vla_setup.py"
import torch.nn as nn
import dmuon
from torch.distributed.fsdp import fully_shard

class VisionTower(nn.Module):
    def __init__(self, d=1024, n=24):
        super().__init__()
        self.blocks = nn.ModuleList([VitBlock(d) for _ in range(n)])

class DecoderLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, 4 * d, bias=False)
        self.down_proj = nn.Linear(4 * d, d, bias=False)

class ActionHead(nn.Module):
    def __init__(self, d, n_actions=7):
        super().__init__()
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, n_actions, bias=False)

class ToyVLA(nn.Module):
    def __init__(self, d=1024, n_vit=24, n_dec=28):
        super().__init__()
        self.visual = VisionTower(d, n_vit)
        self.layers = nn.ModuleList([DecoderLayer(d) for _ in range(n_dec)])
        self.action_head = ActionHead(d)

model = ToyVLA().cuda()

def boundary(m):
    return isinstance(m, (VisionTower, DecoderLayer, ActionHead))

dmuon.dedicate_params(
    model, mesh,
    predicate=lambda n, p: p.ndim == 2,
    hook_boundary_predicate=boundary,
    hook_boundary_strict=True,
)
for layer in model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model.visual, mesh=mesh)
fully_shard(model.action_head, mesh=mesh)
fully_shard(model, mesh=mesh)
optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)
```

---

### 示例 2：MoE 模型

每个 expert 是独立 hook 站点；router 留在外层 `MoELayer`。router 通过 `predicate` 排除
在 dedicated ownership 之外。

```python title="moe_setup.py"
import torch.nn as nn
import dmuon
from torch.distributed.fsdp import fully_shard

class Expert(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

class MoELayer(nn.Module):
    def __init__(self, d, d_ff, n_experts=8):
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d, d_ff) for _ in range(n_experts)])
        self.o_proj = nn.Linear(d, d, bias=False)

model = MoEModel().cuda()  # MoEModel 包含 MoELayer × n_layers

dmuon.dedicate_params(
    model, mesh,
    predicate=lambda n, p: p.ndim == 2 and "router" not in n,
    hook_boundary_predicate=lambda m: isinstance(m, (Expert, MoELayer)),
    hook_boundary_strict=True,
)
for layer in model.layers:
    for expert in layer.experts:
        fully_shard(expert, mesh=mesh)
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)
optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)
```

---

### 示例 3：嵌套 Qwen-VL 风格模型

```python title="qwenvl_setup.py"
import torch
import dmuon
from torch.distributed.fsdp import fully_shard
from transformers import Qwen2VLForConditionalGeneration

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16,
).cuda()

def boundary(m):
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VLDecoderLayer, Qwen2VLVisionBlock,
    )
    return isinstance(m, (Qwen2VLDecoderLayer, Qwen2VLVisionBlock))

dmuon.dedicate_params(
    model, mesh,
    predicate=lambda n, p: p.ndim == 2 and any(
        k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj")
    ),
    hook_boundary_predicate=boundary,
    hook_boundary_strict=True,
)
for layer in model.model.layers:
    fully_shard(layer, mesh=mesh)
for block in model.visual.blocks:
    fully_shard(block, mesh=mesh)
fully_shard(model, mesh=mesh)
optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)
```

---

### 示例 4：nn.Sequential —— 不需要 hook_boundary_predicate 的情况

扁平的 `nn.Sequential` 没有逻辑分组，默认启发式已给出 per-`Linear` hook，这就是正确
行为，不必强行添加谓词。

```python title="sequential_ok.py"
import torch.nn as nn
import dmuon
from torch.distributed.fsdp import fully_shard

model = nn.Sequential(
    nn.Linear(1024, 4096, bias=False),
    nn.GELU(),
    nn.Linear(4096, 1024, bias=False),
).cuda()

# 不需要 hook_boundary_predicate，默认 per-Linear hook 就是最优的
dmuon.dedicate_params(model, mesh, predicate=lambda n, p: p.ndim == 2)
fully_shard(model, mesh=mesh)
optimizer = dmuon.Muon(model, lr=0.02, adamw_lr=1e-3)
```

强行写 `lambda m: isinstance(m, nn.Sequential)` 会把所有参数压到根，消除 layer 级别流水线。

---

## 与 `fully_shard` 边界对齐

谓词定义一次，`hook_boundary_predicate` 和 `fully_shard` 复用同一个。对齐边界让 forward
顺序更可预测，prefetch 流水线效果更好。

```python title="aligned_boundaries.py"
import dmuon
from torch.distributed.fsdp import fully_shard

def is_transformer_layer(m):
    return isinstance(m, TransformerLayer)

dmuon.dedicate_params(
    model, mesh, predicate=lambda n, p: p.ndim == 2,
    hook_boundary_predicate=is_transformer_layer, hook_boundary_strict=True,
)
for module in model.modules():
    if is_transformer_layer(module):
        fully_shard(module, mesh=mesh)
fully_shard(model, mesh=mesh)
```

---

## `strict=True` vs `strict=False`

`strict=True` 在有 dedicated 参数没有匹配祖先时立刻报错——捕捉拼写错误、类未导入、层类型
遗漏等问题。`strict=False` 静默退回直接父模块，仅探索阶段排查时使用。生产代码始终用
`strict=True`。

---

## 常见陷阱

**谓词过窄**——strict 报错，lenient 静默退化为 per-`Linear` hook。扩展谓词到所有包含
dedicated 参数的模块类型。

**谓词过宽**——`lambda m: isinstance(m, nn.Module)` 匹配所有模块含根，所有参数压到根，
layer 级别流水线失效。谓词应指向叶级或中间层逻辑单元。

**谓词有副作用**——`_find_hook_module` 对每个参数的每个祖先调一次谓词，不要在谓词里修改
状态或做昂贵 I/O。

**MoE 权重共享的 expert**——两个层引用同一对象时，两组参数映射到同一 hook 站点。参数按
identity 去重，但请提前核实分配结果。

---

## 相关文档

- [HSDP 训练](hsdp.md) —— HSDP 下的 hook 边界注意事项
- [Z2 与 Z3 模式](z2-z3-modes.md) —— `reshard_after_forward` 与 hook 粒度的交互
- [架构](../design/architecture.md) —— hook 如何与 FSDP2 组合
- [API 文档](../reference/api.md) —— 完整 `dedicate_params` 签名
