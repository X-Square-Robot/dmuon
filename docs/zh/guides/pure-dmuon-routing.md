# Pure DMuon 路由

Pure DMuon 指所有可训练参数都可以进入 DMuon 的 ownership runtime，而不是把非 Muon 参数留在普通 FSDP2/AdamW 路径上。这个模式适合训练栈希望由同一个 runtime 控制全模型的参数放置、梯度通信和优化器 step。关键是优化器数学和通信放置是两件不同的事。

`predicate` 决定一个参数是否由 DMuon 管理。`param_policy` 决定 DMuon 如何处理这个被管理的参数。`route` 选择优化器数学和通信放置：

| Route | 优化器数学 | 前向参数通信 | 反向梯度通信 | 典型参数 |
|-------|------------|--------------|--------------|----------|
| `"muon"` | Muon + Newton-Schulz | owner `broadcast` | `reduce` 到 owner | attention 和 MLP 投影矩阵 |
| `"adamw"` | AdamW | owner `broadcast` | `reduce` 到 owner | LayerNorm、bias、小 AdamW 参数 |
| `"sharded_adamw"` | AdamW | 从参数 shard `all_gather` | `reduce_scatter` 到优化器 shard | embedding、`lm_head` 等很大的 AdamW 参数 |

`"adamw"` 和 `"sharded_adamw"` 都运行 AdamW。它们只是在参数和梯度如何分布上不同。小 AdamW 参数通常应该使用 `"adamw"`：一个 owner 更新参数，再通过 broadcast 发布更新后的值。这个更新语义和 sharded AdamW 等价，但不会在每个 transformer block 内产生大量小 all-gather 和 reduce-scatter collectives。

末端的大 AdamW 参数不同。输入 embedding 在 forward 开头被读取，`lm_head` 这类输出头的梯度通常出现在 backward 尾部。如果这些大张量由单个 owner 通过 broadcast 和 reduce 处理，这个 owner 容易成为 forward 开头或 backward 末尾的短板。把这些张量路由到 `"sharded_adamw"` 可以让 forward all-gather 和 backward reduce-scatter 由所有 rank 分担。

## 控制入口

策略控制分两步。第一步，`predicate` 决定哪些参数进入 DMuon runtime。
Pure DMuon 通常使用一个很宽的 predicate：

```python
predicate=lambda name, param: param.requires_grad
```

返回 `False` 的可训练参数会留在 DMuon 外部，必须由外层 FSDP/DDP runtime
处理。返回 `True` 的可训练参数会被 DMuon placeholder 替换，并继续从
`param_policy` 获得结构化策略。

第二步，`param_policy` 从 `defaults` 开始，按顺序应用命中的 `overrides`。
每个 override 的 `name` token 匹配完整参数名，并且只覆盖 `set` 中声明的字段。
`contains` 仍作为旧别名保留，但新的集成建议使用 `name`。主要 route 是：

- 对需要 Muon 更新的投影矩阵返回 `"muon"`。
- 对 LayerNorm weight、bias、其它 scalar/vector 等小 AdamW 参数返回
  `"adamw"`。这些参数走 DMuon owner `broadcast` 和 `reduce` 路径。
- 只对需要所有 rank 分担通信的大 AdamW 张量返回 `"sharded_adamw"`，
  通常是输入 embedding 和 `lm_head`。

对于任何已经被 `predicate` 纳入 DMuon 的可训练参数，都要确保最终 `route`
是这三个字符串之一。缺省 route 会被归一化成默认 `"muon"` route，所以如果
LayerNorm 或 bias 参数没有命中 AdamW override，它会进入错误的优化器路径。

最终策略会在 `dedicate_params()` 仍然能访问原始 full parameter tensor
时被消费。`"sharded_adamw"` 尤其依赖这个时机，因为 DMuon 必须在参数被
placeholder 替换之前构造每个 rank 的 shard storage。Optimizer param
groups 后续可以给 dedicated 参数分组和设置超参，但不能在
`dedicate_params()` 结束后再创建 `"sharded_adamw"` storage。

Optimizer `param_groups` 默认也不会覆盖这些逐参数 route hint。同一个语义组
可以同时包含 Muon 矩阵和 AdamW-route 的小参数；DMuon 会把这个语义组拆成
route-specific optimizer subgroups，同时保留 `param_policy` 选出的 route。
只有当你确实想进行整组 route override 时，才在用户组上设置 `dmuon_route`、
`dmuon_optimizer` 或 `matrix_optimizer`。

dtype 字段沿用 FSDP2 mixed precision 的语义。`param_dtype` 是 forward/backward
compute 使用的 materialized parameter dtype，也是 forward 参数通信 dtype。
`grad_dtype` 是 DMuon 梯度 buffer 和梯度通信 dtype。`output_dtype` 在模块边界
cast 浮点输出，`cast_forward_inputs=True` 会把浮点输入 cast 到 `param_dtype`。
`master_dtype` 和 `optim_dtype` 描述 DMuon storage 和 optimizer update 精度，
与 forward compute dtype 分开。

## 路由策略

一个实用的 LLM 策略是：

- 默认让适合矩阵优化的参数走 `"muon"`。
- embedding 和 `lm_head` 权重返回 `"sharded_adamw"`。
- LayerNorm、bias、小 AdamW 参数返回 `"adamw"`。

```python
dmuon.dedicate_params(
    model,
    mesh,
    predicate=lambda name, param: param.requires_grad,
    param_policy={
        "defaults": {
            "route": "muon",
            "param_dtype": torch.bfloat16,
            "grad_dtype": None,
            "master_dtype": torch.float32,
            "optim_dtype": torch.float32,
        },
        "overrides": [
            {
                "name": ["embed_tokens", "word_embeddings", "wte", "lm_head"],
                "set": {"route": "sharded_adamw"},
            },
            {
                "name": ["norm", "ln_", ".bias"],
                "set": {"route": "adamw"},
            },
            {
                "name": ["action_head", "action_decoder"],
                "set": {
                    "param_dtype": torch.float32,
                    "grad_dtype": torch.float32,
                    "output_dtype": torch.float32,
                    "cast_forward_inputs": True,
                },
            },
        ],
    },
)
```

使用这个策略时，LayerNorm 参数不需要额外用 FSDP wrapper 特判来避开
all-gather/reduce-scatter。它们命中 `"adamw"` override，因此走 DMuon owner
broadcast/reduce 路径。未命中特殊 override 的投影矩阵保留默认 `"muon"` route。
`action_head` 下的参数可以保持 fp32 forward compute，同时模型其它部分继续 bf16
compute。

## Hook 边界

路由选择和 hook 放置是独立的。DMuon 仍然需要一个模块边界，在 forward 前准备 full parameter，并在 backward 后收集梯度。Decoder block 通常可以作为投影矩阵和 normalization 权重的边界。Embedding 和输出头经常在 decoder block 外部被调用，所以 pure DMuon 集成应该把这些模块加入 `hook_boundary_predicate`。

```python
terminal_module_ids = {
    id(module)
    for module in (
        getattr(model, "lm_head", None),
        getattr(getattr(model, "model", None), "embed_tokens", None),
    )
    if isinstance(module, (torch.nn.Embedding, torch.nn.Linear))
}


def hook_boundary(module):
    if id(module) in terminal_module_ids:
        return True
    return isinstance(module, TransformerDecoderLayer)
```

`hook_boundary_predicate` 控制 DMuon communication 挂在哪个模块边界。`param_policy` 控制该边界上每个参数使用哪种通信原语和 dtype policy。

Activation cast 是 hook-boundary 级别的操作。如果同一个 hook boundary
里存在多个不同的有效 `param_dtype`，并且任意 group 开启了
`cast_forward_inputs`，DMuon 会在 setup 阶段直接报错。需要把 action head
切成单独 hook boundary，或者设置 `cast_forward_inputs=False`，由模块内部显式
处理 activation cast。

## 验证

构造 optimizer 后，先检查 route split，再启动长任务：

```python
summary = dmuon.summarize_param_groups(model, optimizer)
print(summary)
```

预期结果是 Muon-owned 投影矩阵出现在 Muon route，LayerNorm 和 bias 参数出现在 AdamW route，只有很大的 embedding/head 张量出现在 sharded AdamW route。参数行也会显示 `param_dtype`、`grad_dtype`、`output_dtype` 和命中的 policy override index。如果几百个小 AdamW 参数出现在 sharded AdamW route，说明路由策略太宽。

`route_hint_fn` 仍然保留给旧的 route-only 集成。新的 Pure DMuon 集成应优先使用
`param_policy`，因为 route、参数 dtype、梯度 dtype 和模块边界 cast 会在
DMuon 替换原始参数前一起解析。
