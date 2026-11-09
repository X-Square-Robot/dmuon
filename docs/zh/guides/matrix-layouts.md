# 矩阵布局 {#matrix-layouts}

<p class="dmuon-lead">通过 MatrixLayout 指定一个现有参数中的哪些二维矩阵分别接受 Muon 更新。模型的模块、参数名称和前向形状保持原样。</p>

`MatrixLayout` 通过 `param_policy` 或 `param_policy_fn` 配置，适用于融合
QKV、逐 head 更新、堆叠 MoE 专家和交错存储，不需要为每个模型编写专用适配器。

## 先记住一个规则：最后两维是一张矩阵 {#matrix-rule}

完成声明的布局操作后，**最后两个维度**构成一张 Newton–Schulz（NS）矩阵。
前面的维度用于枚举独立更新，每张矩阵有自己的动量状态和 owner 分配。

```python
from dmuon import MatrixLayout

# 原参数形状：[E, 2 * I, H]
layout = MatrixLayout.view(E, 2, I, H)
# 逻辑更新：2 * E 张独立矩阵，每张形状为 [I, H]
```

这是优化器的配置，不要求你在 `forward()` 中改写模型。
不同逻辑矩阵可以分配给不同 owner；同一个 rank 也可以承接多张矩阵，
具体分配由 DMuon 的 placement 策略决定。

| 配置 | 决定什么 |
|---|---|
| `route` | 被选中的参数使用 Muon、AdamW 还是 sharded AdamW |
| `hook_boundary_predicate` | 在哪个模块边界物化参数、组织通信 |
| `matrix_layout` | 哪些完整矩阵分别接受 Muon 更新 |

## 给模型配置布局 {#configure-model}

下面显式选择融合专家权重，其他受管理参数暂时使用 AdamW，便于看清布局配置。
实际训练时可继续为 attention projection 和大 embedding 添加策略，见
[Pure DMuon 路由](pure-dmuon-routing.md)。

假设原参数名为 `model.layers.N.mlp.experts.gate_up_proj` 和 `down_proj`。
请按模型的实际全限定名配置；`wrap_decoder_layer` 是训练代码中用于选择
decoder 层 Hook 边界的模块谓词。

```python
import torch
import dmuon
from dmuon import MatrixLayout

cfg = model.config
E = cfg.num_experts
I = cfg.moe_intermediate_size
H = cfg.hidden_size

dmuon.dedicate_params(
    model,
    mesh,
    predicate=lambda name, p: p.requires_grad,
    hook_boundary_predicate=wrap_decoder_layer,
    param_policy={
        "defaults": {
            "route": "adamw",
            "param_dtype": torch.bfloat16,
            "grad_dtype": torch.bfloat16,
            "master_dtype": torch.float32,
            "optim_dtype": torch.float32,
        },
        "overrides": [
            {
                "name_glob": "model.layers.*.mlp.experts.gate_up_proj",
                "route": "muon",
                "matrix_layout": MatrixLayout.view(E, 2, I, H),
            },
            {
                "name_glob": "model.layers.*.mlp.experts.down_proj",
                "route": "muon",
                "matrix_layout": MatrixLayout.view(E, H, I),
            },
        ],
    },
)

optimizer = dmuon.Muon(model, lr=0.002, adamw_lr=0.0003)
```

示例使用 FP32 主权重和优化器状态、BF16 计算与梯度通信。精度策略与矩阵布局
是两个独立选择。与 FSDP2 组合时，在 `fully_shard()` 之前调用 DMuon 配置。

`name_glob` 对完整参数名进行大小写敏感的 glob 匹配。override 按顺序应用，
后匹配的字段覆盖先前设置；运行时不会根据模型名推断布局。
原有 `name` / `contains` 选择器则使用子串匹配。

### 用回调读取实际维度 {#policy-callback}

不同层的专家维度不一致时，可以直接从参数形状生成配置。
使用 `param_policy_fn=expert_policy` 替代 `param_policy`：

```python
def expert_policy(name, p):
    if name.endswith(".experts.gate_up_proj"):
        experts, twice_i, hidden = p.shape
        return {
            "route": "muon",
            "matrix_layout": MatrixLayout.view(
                experts, 2, twice_i // 2, hidden
            ),
        }
    if name.endswith(".experts.down_proj"):
        return {"route": "muon", "matrix_layout": MatrixLayout.view(*p.shape)}
    return {"route": "adamw"}
```

这段回调只展示路由与布局，精度字段可一并放入返回值。参数名和各轴含义由调用方
明确指定；上例约定融合轴依次存放等长的 gate 和 up。

## 融合 QKV：一起更新、分别更新、逐 head 更新 {#qkv-updates}

对于原形状 `[3 * H, H]`、行顺序为全部 Q、全部 K、全部 V 的权重：

| 期望的更新方式 | 配置 | NS 矩阵 |
|---|---|---|
| 融合 QKV 一起更新 | `MatrixLayout.whole()` | 一张 `[3H, H]` |
| Q、K、V 分别更新 | `MatrixLayout.view(3, H, H)` | 三张 `[H, H]` |
| 每个分量的每个 head 分别更新 | `MatrixLayout.view(3, num_heads, head_dim, H)` | `3 * num_heads` 张 `[head_dim, H]` |
| Q/K/V 行数不同 | `MatrixLayout.rows(q_rows, k_rows, v_rows)` | 三张各自对应行数的矩阵 |

其中 `num_heads * head_dim == H`，行分区必须恰好覆盖整个参数。
融合 NS 与分别 NS 是不同的优化器选择；DMuon 让调用方决定是否拆分。

MoE 同理：`view(E, 2, I, H)` 在每个专家内分别更新 gate/up；
`view(E, 2 * I, H)` 则保留每个专家的 gate/up 融合更新。
希望逐专家更新时不要使用 `whole()`：对于三维参数，它沿用
`[shape[0], product(shape[1:])]` 的整体矩阵解释，并不表示逐专家更新。

## 交错存储：声明元素的真实顺序 {#interleaved}

仅凭参数形状，无法判断 QKV 是按分量连续存放，还是按 head 交错存放。
布局应当描述权重的实际元素顺序。

<div class="dmuon-diagram" markdown>

![按 head 交错的物理 QKV 存储与三张独立逻辑矩阵](../assets/matrix-layout.svg)

</div>

假设 `[24, 8]` 的行顺序是 `Q0, K0, V0, Q1, K1, V1`，每一项占四行：

```python
# 分别更新 Q、K、V，每个分量合并两个 head。
layout = (
    MatrixLayout.view(2, 3, 4, 8)
    .permute(1, 0, 2, 3)
    .reshape(3, 8, 8)
)

# 或者，把六张 head/分量矩阵分别更新。
per_head = MatrixLayout.view(2, 3, 4, 8)
```

对这个物理形状，`view(3, 8, 8)` 也能通过编译，但选中的元素不符合上述 QKV
语义。编译器验证覆盖范围和几何形状；调用方仍需确认模型实际的元素排列。

### GQA 分组和按列融合 {#grouped-query-and-column-layouts}

假设有两个 KV group，每组依次存两个 Q head、一个 K head、一个 V head，
head dimension 为 2，输入宽度为 8：

```python
# 原参数形状：[16, 8]
gqa = (
    MatrixLayout.view(2, 4, 2, 8)
    .split((2, 1, 1), dim=1)
    .reshape(-1, 8)
)
# Q: [8, 8]；K: [4, 8]；V: [4, 8]

# QKV 按列融合，原参数形状：[8, 24]
column_fused = MatrixLayout.view(8, 3, 8).permute(1, 2, 0)
# 三张 [8, 8]，按优化器所需的输出/输入方向解释。
```

`split` 之后的操作分别作用于每个分支。矩阵按照 split 分支顺序，
再按各分支的前导维度枚举。这里的 view/reshape 保持逻辑元素顺序，
不承诺运行时一定能生成零拷贝的 Tensor view。

## 在训练前检查布局 {#inspect-layout}

```python
plan = layout.compile((24, 8))
print([matrix.shape for matrix in plan.matrices])
# [(8, 8), (8, 8), (8, 8)]
```

先用 `model.named_parameters()` 核对名称，再结合模型前向实现确认形状与元素
顺序。对填入 `arange` 的小张量，`plan.pack(tensor)` 可以直观看出每张逻辑矩阵
包含哪些元素，`plan.unpack(...)` 则将其还原。

在仓库根目录运行 CPU 示例：

```bash
PYTHONPATH=. python examples/matrix_layouts.py
```

## CUDA 搬运路径做了什么 {#cuda-copy}

布局编译器生成地址映射元数据，不增加模型模块，也不会为每个专家各发射一个
CUDA kernel。DMuon 在 wrapping 边界将兼容的转换与布局任务合并，在调用方的
stream 上发射批量搬运。

| 阶段 | 处理方式 |
|---|---|
| 主权重 copy-in | 从 owner 主权重直接转换到 DMuon 通信缓冲区；连续 FP32→BF16/FP16 使用向量读取和打包写入。 |
| 恢复原生布局 | 广播完成后，在 broadcast stream 上将交错或转置数据还原到模型缓冲区；可直接对应原生连续区间的矩阵使用缓冲区视图。 |
| 梯度准备 | 借用兼容的连续梯度，或直接 gather/转换到规约输入；owner 仍对完整的逻辑矩阵执行 Muon。 |

compute stream 在使用参数前等待 ready event。预取与布局还原沿用 broadcast
stream，可以与其他工作重叠，因此拷贝和通信 kernel 的累计时长不等于它们对
整步耗时的贡献。

### 编译与后端选择 {#build-and-backends}

在已安装 CUDA 版 PyTorch、并有 CUDA toolkit 的环境中编译：

```bash
DMUON_BUILD_LAYOUT_COPY=1 pip install -e . --no-build-isolation
```

`DMUON_BUILD_LAYOUT_COPY=0` 可以单独禁用此扩展。
另一个可选 fast-clip 扩展由 `DMUON_BUILD_FAST_CLIP` 独立控制。

| 设置 | 含义 |
|---|---|
| `DMUON_COPY_BACKEND=auto` | 默认；扩展可用且任务符合条件时使用 CUDA，否则使用 Torch 参考路径。 |
| `DMUON_COPY_BACKEND=torch` | 使用参考搬运路径，便于对照与调试。 |
| `DMUON_COPY_BACKEND=cuda` | 要求使用可选 CUDA 路径，不可用时明确报错。 |
| `DMUON_COPY_MAX_CTAS=96` | 每次 copy launch 默认最多 96 个 block，实际还受设备 SM 数和任务 tile 数限制。 |

CUDA 搬运支持 FP32、FP16 和 BF16。CTA 预算限制单次 copy launch，不代表预留
对应数量的 SM，也不限制所有通信 kernel。默认预算在 132-SM 设备上选取，
其他设备和负载可以自行调整。

## 支持范围与检查点 {#scope-and-checkpoints}

显式布局目前支持 FSDP2/HSDP 中已物化、连续、非 TP 的参数，要求使用 Muon
路由和 owner-broadcast 前向发布方式。请在 DTensor/TP placement 之前配置。
DDP setup 接口和实验性的 Muon all-gather 发布方式尚未支持这套显式布局。

各 rank 应使用相同的布局声明。

模型检查点保留原参数名与形状。优化器检查点记录每张逻辑矩阵的状态和布局签名，
逻辑矩阵划分或元素映射不一致时会拒绝加载。恢复训练时保持相同布局，并使用 DMuon 的
[检查点辅助接口](checkpoint.md)。
