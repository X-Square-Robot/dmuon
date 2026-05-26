# DMuon

**面向 PyTorch DDP、FSDP2 和 HSDP 的矩阵优化器专属所有权方案。**

*一个所有者。一次 Newton-Schulz。零优化器 all-gather。*

---

<div class="dmuon-hero" markdown>

DMuon 为每个矩阵参数指定唯一的**所有者 rank**。所有者存储完整参数，从其他 rank 汇聚梯度，并独立运行 Newton-Schulz——彻底消除了让朴素 FSDP2+Muon 比 AdamW 慢 3–4 倍的 all-gather 和冗余计算。

从单节点到多节点 HSDP 集群，只需两行 API 变更。二维 Mesh、两阶段 reduce 以及异步 forward 隐藏广播均由内部自动处理。

</div>

---

## DMuon 能做什么

??? abstract "LLM 预训练 — 在 FSDP2/HSDP 上训练 Llama、Qwen、Mistral"

    以接近 AdamW 的代价使用 Muon 训练 Transformer 语言模型。
    专属所有权将每个投影参数路由到单一所有者；
    Newton-Schulz 每步只运行一次，无需优化器 all-gather。
    已在 Qwen2.5（1.5B–7B）和 Llama-3（3B–8B）的 8×A800 上测试，
    相比 FSDP2+AdamW 的总步时开销仅为 4–13%。

    ```python
    import dmuon
    from torch.distributed.fsdp import fully_shard

    dmuon.dedicate_params(
        model, mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    optimizer = dmuon.Muon(model, lr=0.02, ns_steps=5, adamw_lr=1e-3)
    ```

??? abstract "多节点 HSDP — 带异步广播隐藏的二维 Mesh"

    通过 `(replicate, shard)` 二维设备 Mesh 跨节点扩展。
    DMuon 执行两阶段梯度 reduce（shard → replicate），
    并在专用 CUDA 流上分发步后 replicate 广播，
    将其隐藏在下一轮迭代的 forward 计算中。
    与同步基线逐位一致；无法隐藏时自动回退。

    ```python
    hsdp = init_device_mesh(
        "cuda", (replicate_size, shard_size),
        mesh_dim_names=("replicate", "shard"),
    )
    dmuon.dedicate_params(
        model, hsdp["shard"],
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
        replicate_mesh=hsdp["replicate"],
    )
    optimizer = dmuon.Muon(model, lr=0.02, replicate_async=True)
    ```

??? abstract "VLA 与 VLM — 视觉-语言-动作及视觉-语言模型"

    DMuon 的谓词选择适用于任意架构。对于 VLM/VLA，建议先将
    predicate 限定在需要 Muon 的可训练 attention/MLP projection 层。
    未被 predicate 选中的参数，例如 embedding、冻结的 vision tower
    或任务头，会继续由标准 FSDP2 管理。如果 vision encoder 本身可训练
    且包含兼容的 projection 层，也可以通过扩展 predicate 和 hook
    boundary 纳入 DMuon。通过 Gram Newton-Schulz（O(d_model²) 通信量）
    实现 TP 兼容，可与列/行并行张量并行配合使用。

??? abstract "MoE — 专家并行布局下的混合专家模型"

    使用 `hook_boundary_predicate` 将 Hook 边界与专家模块对齐。
    每个专家的投影参数独立分配给一个所有者 rank；均衡划分确保
    各专家组间不出现掉队 rank。

---

## 核心特性

<div class="grid cards" markdown>

-   :material-server-network:{ .lg .middle } **原生 HSDP 支持**

    内置二维 `(replicate, shard)` Mesh，支持两阶段 reduce 和
    异步 forward 隐藏广播。从一维 shard-only 迁移只需一行改动。

-   :material-layers-triple:{ .lg .middle } **DMuon-Z2 / DMuon-Z3**

    为 Muon 目标参数提供与 FSDP2 `reshard_after_forward` 对应的
    内存与通信权衡。Z3（默认）节省内存；Z2 每层减少一次广播。

-   :material-transit-connection-variant:{ .lg .middle } **Hook 边界控制**

    `hook_boundary_predicate` 将 Hook 挂载点与参数划分解耦，
    可精确对齐 `fully_shard()` 边界，适配任意架构。

-   :material-check-decagram:{ .lg .middle } **逐位一致的正确性**

    异步和同步 HSDP 路径产生完全相同的损失轨迹。
    已在 4-GPU（G=2, R=2）上验证，并通过检查点重启测试。

-   :material-puzzle:{ .lg .middle } **FSDP2 兼容**

    不修改 FSDP2 内部实现。导入时安装的轻量 monkey-patch
    使 `fully_shard()` 自动跳过专属参数。

-   :material-scale-balance:{ .lg .middle } **Apache 2.0 许可**

    宽松开源协议，学术研究和生产部署均可自由使用。

</div>

---

## 基准测试

当前快照：A800-SXM4-80GB，bf16，LLM runs 使用 seq=4096、随机初始化和 synthetic data。MFU 使用 step-start interval 计算，避免把跨 step 重叠的通信重复计入步时。这些数据是受控 A800 环境下的 research-preview 摘要，适合作为相对性能参考，不应视为公开复现实验说明。

### LLM Z2/Z3 扩展性

| 模型 / 128GPU | AdamW MFU 区间 | DMuon MFU 区间 | 最优 DMuon 拓扑 |
|:--------------|--------------:|--------------:|:----------------|
| Qwen2.5-1.5B | 36.3–43.6% | 38.4–43.1% | HSDP-Z2，43.1% |
| Qwen2.5-7B | 43.0–48.3% | 39.7–48.0% | FSDP-Z2，48.0% |
| Llama-3.2-3B | 46.1–48.1% | 46.4–48.6% | HSDP-Z2，48.6% |
| Llama-3.1-8B | 47.0–49.9% | 41.1–46.2% | FSDP-Z2，46.2% |

128GPU 行覆盖 FSDP-Z2、FSDP-Z3、HSDP-Z2 和 HSDP-Z3。

---

## 快速入门

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **安装**

    从源码安装 DMuon，验证 CUDA 环境。

    [:octicons-arrow-right-24: 安装](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **快速开始**

    针对 DDP 风格、FSDP2 和 HSDP 的完整运行脚本，按需选择拓扑。

    [:octicons-arrow-right-24: 快速开始](getting-started/quickstart.md)

-   :material-head-lightbulb:{ .lg .middle } **核心概念**

    专属所有权、Z2/Z3 模式、Hook 边界与 HSDP 设计。

    [:octicons-arrow-right-24: 核心概念](getting-started/concepts.md)

-   :material-server-network:{ .lg .middle } **HSDP 指南**

    二维 Mesh、异步模式与检查点的完整说明。

    [:octicons-arrow-right-24: HSDP 指南](guides/hsdp.md)

</div>

---

DMuon 建立在 ZeRO-1（Rajbhandari 等，2020）和 Distributed Shampoo（Shi 等，2023）
所开创的专属所有权原语之上。Gram Newton-Schulz 核改编自 Dao 等，2026。

GitHub：[StarrickLiu/dmuon](https://github.com/StarrickLiu/dmuon) &nbsp;·&nbsp; arXiv 预印本：[TBD]
