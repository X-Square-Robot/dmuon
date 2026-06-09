# 常见问题

!!! tip "TL;DR"
    常见采用问题的快速解答。每条条目均交叉链接到相关指南。如果你的问题
    不在此列，请开一个 GitHub Discussion 或查看[故障排查](../../troubleshooting.md)。

---

??? question "DMuon 和 ZeRO-1 有什么区别？"
    **ZeRO-1** 将优化器状态分片到所有 rank，每个 rank 独立更新自己的 1/N
    参数切片。对 Adam 来说这很高效：每个 rank 可独立更新自己的切片。

    **矩阵优化器的问题**在于 Newton-Schulz 无法在参数切片上运行——它需要
    完整的 (m, n) 梯度矩阵才能计算有意义的正交更新。ZeRO-1 持有 1/N 行的
    rank 在不先 all-gather 完整矩阵的情况下无法正确正交化。

    **专属所有权**更进一步：单个 rank 拥有*整个*参数并在本地运行 NS，无需
    any all-gather。代价是其他 rank 必须接收更新后参数的广播——但该广播
    可以隐藏在下一次前向传播中。

    详见[核心概念](../../getting-started/concepts.md)。

---

??? question "我需要 HSDP 吗？"
    **单节点多 GPU** — 1D shard-only mesh 就够用，更简单。将 `mesh` 传给
    `dedicate_params`；省略 `replicate_mesh`。

    **多节点训练**（`replicate_size ≥ 2`）— HSDP 的两阶段 reduce（shard →
    replicate）加上 DMuon 的异步后置广播能发挥作用：replicate 广播与下一次
    前向传播重叠，摊销了节点间 IB 的代价。在跨两个节点 16+ GPU 的场景下，
    异步隐藏通常是值得的。

    将 `replicate_mesh=hsdp["replicate"]` 传给 `dedicate_params`，将
    `replicate_async=True`（默认）传给 `Muon` 即可获得完整 HSDP 收益。

    详见 [HSDP 指南](../../guides/hsdp.md)。

---

??? question "什么时候需要 hook_boundary_predicate？"
    默认启发式从参数的全限定名中查找 `layers.N` 或 `blocks.N` 来确定 hook
    注册的层模块。对于标准的 Llama/Qwen `model.layers.N.mlp.*_proj` 结构
    和标准的 ViT `visual.blocks.N.attn.*_proj` 结构，这个方式有效。

    在以下情况需要 `hook_boundary_predicate`：

    - **VLA 模型**：action head 位于主层堆栈之外
    - **MoE 模型**：expert 参数的 FQN 模式不同
    - **嵌套多模态模型**：视觉编码器和 LLM 有独立的层编号层级
    - **自定义 adapter / LoRA 层**：adapter 名称不匹配 `layers.N`

    VLA action head 示例：

    ```python
    import dmuon

    dmuon.dedicate_params(
        model,
        mesh,
        predicate=lambda n, p: "proj" in n,
        hook_boundary_predicate=lambda m: hasattr(m, "_is_action_layer"),
        hook_boundary_strict=True,
    )
    ```

    详见[自定义 Hook 边界](../../guides/custom-hook-boundaries.md)。

---

??? question "选 Z2 还是 Z3？"
    **默认 Z3**（`reshard_after_forward=True`）适用于 Muon 目标参数超出
    每 rank 可用显存预算的任何模型。每步通信代价为 `3(N-1)/N · P_M` 字节；
    前向后广播缓冲区被释放，每 rank 峰值显存较低。

    **Z2**（`reshard_after_forward=False`）适用于可以在前向+反向过程中
    在每个 rank 保持 P_M 元素常驻的场景。通信代价降至 `2(N-1)/N · P_M`——
    理论上的 ring all-reduce 下界。最适合较小模型（≤ 3B 参数），此时 GPU
    显存不是瓶颈。

    粗略经验：当 Muon 目标参数的总字节数在考虑激活值和优化器状态后
    占每 GPU 显存的 20% 以下时使用 Z2。7B+ 参数几乎总是需要 Z3。

    详见 [Z2 与 Z3 模式](../../guides/z2-z3-modes.md)。

---

??? question "DMuon 可以与 DeepSpeed 混合使用吗？"
    **简短回答：** ZeRO-3 目前不支持。DeepSpeed ZeRO-3 使用与 DMuon 的
    `dedicate_params` + `fully_shard` 合约不兼容的参数存储机制
    （`deepspeed.zero.Init` + 自定义 hook）。

    **DeepSpeed ZeRO-0/1/2** 集成在路线图中——专属所有权原语在原理上是
    兼容的，因为参数在存储层没有被碎片化。欢迎贡献；现有方法见
    [集成方案](../../guides/integration-recipes.md)。

    **当前推荐：** 将 DMuon 与 **PyTorch FSDP2** 配合使用，这是主要的
    测试和支持配置。

---

??? question "有逐比特一致的收敛保证吗？"
    有。DMuon 在 4-GPU 测试环境（`tests/distributed/test_hsdp_correctness.py`）
    上从三个维度验证逐比特一致输出：

    1. **HSDP vs. shard-only：** DMuon-HSDP（G=2, R=2）在 10 步训练中
       与 shard-only DMuon（G=4）产生相同的 loss 值。
    2. **非 TP HSDP 的异步 vs. 同步：** `replicate_async=True` 与
       `replicate_async=False` 产生相同输出；存在 TP-sharded dedicated
       参数时当前会使用同步 publish。
    3. **检查点恢复：** 从检查点恢复训练与不中断的连续训练在相同步数内
       产生相同 loss 值。

    这些测试在每个 PR 上运行。如果观察到与单 GPU 基线的偏差，请查看
    [故障排查](../../troubleshooting.md)。

---

??? question "DMuon 支持张量并行（TP）吗？"
    **1D FSDP + TP** 支持。先应用 TP，再应用 DMuon，最后 FSDP2：

    ```python
    from torch.distributed.tensor.parallel import parallelize_module
    import dmuon
    from torch.distributed.fsdp import fully_shard

    for layer in model.layers:
        parallelize_module(layer.mlp, tp_mesh, {...})   # 先 TP

    dmuon.dedicate_params(model, dp_mesh, ...)          # 再 DMuon

    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)                # 最后 FSDP2
    fully_shard(model, mesh=dp_mesh)
    ```

    在 DP 组内，所有 rank 共享相同的 TP 位置，因此广播 TP 分片是正确的。
    DMuon 使用带 TP 感知的 Gram Newton-Schulz，通过 all-reduce Gram 矩阵
    实现 O(d²) TP 通信。

    **2D HSDP × TP**（3D 并行）尚未验证。详见
    [张量并行](../../guides/tp-support.md)。

---

??? question "需要引用哪些相关工作？"
    **Gram Newton-Schulz**（使用默认 `"gram"` 后端时）：

    ```bibtex
    @misc{GramNewtonSchulz,
      title  = {Gram Newton-Schulz},
      author = {Jack Zhang and Noah Amsel and Berlin Chen and Tri Dao},
      year   = {2026},
      url    = {https://dao-ailab.github.io/blog/2026/gram-newton-schulz/}
    }
    ```

    **Muon 优化器**（Jordan et al., 2024）：arXiv:2502.16982

    **Distributed Shampoo**（Shi et al., 2023）和 **ZeRO-1**
    （Rajbhandari et al., 2020）开创了 DMuon 所扩展的专属所有权原语。

---

## 参见

- [故障排查](../../troubleshooting.md)
- [核心概念](../../getting-started/concepts.md)
- [API 文档](../../reference/api.md)
- [参与贡献](../../contributing.md)
