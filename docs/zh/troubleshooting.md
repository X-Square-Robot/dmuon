# 故障排查

!!! tip "TL;DR"
    大多数问题属于四类：import/安装、训练设置（错误的 predicate 或 mesh 形状）、
    运行时正确性（NaN、发散）或性能（步骤慢、无重叠、OOM）。在症状表中找到
    你的错误，应用修复方案，然后用单 GPU 冒烟测试验证，再重新运行分布式训练。

---

## 安装

??? warning "ImportError: No module named 'dmuon'"
    **原因：** DMuon 未安装在当前 Python 环境中。

    **修复：**
    ```bash
    git clone https://github.com/StarrickLiu/dmuon && cd dmuon
    pip install -e .
    ```
    验证：`python -c "import dmuon; print(dmuon.__version__)"`。

---

??? warning "ImportError: cannot import name 'fully_shard' from 'torch.distributed.fsdp'"
    **原因：** PyTorch 版本过旧。FSDP2（`torch.distributed.fsdp` 中的
    `fully_shard`）需要 PyTorch 2.4+。

    **修复：** 升级 PyTorch。
    ```bash
    pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cu121
    ```

---

??? warning "CUDA 扩展加载失败 / CuteDSL SYRK 不可用"
    **原因：** CUDA 版本不匹配或缺少 CuteDSL 依赖。

    **修复：** DMuon 在 CuteDSL 内核不可用时自动回退到 cuBLAS
    （`torch.mm` / `torch.addmm`）。验证当前后端：
    ```python
    import dmuon
    print(dmuon.get_ns_backend())
    # "Gram NS · kernel=cublas (SM80, universal fallback)" 是可接受状态
    # —— 正确性不受影响，只是 SYRK 加速不在。
    ```
    若需要 A 卡 `cute_sm80` 快路径，安装 `[syrk]` extras；SM90+ 机器
    请 `pip install dmuon[quack]`，`kernel="auto"` 会自动挑中 Tri Dao
    的 quack SYRK。完整的自动检测阶梯见
    [后端分发](reference/newton-schulz.md#backend-dispatch)。

---

## 训练设置

??? warning "TypeError: dedicate_params() got an unexpected keyword argument '...'"
    **原因：** 用户代码与已安装的 DMuon 版本不匹配。常见情况：用户代码
    使用了较新 API 中的 `replicate_mesh=` 或 `hook_boundary_predicate=`，
    但安装的包版本较旧。

    **修复：** 拉取最新代码并重新安装：
    ```bash
    git pull && pip install -e .
    ```
    检查 `dmuon.__version__` 是否符合预期。

---

??? warning "警告：'dedicate_params: no parameters matched the predicate'"
    **原因：** `predicate` 函数对每个参数都返回了 `False`。常见原因：
    predicate 中的字符串错误（如用了 `"proj"` 但模型用的是 `"linear"`），
    或模型结构中没有 2D 投影参数。

    **修复：** 交互式调试你的 predicate：
    ```python
    for name, param in model.named_parameters():
        if param.ndim == 2:
            print(name, param.shape)
    ```
    根据看到的名称调整 predicate。

---

??? warning "mesh 形状错误——HSDP mesh 必须是带名称 ('replicate', 'shard') 的 2D mesh"
    **原因：** HSDP 设置需要带 `mesh_dim_names=("replicate", "shard")` 的
    2D `DeviceMesh`。将未命名或 1D mesh 传给 `replicate_mesh` 会失败。

    **修复：**
    ```python
    from torch.distributed.device_mesh import init_device_mesh

    hsdp = init_device_mesh(
        "cuda",
        (replicate_size, shard_size),
        mesh_dim_names=("replicate", "shard"),
    )
    dmuon.dedicate_params(model, hsdp["shard"], replicate_mesh=hsdp["replicate"])
    ```

---

## 运行时正确性

??? warning "几步后 loss 出现 NaN"
    **最常见原因：** 上游问题，与 DMuon 无关。检查仅使用 AdamW 时是否
    也出现同样的 NaN。若是，问题在于数据加载、模型架构或 dtype 不匹配。

    **若 NaN 仅在 DMuon 下出现：** 检查混合精度不匹配。确保
    `dedicate_params` 中的 `compute_dtype` 与模型的 autocast dtype 一致，
    或留为 `None` 以继承参数 dtype。

    **Gram NS 中的持续 NaN：** 若 NaN 仅在 `"gram"` 后端出现，切到 cuBLAS
    参考内核排查是不是快路径 SYRK 的问题：
    ```python
    ns = dmuon.NewtonSchulz(kernel="cublas")  # 等价于 deterministic=True
    optimizer = dmuon.Muon(model, ns_backend=ns)
    ```
    若 cuBLAS 也 NaN，问题在 Gram 迭代本身（系数、重启位置、输入尺度）
    而非 kernel。

---

??? warning "'forward output type mismatch' / ModelOutput 属性访问丢失"
    **原因：** DMuon 的前向 hook 包装了模块输出；在旧版本中 HuggingFace
    `ModelOutput` 命名元组的属性访问在包装后丢失。

    **修复：** 这在最新 DMuon 中已修复。如果在当前 `main` 上看到此问题，
    请提交包含模型类和 PyTorch 版本的 GitHub issue。

---

??? warning "Loss 与单 GPU 或 AdamW 基线发散"
    **原因——系数不匹配：** 若从 `"gram"` 切换到 `"direct"`，学习率可能
    需要调整。两个后端的有效步长不同。

    **原因——学习率过高：** Muon 的内部缩放为 `0.2 * sqrt(max(m, n))`。
    从 `lr=0.02` 开始，若出现发散则降低。

    **原因——各 rank 使用了不同的 NS 内核：** 确保每个 rank 使用相同的
    `ns_backend` / `kernel=` 配置；在 DP / replicate 轴上混用不同 SYRK
    内核（例如部分 rank 走 `cute_sm80`、部分走 `cublas`）会累积跨 rank
    的数值漂移。建议每个 rank 打印 `dmuon.get_ns_backend()` 并交叉核对。

    **调试：** 先在小模型上对比 `"gram"` 和 `"direct"` 后端的 loss 曲线。

---

## 性能

??? warning "优化步骤很慢（小模型却 >>100 ms）"
    **原因：** owner 负载可能不均衡，或者 post-step publish 太大，
    无法藏进下一轮 forward。

    **修复：** 先用 `dmuon.Muon(..., replicate_async=False/True)` 对比同步
    与异步计时。如果只有少数 owner rank 慢，检查 dedicated 参数分配、
    hook 边界和 owner 策略。

    **诊断：** 打印当前 rank 的 routing 和通信计划摘要：
    ```python
    import json
    import dmuon

    print(json.dumps(
        dmuon.summarize_param_groups(model, optimizer),
        indent=2,
        default=str,
    ))
    print(json.dumps(
        dmuon.summarize_comm_plan(model),
        indent=2,
        default=str,
    ))
    ```

    如需查看 forward-unshard wait 计数器，在 `dmuon.dedicate_params()`
    运行前设置 `DMUON_RECORD_FORWARD_PROFILE=1`，再在诊断边界读取：
    ```python
    profile = dmuon.collect_forward_unshard_profile(
        model,
        synchronize=True,
    )
    ```
    不要把 `synchronize=True` 放进正常 step timing loop；它会强制 CUDA
    同步，改变正在测量的 overlap 行为。

---

??? warning "广播从未与前向重叠 / 未观察到异步加速"
    **原因 1：** 网络带宽是瓶颈——replicate 广播使 IB 饱和，前向来不及
    隐藏它。常见于仅有 NVLink 的节点共享慢速上行链路。

    **原因 2：** 前向传播相对广播太快（小模型、短序列长度），没有足够的
    计算来隐藏通信。

    **修复：** 切换到同步模式，避免不必要的异步开销：
    ```python
    optimizer = dmuon.Muon(model, lr=0.02, replicate_async=False)
    ```

---

??? warning "owner rank 上 OOM"
    **原因：** LPT（最长处理时间）分配可能将过多大参数分配给少数几个
    owner rank，导致显存不均衡。

    **修复：** 验证 `_extract_layer_id` 是否正确识别了模型的层
    结构。对于 ViT 风格的模型（FQN 中含 `blocks.N` 路径），确保
    `blocks.N` 出现在 FQN 中——否则所有参数可能折叠到同一个"层"键。
    详见[设计/架构](design/architecture.md)。

---

## HSDP 特有问题

??? warning "意外中断（KeyboardInterrupt / OOM）时遗留异步 event"
    **原因：** 异步 replicate-broadcast stream 有未消费的 pending event，
    因为训练在下一次前向传播前退出了。

    **修复：** 这是无害的——CUDA 运行时在进程退出时清理 stream。如果
    你在编写优雅退出处理器，在释放模型资源前调用：
    ```python
    dmuon.wait_all_replicate_broadcasts(model)
    ```

---

??? warning "跨不同拓扑的检查点保存/加载失败"
    **原因：** DMuon 状态字典记录了相对于分片坐标系的 owner 分配。用
    G=8 保存的检查点无法直接加载到 G=4 的运行中。

    **修复：** 这是已知限制。通过 `get_model_state_dict`（重建完整未分片
    张量）保存，然后用 `set_model_state_dict` 重新加载。跨拓扑变更时
    不要复用优化器状态字典——仅从模型权重重新开始。

---

## 张量并行

??? warning "`ValueError: DMuon requires named DeviceMesh for TP detection`"
    **原因：** 你传给 `parallelize_module` / `fully_shard` /
    `dedicate_params` 的 mesh 没有 `mesh_dim_names`。DMuon 通过名称
    集合差识别 TP 轴，所以必须传名称。

    **修复：** 构造 mesh 时带上 `mesh_dim_names`：
    ```python
    mesh = init_device_mesh("cuda", (dp_size, tp_size),
                            mesh_dim_names=("dp", "tp"))
    ```

??? warning "`RuntimeError: tp_scatter_delta_async: previous event still pending`"
    **原因：** 两次 `optimizer.step()` 之间没有 forward。async 的 scatter
    event 依赖**下一次 forward** 的 `_pre_forward_wait` 来 drain。常见于
    自定义训练 loop 在一个 iter 内调 `step()` 两次，或者 `step()` 之后
    不做 forward 直接存 checkpoint。

    **修复：** 保证两次 `step()` 之间有一次 forward；或者把 post-step
    通信改为 sync：
    ```python
    optimizer = dmuon.Muon(model, lr=0.02, replicate_async=False)
    ```

??? info "HSDP × TP（3D mesh）— 已支持"
    3D mesh `(replicate, shard, tp)` 已经验证过（见
    [TP 支持指南](guides/tp-support.md) 和内部报告 `tp_design.md` /
    `tp_alignment_report.md`）。sync 与 async post-step 路径产生
    bit-identical loss 轨迹。调用顺序：

    1. `parallelize_module(model, mesh["tp"], plan)`
    2. `dmuon.dedicate_params(model, mesh["shard"], replicate_mesh=mesh["replicate"], ...)`
    3. `fully_shard(model, mesh=mesh["replicate","shard"])`

---

## 参见

- [常见问题](faq/index.md)
- [HSDP 指南](guides/hsdp.md)
- [API 文档](reference/api.md)
