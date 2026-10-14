# 参与贡献

!!! tip "TL;DR"
    欢迎提交 issue、bug 修复和新的适配器集成。重大改动请先开 GitHub issue
    讨论范围，再投入编码时间。所有分布式变更必须通过逐比特正确性测试
    （`test_hsdp_correctness.py`）。

---

## 开发环境配置

```bash
# 克隆并以可编辑模式安装
git clone https://github.com/X-Square-Robot/dmuon
cd dmuon

# 创建环境（conda 或 venv）
conda create -n dmuon-dev python=3.11 && conda activate dmuon-dev

# 安装 PyTorch（根据需要调整 CUDA 版本）
pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu121

# 安装 DMuon 及开发依赖
pip install -e ".[dev]"

```

`[dev]` extra 会安装测试、lint、打包和文档工具，包括
`pytest`、`ruff`、`build`、`mkdocs-material` 和 `mkdocstrings[python]`。
需要 `twine` 的发布检查可额外安装 `pip install -e ".[release]"`。

---

## 运行测试

### 单元测试

```bash
python -m pytest tests/unit/ -v
```

### 分布式测试（需要 4 块 GPU）

```bash
# 逐比特正确性——每次分布式相关 PR 前必须运行
torchrun --nproc_per_node=4 tests/distributed/test_hsdp_correctness.py
```

分布式测试每次约需 2 分钟。先编写小型隔离的单元测试验证逻辑，再使用
分布式测试环境作为调试循环（参见项目关于冒烟测试的笔记）。

### 原生 Ascend 分布式测试（需要 4 块 NPU）

使用版本匹配的 PyTorch、`torch_npu` 和 CANN 环境，并选择四个可见设备。
同一个必测入口提供原生 NPU/HCCL 验证：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    tests/distributed/test_hsdp_correctness.py --device npu \
    --output-dir /path/to/new-hsdp-proof
```

该测试使用相同的初始化和输入，在 1D shard-only 与 2x2 HSDP mesh 之间比较
上游模型和优化器配置。每个 rank 在每次更新后严格检查 loss、完整模型状态与
优化器状态。HSDP 保存一个标准的 rank-0 检查点，重建模型并恢复该检查点后，
再检查两次更新。不使用容差或各 rank 的本地状态覆盖。通过
`--steps 4 --resume-after 2` 可执行每种拓扑四次更新加两次恢复更新，
共十次优化器调用的冒烟检查。默认验证在每种拓扑执行十次更新，加两次恢复
更新，共 22 次分布式优化器调用。上面的 CUDA 命令不变。

### 单测试冒烟验证模式

```python
# 最小隔离测试——在修改分布式逻辑前先写这个
import torch
# ... 不需要 torchrun 的 10 行设置 ...
```

---

## 代码风格

- **格式化工具：** `ruff format`（配置在 `pyproject.toml` 中）
- **Linter：** `ruff check`（配置在 `pyproject.toml` 中）
- **类型提示：** 所有公开函数和类方法必须加
- **文档字符串：** NumPy 格式（mkdocstrings 已配置）；简洁——
  一行摘要、Args、Returns
- **代码中不使用 emoji**

运行风格检查：

```bash
ruff format dmuon/ examples/
ruff check dmuon/ examples/
```

提交变更前请运行以上两项。

---

## PR 检查清单

提交 pull request 前：

- [ ] 基于 `main` rebase（`git fetch origin && git rebase origin/main`）
- [ ] 单元测试通过（`pytest tests/unit/`）
- [ ] 分布式变更：`test_hsdp_correctness.py` 通过
- [ ] 原生 NPU 变更：四块 NPU 上的 `--device npu` 必测入口通过
- [ ] 公开 API 变更已更新文档字符串
- [ ] 行为或 API 接口变更已更新相关文档
- [ ] **逐比特正确性保持** — 相同随机种子下，1D shard-only、HSDP 和
  检查点恢复运行的 loss 值不变

新功能请先开 issue 对齐范围。

---

## 架构导读

在修改核心 ownership 机制（`dmuon/api.py`、`dmuon/_core/`、
`dmuon/_backends/`）之前，请先阅读
[设计/架构](design/architecture.md)。关键不变量：

- `DedicatedParam._owned_data` 是 owner rank 上参数值的唯一真实来源
- Hook 注册顺序（pre-forward → post-forward → pre-backward →
  post-backward）必须与 FSDP2 的 hook 顺序匹配
- 每个异步 collective 必须有对应的 drain 路径（在下一次前向的
  `_pre_forward_wait` 中，或在 `wait_all_replicate_broadcasts` 中）

---

## 发布流程

DMuon 遵循语义化版本：`MAJOR.MINOR.PATCH`。

- **Patch：** bug 修复、文档字符串/文档更新，无 API 变更
- **Minor：** 新增公开 API 符号、新后端、新指南页面
- **Major：** 破坏性 API 变更（罕见；需要讨论）

文档在每次推送到 `main` 时通过 GitHub Actions 自动部署到 GitHub Pages。

---

## 社区

- **Bug 报告和功能请求：** [GitHub Issues](https://github.com/X-Square-Robot/dmuon/issues)
- **设计讨论：** [GitHub Discussions](https://github.com/X-Square-Robot/dmuon/discussions)
- 暂无 Discord

提 issue 前请先搜索已有 issue。

---

## 参见

- [设计/架构](design/architecture.md)
- [常见问题](faq/index.md)
- [故障排查](troubleshooting.md)
