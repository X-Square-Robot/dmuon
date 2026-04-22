# 参与贡献

!!! tip "TL;DR"
    欢迎提交 issue、bug 修复和新的适配器集成。重大改动请先开 GitHub issue
    讨论范围，再投入编码时间。所有分布式变更必须通过逐比特正确性测试
    （`test_hsdp_correctness.py`）。

---

## 开发环境配置

```bash
# 克隆并以可编辑模式安装
git clone https://github.com/StarrickLiu/dmuon
cd dmuon

# 创建环境（conda 或 venv）
conda create -n dmuon-dev python=3.11 && conda activate dmuon-dev

# 安装 PyTorch（根据需要调整 CUDA 版本）
pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cu121

# 安装 DMuon 及开发依赖
pip install -e ".[dev]"

# 安装 pre-commit hook
pre-commit install
```

`[dev]` extra 会安装 `pytest`、`black`、`ruff`、`mkdocs-material` 和
`mkdocstrings[python]`。

---

## 运行测试

### 单元测试

```bash
python -m pytest tests/unit/ -v
```

!!! note
    `tests/unit/test_newton_schulz.py` 存在与你的改动无关的预有失败——
    如有需要可用 `--ignore=tests/unit/test_newton_schulz.py` 跳过。

### 分布式测试（需要 4 块 GPU）

```bash
# 逐比特正确性——每次分布式相关 PR 前必须运行
torchrun --nproc_per_node=4 tests/distributed/test_hsdp_correctness.py
```

分布式测试每次约需 2 分钟。先编写小型隔离的单元测试验证逻辑，再使用
分布式测试环境作为调试循环（参见项目关于冒烟测试的笔记）。

### 单测试冒烟验证模式

```python
# 最小隔离测试——在修改分布式逻辑前先写这个
import torch
# ... 不需要 torchrun 的 10 行设置 ...
```

---

## 代码风格

- **格式化工具：** `black`（行长 99）
- **Linter：** `ruff`（配置在 `pyproject.toml` 中）
- **类型提示：** 所有公开函数和类方法必须加
- **文档字符串：** NumPy 格式（mkdocstrings 已配置）；简洁——
  一行摘要、Args、Returns
- **代码中不使用 emoji**

运行风格检查：

```bash
black dmuon/ tests/
ruff check dmuon/ tests/
```

以上两项也由上面安装的 pre-commit hook 强制执行。

---

## PR 检查清单

提交 pull request 前：

- [ ] 基于 `main` rebase（`git fetch origin && git rebase origin/main`）
- [ ] 单元测试通过（`pytest tests/unit/`）
- [ ] 分布式变更：`test_hsdp_correctness.py` 通过
- [ ] 公开 API 变更已更新文档字符串
- [ ] 行为或 API 接口变更已更新相关文档
- [ ] `CHANGELOG.md` 已添加条目（格式：`- [类型] 简短说明`）
- [ ] **逐比特正确性保持** — 相同随机种子下，1D shard-only、HSDP 和
  检查点恢复运行的 loss 值不变

新功能请先开 issue 对齐范围。

---

## 架构导读

在修改核心 ownership 机制（`dmuon/param.py`、`dmuon/group.py`、
`dmuon/state.py`、`dmuon/api.py`）之前，请先阅读
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
内部文档（`docs/internal/`）不包含在发布站点中。

---

## 社区

- **Bug 报告和功能请求：** [GitHub Issues](https://github.com/StarrickLiu/dmuon/issues)
- **设计讨论：** [GitHub Discussions](https://github.com/StarrickLiu/dmuon/discussions)
- 暂无 Discord

提 issue 前请先搜索已有 issue。

---

## 参见

- [设计/架构](design/architecture.md)
- [常见问题](faq/index.md)
- [故障排查](troubleshooting.md)
