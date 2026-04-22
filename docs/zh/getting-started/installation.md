# 安装

!!! tip "TL;DR"
    通过 `pip install -e .` 从源码安装。SYRK 加速需要
    `pip install -e ".[syrk]"` 并配备 SM80+ GPU（A100/A800/H100）。
    标准环境下两分钟内完成安装。

---

## 环境要求

| 要求 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.10 | 内部使用了 `match` 语法 |
| PyTorch | 2.6 | FSDP2 的 `fully_shard` API 在 2.6 稳定 |
| CUDA | 11.8 / 12.1 / 12.4 | 三个版本均已测试 |
| GPU SM | SM80+ | SYRK 内核所需（可选） |
| NCCL | 随 PyTorch 附带 | 无需单独安装 |

!!! note "SM80+ 与 SYRK"
    CuteDSL SYRK 内核针对 SM80+（A100、A800、H100、H200）。
    在旧版 GPU（SM70 / V100）上 DMuon 仍可正常运行——
    Newton-Schulz 将自动回退到 `@torch.compile` 的纯 PyTorch 实现，
    功能完全正确，优化器步时约慢 1.5 倍。

---

## 安装方式

=== "从源码安装（推荐）"

    ```bash
    git clone https://github.com/StarrickLiu/dmuon
    cd dmuon
    pip install -e .
    ```

    以可编辑模式安装核心库。SYRK 内核扩展**未**编译；
    Newton-Schulz 使用编译后的 PyTorch 后备实现。

=== "pip 安装（即将上线）"

    ```bash
    # 即将上线，暂未发布到 PyPI
    pip install dmuon
    ```

    PyPI 发布计划在研究预览阶段结束后进行。
    在此之前请从源码安装（见"从源码安装"标签页）。

=== "开发模式（可编辑 + 测试依赖）"

    ```bash
    git clone https://github.com/StarrickLiu/dmuon
    cd dmuon
    pip install -e ".[dev]"
    ```

    以可编辑模式安装库，并包含测试依赖
    （`pytest`、`pytest-dist` 等工具）。
    运行单元测试确认一切正常：

    ```bash
    pytest tests/unit/ -v
    ```

---

## 可选：SYRK 内核加速

SYRK 内核利用 Gram 矩阵的对称性，在 Newton-Schulz 上实现约 1.5 倍加速。
需要 SM80+ 硬件和额外的构建依赖：

```bash
pip install -e ".[syrk]"
```

将安装以下依赖：

- `nvidia-cutlass-dsl >= 4.4.2`
- `apache-tvm-ffi`
- `torch-c-dlpack-ext`

首次使用时 JIT 编译通常需要 1–3 分钟，编译产物缓存在 `~/.cache/dmuon/`。

---

## 验证安装

```python title="verify_install.py"
import dmuon
import torch

print(f"DMuon 版本   : {dmuon.__version__}")
print(f"PyTorch 版本 : {torch.__version__}")
print(f"CUDA 可用    : {torch.cuda.is_available()}")
print(f"NS 后端      : {dmuon.get_ns_backend()}")
```

已安装 SYRK 且配备 SM80+ GPU 时的预期输出：

```
DMuon 版本   : 0.2.0
PyTorch 版本 : 2.6.0
CUDA 可用    : True
NS 后端      : syrk_sm80
```

未安装 SYRK 或 GPU 较旧时的预期输出：

```
DMuon 版本   : 0.2.0
PyTorch 版本 : 2.6.0
CUDA 可用    : True
NS 后端      : compiled
```

---

## 常见问题排查

**`ImportError: cannot import name 'fully_shard' from torch.distributed.fsdp`**
: PyTorch 版本低于 2.6。FSDP2 的 `fully_shard` API 在 2.6 才稳定。
  执行 `pip install --upgrade torch`，确认 `torch.__version__` 不低于 `2.6.0`。

**`RuntimeError: NCCL error: unhandled system error`**
: 通常是进程组初始化问题。确认已设置 `MASTER_ADDR` 和 `MASTER_PORT`，
  且 `dist.init_process_group` 在 `dmuon.dedicate_params` 之前调用。
  详见[故障排查](../troubleshooting.md)。

**安装 SYRK 时 `cutlass-dsl` 构建失败**
: 确认使用了 `[syrk]` 额外依赖：`pip install -e ".[syrk]"`。
  若构建仍失败，编译后备方案会自动启用——
  Newton-Schulz 仍可正确运行，只是稍慢。

---

## 另请参见

- [快速开始](quickstart.md) — 运行第一次分布式训练
- [核心概念](concepts.md) — 训练前了解专属所有权
- [故障排查](../troubleshooting.md) — 运行时错误与常见问题
