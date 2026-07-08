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
    git clone https://github.com/X-Square-Robot/dmuon
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
    git clone https://github.com/X-Square-Robot/dmuon
    cd dmuon
    pip install -e ".[dev]"
    ```

    以可编辑模式安装库，并包含测试、打包和文档依赖。
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

## 可选：快速梯度裁剪（CUDA）

DMuon 附带一个可选的 CUDA 内核，把**分段梯度裁剪**——即 `regular` / `muon` /
`adamw` 三个梯度组各自的范数、裁剪系数与就地缩放——融合到一趟计算里。训练语义与
纯 Python 路径完全一致：每个分段仍各自计算范数、各自使用独立的裁剪系数，只是把运算
搬到了 GPU 上。

`torch` **刻意不作为**构建依赖（若固定它，隔离构建会去下载多 GB 的通用 torch 并把
内核链接到它上——有 ABI 错配风险）。因此要编译内核，需在**已装 torch** 的环境里、
`PATH` 带上 CUDA 工具链、**关闭构建隔离**安装：

```bash
# 需要 nvcc / CUDA_HOME 可见，且环境里已装 torch
pip install -e . --no-build-isolation
```

- 加了 `--no-build-isolation` 且有 `CUDA_HOME` 时，`dmuon._fast_clip_cuda` 会针对你
  真实的 torch 编译并自动启用。
- 普通 `pip install -e .`（隔离构建）的构建环境里没有 torch，`setup.py` 会跳过扩展，
  运行时使用等价的纯 Python 裁剪。不会报错——只是把裁剪放到主机端计算。
- 若扩展**编译出来但加载失败**（例如之后升级 torch/CUDA 破坏了 ABI），DMuon 会 warn
  一次并回退到 Python。设 `DMUON_FAST_CLIP_VERBOSE=1` 可改为直接抛出底层错误。

### 编译与运行时开关

| 变量 | 作用 |
|------|------|
| `DMUON_BUILD_FAST_CLIP=0` | 安装时跳过编译该 CUDA 扩展。 |
| `DMUON_FAST_CLIP=0` | 运行时禁用快速路径（改用纯 Python）。 |
| `DMUON_FAST_CLIP_CHUNK_SIZE` | 内核的单张量分块大小（默认 `262144`）。 |
| `DMUON_FAST_CLIP_VERBOSE=1` | 直接抛出导入错误而非静默回退——用于本应编译成功却没生效时排查。 |

运行时路径对于不满足内核契约的输入（非连续、稀疏、不支持的 dtype）或检测到非有限的
分段范数时，也会自动回退到 Python，因此扩展缺失或过时都不会改变结果。

!!! note "编译需要的是编译器，而非特定 GPU"
    编译裁剪内核只需要主机端的 CUDA 编译器（`nvcc`），并不要求 SM80+ GPU——这些
    内核与架构无关。将 CUDA 工具包与你的 PyTorch CUDA 版本对齐即可（11.8 / 12.1 / 12.4）。

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
NS 后端      : Gram NS · kernel=cute_sm80 (SM80, DMuon internal)
```

未安装 SYRK 或 GPU 较旧时的预期输出：

```
DMuon 版本   : 0.2.0
PyTorch 版本 : 2.6.0
CUDA 可用    : True
NS 后端      : Gram NS · kernel=cublas (SM80, universal fallback)
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

**快速裁剪内核未生效（裁剪统计里 `fastpath=False`）**
: `dmuon._fast_clip_cuda` 扩展没有被编译（安装时无 `CUDA_HOME`），或被
  `DMUON_FAST_CLIP=0` 禁用了。请在 `PATH` 中带上 CUDA 工具链重新安装，或设置
  `DMUON_FAST_CLIP_VERBOSE=1` 查看导入错误。其间梯度裁剪仍通过 Python 路径正确执行。

---

## 另请参见

- [快速开始](quickstart.md) — 运行第一次分布式训练
- [核心概念](concepts.md) — 训练前了解专属所有权
- [故障排查](../troubleshooting.md) — 运行时错误与常见问题
