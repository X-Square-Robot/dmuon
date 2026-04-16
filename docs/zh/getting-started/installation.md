# 安装

## 环境要求

- Python >= 3.10
- PyTorch >= 2.4.0
- 支持 CUDA 的多卡 GPU 环境

## 从源码安装

```bash
git clone https://github.com/StarrickLiu/dmuon && cd dmuon
pip install -e .
```

## 可选：SYRK 内核加速

DMuon 包含一个自定义的 [CuteDSL](https://github.com/NVIDIA/cutlass) SYRK 内核，利用 Gram 矩阵的对称性实现 Newton-Schulz 迭代约 1.5 倍加速。需要 SM80+ GPU（A100、A800、H100 等）和额外依赖：

```bash
pip install -e ".[syrk]"
```

安装内容包括：

- `nvidia-cutlass-dsl >= 4.4.2`
- `apache-tvm-ffi`
- `torch-c-dlpack-ext`

!!! note "说明"
    SYRK 内核是可选的。未安装时，DMuon 使用 `@torch.compile` 的纯 PyTorch 实现作为后备方案，功能完整但 Gram NS 迭代速度略慢。

## 验证安装

```python
import dmuon
print(f"DMuon {dmuon.__version__}")
print(f"NS 后端: {dmuon.get_ns_backend()}")
```

预期输出：
```
DMuon 0.2.0
NS 后端: syrk_sm80    # 或 "compiled"（未安装 SYRK 依赖时）
```

## 下一步

[快速开始](quickstart.md) — 运行你的第一次分布式训练。
