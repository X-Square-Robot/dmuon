# Newton-Schulz 算法

DMuon 实现了四种 Newton-Schulz 变体，各自针对不同场景优化。本页说明差异和使用时机。

---

## 概览

| 函数 | 空间 | TP 支持 | SYRK 加速 | 重启 | 适用场景 |
|------|------|---------|-----------|------|---------|
| `newton_schulz()` | Gram | 否（本地） | 是 | 是 | **默认** — 单 rank 或纯 DP |
| `gram_newton_schulz()` | Gram | 是 | 是 | 是 | **TP 参数** — 精确或块对角 |
| `gram_newton_schulz_local()` | Gram | 否（本地） | 是 | 是 | 内部 — 等同 `newton_schulz` |
| `direct_newton_schulz()` | 直接 | 否 | 否 | 否 | **基线** — 经典 Muon/Moonlight 算法 |

## Gram 空间 vs 直接空间

### 直接空间 NS（经典）

来自 Muon/Moonlight 的标准公式。在完整 (m, n) 矩阵上迭代：

$$
X_{k+1} = a_k X_k + b_k (X_k X_k^T) X_k + c_k (X_k X_k^T)^2 X_k
$$

- 中间矩阵大小为 (m, n)——与梯度相同
- 简单、理论成熟
- 无法利用 Gram 矩阵对称性

### Gram 空间 NS（Dao-AILab）

重新公式化，在 Gram 矩阵 R = X @ X^T（m < n 时大小为 m x m）上迭代：

$$
Z_k = b_k R_k + c_k R_k^2
$$
$$
Q_{k+1} = Q_k Z_k + a_k Q_k \quad \text{（累积乘积）}
$$
$$
R_{k+1} \text{ 从 } R_k \text{ 和 } Z_k \text{ 演化}
$$

最终输出：$X_{\text{out}} = Q \cdot X$

**优势：**

- 中间矩阵为 (m, m)（当 m < n 时）——可以显著更小
- R 是对称的 → SYRK 内核节省 50% 的 tile
- 支持重启机制以提升数值稳定性
- TP 兼容：R 可分解为本地 Gram 矩阵之和

## 精度流水线

所有变体使用相同的精度策略：

1. **fp32 归一化**：`X = G.float() / (G.norm() + eps)` — 确保谱范数计算准确
2. **fp16 迭代**：`X = X.half()` — 10 位尾数带来更低的逐步舍入误差

!!! info "为什么用 fp16 而不是 bf16？"
    归一化后，数值被约束在 [0, 1] 附近。fp16 的 10 位尾数在此范围内比 bf16 的 7 位尾数精度更高。fp16 较小的动态范围不成问题，因为归一化后数值已被约束。

## 系数集

DMuon 内置两套系数集，均提供 5 步 NS 迭代：

### POLAR_EXPRESS_COEFFICIENTS（默认）

来自 [Polar Express 论文](https://arxiv.org/pdf/2505.16932)，带 1.05 安全系数：

```python
POLAR_EXPRESS_COEFFICIENTS = [
    (7.8926, -20.3805, 14.9388),
    (3.9115, -2.5444, 0.4704),
    (3.7607, -2.5120, 0.4762),
    (3.1604, -2.1476, 0.4402),
    (2.1911, -1.4409, 0.3614),
]
```

### YOU_COEFFICIENTS

来自 [@YouJiacheng](https://x.com/YouJiacheng/status/1905861218138804534)：

```python
YOU_COEFFICIENTS = [
    [4.0848, -6.8946, 2.9270],
    [3.9505, -6.3029, 2.6377],
    [3.7418, -5.5913, 2.3037],
    [2.8769, -3.1427, 1.2046],
    [2.8366, -3.0525, 1.2012],
]
```

使用不同系数集：

```python
import dmuon

# 传递给单独的 NS 调用
update = dmuon.newton_schulz(G, coefficients=dmuon.YOU_COEFFICIENTS)
```

!!! note "说明"
    Muon 优化器始终使用 `POLAR_EXPRESS_COEFFICIENTS`（默认）。自定义系数可直接传递给 NS 函数用于实验。

## 重启机制

Gram 空间 NS 包含**重启**机制（来自 Dao-AILab）：在指定迭代处，将累积乘积 Q 应用到 X 上，并从头重新计算 Gram 矩阵 R。

这可以防止数值漂移在 Gram 演化方程中逐步累积。

默认重启位置：第 2 次迭代（即步骤 0 和 1 完成后，步骤 2 前重启）。

```python
# 自定义重启
update = dmuon.newton_schulz(G, restart_iterations=[2])  # 默认
update = dmuon.newton_schulz(G, restart_iterations=[1, 3])  # 更多重启
```

## SYRK 加速

Gram 矩阵 R = X @ X^T 是对称的。DMuon 的 CuteDSL SYRK 内核利用这一点：

- 仅计算 R 的下三角
- 将下三角镜像到上三角
- 与通用 GEMM 相比节省约 50% 的 tile

这适用于所有 Gram 空间变体（`newton_schulz`、`gram_newton_schulz`、`gram_newton_schulz_local`）。直接空间变体不受益于 SYRK。

检查当前后端：

```python
print(dmuon.get_ns_backend())
# "syrk_sm80"  — CuteDSL SYRK 内核（SM80+）
# "compiled"   — @torch.compile 后备方案（任意 GPU）
```

## TP 路由总结

当 Muon 优化器遇到 TP 分片参数时，路由到对应的 NS 变体：

```
TP 参数?
├── 否 → newton_schulz()（本地 Gram NS）
└── 是
    ├── per_head_ns=True 且 Shard(0) 且 full_m < full_n
    │   → newton_schulz()（逐头，零 TP 通信）
    ├── block_diagonal_ns=True
    │   → gram_newton_schulz(..., block_diagonal=True)（零 TP 通信）
    └── 其他
        → gram_newton_schulz(..., shard_dim=dp.shard_dim)（精确，TP all-reduce）
```
