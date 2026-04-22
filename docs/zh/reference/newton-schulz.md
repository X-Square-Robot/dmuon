# Newton-Schulz 变体

!!! tip "TL;DR"
    DMuon 提供两种 NS 后端：**Gram 空间**（`"gram"`，默认）和**直接空间**
    （`"direct"`）。Gram 空间更快（SYRK 内核、重启机制、更小的中间矩阵），
    是所有生产场景的首选。直接空间是经典的 Muon/Moonlight 算法，适合基线
    对比和小矩阵场景。两种后端均支持自定义 `(a, b, c)` 系数集
    （默认 `POLAR_EXPRESS_COEFFICIENTS`，可替换为 `YOU_COEFFICIENTS`）。

---

## 概览

| 函数 | 空间 | TP 支持 | SYRK 加速 | 重启 | 适用场景 |
|---|---|---|---|---|---|
| `newton_schulz()` | Gram | 否（本地） | 是 | 是 | **默认** — 单 rank 或纯 DP |
| `gram_newton_schulz()` | Gram | 是 | 是 | 是 | **TP 参数** — 精确或块对角 |
| `NewtonSchulz("gram")` | Gram | 路由 | 是 | 是 | 传入 `Muon(ns_backend=...)` |
| `NewtonSchulz("direct")` | 直接 | 否 | 否 | 否 | 基线 / 消融实验 |
| `direct_newton_schulz()` | 直接 | 否 | 否 | 否 | 直接函数调用 |

---

## 直接空间 NS（经典）

来自 Muon（Jordan et al., 2024）和 Moonlight 的标准公式，在完整 (m, n)
矩阵上迭代：

$$
X_{k+1} = a_k X_k + b_k (X_k X_k^T) X_k + c_k (X_k X_k^T)^2 X_k
$$

特性：

- 中间矩阵大小为 (m, n)——与梯度相同
- 无对称性利用；每步为通用 GEMM 代价
- 无重启机制
- 简单、理论成熟，适合基线对比

使用 `NewtonSchulz("direct")` 或直接调用 `direct_newton_schulz()`。

---

## Gram 空间 NS（Dao-AILab）

重新公式化为在 Gram 矩阵 $R = X X^T$（大小 (m, m)）上迭代，
改编自 [Dao-AILab/gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz)：

$$
Z_k = b_k R_k + c_k R_k^2
$$
$$
Q_{k+1} = Z_k Q_k + a_k Q_k \quad (\text{累积乘积})
$$

$R$ 由 $R_k$ 和 $Z_k$ 经递推演化；在重启步骤 $Q$ 被应用到 $X$ 并从头
重新计算 $R$。最终输出：$X_{\text{out}} = Q \cdot X$。

**相比直接空间的优势：**

- 中间矩阵为 (m, m)；当 m < n 时（典型的宽投影层）显著更小
- $R$ 是对称的——CuteDSL SYRK 内核节省约 50% tile
- 重启机制防止数值漂移
- $R$ 可分解为本地 Gram 矩阵之和——通过 all-reduce 实现精确 TP

---

## 精度流水线

所有变体使用相同的两阶段精度策略：

1. **fp32 归一化**：`X = G.float() / (G.norm() + eps)` — 在迭代前稳定谱范数
2. **fp16 迭代**：`X = X.half()` — 10 位尾数在归一化后的 [0, 1] 范围内
   比 bf16 的 7 位尾数精度更高

!!! info "为什么用 fp16 而不是 bf16？"
    归一化后数值位于 [0, 1] 附近。fp16 更宽的尾数（10 位）在此范围内
    精度更高。归一化步骤已约束了数值范围，fp16 较小的动态范围不成问题。

---

## 系数集

DMuon 内置两套系数集，均提供 5 步 Newton-Schulz 迭代。

### POLAR_EXPRESS_COEFFICIENTS（默认）

来自 Polar Express 论文（arXiv:2505.16932），应用了 1.05 安全系数：

```python
# 应用安全系数后的近似值
POLAR_EXPRESS_COEFFICIENTS = [
    (7.893, -20.381, 14.939),
    (3.912, -2.544,  0.470),
    (3.761, -2.512,  0.476),
    (3.160, -2.148,  0.440),
    (2.191, -1.441,  0.361),
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

### 使用自定义系数集

```python
import dmuon

# 通过 NewtonSchulz 对象覆盖
ns = dmuon.NewtonSchulz("gram", coefficients=dmuon.YOU_COEFFICIENTS)
optimizer = dmuon.Muon(model, lr=0.02, ns_backend=ns)

# 或直接传给独立 NS 函数
update = dmuon.newton_schulz(G, coefficients=dmuon.YOU_COEFFICIENTS)
```

!!! note
    `Muon` 默认使用 `POLAR_EXPRESS_COEFFICIENTS`。You 系数适用于希望使用
    原始 Muon 算法形式的实验场景。

---

## 重启机制

Gram 空间 NS 包含**重启**机制，改编自 Dao-AILab/gram-newton-schulz。在
指定迭代索引处，累积乘积 $Q$ 被应用到 $X$，然后从头重新计算 Gram 矩阵
$R$，防止数值漂移在 Gram 演化递推中逐步累积。

默认重启位置：`[2]`（在第 0、1 步之后，第 2 步之前重启）。

```python
import dmuon

# 默认重启
update = dmuon.newton_schulz(G, restart_iterations=[2])

# 更激进的重启
ns = dmuon.NewtonSchulz("gram", restart_iterations=[1, 3])
```

---

## SYRK 加速

Gram 矩阵 $R = X X^T$ 是对称的。DMuon 的 CuteDSL SYRK 内核
（改编自 [Dao-AILab/quack](https://github.com/Dao-AILab/quack)）利用这一特性：

- 仅计算 $R$ 的下三角半
- 将结果镜像到上三角
- 相比通用 GEMM 节省约 50% tile

适用于所有 Gram 空间变体（`newton_schulz`、`gram_newton_schulz`）。
直接空间变体不使用 SYRK。

查询当前后端：

```python
import dmuon

print(dmuon.get_ns_backend())
# "syrk_sm80"  — CuteDSL SYRK 内核（SM80+，如 A100/H100）
# "compiled"   — @torch.compile 后备方案（任意 CUDA GPU）
```

SYRK 内核在 SM80+ GPU 且 CuteDSL 可用时自动激活，否则回退到
`@torch.compile` PyTorch 实现。

### 确定性模式

SYRK 内核由于浮点累加顺序可能在不同运行间产生非确定性结果。
如需完全可复现：

```python
ns = dmuon.NewtonSchulz(deterministic=True)
optimizer = dmuon.Muon(model, ns_backend=ns)
```

!!! warning "SYRK B != A 已知问题"
    CuteDSL SYRK 内核在 `B != A` 时（Gram 递推中的某些中间计算）存在已知的
    非确定性问题。解决方案是使用 `deterministic=True`，将所有运算路由到
    cuBLAS，代价是约 1.5x 的性能损失。此问题正在追踪中，待后续内核修复。

---

## TP 路由总结

当 `Muon` 遇到 TP 分片参数时，按如下决策树选择 NS 路径：

```
参数是带有 TP 组的 DTensor 吗？
├── 否  → NewtonSchulz.local()  （标准 Gram NS 或 direct，无通信）
└── 是
    ├── per_head_ns=True 且 Shard(0) 且 full_m < full_n
    │   → NewtonSchulz.local()   （逐头，零 TP 通信）
    ├── block_diagonal_ns=True
    │   → NewtonSchulz.tp(..., block_diagonal=True)   （零 TP 通信）
    └── 其他（默认）
        → NewtonSchulz.tp(..., shard_dim=dp.shard_dim)  （精确 Gram，TP all-reduce）
```

对于 **Shard(0)**（行分片）：迭代转置后使用 $G^T G$，可精确分解为
$\sum_i G_i^T G_i$——一次 all-reduce 即得精确 Gram。
对于 **Shard(1)**（列分片）：使用 $G G^T$，分解为 $\sum_i G_i G_i^T$。

---

## 参考文献与致谢

- **Gram Newton-Schulz** — Dao et al., 2026。博客：
  [dao-ailab.github.io/blog/2026/gram-newton-schulz/](https://dao-ailab.github.io/blog/2026/gram-newton-schulz/)。
  源码：[Dao-AILab/gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz)。
  DMuon 的 Gram NS 逻辑、逐步系数、重启机制和 SYRK 对称优化均改编自此工作。
- **SYRK 内核** — 改编自 Tri Dao 等的
  [Dao-AILab/quack](https://github.com/Dao-AILab/quack)。
- **Muon 优化器** — Jordan et al., arXiv:2502.16982, 2024。引入了 DMuon 所
  扩展的动量 + Newton-Schulz 正交化公式。
- **Polar Express 系数** — arXiv:2505.16932。
- **You 系数** — [@YouJiacheng](https://x.com/YouJiacheng/status/1905861218138804534)。

---

## 参见

- [API 文档](api.md)
- [通信成本分析](communication-cost.md)
- [训练指南](../../guides/training.md)
- [张量并行](../../guides/tp-support.md)
