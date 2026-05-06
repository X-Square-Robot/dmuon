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

## 后端分发（Backend dispatch）

Newton-Schulz 有两条正交选择轴：**算法**（Gram 或 direct）以及底层 **SYRK 内核**实现。
DMuon 自动分发两者，也分别暴露为用户可显式覆盖的参数。

### 双轴结构

```
┌─────────────────────────────────────────────────────────────┐
│  User API:  dmuon.NewtonSchulz(                             │
│                 backend="gram",     ← 轴 1：算法             │
│                 kernel="auto",       ← 轴 2：SYRK 内核        │
│             )                                                │
├─────────────────────────────────────────────────────────────┤
│  轴 1 — 算法（Algorithm）                                    │
│     "gram"    → Gram 空间 NS + SYRK 操作 + 重启机制（默认）  │
│     "direct"  → 经典参数空间 NS                               │
├─────────────────────────────────────────────────────────────┤
│  轴 2 — SYRK 内核后端                                        │
│     "auto"       → 自动选择当前 GPU 的最佳路径（默认）        │
│     "quack"      → Tri Dao quack（SM90+，软依赖，opt-in）    │
│     "cute_sm80"  → DMuon 内置 CuteDSL 内核（仅 SM80/87）     │
│     "cublas"     → torch.mm / torch.addmm（通用后备）        │
└─────────────────────────────────────────────────────────────┘
```

两轴互相正交——任意 `backend` × `kernel` 组合均合法。直接空间 NS 不使用 SYRK，
因此 `backend="direct"` 时 `kernel` 参数为无效 no-op。

### 自动检测阶梯

`kernel="auto"`（默认）下，DMuon 按如下阶梯为当前设备选择最快可用后端：

```
在 import 时探测 SM 版本 ─►
    ┌── SM ≥ 90  ─── quack 已安装？  ── 是 ──► quack
    │                              │
    │                              └── 否 ──► cublas + 警告
    │
    ├── SM 80/87 ─── cute_sm80 已编译？── 是 ──► cute_sm80
    │                              │
    │                              └── 否 ──► cublas
    │
    └── SM < 80  ─────────────────────────► cublas
```

分发遵循 **graceful degradation** 原则：`kernel="auto"` 永远选出一条能跑的路径，
启动时日志打印实际选中的内核。在 SM80 设备上显式指定 `kernel="quack"` 会**立即报错**
并给出安装提示。

### 解析优先级

多档开关同时存在时的优先级：

```
NewtonSchulz(kernel=...) 显式参数           ← 最高（永远获胜）
          │
          ▼ 仅当 kernel 仍为 "auto" 时
DMUON_NS_KERNEL 环境变量
          │
          ▼ 仅当环境变量未设置时
deterministic=True                        ← 旧版别名，映射为 "cublas"
          │
          ▼
自动检测结果
```

若同时设置 `deterministic=True` 和 `kernel="cute_sm80"`，DMuon 会发出 warning 并
按显式 `kernel` 生效。

### 查询当前后端

```python
import dmuon

# 人类可读的一行概要——适合启动日志
print(dmuon.get_ns_backend())
# "Gram NS · kernel=cute_sm80 (SM80, DMuon internal)"
# "Gram NS · kernel=quack (SM90, Tri Dao quack)"
# "Gram NS · kernel=cublas (SM80, universal fallback)"

# 完整诊断字典——适合 bug report / 程序化检查
print(dmuon.get_backend_status())
# {
#   "sm_version": 80,
#   "auto_choice": "cute_sm80",
#   "quack_available": False,
#   "cute_sm80_available": True,
#   "cublas_always_available": True,
# }
```

### 强制指定内核

```python
# 强制 cuBLAS 以获得跨运行的 bit-exact 可复现性
ns = dmuon.NewtonSchulz(kernel="cublas")
ns = dmuon.NewtonSchulz(deterministic=True)   # 旧版等价写法

# 强制 SM80 CuteDSL 内核（若未编译则构造时抛错）
ns = dmuon.NewtonSchulz(kernel="cute_sm80")

# 集群级覆盖（只有当代码里写的是 "auto" 时生效）
# export DMUON_NS_KERNEL=cublas
```

!!! info "quack 后端"
    `quack` SYRK 后端在 SM90+ 设备上、已安装 `quack-kernels` 软依赖（`pip install dmuon[quack]`）
    时自动启用。已在 B300（SM103）上端到端验证——详见
    `docs/internal/benchmarks/quack_smoke_b300.md`（含 correctness 矩阵与性能拐点：
    quack 从 M ≈ 4096 开始占优，M ≥ 8192 时领先显著）。

    运行时 circuit-breaker `dmuon.kernels.syrk_quack.ADAPTER_READY`
    可设为 `False` 紧急禁用 quack 路径（无需卸载包），届时 `kernel="auto"`
    会回退到 `cublas`。

    `get_backend_status()["auto_choice"]` 永远反映真正会跑的 kernel，
    一眼看清 ground truth。

---

## TP 路由

NS 核函数（`newton_schulz`、`gram_newton_schulz`、`direct_newton_schulz`）
是**TP 无感**的：它们总是对完整（未分片）矩阵做运算，不接受 `tp_group`
参数。对于 TP-sharded 参数，DMuon runtime 在调用 NS 之前通过 TP gather
把完整矩阵汇聚到指定的 TP owner，NS 运行完后再 scatter 回去：

```
DP reduce → TP gather（dist.gather on reduce_stream）→
    TP owner 在完整 (m, n) 矩阵上跑 Newton-Schulz →
TP scatter（dist.scatter on replicate_broadcast_stream）→
    replicate broadcast
```

对于任何 `device_mesh` 含有非 DP 轴的 `DTensor` 参数，这套流程会**自动
触发**——`dmuon.Muon` 不需要任何显式 TP 开关。TP owner 由
`compute_balanced_assignment` 内部的确定性 LPT 策略选定，在保持 loss 轨迹
一致的同时，把 TP-sharded 的完整矩阵计算分散到本地 TP ranks。

实际影响：

* **NS 精度与是否 TP 无关**——kernel 永远看到完整矩阵。
* **每个 TP-sharded 参数的额外通信**：一次 `dist.gather` + 一次
  `dist.scatter`，每步字节量 `(T−1)/T · |p|`。两者都跑在 DMuon 专用
  comm stream 上，toy 3D HSDP×TP 上实测和 backward compute **~100%**
  overlap。
* **非 TP 参数行为不变**。

具体配置、完整 lifecycle 以及 sync / async 语义见
[TP 支持指南](../guides/tp-support.md)。

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
