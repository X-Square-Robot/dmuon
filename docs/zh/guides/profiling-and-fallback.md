# 性能分析与 Fallback

!!! tip "TL;DR"
    设置 `DMUON_REPLICATE_PROFILE=1` 可以收集每个 group 的 wait 时间直方图，用于诊断
    异步 replicate broadcast 的延迟。Fallback 协议会在某个 group 持续阻塞（连续 3 步超
    过 100 μs）后自动降级为同步模式，通过 `dmuon.reset_replicate_fallback(model)` 重置。

---

## Fallback 协议

异步 replicate broadcast（Phase C）把 post-step 的 IB 传输隐藏进下一次 forward 的计算里。
当 broadcast 能在下一次用到该层之前完成，效果最好。如果 IB 链路慢或拥塞，broadcast 可能
来不及完成，forward hook 会阻塞——比直接用同步模式还慢。

DMuon 的 fallback 协议按 group 监控这个情况，在某个 group 持续缓慢时自动降级为同步。

相关常量定义在 `dmuon/group.py`，可在模块级别访问：

| 常量 | 默认值 | 含义 |
|---|---|---|
| `REPLICATE_WAIT_THRESHOLD_US` | `100.0` μs | blocked wait 超过此值，慢步骤计数加一 |
| `REPLICATE_FALLBACK_CONSECUTIVE_STEPS` | `3` | 连续慢步骤达到此数，group 降级为同步 |

降级是**单向**的：一旦某个 group 退到同步，就会一直保持，直到手动重置：

```python
import dmuon
dmuon.reset_replicate_fallback(model)
```

Fallback 监控仅在 `DMUON_REPLICATE_PROFILE=1` 时生效，因为计时需要 CUDA event 同步。
生产环境不设置环境变量时，没有任何计时开销，fallback 也不会触发（所有 group 保持异步）。

---

## 环境变量

| 变量 | 值 | 效果 |
|---|---|---|
| `DMUON_REPLICATE_PROFILE` | 未设置 / `0` | 完全禁用，热路径零开销 |
| `DMUON_REPLICATE_PROFILE` | `1` | 收集每个 group 的 wait 时间样本；`replicate_profile_report()` 输出表格 |
| `DMUON_REPLICATE_PROFILE` | `2` | level 1 的基础上，在 dispatch 和 wait 阶段插入 NSight range marker |

基本用法：

```bash
DMUON_REPLICATE_PROFILE=1 torchrun --nproc_per_node=4 train.py
```

在训练脚本里，训练循环结束后（或固定步数后）在 rank 0 打印报告：

```python title="train.py"
import dmuon

# ... 初始化和训练循环 ...

# 打印每个 group 的 wait 直方图（仅 rank 0；其他 rank no-op）
dmuon.replicate_profile_report()
```

---

## 解读报告

示例输出：

```
==============================================================================
[DMUON_REPLICATE_PROFILE] per-group wait time summary (μs)
==============================================================================
                         group     n      mean       p50       p90       p99       max
                 ----------------------------------------
                layers.0.mlp    100     14.22     13.80     18.40     22.30     25.10
                layers.1.mlp    100     18.70     17.90     24.10     28.40     31.80
               layers.10.mlp    100     82.10     79.30    118.40    145.20    312.00
               layers.11.mlp    100     91.50     88.60    132.10    178.90    520.40
                layers.2.mlp    100     15.50     14.90     19.20     23.10     28.80
                     ...
==============================================================================
```

**列说明**：`n` = 样本数；`mean` / `p50` / `p90` / `p99` / `max` = wait 时间（微秒）。

**经验法则**：

- 所有 group 的 `p90 < 100 μs`：异步隐藏效果良好，无需操作。
- `p99` 明显大于 `p90`（如 p90 = 25 μs，p99 = 150 μs）：偶发性 IB 抖动或短暂计算不均衡，
  频率低的话可以接受。
- 多个 group 的 `p90 > 100 μs`：replicate broadcast 持续来不及完成。考虑切换到 DMuon-Z3
  （如果当前是 Z2）以减少 IB 流量，或提高阈值接受 sync fallback。
- 任何 group 的 `max > 1 ms`：排查 IB 饱和或 host 端调度干扰（如 forward 中某处大型
  all-gather 与之重叠）。

当 fallback 协议触发时，报告末尾会出现 `Fallback events` 段落，显示哪些 group 在哪一步
发生了降级。

---

## 调整阈值

在训练开始前用 Python 修改阈值。这些常量是 `dmuon.group` 的模块级变量，修改立即对后续所有
`_update_replicate_fallback` 调用生效：

```python title="tune_thresholds.py"
import dmuon._backends.fsdp2.group as g

# 把阈值提高到 250 μs，才算一次"慢步骤"
g.REPLICATE_WAIT_THRESHOLD_US = 250.0      # 默认：100.0

# 连续 5 次慢步骤才降级为同步
g.REPLICATE_FALLBACK_CONSECUTIVE_STEPS = 5 # 默认：3
```

!!! warning "模块全局状态"
    这些常量影响当前进程中所有 `DedicatedParamGroup` 实例。如果在同一个 Python 进程里跑多
    个实验，要在实验之间重置，或者用独立进程分隔。

典型调优流程：

1. 用 `DMUON_REPLICATE_PROFILE=1` 默认阈值跑 100 步，查看报告。
2. 如果最慢的几个 group 的 `p90` 在 150 μs 左右，正式跑前把
   `REPLICATE_WAIT_THRESHOLD_US` 调到 200。
3. 如果 group 频繁 fallback（报告里 fallback 事件很多），要么继续上调阈值，要么在
   `Muon` 构造函数里直接设 `replicate_async=False` 接受同步模式。

---

## NSight 工作流

Level 2（`DMUON_REPLICATE_PROFILE=2`）会在 replicate broadcast 的 dispatch 和 wait 阶段
插入 NSight range marker，让你把 CUDA 时间线和 wait 直方图关联起来。

```bash
DMUON_REPLICATE_PROFILE=2 nsys profile \
    --trace=cuda,nvtx \
    --output=dmuon_trace \
    torchrun --nproc_per_node=4 train.py
```

在 NSight Systems 里打开 `dmuon_trace.nsys-rep`，过滤 `DMUON::` NVTX 命名空间，可以看到
三种 marker：

| Marker | 含义 |
|---|---|
| `DMUON::replicate_dispatch` | broadcast 已在 replicate stream 上入队 |
| `DMUON::replicate_wait` | 当前 stream 阻塞等待 broadcast 完成 |
| `DMUON::replicate_effective` | dispatch 结束到 wait 开始之间的间隔——broadcast 被隐藏的时间 |

异步隐藏良好时，`replicate_effective` 区间应覆盖前几个 forward 层的 kernel 计算，
`replicate_wait` 应是层边界处一根细条。

---

## API 速查

| 函数 | 说明 |
|---|---|
| `dmuon.replicate_profile_report()` | rank 0 打印每个 group 的 wait 直方图；其他 rank 和 profiling 未开启时 no-op。 |
| `dmuon.reset_replicate_fallback(model)` | 清除所有 group 的 sync-fallback 标志，重新开启异步模式。 |
| `dmuon.wait_all_replicate_broadcasts(model)` | 立即 drain 所有 pending 的异步 replicate broadcast。保存 checkpoint 或在 forward hook 之外读取 `_owned_data` 前调用。 |

---

## 常见诊断

### "训练比预期慢"

开启 `DMUON_REPLICATE_PROFILE=1` 跑 200 步，检查每个 group 的 `p90`。如果多个 group 的
`p90 > 100 μs`，说明异步隐藏失效：

- 当前是 DMuon-Z2：切换到 DMuon-Z3（`reshard_after_forward=True`），减少每步 IB 字节数。
- 已经是 DMuon-Z3：检查 FSDP2 的 forward all-gather 是否在打满 IB 带宽。用 NSight level 2
  查找与 replicate broadcast 重叠的大型 collective。
- 最后手段：在 `Muon` 里设 `replicate_async=False`，去掉异步开销，先用同步模式做基准。

### "训练中途 loss 发散或行为异常"

极不可能由 fallback 协议引起（异步和同步路径 bit-identical）。在 `replicate_profile_report()`
里确认 fallback 事件数量。即使多个 group 已降级为同步，optimizer state 仍然完全正确——
只是那些 group 不再隐藏 IB 流量了。

如果怀疑是异步路径导致的正确性问题，用 `replicate_async=False` 跑 50 步对比 loss 曲线。
如果一致，问题与异步路径无关。

### "Replicate broadcast 完全没有与 forward 重叠"

NSight 里 `replicate_effective` 近乎为零，说明 forward 太短，IB 传输藏不进去。选项：

- 增大 batch size，延长 forward 计算时间。
- 显式设 `replicate_async=False`——同步模式能避免"发起异步、立刻阻塞"的额外开销。
- 用更大的 `shard_size` 配置 HSDP，让每层 forward 更长。

### "某个 group 反复触发 fallback"

从报告的 fallback events 部分找出 group 名称。如果总是同一个（如某个大 embedding 或宽
MLP），说明那个 group 的 broadcast 天生比其他的慢。把 `REPLICATE_WAIT_THRESHOLD_US` 调到
略高于该 group `p90` 的值，或通过 `dedicate_params` 的 `predicate` 把该参数排除在
dedicated ownership 之外。

---

## 相关文档

- [HSDP 训练](hsdp.md) —— HSDP 场景下的异步 broadcast
- [Z2 与 Z3 模式](z2-z3-modes.md) —— 减少 IB 字节以帮助异步隐藏
- [故障排查](../troubleshooting.md) —— 一般训练问题
- [API 文档](../reference/api.md) —— 完整函数签名
