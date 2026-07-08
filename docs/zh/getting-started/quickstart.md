# 快速开始

!!! tip "TL;DR"
    三行设置：`dedicate_params` → `fully_shard` → `dmuon.Muon`。
    从下方标签页选择训练拓扑，粘贴到 `train.py`，五分钟内跑通。

---

## 第一步 — 安装

```bash
git clone https://github.com/X-Square-Robot/dmuon && cd dmuon
pip install -e .
```

SYRK 加速及环境要求详见[安装指南](installation.md)。

---

## 第二步 — 选择训练拓扑

两种变体共用同一个模型定义：

```python title="model.py（三个标签页共用）"
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, d: int = 512, ff: int = 2048, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "gate_proj": nn.Linear(d, ff, bias=False),
                "up_proj":   nn.Linear(d, ff, bias=False),
                "down_proj": nn.Linear(ff, d, bias=False),
                "ln":        nn.LayerNorm(d),
            })
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer["ln"](x)
            x = x + layer["down_proj"](layer["gate_proj"](h) * layer["up_proj"](h))
        return self.head(x).sum()
```

=== "单节点 — FSDP2（默认 Z3 模式）"

    在专属所有权之上叠加 `fully_shard`。此处的非专属参数是
    `ln.weight`、`ln.bias`（一维）以及 `head.weight`（二维，但名称
    不含 `proj`，被谓词排除），由 FSDP2 ZeRO-3 分片。
    `import dmuon` 安装的 monkey-patch 会让 `fully_shard()` 跳过
    已被 `dedicate_params` 标记的参数，二者管理不相交的参数集。

    ```python title="train_fsdp2.py"
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    import dmuon
    from model import TinyMLP

    def main() -> None:
        dist.init_process_group("nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)

        mesh = init_device_mesh("cuda", (world_size,))
        torch.manual_seed(42)
        model = TinyMLP().cuda()

        # dedicate_params 必须在 fully_shard 之前
        dmuon.dedicate_params(
            model, mesh,
            predicate=lambda n, p: "proj" in n and p.ndim == 2,
        )
        for layer in model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)

        optimizer = dmuon.Muon(model, lr=0.02, momentum=0.95, ns_steps=5,
                               adamw_lr=1e-3)

        for step in range(20):
            optimizer.zero_grad()
            loss = model(torch.randn(4, 512, device="cuda"))
            loss.backward()
            optimizer.step()
            if rank == 0 and step % 5 == 0:
                print(f"step {step:3d}  loss={loss.item():.4f}")

        dist.destroy_process_group()

    if __name__ == "__main__":
        main()
    ```

=== "多节点 — HSDP（二维 Mesh）"

    通过二维 `(replicate, shard)` Mesh 跨节点扩展。传入
    `replicate_mesh` 以启用两阶段 reduce 和异步 forward 隐藏广播。
    `replicate_async=True` 为默认值。

    ```python title="train_hsdp.py"
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    import dmuon
    from model import TinyMLP

    def main() -> None:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)

        replicate_size = 2
        shard_size = dist.get_world_size() // replicate_size
        hsdp = init_device_mesh(
            "cuda", (replicate_size, shard_size),
            mesh_dim_names=("replicate", "shard"),
        )
        torch.manual_seed(42)
        model = TinyMLP().cuda()

        dmuon.dedicate_params(
            model, hsdp["shard"],
            predicate=lambda n, p: "proj" in n and p.ndim == 2,
            replicate_mesh=hsdp["replicate"],
        )
        for layer in model.layers:
            fully_shard(layer, mesh=hsdp)
        fully_shard(model, mesh=hsdp)

        optimizer = dmuon.Muon(model, lr=0.02, momentum=0.95, ns_steps=5,
                               adamw_lr=1e-3, replicate_async=True)

        for step in range(20):
            optimizer.zero_grad()
            loss = model(torch.randn(4, 512, device="cuda"))
            loss.backward()
            optimizer.step()
            if rank == 0 and step % 5 == 0:
                print(f"step {step:3d}  loss={loss.item():.4f}")

        dist.destroy_process_group()

    if __name__ == "__main__":
        main()
    ```

---

## 第三步 — 运行

```bash title="单节点（8 张 GPU）"
torchrun --nproc_per_node=8 train_fsdp2.py
```

```bash title="多节点 HSDP（2 节点 × 8 张 GPU）"
torchrun \
  --nnodes=2 --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train_hsdp.py
```

预期输出（rank 0）：

```
step   0  loss=3.2145
step   5  loss=1.0832
step  10  loss=0.4217
step  15  loss=0.1583
```

!!! tip "确认快路径"
    在脚本顶部加一行即可确认实际运行的 SYRK 内核——适合 bug report 与新集群上线自检：

    ```python
    import dmuon
    print(dmuon.get_ns_backend())
    # Gram NS · kernel=cute_sm80 (SM80, DMuon internal)   ← A100/A800 快路径
    # Gram NS · kernel=cublas (SM80, universal fallback)   ← CuteDSL 未编译
    # Gram NS · kernel=quack (SM90, Tri Dao quack)         ← H100 快路径（Phase B-H）
    ```

    完整的自动检测阶梯以及 `kernel=` / `DMUON_NS_KERNEL` 覆盖方式见
    [后端分发](../reference/newton-schulz.md#backend-dispatch)。

---

## 刚才发生了什么？

1. **`dedicate_params()`** — 均衡 LPT 划分：每个投影参数分配给单一所有者
   rank。所有者存储完整参数，其他 rank 持有占位符。Hook 在层级别注册。

2. **`fully_shard()`** — FSDP2 分片剩余的非专属参数
   （`ln.weight`、`ln.bias`、`head.weight`）。`import dmuon` 安装的
   monkey-patch 会让 `fully_shard()` 跳过已被 `dedicate_params`
   标记的参数，DMuon 与 FSDP2 管理不相交的参数集。

3. **`dmuon.Muon()`** — 对所有者的专属参数运行 Newton-Schulz，
   对 FSDP2 分片参数运行 AdamW。无需 all-gather。

在 HSDP 模式下，`replicate_mesh` 启用两阶段 reduce 和异步步后广播。
训练循环的其余部分保持不变。

---

## 另请参见

- [核心概念](concepts.md) — 专属所有权的工作原理
- [HSDP 指南](../guides/hsdp.md) — 二维 Mesh、异步模式和 Fallback
- [训练指南](../guides/training.md) — 生产环境完整训练流程
- [API 文档](../reference/api.md) — 完整函数签名
