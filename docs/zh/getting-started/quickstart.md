# 快速开始

!!! tip "TL;DR"
    三行设置：`dedicate_params` → `fully_shard` → `dmuon.Muon`。
    从下方标签页选择训练拓扑，粘贴到 `train.py`，五分钟内跑通。

---

## 第一步 — 安装

```bash
git clone https://github.com/StarrickLiu/dmuon && cd dmuon
pip install -e .
```

SYRK 加速及环境要求详见[安装指南](installation.md)。

---

## 第二步 — 选择训练拓扑

三种变体共用同一个模型定义：

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

=== "单节点 — DDP 风格（一维 Mesh，无 fully_shard）"

    最简配置：一维 Mesh，仅专属所有权，无 FSDP2 参数分片。
    适合快速验证或参数可放入单卡显存的场景。

    ```python title="train_ddp.py"
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    import dmuon
    from model import TinyMLP

    def main() -> None:
        dist.init_process_group("nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)

        mesh = init_device_mesh("cuda", (world_size,))
        torch.manual_seed(42)
        model = TinyMLP().cuda()

        dmuon.dedicate_params(
            model, mesh,
            predicate=lambda n, p: "proj" in n and p.ndim == 2,
        )
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

=== "单节点 — FSDP2（默认 Z3 模式）"

    在专属所有权之上添加 `fully_shard`。非 Muon 参数（LayerNorm）
    使用 FSDP2 ZeRO-3 分片。`import dmuon` 安装的 monkey-patch 使
    专属参数自动跳过 FSDP2。

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

---

## 刚才发生了什么？

1. **`dedicate_params()`** — 均衡 LPT 划分：每个投影参数分配给单一所有者
   rank。所有者存储完整参数，其他 rank 持有占位符。Hook 在层级别注册。

2. **`fully_shard()`** — FSDP2 分片非专属参数（LayerNorm）。
   专属参数由 `import dmuon` 的 monkey-patch 自动跳过。

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
