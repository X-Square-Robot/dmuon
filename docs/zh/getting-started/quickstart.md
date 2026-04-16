# 快速开始

本指南带你在 5 分钟内从零到跑通 DMuon 分布式训练。

## 前提条件

- 已安装 DMuon（[安装指南](installation.md)）
- 单节点上至少 2 张 GPU

## 最小可运行示例

创建 `train_minimal.py`：

```python
"""最小 DMuon 训练示例"""
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

import dmuon  # (1)!


# 简单模型
class TinyMLP(nn.Module):
    def __init__(self, d=512, ff=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "gate_proj": nn.Linear(d, ff, bias=False),  # (2)!
                "up_proj": nn.Linear(d, ff, bias=False),
                "down_proj": nn.Linear(ff, d, bias=False),
                "ln": nn.LayerNorm(d),
            })
            for _ in range(4)
        ])
        self.head = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            h = layer["ln"](x)
            x = x + layer["down_proj"](layer["gate_proj"](h) * layer["up_proj"](h))
        return self.head(x).sum()


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    mesh = init_device_mesh("cuda", (world_size,))

    torch.manual_seed(42)
    model = TinyMLP().cuda()

    # --- DMuon 设置（3 行代码）---
    dmuon.dedicate_params(  # (3)!
        model, mesh,
        predicate=lambda n, p: "proj" in n and p.ndim == 2,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)  # (4)!
    fully_shard(model, mesh=mesh)

    optimizer = dmuon.Muon(  # (5)!
        model, lr=0.02, momentum=0.95, ns_steps=5,
        adamw_lr=1e-3,
    )

    # --- 训练循环 ---
    for step in range(20):
        optimizer.zero_grad()
        x = torch.randn(4, 512, device="cuda")
        loss = model(x)
        loss.backward()
        optimizer.step()

        if rank == 0 and step % 5 == 0:
            print(f"step {step:3d}  loss={loss.item():.4f}")

    dist.destroy_process_group()
    if rank == 0:
        print("完成！")


if __name__ == "__main__":
    main()
```

1. `import dmuon` 会自动 patch FSDP2，使 `fully_shard()` 跳过专属参数。
2. `proj` 层是 2D 矩阵参数——使用 Muon + Newton-Schulz 更新。LayerNorm 是 1D——使用 AdamW。
3. 标记哪些参数使用专属所有权。`predicate` 选中 2D 投影层。
4. 照常使用 FSDP2。专属参数会被自动跳过。
5. `dmuon.Muon` 同时管理两种参数：专属参数用 Newton-Schulz，其余用 AdamW。

## 运行

```bash
torchrun --nproc_per_node=4 train_minimal.py
```

预期输出：

```
step   0  loss=3.2145
step   5  loss=1.0832
step  10  loss=0.4217
step  15  loss=0.1583
完成！
```

## 刚才发生了什么？

在那 3 行设置代码中，DMuon 做了以下事情：

1. **`dedicate_params()`** — 使用均衡分配算法将每个投影层分配给一个所有者 rank。所有者存储完整参数；其他 rank 持有空占位符。

2. **`fully_shard()`** — FSDP2 照常分片所有*非专属*参数（LayerNorm）。专属参数被自动跳过。

3. **`dmuon.Muon()`** — 创建优化器：对所有者的专属参数运行 Newton-Schulz，对每个 rank 的 FSDP2 分片运行 AdamW。

训练中每个前向/反向步骤：

- **前向**：所有者广播完整参数到所有 rank
- **反向**：梯度 reduce 回所有者
- **Step**：所有者对其参数运行 Newton-Schulz；所有 rank 对其 FSDP2 分片运行 AdamW

无 all-gather。无冗余 NS 计算。

## 下一步

要理解*为什么*这样工作以及*如何*与 FSDP2 组合，请阅读[核心概念](concepts.md)。
