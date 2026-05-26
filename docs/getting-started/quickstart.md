# Quick Start

!!! tip "TL;DR"
    Three setup lines: `dedicate_params` → `fully_shard` → `dmuon.Muon`.
    Pick the tab below for your topology and paste into `train.py`.
    Run in under 5 minutes.

---

## Step 1 — Install

```bash
git clone https://github.com/StarrickLiu/dmuon && cd dmuon
pip install -e .
```

See [Installation](installation.md) for SYRK acceleration and requirements.

---

## Step 2 — Choose your topology

Both variants use the same model definition:

```python title="model.py (shared across tabs)"
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

=== "Single Node — FSDP2 (default Z3 mode)"

    Adds `fully_shard` on top of dedicated ownership. The non-dedicated
    params here are `ln.weight`, `ln.bias` (1D), and `head.weight` (2D
    but excluded by the `"proj" in n` predicate) — FSDP2 ZeRO-3 shards
    them. The monkey-patch installed at `import dmuon` makes
    `fully_shard()` skip any parameter already claimed by
    `dedicate_params`, so the two systems partition disjoint sets.

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

        # dedicate_params BEFORE fully_shard — order matters
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

=== "Multi-Node — HSDP (2D mesh)"

    Scale across nodes with a 2D `(replicate, shard)` mesh. Pass
    `replicate_mesh` to enable two-stage reduce and async forward-hidden
    broadcast. `replicate_async=True` is the default.

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

## Step 3 — Run it

```bash title="Single node (8 GPUs)"
torchrun --nproc_per_node=8 train_fsdp2.py
```

```bash title="Multi-node HSDP (2 nodes × 8 GPUs)"
torchrun \
  --nnodes=2 --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train_hsdp.py
```

Expected output (rank 0):

```
step   0  loss=3.2145
step   5  loss=1.0832
step  10  loss=0.4217
step  15  loss=0.1583
```

!!! tip "Confirm the fast path"
    Add a one-liner at the top of your script to verify which SYRK
    kernel will run — handy for bug reports and sanity-checking new
    cluster builds:

    ```python
    import dmuon
    print(dmuon.get_ns_backend())
    # Gram NS · kernel=cute_sm80 (SM80, DMuon internal)   ← A100/A800 fast path
    # Gram NS · kernel=cublas (SM80, universal fallback)   ← CuteDSL not built
    # Gram NS · kernel=quack (SM90, Tri Dao quack)         ← H100 fast path (Phase B-H)
    ```

    See [Backend dispatch](../reference/newton-schulz.md#backend-dispatch)
    for the full auto-detection ladder and the `kernel=` / `DMUON_NS_KERNEL`
    overrides.

---

## What just happened?

1. **`dedicate_params()`** — Balanced LPT partition: each projection parameter
   assigned to one owner rank. Owners store the full parameter; others hold
   placeholders. Forward/backward hooks registered at layer level.

2. **`fully_shard()`** — FSDP2 shards the remaining non-dedicated params
   (`ln.weight`, `ln.bias`, `head.weight`). The monkey-patch installed at
   `import dmuon` makes `fully_shard()` skip any parameter already claimed
   by `dedicate_params`, so DMuon and FSDP2 partition disjoint sets.

3. **`dmuon.Muon()`** — Newton-Schulz on owned dedicated params; AdamW on
   FSDP2-sharded params. No all-gather needed.

In HSDP mode, `replicate_mesh` enables two-stage reduce and async post-step
broadcast. The training loop is otherwise unchanged.

---

## See Also

- [Core Concepts](concepts.md) — why dedicated ownership works
- [HSDP Guide](../guides/hsdp.md) — 2D mesh and async mode
- [Training Guide](../guides/training.md) — production workflow with all options
- [API Reference](../reference/api.md) — complete signatures
