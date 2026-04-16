# Example: Data Parallel Training

A complete, runnable example of DMuon with pure data parallelism (no TP).

---

## The Script

::: details Full source: `examples/basic_dp.py`

```python
--8<-- "examples/basic_dp.py"
```

:::

## Walkthrough

### Model Definition

```python
class Block(nn.Module):
    def __init__(self, d=512, ff=2048):
        super().__init__()
        self.gate_proj = nn.Linear(d, ff, bias=False)   # 2D → Muon
        self.up_proj = nn.Linear(d, ff, bias=False)      # 2D → Muon
        self.down_proj = nn.Linear(ff, d, bias=False)    # 2D → Muon
        self.ln = nn.LayerNorm(d)                        # 1D → AdamW

    def forward(self, x):
        h = self.ln(x)
        return x + self.down_proj(self.gate_proj(h) * self.up_proj(h))
```

A simple SwiGLU MLP block. The projection layers are 2D matrices (good for Muon), while LayerNorm is 1D (uses AdamW).

### DMuon Setup

```python
# Predicate: 2D projection layers → dedicated (Muon)
dmuon.dedicate_params(
    model, mesh,
    predicate=lambda n, p: "proj" in n and p.ndim == 2,
)

# FSDP2: shards LayerNorm and head across ranks
for layer in model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

# Single optimizer handles both types
optimizer = dmuon.Muon(model, lr=0.02, momentum=0.95, adamw_lr=1e-3)
```

### Training Loop

```python
for step in range(100):
    optimizer.zero_grad()
    x = torch.randn(batch_size, d_model, device="cuda")
    loss = model(x)
    loss.backward()
    optimizer.step()
```

Identical to a standard PyTorch training loop.

### Checkpoint

```python
# Save
model_sd = dmuon.get_model_state_dict(model)
optim_sd = dmuon.get_optimizer_state_dict(model, optimizer)
if dist.get_rank() == 0:
    torch.save({"model": model_sd, "optim": optim_sd, "step": step}, path)

# Load
ckpt = torch.load(path, map_location="cpu")
dmuon.set_model_state_dict(model, ckpt["model"])
dmuon.set_optimizer_state_dict(model, optimizer, ckpt["optim"])
```

## Run

```bash
torchrun --nproc_per_node=4 examples/basic_dp.py
```
