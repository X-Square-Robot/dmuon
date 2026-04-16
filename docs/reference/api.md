# API Reference

Complete reference for DMuon's public API.

---

## Core API

### `dmuon.dedicate_params`

```python
dmuon.dedicate_params(
    model: nn.Module,
    mesh: DeviceMesh,
    predicate: Callable[[str, nn.Parameter], bool],
    compute_dtype: torch.dtype = None,
    reshard_after_forward: bool = True,
) -> dict[nn.Parameter, int]
```

Mark parameters for dedicated ownership and register communication hooks.

Parameters satisfying `predicate` are assigned to owner ranks via a balanced partition algorithm. Each marked parameter will be automatically ignored by subsequent `fully_shard()` calls.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `nn.Module` | *required* | The model whose parameters to partition. |
| `mesh` | `DeviceMesh` | *required* | 1D DeviceMesh for the data-parallel dimension. |
| `predicate` | `Callable` | *required* | `(param_name, param) -> bool`. Parameters returning True use dedicated ownership. |
| `compute_dtype` | `torch.dtype` | `None` | Optional dtype for communication (e.g., `torch.bfloat16`). |
| `reshard_after_forward` | `bool` | `True` | If True, reshard after forward (like `FULL_SHARD`). If False, keep unsharded through forward+backward (like `SHARD_GRAD_OP`). |

**Returns:** Dict mapping each dedicated parameter to its owner rank (int).

---

### `dmuon.wait_all_reduces`

```python
dmuon.wait_all_reduces(model: nn.Module) -> None
```

Wait for all pending gradient reduces to complete.

Called automatically by `optimizer.step()`. Only needed if you want to manually access `_reduced_grad` before stepping.

---

### `dmuon.no_sync`

```python
@contextmanager
dmuon.no_sync(model: nn.Module)
```

Context manager to disable gradient reduction for gradient accumulation.

Within this context, backward passes skip reduce communication and accumulate gradients locally. Also disables FSDP2's gradient sync for symmetric parameters.

```python
with dmuon.no_sync(model):
    loss = model(batch).loss / accum_steps
    loss.backward()
```

---

## Optimizer

### `dmuon.Muon`

```python
dmuon.Muon(
    model: nn.Module,
    lr: float = 0.02,
    momentum: float = 0.95,
    weight_decay: float = 0.0,
    ns_steps: int = 5,
    adamw_lr: float = 1e-3,
    adamw_betas: tuple[float, float] = (0.9, 0.999),
    adamw_weight_decay: float = 0.01,
    adamw_eps: float = 1e-8,
    nesterov: bool = True,
    per_head_ns: bool = True,
    block_diagonal_ns: bool = False,
)
```

Combined optimizer for DMuon distributed training.

Manages two parameter groups:

- **Group 0** (dedicated params): Muon — momentum + Newton-Schulz orthogonalization, run by owner only
- **Group 1** (symmetric params): AdamW, run by all ranks on FSDP2 shards

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `nn.Module` | *required* | Model with `dedicate_params` and `fully_shard` already applied. |
| `lr` | `float` | `0.02` | Muon learning rate for dedicated params. Internally scaled by `0.2 * sqrt(max(m,n))`. |
| `momentum` | `float` | `0.95` | Momentum coefficient for dedicated params. |
| `weight_decay` | `float` | `0.0` | Weight decay for dedicated params (decoupled, like AdamW). |
| `ns_steps` | `int` | `5` | Number of Newton-Schulz iterations. |
| `ns_backend` | `str` or `NewtonSchulz` | `"gram"` | NS backend configuration. Pass a string shorthand (`"gram"` or `"direct"`) for default coefficients, or a `NewtonSchulz` object for full control (see below). |
| `nesterov` | `bool` | `True` | Use Nesterov momentum: `ns_input = grad + mu * buf`. |
| `per_head_ns` | `bool` | `True` | Use per-head local NS for narrow Shard(0) params (GQA k/v_proj). |
| `block_diagonal_ns` | `bool` | `False` | Skip Gram all-reduce for all TP params (experimental). |
| `adamw_lr` | `float` | `1e-3` | AdamW learning rate for symmetric params. |
| `adamw_betas` | `tuple` | `(0.9, 0.999)` | AdamW beta coefficients. |
| `adamw_weight_decay` | `float` | `0.01` | AdamW weight decay. |
| `adamw_eps` | `float` | `1e-8` | AdamW epsilon. |

**Methods:**

- `step(closure=None)` — Perform one optimization step. Internally: (1) wait for reduces, (2) Muon on dedicated params, (3) AdamW on FSDP2 params.
- `zero_grad(set_to_none=True)` — Clear gradients for both parameter types.

---

### `dmuon.NewtonSchulz`

```python
dmuon.NewtonSchulz(
    backend: str = "gram",
    coefficients: list[list[float]] = None,
    restart_iterations: list[int] = None,
)
```

Configurable Newton-Schulz backend object. Pass to `Muon(ns_backend=...)` for custom coefficients or algorithm selection.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `backend` | `str` | `"gram"` | `"gram"`: Gram-space NS (SYRK + restarts). `"direct"`: classic parameter-space NS. |
| `coefficients` | `list` | `None` | Per-step `(a, b, c)` coefficients. `None` uses `POLAR_EXPRESS_COEFFICIENTS`. |
| `restart_iterations` | `list[int]` | `None` | Restart positions for Gram-space NS. `None` uses `[2]`. Ignored for `"direct"`. |

**Usage:**

```python
import dmuon

# Default Gram-space NS
optimizer = dmuon.Muon(model, lr=0.02, ns_backend="gram")

# Classic Muon/Moonlight with You coefficients
ns = dmuon.NewtonSchulz("direct", coefficients=dmuon.YOU_COEFFICIENTS)
optimizer = dmuon.Muon(model, lr=0.02, ns_backend=ns)

# Gram-space with You coefficients
ns = dmuon.NewtonSchulz("gram", coefficients=dmuon.YOU_COEFFICIENTS)
optimizer = dmuon.Muon(model, lr=0.02, ns_backend=ns)
```

!!! note
    TP params that require Gram decomposition (exact or block-diagonal) always use `gram_newton_schulz` internally — the `backend` setting only affects local (non-TP) params and TP per-head params. Custom `coefficients` are applied to all paths.

---

## Newton-Schulz Functions

### `dmuon.newton_schulz`

```python
dmuon.newton_schulz(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: list[list[float]] = None,
    restart_iterations: list[int] = None,
) -> Tensor
```

Newton-Schulz orthogonalization (Gram-space backend).

Routes to `gram_newton_schulz_local()` for Gram-space iteration with per-step coefficients, restart mechanism, and SYRK acceleration.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `G` | `Tensor` | *required* | Gradient matrix (m, n), any dtype. |
| `steps` | `int` | `5` | Ignored (determined by `len(coefficients)`). |
| `eps` | `float` | `1e-7` | Normalization epsilon. |
| `coefficients` | `list` | `POLAR_EXPRESS_COEFFICIENTS` | Per-step `(a, b, c)` coefficients. |
| `restart_iterations` | `list[int]` | `[2]` | Restart positions for numerical stability. |

**Returns:** Orthogonalized update, same shape as G.

---

### `dmuon.gram_newton_schulz`

```python
dmuon.gram_newton_schulz(
    G_shard: Tensor,
    tp_group: dist.ProcessGroup,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: list[list[float]] = None,
    restart_iterations: list[int] = None,
    shard_dim: int = None,
    block_diagonal: bool = False,
) -> Tensor
```

Gram Newton-Schulz with TP SYRK decomposition.

Iterates on the Gram matrix instead of the full gradient. The `shard_dim` controls which Gram is used:

- **Shard(0)** (row-sharded): transpose to use R-side G^TG (decomposes as sum of local terms)
- **Shard(1)** (col-sharded): use L-side GG^T (decomposes as sum of local terms)

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `G_shard` | `Tensor` | *required* | TP-sharded gradient on this rank. |
| `tp_group` | `ProcessGroup` | *required* | TP process group for all-reduce. |
| `shard_dim` | `int` | `None` | TP shard dimension (0 or 1). If None, falls back to shape heuristic. |
| `block_diagonal` | `bool` | `False` | If True, skip TP all-reduce (block-diagonal approximation). |

**Returns:** Orthogonalized update shard, same shape as input.

---

### `dmuon.direct_newton_schulz`

```python
dmuon.direct_newton_schulz(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: list[list[float]] = None,
) -> Tensor
```

Standard Newton-Schulz in direct (parameter) space.

Iterates on the full (m, n) matrix: $X_{k+1} = a_k X + b_k (XX^T)X + c_k (XX^T)^2 X$

This is the classic formulation from Muon/Moonlight. Use for baseline comparison or small matrices where Gram-space overhead is not justified.

**Returns:** Orthogonalized update, same shape as G.

---

## Inspection Utilities

### `dmuon.get_dedicated_params`

```python
dmuon.get_dedicated_params(model: nn.Module) -> list[DedicatedParam]
```

Collect all `DedicatedParam` instances from a model (across all ranks).

---

### `dmuon.get_owned_params`

```python
dmuon.get_owned_params(model: nn.Module, rank: int) -> list[DedicatedParam]
```

Collect `DedicatedParam` instances owned by a specific rank.

---

### `dmuon.get_comm_ctx`

```python
dmuon.get_comm_ctx(model: nn.Module) -> Optional[DedicatedCommContext]
```

Get the `DedicatedCommContext` from a model, if it exists.

---

### `dmuon.get_ns_backend`

```python
dmuon.get_ns_backend() -> str
```

Returns the active Newton-Schulz backend: `"syrk_sm80"` or `"compiled"`.

---

## Checkpoint Functions

### `dmuon.get_model_state_dict`

```python
dmuon.get_model_state_dict(
    model: nn.Module,
    *,
    cpu_offload: bool = True,
) -> dict[str, torch.Tensor]
```

Get full model state dict with both dedicated and FSDP2 parameters.

Produces a state dict identical to what a single-GPU model would produce. **Collective operation** — all ranks must call.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `nn.Module` | *required* | Model with DMuon + FSDP2 applied. |
| `cpu_offload` | `bool` | `True` | Move tensors to CPU (recommended for saving). |

---

### `dmuon.set_model_state_dict`

```python
dmuon.set_model_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None
```

Load a full state dict into a DMuon model. Handles both dedicated and FSDP2 parameters.

The state dict should contain full (unsharded) tensors.

---

### `dmuon.get_optimizer_state_dict`

```python
dmuon.get_optimizer_state_dict(
    model: nn.Module,
    optimizer: Muon,
    *,
    cpu_offload: bool = True,
) -> dict
```

Get optimizer state dict for a DMuon Muon optimizer. **Collective operation** — all ranks must call.

Returns a dict with sections: `"fsdp"` (AdamW state), `"dedicated"` (Muon momentum buffers), `"param_groups"` (hyperparameters).

---

### `dmuon.set_optimizer_state_dict`

```python
dmuon.set_optimizer_state_dict(
    model: nn.Module,
    optimizer: Muon,
    state_dict: dict,
) -> None
```

Load optimizer state dict into a DMuon Muon optimizer.

---

## DedicatedParam Properties

`DedicatedParam` instances (returned by `get_dedicated_params` / `get_owned_params`) expose these properties:

| Property | Type | Description |
|----------|------|-------------|
| `is_owner` | `bool` | Whether this rank owns the parameter. |
| `owner_rank` | `int` | The DP-local rank that owns this parameter. |
| `numel` | `int` | Number of elements in the (local) parameter. |
| `param_name` | `str` | Name of the parameter within its parent module (e.g., `"weight"`). |
| `is_dtensor` | `bool` | Whether this is a TP-sharded DTensor parameter. |
| `tp_group` | `ProcessGroup` | TP process group, or None if not a DTensor. |
| `shard_dim` | `int` | TP shard dimension (0 or 1), or None if not a DTensor. |
| `full_shape` | `torch.Size` | Full (unsharded) parameter shape. |
| `_orig_size` | `torch.Size` | Local (sharded) parameter shape. |
| `_owned_data` | `Tensor` | Full parameter data (only on owner rank). |
| `_reduced_grad` | `Tensor` | Reduced gradient (only on owner, after backward + wait). |

---

## Constants

### Coefficient Sets

```python
dmuon.YOU_COEFFICIENTS        # 5-step coefficients from @YouJiacheng
dmuon.POLAR_EXPRESS_COEFFICIENTS  # 5-step coefficients from Polar Express paper (default)
```

Both are lists of `(a, b, c)` tuples. Pass to `coefficients` parameter of NS functions.
