# Pure DMuon Routing

Pure DMuon means every trainable parameter can enter DMuon's ownership runtime instead of leaving non-Muon parameters on the ordinary FSDP2/AdamW path. This mode is useful when a training stack wants one runtime to control parameter placement, gradient movement, and optimizer stepping for the whole model. The important point is that optimizer math and communication placement are separate decisions.

`predicate` decides whether a parameter is DMuon-managed. `route_hint_fn` decides what DMuon does with that managed parameter:

| Route | Optimizer math | Forward parameter movement | Backward gradient movement | Typical parameters |
|-------|----------------|----------------------------|----------------------------|--------------------|
| `"muon"` | Muon + Newton-Schulz | owner `broadcast` | `reduce` to owner | attention and MLP projection matrices |
| `"adamw"` | AdamW | owner `broadcast` | `reduce` to owner | LayerNorm, bias, small AdamW parameters |
| `"sharded_adamw"` | AdamW | `all_gather` from parameter shards | `reduce_scatter` to optimizer shards | very large AdamW parameters such as embeddings and `lm_head` |

The `"adamw"` and `"sharded_adamw"` routes both run AdamW. They differ only in how the parameter and gradient are distributed. Small AdamW parameters should normally use `"adamw"`: one owner updates the parameter and publishes the updated value by broadcast. This has the same update semantics as a sharded AdamW update, but avoids creating many small all-gather and reduce-scatter collectives inside every transformer block.

Large terminal AdamW parameters are different. Input embeddings are read at the beginning of the forward pass, and output heads such as `lm_head` produce gradients near the end of backward. If a single owner rank handles those tensors through broadcast and reduce, the owner can become a front-of-forward or tail-of-backward bottleneck. Routing those tensors to `"sharded_adamw"` spreads both the forward all-gather and the backward reduce-scatter across all ranks.

## Control Surface

Route control has two steps. First, `predicate` decides which parameters enter
DMuon's runtime at all. For pure DMuon this predicate is usually broad:

```python
predicate=lambda name, param: param.requires_grad
```

Every trainable parameter that returns `False` stays outside DMuon and must be
handled by the surrounding FSDP/DDP runtime. Every trainable parameter that
returns `True` is replaced by a DMuon placeholder and receives a route from
`route_hint_fn`.

Second, `route_hint_fn(name, param)` returns the communication and optimizer
route for that DMuon-managed parameter:

- Return `"muon"` for projection matrices that should be updated by Muon.
- Return `"adamw"` for small AdamW parameters such as LayerNorm weights, bias
  tensors, and other scalars/vectors. These parameters use DMuon's owner
  `broadcast` and `reduce` path.
- Return `"sharded_adamw"` only for large AdamW tensors whose communication
  should be split across all ranks, typically input embeddings and `lm_head`.

For any trainable parameter included by `predicate`, return one of these three
strings explicitly. A `None` route hint is normalized to the default `"muon"`
route, so accidentally returning `None` for a LayerNorm or bias parameter would
put that parameter on the wrong optimizer path.

The returned string is consumed while `dedicate_params()` still has access to
the original full parameter tensor. This matters for `"sharded_adamw"` because
DMuon must build per-rank shard storage before the parameter is replaced by its
placeholder. Optimizer param groups may later group dedicated parameters for
hyperparameters, but they cannot create `"sharded_adamw"` storage after
`dedicate_params()` has finished.

Optimizer `param_groups` also do not override these per-parameter route hints by
default. A semantic group may contain both Muon matrices and AdamW-routed small
parameters; DMuon splits that group into route-specific optimizer subgroups while
keeping the route selected by `route_hint_fn`. Set `dmuon_route`,
`dmuon_optimizer`, or `matrix_optimizer` on a user group only when you
intentionally want a group-level route override.

## Routing Policy

A practical LLM policy is:

- Projection matrices that benefit from Muon return `"muon"`.
- Embedding and `lm_head` weights return `"sharded_adamw"`.
- LayerNorm, bias, and small AdamW parameters return `"adamw"`.

```python
SHARDED_ADAMW_NAME_PARTS = ("embed_tokens", "lm_head")
BLOCKED_MUON_NAME_PARTS = ("embed", "lm_head", "norm")


def is_muon_matrix(name, param):
    if not param.requires_grad or param.ndim != 2 or not name.endswith(".weight"):
        return False
    if any(part in name for part in BLOCKED_MUON_NAME_PARTS):
        return False
    return any(
        part in name
        for part in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
    )


def is_large_adamw_terminal_param(name, param):
    if not param.requires_grad or param.ndim != 2 or not name.endswith(".weight"):
        return False
    return any(part in name for part in SHARDED_ADAMW_NAME_PARTS)


def route_hint(name, param):
    if not param.requires_grad:
        return None
    if is_muon_matrix(name, param):
        return "muon"
    if is_large_adamw_terminal_param(name, param):
        return "sharded_adamw"
    return "adamw"


dmuon.dedicate_params(
    model,
    mesh,
    predicate=lambda name, param: param.requires_grad,
    route_hint_fn=route_hint,
)
```

If this policy is active, LayerNorm parameters do not need special-case FSDP
wrapping to avoid all-gather/reduce-scatter. They match the final `return
"adamw"` branch and therefore use DMuon's owner broadcast/reduce route.

## Hook Boundaries

Route selection is independent from hook placement. DMuon still needs a module boundary where it can prepare full parameters before forward and collect gradients after backward. Decoder blocks usually provide that boundary for projection matrices and normalization weights. Embeddings and output heads are often called outside decoder blocks, so a pure DMuon integration should include those modules in `hook_boundary_predicate`.

```python
terminal_module_ids = {
    id(module)
    for module in (
        getattr(model, "lm_head", None),
        getattr(getattr(model, "model", None), "embed_tokens", None),
    )
    if isinstance(module, (torch.nn.Embedding, torch.nn.Linear))
}


def hook_boundary(module):
    if id(module) in terminal_module_ids:
        return True
    return isinstance(module, TransformerDecoderLayer)
```

`hook_boundary_predicate` controls where DMuon communication is attached. `route_hint_fn` controls which communication primitive is used for each parameter at that boundary.

## Validation

After constructing the optimizer, inspect the route split before running a long job:

```python
summary = dmuon.summarize_param_groups(model, optimizer)
print(summary)
```

The expected pattern is that Muon-owned projection matrices appear under the Muon route, LayerNorm and bias parameters appear under AdamW, and only the large embedding/head tensors appear under sharded AdamW. If hundreds of small AdamW parameters appear under sharded AdamW, the route policy is too broad.
