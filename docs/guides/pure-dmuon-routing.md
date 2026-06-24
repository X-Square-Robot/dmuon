# Pure DMuon Routing

Pure DMuon means every trainable parameter can enter DMuon's ownership runtime instead of leaving non-Muon parameters on the ordinary FSDP2/AdamW path. This mode is useful when a training stack wants one runtime to control parameter placement, gradient movement, and optimizer stepping for the whole model. The important point is that optimizer math and communication placement are separate decisions.

`predicate` decides whether a parameter is DMuon-managed. `param_policy` decides what DMuon does with that managed parameter. The route selects optimizer math and communication placement:

| Route | Optimizer math | Forward parameter movement | Backward gradient movement | Typical parameters |
|-------|----------------|----------------------------|----------------------------|--------------------|
| `"muon"` | Muon + Newton-Schulz | owner `broadcast` | `reduce` to owner | attention and MLP projection matrices |
| `"adamw"` | AdamW | owner `broadcast` | `reduce` to owner | LayerNorm, bias, small AdamW parameters |
| `"sharded_adamw"` | AdamW | `all_gather` from parameter shards | `reduce_scatter` to optimizer shards | very large AdamW parameters such as embeddings and `lm_head` |

The `"adamw"` and `"sharded_adamw"` routes both run AdamW. They differ only in how the parameter and gradient are distributed. Small AdamW parameters should normally use `"adamw"`: one owner updates the parameter and publishes the updated value by broadcast. This has the same update semantics as a sharded AdamW update, but avoids creating many small all-gather and reduce-scatter collectives inside every transformer block.

Large terminal AdamW parameters are different. Input embeddings are read at the beginning of the forward pass, and output heads such as `lm_head` produce gradients near the end of backward. If a single owner rank handles those tensors through broadcast and reduce, the owner can become a front-of-forward or tail-of-backward bottleneck. Routing those tensors to `"sharded_adamw"` spreads both the forward all-gather and the backward reduce-scatter across all ranks.

## Control Surface

Policy control has two steps. First, `predicate` decides which parameters enter
DMuon's runtime at all. For pure DMuon this predicate is usually broad:

```python
predicate=lambda name, param: param.requires_grad
```

Every trainable parameter that returns `False` stays outside DMuon and must be
handled by the surrounding FSDP/DDP runtime. Every trainable parameter that
returns `True` is replaced by a DMuon placeholder and receives a structured
policy from `param_policy`.

Second, `param_policy` starts from `defaults` and applies ordered `overrides`
whose `name` tokens match the full parameter name. Each override updates
only the fields it sets. `contains` is still accepted as a legacy alias for
`name`, but new integrations should use `name`. The main route choices are:

- Return `"muon"` for projection matrices that should be updated by Muon.
- Return `"adamw"` for small AdamW parameters such as LayerNorm weights, bias
  tensors, and other scalars/vectors. These parameters use DMuon's owner
  `broadcast` and `reduce` path.
- Return `"sharded_adamw"` only for large AdamW tensors whose communication
  should be split across all ranks, typically input embeddings and `lm_head`.

For any trainable parameter included by `predicate`, make sure the final
`route` is one of these three strings. A missing route is normalized to the
default `"muon"` route, so accidentally leaving LayerNorm or bias parameters
without an AdamW override would put them on the wrong optimizer path.

The final policy is consumed while `dedicate_params()` still has access to the
original full parameter tensor. This matters for `"sharded_adamw"` because DMuon
must build per-rank shard storage before the parameter is replaced by its
placeholder. Optimizer param groups may later group dedicated parameters for
hyperparameters, but they cannot create `"sharded_adamw"` storage after
`dedicate_params()` has finished.

Optimizer `param_groups` also do not override these per-parameter route hints by
default. A semantic group may contain both Muon matrices and AdamW-routed small
parameters; DMuon splits that group into route-specific optimizer subgroups while
keeping the route selected by `param_policy`. Set `dmuon_route`,
`dmuon_optimizer`, or `matrix_optimizer` on a user group only when you
intentionally want a group-level route override.

The dtype fields follow FSDP2 mixed-precision terminology. `param_dtype` is the
dtype of the materialized parameter used by forward/backward compute and forward
parameter communication. `grad_dtype` is the dtype for DMuon's gradient buffers
and gradient reduction. `output_dtype` casts floating-point outputs at the
module boundary, and `cast_forward_inputs=True` casts floating-point inputs to
`param_dtype`. `master_dtype` and `optim_dtype` describe DMuon's storage and
optimizer-update precision and stay separate from forward compute dtype.

## Routing Policy

A practical LLM policy is:

- Parameters suitable for matrix optimization use `"muon"` by default.
- Embedding and `lm_head` weights return `"sharded_adamw"`.
- LayerNorm, bias, and small AdamW parameters return `"adamw"`.

```python
dmuon.dedicate_params(
    model,
    mesh,
    predicate=lambda name, param: param.requires_grad,
    param_policy={
        "defaults": {
            "route": "muon",
            "param_dtype": torch.bfloat16,
            "grad_dtype": None,
            "master_dtype": torch.float32,
            "optim_dtype": torch.float32,
        },
        "overrides": [
            {
                "name": ["embed_tokens", "word_embeddings", "wte", "lm_head"],
                "set": {"route": "sharded_adamw"},
            },
            {
                "name": ["norm", "ln_", ".bias"],
                "set": {"route": "adamw"},
            },
            {
                "name": ["action_head", "action_decoder"],
                "set": {
                    "param_dtype": torch.float32,
                    "grad_dtype": torch.float32,
                    "output_dtype": torch.float32,
                    "cast_forward_inputs": True,
                },
            },
        ],
    },
)
```

If this policy is active, LayerNorm parameters do not need special-case FSDP
wrapping to avoid all-gather/reduce-scatter. They match the `"adamw"` override
and therefore use DMuon's owner broadcast/reduce path. Projection matrices that
do not match a special override keep the default `"muon"` route. Parameters
under `action_head` can keep fp32 forward compute without forcing the rest of
the model out of bf16 compute.

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

`hook_boundary_predicate` controls where DMuon communication is attached. `param_policy` controls which communication primitive and dtype policy is used for each parameter at that boundary.

Activation casts are hook-boundary operations. If one hook boundary contains
parameters with different effective `param_dtype` values and any group enables
`cast_forward_inputs`, DMuon raises during setup. Split the action head into its
own hook boundary, or set `cast_forward_inputs=False` and make the module handle
activation casts explicitly.

## Validation

After constructing the optimizer, inspect the route split before running a long job:

```python
summary = dmuon.summarize_param_groups(model, optimizer)
print(summary)
```

The expected pattern is that Muon-owned projection matrices appear under the Muon route, LayerNorm and bias parameters appear under AdamW, and only the large embedding/head tensors appear under sharded AdamW. The parameter rows also show `param_dtype`, `grad_dtype`, `output_dtype`, and matched policy override indices. If hundreds of small AdamW parameters appear under sharded AdamW, the route policy is too broad.

`route_hint_fn` remains available for legacy route-only integrations. New pure
DMuon integrations should prefer `param_policy` because route, parameter dtype,
gradient dtype, and module-boundary casting are resolved together before DMuon
replaces the original parameters.
