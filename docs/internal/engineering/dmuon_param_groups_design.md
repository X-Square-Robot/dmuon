# DMuon Param Groups Engineering Design

Status: implementation in progress

Last updated: 2026-05-11

Current worktree status:

- Implemented: constructor lowering, subgroup sidecar mappings, Muon/AdamW
  subgroup update loops, async communication-order preservation, checkpoint
  structural validation, explicit checkpoint metadata tests, and DDP/FSDP2/TP2
  plus HSDP*TP2 tests.
- Verified: default API compatibility on DDP/FSDP2/HSDP paths; DDP semantic
  param groups; FSDP2 semantic param groups; scheduler LR mutation over
  lowered subgroups; checkpoint metadata roundtrip/mismatch behavior; TP2
  and HSDP*TP2 different-LR delta behavior on TP-sharded dedicated parameters;
  HSDP*TP2 sync/async loss consistency with semantic param groups; local
  diagnostics summaries for DDP/FSDP2/TP/HSDP*TP.
- Not finished: public API docs, Wall-X/VLA smoke, and a focused stale
  pre-wrapping parameter validation test.

Owner: DMuon optimizer runtime

Motivation source: Wall-X/VLA training needs PyTorch-style business parameter
groups, especially action-expert parameters with a larger learning rate, while
still letting DMuon internally split parameters between Muon and AdamW update
paths.

## 1. Problem

Current `dmuon.Muon` exposes two internal optimizer groups:

1. dedicated parameters updated by Muon/Newton-Schulz;
2. non-dedicated parameters updated by AdamW.

This is not enough for training stacks that already express semantic learning
rate groups:

```python
optimizer = AdamW(
    [
        {"params": base_params, "lr": 5e-5, "group_name": "base"},
        {"params": action_params, "lr": 1e-4, "group_name": "action"},
    ],
    lr=5e-5,
)
```

For DMuon, a single semantic group can contain both dedicated Muon parameters
and non-dedicated AdamW parameters. The optimizer must therefore accept
user-facing semantic groups and lower each semantic group into internal Muon
and AdamW subgroups.

## 2. Goals

1. Preserve existing `dmuon.Muon(model, lr=..., adamw_lr=...)` behavior.
2. Add optional PyTorch-style `param_groups` to `dmuon.Muon`.
3. Keep `optimizer.param_groups` valid for PyTorch schedulers, checkpointing,
   and framework LR logging.
4. Keep users unaware of dedicated placeholders, `DedicatedParam`, FSDP2
   `DTensor`, and DDP replicate internals.
5. Support different LR/WD/momentum/betas per semantic group.
6. Preserve existing DDP, FSDP2, HSDP, TP, and async post-step communication
   semantics.

## 3. Non-Goals

1. Do not add per-group gradient clipping in the first implementation.
   Existing DMuon grad clipping remains one global Muon/dedicated gradient
   norm.
2. Do not expose `DedicatedParam` objects in `optimizer.param_groups`.
3. Do not support DDP+TP in this feature. It remains unsupported by
   `dedicate_params_ddp`.
4. Do not change owner assignment, TP owner assignment, or FSDP2 wrapping
   order.

## 4. User API

Existing API remains valid:

```python
optimizer = dmuon.Muon(
    model,
    lr=5e-5,
    momentum=0.95,
    weight_decay=0.0,
    adamw_lr=5e-5,
    adamw_weight_decay=0.01,
)
```

New optional API:

```python
optimizer = dmuon.Muon(
    model,
    lr=5e-5,
    adamw_lr=5e-5,
    param_groups=[
        {"params": base_params, "lr": 5e-5, "group_name": "base"},
        {"params": action_params, "lr": 1e-4, "group_name": "action"},
    ],
)
```

Advanced per-path overrides:

```python
{
    "params": action_params,
    "group_name": "action",
    "lr": 1e-4,
    "muon_lr": 8e-5,
    "adamw_lr": 1e-4,
    "muon_weight_decay": 0.0,
    "adamw_weight_decay": 0.01,
    "momentum": 0.95,
    "adamw_betas": (0.9, 0.999),
    "adamw_eps": 1e-8,
}
```

The implementation must resolve values by key presence, not truthiness. For
example, `0.0` is a valid weight decay.

Muon subgroup precedence:

```text
lr           = group["muon_lr"] if present else group["lr"] if present else constructor lr
weight_decay = group["muon_weight_decay"] if present else group["weight_decay"] if present else constructor weight_decay
momentum     = group["momentum"] if present else constructor momentum
```

AdamW subgroup precedence:

```text
lr           = group["adamw_lr"] if present else group["lr"] if present else constructor adamw_lr
weight_decay = group["adamw_weight_decay"] if present else group["weight_decay"] if present else constructor adamw_weight_decay
betas        = group["adamw_betas"] if present else constructor adamw_betas
eps          = group["adamw_eps"] if present else constructor adamw_eps
```

## 5. Call Order Contract

For FSDP2/HSDP usage, users should build `param_groups` from the model after
the DMuon/FSDP2 wrapping sequence is complete:

```python
dmuon.dedicate_params(...)
fully_shard(...)
param_groups = build_groups_from_current_model_named_parameters(model)
optimizer = dmuon.Muon(model, param_groups=param_groups, ...)
```

This is the only fully reliable object-identity contract. Parameters can be
replaced by dedicated placeholders or FSDP2 `DTensor` objects during wrapping.

Optional future extension: accept name-based groups:

```python
param_groups=[{"param_names": [...], "lr": ...}]
```

That is useful for frameworks that compute grouping before wrapping, but it is
out of scope for the first implementation.

## 6. Internal Representation

`optimizer.param_groups` should contain only tensors/parameters in
`group["params"]`. Never put `DedicatedParam` into `param_groups`.

Every user semantic group lowers to a stable pair of subgroups:

```text
<group_name>/muon
<group_name>/adamw
```

Subgroups with no local trainable tensor on a rank use a local dummy parameter
so that:

1. every rank has identical `len(optimizer.param_groups)`;
2. LR schedulers mutate the same group indices on all ranks;
3. checkpoint param-group metadata is structurally comparable.

Only a subgroup that is globally known to be absent may be omitted. The first
implementation should prefer always creating both subgroups per user group.

Internal sidecar state:

```python
self._muon_group_dps: dict[int, list[DedicatedParam]]
self._adamw_group_params: dict[int, list[nn.Parameter]]
self._dp_to_muon_group_idx: dict[int, int]
self._adamw_param_to_group_idx: dict[int, int]
self._dummy_params: list[nn.Parameter]
```

The keys are indices into `self.param_groups`.

## 7. Discovery and Mapping

Muon construction already discovers:

```python
self._dedicated_params  # currently owner-only
self._fsdp_params       # FSDP2 sharded params and DDP replicated params
```

Param-group support needs two dedicated lists:

```python
self._all_dedicated_params
self._owned_dedicated_params
```

Reason: optimizer update uses only owner parameters, but user group mapping and
checkpoint metadata need to reason about all dedicated params consistently.

Build mapping:

```python
param_to_dp = {}
for dp in self._all_dedicated_params:
    if hasattr(dp, "_orig_param"):
        param_to_dp[id(dp._orig_param)] = dp
    if hasattr(dp, "_placeholder"):
        param_to_dp[id(dp._placeholder)] = dp

adamw_param_by_id = {id(p): p for p in self._fsdp_params}
```

DDP replicate params are included in `self._fsdp_params` today, so they are
covered by `adamw_param_by_id`.

Validation:

1. `params` accepts a tensor, list, tuple, or generator.
2. A trainable parameter may appear in only one user semantic group.
3. `requires_grad=False` parameters are ignored and counted in debug logs.
4. Any trainable parameter that is neither dedicated nor AdamW-managed causes
   a fail-loud `RuntimeError`.
5. If user groups omit trainable parameters, fail loud by default. This matches
   the safety posture of current DMuon, where non-dedicated params must be
   managed by FSDP2 or `dmuon.replicate`.

## 8. Default Compatibility Mode

When `param_groups is None`, construct exactly the old logical layout:

```text
default/muon
default/adamw
```

For compatibility, these groups should retain the old hyperparameter values:

```text
default/muon:  lr=lr,       momentum=momentum, weight_decay=weight_decay
default/adamw: lr=adamw_lr, betas=adamw_betas, weight_decay=adamw_weight_decay, eps=adamw_eps
```

If external tests assert `len(optimizer.param_groups) == 2`, they should still
pass.

## 9. Step Semantics

### 9.1 AdamW Path

AdamW can be implemented by iterating AdamW subgroups:

```python
for group_idx, group in enumerate(self.param_groups):
    if group["use_muon"]:
        continue
    for p in self._adamw_group_params[group_idx]:
        step_one_adamw_param(p, group)
```

State remains keyed by parameter object, as today.

### 9.2 Sync Muon Path

The sync path can also iterate Muon subgroups directly:

```python
for group_idx, group in enumerate(self.param_groups):
    if not group["use_muon"]:
        continue
    self._step_muon_params(self._muon_group_dps[group_idx], group)
```

After all Muon and AdamW updates, sync mode still calls
`broadcast_all_updates(model)`.

### 9.3 Async Pipelined Muon Path

This is the critical part. Async post-step publishing must remain ordered by
communication group, not by optimizer subgroup.

Current async pipeline:

```text
for communication_group in _ordered_post_step_groups(model):
    update owner params in this communication_group
    dispatch post-step communication for this communication_group
```

Param-group support must preserve this ordering:

```python
for comm_group in _ordered_post_step_groups(self.model):
    owned_params = [
        dp for dp in comm_group.params
        if dp.is_owner and dp._reduced_grad is not None
    ]
    self._step_muon_comm_group_params(owned_params)
    _dispatch_post_step_async(comm_group)
```

Inside `_step_muon_comm_group_params`, each `dp` looks up its Muon subgroup:

```python
group = self.param_groups[self._dp_to_muon_group_idx[id(dp)]]
step_one_muon_dp(dp, group)
```

This preserves:

1. layer/forward-order publish priority;
2. DDP broadcast ordering;
3. HSDP replicate-broadcast ordering;
4. TP scatter-before-replicate dependency;
5. overlap between early-layer compute and later-layer communication.

Do not implement async Muon as:

```python
for optimizer_group in muon_groups:
    for dp in optimizer_group:
        step_one_muon_dp(dp, optimizer_group)
```

That would reorder post-step communication and can regress overlap or deadlock
when ranks disagree on collective order.

## 10. TP-Specific Requirements

For TP-sharded dedicated params, the per-param Muon group hyperparameters must
drive both the TP owner and non-owner paths:

```python
dp._tp_wd_factor = 1.0 - group["lr"] * group["weight_decay"]
update_full.mul_(-group["lr"] * scale)
```

Every DP-owner TP rank must compute the same `dp._tp_wd_factor` for the same
parameter. That implies every rank must map that parameter to the same semantic
Muon subgroup metadata, even if only the TP owner runs Newton-Schulz.

Tests must include at least one TP topology with different base/action LR and
verify the actual update ratio or delta norm.

## 11. Scheduler Semantics

Schedulers mutate `optimizer.param_groups[*]["lr"]`. Since DMuon lowers each
semantic group into `/muon` and `/adamw` subgroups, scheduler behavior is:

```text
base/muon   lr 5e-5 -> scheduled value
base/adamw  lr 5e-5 -> scheduled value
action/muon lr 1e-4 -> scheduled value
action/adamw lr 1e-4 -> scheduled value
```

If one scheduler scale applies to all groups, ratios are preserved naturally.

`initial_lr` should be populated by PyTorch scheduler on first use. DMuon does
not need custom scheduler support as long as `optimizer.param_groups` is
standard.

## 12. Checkpoint Semantics

`get_optimizer_state_dict()` already saves param group metadata without tensor
refs:

```python
{k: v for k, v in group.items() if k != "params"}
```

This remains valid. New group metadata must be JSON/pickle friendly:

```text
group_name
use_muon
semantic_group_name
subgroup_type
lr
momentum
betas
weight_decay
eps
```

`set_optimizer_state_dict()` should add structure checks:

1. saved/current group count mismatch: warning; restore only matching prefix;
2. `group_name` mismatch: warning;
3. `use_muon` mismatch: raise;
4. `subgroup_type` mismatch: raise.

Dedicated momentum state remains FQN-keyed and does not depend on group index.
AdamW state remains parameter-keyed internally and FQN-keyed in DMuon's exported
state dict.

## 13. Logging and Debuggability

On initialization, rank 0 should log a compact summary:

```text
DMuon param groups:
  0 base/muon:   use_muon=True  lr=5e-5  local_dps=12 params=1(dummy)
  1 base/adamw:  use_muon=False lr=5e-5  local_params=34
  2 action/muon: use_muon=True  lr=1e-4  local_dps=4 params=1(dummy)
  3 action/adamw use_muon=False lr=1e-4  local_params=8
```

Add helper:

```python
dmuon.summarize_param_groups(model, optimizer, max_rows=100)
```

Suggested output rows:

```text
fqn
shape
path = muon | adamw
group_name
lr
owner_rank
is_owner
is_tp_owner
```

This helper is useful for Wall-X audits and for verifying VLM/VLA parameter
coverage.

## 14. Failure Modes

Fail loud on:

1. duplicate trainable parameter across user groups;
2. trainable parameter omitted from all user groups;
3. trainable parameter not managed by dedicated/FSDP2/DDP replicate path;
4. user passes `DedicatedParam` in `params`;
5. user passes stale pre-wrapping params that cannot be mapped;
6. inconsistent subgroup structure when loading optimizer state.

Warn on:

1. ignored `requires_grad=False` params;
2. empty semantic user group;
3. checkpoint group-name mismatch with same structural type.

## 15. Implementation Plan

### Phase A: Constructor Lowering

Status: implemented in worktree.

1. Done: add `param_groups: Optional[list[dict]] = None` to `Muon.__init__`.
2. Done: split dedicated discovery into all-dedicated vs owned-dedicated.
3. Done: build object-id maps for dedicated placeholders/original params and AdamW
   params.
4. Done: normalize user groups and validate duplicate/missing coverage.
5. Done: lower semantic groups into stable `/muon` and `/adamw` optimizer
   groups.
6. Done: preserve current two-group behavior when `param_groups is None`.

### Phase B: Update Loops

Status: implemented in worktree.

1. Done: refactor `_step_muon(params=None)` into a lower-level
   `_step_muon_params(params, group)`.
2. Done: refactor `_step_adamw()` to iterate AdamW subgroups.
3. Done: update sync path to iterate Muon subgroups.
4. Done: update async path to keep `_ordered_post_step_groups(model)` as the outer
   loop and look up per-param Muon subgroup metadata inside the communication
   group.

### Phase C: Checkpoint and Scheduler

Status: implemented in worktree.

1. Done: store subgroup metadata in `param_groups`.
2. Done: add mismatch validation in `set_optimizer_state_dict()` for
   `use_muon` and `subgroup_type`, with warning-only behavior for group-name
   mismatch and group-count mismatch.
3. Done: add scheduler coverage using `LambdaLR` in the DDP distributed
   correctness test.
4. Done: add explicit param-group checkpoint roundtrip coverage.
5. Done: add explicit checkpoint mismatch tests for `use_muon`,
   `subgroup_type`, and `group_name`.

### Phase D: Diagnostics

Status: implemented in worktree.

1. Done: add `summarize_param_groups(model, optimizer, max_rows=...)`.
   The helper is local-rank only and performs no collective communication.
2. Done: add `format_param_group_summary(summary)` for compact human-readable
   logs.
3. Done: include subgroup-level fields: group name, semantic group name,
   subgroup type, LR/WD/momentum/betas/eps, local parameter count, dummy count,
   dedicated count, owned dedicated count, TP-sharded dedicated count, and
   AdamW local parameter count.
4. Done: include parameter-detail rows with FQN, route (`muon` or `adamw`),
   group, local/full shapes, owner metadata, and TP metadata, bounded by
   `max_rows`.
5. Done: add distributed assertions in DDP, FSDP2, DP2*TP2, and HSDP*TP2 tests
   that the diagnostics expose action groups and TP-sharded Muon groups.
6. Remaining: public docs / Wall-X training-log example once the real Wall-X
   smoke is wired.

### Phase E: Distributed Validation

Status: partially complete.

Run CPU/unit tests first, then distributed tests on remote GPU:

1. Done: 2-GPU DDP param group split and full DDP P1 suite.
2. Done: 2-GPU FSDP2 param group split.
3. Done: 4-GPU HSDP async vs sync loss consistency.
4. Done: 4-GPU DP2*TP2 different-LR delta check for TP-sharded dedicated
   parameters.
5. Done: 8-GPU HSDP2*Shard2*TP2 param-group regression, including different-LR
   delta and sync/async loss consistency.

## 16. Test Plan

### Unit Tests

1. Covered by distributed regression: `param_groups=None` preserves two groups
   and current step behavior.
2. Covered by distributed tests: user semantic groups lower to stable `/muon`
   and `/adamw` subgroups.
3. Covered by DDP test: duplicate parameter raises.
4. Covered by DDP test: missing trainable parameter raises.
5. Covered by constructor validation path; needs a focused test for stale
   pre-wrapping params.
6. Covered by DDP test: scheduler scales all subgroup LR values while
   preserving ratios.
7. Covered by DDP test: checkpoint restores subgroup hyperparameters.
8. Covered by DDP test: checkpoint raises on `use_muon`/`subgroup_type`
   mismatch and warns on same-structure `group_name` mismatch.
9. Covered by DDP/FSDP2/TP/HSDP*TP tests: diagnostics expose lowered action
   groups, route metadata, and TP-sharded dedicated counts.

### Distributed Tests

1. Done: DDP action Muon subgroup and action AdamW subgroup both use 2x LR.
2. Partial: FSDP2 split is covered; FSDP2 scheduler behavior is indirectly
   covered by standard PyTorch optimizer group mutation and still needs a
   direct FSDP2 assertion if required.
3. Done: HSDP async loss matches sync bit-for-bit in the existing 4-GPU test.
4. Done: TP2 `update_full.mul_(-lr * scale)` uses per-param group LR. The
   `tests/distributed/test_tp_muon_step.py param_groups_lr` scenario compares
   identical initial weights/input with action LR 1x vs 2x and checks that
   action TP-sharded dedicated deltas scale by 2 while base deltas stay 1x.
5. Done: HSDP*TP2 replicate broadcast remains ordered and loss consistent with
   param groups. The `tests/distributed/test_tp_muon_step.py
   hsdp_tp_param_groups` scenario runs on an HSDP2*Shard2*TP2 mesh, checks
   action/base delta ratios, then compares sync vs async loss trajectories.

### Wall-X Smoke

Status: not started.

Run one short Wall-X/VLA job with DMuon param groups and verify:

1. LR logs expose `base/muon`, `base/adamw`, `action/muon`, `action/adamw`.
2. action groups keep the configured LR ratio after scheduler warmup.
3. action encoder/decoder projection weights are in Muon groups.
4. ViT/modulation/non-dedicated weights are in AdamW groups.
5. checkpoint save/load preserves group metadata.

## 17. Acceptance Criteria

The feature is ready when:

1. Existing DMuon API and tests pass unchanged.
2. `param_groups` supports Wall-X action LR without framework-specific DMuon
   internals.
3. DDP/FSDP2/HSDP/TP communication ordering is unchanged.
4. Scheduler and checkpoint behavior are deterministic across ranks.
5. A real Wall-X 1-step smoke shows expected LR group logs and no parameter
   coverage gaps.
