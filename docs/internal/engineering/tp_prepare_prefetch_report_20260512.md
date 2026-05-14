# TP Prepare-Grads Split and Prefetch Report

Date: 2026-05-12

Branch: `codex/tp-overlap-forward-order`

Worktree:
`/mnt/data/x2robot_v2/liuxingchen/codes/dmuon_worktrees/group-pipelined-post-step-broadcast`

## Summary

The optimizer step now separates the semantic "prepare Muon gradients" phase
from the narrower "wait reduce tail" operation. For TP-bearing dedicated
parameters, async mode prepares gradients one group ahead in forward order:

```text
prepare group 0 on reduce_stream
for group_i in forward_order:
    prepare group_{i+1} on reduce_stream
    wait group_i readiness event on compute stream
    run Muon / NS for group_i
    dispatch that group's TP scatter and replicate publish
```

This keeps the collective order group-coalesced and forward-ordered, avoids a
single CPU launch cluster for all TP grad gathers, and gives later groups'
reduce-unpack plus TP gather work a chance to overlap with current-group NS
compute.

## Problem

The old profiler range `dmuon.optimizer.wait_reduces` was misleading. It did
not only wait for DP/HSDP reduce completion. On TP paths it also launched
`tp_gather_grads`, so profiles could show `c10d::gather_` under
`wait_reduces`.

That was confusing for two reasons:

1. The name implied a passive wait, but the range actively enqueued TP
   collective work.
2. The global location made all group TP gather launches happen together
   before Muon compute, which is the worst shape for CPU launch-bound overlap.

## Design

The new public operation is `prepare_muon_grads(model)`. It means "make owner
side gradients ready for Muon", which may include both reduce-tail waits and TP
grad gather.

The lower-level helpers are:

- `prepare_group_muon_grads(group, use_reduce_stream=False)`: waits the
  group's reduce tail, optionally on the reduce stream, then launches TP grad
  gather if the group has TP shards.
- `wait_group_muon_grads(group)`: consumes the group's readiness event before
  Muon reads `_reduced_grad` or `_tp_full_grad`.
- `wait_all_reduces(model)`: kept only as a backward-compatible alias for
  `prepare_muon_grads(model)`.

Profiler ranges are now explicit:

- `dmuon.prepare_muon_grads.<group>`
- `dmuon.wait_reduce_tail.<group>`
- `dmuon.tp_gather_grads.<group>`
- `dmuon.optimizer.prepare_muon_grads`
- `dmuon.optimizer.group_pipeline`
- `dmuon.optimizer.muon`

For TP async mode, `Muon._step_muon_and_dispatch_groups_async()` uses
one-group lookahead. It prepares the next group before waiting and computing the
current group. The prefetch path is enabled only when the optimizer owns at
least one TP-sharded dedicated parameter. Pure DDP and non-TP FSDP2 preserve
the old global prepare-before-publish behavior.

## Communication Order

The required order is still:

```text
backward
  -> reduce grads on DP/HSDP axes
  -> TP gather reduced grads to TP owner
  -> Muon / NS on the owner
  -> TP scatter full-matrix delta
  -> replicate broadcast, if HSDP replicate is enabled
next forward
  -> group pre-forward wait drains TP scatter before replicate publish
```

The implementation keeps that order by using one ordered group list from
`_ordered_post_step_groups(model)`. Rank-local ownership only changes who
computes a buffer; it does not change which group collectives are dispatched.

`tp_gather_grads(wait_current_stream=False)` is used only when the reduce tail
has already been consumed on `reduce_stream`. That prevents the reduce stream
from depending on current compute work and lets the gather launch as a true
prefetch.

## Bugs Found During Hardening

### Transient Buffer Lifetime

Initial TP prefetch changed timing enough to expose a lifetime bug:

- TP2 async had a nonzero step-1 loss gap against sync.
- TP2 async-drain could produce NaN.

The root cause was allocator reuse of transient reduce/gather buffers before
the reduce stream had finished using them. The fix pins:

- reduce-unpack source tensors in `_muon_grad_ready_refs`;
- TP gather source/receive tensors in `_tp_gather_refs`.

Those refs are released only after `wait_group_muon_grads()` or
`wait_for_tp_gather()` consumes the corresponding event.

### Non-TP DDP Should Not Use TP Prefetch Scheduling

The DDP param-group async-vs-sync loss test failed with the first version of
group-local prepare. DDP has no TP gather to hide, so changing the prepare
scheduler there only added risk. The optimizer now gates group prefetch on
`self._has_tp_dedicated`. Non-TP async still runs global `prepare_muon_grads`
before group publish.

## Validation

Local checks, current worktree:

```bash
/mnt/data/x2robot_v2/liuxingchen/miniforge3/envs/wallx35/bin/python \
  -m compileall dmuon tests/unit/test_priority_order.py tests/distributed/test_tp_comm_order.py

ruff check \
  dmuon/__init__.py dmuon/utils.py dmuon/optim/muon.py \
  dmuon/_backends/ddp/group.py dmuon/_backends/fsdp2/group.py \
  tests/unit/test_priority_order.py tests/distributed/test_tp_comm_order.py

/mnt/data/x2robot_v2/liuxingchen/miniforge3/envs/wallx35/bin/python \
  -m pytest -q tests/unit/test_priority_order.py
```

Results:

- compileall: passed.
- ruff: passed.
- `tests/unit/test_priority_order.py`: 8 passed.

Remote GPU checks on `liuxingchen@22.22.148.138`:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
DMUON_COMM_TOPOLOGY=tp2 \
DMUON_COMM_MODE=async \
DMUON_COMM_MODEL=tiny \
DMUON_COMM_STEPS=2 \
DMUON_COMM_OUT=/tmp/dmuon_comm_tp2_async_current.json \
~/miniforge3/envs/wallx35/bin/torchrun --standalone --nproc_per_node=2 \
  tests/distributed/test_tp_comm_order.py
```

Result:

- `[tp2/async/tiny] step=0 loss=-0.0048700981 refs=30 replicate_states=0`
- `[tp2/async/tiny] step=1 loss=-0.0092021786 refs=30 replicate_states=0`

The matching TP2 sync state-machine gate also passed:

- `[tp2/sync/tiny] step=0 loss=-0.0048700981 refs=30 replicate_states=0`
- `[tp2/sync/tiny] step=1 loss=-0.0092021786 refs=30 replicate_states=0`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DMUON_COMM_TOPOLOGY=hsdp_tp2 \
DMUON_COMM_MODE=async \
DMUON_COMM_MODEL=tiny \
DMUON_COMM_STEPS=2 \
DMUON_COMM_OUT=/tmp/dmuon_comm_hsdp_tp2_async_current.json \
~/miniforge3/envs/wallx35/bin/torchrun --standalone --nproc_per_node=8 \
  tests/distributed/test_tp_comm_order.py
```

Result:

- `[hsdp_tp2/async/tiny] step=0 loss=-0.0048700981 refs=30 replicate_states=24`
- `[hsdp_tp2/async/tiny] step=1 loss=-0.0092021786 refs=30 replicate_states=24`

```bash
CUDA_VISIBLE_DEVICES=0,1 \
~/miniforge3/envs/wallx35/bin/torchrun --standalone --nproc_per_node=2 \
  tests/distributed/test_ddp_correctness.py
```

Result:

- `ALL DDP P1 TESTS PASSED`

Earlier targeted parity checks after the lifetime fix also passed:

- TP2 sync vs async vs async-drain: loss gap 0, final digest gap 0.
- HSDP*TP2 sync vs async: loss gap 0, final digest gap 0.

## Remaining Risk

The core ordering and correctness gates now cover TP2, HSDP*TP2, and non-TP
DDP. The remaining risk is performance shape at larger scale, especially the
relative launch overhead and overlap quality on IB instead of only local
NVLink. That belongs to the next 16-GPU/PAI smoke and 256-GPU readiness window,
not to this correctness split.
