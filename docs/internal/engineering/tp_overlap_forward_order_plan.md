# TP Overlap Forward-Order Engineering Plan

Status: Phase 4 profile/benchmark landed, Phase 4.5 prepare/prefetch hardening landed

Last updated: 2026-05-12

Owner: DMuon optimizer runtime

## 1. Target

DMuon TP post-step communication should overlap with the next forward while
preserving correctness for TP, HSDP+TP, and existing non-TP paths.

The chosen design is **group-coalesced + forward-order**:

1. Dispatch communication at the dedicated parameter group level, not at
   per-parameter granularity.
2. Reuse the previous iteration's recorded forward order as the post-step
   dispatch order.
3. Within each group, finish that group's Muon update before publishing it.
4. For TP groups, always run TP scatter before replicate-axis broadcast.
5. Defer the wait to the next forward pre-hook of the same group, so earlier
   layers can compute while later-layer post-step communication is still
   in flight.

## 2. Communication Contract

The required per-step order is:

```text
backward
  -> wait DP/HSDP reduce
  -> TP gather reduced grads to TP owner
optimizer.step
  -> for group in previous_forward_order:
       owner ranks run Muon/NS for that group
       group dispatches TP scatter of the full-matrix delta, if TP-sharded
       group dispatches replicate broadcast, if HSDP replicate is enabled
  -> AdamW updates non-dedicated params
next forward
  -> each group's pre-forward hook waits its own pending TP scatter
  -> the same hook waits its own pending replicate broadcast
  -> unshard reads fresh _owned_data
```

NCCL ordering is part of the correctness contract. Every rank must enter the
same group sequence for every collective-bearing path. Rank-local ownership
only gates the computation that prepares send buffers; it must not change the
collective dispatch sequence.

## 3. Current Baseline

The worktree already contains most of the runtime foundation:

- `dmuon.utils._ordered_post_step_groups()` returns the previous forward order,
  falling back to module walk order for the first step or skipped modules.
- `Muon._step_muon_and_dispatch_groups_async()` walks that group order and
  dispatches a group immediately after its local Muon update.
- `DedicatedParamGroup._tp_scatter_dispatch()` coalesces TP scatter by TP
  process group inside one dedicated parameter group.
- `tp_scatter_delta_async()` and `replicate_broadcast_async()` enqueue on
  `replicate_broadcast_stream`.
- `DedicatedParamGroup._pre_forward_wait()` drains TP scatter before replicate
  broadcast before unshard reads `_owned_data`.
- Phase 1 adds `summarize_post_step_groups()` and
  `format_post_step_group_summary()` so logs/tests can inspect effective group
  order and pending async state without issuing collectives.

The remaining work is mostly hardening, observability, correctness coverage,
and profile validation.

## 4. Invariants

1. Async and sync modes must produce the same loss trajectory for the same
   seed, topology, model, and data.
2. TP scatter must publish the post-NS delta before any replicate peer can
   consume the updated `_owned_data`.
3. HSDP+TP must keep scatter and replicate broadcast on the same stream unless
   there is an explicit event dependency.
4. DDP and non-TP FSDP2 paths must remain behavior-compatible.
5. Semantic optimizer `param_groups` must keep their per-group hyperparameters
   even when execution order is communication-group order.
6. Checkpoint/state-dict reads must drain all pending post-step states.
7. Fallback-to-sync must be per group and must not change collective ordering.

## 5. Phased Plan

### Phase 0: Plan and Branch Hygiene

- Create a dedicated feature branch from the current clean worktree.
- Land this plan under `docs/internal/engineering/`.
- Keep Wall-X-side changes out of this branch.

Acceptance:

- Worktree branch is isolated for TP overlap work.
- The engineering contract is explicit enough to review before core changes.

### Phase 1: Order and State Observability

Status: landed in this branch.

- Add a read-only diagnostic summary for post-step groups.
- Report, per group:
  - forward-order index;
  - module names using the group;
  - dedicated param count;
  - TP-sharded param count;
  - local owner count;
  - pending TP scatter state;
  - pending replicate broadcast state;
  - fallback flags;
  - last measured wait times.
- Export a compact formatter next to existing diagnostics.

Acceptance:

- Done: users and tests can print the effective forward-order post-step
  schedule.
- Done: no distributed collective is issued by the diagnostic helper.
- Done: the helper works before training starts and is designed to be safe
  after async dispatch.

### Phase 2: Runtime Hardening

Status: partially landed in this branch.

- Make the group dispatch label explicit in step profiling so TP scatter and
  replicate broadcast can be identified separately from Muon compute.
- Audit all post-step dispatch sites so sync and async modes use the same
  group ordering where collectives are involved.
- Strengthen double-dispatch errors with group/module context.
- Keep TP scatter and replicate broadcast in one ordered group pipeline.

Acceptance:

- Pending: a profile can distinguish Muon compute, TP scatter dispatch, and replicate
  broadcast dispatch at group granularity.
- Done: sync and async post-step publish paths share the same forward-order
  group sequence.
- Done: double-dispatch failures identify the group that missed its
  pre-forward wait.
- Done: no new per-parameter collective path is introduced.

### Phase 3: Correctness Tests

Status: landed for the current 8-GPU validation window.

Extend existing distributed tests instead of adding duplicate test harnesses.
Required matrix:

| Topology | Size | Checks |
| --- | ---: | --- |
| TP only | TP2, TP4 | sync vs async loss, state drain, TP owner scatter |
| HSDP+TP | HSDP2*TP2, HSDP2*TP4 when available | sync vs async loss, scatter before replicate |
| FSDP2 non-TP | DP2/DP4 | no regression in async replicate path |
| semantic param groups + TP | TP2, HSDP2*TP2 | per-group LR effect preserved |

Cross-topology loss comparisons should use the same seed, model structure, and
synthetic data. Exact bitwise equality is required only when topology and
execution mode are supposed to execute the same floating-point order. Across
different topologies, compare bounded drift and monotonic trend rather than
forcing bitwise equality.

Acceptance:

- Done: `test_tp_comm_order.py` now dispatches with
  `_ordered_post_step_groups()` and records `summarize_post_step_groups()` in
  JSON output.
- Done on remote GPU, 2026-05-11:
  - TP4 async, tiny model, 2 steps: passed; pinned TP refs=24.
  - TP4 sync, tiny model, 2 steps: passed; losses match async exactly.
  - HSDP2*Shard2*TP2 async, tiny model, 2 steps: passed; pinned TP refs=12,
    pending replicate states=24.
  - HSDP2*Shard2*TP2 sync, tiny model, 2 steps: passed; losses match async
    exactly.
- Done on remote GPU, 2026-05-11:
  - Full Llama-shaped matrix passed with strict sync-vs-async bit equality for
    losses and final digests: TP2, TP4, DP*TP2, DP*TP4, HSDP*TP2.
  - Full Qwen-shaped matrix passed with strict sync-vs-async bit equality for
    losses and final digests: TP2, DP*TP2, HSDP*TP2.
  - Qwen TP4 full was rejected as an expected invalid topology because
    `kv_heads=2` is not divisible by `tp_size=4`.
  - Cross-topology loss drift checks passed for TP-only, DP-only, DP*TP, and
    HSDP*TP comparisons.
  - Artifact directory:
    `docs/internal/report/tp_overlap_forward_order/phase3_llm_full_bit_exact_20260511`.
- Done: the distributed alignment harness now runs deterministic cublas NS,
  disables TF32, uses eager attention, retries occupied rendezvous ports, and
  avoids inserting per-step world collectives into the raw async hot path.
- Done: targeted TP4 final param digest debug passed with zero loss,
  final-weight-digest, and per-parameter-digest gaps. Artifact directory:
  `docs/internal/report/tp_overlap_forward_order/debug_tp4_bit_exact_20260511`.

### Phase 4: Profile and Benchmark Validation

Status: landed for the current 8-GPU validation window.

- Capture remote GPU profiles for Qwen2.5-1.5B-shaped synthetic training with
  batch size 2 and sequence length 4096.
- Run at least:
  - DDP baseline;
  - HSDP baseline;
  - TP2;
  - HSDP2*TP2.
- Save rank-0 torch profiler traces and concise timing summaries.

Acceptance:

- Done: `benchmarks/bench_tp_llm.py` now supports rank-0 torch profiler traces,
  profiler markers for forward/backward/optimizer/post-step dispatch, no-TP
  DP/HSDP baselines, and a `phase4` benchmark matrix.
- Done on remote GPU, 2026-05-11:
  - Qwen2.5-1.5B original shape, random init, synthetic data, batch size 2,
    sequence length 4096, full forward/backward/optimizer.
  - 8-GPU matrix artifacts:
    `docs/internal/report/tp_overlap_forward_order/phase4_qwen1b_4096_20260511`.
  - Rank-0 torch profiler traces saved under:
    `docs/internal/report/tp_overlap_forward_order/phase4_qwen1b_4096_20260511/profiles`.
  - HSDP*TP2 async reached 1476 ms p50 vs 1929 ms sync p50, a 1.31x
    speedup in the matrix run.
  - DP*TP2 repeat run reached 545 ms p50 async vs 569 ms sync, a 1.04x
    speedup after short-matrix variance was removed.
  - TP2 async reached 610 ms p50 vs 623 ms sync p50, a 1.02x speedup.
- Done: profiler traces include `dmuon.tp_scatter_delta.*`,
  `dmuon.replicate_broadcast.*`, `dmuon.pre_forward_wait.*`,
  `dmuon.optimizer.*`, and `dmuon.bench.*` user ranges.
- Done: MFU/step-time summaries are from complete LLM forward+backward+step
  runs, not MLP microbenchmarks. MFU uses a simple `6 * params * tokens`
  estimate and should be treated as an approximate utilization indicator.

### Phase 4.5: Prepare-Grads Split and TP Gather Prefetch

Status: landed in this branch.

Motivation:

- `dmuon.optimizer.wait_reduces` was misleading because it contained both
  reduce-tail waits and TP grad gather launches.
- For TP, launching every gather under one global prepare point creates a CPU
  launch cluster and gives no chance to overlap later groups' TP gather with
  current-group NS compute.
- Non-TP DDP/FSDP2 paths do not benefit from TP gather prefetch and should keep
  the old global prepare behavior.

Implementation:

- Add `prepare_muon_grads(model)` as the explicit public operation after
  backward. Keep `wait_all_reduces(model)` as a compatibility alias.
- Add per-group profile ranges:
  - `dmuon.prepare_muon_grads.<group>`;
  - `dmuon.wait_reduce_tail.<group>`;
  - `dmuon.tp_gather_grads.<group>`.
- Add `prepare_group_muon_grads()` and `wait_group_muon_grads()` helpers.
- Let `wait_for_reduce(stream=...)` unpack owner grads on `reduce_stream` for
  the TP prefetch path without inserting a compute-stream wait.
- Let `tp_gather_grads(wait_current_stream=False)` skip
  `reduce_stream.wait_stream(current_stream)` when the reduced grad was already
  produced on `reduce_stream`.
- In async TP optimizer steps, use one-group lookahead:
  1. prepare group 0;
  2. for each group in forward order, prepare group `i+1`, wait group `i`,
     run group-local Muon, then dispatch that group's TP scatter/replicate
     publish.
- Gate this prefetch path on `self._has_tp_dedicated`. Pure DDP/non-TP async
  keeps the old global `prepare_muon_grads` before group publish.

Hardening:

- Pin reduce-unpack source tensors in `_muon_grad_ready_refs` until the ready
  event is consumed.
- Pin TP gather receive/send tensors in `_tp_gather_refs` until the TP gather
  event is consumed.
- This fixed a pure TP async regression where `tp2 async` had a step-1 loss
  gap and `async_drain` could produce NaN because transient gather buffers were
  eligible for allocator reuse before the reduce stream finished.
- The DDP param-group async test exposed that non-TP DDP should not inherit
  the TP prefetch scheduler. The runtime now restricts group prefetch to
  models with TP-sharded dedicated params.

Validation on remote GPU, 2026-05-12:

- Detailed issue report:
  `docs/internal/engineering/tp_prepare_prefetch_report_20260512.md`.
- Local/static:
  - `python -m compileall dmuon tests/unit/test_priority_order.py tests/distributed/test_tp_comm_order.py`
  - `ruff check dmuon/__init__.py dmuon/utils.py dmuon/optim/muon.py dmuon/_backends/ddp/group.py dmuon/_backends/fsdp2/group.py tests/unit/test_priority_order.py tests/distributed/test_tp_comm_order.py`
  - `pytest -q tests/unit/test_priority_order.py`: 8 passed.
- Remote state-machine:
  - `tp2 async`, tiny, 2 steps: passed.
  - `tp2 sync`, tiny, 2 steps: passed.
  - `hsdp_tp2 async`, tiny, 2 steps: passed with pending replicate states.
- Remote optimizer-step loss parity:
  - `tp2 sync` vs `tp2 async` vs `tp2 async_drain`: loss gap 0,
    final digest gap 0.
  - `hsdp_tp2 sync` vs `hsdp_tp2 async`: loss gap 0, final digest gap 0.
  - DDP P1 script: all tests passed, including param-group async-vs-sync
    loss trajectory.

### Phase 5: 16-GPU Smoke and 256-GPU Readiness

- Use 16 GPUs as the largest pre-window validation target when 32 GPUs are not
  schedulable.
- Feed completed smoke results into the experiment dashboard tables only after
  real runs finish.
- Update the 256-GPU engineering plan with pass/fail status and blockers.

Acceptance:

- 16-GPU smoke covers at least one TP-bearing topology.
- The 256-GPU runbook has no open code-path unknowns for TP overlap.
- Remaining risks are cluster capacity or scale-specific performance, not
  missing correctness coverage.

## 6. Risk Register

| Risk | Mitigation |
| --- | --- |
| Collective order divergence across ranks | Single `_ordered_post_step_groups()` source of truth plus diagnostics |
| Async state reused before wait | Per-group pending-state errors and checkpoint drains |
| Scatter/broadcast dependency bug | Same stream dispatch and pre-forward wait order |
| Param-group LR lost in group-ordered execution | Continue calling Muon update through subgroup metadata |
| Profile instrumentation changes timing | Keep detailed timing behind explicit env/profile flags |
| TP4 unavailable locally | Skip with clear reason locally, require remote/PAI coverage before merge |

## 7. Development Rule

Every phase should end with:

1. code or documentation diff reviewed locally;
2. the smallest meaningful test/check run;
3. progress synchronized back into this document if scope or status changes.
