# Communication Cost Analysis

!!! tip "TL;DR"
    DMuon reaches the PyTorch-DP communication lower bound per step in
    **DMuon-Z2** mode — `2(N-1)/N · P_M` bytes, identical to a ring all-reduce.
    In **DMuon-Z3** mode it uses one extra `(N-1)/N · P_M` bytes (memory-vs-comm
    tradeoff matching ZeRO-3 convention).  Both modes eliminate the optimizer-step
    all-gather that naive FSDP2+Muon requires, and reduce NS compute from R replicas
    to 1.

---

## Notation

| Symbol | Meaning |
|---|---|
| N | Shard group size (DP degree in 1D; shard-dim size in HSDP) |
| R | Replicate group size (1 in 1D; replicate-dim size in HSDP) |
| P_M | Total number of elements in Muon-target (dedicated) parameters |
| P_p | Number of elements in one parameter |
| Ring all-reduce cost | `2(N-1)/N · P` (reduce-scatter + all-gather) |
| Broadcast / reduce cost | `(N-1)/N · P` (one direction only) |

All byte counts are per-parameter-element; multiply by `sizeof(dtype)` for
actual bytes on the wire.

---

## Four theorems — DP-family coverage

### Theorem 1: DDP

| | Naive Muon (DDP) | DMuon (DDP) |
|---|---|---|
| Backward | `2(N-1)/N · P_M` all-reduce | `(N-1)/N · P_M` reduce to owner |
| Forward broadcast | — | `(N-1)/N · P_M` broadcast from owner |
| **Total** | `2(N-1)/N · P_M` | `2(N-1)/N · P_M` |
| NS compute | N copies | **1 copy** |

**Result:** identical communication bytes; eliminates N-1 redundant NS
computations.  Worthwhile for any N > 1.

---

### Theorem 2a: DMuon-Z2 (FSDP, reshard_after_forward=False)

Naive FSDP2 + Muon requires three collectives:

1. Forward all-gather: `(N-1)/N · P_M`
2. Backward reduce-scatter: `(N-1)/N · P_M`
3. Optimizer all-gather (to reconstruct full gradient for NS): `(N-1)/N · P_M`

**Naive total:** `3(N-1)/N · P_M`

DMuon-Z2 replaces all three with:

1. Forward broadcast from owner: `(N-1)/N · P_M`
2. Backward reduce to owner: `(N-1)/N · P_M`

**DMuon-Z2 total:** `2(N-1)/N · P_M`

This equals the ring all-reduce lower bound for N ranks exchanging P_M
elements.  DMuon-Z2 hits the theoretical floor.

**Memory cost:** each rank stores P_M elements resident (the full parameter
on the owner; a broadcast-populated copy on non-owners retained through
forward+backward).

---

### Theorem 2b: DMuon-Z3 (FSDP, reshard_after_forward=True — default)

Naive FSDP2 + Muon requires four collectives:

1. Forward all-gather: `(N-1)/N · P_M`
2. Backward all-gather (re-materialize for gradient computation): `(N-1)/N · P_M`
3. Backward reduce-scatter: `(N-1)/N · P_M`
4. Optimizer all-gather: `(N-1)/N · P_M`

**Naive total:** `4(N-1)/N · P_M`

DMuon-Z3 replaces all four with:

1. Forward broadcast from owner: `(N-1)/N · P_M`
2. Re-broadcast in backward (parameters resharded after forward): `(N-1)/N · P_M`
3. Backward reduce to owner: `(N-1)/N · P_M`

**DMuon-Z3 total:** `3(N-1)/N · P_M`

Saves one full all-gather vs. naive FSDP2+Muon, plus eliminates redundant NS
compute.

**Memory cost:** non-owner ranks free the broadcast buffer after each forward;
only the owner holds P_M resident.  Per-layer packed buffer is transient.

---

### Theorem 3: HSDP (2D mesh, shard size N, replicate size R)

HSDP introduces a replicate dimension.  DMuon's two-stage protocol:

**Backward:** reduce gradient within shard group (`(N-1)/N · P_M`), then
AVG reduce across replicate group (`(R-1)/R · P_M`).  Total divisor = N·R,
matching a single world all-reduce.

**Post-step:** async broadcast of `_owned_data` from the global owner to
R-1 replicate peers.  This hides inside the next forward pass.

**Total per-step bytes:**

| Phase | Bytes |
|---|---|
| Shard-dim reduce (bwd) | `(N-1)/N · P_M` |
| Replicate-dim reduce (bwd) | `(R-1)/R · P_M` |
| Replicate broadcast (async, post-step / pre-fwd) | `(R-1)/R · P_M` |
| Shard broadcast (fwd) | `(N-1)/N · P_M` |

This matches the communication pattern of native HSDP (AG + RS + AR) while
cutting NS compute from N·R replicas down to 1.

---

## The lower bound

The ring all-reduce lower bound for N ranks exchanging P elements is
`2(N-1)/N · P`.  This is tight — any algorithm that requires every rank to
hold the updated parameter at the end of the step must communicate at least
this many elements.

**DMuon-Z2** achieves `2(N-1)/N · P_M` and thus hits the lower bound for
Muon-target parameters.

**DMuon-Z3** uses `3(N-1)/N · P_M`, exceeding the lower bound by one
`(N-1)/N` term.  This is the same overhead accepted by FSDP ZeRO-3 for
non-optimizer parameters: the extra communication buys reduced peak memory
by resharding parameters after each forward pass.

---

## Memory cost

| Mode | Memory per rank for Muon-target params |
|---|---|
| DMuon-Z2 (`reshard_after_forward=False`) | P_M per rank (full copy, resident) |
| DMuon-Z3 (`reshard_after_forward=True`, default) | P_M / N on owner; one packed layer buffer (transient) on non-owners during forward |

For large models at tight GPU memory budgets, Z3 is preferred.  For maximum
communication efficiency at the cost of memory, Z2 eliminates one broadcast
direction.

Choose via `dedicate_params(..., reshard_after_forward=False)` for Z2.

---

## Relation to Canzona

Canzona (Wang et al., arXiv:2602.06079) extends the dedicated-ownership
primitive to Megatron Tensor Parallelism + ZeRO-1 with Micro-Group Scheduling
and All-to-All communication.  DMuon and Canzona are sibling extensions of the
same primitive, independently pioneered by Distributed Shampoo (Shi et al.,
2023) and ZeRO-1 (Rajbhandari et al., 2020).

The key distinction is the target stack: **Canzona targets Megatron-LM** with
its TP+PP+ZeRO1 combination; **DMuon targets PyTorch DDP/FSDP2/HSDP** with no
Megatron dependency.  There is no direct head-to-head benchmark between the two
at this time.  Both systems can be cited when discussing the dedicated-ownership
primitive.

---

## Reproducing these numbers

Bit-identical correctness of HSDP communication is validated in
`tests/distributed/test_hsdp_correctness.py`: DMuon-HSDP matches shard-only
DMuon to bit precision over 10 training steps on a 4-GPU (G=2, R=2) harness.

Per-byte NCCL trace verification (Phase D) is planned; see `[TBD Phase D]` in
the roadmap.

---

## See also

- [HSDP guide](../guides/hsdp.md)
- [Z2 vs Z3 Modes](../guides/z2-z3-modes.md)
- [Design / Architecture](../design/architecture.md)
