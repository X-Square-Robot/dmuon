# Z2 vs Z3 Modes

!!! tip "TL;DR"
    DMuon-Z3 (the default) frees the packed buffer after each forward pass — lower
    memory, matches FSDP2 ZeRO-3 convention. DMuon-Z2 keeps the buffer resident
    through forward and backward — one fewer broadcast, lower communication cost.
    For most models above 3B parameters, stay with Z3.

---

## The two packed-buffer lifecycles

Every group of Muon-target parameters that share an owner is packed into a single
contiguous buffer for the shard-dimension broadcast. The *lifecycle* of that buffer
differs between the two modes:

| | **DMuon-Z3** (default) | **DMuon-Z2** |
|---|---|---|
| `reshard_after_forward` | `True` | `False` |
| After forward pass | buffer storage freed; placeholder tensor installed | buffer stays resident |
| Before backward pass | owner re-broadcasts from `_owned_data` | backward reuses the resident buffer |
| Number of shard broadcasts per step | 2 (one fwd, one bwd) | 1 (fwd only) |
| Steady-state memory per shard rank | one layer's packed buffer transient | full `P_M` resident |

The two modes are bit-identical: the same gradient values arrive at the owner. The
difference is purely in communication count and memory footprint.

---

## Byte cost per step

These figures count bytes moved on the **shard** process group (N ranks). `P_M`
is the total number of elements in all Muon-target parameters.

| Configuration | Comm bytes/step |
|---|---|
| Naive Muon on FSDP2-Z2 (no DMuon) | `3(N-1)/N · P_M` |
| **DMuon-Z2** | `2(N-1)/N · P_M` |
| Naive Muon on FSDP2-Z3 (no DMuon) | `4(N-1)/N · P_M` |
| **DMuon-Z3** | `3(N-1)/N · P_M` |

DMuon saves one gradient all-gather relative to the naive baseline in both modes
(the owner already has the gradient after the backward reduce, so no all-gather is
needed for the optimizer step). DMuon-Z2 additionally saves the backward re-broadcast,
reaching the communication-optimal `2(N-1)/N · P_M`.

Both modes also eliminate the `(N-1)x` redundant Newton-Schulz compute that naive
FSDP2+Muon performs (every rank runs NS on the full gradient in the naive case; with
dedicated ownership, only the single owner runs NS once).

---

## Memory cost

**DMuon-Z3**: at any point during training, only one layer's packed buffer is
allocated on the broadcast stream. Steady-state memory cost for Muon-target params:

```
memory ≈ max_layer_P_M  (transient, one layer at a time)
```

where `max_layer_P_M` is the largest single-layer packed buffer across all owned
groups on this rank.

**DMuon-Z2**: all packed buffers are resident simultaneously from the first forward
pass through the optimizer step. Steady-state memory cost:

```
memory ≈ P_M / N  (all owned packed buffers, one per owner shard column)
```

For a 7B model with 50% of parameters in Muon-target projections, `P_M ≈ 3.5B`
parameters (`≈ 7 GB` in bf16). On 8 shards, DMuon-Z2 adds `≈ 875 MB` per rank in
packed buffers — significant but not prohibitive for an 80 GB GPU.

---

## Decision tree

Treat these thresholds as starting points, not rules. The right choice still
depends on the Muon-target parameter ratio, free GPU memory, network topology,
activation memory, gradient accumulation settings, and measured throughput.

- **Model > 10B parameters?** → Start with DMuon-Z3 (default). In most cases, the
  extra forward broadcast is easier to accept than the additional resident memory.
- **Model < 3B and 8+ GPUs?** → DMuon-Z2 is a candidate. The backward broadcast
  savings are meaningful when communication dominates compute.
- **OOM with DMuon-Z3?** → Do not switch to Z2 first; packed buffers are transient
  in Z3. Check activation memory and gradient accumulation buffer size first.
- **Paired with `fully_shard(..., reshard_after_forward=X)`?** → Mirror the same
  value in `dedicate_params`. Symmetric configs keep the memory model predictable
  across Muon-target and non-Muon parameters.
- **HSDP (multi-node)?** → The choice applies equally. The shard-dimension broadcasts
  are counted above; the replicate-dimension broadcast is a separate post-step fan-out
  and is not affected by Z2/Z3.

---

## Asymmetric configurations

DMuon-Z2 combined with FSDP2-Z3 (or vice versa) is valid and occasionally the optimal
choice:

- **DMuon-Z2 + FSDP2-Z3**: useful when Muon-target params are few but individually
  large (e.g. a single giant projection), and non-Muon params (embeddings, norms) are
  numerous and small. Keeping the large Muon buffer resident saves one broadcast;
  FSDP2-Z3 keeps the many small params from bloating memory.
- **DMuon-Z3 + FSDP2-Z2**: less common. Reasonable if non-Muon params dominate
  memory and Muon-target params are small enough that the extra backward broadcast is
  cheap.

Asymmetric configs add mental overhead. Start with the symmetric configuration and
profile before experimenting.

---

## Relation to ZeRO-2 / ZeRO-3

The Z2/Z3 naming in DMuon follows the same convention as PyTorch FSDP2's
`reshard_after_forward`:

- **ZeRO-2 style** (`reshard_after_forward=False`): parameters are kept
  unsharded (gathered) through the forward and backward pass.
- **ZeRO-3 style** (`reshard_after_forward=True`): parameters are resharded
  (freed) after each forward pass and re-gathered as needed.

DMuon lifts Muon-target parameters *out of* FSDP2's sharded-state mechanism entirely —
they are not sharded in the FSDP2 sense at all. Instead, each parameter has a single
owner that holds the authoritative `_owned_data`, and the packed buffer is broadcast
from that owner. The Z2/Z3 flag then governs the *lifecycle of that packed buffer*,
mirroring the FSDP2 semantics on DMuon's own storage.

This means DMuon-Z2/Z3 and FSDP2-Z2/Z3 are independent knobs. They control different
storage paths and can be set independently, as described in the asymmetric section above.

---

## See also

- [HSDP Guide](hsdp.md) — Z2/Z3 in the context of HSDP multi-node training
- [Custom Hook Boundaries](custom-hook-boundaries.md) — hook granularity affects
  how many packed buffers exist simultaneously
- [Training Guide](training.md) — full single-node workflow
- [Communication Cost Analysis](../reference/communication-cost.md) — detailed
  byte-cost derivations
