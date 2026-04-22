"""Temporary instrumentation for investigating optimizer rank-load imbalance.

Gated behind the ``DMUON_PROFILE_BALANCE`` env var:
    * unset / "0"  — disabled (zero overhead)
    * "1"          — per-step rank-level muon/adamw timing
    * "2"          — also per-param timing inside _step_muon

Dumps partition assignment once at setup time when enabled.

This module is intended to be removed after the investigation concludes.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


def _level() -> int:
    raw = os.environ.get("DMUON_PROFILE_BALANCE", "0")
    try:
        return int(raw)
    except ValueError:
        return 0


def enabled() -> bool:
    return _level() >= 1


def per_param_enabled() -> bool:
    return _level() >= 2


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def ns_flops_estimate(shape: tuple[int, ...], ns_steps: int) -> int:
    """Rough NS-iteration FLOPs for a 2D (m, n) param.

    Per step the gram-space NS does:
      * 1 SYRK:   m·n·min(m,n)       (symmetric output, half of matmul)
      * 2 GEMMs:  2 · min(m,n)² · max(m,n)
    Total per NS step: min(m,n) · (m·n + 2·max(m,n)·min(m,n))
    """
    if len(shape) < 2:
        return int(shape[0]) * ns_steps
    m = int(shape[0])
    n = 1
    for d in shape[1:]:
        n *= int(d)
    k = min(m, n)
    big = max(m, n)
    per_step = k * (m * n + 2 * big * k)
    return per_step * ns_steps


# --- Partition dump --------------------------------------------------------

def dump_assignment(
    alloc_units,
    assignment,
    rank_loads,
    shard_size: int,
    replicate_size: int = 1,
    ns_steps: int = 5,
) -> None:
    """Print partition stats from rank 0. Called once at setup.

    Owner coords are 2D ``(shard, replicate)`` tuples.  When ``replicate_size
    == 1`` the output collapses to the old 1D listing.

    Args:
        alloc_units:     list of (params_list, layer_id, total_numel)
        assignment:      dict[param, (shard, replicate)]
        rank_loads:      dict[(shard, replicate), int] numel-weighted loads
        shard_size:      size of the shard (dp) dimension
        replicate_size:  size of the HSDP replicate dimension (1 in shard-only mode)
        ns_steps:        NS step count for flops estimate
    """
    if not enabled():
        return
    if _rank() != 0:
        return

    slots: list[tuple[int, int]] = [
        (s, r) for s in range(shard_size) for r in range(replicate_size)
    ]
    rank_flops: dict[tuple[int, int], int] = {slot: 0 for slot in slots}
    rank_param_count: dict[tuple[int, int], int] = {slot: 0 for slot in slots}
    rank_items: dict[
        tuple[int, int], list[tuple[str, tuple[int, ...], int, int]]
    ] = {slot: [] for slot in slots}

    # Build a param → (layer_id, numel) lookup so we can print per-param detail
    param_to_meta = {}
    for params_list, layer_id, total_numel in alloc_units:
        for p in params_list:
            numel = p.numel() if not hasattr(p, "_local_tensor") else p._local_tensor.numel()
            param_to_meta[id(p)] = (layer_id, numel)

    for p, owner in assignment.items():
        shape = tuple(p.shape)
        numel = p.numel() if not hasattr(p, "_local_tensor") else p._local_tensor.numel()
        flops = ns_flops_estimate(shape, ns_steps)
        rank_flops[owner] += flops
        rank_param_count[owner] += 1
        layer_id, _ = param_to_meta.get(id(p), (None, numel))
        rank_items[owner].append((str(layer_id), shape, numel, flops))

    total_slots = shard_size * replicate_size
    print("\n" + "=" * 78)
    print("[DMUON_PROFILE_BALANCE] Partition assignment summary")
    print("=" * 78)
    print(
        f"  shard={shard_size}, replicate={replicate_size}, "
        f"total_slots={total_slots}, ns_steps={ns_steps}"
    )
    print(f"  alloc units = {len(alloc_units)}, total params = {len(assignment)}")
    print()
    header = (
        f"  {'slot (s,r)':>12} {'n_params':>9} "
        f"{'numel (M)':>12} {'ns_flops (G)':>14}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for slot in slots:
        numel_m = rank_loads[slot] / 1e6
        flops_g = rank_flops[slot] / 1e9
        print(
            f"  {str(slot):>12} {rank_param_count[slot]:>9d} "
            f"{numel_m:>12.2f} {flops_g:>14.2f}"
        )

    def _stats(xs):
        if not xs:
            return 0.0, 0.0, 0.0, 0.0
        mx, mn = max(xs), min(xs)
        mean = sum(xs) / len(xs)
        return mx, mn, (mx / mn if mn > 0 else float("inf")), mean

    numel_vals = [rank_loads[s] for s in slots]
    flops_vals = [rank_flops[s] for s in slots]
    mx, mn, ratio, _ = _stats(numel_vals)
    print(f"\n  numel     : max={mx/1e6:.2f}M  min={mn/1e6:.2f}M  max/min={ratio:.3f}")
    mx, mn, ratio, _ = _stats(flops_vals)
    print(f"  ns_flops  : max={mx/1e9:.2f}G  min={mn/1e9:.2f}G  max/min={ratio:.3f}")
    print()
    print("  Per-slot param list (layer, shape, numel, ns_flops):")
    for slot in slots:
        print(f"  --- slot {slot} ---")
        for layer_id, shape, numel, flops in rank_items[slot]:
            print(
                f"    {layer_id:>12}  shape={str(shape):<22} "
                f"numel={numel/1e6:>7.2f}M  flops={flops/1e9:>7.2f}G"
            )
    print("=" * 78 + "\n", flush=True)


# --- Per-step timing -------------------------------------------------------

class StepTimer:
    """Per-step sync+timer. Use via:

        st = StepTimer()
        with st.phase("muon"):
            ...
        with st.phase("adamw"):
            ...
        st.report(step_idx, extra={"n_owned": ..., "numel": ...})
    """

    def __init__(self):
        self.timings: dict[str, float] = {}

    class _Phase:
        def __init__(self, parent: "StepTimer", name: str):
            self.parent = parent
            self.name = name
            self.t0 = 0.0

        def __enter__(self):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.parent.timings[self.name] = (time.perf_counter() - self.t0) * 1000.0

    def phase(self, name: str) -> "_Phase":
        return StepTimer._Phase(self, name)

    def report(self, step_idx: int, extra: Optional[dict] = None) -> None:
        """Gather rank timings onto rank 0 and print a table."""
        if not enabled():
            return
        if not (dist.is_available() and dist.is_initialized()):
            # Single-rank fallback: print directly
            msg = f"[balance step={step_idx}] rank=0 " + " ".join(
                f"{k}={v:.2f}ms" for k, v in self.timings.items()
            )
            if extra:
                msg += " " + " ".join(f"{k}={v}" for k, v in extra.items())
            print(msg, flush=True)
            return

        world = _world_size()
        rank = _rank()

        # Serialize timings + extras into a fixed tensor layout.
        phase_names = sorted(self.timings.keys())
        extra = extra or {}
        extra_names = sorted(extra.keys())
        # layout: [phase values...] [extra values...]
        local = torch.tensor(
            [self.timings[k] for k in phase_names]
            + [float(extra[k]) for k in extra_names],
            dtype=torch.float64,
            device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
        )
        gathered = [torch.zeros_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)

        if rank != 0:
            return

        n_phases = len(phase_names)
        print(f"[balance step={step_idx}]")
        header_cols = ["rank"] + phase_names + extra_names
        col_w = max(10, max(len(c) for c in header_cols) + 1)
        print("  " + "".join(f"{c:>{col_w}}" for c in header_cols))

        phase_vals = [[] for _ in phase_names]
        for r, t in enumerate(gathered):
            vals = t.tolist()
            phase = vals[:n_phases]
            for i, v in enumerate(phase):
                phase_vals[i].append(v)
            extras = vals[n_phases:]
            row = [f"{r}"]
            for v in phase:
                row.append(f"{v:.2f}")
            for v in extras:
                row.append(f"{int(v)}")
            print("  " + "".join(f"{c:>{col_w}}" for c in row))

        # Imbalance summary
        for name, vals in zip(phase_names, phase_vals):
            mx, mn = max(vals), min(vals)
            ratio = mx / mn if mn > 0 else float("inf")
            print(
                f"  {name}: max={mx:.2f}ms min={mn:.2f}ms "
                f"max/min={ratio:.3f}  (imbalance={mx - mn:.2f}ms)"
            )
        print(flush=True)


# --- Per-param timing (level 2) --------------------------------------------

class ParamTimer:
    """Per-param timing inside _step_muon. Only active at level >= 2."""

    def __init__(self):
        self.records: list[tuple[str, tuple[int, ...], float]] = []
        self._t0 = 0.0
        self._name = ""
        self._shape: tuple[int, ...] = ()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def start(self, name: str, shape: tuple[int, ...]) -> None:
        if not per_param_enabled():
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        self._name = name
        self._shape = shape

    def end(self) -> None:
        if not per_param_enabled():
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = (time.perf_counter() - self._t0) * 1000.0
        self.records.append((self._name, self._shape, ms))

    def report(self, step_idx: int) -> None:
        if not per_param_enabled() or not self.records:
            return
        rank = _rank()
        total = sum(ms for _, _, ms in self.records)
        print(
            f"[balance step={step_idx} rank={rank}] per-param "
            f"(total_muon={total:.2f}ms, n={len(self.records)}):",
            flush=True,
        )
        for name, shape, ms in self.records:
            print(f"    rank={rank}  {name:<48}  shape={str(shape):<22}  {ms:>8.2f}ms")
