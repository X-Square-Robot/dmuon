"""Unit tests for the 2D (HSDP) partition path — Phase A.7."""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import pytest
import torch.nn as nn

from dmuon._core.partition import SMALL_PARAM_THRESHOLD, compute_balanced_assignment


class FakeDeviceMesh:
    """Minimal stub for DeviceMesh; partition only touches ``.size()``."""

    mesh_dim_names = None

    def __init__(self, world_size: int):
        self._world_size = world_size

    def size(self):
        return self._world_size


class MiniTransformerBlock(nn.Module):
    def __init__(self, hidden=512, intermediate=2048):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden // 4, bias=False)
        self.v_proj = nn.Linear(hidden, hidden // 4, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.ln = nn.LayerNorm(hidden)


class MiniModel(nn.Module):
    def __init__(self, num_layers=4, hidden=512, intermediate=2048):
        super().__init__()
        self.embed = nn.Embedding(1000, hidden)
        self.layers = nn.ModuleDict(
            {
                str(i): MiniTransformerBlock(hidden, intermediate)
                for i in range(num_layers)
            }
        )
        self.norm = nn.LayerNorm(hidden)


def _model(
    num_layers: int = 6, hidden: int = 512, intermediate: int = 2048
) -> nn.Module:
    return MiniModel(num_layers=num_layers, hidden=hidden, intermediate=intermediate)


# --- 1D backward compatibility ---------------------------------------------


def test_1d_returns_int_owners():
    """When replicate_mesh is None, the assignment must still return ints so
    existing call sites and checkpoints stay untouched.  This pins the
    backward-compat contract described in hsdp_native_dev_plan §6."""
    mesh = FakeDeviceMesh(8)
    model = _model()
    assignment = compute_balanced_assignment(
        model, mesh, predicate=lambda n, p: "proj" in n
    ).dp_owners
    assert assignment, "expected non-empty assignment"
    for owner in assignment.values():
        assert isinstance(owner, int), f"shard-only must return int, got {type(owner)}"


# --- 2D owner shape --------------------------------------------------------


def test_2d_returns_tuple_owners():
    shard_mesh = FakeDeviceMesh(4)
    replicate_mesh = FakeDeviceMesh(2)
    model = _model()
    assignment = compute_balanced_assignment(
        model,
        shard_mesh,
        predicate=lambda n, p: "proj" in n,
        replicate_mesh=replicate_mesh,
    ).dp_owners
    assert assignment
    for owner in assignment.values():
        assert (
            isinstance(owner, tuple) and len(owner) == 2
        ), f"HSDP path must return (shard, replicate) tuple; got {owner!r}"
        s, r = owner
        assert 0 <= s < 4 and 0 <= r < 2


# --- Load balance ----------------------------------------------------------


@pytest.mark.parametrize(
    "shard,replicate,num_layers",
    [
        (4, 2, 6),  # 8 slots
        (2, 4, 6),  # 8 slots, tall replicate
        (8, 2, 8),  # 16 slots
        (4, 4, 8),  # 16 slots, square
    ],
)
def test_2d_load_balance(shard, replicate, num_layers):
    """Global LPT over G*R slots should keep per-slot numel imbalance low."""
    shard_mesh = FakeDeviceMesh(shard)
    replicate_mesh = FakeDeviceMesh(replicate)
    model = _model(num_layers=num_layers)
    assignment = compute_balanced_assignment(
        model,
        shard_mesh,
        predicate=lambda n, p: "proj" in n,
        replicate_mesh=replicate_mesh,
    ).dp_owners

    slot_loads: dict[tuple[int, int], int] = {
        (s, r): 0 for s in range(shard) for r in range(replicate)
    }
    for param, coord in assignment.items():
        slot_loads[coord] += param.numel()

    loads = [v for v in slot_loads.values() if v > 0]
    assert len(loads) >= 1
    mx = max(loads)
    mn = min(loads)
    imbalance = (mx - mn) / max(mx, 1)
    # 10% is the bar documented in hsdp_native_dev_plan §3.1.
    assert imbalance < 0.10, (
        f"HSDP load imbalance too high: {imbalance:.1%} "
        f"(loads={sorted(slot_loads.items())})"
    )


def test_hsdp_lpt_balances_large_root_params_across_shard_columns():
    """Large non-layer tensors should not collapse onto one shard column.

    HSDP owner coords are 2D, but stage-2 reduce and post-step replicate
    publish are per shard column.  A plain 2D slot LPT can put two equal-size
    root tensors on different replicate coords of the same shard, which looks
    balanced per owner but serializes the inter-node bytes on one column.
    """
    shard_mesh = FakeDeviceMesh(4)
    replicate_mesh = FakeDeviceMesh(2)
    model = nn.Module()
    model.embed_tokens = nn.Embedding(1024, 8192)
    model.lm_head = nn.Linear(8192, 1024, bias=False)

    assignment = compute_balanced_assignment(
        model,
        shard_mesh,
        predicate=lambda _n, _p: True,
        replicate_mesh=replicate_mesh,
    ).dp_owners

    embed_owner = assignment[model.embed_tokens.weight]
    head_owner = assignment[model.lm_head.weight]
    assert embed_owner[0] != head_owner[0], (
        f"large root params should use different shard columns, got "
        f"embed={embed_owner}, lm_head={head_owner}"
    )


def test_hsdp_column_balance_can_be_disabled_for_ablation():
    shard_mesh = FakeDeviceMesh(4)
    replicate_mesh = FakeDeviceMesh(2)
    model = nn.Module()
    model.embed_tokens = nn.Embedding(1024, 8192)
    model.lm_head = nn.Linear(8192, 1024, bias=False)

    assignment = compute_balanced_assignment(
        model,
        shard_mesh,
        predicate=lambda _n, _p: True,
        replicate_mesh=replicate_mesh,
        hsdp_column_balance=False,
    ).dp_owners

    embed_owner = assignment[model.embed_tokens.weight]
    head_owner = assignment[model.lm_head.weight]
    assert embed_owner[0] == head_owner[0]
    assert embed_owner != head_owner


# --- Same-layer concurrency in 2D ------------------------------------------


def test_same_layer_distinct_slots_when_enough_slots():
    """Same-layer large params must occupy distinct 2D owner coords whenever
    the total number of slots exceeds the number of large params per layer.
    This preserves shard-dim broadcast concurrency in HSDP mode."""
    shard_mesh = FakeDeviceMesh(4)
    replicate_mesh = FakeDeviceMesh(2)  # 8 slots, layer has 5 large params
    # Bump dims so proj weights exceed the SMALL_PARAM_THRESHOLD merge bar.
    model = _model(num_layers=4, hidden=1024, intermediate=8192)
    assignment = compute_balanced_assignment(
        model,
        shard_mesh,
        predicate=lambda n, p: "proj" in n,
        replicate_mesh=replicate_mesh,
    ).dp_owners

    for layer_idx in range(4):
        layer_large_slots: list[tuple[int, int]] = []
        for name, param in model.named_parameters():
            if (
                f"layers.{layer_idx}" in name
                and "proj" in name
                and param.numel() >= SMALL_PARAM_THRESHOLD
            ):
                layer_large_slots.append(assignment[param])
        if not layer_large_slots:
            continue
        assert len(layer_large_slots) == len(set(layer_large_slots)), (
            f"Layer {layer_idx}: same-layer large params share slot "
            f"{layer_large_slots}"
        )


# --- Same-layer small-param merging ----------------------------------------


def test_small_params_merged_in_hsdp():
    """k_proj + v_proj (both small) in the same layer must share one 2D
    slot, so a single packed broadcast covers both."""
    shard_mesh = FakeDeviceMesh(4)
    replicate_mesh = FakeDeviceMesh(2)
    model = _model(num_layers=4)
    assignment = compute_balanced_assignment(
        model,
        shard_mesh,
        predicate=lambda n, p: "proj" in n,
        replicate_mesh=replicate_mesh,
    ).dp_owners
    for layer_idx in range(4):
        k = next(
            (
                p
                for n, p in model.named_parameters()
                if f"layers.{layer_idx}" in n and "k_proj" in n
            ),
            None,
        )
        v = next(
            (
                p
                for n, p in model.named_parameters()
                if f"layers.{layer_idx}" in n and "v_proj" in n
            ),
            None,
        )
        if k is None or v is None:
            continue
        if k.numel() < SMALL_PARAM_THRESHOLD and v.numel() < SMALL_PARAM_THRESHOLD:
            assert assignment[k] == assignment[v], (
                f"Layer {layer_idx}: small k/v on different slots "
                f"({assignment[k]} vs {assignment[v]})"
            )


def test_pack_small_params_can_be_disabled_in_hsdp():
    shard_mesh = FakeDeviceMesh(4)
    replicate_mesh = FakeDeviceMesh(2)
    model = _model(num_layers=1)
    proj_params = [
        p
        for name, p in model.named_parameters()
        if "layers.0" in name and "proj" in name
    ]

    packed = compute_balanced_assignment(
        model,
        shard_mesh,
        predicate=lambda n, p: "proj" in n,
        replicate_mesh=replicate_mesh,
        owner_strategy="round_robin",
    )
    no_pack = compute_balanced_assignment(
        model,
        shard_mesh,
        predicate=lambda n, p: "proj" in n,
        replicate_mesh=replicate_mesh,
        owner_strategy="round_robin",
        pack_small_params=False,
    )

    assert packed.allocation_unit_count == 1
    assert len(set(packed.dp_owners[p] for p in proj_params)) == 1
    assert no_pack.allocation_unit_count == len(proj_params)
    assert len(set(no_pack.dp_owners[p] for p in proj_params)) > 1


# --- Owner coord coverage --------------------------------------------------


def test_all_slots_used_when_enough_params():
    """With G=4, R=2 (8 slots) and 6 layers × 5 large proj, every slot
    should hold at least one large param.  The hidden / intermediate sizes
    are bumped so each proj exceeds SMALL_PARAM_THRESHOLD (otherwise they
    merge into one alloc unit per layer and under-fill the slot grid)."""
    shard_mesh = FakeDeviceMesh(4)
    replicate_mesh = FakeDeviceMesh(2)
    # Use larger dims so gate/up/down/q/o each cross the 5M threshold.
    model = _model(num_layers=6, hidden=1024, intermediate=8192)
    assignment = compute_balanced_assignment(
        model,
        shard_mesh,
        predicate=lambda n, p: "proj" in n,
        replicate_mesh=replicate_mesh,
    ).dp_owners
    used = set(assignment.values())
    expected = {(s, r) for s in range(4) for r in range(2)}
    assert used == expected, f"unused slots: {expected - used}"


# --- Degenerate: replicate_size = 1 behaves like shard-only -----------------


def test_replicate_size_1_matches_shard_only():
    """Passing replicate_mesh with size 1 must be semantically equivalent to
    passing no replicate_mesh (all replicate coords = 0)."""
    shard_mesh = FakeDeviceMesh(8)
    trivial_replicate = FakeDeviceMesh(1)
    model = _model(num_layers=4)

    flat = compute_balanced_assignment(
        model, shard_mesh, predicate=lambda n, p: "proj" in n
    ).dp_owners
    twod = compute_balanced_assignment(
        model,
        shard_mesh,
        predicate=lambda n, p: "proj" in n,
        replicate_mesh=trivial_replicate,
    ).dp_owners

    assert set(flat.keys()) == set(twod.keys())
    for p in flat:
        assert twod[p] == (flat[p], 0), f"mismatch: flat={flat[p]!r}, twod={twod[p]!r}"
