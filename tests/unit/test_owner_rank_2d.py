"""Phase A.0 smoke test: lock the behavioural contract for 2D ``owner_rank``.

Per ``hsdp_native_dev_plan.md`` §2 Phase A and memory
``feedback_smoke_test_before_refactor``, this test is written BEFORE
modifying ``dmuon/param.py`` to pin down the expected tuple semantics,
backwards compatibility with plain-int ``owner_rank``, and the rank-shift
API (``get_global_rank`` is still 1D over ``dp_group``).

No GPU / distributed required — the test operates on lightweight helpers
that encode the coordinate convention only.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from dmuon._core.owner_rank import normalize_owner_rank as _normalize_owner_rank


def test_int_is_promoted_to_2d():
    assert _normalize_owner_rank(0) == (0, 0)
    assert _normalize_owner_rank(3) == (3, 0)


def test_tuple_passthrough():
    assert _normalize_owner_rank((0, 0)) == (0, 0)
    assert _normalize_owner_rank((2, 1)) == (2, 1)
    assert _normalize_owner_rank((7, 3)) == (7, 3)


def test_tuple_shape_validation():
    with pytest.raises(TypeError):
        _normalize_owner_rank((1,))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _normalize_owner_rank((1, 2, 3))  # type: ignore[arg-type]


def test_tuple_element_type_validation():
    with pytest.raises(TypeError):
        _normalize_owner_rank(("0", 0))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _normalize_owner_rank((0, 1.5))  # type: ignore[arg-type]


def test_non_negative_invariant():
    with pytest.raises(ValueError):
        _normalize_owner_rank(-1)
    with pytest.raises(ValueError):
        _normalize_owner_rank((-1, 0))
    with pytest.raises(ValueError):
        _normalize_owner_rank((0, -1))


def test_rejects_non_int_non_tuple():
    with pytest.raises(TypeError):
        _normalize_owner_rank(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _normalize_owner_rank([0, 0])  # type: ignore[arg-type]


def test_by_owner_grouping_key_shape():
    """Downstream: ``DedicatedParamGroup._by_owner`` keys by 2D tuple.

    Assigning three params to ranks (0,0), (1,0), (0,0) should collapse into
    two owner buckets: {(0,0): [p0, p2], (1,0): [p1]}.
    """
    assignments = [
        ("p0", 0),
        ("p1", 1),
        ("p2", (0, 0)),
        ("p3", (1, 0)),
    ]
    buckets: dict[Tuple[int, int], list[str]] = {}
    for name, raw in assignments:
        key = _normalize_owner_rank(raw)
        buckets.setdefault(key, []).append(name)
    assert buckets == {(0, 0): ["p0", "p2"], (1, 0): ["p1", "p3"]}


def test_phase_a_default_replicate_is_zero():
    """Phase A behavioural invariant: without ``replicate_mesh``, the
    2D representation must always end in replicate dim 0 so that Phase A
    behaviour is identical to the pre-existing 1D shard-only path.
    """
    for shard in range(8):
        shard_coord, replicate_coord = _normalize_owner_rank(shard)
        assert shard_coord == shard
        assert replicate_coord == 0
