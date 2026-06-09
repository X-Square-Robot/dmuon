"""Canonical ``owner_rank`` normalisation.

HSDP support extends ``owner_rank`` from a bare shard-group index to a 2D coordinate
``(owner_shard, owner_replicate)`` so the same data structures can describe
both shard-only (1D) and HSDP (2D) layouts.  All code that constructs, groups
or compares ``owner_rank`` should route through :func:`normalize_owner_rank`
to stay consistent with the contract pinned by
``tests/unit/test_owner_rank_2d.py``.
"""

from __future__ import annotations

from typing import Tuple, Union

OwnerRankLike = Union[int, Tuple[int, int]]
OwnerCoord = Tuple[int, int]


def normalize_owner_rank(owner_rank: OwnerRankLike) -> OwnerCoord:
    """Normalise ``owner_rank`` to ``(owner_shard, owner_replicate)``.

    - ``int``: legacy 1D shard-only form → ``(int, 0)``.
    - ``Tuple[int, int]``: already 2D → passthrough after validation.
    """
    if isinstance(owner_rank, tuple):
        if len(owner_rank) != 2:
            raise TypeError(
                f"owner_rank tuple must be (shard, replicate); got {owner_rank!r}"
            )
        shard, replicate = owner_rank
        if not (isinstance(shard, int) and isinstance(replicate, int)):
            raise TypeError(
                f"owner_rank tuple entries must be int; got {owner_rank!r}"
            )
        if shard < 0 or replicate < 0:
            raise ValueError(f"owner_rank must be non-negative; got {owner_rank!r}")
        return shard, replicate
    # ``bool`` is a subclass of ``int`` — reject it explicitly so a stray
    # ``True`` / ``False`` does not silently degrade to rank 1 / 0.
    if isinstance(owner_rank, bool):
        raise TypeError(
            f"owner_rank must be int or Tuple[int, int]; got bool {owner_rank!r}"
        )
    if isinstance(owner_rank, int):
        if owner_rank < 0:
            raise ValueError(f"owner_rank must be non-negative; got {owner_rank!r}")
        return owner_rank, 0
    raise TypeError(
        f"owner_rank must be int or Tuple[int, int]; got {type(owner_rank).__name__}"
    )
