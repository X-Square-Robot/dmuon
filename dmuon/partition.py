"""Back-compat re-export shim.

The partition module now lives at :mod:`dmuon._core.partition`.  This
module preserves the pre-refactor import path so older tests and
downstream callers continue to work unchanged::

    from dmuon.partition import compute_balanced_assignment, SMALL_PARAM_THRESHOLD
"""

from dmuon._core.partition import (  # noqa: F401
    AssignmentResult,
    OwnerCoord,
    OwnerValue,
    SMALL_PARAM_THRESHOLD,
    _extract_layer_id,
    _param_numel,
    compute_balanced_assignment,
)

__all__ = [
    "AssignmentResult",
    "OwnerCoord",
    "OwnerValue",
    "SMALL_PARAM_THRESHOLD",
    "compute_balanced_assignment",
]
