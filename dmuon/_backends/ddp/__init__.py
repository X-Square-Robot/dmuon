"""DDP backend: dedicated params live on every rank.

Exports :class:`DedicatedParamDDP`, :class:`DedicatedParamGroupDDP`,
and :func:`replicate` — the companion helper that all-reduces
gradients on non-dedicated parameters.
"""

from .group import DedicatedParamGroupDDP
from .param import DedicatedParamDDP
from .replicate import replicate, replicate_tp

__all__ = [
    "DedicatedParamDDP",
    "DedicatedParamGroupDDP",
    "replicate",
    "replicate_tp",
]
