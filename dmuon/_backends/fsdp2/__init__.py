"""FSDP2 backend: dedicated params with packed-buffer broadcast.

Exports :class:`DedicatedParam`, :class:`DedicatedParamGroup`, and
the FSDP2 monkey-patch installer. Combined with ``fully_shard`` on
non-dedicated parameters.
"""

from .group import DedicatedParamGroup
from .param import DedicatedParam
from .patch import install_patch, uninstall_patch

__all__ = [
    "DedicatedParam",
    "DedicatedParamGroup",
    "install_patch",
    "uninstall_patch",
]
