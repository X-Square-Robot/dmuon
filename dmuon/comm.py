"""DedicatedCommContext: shared communication state for dedicated parameter groups.

Analogous to FSDP2's FSDPCommContext — holds dedicated CUDA streams for
broadcast/reduce and tracks post-forward ordering for backward prefetch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .group import DedicatedParamGroup
    from .state import DedicatedState


class DedicatedCommContext:
    """Shared communication context across all DedicatedParamGroups.

    Streams:
        broadcast_stream: high-priority stream for NCCL broadcast kernels
        reduce_stream: high-priority stream for NCCL reduce kernels

    Prefetch state:
        post_forward_order: records which groups ran forward, in order.
            Used in backward to determine the next group to prefetch.

    Root post-backward fallback:
        all_states: every DedicatedState registers itself here so the
            autograd-engine callback can iterate and force-fire any group
            whose fast-path post-backward did not run (e.g., when no input
            tensor required gradient).
        post_backward_final_callback_queued: guards the callback so it is
            queued at most once per backward pass.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.broadcast_stream = torch.cuda.Stream(device=device, priority=-1)
        self.reduce_stream = torch.cuda.Stream(device=device, priority=-1)
        self.post_forward_order: list[DedicatedParamGroup] = []
        self.all_states: list[DedicatedState] = []
        self.post_backward_final_callback_queued: bool = False

    def reset_post_forward_order(self) -> None:
        """Clear post-forward order. Call at the start of each forward pass."""
        self.post_forward_order.clear()
