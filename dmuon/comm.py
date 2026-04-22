"""DedicatedCommContext: shared communication state for dedicated parameter groups.

Analogous to FSDP2's FSDPCommContext — holds dedicated CUDA streams for
broadcast/reduce and tracks post-forward ordering for backward prefetch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from .group import DedicatedParamGroup
    from .state import DedicatedState


class DedicatedCommContext:
    """Shared communication context across all DedicatedParamGroups.

    Streams:
        broadcast_stream: high-priority stream for NCCL broadcast kernels on
            the shard (``dp_group``) dimension.
        reduce_stream: high-priority stream for NCCL reduce kernels on the
            shard dimension.
        replicate_broadcast_stream: default-priority stream reserved for the
            inter-replicate-group broadcast used by HSDP-native Muon
            (Phase C).  Initialised here so Phase A code can reference it
            without conditional guards; it stays idle until Phase C wires
            async forward broadcast.

    Process groups:
        replicate_group: ``ProcessGroup`` spanning the replicate dimension of
            the HSDP 2D mesh.  ``None`` in 1D shard-only mode, which is the
            Phase A default — downstream code falls back to the previous
            single-dimension collectives when this is None.

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

    def __init__(
        self,
        device: torch.device,
        replicate_group: Optional[dist.ProcessGroup] = None,
    ):
        """Build the shared communication context for one model's
        dedicated-ownership groups.

        Args:
            device: CUDA device the streams and collectives run on.
            replicate_group: ``ProcessGroup`` spanning the replicate
                dimension of the HSDP 2D mesh. ``None`` in 1D shard-only
                mode (the default); downstream code short-circuits the
                replicate-dim reduce/broadcast when this is ``None``.

        Normally constructed by :func:`dmuon.dedicate_params`, not by
        user code directly.
        """
        self.device = device
        self.broadcast_stream = torch.cuda.Stream(device=device, priority=-1)
        self.reduce_stream = torch.cuda.Stream(device=device, priority=-1)
        # FSDP2 sets the all-reduce (i.e. replicate-dim) stream at default
        # priority because inter-node AR traffic uses different network
        # resources than intra-node AG/RS.  We follow the same convention so
        # the Phase C scheduler can issue broadcasts on this stream without
        # starving shard-dim collectives.
        self.replicate_broadcast_stream = torch.cuda.Stream(device=device)
        self.replicate_group: Optional[dist.ProcessGroup] = replicate_group
        self.post_forward_order: list[DedicatedParamGroup] = []
        self.all_states: list[DedicatedState] = []
        self.post_backward_final_callback_queued: bool = False

    def reset_post_forward_order(self) -> None:
        """Clear post-forward order. Call at the start of each forward pass."""
        self.post_forward_order.clear()
