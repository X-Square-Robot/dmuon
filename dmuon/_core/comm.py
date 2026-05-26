"""DedicatedCommContext: shared communication state for dedicated parameter groups.

Analogous to FSDP2's FSDPCommContext — holds dedicated CUDA streams for
broadcast/reduce and tracks post-forward ordering for backward prefetch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from dmuon._core.state import DedicatedState

    # ``post_forward_order`` is shared across both backends; the list holds
    # whichever concrete group type is registered (FSDP2 or DDP), so we
    # leave it untyped rather than enumerate backends here.


class DedicatedCommContext:
    """Shared communication context across all DedicatedParamGroups.

    Streams:
        broadcast_stream: high-priority stream for NCCL broadcast kernels on
            the shard (``dp_group``) dimension.
        reduce_stream: high-priority stream for NCCL reduce kernels on the
            shard dimension.
        replicate_reduce_stream: default-priority stream for HSDP Stage-2
            replicate-axis reduce/all-reduce. It is intentionally separate
            from the post-step publish stream so large AdamW-route all-reduces
            do not sit in front of next-forward Muon publishes.
        replicate_broadcast_stream: default-priority stream reserved for the
            post-step inter-replicate-group broadcast used by HSDP-native Muon
            and TP post-step scatter. Initialised here so Phase A code can
            reference it without conditional guards; it stays idle until those
            paths are enabled.

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

    Rolling Stage-1 drain (1-outstanding):
        last_reduced_group: most recently ``reduce_grads``-dispatched group
            whose Stage-1 shard reduce has not been safety-drained yet.  Each
            new ``_run_post_backward`` waits only on that Stage-1 event before
            dispatching its own reduce, mirroring FSDP2's
            ``reduce_scatter_state`` buffer-lifetime wait.  HSDP Stage-2
            replicate reduce/all-reduce is intentionally *not* waited here; it
            is drained at the optimizer/root-post-backward boundary.  Reset by
            the root post-backward callback at the end of every backward pass.
    """

    def __init__(
        self,
        device: torch.device,
        replicate_group: Optional[dist.ProcessGroup] = None,
        *,
        tp_buffer_reuse: bool | str = False,
        replicate_broadcast_bucket_mb: float = 0.0,
    ):
        """Build the shared communication context for one model's
        dedicated-ownership groups.

        Args:
            device: CUDA device the streams and collectives run on.
            replicate_group: ``ProcessGroup`` spanning the replicate
                dimension of the HSDP 2D mesh. ``None`` in 1D shard-only
                mode (the default); downstream code short-circuits the
                replicate-dim reduce/broadcast when this is ``None``.
            tp_buffer_reuse: Whether TP gather/scatter should reuse per-param
                scratch buffers. Accepts ``False``/``True`` or ``"gather"``,
                ``"scatter"``, ``"all"``.
            replicate_broadcast_bucket_mb: Optional HSDP post-step publish
                bucket size in MiB. ``0`` keeps one coalesced publish per group.

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
        self.replicate_reduce_stream = torch.cuda.Stream(device=device)
        self.replicate_broadcast_stream = torch.cuda.Stream(device=device)
        self.replicate_group: Optional[dist.ProcessGroup] = replicate_group
        self.tp_buffer_reuse = self._normalize_tp_buffer_reuse(tp_buffer_reuse)
        if replicate_broadcast_bucket_mb <= 0:
            self.replicate_broadcast_bucket_bytes = 0
        else:
            self.replicate_broadcast_bucket_bytes = int(
                float(replicate_broadcast_bucket_mb) * 1024 * 1024
            )
        self.post_forward_order: list = []  # DedicatedParamGroup | DedicatedParamGroupDDP
        self.all_states: list[DedicatedState] = []
        self.post_backward_final_callback_queued: bool = False
        # Rolling-drain pointer for the 1-outstanding reduce policy; see the
        # class docstring's "Rolling reduce drain" section.  Untyped because
        # it holds whichever concrete group type the active backend uses
        # (FSDP2 or DDP), same convention as ``post_forward_order``.
        self.last_reduced_group = None

    @staticmethod
    def _normalize_tp_buffer_reuse(value: bool | str) -> str:
        if isinstance(value, bool):
            return "all" if value else "off"
        mode = str(value).strip().lower()
        aliases = {
            "": "off",
            "0": "off",
            "false": "off",
            "no": "off",
            "off": "off",
            "1": "all",
            "true": "all",
            "yes": "all",
            "on": "all",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"off", "all", "gather", "scatter"}:
            raise ValueError(
                "tp_buffer_reuse must be one of False, True, 'off', "
                "'gather', 'scatter', or 'all'"
            )
        return mode

    def tp_gather_buffer_reuse_enabled(self) -> bool:
        return self.tp_buffer_reuse in {"all", "gather"}

    def tp_scatter_buffer_reuse_enabled(self) -> bool:
        return self.tp_buffer_reuse in {"all", "scatter"}

    def reset_post_forward_order(self) -> None:
        """Clear post-forward order. Call at the start of each forward pass."""
        self.post_forward_order.clear()
        for state in self.all_states:
            indices = getattr(state.group, "_post_forward_indices", None)
            if indices is not None:
                indices.clear()
