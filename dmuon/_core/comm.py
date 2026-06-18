"""DedicatedCommContext: shared communication state for dedicated parameter groups.

Analogous to FSDP2's FSDPCommContext — holds dedicated CUDA streams for
broadcast/reduce and tracks post-forward ordering for backward prefetch.
"""

from __future__ import annotations

import os
from collections import defaultdict
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
        self.sharded_adamw_unshard_stream = torch.cuda.Stream(
            device=device, priority=-1
        )
        self.sharded_adamw_unshard_separate_stream_enabled = False
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
        self.post_forward_order: list = (
            []
        )  # DedicatedParamGroup | DedicatedParamGroupDDP
        self.all_states: list[DedicatedState] = []
        self.forward_prefetch_depth = 1
        self.post_backward_final_callback_queued: bool = False
        # Rolling-drain pointer for the 1-outstanding reduce policy; see the
        # class docstring's "Rolling reduce drain" section.  Untyped because
        # it holds whichever concrete group type the active backend uses
        # (FSDP2 or DDP), same convention as ``post_forward_order``.
        self.last_reduced_group = None
        self.record_forward_profile = _env_flag("DMUON_RECORD_FORWARD_PROFILE")
        self.record_post_step_profile = self.record_forward_profile or _env_flag(
            "DMUON_RECORD_POST_STEP_PROFILE"
        )
        self._forward_profile_events: list[dict[str, object]] = []
        self._post_step_profile_events: list[dict[str, object]] = []
        self._forward_profile_counts: dict[str, dict[str, object]] = defaultdict(
            lambda: {
                "dispatch_calls": 0,
                "prefetch_dispatch_calls": 0,
                "demand_dispatch_calls": 0,
                "prefetch_hits": 0,
                "already_unsharded": 0,
                "owner_broadcast_bytes": 0,
                "owner_broadcast_max_bucket_bytes": 0,
                "owner_broadcast_bucket_count": 0,
                "sharded_adamw_all_gather_bytes": 0,
                "sharded_adamw_param_count": 0,
                "sharded_muon_all_gather_bytes": 0,
                "sharded_muon_param_count": 0,
                "wait_calls": 0,
                "tp_publish_waits": 0,
                "replicate_publish_waits": 0,
                "sharded_muon_publish_waits": 0,
                "prefetch_publish_not_ready_skips": 0,
                "prefetch_tp_publish_not_ready_skips": 0,
                "prefetch_replicate_publish_not_ready_skips": 0,
                "prefetch_sharded_muon_publish_not_ready_skips": 0,
                "prefetch_unready_publish_waits_queued": 0,
                "prefetch_unready_tp_publish_waits_queued": 0,
                "prefetch_unready_replicate_publish_waits_queued": 0,
                "prefetch_unready_sharded_muon_publish_waits_queued": 0,
            }
        )

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

    def record_forward_unshard_counter(self, group_name: str, **values: int) -> None:
        if not self.record_forward_profile:
            return
        stats = self._forward_profile_counts[group_name]
        for key, value in values.items():
            if key.endswith("_max_bucket_bytes"):
                stats[key] = max(int(stats.get(key, 0)), int(value))
            else:
                stats[key] = int(stats.get(key, 0)) + int(value)

    def record_forward_unshard_event(
        self,
        *,
        group_name: str,
        phase: str,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
        bytes: int = 0,
        prefetch: bool = False,
    ) -> None:
        if not self.record_forward_profile:
            return
        self._forward_profile_events.append(
            {
                "group": group_name,
                "phase": phase,
                "bytes": int(bytes),
                "prefetch": bool(prefetch),
                "start": start,
                "end": end,
            }
        )

    def record_post_step_event(
        self,
        *,
        group_name: str,
        phase: str,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
        bytes: int = 0,
    ) -> None:
        if not self.record_post_step_profile:
            return
        self._post_step_profile_events.append(
            {
                "group": group_name,
                "phase": phase,
                "bytes": int(bytes),
                "start": start,
                "end": end,
            }
        )

    def _consume_event_profile(
        self,
        events: list[dict[str, object]],
        *,
        clear: bool,
    ) -> dict[str, object]:
        events_ready = 0
        events_pending = 0
        by_group: dict[str, dict[str, object]] = {}
        by_phase: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"count": 0, "ms": 0.0, "bytes": 0}
        )

        for item in events:
            end = item["end"]
            if isinstance(end, torch.cuda.Event) and not end.query():
                events_pending += 1
                continue
            start = item["start"]
            if not isinstance(start, torch.cuda.Event) or not isinstance(
                end, torch.cuda.Event
            ):
                events_pending += 1
                continue
            try:
                elapsed_ms = float(start.elapsed_time(end))
            except RuntimeError:
                events_pending += 1
                continue

            events_ready += 1
            group = str(item["group"])
            phase = str(item["phase"])
            bytes_value = int(item.get("bytes", 0))
            group_stats = by_group.setdefault(group, {})
            key = f"{phase}_ms"
            group_stats[key] = float(group_stats.get(key, 0.0)) + elapsed_ms
            group_stats[f"{phase}_events"] = (
                int(group_stats.get(f"{phase}_events", 0)) + 1
            )
            if bytes_value:
                group_stats[f"{phase}_bytes"] = (
                    int(group_stats.get(f"{phase}_bytes", 0)) + bytes_value
                )

            phase_stats = by_phase[phase]
            phase_stats["count"] = int(phase_stats["count"]) + 1
            phase_stats["ms"] = float(phase_stats["ms"]) + elapsed_ms
            phase_stats["bytes"] = int(phase_stats["bytes"]) + bytes_value

        for group_stats in by_group.values():
            for key, value in list(group_stats.items()):
                if key.endswith("_ms"):
                    group_stats[key] = round(float(value), 6)

        result: dict[str, object] = {
            "events_ready": events_ready,
            "events_pending": events_pending,
            "by_group": by_group,
            "by_phase": {
                phase: {
                    "count": int(values["count"]),
                    "ms": round(float(values["ms"]), 6),
                    "bytes": int(values["bytes"]),
                }
                for phase, values in by_phase.items()
            },
        }
        if clear:
            events.clear()
        return result

    def consume_forward_unshard_profile(
        self, *, clear: bool = True
    ) -> dict[str, object]:
        """Return aggregate CUDA-event timings for forward unshard diagnostics."""

        event_profile = self._consume_event_profile(
            self._forward_profile_events,
            clear=clear,
        )
        events_ready = int(event_profile["events_ready"])
        events_pending = int(event_profile["events_pending"])
        by_group: dict[str, dict[str, object]] = {
            group: dict(values)
            for group, values in self._forward_profile_counts.items()
        }
        for group, profile_stats in event_profile["by_group"].items():
            group_stats = by_group.setdefault(group, {})
            group_stats.update(profile_stats)

        for group_stats in by_group.values():
            for key, value in list(group_stats.items()):
                if key.endswith("_ms"):
                    group_stats[key] = round(float(value), 6)

        result: dict[str, object] = {
            "enabled": self.record_forward_profile,
            "events_ready": events_ready,
            "events_pending": events_pending,
            "by_group": by_group,
            "by_phase": event_profile["by_phase"],
            "post_step_profile": self.consume_post_step_profile(clear=clear),
        }
        if clear:
            self._forward_profile_counts.clear()
        return result

    def consume_post_step_profile(self, *, clear: bool = True) -> dict[str, object]:
        result = self._consume_event_profile(
            self._post_step_profile_events,
            clear=clear,
        )
        result["enabled"] = self.record_post_step_profile
        return result


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}
