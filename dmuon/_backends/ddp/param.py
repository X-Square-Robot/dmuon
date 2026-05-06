"""DedicatedParamDDP: DDP-path variant of DedicatedParam.

Differs from :class:`dmuon.param.DedicatedParam` in three axes
(storage / compute / comm are orthogonal — see
``docs/internal/research/ddp_adapter_plan.md`` §2):

* **Storage**: every rank keeps the full parameter as a live
  ``nn.Parameter`` on the module, AND every rank clones the current value
  into ``_owned_data`` as the authoritative copy for NS update on the
  owner. The original ``nn.Parameter`` is NOT replaced by a 0-size
  placeholder.
* **Compute**: only the owner rank runs Newton-Schulz (identical to the
  FSDP2 path).
* **Communication**: after backward, gradient is ``dist.reduce``-d to
  the owner (dst=owner). After ``optim.step``, the owner broadcasts the
  updated ``_owned_data`` back to all ranks which then ``copy_`` it
  into their local ``nn.Parameter.data``.

The public attribute surface is kept intentionally compatible with
``DedicatedParam`` so the Muon optimizer, checkpoint code, and
``DedicatedState`` hooks can treat both types identically via duck
typing. Attributes that are meaningful only on the FSDP2 path
(``_placeholder``, ``_unsharded_param``, packed-buffer binding) are
absent here, and no FSDP2-path method touches them for a DDP group
since the group's ``unshard`` / ``reshard`` / ``wait_for_unshard`` are
no-ops.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from dmuon._core.owner_rank import OwnerCoord, OwnerRankLike, normalize_owner_rank


class DedicatedParamDDP:
    """DDP-path dedicated parameter.

    Every rank keeps the full parameter live; the owner additionally
    runs Newton-Schulz on its ``_owned_data`` and broadcasts the updated
    value back every step.
    """

    def __init__(
        self,
        param: nn.Parameter,
        module: nn.Module,
        param_name: str,
        owner_rank: OwnerRankLike,
        dp_group: dist.ProcessGroup,
        device: torch.device,
        compute_dtype: Optional[torch.dtype] = None,
    ):
        self.module = module
        self.param_name = param_name

        coord: OwnerCoord = normalize_owner_rank(owner_rank)
        self.owner_rank: OwnerCoord = coord
        self.owner_shard: int = coord[0]
        self.owner_replicate: int = coord[1]
        self.dp_group = dp_group
        self.replicate_group: Optional[dist.ProcessGroup] = None  # DDP path: always 1D
        self.device = device

        self.is_owner: bool = dp_group.rank() == self.owner_shard
        self._owner_global_rank: int = dist.get_global_rank(dp_group, self.owner_shard)
        self._owner_replicate_global_rank: Optional[int] = None  # unused in 1D

        # DDP path does not support DTensor on dedicated params (TP + DDP is
        # out of scope). Keep the attribute for interface parity with
        # DedicatedParam so downstream code (Muon, checkpoint) does not
        # special-case.
        self.is_dtensor: bool = False
        self._tp_spec = None

        local_data = param.data
        self._orig_size: torch.Size = local_data.size()
        self._orig_dtype: torch.dtype = local_data.dtype
        self._compute_dtype: Optional[torch.dtype] = compute_dtype
        self._requires_grad: bool = param.requires_grad

        self.numel: int = self._orig_size.numel()
        self.shard_dim: Optional[int] = None
        self.full_shape: torch.Size = self._orig_size
        self.tp_group: Optional[dist.ProcessGroup] = None

        # Storage: every rank clones the current parameter value as the
        # authoritative copy for NS update. On non-owner ranks this is a
        # redundant replica, but it keeps the broadcast/read interface
        # uniform with the FSDP2 path (which also keeps ``_owned_data`` on
        # every shard-peer of the owner, see DedicatedParam.__init__).
        self._owned_data: torch.Tensor = local_data.detach().clone()

        # Keep a stable reference to the live module parameter. Used by
        # ``reduce_grad`` to read ``.grad`` after backward, and by
        # ``sync_from_owner`` to ``copy_`` updated data back.
        self._orig_param: nn.Parameter = param

        # Reduced gradient (on owner, after backward)
        self._reduced_grad: Optional[torch.Tensor] = None

        # Accumulated gradient for no_sync gradient accumulation (all ranks)
        self._accumulated_grad: Optional[torch.Tensor] = None

    # ---- interface parity with DedicatedParam (all no-ops on DDP path) ----

    def finish_unshard(self) -> None:  # noqa: D401 — matches DedicatedParam
        """No-op: DDP path keeps the original nn.Parameter live."""

    def reshard(self) -> None:
        """No-op: DDP path does not reshard."""

    @property
    def _is_unsharded(self) -> bool:
        # DDP path is always "unsharded" in the sense that the live
        # parameter is always accessible on every rank.
        return True

    @_is_unsharded.setter
    def _is_unsharded(self, value: bool) -> None:
        # Accept writes from shared code paths; no state to track.
        pass

    @property
    def _unsharded_param(self) -> nn.Parameter:
        # Some shared code paths read ``_unsharded_param`` — on the DDP
        # path the live parameter IS the unsharded view.
        return self._orig_param

    # ---- gradient reduction ----

    def reduce_grad(self, async_op: bool = False) -> Optional[dist.Work]:
        """Reduce this parameter's gradient to the owner rank.

        Reads ``.grad`` directly from the live ``nn.Parameter`` (autograd
        wrote it there during backward). Uses ``op=AVG`` on the 1D
        ``dp_group`` so the owner receives the averaged gradient and
        non-owners hold garbage (discarded immediately by
        :meth:`save_grad_on_owner`).
        """
        grad = self._orig_param.grad
        if grad is None:
            return None
        grad = grad.data.contiguous()
        return dist.reduce(
            grad,
            dst=self._owner_global_rank,
            op=dist.ReduceOp.AVG,
            group=self.dp_group,
            async_op=async_op,
        )

    def save_grad_on_owner(self) -> None:
        """Owner caches the reduced gradient; all ranks clear ``.grad``."""
        grad = self._orig_param.grad
        if self.is_owner and grad is not None:
            self._reduced_grad = grad.data.clone()
        self._orig_param.grad = None

    def clear_reduced_grad(self) -> None:
        """Clear the owner-side reduced gradient after the optimizer step."""
        self._reduced_grad = None

    # ---- post-step broadcast (owner → all ranks) ----

    def sync_from_owner(self) -> None:
        """Broadcast the updated ``_owned_data`` from owner to every rank and
        copy it into the live ``nn.Parameter.data``.

        Normally **not** called in isolation — :class:`DedicatedParamGroupDDP`
        batches this across all params via a single coalesced broadcast.
        Exposed for tests and for the sync fallback path.
        """
        dist.broadcast(
            self._owned_data,
            src=self._owner_global_rank,
            group=self.dp_group,
        )
        self._orig_param.data.copy_(self._owned_data)
