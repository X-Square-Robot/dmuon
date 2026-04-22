"""DedicatedParam: manages one parameter under dedicated ownership."""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ._internal_utils import unsafe_setattr_param
from ._owner_rank import OwnerCoord, OwnerRankLike, normalize_owner_rank

try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._dtensor_spec import DTensorSpec
except ImportError:
    DTensor = None
    DTensorSpec = None

try:
    from torch.distributed.tensor import Shard as _Shard
except ImportError:
    _Shard = None


def _from_local_no_grad(tensor: torch.Tensor, spec: "DTensorSpec") -> "DTensor":
    """Wrap a local tensor as DTensor without gradient tracking."""
    return DTensor.from_local(tensor, spec.mesh, spec.placements, run_check=False)


def _make_contiguous_stride(size: torch.Size) -> tuple[int, ...]:
    """Compute row-major contiguous stride for ``size``."""
    stride: list[int] = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        stride[i] = stride[i + 1] * size[i + 1]
    return tuple(stride)


class DedicatedParam:
    """Manages one dedicated-ownership parameter.

    The owner rank stores the full parameter. Other ranks store nothing.
    Communication uses broadcast (forward) and reduce (backward).

    This class handles DTensor parameters (TP-sharded) by operating on
    the local tensor and re-wrapping after broadcast.
    """

    def __init__(
        self,
        param: nn.Parameter,
        module: nn.Module,
        param_name: str,
        owner_rank: OwnerRankLike,
        dp_group: dist.ProcessGroup,
        device: torch.device,
        compute_dtype: torch.dtype = None,
        replicate_group: Optional[dist.ProcessGroup] = None,
    ):
        self.module = module
        self.param_name = param_name
        # Phase A (HSDP-native refactor): ``owner_rank`` is a 2D coordinate
        # ``(owner_shard, owner_replicate)``.  Plain ``int`` is still accepted
        # for the 1D shard-only path and is promoted to ``(int, 0)``.
        coord: OwnerCoord = normalize_owner_rank(owner_rank)
        self.owner_rank: OwnerCoord = coord
        self.owner_shard: int = coord[0]
        self.owner_replicate: int = coord[1]
        self.dp_group = dp_group          # shard group (1D in Phase A)
        self.replicate_group = replicate_group  # None until Phase B wires HSDP
        self.device = device

        # ``is_owner`` decides whether this rank holds ``_owned_data`` and runs
        # the optimizer step.  Phase A keeps ``replicate_group=None`` so this
        # reduces to the previous shard-only check; Phase B will enable the
        # second-dim check once ``replicate_group`` is wired through.
        shard_hit = dp_group.rank() == self.owner_shard
        replicate_hit = (
            replicate_group is None or replicate_group.rank() == self.owner_replicate
        )
        self.is_owner = shard_hit and replicate_hit

        # Convert dp-local owner_rank to global rank for NCCL collectives.
        # The shard coordinate is always the intra-``dp_group`` rank, so using
        # ``owner_shard`` keeps behaviour identical in both 1D and 2D cases.
        self._owner_global_rank = dist.get_global_rank(dp_group, self.owner_shard)
        # Phase B: Stage-2 ``dist.reduce(dst=...)`` along the replicate axis
        # also needs a global rank.  ``replicate_group`` is a process group
        # spanning the replicate peers of the current shard column, so the
        # translation goes through that group (not ``dp_group``).
        if replicate_group is not None:
            self._owner_replicate_global_rank = dist.get_global_rank(
                replicate_group, self.owner_replicate
            )
        else:
            self._owner_replicate_global_rank = None

        # DTensor awareness (for TP compatibility)
        self.is_dtensor = DTensor is not None and isinstance(param, DTensor)
        if self.is_dtensor:
            self._tp_spec = param._spec
            local_data = param._local_tensor
        else:
            self._tp_spec = None
            local_data = param.data

        self._orig_size = local_data.size()
        self._orig_dtype = local_data.dtype
        self._compute_dtype = compute_dtype  # e.g. bf16 for mixed precision
        self._requires_grad = param.requires_grad

        # Cached attrs (were @property, now computed once).
        # Dependencies: is_dtensor, _tp_spec, _orig_size — all set above.
        self.numel: int = self._orig_size.numel()
        self.shard_dim: Optional[int] = self._compute_shard_dim()
        self.full_shape: torch.Size = self._compute_full_shape()
        self.tp_group: Optional[dist.ProcessGroup] = self._compute_tp_group()

        # Storage: every rank in the owner's shard column keeps a populated
        # ``_owned_data``.
        #
        # Why not just the global owner?  In HSDP mode each replicate row is
        # a full model instance with its own shard_group; the shard-dim
        # broadcast that reconstitutes ``_unsharded_param`` during forward
        # fires inside each row independently.  So every row's shard-owner
        # rank must be a valid *sender* of that broadcast, which means it
        # needs its own populated ``_owned_data``.  At construction time the
        # model's state_dict has just been loaded on every rank, so the
        # local Parameter already holds the correct value — we simply clone
        # it here.  The global owner (one rank in the shard column) also
        # owns the replicate-dim broadcast and the optimizer-state update;
        # its ``_owned_data`` is what the Phase B.2 post-step broadcast
        # fans out to the other rows' shard-owner ranks.
        #
        # Ranks outside the owner's shard column never need this buffer.
        is_in_owner_shard_column = dp_group.rank() == self.owner_shard
        if is_in_owner_shard_column:
            self._owned_data = local_data.detach().clone()
        else:
            self._owned_data = None

        # Placeholder for sharded state (empty tensor)
        self._placeholder = nn.Parameter(
            torch.empty(0, dtype=self._orig_dtype, device=device),
            requires_grad=self._requires_grad,
        )
        # Mark placeholder so the FSDP2 patch also ignores it.  ``patch.py``
        # only checks for the attribute's presence, so storing the 2D coord
        # here is safe and keeps bookkeeping consistent with ``self.owner_rank``.
        self._placeholder._dedicated_owner_rank = coord

        # Phase 2: _unsharded_param is created LATER by DedicatedParamGroup
        # (see ``bind_to_packed_buffer``) as an ``as_strided`` view into the
        # group's persistent per-owner packed buffer. This matches FSDP2's
        # pattern where all an owner's params share one buffer and each
        # Parameter is a stride-metadata view.
        #
        # Storage resizing happens on the group's packed buffer (one alloc /
        # free per broadcast), not on individual Parameter storages.
        self._unsharded_param: Optional[nn.Parameter] = None
        # True while group's packed buffer is allocated; False after reshard.
        self._is_unsharded: bool = False

        # Reduced gradient (on owner, after backward)
        self._reduced_grad: Optional[torch.Tensor] = None

        # Accumulated gradient for no_sync gradient accumulation (all ranks)
        self._accumulated_grad: Optional[torch.Tensor] = None

        # Set module to sharded state
        unsafe_setattr_param(self.module, self.param_name, self._placeholder)

    def bind_to_packed_buffer(
        self, packed_buf: torch.Tensor, storage_offset: int
    ) -> None:
        """Install ``_unsharded_param`` as an as_strided view into the group's
        packed broadcast buffer at ``storage_offset``.

        Called once by :class:`DedicatedParamGroup` after both are constructed.
        The Parameter shares storage with ``packed_buf``; when the group
        alloc/free's the packed buffer's storage, this view's storage also
        resizes (it's the same Storage object).
        """
        contiguous_stride = _make_contiguous_stride(self._orig_size)
        view = torch.as_strided(packed_buf, self._orig_size, contiguous_stride, storage_offset)
        if self.is_dtensor:
            wrapped = _from_local_no_grad(view, self._tp_spec)
        else:
            wrapped = view
        self._unsharded_param = nn.Parameter(wrapped, requires_grad=self._requires_grad)

    # ---- cached-attr helpers (called once from __init__) ----

    def _compute_tp_group(self) -> Optional[dist.ProcessGroup]:
        if self.is_dtensor and self._tp_spec is not None:
            return self._tp_spec.mesh.get_group(mesh_dim=0)
        return None

    def _compute_shard_dim(self) -> Optional[int]:
        """TP shard dimension: 0 (row-sharded) or 1 (col-sharded).

        Returns None for non-DTensor params. Used by Gram NS to decide
        which Gram matrix (L-side or R-side) decomposes under TP.
        """
        if self.is_dtensor and self._tp_spec is not None and _Shard is not None:
            for p in self._tp_spec.placements:
                if isinstance(p, _Shard):
                    return p.dim
        return None

    def _compute_full_shape(self) -> torch.Size:
        """Full (unsharded) shape of the parameter.

        For DTensor params, reconstructs the shape before TP sharding.
        For non-DTensor params, returns ``_orig_size`` as-is.
        """
        if self.is_dtensor and self._tp_spec is not None and _Shard is not None:
            shape = list(self._orig_size)
            for p in self._tp_spec.placements:
                if isinstance(p, _Shard):
                    tp_size = self._tp_spec.mesh.size(0)
                    shape[p.dim] *= tp_size
            return torch.Size(shape)
        return self._orig_size

    # ---- unshard (broadcast) ----
    # Phase 2: storage lives on the group's packed buffer. DedicatedParam
    # only attaches/detaches its _unsharded_param from the module.

    def finish_unshard(self):
        """Attach persistent _unsharded_param to the module.

        Group's unshard path has already alloc'd the packed buffer and
        written broadcast data. Our view (Parameter) shares that storage,
        so setattr makes the new data visible to forward.
        """
        unsafe_setattr_param(self.module, self.param_name, self._unsharded_param)
        self._is_unsharded = True

    # ---- reshard ----

    def reshard(self):
        """Detach _unsharded_param from the module, restore placeholder.

        The Parameter object stays alive; its storage will be freed when
        the group calls ``free_storage`` on the packed buffer.
        """
        if not self._is_unsharded:
            return
        unsafe_setattr_param(self.module, self.param_name, self._placeholder)
        self._is_unsharded = False

    # ---- gradient reduction ----

    def reduce_grad(self, async_op: bool = False) -> Optional[dist.Work]:
        """Reduce gradient to owner rank."""
        if not self._is_unsharded or self._unsharded_param.grad is None:
            return None
        grad = self._unsharded_param.grad.data
        if self.is_dtensor and DTensor is not None and isinstance(grad, DTensor):
            grad = grad._local_tensor
        grad = grad.contiguous()
        return dist.reduce(
            grad,
            dst=self._owner_global_rank,
            op=dist.ReduceOp.AVG,
            group=self.dp_group,
            async_op=async_op,
        )

    def save_grad_on_owner(self):
        """Owner saves the reduced gradient; all ranks clear grad."""
        if not self._is_unsharded:
            return
        if self.is_owner:
            grad = self._unsharded_param.grad
            if grad is not None:
                if self.is_dtensor and DTensor is not None and isinstance(grad, DTensor):
                    grad = grad._local_tensor
                self._reduced_grad = grad.data.clone()
        self._unsharded_param.grad = None

    def clear_reduced_grad(self):
        """Clear saved gradient (after optimizer step)."""
        self._reduced_grad = None
