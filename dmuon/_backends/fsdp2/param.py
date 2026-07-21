"""DedicatedParam: manages one parameter under dedicated ownership."""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from dmuon._core.internal_utils import unsafe_setattr_param
from dmuon._core.owner_rank import OwnerCoord, OwnerRankLike, normalize_owner_rank

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


def _normalize_dmuon_route_hint(route: Optional[str]) -> str:
    if route is None:
        return "muon"
    route = str(route).strip().lower()
    aliases = {
        "matrix": "muon",
        "matrix_optimizer": "muon",
        "base": "adamw",
        "base_adamw": "adamw",
        "dedicated_adamw": "adamw",
        "sharded": "sharded_adamw",
        "base_sharded": "sharded_adamw",
        "sharded_collective": "sharded_adamw",
        "base_sharded_adamw": "sharded_adamw",
    }
    route = aliases.get(route, route)
    if route not in {"muon", "adamw", "sharded_adamw"}:
        raise ValueError(
            "DMuon route hint must be one of 'muon', 'adamw', or "
            f"'sharded_adamw', got {route!r}"
        )
    return route


def _normalize_muon_forward_unshard(mode: Optional[str]) -> str:
    if mode is None:
        return "broadcast"
    mode = str(mode).strip().lower()
    aliases = {
        "owner_broadcast": "broadcast",
        "bcast": "broadcast",
        "sharded": "all_gather",
        "sharded_publish": "all_gather",
        "scatter_all_gather": "all_gather",
        "ag": "all_gather",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"broadcast", "all_gather"}:
        raise ValueError(
            "DMuon Muon forward unshard mode must be 'broadcast' or "
            f"'all_gather', got {mode!r}"
        )
    return mode


def _normalize_tp_local_grad(param: "DedicatedParam", grad: torch.Tensor) -> torch.Tensor:
    """Return the TP-local gradient shard expected by dedicated reduce.

    Normal DTensor autograd for a sharded parameter gives a local tensor whose
    shape matches ``param._orig_size``.  Some custom/redistribute paths can
    instead materialize a replicated full logical gradient.  Dedicated reduce
    and the subsequent TP gather both operate on local shards, so slice that
    full gradient back to this TP rank's shard before enqueueing collectives.
    """
    orig_size = torch.Size(param._orig_size)
    grad_size = torch.Size(grad.shape)
    if grad_size == orig_size:
        return grad

    if (
        param.tp_group is not None
        and param.shard_dim is not None
        and grad_size == torch.Size(param.full_shape)
    ):
        shard_dim = param.shard_dim
        local_extent = orig_size[shard_dim]
        start = param.tp_group.rank() * local_extent
        return grad.narrow(shard_dim, start, local_extent)

    if grad.numel() == orig_size.numel():
        return grad.reshape(orig_size)

    raise RuntimeError(
        f"{param.param_name}: gradient shape {tuple(grad_size)} cannot be "
        f"normalized to TP-local shape {tuple(orig_size)}"
    )


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
        param_dtype: torch.dtype = None,
        grad_dtype: torch.dtype = None,
        output_dtype: torch.dtype = None,
        cast_forward_inputs: bool = False,
        master_dtype: torch.dtype = torch.float32,
        optim_dtype: torch.dtype = torch.float32,
        replicate_group: Optional[dist.ProcessGroup] = None,
        tp_owner_local_rank: int = 0,
        tp_group: Optional[dist.ProcessGroup] = None,
        route_hint: Optional[str] = None,
        muon_forward_unshard: str = "broadcast",
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
        if param_dtype is None:
            param_dtype = compute_dtype
        self._param_dtype = param_dtype  # materialized forward/backward param dtype
        self._grad_dtype = grad_dtype  # optional gradient reduction/buffer dtype
        self._output_dtype = output_dtype
        self._cast_forward_inputs = bool(cast_forward_inputs)
        self._master_dtype = master_dtype
        self._optim_dtype = optim_dtype
        self._compute_dtype = self._param_dtype  # legacy alias
        self._requires_grad = param.requires_grad
        self._dmuon_route: str = _normalize_dmuon_route_hint(route_hint)
        self._muon_forward_unshard: str = _normalize_muon_forward_unshard(
            muon_forward_unshard
        )
        self._tp_group_override = tp_group

        # Cached attrs (were @property, now computed once).
        # Dependencies: is_dtensor, _tp_spec, _orig_size — all set above.
        self.numel: int = self._orig_size.numel()
        self.shard_dim: Optional[int] = self._compute_shard_dim()
        self.full_shape: torch.Size = self._compute_full_shape()
        self.tp_group: Optional[dist.ProcessGroup] = self._compute_tp_group()
        # T2: TP owner selection.  The partitioner assigns a per-parameter
        # TP owner via LPT and API plumbing passes it here.  ``is_tp_owner``
        # is True on exactly one rank per (DP-owner × TP group) combo — that
        # rank runs NS on the full matrix gathered by ``tp_gather_grads``.
        # ``_tp_owner_global_rank`` is the global rank to pass to
        # ``dist.gather(dst=...)`` / ``dist.scatter(src=...)``.
        if self.tp_group is not None:
            if not 0 <= tp_owner_local_rank < self.tp_group.size():
                raise ValueError(
                    f"{param_name}: tp_owner_local_rank={tp_owner_local_rank} "
                    f"outside TP group size {self.tp_group.size()}"
                )
            self._tp_owner_local_rank: int = tp_owner_local_rank
            self._tp_owner_global_rank: Optional[int] = dist.get_global_rank(
                self.tp_group, self._tp_owner_local_rank
            )
            self.is_tp_owner: bool = (
                self.tp_group.rank() == self._tp_owner_local_rank
            )
        else:
            self._tp_owner_local_rank = 0
            self._tp_owner_global_rank = None
            self.is_tp_owner = False

        # Storage: every rank in the owner's shard column keeps a populated
        # ``_owned_data`` for owner-broadcast placement.
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
        #
        # Sharded AdamW is different: every shard rank owns its flat shard in
        # ``_sharded_adamw_data`` and reconstructs the forward view via
        # all-gather.  It must not also keep a full fp32 owner copy here.
        is_in_owner_shard_column = dp_group.rank() == self.owner_shard
        if is_in_owner_shard_column and self._dmuon_route != "sharded_adamw":
            master_dtype = self._master_dtype or self._orig_dtype
            self._owned_data = local_data.detach().to(
                device=device,
                dtype=master_dtype,
            ).clone()
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

        # DMuon-managed sharded AdamW route.  This is the base-param path used
        # for large non-Muon tensors such as embeddings and lm_head: every
        # shard-rank owns one flat shard, DMuon reduce-scatters gradients into
        # that shard, AdamW updates it locally, and the next forward all-gathers
        # the full view.  FSDP2 is not involved in this path.
        self._sharded_adamw_numel: int = int(local_data.numel())
        self._sharded_adamw_chunk_numel: int = 0
        self._sharded_adamw_valid_numel: int = 0
        self._sharded_adamw_data: Optional[torch.Tensor] = None
        self._sharded_adamw_grad: Optional[torch.Tensor] = None
        self._sharded_adamw_full_padded: Optional[torch.Tensor] = None
        self._sharded_adamw_comm_shard: Optional[torch.Tensor] = None
        self._sharded_adamw_reduce_input: Optional[torch.Tensor] = None
        if self._dmuon_route == "sharded_adamw":
            self._init_sharded_adamw_storage(local_data)

        # Optional Muon forward placement: the optimizer still updates the
        # owner full matrix, but post-step publish scatters updated shards to
        # every rank and the next forward reconstructs with all-gather.  This
        # mirrors FSDP's forward communication shape and avoids owner-to-all
        # broadcast on the forward critical path.
        self._sharded_muon_numel: int = int(local_data.numel())
        self._sharded_muon_chunk_numel: int = 0
        self._sharded_muon_valid_numel: int = 0
        self._sharded_muon_data: Optional[torch.Tensor] = None
        self._sharded_muon_full_padded: Optional[torch.Tensor] = None
        self._sharded_muon_comm_shard: Optional[torch.Tensor] = None
        self._sharded_muon_scatter_input: Optional[torch.Tensor] = None
        self._sharded_muon_initialized: bool = False
        if self.uses_sharded_muon_forward():
            self._init_sharded_muon_storage(local_data)

        # Reduced gradient (on owner, after backward)
        self._reduced_grad: Optional[torch.Tensor] = None

        # Optimizer route metadata.  ``route_hint`` gives the communication
        # layer enough information at construction time to allocate storage
        # for sharded base AdamW. ``Muon`` validates and may reassign semantic
        # group indices later, but it must not move a param out of
        # ``sharded_adamw`` after construction without rebuilding storage.
        self._dmuon_adamw_replicate_allreduce: bool = False

        # Accumulated gradient for no_sync gradient accumulation (all ranks)
        self._accumulated_grad: Optional[torch.Tensor] = None

        # T2: TP full-matrix buffers.  These only carry data on the TP owner
        # rank for TP-sharded params; every other rank keeps them at None.
        #   * ``_tp_full_grad``  — populated by ``tp_gather_grads`` with the
        #     reassembled (M, N) gradient; consumed by the optimizer step.
        #   * ``_tp_full_delta`` — populated by the optimizer with the
        #     **pre-scaled** update ``-lr*scale*NS_output``; consumed by
        #     ``tp_scatter_delta`` which chops it back into TP-local
        #     shards and delivers to every DP-owner TP rank.
        #   * ``_tp_wd_factor``  — ``(1 - lr*wd)``; broadcast via a simple
        #     Python attribute write before scatter so every receiving
        #     rank can do ``_owned_data.mul_(wd_factor).add_(shard)`` to
        #     finish Moonlight's weight-decay + update fuse in place.
        # For non-TP params (``tp_group is None``) these stay None / 1.0.
        self._tp_full_grad: Optional[torch.Tensor] = None
        self._tp_full_delta: Optional[torch.Tensor] = None
        self._tp_wd_factor: float = 1.0

        # Set module to sharded state
        unsafe_setattr_param(self.module, self.param_name, self._placeholder)

    def uses_sharded_adamw(self) -> bool:
        return self._dmuon_route == "sharded_adamw"

    def uses_sharded_muon_forward(self) -> bool:
        return (
            self._dmuon_route == "muon"
            and self._muon_forward_unshard == "all_gather"
        )

    def _init_sharded_storage_common(
        self,
        local_data: torch.Tensor,
        *,
        allow_meta_source: bool = False,
        scratch_dtype: torch.dtype = None,
    ) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.tp_group is not None:
            raise NotImplementedError(
                f"{self.param_name}: sharded forward placement does not yet "
                "support TP-sharded DTensor parameters"
            )
        shard_world = int(self.dp_group.size())
        shard_rank = int(self.dp_group.rank())
        numel = int(local_data.numel())
        chunk = (numel + shard_world - 1) // shard_world
        start = shard_rank * chunk
        stop = min(start + chunk, numel)
        valid = max(0, stop - start)

        master_dtype = self._master_dtype or self._orig_dtype
        data_shard = torch.zeros(chunk, dtype=master_dtype, device=self.device)
        if valid and not bool(getattr(local_data, "is_meta", False)):
            data_shard[:valid].copy_(
                local_data.detach().reshape(-1)[start:stop].to(dtype=master_dtype)
            )
        elif valid and not allow_meta_source:
            raise NotImplementedError(
                f"{self.param_name}: cannot initialize sharded storage from a "
                "meta tensor"
            )

        full_padded_numel = chunk * shard_world
        comm_dtype = self._param_dtype or self._orig_dtype
        scratch_dtype = scratch_dtype or comm_dtype
        full_padded = torch.empty(
            full_padded_numel, dtype=comm_dtype, device=self.device
        )
        comm_shard = torch.empty(chunk, dtype=comm_dtype, device=self.device)
        full_comm = torch.empty(
            full_padded_numel,
            dtype=scratch_dtype,
            device=self.device,
        )
        return chunk, valid, data_shard, full_padded, comm_shard, full_comm

    def _init_sharded_adamw_storage(self, local_data: torch.Tensor) -> None:
        """Initialize DMuon-owned shard storage for base AdamW tensors."""
        numel = int(local_data.numel())
        (
            chunk,
            valid,
            shard,
            full_padded,
            comm_shard,
            reduce_input,
        ) = self._init_sharded_storage_common(
            local_data,
            scratch_dtype=self._grad_dtype or self._param_dtype or self._orig_dtype,
        )
        self._sharded_adamw_chunk_numel = int(chunk)
        self._sharded_adamw_valid_numel = int(valid)
        self._sharded_adamw_data = shard

        self._sharded_adamw_full_padded = full_padded
        self._sharded_adamw_comm_shard = comm_shard
        self._sharded_adamw_reduce_input = reduce_input

        contiguous_stride = _make_contiguous_stride(self._orig_size)
        view = torch.as_strided(
            self._sharded_adamw_full_padded[:numel],
            self._orig_size,
            contiguous_stride,
            0,
        )
        if self.is_dtensor:
            wrapped = _from_local_no_grad(view, self._tp_spec)
        else:
            wrapped = view
        self._unsharded_param = nn.Parameter(
            wrapped, requires_grad=self._requires_grad
        )
        from dmuon._core.internal_utils import free_storage

        free_storage(self._sharded_adamw_full_padded)

    def _init_sharded_muon_storage(self, local_data: torch.Tensor) -> None:
        """Initialize shard storage for Muon forward all-gather placement."""
        numel = int(local_data.numel())
        (
            chunk,
            valid,
            shard,
            full_padded,
            comm_shard,
            scatter_input,
        ) = self._init_sharded_storage_common(
            local_data,
            allow_meta_source=True,
        )
        self._sharded_muon_chunk_numel = int(chunk)
        self._sharded_muon_valid_numel = int(valid)
        self._sharded_muon_data = shard
        self._sharded_muon_full_padded = full_padded
        self._sharded_muon_comm_shard = comm_shard
        self._sharded_muon_scatter_input = scatter_input

        contiguous_stride = _make_contiguous_stride(self._orig_size)
        view = torch.as_strided(
            self._sharded_muon_full_padded[:numel],
            self._orig_size,
            contiguous_stride,
            0,
        )
        if self.is_dtensor:
            wrapped = _from_local_no_grad(view, self._tp_spec)
        else:
            wrapped = view
        self._unsharded_param = nn.Parameter(
            wrapped, requires_grad=self._requires_grad
        )
        from dmuon._core.internal_utils import free_storage

        free_storage(self._sharded_muon_full_padded)

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
        if self.uses_sharded_adamw() or self.uses_sharded_muon_forward():
            # Sharded base AdamW owns its full forward view via
            # ``_sharded_adamw_full_padded``; optional Muon sharded forward
            # placement owns the same kind of view via
            # ``_sharded_muon_full_padded``.  Both are populated by
            # all-gather, not owner broadcast, so keep their init-time views.
            return
        contiguous_stride = _make_contiguous_stride(self._orig_size)
        view = torch.as_strided(packed_buf, self._orig_size, contiguous_stride, storage_offset)
        if self.is_dtensor:
            wrapped = _from_local_no_grad(view, self._tp_spec)
        else:
            wrapped = view
        self._unsharded_param = nn.Parameter(wrapped, requires_grad=self._requires_grad)

    # ---- cached-attr helpers (called once from __init__) ----

    def _compute_tp_group(self) -> Optional[dist.ProcessGroup]:
        if self._tp_group_override is not None:
            return self._tp_group_override
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
        for alias_module, alias_name in getattr(self, "_alias_modules", ()):
            unsafe_setattr_param(alias_module, alias_name, self._unsharded_param)
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
        for alias_module, alias_name in getattr(self, "_alias_modules", ()):
            unsafe_setattr_param(alias_module, alias_name, self._placeholder)
        self._is_unsharded = False

    # ---- gradient reduction ----

    def reduce_grad(self, async_op: bool = False) -> Optional[dist.Work]:
        """Reduce gradient to owner rank."""
        if not self._is_unsharded or self._unsharded_param.grad is None:
            return None
        grad = self._unsharded_param.grad.data
        grad = self.local_grad_for_reduce(grad)
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
                grad = self.local_grad_for_reduce(grad)
                self._reduced_grad = grad.data.clone()
        self._unsharded_param.grad = None

    def clear_reduced_grad(self):
        """Clear saved gradient (after optimizer step)."""
        self._reduced_grad = None

    def local_grad_for_reduce(self, grad: torch.Tensor) -> torch.Tensor:
        """Convert a parameter grad to the TP-local tensor reduce expects."""
        if self.is_dtensor and DTensor is not None and isinstance(grad, DTensor):
            if (
                self._tp_spec is not None
                and tuple(grad._spec.placements) != tuple(self._tp_spec.placements)
            ):
                grad = grad.redistribute(placements=self._tp_spec.placements)
            grad = grad._local_tensor
        return _normalize_tp_local_grad(self, grad)
