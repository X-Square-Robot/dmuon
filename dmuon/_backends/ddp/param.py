"""DedicatedParamDDP: DDP-path variant of DedicatedParam.

Differs from :class:`dmuon.param.DedicatedParam` in three orthogonal axes:

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
from dmuon.policy import normalize_route

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


def _normalize_tp_local_grad(param: "DedicatedParamDDP", grad: torch.Tensor) -> torch.Tensor:
    """Return the TP-local gradient shard expected by DDP reduce."""
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
        param_dtype: Optional[torch.dtype] = None,
        grad_dtype: Optional[torch.dtype] = None,
        output_dtype: Optional[torch.dtype] = None,
        cast_forward_inputs: bool = True,
        master_dtype: Optional[torch.dtype] = torch.float32,
        optim_dtype: Optional[torch.dtype] = torch.float32,
        route_hint: Optional[str] = None,
        tp_owner_local_rank: int = 0,
        tp_group: Optional[dist.ProcessGroup] = None,
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

        self.is_dtensor: bool = DTensor is not None and isinstance(param, DTensor)
        if self.is_dtensor:
            self._tp_spec = param._spec
            local_data = param._local_tensor
        else:
            self._tp_spec = None
            local_data = param.data

        self._orig_size: torch.Size = local_data.size()
        self._orig_dtype: torch.dtype = local_data.dtype
        if param_dtype is None:
            param_dtype = compute_dtype
        self._param_dtype: Optional[torch.dtype] = param_dtype
        self._grad_dtype: Optional[torch.dtype] = grad_dtype
        self._output_dtype: Optional[torch.dtype] = output_dtype
        self._cast_forward_inputs: bool = bool(cast_forward_inputs)
        self._master_dtype: Optional[torch.dtype] = master_dtype
        self._optim_dtype: Optional[torch.dtype] = optim_dtype
        self._compute_dtype: Optional[torch.dtype] = self._param_dtype  # legacy alias
        self._dmuon_route: str = normalize_route(route_hint) or "muon"
        self._requires_grad: bool = param.requires_grad
        self._tp_group_override = tp_group

        self.numel: int = self._orig_size.numel()
        self.shard_dim: Optional[int] = self._compute_shard_dim()
        self.full_shape: torch.Size = self._compute_full_shape()
        self.tp_group: Optional[dist.ProcessGroup] = self._compute_tp_group()
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

        # Storage: every rank clones the current parameter value as the
        # authoritative copy for NS update. On non-owner ranks this is a
        # redundant replica, but it keeps the broadcast/read interface
        # uniform with the FSDP2 path (which also keeps ``_owned_data`` on
        # every shard-peer of the owner, see DedicatedParam.__init__).
        master_dtype = self._master_dtype or self._orig_dtype
        self._owned_data: torch.Tensor = local_data.detach().to(
            device=device,
            dtype=master_dtype,
        ).clone()

        # Keep a stable reference to the live module parameter. Used by
        # ``reduce_grad`` to read ``.grad`` after backward, and by
        # ``sync_from_owner`` to ``copy_`` updated data back.
        self._orig_param: nn.Parameter = param
        live_dtype = self._param_dtype or self._orig_dtype
        if live_dtype != local_data.dtype:
            self._cast_live_param_(live_dtype)

        # Reduced gradient (on owner, after backward)
        self._reduced_grad: Optional[torch.Tensor] = None

        # Accumulated gradient for no_sync gradient accumulation (all ranks)
        self._accumulated_grad: Optional[torch.Tensor] = None

        # TP full-matrix buffers.  Same duck-typed surface as the FSDP2
        # DedicatedParam so Muon can reuse the TP gather / NS / scatter path.
        self._tp_full_grad: Optional[torch.Tensor] = None
        self._tp_full_delta: Optional[torch.Tensor] = None
        self._tp_wd_factor: float = 1.0

    # ---- cached-attr helpers ----------------------------------------------

    def _compute_tp_group(self) -> Optional[dist.ProcessGroup]:
        if self._tp_group_override is not None:
            return self._tp_group_override
        if self.is_dtensor and self._tp_spec is not None:
            return self._tp_spec.mesh.get_group(mesh_dim=0)
        return None

    def _compute_shard_dim(self) -> Optional[int]:
        if self.is_dtensor and self._tp_spec is not None and _Shard is not None:
            for p in self._tp_spec.placements:
                if isinstance(p, _Shard):
                    return p.dim
        return None

    def _compute_full_shape(self) -> torch.Size:
        if self.is_dtensor and self._tp_spec is not None and _Shard is not None:
            shape = list(self._orig_size)
            for p in self._tp_spec.placements:
                if isinstance(p, _Shard):
                    tp_size = self._tp_spec.mesh.size(0)
                    shape[p.dim] *= tp_size
            return torch.Size(shape)
        return self._orig_size

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

    def local_grad_for_reduce(self, grad: torch.Tensor) -> torch.Tensor:
        """Convert a live parameter grad to the local tensor reduce expects."""
        if self.is_dtensor and DTensor is not None and isinstance(grad, DTensor):
            if (
                self._tp_spec is not None
                and tuple(grad._spec.placements) != tuple(self._tp_spec.placements)
            ):
                grad = grad.redistribute(placements=self._tp_spec.placements)
            grad = grad._local_tensor
        return _normalize_tp_local_grad(self, grad)

    def clear_live_grad(self) -> None:
        self._orig_param.grad = None

    def _live_tensor(self) -> torch.Tensor:
        return self._orig_param._local_tensor if self.is_dtensor else self._orig_param.data

    def _cast_live_param_(self, dtype: torch.dtype) -> None:
        """Materialize the DDP live parameter in the requested compute dtype."""
        if self.is_dtensor:
            self._orig_param._local_tensor.data = self._orig_param._local_tensor.data.to(
                device=self.device,
                dtype=dtype,
            )
        else:
            self._orig_param.data = self._orig_param.data.to(
                device=self.device,
                dtype=dtype,
            )

    def copy_owned_to_live(self) -> None:
        """Copy ``_owned_data`` into the live Tensor or DTensor local shard."""
        live = self._live_tensor()
        live.copy_(self._owned_data.to(device=live.device, dtype=live.dtype))

    def set_live_grad_from_local(self, grad: torch.Tensor) -> None:
        """Install a local accumulated grad back onto the live parameter."""
        if self.is_dtensor:
            self._orig_param.grad = _from_local_no_grad(grad, self._tp_spec)
        else:
            self._orig_param.grad = grad

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
        grad = self.local_grad_for_reduce(grad)
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
            grad = self.local_grad_for_reduce(grad)
            self._reduced_grad = grad.data.clone()
        self.clear_live_grad()

    def clear_reduced_grad(self) -> None:
        """Clear the owner-side reduced gradient after the optimizer step."""
        self._reduced_grad = None

    # ---- post-step broadcast (owner → all ranks) ----

    def sync_from_owner(self) -> None:
        """Broadcast the updated ``_owned_data`` from owner to every rank and
        copy it into the live ``nn.Parameter.data``.

        Normally **not** called in isolation — :class:`DedicatedParamGroupDDP`
        batches this across all params via a single coalesced broadcast.
        Exposed for tests and synchronous post-step publish.
        """
        dist.broadcast(
            self._owned_data,
            src=self._owner_global_rank,
            group=self.dp_group,
        )
        self.copy_owned_to_live()
