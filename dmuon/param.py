"""DedicatedParam: manages one parameter under dedicated ownership."""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ._internal_utils import unsafe_setattr_param

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
        owner_rank: int,
        dp_group: dist.ProcessGroup,
        device: torch.device,
        compute_dtype: torch.dtype = None,
    ):
        self.module = module
        self.param_name = param_name
        self.owner_rank = owner_rank  # rank within dp_group (local)
        self.is_owner = dp_group.rank() == owner_rank
        self.dp_group = dp_group
        self.device = device

        # Convert dp-local owner_rank to global rank for NCCL collectives
        self._owner_global_rank = dist.get_global_rank(dp_group, owner_rank)

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

        # Storage: owner keeps full data, others release
        if self.is_owner:
            self._owned_data = local_data.detach().clone()
        else:
            self._owned_data = None

        # Placeholder for sharded state (empty tensor)
        self._placeholder = nn.Parameter(
            torch.empty(0, dtype=self._orig_dtype, device=device),
            requires_grad=self._requires_grad,
        )
        # Mark placeholder so the FSDP2 patch also ignores it
        self._placeholder._dedicated_owner_rank = owner_rank

        # Unsharded parameter (populated after broadcast)
        self._unsharded_param: Optional[nn.Parameter] = None

        # Broadcast buffer (temporary, between alloc and finish)
        self._broadcast_buf: Optional[torch.Tensor] = None

        # Reduced gradient (on owner, after backward)
        self._reduced_grad: Optional[torch.Tensor] = None

        # Accumulated gradient for no_sync gradient accumulation (all ranks)
        self._accumulated_grad: Optional[torch.Tensor] = None

        # Set module to sharded state
        self._set_module_param(self._placeholder)

    def _set_module_param(self, param: nn.Parameter):
        """Set parameter on the module, handling DTensor wrapping."""
        if self.is_dtensor and param.numel() > 0:
            param = nn.Parameter(
                _from_local_no_grad(param.data, self._tp_spec),
                requires_grad=param.requires_grad,
            )
        unsafe_setattr_param(self.module, self.param_name, param)

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

    def alloc_and_broadcast(self, async_op: bool = False) -> Optional[dist.Work]:
        """Allocate buffer and broadcast from owner.

        When called from DedicatedParamGroup, async_op=False is used because the
        caller is already on the dedicated broadcast_stream. Stream-based dispatch
        replaces Work-based async.
        """
        # Use compute_dtype (bf16) for communication if available, else orig_dtype
        comm_dtype = self._compute_dtype or self._orig_dtype
        buf = torch.empty(self._orig_size, dtype=comm_dtype, device=self.device)
        if self.is_owner:
            assert self._owned_data is not None, "Owner must have _owned_data"
            # copy_ handles dtype conversion inline — no .to() intermediate tensor
            buf.copy_(self._owned_data)
        work = dist.broadcast( 
            buf, src=self._owner_global_rank, group=self.dp_group, async_op=async_op
        )
        self._broadcast_buf = buf
        return work

    def finish_unshard(self):
        """Complete unshard after broadcast finishes."""
        self._unsharded_param = nn.Parameter(self._broadcast_buf, requires_grad=self._requires_grad)
        self._broadcast_buf = None
        self._set_module_param(self._unsharded_param)
        # For DTensor params, _set_module_param creates a NEW DTensor-wrapped
        # Parameter on the module. Update _unsharded_param to point to that actual
        # module parameter so autograd gradients are visible via _unsharded_param.grad.
        if self.is_dtensor:
            self._unsharded_param = getattr(self.module, self.param_name)

    # ---- reshard ----

    def reshard(self):
        """Reshard: all ranks free unsharded buffer and restore placeholder.

        Note: no owner copy-back is needed. ``_owned_data`` is the
        authoritative fp32 copy and is written only by the optimizer step.
        ``_unsharded_param`` is the bf16 broadcast buffer that forward reads
        but never writes (backward writes only ``.grad``), so copying its
        contents back would just re-quantize fp32 owned data down to bf16
        precision and waste HBM bandwidth on every forward.
        """
        if self._unsharded_param is None:
            return
        self._unsharded_param = None
        self._set_module_param(self._placeholder)

    # ---- gradient reduction ----

    def reduce_grad(self, async_op: bool = False) -> Optional[dist.Work]:
        """Reduce gradient to owner rank."""
        if self._unsharded_param is None or self._unsharded_param.grad is None:
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
        if self.is_owner and self._unsharded_param is not None:
            grad = self._unsharded_param.grad
            if grad is not None:
                if self.is_dtensor and DTensor is not None and isinstance(grad, DTensor):
                    grad = grad._local_tensor
                elif isinstance(grad, torch.Tensor):
                    pass
                self._reduced_grad = grad.data.clone()
        if self._unsharded_param is not None:
            self._unsharded_param.grad = None

    def clear_reduced_grad(self):
        """Clear saved gradient (after optimizer step)."""
        self._reduced_grad = None
