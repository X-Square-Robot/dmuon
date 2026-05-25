"""DDP-style replication for non-dedicated parameters.

Companion to :func:`dmuon.dedicate_params_ddp`. ``dedicate_params_ddp``
handles parameters selected by the predicate (``*.proj.weight``, ...);
everything else (``LayerNorm``, ``head``, embeddings, ...) is handled
here by attaching a post-accumulate-grad hook that averages gradients
across the data-parallel world at end-of-backward.

The design mirrors FSDP2's post-backward pattern and DMuon's own
``DedicatedState._queue_root_post_backward_callback`` (``state.py``):
per-param hooks enqueue the param, and the autograd root callback
flushes the whole batch in one ``dist._coalescing_manager`` so NCCL
fuses the per-bucket all-reduces into a single kernel.

``replicate`` is intentionally **separate** from FSDP2's
``fully_shard``. The two are per-parameter mutually exclusive; calling
``replicate`` on a model where FSDP2 already manages the same param
raises. This is the cleanest boundary — users pick one path per
non-dedicated param.

2D mesh (HSDP-minimal — every rank keeps a full replica of non-dedicated
params and averages across the 2D world) is explicitly NOT supported in
P1. Use ``fully_shard(mesh=hsdp)`` for HSDP; use ``replicate(mesh=1D)``
for pure DDP.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Variable
try:
    from torch.distributed import DeviceMesh
except ImportError:  # Older PyTorch exposes DeviceMesh only from this module.
    from torch.distributed.device_mesh import DeviceMesh


def _is_tp_only_dtensor(p: nn.Parameter, dp_mesh_dim_names: frozenset[str]) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    if not isinstance(p, DTensor):
        return False
    names = p.device_mesh.mesh_dim_names
    if names is None:
        raise ValueError(
            "replicate_tp() requires named DTensor meshes so TP-only "
            "parameters can be distinguished from FSDP-managed DTensors."
        )
    if any(name in dp_mesh_dim_names for name in names):
        return False
    if "tp" not in names:
        return False
    return p.device_mesh["tp"].size() > 1


def _is_fsdp_managed(
    p: nn.Parameter,
    *,
    dp_mesh_dim_names: frozenset[str],
    allow_tp_dtensor: bool,
) -> bool:
    """Best-effort detection of a parameter already under FSDP2's reducer.

    FSDP2 replaces managed parameters with ``DTensor`` instances whose
    ``_spec`` references the DP mesh; treating any DTensor as
    "already managed" is conservative but correct for the P1 scope
    (we never need to re-replicate a DTensor).
    """
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    if not isinstance(p, DTensor):
        return False
    if allow_tp_dtensor and _is_tp_only_dtensor(p, dp_mesh_dim_names):
        return False
    return True


class _ReplicatedGroup:
    """Per-model bundle of replicated parameters and their comm state.

    Attached on the model as ``model._replicated_group`` so the Muon
    optimizer can discover the parameters to update with AdamW.
    """

    def __init__(
        self,
        model: nn.Module,
        mesh: DeviceMesh,
        *,
        allow_tp_dtensor: bool = False,
    ):
        if mesh.ndim != 1:
            raise ValueError(
                f"replicate() requires a 1D mesh; got ndim={mesh.ndim}. "
                "For HSDP, use fully_shard(mesh=hsdp) instead."
            )

        self.mesh = mesh
        self._group: dist.ProcessGroup = mesh.get_group()
        self._device = torch.device("cuda", torch.cuda.current_device())
        dp_names = set(mesh.mesh_dim_names or ())
        self._dp_mesh_dim_names = frozenset(dp_names)
        self._allow_tp_dtensor = allow_tp_dtensor

        self.params: list[nn.Parameter] = []
        for p in model.parameters():
            if hasattr(p, "_dedicated_owner_rank"):
                continue  # dedicated (FSDP2 or DDP path) or its placeholder
            if _is_fsdp_managed(
                p,
                dp_mesh_dim_names=self._dp_mesh_dim_names,
                allow_tp_dtensor=allow_tp_dtensor,
            ):
                raise RuntimeError(
                    "replicate(): parameter is already managed by FSDP2 "
                    "(DTensor detected). replicate and fully_shard are "
                    "mutually exclusive for the same parameter. Use "
                    "replicate_tp() only for TP-only DTensor parameters."
                )
            if p.requires_grad:
                self.params.append(p)

        self._sync_enabled: bool = True

        # Per-param hook handles (removed in destroy()).
        self._hook_handles: list = []
        for p in self.params:
            h = p.register_post_accumulate_grad_hook(self._post_accum_hook)
            self._hook_handles.append(h)

        # End-of-backward flush state.
        self._pending: list[nn.Parameter] = []
        self._final_cb_queued: bool = False

    # ---- hooks --------------------------------------------------------------

    def _post_accum_hook(self, p: nn.Parameter) -> None:
        """Queue ``p`` for end-of-backward all-reduce.

        Fires immediately after autograd accumulates the grad on ``p``.
        The actual collective is deferred to ``_flush``, which runs once
        per backward via an autograd-engine root callback so every queued
        param is visible when the batch is coalesced.
        """
        if not self._sync_enabled:
            return
        self._pending.append(p)
        if not self._final_cb_queued:
            Variable._execution_engine.queue_callback(self._flush)
            self._final_cb_queued = True

    def _flush(self) -> None:
        """Coalesced all-reduce of every queued param.

        Buckets by ``(dtype, device)`` and issues all reduces inside one
        ``dist._coalescing_manager`` per bucket so NCCL can fuse them.
        Uses ``op=AVG`` — matches FSDP2's default reduce-scatter
        semantics for gradient averaging.
        """
        try:
            buckets: dict[tuple[torch.dtype, torch.device], list[nn.Parameter]] = (
                defaultdict(list)
            )
            for p in self._pending:
                if p.grad is None:
                    continue
                grad = p.grad._local_tensor if hasattr(p.grad, "_local_tensor") else p.grad
                buckets[(grad.dtype, grad.device)].append(p)

            for (dtype, device), plist in buckets.items():
                with dist._coalescing_manager(group=self._group, device=device):
                    for p in plist:
                        grad = (
                            p.grad._local_tensor
                            if hasattr(p.grad, "_local_tensor")
                            else p.grad.data
                        )
                        dist.all_reduce(
                            grad, op=dist.ReduceOp.AVG, group=self._group
                        )
        finally:
            self._pending.clear()
            self._final_cb_queued = False

    # ---- no_sync ------------------------------------------------------------

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """Defer gradient averaging across backward calls.

        Inside the context, per-param hooks short-circuit so grads
        accumulate locally on each rank. Leaving the context restores
        normal averaging — the NEXT backward call's grads are averaged
        as usual; the accumulator remains in ``.grad`` and is part of
        that average implicitly (autograd sums into ``.grad`` across
        backward calls when grads are not zeroed).
        """
        prev = self._sync_enabled
        self._sync_enabled = False
        try:
            yield
        finally:
            self._sync_enabled = prev

    # ---- teardown -----------------------------------------------------------

    def destroy(self) -> None:
        """Remove all hooks. Safe to call multiple times."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []


def replicate(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    """Install DDP-style replication for non-dedicated parameters.

    Must be called **after** :func:`dedicate_params_ddp` (or
    :func:`dedicate_params`; though in the FSDP2 path users normally
    call ``fully_shard`` instead). Dedicated parameters are skipped
    automatically.

    Example::

        mesh = init_device_mesh("cuda", (world_size,))
        dmuon.dedicate_params_ddp(model, mesh, predicate=...)
        dmuon.replicate(model, mesh=mesh)
        optimizer = dmuon.Muon(model, lr=..., adamw_lr=...)

    The returned object is the input ``model`` itself (identity) — no
    wrapping, no ``.module`` unwrap needed downstream. This mirrors
    :func:`torch.distributed.fsdp.fully_shard` UX.

    Args:
        model: The model to attach replicated-group state to.
        mesh: 1D DeviceMesh for the data-parallel world.

    Raises:
        ValueError: If ``mesh`` is not 1D.
        RuntimeError: If ``replicate`` was already called on ``model``,
            or if a managed parameter is already under FSDP2.

    Returns:
        ``model`` (for call chaining).
    """
    if hasattr(model, "_replicated_group"):
        raise RuntimeError("replicate() has already been called on this model")
    group = _ReplicatedGroup(model, mesh, allow_tp_dtensor=False)
    model._replicated_group = group

    # Expose no_sync at the model level to match FSDP2 UX.
    if not hasattr(model, "no_sync"):
        model.no_sync = group.no_sync
    return model


def replicate_tp(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    """Install DDP-style replication for TP-only DTensor parameters.

    This is the TP-aware companion to :func:`dmuon.dedicate_params_ddp_tp`.
    It keeps the conservative FSDP-DTensor rejection in :func:`replicate`
    intact while allowing tensor-parallel DTensors to all-reduce their local
    gradient shards over the data-parallel mesh.
    """
    if hasattr(model, "_replicated_group"):
        raise RuntimeError("replicate_tp() has already been called on this model")
    group = _ReplicatedGroup(model, mesh, allow_tp_dtensor=True)
    model._replicated_group = group
    if not hasattr(model, "no_sync"):
        model.no_sync = group.no_sync
    return model


def get_replicated_group(model: nn.Module) -> Optional[_ReplicatedGroup]:
    """Return the ``_ReplicatedGroup`` attached to ``model``, or ``None``."""
    return getattr(model, "_replicated_group", None)
