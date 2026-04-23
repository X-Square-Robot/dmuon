"""Muon optimizer with dedicated ownership for distributed training.

Combines Newton-Schulz orthogonalization on dedicated parameters with
AdamW on symmetric (FSDP2-managed) parameters in a single optimizer.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer

from typing import Union

from .. import _balance_profile
from ..utils import (
    broadcast_all_updates,
    broadcast_all_updates_async,
    get_owned_params,
    update_replicate_fallback,
    wait_all_reduces,
)
from .newton_schulz import NewtonSchulz


class Muon(Optimizer):
    """Muon optimizer for DMuon distributed training.

    Manages two types of parameters:
    - **Dedicated params** (proj layers): Newton-Schulz orthogonalization with
      momentum. Only the owner rank computes the update.
    - **Symmetric params** (layernorm, embedding): AdamW, updated by all ranks
      on their FSDP2 shards.

    Args:
        model: Model with ``dedicate_params`` and ``fully_shard`` already applied.
        lr: Muon learning rate for dedicated params.
        momentum: Momentum coefficient for dedicated params.
        weight_decay: Weight decay for dedicated params.
        ns_steps: Number of Newton-Schulz iterations.
        adamw_lr: AdamW learning rate for symmetric params.
        adamw_betas: AdamW beta coefficients.
        adamw_weight_decay: AdamW weight decay.
        adamw_eps: AdamW epsilon.
        ns_backend: Newton-Schulz backend configuration. Accepts a string
            shorthand (``"gram"`` or ``"direct"``) or a fully configured
            :class:`~dmuon.NewtonSchulz` object for custom coefficients::

                # String shorthand (default coefficients)
                optimizer = dmuon.Muon(model, ns_backend="gram")

                # Custom coefficients
                ns = dmuon.NewtonSchulz("direct", coefficients=dmuon.YOU_COEFFICIENTS)
                optimizer = dmuon.Muon(model, ns_backend=ns)

            ``"gram"`` uses Gram-space NS with SYRK acceleration and restarts.
            ``"direct"`` uses classic parameter-space NS (Muon/Moonlight).
            NS always runs on the full (un-sharded) matrix: for TP-sharded
            parameters the runtime reassembles the matrix via All-to-All
            before calling NS (see ``tp_design.md``).
        nesterov: If True (default), use Nesterov momentum lookahead
            before NS orthogonalization: ``ns_input = grad + μ * buf``.
            Recommended by original Muon paper and used by Moonlight.

    Example::

        import dmuon
        from torch.distributed.fsdp import fully_shard

        dmuon.dedicate_params(model, mesh, predicate=lambda n, p: "proj" in n)
        for layer in model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)

        optimizer = dmuon.Muon(model, lr=0.02, momentum=0.95)

        for batch in dataloader:
            optimizer.zero_grad()
            loss = model(batch).loss
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        adamw_lr: float = 1e-3,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_weight_decay: float = 0.01,
        adamw_eps: float = 1e-8,
        ns_backend: Union[str, NewtonSchulz] = "gram",
        nesterov: bool = True,
        replicate_async: bool = True,
    ):
        if isinstance(ns_backend, str):
            ns_backend = NewtonSchulz(backend=ns_backend)
        if not isinstance(ns_backend, NewtonSchulz):
            raise TypeError(
                f"ns_backend must be 'gram', 'direct', or a NewtonSchulz instance, "
                f"got {type(ns_backend).__name__}"
            )
        self.model = model
        self._ns_steps = ns_steps
        self._ns = ns_backend
        self._nesterov = nesterov
        # Phase C: toggle between async (default, hides broadcast inside
        # the next forward) and Phase B sync (simpler, always-correct).
        # When True, each group's pending event is consumed by its own
        # ``_pre_forward_wait`` hook; when False, the full fan-out is
        # waited synchronously at the end of step().
        self._replicate_async = replicate_async

        # Discover dedicated params owned by this rank
        comm_ctx = getattr(model, "_dedicated_comm_ctx", None)
        if comm_ctx is None:
            raise ValueError(
                "Model has no _dedicated_comm_ctx. Call dmuon.dedicate_params() first."
            )
        self._comm_ctx = comm_ctx
        self._dedicated_params = []
        for module in model.modules():
            if hasattr(module, "_dedicated_state"):
                for dp in module._dedicated_state.group.params:
                    if dp.is_owner:
                        self._dedicated_params.append(dp)

        # Discover FSDP2-managed params AND DDP-replicated params. Both go
        # into the same AdamW param group downstream. ``_fsdp_params`` keeps
        # its name for backwards compat with checkpoint.py even though the
        # list may now contain plain ``nn.Parameter`` (DDP path) alongside
        # FSDP2's sharded params.
        self._fsdp_params: list[nn.Parameter] = []
        fsdp_hit = False
        for module in model.modules():
            fsdp_state = getattr(module, "_get_fsdp_state", lambda: None)()
            if fsdp_state is not None and fsdp_state._fsdp_param_group is not None:
                for fp in fsdp_state._fsdp_param_group.fsdp_params:
                    self._fsdp_params.append(fp.sharded_param)
                fsdp_hit = True

        rep_group = getattr(model, "_replicated_group", None)
        replicate_hit = rep_group is not None
        if replicate_hit:
            self._fsdp_params.extend(rep_group.params)

        # Non-dedicated parameters must be covered by either fully_shard or
        # replicate — otherwise their grads are never synced and they never
        # receive an AdamW update. Fail loudly.
        non_dedicated_exists = any(
            not hasattr(p, "_dedicated_owner_rank") and p.requires_grad
            for p in model.parameters()
        )
        if (
            self._dedicated_params
            and non_dedicated_exists
            and not (fsdp_hit or replicate_hit)
        ):
            raise RuntimeError(
                "dmuon.Muon: model has non-dedicated parameters but neither "
                "fully_shard nor dmuon.replicate was called. Those parameters "
                "would not be synced across ranks nor updated. Call one of "
                "fully_shard() or dmuon.replicate(model, mesh=...) before "
                "constructing Muon."
            )

        # Build param_groups for Optimizer base class.
        # Group 0: placeholders for dedicated params (for LR scheduler compat).
        #   FSDP2 path uses the 0-size _placeholder; DDP path uses the live
        #   nn.Parameter (there is no separate placeholder, and the LR
        #   scheduler does not care about the tensor content — it only needs
        #   a stable Parameter ref).
        # Group 1: non-dedicated params (FSDP2-sharded or DDP-replicated).
        dedicated_placeholders = [
            # FSDP2 path has ``_placeholder`` (0-size nn.Parameter); DDP path
            # has ``_orig_param`` (live full Parameter). Pick whichever exists
            # — ``or`` cannot be used because bool() on an empty Tensor is
            # ambiguous.
            dp._placeholder if hasattr(dp, "_placeholder") else dp._orig_param
            for dp in self._dedicated_params
        ]
        param_groups = [
            {
                "params": dedicated_placeholders if dedicated_placeholders else [torch.zeros(1)],
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
            },
            {
                "params": list(self._fsdp_params) if self._fsdp_params else [torch.zeros(1)],
                "lr": adamw_lr,
                "betas": adamw_betas,
                "weight_decay": adamw_weight_decay,
                "eps": adamw_eps,
            },
        ]
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

        self._profile_step_idx = 0
        self._profile_param_timer: _balance_profile.ParamTimer = _balance_profile.ParamTimer()

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Internally:
        1. Waits for all async gradient reduces to complete (shard + replicate
           in HSDP mode).
        2. Runs Muon (momentum + NS + update) on owned dedicated params.
        3. Runs AdamW on FSDP2 symmetric params.
        4. In HSDP mode, broadcasts the updated ``_owned_data`` from the
           global owner to the replicate peers; no-op in 1D shard-only mode.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        broadcast_fn = (
            broadcast_all_updates_async
            if self._replicate_async
            else broadcast_all_updates
        )

        if _balance_profile.enabled():
            timer = _balance_profile.StepTimer()
            with timer.phase("wait_reduces"):
                wait_all_reduces(self.model)
            with timer.phase("muon"):
                self._step_muon()
            with timer.phase("adamw"):
                self._step_adamw()
            # Phase C.4: flip any slow group to sync BEFORE dispatching the
            # next broadcast so the new decision takes effect immediately.
            update_replicate_fallback(self.model)
            with timer.phase("replicate_broadcast"):
                broadcast_fn(self.model)

            owned_numel = sum(
                dp._owned_data.numel() for dp in self._dedicated_params
            )
            timer.report(
                self._profile_step_idx,
                extra={
                    "n_owned": len(self._dedicated_params),
                    "owned_numel": owned_numel,
                },
            )
            if _balance_profile.per_param_enabled():
                self._profile_param_timer.report(self._profile_step_idx)
                self._profile_param_timer = _balance_profile.ParamTimer()
            self._profile_step_idx += 1
            return loss

        # 1. Wait for all pending async reduces from backward
        wait_all_reduces(self.model)

        # 2. Muon update on dedicated params
        self._step_muon()

        # 3. AdamW update on FSDP2 params
        self._step_adamw()

        # 4. Advance the per-group async→sync fallback state machine
        # (Phase C.4).  Reads ``_last_replicate_wait_us`` populated during
        # the previous forward's ``_pre_forward_wait`` (only when
        # ``DMUON_REPLICATE_PROFILE`` is set).  Must run BEFORE dispatch so
        # a just-tripped flag affects this iteration.
        update_replicate_fallback(self.model)

        # 5. Fan updated _owned_data from global owner to replicate peers.
        # Phase C: by default dispatch async and let ``_pre_forward_wait``
        # consume the event at the start of the next forward; set
        # ``replicate_async=False`` on Muon construction to fall back to the
        # Phase B sync path.  No-op in 1D shard-only mode either way.
        broadcast_fn(self.model)

        return loss

    def _step_muon(self):
        """Newton-Schulz orthogonalization with momentum on dedicated params."""
        group = self.param_groups[0]
        lr = group["lr"]
        mu = group["momentum"]
        wd = group["weight_decay"]

        pt = self._profile_param_timer
        per_param = _balance_profile.per_param_enabled()

        # T2: if any TP-sharded param is present, ``tp_gather_grads`` on
        # ``reduce_stream`` produced ``_tp_full_grad`` for the TP owner.
        # Synchronise the compute stream once before the loop so the NS
        # read below sees that buffer.  No-op when no TP path is live
        # (reduce_stream is idle or already drained).
        if any(dp.tp_group is not None for dp in self._dedicated_params):
            torch.cuda.current_stream().wait_stream(self._comm_ctx.reduce_stream)

        for dp in self._dedicated_params:
            if dp._reduced_grad is None:
                continue

            if per_param:
                pt.start(
                    getattr(dp, "param_name", "<unknown>"),
                    tuple(dp._owned_data.shape),
                )

            # TP path (All-to-All): only the TP owner runs NS on the
            # reassembled full matrix; other DP-owner TP ranks produced
            # ``_reduced_grad`` too but leave NS to the owner — their
            # ``_owned_data`` will be overwritten by ``tp_scatter_delta``
            # with the owner's scattered shard.
            is_tp = dp.is_dtensor and dp.tp_group is not None

            if is_tp and not dp.is_tp_owner:
                # Nothing to compute here; tp_scatter_delta will deliver
                # the update.  Momentum state is owned solely by the TP
                # owner; non-owner ranks do not track it.  Still publish
                # the weight-decay factor so the scatter's in-place fuse
                # on this rank sees a matching wd — every DP-owner rank
                # computes wd identically from the same lr, wd config.
                #
                # DO NOT clear ``_reduced_grad`` here: ``tp_scatter_delta``
                # uses it as the per-rank "participate in collective"
                # gate (every rank must agree on the participant set).
                # It is cleared at the end of the scatter alongside
                # ``_tp_full_grad`` / ``_tp_full_delta``.
                dp._tp_wd_factor = (1.0 - lr * wd) if wd > 0 else 1.0
                if per_param:
                    pt.end()
                continue

            if is_tp:
                assert dp._tp_full_grad is not None, (
                    f"{getattr(dp, 'param_name', '?')}: tp_gather_grads did "
                    "not populate _tp_full_grad on TP owner"
                )
                grad = dp._tp_full_grad.view(dp._tp_full_grad.shape[0], -1)
            else:
                grad = dp._reduced_grad.view(dp._reduced_grad.shape[0], -1)

            # Momentum accumulation: buf = μ * buf + grad.  For TP-sharded
            # params the buf lives only on the TP owner and is sized to
            # the full matrix; non-owner ranks never touch this dict.
            dp_id = id(dp)
            if dp_id not in self.state:
                self.state[dp_id] = {}
            state = self.state[dp_id]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = grad.clone()
            else:
                state["momentum_buffer"].mul_(mu).add_(grad)
            buf = state["momentum_buffer"]

            # Nesterov lookahead: ns_input = grad + μ * buf
            ns_input = grad.add(buf, alpha=mu) if self._nesterov else buf

            # Newton-Schulz on the full (un-sharded) matrix.
            update = self._ns.local(ns_input, self._ns_steps)

            if is_tp:
                # Produce the **pre-scaled** full-matrix update that
                # tp_scatter_delta will chop and hand to each DP-owner
                # TP shard.  Each receiving rank does
                # ``_owned_data.mul_(wd_factor).add_(scatter_shard)``
                # to finish the Moonlight update in place — no further
                # NS-related work on the receivers.
                m_full = dp.full_shape[0]
                n_full = (
                    dp.full_shape[1] if len(dp.full_shape) > 1 else m_full
                )
                scale = 0.2 * (max(m_full, n_full) ** 0.5)
                update_full = update.view(dp.full_shape).to(dp._owned_data.dtype)
                update_full.mul_(-lr * scale)
                dp._tp_full_delta = update_full
                dp._tp_wd_factor = (1.0 - lr * wd) if wd > 0 else 1.0

            else:
                # Non-TP path: owner holds full tensor locally; apply in place.
                owned = dp._owned_data
                m = owned.shape[0]
                n = owned.view(m, -1).shape[1]
                scale = 0.2 * (max(m, n) ** 0.5)

                if wd > 0:
                    owned.mul_(1.0 - lr * wd)
                owned.add_(
                    update.view(owned.shape).to(owned.dtype), alpha=-lr * scale
                )

            # Non-TP params: clear now.  TP params defer the clear to
            # ``tp_scatter_delta`` so every TP group peer still sees the
            # "I had a grad this step" signal when scatter builds its
            # per-param work list.
            if not is_tp:
                dp._reduced_grad = None

            if per_param:
                pt.end()

    def _step_adamw(self):
        """AdamW update on FSDP2-managed symmetric params."""
        group = self.param_groups[1]
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        wd = group["weight_decay"]
        eps = group["eps"]

        for p in self._fsdp_params:
            if p.grad is None:
                continue

            grad = p.grad._local_tensor if hasattr(p.grad, "_local_tensor") else p.grad
            param = p._local_tensor if hasattr(p, "_local_tensor") else p.data

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # Decoupled weight decay
            if wd > 0:
                param.mul_(1.0 - lr * wd)

            # Adam moment updates
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            # Bias correction
            bc1 = 1.0 - beta1 ** state["step"]
            bc2 = 1.0 - beta2 ** state["step"]
            step_size = lr / bc1

            # Update
            denom = (exp_avg_sq.sqrt() / (bc2**0.5)).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)

            p.grad = None

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients.

        Clears FSDP2 params' gradients and dedicated params' accumulated
        gradients (from gradient accumulation).  Dedicated params' _reduced_grad
        is normally cleared in step(), but is also cleared here for safety.
        """
        for p in self._fsdp_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()
        for dp in self._dedicated_params:
            dp._reduced_grad = None
            dp._accumulated_grad = None
