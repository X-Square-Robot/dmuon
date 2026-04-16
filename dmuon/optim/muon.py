"""Muon optimizer with dedicated ownership for distributed training.

Combines Newton-Schulz orthogonalization on dedicated parameters with
AdamW on symmetric (FSDP2-managed) parameters in a single optimizer.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer

from typing import Union

from ..utils import get_owned_params, wait_all_reduces
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
            TP params requiring Gram decomposition always use
            ``gram_newton_schulz`` regardless.
        nesterov: If True (default), use Nesterov momentum lookahead
            before NS orthogonalization: ``ns_input = grad + μ * buf``.
            Recommended by original Muon paper and used by Moonlight.
        per_head_ns: If True (default), use per-head local NS for narrow
            Shard(0) params (GQA k/v_proj) where full m < n. Each rank
            orthogonalizes its heads independently with zero TP communication.
        block_diagonal_ns: If True, skip Gram all-reduce for ALL TP params
            and use local Gram only (block-diagonal approximation). Eliminates
            all TP optimizer communication. Experimental — needs convergence
            validation.

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
        per_head_ns: bool = True,
        block_diagonal_ns: bool = False,
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
        self._per_head_ns = per_head_ns
        self._block_diagonal_ns = block_diagonal_ns

        # Discover dedicated params owned by this rank
        comm_ctx = getattr(model, "_dedicated_comm_ctx", None)
        if comm_ctx is None:
            raise ValueError(
                "Model has no _dedicated_comm_ctx. Call dmuon.dedicate_params() first."
            )
        self._dedicated_params = []
        for module in model.modules():
            if hasattr(module, "_dedicated_state"):
                for dp in module._dedicated_state.group.params:
                    if dp.is_owner:
                        self._dedicated_params.append(dp)

        # Discover FSDP2-managed params
        self._fsdp_params = []
        for module in model.modules():
            fsdp_state = getattr(module, "_get_fsdp_state", lambda: None)()
            if fsdp_state is not None and fsdp_state._fsdp_param_group is not None:
                for fp in fsdp_state._fsdp_param_group.fsdp_params:
                    self._fsdp_params.append(fp.sharded_param)

        # Build param_groups for Optimizer base class
        # Group 0: placeholders for dedicated params (for LR scheduler compat)
        # Group 1: FSDP2 params (AdamW)
        dedicated_placeholders = [dp._placeholder for dp in self._dedicated_params]
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

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Internally:
        1. Waits for all async gradient reduces to complete.
        2. Runs Muon (momentum + NS + update) on owned dedicated params.
        3. Runs AdamW on FSDP2 symmetric params.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 1. Wait for all pending async reduces from backward
        wait_all_reduces(self.model)

        # 2. Muon update on dedicated params
        self._step_muon()

        # 3. AdamW update on FSDP2 params
        self._step_adamw()

        return loss

    def _step_muon(self):
        """Newton-Schulz orthogonalization with momentum on dedicated params."""
        group = self.param_groups[0]
        lr = group["lr"]
        mu = group["momentum"]
        wd = group["weight_decay"]

        for dp in self._dedicated_params:
            if dp._reduced_grad is None:
                continue

            grad = dp._reduced_grad.view(dp._reduced_grad.shape[0], -1)

            # Momentum accumulation: buf = μ * buf + grad
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
            # (standard Muon/Moonlight convention, gives better direction estimate)
            ns_input = grad.add(buf, alpha=mu) if self._nesterov else buf

            # Newton-Schulz orthogonalization — TP-aware routing:
            #   Per-head NS: narrow Shard(0) (GQA k/v_proj) → local NS, zero TP comm
            #   Block-diag NS: skip all-reduce, use local Gram only (experimental)
            #   Exact Gram NS: all-reduce decomposable Gram → standard path
            #   Non-TP: local NS (pure DP, owner has full gradient)
            ns = self._ns

            if dp.is_dtensor and dp.tp_group is not None:
                shard_dim = dp.shard_dim
                full_shape = dp.full_shape
                m_full = full_shape[0]
                n_full = full_shape[1] if len(full_shape) > 1 else m_full

                if self._per_head_ns and shard_dim == 0 and m_full < n_full:
                    # Per-head NS: each rank has complete KV heads
                    update = ns.local(ns_input, self._ns_steps)
                elif self._block_diagonal_ns:
                    # Block-diagonal NS: zero TP comm (experimental)
                    update = ns.tp(
                        ns_input, dp.tp_group, self._ns_steps,
                        shard_dim=shard_dim, block_diagonal=True,
                    )
                else:
                    # Exact Gram NS with TP all-reduce
                    update = ns.tp(
                        ns_input, dp.tp_group, self._ns_steps, shard_dim=shard_dim,
                    )
            else:
                update = ns.local(ns_input, self._ns_steps)

            # Per-param scaling (Moonlight): 0.2 * sqrt(max(m, n))
            owned = dp._owned_data
            m = owned.shape[0]
            n = owned.view(m, -1).shape[1]
            scale = 0.2 * (max(m, n) ** 0.5)

            # Weight decay (decoupled, like AdamW)
            if wd > 0:
                owned.mul_(1.0 - lr * wd)

            # Apply update
            owned.add_(update.view(owned.shape).to(owned.dtype), alpha=-lr * scale)

            # Clear gradient
            dp._reduced_grad = None

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
