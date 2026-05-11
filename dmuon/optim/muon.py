"""Muon optimizer with dedicated ownership for distributed training.

Combines Newton-Schulz orthogonalization on dedicated parameters with
AdamW on symmetric (FSDP2-managed) parameters in a single optimizer.
"""

import os
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from .. import _balance_profile
from ..grad_clip import (
    MuonGradClipStats,
    _clip_ready_muon_grad_norm_,
)
from ..utils import (
    _dispatch_post_step_async,
    _ordered_post_step_groups,
    broadcast_all_updates,
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
        param_groups: Optional PyTorch-style semantic parameter groups. Each
            user group is lowered into a Muon subgroup and an AdamW subgroup
            so schedulers and checkpoint metadata can keep per-group
            hyperparameters without exposing DMuon internals.
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
        param_groups: Optional[list[dict]] = None,
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
        self._grads_ready = False
        self._last_muon_grad_clip_stats: Optional[MuonGradClipStats] = None
        # Phase C: toggle between async (default, hides broadcast inside
        # the next forward) and Phase B sync (simpler, always-correct).
        # When True, each group's pending event is consumed by its own
        # ``_pre_forward_wait`` hook; when False, the full fan-out is
        # waited synchronously at the end of step().
        self._replicate_async = replicate_async

        # Discover all dedicated params, and the subset owned by this rank.
        comm_ctx = getattr(model, "_dedicated_comm_ctx", None)
        if comm_ctx is None:
            raise ValueError(
                "Model has no _dedicated_comm_ctx. Call dmuon.dedicate_params() first."
            )
        self._comm_ctx = comm_ctx
        self._all_dedicated_params = []
        self._dedicated_params = []
        seen_dps: set[int] = set()
        for module in model.modules():
            if hasattr(module, "_dedicated_state"):
                for dp in module._dedicated_state.group.params:
                    if id(dp) in seen_dps:
                        continue
                    seen_dps.add(id(dp))
                    self._all_dedicated_params.append(dp)
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
            self._all_dedicated_params
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

        self._dummy_params: list[nn.Parameter] = []
        self._muon_group_dps: dict[int, list] = {}
        self._adamw_group_params: dict[int, list[nn.Parameter]] = {}
        self._dp_to_muon_group_idx: dict[int, int] = {}
        self._adamw_param_to_group_idx: dict[int, int] = {}

        optimizer_groups = self._build_optimizer_param_groups(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_weight_decay=adamw_weight_decay,
            adamw_eps=adamw_eps,
        )
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(optimizer_groups, defaults)

        self._profile_step_idx = 0
        self._profile_param_timer: _balance_profile.ParamTimer = _balance_profile.ParamTimer()
        self._step_profile_enabled = False
        self._last_step_profile: dict[str, object] = {}
        self._last_step_profile_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def _dummy_param(self) -> nn.Parameter:
        device = torch.device("cpu")
        dtype = torch.float32
        for p in self.model.parameters():
            device = getattr(p, "device", device)
            dtype = getattr(p, "dtype", dtype)
            break
        dummy = nn.Parameter(torch.zeros(1, device=device, dtype=dtype), requires_grad=False)
        self._dummy_params.append(dummy)
        return dummy

    @staticmethod
    def _param_ref_for_dp(dp):
        if hasattr(dp, "_placeholder"):
            return dp._placeholder
        return dp._orig_param

    @staticmethod
    def _param_refs_for_dp(dp) -> list[torch.Tensor]:
        refs = []
        if hasattr(dp, "_placeholder"):
            refs.append(dp._placeholder)
        if hasattr(dp, "_orig_param"):
            refs.append(dp._orig_param)
        return refs

    @staticmethod
    def _group_value(group: dict, primary: str, secondary: Optional[str], default):
        if primary in group:
            return group[primary]
        if secondary is not None and secondary in group:
            return group[secondary]
        return default

    @staticmethod
    def _normalize_group_params(params) -> list[torch.Tensor]:
        if isinstance(params, torch.Tensor):
            return [params]
        try:
            return list(params)
        except TypeError as exc:
            raise TypeError(
                "dmuon.Muon param_groups entries must use a Tensor or an "
                "iterable of Tensors under the 'params' key"
            ) from exc

    def _make_muon_group(
        self,
        *,
        params: list[torch.Tensor],
        semantic_name: str,
        lr: float,
        momentum: float,
        weight_decay: float,
    ) -> dict:
        return {
            "params": params if params else [self._dummy_param()],
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "use_muon": True,
            "group_name": f"{semantic_name}/muon",
            "semantic_group_name": semantic_name,
            "subgroup_type": "muon",
        }

    def _make_adamw_group(
        self,
        *,
        params: list[nn.Parameter],
        semantic_name: str,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        eps: float,
    ) -> dict:
        return {
            "params": params if params else [self._dummy_param()],
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
            "use_muon": False,
            "group_name": f"{semantic_name}/adamw",
            "semantic_group_name": semantic_name,
            "subgroup_type": "adamw",
        }

    def _build_optimizer_param_groups(
        self,
        user_param_groups: Optional[list[dict]],
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        adamw_lr: float,
        adamw_betas: tuple[float, float],
        adamw_weight_decay: float,
        adamw_eps: float,
    ) -> list[dict]:
        if user_param_groups is None:
            return self._build_default_param_groups(
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                adamw_lr=adamw_lr,
                adamw_betas=adamw_betas,
                adamw_weight_decay=adamw_weight_decay,
                adamw_eps=adamw_eps,
            )
        return self._build_semantic_param_groups(
            user_param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_weight_decay=adamw_weight_decay,
            adamw_eps=adamw_eps,
        )

    def _build_default_param_groups(
        self,
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        adamw_lr: float,
        adamw_betas: tuple[float, float],
        adamw_weight_decay: float,
        adamw_eps: float,
    ) -> list[dict]:
        muon_refs = [self._param_ref_for_dp(dp) for dp in self._dedicated_params]
        groups = [
            self._make_muon_group(
                params=muon_refs,
                semantic_name="default",
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            ),
            self._make_adamw_group(
                params=list(self._fsdp_params),
                semantic_name="default",
                lr=adamw_lr,
                betas=adamw_betas,
                weight_decay=adamw_weight_decay,
                eps=adamw_eps,
            ),
        ]
        self._muon_group_dps[0] = list(self._dedicated_params)
        self._adamw_group_params[1] = list(self._fsdp_params)
        for dp in self._all_dedicated_params:
            self._dp_to_muon_group_idx[id(dp)] = 0
        for p in self._fsdp_params:
            self._adamw_param_to_group_idx[id(p)] = 1
        return groups

    def _build_semantic_param_groups(
        self,
        user_param_groups,
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        adamw_lr: float,
        adamw_betas: tuple[float, float],
        adamw_weight_decay: float,
        adamw_eps: float,
    ) -> list[dict]:
        if isinstance(user_param_groups, dict):
            raise TypeError("dmuon.Muon param_groups must be a list of dicts, not a dict")
        semantic_groups = list(user_param_groups)
        if not semantic_groups:
            raise ValueError("dmuon.Muon param_groups must not be empty")

        param_to_dp: dict[int, object] = {}
        for dp in self._all_dedicated_params:
            for ref in self._param_refs_for_dp(dp):
                param_to_dp[id(ref)] = dp
        adamw_param_by_id = {id(p): p for p in self._fsdp_params}
        model_param_names = {id(p): name for name, p in self.model.named_parameters()}

        optimizer_groups: list[dict] = []
        assigned: dict[tuple[str, int], str] = {}

        for semantic_idx, user_group in enumerate(semantic_groups):
            if not isinstance(user_group, dict):
                raise TypeError(
                    f"dmuon.Muon param_groups[{semantic_idx}] must be a dict"
                )
            if "params" not in user_group:
                raise ValueError(
                    f"dmuon.Muon param_groups[{semantic_idx}] is missing 'params'"
                )

            semantic_name = str(
                user_group.get("group_name", user_group.get("name", f"group_{semantic_idx}"))
            )
            raw_params = self._normalize_group_params(user_group["params"])
            muon_refs: list[torch.Tensor] = []
            owned_muon_dps = []
            all_muon_dps = []
            adamw_params: list[nn.Parameter] = []

            for param in raw_params:
                if not isinstance(param, torch.Tensor):
                    if hasattr(param, "_owned_data") and hasattr(param, "param_name"):
                        raise TypeError(
                            "dmuon.Muon param_groups must contain tensors, not "
                            "DedicatedParam objects"
                        )
                    raise TypeError(
                        f"dmuon.Muon param_groups[{semantic_idx}] contains "
                        f"{type(param).__name__}, expected Tensor"
                    )
                if not getattr(param, "requires_grad", False):
                    continue

                param_id = id(param)
                if param_id in param_to_dp:
                    dp = param_to_dp[param_id]
                    assignment_key = ("muon", id(dp))
                    if assignment_key in assigned:
                        raise ValueError(
                            f"trainable parameter appears in multiple DMuon "
                            f"param_groups: {assigned[assignment_key]} and {semantic_name}"
                        )
                    assigned[assignment_key] = semantic_name
                    muon_refs.append(param)
                    all_muon_dps.append(dp)
                    if getattr(dp, "is_owner", False):
                        owned_muon_dps.append(dp)
                elif param_id in adamw_param_by_id:
                    p = adamw_param_by_id[param_id]
                    assignment_key = ("adamw", id(p))
                    if assignment_key in assigned:
                        raise ValueError(
                            f"trainable parameter appears in multiple DMuon "
                            f"param_groups: {assigned[assignment_key]} and {semantic_name}"
                        )
                    assigned[assignment_key] = semantic_name
                    adamw_params.append(p)
                else:
                    name = model_param_names.get(param_id, "<not in current model>")
                    raise RuntimeError(
                        "dmuon.Muon param_groups contains a trainable parameter "
                        f"that DMuon does not manage: {name}. Build param_groups "
                        "from the wrapped model passed to Muon."
                    )

            muon_group_idx = len(optimizer_groups)
            optimizer_groups.append(
                self._make_muon_group(
                    params=muon_refs,
                    semantic_name=semantic_name,
                    lr=self._group_value(user_group, "muon_lr", "lr", lr),
                    momentum=self._group_value(user_group, "momentum", None, momentum),
                    weight_decay=self._group_value(
                        user_group, "muon_weight_decay", "weight_decay", weight_decay
                    ),
                )
            )
            self._muon_group_dps[muon_group_idx] = owned_muon_dps
            for dp in all_muon_dps:
                self._dp_to_muon_group_idx[id(dp)] = muon_group_idx

            adamw_group_idx = len(optimizer_groups)
            optimizer_groups.append(
                self._make_adamw_group(
                    params=adamw_params,
                    semantic_name=semantic_name,
                    lr=self._group_value(user_group, "adamw_lr", "lr", adamw_lr),
                    betas=self._group_value(user_group, "adamw_betas", None, adamw_betas),
                    weight_decay=self._group_value(
                        user_group,
                        "adamw_weight_decay",
                        "weight_decay",
                        adamw_weight_decay,
                    ),
                    eps=self._group_value(user_group, "adamw_eps", None, adamw_eps),
                )
            )
            self._adamw_group_params[adamw_group_idx] = adamw_params
            for p in adamw_params:
                self._adamw_param_to_group_idx[id(p)] = adamw_group_idx

        missing: list[str] = []
        unmanaged: list[str] = []
        for name, param in self.model.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            param_id = id(param)
            if param_id in param_to_dp:
                key = ("muon", id(param_to_dp[param_id]))
            elif param_id in adamw_param_by_id:
                key = ("adamw", id(adamw_param_by_id[param_id]))
            else:
                unmanaged.append(name)
                continue
            if key not in assigned:
                missing.append(name)

        if unmanaged:
            preview = ", ".join(unmanaged[:8])
            raise RuntimeError(
                "dmuon.Muon found trainable model parameters that are not "
                f"managed by DMuon: {preview}"
            )
        if missing:
            preview = ", ".join(missing[:8])
            raise RuntimeError(
                "dmuon.Muon param_groups omitted trainable model parameters: "
                f"{preview}"
            )

        return optimizer_groups

    @property
    def last_muon_grad_clip_stats(self) -> Optional[MuonGradClipStats]:
        """Stats from the most recent Muon gradient clipping call."""

        return self._last_muon_grad_clip_stats

    def _profile_requested(self) -> bool:
        return bool(int(os.environ.get("DMUON_STEP_PROFILE", "0") or 0))

    def _profile_begin_step(self) -> None:
        self._step_profile_enabled = self._profile_requested() and torch.cuda.is_available()
        self._last_step_profile_events = []
        self._last_step_profile = {
            "enabled": self._step_profile_enabled,
            "ns_matrix_count": 0,
            "ns_input_numel": 0,
        }

    def _profile_event_start(self, name: str):
        if not self._step_profile_enabled:
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return name, start, end

    def _profile_event_end(self, token) -> None:
        if token is None:
            return
        _name, _start, end = token
        end.record()
        self._last_step_profile_events.append(token)

    def _profile_add(self, key: str, value: int | float) -> None:
        if not self._step_profile_enabled:
            return
        current = self._last_step_profile.get(key, 0)
        self._last_step_profile[key] = current + value

    def consume_last_step_profile(self) -> dict[str, object]:
        """Return the last step's CUDA event timings after the caller synced."""

        profile = dict(self._last_step_profile)
        if not profile:
            return {}
        if not profile.get("enabled"):
            self._last_step_profile_events = []
            return profile
        event_totals: dict[str, float] = {}
        for name, start, end in self._last_step_profile_events:
            event_totals[name] = event_totals.get(name, 0.0) + float(
                start.elapsed_time(end)
            )
        for name, value in event_totals.items():
            profile[f"{name}_ms"] = round(value, 6)
        profile["ns_compute_ms"] = round(event_totals.get("ns_compute", 0.0), 6)
        self._last_step_profile_events = []
        return profile

    def _ensure_grads_ready(self) -> None:
        """Wait for DMuon reduce/gather once before clipping or stepping."""

        if self._grads_ready:
            return
        wait_all_reduces(self.model)
        self._grads_ready = True

    @torch.no_grad()
    def clip_grad_norm_(
        self,
        max_norm: Optional[float],
        *,
        norm_type: Optional[float] = None,
        error_if_nonfinite: Optional[bool] = None,
        foreach: Optional[bool] = None,
        strategy: Optional[object] = None,
    ) -> MuonGradClipStats:
        """Clip DMuon dedicated/Muon gradients only.

        Ordinary AdamW parameters are intentionally excluded.  Use
        ``torch.nn.utils.clip_grad_norm_`` for those ``param.grad`` tensors and
        call this method as the DMuon-specific extra line.
        """

        self._ensure_grads_ready()
        stats = _clip_ready_muon_grad_norm_(
            self,
            max_norm,
            norm_type=2.0 if norm_type is None else norm_type,
            error_if_nonfinite=False if error_if_nonfinite is None else error_if_nonfinite,
            foreach=foreach,
            strategy="global_norm" if strategy is None else strategy,
        )
        return stats

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Internally:
        1. Waits for all async gradient reduces to complete (shard + replicate
           in HSDP mode).
        2. Runs Muon (momentum + NS + update) on owned dedicated params.
        3. Runs AdamW on FSDP2 symmetric params.
        4. Publishes updated dedicated params. In sync mode this happens as
           one post-step phase; in async mode each group dispatches its
           scatter/broadcast immediately after that group's Muon update, and
           the next forward consumes the pending event before reading it.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._profile_begin_step()
        if _balance_profile.enabled():
            timer = _balance_profile.StepTimer()
            with timer.phase("wait_reduces"):
                profile_token = self._profile_event_start("wait_reduces")
                try:
                    self._ensure_grads_ready()
                finally:
                    self._profile_event_end(profile_token)
            if self._replicate_async:
                update_replicate_fallback(self.model)
                with timer.phase("muon"):
                    profile_token = self._profile_event_start("muon")
                    try:
                        self._step_muon_and_dispatch_groups_async()
                    finally:
                        self._profile_event_end(profile_token)
            else:
                with timer.phase("muon"):
                    profile_token = self._profile_event_start("muon")
                    try:
                        self._step_muon()
                    finally:
                        self._profile_event_end(profile_token)
            with timer.phase("adamw"):
                profile_token = self._profile_event_start("adamw")
                try:
                    self._step_adamw()
                finally:
                    self._profile_event_end(profile_token)
            if not self._replicate_async:
                # Phase C.4: flip any slow group to sync BEFORE dispatching the
                # next broadcast so the new decision takes effect immediately.
                update_replicate_fallback(self.model)
                with timer.phase("replicate_broadcast"):
                    profile_token = self._profile_event_start("replicate_broadcast")
                    try:
                        broadcast_all_updates(self.model)
                    finally:
                        self._profile_event_end(profile_token)

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
            self._grads_ready = False
            return loss

        # 1. Wait for all pending async reduces from backward
        profile_token = self._profile_event_start("wait_reduces")
        try:
            self._ensure_grads_ready()
        finally:
            self._profile_event_end(profile_token)

        if self._replicate_async:
            # Phase C.4 fallback must be advanced before dispatching this
            # step's post-update collectives so a tripped group degrades
            # immediately.  Then pipeline each group: update it, enqueue its
            # post-step scatter/broadcast, and continue with later groups.
            update_replicate_fallback(self.model)
            profile_token = self._profile_event_start("muon")
            try:
                self._step_muon_and_dispatch_groups_async()
            finally:
                self._profile_event_end(profile_token)
        else:
            # 2. Muon update on dedicated params
            profile_token = self._profile_event_start("muon")
            try:
                self._step_muon()
            finally:
                self._profile_event_end(profile_token)

        # 3. AdamW update on FSDP2 params
        profile_token = self._profile_event_start("adamw")
        try:
            self._step_adamw()
        finally:
            self._profile_event_end(profile_token)

        if not self._replicate_async:
            # 4. Advance the per-group async→sync fallback state machine
            # (Phase C.4).  Reads ``_last_replicate_wait_us`` populated during
            # the previous forward's ``_pre_forward_wait`` (only when
            # ``DMUON_REPLICATE_PROFILE`` is set).  Must run BEFORE dispatch so
            # a just-tripped flag affects this iteration.
            update_replicate_fallback(self.model)

            # 5. Fan updated _owned_data from global owner to replicate peers.
            # Sync mode preserves the old full-step dispatch+wait contract.
            profile_token = self._profile_event_start("replicate_broadcast")
            try:
                broadcast_all_updates(self.model)
            finally:
                self._profile_event_end(profile_token)

        self._grads_ready = False
        return loss

    def _step_muon_and_dispatch_groups_async(self) -> None:
        """Pipeline group-local Muon updates with post-step communication.

        Group order follows the previous forward order.  Every rank dispatches
        every group in the same order, while only the owner ranks do the local
        Newton-Schulz work before their group's collective is enqueued.
        """
        for group in _ordered_post_step_groups(self.model):
            owned_params = [
                dp for dp in getattr(group, "params", ())
                if getattr(dp, "is_owner", False)
            ]
            if owned_params:
                self._step_muon(owned_params)

            profile_token = self._profile_event_start("replicate_broadcast")
            try:
                _dispatch_post_step_async(group)
            finally:
                self._profile_event_end(profile_token)

    def _step_muon(self, params=None):
        """Newton-Schulz orthogonalization with momentum on dedicated params.

        With ``params=None`` this walks Muon optimizer subgroups.  With a
        communication-group-local ``params`` list, it keeps that outer
        communication order and only uses subgroup metadata for per-param
        hyperparameters.
        """
        if params is None:
            for group_idx, dedicated_params in self._muon_group_dps.items():
                self._step_muon_params(dedicated_params, self.param_groups[group_idx])
            return

        by_group: dict[int, list] = {}
        for dp in params:
            group_idx = self._dp_to_muon_group_idx.get(id(dp))
            if group_idx is None:
                raise RuntimeError(
                    f"DMuon dedicated param {getattr(dp, 'param_name', '<unknown>')} "
                    "is not assigned to a Muon param group"
                )
            by_group.setdefault(group_idx, []).append(dp)

        for group_idx, dedicated_params in by_group.items():
            self._step_muon_params(dedicated_params, self.param_groups[group_idx])

    def _step_muon_params(self, dedicated_params, group: dict) -> None:
        """Apply one Muon subgroup's hyperparameters to dedicated params."""
        lr = group["lr"]
        mu = group["momentum"]
        wd = group["weight_decay"]
        dedicated_params = list(dedicated_params)

        pt = self._profile_param_timer
        per_param = _balance_profile.per_param_enabled()

        # T2: if any TP-sharded param is present, ``tp_gather_grads`` on
        # ``reduce_stream`` produced ``_tp_full_grad`` for the TP owner.
        # Synchronise the compute stream once before the loop so the NS
        # read below sees that buffer.  No-op when no TP path is live
        # (reduce_stream is idle or already drained).
        if any(dp.tp_group is not None for dp in dedicated_params):
            torch.cuda.current_stream().wait_stream(self._comm_ctx.reduce_stream)

        for dp in dedicated_params:
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
            profile_token = self._profile_event_start("ns_compute")
            try:
                update = self._ns.local(ns_input, self._ns_steps)
            finally:
                self._profile_event_end(profile_token)
            self._profile_add("ns_matrix_count", 1)
            self._profile_add("ns_input_numel", int(ns_input.numel()))

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
                update_full = update.view(dp.full_shape).to(
                    device=dp._owned_data.device,
                    dtype=dp._owned_data.dtype,
                )
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
                    update.view(owned.shape).to(device=owned.device, dtype=owned.dtype),
                    alpha=-lr * scale,
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
        for group_idx, params in self._adamw_group_params.items():
            self._step_adamw_params(params, self.param_groups[group_idx])

    def _step_adamw_params(self, params, group: dict) -> None:
        """Apply one AdamW subgroup's hyperparameters to managed params."""
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        wd = group["weight_decay"]
        eps = group["eps"]

        for p in params:
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
        self._grads_ready = False
        self._last_muon_grad_clip_stats = None
        for p in self._fsdp_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()
        for dp in self._dedicated_params:
            dp._reduced_grad = None
            dp._accumulated_grad = None
