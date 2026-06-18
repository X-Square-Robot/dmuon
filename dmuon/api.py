"""Public API: dedicate_params and DDP variants."""

import logging
import os
from collections import defaultdict
from typing import Callable, Optional

import torch
import torch.nn as nn

try:
    from torch.distributed import DeviceMesh
except ImportError:  # Older PyTorch exposes DeviceMesh only from this module.
    from torch.distributed.device_mesh import DeviceMesh

from ._backends.ddp import DedicatedParamDDP, DedicatedParamGroupDDP
from ._backends.fsdp2 import DedicatedParam, DedicatedParamGroup
from ._core.comm import DedicatedCommContext
from ._core.internal_utils import unsafe_setattr_param
from ._core.owner_rank import normalize_owner_rank
from ._core.partition import _extract_layer_id, compute_balanced_assignment
from ._core.state import DedicatedState
from ._core.tp import is_tp_sharded

logger = logging.getLogger(__name__)


def _find_parent_module(
    model: nn.Module, target_param: nn.Parameter
) -> tuple[nn.Module, str]:
    """Find the direct parent module and param name for a parameter."""
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param is target_param:
                return module, param_name
    raise ValueError(f"Parameter not found in model: {target_param.shape}")


def _find_parent_modules(
    model: nn.Module, target_param: nn.Parameter
) -> list[tuple[nn.Module, str]]:
    """Find all direct module attributes that reference a parameter.

    Tied embeddings expose the same ``nn.Parameter`` through multiple parent
    modules.  Dedicated ownership must replace every alias with the same
    placeholder/full-matrix parameter, otherwise one alias remains outside both
    DMuon and FSDP management.
    """

    parents: list[tuple[nn.Module, str]] = []
    for _module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param is target_param:
                parents.append((module, param_name))
    if not parents:
        raise ValueError(f"Parameter not found in model: {target_param.shape}")
    return parents


def _module_fqn_map(model: nn.Module) -> dict[nn.Module, str]:
    return {module: name for name, module in model.named_modules()}


def _link_forward_prefetch_states(
    model: nn.Module,
    states: list[DedicatedState],
    comm_ctx: DedicatedCommContext,
) -> None:
    """Link DedicatedState objects in model registration order.

    Owner assignment may iterate parameters by load-balance order, which is not
    necessarily the order in which modules run forward.  Forward prefetch must
    target the next module in the actual model walk; otherwise unshard
    collectives are issued too late and show up as exposed communication before
    each layer.
    """

    module_order = {id(module): idx for idx, module in enumerate(model.modules())}
    ordered = sorted(
        states,
        key=lambda state: module_order.get(id(state.module), len(module_order)),
    )
    for state in ordered:
        state._next_group = None
        state._next_groups = []
    for idx in range(len(ordered) - 1):
        ordered[idx]._next_group = ordered[idx + 1].group
        ordered[idx]._next_groups = [
            later_state.group for later_state in ordered[idx + 1 :]
        ]
    comm_ctx.all_states[:] = ordered


def _lowest_common_ancestor_module(
    model: nn.Module, modules: list[nn.Module]
) -> nn.Module:
    """Return the lowest module that contains every module in ``modules``.

    Tied parameters can be read through aliases that live in different parts of
    the model, for example input embeddings and the output head.  A Z3-style
    reshard-after-forward policy is only valid if the hook boundary spans every
    alias use.
    """

    if not modules:
        return model

    fqns = _module_fqn_map(model)
    paths = [fqns[module].split(".") if fqns[module] else [] for module in modules]
    prefix: list[str] = []
    for parts in zip(*paths):
        if all(part == parts[0] for part in parts):
            prefix.append(parts[0])
        else:
            break
    common_path = ".".join(prefix)
    return model.get_submodule(common_path) if common_path else model


def _find_layer_module(
    model: nn.Module, target_param: nn.Parameter
) -> tuple[nn.Module, nn.Module, str]:
    """Find (layer_module, parent_module, param_name) for a parameter.

    layer_module: the transformer layer (e.g., model.layers[i]) for hook registration.
    parent_module: the direct parent (e.g., q_proj Linear) for _set_module_param.
    param_name: name within parent_module.

    Falls back to parent_module as layer_module if no layer structure found.
    """
    parent_module, param_name = _find_parent_module(model, target_param)

    # Find the fully qualified name of parent_module
    parent_fqn: Optional[str] = None
    for name, mod in model.named_modules():
        if mod is parent_module:
            parent_fqn = name
            break

    if parent_fqn is None:
        return parent_module, parent_module, param_name

    # Extract layer prefix: "model.layers.3.self_attn.q_proj" → "layers.3"
    layer_id = _extract_layer_id(parent_fqn)
    if layer_id is None:
        return parent_module, parent_module, param_name

    # Build the full path to the layer module
    # e.g., parent_fqn = "model.layers.3.self_attn.q_proj", layer_id = "layers.3"
    layer_path_id = layer_id.removeprefix("_root.")
    idx = parent_fqn.find(layer_path_id)
    if idx < 0:
        return parent_module, parent_module, param_name
    layer_path = parent_fqn[: idx + len(layer_path_id)]
    try:
        layer_module = model.get_submodule(layer_path)
    except AttributeError:
        return parent_module, parent_module, param_name

    return layer_module, parent_module, param_name


def _find_hook_module(
    model: nn.Module,
    target_param: nn.Parameter,
    hook_boundary_predicate: Optional[Callable[[nn.Module], bool]],
    strict: bool = True,
) -> Optional[nn.Module]:
    """Return the lowest ancestor of ``target_param`` where the predicate is True.

    When ``hook_boundary_predicate`` is None, returns None — caller should fall
    back to the ``_find_layer_module`` path. This split keeps the two semantics
    (partition layer key vs. hook attachment module) decoupled.

    Args:
        model: Root module to search within.
        target_param: Parameter whose ancestor we want.
        hook_boundary_predicate: Callable ``(module) -> bool``. None means the
            caller opted out of explicit hook boundaries.
        strict: When True (default), raise if no ancestor matches the predicate.
            When False, fall back to the parameter's direct parent module.

    Returns:
        The chosen hook module, or None when predicate is None.
    """
    if hook_boundary_predicate is None:
        return None

    parent_module, _ = _find_parent_module(model, target_param)

    # Find parent's FQN (relative to `model`)
    parent_fqn: Optional[str] = None
    for name, mod in model.named_modules():
        if mod is parent_module:
            parent_fqn = name
            break

    # Walk from longest prefix down to root, checking the submodule at each
    # prefix against the predicate.
    parts = parent_fqn.split(".") if parent_fqn else []
    for depth in range(len(parts), -1, -1):
        path = ".".join(parts[:depth])
        try:
            mod = model.get_submodule(path) if path else model
        except AttributeError:
            continue
        if hook_boundary_predicate(mod):
            return mod

    if strict:
        raise ValueError(
            f"param of shape {tuple(target_param.shape)} "
            f"(parent_fqn={parent_fqn!r}): no ancestor matched "
            f"hook_boundary_predicate. Either extend the predicate to cover this "
            f"module, or exclude the param via the dedicate_params `predicate` arg."
        )
    return parent_module


HookBoundaryResolver = Callable[[str, nn.Parameter], Optional[nn.Module]]
AssignmentGroupKeyFn = Callable[[str, nn.Parameter], Optional[str]]
RouteHintFn = Callable[[str, nn.Parameter], Optional[str]]


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {value!r}")


def _resolve_owner_strategy(owner_strategy: Optional[str]) -> str:
    resolved = owner_strategy or os.environ.get("DMUON_OWNER_STRATEGY", "lpt")
    if resolved not in {"lpt", "round_robin", "rank0"}:
        raise ValueError(
            f"Unsupported owner_strategy: {resolved!r}; "
            "expected 'lpt', 'round_robin', or 'rank0'."
        )
    return resolved


def _resolve_owner_cost_model(owner_cost_model: Optional[str]) -> str:
    resolved = owner_cost_model or os.environ.get("DMUON_OWNER_COST_MODEL", "optimizer")
    if resolved not in {"optimizer", "numel"}:
        raise ValueError(
            f"Unsupported owner_cost_model: {resolved!r}; "
            "expected 'optimizer' or 'numel'."
        )
    return resolved


def _resolve_owner_pack_small_params(pack_small_params: Optional[bool]) -> bool:
    if pack_small_params is None:
        return _env_flag("DMUON_OWNER_PACK_SMALL_PARAMS", True)
    return bool(pack_small_params)


def _resolve_hook_module(
    model: nn.Module,
    target_param: nn.Parameter,
    *,
    param_fqn: str,
    hook_boundary_predicate: Optional[Callable[[nn.Module], bool]],
    hook_boundary_resolver: Optional[HookBoundaryResolver],
    strict: bool,
) -> nn.Module:
    """Resolve the forward boundary that should own a dedicated parameter."""
    if hook_boundary_resolver is not None:
        module = hook_boundary_resolver(param_fqn, target_param)
        if module is not None:
            return module
        if strict:
            raise ValueError(
                f"{param_fqn}: hook_boundary_resolver returned None. Either "
                "extend the resolver to cover this parameter or exclude it via "
                "the dedicate_params predicate."
            )
        parent_module, _ = _find_parent_module(model, target_param)
        return parent_module

    if hook_boundary_predicate is not None:
        module = _find_hook_module(
            model, target_param, hook_boundary_predicate, strict=strict
        )
        assert module is not None
        return module

    layer_module, _parent_module, _param_name = _find_layer_module(model, target_param)
    return layer_module


def dedicate_params(
    model: nn.Module,
    mesh: DeviceMesh,
    predicate: Callable[[str, nn.Parameter], bool],
    compute_dtype: torch.dtype = None,
    reshard_after_forward: bool = True,
    replicate_mesh: Optional[DeviceMesh] = None,
    hook_boundary_predicate: Optional[Callable[[nn.Module], bool]] = None,
    hook_boundary_strict: bool = True,
    hook_boundary_resolver: Optional[HookBoundaryResolver] = None,
    assignment_group_key_fn: Optional[AssignmentGroupKeyFn] = None,
    route_hint_fn: Optional[RouteHintFn] = None,
    max_owners_per_group: Optional[int] = None,
    owner_strategy: Optional[str] = None,
    owner_cost_model: Optional[str] = None,
    hsdp_column_balance: Optional[bool] = None,
    pack_small_params: Optional[bool] = None,
    tp_buffer_reuse: bool | str = False,
    replicate_broadcast_bucket_mb: float = 0.0,
    muon_forward_unshard: str = "broadcast",
    delay_stage2_to_optimizer: bool = True,
) -> dict[nn.Parameter, int]:
    """Mark parameters for dedicated ownership and register communication hooks.

    Parameters satisfying ``predicate`` are assigned to owner ranks via a
    balanced partition algorithm. Each marked parameter will be automatically
    ignored by subsequent ``fully_shard()`` calls (requires the monkey-patch
    installed automatically by :mod:`dmuon`).

    Communication hooks are registered at the **layer level** (e.g., on
    ``model.layers[i]``), not on individual sub-modules. This minimizes
    CPU launch overhead by batching all broadcasts/reduces per layer.

    **Tensor parallelism** is supported transparently: if a parameter is a
    ``DTensor`` sharded on a mesh dim that is NOT named in ``mesh`` or
    ``replicate_mesh`` (i.e. the TP axis), DMuon will — at optimizer step
    time — gather the full matrix at a designated TP owner, run
    Newton-Schulz locally, and scatter the per-shard update back to
    the TP group.  No TP mesh argument is required; simply pass the DP
    slice of your 3D mesh as ``mesh`` / ``replicate_mesh``, matching the
    FSDP2 convention (``fully_shard(mesh=mesh["replicate","shard"])``).

    Typical 3D (replicate × shard × tp) call order::

        mesh3d = init_device_mesh("cuda", (R, G, T),
                                  mesh_dim_names=("replicate","shard","tp"))

        parallelize_module(model, mesh3d["tp"], {...})   # TP first
        dmuon.dedicate_params(                           # DMuon BEFORE fully_shard
            model,
            mesh=mesh3d["shard"],
            replicate_mesh=mesh3d["replicate"],
            predicate=...,
        )
        fully_shard(model, mesh=mesh3d["replicate","shard"])

    Args:
        model: The model whose parameters to partition.
        mesh: 1D DeviceMesh over the *shard* dimension (a.k.a. ``dp_group``).
            When ``replicate_mesh`` is provided, this becomes the shard axis
            of the HSDP 2D mesh.  Must be a named sub-mesh (constructed via
            ``init_device_mesh(..., mesh_dim_names=...)``) whenever TP is
            present so the TP axis can be inferred by name-set difference.
        predicate: Callable ``(param_name, param) -> bool``. Parameters
            returning True will use dedicated ownership.
        compute_dtype: Optional dtype for communication (e.g., torch.bfloat16).
        reshard_after_forward: If True (default), reshard dedicated params after
            forward and re-broadcast in backward. If False (SHARD_GRAD_OP mode),
            keep params unsharded through forward+backward, eliminating backward
            broadcasts at the cost of higher memory.
        replicate_mesh: Optional 1D DeviceMesh over the *replicate* dimension.
            When provided, dedicate_params accepts a HSDP-style 2D layout; the
            LPT partition then balances globally over ``G·R`` owner slots
            (G = shard size, R = replicate size).
        hook_boundary_predicate: Optional ``(module) -> bool`` selector for the
            hook attachment module. When set, DMuon registers its pre/post
            forward hooks on the **lowest ancestor** of each dedicated param
            where the predicate is True. Use this to align hook boundaries
            with your FSDP2 ``fully_shard`` boundaries — e.g. treat the whole
            ViT as one hook site even though its parameters are distributed
            across ranks. Leave as None to use the built-in ``layers.N`` /
            ``blocks.N`` heuristic (:func:`_find_layer_module`).
        hook_boundary_strict: When True (default) and ``hook_boundary_predicate``
            is given, raise if a dedicated param has no ancestor matching the
            predicate. When False, fall back to the param's direct parent
            module. Strict is recommended to avoid silent per-sub-module hooks.
        hook_boundary_resolver: Optional ``(param_fqn, param) -> module`` hook
            selector for models whose execution boundary is not an ancestor of
            the parameter. It takes precedence over ``hook_boundary_predicate``.
        assignment_group_key_fn: Optional ``(param_fqn, param) -> key`` selector
            used by the owner assignment pass. Use this to align owner packing
            with communication boundaries.
        route_hint_fn: Optional ``(param_fqn, param) -> route`` selector used
            before parameters are replaced by DMuon placeholders.  This is
            required for DMuon-managed sharded base AdamW parameters because
            their initial local shard must be captured while the full tensor
            is still present on the rank.  Supported route hints are
            ``"muon"``, ``"adamw"``, and ``"sharded_adamw"``.
        max_owners_per_group: Optional cap on distinct owner slots used by one
            assignment group.
        owner_strategy: DP/HSDP owner assignment strategy. ``"lpt"`` is the
            production default; ``"round_robin"`` and ``"rank0"`` are intended
            for benchmark ablations. When omitted, ``DMUON_OWNER_STRATEGY`` can
            override the default ``"lpt"`` for benchmark jobs.
        owner_cost_model: LPT cost model. ``"optimizer"`` uses shape-aware
            matrix optimizer cost plus footprint. ``"numel"`` is a diagnostic
            numel-only ablation. When omitted, ``DMUON_OWNER_COST_MODEL`` can
            override the default ``"optimizer"`` for benchmark jobs.
        hsdp_column_balance: Whether HSDP LPT should balance shard-column load
            before per-owner load. Disable only for placement ablations. When
            omitted, ``DMUON_HSDP_COLUMN_BALANCE`` can override the default
            ``True``.
        pack_small_params: Whether same-layer small parameters are merged into
            packed allocation units before owner assignment. Disable only for
            true per-matrix baseline jobs. When omitted,
            ``DMUON_OWNER_PACK_SMALL_PARAMS`` can override the default ``True``.
        tp_buffer_reuse: Optional TP gather/scatter scratch-buffer reuse policy.
            Accepts ``False``/``True`` or ``"gather"``, ``"scatter"``, ``"all"``.
        replicate_broadcast_bucket_mb: Optional HSDP post-step publish bucket
            size in MiB. ``0`` keeps one coalesced publish per hook group.
        muon_forward_unshard: Forward unshard placement for Muon-routed
            parameters. ``"broadcast"`` keeps the existing owner-to-all
            publish path. ``"all_gather"`` keeps the owner-side full-matrix
            update, then scatters the updated tensor into rank-local shards
            after the optimizer step and reconstructs the forward view with an
            FSDP-style all-gather. The all-gather mode is experimental and is
            currently supported only for non-TP Muon parameters.
        delay_stage2_to_optimizer: When True, backward only waits the Stage-1
            shard reduce for buffer lifetime; HSDP Stage-2 replicate reduce is
            waited by per-group optimizer preparation.  This is the default
            because it preserves overlap between late reduce tails and earlier
            optimizer/publish work.

    Returns:
        Assignment dict mapping each dedicated parameter to its DP owner
        coord (``int`` in 1D shard-only mode, ``Tuple[int, int]`` in HSDP
        mode).  TP owner selection is **not** returned — it is intrinsic
        to each DTensor's TP group and resolved at hook-registration time
        (``DedicatedParam.is_tp_owner``).
    """
    # 1. Compute balanced assignment (DP LPT + TP owner pass; the result
    #    carries ``dp_owners`` with the same int / tuple shape as pre-TP
    #    code plus a sparse ``tp_owners`` dict populated only for the
    #    subset of params that are TP-sharded DTensors).
    owner_strategy = _resolve_owner_strategy(owner_strategy)
    owner_cost_model = _resolve_owner_cost_model(owner_cost_model)
    hsdp_column_balance = (
        _env_flag("DMUON_HSDP_COLUMN_BALANCE", True)
        if hsdp_column_balance is None
        else bool(hsdp_column_balance)
    )
    pack_small_params = _resolve_owner_pack_small_params(pack_small_params)
    result = compute_balanced_assignment(
        model,
        mesh,
        predicate,
        replicate_mesh=replicate_mesh,
        owner_strategy=owner_strategy,
        assignment_group_key_fn=assignment_group_key_fn,
        max_owners_per_group=max_owners_per_group,
        owner_cost_model=owner_cost_model,
        hsdp_column_balance=hsdp_column_balance,
        pack_small_params=pack_small_params,
    )
    assignment = result.dp_owners
    if not assignment:
        logger.warning("dedicate_params: no parameters matched the predicate")
        return assignment

    dp_names: set[str] = set()
    if mesh.mesh_dim_names:
        dp_names |= set(mesh.mesh_dim_names)
    if replicate_mesh is not None and replicate_mesh.mesh_dim_names:
        dp_names |= set(replicate_mesh.mesh_dim_names)
    dp_mesh_dim_names = frozenset(dp_names)

    # Normalize every assignment value to a 2D ``(shard, replicate)`` coord.
    # In shard-only mode (``replicate_mesh is None``) partition.py returns
    # ints, which get promoted to ``(int, 0)`` — identical behaviour to
    # pre-Phase-A code paths.
    normalized: dict[nn.Parameter, tuple[int, int]] = {
        param: normalize_owner_rank(owner) for param, owner in assignment.items()
    }

    # Log assignment summary
    shard_size = mesh.size()
    replicate_size = replicate_mesh.size() if replicate_mesh is not None else 1
    total_slots = shard_size * replicate_size
    rank_loads: dict[tuple[int, int], int] = defaultdict(int)
    for param, coord in normalized.items():
        rank_loads[coord] += param.numel()
    loads_list = [
        rank_loads.get((s, r), 0)
        for s in range(shard_size)
        for r in range(replicate_size)
    ]
    max_load = max(loads_list) if loads_list else 0
    min_load = min(loads_list) if loads_list else 0
    imbalance = (max_load - min_load) / max(max_load, 1)
    mode = "HSDP" if replicate_mesh is not None else "shard-only"
    tp_count = len(result.tp_owners)
    tp_tag = f", TP-sharded={tp_count}" if tp_count else ""
    logger.info(
        f"dedicate_params[{mode}]: {len(normalized)} params over {total_slots} "
        f"owner slots (shard={shard_size}, replicate={replicate_size}), "
        f"owner_strategy={owner_strategy}, owner_cost_model={owner_cost_model}, "
        f"pack_small_params={pack_small_params}, "
        f"allocation_units={result.allocation_unit_count}, "
        f"packed_units={result.packed_allocation_unit_count}, "
        f"imbalance={imbalance:.1%}, loads={loads_list}{tp_tag}"
    )

    # 2. Mark parameters
    for param, coord in normalized.items():
        param._dedicated_owner_rank = coord

    # 3. Create shared communication context
    dp_group = mesh.get_group()
    replicate_group = replicate_mesh.get_group() if replicate_mesh is not None else None
    device_type = mesh.device_type
    device = torch.device(device_type, torch.cuda.current_device())
    comm_ctx = DedicatedCommContext(
        device,
        replicate_group=replicate_group,
        tp_buffer_reuse=tp_buffer_reuse,
        replicate_broadcast_bucket_mb=replicate_broadcast_bucket_mb,
    )

    # 4. Group by layer module and create DedicatedParam + DedicatedState
    layer_to_dparams: dict[nn.Module, list[DedicatedParam]] = defaultdict(list)
    param_to_fqn = {param: name for name, param in model.named_parameters()}

    for param, coord in normalized.items():
        parent_modules = _find_parent_modules(model, param)
        parent_module, param_name = parent_modules[0]
        if len(parent_modules) > 1 and reshard_after_forward:
            layer_module = _lowest_common_ancestor_module(
                model, [module for module, _name in parent_modules]
            )
        else:
            layer_module = _resolve_hook_module(
                model,
                param,
                param_fqn=param_to_fqn[param],
                hook_boundary_predicate=hook_boundary_predicate,
                hook_boundary_resolver=hook_boundary_resolver,
                strict=hook_boundary_strict,
            )
        if is_tp_sharded(param, dp_mesh_dim_names) and param not in result.tp_owners:
            raise RuntimeError(
                f"{param_name}: TP-sharded dedicated parameter is missing "
                "from AssignmentResult.tp_owners"
            )
        d_param = DedicatedParam(
            param=param,
            module=parent_module,
            param_name=param_name,
            owner_rank=coord,
            dp_group=dp_group,
            device=device,
            compute_dtype=compute_dtype,
            replicate_group=replicate_group,
            tp_owner_local_rank=(
                result.tp_owners[param] if param in result.tp_owners else 0
            ),
            route_hint=(
                route_hint_fn(param_to_fqn[param], param)
                if route_hint_fn is not None
                else getattr(param, "_dmuon_route_hint", None)
            ),
            muon_forward_unshard=muon_forward_unshard,
        )
        aliases = [(module, name) for module, name in parent_modules[1:]]
        if aliases:
            d_param._alias_modules = aliases
            for alias_module, alias_name in aliases:
                unsafe_setattr_param(alias_module, alias_name, d_param._placeholder)
        layer_to_dparams[layer_module].append(d_param)

    all_states: list[DedicatedState] = []
    module_fqns = {
        id(module): name or "<root>" for name, module in model.named_modules()
    }
    for layer_module, d_params in layer_to_dparams.items():
        group = DedicatedParamGroup(
            d_params,
            comm_ctx,
            delay_stage2_to_optimizer=delay_stage2_to_optimizer,
        )
        group._debug_name = module_fqns.get(id(layer_module), "<unknown>")
        state = DedicatedState(layer_module, group, comm_ctx, reshard_after_forward)
        layer_module._dedicated_state = state
        all_states.append(state)

    # Link states for forward prefetch in actual module order, not owner
    # assignment order.
    _link_forward_prefetch_states(model, all_states, comm_ctx)

    # Store comm_ctx on model for external access (e.g., reset in training loop)
    model._dedicated_comm_ctx = comm_ctx

    return assignment


def dedicate_params_ddp(
    model: nn.Module,
    mesh: DeviceMesh,
    predicate: Callable[[str, nn.Parameter], bool],
    compute_dtype: Optional[torch.dtype] = None,
    hook_boundary_predicate: Optional[Callable[[nn.Module], bool]] = None,
    hook_boundary_strict: bool = True,
    hook_boundary_resolver: Optional[HookBoundaryResolver] = None,
    assignment_group_key_fn: Optional[AssignmentGroupKeyFn] = None,
    max_owners_per_group: Optional[int] = None,
    owner_strategy: Optional[str] = None,
    owner_cost_model: Optional[str] = None,
    pack_small_params: Optional[bool] = None,
) -> dict[nn.Parameter, int]:
    """DDP-path variant of :func:`dedicate_params`.

    Every rank keeps the full parameter live on the module. Ownership
    applies only to (1) who runs Newton-Schulz, (2) the ``dist.reduce``
    destination after backward, and (3) the ``dist.broadcast`` source
    after ``optim.step``.

    Compared with :func:`dedicate_params`, this entry:

    * Accepts a **1D mesh only**. 2D mesh (HSDP) raises — HSDP users
      should continue to use ``dedicate_params`` + ``fully_shard``.
    * Does **not** replace dedicated parameters with 0-size placeholders
      on non-owner ranks; the original ``nn.Parameter`` stays live.
    * Does **not** register forward-broadcast / reshard hooks.
      ``DedicatedState`` still attaches hooks at the layer boundary, but
      ``unshard`` / ``reshard`` degrade to no-op on DDP groups; the real
      work is ``reduce_grads`` after backward + ``post_step_broadcast``
      after ``optim.step``.

    Args:
        model: Model to partition.
        mesh: 1D DeviceMesh over the data-parallel world.
        predicate: Callable ``(param_name, param) -> bool`` selecting
            dedicated parameters.
        compute_dtype: Optional dtype for communication.
        hook_boundary_predicate: Same semantics as in
            :func:`dedicate_params`. Default heuristic uses
            ``layers.N`` / ``blocks.N``.
        hook_boundary_strict: Same as :func:`dedicate_params`.
        hook_boundary_resolver: Same as :func:`dedicate_params`.
        assignment_group_key_fn: Same as :func:`dedicate_params`.
        max_owners_per_group: Same as :func:`dedicate_params`.
        owner_strategy: Same as :func:`dedicate_params`.
        owner_cost_model: Same as :func:`dedicate_params`.
        pack_small_params: Same as :func:`dedicate_params`.

    Returns:
        Assignment dict mapping each dedicated parameter to its owner
        rank (``int`` — 1D shard-only).
    """
    if mesh.ndim != 1:
        raise ValueError(
            f"dedicate_params_ddp requires a 1D mesh; got ndim={mesh.ndim}. "
            "For HSDP use dedicate_params + fully_shard with replicate_mesh."
        )

    owner_strategy = _resolve_owner_strategy(owner_strategy)
    owner_cost_model = _resolve_owner_cost_model(owner_cost_model)
    pack_small_params = _resolve_owner_pack_small_params(pack_small_params)
    result = compute_balanced_assignment(
        model,
        mesh,
        predicate,
        owner_strategy=owner_strategy,
        assignment_group_key_fn=assignment_group_key_fn,
        max_owners_per_group=max_owners_per_group,
        owner_cost_model=owner_cost_model,
        pack_small_params=pack_small_params,
    )
    assignment = result.dp_owners
    if not assignment:
        logger.warning("dedicate_params_ddp: no parameters matched the predicate")
        return assignment
    if result.tp_owners:
        raise NotImplementedError(
            "dedicate_params_ddp does not support TP-sharded DTensor "
            "dedicated parameters; use dedicate_params_ddp_tp(...) for "
            "DDP+TP or dedicate_params(..., mesh=DP, replicate_mesh=...) "
            "for the FSDP2/HSDP TP path."
        )

    normalized: dict[nn.Parameter, tuple[int, int]] = {
        param: normalize_owner_rank(owner) for param, owner in assignment.items()
    }

    shard_size = mesh.size()
    rank_loads: dict[tuple[int, int], int] = defaultdict(int)
    for param, coord in normalized.items():
        rank_loads[coord] += param.numel()
    loads_list = [rank_loads.get((s, 0), 0) for s in range(shard_size)]
    max_load = max(loads_list) if loads_list else 0
    min_load = min(loads_list) if loads_list else 0
    imbalance = (max_load - min_load) / max(max_load, 1)
    logger.info(
        f"dedicate_params_ddp: {len(normalized)} params over {shard_size} ranks, "
        f"owner_strategy={owner_strategy}, owner_cost_model={owner_cost_model}, "
        f"pack_small_params={pack_small_params}, "
        f"allocation_units={result.allocation_unit_count}, "
        f"packed_units={result.packed_allocation_unit_count}, "
        f"imbalance={imbalance:.1%}, loads={loads_list}"
    )

    # Mark params: ``_dedicated_owner_rank`` shared with FSDP2 path so the
    # monkey-patch / ``replicate`` filter still apply; ``_dedicated_mode``
    # discriminates the two paths for checkpoint branching.
    for param, coord in normalized.items():
        param._dedicated_owner_rank = coord
        param._dedicated_mode = "ddp"

    dp_group = mesh.get_group()
    device_type = mesh.device_type
    device = torch.device(device_type, torch.cuda.current_device())
    comm_ctx = DedicatedCommContext(device, replicate_group=None)

    layer_to_dparams: dict[nn.Module, list[DedicatedParamDDP]] = defaultdict(list)
    param_to_fqn = {param: name for name, param in model.named_parameters()}

    for param, coord in normalized.items():
        parent_module, param_name = _find_parent_module(model, param)
        layer_module = _resolve_hook_module(
            model,
            param,
            param_fqn=param_to_fqn[param],
            hook_boundary_predicate=hook_boundary_predicate,
            hook_boundary_resolver=hook_boundary_resolver,
            strict=hook_boundary_strict,
        )

        d_param = DedicatedParamDDP(
            param=param,
            module=parent_module,
            param_name=param_name,
            owner_rank=coord,
            dp_group=dp_group,
            device=device,
            compute_dtype=compute_dtype,
        )
        layer_to_dparams[layer_module].append(d_param)

    all_states: list[DedicatedState] = []
    module_fqns = {
        id(module): name or "<root>" for name, module in model.named_modules()
    }
    for layer_module, d_params in layer_to_dparams.items():
        group = DedicatedParamGroupDDP(d_params, comm_ctx)
        group._debug_name = module_fqns.get(id(layer_module), "<unknown>")
        # ``reshard_after_forward`` is meaningless on the DDP path (no
        # storage to free). Pass False so DedicatedState skips the
        # ``group.reshard()`` call in post-forward. On DDP groups
        # ``reshard`` is a no-op either way — this just avoids an
        # unnecessary attribute touch.
        state = DedicatedState(
            layer_module, group, comm_ctx, reshard_after_forward=False
        )
        layer_module._dedicated_state = state
        all_states.append(state)

    _link_forward_prefetch_states(model, all_states, comm_ctx)

    model._dedicated_comm_ctx = comm_ctx
    return assignment


def dedicate_params_ddp_tp(
    model: nn.Module,
    mesh: DeviceMesh,
    predicate: Callable[[str, nn.Parameter], bool],
    compute_dtype: Optional[torch.dtype] = None,
    hook_boundary_predicate: Optional[Callable[[nn.Module], bool]] = None,
    hook_boundary_strict: bool = True,
    hook_boundary_resolver: Optional[HookBoundaryResolver] = None,
    assignment_group_key_fn: Optional[AssignmentGroupKeyFn] = None,
    max_owners_per_group: Optional[int] = None,
    owner_strategy: Optional[str] = None,
    owner_cost_model: Optional[str] = None,
    pack_small_params: Optional[bool] = None,
    tp_buffer_reuse: bool | str = False,
) -> dict[nn.Parameter, int]:
    """DDP-path variant that allows TP-sharded DTensor dedicated params.

    This API is intentionally separate from :func:`dedicate_params_ddp` so
    pure DDP keeps rejecting DTensor parameters by default.  It is the
    supported ``DDP + TP`` path:

    1. Apply tensor parallelism first.
    2. Call ``dedicate_params_ddp_tp(model, mesh["dp"], ...)``.
    3. Call ``replicate_tp(model, mesh["dp"])`` for non-dedicated params.

    Each TP-sharded dedicated parameter still has a DP owner.  All TP ranks
    in that DP-owner replica gather their local reduced grads to a TP owner
    for full-matrix Newton-Schulz, scatter update shards back, then DDP
    broadcasts those updated shards across the DP mesh.
    """
    if mesh.ndim != 1:
        raise ValueError(
            f"dedicate_params_ddp_tp requires a 1D DP mesh; got ndim={mesh.ndim}."
        )

    owner_strategy = _resolve_owner_strategy(owner_strategy)
    owner_cost_model = _resolve_owner_cost_model(owner_cost_model)
    pack_small_params = _resolve_owner_pack_small_params(pack_small_params)
    result = compute_balanced_assignment(
        model,
        mesh,
        predicate,
        owner_strategy=owner_strategy,
        assignment_group_key_fn=assignment_group_key_fn,
        max_owners_per_group=max_owners_per_group,
        owner_cost_model=owner_cost_model,
        pack_small_params=pack_small_params,
    )
    assignment = result.dp_owners
    if not assignment:
        logger.warning("dedicate_params_ddp_tp: no parameters matched the predicate")
        return assignment

    dp_names = set(mesh.mesh_dim_names or ())
    dp_mesh_dim_names = frozenset(dp_names)
    normalized: dict[nn.Parameter, tuple[int, int]] = {
        param: normalize_owner_rank(owner) for param, owner in assignment.items()
    }

    shard_size = mesh.size()
    rank_loads: dict[tuple[int, int], int] = defaultdict(int)
    for param, coord in normalized.items():
        rank_loads[coord] += param.numel()
    loads_list = [rank_loads.get((s, 0), 0) for s in range(shard_size)]
    max_load = max(loads_list) if loads_list else 0
    min_load = min(loads_list) if loads_list else 0
    imbalance = (max_load - min_load) / max(max_load, 1)
    logger.info(
        f"dedicate_params_ddp_tp: {len(normalized)} params over {shard_size} "
        f"DP ranks, TP-sharded={len(result.tp_owners)}, "
        f"owner_strategy={owner_strategy}, owner_cost_model={owner_cost_model}, "
        f"pack_small_params={pack_small_params}, "
        f"allocation_units={result.allocation_unit_count}, "
        f"packed_units={result.packed_allocation_unit_count}, "
        f"imbalance={imbalance:.1%}, loads={loads_list}"
    )

    for param, coord in normalized.items():
        param._dedicated_owner_rank = coord
        param._dedicated_mode = "ddp_tp"

    dp_group = mesh.get_group()
    device_type = mesh.device_type
    device = torch.device(device_type, torch.cuda.current_device())
    comm_ctx = DedicatedCommContext(
        device,
        replicate_group=None,
        tp_buffer_reuse=tp_buffer_reuse,
    )

    layer_to_dparams: dict[nn.Module, list[DedicatedParamDDP]] = defaultdict(list)
    param_to_fqn = {param: name for name, param in model.named_parameters()}

    for param, coord in normalized.items():
        parent_module, param_name = _find_parent_module(model, param)
        layer_module = _resolve_hook_module(
            model,
            param,
            param_fqn=param_to_fqn[param],
            hook_boundary_predicate=hook_boundary_predicate,
            hook_boundary_resolver=hook_boundary_resolver,
            strict=hook_boundary_strict,
        )
        if is_tp_sharded(param, dp_mesh_dim_names) and param not in result.tp_owners:
            raise RuntimeError(
                f"{param_name}: TP-sharded DDP+TP dedicated parameter is "
                "missing from AssignmentResult.tp_owners"
            )
        d_param = DedicatedParamDDP(
            param=param,
            module=parent_module,
            param_name=param_name,
            owner_rank=coord,
            dp_group=dp_group,
            device=device,
            compute_dtype=compute_dtype,
            tp_owner_local_rank=(
                result.tp_owners[param] if param in result.tp_owners else 0
            ),
        )
        layer_to_dparams[layer_module].append(d_param)

    all_states: list[DedicatedState] = []
    module_fqns = {
        id(module): name or "<root>" for name, module in model.named_modules()
    }
    for layer_module, d_params in layer_to_dparams.items():
        group = DedicatedParamGroupDDP(d_params, comm_ctx)
        group._debug_name = module_fqns.get(id(layer_module), "<unknown>")
        state = DedicatedState(
            layer_module, group, comm_ctx, reshard_after_forward=False
        )
        layer_module._dedicated_state = state
        all_states.append(state)

    _link_forward_prefetch_states(model, all_states, comm_ctx)

    model._dedicated_comm_ctx = comm_ctx
    return assignment
