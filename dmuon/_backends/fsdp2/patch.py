"""Monkey-patch FSDP2 to auto-ignore dedicated parameters."""

import contextlib
import inspect

_installed = False
_original_fn = None


def _collect_dedicated_params(modules):
    dedicated_params = set()
    for module in modules:
        for _, param in module.named_parameters(recurse=False):
            if hasattr(param, "_dedicated_owner_rank"):
                dedicated_params.add(param)
    return dedicated_params


def _call_get_managed_states(original_fn, modules, ignored_params=None):
    modules = tuple(modules)
    ignored_params = set() if ignored_params is None else set(ignored_params)
    ignored_params.update(_collect_dedicated_params(modules))

    signature = inspect.signature(original_fn)
    if len(signature.parameters) >= 2:
        return original_fn(modules, ignored_params)

    params, buffers = original_fn(modules)
    if ignored_params:
        params = [param for param in params if param not in ignored_params]
    return params, buffers


@contextlib.contextmanager
def coalescing_manager(*, group, device, async_ops: bool = False):
    """Drop-in replacement for ``dist._coalescing_manager`` that works
    around a latent bug present in torch <= 2.7.

    On torch <= 2.7 the exit branch of ``dist._coalescing_manager``
    unconditionally calls ``group._end_coalescing(device).wait()``
    whenever ``device=`` is passed. For collectives outside its Python
    fast-path list (``all_reduce`` / ``all_gather_into_tensor`` /
    ``reduce_scatter_tensor``) — for example the ``dist.reduce`` calls
    DMuon emits for owner-rank gradient reduction — ``_end_coalescing``
    returns ``None`` and the manager crashes with
    ``AttributeError: 'NoneType' object has no attribute 'wait'``.
    torch 2.8 fixed this by initializing ``work = None`` and gating the
    wait on ``work is not None`` (with the note "Backward compatible
    with backends that don't sync at CPP level").

    This helper drives ``ProcessGroup._start_coalescing`` /
    ``_end_coalescing`` directly (the API the NCCL backend actually
    consumes for ``ncclGroupStart`` / ``ncclGroupEnd`` batching) and
    only waits when the backend hands back a real work handle. It
    wraps the same C++ coalescing API stock ``_coalescing_manager``
    uses internally, so semantics match on the unaffected torch
    versions; the only practical difference is that this version also
    tolerates an empty or non-fast-path coalesce block.
    """
    group._start_coalescing(device)
    try:
        yield
    except BaseException:
        # Close the C++ coalesce group on exception so subsequent
        # collectives are not stuck in coalesce mode.
        try:
            group._end_coalescing(device)
        except BaseException:
            pass
        raise

    work = group._end_coalescing(device)
    if work is not None and not async_ops:
        work.wait()


def install_patch():
    """Install the FSDP2 monkey-patch that makes ``fully_shard`` skip
    dedicated parameters.

    Called automatically on ``import dmuon`` — users do not normally
    invoke this directly. After patching, any subsequent call to
    ``fully_shard()`` filters out parameters previously marked with
    ``_dedicated_owner_rank`` by :func:`dedicate_params`, leaving them
    under DMuon's ownership instead of FSDP2's uniform sharding.

    Safe to call repeatedly (idempotent). The reverse operation is
    :func:`uninstall_patch`.

    Patched function: ``torch.distributed.fsdp._fully_shard._fsdp_init.
    _get_managed_states``.
    """
    global _installed, _original_fn
    if _installed:
        return

    try:
        import torch.distributed.fsdp._fully_shard._fsdp_init as _fsdp_init
        import torch.distributed.fsdp._fully_shard._fully_shard as _fully_shard_mod
    except ModuleNotFoundError:
        # Older local test environments may not ship FSDP2 internals.  Keep
        # import dmuon usable; actual FSDP2 users will still fail when they
        # import/call fully_shard from an unsupported PyTorch build.
        return

    _original_fn = _fsdp_init._get_managed_states

    def _patched_get_managed_states(modules, ignored_params=None):
        return _call_get_managed_states(_original_fn, modules, ignored_params)

    # Patch both the definition site and the import site
    _fsdp_init._get_managed_states = _patched_get_managed_states
    _fully_shard_mod._get_managed_states = _patched_get_managed_states
    _installed = True


def uninstall_patch():
    """Uninstall monkey-patch, restoring original FSDP2 behavior."""
    global _installed, _original_fn
    if not _installed:
        return

    try:
        import torch.distributed.fsdp._fully_shard._fsdp_init as _fsdp_init
        import torch.distributed.fsdp._fully_shard._fully_shard as _fully_shard_mod
    except ModuleNotFoundError:
        return

    _fsdp_init._get_managed_states = _original_fn
    _fully_shard_mod._get_managed_states = _original_fn
    _original_fn = None
    _installed = False
