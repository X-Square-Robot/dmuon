"""Monkey-patch FSDP2 to auto-ignore dedicated parameters."""

_installed = False
_original_fn = None


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

    import torch.distributed.fsdp._fully_shard._fsdp_init as _fsdp_init
    import torch.distributed.fsdp._fully_shard._fully_shard as _fully_shard_mod

    _original_fn = _fsdp_init._get_managed_states

    def _patched_get_managed_states(modules, ignored_params=None):
        if ignored_params is None:
            ignored_params = set()
        for module in modules:
            for _, param in module.named_parameters(recurse=False):
                if hasattr(param, "_dedicated_owner_rank"):
                    ignored_params.add(param)
        return _original_fn(modules, ignored_params)

    # Patch both the definition site and the import site
    _fsdp_init._get_managed_states = _patched_get_managed_states
    _fully_shard_mod._get_managed_states = _patched_get_managed_states
    _installed = True


def uninstall_patch():
    """Uninstall monkey-patch, restoring original FSDP2 behavior."""
    global _installed, _original_fn
    if not _installed:
        return

    import torch.distributed.fsdp._fully_shard._fsdp_init as _fsdp_init
    import torch.distributed.fsdp._fully_shard._fully_shard as _fully_shard_mod

    _fsdp_init._get_managed_states = _original_fn
    _fully_shard_mod._get_managed_states = _original_fn
    _original_fn = None
    _installed = False
