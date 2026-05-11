"""DMuon: Dedicated parameter ownership for distributed training with matrix optimizers.

DMuon enables efficient use of Muon and other matrix optimizers with PyTorch FSDP2.
Each parameter is assigned to a single owner rank that stores the complete parameter
and performs Newton-Schulz orthogonalization locally — zero extra communication, 1/R compute.

Usage::

    import dmuon
    from torch.distributed.fsdp import fully_shard

    dmuon.dedicate_params(model, dp_mesh, predicate=lambda n, p: "proj" in n)
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)

    optimizer = dmuon.Muon(model, lr=0.02, momentum=0.95)

    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
"""

__version__ = "0.2.0"

from ._backends.ddp import replicate
from ._backends.fsdp2 import install_patch
from ._core.comm import DedicatedCommContext
from ._replicate_profile import replicate_profile_report
from .api import dedicate_params, dedicate_params_ddp
from .checkpoint import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from .diagnostics import format_param_group_summary, summarize_param_groups
from .grad_clip import (
    MuonGradClipStats,
    clip_grad_norm_,
    register_muon_grad_clip_strategy,
)
from .optim import (
    POLAR_EXPRESS_COEFFICIENTS,
    YOU_COEFFICIENTS,
    Muon,
    NewtonSchulz,
    get_backend_status,
    get_ns_backend,
    gram_newton_schulz,
    newton_schulz,
)
from .utils import (
    broadcast_all_updates,
    broadcast_all_updates_async,
    get_comm_ctx,
    get_dedicated_params,
    get_owned_params,
    no_sync,
    reset_replicate_fallback,
    wait_all_post_step_broadcasts,
    wait_all_reduces,
    wait_all_replicate_broadcasts,
)

# Auto-install monkey-patch so fully_shard() skips dedicated params
install_patch()

__all__ = [
    "dedicate_params",
    "dedicate_params_ddp",
    "replicate",
    "Muon",
    "NewtonSchulz",
    "newton_schulz",
    "gram_newton_schulz",
    "get_backend_status",
    "get_ns_backend",
    "get_comm_ctx",
    "get_dedicated_params",
    "get_owned_params",
    "DedicatedCommContext",
    "clip_grad_norm_",
    "register_muon_grad_clip_strategy",
    "MuonGradClipStats",
    "no_sync",
    "wait_all_reduces",
    "wait_all_replicate_broadcasts",
    "wait_all_post_step_broadcasts",
    "broadcast_all_updates",
    "broadcast_all_updates_async",
    "reset_replicate_fallback",
    "replicate_profile_report",
    "get_model_state_dict",
    "set_model_state_dict",
    "get_optimizer_state_dict",
    "set_optimizer_state_dict",
    "summarize_param_groups",
    "format_param_group_summary",
    "YOU_COEFFICIENTS",
    "POLAR_EXPRESS_COEFFICIENTS",
]
