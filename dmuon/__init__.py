"""DMuon: Dedicated parameter ownership for distributed training with matrix optimizers.

DMuon enables efficient use of Muon and other matrix optimizers with PyTorch DDP,
FSDP2, HSDP, and Tensor Parallelism.  Each matrix parameter is assigned to a
single owner rank that materializes the logical matrix for the optimizer update
and publishes the result back to the active distributed placement.

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

from ._backends.ddp import replicate, replicate_tp
from ._backends.fsdp2 import install_patch
from ._core.comm import DedicatedCommContext
from .api import dedicate_params, dedicate_params_ddp, dedicate_params_ddp_tp
from .checkpoint import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
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
    collect_forward_unshard_profile,
    get_comm_ctx,
    get_dedicated_params,
    get_owned_params,
    no_sync,
    prepare_muon_grads,
    wait_all_post_step_broadcasts,
    wait_all_reduces,
    wait_all_replicate_broadcasts,
)

# Auto-install monkey-patch so fully_shard() skips dedicated params
install_patch()

__all__ = [
    "dedicate_params",
    "dedicate_params_ddp",
    "dedicate_params_ddp_tp",
    "replicate",
    "replicate_tp",
    "install_patch",
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
    "prepare_muon_grads",
    "wait_all_reduces",
    "wait_all_replicate_broadcasts",
    "wait_all_post_step_broadcasts",
    "broadcast_all_updates",
    "broadcast_all_updates_async",
    "collect_forward_unshard_profile",
    "get_model_state_dict",
    "set_model_state_dict",
    "get_optimizer_state_dict",
    "set_optimizer_state_dict",
    "YOU_COEFFICIENTS",
    "POLAR_EXPRESS_COEFFICIENTS",
]
