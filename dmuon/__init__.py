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

from .api import dedicate_params
from .checkpoint import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from .comm import DedicatedCommContext
from .optim import (
    POLAR_EXPRESS_COEFFICIENTS,
    YOU_COEFFICIENTS,
    Muon,
    NewtonSchulz,
    get_ns_backend,
    gram_newton_schulz,
    newton_schulz,
)
from .patch import install_patch
from .utils import get_comm_ctx, get_dedicated_params, get_owned_params, no_sync, wait_all_reduces

# Auto-install monkey-patch so fully_shard() skips dedicated params
install_patch()

__all__ = [
    "dedicate_params",
    "Muon",
    "NewtonSchulz",
    "newton_schulz",
    "gram_newton_schulz",
    "get_ns_backend",
    "get_comm_ctx",
    "get_dedicated_params",
    "get_owned_params",
    "DedicatedCommContext",
    "no_sync",
    "wait_all_reduces",
    "get_model_state_dict",
    "set_model_state_dict",
    "get_optimizer_state_dict",
    "set_optimizer_state_dict",
    "YOU_COEFFICIENTS",
    "POLAR_EXPRESS_COEFFICIENTS",
]
