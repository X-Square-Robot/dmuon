"""DMuon: Dedicated parameter ownership for distributed training with matrix optimizers.

DMuon enables efficient use of Muon and other matrix optimizers with PyTorch FSDP2.
Each parameter is assigned to a single owner rank that stores the complete parameter
and performs Newton-Schulz orthogonalization locally — zero extra communication, 1/R compute.

Usage::

    from dmuon import dedicate_params
    from torch.distributed.fsdp import fully_shard

    dedicate_params(model, dp_mesh, predicate=lambda n, p: "proj" in n)
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)
"""

__version__ = "0.1.0"

from .api import dedicate_params
from .patch import install_patch
from .utils import get_dedicated_params, get_owned_params

# Auto-install monkey-patch so fully_shard() skips dedicated params
install_patch()

__all__ = ["dedicate_params", "get_dedicated_params", "get_owned_params"]
