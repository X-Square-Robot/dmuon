from types import SimpleNamespace

import torch
import torch.nn as nn

from dmuon.checkpoint import set_optimizer_state_dict


class FakeDedicatedParam:
    pass


def test_set_optimizer_state_dict_restores_muon_momentum_to_compute_dtype():
    model = nn.Module()
    model.proj = nn.Linear(2, 2, bias=False)

    dp = FakeDedicatedParam()
    dp.module = model.proj
    dp.param_name = "weight"
    dp._compute_dtype = torch.bfloat16
    dp._orig_dtype = torch.float32
    dp._orig_size = torch.Size([2, 2])
    dp.device = torch.device("cpu")
    dp.is_owner = True
    dp.tp_group = None
    dp._owned_data = None
    model.proj._dedicated_state = SimpleNamespace(group=SimpleNamespace(params=[dp]))
    optimizer = SimpleNamespace(state={})
    state_dict = {
        "dedicated": {
            "proj.weight": {"momentum_buffer": torch.ones(2, 2, dtype=torch.float32)}
        }
    }

    set_optimizer_state_dict(model, optimizer, state_dict)

    momentum = optimizer.state[id(dp)]["momentum_buffer"]
    assert momentum.dtype == torch.bfloat16
    assert momentum.shape == (2, 2)
