from types import SimpleNamespace

import torch.nn as nn

from dmuon.optim.muon import Muon


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.block = nn.Module()
        self.block.proj = nn.Linear(4, 4, bias=False)
        self.head = nn.Linear(4, 2, bias=False)


def test_first_step_fqn_mapping_uses_parent_module_prefix() -> None:
    model = TinyModel()
    dedicated_params = [
        SimpleNamespace(module=model.block.proj, param_name="weight"),
        SimpleNamespace(module=model.embed, param_name="weight"),
        SimpleNamespace(module=model.head, param_name="weight"),
    ]

    names = Muon._compute_dedicated_param_fqns(model, dedicated_params)

    assert names[id(dedicated_params[0])] == "block.proj.weight"
    assert names[id(dedicated_params[1])] == "embed.weight"
    assert names[id(dedicated_params[2])] == "head.weight"


def test_first_step_fqn_mapping_falls_back_to_local_param_name() -> None:
    model = TinyModel()
    detached_module = nn.Linear(4, 4, bias=False)
    dedicated_param = SimpleNamespace(module=detached_module, param_name="weight")

    names = Muon._compute_dedicated_param_fqns(model, [dedicated_param])

    assert names[id(dedicated_param)] == "weight"
