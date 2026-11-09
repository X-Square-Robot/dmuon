"""Differentiable reduced architectures; CPU transport is explicitly simulated."""

import pytest
import torch

from tests.helpers.attention_training import exercise
from tests.helpers.attention_models import (
    TinyAttentionStack,
    install,
    checkpoint_bytes,
    restore_bytes,
)
from tests.unit.test_matrix_owner_integration import Mesh, cpu_transport  # noqa: F401


@pytest.mark.parametrize("kind", ["mla", "dsa", "csa2", "kda", "moba"])
@pytest.mark.parametrize("mode", ["whole", "components", "per_head"])
@pytest.mark.parametrize(
    "reshard,recompute,dtype",
    [
        (True, False, torch.float32),
        (False, False, torch.float32),
        (True, True, torch.float32),
        (True, False, torch.bfloat16),
    ],
)
def test_attention_training_and_checkpoint(cpu_transport, kind, mode, reshard, recompute, dtype):  # noqa: F811
    exercise(kind, mode, dtype, reshard, recompute, Mesh())


@pytest.mark.parametrize("kind", ["mla", "dsa", "csa2", "kda", "moba"])
def test_attention_production_ns(cpu_transport, kind):  # noqa: F811
    exercise(kind, "per_head", torch.float32, True, False, Mesh(), native_ns=True)


def test_checkpoint_preserves_absent_muon_and_adamw_state(cpu_transport):  # noqa: F811
    import io
    import dmuon

    def build():
        model = TinyAttentionStack("mla")
        model.layers[0].unused_adam = torch.nn.Parameter(torch.ones(16))
        return model, install(model, Mesh(), dtype=torch.float32)

    model, optimizer = build()
    output, aux = model(torch.randn(1, 6, 16, requires_grad=True))
    (output.square().mean() + aux).backward()
    optimizer.step()
    data = checkpoint_bytes(model, optimizer)
    saved = torch.load(io.BytesIO(data), weights_only=False)["optimizer"]
    for category in ("dedicated", "dedicated_adamw"):
        assert saved[category]
        assert not any("unused" in name for name in saved[category])
    resumed, resumed_optimizer = build()
    restore_bytes(resumed, resumed_optimizer, data)
    for unit in dmuon.get_dedicated_params(resumed):
        if "unused" in unit.param_name:
            assert id(unit) not in resumed_optimizer.state
