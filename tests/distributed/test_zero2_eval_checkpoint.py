"""Check post-step eval freshness and same-model ZeRO-2 checkpoint reload.

Run: torchrun --standalone --nproc_per_node=4 tests/distributed/test_zero2_eval_checkpoint.py
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dmuon


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 32, bias=False)
        self.residual = nn.Linear(32, 32, bias=False)
        self.scale = nn.Parameter(torch.ones(32))

    def forward(self, hidden):
        return (self.proj(hidden) + self.residual(hidden)) * self.scale


def canonical_state(model):
    return dmuon.get_model_state_dict(model, rank0_only=False)


def run_fsdp_boundary_case(device, mesh, dtype):
    """Exercise real FSDP boundaries; also runnable on CPU/Gloo with PyTorch 2.10."""
    model = nn.Linear(4, 4, bias=False, device=device)
    with torch.no_grad():
        model.weight.fill_(0.123456)
    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=False,
        mp_policy=MixedPrecisionPolicy(param_dtype=dtype),
    )
    hidden = torch.ones(2, 4, device=device)
    master = model.weight
    group = model._get_fsdp_state()._fsdp_param_group
    model.eval()
    with torch.no_grad():
        model(hidden)
    saved = dmuon.get_model_state_dict(model, cpu_offload=False, rank0_only=False)
    assert saved["weight"].dtype == torch.float32
    torch.testing.assert_close(
        saved["weight"], torch.full_like(saved["weight"], 0.123456), rtol=0, atol=0
    )

    model.unshard(async_op=True)
    assert group._all_gather_result is not None
    dmuon.set_model_state_dict(model, {"weight": torch.full((4, 4), 2.0, device=device)})
    assert group._all_gather_result is None
    # Check the next forward before any getter could incidentally reshard.
    with torch.no_grad():
        actual = model(hidden)
    torch.testing.assert_close(actual, torch.full_like(actual, 8), rtol=0, atol=0)

    # Standalone optimizer save/load must resolve persistent parameter IDs too.
    optimizer = SimpleNamespace(
        _fsdp_params=[master],
        param_groups=[],
        state={master: {"exp_avg": torch.ones_like(master._local_tensor), "step": 3}},
    )
    optim_sd = dmuon.get_optimizer_state_dict(
        model,
        optimizer,
        cpu_offload=False,
        rank0_only=False,
    )
    assert optim_sd["fsdp"]["weight"]["step"] == 3
    assert optim_sd["fsdp"]["weight"]["exp_avg"].shape == (4, 4)
    with torch.no_grad():
        model(hidden)
    optimizer.state.clear()
    dmuon.set_optimizer_state_dict(model, optimizer, optim_sd)
    assert master in optimizer.state
    torch.testing.assert_close(
        optimizer.state[master]["exp_avg"], torch.ones_like(master._local_tensor), rtol=0, atol=0
    )

    model.unshard(async_op=True)
    assert group._all_gather_result is not None
    saved = dmuon.get_model_state_dict(model, cpu_offload=False, rank0_only=True)
    assert group._all_gather_result is None
    if dist.get_rank() == 0:
        torch.testing.assert_close(
            saved["weight"], torch.full_like(saved["weight"], 2), rtol=0, atol=0
        )
        print(f"PASS FSDP pending prefetch, master export, optimizer IDs: {dtype=}")
    else:
        assert saved == {}
    dist.barrier()


def reference(hidden, state, device, dtype):
    hidden = hidden.to(dtype)
    return (
        nn.functional.linear(hidden, state["proj.weight"].to(device=device, dtype=dtype))
        + nn.functional.linear(hidden, state["residual.weight"].to(device=device, dtype=dtype))
    ) * state["scale"].to(device=device, dtype=dtype)


def assert_persistent_state(model, expected):
    actual = canonical_state(model)
    for name, value in expected.items():
        torch.testing.assert_close(
            actual[name],
            value.to(device=actual[name].device, dtype=actual[name].dtype),
            rtol=0,
            atol=0,
            msg=lambda msg: f"Persistent state {name}: {msg}",
        )


def run_case(device, mesh, dtype, replicate_async, matrix_layout):
    torch.manual_seed(42)
    model = Probe().to(device)
    policy = {"param_dtype": dtype, "master_dtype": torch.float32}
    if matrix_layout:
        policy["matrix_layout"] = dmuon.MatrixLayout.whole()
    dmuon.dedicate_params(
        model,
        mesh,
        predicate=lambda name, _param: name == "proj.weight",
        reshard_after_forward=False,
        param_policy={"defaults": policy},
    )
    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=False,
        mp_policy=MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32),
    )
    optimizer = dmuon.Muon(
        model,
        lr=0.1,
        momentum=0.0,
        ns_backend="direct",
        replicate_async=replicate_async,
        first_step_progress_log=False,
    )
    hidden = torch.randn(3, 32, device=device)
    model.eval()
    with torch.no_grad():
        baseline = model(hidden).clone()
    saved = canonical_state(model)
    previous = baseline
    for _ in range(3):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        model(hidden).float().square().mean().backward()
        optimizer.step()
        # No explicit sync, checkpoint load, or materialization before eval.
        model.eval()
        with torch.no_grad():
            actual = model(hidden).clone()
        latest = canonical_state(model)
        expected = reference(hidden, latest, device, dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not torch.equal(actual, previous), "Eval still uses pre-step weights"
        previous = actual
    # The preceding eval left the dedicated BF16/FP32 copies resident.
    # Re-enter forward so symmetric FSDP copies are resident too.
    with torch.no_grad():
        model(hidden)
    dmuon.set_model_state_dict(model, saved)
    with torch.no_grad():
        restored = model(hidden)
    torch.testing.assert_close(restored, baseline, rtol=0, atol=0)
    assert_persistent_state(model, saved)

    # Aliased FSDP input must survive release, even when it targets a dedicated
    # parameter (including a MatrixLayout parent captured before the release).
    with torch.no_grad():
        model(hidden)
        aliased = {
            "proj.weight": model.residual.weight.detach(),
            "residual.weight": model.residual.weight.detach(),
            "scale": model.scale.detach(),
        }
        expected_alias = {name: value.clone() for name, value in aliased.items()}
    dmuon.set_model_state_dict(model, aliased)
    with torch.no_grad():
        actual = model(hidden)
    torch.testing.assert_close(
        actual, reference(hidden, expected_alias, device, dtype), rtol=0, atol=0
    )
    assert_persistent_state(model, expected_alias)

    # No dedicated parameter is loaded in this case, but the live FSDP scale
    # must still be restored into its persistent shard.
    with torch.no_grad():
        model(hidden)
        scale_only = {"scale": torch.full_like(model.scale, 2.0)}
    dmuon.set_model_state_dict(model, scale_only)
    assert_persistent_state(model, scale_only)
    if dist.get_rank() == 0:
        print(
            f"PASS eval, full/aliased/partial restore: {dtype=}, {replicate_async=}, {matrix_layout=}"
        )
    dist.barrier()
    del model, optimizer
    torch.cuda.empty_cache()


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    for dtype in (torch.float32, torch.bfloat16):
        run_fsdp_boundary_case(device, mesh, dtype)
        for replicate_async in (False, True):
            for matrix_layout in (False, True):
                run_case(device, mesh, dtype, replicate_async, matrix_layout)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
