"""Shared single-process and real-collective training experiment."""

from contextlib import nullcontext
from copy import deepcopy

import torch
import torch.distributed as dist

import dmuon
from tests.helpers.attention_models import (
    ReferenceOptimizer,
    TinyAttentionStack,
    assert_model_weights,
    assert_owner_gradients,
    assert_owner_states,
    checkpoint_bytes,
    install,
    restore_bytes,
)


def exercise(
    kind,
    mode,
    dtype,
    reshard,
    recompute,
    mesh,
    *,
    device="cpu",
    replicate=None,
    world=1,
    rank=0,
    steps=3,
    native_ns=False,
):
    torch.manual_seed(812)
    model = TinyAttentionStack(kind, dtype=dtype, layout_mode=mode).to(device)
    reference = deepcopy(model)
    ref_optimizer = ReferenceOptimizer(reference, native_ns=native_ns)
    optimizer = install(
        model, mesh, dtype=dtype, reshard=reshard, replicate=replicate, native_ns=native_ns
    )
    models = [(model, optimizer)]
    rtol, atol = (0.04, 0.00015) if dtype == torch.bfloat16 else (0.0003, 0.000003)
    if native_ns:
        rtol, atol = 0.05, 0.0003  # production direct NS internally uses FP16
    for step in range(steps):
        for active, optim in models:
            optim.zero_grad()
        reference.zero_grad(set_to_none=True)
        for micro in range(2):
            torch.manual_seed(1000 + 10 * step + micro + 100 * rank)
            hidden = (torch.randn(1, 6, 16, device=device) * 0.2).requires_grad_()
            expected, aux = reference(hidden, expert=(step + micro) % 2)
            ((expected.square().mean() + aux) / 2).backward()
            for active, _ in models:
                actual_input = hidden.detach().clone().requires_grad_()
                with dmuon.no_sync(active) if micro == 0 else nullcontext():
                    actual, auxiliary = active(
                        actual_input, expert=(step + micro) % 2, recompute=recompute
                    )
                    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                    torch.testing.assert_close(auxiliary, aux, rtol=rtol, atol=atol)
                    ((actual.square().mean() + auxiliary) / 2).backward()
                torch.testing.assert_close(actual_input.grad, hidden.grad, rtol=rtol, atol=atol)
        if world > 1:
            for param in reference.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad.div_(world)
        for active, _ in models:
            dmuon.prepare_muon_grads(active)
            assert_owner_gradients(
                active,
                reference,
                rtol=rtol,
                atol=1e-7 if native_ns or dtype == torch.bfloat16 else 1e-10,
            )
        ref_optimizer.step()
        for active, optim in models:
            optim.step()
            assert_owner_states(active, optim, ref_optimizer, rtol=rtol, atol=atol)
            assert_model_weights(active, reference, rtol=rtol, atol=max(atol, 0.00002))
        if len(models) == 2:
            uninterrupted = dmuon.get_model_state_dict(models[0][0], rank0_only=False)
            resumed_state = dmuon.get_model_state_dict(models[1][0], rank0_only=False)
            for name in uninterrupted:
                torch.testing.assert_close(
                    uninterrupted[name],
                    resumed_state[name],
                    rtol=0,
                    atol=0,
                    msg=f"checkpoint continuation: {name}",
                )
        if step == 0:
            data = checkpoint_bytes(model, optimizer)
            resumed = TinyAttentionStack(kind, dtype=dtype, layout_mode=mode).to(device)
            resumed_optimizer = install(
                resumed,
                mesh,
                dtype=dtype,
                reshard=reshard,
                replicate=replicate,
                native_ns=native_ns,
            )
            restore_bytes(resumed, resumed_optimizer, data)
            assert_model_weights(resumed, reference, rtol=rtol, atol=atol)
            models.append((resumed, resumed_optimizer))
    return {
        "kind": kind,
        "mode": mode,
        "dtype": str(dtype),
        "reshard": reshard,
        "recompute": recompute,
        "steps": steps,
        "microbatches": 2,
        "wrapping_groups": 3,
        "checkpoint_continuation": steps - 1,
        "world": world,
        "ns": "production direct FP16" if native_ns else "FP32 test backend",
    }
