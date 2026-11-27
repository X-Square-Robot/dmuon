"""Owner-local optimizer arithmetic for Ascend. Communication stays in the optimizer."""

from contextlib import contextmanager

import torch
import torch_npu

from dmuon.backend import torch as reference

from . import ns
from .adamw_scalar import adamw_one_scalar
from .direct_ns import direct_newton_schulz_tilelang
from .kernels import elementwise
from .tp import cast_then_scale


_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _dense(tensor):
    return (
        tensor.device.type == "npu"
        and tensor.dtype in _DTYPES
        and tensor.is_contiguous()
        and not tensor.is_neg()
        and torch_npu.get_npu_format(tensor) in (0, 2)
    )


@contextmanager
def momentum_input(grad, state, state_dtype, momentum, nesterov):
    buffer = state.get("momentum_buffer")
    if (
        not _dense(grad)
        or grad.numel() == 0
        or state_dtype not in _DTYPES
        or (
            buffer is not None
            and (not _dense(buffer) or buffer.device != grad.device or buffer.shape != grad.shape)
        )
    ):
        with reference.momentum_input(grad, state, state_dtype, momentum, nesterov) as value:
            yield value
        return

    init = buffer is None
    if init:
        buffer = torch.empty_like(grad, dtype=state_dtype)
    ns_dtype = torch.promote_types(grad.dtype, buffer.dtype) if nesterov else buffer.dtype
    value = ns.optimizer_input(grad.shape, ns_dtype, grad.device)
    partials = None
    if ns._momentum_norm_enabled():
        partials = ns.optimizer_partials(value.numel(), value.device)
        elementwise.momentum_norm(grad, buffer, value, partials, momentum, nesterov, init=init)
    elif init:
        elementwise.momentum_init(grad, buffer, value, momentum, nesterov)
    else:
        elementwise.momentum_fused(grad, buffer, value, momentum, nesterov)
    if init:
        state["momentum_buffer"] = buffer
    with ns.optimizer_borrow(value, partials):
        yield value


def apply_update_(param, update, wd_factor, alpha):
    if not (_dense(param) and _dense(update) and param.device == update.device):
        return reference.apply_update_(param, update, wd_factor, alpha)
    elementwise.apply_update(param, update.view_as(param), wd_factor, alpha)


def scaled_delta(update, shape, device, dtype, alpha):
    if not (_dense(update) and update.device == device and dtype in _DTYPES and update.numel()):
        # A borrowed NS graph output may be reused by the next dispatch.
        return update.reshape(shape).to(device=device, dtype=dtype, copy=True).mul_(alpha)
    result = torch.empty(shape, device=device, dtype=dtype)
    cast_then_scale(update, result, alpha)
    return result


def adamw_bucket_(items, step, lr, betas, weight_decay, eps):
    if not all(_dense(tensor) for item in items for tensor in item):
        return reference.adamw_bucket_(items, step, lr, betas, weight_decay, eps)
    beta1, beta2 = betas
    step_size = lr / (1 - beta1**step)
    bc2_sqrt = (1 - beta2**step) ** 0.5
    wd_factor = 1 - lr * weight_decay if weight_decay > 0 else None
    for param, grad, first, second in items:
        adamw_one_scalar(
            param,
            grad,
            first,
            second,
            wd_factor=wd_factor,
            beta1=beta1,
            beta2=beta2,
            bc2_sqrt=bc2_sqrt,
            eps=eps,
            step_size=step_size,
        )


def gram_newton_schulz(G, steps, eps, coefficients, restart_iterations, deterministic):
    if not _dense(G) or (coefficients is not None and len(coefficients) == 0):
        return reference.gram_newton_schulz(
            G, steps, eps, coefficients, restart_iterations, deterministic
        )
    return ns.gram_newton_schulz_tilelang(
        G, steps, eps, coefficients, restart_iterations, deterministic
    )


def direct_newton_schulz(G, steps, eps, coefficients):
    if not _dense(G):
        return reference.direct_newton_schulz(G, steps, eps, coefficients)
    return direct_newton_schulz_tilelang(G, steps, eps, coefficients)
