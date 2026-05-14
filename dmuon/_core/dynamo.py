"""Small helpers for keeping DMuon control-plane code out of TorchDynamo."""

from __future__ import annotations

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

try:
    from torch._dynamo import disable as _torch_dynamo_disable
except Exception:  # pragma: no cover - torch._dynamo is optional across builds
    _torch_dynamo_disable = None


def dynamo_disable(fn: F) -> F:
    """Mark dynamic communication/scheduler code as eager-only when available."""
    if _torch_dynamo_disable is None:
        return fn
    return _torch_dynamo_disable(fn)  # type: ignore[return-value]
