"""Native device operations shared by the distributed backends."""

from importlib import import_module
from types import ModuleType

import torch


def get_device_module(device: torch.device | str) -> ModuleType:
    """Return the native CUDA or NPU runtime module.

    Parameters
    ----------
    device : torch.device or str
        Device whose stream, event, and allocator APIs are required.

    Returns
    -------
    types.ModuleType
        ``torch.cuda`` or the registered native ``torch.npu`` module.
    """
    device_type = device.type if isinstance(device, torch.device) else device.split(":", 1)[0]
    if device_type == "cuda":
        return torch.cuda
    if device_type == "npu":
        # torch_npu registers the device and allocator without the global
        # CUDA compatibility patch in torch_npu.contrib.transfer_to_npu.
        if not hasattr(torch, "npu"):
            import_module("torch_npu")
        return torch.npu
    raise ValueError(f"DMuon requires a CUDA or NPU device, got {device!r}")


def current_device(device_type: str) -> torch.device:
    """Return the current device for a native runtime or CPU.

    Parameters
    ----------
    device_type : str
        ``"cuda"``, ``"npu"``, or ``"cpu"``.

    Returns
    -------
    torch.device
        The runtime's current indexed device, or CPU.
    """
    if device_type == "cpu":
        return torch.device("cpu")
    module = get_device_module(device_type)
    return torch.device(device_type, module.current_device())


def cuda_is_available() -> bool:
    """Check that a genuine CUDA build and CUDA device are available.

    Returns
    -------
    bool
        Whether optional CUDA kernels can use the actual CUDA runtime.
    """
    # A CUDA compatibility shim can report NPU availability through CUDA.
    # Optional CUDA kernels require an actual CUDA-enabled PyTorch build.
    return torch.version.cuda is not None and torch.cuda.is_available()


def default_device() -> torch.device:
    """Select an available native CUDA or NPU device, falling back to CPU.

    Returns
    -------
    torch.device
        Current CUDA device, current registered NPU device, or CPU.
    """
    if cuda_is_available():
        return current_device("cuda")
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        return current_device("npu")
    return torch.device("cpu")
