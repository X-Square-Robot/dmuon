"""Importing DMuon must isolate CUDA-only dependencies and transfer shims."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("hide_cuda", [False, True], ids=["native-runtime", "without-cuda"])
def test_import_keeps_torch_device_apis_and_optional_dependencies_isolated(hide_cuda):
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        import torch

        genuine_cuda = torch.version.cuda is not None and torch.cuda.is_available()
        protected = (
            torch.cuda, torch.cuda.is_available, torch.cuda.current_device,
            torch.cuda.Stream, torch.cuda.Event, torch.Tensor.cuda,
            torch.Tensor.to, torch.Tensor.is_cuda, torch.nn.Module.cuda,
            torch.distributed.init_process_group,
        )
        cuda_optional = ('cutlass', 'quack', 'dmuon.kernels.syrk_quack', 'dmuon.kernels.syrk_sm80')
        # PyTorch's cross-device Dynamo/Triton imports are allowed. Only
        # CUDA-specific dependencies and the global transfer shim are blocked.
        # Internal adapters are shipped by DMuon. On CUDA they must be
        # allowed to load and handle their missing external dependencies.
        blocked = (('cutlass', 'quack') if genuine_cuda else cuda_optional)
        blocked += ('torch_npu.contrib.transfer_to_npu',)
        attempted_optional_imports = []

        class RejectOptionalDependencies(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
                    attempted_optional_imports.append(fullname)
                    raise ImportError('Unavailable optional dependency: ' + fullname)

        sys.meta_path.insert(0, RejectOptionalDependencies())
        import dmuon
        assert protected == (
            torch.cuda, torch.cuda.is_available, torch.cuda.current_device,
            torch.cuda.Stream, torch.cuda.Event, torch.Tensor.cuda,
            torch.Tensor.to, torch.Tensor.is_cuda, torch.nn.Module.cuda,
            torch.distributed.init_process_group,
        )
        assert 'torch_npu.contrib.transfer_to_npu' not in sys.modules
        assert not any(name.startswith('torch_npu.contrib.transfer_to_npu')
                       for name in attempted_optional_imports)
        if not genuine_cuda:
            assert not attempted_optional_imports, attempted_optional_imports
        import dmuon.kernels
        assert dmuon.kernels.__all__ == []
        assert not any(name == 'cutlass' or name.startswith('cutlass.') for name in sys.modules)
        assert dmuon.NewtonSchulz(kernel='cublas').kernel.value == 'cublas'
        if genuine_cuda:
            # CUDA may probe optional kernels, but missing dependencies must
            # leave the package importable and route auto selection to cuBLAS.
            assert dmuon.NewtonSchulz(kernel='auto').kernel.value == 'cublas'
    """)
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root), TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    if hide_cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run([sys.executable, "-c", script], cwd=root, env=env, check=True)
