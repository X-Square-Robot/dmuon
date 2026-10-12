"""Importing DMuon must isolate CUDA-only dependencies and transfer shims."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_import_keeps_torch_device_apis_and_optional_dependencies_isolated():
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        import torch

        protected = (
            torch.cuda, torch.cuda.is_available, torch.cuda.current_device,
            torch.cuda.Stream, torch.cuda.Event, torch.Tensor.cuda,
            torch.Tensor.to, torch.Tensor.is_cuda, torch.nn.Module.cuda,
            torch.distributed.init_process_group,
        )
        cuda_optional = ('cutlass', 'quack', 'dmuon.kernels.syrk_quack', 'dmuon.kernels.syrk_sm80')
        # PyTorch's cross-device Dynamo/Triton imports are allowed. Only
        # CUDA-specific dependencies and the global transfer shim are blocked.
        blocked = cuda_optional + ('torch_npu.contrib.transfer_to_npu',)
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
        assert not attempted_optional_imports, attempted_optional_imports
        import dmuon.kernels
        assert dmuon.kernels.__all__ == []
        assert not any(name == 'cutlass' or name.startswith('cutlass.') for name in sys.modules)
        assert dmuon.NewtonSchulz(kernel='cublas').kernel.value == 'cublas'
    """)
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root), TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    subprocess.run([sys.executable, "-c", script], cwd=root, env=env, check=True)
