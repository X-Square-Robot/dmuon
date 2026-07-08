from __future__ import annotations

import os

from setuptools import setup


def _extension_config():
    if os.environ.get("DMUON_BUILD_FAST_CLIP", "1") == "0":
        return [], {}
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except Exception as exc:
        print(f"DMuon fast clip CUDA extension disabled: {exc}")
        return [], {}

    if CUDA_HOME is None:
        print("DMuon fast clip CUDA extension disabled: CUDA_HOME is not available")
        return [], {}

    ext_modules = [
        CUDAExtension(
            name="dmuon._fast_clip_cuda",
            sources=["dmuon/csrc/fast_clip_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ]
    return ext_modules, {"build_ext": BuildExtension}


ext_modules, cmdclass = _extension_config()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
