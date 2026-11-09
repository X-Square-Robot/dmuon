from __future__ import annotations

import os

from setuptools import setup


def _extension_config():
    enabled = [
        ("FAST_CLIP", "_fast_clip_cuda", "fast_clip_kernel.cu"),
        ("LAYOUT_COPY", "_layout_copy_cuda", "layout_copy_kernel.cu"),
    ]
    enabled = [entry for entry in enabled if os.environ.get(f"DMUON_BUILD_{entry[0]}", "1") != "0"]
    if not enabled:
        return [], {}
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except Exception as exc:
        print(f"DMuon optional CUDA extensions disabled: {exc}")
        return [], {}

    if CUDA_HOME is None:
        print("DMuon optional CUDA extensions disabled: CUDA_HOME is not available")
        return [], {}

    ext_modules = [
        CUDAExtension(
            name=f"dmuon.{name}",
            sources=[f"dmuon/csrc/{source}"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
        for _, name, source in enabled
    ]
    return ext_modules, {"build_ext": BuildExtension}


ext_modules, cmdclass = _extension_config()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
