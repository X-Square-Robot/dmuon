"""CuteDSL kernels for dmuon.

Requires nvidia-cutlass-dsl >= 4.4.2 and apache-tvm-ffi.
Install via: pip install dmuon[syrk]
"""

from dmuon._core.device import cuda_is_available

__all__ = []
if cuda_is_available():
    try:
        from dmuon.kernels.syrk_sm80 import syrk_sm80

        __all__ = ["syrk_sm80"]
    except ImportError:
        __all__ = []
