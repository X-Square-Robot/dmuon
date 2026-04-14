"""CuteDSL kernels for dmuon.

Requires nvidia-cutlass-dsl >= 4.4.2 and apache-tvm-ffi.
Install via: pip install dmuon[syrk]
"""

try:
    from dmuon.kernels.syrk_sm80 import syrk_sm80

    __all__ = ["syrk_sm80"]
except ImportError:
    __all__ = []
