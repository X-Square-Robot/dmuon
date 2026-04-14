# Extracted from quack/utils.py — only the sqrt function needed by syrk_sm80.
# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

from cutlass import Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def sqrt(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "sqrt.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
        )
    )
