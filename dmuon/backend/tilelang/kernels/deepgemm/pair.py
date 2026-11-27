"""Grouped two-problem fused-poly GEMM in one persistent MixedKernel launch.

The Newton-Schulz iteration contains an independent product pair
(Q_{k} = poly(Z, Q) and RZ = poly(R, Z) both depend only on Z_{k}) which
otherwise pays two launches and one inter-kernel boundary per iteration.

Single-scheduler form: one AscendTileScheduler over a doubled M axis, with
a branch on the element offset selecting the problem.  Distinct tensor
type names (``a1``..``d2``) are mandatory — TileLang codegen merges
same-named same-shaped parameters into one pointer.

Coordinate contract: ``scheduler.tile()`` returns ELEMENT offsets along
M/N.  The problem boundary is at ``M`` (the per-problem height in
elements), and the second problem's row offset is ``m_idx - M``.  Earlier
designs subtracted tile counts from element offsets, producing reads
one row off; the symptom (zeroed second problem, collapsed extent
``rows - m_idx + 1``) was misdiagnosed as a compiler bug.
"""

import tilelang
import tilelang.ascend.language as T

from .common import COMPILE_FLAGS, MAX_EPILOGUE_STAGES
from .config import Major
from .copy import make_gm_to_l1_copy
from .epilogue import make_store_output
from .compute import make_compute


def _build_pair_kernel(m, n, k, out_dtype, config, with_alpha=True, with_c=True):
    BLOCK_M, BLOCK_N, BLOCK_K = config.block_m, config.block_n, config.block_k
    MAD_M, MAD_N, MAD_K = config.mad_m, config.mad_n, config.mad_k
    ub_dtype = "float32" if with_alpha else out_dtype
    output_m = MAD_M * (2 if with_alpha and out_dtype in ("bfloat16", "float16") else 1)
    num_mad_m_tiles = BLOCK_M // MAD_M
    num_mad_n_tiles = BLOCK_N // MAD_N
    num_k_blocks = (k + BLOCK_K - 1) // BLOCK_K

    copy_a = make_gm_to_l1_copy(Major.K, config.l2_ctrl_a)
    copy_b = make_gm_to_l1_copy(Major.K, config.l2_ctrl_b)
    compute_k_block = make_compute(config, Major.K, Major.K)
    store_output = make_store_output(config, Major.K, Major.K, with_alpha=with_alpha, with_c=with_c)

    def tensor(name, shape, dtype):
        return T.StridedTensor(shape, (shape[-1], 1), dtype)

    a1_type = tensor("a1", (m, k), "float16")
    b1_type = tensor("b1", (n, k), "float16")
    c1_type = tensor("c1", (m, n), "float16" if with_c else out_dtype)
    d1_type = tensor("d1", (m, n), out_dtype)
    a2_type = tensor("a2", (m, k), "float16")
    b2_type = tensor("b2", (n, k), "float16")
    c2_type = tensor("c2", (m, n), "float16" if with_c else out_dtype)
    d2_type = tensor("d2", (m, n), out_dtype)

    @T.prim_func
    def kernel(
        gm_a1: a1_type,
        gm_b1: b1_type,
        gm_c1: c1_type,
        gm_d1: d1_type,
        gm_a2: a2_type,
        gm_b2: b2_type,
        gm_c2: c2_type,
        gm_d2: d2_type,
        alpha1: T.float32,
        beta1: T.float32,
        alpha2: T.float32,
        beta2: T.float32,
    ):
        with T.MixedKernel(config.num_blocks, sids=1) as (block_idx, _sub_id):
            l1a = T.alloc_l1((BLOCK_M, BLOCK_K), "float16")
            l1b = T.alloc_l1((BLOCK_N, BLOCK_K), "float16")
            l0a = T.alloc_l0a((MAD_M, MAD_K), "float16")
            l0b = T.alloc_l0b((MAD_N, MAD_K), "float16")
            l0c = T.alloc_l0c((num_mad_m_tiles, num_mad_n_tiles, MAD_M, MAD_N), "float32")
            epilogue_ub = T.alloc_shared((MAD_M, MAD_N), ub_dtype)
            c_ub = T.alloc_shared((MAD_M, MAD_N), "float16") if with_c else None
            output_ub = T.view(epilogue_ub, (output_m, MAD_N), dtype=out_dtype)
            T.annotate_buffer_versions(
                {
                    l1a: config.num_l1_stages,
                    l1b: config.num_l1_stages,
                    l0a: config.num_l0_stages,
                    l0b: config.num_l0_stages,
                    epilogue_ub: min(config.num_epilogue_stages, MAX_EPILOGUE_STAGES),
                }
            )
            T.set_mmad_direction("n")
            scheduler = T.AscendTileScheduler(
                block_m=BLOCK_M,
                block_n=BLOCK_N,
                num_cores=config.num_blocks,
                shape_m=2 * m,
                shape_n=n,
            )
            scheduler.init(block_idx)
            while scheduler.valid():
                m_idx, n_idx, actual_m, actual_n = scheduler.tile()
                if m_idx < m:
                    for k_block_idx in range(num_k_blocks):
                        k_idx = k_block_idx * BLOCK_K
                        actual_k = T.min(k - k_idx, BLOCK_K)
                        with T.Task():
                            copy_a(gm_a1, l1a, m_idx, k_idx, actual_m, actual_k, 0)
                            copy_b(gm_b1, l1b, n_idx, k_idx, actual_n, actual_k, 0)
                        compute_k_block(
                            l1a,
                            l1b,
                            l0a,
                            l0b,
                            l0c,
                            k_block_idx,
                            num_k_blocks,
                            actual_m,
                            actual_n,
                            actual_k,
                        )
                    store_output(
                        l0c,
                        epilogue_ub,
                        c_ub,
                        None,
                        output_ub,
                        gm_d1,
                        gm_c1,
                        gm_d1,
                        m_idx,
                        n_idx,
                        actual_m,
                        actual_n,
                        alpha=alpha1,
                        beta=beta1,
                    )
                else:
                    m2 = m_idx - m
                    for k_block_idx in range(num_k_blocks):
                        k_idx = k_block_idx * BLOCK_K
                        actual_k = T.min(k - k_idx, BLOCK_K)
                        with T.Task():
                            copy_a(gm_a2, l1a, m2, k_idx, actual_m, actual_k, 0)
                            copy_b(gm_b2, l1b, n_idx, k_idx, actual_n, actual_k, 0)
                        compute_k_block(
                            l1a,
                            l1b,
                            l0a,
                            l0b,
                            l0c,
                            k_block_idx,
                            num_k_blocks,
                            actual_m,
                            actual_n,
                            actual_k,
                        )
                    store_output(
                        l0c,
                        epilogue_ub,
                        c_ub,
                        None,
                        output_ub,
                        gm_d2,
                        gm_c2,
                        gm_d2,
                        m2,
                        n_idx,
                        actual_m,
                        actual_n,
                        alpha=alpha2,
                        beta=beta2,
                    )
                scheduler.next_block()

    return kernel


@tilelang.jit(execution_backend="tvm_ffi", compile_flags=COMPILE_FLAGS)
def build_fp16_gemm_pair(m, n, k, out_dtype, config, with_alpha=True, with_c=True):
    return _build_pair_kernel(m, n, k, out_dtype, config, with_alpha, with_c)
