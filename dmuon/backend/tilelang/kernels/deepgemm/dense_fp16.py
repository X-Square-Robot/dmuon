"""DeepGEMM-style fp16 dense GEMM for Ascend 950DT.

Adapted from the official tilelang examples/ascend/deepgemm MNK BF16 kernel
(DeepSeek/tilelang). The scheduling and MAD pipeline are kept intact:
MixedKernel + AscendTileScheduler, explicit L1/L0A/L0B/L0C buffers, K-block
software pipelining, and per-MAD-tile epilogue copies.  Only the input dtype
is changed to FP16; the accumulator remains FP32.
"""

import tilelang
import tilelang.ascend.language as T

from .common import COMPILE_FLAGS, MAX_EPILOGUE_STAGES, physical_shape
from .config import Major
from .copy import make_gm_to_l1_copy
from .epilogue import make_store_output
from .compute import make_compute
from .triangular import CompactTriangularScheduler


def _build_kernel(
    m,
    n,
    k,
    out_dtype,
    config,
    with_alpha=False,
    with_c=False,
    with_diag=False,
    major_a=Major.K,
    major_b=Major.K,
    triangular=False,
    with_mirror=False,
    compact_triangular=False,
    mirror_offdiag_only=False,
    mirror_counter=False,
    mirror_padding=0,
    mirror_simd=False,
):
    BLOCK_M, BLOCK_N, BLOCK_K = config.block_m, config.block_n, config.block_k
    MAD_M, MAD_N, MAD_K = config.mad_m, config.mad_n, config.mad_k
    if mirror_simd and with_mirror and (MAD_N != 128 or MAD_M % 16):
        raise ValueError("SIMD mirror requires a 128-column MAD tile and 16-aligned rows")
    ub_dtype = "float32" if with_alpha else out_dtype
    # With alpha, the FP32 UB slot is viewed as twice as many FP16 elements;
    # apply_alpha packs the converted values in place before the GM store.
    output_m = (
        MAD_M
        if with_c
        else MAD_M * (2 if with_alpha and out_dtype in ("bfloat16", "float16") else 1)
    )
    num_mad_m_tiles = BLOCK_M // MAD_M
    num_mad_n_tiles = BLOCK_N // MAD_N

    copy_a = make_gm_to_l1_copy(major_a, config.l2_ctrl_a)
    copy_b = make_gm_to_l1_copy(major_b, config.l2_ctrl_b)
    compute_k_block = make_compute(config, major_a, major_b)
    store_output = make_store_output(
        config,
        Major.K,
        major_b,
        with_alpha=with_alpha,
        with_c=with_c,
        with_diag=with_diag,
        with_mirror=with_mirror,
        mirror_offdiag_only=mirror_offdiag_only,
        mirror_counter=mirror_counter,
        mirror_simd=mirror_simd,
    )

    def tensor(name, shape, dtype):
        # DMuon NS operands are contiguous.  Static outer strides avoid
        # dynamic-stride ABI handling in every launch while preserving the
        # same physical layout and arbitrary M/N/K extents.
        return T.StridedTensor(shape, (shape[-1], 1), dtype)

    a_type = tensor("a", physical_shape(m, k, major_a), "float16")
    b_type = tensor("b", physical_shape(n, k, major_b), "float16")
    d_type = tensor("d", (m, n), out_dtype)
    c_type = tensor("c", (m, n), "float16" if with_c else out_dtype)
    q_type = tensor("q", (m, n), out_dtype)

    scheduler_type = CompactTriangularScheduler if compact_triangular else T.AscendTileScheduler

    @T.prim_func
    def kernel(
        gm_a: a_type,
        gm_b: b_type,
        gm_c: c_type,
        gm_d: d_type,
        gm_q: q_type,
        alpha: T.float32,
        beta: T.float32,
        diag_add: T.float32,
    ):
        with T.MixedKernel(config.num_blocks, sids=1) as (block_idx, _sub_id):
            scheduler = scheduler_type(
                block_m=BLOCK_M,
                block_n=BLOCK_N,
                num_cores=config.num_blocks,
                shape_m=m,
                shape_n=n,
            )
            # The L1 tile is indexed [MN, K] for MajorK and [K, MN] for
            # MajorMN (copy.py and compute.py both slice l1 along K first
            # for MN-major), so the declaration must follow the major.
            l1a = T.alloc_l1(
                (BLOCK_K, BLOCK_M) if major_a == Major.MN else (BLOCK_M, BLOCK_K), "float16"
            )
            l1b = T.alloc_l1(
                (BLOCK_K, BLOCK_N) if major_b == Major.MN else (BLOCK_N, BLOCK_K), "float16"
            )
            l0a = T.alloc_l0a((MAD_M, MAD_K), "float16")
            l0b = T.alloc_l0b((MAD_N, MAD_K), "float16")
            l0c = T.alloc_l0c((num_mad_m_tiles, num_mad_n_tiles, MAD_M, MAD_N), "float32")
            epilogue_ub = T.alloc_shared((MAD_M, MAD_N), ub_dtype)
            c_ub = T.alloc_shared((MAD_M, MAD_N), "float16") if with_c else None
            mirror_ub = (
                T.alloc_shared((MAD_N, MAD_M + mirror_padding), "float16") if with_mirror else None
            )
            output_ub = (
                T.alloc_shared((MAD_M, MAD_N), "float16")
                if with_c
                else T.view(epilogue_ub, (output_m, MAD_N), dtype=out_dtype)
            )
            T.annotate_buffer_versions(
                {
                    l1a: config.num_l1_stages,
                    l1b: config.num_l1_stages,
                    l0a: config.num_l0_stages,
                    l0b: config.num_l0_stages,
                    epilogue_ub: min(config.num_epilogue_stages, MAX_EPILOGUE_STAGES),
                    **({mirror_ub: (2, "counter") if mirror_counter else 1} if with_mirror else {}),
                }
            )

            T.set_mmad_direction("n")
            scheduler.init(block_idx)
            while scheduler.valid():
                num_k_blocks = T.ceildiv(k, BLOCK_K)
                m_idx, n_idx, actual_m, actual_n = scheduler.tile()
                if not triangular or n_idx <= m_idx:
                    for k_block_idx in range(num_k_blocks):
                        k_idx = k_block_idx * BLOCK_K
                        actual_k = T.min(k - k_idx, BLOCK_K)
                        with T.Task():
                            copy_a(gm_a, l1a, m_idx, k_idx, actual_m, actual_k, 0)
                            copy_b(gm_b, l1b, n_idx, k_idx, actual_n, actual_k, 0)
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
                        mirror_ub,
                        output_ub,
                        gm_d,
                        gm_c,
                        gm_q,
                        m_idx,
                        n_idx,
                        actual_m,
                        actual_n,
                        alpha=alpha,
                        beta=beta,
                        diag_add=diag_add,
                    )
                scheduler.next_block()

    return kernel


@tilelang.jit(execution_backend="tvm_ffi", compile_flags=COMPILE_FLAGS)
def build_fp16_gemm(
    m,
    n,
    k,
    out_dtype,
    config,
    with_alpha=False,
    with_c=False,
    with_diag=False,
    major_a=Major.K,
    major_b=Major.K,
    triangular=False,
    with_mirror=False,
    compact_triangular=False,
    mirror_offdiag_only=False,
    mirror_counter=False,
    mirror_padding=0,
    mirror_simd=False,
):
    return _build_kernel(
        m,
        n,
        k,
        out_dtype,
        config,
        with_alpha,
        with_c,
        with_diag,
        major_a,
        major_b,
        triangular,
        with_mirror,
        compact_triangular,
        mirror_offdiag_only,
        mirror_counter,
        mirror_padding,
        mirror_simd,
    )
