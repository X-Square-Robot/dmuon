"""Output tiling and AIV transforms shared by the BF16 and FP8 kernel families."""

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from .config import Major
from .common import clamp, matrix_region


@T.macro
def apply_alpha(values, output, alpha):
    """Scale in FP32, then pack BF16 in place within each MAD tile's UB slot."""
    with T.SimdVF():
        # Expose the linear offset directly; equivalent row/column indexing can
        # leave division and remainder in generated code (e.g. MAD_N=48).
        flat_values = T.reshape(values, (values.shape[0] * values.shape[1],))
        flat_output = T.reshape(output, (output.shape[0] * output.shape[1],))
        mask = T.simd.pset(32)
        for i in T.serial(values.shape[0] * values.shape[1] // 64):
            value = T.simd.vld(flat_values[i * 64])
            scaled = T.simd.vmuls(value, alpha, mask)
            if output.dtype in ("bfloat16", "float16"):
                converted = T.simd.vcvt(scaled, output.dtype)
                T.simd.vsts(flat_output[i * 64], converted, mask, dist="PK_B32")
            else:
                T.simd.vsts(flat_output[i * 64], scaled, mask)


@T.macro
def apply_axpby(values, c_values, output, alpha, beta):
    """out = rint16(alpha * acc_fp32 + beta * c_fp16) on independent UB."""
    with T.SimdVF():
        flat_values = T.reshape(values, (values.shape[0] * values.shape[1],))
        flat_c = T.reshape(c_values, (c_values.shape[0] * c_values.shape[1],))
        flat_output = T.reshape(output, (output.shape[0] * output.shape[1],))
        mask = T.simd.pset(32)
        for i in T.serial(values.shape[0] * values.shape[1] // 64):
            value = T.simd.vld(flat_values[i * 64])
            c_value = T.simd.vld(flat_c[i * 64], dist="US_B16")
            c_fp32 = T.simd.vcvt(c_value, "float32", part=0)
            scaled = T.simd.vadd(
                T.simd.vmuls(value, alpha, mask),
                T.simd.vmuls(c_fp32, beta, mask),
                mask,
            )
            converted = T.simd.vcvt(scaled, "float16")
            T.simd.vsts(flat_output[i * 64], converted, mask, dist="PK_B32")


@T.macro
def transpose_16rows_128cols(src, dst, row, stride, mask):
    """Bit-preserving SIMD transpose adapted from official DeepGEMM transform_sf.

    Four register-interleave levels replace the scalar SIMT UB scatter.
    This hardware tile primitive handles any matrix shape through MAD tails.
    """
    regs = S.alloc_local((16,), "uint16")
    src_ptr = S.make_ubuf_ptr(T.access_ptr(src[row, 0], "r", extent=2048), "uint16")
    for i in T.Unroll(16, explicit=True):
        regs[i], src_ptr = S.vld(src_ptr, post_inc=128)
    for level in T.Unroll(4, explicit=True):
        offset = 8 >> level
        for pair in T.Unroll(8, explicit=True):
            a = (pair // offset) * (2 * offset) + pair % offset
            b = a + offset
            regs[a], regs[b] = S.vintlv(regs[a], regs[b])
    dst_ptr = S.make_ubuf_ptr(T.access_ptr(dst[0, row], "w", extent=127 * stride + 16), "uint16")
    for i in T.Unroll(16, explicit=True):
        dst_ptr = S.vsstb(
            regs[i], dst_ptr, T.int32(((stride // 16) << 16) | (stride // 2)), mask, update=True
        )


def make_store_output(
    config,
    major_a,
    major_b,
    *,
    blockscaled=False,
    with_alpha=False,
    with_c=False,
    with_diag=False,
    diag_scalar=False,
    with_mirror=False,
    mirror_offdiag_only=False,
    mirror_counter=False,
    mirror_simd=False,
):
    """Drain all MAD tiles in a block; GM stores exclude FP8 load/fixpipe padding."""
    MAD_M, MAD_N = config.mad_m, config.mad_n
    num_mad_m_tiles = config.block_m // MAD_M
    num_mad_n_tiles = config.block_n // MAD_N

    @T.macro
    def store_output(
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
        alpha=1.0,
        beta=1.0,
        diag_add=0.0,
        batch_idx=0,
    ):
        for m_mad_tile_idx in T.Unroll(num_mad_m_tiles, explicit=True):
            m_mad_idx = m_mad_tile_idx * MAD_M
            if m_mad_idx < actual_m:
                eff_mad_m = clamp(actual_m - m_mad_idx, 0, MAD_M)
                load_mad_m = (
                    T.ceildiv(eff_mad_m, 32) * 32
                    if blockscaled and major_a == Major.MN
                    else eff_mad_m
                )
                for n_mad_tile_idx in T.Unroll(num_mad_n_tiles, explicit=True):
                    n_mad_idx = n_mad_tile_idx * MAD_N
                    if n_mad_idx < actual_n:
                        eff_mad_n = clamp(actual_n - n_mad_idx, 0, MAD_N)
                        load_mad_n = (
                            T.ceildiv(eff_mad_n, 32) * 32
                            if blockscaled and major_b == Major.MN
                            else eff_mad_n
                        )
                        # AutoSchedule assigns ownership per loop body; isolate each MAD tile,
                        # including both views of the in-place UB storage.
                        for _ in range(1):
                            T.assume_no_conflict(gm_d)
                            T.copy(
                                l0c[m_mad_tile_idx, n_mad_tile_idx, 0:load_mad_m, 0:load_mad_n],
                                epilogue_ub[0:load_mad_m, 0:load_mad_n],
                                unit_flag_ctrl=3,
                            )
                            if with_c:
                                T.copy(
                                    matrix_region(
                                        gm_c,
                                        m_idx + m_mad_idx,
                                        n_idx + n_mad_idx,
                                        eff_mad_m,
                                        eff_mad_n,
                                        batch_idx,
                                    ),
                                    c_ub[0:eff_mad_m, 0:eff_mad_n],
                                )
                            if with_c:
                                apply_axpby(epilogue_ub, c_ub, output_ub, alpha, beta)
                            elif with_alpha:
                                apply_alpha(epilogue_ub, output_ub, alpha)
                            T.copy(
                                output_ub[0:eff_mad_m, 0:eff_mad_n],
                                matrix_region(
                                    gm_d,
                                    m_idx + m_mad_idx,
                                    n_idx + n_mad_idx,
                                    eff_mad_m,
                                    eff_mad_n,
                                    batch_idx,
                                ),
                                l2_cache_ctrl=config.l2_ctrl_store_cd,
                            )
                            if with_mirror and (
                                m_idx != n_idx
                                if mirror_offdiag_only
                                else m_idx + m_mad_idx != n_idx + n_mad_idx
                            ):
                                if mirror_counter:
                                    with T.Stage(0):
                                        if mirror_simd:
                                            src_bits = T.view(output_ub, dtype="uint16")
                                            dst_bits = T.view(mirror_ub, dtype="uint16")
                                            with T.SimdVF():
                                                full16 = S.pset(16)
                                                for row_block in T.serial(MAD_M // 16):
                                                    transpose_16rows_128cols(
                                                        src_bits,
                                                        dst_bits,
                                                        row_block * 16,
                                                        mirror_ub.shape[1],
                                                        full16,
                                                    )
                                        else:
                                            with T.SimtVF(threads=256):
                                                mirror_frag = T.alloc_fragment(
                                                    (MAD_M, MAD_N), "float16"
                                                )
                                                for i, j in T.Parallel(MAD_M, MAD_N):
                                                    mirror_frag[i, j] = output_ub[i, j]
                                                for i, j in T.Parallel(MAD_M, MAD_N):
                                                    mirror_ub[j, i] = mirror_frag[i, j]
                                        T.copy(
                                            mirror_ub[0:eff_mad_n, 0:eff_mad_m],
                                            matrix_region(
                                                gm_d,
                                                n_idx + n_mad_idx,
                                                m_idx + m_mad_idx,
                                                eff_mad_n,
                                                eff_mad_m,
                                                batch_idx,
                                            ),
                                            l2_cache_ctrl=config.l2_ctrl_store_cd,
                                        )
                                else:
                                    if mirror_simd:
                                        src_bits = T.view(output_ub, dtype="uint16")
                                        dst_bits = T.view(mirror_ub, dtype="uint16")
                                        with T.SimdVF():
                                            full16 = S.pset(16)
                                            for row_block in T.serial(MAD_M // 16):
                                                transpose_16rows_128cols(
                                                    src_bits,
                                                    dst_bits,
                                                    row_block * 16,
                                                    mirror_ub.shape[1],
                                                    full16,
                                                )
                                    else:
                                        with T.SimtVF(threads=256):
                                            mirror_frag = T.alloc_fragment(
                                                (MAD_M, MAD_N), "float16"
                                            )
                                            for i, j in T.Parallel(MAD_M, MAD_N):
                                                mirror_frag[i, j] = output_ub[i, j]
                                            for i, j in T.Parallel(MAD_M, MAD_N):
                                                mirror_ub[j, i] = mirror_frag[i, j]
                                    T.copy(
                                        mirror_ub[0:eff_mad_n, 0:eff_mad_m],
                                        matrix_region(
                                            gm_d,
                                            n_idx + n_mad_idx,
                                            m_idx + m_mad_idx,
                                            eff_mad_n,
                                            eff_mad_m,
                                            batch_idx,
                                        ),
                                        l2_cache_ctrl=config.l2_ctrl_store_cd,
                                    )
                            if with_diag:
                                if diag_scalar:
                                    # SimtVF reserves a 32 KB UB thread stack at
                                    # runtime; with a 240 KB dynamic arena that
                                    # overflows physical UB and the launch is
                                    # rejected.  A tile-guarded scalar loop only
                                    # touches the one diagonal element per row
                                    # and skips off-diagonal tiles entirely.
                                    if (
                                        m_idx + m_mad_idx + MAD_M > n_idx + n_mad_idx
                                        and n_idx + n_mad_idx + MAD_N > m_idx + m_mad_idx
                                    ):
                                        for i in T.serial(MAD_M):
                                            j = m_idx + m_mad_idx + i - (n_idx + n_mad_idx)
                                            if i < eff_mad_m and j >= 0 and j < eff_mad_n:
                                                output_ub[i, j] = T.cast(
                                                    T.cast(output_ub[i, j], "float32") + diag_add,
                                                    "float16",
                                                )
                                else:
                                    with T.SimtVF(threads=256):
                                        for i, j in T.Parallel(MAD_M, MAD_N):
                                            if (
                                                i < eff_mad_m
                                                and j < eff_mad_n
                                                and m_idx + m_mad_idx + i == n_idx + n_mad_idx + j
                                            ):
                                                output_ub[i, j] = T.cast(
                                                    T.cast(output_ub[i, j], "float32") + diag_add,
                                                    "float16",
                                                )
                                T.copy(
                                    output_ub[0:eff_mad_m, 0:eff_mad_n],
                                    matrix_region(
                                        gm_q,
                                        m_idx + m_mad_idx,
                                        n_idx + n_mad_idx,
                                        eff_mad_m,
                                        eff_mad_n,
                                        batch_idx,
                                    ),
                                    l2_cache_ctrl=config.l2_ctrl_store_cd,
                                )
                                if with_mirror and m_idx != n_idx:
                                    T.copy(
                                        mirror_ub[0:eff_mad_n, 0:eff_mad_m],
                                        matrix_region(
                                            gm_q,
                                            n_idx + n_mad_idx,
                                            m_idx + m_mad_idx,
                                            eff_mad_n,
                                            eff_mad_m,
                                            batch_idx,
                                        ),
                                        l2_cache_ctrl=config.l2_ctrl_store_cd,
                                    )

    return store_output
