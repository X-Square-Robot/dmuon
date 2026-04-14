# Copyright (c) 2025-2026, Tri Dao.
# SM80 symmetric GEMM kernel: D = alpha * A @ B^T + beta * C
# When B=A this is SYRK (D = alpha * A @ A^T + beta * C).
# Self-contained implementation using cp.async + warp-level MMA (no TMA).
# Only launches lower-triangle tiles + mirror write for symmetric output.
# Reference: fused-muon/csrc/syrk_common.cuh

from typing import Optional

from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.utils import LayoutEnum
from cutlass import Int32, Float32, const_expr

from dmuon.kernels.compile_utils import make_fake_tensor as fake_tensor
from dmuon.kernels.cute_dsl_utils import torch2cute_dtype_map
from dmuon.kernels.cache_utils import jit_cache
from dmuon.kernels import sm80_utils
from dmuon.kernels import dsl_utils as utils


# ── Tile / pipeline constants ────────────────────────────────────────────────

MMA_INST_MNK = (16, 8, 16)
NUM_THREADS = 128  # 4 warps

# Default configuration
DEFAULT_TILE_M = 128
DEFAULT_TILE_K = 32
DEFAULT_NUM_STAGES = 4

# All valid (tile_m, tile_k, num_stages) configs for autotuning
SYRK_SM80_CONFIGS = [
    (64, 32, 4),
    (64, 32, 3),
    (64, 64, 3),
    (128, 32, 4),
    (128, 32, 3),
    (128, 64, 3),
]


def _atom_layout_for_tile(tile_m):
    """Atom layout and permutation for the given tile_m.

    Always uses 4 warps (128 threads). The tiled_mma covers 32×32 per step,
    with the fragment repeating as needed for the full tile_m × tile_m output.
    """
    # 4 warps in 2×2 layout: each warp handles 16×8 MMA atom
    # perm (32, 32, 16): extends coverage to 32×32 per tiled_mma step
    return (2, 2, 1), (32, 32, 16)


class SyrkSm80:
    """SM80 symmetric rank-k update kernel.

    Uses cp.async pipeline for G2S, warp-level MMA (16×8×16) for compute,
    and dual-write mirror epilogue for symmetric output.
    """

    arch = 80

    def __init__(
        self,
        acc_dtype,
        a_dtype,
        has_C,
        alpha_mode,
        beta_mode,
        diag_add_mode=0,
        tile_m=DEFAULT_TILE_M,
        tile_k=DEFAULT_TILE_K,
        num_stages=DEFAULT_NUM_STAGES,
    ):
        self.acc_dtype = acc_dtype
        self.a_dtype = a_dtype
        self.has_C = has_C
        self.alpha_mode = alpha_mode
        self.beta_mode = beta_mode
        self.has_diag_add = diag_add_mode == 1
        self.tile_m = tile_m
        self.tile_k = tile_k
        self.num_stages = num_stages
        self.threads_per_cta = NUM_THREADS
        self.atom_layout_mnk, self.perm_mnk = _atom_layout_for_tile(tile_m)
        self.padded_n = tile_m + 8

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # (m, k, l)  K-contiguous
        mB: cute.Tensor,  # (m, k, l)  K-contiguous (same as A for SYRK)
        mD: cute.Tensor,  # (m, m, l)  N-contiguous
        mC: Optional[cute.Tensor],  # (m, m, l) or None
        alpha: Optional[Float32],
        beta: Optional[Float32],
        diag_add: Optional[Float32],
        stream,
    ):
        a_dtype = mA.element_type

        # ── Tiled MMA ────────────────────────────────────────────────────
        mma_op = warp.MmaF16BF16Op(a_dtype, self.acc_dtype, MMA_INST_MNK)
        atom_layout = cute.make_layout(self.atom_layout_mnk)
        tiled_mma = cute.make_tiled_mma(mma_op, atom_layout, permutation_mnk=self.perm_mnk)

        # ── G2S tiled copy (cp.async 128-bit) ────────────────────────────
        copy_op = cpasync.CopyG2SOp()
        num_copy_elems = 128 // a_dtype.width  # 8 for bf16/fp16
        copy_atom = cute.make_copy_atom(copy_op, a_dtype, num_bits_per_copy=128)
        thr_layout = cute.make_ordered_layout((32, 4), order=(1, 0))
        val_layout = cute.make_layout((1, num_copy_elems))
        tiled_copy_g2s = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        # ── S2R ldmatrix atoms ───────────────────────────────────────────
        ldmatrix_atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), a_dtype)
        smem_tiled_copy_A = cute.make_tiled_copy_A(ldmatrix_atom, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(ldmatrix_atom, tiled_mma)

        # ── SMEM layouts (swizzled, staged) ──────────────────────────────
        # Reuse SM90 helper to create bank-conflict-free swizzled smem atom
        # (same swizzle pattern works for SM80 ldmatrix)
        raw_atom = sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, a_dtype, self.tile_k)
        smem_atom = warpgroup.make_smem_layout_atom(raw_atom, a_dtype)
        smem_layout_A = cute.tile_to_shape(
            smem_atom, (self.tile_m, self.tile_k, self.num_stages), order=(0, 1, 2)
        )
        smem_layout_B = cute.tile_to_shape(
            smem_atom, (self.tile_m, self.tile_k, self.num_stages), order=(0, 1, 2)
        )

        # ── Grid ─────────────────────────────────────────────────────────
        M = mA.shape[0]
        L = mA.shape[2]
        Mt = cute.ceil_div(M, self.tile_m)
        num_tri_tiles = Mt * (Mt + 1) // 2

        # ── Shared storage ───────────────────────────────────────────────
        # sA must fit both mainloop (tile_m × tile_k × stages) AND mirror
        # transpose buffer (tile_m × padded_n). Reuse same memory since they
        # are used at different phases.
        mainloop_cosize_A = cute.cosize(smem_layout_A)
        mirror_buf_size = self.tile_m * self.padded_n
        sA_alloc = max(mainloop_cosize_A, mirror_buf_size)

        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[cute.struct.MemRange[a_dtype, sA_alloc], 128]
            sB: cute.struct.Align[cute.struct.MemRange[a_dtype, cute.cosize(smem_layout_B)], 128]

        self.shared_storage = SharedStorage

        # ── Launch ───────────────────────────────────────────────────────
        self.kernel(
            tiled_mma,
            tiled_copy_g2s,
            smem_tiled_copy_A,
            smem_tiled_copy_B,
            mA,
            mB,
            mD,
            mC,
            alpha,
            beta,
            diag_add,
            smem_layout_A,
            smem_layout_B,
        ).launch(
            grid=(num_tri_tiles, L, 1),
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_copy_g2s: cute.TiledCopy,
        smem_tiled_copy_A: cute.TiledCopy,
        smem_tiled_copy_B: cute.TiledCopy,
        mA_mkl: cute.Tensor,
        mB_mkl: cute.Tensor,  # same as mA for SYRK, different for general sym GEMM
        mD_mnl: cute.Tensor,
        mC_mnl: Optional[cute.Tensor],
        alpha: Optional[Float32],
        beta: Optional[Float32],
        diag_add: Optional[Float32],
        smem_layout_A: cute.ComposedLayout,
        smem_layout_B: cute.ComposedLayout,
    ):
        has_C = const_expr(mC_mnl is not None)
        has_alpha = const_expr(alpha is not None)
        has_diag_add = const_expr(diag_add is not None)
        has_beta = const_expr(beta is not None)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        # ── Lower-triangle tile index ────────────────────────────────────
        i_tile = Int32(
            (utils.sqrt(Float32(1.0) + Float32(8.0) * Float32(bidx)) - Float32(1.0)) * Float32(0.5)
        )
        if i_tile * (i_tile + 1) // 2 > bidx:
            i_tile = i_tile - 1
        if (i_tile + 1) * (i_tile + 2) // 2 <= bidx:
            i_tile = i_tile + 1
        j_tile = bidx - i_tile * (i_tile + 1) // 2

        batch_idx = bidy
        M = mA_mkl.shape[0]

        # ── Batch slice → 2D tensors ────────────────────────────────────
        mA_mk = mA_mkl[None, None, batch_idx]  # (M, K)
        mB_mk = mB_mkl[None, None, batch_idx]  # (M, K) — same as mA for SYRK
        mD_mn = mD_mnl[None, None, batch_idx]  # (M, M)

        cta_mk = (self.tile_m, self.tile_k)
        cta_mn = (self.tile_m, self.tile_m)

        gA = cute.local_tile(mA_mk, cta_mk, (i_tile, None))  # (TM, TK, RestK)
        gB = cute.local_tile(mB_mk, cta_mk, (j_tile, None))  # (TM, TK, RestK)
        gD = cute.local_tile(mD_mn, cta_mn, (i_tile, j_tile))  # (TM, TM)

        # ── Shared memory ────────────────────────────────────────────────
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(smem_layout_A.outer, swizzle=smem_layout_A.inner)
        sB = storage.sB.get_tensor(smem_layout_B.outer, swizzle=smem_layout_B.inner)

        # ── G2S partitions ───────────────────────────────────────────────
        thr_g2s = tiled_copy_g2s.get_slice(tidx)
        tAgA = thr_g2s.partition_S(gA)
        tAsA = thr_g2s.partition_D(sA)
        tBgB = thr_g2s.partition_S(gB)
        tBsB = thr_g2s.partition_D(sB)

        k_count = cute.size(tAgA, mode=[3])

        # ── S2R partitions ───────────────────────────────────────────────
        thr_s2r_A = smem_tiled_copy_A.get_slice(tidx)
        thr_s2r_B = smem_tiled_copy_B.get_slice(tidx)
        tCsA_copy = thr_s2r_A.partition_S(sA)
        tCsB_copy = thr_s2r_B.partition_S(sB)

        # ── MMA fragments ────────────────────────────────────────────────
        thr_mma = tiled_mma.get_slice(tidx)
        acc, _, _, tCrA, tCrB = sm80_utils.partition_fragment_ABC(
            thr_mma, (self.tile_m, self.tile_m, self.tile_k), sA, sB
        )
        tCrA_copy = smem_tiled_copy_A.retile(tCrA)
        tCrB_copy = smem_tiled_copy_B.retile(tCrB)
        K_BLK = cute.size(tCrA, mode=[2])

        acc.fill(0.0)

        # ══════════════════════════════════════════════════════════════════
        #  Mainloop: cp.async pipeline
        # ══════════════════════════════════════════════════════════════════
        k_next = Int32(0)

        # Prologue: fill stages 0..num_stages-2
        for p in cutlass.range_constexpr(self.num_stages - 1):
            cute.copy(tiled_copy_g2s, tAgA[None, None, None, k_next], tAsA[None, None, None, p])
            cute.copy(tiled_copy_g2s, tBgB[None, None, None, k_next], tBsB[None, None, None, p])
            cute.arch.cp_async_commit_group()
            k_count = k_count - 1
            if k_count > 0:
                k_next = k_next + 1

        pipe_r = Int32(0)
        pipe_w = Int32(self.num_stages - 1)

        # Pre-load first k-block of stage 0 into registers
        tCsA_p = tCsA_copy[None, None, None, pipe_r]
        tCsB_p = tCsB_copy[None, None, None, pipe_r]
        if const_expr(K_BLK > 1):
            cute.arch.cp_async_wait_group(self.num_stages - 2)
            cute.arch.sync_threads()
            cute.copy(smem_tiled_copy_A, tCsA_p[None, None, 0], tCrA_copy[None, None, 0])
            cute.copy(smem_tiled_copy_B, tCsB_p[None, None, 0], tCrB_copy[None, None, 0])

        # Steady-state loop
        while k_count > -(self.num_stages - 1):
            for kb in cutlass.range_constexpr(K_BLK):
                if const_expr(kb == K_BLK - 1):
                    # Switch to next stage in read pipe
                    tCsA_p = tCsA_copy[None, None, None, pipe_r]
                    tCsB_p = tCsB_copy[None, None, None, pipe_r]
                    cute.arch.cp_async_wait_group(self.num_stages - 2)
                    cute.arch.sync_threads()
                # Pre-fetch next k-block into registers
                kn = (kb + 1) % K_BLK
                cute.copy(smem_tiled_copy_A, tCsA_p[None, None, kn], tCrA_copy[None, None, kn])
                cute.copy(smem_tiled_copy_B, tCsB_p[None, None, kn], tCrB_copy[None, None, kn])
                if const_expr(kb == 0):
                    # Issue next G2S cp.async
                    cute.copy(
                        tiled_copy_g2s,
                        tAgA[None, None, None, k_next],
                        tAsA[None, None, None, pipe_w],
                    )
                    cute.copy(
                        tiled_copy_g2s,
                        tBgB[None, None, None, k_next],
                        tBsB[None, None, None, pipe_w],
                    )
                    cute.arch.cp_async_commit_group()
                    k_count = k_count - 1
                    if k_count > 0:
                        k_next = k_next + 1
                    pipe_w = pipe_r
                    pipe_r = pipe_r + 1
                    if pipe_r == self.num_stages:
                        pipe_r = Int32(0)
                # MMA
                cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)

        # ══════════════════════════════════════════════════════════════════
        #  Epilogue
        # ══════════════════════════════════════════════════════════════════
        trs = i_tile * self.tile_m
        tcs = j_tile * self.tile_m
        is_diagonal = i_tile == j_tile

        # Per-thread coordinate tensor
        cD = cute.make_identity_tensor((self.tile_m, self.tile_m))
        tCcD = thr_mma.partition_C(cD)

        # Alpha/beta scaling + optional C
        if const_expr(has_C):
            mC_mn = mC_mnl[None, None, batch_idx]
            gC = cute.local_tile(mC_mn, cta_mn, (i_tile, j_tile))
            tCgC_load = thr_mma.partition_C(gC)
            for i in cutlass.range_constexpr(cute.size(acc)):
                a_val = acc[i]
                if const_expr(has_alpha):
                    a_val = alpha * a_val
                c_val = Float32(tCgC_load[i])
                if const_expr(has_beta):
                    acc[i] = a_val + beta * c_val
                else:
                    acc[i] = a_val + c_val
        else:
            if const_expr(has_alpha):
                for i in cutlass.range_constexpr(cute.size(acc)):
                    acc[i] = alpha * acc[i]

        # Diagonal add: acc[i] += diag_add for diagonal elements (only on diagonal tiles)
        if const_expr(has_diag_add):
            if is_diagonal:
                for i in cutlass.range_constexpr(cute.size(acc)):
                    if Int32(tCcD[i][0]) == Int32(tCcD[i][1]):
                        acc[i] = acc[i] + diag_add

        # Convert fp32 acc → output dtype
        d_dtype = mD_mnl.element_type
        out = cute.make_rmem_tensor(acc.shape, d_dtype)
        for i in cutlass.range_constexpr(cute.size(acc)):
            out[i] = d_dtype(acc[i])

        # ── Lower-triangle store ─────────────────────────────────────────
        tCgD = thr_mma.partition_C(gD)
        cute.autovec_copy(out, tCgD)

        # ── Mirror (dual-write upper triangle) ───────────────────────────
        # Reuse sA smem as transpose buffer (mainloop is done)
        mirror_smem = storage.sA.get_tensor((self.tile_m, self.padded_n))

        if is_diagonal:
            # Diagonal tiles: element-wise mirror (few tiles, not perf-critical)
            for i in cutlass.range_constexpr(cute.size(out)):
                gr = trs + Int32(tCcD[i][0])
                gc = tcs + Int32(tCcD[i][1])
                if gr > gc:
                    mD_mn[gc, gr] = out[i]
        else:
            # Off-diagonal tiles: smem transpose + coalesced global write
            # Phase 1: scatter MMA output into smem (row-major with padding)
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(cute.size(out)):
                lr = Int32(tCcD[i][0])
                lc = Int32(tCcD[i][1])
                mirror_smem[lr, lc] = out[i]
            cute.arch.sync_threads()
            # Phase 2: cooperative coalesced store — each thread writes one
            # column (trs+tidx) across all rows (tcs..tcs+tile_m-1)
            if tidx < self.tile_m:
                for mr in cutlass.range(self.tile_m):
                    mD_mn[tcs + mr, trs + tidx] = mirror_smem[tidx, mr]


# ── Compilation ──────────────────────────────────────────────────────────────


@jit_cache
def _compile_syrk_sm80(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    alpha_mode,
    beta_mode,
    diag_add_mode=0,
    tile_m=DEFAULT_TILE_M,
    tile_k=DEFAULT_TILE_K,
    num_stages=DEFAULT_NUM_STAGES,
):
    m, k, l = cute.sym_int(), cute.sym_int(), cute.sym_int()
    div_a = 128 // a_dtype.width
    div_b = 128 // b_dtype.width
    div_d = 128 // d_dtype.width

    mA = fake_tensor(a_dtype, (m, k, l), leading_dim=1, divisibility=div_a)
    mB = fake_tensor(b_dtype, (m, k, l), leading_dim=1, divisibility=div_b)
    mD = fake_tensor(d_dtype, (m, m, l), leading_dim=1, divisibility=div_d)
    mC = None
    if c_dtype is not None:
        div_c = 128 // c_dtype.width
        mC = fake_tensor(c_dtype, (m, m, l), leading_dim=1, divisibility=div_c)

    def fake_scalar(mode):
        return Float32(1.0) if mode else None

    syrk = SyrkSm80(
        Float32,
        a_dtype,
        c_dtype is not None,
        alpha_mode,
        beta_mode,
        diag_add_mode=diag_add_mode,
        tile_m=tile_m,
        tile_k=tile_k,
        num_stages=num_stages,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        syrk,
        mA,
        mB,
        mD,
        mC,
        fake_scalar(alpha_mode),
        fake_scalar(beta_mode),
        fake_scalar(diag_add_mode),
        stream,
        options="--enable-tvm-ffi",
    )


# ── Public API ───────────────────────────────────────────────────────────────


def syrk_sm80(
    A: Tensor,  # (L, M, K) or (M, K)
    D: Tensor,  # (L, M, M) or (M, M) — output, modified in-place
    B: Optional[Tensor] = None,  # (L, M, K) or (M, K) — if None, uses A (SYRK)
    C: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    diag_add: float = 0.0,
    tile_m: int = DEFAULT_TILE_M,
    tile_k: int = DEFAULT_TILE_K,
    num_stages: int = DEFAULT_NUM_STAGES,
) -> None:
    """SM80 symmetric GEMM: D = alpha * A @ B^T + beta * C + diag_add * I.

    When B is None, computes SYRK: D = alpha * A @ A^T + beta * C + diag_add * I.
    Only computes lower triangle + mirror write for symmetric output.
    Requires that A @ B^T produces a symmetric result.
    """
    squeeze = A.ndim == 2
    if squeeze:
        A = A.unsqueeze(0)
        D = D.unsqueeze(0)
        if B is not None:
            B = B.unsqueeze(0)
        if C is not None:
            C = C.unsqueeze(0)

    if B is None:
        B = A

    # Ensure K-contiguous
    if A.stride(-1) != 1:
        A = A.contiguous()
    if B.stride(-1) != 1:
        B = B.contiguous()
    assert D.is_contiguous(), "Output D must be contiguous"

    assert A.shape[1] % tile_m == 0, f"M ({A.shape[1]}) must be divisible by tile_m ({tile_m})"
    assert A.shape[2] == B.shape[2], f"K mismatch: A has K={A.shape[2]}, B has K={B.shape[2]}"
    assert A.shape[1] == B.shape[1], f"M mismatch: A has M={A.shape[1]}, B has M={B.shape[1]}"

    # Permute (L, M, K) → (M, K, L) ; (L, M, M) → (M, M, L)
    A_p = A.permute(1, 2, 0)
    B_p = B.permute(1, 2, 0)
    D_p = D.permute(1, 2, 0)
    C_p = C.permute(1, 2, 0) if C is not None else None

    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype]
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None

    alpha_mode = 1 if alpha != 1.0 else 0
    beta_mode = 1 if beta != 1.0 else 0
    diag_add_mode = 1 if diag_add != 0.0 else 0

    compiled_fn = _compile_syrk_sm80(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        alpha_mode,
        beta_mode,
        diag_add_mode=diag_add_mode,
        tile_m=tile_m,
        tile_k=tile_k,
        num_stages=num_stages,
    )

    from dmuon.kernels.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    alpha_arg = Float32(alpha) if alpha_mode else None
    beta_arg = Float32(beta) if beta_mode else None
    diag_add_arg = Float32(diag_add) if diag_add_mode else None

    compiled_fn(A_p, B_p, D_p, C_p, alpha_arg, beta_arg, diag_add_arg)
    # D is modified in-place through the permuted view — no copy needed
