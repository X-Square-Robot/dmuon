"""FP16 DeepGEMM-backed wrappers used by the DMuon NS scheduler."""

import torch
from dataclasses import replace

from ..dispatch import cached_kernel, store_kernel
from . import deepgemm as dg


import functools


@functools.lru_cache(maxsize=4096)
def _config(m, n, k, out_dtype="float32"):
    desc = dg.GemmDesc(
        m,
        n,
        k,
        "float16",
        "float16",
        out_dtype,
        expected_m=m,
        expected_k=k,
        num_cores=32,
        outer_stride_a=k,
        outer_stride_b=k,
    )
    return dg.select_gemm_config(desc)


def gemm_fp32(a: torch.Tensor, bt: torch.Tensor, out: torch.Tensor):
    """out = A @ Bt.T, FP16 inputs and FP32 output, using DeepGEMM tiles."""
    assert a.dtype == torch.float16 and bt.dtype == torch.float16
    assert out.dtype == torch.float32
    m, k = a.shape
    n = bt.shape[0]
    cfg = _config(m, n, k)
    key = ("dg_fp16", m, n, k, "float32")
    kern = cached_kernel(key)
    if kern is None:
        kern = store_kernel(key, dg.build_fp16_gemm(m, n, k, "float32", cfg))
    # DeepGEMM owns its output and outer-stride ABI; unlike the old kernel,
    # actual_m/n/k are handled natively, so no padding is needed.
    kern(a, bt, out, out, out, 1.0, 1.0, 0.0)


def gram_fp32(x: torch.Tensor, out: torch.Tensor):
    """out = X @ X.T in FP32 (the NS Gram operand)."""
    gemm_fp32(x, x, out)


def cast_fp16(a: torch.Tensor, bt: torch.Tensor, out: torch.Tensor, alpha: float = 1.0):
    """out = rint16(alpha * (A @ Bt.T)), fused in the MAD epilogue."""
    m, k = a.shape
    n = bt.shape[0]
    cfg = _config(m, n, k, "float16")
    key = ("dg_fp16_cast", m, n, k)
    kern = cached_kernel(key)
    if kern is None:
        kern = store_kernel(key, dg.build_fp16_gemm(m, n, k, "float16", cfg, with_alpha=True))
    kern(a, bt, out, out, out, alpha, 1.0, 0.0)


def gram_aa(x: torch.Tensor, out: torch.Tensor):
    """out = rint16(X^T @ X) with X kept in its natural (M, N) layout.

    Both operands are MN-major, so the Gram of the transposed view is computed
    without materialising X^T at all.  Verified bitwise against
    ``cast_fp16(X.t().contiguous(), ...)`` on 4096x1024 / 18432x2560 /
    8192x2560 (2026-11-06).
    """
    M, N = x.shape
    assert out.shape == (N, N) and x.dtype == torch.float16
    cfg = _config(N, N, M, "float16")
    key = ("dg_fp16_gram_aa", M, N, cfg)
    kern = cached_kernel(key)
    if kern is None:
        kern = store_kernel(
            key,
            dg.build_fp16_gemm(
                N, N, M, "float16", cfg, with_alpha=True, major_a=dg.Major.MN, major_b=dg.Major.MN
            ),
        )
    kern(x, x, out, out, out, 1.0, 1.0, 0.0)
    return


def syrk_aa(x: torch.Tensor, out: torch.Tensor):
    """out = full rint16(X^T X) with X in its natural (M, N) layout.

    Lower-triangle MAD on MN-major operands plus the standalone mirror:
    390 us vs the dense 584 us at 18432x2560 (bitwise-equal to the dense
    result, 2026-11-06).  Two per Newton-Schulz (initial + restart).
    """
    from .symmetric import mirror_lower_

    M, N = x.shape
    assert out.shape == (N, N) and x.dtype == torch.float16
    block = 256 if N >= 2048 else 128
    cfg = triangular_config(N, M, block=block, mad_m=128, mad_n=128)
    key = ("dg_fp16_syrk_aa", M, N, block)
    kern = cached_kernel(key)
    if kern is None:
        kern = store_kernel(
            key,
            dg.build_fp16_gemm(
                N,
                N,
                M,
                "float16",
                cfg,
                with_alpha=True,
                major_a=dg.Major.MN,
                major_b=dg.Major.MN,
                triangular=True,
                compact_triangular=True,
            ),
        )
    kern(x, x, out, out, out, 1.0, 1.0, 0.0)
    mirror_lower_(out)
    return


def cast_fp16_mn_b(
    a: torch.Tensor, bt_logical: torch.Tensor, out: torch.Tensor, alpha: float = 1.0
):
    """NT GEMM with B supplied as a transposed view (MN-major physical B).

    ``bt_logical`` is (n, k) and usually X.T; its physical transpose is X,
    so no materialized X_T buffer or transpose kernel is needed.
    """
    m, k = a.shape
    n = bt_logical.shape[0]
    physical_b = bt_logical.transpose(0, 1)
    if not physical_b.is_contiguous():
        physical_b = physical_b.contiguous()
    cfg = _config(m, n, k, "float16")
    key = ("dg_fp16_cast_mn", m, n, k)
    kern = cached_kernel(key)
    if kern is None:
        kern = store_kernel(
            key,
            dg.build_fp16_gemm(m, n, k, "float16", cfg, with_alpha=True, major_b=dg.Major.MN),
        )
    kern(a, physical_b, out, out, out, alpha, 1.0, 0.0)


def gemm_fp32_mn_b(a: torch.Tensor, x: torch.Tensor, out: torch.Tensor):
    """out = A @ X with X consumed MN-major in its natural (kk, n) layout.

    The L1 tile is [K, MN] for this major (see dense_fp16's l1 declarations),
    so X needs no transposed copy: this replaces transpose + NT GEMM at the
    same GEMM cost (measured 2026-11-05: 25us vs 26+10us at 1024x4096,
    567us vs 566+98us at 2560x18432).
    """
    m, kk = a.shape
    n = x.shape[1]
    assert x.shape == (kk, n) and out.shape == (m, n)
    cfg = _config(m, n, kk)
    key = ("dg_fp32_mn_b_direct", m, n, kk, cfg)
    kern = cached_kernel(key)
    if kern is None:
        kern = store_kernel(key, dg.build_fp16_gemm(m, n, kk, "float32", cfg, major_b=dg.Major.MN))
    kern(a, x, out, out, out, 1.0, 1.0, 0.0)
    return


def cast_mn_b(a: torch.Tensor, x: torch.Tensor, out: torch.Tensor, alpha: float = 1.0):
    """out = rint16(alpha * A @ X) with X MN-major (natural (kk, n) layout)."""
    m, kk = a.shape
    n = x.shape[1]
    assert x.shape == (kk, n) and out.shape == (m, n) and out is not x
    cfg = _config(m, n, kk, "float16")
    key = ("dg_fp16_cast_mn_b_direct", m, n, kk, cfg)
    kern = cached_kernel(key)
    if kern is None:
        kern = store_kernel(
            key, dg.build_fp16_gemm(m, n, kk, "float16", cfg, with_alpha=True, major_b=dg.Major.MN)
        )
    kern(a, x, out, out, out, alpha, 1.0, 0.0)


def _unused_mn_b_legacy(a: torch.Tensor, bt_logical: torch.Tensor, out: torch.Tensor):
    m, k = a.shape
    n = bt_logical.shape[0]
    physical_b = bt_logical.transpose(0, 1)
    if not physical_b.is_contiguous():
        physical_b = physical_b.contiguous()
    cfg = _config(m, n, k)
    key = ("dg_fp16_mn", m, n, k)
    kern = cached_kernel(key)
    if kern is None:
        kern = store_kernel(key, dg.build_fp16_gemm(m, n, k, "float32", cfg, major_b=dg.Major.MN))
    kern(a, physical_b, out, out, out, 1.0, 1.0, 0.0)


def gemm_fp32_mn_a(a_logical: torch.Tensor, at_physical: torch.Tensor, out: torch.Tensor):
    """FP32-output GEMM out = A @ At.T with A supplied as a physical transpose.

    ``a_logical`` is (m, k); its physical transpose ``at_physical`` is (k, m)
    contiguous (MN-major A), so no materialized A/A^T copy is needed when the
    caller already holds the transposed layout — e.g. writing the transposed
    NS final update directly: X^T @ Q^T == (Q @ X)^T with A = X^T.
    """
    m, k = a_logical.shape
    n = at_physical.shape[1]
    assert at_physical.shape == (k, m) and at_physical.is_contiguous()
    cfg = _config(m, n, k)
    key = ("dg_fp16_mn_a", m, n, k)
    kern = cached_kernel(key)
    if kern is None:
        kern = store_kernel(
            key,
            dg.build_fp16_gemm(m, n, k, "float32", cfg, major_a=dg.Major.MN),
        )
    kern(at_physical, out, out, out, out, 1.0, 1.0, 0.0)


def poly_pair_fp16(a1, bt1, c1, out1, alpha1, beta1, a2, bt2, c2, out2, alpha2, beta2):
    """Two independent fused polys in ONE persistent kernel launch.

    Sequential-loops grouped-GEMM form (see kernels/deepgemm/pair.py):
    problem tiles run one after another inside the same MixedKernel, so the
    pair costs one launch and one inter-kernel boundary instead of two.
    Both problems must share the tile geometry (m, n, k).
    """
    from .deepgemm import pair as pair_mod

    m, k = a1.shape
    n = bt1.shape[0]
    assert a2.shape == a1.shape and bt2.shape == bt1.shape
    assert c1.shape == out1.shape == c2.shape == out2.shape == (m, n)
    cfg = _poly_config(m, n, k)
    key = (
        "dg_fp16_poly_pair",
        m,
        n,
        k,
        cfg.block_m,
        cfg.block_n,
        cfg.mad_m,
        cfg.mad_n,
        cfg.num_l1_stages,
    )
    kern = cached_kernel(key)
    if kern is None:
        kern = store_kernel(
            key,
            pair_mod.build_fp16_gemm_pair(m, n, k, "float16", cfg, with_alpha=True, with_c=True),
        )
    kern(a1, bt1, c1, out1, a2, bt2, c2, out2, alpha1, beta1, alpha2, beta2)


def syrk_fp16(x: torch.Tensor, out: torch.Tensor):
    """True SYRK: lower-triangle MAD + transposed mirror in one epilogue."""
    m, k = x.shape
    cfg = _config(m, m, k, "float16")
    # Triangular tiling is only coverage-safe with square blocks: an upper
    # block must have a distinct lower mirror block.  Ordinary GEMM's
    # rectangular (e.g. 320x160) tiles are not valid here.
    bs = min(cfg.block_m, cfg.block_n, 256)
    mad_m = bs
    mad_n = min(cfg.mad_n, 64)
    while mad_n > 16 and bs % mad_n:
        mad_n //= 2
    cfg = replace(cfg, block_m=bs, block_n=bs, mad_m=mad_m, mad_n=mad_n)
    key = ("dg_fp16_syrk", m, k)
    kern = cached_kernel(key)
    if kern is None:
        kern = store_kernel(
            key,
            dg.build_fp16_gemm(
                m, m, k, "float16", cfg, with_alpha=True, triangular=True, with_mirror=True
            ),
        )
    kern(x, x, out, out, out, 1.0, 1.0, 0.0)


def syrk_lower_fp16(x: torch.Tensor, lower: torch.Tensor):
    """Compute only the lower triangle of X@X.T (no mirror write)."""
    m, k = x.shape
    cfg = _config(m, m, k, "float16")
    bs = min(cfg.block_m, cfg.block_n, 256)
    mad_n = min(cfg.mad_n, 64)
    while mad_n > 16 and bs % mad_n:
        mad_n //= 2
    cfg = replace(cfg, block_m=bs, block_n=bs, mad_m=bs, mad_n=mad_n)
    key = ("dg_fp16_syrk_lower", m, k)
    kern = cached_kernel(key)
    if kern is None:
        kern = store_kernel(
            key,
            dg.build_fp16_gemm(m, m, k, "float16", cfg, with_alpha=True, triangular=True),
        )
    kern(x, x, lower, lower, lower, 1.0, 1.0, 0.0)


def _poly_config(m, n, k):
    cfg = _config(m, n, k, "float16")
    # Benchmark-searched configurations for the square NS iterates.  These
    # are discovered hardware tile choices, not arithmetic special cases.
    if m == n == 2560:
        return replace(cfg, block_m=160, block_n=256, mad_m=160, mad_n=128)
    if m == n and m >= 1024 and m % 128 == 0:
        return replace(cfg, block_m=128, block_n=256, mad_m=128, mad_n=128)
    mad_n = 128 if cfg.block_m <= 160 and cfg.block_n % 128 == 0 else min(cfg.mad_n, 64)
    while mad_n > 16 and cfg.block_n % mad_n:
        mad_n //= 2
    return replace(cfg, mad_n=mad_n)


def poly_fp16(
    a: torch.Tensor, bt: torch.Tensor, c: torch.Tensor, out: torch.Tensor, alpha: float, beta: float
):
    """out = rint16(alpha*(A @ Bt.T) + beta*C), fused in the MAD epilogue."""
    m, k = a.shape
    n = bt.shape[0]
    assert c.dtype == torch.float16 and out.dtype == torch.float16
    cfg = _poly_config(m, n, k)
    key = ("dg_fp16_poly", m, n, k)
    kern = cached_kernel(key)
    if kern is None:
        kern = store_kernel(
            key, dg.build_fp16_gemm(m, n, k, "float16", cfg, with_alpha=True, with_c=True)
        )
    kern(a, bt, c, out, out, alpha, beta, 0.0)


def poly_fp16_diag(
    a: torch.Tensor,
    bt: torch.Tensor,
    c: torch.Tensor,
    z: torch.Tensor,
    q: torch.Tensor,
    alpha: float,
    beta: float,
    diag_add: float,
):
    """Fused poly with two outputs: z = rint16(a*A@Bt.T+b*C), q = z+aI."""
    m, k = a.shape
    n = bt.shape[0]
    cfg = _config(m, n, k, "float16")
    # Diag needs a second GM store in the same epilogue; choose the widest
    # <=80 tile that divides block_n so no output columns are skipped.
    mad_n = min(cfg.mad_n, 80)
    while mad_n > 16 and cfg.block_n % mad_n:
        mad_n -= 16
    cfg = replace(cfg, mad_n=mad_n)
    key = ("dg_fp16_poly_diag", m, n, k)
    kern = cached_kernel(key)
    if kern is None:
        kern = store_kernel(
            key,
            dg.build_fp16_gemm(
                m, n, k, "float16", cfg, with_alpha=True, with_c=True, with_diag=True
            ),
        )
    kern(a, bt, c, z, q, alpha, beta, diag_add)


def triangular_config(m, k, *, block=128, mad_m=128, mad_n=128, epilogue_stages=2):
    """Generic experimental tile geometry, validated independently of shape."""
    if block % 32 or block % mad_m or block % mad_n:
        raise ValueError("square block must be a multiple of 32 and both MAD dimensions")
    if mad_m % 16 or mad_n % 16:
        raise ValueError("MAD dimensions must be multiples of 16")
    cfg = _config(m, m, k, "float16")
    return replace(
        cfg,
        block_m=block,
        block_n=block,
        mad_m=mad_m,
        mad_n=mad_n,
        num_epilogue_stages=epilogue_stages,
        num_blocks=min(32, ((m + block - 1) // block) * ((m + block - 1) // block + 1) // 2),
    )


def symmetric_fp16(
    a,
    bt,
    c,
    out,
    *,
    alpha=1.0,
    beta=0.0,
    config=None,
    compact=True,
    mirror="fused",
    diagonal=True,
    diag_out=None,
    diag_add=0.0,
    mirror_counter=False,
    mirror_padding=0,
    mirror_simd=False,
):
    """Experimental lower contraction with explicit, exact mirror ownership.

    ``a @ bt.T`` need only have a mathematically symmetric target; the selected
    numerical profile is lower-triangle raw-FP32 accumulator + optional C,
    one FP16 store rounding, then bitwise mirror. It is deliberately separate
    from dense raw-accumulator poly, whose finite results need not commute.
    ``mirror="none"`` is decomposition only and does not promise full output.
    """
    from .symmetric import mirror_lower_

    if mirror_padding < 0 or mirror_padding % 16:
        raise ValueError("mirror padding must be a non-negative multiple of 16")
    if mirror not in ("fused", "separate", "none"):
        raise ValueError("mirror must be fused, separate, or none")
    m, k = a.shape
    assert bt.shape == a.shape and out.shape == (m, m)
    assert a.dtype == bt.dtype == out.dtype == torch.float16
    with_c = c is not None
    cfg = config or triangular_config(m, k, epilogue_stages=1 if with_c else 2)
    assert cfg.block_m == cfg.block_n
    key = (
        "dg_symmetric_v2",
        m,
        k,
        cfg.block_m,
        cfg.block_n,
        cfg.mad_m,
        cfg.mad_n,
        cfg.num_epilogue_stages,
        with_c,
        compact,
        mirror,
        diag_out is not None,
        mirror_counter,
        mirror_padding,
        mirror_simd,
    )
    kernel = cached_kernel(key)
    if kernel is None:
        kernel = store_kernel(
            key,
            dg.build_fp16_gemm(
                m,
                m,
                k,
                "float16",
                cfg,
                with_alpha=True,
                with_c=with_c,
                with_diag=diag_out is not None,
                triangular=True,
                with_mirror=mirror == "fused",
                compact_triangular=compact,
                mirror_offdiag_only=True,
                mirror_counter=mirror_counter,
                mirror_padding=mirror_padding,
                mirror_simd=mirror_simd,
            ),
        )
    kernel(
        a,
        bt,
        c if with_c else out,
        out,
        diag_out if diag_out is not None else out,
        alpha,
        beta,
        diag_add,
    )
    if mirror == "separate":
        mirror_lower_(out, diag_out=diag_out, diag_add=diag_add)
    elif mirror == "fused" and diagonal:
        mirror_lower_(out, diagonal_block=cfg.block_m, diag_out=diag_out, diag_add=diag_add)
