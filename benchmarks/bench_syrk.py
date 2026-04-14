"""SYRK kernel benchmark: per-operation speedup and Gram NS end-to-end.

Subcommands:
  summary  — Per-shape CuteDSL SYRK vs cuBLAS speedup + Gram NS E2E (for docs)
  detail   — Sub-operation breakdown for selected shapes

Run:
  CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_syrk.py summary
  CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_syrk.py detail
  CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_syrk.py           # both
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# ── LLM shapes ────────────────────────────────────────────────────────────────
# (model, proj_type, m_orig, n_orig)

MODEL_SHAPES = [
    # Llama-3.2 1B (hidden=2048, inter=8192, heads=32, kv=8, head_dim=64)
    ("Llama-1B", "q_proj",    2048, 2048),
    ("Llama-1B", "k_proj",    2048,  512),
    ("Llama-1B", "v_proj",    2048,  512),
    ("Llama-1B", "o_proj",    2048, 2048),
    ("Llama-1B", "gate_proj", 2048, 8192),
    ("Llama-1B", "up_proj",   2048, 8192),
    ("Llama-1B", "down_proj", 8192, 2048),

    # Llama-3.2 3B (hidden=3072, inter=8192, heads=24, kv=8, head_dim=128)
    ("Llama-3B", "q_proj",    3072, 3072),
    ("Llama-3B", "k_proj",    3072, 1024),
    ("Llama-3B", "v_proj",    3072, 1024),
    ("Llama-3B", "o_proj",    3072, 3072),
    ("Llama-3B", "gate_proj", 3072, 8192),
    ("Llama-3B", "up_proj",   3072, 8192),
    ("Llama-3B", "down_proj", 8192, 3072),

    # Llama-3.1 8B (hidden=4096, inter=14336, heads=32, kv=8, head_dim=128)
    ("Llama-8B", "q_proj",     4096,  4096),
    ("Llama-8B", "k_proj",     4096,  1024),
    ("Llama-8B", "v_proj",     4096,  1024),
    ("Llama-8B", "o_proj",     4096,  4096),
    ("Llama-8B", "gate_proj",  4096, 14336),
    ("Llama-8B", "up_proj",    4096, 14336),
    ("Llama-8B", "down_proj", 14336,  4096),

    # Qwen-2.5 3B (hidden=2048, inter=11008, heads=16, kv=2, head_dim=128)
    ("Qwen-3B", "q_proj",    2048, 2048),
    ("Qwen-3B", "k_proj",    2048,  256),
    ("Qwen-3B", "v_proj",    2048,  256),
    ("Qwen-3B", "o_proj",    2048, 2048),
    ("Qwen-3B", "gate_proj", 2048, 11008),
    ("Qwen-3B", "up_proj",   2048, 11008),
    ("Qwen-3B", "down_proj", 11008, 2048),

    # Qwen-2.5 7B (hidden=3584, inter=18944, heads=28, kv=4, head_dim=128)
    ("Qwen-7B", "q_proj",     3584,  3584),
    ("Qwen-7B", "k_proj",     3584,   512),
    ("Qwen-7B", "v_proj",     3584,   512),
    ("Qwen-7B", "o_proj",     3584,  3584),
    ("Qwen-7B", "gate_proj",  3584, 18944),
    ("Qwen-7B", "up_proj",    3584, 18944),
    ("Qwen-7B", "down_proj", 18944,  3584),

    # Qwen-2.5 14B (hidden=5120, inter=13824, heads=40, kv=8, head_dim=128)
    ("Qwen-14B", "q_proj",     5120,  5120),
    ("Qwen-14B", "k_proj",     5120,  1024),
    ("Qwen-14B", "v_proj",     5120,  1024),
    ("Qwen-14B", "o_proj",     5120,  5120),
    ("Qwen-14B", "gate_proj",  5120, 13824),
    ("Qwen-14B", "up_proj",    5120, 13824),
    ("Qwen-14B", "down_proj", 13824,  5120),

    # Mistral 7B (hidden=4096, inter=14336, heads=32, kv=8, head_dim=128)
    ("Mistral-7B", "q_proj",     4096,  4096),
    ("Mistral-7B", "k_proj",     4096,  1024),
    ("Mistral-7B", "v_proj",     4096,  1024),
    ("Mistral-7B", "o_proj",     4096,  4096),
    ("Mistral-7B", "gate_proj",  4096, 14336),
    ("Mistral-7B", "up_proj",    4096, 14336),
    ("Mistral-7B", "down_proj", 14336,  4096),

    # DeepSeek-V3/R1 (hidden=7168, inter=18432, heads=128, kv=?, head_dim=128)
    ("DeepSeek-V3", "q_proj",     7168,  7168),
    ("DeepSeek-V3", "gate_proj",  7168, 18432),
    ("DeepSeek-V3", "down_proj", 18432,  7168),
]

# Shapes for detail sub-operation breakdown
DETAIL_SHAPES = [
    (3072, 3072),
    (3072, 8192),
    (4096, 4096),
    (4096, 14336),
    (5120, 5120),
    (7168, 7168),
]


# ── Benchmark utility ─────────────────────────────────────────────────────────

def bench(fn, warmup=10, repeat=50):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]


# ── Subcommand: summary ──────────────────────────────────────────────────────

def run_summary(device, syrk_sm80, has_syrk, ns_mod, gram_ns_local):
    coefficients = ns_mod.POLAR_EXPRESS_COEFFICIENTS
    best_tm, best_tk, best_ns = 128, 64, 3

    # Table 1: Per-shape SYRK speedup
    print("=" * 100)
    print("TABLE 1: CuteDSL SYRK vs cuBLAS — Per-Shape Speedup")
    print("=" * 100)
    print(f"{'Model':<14s} {'Proj':<11s} {'Original':>12s} {'NS shape':>12s} "
          f"{'cuBLAS':>9s} {'SYRK':>9s} {'Speedup':>8s}  "
          f"{'Z cuBLAS':>9s} {'Z SYRK':>9s} {'Z Spdup':>8s}")
    print("-" * 100)

    seen_ns_shapes = {}

    for model, proj, m_orig, n_orig in MODEL_SHAPES:
        m_ns = min(m_orig, n_orig)
        k_ns = max(m_orig, n_orig)

        X = torch.randn(m_ns, k_ns, device=device, dtype=torch.float32)
        X = X / (X.norm() + 1e-7)
        X = X.to(torch.bfloat16)

        t_cublas_r = bench(lambda: X @ X.T)

        t_syrk_r = float('inf')
        if has_syrk and m_ns % best_tm == 0:
            R_buf = torch.empty(m_ns, m_ns, device=device, dtype=torch.bfloat16)
            t_syrk_r = bench(lambda: syrk_sm80(X, R_buf, tile_m=best_tm, tile_k=best_tk, num_stages=best_ns))
        spd_r = t_cublas_r / t_syrk_r if t_syrk_r < float('inf') else 0

        if m_ns not in seen_ns_shapes:
            R = (X @ X.T).contiguous()
            a, b, c = coefficients[0]
            t_cublas_z = bench(lambda: torch.addmm(R, R, R, alpha=c, beta=b))

            t_syrk_z = float('inf')
            if has_syrk and m_ns % best_tm == 0:
                Z_buf = torch.empty_like(R)
                t_syrk_z = bench(lambda: syrk_sm80(R, Z_buf, C=R, alpha=c, beta=b,
                                                     tile_m=best_tm, tile_k=best_tk, num_stages=best_ns))
            spd_z = t_cublas_z / t_syrk_z if t_syrk_z < float('inf') else 0
            seen_ns_shapes[m_ns] = (t_cublas_z, t_syrk_z, spd_z)
        else:
            t_cublas_z, t_syrk_z, spd_z = seen_ns_shapes[m_ns]

        syrk_str = f"{t_syrk_r:>7.0f}us" if t_syrk_r < float('inf') else "    N/A  "
        spd_r_str = f"{spd_r:>6.2f}x" if spd_r > 0 else "   N/A "
        z_syrk_str = f"{t_syrk_z:>7.0f}us" if t_syrk_z < float('inf') else "    N/A  "
        z_spd_str = f"{spd_z:>6.2f}x" if spd_z > 0 else "   N/A "

        print(f"{model:<14s} {proj:<11s} ({m_orig:>5d},{n_orig:>5d}) ({m_ns:>5d},{k_ns:>5d}) "
              f"{t_cublas_r:>7.0f}us {syrk_str} {spd_r_str}  "
              f"{t_cublas_z:>7.0f}us {z_syrk_str} {z_spd_str}")

    # Table 2: Gram NS End-to-End
    print()
    print("=" * 80)
    print("TABLE 2: Gram NS End-to-End — CuteDSL SYRK vs cuBLAS (torch.addmm)")
    print("=" * 80)
    print(f"{'Model':<14s} {'Proj':<11s} {'NS shape':>12s} "
          f"{'cuBLAS NS':>10s} {'SYRK NS':>10s} {'Speedup':>8s}")
    print("-" * 80)

    import dmuon.optim.syrk_dispatch as _dispatch

    seen_e2e = {}
    for model, proj, m_orig, n_orig in MODEL_SHAPES:
        m_ns = min(m_orig, n_orig)
        k_ns = max(m_orig, n_orig)
        key = (m_ns, k_ns)

        if key not in seen_e2e:
            G = torch.randn(m_ns, k_ns, device=device, dtype=torch.bfloat16)

            saved = _dispatch.HAS_SYRK
            _dispatch.HAS_SYRK = False
            t_cublas = bench(lambda: gram_ns_local(G))
            _dispatch.HAS_SYRK = saved

            if has_syrk:
                _dispatch.HAS_SYRK = True
                t_syrk = bench(lambda: gram_ns_local(G))
                _dispatch.HAS_SYRK = saved
            else:
                t_syrk = t_cublas

            spd = t_cublas / t_syrk
            seen_e2e[key] = (t_cublas, t_syrk, spd)

        t_cublas, t_syrk, spd = seen_e2e[key]
        print(f"{model:<14s} {proj:<11s} ({m_ns:>5d},{k_ns:>5d}) "
              f"{t_cublas:>8.0f}us {t_syrk:>8.0f}us {spd:>6.2f}x")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for (m, k), (tc, ts, sp) in sorted(seen_e2e.items()):
        label = "cuBLAS" if sp <= 1.01 else f"SYRK {sp:.2f}x"
        print(f"  ({m:>5d},{k:>5d}): {label}")


# ── Subcommand: detail ────────────────────────────────────────────────────────

def run_detail(device, syrk_sm80, has_syrk, ns_mod, gram_ns_local):
    coefficients = ns_mod.POLAR_EXPRESS_COEFFICIENTS
    restart_iterations = ns_mod.DEFAULT_RESTART_ITERATIONS

    for m, n in DETAIL_SHAPES:
        G = torch.randn(m, n, device=device, dtype=torch.bfloat16)
        X = G.float()
        if m > n:
            X = X.T
            m_eff, n_eff = n, m
        else:
            m_eff, n_eff = m, n
        X = X / (X.norm() + 1e-7)
        X = X.half()

        _use_syrk = has_syrk and m_eff % 64 == 0

        print(f"\n{'='*60}")
        print(f"DETAIL: ({m},{n}) -> NS shape ({m_eff},{n_eff})")
        print(f"{'='*60}")

        # R = X @ X^T
        if _use_syrk:
            R_buf = torch.empty(m_eff, m_eff, device=device, dtype=X.dtype)
            t_syrk = bench(lambda: syrk_sm80(X, R_buf), warmup=20, repeat=100)
            syrk_sm80(X, R_buf)
            R = R_buf
        else:
            t_syrk = bench(lambda: X @ X.T, warmup=20, repeat=100)
            R = X @ X.T

        t_cublas = bench(lambda: X @ X.T, warmup=20, repeat=100)
        label = "SYRK" if _use_syrk else "cuBLAS"
        print(f"  R=X@X^T  {label}: {t_syrk:>8.0f}us  cuBLAS: {t_cublas:>8.0f}us", end="")
        if _use_syrk:
            print(f"  speedup: {t_cublas/t_syrk:.2f}x")
        else:
            print()

        # Z = c*R^2 + b*R
        a0, b0, c0 = coefficients[0]
        if _use_syrk:
            Z_buf = torch.empty_like(R)
            t_z_syrk = bench(lambda: syrk_sm80(R, Z_buf, C=R, alpha=c0, beta=b0), warmup=20, repeat=100)
            syrk_sm80(R, Z_buf, C=R, alpha=c0, beta=b0)
            Z = Z_buf
        else:
            t_z_syrk = None
            Z = torch.addmm(R, R, R, alpha=c0, beta=b0)
        t_z_cublas = bench(lambda: torch.addmm(R, R, R, alpha=c0, beta=b0), warmup=20, repeat=100)

        if _use_syrk:
            print(f"  Z=c*R²+b*R  SYRK: {t_z_syrk:>8.0f}us  cuBLAS: {t_z_cublas:>8.0f}us  speedup: {t_z_cublas/t_z_syrk:.2f}x")
        else:
            print(f"  Z=c*R²+b*R  cuBLAS: {t_z_cublas:>8.0f}us")

        # Q = Z + a*I
        I = torch.eye(m_eff, device=device, dtype=X.dtype)
        t_q_init = bench(lambda: Z + a0 * I, warmup=20, repeat=100)
        Q = Z + a0 * I
        print(f"  Q=Z+a*I (init):     {t_q_init:>8.0f}us")

        # Q = a*Q + Q@Z
        if _use_syrk:
            Q2 = torch.empty_like(Q)
            t_q_syrk = bench(lambda: syrk_sm80(Q, Q2, B=Z, C=Q, beta=a0), warmup=20, repeat=100)
            t_q_cublas = bench(lambda: torch.addmm(Q, Q, Z.T, beta=a0), warmup=20, repeat=100)
            print(f"  Q=Q@Z+a*Q  SYRK: {t_q_syrk:>8.0f}us  cuBLAS: {t_q_cublas:>8.0f}us  speedup: {t_q_cublas/t_q_syrk:.2f}x")
        else:
            t_q_cublas = bench(lambda: torch.addmm(Q, Z, Q, beta=a0), warmup=20, repeat=100)
            print(f"  Q=a*Q+Z@Q  cuBLAS: {t_q_cublas:>8.0f}us")

        # RZ = R@Z + a*R
        if _use_syrk:
            RZ_buf = torch.empty_like(R)
            t_rz_syrk = bench(lambda: syrk_sm80(R, RZ_buf, B=Z, C=R, beta=a0), warmup=20, repeat=100)
            syrk_sm80(R, RZ_buf, B=Z, C=R, beta=a0)
            RZ = RZ_buf
            t_rz_cublas = bench(lambda: torch.addmm(R, R, Z.T, beta=a0), warmup=20, repeat=100)
            print(f"  RZ=R@Z+a*R  SYRK: {t_rz_syrk:>8.0f}us  cuBLAS: {t_rz_cublas:>8.0f}us  speedup: {t_rz_cublas/t_rz_syrk:.2f}x")
        else:
            RZ = torch.addmm(R, R, Z, beta=a0)
            t_rz_cublas = bench(lambda: torch.addmm(R, R, Z, beta=a0), warmup=20, repeat=100)
            print(f"  RZ=a*R+R@Z  cuBLAS: {t_rz_cublas:>8.0f}us")

        # R = Z@RZ + a*RZ
        if _use_syrk:
            R2 = torch.empty_like(R)
            t_r_syrk = bench(lambda: syrk_sm80(Z, R2, B=RZ, C=RZ, beta=a0), warmup=20, repeat=100)
            t_r_cublas = bench(lambda: torch.addmm(RZ, Z, RZ.T, beta=a0), warmup=20, repeat=100)
            print(f"  R=Z@RZ+a*RZ  SYRK: {t_r_syrk:>8.0f}us  cuBLAS: {t_r_cublas:>8.0f}us  speedup: {t_r_cublas/t_r_syrk:.2f}x")
        else:
            t_r_cublas = bench(lambda: torch.addmm(RZ, Z, RZ, beta=a0), warmup=20, repeat=100)
            print(f"  R=a*RZ+Z@RZ  cuBLAS: {t_r_cublas:>8.0f}us")

        # Q @ X (rectangular, cuBLAS always)
        t_proj = bench(lambda: Q @ X, warmup=20, repeat=100)
        print(f"  Q@X (project):      {t_proj:>8.0f}us")

        # Actual E2E
        t_e2e = bench(lambda: gram_ns_local(G), warmup=10, repeat=50)
        print(f"  E2E actual:         {t_e2e:>8.0f}us")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    gpu_name = torch.cuda.get_device_name(0)

    try:
        from dmuon.kernels.syrk_sm80 import syrk_sm80
        has_syrk = True
    except ImportError:
        syrk_sm80 = None
        has_syrk = False

    import importlib
    ns_mod = importlib.import_module("dmuon.optim.newton_schulz")
    gram_ns_local = ns_mod.gram_newton_schulz_local

    print(f"GPU: {gpu_name}")
    print(f"CuteDSL SYRK: {has_syrk}")
    from dmuon.optim.syrk_dispatch import get_ns_backend
    print(f"NS Backend: {get_ns_backend()}")
    print()

    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"

    if cmd in ("summary", "all"):
        run_summary(device, syrk_sm80, has_syrk, ns_mod, gram_ns_local)
    if cmd in ("detail", "all"):
        run_detail(device, syrk_sm80, has_syrk, ns_mod, gram_ns_local)


if __name__ == "__main__":
    main()
