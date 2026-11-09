"""Source-pinned attention layouts with independent native-index selectors.

These are test data, not model adapters. Optional matrix boundaries are an
explicit optimizer choice. ``stacked_*`` and column-oriented variants are
stress layouts; the official K3 HF experts are separate modules.
No upstream code, network access, or large checkpoints are required.
"""

from dataclasses import dataclass
from typing import Callable

import torch

from dmuon import MatrixLayout


SOURCES = {
    "deepseek_v41": "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py",
    "kimi_k3": "https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py",
    "kimi_vision": "https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_k3.py",
    "vllm_dsa": "https://github.com/vllm-project/vllm/blob/73864f836099d51062981977807d1e7ffa43759f/vllm/model_executor/models/deepseek_v2.py",
    "moba": "https://github.com/MoonshotAI/MoBA/blob/b5d58363311d3ca946f1ec444182727c15e338b5/moba/moba_naive.py",
}


@dataclass(frozen=True)
class LayoutCase:
    name: str
    shape: tuple[int, ...]
    layout: MatrixLayout
    blocks: Callable[[torch.Tensor], list[torch.Tensor]]
    origin: str
    stress_variant: bool = False

    def indices(self, device="cpu"):
        # This oracle never calls MatrixLayout.pack/unpack or MatrixMap.indices.
        native = torch.arange(torch.tensor(self.shape).prod().item(), device=device)
        return self.blocks(native.reshape(self.shape))


def row_blocks(*sizes):
    return lambda t: list(t.split(sizes, dim=0))


def head_blocks(heads, left, right, rank, *, whole):
    def select(t):
        native = t.reshape(heads, left + right, rank)
        if whole:
            return [native[:, :left].reshape(-1, rank), native[:, left:].reshape(-1, rank)]
        return [native[h, :left] for h in range(heads)] + [native[h, left:] for h in range(heads)]

    return select


def _generic_blocks(t, k):
    native = t.reshape(2, 4, 3, 4 * k).permute(2, 0, 3, 1).reshape(3, 8 * k, 4)
    return [
        matrix
        for first in native.split((1, 2), dim=0)
        for branch in first.split((3 * k, 5 * k), dim=1)
        for matrix in branch.permute(0, 2, 1).unbind(0)
    ]


def attention_layout_cases(scale="small", seed=0):
    """small: odd nonpowers; native: true head/rank dims, experts bounded at 2.

    metadata uses the actual 896-expert stacked stress shape without allocating
    that tensor. This mode is ONLY for metadata coverage, never a copy default.
    """
    if scale not in ("small", "native", "metadata"):
        raise ValueError(scale)
    full = scale != "small"
    # Seeded small geometry changes boundaries without changing source semantics.
    h = 7168 if full else 11 + 2 * (seed % 3)
    n, qrank, krank = (96, 1536, 512) if full else (3 + seed % 2, 11, 7)
    nope, rope, vdim = (128, 64, 128) if full else (5, 3, 4)
    idx_dim, idx_heads = (128, 64) if full else (7, 3)
    qa, kva, kr = (1536, 576, 512) if full else (9, 6, 4)
    cases = [
        LayoutCase(
            "dsa_fused_wk_weights",
            (idx_dim + idx_heads, h),
            MatrixLayout.rows(idx_dim, idx_heads),
            row_blocks(idx_dim, idx_heads),
            "vllm_dsa",
        ),
        LayoutCase(
            "mla_fused_q_kv_a",
            (qa + kva, h),
            MatrixLayout.rows(qa, kva),
            row_blocks(qa, kva),
            "vllm_dsa",
        ),
        LayoutCase(
            "mla_fused_q_latent_rope",
            (qa + kva, h),
            MatrixLayout.rows(qa, kr, kva - kr),
            row_blocks(qa, kr, kva - kr),
            "vllm_dsa",
        ),
    ]
    kd = 128 if full else 5
    cases.append(
        LayoutCase(
            "k3_kda_separate_q_k_v",
            (n * kd, h),
            MatrixLayout.view(n, kd, h),
            lambda t: list(t.reshape(n, kd, h).unbind(0)),
            "kimi_k3",
        )
    )
    for whole in (False, True):
        name = "whole" if whole else "per_head"
        layout = MatrixLayout.view(n, nope + rope, qrank).split((nope, rope), dim=1)
        kv_layout = MatrixLayout.view(n, nope + vdim, krank).split((nope, vdim), dim=1)
        if whole:
            layout = layout.reshape(-1, qrank)
            kv_layout = kv_layout.reshape(-1, krank)
        cases += [
            LayoutCase(
                f"k3_q_b_{name}",
                (n * (nope + rope), qrank),
                layout,
                head_blocks(n, nope, rope, qrank, whole=whole),
                "kimi_k3",
            ),
            LayoutCase(
                f"k3_kv_b_{name}",
                (n * (nope + vdim), krank),
                kv_layout,
                head_blocks(n, nope, vdim, krank, whole=whole),
                "kimi_k3",
            ),
        ]
    groups, out, inp = (8, 1024, 4096) if full else (3, 5, h)
    heads, dim, rank = (64, 512, 1280) if full else (3, 10, 7)
    tail = 64 if full else 2
    cases += [
        LayoutCase(
            "v41_grouped_output",
            (groups * out, inp),
            MatrixLayout.view(groups, out, inp),
            lambda t: list(t.reshape(groups, out, inp).unbind(0)),
            "deepseek_v41",
        ),
        LayoutCase(
            "v41_q_b_rope_tail",
            (heads * dim, rank),
            MatrixLayout.view(heads, dim, rank).split((dim - tail, tail), dim=1).reshape(-1, rank),
            head_blocks(heads, dim - tail, tail, rank, whole=True),
            "deepseek_v41",
        ),
    ]
    vn, vd, vh = (12, 128, 1024) if full else (3, 4, h)
    cases.append(
        LayoutCase(
            "k3_vision_qkv",
            (3 * vn * vd, vh),
            MatrixLayout.view(3, vn, vd, vh),
            lambda t: [t.reshape(3, vn, vd, vh)[q, head] for q in range(3) for head in range(vn)],
            "kimi_vision",
        )
    )
    experts = (896 if scale == "metadata" else 2) if full else 3
    inter, latent = (3072, 3584) if full else (5, 7)
    cases += [
        LayoutCase(
            "stacked_latent_moe",
            (experts, 2 * inter, latent),
            MatrixLayout.view(experts, 2, inter, latent),
            lambda t: [t[e, b * inter : (b + 1) * inter] for e in range(experts) for b in range(2)],
            "kimi_k3",
            True,
        ),
        LayoutCase(
            "column_latent_moe",
            (experts, latent, 2 * inter),
            MatrixLayout.view(experts, latent, 2, inter).permute(0, 2, 3, 1),
            lambda t: [
                t[e, :, b * inter : (b + 1) * inter].t() for e in range(experts) for b in range(2)
            ],
            "kimi_k3",
            True,
        ),
    ]
    if not full:
        for side, rows, cols in [(128, 130, 258), (256, 258, 514), (64, 129, 257)]:
            cases.append(
                LayoutCase(
                    f"transpose_tail_{side}",
                    (2, rows, 2 * cols),
                    MatrixLayout.view(2, rows, 2, cols).permute(0, 2, 3, 1),
                    lambda t, r=rows, c=cols: [
                        t[e, :, b * c : (b + 1) * c].t() for e in range(2) for b in range(2)
                    ],
                    "synthetic_transpose_tails",
                    True,
                )
            )
        k = 31 + seed * 2
        cases.append(
            LayoutCase(
                "four_stage_index",
                (8, 12 * k),
                MatrixLayout.view(2, 4, 3, 4 * k)
                .permute(2, 0, 3, 1)
                .reshape(3, 8 * k, 4)
                .split((1, 2), dim=0)
                .split((3 * k, 5 * k), dim=-2)
                .permute(0, 2, 1),
                lambda t: _generic_blocks(t, k),
                "synthetic_four_stage",
                True,
            )
        )
    return cases


def oracle_scatter(case, matrices, *, initial=None):
    """Inverse by independent integer indexing, not inverse layout compilation."""
    output = (
        torch.empty(case.shape, dtype=matrices[0].dtype, device=matrices[0].device)
        if initial is None
        else initial.clone()
    )
    for positions, matrix in zip(case.indices(output.device), matrices):
        if initial is None:
            output.flatten()[positions.flatten()] = matrix.flatten()
        else:
            output.flatten()[positions.flatten()] += matrix.flatten().to(output.dtype)
    return output
