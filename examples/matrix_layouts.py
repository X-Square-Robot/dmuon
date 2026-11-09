"""Matrix policies applied to the original model without model adapters.

Use configure_moe with the model, mesh, and wrapping predicate from your trainer.
Run PYTHONPATH=. python examples/matrix_layouts.py to inspect example layouts on CPU.
The MoE rule paths below are examples; match the actual FQNs in your model.

The optional CUDA copy extension batches each wrapping group's local copies.
DMUON_COPY_MAX_CTAS=96 (default) caps each copy launch, not all communication SMs.
The actual grid is also capped by the device SM count and available copy tiles.
DMUON_COPY_BACKEND=torch selects the reference; cuda requires the extension.
Build independently with DMUON_BUILD_LAYOUT_COPY=1; 0 disables this extension.
"""

import dmuon
from dmuon import MatrixLayout


def configure_moe(model, dp_mesh, wrap_decoder_layer):
    cfg = model.config
    E, I, H = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
    return dmuon.dedicate_params(
        model,
        dp_mesh,
        predicate=lambda name, p: p.requires_grad,
        hook_boundary_predicate=wrap_decoder_layer,
        param_policy={
            "defaults": {"route": "adamw"},
            "overrides": [
                {
                    "name_glob": "model.layers.*.mlp.experts.gate_up_proj",
                    "route": "muon",
                    "matrix_layout": MatrixLayout.view(E, 2, I, H),
                },
                {
                    "name_glob": "model.layers.*.mlp.experts.down_proj",
                    "route": "muon",
                    "matrix_layout": MatrixLayout.view(E, H, I),
                },
            ],
        },
    )


def main():
    examples = [
        ("whole QKV", (24, 8), MatrixLayout.whole()),
        ("Q/K/V", (24, 8), MatrixLayout.view(3, 8, 8)),
        ("per-head QKV", (24, 8), MatrixLayout.view(3, 2, 4, 8)),
        ("unequal Q/K/V", (24, 8), MatrixLayout.rows(16, 4, 4)),
        ("expert gate/up split", (2, 8, 8), MatrixLayout.view(2, 2, 4, 8)),
        ("expert gate/up combined", (2, 8, 8), MatrixLayout.view(2, 8, 8)),
        # Physical order: Q0,K0,V0,Q1,K1,V1, with four rows per head.
        (
            "interleaved full Q/K/V",
            (24, 8),
            MatrixLayout.view(2, 3, 4, 8).permute(1, 0, 2, 3).reshape(3, 8, 8),
        ),
        ("interleaved per-head QKV", (24, 8), MatrixLayout.view(2, 3, 4, 8)),
        # Two KV groups, each containing two Q heads, one K and one V.
        (
            "grouped GQA full Q/K/V",
            (16, 8),
            MatrixLayout.view(2, 4, 2, 8).split((2, 1, 1), dim=1).reshape(-1, 8),
        ),
        # Output/input orientation for a weight stored as [input, packed_output].
        ("column-fused Q/K/V", (8, 24), MatrixLayout.view(8, 3, 8).permute(1, 2, 0)),
        (
            "column-fused expert gate/up",
            (2, 8, 8),
            MatrixLayout.view(2, 8, 2, 4).permute(0, 2, 3, 1),
        ),
    ]
    for name, shape, layout in examples:
        plan = layout.compile(shape)
        print(
            name,
            "original",
            shape,
            "NS matrices",
            [m.shape for m in plan.matrices],
            "pack/unpack",
            not plan.is_contiguous,
        )


if __name__ == "__main__":
    main()
