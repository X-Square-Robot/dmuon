"""Pure layout compilation and per-matrix owner planning."""

import pytest
import torch
from torch import nn

from dmuon import MatrixLayout
from dmuon._core.partition import compute_balanced_assignment
from dmuon.policy import resolve_param_policies


@pytest.mark.parametrize(
    "shape,layout,expected",
    [
        ((12, 8), MatrixLayout.whole(), [(12, 8)]),
        ((12, 8), MatrixLayout.view(3, 4, 8), [(4, 8)] * 3),
        ((12, 8), MatrixLayout.view(3, 2, 2, 8), [(2, 8)] * 6),
        ((12, 8), MatrixLayout.rows(8, 2, 2), [(8, 8), (2, 8), (2, 8)]),
        ((3, 8, 4), MatrixLayout.view(3, 2, 4, 4), [(4, 4)] * 6),
        ((3, 8, 4), MatrixLayout.view(-1, 8, 4), [(8, 4)] * 3),
        ((3, 8, 4), MatrixLayout.rows(4, 4), [(4, 4)] * 6),
    ],
)
def test_layout_offsets_cover_source_once(shape, layout, expected):
    compiled = layout.compile(shape)
    assert [matrix.shape for matrix in compiled.matrices] == expected
    reference = torch.arange(torch.tensor(shape).prod()).view(shape)
    parts = [
        reference.reshape(-1)[m.offset : m.offset + m.numel].view(m.shape)
        for m in compiled.matrices
    ]
    assert torch.equal(torch.cat([part.flatten() for part in parts]), reference.flatten())
    assert len(compiled.signature) == 16


@pytest.mark.parametrize(
    "layout",
    [
        MatrixLayout.view(2, 3),
        MatrixLayout.rows(3, 3),
        MatrixLayout.view(-1, -1, 4),
        MatrixLayout.view(0, 4),
        MatrixLayout.rows(),
        MatrixLayout.rows(-1, 9),
        MatrixLayout.view(4.0, 4),
        MatrixLayout("unknown"),
    ],
)
def test_invalid_layout_rejected(layout):
    with pytest.raises(ValueError):
        layout.compile((8, 4))


class Mesh:
    mesh_dim_names = None

    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


@pytest.mark.parametrize("hsdp", [False, True])
def test_planner_uses_units_without_altering_original_model(hsdp):
    model = nn.Module()
    model.weight = nn.Parameter(torch.empty(4, 8, 4))
    model.a = nn.Parameter(torch.empty(4, 4))
    model.b = nn.Parameter(torch.empty(4, 4))
    original = list(model.named_parameters())
    layout = MatrixLayout.view(4, 8, 4).compile(model.weight.shape)
    result = compute_balanced_assignment(
        model,
        Mesh(2 if hsdp else 4),
        lambda _n, _p: True,
        replicate_mesh=Mesh(2) if hsdp else None,
        matrix_layouts={model.weight: layout},
    )
    assert result.allocation_unit_count == 5
    assert len(result.matrix_owners) == 4
    assert model.weight not in result.dp_owners
    assert len(set(result.matrix_owners.values())) > 1
    assert result.dp_owners[model.a] == result.dp_owners[model.b]
    assert all(a is b and an == bn for (an, a), (bn, b) in zip(original, model.named_parameters()))


def test_inline_policy_glob_and_callback_are_precise():
    model = nn.Module()
    model.qkv = nn.Linear(4, 12, bias=False)
    model.other = nn.Linear(4, 4, bias=False)
    params = {p: name for name, p in model.named_parameters()}
    policy = {
        "defaults": {"route": "adamw"},
        "overrides": [
            {
                "name_glob": "qkv.weight",
                "route": "muon",
                "matrix_layout": MatrixLayout.view(3, 4, 4),
            }
        ],
    }
    resolved = resolve_param_policies(params=params, param_to_fqn=params, param_policy=policy)
    assert resolved[model.qkv.weight].matrix_layout == MatrixLayout.view(3, 4, 4)
    assert resolved[model.other.weight].route == "adamw"
    assert resolved[model.other.weight].matrix_layout is None


@pytest.mark.parametrize("per_head", [False, True])
def test_interleaved_qkv_has_the_expected_rows_and_inverse(per_head):
    source = torch.arange(24 * 8).view(24, 8)
    layout = MatrixLayout.view(2, 3, 4, 8)
    if not per_head:
        layout = layout.permute(1, 0, 2, 3).reshape(3, 8, 8)
    plan = layout.compile(source.shape)
    packed = plan.pack(source).clone()
    matrices = [packed[m.offset : m.offset + m.numel].view(m.shape) for m in plan.matrices]
    if per_head:
        expected = list(source.split(4))
    else:
        expected = [
            source[rows]
            for rows in [
                [0, 1, 2, 3, 12, 13, 14, 15],
                [4, 5, 6, 7, 16, 17, 18, 19],
                [8, 9, 10, 11, 20, 21, 22, 23],
            ]
        ]
    for actual, wanted in zip(matrices, expected):
        torch.testing.assert_close(actual, wanted)
    torch.testing.assert_close(plan.unpack(packed), source)
    # An asymmetric update must go back to the matching original Q rows.
    matrices[0].add_(1000)
    wanted = source.clone()
    wanted[:4].add_(1000)
    if not per_head:
        wanted[12:16].add_(1000)
    torch.testing.assert_close(plan.unpack(packed), wanted)


def test_grouped_gqa_unequal_branches_and_inferred_shapes():
    source = torch.arange(16 * 5).view(16, 5)
    plan = (
        MatrixLayout.view(2, 4, 2, 5).split((2, 1, 1), dim=1).reshape(-1, 5).compile(source.shape)
    )
    assert [m.shape for m in plan.matrices] == [(8, 5), (4, 5), (4, 5)]
    expected_rows = [[0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 12, 13], [6, 7, 14, 15]]
    packed = torch.empty(source.numel(), dtype=source.dtype)
    assert plan.pack(source, out=packed) is packed
    for spec, rows in zip(plan.matrices, expected_rows):
        torch.testing.assert_close(
            packed[spec.offset : spec.offset + spec.numel].view(spec.shape), source[rows]
        )
    restored = torch.empty_like(source)
    assert plan.unpack(packed, out=restored) is restored
    torch.testing.assert_close(restored, source)


@pytest.mark.parametrize("moe", [False, True])
def test_column_fused_matrices_are_transposed_to_output_input_order(moe):
    if moe:
        source = torch.arange(2 * 3 * 8).view(2, 3, 8)
        layout = MatrixLayout.view(2, 3, 2, 4).permute(0, 2, 3, 1)
        expected = [source[e, :, start : start + 4].t() for e in range(2) for start in (0, 4)]
    else:
        source = torch.arange(3 * 12).view(3, 12)
        layout = MatrixLayout.view(3, 3, 4).permute(1, 2, 0)
        expected = [source[:, start : start + 4].t() for start in (0, 4, 8)]
    plan = layout.compile(source.shape)
    packed = plan.pack(source)
    for spec, wanted in zip(plan.matrices, expected):
        torch.testing.assert_close(
            packed[spec.offset : spec.offset + spec.numel].view(spec.shape), wanted
        )
    torch.testing.assert_close(plan.unpack(packed), source)


def test_multiple_splits_and_noncontiguous_reshape_are_reversible():
    source = torch.arange(8 * 12).view(8, 12)
    plan = (
        MatrixLayout.view(2, 4, 3, 4)
        .permute(2, 0, 3, 1)
        .reshape(3, 8, 4)
        .split((1, 2), dim=0)
        .split((3, 5), dim=-2)
        .permute(0, 2, 1)
        .compile(source.shape)
    )
    packed = plan.pack(source)
    assert torch.equal(packed.sort().values, source.flatten())
    torch.testing.assert_close(plan.unpack(packed), source)
    # Also exercises several source roots when chaining after legacy rows().
    rows = MatrixLayout.rows(3, 5).permute(1, 0).compile((2, 8, 12))
    expert = torch.arange(2 * 8 * 12).view(2, 8, 12)
    torch.testing.assert_close(rows.unpack(rows.pack(expert)), expert)


@pytest.mark.parametrize(
    "shape,layout",
    [
        ((24, 8), MatrixLayout.view(2, 3, 4, 8).permute(1, 0, 2, 3).reshape(3, 8, 8)),
        ((16, 8), MatrixLayout.view(2, 4, 2, 8).split((2, 1, 1), dim=1).reshape(-1, 8)),
        ((8, 24), MatrixLayout.view(8, 3, 8).permute(1, 2, 0)),
        ((2, 8, 8), MatrixLayout.view(2, 8, 2, 4).permute(0, 2, 3, 1)),
    ],
)
def test_common_mappings_copy_directly_without_materializing_intermediate_tensors(
    monkeypatch, shape, layout
):
    source = torch.arange(torch.tensor(shape).prod()).view(shape)
    plan = layout.compile(shape)
    native_reshape = torch.Tensor.reshape

    def reshape(tensor, *dimensions):
        assert tensor.is_contiguous(), "Unexpected temporary from a noncontiguous reshape"
        return native_reshape(tensor, *dimensions)

    def cat(*_args, **_kwargs):
        raise AssertionError("Unexpected concatenation temporary during layout restoration")

    monkeypatch.setattr(torch.Tensor, "reshape", reshape)
    monkeypatch.setattr(torch, "cat", cat)
    packed = plan.pack(source)
    restored = plan.unpack(packed)
    torch.testing.assert_close(restored, source)


def test_layout_signature_includes_mapping_and_legacy_state_is_stable():
    whole = MatrixLayout.view(3, 8, 8).compile((24, 8))
    interleaved = (
        MatrixLayout.view(2, 3, 4, 8).permute(1, 0, 2, 3).reshape(3, 8, 8).compile((24, 8))
    )
    assert whole.matrices == interleaved.matrices
    assert whole.signature != interleaved.signature
    assert whole.state()["version"] == 1 and "mapping" not in whole.state()
    assert interleaved.state()["version"] == 2 and "mapping" in interleaved.state()
    assert whole.is_contiguous and not interleaved.is_contiguous
    # Compilation stores only operation/shape metadata for multi-billion elements.
    large = MatrixLayout.view(64, 4096, 2, 4096).permute(0, 2, 3, 1)
    assert len(large.compile((64, 4096, 8192)).matrices) == 128


@pytest.mark.parametrize(
    "layout",
    [
        MatrixLayout.view(8, 4).permute(0, 0),
        MatrixLayout.view(8, 4).permute(0, 2),
        MatrixLayout.view(8, 4).permute(0),
        MatrixLayout.view(8, 4).permute(0, True),
        MatrixLayout.view(8, 4).reshape(3, -1),
        MatrixLayout.view(8, 4).reshape(-1),
        MatrixLayout.view(8, 4).reshape(-1, -1),
        MatrixLayout.view(8, 4).split((3, 4), dim=0),
        MatrixLayout.view(8, 4).split((8, 0), dim=0),
        MatrixLayout.view(8, 4).split((), dim=0),
        MatrixLayout.view(8, 4).split((8,), dim=2),
        MatrixLayout.view(8, 4).split((8,), dim=True),
        MatrixLayout.view(8, 4).split((4.0, 4), dim=0),
    ],
)
def test_invalid_layout_operations_rejected_before_execution(layout):
    with pytest.raises(ValueError):
        layout.compile((8, 4))
