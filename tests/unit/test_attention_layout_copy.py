"""Independent attention-model layout semantics on Torch and real CUDA paths."""

import pytest
import torch

from dmuon._core.layout_copy import CopyExecutor, CopyTask, _extension
from tests.helpers.attention_layouts import attention_layout_cases, oracle_scatter


CASES = attention_layout_cases()
DTYPES = [
    (torch.float32, torch.float32),
    (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16),
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
]


@pytest.fixture(params=["cpu", "cuda"])
def copy_device(request, monkeypatch):
    if request.param == "cuda":
        if not torch.cuda.is_available() or _extension() is None:
            pytest.skip("requires CUDA and the optional layout-copy extension")
    monkeypatch.setenv("DMUON_COPY_BACKEND", "cuda" if request.param == "cuda" else "torch")
    monkeypatch.setenv("DMUON_COPY_MAX_CTAS", "8")
    return request.param


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_independent_native_indices_cover_exactly_once(case):
    plan = case.layout.compile(case.shape)
    oracle = case.indices()
    flat = torch.cat([part.flatten() for part in oracle])
    torch.testing.assert_close(flat.sort()[0], torch.arange(flat.numel()), rtol=0, atol=0)
    assert [tuple(part.shape) for part in oracle] == [m.shape for m in plan.matrices]
    # Compare each mapping independently; a matching pack/unpack bug cannot cancel out.
    for mapping, expected in zip(plan.copy_maps, oracle):
        torch.testing.assert_close(mapping.indices(torch.arange(mapping.numel)), expected.flatten())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("sd,dd", DTYPES)
@pytest.mark.parametrize("add", [False, True])
def test_attention_layout_copy_exact(copy_device, case, sd, dd, add):
    device = copy_device
    plan = case.layout.compile(case.shape)
    torch.manual_seed(410)
    source = torch.randn(case.shape, dtype=sd, device=device)
    expected = [part.to(dd) for part in case.blocks(source)]
    outputs = [torch.randn(spec.shape, device=device, dtype=dd) for spec in plan.matrices]
    initial = [x.clone() for x in outputs]
    executor = CopyExecutor()
    executor.run([CopyTask(source, out, m, add=add) for out, m in zip(outputs, plan.copy_maps)])
    for actual, want, before in zip(outputs, expected, initial):
        torch.testing.assert_close(actual, before + want if add else want, rtol=0, atol=0)
    changed = [(x + (index + 1) * 0.125).contiguous() for index, x in enumerate(expected)]
    native_initial = torch.randn(case.shape, dtype=dd, device=device)
    actual = native_initial.clone() if add else torch.full_like(native_initial, float("nan"))
    executor.run(
        [CopyTask(x, actual, m, scatter=True, add=add) for x, m in zip(changed, plan.copy_maps)]
    )
    wanted = oracle_scatter(case, changed, initial=native_initial if add else None)
    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    if device == "cuda":
        assert executor.last_stats["backend"] == "cuda"
        assert executor.last_stats["launches"] == 1
        assert executor.last_stats["ctas"] <= 8
        if case.name.startswith("transpose_tail_") and sd == dd == torch.bfloat16 and not add:
            assert executor.last_stats["transpose_tile"] == int(case.name.rsplit("_", 1)[1])


@pytest.mark.parametrize("seed", range(8))
def test_seeded_model_boundaries_have_independent_oracles(seed):
    for case in attention_layout_cases(seed=seed):
        if case.name.startswith("transpose_tail_"):
            continue  # fixed-size shared-memory tails have their own CUDA tests
        plan = case.layout.compile(case.shape)
        oracle = case.indices()
        for mapping, positions in zip(plan.copy_maps, oracle):
            torch.testing.assert_close(
                mapping.indices(torch.arange(mapping.numel)), positions.flatten()
            )


@pytest.mark.parametrize(
    "case",
    [
        c
        for c in CASES
        if c.name
        in ("k3_q_b_whole", "four_stage_index", "transpose_tail_128", "transpose_tail_256")
    ],
    ids=lambda c: c.name,
)
def test_attention_unaligned_reused_pointers_and_strided_gradients(copy_device, case):
    plan = case.layout.compile(case.shape)
    executor = CopyExecutor()
    for offset in (1, 0, 3):
        storage = torch.randn(
            torch.tensor(case.shape).prod().item() + offset,
            dtype=torch.bfloat16,
            device=copy_device,
        )
        source = storage[offset:].reshape(case.shape)
        expected = [x.clone() for x in case.blocks(source)]
        outputs = [
            torch.empty(x.numel() + offset, dtype=x.dtype, device=x.device)[offset:].view(x.shape)
            for x in expected
        ]
        executor.run(
            [CopyTask(source, out, mapping) for out, mapping in zip(outputs, plan.copy_maps)]
        )
        for actual, want in zip(outputs, expected):
            torch.testing.assert_close(actual, want, rtol=0, atol=0)
    # A noncontiguous native grad must preserve logical element semantics.
    axes = tuple(reversed(range(len(case.shape))))
    source = source.permute(axes).contiguous().permute(axes)
    executor.run([CopyTask(source, out, mapping) for out, mapping in zip(outputs, plan.copy_maps)])
    for actual, want in zip(outputs, case.blocks(source)):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)


def test_real_model_shapes_compile_without_materializing_weights():
    for case in attention_layout_cases("metadata"):
        compiled = case.layout.compile(case.shape)
        assert sum(m.numel for m in compiled.matrices) == torch.tensor(case.shape).prod().item()
        if case.name == "stacked_latent_moe":
            assert len(compiled.matrices) == 1792
            assert compiled.matrices[-1].offset > 2**32


@pytest.mark.parametrize("budget", [1, 8, 132])
def test_attention_mixed_batch_on_a_consumer_stream(copy_device, monkeypatch, budget):
    monkeypatch.setenv("DMUON_COPY_MAX_CTAS", str(budget))
    cases = [
        c
        for c in CASES
        if c.name
        in ("dsa_fused_wk_weights", "k3_q_b_whole", "transpose_tail_128", "four_stage_index")
    ]
    tasks, pairs = [], []
    for case in cases:
        source = torch.randn(case.shape, device=copy_device, dtype=torch.bfloat16)
        for spec, mapping, want in zip(
            case.layout.compile(case.shape).matrices,
            case.layout.compile(case.shape).copy_maps,
            case.blocks(source),
        ):
            out = torch.full(spec.shape, float("nan"), device=copy_device, dtype=torch.float32)
            tasks.append(CopyTask(source, out, mapping))
            pairs.append((out, want.float()))
    executor = CopyExecutor()
    if copy_device == "cuda":
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            executor.run(tasks)
            ready = producer.record_event()
        with torch.cuda.stream(consumer):
            consumer.wait_event(ready)
            copies = [(a.clone(), b.clone()) for a, b in pairs]
        consumer.synchronize()
        assert executor.last_stats["path"] == "packed16-widen"
        assert executor.last_stats["ctas"] <= budget
    else:
        executor.run(tasks)
        copies = pairs
    for a, b in copies:
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("side", [128, 256, 64])
def test_attention_sanitizer_transpose_paths(copy_device, side):
    """Bounded representative selector for memcheck/racecheck/synccheck."""
    case = next(c for c in CASES if c.name == f"transpose_tail_{side}")
    plan = case.layout.compile(case.shape)
    source = torch.randn(case.shape, device=copy_device, dtype=torch.bfloat16)
    values = [x.contiguous() for x in case.blocks(source)]
    target = torch.full_like(source, float("nan"))
    executor = CopyExecutor()
    executor.run([CopyTask(x, target, m, scatter=True) for x, m in zip(values, plan.copy_maps)])
    torch.testing.assert_close(target, source, rtol=0, atol=0)
