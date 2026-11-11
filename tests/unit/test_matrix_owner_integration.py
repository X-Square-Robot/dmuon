"""Real API/hooks/Muon math on CPU; device streams and transport are stubbed.

This checks the module seams without claiming distributed CUDA coverage.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import dmuon
from dmuon import MatrixLayout
from copy import deepcopy


class Event:
    def __init__(self, *args, **kwargs):
        pass

    def record(self, *args):
        pass

    def query(self):
        return True


class Stream:
    def __init__(self, *args, **kwargs):
        pass

    def wait_stream(self, *args):
        pass

    def wait_event(self, *args):
        pass

    def record_event(self):
        return Event()


class Mesh:
    mesh_dim_names = ("shard",)
    device_type = "cpu"

    def size(self):
        return 1

    def get_group(self):
        return self

    def rank(self):
        return 0


class Model(nn.Module):
    def __init__(self, weight, column=False):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.column = column

    def forward(self, hidden):
        if self.column:
            if self.weight.ndim == 3:
                return torch.einsum("bh,ehi->bei", hidden, self.weight).sum(1)
            return hidden @ self.weight
        if self.weight.ndim == 3:
            return torch.einsum("bh,eih->bei", hidden, self.weight).sum(1)
        return nn.functional.linear(hidden, self.weight)


@pytest.fixture
def cpu_transport(monkeypatch):
    runtime = SimpleNamespace(
        Stream=Stream,
        Event=Event,
        current_stream=lambda: Stream(),
        stream=lambda _stream: nullcontext(),
    )
    for path in [
        "dmuon._core.comm.get_device_module",
        "dmuon._backends.fsdp2.group.get_device_module",
        "dmuon.optim.muon.get_device_module",
        "dmuon.utils.get_device_module",
    ]:
        monkeypatch.setattr(path, lambda _device: runtime)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda _tensor, _stream: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda _group, rank: rank)
    monkeypatch.setattr(torch.distributed, "_coalescing_manager", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(torch.distributed, "reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_args, **_kwargs: None)
    broadcasts = []
    monkeypatch.setattr(
        torch.distributed, "broadcast", lambda tensor, **_kwargs: broadcasts.append(tensor.numel())
    )
    return broadcasts


@pytest.mark.parametrize(
    "shape,layout,column",
    [
        ((12, 8), MatrixLayout.whole(), False),
        ((12, 8), MatrixLayout.view(3, 4, 8), False),
        ((12, 8), MatrixLayout.view(3, 2, 2, 8), False),
        ((12, 8), MatrixLayout.rows(8, 2, 2), False),
        ((3, 8, 8), MatrixLayout.view(3, 2, 4, 8), False),
        ((24, 8), MatrixLayout.view(2, 3, 4, 8).permute(1, 0, 2, 3).reshape(3, 8, 8), False),
        ((24, 8), MatrixLayout.view(2, 3, 4, 8).permute(1, 0, 2, 3), False),
        ((16, 8), MatrixLayout.view(2, 4, 2, 8).split((2, 1, 1), dim=1).reshape(-1, 8), False),
        ((8, 24), MatrixLayout.view(8, 3, 8).permute(1, 2, 0), True),
        ((2, 8, 8), MatrixLayout.view(2, 8, 2, 4).permute(0, 2, 3, 1), True),
    ],
)
@pytest.mark.parametrize("reshard", [True, False])
def test_original_model_api_backward_and_actual_muon(cpu_transport, shape, layout, column, reshard):
    torch.manual_seed(11)
    model = Model(torch.randn(shape), column=column)
    reference = deepcopy(model)
    original_names = list(dict(model.named_parameters()))
    modules = [(name, type(module)) for name, module in model.named_modules()]
    result = dmuon.dedicate_params(
        model,
        Mesh(),
        predicate=lambda _n, _p: True,
        reshard_after_forward=reshard,
        compute_dtype=torch.float32,
        process_group_policy="shared",
        param_policy={"defaults": {"route": "muon", "matrix_layout": layout}},
    )
    assert len(result.matrix_owners) == len(layout.compile(shape).matrices)
    assert list(dict(model.named_parameters())) == original_names
    assert [(name, type(module)) for name, module in model.named_modules()] == modules
    optimizer = dmuon.Muon(
        model, lr=0.01, momentum=0.0, weight_decay=0.0, first_step_progress_log=False
    )
    hidden = torch.randn(3, 8, requires_grad=True)
    ref_hidden = hidden.detach().clone().requires_grad_()
    actual, expected = model(hidden), reference(ref_hidden)
    torch.testing.assert_close(actual, expected)
    for state in model._dedicated_comm_ctx.all_states:
        matrices = state.group._matrix_params
        for parent, packed in matrices.packed_buffers.items():
            if packed is not matrices.buffers[parent]:
                # Packing storage is temporary even with reshard_after_forward=False.
                assert packed.untyped_storage().nbytes() == 0
            if all(m.contiguous_offset is not None for m in parent.layout.copy_maps):
                assert packed is matrices.buffers[parent]
    actual.square().sum().backward()
    expected.square().sum().backward()
    dmuon.prepare_muon_grads(model)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad)
    plan = layout.compile(shape)
    packed_gradient = plan.pack(reference.weight.grad)
    for unit in dmuon.get_dedicated_params(model):
        spec = unit.matrix_spec
        expected_grad = packed_gradient[spec.offset : spec.offset + spec.numel].view(spec.shape)
        torch.testing.assert_close(unit._reduced_grad, expected_grad)
    optimizer.step()
    with torch.no_grad():
        packed_weight = plan.pack(reference.weight).clone()
        for spec in plan.matrices:
            value = packed_weight[spec.offset : spec.offset + spec.numel].view(spec.shape)
            grad = packed_gradient[spec.offset : spec.offset + spec.numel].view(spec.shape)
            value.add_(dmuon.newton_schulz(grad), alpha=-0.01 * 0.2 * max(spec.shape) ** 0.5)
        reference.weight.copy_(plan.unpack(packed_weight))
    torch.testing.assert_close(model(hidden), reference(hidden))
    assert cpu_transport


@pytest.mark.parametrize(
    "layout",
    [
        MatrixLayout.view(3, 4, 8),
        MatrixLayout.view(2, 3, 2, 8).permute(1, 0, 2, 3).reshape(3, 4, 8),
    ],
)
def test_no_sync_and_original_parameter_semantic_groups(cpu_transport, layout):
    model = Model(torch.randn(12, 8))
    reference = deepcopy(model)
    dmuon.dedicate_params(
        model,
        Mesh(),
        predicate=lambda _n, _p: True,
        process_group_policy="shared",
        param_policy_fn=lambda _n, _p: {"matrix_layout": layout},
    )
    optimizer = dmuon.Muon(
        model,
        lr=0.1,
        momentum=0,
        first_step_progress_log=False,
        param_groups=[{"params": [model.weight], "lr": 0.007, "weight_decay": 0}],
    )
    for microbatch in range(2):
        hidden = torch.randn(3, 8, requires_grad=True)
        with dmuon.no_sync(model) if microbatch == 0 else nullcontext():
            model(hidden).sum().backward()
        reference(hidden).sum().backward()
    dmuon.prepare_muon_grads(model)
    plan = layout.compile(reference.weight.shape)
    gradient = plan.pack(reference.weight.grad)
    for unit in dmuon.get_dedicated_params(model):
        spec = unit.matrix_spec
        torch.testing.assert_close(
            unit._reduced_grad, gradient[spec.offset : spec.offset + spec.numel].view(spec.shape)
        )
    optimizer.step()
    with torch.no_grad():
        packed = plan.pack(reference.weight).clone()
        for spec in plan.matrices:
            value = packed[spec.offset : spec.offset + spec.numel].view(spec.shape)
            grad = gradient[spec.offset : spec.offset + spec.numel].view(spec.shape)
            value.add_(dmuon.newton_schulz(grad), alpha=-0.007 * 0.2 * max(spec.shape) ** 0.5)
        reference.weight.copy_(plan.unpack(packed))
    torch.testing.assert_close(model(hidden), reference(hidden))
    optimizer.zero_grad()
    assert all(
        unit._reduced_grad is None and unit._accumulated_grad is None
        for unit in dmuon.get_dedicated_params(model)
    )


class MoE(nn.Module):
    def __init__(self, column=False):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(2, 8, 8) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(2, 8, 4) * 0.1)
        self.router = nn.Linear(8, 8, bias=False)
        self.column = column
        if column:
            self.gate_up_proj = nn.Parameter(self.gate_up_proj.transpose(1, 2).contiguous())
            self.down_proj = nn.Parameter(self.down_proj.transpose(1, 2).contiguous())

    def forward(self, hidden):
        if self.column:
            gate, up = torch.einsum("bh,ehi->bei", hidden, self.gate_up_proj).chunk(2, dim=-1)
            return torch.einsum(
                "bei,eih->beh", torch.nn.functional.silu(gate) * up, self.down_proj
            ).sum(1) + self.router(hidden)
        gate, up = torch.einsum("bh,eih->bei", hidden, self.gate_up_proj).chunk(2, dim=-1)
        return torch.einsum(
            "bei,ehi->beh", torch.nn.functional.silu(gate) * up, self.down_proj
        ).sum(1) + self.router(hidden)


def _moe_layouts(model):
    if model.column:
        return [
            ("gate_up_proj", MatrixLayout.view(2, 8, 2, 4).permute(0, 2, 3, 1)),
            ("down_proj", MatrixLayout.view(2, 4, 8).permute(0, 2, 1)),
        ]
    return [
        ("gate_up_proj", MatrixLayout.view(2, 2, 4, 8)),
        ("down_proj", MatrixLayout.view(2, 8, 4)),
    ]


def _install_moe(model):
    layouts = dict(_moe_layouts(model))
    return dmuon.dedicate_params(
        model,
        Mesh(),
        predicate=lambda _n, _p: True,
        hook_boundary_predicate=lambda module: isinstance(module, MoE),
        process_group_policy="shared",
        param_policy={
            "defaults": {"route": "adamw"},
            "overrides": [
                {
                    "name_glob": "gate_up_proj",
                    "route": "muon",
                    "matrix_layout": layouts["gate_up_proj"],
                },
                {
                    "name_glob": "down_proj",
                    "route": "muon",
                    "matrix_layout": layouts["down_proj"],
                },
            ],
        },
    )


@pytest.mark.parametrize("column", [False, True])
def test_unmodified_moe_with_muon_and_adamw(cpu_transport, column):
    model = MoE(column=column)
    reference = deepcopy(model)
    modules = list(model.named_modules())
    _install_moe(model)
    assert list(model.named_modules()) == modules
    optimizer = dmuon.Muon(
        model,
        lr=0.01,
        momentum=0,
        adamw_lr=0.003,
        adamw_weight_decay=0,
        first_step_progress_log=False,
    )
    ref_adamw = torch.optim.AdamW(reference.router.parameters(), lr=0.003, weight_decay=0)
    hidden = torch.randn(3, 8, requires_grad=True)
    actual, expected = model(hidden), reference(hidden)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    optimizer.step()
    with torch.no_grad():
        for name, layout in _moe_layouts(reference):
            parameter = getattr(reference, name)
            plan = layout.compile(parameter.shape)
            packed, gradient = plan.pack(parameter).clone(), plan.pack(parameter.grad)
            for spec in plan.matrices:
                value = packed[spec.offset : spec.offset + spec.numel].view(spec.shape)
                grad = gradient[spec.offset : spec.offset + spec.numel].view(spec.shape)
                value.add_(dmuon.newton_schulz(grad), alpha=-0.01 * 0.2 * max(spec.shape) ** 0.5)
            parameter.copy_(plan.unpack(packed))
    ref_adamw.step()
    torch.testing.assert_close(model(hidden), reference(hidden))
    state = dmuon.get_model_state_dict(model)
    assert set(state) == set(reference.state_dict())
    for name, value in state.items():
        torch.testing.assert_close(value, reference.state_dict()[name])


def _installed_model(layout, shape=(3, 8, 8)):
    model = Model(torch.randn(shape))
    dmuon.dedicate_params(
        model,
        Mesh(),
        lambda _n, _p: True,
        process_group_policy="shared",
        param_policy_fn=lambda _n, _p: {"matrix_layout": layout},
    )
    optimizer = dmuon.Muon(model, lr=0.01, momentum=0.9, first_step_progress_log=False)
    return model, optimizer


@pytest.mark.parametrize(
    "layout",
    [
        MatrixLayout.view(3, 2, 4, 8),
        MatrixLayout.view(3, 8, 2, 4).permute(0, 2, 3, 1),
        MatrixLayout.view(3, 8, 8).split((4, 4), dim=-2),
    ],
)
def test_original_model_and_unique_matrix_momentum_checkpoint_resume(cpu_transport, layout):
    model, optimizer = _installed_model(layout)
    hidden = torch.randn(3, 8, requires_grad=True)
    model(hidden).square().sum().backward()
    optimizer.step()
    weights = dmuon.get_model_state_dict(model)
    saved = dmuon.get_optimizer_state_dict(model, optimizer)
    assert set(weights) == {"weight"}
    assert weights["weight"].shape == (3, 8, 8)
    assert len(saved["dedicated"]) == 6
    assert all("::matrix[" in key for key in saved["dedicated"])
    assert saved["matrix_layouts"] == {"weight": layout.compile((3, 8, 8)).state()}
    resumed, resumed_optimizer = _installed_model(layout)
    dmuon.set_model_state_dict(resumed, weights)
    dmuon.set_optimizer_state_dict(resumed, resumed_optimizer, saved)
    for current, opt in [(model, optimizer), (resumed, resumed_optimizer)]:
        opt.zero_grad()
        current(hidden).square().sum().backward()
        opt.step()
    torch.testing.assert_close(
        dmuon.get_model_state_dict(model)["weight"], dmuon.get_model_state_dict(resumed)["weight"]
    )
    changed, changed_optimizer = _installed_model(MatrixLayout.view(3, 8, 8))
    dmuon.set_model_state_dict(changed, weights)
    with pytest.raises(ValueError, match="NS boundaries"):
        dmuon.set_optimizer_state_dict(changed, changed_optimizer, saved)
    assert not changed_optimizer.state


def test_model_load_invalidates_live_compute_slot(cpu_transport):
    model, _optimizer = _installed_model(MatrixLayout.view(3, 4, 8), shape=(12, 8))
    hidden = torch.randn(3, 8)
    old = model(hidden).detach().clone()
    dmuon.set_model_state_dict(model, {"weight": torch.zeros(12, 8)})
    assert torch.count_nonzero(old) > 0
    assert torch.count_nonzero(model(hidden)) == 0


@pytest.mark.parametrize("eval_context", [torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize(
    "route,placement,layout",
    [
        ("muon", "broadcast", None),
        ("adamw", "broadcast", None),
        ("sharded_adamw", "broadcast", None),
        ("muon", "all_gather", None),
        ("muon", "broadcast", MatrixLayout.whole()),
        ("muon", "broadcast", MatrixLayout.view(3, 4, 8)),
        (
            "muon", "broadcast",
            MatrixLayout.view(2, 3, 2, 8).permute(1, 0, 2, 3).reshape(3, 4, 8),
        ),
    ],
)
def test_model_load_refreshes_zero2_eval_copies(
    cpu_transport, monkeypatch, eval_context, route, placement, layout
):
    # Real copy-in, placement, checkpoint and forward hooks; only the one-rank
    # transport and device streams are stubbed, like the other integration tests.
    transfers = []

    def gather(output, source, **_kwargs):
        transfers.append("gather")
        output.copy_(source)

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    monkeypatch.setattr(
        torch.distributed, "reduce_scatter_tensor",
        lambda output, source, **_kwargs: output.copy_(source),
    )
    model = Model(torch.ones(12, 8)).eval()
    policy = dict(route=route, param_dtype=torch.bfloat16, grad_dtype=torch.bfloat16,
                  master_dtype=torch.float32)
    if layout is not None:
        policy["matrix_layout"] = layout
    dmuon.dedicate_params(
        model, Mesh(), lambda _n, _p: True, reshard_after_forward=False,
        compute_dtype=torch.bfloat16, process_group_policy="shared",
        muon_forward_unshard=placement, param_policy={"defaults": policy},
    )
    group = model._dedicated_comm_ctx.all_states[0].group
    hidden = torch.ones(3, 8, dtype=torch.bfloat16)
    target = torch.full((12, 8), 2.0)
    with eval_context():
        old = model(hidden).clone()
    assert group._is_unsharded
    dmuon.set_model_state_dict(model, {"weight": target})
    assert not group._is_unsharded
    torch.testing.assert_close(
        dmuon.get_model_state_dict(model, rank0_only=False)["weight"],
        target, rtol=0, atol=0,
    )
    with eval_context():
        actual = model(hidden).clone()
        expected = nn.functional.linear(hidden, target.to(torch.bfloat16))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not torch.equal(actual, old)
        counts = len(cpu_transport), len(transfers)
        torch.testing.assert_close(model(hidden), expected, rtol=0, atol=0)
        assert (len(cpu_transport), len(transfers)) == counts
    assert not model._dedicated_comm_ctx.all_states[0].reshard_after_forward
    group.reshard()


def test_model_load_partial_only_invalidates_affected_group(cpu_transport):
    model = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False)).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.)
    dmuon.dedicate_params(
        model, Mesh(), lambda _n, _p: True, reshard_after_forward=False,
        compute_dtype=torch.bfloat16, process_group_policy="shared",
        hook_boundary_predicate=lambda module: isinstance(module, nn.Linear),
    )
    groups = [state.group for state in model._dedicated_comm_ctx.all_states]
    hidden = torch.ones(2, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        model(hidden)
        assert all(group._is_unsharded for group in groups)
        dmuon.set_model_state_dict(model, {"0.weight": torch.full((8, 8), 2.)})
        assert not groups[0]._is_unsharded
        assert groups[1]._is_unsharded
        count = len(cpu_transport)
        torch.testing.assert_close(model(hidden), torch.full((2, 8), 128., dtype=torch.bfloat16))
        assert len(cpu_transport) == count + 1
        count = len(cpu_transport)
        dmuon.set_model_state_dict(model, {})
        assert all(group._is_unsharded for group in groups)
        model(hidden)
        assert len(cpu_transport) == count
    for group in groups:
        group.reshard()


@pytest.mark.parametrize("layout", [None, MatrixLayout.whole(), MatrixLayout.view(3, 4, 8)])
def test_model_load_reads_live_checkpoint_views_before_releasing_storage(cpu_transport, layout):
    model = Model(torch.ones(12, 8)).eval()
    policy = {} if layout is None else {"matrix_layout": layout}
    dmuon.dedicate_params(
        model, Mesh(), lambda _n, _p: True, reshard_after_forward=False,
        compute_dtype=torch.bfloat16, process_group_policy="shared",
        param_policy={"defaults": policy},
    )
    with torch.no_grad():
        hidden = torch.ones(2, 8, dtype=torch.bfloat16)
        expected = model(hidden).clone()
        # Deliberately no clone: these checkpoint bytes share live storage.
        checkpoint = {"weight": model.weight.detach()}
        for dp in dmuon.get_dedicated_params(model):
            dp._owned_data.fill_(3.)
        dmuon.set_model_state_dict(model, checkpoint)
        torch.testing.assert_close(model(hidden), expected, rtol=0, atol=0)
    model._dedicated_comm_ctx.all_states[0].group.reshard()


def test_model_load_drains_old_publish_and_prefetch_before_writing(cpu_transport, monkeypatch):
    class PendingEvent:
        def __init__(self, callback):
            self.callback = callback

        def finish(self):
            if self.callback is not None:
                callback, self.callback = self.callback, None
                callback()

    def wait_event(_stream, event):
        if isinstance(event, PendingEvent):
            event.finish()

    monkeypatch.setattr(Stream, "wait_event", wait_event)
    model = Model(torch.ones(4, 4)).eval()
    dmuon.dedicate_params(
        model, Mesh(), lambda _n, _p: True, reshard_after_forward=False,
        compute_dtype=torch.bfloat16, process_group_policy="shared",
    )
    group = model._dedicated_comm_ctx.all_states[0].group
    dp = dmuon.get_dedicated_params(model)[0]
    group.unshard(prefetch=True)
    assert group._broadcast_event is not None and not group._is_unsharded
    trace = []

    def finish_old_publish():
        trace.append("publish")
        dp._owned_data.fill_(3.)

    def finish_old_prefetch():
        trace.append("prefetch")
        assert torch.all(dp._owned_data == 3.)  # checkpoint has not overwritten it yet

    group._replicate_broadcast_state = SimpleNamespace(
        event=PendingEvent(finish_old_publish), replicate_input=dp._owned_data,
    )
    group._broadcast_event = PendingEvent(finish_old_prefetch)
    dmuon.set_model_state_dict(model, {"weight": torch.full((4, 4), 2.)})
    assert trace == ["publish", "prefetch"]
    assert group._replicate_broadcast_state is None and group._broadcast_event is None
    assert torch.all(dp._owned_data == 2.)
    with torch.no_grad():
        torch.testing.assert_close(
            model(torch.ones(2, 4, dtype=torch.bfloat16)),
            torch.full((2, 4), 8., dtype=torch.bfloat16), rtol=0, atol=0,
        )
    group.reshard()


def test_layout_clip_counts_each_gradient_once(cpu_transport):
    model, optimizer = _installed_model(MatrixLayout.view(3, 4, 8), shape=(12, 8))
    reference = dmuon.get_model_state_dict(model)["weight"].requires_grad_()
    hidden = torch.randn(3, 8, requires_grad=True)
    model(hidden).square().sum().backward()
    nn.functional.linear(hidden, reference).square().sum().backward()
    stats = dmuon.clip_grad_norm_(optimizer, None, foreach=False)
    assert stats.param_count == 3
    assert stats.total_norm == pytest.approx(float(reference.grad.norm()), rel=1e-5)


def test_tied_original_attributes_remain_aliased_and_checkpoint_keys_preserved(cpu_transport):
    class Tied(Model):
        def __init__(self):
            super().__init__(torch.randn(12, 8))
            self.alias = self.weight

        def forward(self, hidden):
            assert self.weight is self.alias
            return nn.functional.linear(hidden, self.weight) + nn.functional.linear(
                hidden, self.alias
            )

    model = Tied()
    expected = deepcopy(model)
    names = set(model.state_dict())
    dmuon.dedicate_params(
        model,
        Mesh(),
        lambda _n, _p: True,
        process_group_policy="shared",
        param_policy_fn=lambda _n, _p: {"matrix_layout": MatrixLayout.view(3, 4, 8)},
    )
    hidden = torch.randn(3, 8, requires_grad=True)
    torch.testing.assert_close(model(hidden), expected(hidden))
    state = dmuon.get_model_state_dict(model)
    assert set(state) == names
    torch.testing.assert_close(state["weight"], state["alias"])


@pytest.mark.parametrize(
    "layout, message",
    [
        (MatrixLayout.view(2, 8, 8), "exactly cover"),
        (MatrixLayout.view(3, 4, 8).permute(0, 0, 2), "every axis"),
        (MatrixLayout.view(12, 8).split((5, 6), dim=0), "exactly cover"),
    ],
)
def test_bad_layout_fails_before_model_attribute_replacement(cpu_transport, layout, message):
    model = Model(torch.randn(12, 8))
    original = model.weight
    with pytest.raises(ValueError, match=message):
        dmuon.dedicate_params(
            model,
            Mesh(),
            lambda _n, _p: True,
            process_group_policy="shared",
            param_policy_fn=lambda _n, _p: {"matrix_layout": layout},
        )
    assert model.weight is original


@pytest.mark.parametrize(
    "layout",
    [
        MatrixLayout.view(3, 4, 8),
        MatrixLayout.view(2, 3, 2, 8).permute(1, 0, 2, 3).reshape(3, 4, 8),
    ],
)
def test_returned_plan_does_not_retain_original_full_parameter_storage(cpu_transport, layout):
    import gc
    import weakref

    model = Model(torch.randn(12, 8))
    original = weakref.ref(model.weight)
    plan = dmuon.dedicate_params(
        model,
        Mesh(),
        lambda _n, _p: True,
        process_group_policy="shared",
        param_policy_fn=lambda _n, _p: {"matrix_layout": layout},
    )
    gc.collect()
    assert original() is None
    assert plan and len(plan) == 3
    assert len(list(plan.items())) == 3
    assert all(matrix.fqn == "weight" and matrix.shape == (4, 8) for matrix in plan.matrix_owners)


@pytest.mark.parametrize("route,placement", [("adamw", "broadcast"), ("muon", "all_gather")])
def test_incompatible_layout_policy_rejected_before_wrapping(cpu_transport, route, placement):
    model = Model(torch.randn(12, 8))
    original = model.weight
    with pytest.raises(ValueError):
        dmuon.dedicate_params(
            model,
            Mesh(),
            lambda _n, _p: True,
            process_group_policy="shared",
            muon_forward_unshard=placement,
            param_policy_fn=lambda _n, _p: {
                "route": route,
                "matrix_layout": MatrixLayout.view(3, 4, 8),
            },
        )
    assert model.weight is original


def test_layout_callback_receives_original_shape_and_runs_once(cpu_transport):
    model = Model(torch.randn(3, 8, 8))
    predicates, policies = [], []

    def predicate(name, parameter):
        predicates.append((name, tuple(parameter.shape)))
        return True

    def policy(name, parameter):
        policies.append((name, tuple(parameter.shape)))
        experts, rows, hidden = parameter.shape
        return {"route": "muon", "matrix_layout": MatrixLayout.view(experts, 2, rows // 2, hidden)}

    dmuon.dedicate_params(
        model, Mesh(), predicate, process_group_policy="shared", param_policy_fn=policy
    )
    assert predicates == policies == [("weight", (3, 8, 8))]


def test_optimizer_checkpoint_rejects_same_shape_different_element_mapping(cpu_transport):
    ordinary = MatrixLayout.view(3, 4, 8)
    interleaved = MatrixLayout.view(2, 3, 2, 8).permute(1, 0, 2, 3).reshape(3, 4, 8)
    model, optimizer = _installed_model(ordinary, shape=(12, 8))
    saved = dmuon.get_optimizer_state_dict(model, optimizer)
    changed, changed_optimizer = _installed_model(interleaved, shape=(12, 8))
    assert ordinary.compile((12, 8)).matrices == interleaved.compile((12, 8)).matrices
    with pytest.raises(ValueError, match="NS boundaries"):
        dmuon.set_optimizer_state_dict(changed, changed_optimizer, saved)
    assert not changed_optimizer.state


@pytest.mark.parametrize("identity_operations", [False, True])
def test_contiguous_expert_gradients_are_borrowed_without_copy(
    cpu_transport, monkeypatch, identity_operations
):
    from dmuon._backends.fsdp2.matrix_group import MatrixParamGroup

    layout = MatrixLayout.view(3, 2, 4, 8)
    if identity_operations:
        layout = layout.permute(1, 0, 2, 3).permute(1, 0, 2, 3)
    model, _optimizer = _installed_model(layout)
    split = MatrixParamGroup.split_gradients
    captured = []

    def inspect(group):
        original = {p: p._unsharded_param.grad for p in group.entries}
        split(group)
        for parent, units in group.entries.items():
            gradient = original[parent]
            for unit in units:
                assert (
                    unit._layout_grad.untyped_storage().data_ptr()
                    == gradient.untyped_storage().data_ptr()
                )
                assert (
                    unit._layout_grad.storage_offset()
                    == gradient.storage_offset() + unit.matrix_spec.offset
                )
                captured.append(unit)
        assert group.copy_executor.last_stats["tasks"] == 0

    monkeypatch.setattr(MatrixParamGroup, "split_gradients", inspect)
    model(torch.randn(3, 8, requires_grad=True)).square().sum().backward()
    assert len(captured) == 6


@pytest.mark.parametrize(
    "layout",
    [
        MatrixLayout.view(2, 3, 2, 8).permute(1, 0, 2, 3).reshape(3, 4, 8),
        MatrixLayout.view(12, 8).permute(1, 0).reshape(3, 4, 8),
    ],
)
def test_runtime_does_not_materialize_full_packed_gradients(cpu_transport, monkeypatch, layout):
    from dmuon.matrix_layout import CompiledMatrixLayout

    model, optimizer = _installed_model(layout, shape=(12, 8))
    reference = Model(dmuon.get_model_state_dict(model)["weight"])
    hidden = torch.randn(3, 8, requires_grad=True)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Runtime must copy directly between final buffers")

    with monkeypatch.context() as patch:
        patch.setattr(CompiledMatrixLayout, "pack", forbidden)
        patch.setattr(CompiledMatrixLayout, "unpack", forbidden)
        torch.testing.assert_close(model(hidden), reference(hidden))
        model(hidden).square().sum().backward()
        reference(hidden).square().sum().backward()
        dmuon.prepare_muon_grads(model)
        packed = layout.compile((12, 8)).copy_maps
        for unit, mapping in zip(
            sorted(dmuon.get_dedicated_params(model), key=lambda p: p.matrix_spec.index), packed
        ):
            expected = reference.weight.grad.flatten()[mapping.indices(torch.arange(mapping.numel))]
            torch.testing.assert_close(unit._reduced_grad, expected.view(mapping.shape))


@pytest.mark.parametrize("interleaved", [False, True])
def test_gradient_mapping_casts_directly_to_reduction_dtype(cpu_transport, interleaved):
    layout = MatrixLayout.view(3, 4, 8)
    if interleaved:
        layout = MatrixLayout.view(2, 3, 2, 8).permute(1, 0, 2, 3).reshape(3, 4, 8)
    model = Model(torch.randn(12, 8))
    reference = deepcopy(model).to(dtype=torch.bfloat16)
    dmuon.dedicate_params(
        model,
        Mesh(),
        lambda _n, _p: True,
        process_group_policy="shared",
        param_policy_fn=lambda _n, _p: {
            "matrix_layout": layout,
            "param_dtype": torch.bfloat16,
            "grad_dtype": torch.float32,
        },
    )
    hidden = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    model(hidden).square().sum().backward()
    reference(hidden).square().sum().backward()
    dmuon.prepare_muon_grads(model)
    expected = layout.compile((12, 8)).pack(reference.weight.grad).float()
    for unit in dmuon.get_dedicated_params(model):
        spec = unit.matrix_spec
        assert unit._reduced_grad.dtype == torch.float32
        torch.testing.assert_close(
            unit._reduced_grad, expected[spec.offset : spec.offset + spec.numel].view(spec.shape)
        )
