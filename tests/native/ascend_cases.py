"""Small, deterministic cases shared by native Ascend and migration tests."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch import nn


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def tensor_deltas(actual, expected, *, path="root"):
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return [{"path": path, "actual_shape": list(actual.shape),
                     "expected_shape": list(expected.shape),
                     "actual_dtype": str(actual.dtype), "expected_dtype": str(expected.dtype)}]
        if torch.allclose(actual, expected, rtol=0, atol=0, equal_nan=True):
            return []
        delta = actual.float() - expected.float()
        return [{"path": path, "dtype": str(actual.dtype), "shape": list(actual.shape),
                 "max_abs_diff": float(delta.abs().max()),
                 "relative_l2_diff": float(delta.norm() / expected.float().norm().clamp_min(1e-30))}]
    if isinstance(expected, dict) and isinstance(actual, dict):
        return [item for key in expected.keys() & actual.keys()
                for item in tensor_deltas(actual[key], expected[key], path=f"{path}.{key}")]
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return [item for index, (left, right) in enumerate(zip(actual, expected))
                for item in tensor_deltas(left, right, path=f"{path}[{index}]")]
    return []


def master_snapshot(dmuon, model):
    module_names = {id(module): name for name, module in model.named_modules()}
    return {
        f"{module_names[id(dp.module)]}.{dp.param_name}": {
            "original_dtype": str(dp._orig_dtype), "master_dtype": str(dp._master_dtype),
            "owned_data": cpu_tree(getattr(dp, "_owned_data", None)),
            "sharded_adamw_data": cpu_tree(getattr(dp, "_sharded_adamw_data", None)),
        }
        for dp in dmuon.get_dedicated_params(model)
    }


def ns_cases():
    shapes = [(16, 16), (32, 8), (8, 32), (1, 16), (16, 1)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for shape in shapes:
            for family in ("random", "zero", "rank1"):
                yield shape, dtype, family
    # Representative attention and MLP projection shapes.
    for shape in ((1024, 1024), (3072, 1024), (1024, 3072)):
        yield shape, torch.bfloat16, "random"


def run_ns(dmuon, device, shape, dtype, family):
    generator = torch.Generator().manual_seed(10233)
    if family == "zero":
        matrix = torch.zeros(shape)
    elif family == "rank1":
        matrix = torch.randn(shape[0], 1, generator=generator) @ torch.randn(
            1, shape[1], generator=generator
        )
    else:
        matrix = torch.randn(shape, generator=generator)
    matrix = matrix.to(device=device, dtype=dtype)
    output = dmuon.NewtonSchulz().local(matrix, 5)
    assert output.shape == matrix.shape
    assert output.dtype == matrix.dtype
    assert torch.isfinite(output).all()
    if family == "zero":
        assert torch.count_nonzero(output) == 0
    return cpu_tree(output)


def run_clip(dmuon, device, value, max_norm, *, segmented, dtype=torch.float32):
    grad = torch.tensor(value, device=device, dtype=dtype)
    original = grad.clone()
    if segmented:
        result = dmuon.clip_grad_norm_buckets_(
            [dmuon.GradClipBucket("muon", [grad], reduce=False),
             dmuon.GradClipBucket("empty", [], reduce=False)], max_norm,
        )
        assert not result.fastpath
        stats = result.stats_by_name["muon"]
    else:
        dp = SimpleNamespace(_reduced_grad=grad, param_name="proj", is_dtensor=False)
        optimizer = SimpleNamespace(
            _dedicated_params=[dp], _comm_ctx=SimpleNamespace(device=device),
            _ensure_grads_ready=lambda: None,
        )
        stats = dmuon.clip_grad_norm_(optimizer, max_norm)
    # Stats-only calls retain Python coefficient/flag unions on the public
    # Muon API; the segmented API returns device tensors for those fields.
    assert isinstance(stats.total_norm, torch.Tensor) and stats.total_norm.ndim == 0
    assert isinstance(stats.found_inf, torch.Tensor) and stats.found_inf.ndim == 0
    assert isinstance(stats.clip_coef, (torch.Tensor, float))
    assert isinstance(stats.clipped, (torch.Tensor, bool))
    assert stats.total_norm.device == device
    assert stats.found_inf.device == device
    assert stats.param_count == 1
    if max_norm is None:
        torch.testing.assert_close(grad, original, rtol=0, atol=0, equal_nan=True)
        assert not bool(stats.clipped)
    assert bool(stats.found_inf) == (not bool(torch.isfinite(original).all()))
    return {"grad": cpu_tree(grad), "stats": cpu_tree(stats.as_dict()),
            "field_types": {name: type(getattr(stats, name)).__name__
                            for name in ("total_norm", "clip_coef", "clipped", "found_inf")}}


def run_empty_clip(dmuon, device):
    optimizer = SimpleNamespace(
        _dedicated_params=[], _comm_ctx=SimpleNamespace(device=device),
        _ensure_grads_ready=lambda: None,
    )
    stats = dmuon.clip_grad_norm_(optimizer, None)
    assert stats.total_norm.device == device
    assert stats.total_norm.item() == 0 and stats.param_count == 0
    assert stats.clip_coef == 1.0 and stats.clipped is False
    assert not bool(stats.found_inf)
    segmented = dmuon.clip_grad_norm_buckets_([], None)
    assert segmented.total_norm.device == device
    assert segmented.total_norm.item() == 0 and segmented.bucket_stats == ()
    return {"muon": cpu_tree(stats.as_dict()), "segmented": cpu_tree(segmented.total_norm)}


def run_nonfinite_clip_error(dmuon, device, value, *, dtype=torch.float32):
    dp = SimpleNamespace(_reduced_grad=torch.tensor([value], device=device, dtype=dtype),
                         param_name="proj", is_dtensor=False)
    optimizer = SimpleNamespace(
        _dedicated_params=[dp], _comm_ctx=SimpleNamespace(device=device),
        _ensure_grads_ready=lambda: None,
    )
    try:
        dmuon.clip_grad_norm_(optimizer, 1.0, error_if_nonfinite=True)
    except RuntimeError as error:
        assert "non-finite" in str(error)
        return True
    raise AssertionError("non-finite gradient was accepted")


def run_stream_lifetime(dmuon, device):
    context = dmuon.DedicatedCommContext(device)
    runtime = getattr(context, "_device_module", torch.cuda)
    consumer = runtime.current_stream(device)
    outputs = []
    for value in range(8):
        with runtime.stream(context.broadcast_stream):
            temporary = torch.full((1024,), float(value), device=device)
            produced = temporary + 1
            event = runtime.Event()
            event.record(context.broadcast_stream)
        consumer.wait_event(event)
        produced.record_stream(consumer)
        output = produced.clone()
        del temporary, produced
        # Reuse allocator-sized buffers while the consumer retains its own
        # asynchronous work, then check the values after synchronization.
        scratch = [torch.empty(1024, device=device) for _ in range(8)]
        outputs.append(output)
        del scratch
    runtime.synchronize(device)
    expected = torch.arange(1, 9, dtype=torch.float32).unsqueeze(1).expand(8, 1024)
    result = cpu_tree(torch.stack(outputs))
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    return result


class Projection(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8, dtype=torch.float32) * 0.05)
        self.compute_dtype = dtype
        self.seen_dtypes = None

    def forward(self, inputs, *, options):
        x, index = inputs
        self.seen_dtypes = (str(x.dtype), str(options["nested"][0].dtype), str(index.dtype))
        assert index.dtype == torch.int64
        assert options["nested"][1][1] == "unchanged"
        # DDP retains FP32 parameter storage; autocast supplies its compute
        # precision. FSDP2 additionally materializes the policy dtype.
        with torch.autocast(device_type=x.device.type, dtype=self.compute_dtype,
                            enabled=self.compute_dtype != torch.float32):
            return (x + options["nested"][0] * 0.125) @ self.weight.T


class TinyModel(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.proj = Projection(dtype)
        self.compute_dtype = dtype
        self.owner = Projection(torch.float32)
        self.sharded = Projection(torch.float32)
        self.norm = nn.LayerNorm(8, dtype=torch.float32)
        self.head = nn.Linear(8, 1, dtype=torch.float32)
        self.optional_bias = nn.Parameter(torch.full((1,), 0.01, dtype=torch.float32))
        self.unused = nn.Parameter(torch.ones(3, dtype=torch.float32))

    def forward(self, inputs, *, options):
        if options["use_muon"]:
            projected = self.proj(inputs, options=options)
        else:
            projected = inputs[0].to(self.compute_dtype)
        fp32 = projected.float()
        owner_options = {**options, "nested": [fp32, options["nested"][1]]}
        owner_inputs = (fp32, inputs[1])
        owner = self.owner(owner_inputs, options=owner_options)
        sharded = self.sharded(owner_inputs, options=owner_options)
        projected = projected + ((owner + sharded) * 0.03125).to(projected.dtype)
        with torch.autocast(device_type=projected.device.type, dtype=self.compute_dtype,
                            enabled=self.compute_dtype != torch.float32):
            output = self.head(self.norm(projected))
        if options["use_extra"]:
            output = output + self.optional_bias
        return output.float().square().mean()


def build_model(dmuon, mesh, device, dtype, *, cast, pathway, asynchronous,
                sharded_adamw=False, grad_dtype=None):
    torch.manual_seed(10233)
    model = TinyModel(dtype).to(device)
    dedicate = dmuon.dedicate_params_ddp if pathway == "ddp" else dmuon.dedicate_params
    policy = {"param_dtype": dtype, "grad_dtype": dtype if grad_dtype is None else grad_dtype,
              "master_dtype": torch.float32, "optim_dtype": torch.float32,
              "cast_forward_inputs": cast}
    routes = {"proj.weight": "muon", "owner.weight": "adamw",
              "sharded.weight": "sharded_adamw" if pathway == "fsdp2" and sharded_adamw else "adamw"}

    def parameter_policy(name, param):
        return {**policy, "route": routes[name],
                "param_dtype": dtype if name == "proj.weight" else torch.float32,
                "grad_dtype": policy["grad_dtype"] if name == "proj.weight" else torch.float32}

    dedicate(model, mesh, predicate=lambda name, param: name in routes,
             param_policy_fn=parameter_policy)
    module_names = {id(module): name for name, module in model.named_modules()}
    resolved = {f"{module_names[id(dp.module)]}.{dp.param_name}": dp._dmuon_route
                for dp in dmuon.get_dedicated_params(model)}
    assert resolved == routes
    if pathway == "ddp":
        dmuon.replicate(model, mesh=mesh)
    else:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        precision = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
        fully_shard(model.norm, mesh=mesh, mp_policy=precision)
        fully_shard(model.head, mesh=mesh, mp_policy=precision)
        fully_shard(model, mesh=mesh)
    optimizer = dmuon.Muon(
        model, lr=0.02, adamw_lr=1e-3, ns_steps=5, weight_decay=0.01,
        replicate_async=asynchronous, first_step_progress_log=False,
    )
    return model, optimizer


def run_update(dmuon, model, optimizer, device, dtype, *, index, accumulate, no_sync):
    optimizer.zero_grad()
    losses = []
    for micro in range(accumulate):
        generator = torch.Generator().manual_seed(20000 + index * 8 + micro)
        x = torch.randn(2, 8, generator=generator).to(device=device, dtype=dtype)
        indices = torch.tensor([0, 1], device=device)
        options = {"nested": [x.clone(), (indices, "unchanged")],
                   "use_extra": index % 2 == 0, "use_muon": index % 2 == 0}
        context = dmuon.no_sync(model) if no_sync and micro < accumulate - 1 else nullcontext()
        with context:
            loss = model((x, indices), options=options) / accumulate
            loss.backward()
        losses.append(loss.detach().cpu())
    dmuon.wait_all_reduces(model)
    gradients = {}
    module_names = {id(module): name for name, module in model.named_modules()}
    for dp in dmuon.get_owned_params(model, 0):
        grad = dp._reduced_grad
        if grad is None:
            grad = getattr(dp, "_sharded_adamw_grad", None)
        if grad is not None:
            gradients[f"{module_names[id(dp.module)]}.{dp.param_name}"] = cpu_tree(grad)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grad = parameter.grad.to_local() if hasattr(parameter.grad, "to_local") else parameter.grad
            gradients[name] = cpu_tree(grad)
    stats = dmuon.clip_grad_norm_(optimizer, None)
    assert stats.param_count == (1 if index % 2 == 0 else 0)
    assert not bool(stats.found_inf)
    optimizer.step()
    dmuon.wait_all_post_step_broadcasts(model)
    live_momentum = {}
    for dp in dmuon.get_owned_params(model, 0):
        state = optimizer.state.get(id(dp), {})
        for key in ("momentum_buffer", "tp_local_momentum_buffer"):
            if key not in state:
                continue
            buffer = state[key]
            expected_dtype = (
                dp._optim_dtype or dp._compute_dtype or dp._orig_dtype or buffer.dtype
            )
            assert buffer.dtype == expected_dtype
            name = f"{module_names[id(dp.module)]}.{dp.param_name}:{key}"
            live_momentum[name] = cpu_tree(buffer)
    model_state = dmuon.get_model_state_dict(model)
    optimizer_state = dmuon.get_optimizer_state_dict(model, optimizer)
    assert "dedicated" in optimizer_state and "fsdp" in optimizer_state
    assert set(optimizer_state["dedicated"]) == {"proj.weight"}
    assert set(optimizer_state["dedicated_adamw"]) == {"owner.weight", "sharded.weight"}
    for state in optimizer_state["dedicated_adamw"].values():
        assert state["step"] == index + 1
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32
    assert model.unused.grad is None
    assert "unused" not in optimizer_state["fsdp"]
    assert optimizer_state["fsdp"]["head.weight"]["step"] == index + 1
    assert optimizer_state["fsdp"]["optional_bias"]["step"] == index // 2 + 1
    return {"losses": losses, "model": cpu_tree(model_state),
            "optimizer": cpu_tree(optimizer_state), "clip": cpu_tree(stats.as_dict()),
            "input_dtypes": model.proj.seen_dtypes if index % 2 == 0 else None,
            "gradients": gradients, "live_momentum": live_momentum}


def run_training(dmuon, mesh, device, *, dtype, cast, pathway, asynchronous, no_sync,
                 sharded_adamw=False, grad_dtype=None, verify_restore=True):
    model, optimizer = build_model(
        dmuon, mesh, device, dtype, cast=cast, pathway=pathway, asynchronous=asynchronous,
        sharded_adamw=sharded_adamw, grad_dtype=grad_dtype,
    )
    result = []
    input_dtype = torch.bfloat16 if cast and dtype == torch.float32 else dtype
    for index in range(3):
        result.append(run_update(dmuon, model, optimizer, device, input_dtype,
                                 index=index, accumulate=4, no_sync=no_sync))
    assert result[-1]["input_dtypes"] == (str(dtype), str(dtype), "torch.int64")
    assert_tree_equal(result[0]["model"]["proj.weight"], result[1]["model"]["proj.weight"])
    assert_tree_equal(result[0]["optimizer"]["dedicated"], result[1]["optimizer"]["dedicated"])
    saved = result[-1]
    if verify_restore:
        original_master = master_snapshot(dmuon, model)
        restored, restored_optimizer = build_model(
            dmuon, mesh, device, dtype, cast=cast, pathway=pathway, asynchronous=asynchronous,
            sharded_adamw=sharded_adamw, grad_dtype=grad_dtype,
        )
        dmuon.set_model_state_dict(restored, saved["model"])
        dmuon.set_optimizer_state_dict(restored, restored_optimizer, saved["optimizer"])
        restored_initial = {"model": cpu_tree(dmuon.get_model_state_dict(restored)),
                            "optimizer": cpu_tree(dmuon.get_optimizer_state_dict(restored, restored_optimizer)),
                            "master": master_snapshot(dmuon, restored)}
    for index in (3, 4):
        next_original = run_update(dmuon, model, optimizer, device, input_dtype,
                                   index=index, accumulate=4, no_sync=no_sync)
        if verify_restore:
            next_restored = run_update(dmuon, restored, restored_optimizer, device, input_dtype,
                                       index=index, accumulate=4, no_sync=no_sync)
            try:
                assert_tree_equal(next_original, next_restored)
            except AssertionError as error:
                error.restore_failure = {
                    "stage": "checkpoint_next_update", "dtype": str(dtype),
                    "pathway": pathway, "update_index": index, "error": str(error),
                    "sharded_adamw": sharded_adamw, "grad_dtype": str(grad_dtype),
                    "saved": saved, "original_master": original_master,
                    "restored_initial": restored_initial,
                    "expected": next_original, "actual": next_restored,
                    "deltas": tensor_deltas(next_restored, next_original),
                }
                raise
        result.append(next_original)
    return result


def assert_tree_equal(actual, expected, *, path="root"):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True,
                                   msg=lambda detail: f"{path}: {detail}")
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_tree_equal(actual[key], expected[key], path=f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_tree_equal(left, right, path=f"{path}[{index}]")
    else:
        assert actual == expected, path
