import torch
import torch.nn as nn
import pytest

from dmuon.api import _validate_policy_hook_boundaries
from dmuon.policy import DMuonParamPolicy, resolve_param_policies


class TinyPolicyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.q_proj = nn.Linear(4, 4, bias=False)
        self.action_head = nn.Module()
        self.action_head.proj = nn.Linear(4, 4, bias=False)
        self.action_head.norm = nn.LayerNorm(4)
        self.embed_tokens = nn.Embedding(8, 4)


def _policy_by_name(model: nn.Module, **kwargs):
    params = dict(model.named_parameters())
    param_to_fqn = {param: name for name, param in params.items()}
    resolved = resolve_param_policies(
        params=params.values(),
        param_to_fqn=param_to_fqn,
        **kwargs,
    )
    return {name: resolved[param] for name, param in params.items()}


def test_param_policy_ordered_contains_overrides_route_and_dtype() -> None:
    model = TinyPolicyModel()

    policies = _policy_by_name(
        model,
        param_policy={
            "defaults": {
                "route": "adamw",
                "param_dtype": "bf16",
                "grad_dtype": None,
            },
            "overrides": [
                {
                    "name": ["q_proj", "proj"],
                    "set": {"route": "muon"},
                },
                {
                    "name": ["embed_tokens", "lm_head"],
                    "set": {"route": "sharded_adamw"},
                },
                {
                    "name": ["action_head"],
                    "set": {
                        "param_dtype": torch.float32,
                        "grad_dtype": torch.float32,
                        "output_dtype": torch.float32,
                        "cast_forward_inputs": True,
                    },
                },
                {
                    "name": ["norm"],
                    "set": {"route": "adamw"},
                },
            ],
        },
    )

    assert policies["backbone.q_proj.weight"].route == "muon"
    assert policies["backbone.q_proj.weight"].param_dtype is torch.bfloat16

    action_proj = policies["action_head.proj.weight"]
    assert action_proj.route == "muon"
    assert action_proj.param_dtype is torch.float32
    assert action_proj.grad_dtype is torch.float32
    assert action_proj.output_dtype is torch.float32
    assert action_proj.matched_overrides == (0, 2)

    action_norm = policies["action_head.norm.weight"]
    assert action_norm.route == "adamw"
    assert action_norm.param_dtype is torch.float32
    assert action_norm.grad_dtype is torch.float32
    assert action_norm.matched_overrides == (2, 3)

    assert policies["embed_tokens.weight"].route == "sharded_adamw"
    assert policies["embed_tokens.weight"].param_dtype is torch.bfloat16


def test_param_policy_contains_alias_remains_supported() -> None:
    model = TinyPolicyModel()

    policies = _policy_by_name(
        model,
        param_policy={
            "defaults": {"route": "adamw"},
            "overrides": [
                {
                    "contains": "q_proj",
                    "set": {"route": "matrix"},
                },
            ],
        },
    )

    assert policies["backbone.q_proj.weight"].route == "muon"


def test_legacy_compute_dtype_and_route_hint_fn() -> None:
    model = TinyPolicyModel()

    policies = _policy_by_name(
        model,
        compute_dtype=torch.bfloat16,
        route_hint_fn=lambda name, _param: (
            "sharded_adamw" if "embed_tokens" in name else "adamw"
        ),
        default_muon_forward_unshard="all_gather",
    )

    assert policies["backbone.q_proj.weight"].route == "adamw"
    assert policies["backbone.q_proj.weight"].param_dtype is torch.bfloat16
    assert policies["backbone.q_proj.weight"].muon_forward_unshard == "all_gather"
    assert policies["embed_tokens.weight"].route == "sharded_adamw"


def test_param_policy_fn_can_return_partial_mapping_or_policy_object() -> None:
    model = TinyPolicyModel()

    def policy_fn(name, _param):
        if "action_head" in name:
            return DMuonParamPolicy(route="muon", param_dtype=torch.float32)
        if "embed_tokens" in name:
            return {"route": "sharded"}
        return None

    policies = _policy_by_name(
        model,
        compute_dtype=torch.bfloat16,
        param_policy_fn=policy_fn,
    )

    assert policies["action_head.proj.weight"].route == "muon"
    assert policies["action_head.proj.weight"].param_dtype is torch.float32
    assert policies["embed_tokens.weight"].route == "sharded_adamw"
    assert policies["backbone.q_proj.weight"].param_dtype is torch.bfloat16


def test_param_policy_rejects_legacy_route_hint_conflict() -> None:
    model = TinyPolicyModel()

    with pytest.raises(ValueError, match="route_hint_fn"):
        _policy_by_name(
            model,
            route_hint_fn=lambda _n, _p: "muon",
            param_policy={"defaults": {"route": "adamw"}},
        )


def test_param_policy_rejects_unknown_override_selector_field() -> None:
    model = TinyPolicyModel()

    with pytest.raises(ValueError, match="unknown selector"):
        _policy_by_name(
            model,
            param_policy={
                "defaults": {"route": "adamw"},
                "overrides": [
                    {
                        "names": "q_proj",
                        "set": {"route": "muon"},
                    },
                ],
            },
        )


class FakeDParam:
    def __init__(
        self,
        *,
        param_dtype: torch.dtype,
        orig_dtype: torch.dtype = torch.float32,
        output_dtype=None,
        cast_forward_inputs: bool = True,
    ) -> None:
        self._param_dtype = param_dtype
        self._orig_dtype = orig_dtype
        self._output_dtype = output_dtype
        self._cast_forward_inputs = cast_forward_inputs


def test_policy_hook_boundary_rejects_ambiguous_input_casts() -> None:
    module = nn.Linear(2, 2)
    layer_to_dparams = {
        (module, ("bf16",)): [FakeDParam(param_dtype=torch.bfloat16)],
        (module, ("fp32",)): [FakeDParam(param_dtype=torch.float32)],
    }

    with pytest.raises(ValueError, match="mixed param dtypes"):
        _validate_policy_hook_boundaries(
            layer_to_dparams,
            {id(module): "action_head"},
        )


def test_policy_hook_boundary_allows_mixed_params_when_activation_cast_is_external() -> None:
    module = nn.Linear(2, 2)
    layer_to_dparams = {
        (module, ("bf16",)): [
            FakeDParam(param_dtype=torch.bfloat16, cast_forward_inputs=False)
        ],
        (module, ("fp32",)): [
            FakeDParam(param_dtype=torch.float32, cast_forward_inputs=False)
        ],
    }

    _validate_policy_hook_boundaries(layer_to_dparams, {id(module): "action_head"})


def test_policy_hook_boundary_allows_parallel_groups_with_shared_activation_policy() -> None:
    module = nn.Linear(2, 2)
    layer_to_dparams = {
        (module, ("bf16", "bf16_grad")): [
            FakeDParam(param_dtype=torch.bfloat16, output_dtype=torch.bfloat16)
        ],
        (module, ("bf16", "fp32_grad")): [
            FakeDParam(param_dtype=torch.bfloat16, output_dtype=torch.bfloat16)
        ],
    }

    _validate_policy_hook_boundaries(layer_to_dparams, {id(module): "action_head"})
