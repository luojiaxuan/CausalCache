from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import scripts.train_set_utility_set_transformer_control as control_training

torch = pytest.importorskip("torch")
load_file = pytest.importorskip("safetensors.torch").load_file
save_file = pytest.importorskip("safetensors.torch").save_file

from causalcache.set_utility_token_models import (
    TokenConditionalMarginalPredictor,
    TokenUtilityModelConfig,
)
from scripts.train_set_utility_structured_marginal import (
    _read_signed_json,
    _sha256_file,
    _signed,
    _write_atomic,
)
from scripts.train_set_utility_token_adapter_control import (
    PHASE_ADAPTER_ONLY,
    PHASE_JOINT,
    TokenAdapterControlRuntime,
)


def _model_config() -> dict[str, object]:
    return {
        "dropout": 0.0,
        "family": "set_transformer",
        "hidden_size": 16,
        "latent_count": 2,
        "num_heads": 4,
        "numeric_feature_size": 5,
        "preserve_entity_latents": True,
        "resampler_layers": 1,
        "set_layers": 1,
        "source_hidden_size": 64,
    }


def _selector_config(checkpoint_sha256: str) -> dict[str, object]:
    return {
        "arms": {
            "post_final_token_adapter": {
                "adapter_bottleneck_size": 64,
                "adapter_learning_rate": 1e-4,
                "head_learning_rate": 2e-4,
            }
        },
        "checkpoint_selection": {"minimum_delta": 0.005, "patience": 2},
        "firewall": {
            "action_policy_modified": False,
            "teacher_labels_regenerated": False,
            "untouched_evaluation_access": False,
        },
        "head": _model_config(),
        "input": {
            "initial_head_checkpoint_sha256": checkpoint_sha256,
            "split_manifest_content_sha256": "b" * 64,
            "training_input_content_sha256": "c" * 64,
        },
        "training": {
            "loss": {
                "conditional_listwise": 2.0,
                "conditional_marginal": 0.5,
                "conditional_marginal_smooth_l1_beta": 0.25,
                "decision_regret": 2.0,
                "decision_temperature": 0.05,
                "sign_classification": 0.25,
            },
            "maximum_base_cardinality": 3,
            "maximum_gradient_norm": 1.0,
            "minimum_lr_ratio": 0.1,
            "normalization_floor": 0.01,
            "phase_1": {"epochs": 1, "head_trainable": False},
            "phase_2": {"head_trainable": True, "maximum_epochs": 4},
            "seed": 17,
            "warmup_ratio": 0.05,
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _initial_checkpoint(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    model = TokenConditionalMarginalPredictor(
        TokenUtilityModelConfig(**_model_config())
    )
    state = {
        key: value.detach().contiguous() for key, value in model.state_dict().items()
    }
    path = tmp_path / "initial.safetensors"
    save_file(state, str(path))
    return path, state


def _runtime(
    tmp_path: Path,
    *,
    phase: str,
    parent_checkpoint: Path | None = None,
    parent_summary: Path | None = None,
) -> tuple[TokenAdapterControlRuntime, Path, Path]:
    initial, _ = _initial_checkpoint(tmp_path)
    selector = tmp_path / "selector.json"
    control = tmp_path / "control.json"
    _write_json(selector, _selector_config(_sha256_file(initial)))
    _write_json(control, {"fixture": True})
    return (
        TokenAdapterControlRuntime(
            phase=phase,
            selector_config_path=selector,
            control_config_path=control,
            variant_name="tiny",
            initial_checkpoint=initial,
            parent_checkpoint=parent_checkpoint,
            parent_summary=parent_summary,
        ),
        initial,
        control,
    )


def test_adapter_only_runtime_is_zero_init_frozen_and_fp32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, initial, _ = _runtime(tmp_path, phase=PHASE_ADAPTER_ONLY)
    variant = {
        "model": _model_config(),
        "weight_decay": 0.01,
    }
    model = runtime.build_model(variant=variant, device="cpu", torch=torch)
    assert not any(
        parameter.requires_grad for parameter in model.predictor.parameters()
    )
    assert all(
        parameter.dtype == torch.float32 for parameter in model.adapter.parameters()
    )
    assert torch.count_nonzero(model.adapter.up.weight) == 0
    initial_state = load_file(str(initial))
    assert all(
        torch.equal(value, initial_state[name])
        for name, value in model.predictor.state_dict().items()
    )
    groups = runtime.optimizer_parameter_groups(model, variant=variant)
    assert [group["name"] for group in groups] == ["adapter"]
    runtime.set_training_mode(model)
    assert model.adapter.training is True
    assert model.predictor.training is False
    assert runtime.ddp_find_unused_parameters() is False

    execution = runtime.execution_config(
        {
            "checkpoint_selection": {"minimum_delta": 0.1, "patience": 3},
            "training": {
                "balancing": {},
                "exact_optimizer_inventory": {},
                "minimum_inventory": {},
            },
        }
    )
    assert execution["training"]["epochs"] == 1
    assert execution["checkpoint_selection"]["patience"] == 3
    assert execution["checkpoint_selection"]["minimum_delta"] == 0.1

    monkeypatch.setattr(
        control_training,
        "merge_control_truth_schedule",
        lambda _plans: {"missing_coalition_count": 1},
    )
    schedule = control_training._publish_schedule(
        tmp_path,
        ({"epoch": 1},),
        model_family=runtime.model_family,
        complete_status=runtime.truth_complete_status,
        pending_status=runtime.truth_pending_status,
    )
    assert schedule["model_family"] == "selector_token_adapter_v1_adapter_only"
    assert schedule["status"] == "PENDING_SELECTOR_TOKEN_ADAPTER_HELDOUT_TRUTH"


def test_joint_runtime_strictly_binds_and_loads_parent(tmp_path: Path) -> None:
    initial, initial_state = _initial_checkpoint(tmp_path)
    selector = tmp_path / "selector.json"
    control = tmp_path / "control.json"
    _write_json(selector, _selector_config(_sha256_file(initial)))
    _write_json(control, {"fixture": True})

    phase_one = TokenAdapterControlRuntime(
        phase=PHASE_ADAPTER_ONLY,
        selector_config_path=selector,
        control_config_path=control,
        variant_name="tiny",
        initial_checkpoint=initial,
    )
    variant = {"model": _model_config(), "weight_decay": 0.01}
    parent_model = phase_one.build_model(variant=variant, device="cpu", torch=torch)
    parent_model.adapter.up.weight.data.fill_(0.125)
    parent_checkpoint = tmp_path / "adapter-only.safetensors"
    save_file(
        {
            key: value.detach().contiguous()
            for key, value in parent_model.state_dict().items()
        },
        str(parent_checkpoint),
    )
    parent_summary = tmp_path / "summary.json"
    _write_atomic(
        parent_summary,
        _signed(
            {
                "identity": {
                    "config_sha256": _sha256_file(control),
                    "initial_checkpoint_sha256": _sha256_file(initial),
                    "model_family": "selector_token_adapter_v1_adapter_only",
                    "selector_config_sha256": _sha256_file(selector),
                    "variant": "tiny",
                },
                "runtime": {"phase": PHASE_ADAPTER_ONLY},
                "selected_checkpoint": {"sha256": _sha256_file(parent_checkpoint)},
                "status": (
                    "COMPLETED_SELECTOR_TOKEN_ADAPTER_SELECTED_BY_TRUE_RECOVERY"
                ),
            }
        ),
    )
    runtime = TokenAdapterControlRuntime(
        phase=PHASE_JOINT,
        selector_config_path=selector,
        control_config_path=control,
        variant_name="tiny",
        initial_checkpoint=initial,
        parent_checkpoint=parent_checkpoint,
        parent_summary=parent_summary,
    )
    model = runtime.build_model(variant=variant, device="cpu", torch=torch)
    assert torch.all(model.adapter.up.weight == 0.125)
    assert all(
        torch.equal(value, initial_state[name])
        for name, value in model.predictor.state_dict().items()
    )
    groups = runtime.optimizer_parameter_groups(model, variant=variant)
    assert [group["name"] for group in groups] == ["adapter", "head"]
    runtime.set_training_mode(model)
    assert model.predictor.training is True
    assert (
        runtime.execution_config(
            {
                "checkpoint_selection": {},
                "training": {
                    "balancing": {},
                    "exact_optimizer_inventory": {},
                    "minimum_inventory": {},
                },
            }
        )["training"]["epochs"]
        == 4
    )

    bad_summary = copy.deepcopy(_read_signed_json(parent_summary))
    bad_summary.pop("content_sha256")
    bad_summary["selected_checkpoint"]["sha256"] = "0" * 64
    _write_atomic(parent_summary, _signed(bad_summary))
    with pytest.raises(ValueError, match="parent binding"):
        TokenAdapterControlRuntime(
            phase=PHASE_JOINT,
            selector_config_path=selector,
            control_config_path=control,
            variant_name="tiny",
            initial_checkpoint=initial,
            parent_checkpoint=parent_checkpoint,
            parent_summary=parent_summary,
        )
