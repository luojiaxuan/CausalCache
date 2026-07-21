#!/usr/bin/env python3
"""Train the post-final-hidden token-adapter selector control."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

from causalcache.set_utility_token_adapter import (
    TokenAdapterConditionalMarginalPredictor,
    TokenAdapterConfig,
)
from causalcache.set_utility_token_models import TokenUtilityModelConfig
from scripts.train_set_utility_set_transformer_control import (
    SetTransformerControlRuntime,
    _finalize,
    _fit,
)
from scripts.train_set_utility_structured_marginal import (
    _read_signed_json,
    _sha256_file,
)
from scripts.train_set_utility_token_predictor import _read_json

PHASE_ADAPTER_ONLY = "adapter_only"
PHASE_JOINT = "joint"
PHASES = (PHASE_ADAPTER_ONLY, PHASE_JOINT)


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


class TokenAdapterControlRuntime(SetTransformerControlRuntime):
    """Specialize the shared control loop without changing its truth barrier."""

    progress_schema = "causalcache.selector_token_adapter_progress.v1"
    summary_schema = "causalcache.selector_token_adapter_training.v1"
    completed_status = "COMPLETED_SELECTOR_TOKEN_ADAPTER_SELECTED_BY_TRUE_RECOVERY"
    finalized_status = "COMPLETED_SELECTOR_TOKEN_ADAPTER_TRUE_RECOVERY_SELECTION"
    pending_finalized_status = "PENDING_SELECTOR_TOKEN_ADAPTER_TRUE_RECOVERY"
    selection_schema = "causalcache.selector_token_adapter_selection.v1"

    def __init__(
        self,
        *,
        phase: str,
        selector_config_path: Path,
        control_config_path: Path,
        variant_name: str,
        initial_checkpoint: Path,
        parent_checkpoint: Path | None = None,
        parent_summary: Path | None = None,
    ) -> None:
        if phase not in PHASES:
            raise ValueError("token-adapter phase is invalid")
        self.phase = phase
        self.selector_config_path = selector_config_path.resolve()
        self.control_config_path = control_config_path.resolve()
        self.variant_name = variant_name
        self.selector_config = _read_json(self.selector_config_path)
        self.selector_config_sha256 = _sha256_file(self.selector_config_path)
        self.control_config_sha256 = _sha256_file(self.control_config_path)
        self.initial_checkpoint = initial_checkpoint.resolve()
        self.initial_checkpoint_sha256 = _sha256_file(self.initial_checkpoint)
        self.parent_checkpoint = (
            None if parent_checkpoint is None else parent_checkpoint.resolve()
        )
        self.parent_summary = (
            None if parent_summary is None else parent_summary.resolve()
        )
        self.parent_checkpoint_sha256 = None
        self.parent_summary_sha256 = None
        self._validate_parent_artifacts()
        self.model_family = f"selector_token_adapter_v1_{phase}"

    def _validate_parent_artifacts(self) -> None:
        if self.phase == PHASE_ADAPTER_ONLY:
            if self.parent_checkpoint is not None or self.parent_summary is not None:
                raise ValueError("adapter-only phase cannot accept a parent artifact")
            return
        if self.parent_checkpoint is None or self.parent_summary is None:
            raise ValueError(
                "joint token-adapter phase requires its selected adapter-only "
                "checkpoint and summary"
            )
        self.parent_checkpoint_sha256 = _sha256_file(self.parent_checkpoint)
        self.parent_summary_sha256 = _sha256_file(self.parent_summary)
        summary = _read_signed_json(self.parent_summary)
        runtime = summary.get("runtime")
        identity = summary.get("identity")
        selected = summary.get("selected_checkpoint")
        if (
            summary.get("status")
            != "COMPLETED_SELECTOR_TOKEN_ADAPTER_SELECTED_BY_TRUE_RECOVERY"
            or not isinstance(runtime, Mapping)
            or runtime.get("phase") != PHASE_ADAPTER_ONLY
            or not isinstance(identity, Mapping)
            or identity.get("model_family")
            != f"selector_token_adapter_v1_{PHASE_ADAPTER_ONLY}"
            or identity.get("config_sha256") != self.control_config_sha256
            or identity.get("selector_config_sha256") != self.selector_config_sha256
            or identity.get("initial_checkpoint_sha256")
            != self.initial_checkpoint_sha256
            or identity.get("variant") != self.variant_name
            or not isinstance(selected, Mapping)
            or selected.get("sha256") != self.parent_checkpoint_sha256
        ):
            raise ValueError("joint token-adapter parent binding drifted")

    def validate_config(
        self, config: Mapping[str, Any], variant_name: str
    ) -> dict[str, Any]:
        if _sha256_file(self.control_config_path) != self.control_config_sha256:
            raise ValueError("token-adapter control config changed after binding")
        if variant_name != self.variant_name:
            raise ValueError("token-adapter variant drifted")
        variant = super().validate_config(config, variant_name)
        selector = self.selector_config
        arm = selector.get("arms", {}).get("post_final_token_adapter")
        training = selector.get("training")
        head = selector.get("head")
        input_contract = selector.get("input")
        firewall = selector.get("firewall")
        if not all(
            isinstance(value, Mapping)
            for value in (arm, training, head, input_contract, firewall)
        ):
            raise ValueError("token-adapter selector contract is incomplete")
        for key in (
            "training_input_content_sha256",
            "initial_head_checkpoint_sha256",
            "split_manifest_content_sha256",
        ):
            _sha256(input_contract.get(key), label=f"token-adapter {key}")
        if (
            input_contract["training_input_content_sha256"]
            != config["input"]["training_input_content_sha256"]
            or input_contract["split_manifest_content_sha256"]
            != config["input"]["train_heldout_manifest_content_sha256"]
            or input_contract["initial_head_checkpoint_sha256"]
            != self.initial_checkpoint_sha256
            or dict(head) != dict(variant["model"])
            or int(arm.get("adapter_bottleneck_size", -1)) != 64
            or int(training.get("phase_1", {}).get("epochs", -1)) != 1
            or int(training.get("phase_2", {}).get("maximum_epochs", -1)) != 4
            or training.get("phase_1", {}).get("head_trainable") is not False
            or training.get("phase_2", {}).get("head_trainable") is not True
            or firewall.get("untouched_evaluation_access") is not False
            or firewall.get("action_policy_modified") is not False
            or firewall.get("teacher_labels_regenerated") is not False
        ):
            raise ValueError("token-adapter architecture, phase, or firewall drifted")
        if tuple(variant.get("allowed_world_sizes", ())) != (1, 2, 4, 6):
            raise ValueError("token-adapter DDP sizes must be exactly 1/2/4/6")
        return variant

    def execution_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(dict(config))
        selector_training = self.selector_config["training"]
        result["training"].update(
            {
                "epochs": (
                    int(selector_training["phase_1"]["epochs"])
                    if self.phase == PHASE_ADAPTER_ONLY
                    else int(selector_training["phase_2"]["maximum_epochs"])
                ),
                "loss": copy.deepcopy(selector_training["loss"]),
                "maximum_base_cardinality": int(
                    selector_training["maximum_base_cardinality"]
                ),
                "maximum_gradient_norm": float(
                    selector_training["maximum_gradient_norm"]
                ),
                "minimum_lr_ratio": float(selector_training["minimum_lr_ratio"]),
                "normalization_floor": float(selector_training["normalization_floor"]),
                "seed": int(selector_training["seed"]),
                "warmup_ratio": float(selector_training["warmup_ratio"]),
            }
        )
        # note (luojiaxuan): Checkpoint selection remains the unchanged frozen
        # control contract. The earlier extraction config used patience=2, but
        # it is not an executable training contract and must not override the
        # trajectory-holdout manifest's patience=3 binding.
        return result

    def build_model(
        self,
        *,
        variant: Mapping[str, Any],
        device: Any,
        torch: Any,
    ) -> TokenAdapterConditionalMarginalPredictor:
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as error:  # pragma: no cover
            raise RuntimeError("token-adapter training needs safetensors") from error
        arm = self.selector_config["arms"]["post_final_token_adapter"]
        model = TokenAdapterConditionalMarginalPredictor(
            TokenUtilityModelConfig(**variant["model"]),
            TokenAdapterConfig(
                source_hidden_size=int(variant["model"]["source_hidden_size"]),
                bottleneck_size=int(arm["adapter_bottleneck_size"]),
            ),
        ).to(device)
        initial_state = load_file(str(self.initial_checkpoint), device=str(device))
        model.load_predictor_state_dict(initial_state)
        if self.phase == PHASE_JOINT:
            assert self.parent_checkpoint is not None
            model.load_state_dict(
                load_file(str(self.parent_checkpoint), device=str(device)), strict=True
            )
            model.unfreeze_head()
        else:
            if bool(model.adapter.up.weight.detach().count_nonzero()):
                raise RuntimeError("zero-init token adapter is not an exact identity")
            model.freeze_head()
        if any(
            parameter.dtype != torch.float32 for parameter in model.adapter.parameters()
        ):
            raise RuntimeError("token-adapter master weights must remain FP32")
        return model

    def optimizer_parameter_groups(
        self,
        model: TokenAdapterConditionalMarginalPredictor,
        *,
        variant: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        del variant
        arm = self.selector_config["arms"]["post_final_token_adapter"]
        return model.optimizer_parameter_groups(
            adapter_lr=float(arm["adapter_learning_rate"]),
            head_lr=float(arm["head_learning_rate"]),
        )

    def set_training_mode(
        self, model: TokenAdapterConditionalMarginalPredictor
    ) -> None:
        model.train()
        model.adapter.train()
        if self.phase == PHASE_ADAPTER_ONLY:
            model.predictor.eval()
            if any(
                parameter.requires_grad for parameter in model.predictor.parameters()
            ):
                raise RuntimeError(
                    "adapter-only predictor unexpectedly became trainable"
                )
        else:
            model.predictor.train()

    def load_resume_model_state(
        self,
        model: TokenAdapterConditionalMarginalPredictor,
        state: Mapping[str, Any],
    ) -> None:
        model.load_state_dict(state, strict=True)
        if any(
            parameter.dtype != model.adapter.up.weight.dtype
            for parameter in model.adapter.parameters()
        ):
            raise RuntimeError("resumed token-adapter master dtype drifted")

    def identity_fields(self) -> dict[str, Any]:
        return {
            "initial_checkpoint_sha256": self.initial_checkpoint_sha256,
            "initialization": "strict_current_best_set_epoch_2",
            "parent_selector_checkpoint_sha256": self.parent_checkpoint_sha256,
            "parent_selector_summary_sha256": self.parent_summary_sha256,
            "phase": self.phase,
            "selector_config_sha256": self.selector_config_sha256,
        }

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "adapter_bottleneck_size": 64,
            "adapter_master_dtype": "torch.float32",
            "initial_checkpoint_sha256": self.initial_checkpoint_sha256,
            "parent_selector_checkpoint_sha256": self.parent_checkpoint_sha256,
            "phase": self.phase,
            "truth_denominator": "fixed_256_state_train_trajectory_holdout",
            "untouched_evaluation_access": False,
        }


def _runtime(args: argparse.Namespace) -> TokenAdapterControlRuntime:
    return TokenAdapterControlRuntime(
        phase=args.phase,
        selector_config_path=args.selector_config,
        control_config_path=args.config,
        variant_name=args.variant,
        initial_checkpoint=args.initial_checkpoint,
        parent_checkpoint=args.parent_selector_checkpoint,
        parent_summary=args.parent_selector_summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "finalize"):
        child = subparsers.add_parser(command)
        child.add_argument("--input-root", type=Path, required=True)
        child.add_argument("--cache-root", type=Path, required=True)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--selector-config", type=Path, required=True)
        child.add_argument("--variant", required=True)
        child.add_argument("--phase", choices=PHASES, required=True)
        child.add_argument("--split-manifest", type=Path, required=True)
        child.add_argument("--initial-checkpoint", type=Path, required=True)
        child.add_argument("--parent-selector-checkpoint", type=Path)
        child.add_argument("--parent-selector-summary", type=Path)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument("--truth-source", action="append", default=[])
    fit = subparsers.choices["fit"]
    fit.add_argument("--device", required=True)
    fit.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    runtime = _runtime(args)
    if args.command == "fit":
        _fit(args, runtime=runtime)
    else:
        _finalize(args, runtime=runtime)


if __name__ == "__main__":
    main()
