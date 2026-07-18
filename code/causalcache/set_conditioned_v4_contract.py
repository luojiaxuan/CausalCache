"""Fail-closed Source-A/B contract for frozen-base residual development v4."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_conditioned_v4_frozen_base_residual_development_v1"
SOURCE_STATUS = "source_a_frozen_before_set_conditioned_v4_execution"
VALIDATION_STATUS = "VALID_SET_CONDITIONED_V4_FROZEN_BASE_RESIDUAL_SOURCE_A_V1"
EXECUTION_B_VALIDATION_STATUS = (
    "VALID_SET_CONDITIONED_V4_FROZEN_BASE_RESIDUAL_EXECUTION_B_V1"
)
EXECUTION_B_FREEZE_STATUS = "frozen_execution_b_before_v4_training"
CANONICAL_CONFIG_PATH = (
    "code/configs/"
    "causalcache_set_conditioned_v4_frozen_base_residual_development_v1.json"
)
EXECUTION_B_RUNNER_FREEZE_PATH = (
    "code/configs/"
    "causalcache_set_conditioned_v4_frozen_base_residual_runner_v1.json"
)
REQUIRED_BRANCH = "luojiaxuan/set-conditioned-v4-frozen-base-residual"
BASE_GIT_COMMIT = "a50245b0edc681ec3c7f9ec06277d789586d32a8"
ORIGIN_URL = "https://github.com/luojiaxuan/CausalCache.git"

# note (luojiaxuan): The main task replaces this sentinel only after every
# Source-A file is final. Loading remains useful for contract tests, while both
# Source-A and Execution-B validation fail closed until the replacement occurs.
FROZEN_CONFIG_SHA256 = (
    "474df3cfebad7b3a6de9a31769650c1695a0939008f542ec4fd8f31db2aa6c3f"
)
FROZEN_CONFIG_SHA256_PLACEHOLDER = "TO_BE_MECHANICALLY_FROZEN_AT_SOURCE_A"

FORMAL_CACHE_REVISION = "a61b31bf2e69be00f94469f4a2f2d6b336fcc386"
FRESH16_REVISION = "3541fe1ea2c46e555c29cc53483e6f3b809f8f81"
FROZEN_BASE_REVISION = "23f6786075c7bff91f93fd7e8a878e070efb72a9"
FRESH16_FEATURE_SHA256 = (
    "deacc63480ed706d44e7690d974250860d1bc3e5967d4c2a3315090fe4b93939"
)
FRESH16_LABEL_SHA256 = (
    "832096b98011264a6b7fee74b2e78798876ea0d5ffc5c9c9baf7642be8e92382"
)
HISTORICAL_INDEPENDENT_SHA256 = (
    "2359c0f4b03e7009291c8912bea7193a4dfebd5f01d4876efd312bc595404dd2"
)
FROZEN_BASE_SELECTION_SHA256 = (
    "b5889a98adf08c3cf35b647a519b0d08f832879374a9ea8c4b5268c5a1c4b8dd"
)
LABEL_BLIND_SEAL_STATUS = (
    "SEALED_LABEL_BLIND_SET_CONDITIONED_V4_FROZEN_BASE_RESIDUAL_V1"
)

EXPECTED_BASE_EPOCHS = (60, 51, 6, 56, 54)
EXPECTED_BASE_OOF_RATIOS = (
    0.9012143403172107,
    0.9044738982091675,
    0.9164728647614823,
    0.9105838341486948,
    0.8991374084054389,
)
EXPECTED_CHECKPOINTS = (
    (
        0,
        60,
        "7088a1cb4224dd1e7013868249ed1c99c20782eb63c2a762e3059cfb939d4a49",
        "0198ad3cca7bd9e4eef2f6c40c9ac6349732f21935c61ec3434f058ee40b972a",
    ),
    (
        1,
        51,
        "8a25b8bad27e4c6f20a9a93273602e82a276ee7f9275ff672d6b0ed64dd8fd79",
        "454b0732389eac920f2aa69edb68317e3746dc57e1d2c6c7269f792257255af7",
    ),
    (
        2,
        6,
        "b8c38619db06bd479d278da62663a3b451cff951dff0747bf156835234216211",
        "c4d508933928dc7c6a24501090e3afd2a7d50a4977f37b3b3cb6cd818d28eb13",
    ),
    (
        3,
        56,
        "b7b175c0b513508f10479bb228fad6ffb6b4e330b050c981d415c2a54b8c11a2",
        "1f0d319411319dd2348e0a3da8e7d2b18e5b566e468278853d890a761639da60",
    ),
    (
        4,
        54,
        "8e5eee31c85653a210b9bf05c4cf4d94a8b4fabdb6bdab455caa446cc655df38",
        "51fc5aaa157ec9feb97a7c40cd7d532cc248ecb076feacd880dc9bba38007def",
    ),
)
EXPECTED_STATES = (
    "source_a_identity",
    "frozen_base_and_formal_input_identity",
    "fold_base_replay",
    "formal_residual_oof",
    "five_residual_checkpoint_completion",
    "base_zero_residual_invariant",
    "fresh_feature_only_prediction_completion",
    "label_blind_seal",
    "consumed_development_label_access_claim",
    "single_fresh_label_decode_and_join",
    "development_report_completion",
    "report_only_self_consistency_validation",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class SetConditionedV4Contract:
    data: Mapping[str, Any]
    sha256: str
    mechanically_frozen: bool
    repository_root: Path
    source_path: Path

    @property
    def formal_input(self) -> Mapping[str, Any]:
        return _mapping(self.data["formal58_input"], "formal58 input")

    @property
    def fresh_input(self) -> Mapping[str, Any]:
        return _mapping(
            self.data["fresh16_consumed_development_input"],
            "fresh16 consumed development input",
        )

    @property
    def frozen_base(self) -> Mapping[str, Any]:
        return _mapping(self.data["frozen_independent_base"], "frozen base")

    @property
    def training(self) -> Mapping[str, Any]:
        return _mapping(self.data["training_contract"], "training contract")

    @property
    def selection(self) -> Mapping[str, Any]:
        return _mapping(self.data["selection_contract"], "selection contract")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a sequence")
    return value


def _exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} drifted: expected {expected!r}, got {value!r}")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys drifted: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be nonempty text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key in {label}: {key}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite constant in {label}: {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    return _mapping(decoded, label)


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    fingerprint = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if fingerprint(before) != fingerprint(after) or len(payload) != before.st_size:
        raise ValueError(f"{label} changed while being read")
    return payload


def _validate_source_freeze(data: Mapping[str, Any]) -> None:
    source = _mapping(data["source_freeze"], "source freeze")
    _exact(source.get("required_branch"), REQUIRED_BRANCH, "source branch")
    _exact(source.get("required_base_ancestor"), BASE_GIT_COMMIT, "source base")
    _exact(source.get("network_call_count"), 0, "source network count")
    _exact(
        source.get("execution_b_runner_freeze_path"),
        EXECUTION_B_RUNNER_FREEZE_PATH,
        "Execution-B runner path",
    )
    for field in (
        "execution_b_must_be_direct_single_parent_of_source_a",
        "execution_b_runner_freeze_must_be_unique_diff",
        "execution_b_runner_freeze_must_be_absent_during_source_a_validation",
        "execution_b_must_be_clean_and_pushed",
        "execution_requires_clean_pushed_source_a_commit",
    ):
        _exact(source.get(field), True, field)
    _exact(source.get("source_validator_authorizes_execution"), False, "validator auth")
    _exact(
        source.get("source_inventory_sha256_location"),
        "mechanically_computed_and_frozen_only_in_execution_b_runner",
        "source inventory location",
    )
    paths = tuple(
        _safe_relative(item, "required Source-A path")
        for item in _sequence(source.get("required_source_a_paths"), "source paths")
    )
    required = {
        CANONICAL_CONFIG_PATH,
        "code/causalcache/set_conditioned_v4_contract.py",
        "code/causalcache/set_conditioned_v4_data.py",
        "code/causalcache/set_conditioned_v4_training.py",
        "code/causalcache/set_conditioned_v4_exploration.py",
        "code/scripts/validate_set_conditioned_v4_contract.py",
        "code/scripts/run_set_conditioned_v4_exploration.py",
        "code/tests/test_set_conditioned_v4_contract.py",
        "code/tests/test_set_conditioned_v4_data.py",
        "code/tests/test_set_conditioned_v4_training.py",
        "code/tests/test_set_conditioned_v4_exploration.py",
        "docs/set_conditioned_v4_frozen_base_residual.md",
    }
    if len(paths) != len(set(paths)) or not required.issubset(paths):
        raise ValueError("required Source-A path inventory drifted")
    prerequisites = _sequence(source.get("git_prerequisites"), "Git prerequisites")
    if len(prerequisites) != 8:
        raise ValueError("Git prerequisite inventory drifted")
    for raw in prerequisites:
        item = _mapping(raw, "Git prerequisite")
        _exact_keys(item, {"path", "sha256", "size_bytes"}, "Git prerequisite")
        _safe_relative(item["path"], "Git prerequisite path")
        if _SHA256.fullmatch(str(item["sha256"])) is None:
            raise ValueError("Git prerequisite SHA-256 is malformed")
        if type(item["size_bytes"]) is not int or item["size_bytes"] <= 0:
            raise ValueError("Git prerequisite size is malformed")


def _validate_inputs_and_base(data: Mapping[str, Any]) -> None:
    formal = _mapping(data["formal58_input"], "formal58 input")
    _exact(formal.get("immutable_revision"), FORMAL_CACHE_REVISION, "formal revision")
    _exact(
        {
            "role": formal.get("role"),
            "trajectory_count": formal.get("trajectory_count"),
            "state_count": formal.get("state_count"),
            "candidate_counts": formal.get("candidate_counts"),
            "budget": formal.get("budget"),
            "fold_count": formal.get("fold_count"),
            "fold_sizes": formal.get("fold_sizes"),
        },
        {
            "role": "only_training_and_train_only_model_selection_source",
            "trajectory_count": 58,
            "state_count": 174,
            "candidate_counts": [2, 3, 4],
            "budget": 2,
            "fold_count": 5,
            "fold_sizes": [12, 12, 12, 11, 11],
        },
        "formal geometry",
    )
    formal_files = {
        item["kind"]: item
        for item in _sequence(formal.get("exact_files"), "formal files")
        if isinstance(item, Mapping)
    }
    _exact(set(formal_files), {"feature_cache", "label_cache", "bundle_manifest"}, "formal files")

    fresh = _mapping(
        data["fresh16_consumed_development_input"], "fresh16 input"
    )
    _exact(fresh.get("immutable_revision"), FRESH16_REVISION, "fresh revision")
    _exact(fresh.get("role"), "consumed_development_evidence_not_holdout_not_test", "fresh role")
    files = {
        item["kind"]: item
        for item in _sequence(fresh.get("exact_files"), "fresh files")
        if isinstance(item, Mapping)
    }
    _exact(
        set(files),
        {"bundle_manifest", "feature_states", "label_states", "historical_independent_decisions"},
        "fresh files",
    )
    _exact(files["feature_states"].get("sha256"), FRESH16_FEATURE_SHA256, "fresh feature hash")
    _exact(files["label_states"].get("sha256"), FRESH16_LABEL_SHA256, "fresh label hash")
    _exact(
        files["historical_independent_decisions"].get("sha256"),
        HISTORICAL_INDEPENDENT_SHA256,
        "historical decision hash",
    )
    _exact(
        files["historical_independent_decisions"].get("access_stage"),
        "evaluate_after_v4_label_blind_seal",
        "historical decision access stage",
    )
    for field in (
        "may_be_called_holdout",
        "may_be_called_test",
        "may_authorize_go",
        "may_change_v1_or_v3_verdict",
        "may_tune_after_label_join",
    ):
        _exact(fresh.get(field), False, f"fresh {field}")
    _exact(fresh.get("maximum_new_claim_count"), 1, "fresh claim count")
    _exact(fresh.get("maximum_new_semantic_decode_attempt_count"), 1, "fresh decode attempts")
    _exact(fresh.get("maximum_new_decoded_state_count"), 48, "fresh decoded rows")

    base = _mapping(data["frozen_independent_base"], "frozen base")
    _exact(base.get("immutable_revision"), FROZEN_BASE_REVISION, "base revision")
    _exact(base.get("selected_learning_rate"), 0.0003, "base LR")
    _exact(tuple(base.get("selected_epochs_by_seed", ())), EXPECTED_BASE_EPOCHS, "base epochs")
    _exact(base.get("selection_sha256"), FROZEN_BASE_SELECTION_SHA256, "base selection")
    checkpoints = _sequence(base.get("checkpoints"), "base checkpoints")
    observed = tuple(
        (
            item.get("seed"),
            item.get("selected_epoch"),
            item.get("sha256"),
            item.get("model_state_sha256"),
        )
        for item in checkpoints
        if isinstance(item, Mapping)
    )
    _exact(observed, EXPECTED_CHECKPOINTS, "base checkpoint inventory")
    report = _mapping(base.get("oof_report"), "base OOF report")
    _exact(tuple(report.get("seed_best_raw_ratios", ())), EXPECTED_BASE_OOF_RATIOS, "base OOF ratios")
    for field in (
        "optimizer_may_contain_base_parameter",
        "all_parameters_require_grad",
    ):
        _exact(base.get(field), False, field)
    for field in (
        "training_pre_post_checkpoint_bytes_must_match",
        "training_pre_post_model_state_sha256_must_match",
        "zero_residual_must_replay_canonical_independent_selector",
    ):
        _exact(base.get(field), True, field)
    _exact(
        base.get("fresh16_historical_selection_sha256"),
        HISTORICAL_INDEPENDENT_SHA256,
        "base Fresh16 selection",
    )


def _validate_science_model_and_crossfit(data: Mapping[str, Any]) -> None:
    science = _mapping(data["scientific_contract"], "science")
    _exact(
        science.get("pair_correction_target"),
        "r_star_m_ij=U({i,j})/D(empty)-(g_m_i+g_m_j)",
        "pair correction target",
    )
    _exact(science.get("target_eligibility"), "D(empty)>1e-12", "target eligibility")
    _exact(science.get("target_name"), "set_utility_correction_residual", "target name")
    _exact(science.get("target_is_claimed_to_be_pure_second_order_interaction"), False, "pure interaction claim")
    _exact(science.get("target_may_absorb_frozen_base_prediction_error"), True, "base error acknowledgement")
    _exact(science.get("policy_rerun_count"), 0, "policy rerun")
    _exact(science.get("restoration_label_regeneration_count"), 0, "label regeneration")

    model = _mapping(data["model_contract"], "model")
    _exact(model.get("event_embedding_dimension"), 88, "event embedding")
    _exact(model.get("query_dimension"), 64, "query dimension")
    pair = _mapping(model.get("pair_features"), "pair features")
    _exact(pair.get("dimension"), 416, "pair feature dimension")
    _exact(
        pair.get("concatenation"),
        ["z_i", "z_j", "z_i_times_z_j", "abs_z_i_minus_z_j", "q64"],
        "pair features",
    )
    _exact(
        model.get("residual_head_layers"),
        ["Linear(416,64)", "GELU(approximate=none)", "Linear(64,1)"],
        "residual head",
    )
    _exact(model.get("trainable_parameter_count"), 26753, "parameter count")
    _exact(model.get("final_linear_weight_initialization"), "all_zeros", "zero weight")
    _exact(model.get("final_linear_bias_initialization"), "all_zeros", "zero bias")
    _exact(model.get("epoch_zero_output"), "exact_zero_for_every_pair", "epoch zero output")

    crossfit = _mapping(data["cross_fitting_contract"], "cross fitting")
    _exact(crossfit.get("fold_sizes"), [12, 12, 12, 11, 11], "cross-fit folds")
    _exact(crossfit.get("base_replay_learning_rate"), 0.0003, "base replay LR")
    _exact(tuple(crossfit.get("base_replay_epochs_by_seed", ())), EXPECTED_BASE_EPOCHS, "base replay epochs")
    _exact(
        crossfit.get(
            "base_replay_must_reproduce_five_historical_raw_oof_ratios_and_mean_within_1e_12"
        ),
        True,
        "base replay",
    )
    _exact(crossfit.get("base_replay_is_frozen_before_residual_optimization"), True, "base freeze")
    _exact(crossfit.get("full_fit_base_may_score_oof_heldout_fold"), False, "OOF leakage")
    _exact(crossfit.get("heldout_or_fresh_label_may_update_base"), False, "base update firewall")


def _validate_training_selection_and_evaluation(data: Mapping[str, Any]) -> None:
    training = _mapping(data["training_contract"], "training")
    _exact(training.get("device"), "cpu", "training device")
    _exact(training.get("dtype"), "float32", "training dtype")
    _exact(training.get("learning_rates_in_grid_order"), [0.0003, 0.001], "LR grid")
    _exact(training.get("seeds_in_grid_order"), [0, 1, 2, 3, 4], "seed grid")
    _exact(training.get("maximum_epochs"), 500, "maximum epochs")
    _exact(training.get("patience_epochs"), 50, "patience")
    _exact(training.get("epoch_improvement_epsilon"), 0.0001, "epoch epsilon")
    _exact(training.get("epoch_zero_is_a_model_selection_candidate"), True, "epoch-zero candidate")
    _exact(training.get("epoch_zero_is_exact_frozen_base"), True, "epoch-zero base")
    _exact(
        training.get("epoch_selection_metric"),
        "minimum_of_n4_raw_utility_over_exact_and_n4_normalized_recovery_over_exact",
        "OOF metric",
    )
    loss = _mapping(training.get("loss"), "loss")
    _exact(loss.get("ranking_comparisons"), "all_unordered_feasible_set_pairs_with_at_least_one_pair_set", "ranking comparisons")
    _exact(loss.get("ranking_raw_tie_epsilon"), 1e-12, "ranking tie")
    _exact(loss.get("ranking_weight"), 0.25, "ranking weight")
    _exact(loss.get("singleton_regression_weight"), 0.0, "singleton regression")
    optimizer = _mapping(training.get("optimizer"), "optimizer")
    _exact(optimizer.get("parameter_scope"), "residual_head_only", "optimizer scope")

    selection = _mapping(data["selection_contract"], "selection")
    _exact(
        selection.get("variants"),
        ["frozen_base", "unguarded_frozen_base_residual", "safe_frozen_base_residual"],
        "variants",
    )
    _exact(selection.get("primary_selector"), "safe_frozen_base_residual", "primary selector")
    _exact(selection.get("safe_vote_threshold"), 4, "safe vote")
    _exact(selection.get("safe_vote_total"), 5, "safe vote total")
    _exact(selection.get("safe_positive_margin_threshold"), 4, "safe margin")
    _exact(
        selection.get("safe_positive_margin_definition"),
        "individual_seed_total(pair_candidate)-total(base_candidate)_strictly_greater_than_zero",
        "safe margin definition",
    )
    _exact(selection.get("fallback"), "canonical_five_seed_mean_frozen_independent_selection", "fallback")
    _exact(selection.get("tie_epsilon"), 0.0, "tie epsilon")

    evaluation = _mapping(data["evaluation_contract"], "evaluation")
    _exact(evaluation.get("fresh16_is_consumed_development_only"), True, "development role")
    _exact(
        evaluation.get("primary_contrast"),
        "safe_frozen_base_residual_minus_frozen_base",
        "primary contrast",
    )
    rule = _mapping(evaluation.get("development_interpretation_rule"), "route rule")
    _exact(rule.get("all_conditions_required_for_promising"), True, "all checks")
    _exact(rule.get("minimum_mean_normalized_delta"), 0.01, "normalized threshold")
    _exact(rule.get("normalized_paired_bootstrap_lower_strictly_greater_than"), 0.0, "bootstrap threshold")
    _exact(rule.get("mean_raw_delta_strictly_greater_than"), 0.0, "raw threshold")
    _exact(rule.get("n4_mean_normalized_delta_strictly_greater_than"), 0.0, "n4 threshold")
    _exact(rule.get("minimum_positive_trajectory_count"), 8, "positive trajectory threshold")
    _exact(evaluation.get("result_may_automatically_open_confirm20"), False, "confirm route")
    _exact(evaluation.get("same_formal58_fresh16_may_support_another_v5_retune"), False, "v5 retune")
    _exact(set(evaluation.get("forbidden_interpretations", ())), {"GO", "CONFIRMED", "HELDOUT_PASS", "TEST_PASS"}, "forbidden interpretations")


def _validate_firewall_runtime_and_authorization(data: Mapping[str, Any]) -> None:
    firewall = _mapping(data["access_firewall"], "firewall")
    source_counts = _mapping(firewall.get("source_a_counts"), "Source-A counts")
    if not source_counts or any(value != 0 for value in source_counts.values()):
        raise ValueError("Source-A access counts must all be zero")
    train = _mapping(firewall.get("train_seal_maximum_counts"), "train-seal counts")
    for field in (
        "fresh16_label_access_claim",
        "fresh16_label_semantic_decode_attempt",
        "fresh16_decoded_state",
        "confirm20_access",
        "legacy_dev5_access",
        "matched_nll_evaluation",
        "closed_loop_episode",
        "raw_gui_access",
        "policy_forward",
        "gpu_job",
    ):
        _exact(train.get(field), 0, f"train-seal {field}")
    evaluate = _mapping(firewall.get("evaluate_maximum_new_counts"), "evaluate counts")
    expected_positive = {
        "fresh16_label_access_claim": 1,
        "fresh16_label_semantic_decode_attempt": 1,
        "fresh16_decoded_state": 48,
        "fresh16_development_join": 1,
        "fresh16_development_report": 1,
    }
    for field, expected in expected_positive.items():
        _exact(evaluate.get(field), expected, f"evaluate {field}")
    for field in (
        "confirm20_access",
        "legacy_dev5_access",
        "matched_nll_evaluation",
        "closed_loop_episode",
        "raw_gui_access",
        "policy_forward",
        "gpu_job",
    ):
        _exact(evaluate.get(field), 0, f"evaluate {field}")
    validation = _mapping(
        firewall.get("post_report_validation_maximum_new_counts"),
        "validation counts",
    )
    if not validation or any(value != 0 for value in validation.values()):
        raise ValueError("post-report validation must not reopen Fresh16 labels")
    historical = _mapping(firewall.get("historical_accounting"), "historical accounting")
    _exact(historical.get("v1_primary_unit"), "decoded_state_rows", "v1 unit")
    _exact(historical.get("v1_primary_value"), 48, "v1 rows")
    _exact(historical.get("v3_parent_claim_count"), 1, "v3 claims")
    _exact(historical.get("v3_parent_semantic_decode_attempt_count"), 2, "v3 attempts")
    _exact(historical.get("v3_to_v4_lineage_claim_count_after_success"), 2, "lineage claims")
    _exact(historical.get("v3_to_v4_lineage_semantic_decode_attempt_count_after_success"), 3, "lineage attempts")
    _exact(historical.get("lineage_totals_may_be_called_project_global"), False, "global accounting")
    for field in (
        "checkpoint_and_prediction_seal_must_precede_label_access_claim",
        "execution_b_validation_must_precede_input_read_or_output_mutation",
        "label_access_claim_must_precede_label_open_parse_or_semantic_decode",
        "post_report_validation_must_not_reopen_label",
        "no_post_label_training_prediction_guard_or_threshold_change",
        "decode_failure_requires_versioned_repair_and_cumulative_attempt_increment",
    ):
        _exact(firewall.get(field), True, field)

    machine = _mapping(data["execution_state_machine"], "state machine")
    _exact(tuple(machine.get("ordered_states", ())), EXPECTED_STATES, "state order")
    _exact(machine.get("train_seal_terminal_state"), "label_blind_seal", "train-seal terminal")
    seal = _mapping(machine.get("label_blind_seal"), "label-blind seal")
    _exact(seal.get("status"), LABEL_BLIND_SEAL_STATUS, "seal status")
    for counter in (
        "fresh_label_semantic_decode_attempt_count",
        "confirm20_access_count",
        "policy_forward_count",
        "gpu_job_count",
    ):
        _exact(seal.get(counter), 0, f"seal {counter}")
    _exact(machine.get("validate_reads_report_only"), True, "report-only validation")

    output = _mapping(data["output_contract"], "output contract")
    for field in (
        "reusable_model_and_state_artifacts_go_to_private_hugging_face",
        "post_report_publication_is_not_a_scientific_state_transition",
        "private_hf_publication_requires_manifest_and_card_per_repo",
        "tag_resolution_and_fresh_download_hash_replay_required_before_git_completion",
    ):
        _exact(output.get(field), True, field)

    runtime = _mapping(data["runtime_contract"], "runtime")
    _exact(
        {
            "device": runtime.get("device"),
            "dtype": runtime.get("dtype"),
            "gpu_count": runtime.get("gpu_count"),
            "gpu_preflight_required": runtime.get("gpu_preflight_required"),
            "policy_model_runtime_allowed": runtime.get("policy_model_runtime_allowed"),
        },
        {
            "device": "cpu",
            "dtype": "float32",
            "gpu_count": 0,
            "gpu_preflight_required": False,
            "policy_model_runtime_allowed": False,
        },
        "CPU runtime",
    )
    operations = _mapping(data["source_only_operation_contract"], "source operations")
    if not operations or any(value != 0 for value in operations.values()):
        raise ValueError("source-only operations must all be zero")
    authorization = _mapping(data["authorization"], "authorization")
    for field in (
        "fresh16_may_authorize_confirm",
        "confirm20_access_authorized",
        "legacy_dev5_access_authorized",
        "matched_nll_authorized",
        "closed_loop_authorized",
        "gpu_authorized",
        "policy_forward_authorized",
        "mainline_or_v3_artifact_mutation_authorized",
    ):
        _exact(authorization.get(field), False, field)


def validate_contract_data(value: Mapping[str, Any]) -> None:
    data = _mapping(value, "contract")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "lineage",
        "source_freeze",
        "formal58_input",
        "fresh16_consumed_development_input",
        "frozen_independent_base",
        "scientific_contract",
        "model_contract",
        "cross_fitting_contract",
        "training_contract",
        "selection_contract",
        "evaluation_contract",
        "access_firewall",
        "execution_state_machine",
        "runtime_contract",
        "namespace_contract",
        "output_contract",
        "source_only_operation_contract",
        "authorization",
    }
    _exact_keys(data, expected_keys, "contract")
    _exact(data.get("schema_version"), SCHEMA_VERSION, "schema version")
    _exact(data.get("protocol_id"), PROTOCOL_ID, "protocol id")
    _exact(data.get("status"), SOURCE_STATUS, "source status")
    lineage = _mapping(data["lineage"], "lineage")
    _exact(lineage.get("branch"), REQUIRED_BRANCH, "lineage branch")
    _exact(lineage.get("base_git_commit"), BASE_GIT_COMMIT, "base commit")
    _exact(lineage.get("origin_url"), ORIGIN_URL, "origin URL")
    _exact(lineage.get("study_role"), "non_mainline_final_frozen_base_set_conditioning_development_ablation", "study role")
    _exact(lineage.get("v3_result_is_immutable"), True, "v3 immutability")
    _exact(lineage.get("historical_v1_verdict_is_immutable"), True, "v1 immutability")
    v3 = _mapping(lineage.get("v3_result"), "v3 result")
    _exact(v3.get("required_outcome"), "stop_after_consumed_development_no_go", "v3 outcome")
    _validate_source_freeze(data)
    _validate_inputs_and_base(data)
    _validate_science_model_and_crossfit(data)
    _validate_training_selection_and_evaluation(data)
    _validate_firewall_runtime_and_authorization(data)


def load_frozen_set_conditioned_v4_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path | None = None,
) -> SetConditionedV4Contract:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = root / source_path
    payload = _regular_file_bytes(source_path, label="v4 Source-A contract")
    digest = sha256_bytes(payload)
    frozen = _SHA256.fullmatch(FROZEN_CONFIG_SHA256) is not None
    if frozen and digest != FROZEN_CONFIG_SHA256:
        raise ValueError(
            "v4 Source-A contract hash drifted: "
            f"expected {FROZEN_CONFIG_SHA256}, got {digest}"
        )
    if not frozen and FROZEN_CONFIG_SHA256 != FROZEN_CONFIG_SHA256_PLACEHOLDER:
        raise ValueError("v4 Source-A config freeze constant is malformed")
    data = _strict_json(payload, label="v4 Source-A contract")
    validate_contract_data(data)
    return SetConditionedV4Contract(
        data=data,
        sha256=digest,
        mechanically_frozen=frozen,
        repository_root=root,
        source_path=source_path.resolve(),
    )


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()
        raise ValueError(
            f"Git identity check failed: {' '.join(arguments)}"
            + (f": {detail}" if detail else "")
        )
    return result.stdout.strip()


def _source_inventory(
    contract: SetConditionedV4Contract,
) -> tuple[tuple[dict[str, Any], ...], str]:
    source = _mapping(contract.data["source_freeze"], "source freeze")
    inventory: list[dict[str, Any]] = []
    for relative in _sequence(source["required_source_a_paths"], "required paths"):
        path_value = _safe_relative(relative, "required source path")
        payload = _regular_file_bytes(
            contract.repository_root / path_value,
            label=path_value,
        )
        inventory.append(
            {
                "path": path_value,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    prerequisites = {
        item["path"]: item
        for item in _sequence(source["git_prerequisites"], "Git prerequisites")
    }
    observed = {item["path"]: item for item in inventory}
    for relative, expected in prerequisites.items():
        if observed.get(relative) != expected:
            raise ValueError(f"frozen Git prerequisite bytes drifted: {relative}")
    result = tuple(inventory)
    return result, sha256_bytes(canonical_json_bytes(result))


def _require_mechanical_freeze(contract: SetConditionedV4Contract) -> None:
    if not contract.mechanically_frozen:
        raise ValueError(
            "v4 Source-A config SHA-256 has not been mechanically frozen"
        )


def _validate_clean_pushed_branch(root: Path) -> tuple[str, str]:
    branch = _git(root, "branch", "--show-current")
    if branch != REQUIRED_BRANCH:
        raise ValueError("v4 validation is running on the wrong branch")
    head = _git(root, "rev-parse", "HEAD")
    if _COMMIT.fullmatch(head) is None:
        raise ValueError("Git HEAD is not a full lowercase commit")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("v4 execution source must have a clean worktree")
    if _git(root, "rev-parse", f"origin/{REQUIRED_BRANCH}") != head:
        raise ValueError("v4 execution source must equal its pushed origin branch")
    if _git(root, "remote", "get-url", "origin") != ORIGIN_URL:
        raise ValueError("origin URL differs from the frozen repository")
    return branch, head


def _live_remote_branch(root: Path) -> str:
    reference = f"refs/heads/{REQUIRED_BRANCH}"
    output = _git(root, "ls-remote", "--exit-code", "origin", reference)
    lines = output.splitlines()
    suffix = f"\t{reference}"
    if len(lines) != 1 or not lines[0].endswith(suffix):
        raise ValueError("live origin branch response is malformed")
    commit = lines[0][: -len(suffix)]
    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("live origin branch commit is malformed")
    return commit


def execution_b_freeze_payload(
    contract: SetConditionedV4Contract,
    *,
    source_a_git_commit: str,
    source_inventory_sha256: str,
) -> dict[str, Any]:
    _require_mechanical_freeze(contract)
    if _COMMIT.fullmatch(source_a_git_commit) is None:
        raise ValueError("Source-A commit is malformed")
    if _SHA256.fullmatch(source_inventory_sha256) is None:
        raise ValueError("Source-A inventory SHA-256 is malformed")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": EXECUTION_B_FREEZE_STATUS,
        "source_a_git_commit": source_a_git_commit,
        "source_inventory_sha256": source_inventory_sha256,
        "contract_sha256": contract.sha256,
        "runner_freeze_path": EXECUTION_B_RUNNER_FREEZE_PATH,
        "execution_b_direct_single_parent_required": True,
        "execution_b_unique_diff": [EXECUTION_B_RUNNER_FREEZE_PATH],
        "runtime_device": "cpu",
        "gpu_count": 0,
        "formal58_role": "only_training_and_train_only_model_selection_source",
        "fresh16_role": "consumed_development_not_holdout_or_test",
        "frozen_base_revision": FROZEN_BASE_REVISION,
        "fresh16_maximum_new_semantic_decode_attempt_count": 1,
        "confirm20_access_count": 0,
    }


def validate_source_a(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    contract = load_frozen_set_conditioned_v4_contract(
        path, repository_root=repository_root
    )
    _require_mechanical_freeze(contract)
    root = contract.repository_root
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("repository root differs from Git top level")
    runner = root / EXECUTION_B_RUNNER_FREEZE_PATH
    if runner.exists() or runner.is_symlink():
        raise ValueError("Execution-B runner freeze must be absent from Source-A")
    branch, head = _validate_clean_pushed_branch(root)
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", BASE_GIT_COMMIT, head],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if ancestor.returncode != 0:
        raise ValueError("frozen base commit is not an ancestor of Source-A")
    inventory, inventory_sha256 = _source_inventory(contract)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": VALIDATION_STATUS,
        "contract_sha256": contract.sha256,
        "source_a": {
            "branch": branch,
            "head": head,
            "base_git_commit": BASE_GIT_COMMIT,
            "origin_url": ORIGIN_URL,
            "required_path_count": len(inventory),
            "source_inventory_sha256": inventory_sha256,
            "source_inventory": list(inventory),
        },
        "execution_authorized_by_validator": False,
        "network_call_count": 0,
        "policy_forward_count": 0,
        "gpu_job_count": 0,
        "confirm20_access_count": 0,
    }


def _git_blob(root: Path, specification: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), "show", specification],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError("committed runner-freeze blob is unavailable")
    return result.stdout


def validate_execution_b(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path | None = None,
    source_a_git_commit: str,
    execution_b_git_commit: str,
) -> dict[str, Any]:
    if _COMMIT.fullmatch(source_a_git_commit) is None:
        raise ValueError("requested Source-A commit is malformed")
    if _COMMIT.fullmatch(execution_b_git_commit) is None:
        raise ValueError("requested Execution-B commit is malformed")
    contract = load_frozen_set_conditioned_v4_contract(
        path, repository_root=repository_root
    )
    _require_mechanical_freeze(contract)
    root = contract.repository_root
    branch, head = _validate_clean_pushed_branch(root)
    if head != execution_b_git_commit or _live_remote_branch(root) != head:
        raise ValueError("Execution-B identity differs from local or live origin")
    parents = _git(root, "rev-list", "--parents", "-n", "1", head).split()
    if parents != [head, source_a_git_commit]:
        raise ValueError("Execution-B must be direct single-parent child of Source-A")
    changed = tuple(
        line
        for line in _git(
            root,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            source_a_git_commit,
            head,
        ).splitlines()
        if line
    )
    expected_change = f"A\t{EXECUTION_B_RUNNER_FREEZE_PATH}"
    if changed != (expected_change,):
        raise ValueError("Execution-B must add only the runner freeze")
    inventory, inventory_sha256 = _source_inventory(contract)
    expected = execution_b_freeze_payload(
        contract,
        source_a_git_commit=source_a_git_commit,
        source_inventory_sha256=inventory_sha256,
    )
    expected_payload = canonical_json_bytes(expected) + b"\n"
    runner_path = root / EXECUTION_B_RUNNER_FREEZE_PATH
    observed = _regular_file_bytes(runner_path, label="Execution-B runner freeze")
    if observed != expected_payload:
        raise ValueError("Execution-B runner freeze differs from canonical bytes")
    if _git_blob(root, f"{head}:{EXECUTION_B_RUNNER_FREEZE_PATH}") != expected_payload:
        raise ValueError("Execution-B runner freeze differs from committed blob")
    tree_record = _git(root, "ls-tree", head, "--", EXECUTION_B_RUNNER_FREEZE_PATH)
    if not tree_record.startswith("100644 blob ") or not tree_record.endswith(
        f"\t{EXECUTION_B_RUNNER_FREEZE_PATH}"
    ):
        raise ValueError("Execution-B runner freeze must be regular 100644 blob")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": EXECUTION_B_VALIDATION_STATUS,
        "contract_sha256": contract.sha256,
        "source_a_git_commit": source_a_git_commit,
        "execution_b_git_commit": head,
        "branch": branch,
        "direct_single_parent": True,
        "unique_diff": expected_change,
        "runner_freeze_sha256": sha256_bytes(expected_payload),
        "source_inventory_sha256": inventory_sha256,
        "required_path_count": len(inventory),
        "runner_freeze": expected,
        "confirm20_access_count": 0,
        "gpu_job_count": 0,
    }


__all__ = [
    "BASE_GIT_COMMIT",
    "CANONICAL_CONFIG_PATH",
    "EXECUTION_B_FREEZE_STATUS",
    "EXECUTION_B_RUNNER_FREEZE_PATH",
    "EXPECTED_BASE_EPOCHS",
    "EXPECTED_BASE_OOF_RATIOS",
    "EXPECTED_CHECKPOINTS",
    "EXPECTED_STATES",
    "FRESH16_FEATURE_SHA256",
    "FRESH16_LABEL_SHA256",
    "FRESH16_REVISION",
    "FROZEN_BASE_REVISION",
    "FROZEN_BASE_SELECTION_SHA256",
    "FROZEN_CONFIG_SHA256",
    "FROZEN_CONFIG_SHA256_PLACEHOLDER",
    "HISTORICAL_INDEPENDENT_SHA256",
    "LABEL_BLIND_SEAL_STATUS",
    "PROTOCOL_ID",
    "REQUIRED_BRANCH",
    "SCHEMA_VERSION",
    "SOURCE_STATUS",
    "SetConditionedV4Contract",
    "canonical_json_bytes",
    "execution_b_freeze_payload",
    "load_frozen_set_conditioned_v4_contract",
    "sha256_bytes",
    "validate_contract_data",
    "validate_execution_b",
    "validate_source_a",
]
