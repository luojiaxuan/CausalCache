"""Fail-closed Source-A contract for the non-mainline v3 pair-residual study."""

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
PROTOCOL_ID = "causalcache_set_conditioned_v3_pair_residual_exploration_v1"
SOURCE_STATUS = "source_a_frozen_before_set_conditioned_v3_execution"
VALIDATION_STATUS = "VALID_SET_CONDITIONED_V3_PAIR_RESIDUAL_SOURCE_A_V1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_set_conditioned_v3_pair_residual_exploration_v1.json"
)
EXECUTION_B_RUNNER_FREEZE_PATH = (
    "code/configs/causalcache_set_conditioned_v3_pair_residual_runner_v1.json"
)
EXECUTION_B_FREEZE_STATUS = "frozen_execution_b_before_v3_training"
FROZEN_CONFIG_SHA256 = (
    "8e1439f981212d8d26b835f1a5994904b425f014037f8a1cebb3e364cf5bdf2f"
)
REQUIRED_BRANCH = "luojiaxuan/set-conditioned-v3-pair-residual"
BASE_GIT_COMMIT = "ed8772eaa1c22794768e99e6cbe51a47f8137ce0"
ORIGIN_URL = "https://github.com/luojiaxuan/CausalCache.git"

FORMAL_CACHE_REVISION = "a61b31bf2e69be00f94469f4a2f2d6b336fcc386"
FRESH16_REVISION = "3541fe1ea2c46e555c29cc53483e6f3b809f8f81"
FRESH16_FEATURE_SHA256 = (
    "deacc63480ed706d44e7690d974250860d1bc3e5967d4c2a3315090fe4b93939"
)
FRESH16_LABEL_SHA256 = (
    "832096b98011264a6b7fee74b2e78798876ea0d5ffc5c9c9baf7642be8e92382"
)
LABEL_BLIND_SEAL_STATUS = "SEALED_LABEL_BLIND_SET_CONDITIONED_V3_V1"

EXPECTED_FORMAL_FILES = (
    {
        "kind": "feature_cache",
        "path": "formal58-transport-repair/v1/feature-cache-v1.tar",
        "sha256": "81fded50c4450700220742d3e0be9a5585d1bc51086150515b463bbdf4b4df8e",
        "size_bytes": 993280,
    },
    {
        "kind": "label_cache",
        "path": "formal58-transport-repair/v1/label-cache-v1.tar",
        "sha256": "4f9ef172aaa94c3ea8ce53aa43336c9fcb7800c24e181e1462d8239e31053cee",
        "size_bytes": 163840,
    },
    {
        "kind": "bundle_manifest",
        "path": "formal58-transport-repair/v1/cache-bundle-manifest-v1.json",
        "sha256": "15c8bf56ddad4f6f278599c32aaadcd813e0e016db0d523b4892eb47f8d9d144",
        "size_bytes": 13856,
    },
)

EXPECTED_STATES = (
    "source_a_identity",
    "formal_input_identity",
    "formal_training_completion",
    "checkpoint_completion",
    "fresh_feature_only_identity",
    "fresh_feature_only_prediction_completion",
    "label_blind_seal",
    "label_access_claim",
    "fresh_label_and_historical_reference_identity",
    "one_time_consumed_development_join",
    "development_report_completion",
    "immutable_replay_completion",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class SetConditionedV3Contract:
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path

    @property
    def formal_input(self) -> Mapping[str, Any]:
        return _mapping(self.data["formal58_input"], "formal58 input")

    @property
    def development_input(self) -> Mapping[str, Any]:
        return _mapping(
            self.data["fresh16_consumed_development_input"],
            "fresh16 consumed development input",
        )

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


def _validate_paths_and_prerequisites(source: Mapping[str, Any]) -> None:
    _exact(
        source.get("execution_b_runner_freeze_path"),
        EXECUTION_B_RUNNER_FREEZE_PATH,
        "Execution-B runner-freeze path",
    )
    for field in (
        "execution_b_must_be_direct_single_parent_of_source_a",
        "execution_b_runner_freeze_must_be_unique_diff",
        "execution_b_runner_freeze_must_be_absent_during_source_a_validation",
        "execution_b_must_be_clean_and_pushed",
    ):
        _exact(source.get(field), True, field)
    required = tuple(
        _safe_relative(path, "required source path")
        for path in _sequence(
            source.get("required_source_a_paths"), "required source paths"
        )
    )
    if len(required) != len(set(required)) or len(required) != 25:
        raise ValueError("required Source-A path inventory drifted")
    expected_new = {
        "README.md",
        CANONICAL_CONFIG_PATH,
        "code/causalcache/set_conditioned_v3_contract.py",
        "code/causalcache/set_conditioned_v3_data.py",
        "code/causalcache/set_conditioned_v3_training.py",
        "code/causalcache/set_conditioned_v3_exploration.py",
        "code/scripts/validate_set_conditioned_v3_contract.py",
        "code/scripts/run_set_conditioned_v3_exploration.py",
        "code/tests/test_set_conditioned_v3_contract.py",
        "code/tests/test_set_conditioned_v3_data.py",
        "code/tests/test_set_conditioned_v3_training.py",
        "code/tests/test_set_conditioned_v3_exploration.py",
        "docs/set_conditioned_v3_pair_residual.md",
        "docs/progress.md",
    }
    if not expected_new.issubset(required):
        raise ValueError("required Source-A implementation inventory drifted")
    prerequisites = _sequence(source.get("git_prerequisites"), "Git prerequisites")
    if len(prerequisites) != 11:
        raise ValueError("Git prerequisite inventory drifted")
    for index, raw in enumerate(prerequisites):
        item = _mapping(raw, f"Git prerequisite {index}")
        _exact_keys(item, {"path", "sha256", "size_bytes"}, "Git prerequisite")
        _safe_relative(item["path"], "Git prerequisite path")
        if _SHA256.fullmatch(str(item["sha256"])) is None:
            raise ValueError("Git prerequisite SHA-256 is malformed")
        if type(item["size_bytes"]) is not int or item["size_bytes"] <= 0:
            raise ValueError("Git prerequisite size is malformed")


def _validate_inputs(data: Mapping[str, Any]) -> None:
    formal = _mapping(data["formal58_input"], "formal58 input")
    _exact(formal.get("immutable_revision"), FORMAL_CACHE_REVISION, "formal revision")
    _exact(tuple(formal.get("exact_files", ())), EXPECTED_FORMAL_FILES, "formal files")
    _exact(
        {
            "trajectory_count": formal.get("trajectory_count"),
            "state_count": formal.get("state_count"),
            "candidate_counts": formal.get("candidate_counts"),
            "budget": formal.get("budget"),
            "fold_count": formal.get("fold_count"),
            "fold_sizes": formal.get("fold_sizes"),
        },
        {
            "trajectory_count": 58,
            "state_count": 174,
            "candidate_counts": [2, 3, 4],
            "budget": 2,
            "fold_count": 5,
            "fold_sizes": [12, 12, 12, 11, 11],
        },
        "formal geometry",
    )

    fresh = _mapping(
        data["fresh16_consumed_development_input"], "fresh16 input"
    )
    _exact(fresh.get("immutable_revision"), FRESH16_REVISION, "fresh16 revision")
    files = tuple(
        _mapping(item, "fresh16 exact file")
        for item in _sequence(fresh.get("exact_files"), "fresh16 exact files")
    )
    if len(files) != 4:
        raise ValueError("fresh16 exact-file inventory drifted")
    by_kind = {str(item.get("kind")): item for item in files}
    if set(by_kind) != {
        "bundle_manifest",
        "feature_states",
        "label_states",
        "historical_independent_decisions",
    }:
        raise ValueError("fresh16 exact-file kinds drifted")
    _exact(
        by_kind["feature_states"].get("sha256"),
        FRESH16_FEATURE_SHA256,
        "fresh16 feature hash",
    )
    _exact(
        by_kind["label_states"].get("sha256"),
        FRESH16_LABEL_SHA256,
        "fresh16 label hash",
    )
    _exact(
        by_kind["feature_states"].get("access_stage"),
        "train_seal",
        "fresh16 feature access stage",
    )
    _exact(
        by_kind["label_states"].get("access_stage"),
        "evaluate_after_label_blind_seal",
        "fresh16 label access stage",
    )
    for field in (
        "may_be_called_holdout",
        "may_be_called_test",
        "may_authorize_go",
        "may_change_v1_verdict",
        "may_tune_after_label_join",
    ):
        _exact(fresh.get(field), False, f"fresh16 {field}")
    _exact(fresh.get("one_label_join_total"), True, "fresh16 one-time join")


def _validate_science_and_model(data: Mapping[str, Any]) -> None:
    science = _mapping(data["scientific_contract"], "scientific contract")
    _exact(science.get("singleton_target"), "U_i=U({i})", "singleton target")
    _exact(
        science.get("pair_residual_target"),
        "R_ij=U({i,j})-U({i})-U({j})",
        "pair residual target",
    )
    _exact(science.get("candidate_event_maximum"), 4, "candidate maximum")
    _exact(science.get("budget"), 2, "budget")
    _exact(science.get("maximum_feasible_subset_count"), 11, "subset count")
    _exact(science.get("policy_rerun_count"), 0, "policy rerun count")

    model = _mapping(data["model_contract"], "model contract")
    _exact(model.get("event_input_dimension"), 200, "event input dimension")
    _exact(model.get("query_dimension"), 64, "query dimension")
    _exact(
        model.get("event_encoder"),
        {
            "layers": ["Linear(200,64)", "GELU(approximate=none)"],
            "output_dimension": 64,
        },
        "event encoder",
    )
    _exact(
        model.get("singleton_head"),
        {
            "layers": ["Linear(64,1)"],
            "output": "scaled_raw_singleton_utility",
        },
        "singleton head",
    )
    pair = _mapping(model.get("pair_features"), "pair features")
    _exact(pair.get("canonical_order"), "event_step_i_strictly_less_than_j", "pair order")
    _exact(
        pair.get("concatenation"),
        ["z_i", "z_j", "z_i_times_z_j", "abs_z_i_minus_z_j", "q64"],
        "pair feature concatenation",
    )
    _exact(pair.get("dimension"), 320, "pair feature dimension")
    _exact(
        model.get("pair_residual_head"),
        {
            "layers": [
                "Linear(320,64)",
                "GELU(approximate=none)",
                "Linear(64,1)",
            ],
            "output": "scaled_raw_pair_residual",
        },
        "pair residual head",
    )
    _exact(
        model.get("chronological_pair_features_are_intentionally_not_permutation_invariant"),
        True,
        "chronological pair parameterization",
    )
    _exact(model.get("iterative_conditional_mlp_reused"), False, "v1 model reuse")


def _validate_training_and_selection(data: Mapping[str, Any]) -> None:
    training = _mapping(data["training_contract"], "training contract")
    _exact(training.get("device"), "cpu", "training device")
    _exact(training.get("dtype"), "float32", "training dtype")
    _exact(training.get("learning_rates_in_grid_order"), [0.0003, 0.001], "LR grid")
    _exact(training.get("seeds_in_grid_order"), [0, 1, 2, 3, 4], "seed grid")
    _exact(training.get("maximum_epochs"), 500, "maximum epochs")
    _exact(training.get("patience_epochs"), 50, "patience")
    scaling = _mapping(training.get("target_scaling"), "target scaling")
    _exact(scaling.get("scope"), "fold_training_partition_only", "scale scope")
    _exact(
        scaling.get("formula"),
        "sqrt((equal_mean_Ui_squared+equal_mean_Uij_squared+equal_mean_Rij_squared)/3)",
        "RMS formula",
    )
    _exact(scaling.get("minimum_scale"), 1e-12, "RMS clamp")
    _exact(
        scaling.get("heldout_or_fresh_values_may_fit_scale"),
        False,
        "scale leakage firewall",
    )
    loss = _mapping(training.get("loss"), "loss")
    _exact(
        loss,
        {
            "singleton": "equal_weight_SmoothL1(Uhat_i_scaled,U_i_scaled,beta=1)",
            "pair_utility": "equal_weight_SmoothL1(Uhat_ij_scaled,U_ij_scaled,beta=1)",
            "pair_residual": "equal_weight_SmoothL1(Rhat_ij_scaled,R_ij_scaled,beta=1)",
            "regression": "(L_singleton+L_pair_utility+L_pair_residual)/3",
            "all_feasible_set_ranking": "equal_weight_softplus(-sign(U_left-U_right)*(Uhat_left-Uhat_right))",
            "ranking_comparisons": "all_unordered_pairs_among_empty_singletons_pairs",
            "ranking_equal_weight_order": "trajectory_then_state_then_untied_comparison",
            "ranking_raw_tie_epsilon": 1e-12,
            "ranking_weight": 0.25,
            "normalized_regression_weight": 0.0,
            "extra_sign_loss_weight": 0.0,
        },
        "loss contract",
    )
    _exact(
        training.get("single_seed_early_stop_and_lr_selection_variant"),
        "unguarded_pair_residual",
        "single-seed selection variant",
    )
    _exact(
        training.get("epoch_selection_metric"),
        "minimum_of_n4_raw_utility_over_exact_and_n4_normalized_recovery_over_exact",
        "OOF epoch metric",
    )
    bootstrap = _mapping(training.get("bootstrap"), "bootstrap")
    _exact(
        bootstrap,
        {
            "unit": "trajectory",
            "resamples": 10000,
            "confidence": 0.9,
            "interval": "percentile",
            "seed": 271828,
        },
        "bootstrap",
    )

    selection = _mapping(data["selection_contract"], "selection contract")
    _exact(
        selection.get("feasible_sets"),
        "empty_plus_all_singletons_plus_all_pairs",
        "feasible sets",
    )
    _exact(
        selection.get("tie_break"),
        [
            "higher_predicted_utility",
            "lower_cardinality",
            "lexicographically_lower_event_step_ids",
        ],
        "selection tie break",
    )
    _exact(selection.get("tie_epsilon"), 0.0, "selection tie epsilon")
    _exact(selection.get("safe_vote_threshold"), 4, "safe vote threshold")
    _exact(selection.get("safe_vote_total"), 5, "safe vote total")
    _exact(
        selection.get("safe_pair_candidate"),
        "argmax_of_five_seed_mean_unguarded_set_utility",
        "safe pair candidate",
    )
    _exact(
        selection.get("safe_base_candidate"),
        "argmax_of_five_seed_mean_additive_set_utility",
        "safe base candidate",
    )
    _exact(
        selection.get("safe_positive_margin_threshold"),
        4,
        "safe positive-margin threshold",
    )
    _exact(
        selection.get("safe_positive_margin_definition"),
        "individual_seed_F_m(pair_candidate)-F_m(base_candidate)_strictly_greater_than_zero",
        "safe positive-margin definition",
    )
    _exact(
        selection.get("safe_identical_candidate_rule"),
        "if_pair_candidate_equals_base_candidate_return_that_subset_without_switch_test",
        "safe identical-candidate rule",
    )
    _exact(
        selection.get("fallback"),
        "five_seed_mean_additive_exact_enumeration",
        "safe fallback",
    )
    _exact(selection.get("primary_selector"), "safe_pair_residual", "primary selector")
    _exact(
        selection.get("safe_selector_does_not_participate_in_single_seed_early_stopping"),
        True,
        "safe selector model-selection isolation",
    )
    _exact(
        selection.get("safe_trace_required_fields"),
        [
            "pair_candidate",
            "base_candidate",
            "seedwise_unguarded_argmax",
            "seedwise_pair_minus_base_margins",
            "pair_candidate_vote_count",
            "strictly_positive_margin_count",
            "used_pair_candidate",
        ],
        "safe trace",
    )
    variants = _sequence(selection.get("variants"), "selection variants")
    _exact(
        [item.get("name") for item in variants if isinstance(item, Mapping)],
        ["additive", "unguarded_pair_residual", "safe_pair_residual"],
        "selection variants",
    )


def _validate_firewall_and_outputs(data: Mapping[str, Any]) -> None:
    evaluation = _mapping(data["evaluation_contract"], "evaluation contract")
    _exact(evaluation.get("fresh16_is_development_only"), True, "development role")
    _exact(
        evaluation.get("development_interpretation_rule"),
        {
            "primary_contrast": "safe_pair_residual_minus_additive",
            "all_conditions_required_for_promising": True,
            "minimum_mean_normalized_delta": 0.01,
            "normalized_paired_bootstrap_lower_strictly_greater_than": 0.0,
            "mean_raw_delta_strictly_greater_than": 0.0,
            "n4_mean_normalized_delta_strictly_greater_than": 0.0,
            "minimum_positive_trajectory_count": 8,
            "failure_interpretation": "NO_DEVELOPMENT_EVIDENCE_TO_CONTINUE_SET_CONDITIONING",
            "pass_interpretation": "PROMISING_DEVELOPMENT_SIGNAL_FOR_A_SEPARATELY_FROZEN_FUTURE_STUDY",
        },
        "development interpretation rule",
    )
    forbidden = set(_sequence(evaluation.get("forbidden_interpretations"), "forbidden interpretations"))
    if forbidden != {"GO", "CONFIRMED", "HELDOUT_PASS", "TEST_PASS"}:
        raise ValueError("fresh16 forbidden interpretations drifted")

    firewall = _mapping(data["access_firewall"], "access firewall")
    source_counts = _mapping(firewall.get("source_a_counts"), "Source-A counts")
    if not source_counts or any(value != 0 for value in source_counts.values()):
        raise ValueError("Source-A access counts must all remain zero")
    train_counts = _mapping(
        firewall.get("train_seal_maximum_counts"), "train-seal counts"
    )
    for forbidden_field in (
        "fresh16_label_semantic_decode",
        "legacy_dev5_access",
        "confirm20_access",
        "matched_nll_evaluation",
        "closed_loop_episode",
        "raw_gui_access",
        "policy_forward",
        "gpu_job",
    ):
        _exact(train_counts.get(forbidden_field), 0, f"train-seal {forbidden_field}")
    evaluate_counts = _mapping(
        firewall.get("evaluate_maximum_counts"), "evaluate counts"
    )
    _exact(evaluate_counts.get("fresh16_label_access_claim"), 1, "label claim count")
    _exact(evaluate_counts.get("fresh16_label_file_decode"), 1, "label decode count")
    _exact(evaluate_counts.get("fresh16_join"), 1, "fresh16 join count")
    for forbidden_field in (
        "legacy_dev5_access",
        "confirm20_access",
        "matched_nll_evaluation",
        "closed_loop_episode",
        "raw_gui_access",
        "policy_forward",
        "gpu_job",
    ):
        _exact(evaluate_counts.get(forbidden_field), 0, f"evaluate {forbidden_field}")
    for field in (
        "checkpoint_and_prediction_seal_must_precede_label_access_claim",
        "execution_b_validation_must_precede_input_read_or_output_mutation",
        "transport_byte_possession_is_not_semantic_access",
        "label_transport_download_may_precede_claim",
        "label_access_claim_must_precede_label_open_parse_or_semantic_decode",
        "no_post_label_training_or_prediction_change",
    ):
        _exact(firewall.get(field), True, field)

    machine = _mapping(data["execution_state_machine"], "execution state machine")
    _exact(tuple(machine.get("ordered_states", ())), EXPECTED_STATES, "state order")
    _exact(machine.get("train_seal_terminal_state"), "label_blind_seal", "train-seal terminal")
    seal = _mapping(machine.get("label_blind_seal"), "label-blind seal")
    _exact(seal.get("path"), "label-blind-seal.json", "seal path")
    _exact(seal.get("status"), LABEL_BLIND_SEAL_STATUS, "seal status")
    _exact(
        seal.get("must_bind"),
        [
            "five_seed_safetensors",
            "formal_training_report",
            "fresh_feature_only_predictions_for_all_variants",
            "source_a_git_commit",
            "execution_b_git_commit",
            "runner_freeze_sha256",
            "contract_sha256",
        ],
        "label-blind seal bindings",
    )
    for counter in (
        "fresh_label_semantic_decode_count",
        "confirm20_access_count",
        "policy_forward_count",
        "gpu_job_count",
    ):
        _exact(seal.get(counter), 0, f"seal {counter}")
    _exact(machine.get("evaluate_requires_byte_identical_label_blind_seal"), True, "seal replay")
    _exact(machine.get("evaluate_may_not_retrain_or_repredict"), True, "post-seal mutation")

    runtime = _mapping(data["runtime_contract"], "runtime contract")
    _exact(
        {
            "device": runtime.get("device"),
            "gpu_count": runtime.get("gpu_count"),
            "gpu_preflight_required": runtime.get("gpu_preflight_required"),
            "torch_cuda_available_required": runtime.get("torch_cuda_available_required"),
            "policy_model_runtime_allowed": runtime.get("policy_model_runtime_allowed"),
        },
        {
            "device": "cpu",
            "gpu_count": 0,
            "gpu_preflight_required": False,
            "torch_cuda_available_required": False,
            "policy_model_runtime_allowed": False,
        },
        "CPU-only runtime",
    )
    namespace = _mapping(data["namespace_contract"], "namespace contract")
    _exact(
        namespace.get("execution_namespace"),
        "set-conditioned-v3-pair-residual-exploration-v1",
        "execution namespace",
    )
    _exact(namespace.get("must_not_write_mainline_namespaces"), True, "namespace isolation")

    operations = _mapping(
        data["source_only_operation_contract"], "source-only operations"
    )
    if not operations or any(value != 0 for value in operations.values()):
        raise ValueError("source-only operations must all remain zero")
    authorization = _mapping(data["authorization"], "authorization")
    _exact(
        authorization.get("execution_b_formal58_training_authorized_after_clean_pushed_source_a"),
        True,
        "formal execution authorization",
    )
    _exact(
        authorization.get("execution_b_one_fresh16_consumed_development_authorized_after_label_blind_seal"),
        True,
        "fresh16 execution authorization",
    )
    for field in (
        "fresh16_may_authorize_confirm",
        "confirm20_access_authorized",
        "legacy_dev5_access_authorized",
        "matched_nll_authorized",
        "closed_loop_authorized",
        "gpu_authorized",
        "policy_forward_authorized",
        "mainline_artifact_mutation_authorized",
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
        "scientific_contract",
        "model_contract",
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
    _exact(lineage.get("study_role"), "non_mainline_set_conditioning_exploration", "study role")
    _exact(lineage.get("v1_failure_is_immutable"), True, "v1 verdict immutability")
    failure = _mapping(lineage.get("v1_failure_summary"), "v1 failure summary")
    _exact(failure.get("required_outcome"), "NO_V2_CONDITIONAL_RESCUE", "v1 outcome")

    source = _mapping(data["source_freeze"], "source freeze")
    _exact(source.get("required_branch"), REQUIRED_BRANCH, "source branch")
    _exact(source.get("required_base_ancestor"), BASE_GIT_COMMIT, "source base")
    _exact(source.get("network_call_count"), 0, "source network count")
    _exact(source.get("execution_requires_clean_pushed_source_a_commit"), True, "execution freeze")
    _exact(source.get("source_validator_authorizes_execution"), False, "validator authorization")
    _validate_paths_and_prerequisites(source)
    _validate_inputs(data)
    _validate_science_and_model(data)
    _validate_training_and_selection(data)
    _validate_firewall_and_outputs(data)


def load_frozen_set_conditioned_v3_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path | None = None,
) -> SetConditionedV3Contract:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = root / source_path
    payload = _regular_file_bytes(source_path, label="v3 Source-A contract")
    digest = sha256_bytes(payload)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError(
            f"v3 Source-A contract hash drifted: expected {FROZEN_CONFIG_SHA256}, got {digest}"
        )
    data = _strict_json(payload, label="v3 Source-A contract")
    validate_contract_data(data)
    return SetConditionedV3Contract(
        data=data,
        sha256=digest,
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
    contract: SetConditionedV3Contract,
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
    inventory_by_path = {item["path"]: item for item in inventory}
    for relative, expected in prerequisites.items():
        observed = inventory_by_path.get(relative)
        if observed != expected:
            raise ValueError(f"frozen Git prerequisite bytes drifted: {relative}")
    frozen = tuple(inventory)
    return frozen, sha256_bytes(canonical_json_bytes(frozen))


def _validate_clean_pushed_branch(root: Path) -> tuple[str, str]:
    branch = _git(root, "branch", "--show-current")
    if branch != REQUIRED_BRANCH:
        raise ValueError("v3 validation is running on the wrong branch")
    head = _git(root, "rev-parse", "HEAD")
    if _COMMIT.fullmatch(head) is None:
        raise ValueError("Git HEAD is not a full lowercase commit")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ValueError("v3 execution source must have a clean worktree")
    tracking = _git(root, "rev-parse", f"origin/{REQUIRED_BRANCH}")
    if tracking != head:
        raise ValueError("v3 execution source must equal its pushed origin branch")
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


def execution_b_freeze_payload(
    contract: SetConditionedV3Contract,
    *,
    source_a_git_commit: str,
    source_inventory_sha256: str,
) -> dict[str, Any]:
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
        "confirm20_access_count": 0,
        "fresh16_role": "consumed_development_not_holdout_or_test",
    }


def _load_execution_b_freeze(
    contract: SetConditionedV3Contract,
    *,
    source_a_git_commit: str,
    source_inventory_sha256: str,
) -> Mapping[str, Any]:
    payload = _regular_file_bytes(
        contract.repository_root / EXECUTION_B_RUNNER_FREEZE_PATH,
        label="Execution-B runner freeze",
    )
    value = _strict_json(payload, label="Execution-B runner freeze")
    expected = execution_b_freeze_payload(
        contract,
        source_a_git_commit=source_a_git_commit,
        source_inventory_sha256=source_inventory_sha256,
    )
    if value != expected or payload != canonical_json_bytes(expected) + b"\n":
        raise ValueError("Execution-B runner freeze differs from canonical frozen bytes")
    return value


def validate_source_a(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    contract = load_frozen_set_conditioned_v3_contract(
        path, repository_root=repository_root
    )
    root = contract.repository_root
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("repository root differs from the Git top level")
    runner = root / EXECUTION_B_RUNNER_FREEZE_PATH
    if runner.exists() or runner.is_symlink():
        raise ValueError("Execution-B runner freeze must be absent from Source-A")
    branch, head = _validate_clean_pushed_branch(root)
    result = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", BASE_GIT_COMMIT, head],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("the frozen base commit is not an ancestor of Source-A")
    if _git(root, "remote", "get-url", "origin") != ORIGIN_URL:
        raise ValueError("origin URL differs from the frozen repository")

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
    }


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
    contract = load_frozen_set_conditioned_v3_contract(
        path,
        repository_root=repository_root,
    )
    root = contract.repository_root
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("repository root differs from the Git top level")
    branch, head = _validate_clean_pushed_branch(root)
    if head != execution_b_git_commit:
        raise ValueError("requested Execution-B commit differs from validated HEAD")
    if _live_remote_branch(root) != head:
        raise ValueError("Execution-B differs from the live origin branch")
    _git(root, "merge-base", "--is-ancestor", BASE_GIT_COMMIT, source_a_git_commit)
    parents = _git(root, "rev-list", "--parents", "-n", "1", head).split()
    if parents != [head, source_a_git_commit]:
        raise ValueError("Execution-B must be the direct single-parent child of Source-A")
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
        raise ValueError("Execution-B must differ from Source-A only by runner freeze")
    inventory, inventory_sha256 = _source_inventory(contract)
    freeze = _load_execution_b_freeze(
        contract,
        source_a_git_commit=source_a_git_commit,
        source_inventory_sha256=inventory_sha256,
    )
    runner_payload = canonical_json_bytes(freeze) + b"\n"
    if _git_blob(root, f"{head}:{EXECUTION_B_RUNNER_FREEZE_PATH}") != runner_payload:
        raise ValueError("Execution-B runner freeze differs from its committed blob")
    tree_record = _git(
        root,
        "ls-tree",
        head,
        "--",
        EXECUTION_B_RUNNER_FREEZE_PATH,
    )
    if not tree_record.startswith("100644 blob ") or not tree_record.endswith(
        f"\t{EXECUTION_B_RUNNER_FREEZE_PATH}"
    ):
        raise ValueError("Execution-B runner freeze must be a regular 100644 blob")
    runner_freeze_sha256 = sha256_bytes(runner_payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "VALID_SET_CONDITIONED_V3_PAIR_RESIDUAL_EXECUTION_B_V1",
        "contract_sha256": contract.sha256,
        "source_a_git_commit": source_a_git_commit,
        "execution_b_git_commit": head,
        "branch": branch,
        "direct_single_parent": True,
        "unique_diff": expected_change,
        "runner_freeze_sha256": runner_freeze_sha256,
        "source_inventory_sha256": inventory_sha256,
        "required_path_count": len(inventory),
        "runner_freeze": dict(freeze),
        "confirm20_access_count": 0,
        "gpu_job_count": 0,
    }


__all__ = [
    "BASE_GIT_COMMIT",
    "CANONICAL_CONFIG_PATH",
    "EXPECTED_FORMAL_FILES",
    "EXPECTED_STATES",
    "EXECUTION_B_FREEZE_STATUS",
    "EXECUTION_B_RUNNER_FREEZE_PATH",
    "FORMAL_CACHE_REVISION",
    "FRESH16_FEATURE_SHA256",
    "FRESH16_LABEL_SHA256",
    "FRESH16_REVISION",
    "FROZEN_CONFIG_SHA256",
    "LABEL_BLIND_SEAL_STATUS",
    "PROTOCOL_ID",
    "REQUIRED_BRANCH",
    "SCHEMA_VERSION",
    "SOURCE_STATUS",
    "SetConditionedV3Contract",
    "VALIDATION_STATUS",
    "canonical_json_bytes",
    "execution_b_freeze_payload",
    "load_frozen_set_conditioned_v3_contract",
    "sha256_bytes",
    "validate_contract_data",
    "validate_execution_b",
    "validate_source_a",
]
