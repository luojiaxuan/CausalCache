"""Frozen Source-A contract for the fresh-16 failure decomposition child."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_gate_v1_fresh16_failure_decomposition_v1"
SOURCE_STATUS = "source_only_frozen_before_fresh16_failure_decomposition"
SOURCE_VALIDATION_STATUS = "VALID_GATE_V1_FRESH16_FAILURE_DECOMPOSITION_SOURCE_A_V1"
RUNNER_FREEZE_STATUS = "FROZEN_GATE_V1_FRESH16_FAILURE_DECOMPOSITION_EXECUTION_B_V1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_gate_v1_fresh16_failure_decomposition_v1.json"
)
RUNNER_FREEZE_B_PATH = (
    "code/configs/causalcache_gate_v1_fresh16_failure_decomposition_runner_v1.json"
)
FROZEN_CONFIG_SHA256 = (
    "fa2cd3759150f838fce78b72a987d7a889ef23f5eec5ec41286ae69090104f1f"
)

PARENT_RESULT_COMMIT = "a04049e3b9a314dcbeb478210cb53cd20a207d32"
PARENT_SOURCE_A_COMMIT = "f0dd53b9a0249f259a833b4d8ad3ff26096a0ed1"
PARENT_EXECUTION_B_COMMIT = "ce523ff54ebfdc19a9c2bd49ad21548f0934a634"
PARENT_PAYLOAD_COMMIT = "9f0c61b9437773ca5d3f7e0cabd6e2a987e3c908"
PARENT_REPORT_COMMIT = "3541fe1ea2c46e555c29cc53483e6f3b809f8f81"
PARENT_TAG_OBJECT = "34d5928db4e82532f46f7596702aec3f018c4339"
PARENT_REPO = (
    "gavinlaw/causalcache-gate-v1-fresh16-claim-serialization-repair-mobile"
)
PARENT_TAG = "gate-v1-fresh16-claim-serialization-repair-v1"
PARENT_BUNDLE_PATH = "fresh16-eval/v1/manifests/bundle-manifest-v1.json"
PARENT_LABEL_PATH = "fresh16-eval/v1/caches/label-states-v1.jsonl"
PARENT_STATE_RECORD_PATH = (
    "fresh16-eval/v1/reports/primary-state-records-v1.jsonl"
)
PARENT_BUNDLE_SHA256 = (
    "a11e9d693c2fa33f5f5f97b123649608bdf98583693caa818e166d8009382844"
)
PARENT_BUNDLE_SIZE_BYTES = 2133

CHILD_REPO = "gavinlaw/causalcache-gate-v1-fresh16-failure-decomposition-mobile"
CHILD_TAG = "gate-v1-fresh16-failure-decomposition-v1"

REQUIRED_SOURCE_A_PATHS = (
    CANONICAL_CONFIG_PATH,
    "code/causalcache/gate_v1_fresh16_failure_decomposition_contract.py",
    "code/causalcache/gate_v1_fresh16_failure_decomposition.py",
    "code/causalcache/gate_v1_fresh16_failure_decomposition_runner.py",
    "code/scripts/manage_gate_v1_fresh16_failure_decomposition.py",
    "code/tests/test_gate_v1_fresh16_failure_decomposition_contract.py",
    "code/tests/test_gate_v1_fresh16_failure_decomposition.py",
    "code/tests/test_gate_v1_fresh16_failure_decomposition_runner.py",
    "code/configs/causalcache_gate_v1_fresh16_claim_serialization_repair_v1.json",
    "code/configs/causalcache_gate_v1_fresh16_claim_serialization_repair_runner_v1.json",
    "data/results/gate_v1_fresh16_claim_serialization_repair_v1/summary.json",
    "data/results/restoration_v2_2_selector_geometry_v2_repair/summary.json",
    "code/causalcache/gate_v1_fresh16.py",
    "code/causalcache/gate_v1_data.py",
    "code/causalcache/low_fidelity_v2.py",
    "code/causalcache/restoration_v2_2_label_table.py",
)

PARENT_DOWNLOAD_PATHS = (
    PARENT_BUNDLE_PATH,
    PARENT_LABEL_PATH,
    PARENT_STATE_RECORD_PATH,
)

CHILD_TARGETS = (
    "fresh16-failure-decomposition/v1/state-decomposition-v1.jsonl",
    "fresh16-failure-decomposition/v1/failure-decomposition-report-v1.json",
    "fresh16-failure-decomposition/v1/bundle-manifest-v1.json",
)

ORDERED_LOCAL_STATES = (
    "global_claim",
    "parent_readonly_attestation",
    "local_report_completion",
    "child_remote_base_receipt",
    "child_report_commit_receipt",
    "child_tag_receipt",
    "immutable_replay_completion",
    "completion_staging",
    "final_completion",
)

ALLOWED_PARENT_HF_API_METHODS = (
    "dataset_info",
    "list_repo_commits",
    "list_repo_files",
    "list_repo_refs",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class FailureDecompositionContract:
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path

    @property
    def source(self) -> Mapping[str, Any]:
        return _mapping(self.data["source_freeze"], "source freeze")

    @property
    def parent(self) -> Mapping[str, Any]:
        return _mapping(self.data["parent_primary"], "parent primary")

    @property
    def destination(self) -> Mapping[str, Any]:
        return _mapping(self.data["destination"], "destination")


@dataclass(frozen=True)
class SourceIdentity:
    git_commit: str
    origin_main_git_commit: str
    branch: str
    origin_url: str
    source_inventory: tuple[Mapping[str, Any], ...]
    loaded_module_inventory: tuple[Mapping[str, Any], ...]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
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


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} schema drifted")


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be nonempty text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be canonical relative POSIX")
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
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite constant in {label}: {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    return _mapping(value, label)


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


def _validate_parent(value: Any) -> Mapping[str, Any]:
    parent = dict(_mapping(value, "parent primary"))
    if (
        parent.get("source_a_git_commit") != PARENT_SOURCE_A_COMMIT
        or parent.get("execution_b_git_commit") != PARENT_EXECUTION_B_COMMIT
        or parent.get("result_git_commit") != PARENT_RESULT_COMMIT
    ):
        raise ValueError("parent Git lineage drifted")
    parent_contract = _mapping(parent.get("parent_contract"), "parent contract")
    runner = _mapping(
        parent.get("execution_b_runner_freeze"), "parent runner freeze"
    )
    summary = _mapping(parent.get("result_summary"), "parent result summary")
    if (
        parent_contract
        != {
            "path": "code/configs/causalcache_gate_v1_fresh16_claim_serialization_repair_v1.json",
            "sha256": "3979573be235d630ee2f46dc23be8747a843190c9b57ee81e1b3a17b4416d8c7",
            "size_bytes": 11723,
        }
        or runner
        != {
            "path": (
                "code/configs/causalcache_gate_v1_fresh16_"
                "claim_serialization_repair_runner_v1.json"
            ),
            "sha256": "c35dccd0d540cd01bd0f93fce11289c40dbada577e856e997a9dcd52347403d8",
            "size_bytes": 28936,
        }
        or summary
        != {
            "path": "data/results/gate_v1_fresh16_claim_serialization_repair_v1/summary.json",
            "sha256": "621c09d13c027bfbc2835be894dd3de8fbc69c2f03a8605ab3a951cba753e626",
            "size_bytes": 9419,
            "status": "REVALIDATED_GATE_V1_FRESH16_PRIMARY_EVALUATION_V1",
        }
    ):
        raise ValueError("parent Git artifact identity drifted")
    hf = _mapping(parent.get("hf"), "parent HF")
    bundle = _mapping(hf.get("bundle_manifest"), "parent bundle")
    if (
        hf.get("repo") != PARENT_REPO
        or hf.get("repo_type") != "dataset"
        or hf.get("private") is not True
        or hf.get("tag") != PARENT_TAG
        or hf.get("annotated_tag_object") != PARENT_TAG_OBJECT
        or hf.get("payload_commit") != PARENT_PAYLOAD_COMMIT
        or hf.get("report_commit") != PARENT_REPORT_COMMIT
        or hf.get("tag_resolved_commit") != PARENT_REPORT_COMMIT
        or bundle
        != {
            "path": PARENT_BUNDLE_PATH,
            "sha256": PARENT_BUNDLE_SHA256,
            "size_bytes": PARENT_BUNDLE_SIZE_BYTES,
        }
    ):
        raise ValueError("parent immutable HF identity drifted")
    verdict = _mapping(parent.get("verdict"), "parent verdict")
    if verdict != {
        "go_selector": False,
        "go_set_conditioning_primary": False,
        "implementation_valid": True,
        "reclassification_allowed": False,
    }:
        raise ValueError("parent v1 verdict boundary drifted")
    return parent


def _validate_source_freeze(value: Any) -> Mapping[str, Any]:
    source = dict(_mapping(value, "source freeze"))
    if (
        source.get("branch") != "main"
        or source.get("origin_name") != "origin"
        or source.get("origin_url")
        != "https://github.com/luojiaxuan/CausalCache.git"
        or source.get("head_must_equal_origin_main") is not True
        or source.get("worktree_must_be_clean") is not True
        or source.get("source_a_local_tracking_validation_network_call_count") != 0
    ):
        raise ValueError("Source-A Git policy drifted")
    paths = tuple(_sequence(source.get("required_source_a_paths"), "source paths"))
    if paths != REQUIRED_SOURCE_A_PATHS or len(paths) != len(set(paths)):
        raise ValueError("required Source-A path inventory drifted")
    runner = dict(_mapping(source.get("execution_b_runner_freeze"), "runner freeze"))
    if runner != {
        "bind_git_prerequisites": True,
        "bind_loaded_module_inventory_sha256": True,
        "bind_required_source_a_paths": True,
        "bind_source_a_inventory_sha256": True,
        "direct_single_parent_child_of_source_a": True,
        "must_be_absent_during_source_only_validation": True,
        "only_allowed_execution_b_source_tree_diff": True,
        "path": RUNNER_FREEZE_B_PATH,
    }:
        raise ValueError("Execution-B runner-freeze contract drifted")
    prerequisites = _sequence(source.get("git_prerequisites"), "Git prerequisites")
    if len(prerequisites) != 6:
        raise ValueError("Git prerequisite denominator drifted")
    paths_seen: set[str] = set()
    for index, raw in enumerate(prerequisites):
        record = _mapping(raw, f"Git prerequisite {index}")
        path = _safe_relative(record.get("path"), "Git prerequisite path")
        if path in paths_seen:
            raise ValueError("Git prerequisite path is duplicated")
        paths_seen.add(path)
        if (
            not isinstance(record.get("sha256"), str)
            or _SHA256.fullmatch(record["sha256"]) is None
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] <= 0
        ):
            raise ValueError("Git prerequisite byte identity is malformed")
        commit = record.get("git_commit")
        if commit is not None and (
            not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None
        ):
            raise ValueError("Git prerequisite commit is malformed")
    return source


def _validate_input(value: Any) -> Mapping[str, Any]:
    inputs = dict(_mapping(value, "input contract"))
    targets = _sequence(inputs.get("exact_force_download_targets"), "input targets")
    paths = tuple(_mapping(item, "input target").get("path") for item in targets)
    if (
        paths != PARENT_DOWNLOAD_PATHS
        or len(targets) != 3
        or tuple(inputs.get("allowed_parent_hf_api_methods", ()))
        != ALLOWED_PARENT_HF_API_METHODS
        or inputs.get("parent_commit_chain")
        != [PARENT_PAYLOAD_COMMIT, PARENT_REPORT_COMMIT]
        or inputs.get("parent_remote_mutation_call_count") != 0
        or inputs.get("force_download") is not True
        or inputs.get("fresh_download_directory_must_start_empty") is not True
        or inputs.get("pre_post_parent_identity_stable") is not True
        or inputs.get("selective_reader_only") is not True
    ):
        raise ValueError("selective parent input contract drifted")
    bundle = _mapping(targets[0], "bundle input")
    if bundle != {
        "identity": "direct_frozen_identity",
        "path": PARENT_BUNDLE_PATH,
        "sha256": PARENT_BUNDLE_SHA256,
        "size_bytes": PARENT_BUNDLE_SIZE_BYTES,
    }:
        raise ValueError("direct bundle identity drifted")
    if (
        _mapping(targets[1], "label input")
        != {
            "expected_record_count": 48,
            "identity": "bound_by_frozen_bundle_payload_inventory",
            "path": PARENT_LABEL_PATH,
        }
        or _mapping(targets[2], "state-record input")
        != {
            "expected_record_count": 48,
            "identity": "bound_by_frozen_bundle_report_inventory",
            "path": PARENT_STATE_RECORD_PATH,
        }
    ):
        raise ValueError("bundle-transitive input identities drifted")
    return inputs


def _validate_analysis(value: Any) -> Mapping[str, Any]:
    analysis = dict(_mapping(value, "analysis contract"))
    if (
        analysis.get("budget") != 2
        or analysis.get("candidate_event_counts") != [2, 3, 4]
        or analysis.get("trajectory_equal_aggregation") is not True
        or analysis.get("diagnostic_only") is not True
        or analysis.get("student")
        != "sealed_parent_conditional_ensemble_decisions_only"
        or analysis.get("decomposition_identity")
        != "exact_minus_student_equals_search_plus_distillation"
    ):
        raise ValueError("failure-decomposition objective drifted")
    normalization = _mapping(analysis.get("normalization"), "normalization")
    if normalization != {
        "denominator": "D_empty",
        "eligible_only_when_strictly_greater_than": 1e-12,
        "ineligible_value": None,
    }:
        raise ValueError("normalization threshold drifted")
    interaction = _mapping(analysis.get("interaction_mass"), "interaction mass")
    if interaction.get("historical_train_only_tertile_cutpoints_by_n") != {
        "2": [0.2909362464858063, 0.42337624776450483],
        "3": [0.2760179595936389, 0.3818779740249449],
        "4": [0.19346783303918597, 0.21532511632530205],
    }:
        raise ValueError("historical interaction thresholds drifted")
    strata = _mapping(analysis.get("report_strata"), "report strata")
    if (
        strata.get("decision_step_ids") != [4, 5, 6]
        or strata.get("selection_rounds") != [0, 1]
        or strata.get("candidate_event_count") is not True
        or strata.get("interaction_strength") is not True
        or len(_sequence(strata.get("denominator_bins"), "denominator bins")) != 4
    ):
        raise ValueError("failure-decomposition strata drifted")
    greedy = _mapping(analysis.get("true_conditional_greedy"), "true greedy")
    if greedy != {
        "allow_early_stop": True,
        "rescore_after_each_addition": True,
        "strictly_positive_gain_required": True,
        "tie_break": "lower_event_step_id",
    }:
        raise ValueError("true conditional-greedy definition drifted")
    return analysis


def _validate_routing(value: Any) -> Mapping[str, Any]:
    routing = dict(_mapping(value, "routing contract"))
    aliases = _mapping(routing.get("method_aliases"), "routing method aliases")
    if (
        aliases.get("C") != "sealed_learned_conditional_ensemble"
        or aliases.get("E") != "exact_subset_oracle"
        or aliases.get("G") != "true_conditional_marginal_greedy"
        or aliases.get("I") != "sealed_learned_independent_ensemble"
        or _mapping(aliases.get("J"), "oracle independent J")
        != {
            "budget": 2,
            "definition": "oracle_budget_conditioned_independent",
            "event_score": (
                "0.5_times_empty_marginal_plus_mean_of_other_singleton_base_"
                "conditional_marginals"
            ),
            "strictly_positive_score_required": True,
            "tie_break": "lower_event_step_id",
            "top_b": True,
        }
    ):
        raise ValueError("routing method aliases or oracle J drifted")
    bootstrap = _mapping(routing.get("bootstrap"), "routing bootstrap")
    if bootstrap != {
        "confidence": 0.9,
        "interval": "percentile",
        "resamples": 10000,
        "seed": 271828,
        "unit": "trajectory",
    }:
        raise ValueError("routing bootstrap contract drifted")
    search = _mapping(routing.get("search_pass"), "search pass")
    if (
        search.get("normalized_recovery_G_over_E_minimum") != 0.95
        or search.get("raw_utility_G_over_E_minimum") != 0.95
        or search.get("n3_raw_utility_G_over_E_minimum") != 0.9
        or search.get("n4_raw_utility_G_over_E_minimum") != 0.9
        or len(_sequence(search.get("all_required"), "search requirements")) != 4
    ):
        raise ValueError("search-pass routing thresholds drifted")
    headroom = _mapping(
        routing.get("oracle_set_headroom_pass"), "oracle set headroom"
    )
    if (
        headroom.get("normalized_mean_delta_G_minus_J_minimum") != 0.02
        or headroom.get(
            "paired_trajectory_bootstrap_lower_strictly_greater_than"
        )
        != 0.0
        or headroom.get("positive_trajectory_count_minimum") != 12
        or headroom.get("trajectory_count") != 16
        or len(_sequence(headroom.get("all_required"), "headroom requirements"))
        != 3
    ):
        raise ValueError("oracle-set-headroom routing thresholds drifted")
    student = _mapping(routing.get("material_student_gap"), "material student gap")
    if (
        student.get("normalized_recovery_C_over_G_maximum_exclusive") != 0.9
        or student.get("raw_utility_C_over_G_maximum_exclusive") != 0.95
        or len(_sequence(student.get("any_required"), "student-gap alternatives"))
        != 2
    ):
        raise ValueError("material-student-gap routing thresholds drifted")
    prefix = _mapping(
        routing.get("prefix_completion_dominance"),
        "prefix/completion dominance",
    )
    if prefix != {
        "diagnostic_only": True,
        "minimum_share": 0.6,
        "share_definition": (
            "positive_distillation_regret_from_prefix_or_completion_errors_"
            "divided_by_total_positive_distillation_regret"
        ),
    }:
        raise ValueError("prefix/completion diagnostic threshold drifted")
    denominator = _mapping(
        routing.get("denominator_diagnostic"), "denominator diagnostic"
    )
    if denominator != {
        "bottom_within_n_quartile_exact_raw_utility_share_maximum": 0.2,
        "bottom_within_n_quartile_positive_normalized_regret_share_minimum": 0.5,
        "diagnostic_only_cannot_change_routing": True,
        "raw_minus_normalized_ratio_gap_minimum": 0.15,
        "raw_utility_over_exact_raw_minimum": 0.9,
        "remove_bottom_within_n_quartile_normalized_ratio_improvement_minimum": 0.1,
    }:
        raise ValueError("denominator diagnostic thresholds drifted")
    seed = _mapping(routing.get("seed_isolated_diagnostic"), "seed diagnostic")
    if seed != {
        "drop_worst_seed_population_std_maximum": 0.08,
        "individual_seed_E_ratio_minimum": 0.75,
        "individual_seed_pass_count_minimum": 4,
        "individual_seed_total_count": 5,
        "median_individual_seed_E_ratio_minimum": 0.75,
        "not_a_routing_gate": True,
    }:
        raise ValueError("seed-isolated diagnostic thresholds drifted")
    learned = _mapping(routing.get("learned_C_vs_I_replay"), "learned replay")
    if learned != {
        "diagnostic_only": True,
        "must_equal_parent_sealed_report": True,
    }:
        raise ValueError("learned C-vs-I replay boundary drifted")
    final = _mapping(routing.get("final_routing"), "final routing")
    if final != {
        "all_required": [
            "search_pass",
            "oracle_set_headroom_pass",
            "material_student_gap",
        ],
        "diagnostics_not_routing_gates": [
            "prefix_completion_dominance",
            "denominator_diagnostic",
            "seed_isolated_diagnostic",
            "learned_C_vs_I_replay",
        ],
        "fail_status": "NO_V2_CONDITIONAL_RESCUE",
        "fresh16_already_consumed": True,
        "fresh16_may_be_used_as_v2_holdout": False,
        "parent_v1_verdict_may_change": False,
        "pass_status": "ONE_V2_CONDITIONAL_RESCUE",
    }:
        raise ValueError("final one-rescue routing rule drifted")
    return routing


def _validate_boundaries(data: Mapping[str, Any]) -> None:
    output = _mapping(data["output_contract"], "output contract")
    if (
        tuple(output.get("exact_targets", ())) != CHILD_TARGETS
        or output.get("state_record_count") != 48
        or output.get("trajectory_record_count_embedded_in_report") != 16
        or output.get("report_inherits_v1_verdict_without_reclassification")
        is not True
    ):
        raise ValueError("child output contract drifted")
    destination = _mapping(data["destination"], "destination")
    if (
        destination.get("repo") != CHILD_REPO
        or destination.get("repo_type") != "dataset"
        or destination.get("private") is not True
        or destination.get("tag") != CHILD_TAG
        or destination.get("single_exact_report_commit") is not True
        or destination.get("annotated_tag_required") is not True
        or destination.get("remote_base_receipt_before_first_mutation") is not True
        or destination.get("tag_must_resolve_to_report_commit") is not True
        or destination.get("completed_replay_remote_mutation_count") != 0
        or destination.get("allowed_remote_states")
        != [
            "ABSENT",
            "EMPTY",
            "COMPLETE",
        ]
    ):
        raise ValueError("child HF destination contract drifted")
    local = _mapping(data["local_first_state_machine"], "local state machine")
    if (
        tuple(local.get("ordered_states", ())) != ORDERED_LOCAL_STATES
        or local.get("state_file_mode") != 0o600
        or local.get("artifact_file_mode") != 0o444
        or local.get("o_excl_or_byte_identical_reuse") is not True
        or local.get("crash_recovery_after_remote_mutation") is not False
        or local.get("partial_attempt_requires_versioned_repair") is not True
        or local.get("pre_mutation_byte_identical_restart_only") is not True
        or local.get("no_overwrite") is not True
        or local.get("final_completion_is_hard_link_to_staging") is not True
        or local.get("completion_is_last_first_execution_local_mutation") is not True
    ):
        raise ValueError("local-first state machine drifted")
    source_only = _mapping(
        data["source_only_operation_contract"], "source-only operations"
    )
    if not source_only or any(
        type(value) is not int or value != 0 for value in source_only.values()
    ):
        raise ValueError("Source-A operation counters must all remain zero")
    authorization = _mapping(data["authorization"], "authorization")
    if not authorization or any(value is not False for value in authorization.values()):
        raise ValueError("Source-A authorizations must all remain false")
    execution = _mapping(data["execution_fixed_operation_contract"], "execution counts")
    expected_nonzero = {
        "child_exact_target_count": 3,
        "parent_force_download_count": 3,
        "sealed_fresh16_state_decode_count": 48,
        "state_decomposition_record_count": 48,
        "trajectory_decomposition_record_count": 16,
    }
    for key, value in execution.items():
        if type(value) is not int or value != expected_nonzero.get(key, 0):
            raise ValueError("fixed execution operation contract drifted")
    runtime = _mapping(data["runtime_contract"], "runtime contract")
    if runtime != {
        "cpu_only": True,
        "gpu_count": 0,
        "gpu_preflight_required": False,
        "model_runtime_allowed": False,
        "network_scope": "parent_read_only_then_new_child_publication_only",
        "torch_import_required": False,
    }:
        raise ValueError("CPU-only runtime contract drifted")


def validate_contract_data(value: Any) -> Mapping[str, Any]:
    data = dict(_mapping(value, "failure-decomposition contract"))
    _exact_keys(
        data,
        {
            "schema_version",
            "protocol_id",
            "status",
            "parent_primary",
            "source_freeze",
            "input_contract",
            "analysis_contract",
            "routing_contract",
            "output_contract",
            "destination",
            "local_first_state_machine",
            "runtime_contract",
            "source_only_operation_contract",
            "execution_fixed_operation_contract",
            "authorization",
        },
        "failure-decomposition contract",
    )
    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("protocol_id") != PROTOCOL_ID
        or data.get("status") != SOURCE_STATUS
    ):
        raise ValueError("failure-decomposition source identity drifted")
    _validate_parent(data["parent_primary"])
    _validate_source_freeze(data["source_freeze"])
    _validate_input(data["input_contract"])
    _validate_analysis(data["analysis_contract"])
    _validate_routing(data["routing_contract"])
    _validate_boundaries(data)
    return data


def load_frozen_failure_decomposition_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> FailureDecompositionContract:
    root = Path(repository_root)
    if not root.is_absolute():
        root = root.resolve(strict=True)
    if root.is_symlink() or root.resolve(strict=True) != root.absolute():
        raise ValueError("repository root must be one real canonical directory")
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = root / source_path
    payload = _regular_file_bytes(source_path, label="failure-decomposition config")
    digest = sha256_bytes(payload)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError("failure-decomposition config SHA256 drifted")
    data = validate_contract_data(_strict_json(payload, label="failure-decomposition config"))
    return FailureDecompositionContract(
        data=data,
        sha256=digest,
        repository_root=root,
        source_path=source_path,
    )


def _git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ValueError(f"git {' '.join(arguments)} failed")
    return completed.stdout


def _validate_prerequisites(
    contract: FailureDecompositionContract,
    *,
    head: str,
) -> None:
    for raw in contract.source["git_prerequisites"]:
        record = _mapping(raw, "Git prerequisite")
        relative = _safe_relative(record["path"], "Git prerequisite path")
        payload = _regular_file_bytes(
            contract.repository_root / relative,
            label=f"Git prerequisite {relative}",
        )
        if (
            len(payload) != record["size_bytes"]
            or sha256_bytes(payload) != record["sha256"]
            or payload != _git(contract.repository_root, "show", f"{head}:{relative}")
        ):
            raise ValueError(f"Git prerequisite bytes drifted: {relative}")
        commit = record.get("git_commit")
        if commit is not None:
            _git(contract.repository_root, "merge-base", "--is-ancestor", commit, head)
            if payload != _git(contract.repository_root, "show", f"{commit}:{relative}"):
                raise ValueError(f"Git prerequisite creation bytes drifted: {relative}")


def _source_inventory(
    root: Path,
    *,
    commit: str,
) -> tuple[Mapping[str, Any], ...]:
    records = []
    for relative in REQUIRED_SOURCE_A_PATHS:
        payload = _regular_file_bytes(root / relative, label=f"Source-A {relative}")
        if payload != _git(root, "show", f"{commit}:{relative}"):
            raise ValueError(f"Source-A blob differs from commit: {relative}")
        records.append(
            {
                "path": relative,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    return tuple(records)


def _loaded_module_inventory(
    root: Path,
    *,
    commit: str,
) -> tuple[Mapping[str, Any], ...]:
    code_root = root / "code"
    required = set(REQUIRED_SOURCE_A_PATHS)
    records = []
    for name, module in sorted(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if not isinstance(source, str) or not source.endswith(".py"):
            continue
        path = Path(source)
        if not path.is_absolute():
            continue
        try:
            relative = (Path("code") / path.relative_to(code_root)).as_posix()
        except ValueError:
            continue
        if relative not in required:
            continue
        payload = _regular_file_bytes(path, label=f"loaded module {name}")
        if payload != _git(root, "show", f"{commit}:{relative}"):
            raise ValueError(f"loaded module differs from Source-A: {name}")
        records.append(
            {
                "module": name,
                "path": relative,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    if not records:
        raise ValueError("loaded Source-A module inventory is empty")
    return tuple(records)


def _source_record(source: SourceIdentity) -> Mapping[str, Any]:
    source_inventory = list(source.source_inventory)
    loaded = list(source.loaded_module_inventory)
    return {
        "git_commit": source.git_commit,
        "origin_main_git_commit": source.origin_main_git_commit,
        "branch": source.branch,
        "origin_url": source.origin_url,
        "source_inventory": source_inventory,
        "source_inventory_sha256": sha256_bytes(canonical_json_bytes(source_inventory)),
        "loaded_module_inventory": loaded,
        "loaded_module_inventory_sha256": sha256_bytes(canonical_json_bytes(loaded)),
    }


def validate_source_a(
    contract: FailureDecompositionContract,
    *,
    expected_source_a_git_commit: str | None = None,
) -> Mapping[str, Any]:
    runner_path = contract.repository_root / RUNNER_FREEZE_B_PATH
    if os.path.lexists(runner_path):
        raise ValueError("Execution-B runner freeze must be absent from Source-A")
    source = contract.source
    head = _git(contract.repository_root, "rev-parse", "HEAD").decode().strip()
    tracking = _git(
        contract.repository_root,
        "rev-parse",
        f"{source['origin_name']}/{source['branch']}",
    ).decode().strip()
    branch = _git(
        contract.repository_root, "branch", "--show-current"
    ).decode().strip()
    origin_url = _git(
        contract.repository_root, "remote", "get-url", source["origin_name"]
    ).decode().strip()
    dirty = _git(
        contract.repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if (
        _COMMIT.fullmatch(head) is None
        or (expected_source_a_git_commit is not None and head != expected_source_a_git_commit)
        or head != tracking
        or branch != source["branch"]
        or origin_url != source["origin_url"]
        or dirty
    ):
        raise ValueError("Source-A must be clean pushed canonical main")
    _git(
        contract.repository_root,
        "merge-base",
        "--is-ancestor",
        PARENT_RESULT_COMMIT,
        head,
    )
    _validate_prerequisites(contract, head=head)
    identity = SourceIdentity(
        git_commit=head,
        origin_main_git_commit=tracking,
        branch=branch,
        origin_url=origin_url,
        source_inventory=_source_inventory(contract.repository_root, commit=head),
        loaded_module_inventory=_loaded_module_inventory(
            contract.repository_root, commit=head
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": SOURCE_VALIDATION_STATUS,
        "contract_sha256": contract.sha256,
        "source_a": _source_record(identity),
        **dict(contract.data["source_only_operation_contract"]),
        "failure_decomposition_executed": False,
        "execution_authorized": False,
        "runner_freeze_b_present": False,
        "runner_freeze_b_path": RUNNER_FREEZE_B_PATH,
    }


def _runner_freeze_payload(
    contract: FailureDecompositionContract,
    validation: Mapping[str, Any],
) -> Mapping[str, Any]:
    source = _mapping(validation.get("source_a"), "validated Source-A")
    source_inventory = _sequence(source.get("source_inventory"), "source inventory")
    loaded = _sequence(source.get("loaded_module_inventory"), "loaded modules")
    prerequisites = list(contract.source["git_prerequisites"])
    paths = list(REQUIRED_SOURCE_A_PATHS)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": RUNNER_FREEZE_STATUS,
        "contract_sha256": contract.sha256,
        "source_a_git_commit": source["git_commit"],
        "source_a_origin_main_git_commit": source["origin_main_git_commit"],
        "execution_b_direct_parent_required": source["git_commit"],
        "execution_b_required_unique_diff": [RUNNER_FREEZE_B_PATH],
        "required_source_a_paths": paths,
        "required_source_a_paths_sha256": sha256_bytes(canonical_json_bytes(paths)),
        "git_prerequisites": prerequisites,
        "git_prerequisites_sha256": sha256_bytes(
            canonical_json_bytes(prerequisites)
        ),
        "source_blob_inventory": list(source_inventory),
        "source_blob_inventory_sha256": source["source_inventory_sha256"],
        "source_a_inventory_sha256": source["source_inventory_sha256"],
        "loaded_module_inventory": list(loaded),
        "loaded_module_inventory_sha256": source[
            "loaded_module_inventory_sha256"
        ],
        "parent_result_git_commit": PARENT_RESULT_COMMIT,
        "parent_report_commit": PARENT_REPORT_COMMIT,
        "parent_exact_download_paths": list(PARENT_DOWNLOAD_PATHS),
        "child_repo": CHILD_REPO,
        "child_tag": CHILD_TAG,
        "v1_verdict_reclassification_allowed": False,
        "legacy_dev5_access_authorized": False,
        "confirm20_access_authorized": False,
        "matched_nll_authorized": False,
        "closed_loop_authorized": False,
        "execution_authorized_after_clean_pushed_b_only": True,
    }


def runner_freeze_bytes(
    contract: FailureDecompositionContract,
    validation: Mapping[str, Any],
) -> bytes:
    return pretty_json_bytes(_runner_freeze_payload(contract, validation))


def _exclusive_write(path: Path, payload: bytes) -> None:
    try:
        parent = path.parent.lstat()
    except OSError as error:
        raise ValueError("runner-freeze parent is missing") from error
    if not stat.S_ISDIR(parent.st_mode) or path.parent.is_symlink():
        raise ValueError("runner-freeze parent must be a real directory")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("runner-freeze write made no progress")
            view = view[written:]
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if _regular_file_bytes(path, label="Execution-B runner freeze") != payload:
        raise ValueError("Execution-B runner-freeze readback differs")


def materialize_runner_freeze(
    contract: FailureDecompositionContract,
    *,
    expected_source_a_git_commit: str,
) -> Mapping[str, Any]:
    validation = validate_source_a(
        contract,
        expected_source_a_git_commit=expected_source_a_git_commit,
    )
    payload = runner_freeze_bytes(contract, validation)
    path = contract.repository_root / RUNNER_FREEZE_B_PATH
    _exclusive_write(path, payload)
    status = tuple(
        line
        for line in _git(
            contract.repository_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        .decode("utf-8")
        .splitlines()
        if line
    )
    if status != (f"?? {RUNNER_FREEZE_B_PATH}",):
        raise ValueError("runner freeze is not the unique Source-A worktree diff")
    return _strict_json(payload, label="materialized Execution-B runner freeze")


def load_runner_freeze(
    contract: FailureDecompositionContract,
) -> Mapping[str, Any]:
    path = contract.repository_root / RUNNER_FREEZE_B_PATH
    payload = _regular_file_bytes(path, label="Execution-B runner freeze")
    value = dict(_strict_json(payload, label="Execution-B runner freeze"))
    if payload != pretty_json_bytes(value):
        raise ValueError("Execution-B runner freeze is not canonical pretty JSON")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "contract_sha256",
        "source_a_git_commit",
        "source_a_origin_main_git_commit",
        "execution_b_direct_parent_required",
        "execution_b_required_unique_diff",
        "required_source_a_paths",
        "required_source_a_paths_sha256",
        "git_prerequisites",
        "git_prerequisites_sha256",
        "source_blob_inventory",
        "source_blob_inventory_sha256",
        "source_a_inventory_sha256",
        "loaded_module_inventory",
        "loaded_module_inventory_sha256",
        "parent_result_git_commit",
        "parent_report_commit",
        "parent_exact_download_paths",
        "child_repo",
        "child_tag",
        "v1_verdict_reclassification_allowed",
        "legacy_dev5_access_authorized",
        "confirm20_access_authorized",
        "matched_nll_authorized",
        "closed_loop_authorized",
        "execution_authorized_after_clean_pushed_b_only",
    }
    paths = list(REQUIRED_SOURCE_A_PATHS)
    prerequisites = list(contract.source["git_prerequisites"])
    inventory = value.get("source_blob_inventory")
    loaded = value.get("loaded_module_inventory")
    if (
        set(value) != expected_keys
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != RUNNER_FREEZE_STATUS
        or value.get("contract_sha256") != contract.sha256
        or not isinstance(value.get("source_a_git_commit"), str)
        or _COMMIT.fullmatch(value["source_a_git_commit"]) is None
        or value.get("source_a_origin_main_git_commit")
        != value.get("source_a_git_commit")
        or value.get("execution_b_direct_parent_required")
        != value.get("source_a_git_commit")
        or value.get("execution_b_required_unique_diff")
        != [RUNNER_FREEZE_B_PATH]
        or value.get("required_source_a_paths") != paths
        or value.get("required_source_a_paths_sha256")
        != sha256_bytes(canonical_json_bytes(paths))
        or value.get("git_prerequisites") != prerequisites
        or value.get("git_prerequisites_sha256")
        != sha256_bytes(canonical_json_bytes(prerequisites))
        or not isinstance(inventory, list)
        or len(inventory) != len(paths)
        or [item.get("path") for item in inventory if isinstance(item, Mapping)]
        != paths
        or value.get("source_blob_inventory_sha256")
        != sha256_bytes(canonical_json_bytes(inventory))
        or value.get("source_a_inventory_sha256")
        != value.get("source_blob_inventory_sha256")
        or not isinstance(loaded, list)
        or not loaded
        or value.get("loaded_module_inventory_sha256")
        != sha256_bytes(canonical_json_bytes(loaded))
        or value.get("parent_result_git_commit") != PARENT_RESULT_COMMIT
        or value.get("parent_report_commit") != PARENT_REPORT_COMMIT
        or value.get("parent_exact_download_paths") != list(PARENT_DOWNLOAD_PATHS)
        or value.get("child_repo") != CHILD_REPO
        or value.get("child_tag") != CHILD_TAG
        or value.get("v1_verdict_reclassification_allowed") is not False
        or value.get("legacy_dev5_access_authorized") is not False
        or value.get("confirm20_access_authorized") is not False
        or value.get("matched_nll_authorized") is not False
        or value.get("closed_loop_authorized") is not False
        or value.get("execution_authorized_after_clean_pushed_b_only") is not True
    ):
        raise ValueError("Execution-B runner-freeze identity drifted")
    for index, item in enumerate(inventory):
        record = _mapping(item, f"frozen source blob {index}")
        if (
            set(record) != {"path", "sha256", "size_bytes"}
            or not isinstance(record["sha256"], str)
            or _SHA256.fullmatch(record["sha256"]) is None
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
        ):
            raise ValueError("Execution-B frozen source inventory is malformed")
    for index, item in enumerate(loaded):
        record = _mapping(item, f"frozen loaded module {index}")
        if (
            set(record) != {"module", "path", "sha256", "size_bytes"}
            or not isinstance(record["module"], str)
            or not record["module"]
            or record["path"] not in paths
            or not isinstance(record["sha256"], str)
            or _SHA256.fullmatch(record["sha256"]) is None
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
        ):
            raise ValueError("Execution-B loaded-module inventory is malformed")
    return value


def _live_remote_main(root: Path, remote: str) -> str:
    output = _git(
        root,
        "ls-remote",
        "--exit-code",
        remote,
        "refs/heads/main",
    ).decode("ascii")
    suffix = "\trefs/heads/main"
    lines = output.splitlines()
    if len(lines) != 1 or not lines[0].endswith(suffix):
        raise ValueError("canonical live remote-main response drifted")
    commit = lines[0][: -len(suffix)]
    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("canonical live remote main is malformed")
    return commit


def validate_execution_b_source(
    contract: FailureDecompositionContract,
    *,
    expected_execution_b_git_commit: str,
) -> Mapping[str, Any]:
    if _COMMIT.fullmatch(expected_execution_b_git_commit) is None:
        raise ValueError("expected Execution-B commit is malformed")
    freeze = load_runner_freeze(contract)
    source = contract.source
    root = contract.repository_root
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    tracking = _git(
        root,
        "rev-parse",
        f"{source['origin_name']}/{source['branch']}",
    ).decode().strip()
    branch = _git(root, "branch", "--show-current").decode().strip()
    origin_url = _git(
        root, "remote", "get-url", source["origin_name"]
    ).decode().strip()
    dirty = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if (
        head != expected_execution_b_git_commit
        or head != tracking
        or branch != source["branch"]
        or origin_url != source["origin_url"]
        or dirty
    ):
        raise ValueError("Execution-B must be clean pushed canonical main")
    live = _live_remote_main(root, source["origin_name"])
    if live != head:
        raise ValueError("Execution-B differs from live canonical main")
    source_a = freeze["source_a_git_commit"]
    parents = (
        _git(root, "rev-list", "--parents", "-n", "1", head)
        .decode("ascii")
        .strip()
        .split()
    )
    if parents != [head, source_a]:
        raise ValueError("Execution-B must be the direct single-parent child of Source-A")
    changed = tuple(
        line
        for line in _git(root, "diff", "--name-only", source_a, head)
        .decode("utf-8")
        .splitlines()
        if line
    )
    if changed != (RUNNER_FREEZE_B_PATH,):
        raise ValueError("Execution-B differs from Source-A outside runner freeze")
    runner_payload = _regular_file_bytes(
        root / RUNNER_FREEZE_B_PATH,
        label="Execution-B runner freeze",
    )
    if runner_payload != _git(root, "show", f"{head}:{RUNNER_FREEZE_B_PATH}"):
        raise ValueError("Execution-B runner-freeze bytes differ from its commit")
    _validate_prerequisites(contract, head=head)
    inventory = _source_inventory(root, commit=head)
    if list(inventory) != freeze["source_blob_inventory"]:
        raise ValueError("Execution-B source blobs differ from Source-A freeze")
    loaded = _loaded_module_inventory(root, commit=head)
    frozen_loaded = {
        item["module"]: item for item in freeze["loaded_module_inventory"]
    }
    if {item["module"]: item for item in loaded} != frozen_loaded:
        raise ValueError("Execution-B loaded modules differ from Source-A freeze")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "VALID_GATE_V1_FRESH16_FAILURE_DECOMPOSITION_EXECUTION_B_V1",
        "source_a_git_commit": source_a,
        "execution_b_git_commit": head,
        "origin_main_git_commit": tracking,
        "live_origin_main_git_commit": live,
        "direct_single_parent": True,
        "unique_diff": RUNNER_FREEZE_B_PATH,
        "contract_sha256": contract.sha256,
        "source_inventory_sha256": freeze["source_a_inventory_sha256"],
        "loaded_module_inventory_sha256": freeze[
            "loaded_module_inventory_sha256"
        ],
    }


__all__ = [
    "ALLOWED_PARENT_HF_API_METHODS",
    "CANONICAL_CONFIG_PATH",
    "CHILD_REPO",
    "CHILD_TAG",
    "CHILD_TARGETS",
    "FROZEN_CONFIG_SHA256",
    "FailureDecompositionContract",
    "ORDERED_LOCAL_STATES",
    "PARENT_DOWNLOAD_PATHS",
    "PROTOCOL_ID",
    "REQUIRED_SOURCE_A_PATHS",
    "RUNNER_FREEZE_B_PATH",
    "RUNNER_FREEZE_STATUS",
    "SCHEMA_VERSION",
    "SOURCE_STATUS",
    "SOURCE_VALIDATION_STATUS",
    "load_frozen_failure_decomposition_contract",
    "load_runner_freeze",
    "materialize_runner_freeze",
    "runner_freeze_bytes",
    "validate_contract_data",
    "validate_execution_b_source",
    "validate_source_a",
]
