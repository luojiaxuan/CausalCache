"""Fail-closed source-only contract for the validation-12 closed-loop probe."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_exploratory_closed_loop_validation12_v1"
SOURCE_STATUS = "source_only_frozen_before_exploratory_closed_loop_execution"
VALIDATION_STATUS = "VALID_SOURCE_ONLY_EXPLORATORY_CLOSED_LOOP_VALIDATION12_V1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_exploratory_closed_loop_validation12_v1.json"
)
FROZEN_CONFIG_SHA256 = (
    "1cae85d4e132a69dcf126c6d51274fc03974c3678475080db93c8747b062cfae"
)
ROSTER_MANIFEST_PATH = (
    "data/manifests/exploratory_closed_loop_validation12_v1.json"
)
ROSTER_MANIFEST_SHA256 = (
    "8e68e6eb1adde5627e83ce07cf4e8d7cef4843250a2e1b58baa247af57a07a42"
)
PARENT_NO_GO = "NO_GO_INDEPENDENT_CONFIRM"
ARMS = (
    "independent_B2",
    "recent_B2",
    "ocr_rgb_B2",
    "conditional_B2",
    "summary_B0",
)
ROUTES = (
    "ADVANCE_TO_NEW_TRAIN60_PROTOCOL",
    "STOP_CURRENT_INDEPENDENT_CLOSED_LOOP_DIRECTION",
    "INCONCLUSIVE_DEVELOPMENT_ONLY",
    "INVALID_EXPLORATORY_CLOSED_LOOP_EXECUTION_V1",
)
EXPECTED_SECTION_SHA256 = {
    "scientific_scope": "1b9aab0a03a9478de445105cae41cdfdbd0986ea2d5b65c62ddfceb83ea570ae",
    "parent_no_go": "c9d6ed55221519f936abe1cf3793161ce22157bc0cc865db27384636de424b63",
    "roster": "58b09c40a1c0fe745c62419e1c7f03e143a84eb0a1284e28ad78219a0859dd99",
    "policy": "a4b283ca5509d143e6f9482bad7db9137815da7133831236598d1ded68306bf9",
    "gate": "76b1c65536c4a5cc96c15231fb1db83124d71ecfa7611d31a66c5d0a7c2ad40b",
    "ocr": "7fb7be7e7c39f826813d562e710dfdadec4b4d4839888ef6ecc119e86ce0b53c",
    "memory_and_arms": "6f68f343c18ac4b21a50366b01710f459f2c15e6ac51eb593b13230074c9a939",
    "online_rollout": "959c30251731b87a9ed4fb5383026a9a2a15b84e4fc0128ad1298274b440bc0a",
    "evaluation": "29a418e75096d912ab6429f0490ad33932d2e4558177762b2fbe7f80f58d62e6",
    "validity_and_routing": "0c0d3ac023ed81b2cc3a49383ffcbadda4090ebfef4659cad6b9b762aa4a7f7f",
    "runtime": "b228d839dd85842905b1ec25be1cfea44f37870dd86c5c925dcc3535553b517e",
    "locked_followups": "7030127b5b80092a539cf2927866df219e3c0b12ce81c5bcc3b911c71fd3a50d",
    "source_only_operation_contract": "e6931ff30fedb8b5f572ff5f2764be863ad70d554e1c4a021284c111aa691b4c",
    "authorization": "6f4fc1cee5aef316aa7913e4b2c60d00ecf98573159787060c64c0194b75323d",
}
TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    *EXPECTED_SECTION_SHA256,
}


@dataclass(frozen=True)
class ExploratoryClosedLoopContract:
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path

    @property
    def roster(self) -> Mapping[str, Any]:
        return _mapping(self.data["roster"], "roster")

    @property
    def arms(self) -> Mapping[str, Any]:
        return _mapping(self.data["memory_and_arms"], "memory and arms")

    @property
    def runtime(self) -> Mapping[str, Any]:
        return _mapping(self.data["runtime"], "runtime")

    @property
    def locks(self) -> Mapping[str, Any]:
        return _mapping(self.data["locked_followups"], "locked followups")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
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


def _equal(actual: Any, expected: Any, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(
            f"{label} drifted: expected {expected!r}, got {actual!r}"
        )


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _strict_json_bytes(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {label}: {raw}")
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
            raise ValueError(f"{label} must be a regular file")
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
    if fingerprint(before) != fingerprint(after) or len(payload) != after.st_size:
        raise ValueError(f"{label} changed while being read")
    return payload


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = _mapping(config[name], name.replace("_", " "))
    observed = sha256_bytes(canonical_json_bytes(value))
    expected = EXPECTED_SECTION_SHA256[name]
    if observed != expected:
        raise ValueError(
            f"{name} section drifted: expected {expected}, got {observed}"
        )
    return value


def _validate_parent_and_roster(config: Mapping[str, Any]) -> None:
    parent = _section(config, "parent_no_go")
    result = _mapping(parent["result_summary"], "parent result summary")
    _safe_relative(result["path"], "parent result path")
    _equal(result["scientific_outcome"], PARENT_NO_GO, "parent outcome")
    _equal(parent["historical_result_preserved"], True, "parent preservation")
    for key in (
        "historical_closed_loop_authorized",
        "historical_closed_loop_executed",
        "historical_matched_nll_authorized",
        "historical_sealed_test_authorized",
        "threshold_model_data_seed_or_comparator_change_allowed",
    ):
        _equal(parent[key], False, f"parent lock {key}")

    roster = _section(config, "roster")
    manifest = _mapping(roster["manifest"], "roster manifest")
    _equal(manifest["path"], ROSTER_MANIFEST_PATH, "roster manifest path")
    _equal(manifest["sha256"], ROSTER_MANIFEST_SHA256, "roster manifest SHA256")
    _equal(roster["partition"], "validation", "roster partition")
    _equal(roster["template_count"], 12, "template denominator")
    _equal(roster["arm_count"], len(ARMS), "arm denominator")
    _equal(roster["episode_count"], 60, "episode denominator")
    if roster["template_count"] * roster["arm_count"] != roster["episode_count"]:
        raise ValueError("roster episode denominator is internally inconsistent")


def _validate_policy_gate_and_ocr(config: Mapping[str, Any]) -> None:
    policy = _section(config, "policy")
    _equal(
        policy["repo"], "mPLUG/GUI-Owl-1.5-8B-Instruct", "policy repo"
    )
    _equal(
        policy["revision"],
        "06d5faecff74840bab2be2425e9c42667a5d04fc",
        "policy revision",
    )
    _equal(policy["effective_visual_tokens_per_image"], 2560, "visual tokens")
    _equal(policy["policy_training_allowed"], False, "policy training lock")

    gate = _section(config, "gate")
    checkpoints = tuple(_sequence(gate["checkpoints"], "gate checkpoints"))
    if len(checkpoints) != 10:
        raise ValueError("gate must bind exactly ten checkpoints")
    identities: list[tuple[str, int]] = []
    for index, raw in enumerate(checkpoints):
        checkpoint = _mapping(raw, f"gate checkpoint {index}")
        family = checkpoint.get("family")
        seed = checkpoint.get("seed")
        if family not in {"conditional", "independent"} or type(seed) is not int:
            raise ValueError(f"gate checkpoint {index} identity is invalid")
        identities.append((family, seed))
    expected = [
        *(('conditional', seed) for seed in range(5)),
        *(('independent', seed) for seed in range(5)),
    ]
    if identities != expected:
        raise ValueError("gate checkpoint family or seed order drifted")
    _equal(gate["training_or_optimizer_allowed"], False, "gate training lock")

    ocr = _section(config, "ocr")
    _equal(
        ocr["backend_id"],
        "rapidocr-3.8.4-ppocrv5-mobile-en-cpu-v1",
        "OCR backend",
    )
    _equal(ocr["provider"], "CPUExecutionProvider", "OCR provider")
    _equal(ocr["feature_or_ocr_fallback_allowed"], False, "OCR fallback lock")


def _validate_rollout_and_evaluation(config: Mapping[str, Any]) -> None:
    arms = _section(config, "memory_and_arms")
    _equal(tuple(arms["ordered_arms"]), ARMS, "ordered arm inventory")
    _equal(
        arms["all_candidates_rescored_at_every_policy_decision"],
        True,
        "per-decision reselection",
    )
    _equal(
        arms["current_equivalent_newest_post_state_is_candidate"],
        False,
        "current-equivalent exclusion",
    )
    _equal(arms["maximum_high_fidelity_events"], 2, "B2 capacity")
    _equal(arms["summary_B0_high_fidelity_events"], 0, "B0 capacity")

    rollout = _section(config, "online_rollout")
    _equal(
        rollout["each_task_arm_has_independent_lifecycle"],
        True,
        "arm lifecycle isolation",
    )
    _equal(
        rollout["fixed_trajectory_replay_across_arms_allowed"],
        False,
        "trajectory replay lock",
    )
    for key in (
        "hidden_retry_allowed",
        "outcome_based_early_stop_allowed",
        "top_up_allowed",
        "completed_episode_or_arm_deletion_allowed",
    ):
        _equal(rollout[key], False, f"rollout lock {key}")

    evaluation = _section(config, "evaluation")
    _equal(
        evaluation["primary_contrast"],
        "independent_B2_minus_recent_B2",
        "primary contrast",
    )
    _equal(
        evaluation["statistical_significance_or_long_horizon_claim_allowed"],
        False,
        "exploratory claim lock",
    )


def _validate_validity_runtime_and_locks(config: Mapping[str, Any]) -> None:
    routing = _section(config, "validity_and_routing")
    observed_routes = (
        routing["advance_route"],
        routing["stop_route"],
        routing["fallback_route"],
        routing["invalid_route"],
    )
    _equal(observed_routes, ROUTES, "route inventory")
    validity = _mapping(routing["validity"], "validity")
    _equal(validity["exact_episode_record_count"], 60, "valid episode denominator")
    _equal(
        validity["missing_episode_record_outcome"],
        ROUTES[3],
        "missing episode route",
    )
    _equal(
        routing["conditional_success_may_rescue_independent_route"],
        False,
        "conditional rescue lock",
    )

    runtime = _section(config, "runtime")
    _equal(runtime["policy_host"], "aries", "policy host")
    _equal(runtime["emulator_host"], "aries", "emulator host")
    _equal(runtime["gpu_type"], "NVIDIA_RTX_A6000", "GPU type")
    _equal(runtime["maximum_worker_count"], 4, "maximum worker count")
    _equal(runtime["cross_host_transport_required"], False, "cross-host transport")
    _equal(runtime["hyper_policy_allowed_in_v1"], False, "Hyper policy lock")

    locks = _section(config, "locked_followups")
    for key in (
        "androidworld_train_access_count",
        "sealed_androidworld_test_access_count",
        "matched_nll_evaluation_count",
        "paper_primary_claim_count",
    ):
        _equal(locks[key], 0, f"follow-up lock {key}")
    _equal(locks["sealed_test75_status"], "LOCKED", "sealed test lock")
    _equal(locks["matched_nll_status"], "LOCKED", "matched NLL lock")
    _equal(
        locks["train60_protocol_status"],
        "LOCKED_REQUIRES_ADVANCE_AND_NEW_VERSIONED_CONTRACT",
        "train-60 lock",
    )


def _validate_source_only(config: Mapping[str, Any]) -> None:
    operations = _section(config, "source_only_operation_contract")
    if not operations:
        raise ValueError("source-only operation inventory is empty")
    for key, value in operations.items():
        if type(value) is not int or value != 0:
            raise ValueError(f"source-only operation must be exact zero: {key}")

    authorization = _section(config, "authorization")
    if not authorization:
        raise ValueError("source-only authorization inventory is empty")
    for key, value in authorization.items():
        if value is not False:
            raise PermissionError(f"source-only authorization must remain false: {key}")


def validate_exploratory_closed_loop_contract(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if set(config) != TOP_LEVEL_KEYS:
        raise ValueError("exploratory closed-loop top-level key inventory drifted")
    _equal(config["schema_version"], SCHEMA_VERSION, "schema version")
    _equal(config["protocol_id"], PROTOCOL_ID, "protocol id")
    _equal(config["status"], SOURCE_STATUS, "source status")
    _section(config, "scientific_scope")
    _validate_parent_and_roster(config)
    _validate_policy_gate_and_ocr(config)
    _validate_rollout_and_evaluation(config)
    _validate_validity_runtime_and_locks(config)
    _validate_source_only(config)
    return {
        "status": VALIDATION_STATUS,
        "protocol_id": PROTOCOL_ID,
        "parent_scientific_outcome": PARENT_NO_GO,
        "roster_manifest_path": ROSTER_MANIFEST_PATH,
        "roster_manifest_sha256": ROSTER_MANIFEST_SHA256,
        "partition": "validation",
        "template_count": 12,
        "arm_count": 5,
        "episode_count": 60,
        "ordered_arms": list(ARMS),
        "reselect_every_policy_decision": True,
        "policy_host": "aries",
        "emulator_host": "aries",
        "maximum_worker_count": 4,
        "routes": list(ROUTES),
        "train60_locked": True,
        "sealed_test_locked": True,
        "matched_nll_locked": True,
        "paper_claim_locked": True,
        "execution_authorized": False,
    }


def _canonical_contract_path(root: Path, supplied: str | Path) -> Path:
    raw = Path(supplied)
    candidate = raw if raw.is_absolute() else root / raw
    canonical_raw = root / CANONICAL_CONFIG_PATH
    canonical = canonical_raw.resolve()
    if (
        candidate.resolve() != canonical
        or candidate.is_symlink()
        or canonical_raw.is_symlink()
    ):
        raise ValueError("exploratory closed-loop contract path is not canonical")
    return canonical


def load_frozen_exploratory_closed_loop_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> ExploratoryClosedLoopContract:
    root = Path(repository_root).resolve()
    source = _canonical_contract_path(root, path)
    payload = _regular_file_bytes(source, label="exploratory closed-loop contract")
    observed = sha256_bytes(payload)
    if observed != FROZEN_CONFIG_SHA256:
        raise ValueError(
            "exploratory closed-loop contract bytes drifted: "
            f"expected {FROZEN_CONFIG_SHA256}, got {observed}"
        )
    config = _strict_json_bytes(payload, label="exploratory closed-loop contract")
    validate_exploratory_closed_loop_contract(config)
    return ExploratoryClosedLoopContract(
        data=config,
        sha256=observed,
        repository_root=root,
        source_path=source,
    )


def validate_source_only_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    contract = load_frozen_exploratory_closed_loop_contract(
        path, repository_root=repository_root
    )
    result = validate_exploratory_closed_loop_contract(contract.data)
    operations = _mapping(
        contract.data["source_only_operation_contract"],
        "source-only operation contract",
    )
    return {
        **result,
        "config_path": CANONICAL_CONFIG_PATH,
        "config_sha256": contract.sha256,
        "source_contract_read_count": 1,
        "only_read_path": CANONICAL_CONFIG_PATH,
        "bound_identity_sections": [
            "parent_no_go",
            "roster",
            "policy",
            "gate",
            "ocr",
        ],
        "artifact_access_count": 0,
        "bound_artifacts_opened_or_verified": False,
        "source_only_operation_counts": dict(operations),
    }


__all__ = [
    "ARMS",
    "CANONICAL_CONFIG_PATH",
    "EXPECTED_SECTION_SHA256",
    "ExploratoryClosedLoopContract",
    "FROZEN_CONFIG_SHA256",
    "PARENT_NO_GO",
    "PROTOCOL_ID",
    "ROSTER_MANIFEST_PATH",
    "ROSTER_MANIFEST_SHA256",
    "ROUTES",
    "SCHEMA_VERSION",
    "SOURCE_STATUS",
    "VALIDATION_STATUS",
    "canonical_json_bytes",
    "load_frozen_exploratory_closed_loop_contract",
    "sha256_bytes",
    "validate_exploratory_closed_loop_contract",
    "validate_source_only_contract",
]
