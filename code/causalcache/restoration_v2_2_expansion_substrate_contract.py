"""Source-only freeze contract for the 192-state label-expansion substrate."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "0.1.0"
PROTOCOL_ID = "causalcache_restoration_v2_2_label_expansion_substrate_v1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_expansion_substrate_v1.json"
)
CANONICAL_COMPLETION_PATH = (
    "data/results/restoration_v2_2_label_expansion_derived/artifact.json"
)
EXPANSION_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_label_expansion_v1.json"
)
EXPANSION_SELECTION_PATH = (
    "data/manifests/restoration_v2_2_label_expansion_selection.json"
)
EXPANSION_EXPOSURE_PATH = (
    "data/manifests/restoration_v2_2_label_expansion_exposure.json"
)
COMPLETION_SCHEMA_PATH = (
    "code/configs/causalcache_restoration_v2_2_label_expansion_derived_"
    "completion.schema.json"
)
DERIVED_PAYLOAD_PREFIX = "derived/restoration-v2-label-expansion-v1"
DERIVED_PAYLOAD_MANIFEST_PATH = f"{DERIVED_PAYLOAD_PREFIX}/manifest.json"
DERIVED_ARTIFACT_PATHS = (
    ".gitattributes",
    "README.md",
    f"{DERIVED_PAYLOAD_PREFIX}/images-00000-of-00001.tar",
    DERIVED_PAYLOAD_MANIFEST_PATH,
    f"{DERIVED_PAYLOAD_PREFIX}/ocr-records-00000-of-00001.jsonl",
    f"{DERIVED_PAYLOAD_PREFIX}/trajectories-00000-of-00001.jsonl",
)
DERIVED_REPO = "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile"
EXPANSION_EXPOSURE_SHA256 = (
    "e5f7e7b5ffeb71ac117b5cb4a470eb31648978a734edb9a521125d3bda29a323"
)
PRESERVED_OLD_TAG = "restoration-v2-derived-v1.0.0"
PRESERVED_OLD_REVISION = "89f136abaff797e14fe758a198996e51032a10a6"
EXPECTED_DERIVED_PROTOCOL_ID = (
    "causalcache_restoration_v2_label_expansion_derived_v1"
)
EXPECTED_DERIVED_STATUS = "POLICY_BLIND_LABEL_EXPANSION_DATASET_MATERIALIZED"
EXPECTED_COMPLETION_PROTOCOL_ID = (
    "causalcache_restoration_v2_2_label_expansion_derived_completion_v1"
)
EXPECTED_COMPLETION_STATUS = "VERIFIED_LABEL_EXPANSION_DERIVED_ARTIFACT"
EXPECTED_VALIDATOR_OUTCOME = (
    "PASSED_GUIODYSSEY_RESTORATION_V2_EXPANSION_VALIDATION"
)
ROLE_ORDER = ("gate_train_expansion", "gate_development_expansion")
ROLE_TRAJECTORY_COUNTS = {"gate_train_expansion": 48, "gate_development_expansion": 16}
ROLE_STATE_COUNTS = {"gate_train_expansion": 144, "gate_development_expansion": 48}
EXPECTED_DERIVED_COUNTS = {
    "trajectory_count": 64,
    "event_count": 320,
    "state_count": 192,
    "image_member_count": 384,
    "ocr_record_count": 384,
}
EXPECTED_LABEL_WORKLOAD = {
    "trajectories": 64,
    "states": 192,
    "full_subset_distance_rows": 1792,
    "deployment_conditional_edges": 1856,
    "teacher_forwards": 1984,
}
EXPECTED_SUBSTRATE_COUNTS = {
    "generation_call_count": 384,
    "teacher_forward_count": 576,
    "kl_measurement_count": 384,
}
EXPECTED_STACK = {
    "container_image_digest": (
        "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
    ),
    "python_version": "3.12.3",
    "torch_version": "2.11.0+cu130",
    "torch_cuda_version": "13.0",
    "cudnn_version": 91900,
    "transformers_version": "5.6.0",
    "nvidia_driver_version": "570.172.08",
    "gpu_name": "NVIDIA H200",
}
INHERITED_V2_2_EAGER_SOURCE_LOCK_PATHS = (
    "code/configs/causalcache_restoration_v2_2_eager.json",
    "code/causalcache/restoration_v2_2_eager_contract.py",
    "code/causalcache/policy/gui_owl_v2_2_eager_runtime.py",
    "code/causalcache/restoration_v2_2_eager_artifact.py",
    "code/scripts/run_restoration_v2_2_eager_substrate.py",
    "code/scripts/manage_restoration_v2_2_eager_artifact.py",
    "code/scripts/validate_restoration_v2_2_eager_contract.py",
    "code/configs/causalcache_restoration_v2_1_full_45.json",
    "code/causalcache/restoration_v2_1_full_45_contract.py",
    "code/causalcache/restoration_v2_1_full_45_artifact.py",
    "code/scripts/run_restoration_v2_1_full_45_substrate.py",
    "code/scripts/manage_restoration_v2_1_full_45_artifact.py",
    "code/scripts/validate_restoration_v2_1_full_45_contract.py",
    "code/scripts/run_restoration_v2_1_interface_pilot.py",
    "code/causalcache/restoration_v2_1_contract.py",
    "code/causalcache/restoration_v2_1_pilot_artifact.py",
    "code/scripts/manage_restoration_v2_1_pilot_artifact.py",
    "code/causalcache/restoration_v2_1_processor_audit.py",
    "code/scripts/audit_gui_owl_v2_1_processor.py",
    "code/causalcache/data/restoration_v2_1_processor_inputs.py",
    "code/causalcache/data/restoration_v2_screening.py",
    "code/causalcache/policy/gui_owl_v2_1.py",
    "code/causalcache/policy/gui_owl_v2_1_runtime.py",
    "code/causalcache/restoration_v2_gpu_kl.py",
    "code/causalcache/data/guiodyssey_restoration_v2.py",
    "code/causalcache/data/restoration_v2_selection.py",
    "code/causalcache/data/guiodyssey.py",
    "code/causalcache/data/guiodyssey_independent.py",
    "code/causalcache/low_fidelity_v2.py",
    "code/causalcache/restoration_v2_contract.py",
    "code/causalcache/restoration_v2_text_backend.py",
    "code/causalcache/schema.py",
    "code/causalcache/policy/gui_owl_v2.py",
    "code/causalcache/policy/gui_owl_v2_runtime.py",
    "code/causalcache/policy/gui_owl_v2_vision.py",
    "code/causalcache/restoration_v2_baselines.py",
    "code/configs/causalcache_restoration_v2_1_pilot.json",
    "code/configs/causalcache_restoration_v2.json",
    "code/configs/restoration_v2_ocr_backend.json",
    "code/configs/gui_owl_1_5_8b_snapshot.json",
    "data/manifests/restoration_v2_selection.json",
    "data/manifests/restoration_v2_ocr_backend.json",
    "data/manifests/restoration_v2_derived_artifact.json",
    "data/manifests/restoration_v2_real_screen_source.json",
    "data/results/restoration_v2_1_interface_pilot/artifact.json",
    "data/results/restoration_v2_1_processor_preflight/artifact.json",
    "data/results/restoration_v2_1_full_45_substrate/artifact.json",
    "data/results/restoration_v2_1_full_45_substrate/summary.json",
    "data/results/spatial_reference_audit_v1/artifact.json",
    "data/results/spatial_reference_audit_v1/summary.json",
)
EXPANSION_SOURCE_LOCK_PATHS = (
    COMPLETION_SCHEMA_PATH,
    "code/causalcache/restoration_v2_2_expansion_substrate_contract.py",
    "code/causalcache/restoration_v2_2_expansion_substrate_inputs.py",
    "code/scripts/materialize_restoration_v2_2_expansion_substrate_contract.py",
    "code/scripts/validate_restoration_v2_2_expansion_substrate_contract.py",
    "code/causalcache/data/guiodyssey_restoration_v2_expansion.py",
    "code/causalcache/restoration_v2_2_label_expansion.py",
    "code/scripts/build_guiodyssey_restoration_v2_expansion.py",
    "code/scripts/validate_guiodyssey_restoration_v2_expansion.py",
    "code/scripts/validate_restoration_v2_ocr_backend.py",
    "code/scripts/validate_restoration_v2_real_screen.py",
    "code/configs/independent_reference_gate_v1.json",
    "data/manifests/independent_reference_gate_v1_source_files.json",
)
SOURCE_LOCK_PATHS = tuple(
    dict.fromkeys(
        (*INHERITED_V2_2_EAGER_SOURCE_LOCK_PATHS, *EXPANSION_SOURCE_LOCK_PATHS)
    )
)
RESERVED_EXECUTION_SOURCE_PATHS = (
    "code/causalcache/restoration_v2_2_expansion_substrate_artifact.py",
    "code/scripts/run_restoration_v2_2_expansion_substrate.py",
    "code/scripts/manage_restoration_v2_2_expansion_substrate_artifact.py",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_json_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    value = json.loads(
        payload,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload, value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys drifted")


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _require_git_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or GIT_SHA_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a full lowercase Git SHA")
    return value


def _safe_relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"{name} must be a canonical repository-relative path")
    if path.as_posix() != value:
        raise ValueError(f"{name} must use canonical POSIX syntax")
    return value


def _repo_file(root: Path, relative: str) -> Path:
    canonical = _safe_relative_path(relative, "repository file")
    path = root.joinpath(*PurePosixPath(canonical).parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required repository file is missing: {relative}")
    return path


def _identity(root: Path, relative: str) -> dict[str, Any]:
    path = _repo_file(root, relative)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _git_blob(root: Path, revision: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(f"{relative} is not committed at source Git revision")
    return result.stdout


def _state_projection(selection: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    if selection.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("expansion selection schema drifted")
    if selection.get("protocol_id") != (
        "causalcache_restoration_v2_2_label_expansion_v1"
    ):
        raise ValueError("expansion selection protocol drifted")
    if selection.get("status") != "FROZEN_POLICY_BLIND_STRUCTURAL_SPLIT":
        raise ValueError("expansion selection is not frozen")
    if selection.get("structural_manifest_only") is not True:
        raise ValueError("expansion selection must be structural only")
    for field in ("policy_output_generated", "restoration_output_generated"):
        if selection.get(field) is not False:
            raise ValueError(f"expansion selection {field} must be false")
    if selection.get("expected_workload", {}).get("total") != EXPECTED_LABEL_WORKLOAD:
        raise ValueError("expansion exact-label workload drifted")

    splits = _mapping(selection.get("splits"), "expansion splits")
    if set(splits) != set(ROLE_ORDER):
        raise ValueError("expansion role inventory drifted")
    projection: list[dict[str, Any]] = []
    seen_state_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    for role in ROLE_ORDER:
        split = _mapping(splits[role], f"split {role}")
        _exact_keys(split, {"trajectories", "states"}, f"split {role}")
        trajectories = _sequence(split["trajectories"], f"{role} trajectories")
        states = _sequence(split["states"], f"{role} states")
        if len(trajectories) != ROLE_TRAJECTORY_COUNTS[role]:
            raise ValueError(f"{role} trajectory count drifted")
        if len(states) != ROLE_STATE_COUNTS[role]:
            raise ValueError(f"{role} state count drifted")
        role_source_ids: list[str] = []
        for trajectory in trajectories:
            record = _mapping(trajectory, f"{role} trajectory")
            source_id = record.get("source_id")
            if (
                not isinstance(source_id, str)
                or not source_id
                or source_id in seen_source_ids
            ):
                raise ValueError("expansion source IDs must be non-empty and unique")
            seen_source_ids.add(source_id)
            role_source_ids.append(source_id)
        expected_state_sources = [
            source_id for source_id in role_source_ids for _ in range(3)
        ]
        observed_state_sources: list[str] = []
        for index, state_value in enumerate(states):
            state = _mapping(state_value, f"{role} state {index}")
            source_id = state.get("source_id")
            state_id = state.get("state_id")
            step = state.get("decision_step_id")
            expected_step = 4 + index % 3
            if source_id != expected_state_sources[index] or step != expected_step:
                raise ValueError("expansion state order or decision steps drifted")
            if state_id != f"{source_id}:decision_step:{expected_step:03d}":
                raise ValueError("expansion state ID drifted")
            if state_id in seen_state_ids:
                raise ValueError("expansion state IDs must be unique")
            seen_state_ids.add(str(state_id))
            observed_state_sources.append(str(source_id))
            if state.get("history_event_step_ids") != list(range(1, expected_step)):
                raise ValueError("expansion history geometry drifted")
            if state.get("candidate_event_step_ids") != list(
                range(1, expected_step - 1)
            ):
                raise ValueError("expansion candidate geometry drifted")
            if state.get("current_equivalent_event_step_id") != expected_step - 1:
                raise ValueError("expansion current-equivalence geometry drifted")
            projection.append(
                {
                    "state_index": len(projection),
                    "role": role,
                    "source_id": source_id,
                    "state_id": state_id,
                    "decision_step_id": expected_step,
                    "history_event_step_ids": state["history_event_step_ids"],
                    "candidate_event_step_ids": state["candidate_event_step_ids"],
                    "current_equivalent_event_step_id": expected_step - 1,
                }
            )
        if observed_state_sources != expected_state_sources:
            raise ValueError("expansion state/trajectory order drifted")
    if len(projection) != EXPECTED_DERIVED_COUNTS["state_count"]:
        raise ValueError("expansion state denominator drifted")
    return projection, sha256_bytes(canonical_json_bytes(projection))


def _validate_exposure(exposure: Mapping[str, Any]) -> None:
    if exposure.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("expansion exposure schema drifted")
    if exposure.get("protocol_id") != (
        "causalcache_restoration_v2_2_label_expansion_exposure_v1"
    ):
        raise ValueError("expansion exposure protocol drifted")
    if exposure.get("status") != "FROZEN_PRE_EXPANSION_POLICY_OUTPUT_EXPOSURE":
        raise ValueError("expansion exposure is not frozen")
    overlap = _mapping(exposure.get("overlap_proof"), "overlap proof")
    if overlap.get("all_required_intersections_are_empty") is not True:
        raise ValueError("expansion exposure has a reserved-cohort overlap")
    comparisons = _sequence(overlap.get("comparisons"), "overlap comparisons")
    if len(comparisons) != 6 or any(
        _mapping(record, "overlap comparison").get("intersection_count") != 0
        for record in comparisons
    ):
        raise ValueError("expansion overlap proof drifted")
    assertions = _mapping(
        exposure.get("pre_policy_output_assertions"), "pre-policy assertions"
    )
    for field in (
        "confirm_policy_output_accessed",
        "confirm_restoration_output_accessed",
        "expansion_gate_training_started_before_freeze",
        "expansion_policy_output_generated_before_freeze",
        "expansion_restoration_output_generated_before_freeze",
        "selection_changed_after_expansion_output",
    ):
        if assertions.get(field) is not False:
            raise ValueError(f"expansion exposure assertion must be false: {field}")
    cohort = _mapping(
        _mapping(exposure.get("cohort_inventory"), "cohort inventory").get(
            "label_expansion"
        ),
        "label expansion cohort",
    )
    if cohort.get("trajectory_count") != 64 or cohort.get("source_ids_sha256") != (
        "238a5608d08f6e4594195690b7595b1953e0865da2192e1e4b46d8b5795b3526"
    ):
        raise ValueError("expansion exposure cohort identity drifted")


def _file_witness(
    value: Any, name: str, *, expected_path: str | None = None
) -> Mapping[str, Any]:
    record = _mapping(value, name)
    _exact_keys(record, {"path", "sha256", "size_bytes"}, name)
    path = _safe_relative_path(record.get("path"), f"{name}.path")
    if expected_path is not None and path != expected_path:
        raise ValueError(f"{name}.path drifted")
    _require_sha256(record.get("sha256"), f"{name}.sha256")
    if type(record.get("size_bytes")) is not int or record["size_bytes"] <= 0:
        raise ValueError(f"{name}.size_bytes must be positive")
    return record


def _artifact_file_inventory(
    value: Any, name: str
) -> tuple[list[dict[str, Any]], int, str]:
    records = _sequence(value, name)
    if len(records) != len(DERIVED_ARTIFACT_PATHS):
        raise ValueError(f"{name} must contain the exact six artifact files")
    normalized: list[dict[str, Any]] = []
    for index, (record, expected_path) in enumerate(
        zip(records, DERIVED_ARTIFACT_PATHS, strict=True)
    ):
        witness = _file_witness(
            record,
            f"{name}[{index}]",
            expected_path=expected_path,
        )
        normalized.append(dict(witness))
    total_bytes = sum(record["size_bytes"] for record in normalized)
    return normalized, total_bytes, sha256_bytes(canonical_json_bytes(normalized))


def _external_file_witness(value: Any, name: str) -> None:
    record = _mapping(value, name)
    _exact_keys(record, {"path", "sha256", "size_bytes"}, name)
    if not isinstance(record.get("path"), str) or not record["path"]:
        raise ValueError(f"{name}.path must be non-empty")
    _require_sha256(record.get("sha256"), f"{name}.sha256")
    if type(record.get("size_bytes")) is not int or record["size_bytes"] <= 0:
        raise ValueError(f"{name}.size_bytes must be positive")


def _validate_argv(value: Any, name: str) -> None:
    arguments = _sequence(value, name)
    if not arguments or any(
        not isinstance(item, str) or not item for item in arguments
    ):
        raise ValueError(f"{name} must contain non-empty strings")
    joined = " ".join(arguments).casefold()
    for forbidden in ("hf_token", "hf_key", "authorization:", "bearer "):
        if forbidden in joined:
            raise ValueError(f"{name} contains a secret-bearing argument")


def _validate_execution_metadata(value: Any) -> None:
    execution = _mapping(value, "derived completion execution")
    _exact_keys(
        execution,
        {
            "host",
            "container",
            "utc",
            "commands",
            "logs",
            "exit_codes",
            "runtime_versions",
            "source_preflight",
            "exposure_ledger_sha256",
            "hf_repo_private",
            "old_tags_preserved",
        },
        "derived completion execution",
    )
    host = _mapping(execution["host"], "execution host")
    _exact_keys(host, {"alias", "hostname"}, "execution host")
    if any(not isinstance(host.get(key), str) or not host[key] for key in host):
        raise ValueError("execution host fields must be non-empty")
    container = _mapping(execution["container"], "execution container")
    _exact_keys(
        container,
        {
            "name",
            "id",
            "image",
            "image_digest",
            "device_requests",
            "gpu_used",
        },
        "execution container",
    )
    for field in ("name", "id", "image"):
        if not isinstance(container.get(field), str) or not container[field]:
            raise ValueError(f"execution container {field} must be non-empty")
    digest = container.get("image_digest")
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
    ):
        raise ValueError("execution container image digest is invalid")
    if container.get("device_requests") != [] or container.get("gpu_used") is not False:
        raise ValueError("policy-blind derived build must have no GPU device requests")
    utc = _mapping(execution["utc"], "execution UTC bracket")
    _exact_keys(utc, {"started_at_utc", "ended_at_utc"}, "execution UTC bracket")
    utc_pattern = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[^\s]+Z")
    if any(
        not isinstance(utc.get(field), str)
        or utc_pattern.fullmatch(utc[field]) is None
        for field in utc
    ):
        raise ValueError("execution UTC bracket is invalid")
    if utc["ended_at_utc"] < utc["started_at_utc"]:
        raise ValueError("execution UTC bracket is reversed")
    commands = _mapping(execution["commands"], "execution commands")
    command_names = {
        "build",
        "standalone_preupload_validation",
        "hf_upload",
        "hf_tag_create",
        "hf_download",
        "fresh_download_validation",
    }
    _exact_keys(commands, command_names, "execution commands")
    for name, argv in commands.items():
        _validate_argv(argv, f"execution command {name}")
    logs = _mapping(execution["logs"], "execution logs")
    log_names = {
        "build",
        "standalone_preupload_validation",
        "fresh_download_validation",
    }
    _exact_keys(logs, log_names, "execution logs")
    for name, witness in logs.items():
        _external_file_witness(witness, f"execution log {name}")
    exit_codes = _mapping(execution["exit_codes"], "execution exit codes")
    _exact_keys(exit_codes, command_names, "execution exit codes")
    if any(value != 0 for value in exit_codes.values()):
        raise ValueError("all completion execution commands must exit zero")
    runtime = _mapping(execution["runtime_versions"], "execution runtime versions")
    runtime_names = {
        "python",
        "rapidocr",
        "onnxruntime",
        "Pillow",
        "numpy",
        "pyarrow",
        "huggingface_hub",
    }
    _exact_keys(runtime, runtime_names, "execution runtime versions")
    if any(not isinstance(value, str) or not value for value in runtime.values()):
        raise ValueError("execution runtime versions must be non-empty strings")
    preflight = _mapping(execution["source_preflight"], "source preflight")
    if dict(preflight) != {
        "source_file_rehash_count": 16,
        "selected_row_reload_count": 64,
    }:
        raise ValueError("derived source preflight counts drifted")
    if execution.get("exposure_ledger_sha256") != EXPANSION_EXPOSURE_SHA256:
        raise ValueError("completion exposure-ledger SHA256 drifted")
    if execution.get("hf_repo_private") is not True:
        raise ValueError("derived artifact HF repo must remain private")
    old_tags = _sequence(execution["old_tags_preserved"], "preserved old tags")
    required_old_tag = {
        "tag": PRESERVED_OLD_TAG,
        "immutable_revision": PRESERVED_OLD_REVISION,
    }
    if required_old_tag not in old_tags:
        raise ValueError("the prior derived-artifact immutable tag was not preserved")
    for index, record_value in enumerate(old_tags):
        record = _mapping(record_value, f"preserved old tag {index}")
        _exact_keys(
            record,
            {"tag", "immutable_revision"},
            f"preserved old tag {index}",
        )
        if not isinstance(record.get("tag"), str) or not record["tag"]:
            raise ValueError("preserved old tag name must be non-empty")
        _require_git_sha(record.get("immutable_revision"), "preserved old revision")


def validate_completion_manifest(completion: Mapping[str, Any]) -> None:
    _exact_keys(
        completion,
        {
            "schema_version",
            "protocol_id",
            "status",
            "source",
            "artifact",
            "execution",
            "payload_manifest_snapshot",
            "negative_declarations",
        },
        "derived completion manifest",
    )
    if completion.get("schema_version") != "1.0.0":
        raise ValueError("derived completion schema drifted")
    if completion.get("protocol_id") != EXPECTED_COMPLETION_PROTOCOL_ID:
        raise ValueError("derived completion protocol drifted")
    if completion.get("status") != EXPECTED_COMPLETION_STATUS:
        raise ValueError("derived completion is not verified")

    source = _mapping(completion["source"], "derived completion source")
    _exact_keys(
        source,
        {"builder_git_commit", "clean_checkout", "head_equals_origin_main"},
        "derived completion source",
    )
    _require_git_sha(source.get("builder_git_commit"), "builder Git commit")
    if (
        source.get("clean_checkout") is not True
        or source.get("head_equals_origin_main") is not True
    ):
        raise ValueError("derived artifact was not built from clean canonical main")

    artifact = _mapping(completion["artifact"], "derived artifact")
    _exact_keys(
        artifact,
        {
            "repo",
            "tag",
            "immutable_revision",
            "payload_prefix",
            "preupload",
            "fresh_download",
        },
        "derived artifact",
    )
    if (
        artifact.get("repo") != DERIVED_REPO
        or artifact.get("payload_prefix") != DERIVED_PAYLOAD_PREFIX
    ):
        raise ValueError("derived HF repo or payload prefix drifted")
    if not isinstance(artifact.get("tag"), str) or not artifact["tag"]:
        raise ValueError("derived HF tag must be non-empty")
    _require_git_sha(artifact.get("immutable_revision"), "derived HF revision")
    preupload = _mapping(artifact["preupload"], "preupload validation")
    fresh = _mapping(artifact["fresh_download"], "fresh-download validation")
    validation_keys = {
        "artifact_tree_sha256",
        "artifact_file_count",
        "artifact_total_bytes",
        "files",
        "validator_outcome",
    }
    _exact_keys(preupload, validation_keys, "preupload validation")
    _exact_keys(
        fresh,
        validation_keys
        | {
            "clean_projection_created",
            "direct_repo_snapshot_validation_used",
            "payload_prefix_exact_six_only",
        },
        "fresh-download validation",
    )
    pre_files, pre_total, pre_tree = _artifact_file_inventory(
        preupload["files"], "preupload artifact files"
    )
    fresh_files, fresh_total, fresh_tree = _artifact_file_inventory(
        fresh["files"], "fresh-download artifact files"
    )
    if (
        fresh.get("clean_projection_created") is not True
        or fresh.get("direct_repo_snapshot_validation_used") is not False
        or fresh.get("payload_prefix_exact_six_only") is not True
    ):
        raise ValueError(
            "fresh download must validate a clean exact-six payload projection"
        )
    for phase_name, phase in (("preupload", preupload), ("fresh download", fresh)):
        _require_sha256(phase.get("artifact_tree_sha256"), f"{phase_name} tree")
        if phase.get("artifact_file_count") != 6:
            raise ValueError(f"{phase_name} artifact file count drifted")
        if (
            type(phase.get("artifact_total_bytes")) is not int
            or phase["artifact_total_bytes"] <= 0
        ):
            raise ValueError(f"{phase_name} artifact bytes must be positive")
        if phase.get("validator_outcome") != EXPECTED_VALIDATOR_OUTCOME:
            raise ValueError(f"{phase_name} standalone validator did not pass")
    if (
        preupload["artifact_total_bytes"] != pre_total
        or fresh["artifact_total_bytes"] != fresh_total
        or preupload["artifact_tree_sha256"] != pre_tree
        or fresh["artifact_tree_sha256"] != fresh_tree
    ):
        raise ValueError(
            "artifact aggregate identity does not reproduce its file inventory"
        )
    if (
        preupload["artifact_tree_sha256"] != fresh["artifact_tree_sha256"]
        or preupload["artifact_file_count"] != fresh["artifact_file_count"]
        or preupload["artifact_total_bytes"] != fresh["artifact_total_bytes"]
        or pre_files != fresh_files
    ):
        raise ValueError("preupload and fresh-download artifact identities differ")

    _validate_execution_metadata(completion["execution"])

    snapshot = _mapping(
        completion["payload_manifest_snapshot"], "payload manifest snapshot"
    )
    required_snapshot = {
        "schema_version": "1.0.0",
        "protocol_id": EXPECTED_DERIVED_PROTOCOL_ID,
        "status": EXPECTED_DERIVED_STATUS,
        "counts": EXPECTED_DERIVED_COUNTS,
        "formal_counts_enforced": True,
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "per_decision_view_current_expert_action_payload_included": False,
        "consumer_must_slice_events_by_history_event_step_ids": True,
    }
    for field, expected in required_snapshot.items():
        if snapshot.get(field) != expected:
            raise ValueError(f"derived payload manifest snapshot drifted: {field}")
    if snapshot.get("dataset_repo") != DERIVED_REPO:
        raise ValueError("derived payload dataset repo drifted")
    inputs = _mapping(snapshot.get("inputs"), "derived payload inputs")
    for key, expected_path in (
        ("label_expansion_config", EXPANSION_CONFIG_PATH),
        ("expansion_selection_manifest", EXPANSION_SELECTION_PATH),
    ):
        record = _mapping(inputs.get(key), f"derived payload input {key}")
        if record.get("path") != expected_path:
            raise ValueError(f"derived payload input path drifted: {key}")
        _require_sha256(record.get("sha256"), f"derived payload input {key}")
    generator = _mapping(snapshot.get("generator"), "derived payload generator")
    generator_revision = _require_git_sha(
        generator.get("git_revision"), "derived payload generator revision"
    )
    if generator_revision != source["builder_git_commit"]:
        raise ValueError("derived builder and payload generator revisions differ")
    snapshot_payload = pretty_json_bytes(snapshot)
    payload_index = DERIVED_ARTIFACT_PATHS.index(DERIVED_PAYLOAD_MANIFEST_PATH)
    pre_payload = pre_files[payload_index]
    if (
        sha256_bytes(snapshot_payload) != pre_payload["sha256"]
        or len(snapshot_payload) != pre_payload["size_bytes"]
    ):
        raise ValueError("payload manifest snapshot does not reproduce its witness")

    declarations = _mapping(
        completion["negative_declarations"], "negative declarations"
    )
    expected_declarations = {
        "confirm_state_or_output_accessed": False,
        "policy_module_loaded": False,
        "policy_forward_called": False,
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "gate_training_started": False,
        "matched_nll_started": False,
        "closed_loop_started": False,
    }
    if dict(declarations) != expected_declarations:
        raise ValueError("derived completion negative declarations drifted")


def _worker_topology() -> dict[str, Any]:
    return {
        "worker_count": 2,
        "gpu_model": "NVIDIA H200",
        "one_process_per_device": True,
        "workers": [
            {
                "worker_id": "even",
                "device": "cuda:0",
                "index_parity": 0,
                "state_indices": list(range(0, 192, 2)),
            },
            {
                "worker_id": "odd",
                "device": "cuda:1",
                "index_parity": 1,
                "state_indices": list(range(1, 192, 2)),
            },
        ],
        "cross_worker_state_stealing_allowed": False,
        "worker_failure_invalidates_entire_attempt": True,
        "worker_outputs_must_be_disjoint": True,
        "worker_union_must_equal_fixed_denominator": True,
    }


def build_contract(
    *,
    repository_root: str | Path,
    source_git_commit: str,
    completion_manifest: Mapping[str, Any],
    completion_manifest_sha256: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    _require_git_sha(source_git_commit, "source Git commit")
    _require_sha256(completion_manifest_sha256, "completion manifest SHA256")
    validate_completion_manifest(completion_manifest)
    expansion_payload, expansion = load_json_object(
        _repo_file(root, EXPANSION_SELECTION_PATH)
    )
    exposure_payload, exposure = load_json_object(
        _repo_file(root, EXPANSION_EXPOSURE_PATH)
    )
    config_payload, _ = load_json_object(_repo_file(root, EXPANSION_CONFIG_PATH))
    projection, projection_sha256 = _state_projection(expansion)
    _validate_exposure(exposure)
    if sha256_bytes(expansion_payload) != (
        "4aec4deffc6405c3d06ca3001d082e4fbd85ee44f55785708a0cee573edcd169"
    ):
        raise ValueError("expansion selection SHA256 drifted")
    if sha256_bytes(exposure_payload) != (
        "e5f7e7b5ffeb71ac117b5cb4a470eb31648978a734edb9a521125d3bda29a323"
    ):
        raise ValueError("expansion exposure SHA256 drifted")
    if sha256_bytes(config_payload) != (
        "e74904fb75cd72429e6adb51ab7500adcaa1915fb6cebc8567a2945bc9373947"
    ):
        raise ValueError("expansion split config SHA256 drifted")

    completion = _mapping(completion_manifest, "completion manifest")
    artifact = _mapping(completion["artifact"], "completion artifact")
    preupload = _mapping(artifact["preupload"], "preupload validation")
    payload_index = DERIVED_ARTIFACT_PATHS.index(DERIVED_PAYLOAD_MANIFEST_PATH)
    snapshot = _mapping(completion["payload_manifest_snapshot"], "payload snapshot")
    if snapshot["inputs"]["label_expansion_config"]["sha256"] != sha256_bytes(
        config_payload
    ):
        raise ValueError("derived artifact expansion-config provenance drifted")
    if snapshot["inputs"]["expansion_selection_manifest"][
        "sha256"
    ] != sha256_bytes(expansion_payload):
        raise ValueError("derived artifact expansion-selection provenance drifted")

    source_files = [_identity(root, path) for path in SOURCE_LOCK_PATHS]
    prohibited = {
        "maximum_confirm_state_access_count": 0,
        "maximum_confirm_prompt_or_image_access_count": 0,
        "maximum_current_or_future_expert_action_target_read_count": 0,
        "maximum_current_or_future_expert_action_semantic_consumption_count": 0,
        "maximum_expert_action_backfill_count": 0,
        "maximum_prior_policy_output_import_count": 0,
        "maximum_restoration_coalition_construction_count": 0,
        "maximum_restoration_candidate_prompt_count": 0,
        "maximum_restoration_label_count": 0,
        "maximum_exact_subset_search_count": 0,
        "maximum_gate_training_example_count": 0,
        "maximum_gate_model_forward_count": 0,
        "maximum_matched_nll_evaluation_count": 0,
        "maximum_closed_loop_episode_count": 0,
        "androidworld_test_split_access_allowed": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "preregistration_status": (
            "source_only_frozen_before_any_expansion_policy_or_restoration_output"
        ),
        "freeze": {
            "source_git_commit": source_git_commit,
            "clean_pushed_canonical_main_required_at_materialization": True,
            "policy_output_accessed_to_choose_contract": False,
            "restoration_output_accessed_to_choose_contract": False,
            "confirm_state_or_output_accessed": False,
            "source_only_validator_authorizes_policy_or_gpu_execution": False,
        },
        "immutable_inputs": {
            "label_expansion_config": _identity(root, EXPANSION_CONFIG_PATH),
            "structural_selection": _identity(root, EXPANSION_SELECTION_PATH),
            "exposure_ledger": _identity(root, EXPANSION_EXPOSURE_PATH),
            "derived_completion": {
                "path": CANONICAL_COMPLETION_PATH,
                "sha256": completion_manifest_sha256,
            },
            "derived_artifact": {
                "repo": artifact["repo"],
                "tag": artifact["tag"],
                "immutable_revision": artifact["immutable_revision"],
                "payload_prefix": artifact["payload_prefix"],
                "artifact_tree_sha256": preupload["artifact_tree_sha256"],
                "artifact_file_count": preupload["artifact_file_count"],
                "artifact_total_bytes": preupload["artifact_total_bytes"],
                "files": [dict(record) for record in preupload["files"]],
                "payload_manifest": dict(preupload["files"][payload_index]),
                "preupload_validator_outcome": preupload["validator_outcome"],
                "fresh_download_validator_outcome": artifact["fresh_download"][
                    "validator_outcome"
                ],
                "preupload_and_fresh_download_tree_equal": True,
                "completion_execution_sha256": sha256_bytes(
                    canonical_json_bytes(completion["execution"])
                ),
            },
        },
        "data_projection": {
            "role_order": list(ROLE_ORDER),
            "role_trajectory_counts": ROLE_TRAJECTORY_COUNTS,
            "role_state_counts": ROLE_STATE_COUNTS,
            "fixed_trajectory_denominator": 64,
            "fixed_state_denominator": 192,
            "decision_steps_per_trajectory": [4, 5, 6],
            "state_projection_sha256": projection_sha256,
            "state_projection_record_count": len(projection),
            "fresh_state_output_required": True,
            "state_filtering_top_up_or_replacement_allowed": False,
            "consumer_must_slice_events_by_history_event_step_ids": True,
            "per_decision_view_current_expert_action_payload_included": False,
            "shared_trajectory_storage_contains_later_event_actions": True,
            "decision_view_loader": {
                "module_path": (
                    "code/causalcache/restoration_v2_2_expansion_substrate_inputs.py"
                ),
                "callable": "build_decision_view_input",
                "strict_slice_primitive": "decision_view_events",
                "required_call_count": 192,
                "direct_full_trajectory_events_consumption_allowed": False,
                "required_history_step_ids": "exact_range_1_to_decision_step_exclusive",
                "current_or_future_event_action_exposure_count": 0,
                "runner_must_record_per_state_slice_witness": True,
                "serialized_request_event_steps_must_equal_history_step_ids": True,
                "state_id_must_bind_source_id_and_decision_step": True,
                "request_manifest_validator_callable": (
                    "validate_request_manifest"
                ),
                "request_manifest_validation_requires_actual_included_events": (
                    True
                ),
                "request_manifest_recomputes_included_event_step_ids": True,
                "request_manifest_recomputes_included_event_payload_sha256": True,
                "runner_must_validate_manifest_against_actual_slice_before_request": (
                    True
                ),
                (
                    "excluded_current_or_future_action_canary_must_leave_"
                    "prompt_bytes_unchanged"
                ): True,
                "included_history_action_canary_must_change_prompt_bytes": True,
                "canary_must_pass_before_first_generation": True,
                "excluded_action_canary_state_count": 128,
                "excluded_action_canary_mutation_count": 192,
                "each_excluded_event_action_must_be_mutated_separately": True,
                "included_history_action_canary_state_count": 192,
                "included_history_action_canary_mutation_count": 192,
                "canary_mechanical_action_mutation_is_not_semantic_consumption": (
                    True
                ),
                "canary_mutated_current_or_future_actions_must_not_enter_request": (
                    True
                ),
                "raw_request_manifest_records_actual_included_event_step_ids": True,
                "canonical_action_source": (
                    "frozen_full_history_policy_native_generation_only"
                ),
                "expert_or_adjacent_event_action_backfill_allowed": False,
            },
            "exact_label_workload_frozen_but_not_authorized": EXPECTED_LABEL_WORKLOAD,
        },
        "scientific_source_lock": {
            "source_files": source_files,
            "reserved_execution_source_paths": list(RESERVED_EXECUTION_SOURCE_PATHS),
            "prompt_policy_parser_teacher_kl_semantics_inherited_from_v2_2_eager": True,
            (
                "formal_execution_requires_a_separate_committed_and_pushed_"
                "runner_freeze"
            ): True,
            "scientific_environment_variables_allowed": False,
        },
        "runtime": {
            "dtype": "bfloat16",
            "attention_implementation_requested": "eager",
            "attention_implementation_observed_must_equal": "eager",
            "eager_control_flags": {
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest",
                "seed": 0,
            },
            "strict_cuda_determinism_claimed": False,
            "expected_execution_stack": EXPECTED_STACK,
            "scientific_runtime_metadata_must_be_equal_across_workers": True,
            "container_image_digest_must_be_equal_across_workers": True,
            "worker_identity_fields_required": [
                "device",
                "gpu_name",
                "gpu_uuid",
                "gpu_pci_bus_id",
                "logical_device_index",
                "nvidia_smi_index",
            ],
            "worker_identity_fields_allowed_to_differ": [
                "device",
                "gpu_uuid",
                "gpu_pci_bus_id",
                "logical_device_index",
                "nvidia_smi_index",
            ],
        },
        "substrate_schedule": {
            "per_state_maximum": {
                "full_history_generation_calls": 2,
                "teacher_forward_calls_on_success": 3,
                "kl_measurements_on_success": 2,
            },
            "planned_and_maximum_counts": EXPECTED_SUBSTRATE_COUNTS,
            "mixed_fidelity_generation_call_count": 384,
            "mixed_fidelity_generation_semantics": (
                "frozen_mixed_fidelity_input_builder_with_full_history_"
                "reference_endpoint"
            ),
            "generation_order_per_state": [
                "full_history_reference_repeat_1",
                "full_history_reference_repeat_2",
            ],
            "teacher_order_on_success_per_state": [
                "full_history_reference_1",
                "full_history_reference_2",
                "summary_only",
            ],
            "kl_order_on_success_per_state": [
                "repeat_reference_kl",
                "summary_reference_kl",
            ],
            "automatic_retry_allowed": False,
            "state_retry_count": 0,
            "top_up_count": 0,
            "resume_allowed": False,
            "completed_state_regeneration_allowed": False,
            "sample_mutation_allowed": False,
        },
        "substrate_gate": {
            "fixed_state_denominator": 192,
            "minimum_parse_coverage": 0.99,
            "derived_required_parse_success_count": 191,
            "minimum_finite_logit_coverage": 1.0,
            "derived_required_finite_logit_state_count": 192,
            "minimum_repeat_canonical_action_agreement": 1.0,
            "derived_required_repeat_agreement_count": 192,
            "memory_sensitive_definition": (
                "summary_reference_kl_greater_than_repeat_noise_epsilon"
            ),
            "repeat_noise_epsilon": "max(1e-4,10*mean_repeat_kl)",
            "minimum_memory_sensitive_states": 8,
            "memory_sensitive_fraction_is_reported_not_thresholded": True,
            "failed_states_remain_in_fixed_denominator": True,
            "pass_outcome": "PASS_V2_2_LABEL_EXPANSION_SUBSTRATE_V1",
            "fail_outcome": "NO_GO_V2_2_LABEL_EXPANSION_SUBSTRATE_V1",
            "invalid_outcome": "INVALID_V2_2_LABEL_EXPANSION_SUBSTRATE_V1",
        },
        "prohibited_work": prohibited,
        "execution": {
            "attempt_id": "restoration-v2-2-label-expansion-substrate-v1",
            "canonical_persistent_output_dir": (
                "/data/experiments/causalcache/"
                "restoration-v2-2-label-expansion-substrate-v1"
            ),
            "canonical_global_attempt_ledger": (
                "/data/experiments/causalcache/"
                ".restoration-v2-2-label-expansion-substrate-v1.attempt.json"
            ),
            "required_host_class": "Hyper_H200",
            "worker_topology": _worker_topology(),
            "same_host_and_container_required_for_both_workers": True,
            "same_state_repeats_must_remain_on_one_worker_device": True,
            "cross_host_attempt_allowed": False,
            "alternate_output_or_ledger_allowed": False,
            "state_attempt_marker_written_before_first_generation": True,
            "runtime_import_requires_fresh_immutable_derived_validation": True,
            "execution_locked_until_runner_source_is_committed_and_pushed": True,
        },
        "artifact_destination": {
            "repo": (
                "gavinlaw/causalcache-restoration-v2-2-label-expansion-substrate-mobile"
            ),
            "repo_type": "dataset",
            "visibility": "private",
            "tag": "v2.2-label-expansion-substrate-v1",
            "path": "raw/v2.2-label-expansion-substrate-v1.tar",
            "git_result_dir": (
                "data/results/restoration_v2_2_label_expansion_substrate_v1"
            ),
            "fresh_immutable_download_and_byte_hash_required_before_git_manifest": True,
        },
        "promotion": {
            "pass_authorizes_only": (
                "freeze_exact_subset_and_conditional_marginal_label_execution_source"
            ),
            "pass_does_not_authorize_automatic_restoration_label_execution": True,
            "gate_training_matched_nll_closed_loop_and_confirm_remain_locked": True,
            "no_go_stops_expansion_restoration_labels": True,
            "source_only_validator_authorizes_policy_or_gpu_execution": False,
        },
    }


def validate_contract_data(
    data: Mapping[str, Any],
    *,
    repository_root: str | Path,
    completion_manifest: Mapping[str, Any],
    completion_manifest_sha256: str,
    require_git_blobs: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    freeze = _mapping(data.get("freeze"), "contract freeze")
    source_git_commit = _require_git_sha(
        freeze.get("source_git_commit"), "source Git commit"
    )
    expected = build_contract(
        repository_root=root,
        source_git_commit=source_git_commit,
        completion_manifest=completion_manifest,
        completion_manifest_sha256=completion_manifest_sha256,
    )
    if dict(data) != expected:
        raise ValueError(
            "expansion substrate contract differs from deterministic freeze"
        )
    if require_git_blobs:
        paths = [
            EXPANSION_CONFIG_PATH,
            EXPANSION_SELECTION_PATH,
            EXPANSION_EXPOSURE_PATH,
            CANONICAL_COMPLETION_PATH,
            *SOURCE_LOCK_PATHS,
        ]
        for relative in paths:
            path = _repo_file(root, relative)
            if _git_blob(root, source_git_commit, relative) != path.read_bytes():
                raise ValueError(f"{relative} differs from source Git revision")
    return {
        "contract_valid": True,
        "protocol_id": PROTOCOL_ID,
        "trajectory_count": 64,
        "state_count": 192,
        "generation_call_count": 384,
        "teacher_forward_count": 576,
        "kl_measurement_count": 384,
        "worker_state_counts": {"even": 96, "odd": 96},
        "restoration_label_count": 0,
        "policy_or_gpu_execution_authorized_by_this_validator": False,
        "confirm_access_authorized_by_this_validator": False,
    }


def load_and_validate_contract(
    config_path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    expected_config = (root / CANONICAL_CONFIG_PATH).resolve()
    supplied = Path(config_path)
    actual_config = (
        supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    )
    if (
        actual_config != expected_config
        or actual_config.is_symlink()
        or not actual_config.is_file()
    ):
        raise ValueError(f"expansion substrate config must be {CANONICAL_CONFIG_PATH}")
    config_payload, data = load_json_object(actual_config)
    if config_payload != pretty_json_bytes(data):
        raise ValueError("expansion substrate config must be canonical pretty JSON")
    completion_path = _repo_file(root, CANONICAL_COMPLETION_PATH)
    completion_payload, completion = load_json_object(completion_path)
    validation = validate_contract_data(
        data,
        repository_root=root,
        completion_manifest=completion,
        completion_manifest_sha256=sha256_bytes(completion_payload),
        require_git_blobs=True,
    )
    validation["config_sha256"] = sha256_bytes(config_payload)
    validation["completion_manifest_sha256"] = sha256_bytes(completion_payload)
    return data, validation


__all__ = [
    "CANONICAL_COMPLETION_PATH",
    "CANONICAL_CONFIG_PATH",
    "EXPECTED_COMPLETION_PROTOCOL_ID",
    "EXPECTED_COMPLETION_STATUS",
    "EXPECTED_DERIVED_COUNTS",
    "EXPECTED_SUBSTRATE_COUNTS",
    "PROTOCOL_ID",
    "build_contract",
    "canonical_json_bytes",
    "load_and_validate_contract",
    "load_json_object",
    "pretty_json_bytes",
    "sha256_bytes",
    "validate_completion_manifest",
    "validate_contract_data",
]
