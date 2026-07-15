"""Validate the immutable restoration-v2 derived-artifact completion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_restoration_v2"
ARTIFACT_ID = "causalcache-restoration-v2-derived-artifact-v1"
PAYLOAD_ARTIFACT_ID = "causalcache-guiodyssey-restoration-v2-mobile-v1"
STATUS = (
    "hf_immutable_verified_two_materializations_and_three_exact_replays_passed"
)
SUMMARY_OUTCOME = "PASSED_RESTORATION_V2_DERIVED_ARTIFACT_COMPLETION"
VALIDATOR_OUTCOME = "PASSED_GUIODYSSEY_RESTORATION_V2_ARTIFACT_VALIDATION"
DATASET_REPO = "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile"
HF_TAG = "restoration-v2-derived-v1.0.0"
HF_CLI_PATH = "/Users/luojiaxuan/.local/bin/hf"
HF_UPLOAD_COMMIT_MESSAGE = "Upload CausalCache restoration v2 derived artifact v1"
HF_TAG_MESSAGE = "CausalCache restoration-v2 derived dataset v1.0.0"
PAYLOAD_PREFIX = "derived/restoration-v2-v1"
COMPLETION_MANIFEST_PATH = "data/manifests/restoration_v2_derived_artifact.json"
FROZEN_BUILDER_COMMIT = "1a01f2323647d092cab67f0531ecb877a4a255de"
FROZEN_BUILDER_TREE = "753e42ab9663553d461e8200dcfcdd5576ee209e"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")

EXACT_FILE_ALLOWLIST = (
    ".gitattributes",
    "README.md",
    f"{PAYLOAD_PREFIX}/images-00000-of-00001.tar",
    f"{PAYLOAD_PREFIX}/manifest.json",
    f"{PAYLOAD_PREFIX}/ocr-records-00000-of-00001.jsonl",
    f"{PAYLOAD_PREFIX}/trajectories-00000-of-00001.jsonl",
)
PAYLOAD_SHARD_PATHS = (
    EXACT_FILE_ALLOWLIST[2],
    EXACT_FILE_ALLOWLIST[4],
    EXACT_FILE_ALLOWLIST[5],
)
EXPECTED_COUNTS = {
    "trajectory_count": 35,
    "event_count": 175,
    "state_count": 65,
    "image_member_count": 210,
    "ocr_record_count": 210,
}
EXPECTED_ROLE_COUNTS = {
    "v2_label_train": 10,
    "v2_development": 5,
    "v2_confirm_primary": 20,
}
EXPECTED_VALIDATION_IDS = (
    "repeat-1-builder-integrated-replay",
    "repeat-2-builder-integrated-replay",
    "hf-immutable-redownload-exact-replay",
)
EXPECTED_CHECKOUT_TRANSPORT = {
    "canonical_main_push_verified_on_mac": True,
    "hyper00_https_fetch_outcome": "FAILED_NONINTERACTIVE_GITHUB_CREDENTIALS",
    "fallback": "complete_git_bundle_from_verified_pushed_main",
    "bundle_size_bytes": 1325083,
    "bundle_sha256": (
        "a97563e9ddbbc2ab2242af6070480aae518938fd2100d4c09b808b7b6c5e1857"
    ),
}
EXPECTED_COMPLETION_LIBRARY_VERSIONS = {
    "pyarrow": "24.0.0",
    "Pillow": "12.2.0",
    "rapidocr": "3.8.4",
    "onnxruntime": "1.24.4",
    "huggingface_hub": "1.23.0",
}
COMPLETION_SOURCE_PATHS = (
    "code/scripts/validate_restoration_v2_derived_artifact.py",
    "code/tests/test_restoration_v2_derived_artifact_manifest.py",
    "data/results/restoration_v2_derived_artifact/summary.json",
)
HISTORICAL_INPUT_PATHS = {
    "scientific_contract": "code/configs/causalcache_restoration_v2.json",
    "v1_config": "code/configs/independent_reference_gate_v1.json",
    "source_file_manifest": (
        "data/manifests/independent_reference_gate_v1_source_files.json"
    ),
    "selection_manifest": "data/manifests/restoration_v2_selection.json",
    "exposure_manifest": "data/manifests/restoration_v2_exposure.json",
    "ocr_backend_config": "code/configs/restoration_v2_ocr_backend.json",
    "ocr_backend_manifest": "data/manifests/restoration_v2_ocr_backend.json",
}
HISTORICAL_GENERATOR_PATHS = {
    "module": "code/causalcache/data/guiodyssey_restoration_v2.py",
    "build_cli": "code/scripts/build_guiodyssey_restoration_v2.py",
    "validator": "code/scripts/validate_guiodyssey_restoration_v2.py",
}
HISTORICAL_TEST_PATH = "code/tests/test_guiodyssey_restoration_v2.py"
FROZEN_BUILDER_SOURCE_SHA256 = {
    "code/causalcache/data/guiodyssey_restoration_v2.py": (
        "8caf9418e4212c760b789f660b8b67c9b23a5b88d08a8434d4a1e718f7a0e4ee"
    ),
    "code/configs/causalcache_restoration_v2.json": (
        "9b9b78d9e1902d6ba7c648c939809c56fe55cccc17de58d4e6eed8d9ddf746cc"
    ),
    "code/configs/independent_reference_gate_v1.json": (
        "7b62aa31c80536f28bc4e8a3d684ff535bc2315d44a31cb5f002ed6dcb493fd8"
    ),
    "code/configs/restoration_v2_ocr_backend.json": (
        "51e08a9565f804ecb1ee7f12887cc12742b84c04996fffbbde7c0f304e4bd036"
    ),
    "code/scripts/build_guiodyssey_restoration_v2.py": (
        "142be847ed168690e05e1472e0b12b51c57bcf2f07c3f94d36a12194961b093a"
    ),
    "code/scripts/validate_guiodyssey_restoration_v2.py": (
        "63e590d3f134168cf426a942444a441f777b7fc547ce89665aec9a4f1a384476"
    ),
    "code/tests/test_guiodyssey_restoration_v2.py": (
        "726de392070c66a44348259b2b3e40ca081df676b37ffc23528c7a0bc31fcea5"
    ),
    "data/manifests/independent_reference_gate_v1_source_files.json": (
        "46f2240a07f46b6e283cb90e1f478f0b14a2e1499c52e2cc8e144898065adffc"
    ),
    "data/manifests/restoration_v2_exposure.json": (
        "bc1224826e0642c2374f6cfbebd676389d8c17ce6bf8a789f08903740cf97a95"
    ),
    "data/manifests/restoration_v2_ocr_backend.json": (
        "107478672438b52e9c2ccc9ef8de5d13d4e16329ab1afbe95772e2d3df5bd2a9"
    ),
    "data/manifests/restoration_v2_selection.json": (
        "292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b"
    ),
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_loads(payload: str | bytes) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(payload, object_pairs_hook=pairs_hook)


def load_json_object(path: str | Path) -> tuple[bytes, dict[str, Any]]:
    payload = Path(path).read_bytes()
    value = _strict_json_loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload, value


def _require_exact_keys(value: Any, keys: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{field} fields drifted")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _require_git_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or GIT_SHA_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a full lowercase Git SHA")
    return value


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _safe_relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a safe relative POSIX path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or str(parsed) != value
        or any(part in {".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"{field} must be a safe relative POSIX path")
    return value


def _parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{field} must be an RFC3339 UTC timestamp") from error
    return parsed


def _validate_utc_bracket(record: Mapping[str, Any], field: str) -> None:
    started = _parse_utc(record.get("started_at_utc"), f"{field}.started_at_utc")
    ended = _parse_utc(record.get("ended_at_utc"), f"{field}.ended_at_utc")
    if started >= ended:
        raise ValueError(f"{field} UTC bracket is not increasing")


def _validate_argv(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{field} must be a non-empty string array")
    if any(item == "--token" or item.startswith("--token=") for item in value):
        raise ValueError(f"{field} cannot persist an authentication token")
    return list(value)


def _flag_values(argv: Sequence[str], flag: str, field: str) -> list[str]:
    if any(item.startswith(f"{flag}=") for item in argv):
        raise ValueError(f"{field} must record {flag} as a separate argv token")
    values: list[str] = []
    for index, item in enumerate(argv):
        if item == flag:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"{field} has a missing value for {flag}")
            values.append(argv[index + 1])
    return values


def _require_single_flag_value(
    argv: Sequence[str],
    flag: str,
    expected: str,
    field: str,
) -> None:
    if _flag_values(argv, flag, field) != [expected]:
        raise ValueError(f"{field} {flag} value drifted")


def _validate_flag_tail(
    argv: Sequence[str],
    *,
    start: int,
    value_flags: set[str],
    boolean_flags: set[str],
    repeatable_value_flags: set[str],
    field: str,
) -> None:
    counts: dict[str, int] = {}
    index = start
    while index < len(argv):
        flag = argv[index]
        if flag in value_flags:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"{field} has a missing value for {flag}")
            counts[flag] = counts.get(flag, 0) + 1
            index += 2
            continue
        if flag in boolean_flags:
            counts[flag] = counts.get(flag, 0) + 1
            index += 1
            continue
        raise ValueError(f"{field} contains an unexpected argument: {flag}")
    if any(
        count != 1
        for flag, count in counts.items()
        if flag not in repeatable_value_flags
    ):
        raise ValueError(f"{field} contains a duplicate non-repeatable flag")


def _git_output(repository_root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repository_root,
        stderr=subprocess.STDOUT,
    )


def _historical_blob(repository_root: Path, commit: str, path: str) -> bytes:
    _safe_relative_path(path, "historical source path")
    return _git_output(repository_root, "show", f"{commit}:{path}")


def _validate_file_records(records: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(records, list) or len(records) != len(EXACT_FILE_ALLOWLIST):
        raise ValueError(f"{field} must contain exactly six file records")
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        value = _require_exact_keys(
            record,
            {"path", "size_bytes", "sha256"},
            f"{field}[{index}]",
        )
        path = _safe_relative_path(value.get("path"), f"{field}[{index}].path")
        size = value.get("size_bytes")
        if type(size) is not int or size <= 0:
            raise ValueError(f"{field}[{index}].size_bytes must be positive")
        digest = _require_sha256(value.get("sha256"), f"{field}[{index}].sha256")
        normalized.append({"path": path, "size_bytes": size, "sha256": digest})
    if tuple(record["path"] for record in normalized) != EXACT_FILE_ALLOWLIST:
        raise ValueError(f"{field} exact-six allowlist or order drifted")
    return normalized


def _validate_counts(value: Any, field: str) -> dict[str, int]:
    record = _require_exact_keys(value, set(EXPECTED_COUNTS), field)
    if dict(record) != EXPECTED_COUNTS:
        raise ValueError(f"{field} formal counts drifted")
    return dict(record)


def _validate_source_records(
    records: Any,
    *,
    expected_paths: Sequence[str],
    repository_root: Path | None,
    historical_blobs: Mapping[str, bytes] | None,
    field: str,
) -> dict[str, str]:
    if not isinstance(records, list) or len(records) != len(expected_paths):
        raise ValueError(f"{field} source inventory count drifted")
    observed: dict[str, str] = {}
    for index, record in enumerate(records):
        value = _require_exact_keys(record, {"path", "sha256"}, f"{field}[{index}]")
        path = _safe_relative_path(value.get("path"), f"{field}[{index}].path")
        digest = _require_sha256(value.get("sha256"), f"{field}[{index}].sha256")
        if path in observed:
            raise ValueError(f"{field} contains a duplicate path")
        if historical_blobs is not None:
            payload = historical_blobs.get(path)
            if payload is None or sha256_bytes(payload) != digest:
                raise ValueError(f"{field} historical source SHA256 drifted: {path}")
        elif repository_root is not None:
            source_path = repository_root.joinpath(*PurePosixPath(path).parts)
            if not source_path.is_file() or sha256_file(source_path) != digest:
                raise ValueError(f"{field} current source SHA256 drifted: {path}")
        observed[path] = digest
    if tuple(observed) != tuple(expected_paths):
        raise ValueError(f"{field} exact source inventory or order drifted")
    return observed


def _validate_builder_history(
    builder: Any,
    *,
    repository_root: Path,
    expected_builder_commit: str,
    expected_builder_tree: str,
    expected_builder_source_sha256: Mapping[str, str],
) -> tuple[dict[str, bytes], dict[str, str]]:
    record = _require_exact_keys(
        builder,
        {"git_commit", "git_tree", "source_files"},
        "builder_snapshot",
    )
    commit = _require_git_sha(record.get("git_commit"), "builder_snapshot.git_commit")
    tree = _require_git_sha(record.get("git_tree"), "builder_snapshot.git_tree")
    if commit != expected_builder_commit or tree != expected_builder_tree:
        raise ValueError("builder historical commit or tree drifted")
    object_type = _git_output(repository_root, "cat-file", "-t", commit).decode().strip()
    if object_type != "commit":
        raise ValueError("builder historical object is not a commit")
    observed_tree = _git_output(repository_root, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    if observed_tree != tree:
        raise ValueError("builder historical Git tree identity drifted")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("builder commit is not an ancestor of origin/main")
    expected_paths = tuple(sorted(expected_builder_source_sha256))
    blobs = {
        path: _historical_blob(repository_root, commit, path) for path in expected_paths
    }
    observed = _validate_source_records(
        record.get("source_files"),
        expected_paths=expected_paths,
        repository_root=None,
        historical_blobs=blobs,
        field="builder_snapshot.source_files",
    )
    if observed != dict(expected_builder_source_sha256):
        raise ValueError("builder frozen source identities drifted")
    return blobs, observed


def _validate_payload_manifest(
    content: Any,
    *,
    builder_commit: str,
    historical_blobs: Mapping[str, bytes],
    historical_source_sha256: Mapping[str, str],
    hf_files: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    manifest = _require_exact_keys(
        content,
        {
            "schema_version",
            "protocol_id",
            "artifact_id",
            "status",
            "dataset_repo",
            "license",
            "policy_output_generated",
            "restoration_output_generated",
            "formal_counts_enforced",
            "inputs",
            "generator",
            "ocr_backend",
            "ocr_runtime",
            "source_dataset",
            "role_source_ids",
            "counts",
            "inventories",
            "payload_files",
        },
        "payload_manifest_snapshot.content",
    )
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "artifact_id": PAYLOAD_ARTIFACT_ID,
        "status": "POLICY_BLIND_DERIVED_DATASET_MATERIALIZED",
        "dataset_repo": DATASET_REPO,
        "license": "cc-by-4.0",
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "formal_counts_enforced": True,
    }
    if any(manifest.get(key) != value for key, value in expected_scalars.items()):
        raise ValueError("payload manifest identity or pre-output state drifted")
    _validate_counts(manifest.get("counts"), "payload_manifest.counts")

    inputs = _require_exact_keys(
        manifest.get("inputs"), set(HISTORICAL_INPUT_PATHS), "payload_manifest.inputs"
    )
    for name, path in HISTORICAL_INPUT_PATHS.items():
        identity = _require_exact_keys(
            inputs.get(name), {"path", "sha256"}, f"payload_manifest.inputs.{name}"
        )
        if identity.get("path") != path or identity.get("sha256") != (
            historical_source_sha256[path]
        ):
            raise ValueError(f"payload manifest historical input drifted: {name}")

    generator = _require_exact_keys(
        manifest.get("generator"),
        {
            "git_revision",
            "module_path",
            "module_sha256",
            "build_cli_path",
            "build_cli_sha256",
            "validator_path",
            "validator_sha256",
        },
        "payload_manifest.generator",
    )
    if generator.get("git_revision") != builder_commit:
        raise ValueError("payload manifest generator commit drifted")
    for name, path in HISTORICAL_GENERATOR_PATHS.items():
        if generator.get(f"{name}_path") != path or generator.get(
            f"{name}_sha256"
        ) != historical_source_sha256[path]:
            raise ValueError(f"payload manifest generator source drifted: {name}")

    ocr_backend = _require_exact_keys(
        manifest.get("ocr_backend"),
        {
            "backend_id",
            "backend_config_sha256",
            "hf_model_repo",
            "hf_model_revision",
            "hf_model_tag",
            "real_screen_golden_dataset_revision",
        },
        "payload_manifest.ocr_backend",
    )
    backend_config = _strict_json_loads(historical_blobs[HISTORICAL_INPUT_PATHS["ocr_backend_config"]])
    backend_completion = _strict_json_loads(
        historical_blobs[HISTORICAL_INPUT_PATHS["ocr_backend_manifest"]]
    )
    expected_ocr = {
        "backend_id": backend_config["backend_id"],
        "backend_config_sha256": historical_source_sha256[
            HISTORICAL_INPUT_PATHS["ocr_backend_config"]
        ],
        "hf_model_repo": backend_completion["hf_model_artifact"]["repo"],
        "hf_model_revision": backend_completion["hf_model_artifact"][
            "immutable_revision"
        ],
        "hf_model_tag": backend_completion["hf_model_artifact"]["tag"],
        "real_screen_golden_dataset_revision": backend_completion[
            "real_screen_golden"
        ]["immutable_revision"],
    }
    if dict(ocr_backend) != expected_ocr:
        raise ValueError("payload manifest OCR backend provenance drifted")

    ocr_runtime = _require_exact_keys(
        manifest.get("ocr_runtime"),
        {
            "runtime_packages",
            "rapidocr_package_file_sha256",
            "wheel_sha256",
            "model_sha256",
            "recognizer_character_inventory",
        },
        "payload_manifest.ocr_runtime",
    )
    for name in (
        "rapidocr_package_file_sha256",
        "wheel_sha256",
        "model_sha256",
    ):
        hashes = ocr_runtime.get(name)
        if not isinstance(hashes, Mapping) or not hashes:
            raise ValueError(f"payload_manifest.ocr_runtime.{name} must be non-empty")
        for key, digest in hashes.items():
            _require_nonempty_string(key, f"ocr_runtime.{name} key")
            _require_sha256(digest, f"ocr_runtime.{name}.{key}")
    packages = ocr_runtime.get("runtime_packages")
    if not isinstance(packages, Mapping) or not packages or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in packages.items()
    ):
        raise ValueError("payload manifest OCR runtime package identity drifted")
    inventory = _require_exact_keys(
        ocr_runtime.get("recognizer_character_inventory"),
        {"metadata_key", "entry_count", "utf8_sha256", "canonical_json_sha256"},
        "payload_manifest.ocr_runtime.recognizer_character_inventory",
    )
    _require_nonempty_string(inventory.get("metadata_key"), "recognizer metadata_key")
    if type(inventory.get("entry_count")) is not int or inventory["entry_count"] <= 0:
        raise ValueError("recognizer character entry_count must be positive")
    _require_sha256(inventory.get("utf8_sha256"), "recognizer utf8_sha256")
    _require_sha256(
        inventory.get("canonical_json_sha256"),
        "recognizer canonical_json_sha256",
    )

    v1_config = _strict_json_loads(historical_blobs[HISTORICAL_INPUT_PATHS["v1_config"]])
    source_pool = v1_config["source_pool"]
    expected_source_dataset = {
        key: source_pool[key]
        for key in (
            "upstream_repo",
            "upstream_revision",
            "transport_repo",
            "transport_revision",
            "license",
        )
    }
    source_dataset = _require_exact_keys(
        manifest.get("source_dataset"),
        set(expected_source_dataset),
        "payload_manifest.source_dataset",
    )
    if dict(source_dataset) != expected_source_dataset:
        raise ValueError("payload manifest source dataset identity drifted")

    selection = _strict_json_loads(
        historical_blobs[HISTORICAL_INPUT_PATHS["selection_manifest"]]
    )
    expected_roles = {
        role: [str(record["source_id"]) for record in selection["roles"][role]["trajectories"]]
        for role in EXPECTED_ROLE_COUNTS
    }
    roles = _require_exact_keys(
        manifest.get("role_source_ids"), set(EXPECTED_ROLE_COUNTS), "payload_manifest.role_source_ids"
    )
    if dict(roles) != expected_roles or any(
        len(roles[role]) != count for role, count in EXPECTED_ROLE_COUNTS.items()
    ):
        raise ValueError("payload manifest role source identities drifted")

    inventories = _require_exact_keys(
        manifest.get("inventories"),
        {
            "image_members_sha256",
            "ocr_records_sha256",
            "ocr_record_aggregate_sha256",
            "trajectory_index_sha256",
        },
        "payload_manifest.inventories",
    )
    for key, digest in inventories.items():
        _require_sha256(digest, f"payload_manifest.inventories.{key}")

    payload_files = manifest.get("payload_files")
    if not isinstance(payload_files, list) or len(payload_files) != 3:
        raise ValueError("payload manifest must contain exactly three payload shards")
    hf_by_path = {record["path"]: record for record in hf_files}
    for index, (record, path) in enumerate(zip(payload_files, PAYLOAD_SHARD_PATHS, strict=True)):
        count_key = "member_count" if index == 0 else "record_count"
        value = _require_exact_keys(
            record,
            {"path", "size_bytes", "sha256", count_key},
            f"payload_manifest.payload_files[{index}]",
        )
        expected_count = 210 if index < 2 else 35
        if (
            value.get("path") != path
            or type(value.get(count_key)) is not int
            or value.get(count_key) != expected_count
            or value.get("size_bytes") != hf_by_path[path]["size_bytes"]
            or value.get("sha256") != hf_by_path[path]["sha256"]
        ):
            raise ValueError(f"payload manifest shard identity drifted: {path}")
    return manifest


def _validate_hf_artifact(
    value: Any,
    *,
    builder_commit: str,
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    artifact = _require_exact_keys(
        value,
        {
            "repo",
            "repo_type",
            "visibility",
            "tag",
            "immutable_revision",
            "tag_resolved_revision",
            "uploaded_from_git_commit",
            "payload_prefix",
            "exact_download_allowlist",
            "artifact_tree_sha256",
            "ocr_record_aggregate_sha256",
            "counts",
            "files",
            "upload",
            "redownload",
        },
        "hf_dataset_artifact",
    )
    if (
        artifact.get("repo") != DATASET_REPO
        or artifact.get("repo_type") != "dataset"
        or artifact.get("visibility") != "private"
        or artifact.get("tag") != HF_TAG
        or artifact.get("payload_prefix") != PAYLOAD_PREFIX
    ):
        raise ValueError("HF derived dataset identity drifted")
    revision = _require_git_sha(
        artifact.get("immutable_revision"), "hf_dataset_artifact.immutable_revision"
    )
    if artifact.get("tag_resolved_revision") != revision:
        raise ValueError("HF tag does not resolve to the immutable revision")
    if artifact.get("uploaded_from_git_commit") != builder_commit:
        raise ValueError("HF upload builder commit drifted")
    if tuple(artifact.get("exact_download_allowlist", ())) != EXACT_FILE_ALLOWLIST:
        raise ValueError("HF exact-six download allowlist drifted")
    files = _validate_file_records(artifact.get("files"), "hf_dataset_artifact.files")
    tree = sha256_bytes(canonical_json_bytes(files))
    if artifact.get("artifact_tree_sha256") != tree:
        raise ValueError("HF artifact tree SHA256 drifted")
    aggregate = _require_sha256(
        artifact.get("ocr_record_aggregate_sha256"),
        "hf_dataset_artifact.ocr_record_aggregate_sha256",
    )
    _validate_counts(artifact.get("counts"), "hf_dataset_artifact.counts")

    upload = _require_exact_keys(
        artifact.get("upload"),
        {
            "client",
            "method",
            "source_repeat",
            "source_materialization_output_path",
            "local_upload_source_path",
            "allow_patterns",
            "commit_message",
            "commit_oid",
            "upload_started_at_utc",
            "upload_ended_at_utc",
            "upload_argv",
            "tag_started_at_utc",
            "tag_ended_at_utc",
            "tag_create_argv",
            "tag_resolution_method",
        },
        "hf_dataset_artifact.upload",
    )
    upload_started = _parse_utc(
        upload.get("upload_started_at_utc"), "hf_dataset_artifact.upload start"
    )
    upload_ended = _parse_utc(
        upload.get("upload_ended_at_utc"), "hf_dataset_artifact.upload end"
    )
    tag_started = _parse_utc(
        upload.get("tag_started_at_utc"), "hf_dataset_artifact.tag start"
    )
    tag_ended = _parse_utc(
        upload.get("tag_ended_at_utc"), "hf_dataset_artifact.tag end"
    )
    if upload_started >= upload_ended or tag_started >= tag_ended:
        raise ValueError("HF upload or tag UTC bracket is not increasing")
    if upload_ended >= tag_started:
        raise ValueError("HF tag creation did not follow upload")
    _require_nonempty_string(upload.get("client"), "upload.client")
    if upload.get("commit_message") != HF_UPLOAD_COMMIT_MESSAGE:
        raise ValueError("HF upload commit message drifted from the execution transcript")
    materialization_source = _require_nonempty_string(
        upload.get("source_materialization_output_path"),
        "upload.source_materialization_output_path",
    )
    local_upload_source = _require_nonempty_string(
        upload.get("local_upload_source_path"),
        "upload.local_upload_source_path",
    )
    if materialization_source == local_upload_source:
        raise ValueError("HF upload must distinguish Hyper source and Mac staging paths")
    if (
        upload.get("method") != "hf_cli_upload+hf_cli_tag_create"
        or upload.get("source_repeat") != 1
        or tuple(upload.get("allow_patterns", ())) != EXACT_FILE_ALLOWLIST
        or upload.get("commit_oid") != revision
    ):
        raise ValueError("HF upload immutable provenance drifted")
    if upload.get("tag_resolution_method") != "HfApi.dataset_info(revision=tag).sha":
        raise ValueError("HF tag resolution method drifted")

    upload_argv = _validate_argv(upload.get("upload_argv"), "HF upload argv")
    if upload_argv[:5] != [
        HF_CLI_PATH,
        "upload",
        DATASET_REPO,
        local_upload_source,
        ".",
    ]:
        raise ValueError("HF upload positional repo/root/path drifted")
    first_upload_flag = next(
        (index for index, item in enumerate(upload_argv) if item.startswith("--")),
        len(upload_argv),
    )
    if first_upload_flag != 5:
        raise ValueError("HF upload contains an unexpected positional argument")
    if any(item == "--delete" or item.startswith("--delete=") for item in upload_argv):
        raise ValueError("HF upload argv cannot delete existing repository files")
    _validate_flag_tail(
        upload_argv,
        start=5,
        value_flags={
            "--repo-type",
            "--revision",
            "--include",
            "--commit-message",
            "--format",
        },
        boolean_flags=set(),
        repeatable_value_flags={"--include"},
        field="HF upload argv",
    )
    _require_single_flag_value(
        upload_argv, "--repo-type", "dataset", "HF upload argv"
    )
    _require_single_flag_value(upload_argv, "--revision", "main", "HF upload argv")
    _require_single_flag_value(
        upload_argv,
        "--commit-message",
        str(upload["commit_message"]),
        "HF upload argv",
    )
    _require_single_flag_value(upload_argv, "--format", "json", "HF upload argv")
    if _flag_values(upload_argv, "--include", "HF upload argv") != list(
        EXACT_FILE_ALLOWLIST
    ):
        raise ValueError("HF upload argv exact-six include inventory drifted")
    expected_upload_argv = [
        HF_CLI_PATH,
        "upload",
        DATASET_REPO,
        local_upload_source,
        ".",
        "--repo-type",
        "dataset",
        "--revision",
        "main",
    ]
    for path in EXACT_FILE_ALLOWLIST:
        expected_upload_argv.extend(["--include", path])
    expected_upload_argv.extend(
        ["--commit-message", HF_UPLOAD_COMMIT_MESSAGE, "--format", "json"]
    )
    if upload_argv != expected_upload_argv:
        raise ValueError("HF upload argv order drifted from the execution transcript")

    tag_argv = _validate_argv(upload.get("tag_create_argv"), "HF tag argv")
    if tag_argv[:6] != [
        HF_CLI_PATH,
        "repos",
        "tag",
        "create",
        DATASET_REPO,
        HF_TAG,
    ]:
        raise ValueError("HF tag argv repo or tag drifted")
    first_tag_flag = next(
        (index for index, item in enumerate(tag_argv) if item.startswith("--")),
        len(tag_argv),
    )
    if first_tag_flag != 6:
        raise ValueError("HF tag argv contains an unexpected positional argument")
    _validate_flag_tail(
        tag_argv,
        start=6,
        value_flags={"--repo-type", "--revision", "--message", "--format"},
        boolean_flags=set(),
        repeatable_value_flags=set(),
        field="HF tag argv",
    )
    _require_single_flag_value(tag_argv, "--repo-type", "dataset", "HF tag argv")
    _require_single_flag_value(tag_argv, "--revision", revision, "HF tag argv")
    _require_single_flag_value(tag_argv, "--message", HF_TAG_MESSAGE, "HF tag argv")
    _require_single_flag_value(tag_argv, "--format", "json", "HF tag argv")
    expected_tag_argv = [
        HF_CLI_PATH,
        "repos",
        "tag",
        "create",
        DATASET_REPO,
        HF_TAG,
        "--repo-type",
        "dataset",
        "--revision",
        revision,
        "--message",
        HF_TAG_MESSAGE,
        "--format",
        "json",
    ]
    if tag_argv != expected_tag_argv:
        raise ValueError("HF tag argv order drifted from the execution transcript")

    redownload = _require_exact_keys(
        artifact.get("redownload"),
        {
            "started_at_utc",
            "ended_at_utc",
            "client",
            "method",
            "argv",
            "revision",
            "allow_patterns",
            "force_download",
            "fresh_cache_dir",
            "fresh_cache_dir_preexisted",
            "download_root",
            "projection_root",
            "validated_output_path",
            "observed_files",
            "all_listed_file_hashes_verified",
            "artifact_validator_replayed_on_hyper00",
            "validator_outcome",
            "ocr_replay_record_count",
            "ocr_replay_aggregate_sha256",
            "artifact_tree_sha256",
        },
        "hf_dataset_artifact.redownload",
    )
    _validate_utc_bracket(redownload, "hf_dataset_artifact.redownload")
    _require_nonempty_string(redownload.get("client"), "redownload.client")
    if redownload.get("method") != "hf_cli_download+exact_six_clean_projection":
        raise ValueError("HF redownload method drifted")
    if redownload.get("revision") != revision:
        raise ValueError("HF redownload revision drifted")
    if tuple(redownload.get("allow_patterns", ())) != EXACT_FILE_ALLOWLIST:
        raise ValueError("HF redownload exact-six allowlist drifted")
    if redownload.get("force_download") is not True:
        raise ValueError("HF redownload must force immutable bytes")
    _require_nonempty_string(redownload.get("fresh_cache_dir"), "redownload.fresh_cache_dir")
    download_root = _require_nonempty_string(
        redownload.get("download_root"), "redownload.download_root"
    )
    projection_root = _require_nonempty_string(
        redownload.get("projection_root"), "redownload.projection_root"
    )
    if download_root == projection_root:
        raise ValueError("HF download and clean projection roots must be distinct")
    _require_nonempty_string(
        redownload.get("validated_output_path"), "redownload.validated_output_path"
    )
    if redownload.get("validated_output_path") != projection_root:
        raise ValueError("HF validator must consume the exact-six clean projection")
    redownload_argv = _validate_argv(
        redownload.get("argv"), "HF immutable redownload argv"
    )
    expected_download_prefix = [
        HF_CLI_PATH,
        "download",
        DATASET_REPO,
        *EXACT_FILE_ALLOWLIST,
    ]
    first_download_flag = next(
        (index for index, item in enumerate(redownload_argv) if item.startswith("--")),
        len(redownload_argv),
    )
    if (
        first_download_flag != len(expected_download_prefix)
        or redownload_argv[:first_download_flag] != expected_download_prefix
    ):
        raise ValueError("HF redownload positional exact-six inventory drifted")
    for forbidden in ("--cache-dir", "--include", "--exclude", "--no-force-download"):
        if any(
            item == forbidden or item.startswith(f"{forbidden}=")
            for item in redownload_argv
        ):
            raise ValueError(f"HF redownload argv cannot use {forbidden}")
    _validate_flag_tail(
        redownload_argv,
        start=len(expected_download_prefix),
        value_flags={"--repo-type", "--revision", "--local-dir", "--format"},
        boolean_flags={"--force-download"},
        repeatable_value_flags=set(),
        field="HF redownload argv",
    )
    _require_single_flag_value(
        redownload_argv, "--repo-type", "dataset", "HF redownload argv"
    )
    _require_single_flag_value(
        redownload_argv, "--revision", revision, "HF redownload argv"
    )
    _require_single_flag_value(
        redownload_argv, "--local-dir", download_root, "HF redownload argv"
    )
    _require_single_flag_value(
        redownload_argv, "--format", "json", "HF redownload argv"
    )
    if redownload_argv.count("--force-download") != 1:
        raise ValueError("HF redownload argv must force immutable bytes exactly once")
    expected_redownload_argv = [
        HF_CLI_PATH,
        "download",
        DATASET_REPO,
        *EXACT_FILE_ALLOWLIST,
        "--repo-type",
        "dataset",
        "--revision",
        revision,
        "--local-dir",
        download_root,
        "--force-download",
        "--format",
        "json",
    ]
    if redownload_argv != expected_redownload_argv:
        raise ValueError("HF redownload argv order drifted from the execution transcript")
    if redownload.get("fresh_cache_dir_preexisted") is not False:
        raise ValueError("HF redownload cache was not fresh")
    observed_files = _validate_file_records(
        redownload.get("observed_files"), "hf_dataset_artifact.redownload.observed_files"
    )
    if observed_files != files:
        raise ValueError("HF redownload observed file identities drifted")
    if (
        redownload.get("all_listed_file_hashes_verified") is not True
        or redownload.get("artifact_validator_replayed_on_hyper00") is not True
        or redownload.get("validator_outcome") != VALIDATOR_OUTCOME
        or redownload.get("ocr_replay_record_count") != 210
        or redownload.get("ocr_replay_aggregate_sha256") != aggregate
        or redownload.get("artifact_tree_sha256") != tree
    ):
        raise ValueError("HF immutable redownload replay evidence drifted")
    return artifact, files


def _validate_run_record(
    value: Any,
    *,
    repeat: int,
    builder_commit: str,
    tree: str,
    aggregate: str,
) -> Mapping[str, Any]:
    record = _require_exact_keys(
        value,
        {
            "repeat",
            "started_at_utc",
            "ended_at_utc",
            "argv",
            "output_path",
            "output_dir_preexisted",
            "outcome",
            "artifact_tree_sha256",
            "counts",
            "generated_ocr_record_count",
            "generated_ocr_record_aggregate_sha256",
            "ocr_record_aggregate_sha256",
            "ocr_replay_performed",
            "ocr_replay_record_count",
            "ocr_replay_aggregate_sha256",
            "policy_loaded",
            "policy_output_generated",
            "restoration_output_generated",
        },
        f"materialization_repeats[{repeat - 1}]",
    )
    if record.get("repeat") != repeat:
        raise ValueError("materialization repeat IDs drifted")
    _validate_utc_bracket(record, f"materialization_repeats[{repeat - 1}]")
    argv = record.get("argv")
    if not isinstance(argv, list) or not argv or any(
        not isinstance(item, str) or not item for item in argv
    ):
        raise ValueError("materialization argv must be a non-empty string array")
    if "--git-revision" not in argv or argv[argv.index("--git-revision") + 1] != builder_commit:
        raise ValueError("materialization argv builder revision drifted")
    output_path = _require_nonempty_string(record.get("output_path"), "materialization output_path")
    if "--output-dir" not in argv or argv[argv.index("--output-dir") + 1] != output_path:
        raise ValueError("materialization argv output path drifted")
    if record.get("output_dir_preexisted") is not False:
        raise ValueError("materialization output directory was not fresh")
    _validate_counts(record.get("counts"), "materialization counts")
    if (
        record.get("outcome") != VALIDATOR_OUTCOME
        or record.get("artifact_tree_sha256") != tree
        or record.get("generated_ocr_record_count") != 210
        or record.get("generated_ocr_record_aggregate_sha256") != aggregate
        or record.get("ocr_record_aggregate_sha256") != aggregate
        or record.get("ocr_replay_performed") is not True
        or record.get("ocr_replay_record_count") != 210
        or record.get("ocr_replay_aggregate_sha256") != aggregate
        or record.get("policy_loaded") is not False
        or record.get("policy_output_generated") is not False
        or record.get("restoration_output_generated") is not False
    ):
        raise ValueError("materialization deterministic replay evidence drifted")
    return record


def _validate_validation_record(
    value: Any,
    *,
    validation_id: str,
    tree: str,
    aggregate: str,
) -> Mapping[str, Any]:
    record = _require_exact_keys(
        value,
        {
            "validation_id",
            "source",
            "started_at_utc",
            "ended_at_utc",
            "argv",
            "output_path",
            "outcome",
            "artifact_tree_sha256",
            "ocr_record_aggregate_sha256",
            "ocr_replay_performed",
            "ocr_replay_record_count",
            "ocr_replay_aggregate_sha256",
            "policy_loaded",
            "policy_output_generated",
            "restoration_output_generated",
        },
        f"artifact_validations.{validation_id}",
    )
    expected_source = (
        "standalone_validator" if validation_id.startswith("hf-") else "builder_integrated"
    )
    if record.get("validation_id") != validation_id or record.get("source") != expected_source:
        raise ValueError("artifact validation ID or source drifted")
    _validate_utc_bracket(record, f"artifact_validations.{validation_id}")
    argv = record.get("argv")
    if not isinstance(argv, list) or not argv or any(
        not isinstance(item, str) or not item for item in argv
    ):
        raise ValueError("artifact validation argv must be a non-empty string array")
    _require_nonempty_string(record.get("output_path"), "artifact validation output_path")
    if (
        record.get("outcome") != VALIDATOR_OUTCOME
        or record.get("artifact_tree_sha256") != tree
        or record.get("ocr_record_aggregate_sha256") != aggregate
        or record.get("ocr_replay_performed") is not True
        or record.get("ocr_replay_record_count") != 210
        or record.get("ocr_replay_aggregate_sha256") != aggregate
        or record.get("policy_loaded") is not False
        or record.get("policy_output_generated") is not False
        or record.get("restoration_output_generated") is not False
    ):
        raise ValueError("artifact validation exact replay evidence drifted")
    return record


def validate_completion_manifest(
    *,
    artifact_manifest_path: str | Path,
    repository_root: str | Path,
    expected_builder_commit: str = FROZEN_BUILDER_COMMIT,
    expected_builder_tree: str = FROZEN_BUILDER_TREE,
    expected_builder_source_sha256: Mapping[str, str] = FROZEN_BUILDER_SOURCE_SHA256,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {root}")
    manifest_path = Path(artifact_manifest_path).resolve()
    expected_manifest_path = root.joinpath(
        *PurePosixPath(COMPLETION_MANIFEST_PATH).parts
    ).resolve()
    if manifest_path != expected_manifest_path:
        raise ValueError("completion manifest path drifted")
    manifest_payload, manifest = load_json_object(manifest_path)
    if manifest_payload != pretty_json_bytes(manifest):
        raise ValueError("completion manifest is not canonical pretty JSON")
    manifest = _require_exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "artifact_id",
            "status",
            "builder_snapshot",
            "source_files",
            "summary",
            "hf_dataset_artifact",
            "materialization_repeat_count",
            "artifact_validation_count",
            "dependency_1_closed",
            "policy_loaded_before_manifest",
            "policy_output_generated_before_manifest",
            "restoration_output_generated_before_manifest",
        },
        "completion manifest",
    )
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "artifact_id": ARTIFACT_ID,
        "status": STATUS,
        "materialization_repeat_count": 2,
        "artifact_validation_count": 3,
        "dependency_1_closed": True,
        "policy_loaded_before_manifest": False,
        "policy_output_generated_before_manifest": False,
        "restoration_output_generated_before_manifest": False,
    }
    if any(manifest.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("completion manifest identity or pre-output state drifted")

    historical_blobs, historical_hashes = _validate_builder_history(
        manifest.get("builder_snapshot"),
        repository_root=root,
        expected_builder_commit=expected_builder_commit,
        expected_builder_tree=expected_builder_tree,
        expected_builder_source_sha256=expected_builder_source_sha256,
    )
    completion_source_paths = tuple(COMPLETION_SOURCE_PATHS)
    completion_sources = _validate_source_records(
        manifest.get("source_files"),
        expected_paths=completion_source_paths,
        repository_root=root,
        historical_blobs=None,
        field="completion source_files",
    )
    summary_identity = _require_exact_keys(
        manifest.get("summary"), {"path", "sha256"}, "completion summary"
    )
    if summary_identity.get("path") != COMPLETION_SOURCE_PATHS[-1]:
        raise ValueError("completion summary path drifted")
    summary_sha = _require_sha256(summary_identity.get("sha256"), "completion summary SHA256")
    if completion_sources[COMPLETION_SOURCE_PATHS[-1]] != summary_sha:
        raise ValueError("completion summary/source identity drifted")
    summary_path = root.joinpath(*PurePosixPath(COMPLETION_SOURCE_PATHS[-1]).parts)
    summary_payload, summary = load_json_object(summary_path)
    if summary_payload != pretty_json_bytes(summary):
        raise ValueError("completion summary is not canonical pretty JSON")

    hf_artifact, files = _validate_hf_artifact(
        manifest.get("hf_dataset_artifact"),
        builder_commit=expected_builder_commit,
    )
    summary = _require_exact_keys(
        summary,
        {
            "schema_version",
            "protocol_id",
            "outcome",
            "artifact_id",
            "source",
            "payload_manifest_snapshot",
            "materialization_repeats",
            "deterministic_identity",
            "artifact_validations",
            "hf_dataset_artifact",
            "runtime",
            "negative_declarations",
            "dependency_1_closed",
        },
        "completion summary",
    )
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("protocol_id") != PROTOCOL_ID
        or summary.get("outcome") != SUMMARY_OUTCOME
        or summary.get("artifact_id") != ARTIFACT_ID
        or summary.get("dependency_1_closed") is not True
    ):
        raise ValueError("completion summary identity drifted")
    source = _require_exact_keys(
        summary.get("source"),
        {
            "builder_git_commit",
            "builder_git_tree",
            "clean_checkout",
            "head_equals_origin_main",
            "checkout_transport",
        },
        "completion summary source",
    )
    if source != {
        "builder_git_commit": expected_builder_commit,
        "builder_git_tree": expected_builder_tree,
        "clean_checkout": True,
        "head_equals_origin_main": True,
        "checkout_transport": EXPECTED_CHECKOUT_TRANSPORT,
    }:
        raise ValueError("completion summary historical source state drifted")

    if summary.get("hf_dataset_artifact") != hf_artifact:
        raise ValueError("completion manifest/summary HF evidence drifted")
    tree = str(hf_artifact["artifact_tree_sha256"])
    aggregate = str(hf_artifact["ocr_record_aggregate_sha256"])
    snapshot = _require_exact_keys(
        summary.get("payload_manifest_snapshot"),
        {"path", "sha256", "content"},
        "payload_manifest_snapshot",
    )
    if snapshot.get("path") != EXACT_FILE_ALLOWLIST[3]:
        raise ValueError("payload manifest snapshot path drifted")
    snapshot_sha = _require_sha256(snapshot.get("sha256"), "payload manifest snapshot SHA256")
    manifest_file = files[3]
    if snapshot_sha != manifest_file["sha256"] or sha256_bytes(
        pretty_json_bytes(snapshot.get("content"))
    ) != snapshot_sha:
        raise ValueError("payload manifest snapshot bytes are not bound to the HF file")
    payload_manifest = _validate_payload_manifest(
        snapshot.get("content"),
        builder_commit=expected_builder_commit,
        historical_blobs=historical_blobs,
        historical_source_sha256=historical_hashes,
        hf_files=files,
    )
    if payload_manifest["inventories"]["ocr_record_aggregate_sha256"] != aggregate:
        raise ValueError("payload/HF OCR aggregate identity drifted")

    deterministic = _require_exact_keys(
        summary.get("deterministic_identity"),
        {
            "materialization_repeat_count",
            "canonical_files",
            "artifact_tree_sha256",
            "ocr_record_aggregate_sha256",
            "all_six_files_byte_identical",
            "ocr_record_aggregate_byte_identical",
        },
        "deterministic_identity",
    )
    canonical_files = _validate_file_records(
        deterministic.get("canonical_files"), "deterministic_identity.canonical_files"
    )
    if (
        deterministic.get("materialization_repeat_count") != 2
        or canonical_files != files
        or deterministic.get("artifact_tree_sha256") != tree
        or deterministic.get("ocr_record_aggregate_sha256") != aggregate
        or deterministic.get("all_six_files_byte_identical") is not True
        or deterministic.get("ocr_record_aggregate_byte_identical") is not True
    ):
        raise ValueError("two-build deterministic identity evidence drifted")

    repeats = summary.get("materialization_repeats")
    if not isinstance(repeats, list) or len(repeats) != 2:
        raise ValueError("completion summary requires exactly two materializations")
    validated_repeats = [
        _validate_run_record(
            record,
            repeat=index,
            builder_commit=expected_builder_commit,
            tree=tree,
            aggregate=aggregate,
        )
        for index, record in enumerate(repeats, start=1)
    ]
    if validated_repeats[0]["output_path"] == validated_repeats[1]["output_path"]:
        raise ValueError("materialization repeats must use distinct output paths")
    if _parse_utc(validated_repeats[0]["ended_at_utc"], "repeat 1 end") >= _parse_utc(
        validated_repeats[1]["started_at_utc"], "repeat 2 start"
    ):
        raise ValueError("materialization repeats are not independently ordered")
    upload = hf_artifact["upload"]
    redownload = hf_artifact["redownload"]
    if upload["source_materialization_output_path"] != validated_repeats[0]["output_path"]:
        raise ValueError("HF upload source is not materialization repeat 1")
    if _parse_utc(validated_repeats[1]["ended_at_utc"], "repeat 2 end") >= _parse_utc(
        upload["upload_started_at_utc"], "HF upload start"
    ):
        raise ValueError("HF upload did not follow both materializations")
    if _parse_utc(upload["tag_ended_at_utc"], "HF tag end") >= _parse_utc(
        redownload["started_at_utc"], "HF redownload start"
    ):
        raise ValueError("HF immutable redownload did not follow upload and tag creation")

    validations = summary.get("artifact_validations")
    if not isinstance(validations, list) or len(validations) != 3:
        raise ValueError("completion summary requires exactly three artifact validations")
    if tuple(record.get("validation_id") for record in validations if isinstance(record, Mapping)) != (
        EXPECTED_VALIDATION_IDS
    ):
        raise ValueError("artifact validation IDs or order drifted")
    validated = [
        _validate_validation_record(
            record,
            validation_id=validation_id,
            tree=tree,
            aggregate=aggregate,
        )
        for record, validation_id in zip(validations, EXPECTED_VALIDATION_IDS, strict=True)
    ]
    if validated[0]["output_path"] != validated_repeats[0]["output_path"] or validated[1][
        "output_path"
    ] != validated_repeats[1]["output_path"]:
        raise ValueError("builder-integrated validation output paths drifted")
    for index in (0, 1):
        if (
            validated[index]["argv"] != validated_repeats[index]["argv"]
            or validated[index]["started_at_utc"]
            != validated_repeats[index]["started_at_utc"]
            or validated[index]["ended_at_utc"]
            != validated_repeats[index]["ended_at_utc"]
        ):
            raise ValueError("builder-integrated validation invocation drifted")
    hf_validation = validated[2]
    if hf_validation["output_path"] != redownload["validated_output_path"]:
        raise ValueError("HF redownload validation output path drifted")
    hf_argv = hf_validation["argv"]
    if (
        "--output-dir" not in hf_argv
        or hf_argv[hf_argv.index("--output-dir") + 1]
        != redownload["validated_output_path"]
        or "--git-revision" not in hf_argv
        or hf_argv[hf_argv.index("--git-revision") + 1]
        != expected_builder_commit
    ):
        raise ValueError("HF standalone replay argv drifted")
    if _parse_utc(redownload["ended_at_utc"], "HF redownload end") > _parse_utc(
        hf_validation["started_at_utc"], "HF validation start"
    ):
        raise ValueError("HF standalone replay preceded immutable redownload")

    runtime = _require_exact_keys(
        summary.get("runtime"),
        {
            "host_alias",
            "hostname",
            "architecture",
            "kernel",
            "python",
            "container_name",
            "container_id",
            "container_image",
            "container_image_digest",
            "provider",
            "gpu_used",
            "dtype",
            "seed",
            "library_versions",
        },
        "completion summary runtime",
    )
    for key in (
        "host_alias",
        "hostname",
        "architecture",
        "kernel",
        "python",
        "container_name",
        "container_id",
        "container_image",
        "container_image_digest",
        "provider",
        "dtype",
    ):
        _require_nonempty_string(runtime.get(key), f"runtime.{key}")
    if runtime.get("gpu_used") is not False or runtime.get("seed") is not None:
        raise ValueError("completion runtime must remain CPU policy-blind")
    libraries = _require_exact_keys(
        runtime.get("library_versions"),
        set(EXPECTED_COMPLETION_LIBRARY_VERSIONS),
        "runtime.library_versions",
    )
    if dict(libraries) != EXPECTED_COMPLETION_LIBRARY_VERSIONS:
        raise ValueError("completion runtime library versions drifted")

    expected_negative = {
        "confirm_policy_scores_observed": False,
        "policy_module_loaded": False,
        "policy_forward_called": False,
        "policy_output_generated": False,
        "restoration_output_generated": False,
    }
    negative = _require_exact_keys(
        summary.get("negative_declarations"), set(expected_negative), "negative_declarations"
    )
    if dict(negative) != expected_negative:
        raise ValueError("completion pre-policy negative declarations drifted")

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "outcome": "PASSED_RESTORATION_V2_DERIVED_ARTIFACT_SOURCE_VALIDATION",
        "artifact_manifest_sha256": sha256_bytes(manifest_payload),
        "summary_sha256": summary_sha,
        "builder_git_commit": expected_builder_commit,
        "builder_git_tree": expected_builder_tree,
        "completion_source_file_count": len(completion_sources),
        "hf_dataset_repo": DATASET_REPO,
        "hf_dataset_tag": HF_TAG,
        "hf_dataset_immutable_revision": hf_artifact["immutable_revision"],
        "hf_file_count": len(files),
        "materialization_repeat_count": len(validated_repeats),
        "artifact_validation_count": len(validated),
        "ocr_replay_record_count": 210,
        "dependency_1_closed": True,
        "policy_output_generated_by_this_validation": False,
        "restoration_output_generated_by_this_validation": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_completion_manifest(
        artifact_manifest_path=args.artifact_manifest,
        repository_root=args.repository_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
