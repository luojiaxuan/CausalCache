"""Formal no-GPU runner for the expansion-label scientific repair child."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import causalcache.restoration_v2_2_expansion_labels_artifact as frozen_v1
from causalcache.restoration_v2_2_expansion_labels_invalid_forensic import (
    ARCHIVE_STATUS as P0_ARCHIVE_STATUS,
    InvalidForensicContract,
    InvalidForensicEvidence,
    canonical_json_bytes,
    collect_invalid_forensic_source,
    load_frozen_forensic_contract,
    read_invalid_forensic_archive,
    sha256_bytes,
    strict_pretty_json_object_bytes,
)
from causalcache.restoration_v2_2_expansion_labels_invalid_forensic_tag_resolution import (
    TagResolutionContract,
    load_frozen_tag_resolution_contract,
    validate_parent_publication_state,
)
from causalcache.restoration_v2_2_expansion_labels_scientific_repair import (
    FORMAL_EXTERNAL_REPLAY_TIER,
    FROZEN_DERIVED_ARTIFACT,
    FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION,
    FROZEN_OCR_BACKEND_CONFIG,
    FROZEN_PARENT_SUBSTRATE,
    PASS_STATUS as CORE_PASS_STATUS,
    PROTOCOL_ID as CORE_PROTOCOL_ID,
    FrozenExternalReplayAttestation,
    FrozenExternalReplayExecutor,
    LedgerNeutralScientificValidation,
    validate_ledger_neutral_scientific_payload,
)
from causalcache.restoration_v2_2_expansion_math_audit import (
    audit_normalized_raw_states,
    compare_with_reducer_output,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = (
    "causalcache_restoration_v2_2_expansion_exact_labels_"
    "scientific_repair_runner_v1"
)
SOURCE_STATUS = "source_only_frozen_before_formal_no_gpu_scientific_repair"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_"
    "expansion_labels_scientific_repair_runner_v1.json"
)
FROZEN_CONFIG_SHA256 = (
    "4f4202944674192090cf3df19b864c96e7082628ab25ca8287affa32944e64c2"
)
IMPORT_GUARD_MARKER = (
    "causalcache_formal_no_model_framework_import_guard_v1"
)
TEST_ONLY_STATUS = "TEST_ONLY_EXPANSION_LABEL_SCIENTIFIC_REPAIR_RUNNER_V1"
AUDIT_STATUS = "VALIDATED_EXPANSION_LABEL_SCIENTIFIC_REPAIR_AUDIT_V1"
MANIFEST_STATUS = "IMMUTABLE_REPAIRED_EXPANSION_LABEL_PAYLOAD_MANIFEST_V1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
_CONTAINER_NAME = re.compile(r"sglang-omni-jaxan-[0-9]{8}")


@dataclass(frozen=True)
class ScientificRepairRunnerContract:
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path

    @property
    def output(self) -> Mapping[str, Any]:
        return _mapping(self.data["output_contract"], "output contract")

    @property
    def remote(self) -> Mapping[str, Any]:
        return _mapping(
            self.data["immutable_invalid_forensic_source"],
            "immutable invalid-forensic source",
        )

    @property
    def runtime(self) -> Mapping[str, Any]:
        return _mapping(self.data["no_gpu_runtime"], "no-GPU runtime")


@dataclass(frozen=True)
class SourceIdentity:
    head: str
    origin_main: str
    remote_main: str
    branch: str
    origin_url: str
    inventory: tuple[Mapping[str, Any], ...]
    inventory_sha256: str
    loaded_module_inventory: tuple[Mapping[str, Any], ...]
    loaded_module_inventory_sha256: str


@dataclass(frozen=True)
class BoundStateSnapshot:
    p1_claim: bytes
    tag_claim: bytes
    tag_completion: bytes


@dataclass(frozen=True)
class OriginalInvalidSnapshot:
    source_evidence: InvalidForensicEvidence
    local_archive_evidence: InvalidForensicEvidence
    local_archive_bytes: bytes
    bound_state: BoundStateSnapshot


@dataclass(frozen=True)
class FreshForensicDownload:
    evidence: InvalidForensicEvidence
    archive_bytes: bytes
    sidecar_bytes: bytes
    attestation: Mapping[str, Any]
    formal_hf_download: bool


@dataclass(frozen=True)
class RepairedArchive:
    files: Mapping[str, bytes]
    archive_bytes: bytes
    archive_sha256: str
    archive_size_bytes: int
    inventory: tuple[Mapping[str, Any], ...]
    tree_inventory_sha256: str


@dataclass(frozen=True)
class StableArchiveHandle:
    descriptor: int
    path: Path
    stat_fingerprint: tuple[int, ...]
    payload: bytes
    archive: RepairedArchive


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise TypeError(f"{label} must be a sequence")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} schema drifted")


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _git_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase Git commit")
    return value


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"{label} must be canonical relative POSIX")
    return value


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_json_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be one JSON object")
    return value


def _regular_file_bytes(
    path: str | Path,
    *,
    label: str,
    mode: int | None = None,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> bytes:
    candidate = Path(path)
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
        raise ValueError(f"{label} mode drifted")
    payload = candidate.read_bytes()
    if expected_size is not None and len(payload) != expected_size:
        raise ValueError(f"{label} size drifted")
    if expected_sha256 is not None and sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"{label} SHA256 drifted")
    return payload


def load_frozen_runner_contract(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> ScientificRepairRunnerContract:
    root = Path(repository_root).resolve()
    source = Path(path).resolve()
    expected = (root / CANONICAL_CONFIG_PATH).resolve()
    if source != expected or root.is_symlink() or source.is_symlink():
        raise ValueError("runner config must use the canonical repository path")
    payload = _regular_file_bytes(source, label="runner config")
    digest = sha256_bytes(payload)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError("runner config SHA256 drifted")
    data = dict(_strict_json_object(payload, label="runner config"))
    expected_top = {
        "schema_version",
        "protocol_id",
        "status",
        "parent_scientific_repair_core",
        "git_source_contract",
        "immutable_invalid_forensic_source",
        "p0_forensic_contract",
        "tag_resolution_parent",
        "original_invalid_attempt",
        "external_replay_inputs",
        "no_gpu_runtime",
        "scientific_payload",
        "output_contract",
        "formal_consumption",
        "planned_hf_publication",
    }
    _exact_keys(data, expected_top, "runner config")
    if (
        data["schema_version"] != SCHEMA_VERSION
        or data["protocol_id"] != PROTOCOL_ID
        or data["status"] != SOURCE_STATUS
        or data["parent_scientific_repair_core"]["git_commit"]
        != "c537a1f1c12b9afe1b324cb481a9af7e596e9c5f"
        or data["parent_scientific_repair_core"]["sha256"]
        != "6b1c022758844284a0c8364a33aa12457db1fa08dba9e5567c8fe3a48cfa744c"
        or data["output_contract"]["exact_members"]
        != [
            "audit.json",
            "derived_labels.jsonl",
            "manifest.json",
            "raw_states.jsonl",
        ]
        or data["formal_consumption"]["gate_training_unlocked"] is not False
        or data["planned_hf_publication"]["publication_authorized_by_this_runner"]
        is not False
    ):
        raise ValueError("runner config frozen identity drifted")
    return ScientificRepairRunnerContract(
        data=data,
        sha256=digest,
        repository_root=root,
        source_path=source,
    )


def _git_bytes(root: Path, arguments: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ValueError(f"git {' '.join(arguments)} failed")
    return completed.stdout


def _remote_main_head(root: Path, *, remote_name: str) -> str:
    completed = subprocess.run(
        ["git", "ls-remote", "--exit-code", remote_name, "refs/heads/main"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ValueError("canonical remote main could not be resolved")
    try:
        lines = completed.stdout.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("canonical remote main response is not ASCII") from error
    expected_suffix = "\trefs/heads/main"
    if len(lines) != 1 or not lines[0].endswith(expected_suffix):
        raise ValueError("canonical remote main response schema drifted")
    return _git_sha(lines[0][: -len(expected_suffix)], "canonical remote main")


def _loaded_project_module_inventory(
    root: Path,
    *,
    head: str,
) -> tuple[Mapping[str, Any], ...]:
    code_root = root / "code"
    if code_root.is_symlink() or not code_root.is_dir():
        raise ValueError("repository code root must be one real directory")
    records: list[Mapping[str, Any]] = []
    for module_name, module in sorted(sys.modules.items()):
        if not (
            module_name == "causalcache"
            or module_name.startswith("causalcache.")
            or module_name == "scripts"
            or module_name.startswith("scripts.")
        ):
            continue
        file_value = getattr(module, "__file__", None)
        if not isinstance(file_value, str):
            raise ValueError(f"loaded project module has no source file: {module_name}")
        candidate = Path(file_value)
        if not candidate.is_absolute() or candidate.suffix != ".py":
            raise ValueError(f"loaded project module is not a source .py: {module_name}")
        try:
            relative_to_code = candidate.relative_to(code_root)
        except ValueError as error:
            raise ValueError(
                f"loaded project module escaped repository code: {module_name}"
            ) from error
        if any(part in {"", ".", ".."} for part in relative_to_code.parts):
            raise ValueError(
                f"loaded project module path is not canonical: {module_name}"
            )
        current = code_root
        for component in relative_to_code.parts:
            current /= component
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(
                    f"loaded project module traverses a symlink: {module_name}"
                )
        relative = (Path("code") / relative_to_code).as_posix()
        local = _regular_file_bytes(candidate, label=f"loaded module {module_name}")
        _git_bytes(root, ["ls-files", "--error-unmatch", "--", relative])
        committed = _git_bytes(root, ["show", f"{head}:{relative}"])
        if local != committed:
            raise ValueError(f"loaded project module differs from HEAD: {module_name}")
        records.append(
            {
                "module": module_name,
                "path": relative,
                "sha256": sha256_bytes(local),
                "size_bytes": len(local),
            }
        )
    if not records:
        raise ValueError("loaded project module closure is empty")
    return tuple(records)


def validate_clean_pushed_source(
    contract: ScientificRepairRunnerContract,
    *,
    expected_head: str,
    verify_remote: bool = True,
    expected_remote_main: str | None = None,
) -> SourceIdentity:
    root = contract.repository_root
    source_contract = contract.data["git_source_contract"]
    head = _git_bytes(root, ["rev-parse", "HEAD"]).decode().strip()
    origin_main = _git_bytes(root, ["rev-parse", "origin/main"]).decode().strip()
    branch = _git_bytes(root, ["branch", "--show-current"]).decode().strip()
    origin_url = _git_bytes(root, ["remote", "get-url", "origin"]).decode().strip()
    status_bytes = _git_bytes(root, ["status", "--porcelain=v1", "--untracked-files=all"])
    remote_main = (
        _remote_main_head(root, remote_name=source_contract["origin_name"])
        if verify_remote
        else _git_sha(expected_remote_main, "previously sealed remote main")
    )
    if (
        head != _git_sha(expected_head, "expected source HEAD")
        or head != origin_main
        or head != remote_main
        or branch != source_contract["branch"]
        or origin_url != source_contract["origin_url"]
        or status_bytes
    ):
        raise ValueError("formal source must be clean pushed canonical main")
    parent = contract.data["parent_scientific_repair_core"]["git_commit"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", parent, head],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if ancestor.returncode != 0:
        raise ValueError("formal source is not a descendant of the frozen core")
    inventory: list[Mapping[str, Any]] = []
    for relative in source_contract["required_source_paths"]:
        relative = _safe_relative_path(relative, "source path")
        local = _regular_file_bytes(root / relative, label=f"source {relative}")
        committed = _git_bytes(root, ["show", f"{head}:{relative}"])
        if local != committed:
            raise ValueError(f"source file differs from pushed HEAD: {relative}")
        inventory.append(
            {
                "path": relative,
                "sha256": sha256_bytes(local),
                "size_bytes": len(local),
            }
        )
    core = contract.data["parent_scientific_repair_core"]
    core_bytes = _regular_file_bytes(root / core["path"], label="repair core")
    if (
        len(core_bytes) != core["size_bytes"]
        or sha256_bytes(core_bytes) != core["sha256"]
    ):
        raise ValueError("frozen scientific repair core blob drifted")
    normalized = tuple(inventory)
    loaded_modules = _loaded_project_module_inventory(root, head=head)
    return SourceIdentity(
        head=head,
        origin_main=origin_main,
        remote_main=remote_main,
        branch=branch,
        origin_url=origin_url,
        inventory=normalized,
        inventory_sha256=sha256_bytes(canonical_json_bytes(normalized)),
        loaded_module_inventory=loaded_modules,
        loaded_module_inventory_sha256=sha256_bytes(
            canonical_json_bytes(loaded_modules)
        ),
    )


def _guard_roots(contract: ScientificRepairRunnerContract) -> tuple[str, ...]:
    return tuple(contract.runtime["forbidden_import_roots"])


def validate_import_guard(contract: ScientificRepairRunnerContract) -> Mapping[str, Any]:
    roots = _guard_roots(contract)
    guards = [
        finder
        for finder in sys.meta_path
        if getattr(finder, "causalcache_guard_marker", None) == IMPORT_GUARD_MARKER
    ]
    expected_modules = {
        "__main__",
        "scripts.run_restoration_v2_2_expansion_labels_scientific_repair",
    }
    if (
        len(guards) != 1
        or tuple(getattr(guards[0], "blocked_roots", ())) != roots
        or type(guards[0]).__name__ != "NoModelFrameworkImportGuard"
        or type(guards[0]).__module__ not in expected_modules
        or getattr(type(guards[0]).find_spec, "__qualname__", "")
        != "NoModelFrameworkImportGuard.find_spec"
    ):
        raise ValueError("formal model-framework import guard is missing or drifted")
    imported = sorted(
        name
        for name in sys.modules
        if any(name == root or name.startswith(root + ".") for root in roots)
    )
    if imported:
        raise ValueError("formal repair imported a forbidden model framework")
    for root in roots:
        probe = f"{root}.__causalcache_formal_import_guard_probe__"
        try:
            importlib.util.find_spec(probe)
        except ImportError as error:
            if str(error) != (
                f"model-framework import forbidden in formal CPU repair: {probe}"
            ) and str(error) != (
                f"model-framework import forbidden in formal CPU repair: {root}"
            ):
                raise ValueError("formal import guard probe error drifted") from error
        else:
            raise ValueError("formal import guard did not block an active probe")
    return {
        "guard_marker": IMPORT_GUARD_MARKER,
        "blocked_roots": list(roots),
        "forbidden_modules_imported": imported,
        "pil_allowed": True,
    }


def _nvidia_device_nodes() -> list[str]:
    return sorted(
        {
            str(path)
            for pattern in (
                "/dev/nvidia[0-9]*",
                "/dev/nvidiactl",
                "/dev/nvidia-uvm",
                "/dev/nvidia-uvm-tools",
                "/dev/nvidia-modeset",
                "/dev/nvidia-caps/*",
            )
            for path in Path("/").glob(pattern.lstrip("/"))
            if path.exists() or path.is_symlink()
        }
    )


def validate_no_gpu_launch_receipt(
    contract: ScientificRepairRunnerContract,
    *,
    source: SourceIdentity,
    receipt_path: str | Path,
) -> Mapping[str, Any]:
    runtime = contract.runtime
    path = Path(receipt_path).resolve()
    if path != Path(runtime["launch_receipt_path"]).resolve():
        raise ValueError("no-GPU launch receipt path drifted")
    payload = _regular_file_bytes(
        path,
        label="no-GPU launch receipt",
        mode=runtime["launch_receipt_mode"],
    )
    receipt = dict(strict_pretty_json_object_bytes(payload, label="launch receipt"))
    _exact_keys(
        receipt,
        {
            "schema_version",
            "status",
            "host_alias",
            "host_hostname",
            "container_id",
            "container_name",
            "container_hostname",
            "container_image_digest",
            "runtime",
            "privileged",
            "device_requests_raw",
            "normalized_device_requests",
            "explicit_devices",
            "nvidia_visible_devices",
            "cuda_visible_devices",
            "source_git_commit",
        },
        "no-GPU launch receipt",
    )
    container_id = receipt["container_id"]
    container_name = receipt["container_name"]
    container_hostname = socket.gethostname()
    raw_requests = receipt["device_requests_raw"]
    nodes = _nvidia_device_nodes()
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or receipt["status"] != runtime["launch_receipt_status"]
        or receipt["host_alias"] != runtime["required_host_alias"]
        or receipt["host_hostname"] != runtime["required_host_hostname"]
        or not isinstance(container_id, str)
        or _CONTAINER_ID.fullmatch(container_id) is None
        or not isinstance(container_name, str)
        or _CONTAINER_NAME.fullmatch(container_name) is None
        or receipt["container_hostname"] != container_hostname
        or not container_id.startswith(container_hostname)
        or receipt["container_image_digest"]
        != runtime["required_container_image_digest"]
        or receipt["runtime"] != runtime["required_runtime"]
        or receipt["privileged"] is not False
        or raw_requests not in (None, [])
        or receipt["normalized_device_requests"] != []
        or receipt["explicit_devices"] != []
        or receipt["nvidia_visible_devices"] != "void"
        or receipt["cuda_visible_devices"] != ""
        or receipt["source_git_commit"] != source.head
        or os.environ.get("NVIDIA_VISIBLE_DEVICES") != "void"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or nodes
    ):
        raise ValueError("formal container is not the frozen no-GPU runtime")
    return {
        "receipt_sha256": sha256_bytes(payload),
        "receipt_size_bytes": len(payload),
        "container_id": container_id,
        "container_name": container_name,
        "container_hostname": container_hostname,
        "container_image_digest": receipt["container_image_digest"],
        "docker_runtime": receipt["runtime"],
        "device_requests_raw": raw_requests,
        "normalized_device_requests": [],
        "nvidia_device_nodes": nodes,
        "nvidia_visible_devices": "void",
        "cuda_visible_devices": "",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": "cpu",
    }


def _bound_state_file(
    path: str,
    *,
    label: str,
    expected_sha256: str,
    expected_size: int,
    mode: int,
) -> bytes:
    return _regular_file_bytes(
        Path(path),
        label=label,
        mode=mode,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )


def snapshot_bound_transport_state(
    contract: ScientificRepairRunnerContract,
    *,
    tag_contract: TagResolutionContract,
) -> BoundStateSnapshot:
    frozen = contract.data["tag_resolution_parent"]
    if tag_contract.sha256 != frozen["config_sha256"]:
        raise ValueError("tag-resolution parent config drifted")
    parent = validate_parent_publication_state(tag_contract)
    p1_claim = _bound_state_file(
        frozen["parent_p1_claim_path"],
        label="P1 claim",
        expected_sha256=frozen["parent_p1_claim_sha256"],
        expected_size=frozen["parent_p1_claim_size_bytes"],
        mode=frozen["file_mode"],
    )
    if p1_claim != parent.claim_bytes:
        raise ValueError("P1 claim differs from tag-resolution parent validation")
    p1_completion = Path(frozen["parent_p1_completion_path"])
    if p1_completion.exists() or p1_completion.is_symlink():
        raise ValueError("P1 completion must remain absent")
    tag_claim = _bound_state_file(
        frozen["claim_path"],
        label="tag-resolution claim",
        expected_sha256=frozen["claim_sha256"],
        expected_size=frozen["claim_size_bytes"],
        mode=frozen["file_mode"],
    )
    tag_completion = _bound_state_file(
        frozen["completion_path"],
        label="tag-resolution completion",
        expected_sha256=frozen["completion_sha256"],
        expected_size=frozen["completion_size_bytes"],
        mode=frozen["file_mode"],
    )
    claim_record = strict_pretty_json_object_bytes(
        tag_claim, label="tag-resolution claim"
    )
    completion_record = strict_pretty_json_object_bytes(
        tag_completion, label="tag-resolution completion"
    )
    remote = contract.remote
    if (
        completion_record.get("status") != frozen["completion_status"]
        or completion_record.get("claim_sha256") != frozen["claim_sha256"]
        or completion_record.get("remote_mutation_call_count") != 0
        or completion_record.get("remote_attestation", {})
        .get("identity_post", {})
        .get("immutable_resolved_commit")
        != remote["immutable_revision"]
        or claim_record.get("remote_evidence", {}).get("archive_sha256")
        != remote["archive_sha256"]
        or claim_record.get("formal_consumption", {}).get(
            "formal_label_loader_eligible"
        )
        is not False
    ):
        raise ValueError("tag-resolution seal semantic binding drifted")
    return BoundStateSnapshot(
        p1_claim=p1_claim,
        tag_claim=tag_claim,
        tag_completion=tag_completion,
    )


def snapshot_original_invalid_attempt(
    contract: ScientificRepairRunnerContract,
    *,
    p0_contract: InvalidForensicContract,
    tag_contract: TagResolutionContract,
) -> OriginalInvalidSnapshot:
    frozen = contract.data["original_invalid_attempt"]
    remote = contract.remote
    if p0_contract.sha256 != contract.data["p0_forensic_contract"]["sha256"]:
        raise ValueError("P0 forensic config differs from runner contract")
    source = collect_invalid_forensic_source(
        contract=p0_contract,
        raw_output_dir=frozen["raw_output_root"],
        external_global_ledger=frozen["external_global_ledger_path"],
        external_worker_ledgers=frozen["external_worker_ledger_paths"],
    )
    archive_path = Path(frozen["local_forensic_archive_path"])
    archive_bytes = _regular_file_bytes(
        archive_path,
        label="local invalid-forensic archive",
        expected_sha256=remote["archive_sha256"],
        expected_size=remote["archive_size_bytes"],
    )
    archive = read_invalid_forensic_archive(archive_path, contract=p0_contract)
    if (
        source.files != archive.files
        or source.manifest != archive.manifest
        or source.tree_inventory_sha256 != remote["tree_inventory_sha256"]
        or len(source.files) != remote["archive_member_count"]
        or source.manifest.get("status") != P0_ARCHIVE_STATUS
        or source.manifest.get("source_attempt", {}).get(
            "original_attempt_status"
        )
        != contract.data["p0_forensic_contract"]["producer_invalid_status"]
        or source.manifest.get("formal_consumption", {}).get(
            "formal_label_loader_eligible"
        )
        is not False
    ):
        raise ValueError("original local INVALID attempt identity drifted")
    return OriginalInvalidSnapshot(
        source_evidence=source,
        local_archive_evidence=archive,
        local_archive_bytes=archive_bytes,
        bound_state=snapshot_bound_transport_state(
            contract, tag_contract=tag_contract
        ),
    )


def require_original_snapshot_unchanged(
    before: OriginalInvalidSnapshot,
    after: OriginalInvalidSnapshot,
) -> None:
    if (
        before.source_evidence.files != after.source_evidence.files
        or before.source_evidence.manifest != after.source_evidence.manifest
        or before.source_evidence.tree_inventory_sha256
        != after.source_evidence.tree_inventory_sha256
        or before.local_archive_evidence.files
        != after.local_archive_evidence.files
        or before.local_archive_bytes != after.local_archive_bytes
        or before.bound_state != after.bound_state
    ):
        raise ValueError("original INVALID attempt or transport seals changed")


def _assert_download_path(
    root: Path,
    returned: Path,
    relative: str,
    *,
    label: str,
) -> None:
    expected = root / relative
    if not root.is_absolute() or not returned.is_absolute() or returned != expected:
        raise ValueError(f"{label} returned path drifted")
    current = root
    for component in returned.relative_to(root).parts[:-1]:
        current /= component
        metadata = current.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} contains a symlink directory")
    metadata = returned.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")


def _validate_remote_sidecar(
    payload: bytes,
    *,
    contract: ScientificRepairRunnerContract,
) -> Mapping[str, Any]:
    remote = contract.remote
    p0 = contract.data["p0_forensic_contract"]
    sidecar = dict(
        strict_pretty_json_object_bytes(payload, label="remote forensic sidecar")
    )
    _exact_keys(
        sidecar,
        {
            "schema_version",
            "protocol_id",
            "status",
            "artifact_class",
            "destination",
            "source_archive",
            "publication_source",
            "formal_consumption",
        },
        "remote forensic sidecar",
    )
    source = _mapping(sidecar["source_archive"], "sidecar source archive")
    if (
        sidecar["status"]
        != "IMMUTABLE_INVALID_FORENSIC_ARCHIVE_PUBLICATION_MANIFEST_V1"
        or sidecar["destination"].get("repo") != remote["repo"]
        or sidecar["destination"].get("archive_path") != remote["archive_path"]
        or sidecar["destination"].get("sidecar_path") != remote["sidecar_path"]
        or source.get("p0_config_sha256") != p0["sha256"]
        or source.get("archive_sha256") != remote["archive_sha256"]
        or source.get("archive_size_bytes") != remote["archive_size_bytes"]
        or source.get("tree_inventory_sha256")
        != remote["tree_inventory_sha256"]
        or source.get("member_count") != remote["archive_member_count"]
        or sidecar["formal_consumption"].get("formal_label_loader_eligible")
        is not False
        or sidecar["formal_consumption"].get("gate_training_unlocked") is not False
        or sidecar["formal_consumption"].get("producer_attempt_reclassified")
        is not False
    ):
        raise ValueError("remote forensic sidecar semantic binding drifted")
    return sidecar


def validate_downloaded_forensic_pair(
    *,
    contract: ScientificRepairRunnerContract,
    p0_contract: InvalidForensicContract,
    archive_path: str | Path,
    sidecar_path: str | Path,
    formal_hf_download: bool,
    remote_private_verified: bool,
) -> FreshForensicDownload:
    remote = contract.remote
    archive = Path(archive_path)
    sidecar = Path(sidecar_path)
    archive_bytes = _regular_file_bytes(
        archive,
        label="downloaded invalid-forensic archive",
        expected_sha256=remote["archive_sha256"],
        expected_size=remote["archive_size_bytes"],
    )
    sidecar_bytes = _regular_file_bytes(
        sidecar,
        label="downloaded invalid-forensic sidecar",
        expected_sha256=remote["sidecar_sha256"],
    )
    _validate_remote_sidecar(sidecar_bytes, contract=contract)
    evidence = read_invalid_forensic_archive(archive, contract=p0_contract)
    if (
        evidence.tree_inventory_sha256 != remote["tree_inventory_sha256"]
        or len(evidence.files) != remote["archive_member_count"]
        or evidence.manifest.get("status") != P0_ARCHIVE_STATUS
    ):
        raise ValueError("fresh immutable archive failed strict P0 readback")
    attestation = {
        "repo": remote["repo"],
        "repo_type": remote["repo_type"],
        "private": remote_private_verified,
        "immutable_revision": remote["immutable_revision"],
        "archive_path": remote["archive_path"],
        "archive_sha256": sha256_bytes(archive_bytes),
        "archive_size_bytes": len(archive_bytes),
        "sidecar_path": remote["sidecar_path"],
        "sidecar_sha256": sha256_bytes(sidecar_bytes),
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
        "member_count": len(evidence.files),
        "force_download": formal_hf_download,
        "fresh_download_directory_started_empty": formal_hf_download,
        "p0_strict_archive_readback_passed": True,
        "formal_hf_download": formal_hf_download,
        "remote_mutation_call_count": 0,
    }
    return FreshForensicDownload(
        evidence=evidence,
        archive_bytes=archive_bytes,
        sidecar_bytes=sidecar_bytes,
        attestation=attestation,
        formal_hf_download=formal_hf_download,
    )


def download_fresh_immutable_forensic(
    contract: ScientificRepairRunnerContract,
    *,
    p0_contract: InvalidForensicContract,
    hf_token: str,
    fresh_download_parent: str | Path,
) -> FreshForensicDownload:
    if (
        not isinstance(hf_token, str)
        or not hf_token
        or any(character.isspace() for character in hf_token)
    ):
        raise ValueError("HF token must be one nonempty token")
    parent = Path(fresh_download_parent)
    if not parent.is_absolute() or not parent.is_dir() or parent.is_symlink():
        raise ValueError("fresh-download parent must be an absolute real directory")
    validate_import_guard(contract)
    from huggingface_hub import HfApi, hf_hub_download

    remote = contract.remote
    api = HfApi(token=hf_token)
    pre = api.dataset_info(remote["repo"], revision=remote["immutable_revision"])
    if pre.private is not True or pre.sha != remote["immutable_revision"]:
        raise ValueError("private immutable HF source identity drifted before download")
    with tempfile.TemporaryDirectory(
        prefix="scientific-repair-invalid-forensic-",
        dir=parent,
    ) as temporary:
        directory = Path(temporary)
        if any(directory.iterdir()) or directory.is_symlink():
            raise ValueError("fresh HF download directory did not start empty")
        archive = Path(
            hf_hub_download(
                repo_id=remote["repo"],
                filename=remote["archive_path"],
                repo_type=remote["repo_type"],
                revision=remote["immutable_revision"],
                local_dir=directory,
                force_download=True,
                token=hf_token,
            )
        )
        sidecar = Path(
            hf_hub_download(
                repo_id=remote["repo"],
                filename=remote["sidecar_path"],
                repo_type=remote["repo_type"],
                revision=remote["immutable_revision"],
                local_dir=directory,
                force_download=True,
                token=hf_token,
            )
        )
        _assert_download_path(
            directory,
            archive,
            remote["archive_path"],
            label="downloaded archive",
        )
        _assert_download_path(
            directory,
            sidecar,
            remote["sidecar_path"],
            label="downloaded sidecar",
        )
        downloaded = validate_downloaded_forensic_pair(
            contract=contract,
            p0_contract=p0_contract,
            archive_path=archive,
            sidecar_path=sidecar,
            formal_hf_download=True,
            remote_private_verified=True,
        )
    post = api.dataset_info(remote["repo"], revision=remote["immutable_revision"])
    if post.private is not True or post.sha != remote["immutable_revision"]:
        raise ValueError("private immutable HF source identity drifted after download")
    validate_import_guard(contract)
    return downloaded


def validate_external_replay_inputs(
    contract: ScientificRepairRunnerContract,
) -> Mapping[str, Any]:
    frozen = contract.data["external_replay_inputs"]
    parent = Path(frozen["parent_substrate_archive_path"])
    parent_bytes = _regular_file_bytes(
        parent,
        label="parent substrate archive",
        expected_sha256=frozen["parent_substrate_archive_sha256"],
        expected_size=frozen["parent_substrate_archive_size_bytes"],
    )
    derived = Path(frozen["derived_artifact_root"])
    if not derived.is_absolute() or not derived.is_dir() or derived.is_symlink():
        raise ValueError("derived artifact root must be one real absolute directory")
    from causalcache.data.guiodyssey_restoration_v2_expansion import (
        artifact_tree_identity,
    )

    tree = artifact_tree_identity(derived)
    if tree.get("artifact_tree_sha256") != frozen["derived_artifact_tree_sha256"]:
        raise ValueError("derived artifact tree SHA256 drifted")
    ocr = contract.repository_root / frozen["ocr_backend_config_repository_path"]
    ocr_bytes = _regular_file_bytes(
        ocr,
        label="OCR backend config",
        expected_sha256=frozen["ocr_backend_config_sha256"],
        expected_size=frozen["ocr_backend_config_size_bytes"],
    )
    return {
        "parent_substrate_archive_path": str(parent.resolve()),
        "parent_substrate_archive_sha256": sha256_bytes(parent_bytes),
        "parent_substrate_archive_size_bytes": len(parent_bytes),
        "parent_substrate_tree_inventory_sha256": frozen[
            "parent_substrate_tree_inventory_sha256"
        ],
        "derived_artifact_root": str(derived.resolve()),
        "derived_artifact_repo": frozen["derived_artifact_repo"],
        "derived_artifact_revision": frozen["derived_artifact_revision"],
        "derived_artifact_tree_sha256": tree["artifact_tree_sha256"],
        "derived_artifact_file_count": len(tree["files"]),
        "ocr_backend_config_path": str(ocr.resolve()),
        "ocr_backend_config_sha256": sha256_bytes(ocr_bytes),
        "ocr_backend_config_size_bytes": len(ocr_bytes),
    }


def _source_record(source: SourceIdentity) -> Mapping[str, Any]:
    return {
        "git_commit": source.head,
        "origin_main_git_commit": source.origin_main,
        "externally_resolved_remote_main_git_commit": source.remote_main,
        "branch": source.branch,
        "origin_url": source.origin_url,
        "source_path_count": len(source.inventory),
        "source_inventory_sha256": source.inventory_sha256,
        "loaded_project_module_count": len(source.loaded_module_inventory),
        "loaded_project_module_inventory_sha256": (
            source.loaded_module_inventory_sha256
        ),
    }


def _original_record(snapshot: OriginalInvalidSnapshot) -> Mapping[str, Any]:
    evidence = snapshot.source_evidence
    return {
        "producer_attempt_status": evidence.manifest["source_attempt"][
            "original_attempt_status"
        ],
        "forensic_archive_status": evidence.manifest["status"],
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
        "member_count": len(evidence.files),
        "local_archive_sha256": sha256_bytes(snapshot.local_archive_bytes),
        "local_archive_size_bytes": len(snapshot.local_archive_bytes),
        "p1_claim_sha256": sha256_bytes(snapshot.bound_state.p1_claim),
        "tag_resolution_claim_sha256": sha256_bytes(
            snapshot.bound_state.tag_claim
        ),
        "tag_resolution_completion_sha256": sha256_bytes(
            snapshot.bound_state.tag_completion
        ),
        "parent_p1_completion_observed_absent": True,
        "producer_reclassified": False,
        "original_formal_label_loader_eligible": False,
    }


def _operation_counts() -> Mapping[str, int]:
    return {
        "nvidia_device_query_count": 0,
        "cuda_operation_count": 0,
        "model_load_count": 0,
        "policy_forward_count": 0,
        "policy_vision_forward_count": 0,
        "language_model_forward_count": 0,
        "teacher_forward_count": 0,
        "kl_measurement_count": 0,
        "generation_count": 0,
        "gate_model_forward_count": 0,
        "gate_training_example_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
        "confirm_state_access_count": 0,
        "sealed_test_state_access_count": 0,
        "remote_mutation_call_count": 0,
    }


def build_deterministic_claim(
    contract: ScientificRepairRunnerContract,
    *,
    source: SourceIdentity,
    runtime: Mapping[str, Any],
    original: OriginalInvalidSnapshot,
    fresh: FreshForensicDownload,
    external_inputs: Mapping[str, Any],
    argv: Sequence[str],
) -> Mapping[str, Any]:
    if fresh.formal_hf_download is not True:
        raise ValueError("formal claim requires an actual immutable HF download")
    normalized_argv = list(argv)
    if len(normalized_argv) >= 2 and normalized_argv[1] in {"run", "validate"}:
        normalized_argv[1] = "formal-replay"
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": contract.output["claim_status"],
        "runner_config_sha256": contract.sha256,
        "source": dict(_source_record(source)),
        "runtime": dict(runtime),
        "original_invalid_attempt": dict(_original_record(original)),
        "fresh_immutable_forensic": dict(fresh.attestation),
        "external_replay_inputs": dict(external_inputs),
        "canonical_output_archive": contract.output["canonical_archive_path"],
        "completion_path": contract.output["completion_path"],
        "normalized_formal_argv": normalized_argv,
        "repair_runtime_operation_counts": dict(_operation_counts()),
        "formal_consumption": dict(contract.data["formal_consumption"]),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_or_identical_state(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    mode: int,
) -> tuple[bytes, str, bool]:
    final = Path(path)
    if not final.parent.is_dir() or final.parent.is_symlink():
        raise ValueError("state parent must preexist as a real directory")
    payload = _pretty_json_bytes(record)
    if final.exists() or final.is_symlink():
        existing = _regular_file_bytes(final, label="existing state", mode=mode)
        if existing != payload:
            raise ValueError("existing state is not byte-identical")
        return existing, sha256_bytes(existing), False
    temporary = final.with_name(
        f".{final.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, mode)
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("state write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, final)
        _fsync_directory(final.parent)
        return payload, sha256_bytes(payload), True
    except FileExistsError:
        existing = _regular_file_bytes(final, label="raced state", mode=mode)
        if existing != payload:
            raise ValueError("concurrent state differs from deterministic bytes")
        return existing, sha256_bytes(existing), False
    finally:
        temporary.unlink(missing_ok=True)


def inspect_repair_state(
    contract: ScientificRepairRunnerContract,
) -> Mapping[str, bool]:
    output = Path(contract.output["canonical_archive_path"])
    claim = Path(contract.output["claim_path"])
    completion = Path(contract.output["completion_path"])
    state = {
        "claim_exists": claim.exists() or claim.is_symlink(),
        "output_exists": output.exists() or output.is_symlink(),
        "completion_exists": completion.exists() or completion.is_symlink(),
    }
    if state["completion_exists"] and not state["output_exists"]:
        raise ValueError("completion-without-output is permanently invalid")
    if state["output_exists"] and not state["claim_exists"]:
        raise ValueError("orphan repaired output exists without its claim")
    if state["completion_exists"] and not state["claim_exists"]:
        raise ValueError("orphan completion exists without its claim")
    return state


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _parse_canonical_jsonl(
    payload: bytes,
    *,
    label: str,
    expected_count: int,
) -> tuple[Mapping[str, Any], ...]:
    lines = payload.splitlines(keepends=True)
    if len(lines) != expected_count or any(not line.endswith(b"\n") for line in lines):
        raise ValueError(f"{label} line count or LF termination drifted")
    records: list[Mapping[str, Any]] = []
    for index, line in enumerate(lines):
        raw = line[:-1]
        record = _strict_json_object(raw, label=f"{label} row {index}")
        if canonical_json_bytes(record) != raw:
            raise ValueError(f"{label} row {index} is not canonical JSON")
        records.append(record)
    return tuple(records)


def _inventory(files: Mapping[str, bytes]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "path": name,
            "sha256": sha256_bytes(files[name]),
            "size_bytes": len(files[name]),
        }
        for name in sorted(files)
    )


def deterministic_repaired_ustar_bytes(
    files: Mapping[str, bytes],
    *,
    contract: ScientificRepairRunnerContract,
) -> bytes:
    expected = set(contract.output["exact_members"])
    if set(files) != expected:
        raise ValueError("repaired archive member inventory drifted")
    prefix = contract.output["archive_member_prefix"]
    destination = io.BytesIO()
    with tarfile.open(
        fileobj=destination,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for relative in sorted(files):
            _safe_relative_path(relative, "repaired archive member")
            payload = files[relative]
            if not isinstance(payload, bytes):
                raise TypeError("repaired archive member payload must be bytes")
            info = tarfile.TarInfo(f"{prefix}/{relative}")
            info.size = len(payload)
            info.mode = contract.output["regular_file_mode"]
            info.uid = contract.output["uid"]
            info.gid = contract.output["gid"]
            info.uname = ""
            info.gname = ""
            info.mtime = contract.output["mtime"]
            archive.addfile(info, io.BytesIO(payload))
    return destination.getvalue()


def _validate_repaired_payload_files(
    files: Mapping[str, bytes],
    *,
    contract: ScientificRepairRunnerContract,
    require_formal: bool,
) -> None:
    expected = set(contract.output["exact_members"])
    if set(files) != expected:
        raise ValueError("repaired payload exact-four inventory drifted")
    raw = _parse_canonical_jsonl(
        files["raw_states.jsonl"],
        label="raw states",
        expected_count=contract.data["scientific_payload"]["raw_state_count"],
    )
    derived = _parse_canonical_jsonl(
        files["derived_labels.jsonl"],
        label="derived labels",
        expected_count=contract.data["scientific_payload"]["derived_state_count"],
    )
    audit = strict_pretty_json_object_bytes(files["audit.json"], label="repair audit")
    manifest = strict_pretty_json_object_bytes(
        files["manifest.json"], label="repair manifest"
    )
    if require_formal and audit.get("status") == TEST_ONLY_STATUS:
        raise ValueError("test-only repair payload cannot satisfy formal reader")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "status",
            "runner_config_sha256",
            "source_git_commit",
            "raw_state_count",
            "derived_state_count",
            "payload_inventory",
            "scientific_payload_sha256",
            "original_attempt_remains_invalid",
            "producer_reclassified",
            "formal_consumption",
        },
        "repair manifest",
    )
    expected_audit_keys = (
        {
            "schema_version",
            "protocol_id",
            "status",
            "runner_config_sha256",
            "formal_scientific_repair_pass",
            "source",
            "runtime",
            "import_guard",
            "original_invalid_attempt",
            "fresh_immutable_forensic",
            "external_replay_inputs",
            "claim_sha256",
            "repair_runtime_operation_counts",
            "archived_producer_operation_counts",
            "scientific_repair_core_report",
            "scientific_summary",
            "formal_consumption",
        }
        if audit.get("status") == AUDIT_STATUS
        else {
            "schema_version",
            "protocol_id",
            "status",
            "runner_config_sha256",
            "formal_scientific_repair_pass",
            "original_invalid_attempt",
            "fresh_immutable_forensic",
            "scientific_repair_core_report",
            "formal_consumption",
        }
    )
    _exact_keys(audit, expected_audit_keys, "repair audit")
    payload_inventory = _inventory(
        {name: files[name] for name in ("audit.json", "derived_labels.jsonl", "raw_states.jsonl")}
    )
    expected_scientific_sha = sha256_bytes(
        canonical_json_bytes(
            {
                "raw_states_sha256": sha256_bytes(files["raw_states.jsonl"]),
                "derived_labels_sha256": sha256_bytes(
                    files["derived_labels.jsonl"]
                ),
                "core_report_sha256": sha256_bytes(
                    canonical_json_bytes(audit["scientific_repair_core_report"])
                ),
            }
        )
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != MANIFEST_STATUS
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("runner_config_sha256") != contract.sha256
        or manifest.get("payload_inventory") != list(payload_inventory)
        or manifest.get("scientific_payload_sha256") != expected_scientific_sha
        or manifest.get("raw_state_count") != len(raw)
        or manifest.get("derived_state_count") != len(derived)
        or manifest.get("original_attempt_remains_invalid") is not True
        or manifest.get("producer_reclassified") is not False
        or manifest.get("formal_consumption")
        != contract.data["formal_consumption"]
        or audit.get("schema_version") != SCHEMA_VERSION
        or audit.get("status") not in {AUDIT_STATUS, TEST_ONLY_STATUS}
        or audit.get("protocol_id") != PROTOCOL_ID
        or audit.get("runner_config_sha256") != contract.sha256
        or audit.get("original_invalid_attempt", {}).get("producer_reclassified")
        is not False
        or audit.get("formal_consumption") != contract.data["formal_consumption"]
        or (
            require_formal
            and audit.get("repair_runtime_operation_counts") != _operation_counts()
        )
    ):
        raise ValueError("repaired payload manifest or audit binding drifted")
    if not require_formal:
        return

    source = _mapping(audit["source"], "repair audit source")
    runtime = _mapping(audit["runtime"], "repair audit runtime")
    import_guard = _mapping(audit["import_guard"], "repair audit import guard")
    original = _mapping(
        audit["original_invalid_attempt"], "repair audit original attempt"
    )
    fresh = _mapping(
        audit["fresh_immutable_forensic"], "repair audit fresh forensic"
    )
    external = _mapping(
        audit["external_replay_inputs"], "repair audit external inputs"
    )
    report = _mapping(
        audit["scientific_repair_core_report"], "scientific repair core report"
    )
    summary = _mapping(audit["scientific_summary"], "scientific summary")
    _exact_keys(
        source,
        {
            "git_commit",
            "origin_main_git_commit",
            "externally_resolved_remote_main_git_commit",
            "branch",
            "origin_url",
            "source_path_count",
            "source_inventory_sha256",
            "loaded_project_module_count",
            "loaded_project_module_inventory_sha256",
        },
        "repair audit source",
    )
    _exact_keys(
        runtime,
        {
            "receipt_sha256",
            "receipt_size_bytes",
            "container_id",
            "container_name",
            "container_hostname",
            "container_image_digest",
            "docker_runtime",
            "device_requests_raw",
            "normalized_device_requests",
            "nvidia_device_nodes",
            "nvidia_visible_devices",
            "cuda_visible_devices",
            "python_version",
            "platform",
            "device",
            "import_guard",
        },
        "repair audit runtime",
    )
    _exact_keys(
        import_guard,
        {
            "guard_marker",
            "blocked_roots",
            "forbidden_modules_imported",
            "pil_allowed",
        },
        "repair audit import guard",
    )
    _exact_keys(
        report,
        {
            "schema_version",
            "protocol_id",
            "status",
            "formal_scientific_repair_pass",
            "original_attempt",
            "append_only_execution_log",
            "legacy_validator_negative_control",
            "monitor_validation",
            "raw_distance_reduction",
            "independent_stdlib_math_audit",
            "external_input_replay",
            "forensic_evidence",
            "execution_scope",
        },
        "scientific repair core report",
    )
    remote = contract.remote
    planned = contract.data["scientific_payload"]
    frozen_external = contract.data["external_replay_inputs"]
    reduction_report = _mapping(
        report["raw_distance_reduction"], "core raw-distance reduction"
    )
    reduction_counts = _mapping(
        reduction_report.get("counts"), "core reduction counts"
    )
    reduction_summary = _mapping(
        reduction_report.get("summary"), "core reduction summary"
    )
    run_hashes = {record.get("run_contract_sha256") for record in raw}
    if len(run_hashes) != 1:
        raise ValueError("formal raw states do not bind one run contract")
    run_contract_sha256 = _sha256(
        next(iter(run_hashes)), "formal raw-state run contract"
    )
    expected_states = frozen_v1.validate_state_projections(
        [record.get("state") for record in raw]
    )
    rebuilt = frozen_v1.reduce_raw_distance_states(
        raw,
        expected_states=expected_states,
        run_contract_sha256=run_contract_sha256,
    )
    math_audit = audit_normalized_raw_states(raw)
    math_comparison = compare_with_reducer_output(math_audit, rebuilt)
    expected_summary = {
        "counts": dict(rebuilt["counts"]),
        "summary": dict(rebuilt["summary"]),
        "derived_payload_sha256": rebuilt["derived_payload_sha256"],
        "core_report_sha256": sha256_bytes(canonical_json_bytes(report)),
    }
    expected_counts = {
        "trajectory_count": planned["trajectory_count"],
        "state_count": planned["raw_state_count"],
        "raw_distance_row_count": planned["raw_distance_row_count"],
        "deployment_conditional_edge_count": planned[
            "deployment_conditional_edge_count"
        ],
        "full_hypercube_edge_count": planned["full_hypercube_edge_count"],
        "pair_interaction_count": planned["pair_interaction_count"],
        "exact_permutation_attribution_count": planned[
            "exact_permutation_attribution_count"
        ],
        "primary_exact_subset_oracle_count": planned[
            "primary_exact_subset_oracle_count"
        ],
        "teacher_forward_count": planned["archived_teacher_forward_count"],
        "kl_measurement_count": planned["archived_kl_measurement_count"],
    }
    if (
        audit.get("status") != AUDIT_STATUS
        or audit.get("formal_scientific_repair_pass") is not True
        or manifest.get("source_git_commit") != source.get("git_commit")
        or source.get("git_commit") != source.get("origin_main_git_commit")
        or source.get("git_commit")
        != source.get("externally_resolved_remote_main_git_commit")
        or _GIT_SHA.fullmatch(str(source.get("git_commit"))) is None
        or source.get("branch") != contract.data["git_source_contract"]["branch"]
        or source.get("origin_url")
        != contract.data["git_source_contract"]["origin_url"]
        or source.get("source_path_count")
        != len(contract.data["git_source_contract"]["required_source_paths"])
        or _SHA256.fullmatch(str(source.get("source_inventory_sha256"))) is None
        or not isinstance(source.get("loaded_project_module_count"), int)
        or source.get("loaded_project_module_count", 0) <= 0
        or _SHA256.fullmatch(
            str(source.get("loaded_project_module_inventory_sha256"))
        )
        is None
        or _SHA256.fullmatch(str(runtime.get("receipt_sha256"))) is None
        or not isinstance(runtime.get("receipt_size_bytes"), int)
        or runtime.get("receipt_size_bytes", 0) <= 0
        or runtime.get("container_image_digest")
        != contract.runtime["required_container_image_digest"]
        or runtime.get("docker_runtime") != contract.runtime["required_runtime"]
        or runtime.get("device_requests_raw") not in (None, [])
        or runtime.get("normalized_device_requests") != []
        or runtime.get("nvidia_device_nodes") != []
        or runtime.get("nvidia_visible_devices") != "void"
        or runtime.get("cuda_visible_devices") != ""
        or runtime.get("device") != "cpu"
        or runtime.get("import_guard") != import_guard
        or import_guard.get("guard_marker") != IMPORT_GUARD_MARKER
        or import_guard.get("blocked_roots") != list(_guard_roots(contract))
        or import_guard.get("forbidden_modules_imported") != []
        or import_guard.get("pil_allowed") is not True
        or original.get("tree_inventory_sha256")
        != remote["tree_inventory_sha256"]
        or original.get("member_count") != remote["archive_member_count"]
        or original.get("local_archive_sha256") != remote["archive_sha256"]
        or original.get("local_archive_size_bytes") != remote["archive_size_bytes"]
        or original.get("producer_reclassified") is not False
        or original.get("original_formal_label_loader_eligible") is not False
        or fresh.get("repo") != remote["repo"]
        or fresh.get("private") is not True
        or fresh.get("immutable_revision") != remote["immutable_revision"]
        or fresh.get("archive_sha256") != remote["archive_sha256"]
        or fresh.get("archive_size_bytes") != remote["archive_size_bytes"]
        or fresh.get("sidecar_sha256") != remote["sidecar_sha256"]
        or fresh.get("tree_inventory_sha256")
        != remote["tree_inventory_sha256"]
        or fresh.get("member_count") != remote["archive_member_count"]
        or fresh.get("formal_hf_download") is not True
        or fresh.get("remote_mutation_call_count") != 0
        or external.get("parent_substrate_archive_sha256")
        != frozen_external["parent_substrate_archive_sha256"]
        or external.get("parent_substrate_archive_size_bytes")
        != frozen_external["parent_substrate_archive_size_bytes"]
        or external.get("derived_artifact_repo")
        != frozen_external["derived_artifact_repo"]
        or external.get("derived_artifact_revision")
        != frozen_external["derived_artifact_revision"]
        or external.get("derived_artifact_tree_sha256")
        != frozen_external["derived_artifact_tree_sha256"]
        or external.get("ocr_backend_config_sha256")
        != frozen_external["ocr_backend_config_sha256"]
        or _SHA256.fullmatch(str(audit.get("claim_sha256"))) is None
        or audit.get("repair_runtime_operation_counts") != _operation_counts()
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("protocol_id") != CORE_PROTOCOL_ID
        or report.get("status") != CORE_PASS_STATUS
        or report.get("formal_scientific_repair_pass") is not True
        or report.get("external_input_replay", {}).get("attestation_tier")
        != FORMAL_EXTERNAL_REPLAY_TIER
        or report.get("forensic_evidence", {}).get("forensic_contract_sha256")
        != contract.data["p0_forensic_contract"]["sha256"]
        or report.get("forensic_evidence", {}).get("tree_inventory_sha256")
        != remote["tree_inventory_sha256"]
        or report.get("independent_stdlib_math_audit") != math_comparison
        or list(derived) != rebuilt["states"]
        or dict(reduction_counts) != rebuilt["counts"]
        or dict(reduction_summary) != rebuilt["summary"]
        or reduction_report.get("derived_payload_sha256")
        != rebuilt["derived_payload_sha256"]
        or summary != expected_summary
        or any(rebuilt["counts"].get(key) != value for key, value in expected_counts.items())
        or audit.get("archived_producer_operation_counts")
        != {
            "teacher_forward_count": planned["archived_teacher_forward_count"],
            "kl_measurement_count": planned["archived_kl_measurement_count"],
            "scalar_host_transfer_count": rebuilt["counts"][
                "scalar_host_transfer_count"
            ],
        }
    ):
        raise ValueError("formal repaired payload deep validation failed")


def repaired_archive_from_files(
    files: Mapping[str, bytes],
    *,
    contract: ScientificRepairRunnerContract,
    require_formal: bool,
) -> RepairedArchive:
    normalized = dict(files)
    _validate_repaired_payload_files(
        normalized, contract=contract, require_formal=require_formal
    )
    archive = deterministic_repaired_ustar_bytes(normalized, contract=contract)
    inventory = _inventory(normalized)
    return RepairedArchive(
        files=normalized,
        archive_bytes=archive,
        archive_sha256=sha256_bytes(archive),
        archive_size_bytes=len(archive),
        inventory=inventory,
        tree_inventory_sha256=sha256_bytes(canonical_json_bytes(inventory)),
    )


def _read_repaired_archive_payload(
    payload: bytes,
    *,
    contract: ScientificRepairRunnerContract,
    require_formal: bool,
    expected_files: Mapping[str, bytes] | None,
) -> RepairedArchive:
    files: dict[str, bytes] = {}
    prefix = f"{contract.output['archive_member_prefix']}/"
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != sorted(
                member.name for member in members
            ):
                raise ValueError("repaired USTAR member order drifted")
            for member in members:
                if (
                    not member.isfile()
                    or not member.name.startswith(prefix)
                    or member.mode != contract.output["regular_file_mode"]
                    or member.uid != contract.output["uid"]
                    or member.gid != contract.output["gid"]
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != contract.output["mtime"]
                    or member.pax_headers
                ):
                    raise ValueError("repaired USTAR metadata drifted")
                relative = _safe_relative_path(
                    member.name[len(prefix) :], "repaired USTAR path"
                )
                if relative in files:
                    raise ValueError("repaired USTAR contains duplicate members")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("repaired USTAR member is unreadable")
                member_payload = extracted.read()
                if len(member_payload) != member.size:
                    raise ValueError("repaired USTAR member size drifted")
                files[relative] = member_payload
    except tarfile.TarError as error:
        raise ValueError("repaired scientific archive is not readable USTAR") from error
    rebuilt = repaired_archive_from_files(
        files, contract=contract, require_formal=require_formal
    )
    if rebuilt.archive_bytes != payload:
        raise ValueError("repaired scientific archive is not deterministic USTAR")
    if require_formal:
        if expected_files is None:
            raise ValueError("formal archive readback requires recomputed expected files")
        if rebuilt.files != dict(expected_files):
            raise ValueError("formal archive differs from recomputed scientific payload")
    return rebuilt


def read_repaired_archive(
    path: str | Path,
    *,
    contract: ScientificRepairRunnerContract,
    require_formal: bool = True,
    expected_files: Mapping[str, bytes] | None = None,
) -> RepairedArchive:
    candidate = Path(path)
    payload = _regular_file_bytes(candidate, label="repaired scientific archive")
    return _read_repaired_archive_payload(
        payload,
        contract=contract,
        require_formal=require_formal,
        expected_files=expected_files,
    )


def _stat_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _descriptor_bytes(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def open_stable_repaired_archive(
    path: str | Path,
    *,
    contract: ScientificRepairRunnerContract,
    expected_files: Mapping[str, bytes],
) -> StableArchiveHandle:
    candidate = Path(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise ValueError("canonical repaired archive cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        payload = _descriptor_bytes(descriptor)
        after = os.fstat(descriptor)
        path_metadata = candidate.lstat()
        fingerprint = _stat_fingerprint(before)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != contract.output["regular_file_mode"]
            or fingerprint != _stat_fingerprint(after)
            or before.st_dev != path_metadata.st_dev
            or before.st_ino != path_metadata.st_ino
            or len(payload) != before.st_size
        ):
            raise ValueError("canonical repaired archive changed during stable read")
        archive = _read_repaired_archive_payload(
            payload,
            contract=contract,
            require_formal=True,
            expected_files=expected_files,
        )
        return StableArchiveHandle(
            descriptor=descriptor,
            path=candidate,
            stat_fingerprint=fingerprint,
            payload=payload,
            archive=archive,
        )
    except BaseException:
        os.close(descriptor)
        raise


def revalidate_stable_repaired_archive(handle: StableArchiveHandle) -> None:
    before = os.fstat(handle.descriptor)
    payload = _descriptor_bytes(handle.descriptor)
    after = os.fstat(handle.descriptor)
    path_metadata = handle.path.lstat()
    if (
        _stat_fingerprint(before) != handle.stat_fingerprint
        or _stat_fingerprint(after) != handle.stat_fingerprint
        or before.st_dev != path_metadata.st_dev
        or before.st_ino != path_metadata.st_ino
        or payload != handle.payload
    ):
        raise ValueError("canonical repaired archive changed before completion")


def _scientific_summary(
    validation: LedgerNeutralScientificValidation,
) -> Mapping[str, Any]:
    report = validation.report
    reduction = validation.reduction
    return {
        "counts": dict(reduction["counts"]),
        "summary": dict(reduction["summary"]),
        "derived_payload_sha256": reduction["derived_payload_sha256"],
        "core_report_sha256": sha256_bytes(canonical_json_bytes(report)),
    }


def build_formal_repaired_archive(
    contract: ScientificRepairRunnerContract,
    *,
    validation: LedgerNeutralScientificValidation,
    source: SourceIdentity,
    runtime: Mapping[str, Any],
    original: OriginalInvalidSnapshot,
    fresh: FreshForensicDownload,
    external_inputs: Mapping[str, Any],
    claim_sha256: str,
) -> RepairedArchive:
    report = validation.report
    counts = validation.reduction.get("counts")
    planned = contract.data["scientific_payload"]
    if (
        report.get("status") != CORE_PASS_STATUS
        or report.get("formal_scientific_repair_pass") is not True
        or fresh.formal_hf_download is not True
        or fresh.attestation.get("formal_hf_download") is not True
        or report.get("external_input_replay", {}).get("attestation_tier")
        != FORMAL_EXTERNAL_REPLAY_TIER
        or counts.get("trajectory_count") != planned["trajectory_count"]
        or counts.get("state_count") != planned["raw_state_count"]
        or counts.get("raw_distance_row_count") != planned["raw_distance_row_count"]
        or counts.get("deployment_conditional_edge_count")
        != planned["deployment_conditional_edge_count"]
        or counts.get("full_hypercube_edge_count")
        != planned["full_hypercube_edge_count"]
        or counts.get("pair_interaction_count")
        != planned["pair_interaction_count"]
        or counts.get("exact_permutation_attribution_count")
        != planned["exact_permutation_attribution_count"]
        or counts.get("primary_exact_subset_oracle_count")
        != planned["primary_exact_subset_oracle_count"]
        or counts.get("teacher_forward_count")
        != planned["archived_teacher_forward_count"]
        or counts.get("kl_measurement_count")
        != planned["archived_kl_measurement_count"]
    ):
        raise ValueError("formal scientific repair result or denominator drifted")
    raw_states = _canonical_jsonl(validation.normalized_state_records)
    derived_states = _canonical_jsonl(validation.reduction["states"])
    audit = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": AUDIT_STATUS,
        "runner_config_sha256": contract.sha256,
        "formal_scientific_repair_pass": True,
        "source": dict(_source_record(source)),
        "runtime": dict(runtime),
        "import_guard": dict(validate_import_guard(contract)),
        "original_invalid_attempt": dict(_original_record(original)),
        "fresh_immutable_forensic": dict(fresh.attestation),
        "external_replay_inputs": dict(external_inputs),
        "claim_sha256": _sha256(claim_sha256, "claim SHA256"),
        "repair_runtime_operation_counts": dict(_operation_counts()),
        "archived_producer_operation_counts": {
            "teacher_forward_count": counts["teacher_forward_count"],
            "kl_measurement_count": counts["kl_measurement_count"],
            "scalar_host_transfer_count": counts["scalar_host_transfer_count"],
        },
        "scientific_repair_core_report": dict(report),
        "scientific_summary": dict(_scientific_summary(validation)),
        "formal_consumption": dict(contract.data["formal_consumption"]),
    }
    files: dict[str, bytes] = {
        "raw_states.jsonl": raw_states,
        "derived_labels.jsonl": derived_states,
        "audit.json": _pretty_json_bytes(audit),
    }
    inventory = _inventory(files)
    scientific_payload_sha = sha256_bytes(
        canonical_json_bytes(
            {
                "raw_states_sha256": sha256_bytes(raw_states),
                "derived_labels_sha256": sha256_bytes(derived_states),
                "core_report_sha256": sha256_bytes(canonical_json_bytes(report)),
            }
        )
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": MANIFEST_STATUS,
        "runner_config_sha256": contract.sha256,
        "source_git_commit": source.head,
        "raw_state_count": len(validation.normalized_state_records),
        "derived_state_count": len(validation.reduction["states"]),
        "payload_inventory": list(inventory),
        "scientific_payload_sha256": scientific_payload_sha,
        "original_attempt_remains_invalid": True,
        "producer_reclassified": False,
        "formal_consumption": dict(contract.data["formal_consumption"]),
    }
    files["manifest.json"] = _pretty_json_bytes(manifest)
    return repaired_archive_from_files(files, contract=contract, require_formal=True)


def build_test_only_repaired_archive(
    contract: ScientificRepairRunnerContract,
    *,
    core_report: Mapping[str, Any],
    raw_states: Sequence[Mapping[str, Any]],
    derived_states: Sequence[Mapping[str, Any]],
) -> RepairedArchive:
    raw_payload = _canonical_jsonl(raw_states)
    derived_payload = _canonical_jsonl(derived_states)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": TEST_ONLY_STATUS,
        "runner_config_sha256": contract.sha256,
        "formal_scientific_repair_pass": False,
        "original_invalid_attempt": {"producer_reclassified": False},
        "fresh_immutable_forensic": {"formal_hf_download": False},
        "scientific_repair_core_report": dict(core_report),
        "formal_consumption": dict(contract.data["formal_consumption"]),
    }
    files: dict[str, bytes] = {
        "raw_states.jsonl": raw_payload,
        "derived_labels.jsonl": derived_payload,
        "audit.json": _pretty_json_bytes(audit),
    }
    inventory = _inventory(files)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": MANIFEST_STATUS,
        "runner_config_sha256": contract.sha256,
        "source_git_commit": "0" * 40,
        "raw_state_count": len(raw_states),
        "derived_state_count": len(derived_states),
        "payload_inventory": list(inventory),
        "scientific_payload_sha256": sha256_bytes(
            canonical_json_bytes(
                {
                    "raw_states_sha256": sha256_bytes(raw_payload),
                    "derived_labels_sha256": sha256_bytes(derived_payload),
                    "core_report_sha256": sha256_bytes(
                        canonical_json_bytes(core_report)
                    ),
                }
            )
        ),
        "original_attempt_remains_invalid": True,
        "producer_reclassified": False,
        "formal_consumption": dict(contract.data["formal_consumption"]),
    }
    files["manifest.json"] = _pretty_json_bytes(manifest)
    return repaired_archive_from_files(files, contract=contract, require_formal=False)


def atomic_publish_repaired_archive(
    path: str | Path,
    archive: RepairedArchive,
    *,
    contract: ScientificRepairRunnerContract,
    pre_publish_check: Callable[[], None],
) -> None:
    final = Path(path)
    if final.resolve() != Path(contract.output["canonical_archive_path"]).resolve():
        raise ValueError("repaired archive output path drifted")
    if not final.parent.is_dir() or final.parent.is_symlink():
        raise ValueError("repaired archive parent must preexist")
    if final.exists() or final.is_symlink():
        raise FileExistsError("repaired archive overwrite is forbidden")
    temporary = final.with_name(
        f".{final.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, 0o644)
    try:
        view = memoryview(archive.archive_bytes)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("repaired archive write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        staged = read_repaired_archive(
            temporary,
            contract=contract,
            require_formal=True,
            expected_files=archive.files,
        )
        if staged.archive_bytes != archive.archive_bytes:
            raise ValueError("staged repaired archive differs before publication")
        pre_publish_check()
        os.link(temporary, final)
        _fsync_directory(final.parent)
    except FileExistsError:
        raise FileExistsError("concurrent repaired archive publication won")
    finally:
        temporary.unlink(missing_ok=True)


def build_completion_record(
    contract: ScientificRepairRunnerContract,
    *,
    claim_sha256: str,
    archive: RepairedArchive,
    source: SourceIdentity,
) -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": contract.output["completion_status"],
        "runner_config_sha256": contract.sha256,
        "claim_sha256": _sha256(claim_sha256, "claim SHA256"),
        "source_git_commit": source.head,
        "archive_path": contract.output["canonical_archive_path"],
        "archive_sha256": archive.archive_sha256,
        "archive_size_bytes": archive.archive_size_bytes,
        "member_count": len(archive.inventory),
        "member_inventory": list(archive.inventory),
        "tree_inventory_sha256": archive.tree_inventory_sha256,
        "completion_created_after_strict_output_readback": True,
        "original_attempt_reclassified": False,
        "repair_runtime_operation_counts": dict(_operation_counts()),
        "formal_consumption": dict(contract.data["formal_consumption"]),
    }


def validate_completion_record(
    contract: ScientificRepairRunnerContract,
    *,
    claim_sha256: str,
    archive: RepairedArchive,
    source: SourceIdentity,
) -> Mapping[str, Any]:
    path = Path(contract.output["completion_path"])
    payload = _regular_file_bytes(
        path,
        label="scientific repair completion",
        mode=contract.output["state_file_mode"],
    )
    record = strict_pretty_json_object_bytes(payload, label="repair completion")
    expected = build_completion_record(
        contract,
        claim_sha256=claim_sha256,
        archive=archive,
        source=source,
    )
    if record != expected or _pretty_json_bytes(record) != payload:
        raise ValueError("scientific repair completion binding drifted")
    return record


def atomic_publish_completion_last(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    contract: ScientificRepairRunnerContract,
    stable_archive: StableArchiveHandle,
    pre_publish_check: Callable[[], None],
) -> tuple[str, bool]:
    final = Path(path)
    if final.resolve() != Path(contract.output["completion_path"]).resolve():
        raise ValueError("scientific repair completion path drifted")
    if not final.parent.is_dir() or final.parent.is_symlink():
        raise ValueError("completion parent must preexist as a real directory")
    if final.exists() or final.is_symlink():
        raise FileExistsError("scientific repair completion overwrite is forbidden")
    mode = contract.output["state_file_mode"]
    payload = _pretty_json_bytes(record)
    temporary = final.with_name(
        f".{final.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, mode)
    published = False
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("completion write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        staged, staged_sha = _read_deterministic_state(
            temporary,
            expected=record,
            mode=mode,
            label="staged scientific repair completion",
        )
        if staged != payload:
            raise ValueError("staged completion bytes drifted")
        pre_publish_check()
        revalidate_stable_repaired_archive(stable_archive)
        os.link(temporary, final)
        published = True
        try:
            _fsync_directory(final.parent)
        except OSError:
            pass
        return staged_sha, True
    except FileExistsError:
        raise FileExistsError("concurrent completion publication won")
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            if not published:
                raise
        try:
            os.close(stable_archive.descriptor)
        except OSError:
            if not published:
                raise


def _load_parent_contracts(
    contract: ScientificRepairRunnerContract,
) -> tuple[InvalidForensicContract, TagResolutionContract]:
    p0 = load_frozen_forensic_contract(
        contract.repository_root / contract.data["p0_forensic_contract"]["path"],
        repository_root=contract.repository_root,
    )
    tag = load_frozen_tag_resolution_contract(
        contract.repository_root
        / contract.data["tag_resolution_parent"]["config_path"],
        repository_root=contract.repository_root,
    )
    if (
        p0.sha256 != contract.data["p0_forensic_contract"]["sha256"]
        or tag.sha256 != contract.data["tag_resolution_parent"]["config_sha256"]
    ):
        raise ValueError("scientific repair parent contract identity drifted")
    return p0, tag


def _read_deterministic_state(
    path: str | Path,
    *,
    expected: Mapping[str, Any],
    mode: int,
    label: str,
) -> tuple[bytes, str]:
    payload = _regular_file_bytes(path, label=label, mode=mode)
    record = strict_pretty_json_object_bytes(payload, label=label)
    if record != expected or _pretty_json_bytes(record) != payload:
        raise ValueError(f"{label} deterministic bytes drifted")
    return payload, sha256_bytes(payload)


def _require_fresh_matches_original(
    fresh: FreshForensicDownload,
    original: OriginalInvalidSnapshot,
) -> None:
    if (
        fresh.archive_bytes != original.local_archive_bytes
        or fresh.evidence.files != original.source_evidence.files
        or fresh.evidence.manifest != original.source_evidence.manifest
        or fresh.evidence.tree_inventory_sha256
        != original.source_evidence.tree_inventory_sha256
    ):
        raise ValueError("fresh immutable HF evidence differs from local INVALID source")


def _run_core(
    contract: ScientificRepairRunnerContract,
    *,
    p0_contract: InvalidForensicContract,
    fresh: FreshForensicDownload,
    external_inputs: Mapping[str, Any],
) -> LedgerNeutralScientificValidation:
    attestation = FrozenExternalReplayAttestation(
        implementation=dict(FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION),
        parent_substrate=dict(FROZEN_PARENT_SUBSTRATE),
        derived_artifact=dict(FROZEN_DERIVED_ARTIFACT),
        ocr_backend_config=dict(FROZEN_OCR_BACKEND_CONFIG),
        parent_substrate_archive_path=str(
            external_inputs["parent_substrate_archive_path"]
        ),
        derived_artifact_root=str(external_inputs["derived_artifact_root"]),
        ocr_backend_config_path=str(external_inputs["ocr_backend_config_path"]),
    )
    executor = FrozenExternalReplayExecutor(attestation)
    result = validate_ledger_neutral_scientific_payload(
        fresh.evidence,
        forensic_contract=p0_contract,
        external_input_replay_callback=executor,
        external_replay_attestation=attestation,
    )
    if (
        result.report.get("status") != CORE_PASS_STATUS
        or result.report.get("formal_scientific_repair_pass") is not True
    ):
        raise ValueError("ledger-neutral core did not return its formal PASS")
    return result


def _revalidate_immutable_preconditions(
    contract: ScientificRepairRunnerContract,
    *,
    p0_contract: InvalidForensicContract,
    tag_contract: TagResolutionContract,
    source: SourceIdentity,
    expected_head: str,
    receipt_path: str | Path,
    runtime_before: Mapping[str, Any],
    original_before: OriginalInvalidSnapshot,
    external_inputs_before: Mapping[str, Any],
    allow_loaded_module_growth: bool = False,
) -> SourceIdentity:
    repeated_source = validate_clean_pushed_source(
        contract,
        expected_head=expected_head,
        verify_remote=False,
        expected_remote_main=source.remote_main,
    )
    static_before = (
        source.head,
        source.origin_main,
        source.remote_main,
        source.branch,
        source.origin_url,
        source.inventory,
        source.inventory_sha256,
    )
    static_after = (
        repeated_source.head,
        repeated_source.origin_main,
        repeated_source.remote_main,
        repeated_source.branch,
        repeated_source.origin_url,
        repeated_source.inventory,
        repeated_source.inventory_sha256,
    )
    before_modules = {
        item["module"]: item for item in source.loaded_module_inventory
    }
    after_modules = {
        item["module"]: item for item in repeated_source.loaded_module_inventory
    }
    if (
        static_after != static_before
        or any(after_modules.get(name) != item for name, item in before_modules.items())
        or (
            not allow_loaded_module_growth
            and repeated_source.loaded_module_inventory
            != source.loaded_module_inventory
        )
    ):
        raise ValueError("formal source changed during scientific repair")
    validate_import_guard(contract)
    repeated_runtime = validate_no_gpu_launch_receipt(
        contract, source=source, receipt_path=receipt_path
    )
    if repeated_runtime != runtime_before:
        raise ValueError("no-GPU launch receipt changed during scientific repair")
    repeated_inputs = validate_external_replay_inputs(contract)
    if repeated_inputs != external_inputs_before:
        raise ValueError("external replay inputs changed during scientific repair")
    original_after = snapshot_original_invalid_attempt(
        contract,
        p0_contract=p0_contract,
        tag_contract=tag_contract,
    )
    require_original_snapshot_unchanged(original_before, original_after)
    return repeated_source


def execute_formal_scientific_repair(
    contract: ScientificRepairRunnerContract,
    *,
    mode: str,
    expected_source_git_commit: str,
    receipt_path: str | Path,
    hf_token: str,
    fresh_download_parent: str | Path,
    argv: Sequence[str],
) -> Mapping[str, Any]:
    if mode not in {"run", "validate"}:
        raise ValueError("scientific repair mode must be run or validate")
    initial_state = inspect_repair_state(contract)
    if mode == "validate" and not all(initial_state.values()):
        raise ValueError("read-only validation requires claim, output, and completion")
    p0_contract, tag_contract = _load_parent_contracts(contract)
    external_inputs = validate_external_replay_inputs(contract)
    source = validate_clean_pushed_source(
        contract, expected_head=expected_source_git_commit
    )
    guard = validate_import_guard(contract)
    receipt_runtime = validate_no_gpu_launch_receipt(
        contract, source=source, receipt_path=receipt_path
    )
    runtime = {**dict(receipt_runtime), "import_guard": dict(guard)}
    original = snapshot_original_invalid_attempt(
        contract,
        p0_contract=p0_contract,
        tag_contract=tag_contract,
    )

    # note (luojiaxuan): Network transport is deliberately completed before
    # the deterministic claim. A disconnect therefore cannot poison the sole
    # local repair identity, while the downloaded P0 bytes still remain the
    # only evidence object passed into the scientific core.
    fresh = download_fresh_immutable_forensic(
        contract,
        p0_contract=p0_contract,
        hf_token=hf_token,
        fresh_download_parent=fresh_download_parent,
    )
    _require_fresh_matches_original(fresh, original)
    _revalidate_immutable_preconditions(
        contract,
        p0_contract=p0_contract,
        tag_contract=tag_contract,
        source=source,
        expected_head=expected_source_git_commit,
        receipt_path=receipt_path,
        runtime_before=receipt_runtime,
        original_before=original,
        external_inputs_before=external_inputs,
    )

    claim = build_deterministic_claim(
        contract,
        source=source,
        runtime=runtime,
        original=original,
        fresh=fresh,
        external_inputs=external_inputs,
        argv=argv,
    )
    claim_path = Path(contract.output["claim_path"])
    state_mode = contract.output["state_file_mode"]
    if mode == "run":
        claim_bytes, claim_sha, claim_created = exclusive_or_identical_state(
            claim_path,
            claim,
            mode=state_mode,
        )
    else:
        claim_bytes, claim_sha = _read_deterministic_state(
            claim_path,
            expected=claim,
            mode=state_mode,
            label="scientific repair claim",
        )
        claim_created = False
    if claim_bytes != _pretty_json_bytes(claim):
        raise ValueError("scientific repair claim bytes drifted")

    validation = _run_core(
        contract,
        p0_contract=p0_contract,
        fresh=fresh,
        external_inputs=external_inputs,
    )
    validate_import_guard(contract)
    final_source = _revalidate_immutable_preconditions(
        contract,
        p0_contract=p0_contract,
        tag_contract=tag_contract,
        source=source,
        expected_head=expected_source_git_commit,
        receipt_path=receipt_path,
        runtime_before=receipt_runtime,
        original_before=original,
        external_inputs_before=external_inputs,
        allow_loaded_module_growth=True,
    )
    archive = build_formal_repaired_archive(
        contract,
        validation=validation,
        source=final_source,
        runtime=runtime,
        original=original,
        fresh=fresh,
        external_inputs=external_inputs,
        claim_sha256=claim_sha,
    )
    output_path = Path(contract.output["canonical_archive_path"])
    if output_path.exists() or output_path.is_symlink():
        existing = read_repaired_archive(
            output_path,
            contract=contract,
            require_formal=True,
            expected_files=archive.files,
        )
        if existing.archive_bytes != archive.archive_bytes:
            raise ValueError(
                "existing output-without-completion differs from full recomputation"
            )
        output_created = False
    elif mode == "validate":
        raise ValueError("read-only validation cannot create a missing output")
    else:
        def pre_publish_check() -> None:
            _revalidate_immutable_preconditions(
                contract,
                p0_contract=p0_contract,
                tag_contract=tag_contract,
                source=final_source,
                expected_head=expected_source_git_commit,
                receipt_path=receipt_path,
                runtime_before=receipt_runtime,
                original_before=original,
                external_inputs_before=external_inputs,
            )
            current_claim, _ = _read_deterministic_state(
                claim_path,
                expected=claim,
                mode=state_mode,
                label="scientific repair claim",
            )
            if current_claim != claim_bytes:
                raise ValueError("scientific repair claim changed before publish")

        atomic_publish_repaired_archive(
            output_path,
            archive,
            contract=contract,
            pre_publish_check=pre_publish_check,
        )
        output_created = True
    stable_output = open_stable_repaired_archive(
        output_path,
        contract=contract,
        expected_files=archive.files,
    )
    output_readback = stable_output.archive
    if output_readback.archive_bytes != archive.archive_bytes:
        os.close(stable_output.descriptor)
        raise ValueError("canonical repaired archive differs after strict readback")

    # note (luojiaxuan): Completion is the terminal canonical mutation. The
    # output is read and then re-read from one stable descriptor immediately
    # before the no-replace completion link; no validation follows that link.
    completion = build_completion_record(
        contract,
        claim_sha256=claim_sha,
        archive=output_readback,
        source=final_source,
    )
    completion_path = Path(contract.output["completion_path"])
    completion_sha = sha256_bytes(_pretty_json_bytes(completion))
    completion_created = mode == "run" and not initial_state["completion_exists"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": (
            contract.output["artifact_status"]
            if mode == "run"
            else contract.output["validate_status"]
        ),
        "source_git_commit": source.head,
        "runner_config_sha256": contract.sha256,
        "claim_sha256": claim_sha,
        "claim_created": claim_created,
        "output_created": output_created,
        "completion_sha256": completion_sha,
        "completion_created": completion_created,
        "archive_path": str(output_path),
        "archive_sha256": output_readback.archive_sha256,
        "archive_size_bytes": output_readback.archive_size_bytes,
        "member_count": len(output_readback.inventory),
        "tree_inventory_sha256": output_readback.tree_inventory_sha256,
        "formal_scientific_repair_pass": True,
        "original_attempt_remains_invalid": True,
        "repair_runtime_gpu_or_model_operation_count": 0,
        "gate_training_unlocked": False,
        "remote_mutation_call_count": 0,
    }
    if completion_created:
        def final_pre_completion_check() -> None:
            _revalidate_immutable_preconditions(
                contract,
                p0_contract=p0_contract,
                tag_contract=tag_contract,
                source=final_source,
                expected_head=expected_source_git_commit,
                receipt_path=receipt_path,
                runtime_before=receipt_runtime,
                original_before=original,
                external_inputs_before=external_inputs,
            )
            current_claim, _ = _read_deterministic_state(
                claim_path,
                expected=claim,
                mode=state_mode,
                label="scientific repair claim",
            )
            if current_claim != claim_bytes:
                raise ValueError("scientific repair claim changed before completion")
            if completion_path.exists() or completion_path.is_symlink():
                raise FileExistsError("scientific repair completion appeared concurrently")

        atomic_publish_completion_last(
            completion_path,
            completion,
            contract=contract,
            stable_archive=stable_output,
            pre_publish_check=final_pre_completion_check,
        )
        return result

    try:
        _revalidate_immutable_preconditions(
            contract,
            p0_contract=p0_contract,
            tag_contract=tag_contract,
            source=final_source,
            expected_head=expected_source_git_commit,
            receipt_path=receipt_path,
            runtime_before=receipt_runtime,
            original_before=original,
            external_inputs_before=external_inputs,
        )
        current_claim, _ = _read_deterministic_state(
            claim_path,
            expected=claim,
            mode=state_mode,
            label="scientific repair claim",
        )
        if current_claim != claim_bytes:
            raise ValueError("scientific repair claim changed before completion readback")
        revalidate_stable_repaired_archive(stable_output)
        completion_bytes, observed_completion_sha = _read_deterministic_state(
            completion_path,
            expected=completion,
            mode=state_mode,
            label="scientific repair completion",
        )
        if (
            completion_bytes != _pretty_json_bytes(completion)
            or observed_completion_sha != completion_sha
        ):
            raise ValueError("scientific repair completion bytes drifted")
    finally:
        try:
            os.close(stable_output.descriptor)
        except OSError:
            pass
    return result


__all__ = [
    "AUDIT_STATUS",
    "CANONICAL_CONFIG_PATH",
    "FROZEN_CONFIG_SHA256",
    "IMPORT_GUARD_MARKER",
    "MANIFEST_STATUS",
    "PROTOCOL_ID",
    "SOURCE_STATUS",
    "TEST_ONLY_STATUS",
    "BoundStateSnapshot",
    "FreshForensicDownload",
    "OriginalInvalidSnapshot",
    "RepairedArchive",
    "ScientificRepairRunnerContract",
    "SourceIdentity",
    "atomic_publish_repaired_archive",
    "build_completion_record",
    "build_deterministic_claim",
    "build_formal_repaired_archive",
    "build_test_only_repaired_archive",
    "deterministic_repaired_ustar_bytes",
    "download_fresh_immutable_forensic",
    "exclusive_or_identical_state",
    "execute_formal_scientific_repair",
    "inspect_repair_state",
    "load_frozen_runner_contract",
    "read_repaired_archive",
    "repaired_archive_from_files",
    "require_original_snapshot_unchanged",
    "snapshot_original_invalid_attempt",
    "validate_clean_pushed_source",
    "validate_completion_record",
    "validate_downloaded_forensic_pair",
    "validate_external_replay_inputs",
    "validate_import_guard",
    "validate_no_gpu_launch_receipt",
]
