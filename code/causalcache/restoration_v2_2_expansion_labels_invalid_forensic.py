"""Package the immutable bytes of the invalid expansion-label v1 attempt.

This protocol is transport-only.  It preserves the producer root and the
terminal external high-water ledgers without reclassifying the original
attempt as a valid label artifact.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import tarfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = (
    "causalcache_restoration_v2_2_expansion_exact_labels_invalid_forensic_v1"
)
SOURCE_STATUS = "source_only_frozen_before_any_invalid_forensic_packaging_or_upload"
ARCHIVE_STATUS = "ARCHIVED_INVALID_EXPANSION_EXACT_LABEL_ATTEMPT"
ARTIFACT_CLASS = "invalid_attempt_forensics"
PRODUCER_PROTOCOL_ID = "causalcache_restoration_v2_2_expansion_exact_labels_v1"
PRODUCER_INVALID_STATUS = "INVALID_EXPANSION_EXACT_LABEL_ATTEMPT"
PRODUCER_COMPLETED_STATUS = "COMPLETED_EXPANSION_EXACT_LABEL_ATTEMPT"
PRODUCER_WORKER_COMPLETED_STATUS = "COMPLETED_V2_2_EXPANSION_EXACT_LABEL_WORKER"
PRODUCER_EXECUTION_EVIDENCE_STATUS = (
    "VALIDATED_EXPANSION_EXACT_LABEL_EXECUTION_EVIDENCE"
)
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_"
    "expansion_labels_invalid_forensic_v1.json"
)
FROZEN_CONFIG_SHA256 = (
    "5290a51a250e31be4fcf0a970c77ef31c08c92c5892edd19a12ecb14d9d5a6a2"
)
ATTEMPT_ROOT_NAMESPACE = "attempt_root"
TERMINAL_LEDGER_NAMESPACE = "terminal_external_ledgers"
FORENSIC_MANIFEST_PATH = "forensic_manifest.json"
ARCHIVE_FORMAT = "ustar"
WORKERS = ("even", "odd")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class InvalidForensicContract:
    data: Mapping[str, Any]
    sha256: str
    source_path: Path | None = None

    @property
    def source_attempt(self) -> Mapping[str, Any]:
        return _mapping(self.data["source_attempt"], "source_attempt")

    @property
    def archive_contract(self) -> Mapping[str, Any]:
        return _mapping(self.data["archive_contract"], "archive_contract")


@dataclass(frozen=True)
class InvalidForensicEvidence:
    files: Mapping[str, bytes]
    manifest: Mapping[str, Any]
    inventory: tuple[Mapping[str, Any], ...]
    tree_inventory_sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} schema drifted")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def strict_pretty_json_object_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict) or pretty_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical pretty JSON")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be a UTC timestamp")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be a UTC timestamp") from error


def _safe_relative_path(value: Any, label: str = "archive path") -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be nonempty")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be canonical relative POSIX")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return Path(value)


def _inventory(files: Mapping[str, bytes]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "path": name,
            "sha256": sha256_bytes(files[name]),
            "size_bytes": len(files[name]),
        }
        for name in sorted(files)
    )


def _inventory_sha256(files: Mapping[str, bytes]) -> str:
    return sha256_bytes(canonical_json_bytes(_inventory(files)))


def _validate_binding(value: Any, label: str) -> dict[str, Any]:
    binding = dict(_mapping(value, label))
    _exact_keys(binding, {"path", "sha256", "size_bytes"}, label)
    _absolute_path(binding.get("path"), f"{label} path")
    _sha256(binding.get("sha256"), f"{label} SHA256")
    _positive_int(binding.get("size_bytes"), f"{label} size")
    return binding


def validate_forensic_contract_data(value: Any) -> Mapping[str, Any]:
    data = dict(_mapping(value, "invalid-forensic source contract"))
    _exact_keys(
        data,
        {
            "schema_version",
            "protocol_id",
            "status",
            "artifact_class",
            "source_attempt",
            "failure_contract",
            "archive_contract",
            "formal_consumption",
            "network_contract",
        },
        "invalid-forensic source contract",
    )
    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("protocol_id") != PROTOCOL_ID
        or data.get("status") != SOURCE_STATUS
        or data.get("artifact_class") != ARTIFACT_CLASS
    ):
        raise ValueError("invalid-forensic source identity drifted")

    source = dict(_mapping(data.get("source_attempt"), "source_attempt"))
    _exact_keys(
        source,
        {
            "producer_protocol_id",
            "attempt_id",
            "run_contract_sha256",
            "expected_state_count",
            "output_root",
            "root_file_count",
            "root_total_file_bytes",
            "root_inventory_sha256",
            "critical_root_file_sha256",
            "external_global_ledger",
            "external_worker_ledgers",
            "append_only_execution_log",
        },
        "source_attempt",
    )
    if (
        source.get("producer_protocol_id") != PRODUCER_PROTOCOL_ID
        or not isinstance(source.get("attempt_id"), str)
        or not source["attempt_id"]
    ):
        raise ValueError("producer attempt identity drifted")
    _sha256(source.get("run_contract_sha256"), "run-contract SHA256")
    _positive_int(source.get("expected_state_count"), "expected state count")
    _absolute_path(source.get("output_root"), "output root")
    root_count = _positive_int(source.get("root_file_count"), "root file count")
    _positive_int(source.get("root_total_file_bytes"), "root total bytes")
    _sha256(source.get("root_inventory_sha256"), "root inventory SHA256")
    critical = dict(
        _mapping(source.get("critical_root_file_sha256"), "critical root bindings")
    )
    required_critical = {
        "aggregate.json",
        "execution_evidence.json",
        "global_attempt_ledger.json",
        "logs/execution.log",
        "logs/gpu_utilization_monitor.log",
        "monitor_summary.json",
        "run_manifest.json",
    }
    if set(critical) != required_critical:
        raise ValueError("critical root binding inventory drifted")
    for path, digest in critical.items():
        _safe_relative_path(path, "critical root path")
        _sha256(digest, f"critical root SHA256 for {path}")

    global_binding = _validate_binding(
        source.get("external_global_ledger"), "external global ledger"
    )
    workers = dict(
        _mapping(source.get("external_worker_ledgers"), "external worker ledgers")
    )
    if set(workers) != set(WORKERS):
        raise ValueError("external worker ledger inventory drifted")
    worker_bindings = {
        worker: _validate_binding(workers[worker], f"{worker} external ledger")
        for worker in WORKERS
    }
    all_ledger_paths = [global_binding["path"]] + [
        worker_bindings[worker]["path"] for worker in WORKERS
    ]
    if len(set(all_ledger_paths)) != 3:
        raise ValueError("external ledger paths must be distinct")

    append = dict(
        _mapping(
            source.get("append_only_execution_log"),
            "append-only execution log",
        )
    )
    _exact_keys(
        append,
        {
            "execution_evidence_path",
            "execution_evidence_sha256",
            "final_log_path",
            "final_log_sha256",
            "final_log_size_bytes",
            "pre_failure_prefix_sha256",
            "pre_failure_prefix_size_bytes",
            "terminal_suffix_sha256",
            "terminal_suffix_size_bytes",
        },
        "append-only execution log",
    )
    for key in ("execution_evidence_path", "final_log_path"):
        _safe_relative_path(append.get(key), key)
    for key in (
        "execution_evidence_sha256",
        "final_log_sha256",
        "pre_failure_prefix_sha256",
        "terminal_suffix_sha256",
    ):
        _sha256(append.get(key), key)
    final_size = _positive_int(append.get("final_log_size_bytes"), "final log size")
    prefix_size = _positive_int(
        append.get("pre_failure_prefix_size_bytes"), "pre-failure prefix size"
    )
    suffix_size = _positive_int(
        append.get("terminal_suffix_size_bytes"), "terminal suffix size"
    )
    if prefix_size + suffix_size != final_size:
        raise ValueError("append-only execution-log sizes do not close")
    if (
        critical[append["execution_evidence_path"]]
        != append["execution_evidence_sha256"]
        or critical[append["final_log_path"]] != append["final_log_sha256"]
    ):
        raise ValueError("append-only bindings differ from critical root bindings")

    failure = dict(_mapping(data.get("failure_contract"), "failure contract"))
    _exact_keys(
        failure,
        {
            "exception_type",
            "message",
            "original_attempt_status",
            "root_embedded_snapshot_status",
        },
        "failure contract",
    )
    if (
        failure.get("exception_type") != "ValueError"
        or not isinstance(failure.get("message"), str)
        or not failure["message"]
        or failure.get("original_attempt_status") != PRODUCER_INVALID_STATUS
        or failure.get("root_embedded_snapshot_status")
        != PRODUCER_COMPLETED_STATUS
    ):
        raise ValueError("frozen failure identity drifted")

    archive = dict(_mapping(data.get("archive_contract"), "archive contract"))
    _exact_keys(
        archive,
        {
            "archive_format",
            "archive_member_prefix",
            "canonical_archive_path",
            "attempt_root_namespace",
            "terminal_external_ledger_namespace",
            "forensic_manifest_path",
            "expected_payload_file_count_before_manifest",
            "expected_member_count",
            "deterministic_regular_file_metadata",
            "overwrite_allowed",
        },
        "archive contract",
    )
    metadata = dict(
        _mapping(
            archive.get("deterministic_regular_file_metadata"),
            "deterministic USTAR metadata",
        )
    )
    if (
        archive.get("archive_format") != ARCHIVE_FORMAT
        or not isinstance(archive.get("archive_member_prefix"), str)
        or not archive["archive_member_prefix"]
        or archive.get("attempt_root_namespace") != ATTEMPT_ROOT_NAMESPACE
        or archive.get("terminal_external_ledger_namespace")
        != TERMINAL_LEDGER_NAMESPACE
        or archive.get("forensic_manifest_path") != FORENSIC_MANIFEST_PATH
        or archive.get("overwrite_allowed") is not False
        or metadata
        != {
            "mode": 0o644,
            "uid": 0,
            "gid": 0,
            "uname": "",
            "gname": "",
            "mtime": 0,
            "pax_headers": {},
        }
    ):
        raise ValueError("deterministic USTAR contract drifted")
    _absolute_path(archive.get("canonical_archive_path"), "canonical archive")
    payload_count = _positive_int(
        archive.get("expected_payload_file_count_before_manifest"),
        "payload file count",
    )
    member_count = _positive_int(
        archive.get("expected_member_count"), "archive member count"
    )
    if payload_count != root_count + 3 or member_count != payload_count + 1:
        raise ValueError("archive namespace counts do not close")

    formal = dict(_mapping(data.get("formal_consumption"), "formal consumption"))
    _exact_keys(
        formal,
        {
            "formal_label_loader_eligible",
            "gate_training_unlocked",
            "matched_nll_unlocked",
            "closed_loop_unlocked",
            "reason",
        },
        "formal consumption",
    )
    if (
        any(
            formal.get(key) is not False
            for key in (
                "formal_label_loader_eligible",
                "gate_training_unlocked",
                "matched_nll_unlocked",
                "closed_loop_unlocked",
            )
        )
        or not isinstance(formal.get("reason"), str)
        or not formal["reason"]
    ):
        raise ValueError("invalid forensic artifact cannot unlock formal work")

    network = dict(_mapping(data.get("network_contract"), "network contract"))
    if network != {
        "hf_publish_authorized": False,
        "network_access_required": False,
        "upload_command_present": False,
    }:
        raise ValueError(
            "source-only invalid forensic protocol cannot authorize upload"
        )
    return data


def forensic_contract_from_data(value: Any) -> InvalidForensicContract:
    data = validate_forensic_contract_data(value)
    payload = pretty_json_bytes(data)
    return InvalidForensicContract(data=data, sha256=sha256_bytes(payload))


def load_frozen_forensic_contract(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> InvalidForensicContract:
    supplied = Path(path)
    if not supplied.is_absolute():
        if repository_root is None:
            raise ValueError("relative config requires an explicit repository root")
        supplied = Path(repository_root) / supplied
    payload = _regular_file_bytes(supplied, "frozen invalid-forensic config")
    if sha256_bytes(payload) != FROZEN_CONFIG_SHA256:
        raise ValueError("frozen invalid-forensic config SHA256 drifted")
    data = strict_pretty_json_object_bytes(payload, label="invalid-forensic config")
    validate_forensic_contract_data(data)
    return InvalidForensicContract(
        data=data,
        sha256=FROZEN_CONFIG_SHA256,
        source_path=supplied.resolve(),
    )


def _regular_file_bytes(path: Path, label: str) -> bytes:
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
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} is missing") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise ValueError(f"{label} changed while being read")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise ValueError(f"{label} size changed while being read")
    return payload


def _read_regular_tree(root: Path) -> dict[str, bytes]:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            root,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0),
        )
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise ValueError("producer root is missing") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("producer root must be a non-symlink directory")
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        try:
            metadata = path.lstat()
        except OSError as error:
            raise ValueError("producer root changed during traversal") from error
        if stat.S_ISDIR(metadata.st_mode):
            continue
        relative = _safe_relative_path(
            path.relative_to(root).as_posix(), "producer root path"
        )
        files[relative] = _regular_file_bytes(
            path, f"producer root member {relative}"
        )
    return files


def _validate_root_snapshot(
    files: Mapping[str, bytes], contract: InvalidForensicContract
) -> None:
    source = contract.source_attempt
    inventory = _inventory(files)
    if (
        len(files) != source["root_file_count"]
        or sum(len(payload) for payload in files.values())
        != source["root_total_file_bytes"]
        or sha256_bytes(canonical_json_bytes(inventory))
        != source["root_inventory_sha256"]
    ):
        raise ValueError("producer root count, bytes, or inventory SHA256 drifted")
    for path, digest in source["critical_root_file_sha256"].items():
        if path not in files or sha256_bytes(files[path]) != digest:
            raise ValueError(f"critical producer root bytes drifted: {path}")


def _validate_external_binding(
    payload: bytes, binding: Mapping[str, Any], *, label: str
) -> None:
    if (
        len(payload) != binding["size_bytes"]
        or sha256_bytes(payload) != binding["sha256"]
    ):
        raise ValueError(f"{label} bytes drifted")


def _expected_worker_indices(worker: str, count: int) -> list[int]:
    parity = 0 if worker == "even" else 1
    return list(range(parity, count, 2))


def _validate_worker_ledger(
    payload: bytes,
    *,
    worker: str,
    contract: InvalidForensicContract,
) -> Mapping[str, Any]:
    ledger = strict_pretty_json_object_bytes(payload, label=f"{worker} worker ledger")
    _exact_keys(
        ledger,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract_sha256",
            "worker",
            "attempted_state_indices",
            "completed_state_indices",
            "retry_count",
            "top_up_count",
            "started_at_utc",
            "ended_at_utc",
        },
        f"{worker} worker ledger",
    )
    source = contract.source_attempt
    count = source["expected_state_count"]
    parity = 0 if worker == "even" else 1
    expected_indices = _expected_worker_indices(worker, count)
    expected_worker = {
        "worker_id": worker,
        "device": f"cuda:{parity}",
        "index_parity": parity,
        "state_indices": expected_indices,
    }
    if (
        ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("protocol_id") != PRODUCER_PROTOCOL_ID
        or ledger.get("status") != PRODUCER_WORKER_COMPLETED_STATUS
        or ledger.get("run_contract_sha256") != source["run_contract_sha256"]
        or ledger.get("worker") != expected_worker
        or ledger.get("attempted_state_indices") != expected_indices
        or ledger.get("completed_state_indices") != expected_indices
        or ledger.get("retry_count") != 0
        or ledger.get("top_up_count") != 0
        or _timestamp(ledger.get("started_at_utc"), "worker start")
        > _timestamp(ledger.get("ended_at_utc"), "worker end")
    ):
        raise ValueError(f"{worker} terminal high-water ledger drifted")
    return ledger


def _validate_global_ledger_chain(
    files: Mapping[str, bytes], contract: InvalidForensicContract
) -> Mapping[str, Any]:
    internal_name = f"{ATTEMPT_ROOT_NAMESPACE}/global_attempt_ledger.json"
    external_name = f"{TERMINAL_LEDGER_NAMESPACE}/global_attempt_ledger.json"
    try:
        internal_payload = files[internal_name]
        external_payload = files[external_name]
    except KeyError as error:
        raise ValueError(
            "forensic archive lacks one global ledger namespace"
        ) from error
    internal = strict_pretty_json_object_bytes(
        internal_payload, label="embedded completed global ledger"
    )
    external = strict_pretty_json_object_bytes(
        external_payload, label="terminal external global ledger"
    )
    base_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "attempt_id",
        "run_contract_sha256",
        "output_dir",
        "worker_sibling_ledgers",
        "attempted_state_indices",
        "completed_state_indices",
        "retry_count",
        "top_up_count",
        "started_at_utc",
        "ended_at_utc",
    }
    _exact_keys(internal, base_keys, "embedded completed global ledger")
    _exact_keys(
        external,
        base_keys | {"failure", "claimed_ledger_sha256"},
        "terminal external global ledger",
    )
    source = contract.source_attempt
    count = source["expected_state_count"]
    expected_indices = list(range(count))
    worker_bindings = source["external_worker_ledgers"]
    expected_siblings = {
        worker: worker_bindings[worker]["path"] for worker in WORKERS
    }
    shared_expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PRODUCER_PROTOCOL_ID,
        "attempt_id": source["attempt_id"],
        "run_contract_sha256": source["run_contract_sha256"],
        "output_dir": source["output_root"],
        "worker_sibling_ledgers": expected_siblings,
        "attempted_state_indices": expected_indices,
        "completed_state_indices": expected_indices,
        "retry_count": 0,
        "top_up_count": 0,
    }
    for key, expected in shared_expected.items():
        if internal.get(key) != expected or external.get(key) != expected:
            raise ValueError(f"global ledger chain field drifted: {key}")
    failure = contract.data["failure_contract"]
    if (
        internal.get("status") != PRODUCER_COMPLETED_STATUS
        or external.get("status") != PRODUCER_INVALID_STATUS
        or external.get("failure")
        != {
            "exception_type": failure["exception_type"],
            "message": failure["message"],
        }
        or external.get("claimed_ledger_sha256") != sha256_bytes(internal_payload)
        or sha256_bytes(internal_payload)
        != source["critical_root_file_sha256"]["global_attempt_ledger.json"]
        or _timestamp(internal.get("started_at_utc"), "internal global start")
        > _timestamp(internal.get("ended_at_utc"), "internal global end")
        or _timestamp(external.get("started_at_utc"), "external global start")
        > _timestamp(external.get("ended_at_utc"), "external global end")
        or _timestamp(internal["ended_at_utc"], "internal global end")
        > _timestamp(external["ended_at_utc"], "external global end")
    ):
        raise ValueError("completed-to-invalid global ledger chain drifted")

    worker_sha256: dict[str, str] = {}
    for worker in WORKERS:
        root_worker = (
            f"{ATTEMPT_ROOT_NAMESPACE}/workers/{worker}/worker_attempt_ledger.json"
        )
        root_sibling = (
            f"{ATTEMPT_ROOT_NAMESPACE}/worker_sibling_ledgers/{worker}.json"
        )
        external_worker = (
            f"{TERMINAL_LEDGER_NAMESPACE}/workers/{worker}.json"
        )
        try:
            payloads = (
                files[root_worker],
                files[root_sibling],
                files[external_worker],
            )
        except KeyError as error:
            raise ValueError(f"{worker} ledger namespace is incomplete") from error
        if len(set(payloads)) != 1:
            raise ValueError(f"{worker} root, sibling, and external ledgers differ")
        _validate_worker_ledger(payloads[0], worker=worker, contract=contract)
        worker_sha256[worker] = sha256_bytes(payloads[0])
    return {
        "embedded_completed_global_ledger_sha256": sha256_bytes(internal_payload),
        "terminal_external_invalid_global_ledger_sha256": sha256_bytes(
            external_payload
        ),
        "terminal_claims_embedded_completed_sha256": True,
        "worker_ledger_sha256": worker_sha256,
        "worker_root_sibling_external_byte_equal": True,
    }


def _validate_append_only_execution_log(
    files: Mapping[str, bytes], contract: InvalidForensicContract
) -> Mapping[str, Any]:
    source = contract.source_attempt
    append = source["append_only_execution_log"]
    log_name = f"{ATTEMPT_ROOT_NAMESPACE}/{append['final_log_path']}"
    evidence_name = (
        f"{ATTEMPT_ROOT_NAMESPACE}/{append['execution_evidence_path']}"
    )
    try:
        final_log = files[log_name]
        evidence_payload = files[evidence_name]
    except KeyError as error:
        raise ValueError("append-only execution-log evidence is incomplete") from error
    if (
        len(final_log) != append["final_log_size_bytes"]
        or sha256_bytes(final_log) != append["final_log_sha256"]
        or sha256_bytes(evidence_payload) != append["execution_evidence_sha256"]
    ):
        raise ValueError("final execution-log or execution-evidence bytes drifted")
    prefix_size = append["pre_failure_prefix_size_bytes"]
    prefix = final_log[:prefix_size]
    suffix = final_log[prefix_size:]
    if (
        len(suffix) != append["terminal_suffix_size_bytes"]
        or sha256_bytes(prefix) != append["pre_failure_prefix_sha256"]
        or sha256_bytes(suffix) != append["terminal_suffix_sha256"]
        or not suffix.endswith(b"\n")
        or suffix.count(b"\n") != 1
    ):
        raise ValueError("execution log is not the frozen prefix plus one suffix line")
    try:
        suffix_text = suffix.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("terminal execution-log suffix is not UTF-8") from error
    marker = " attempt invalidated: "
    if marker not in suffix_text:
        raise ValueError("terminal execution-log suffix lacks the invalidation marker")
    timestamp_text, message_with_newline = suffix_text.split(marker, 1)
    if (
        message_with_newline != contract.data["failure_contract"]["message"] + "\n"
    ):
        raise ValueError("terminal execution-log failure message drifted")
    suffix_time = _timestamp(timestamp_text, "terminal execution-log timestamp")

    evidence = strict_pretty_json_object_bytes(
        evidence_payload, label="producer execution evidence"
    )
    _exact_keys(
        evidence,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract_sha256",
            "preflight_log_sha256",
            "execution_log_sha256",
            "utilization_monitor_log_sha256",
            "monitor_ready_sha256",
            "monitor_stop_request_sha256",
            "monitor_summary_sha256",
        },
        "producer execution evidence",
    )
    if (
        evidence.get("schema_version") != SCHEMA_VERSION
        or evidence.get("protocol_id") != PRODUCER_PROTOCOL_ID
        or evidence.get("status") != PRODUCER_EXECUTION_EVIDENCE_STATUS
        or evidence.get("run_contract_sha256") != source["run_contract_sha256"]
        or evidence.get("execution_log_sha256")
        != append["pre_failure_prefix_sha256"]
    ):
        raise ValueError("execution evidence does not bind the pre-failure prefix")
    for key, value in evidence.items():
        if key.endswith("_sha256"):
            _sha256(value, f"execution evidence {key}")

    external = strict_pretty_json_object_bytes(
        files[f"{TERMINAL_LEDGER_NAMESPACE}/global_attempt_ledger.json"],
        label="terminal external global ledger",
    )
    if suffix_time < _timestamp(external["ended_at_utc"], "external global end"):
        raise ValueError("execution-log invalidation suffix predates invalid ledger")
    return {
        "final_log_sha256": sha256_bytes(final_log),
        "pre_failure_prefix_sha256": sha256_bytes(prefix),
        "terminal_suffix_sha256": sha256_bytes(suffix),
        "terminal_suffix_is_exactly_one_line": True,
        "execution_evidence_binds_pre_failure_prefix": True,
    }


def _manifest_payload(
    payload_files: Mapping[str, bytes], contract: InvalidForensicContract
) -> Mapping[str, Any]:
    root_files = {
        name[len(f"{ATTEMPT_ROOT_NAMESPACE}/") :]: payload
        for name, payload in payload_files.items()
        if name.startswith(f"{ATTEMPT_ROOT_NAMESPACE}/")
    }
    external_files = {
        name[len(f"{TERMINAL_LEDGER_NAMESPACE}/") :]: payload
        for name, payload in payload_files.items()
        if name.startswith(f"{TERMINAL_LEDGER_NAMESPACE}/")
    }
    ledger_chain = _validate_global_ledger_chain(payload_files, contract)
    append_chain = _validate_append_only_execution_log(payload_files, contract)
    payload_inventory = _inventory(payload_files)
    source = contract.source_attempt
    archive = contract.archive_contract
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": ARCHIVE_STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "forensic_source_config": {
            "path": CANONICAL_CONFIG_PATH,
            "sha256": contract.sha256,
        },
        "source_attempt": {
            "producer_protocol_id": PRODUCER_PROTOCOL_ID,
            "attempt_id": source["attempt_id"],
            "run_contract_sha256": source["run_contract_sha256"],
            "original_attempt_status": PRODUCER_INVALID_STATUS,
            "root_embedded_snapshot_status": PRODUCER_COMPLETED_STATUS,
        },
        "archive_layout": {
            "format": ARCHIVE_FORMAT,
            "member_prefix": archive["archive_member_prefix"],
            "attempt_root_namespace": ATTEMPT_ROOT_NAMESPACE,
            "terminal_external_ledger_namespace": TERMINAL_LEDGER_NAMESPACE,
            "manifest_path": FORENSIC_MANIFEST_PATH,
            "payload_file_count_before_manifest": len(payload_files),
            "member_count_with_manifest": len(payload_files) + 1,
        },
        "source_inventory": {
            "attempt_root": {
                "file_count": len(root_files),
                "total_file_bytes": sum(len(value) for value in root_files.values()),
                "inventory_sha256": _inventory_sha256(root_files),
            },
            "terminal_external_ledgers": {
                "file_count": len(external_files),
                "total_file_bytes": sum(
                    len(value) for value in external_files.values()
                ),
                "inventory_sha256": _inventory_sha256(external_files),
            },
            "namespaced_payload": {
                "file_count": len(payload_files),
                "total_file_bytes": sum(
                    len(value) for value in payload_files.values()
                ),
                "inventory_sha256": sha256_bytes(
                    canonical_json_bytes(payload_inventory)
                ),
            },
        },
        "ledger_chain": ledger_chain,
        "append_only_execution_log": append_chain,
        "formal_consumption": dict(contract.data["formal_consumption"]),
        "network_contract": dict(contract.data["network_contract"]),
        "negative_declarations": {
            "producer_bytes_modified": False,
            "producer_invalid_status_reclassified": False,
            "formal_label_pass_claimed": False,
            "gate_training_authorized": False,
            "hf_upload_performed_by_this_protocol": False,
        },
    }


def _validate_manifest_identity(
    manifest: Mapping[str, Any], contract: InvalidForensicContract
) -> None:
    _exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "status",
            "artifact_class",
            "forensic_source_config",
            "source_attempt",
            "archive_layout",
            "source_inventory",
            "ledger_chain",
            "append_only_execution_log",
            "formal_consumption",
            "network_contract",
            "negative_declarations",
        },
        "forensic manifest",
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != ARCHIVE_STATUS
        or manifest.get("artifact_class") != ARTIFACT_CLASS
        or manifest.get("forensic_source_config")
        != {"path": CANONICAL_CONFIG_PATH, "sha256": contract.sha256}
        or manifest.get("formal_consumption")
        != contract.data["formal_consumption"]
        or manifest.get("network_contract") != contract.data["network_contract"]
        or manifest.get("negative_declarations")
        != {
            "producer_bytes_modified": False,
            "producer_invalid_status_reclassified": False,
            "formal_label_pass_claimed": False,
            "gate_training_authorized": False,
            "hf_upload_performed_by_this_protocol": False,
        }
    ):
        raise ValueError("forensic manifest identity or negative declarations drifted")


def validate_invalid_forensic_files(
    files: Mapping[str, bytes],
    *,
    contract: InvalidForensicContract,
) -> InvalidForensicEvidence:
    if not isinstance(files, Mapping):
        raise TypeError("forensic files must be a mapping")
    normalized = {_safe_relative_path(name): payload for name, payload in files.items()}
    if any(not isinstance(payload, bytes) for payload in normalized.values()):
        raise TypeError("forensic payloads must be bytes")
    try:
        manifest_payload = normalized[FORENSIC_MANIFEST_PATH]
    except KeyError as error:
        raise ValueError("forensic manifest is missing") from error
    payload_files = {
        name: payload
        for name, payload in normalized.items()
        if name != FORENSIC_MANIFEST_PATH
    }
    prefixes = (
        f"{ATTEMPT_ROOT_NAMESPACE}/",
        f"{TERMINAL_LEDGER_NAMESPACE}/",
    )
    if any(not name.startswith(prefixes) for name in payload_files):
        raise ValueError("forensic payload escaped the two frozen namespaces")
    archive = contract.archive_contract
    if (
        len(payload_files) != archive["expected_payload_file_count_before_manifest"]
        or len(normalized) != archive["expected_member_count"]
    ):
        raise ValueError("forensic archive member counts drifted")
    manifest = strict_pretty_json_object_bytes(
        manifest_payload, label="forensic manifest"
    )
    _validate_manifest_identity(manifest, contract)

    root_files = {
        name[len(f"{ATTEMPT_ROOT_NAMESPACE}/") :]: payload
        for name, payload in payload_files.items()
        if name.startswith(f"{ATTEMPT_ROOT_NAMESPACE}/")
    }
    external_files = {
        name[len(f"{TERMINAL_LEDGER_NAMESPACE}/") :]: payload
        for name, payload in payload_files.items()
        if name.startswith(f"{TERMINAL_LEDGER_NAMESPACE}/")
    }
    _validate_root_snapshot(root_files, contract)
    if set(external_files) != {
        "global_attempt_ledger.json",
        "workers/even.json",
        "workers/odd.json",
    }:
        raise ValueError("terminal external ledger archive inventory drifted")
    source = contract.source_attempt
    bindings = {
        "global_attempt_ledger.json": source["external_global_ledger"],
        "workers/even.json": source["external_worker_ledgers"]["even"],
        "workers/odd.json": source["external_worker_ledgers"]["odd"],
    }
    for name, binding in bindings.items():
        _validate_external_binding(
            external_files[name], binding, label=f"archived {name}"
        )
    rebuilt = _manifest_payload(payload_files, contract)
    if manifest != rebuilt:
        raise ValueError("forensic manifest differs from archived source bytes")
    inventory = _inventory(normalized)
    return InvalidForensicEvidence(
        files=dict(normalized),
        manifest=dict(manifest),
        inventory=inventory,
        tree_inventory_sha256=sha256_bytes(canonical_json_bytes(inventory)),
    )


def collect_invalid_forensic_source(
    *,
    contract: InvalidForensicContract,
    raw_output_dir: str | Path,
    external_global_ledger: str | Path,
    external_worker_ledgers: Mapping[str, str | Path],
) -> InvalidForensicEvidence:
    source = contract.source_attempt
    root = Path(raw_output_dir)
    global_path = Path(external_global_ledger)
    worker_paths = {key: Path(value) for key, value in external_worker_ledgers.items()}
    if (
        root.resolve() != Path(source["output_root"]).resolve()
        or global_path.resolve()
        != Path(source["external_global_ledger"]["path"]).resolve()
        or set(worker_paths) != set(WORKERS)
        or any(
            worker_paths[worker].resolve()
            != Path(source["external_worker_ledgers"][worker]["path"]).resolve()
            for worker in WORKERS
        )
    ):
        raise ValueError("forensic source paths differ from the frozen contract")
    root_files = _read_regular_tree(root)
    _validate_root_snapshot(root_files, contract)
    global_payload = _regular_file_bytes(global_path, "external global ledger")
    _validate_external_binding(
        global_payload,
        source["external_global_ledger"],
        label="external global ledger",
    )
    worker_payloads: dict[str, bytes] = {}
    for worker in WORKERS:
        payload = _regular_file_bytes(
            worker_paths[worker], f"{worker} external ledger"
        )
        _validate_external_binding(
            payload,
            source["external_worker_ledgers"][worker],
            label=f"{worker} external ledger",
        )
        worker_payloads[worker] = payload
    payload_files = {
        f"{ATTEMPT_ROOT_NAMESPACE}/{name}": payload
        for name, payload in root_files.items()
    }
    payload_files[f"{TERMINAL_LEDGER_NAMESPACE}/global_attempt_ledger.json"] = (
        global_payload
    )
    for worker in WORKERS:
        payload_files[
            f"{TERMINAL_LEDGER_NAMESPACE}/workers/{worker}.json"
        ] = worker_payloads[worker]
    manifest = _manifest_payload(payload_files, contract)
    files = dict(payload_files)
    files[FORENSIC_MANIFEST_PATH] = pretty_json_bytes(manifest)
    return validate_invalid_forensic_files(files, contract=contract)


def deterministic_invalid_forensic_ustar_bytes(
    files: Mapping[str, bytes],
    *,
    member_prefix: str,
) -> bytes:
    if not isinstance(member_prefix, str) or not member_prefix:
        raise ValueError("USTAR member prefix must be nonempty")
    destination = io.BytesIO()
    with tarfile.open(
        fileobj=destination, mode="w", format=tarfile.USTAR_FORMAT
    ) as archive:
        for relative in sorted(files):
            _safe_relative_path(relative)
            payload = files[relative]
            if not isinstance(payload, bytes):
                raise TypeError("USTAR payloads must be bytes")
            info = tarfile.TarInfo(f"{member_prefix}/{relative}")
            info.size = len(payload)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return destination.getvalue()


def read_invalid_forensic_archive(
    path: str | Path,
    *,
    contract: InvalidForensicContract,
) -> InvalidForensicEvidence:
    archive_path = Path(path)
    payload = _regular_file_bytes(archive_path, "invalid-forensic archive")
    files: dict[str, bytes] = {}
    prefix = f"{contract.archive_contract['archive_member_prefix']}/"
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != sorted(
                member.name for member in members
            ):
                raise ValueError("USTAR member order drifted")
            for member in members:
                if (
                    not member.isfile()
                    or not member.name.startswith(prefix)
                    or member.mode != 0o644
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or member.pax_headers
                ):
                    raise ValueError("archive has non-canonical USTAR metadata")
                relative = _safe_relative_path(
                    member.name[len(prefix) :], "USTAR member path"
                )
                if relative in files:
                    raise ValueError("USTAR contains duplicate members")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("USTAR member is unreadable")
                member_payload = extracted.read()
                if len(member_payload) != member.size:
                    raise ValueError("USTAR member size differs from extracted bytes")
                files[relative] = member_payload
    except tarfile.TarError as error:
        raise ValueError("invalid-forensic archive is not readable USTAR") from error
    rebuilt = deterministic_invalid_forensic_ustar_bytes(
        files,
        member_prefix=contract.archive_contract["archive_member_prefix"],
    )
    if payload != rebuilt:
        raise ValueError("archive is not byte-canonical deterministic USTAR")
    return validate_invalid_forensic_files(files, contract=contract)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_exclusive_publish_bytes(
    path: str | Path,
    payload: bytes,
    *,
    pre_publish_check: Callable[[], None] | None = None,
    publish_fn: Callable[[Path, Path], Any] = os.link,
) -> None:
    final = Path(path)
    try:
        parent_metadata = final.parent.lstat()
    except OSError as error:
        raise ValueError("archive output parent must preexist") from error
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("archive output parent must be a non-symlink directory")
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {final}")
    temporary = final.with_name(f".{final.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o644)
    try:
        os.fchmod(descriptor, 0o644)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("archive write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        if pre_publish_check is not None:
            pre_publish_check()
        # note (luojiaxuan): A same-directory hard link provides atomic
        # no-replace publication without exposing partially written bytes.
        publish_fn(temporary, final)
        _fsync_directory(final.parent)
    finally:
        temporary.unlink(missing_ok=True)


def package_invalid_forensic_archive(
    *,
    contract: InvalidForensicContract,
    raw_output_dir: str | Path,
    external_global_ledger: str | Path,
    external_worker_ledgers: Mapping[str, str | Path],
    output_archive: str | Path,
) -> Mapping[str, Any]:
    output = Path(output_archive)
    if output.resolve() != Path(
        contract.archive_contract["canonical_archive_path"]
    ).resolve():
        raise ValueError("forensic archive output differs from the frozen path")
    if output.exists() or output.is_symlink():
        raise FileExistsError("invalid-forensic archive overwrite is forbidden")
    evidence = collect_invalid_forensic_source(
        contract=contract,
        raw_output_dir=raw_output_dir,
        external_global_ledger=external_global_ledger,
        external_worker_ledgers=external_worker_ledgers,
    )
    payload = deterministic_invalid_forensic_ustar_bytes(
        evidence.files,
        member_prefix=contract.archive_contract["archive_member_prefix"],
    )

    def source_still_exact() -> None:
        repeated = collect_invalid_forensic_source(
            contract=contract,
            raw_output_dir=raw_output_dir,
            external_global_ledger=external_global_ledger,
            external_worker_ledgers=external_worker_ledgers,
        )
        if repeated.files != evidence.files:
            raise ValueError("forensic source changed before archive publication")

    atomic_exclusive_publish_bytes(
        output,
        payload,
        pre_publish_check=source_still_exact,
    )
    reread = read_invalid_forensic_archive(output, contract=contract)
    if reread.files != evidence.files:
        raise ValueError("published archive differs from collected forensic bytes")
    after_publication = collect_invalid_forensic_source(
        contract=contract,
        raw_output_dir=raw_output_dir,
        external_global_ledger=external_global_ledger,
        external_worker_ledgers=external_worker_ledgers,
    )
    if after_publication.files != evidence.files:
        raise ValueError("forensic source changed after archive publication")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "PACKAGED_INVALID_EXPANSION_EXACT_LABEL_FORENSIC_ARCHIVE",
        "artifact_class": ARTIFACT_CLASS,
        "formal_label_loader_eligible": False,
        "source_config_sha256": contract.sha256,
        "archive_path": str(output.resolve()),
        "archive_sha256": sha256_file(output),
        "archive_size_bytes": output.stat().st_size,
        "member_count": len(reread.files),
        "tree_inventory_sha256": reread.tree_inventory_sha256,
        "producer_attempt_status": PRODUCER_INVALID_STATUS,
    }


def require_formal_label_loader_eligible(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("protocol_id") == PROTOCOL_ID
        or manifest.get("artifact_class") == ARTIFACT_CLASS
        or _mapping(manifest.get("formal_consumption"), "formal consumption").get(
            "formal_label_loader_eligible"
        )
        is not True
    ):
        raise ValueError(
            "invalid-forensic protocol is permanently ineligible for formal labels"
        )


__all__ = [
    "ARCHIVE_FORMAT",
    "ARCHIVE_STATUS",
    "ARTIFACT_CLASS",
    "ATTEMPT_ROOT_NAMESPACE",
    "CANONICAL_CONFIG_PATH",
    "FORENSIC_MANIFEST_PATH",
    "FROZEN_CONFIG_SHA256",
    "InvalidForensicContract",
    "InvalidForensicEvidence",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "SOURCE_STATUS",
    "TERMINAL_LEDGER_NAMESPACE",
    "atomic_exclusive_publish_bytes",
    "canonical_json_bytes",
    "collect_invalid_forensic_source",
    "deterministic_invalid_forensic_ustar_bytes",
    "forensic_contract_from_data",
    "load_frozen_forensic_contract",
    "package_invalid_forensic_archive",
    "pretty_json_bytes",
    "read_invalid_forensic_archive",
    "require_formal_label_loader_eligible",
    "sha256_bytes",
    "sha256_file",
    "strict_pretty_json_object_bytes",
    "validate_forensic_contract_data",
    "validate_invalid_forensic_files",
]
