"""Fail-closed execution wrapper for the fresh-16 claim serialization repair."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache import gate_v1_fresh16_evaluation_runner as base_runner
from causalcache import gate_v1_fresh16_inventory_repair_runner as parent_runner
from causalcache.gate_v1_fresh16_evaluation_contract import canonical_json_bytes


validate_source_a = base_runner.validate_source_a
materialize_runner_freeze = base_runner.materialize_runner_freeze
validate_execution_b_source = base_runner.validate_execution_b_source
capture_docker_inspect_receipt = base_runner.capture_docker_inspect_receipt
validate_retained_v1_failure = parent_runner.validate_retained_v1_failure

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_STATE_NAME = re.compile(r"[a-z0-9_]+")
_RETAINED_STATE_NAMES = (
    "runtime_receipts",
    "global_claim",
    "transport_verification",
    "label_blind_cpu_completion",
    "checkpoint_replay_completion",
    "policy_worker_even_completion",
    "policy_worker_odd_completion",
    "heuristic_local_seal",
)


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


def _parent_failure(contract: Any) -> Mapping[str, Any]:
    repair = getattr(contract, "repair", None)
    if repair is None:
        overlay = _mapping(getattr(contract, "overlay", None), "repair overlay")
        repair = overlay.get("claim_serialization_repair")
    return _mapping(
        _mapping(repair, "claim serialization repair").get("parent_failure"),
        "parent failure",
    )


def _real_data_root(data_root: Path) -> Path:
    supplied = Path(data_root)
    try:
        metadata = supplied.lstat()
    except OSError as error:
        raise ValueError("data root is missing or unsafe") from error
    if supplied.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("data root must be a real directory")
    resolved = supplied.resolve(strict=True)
    if resolved != supplied.absolute():
        raise ValueError("data root must be a canonical real directory")
    return resolved


def _under_data_root(data_root: Path, canonical: Any, *, label: str) -> Path:
    if not isinstance(canonical, str):
        raise ValueError(f"{label} path must be text")
    pure = PurePosixPath(canonical)
    if (
        not pure.is_absolute()
        or pure.parts[:2] != ("/", "data")
        or pure.as_posix() != canonical
        or any(part in {"", ".", ".."} for part in pure.parts[2:])
    ):
        raise ValueError(f"{label} must be canonically rooted under /data")
    root = _real_data_root(data_root)
    result = root.joinpath(*pure.parts[2:])
    current = root
    for part in pure.parts[2:-1]:
        current /= part
        if not os.path.lexists(current):
            break
        metadata = current.lstat()
        if current.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} parent is unsafe")
    return result


def _real_directory(data_root: Path, canonical: Any, *, label: str) -> Path:
    path = _under_data_root(data_root, canonical, label=label)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing or unsafe") from error
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or path.resolve(strict=True) != path.absolute()
    ):
        raise ValueError(f"{label} must be a real canonical directory")
    return path


def _assert_absent(data_root: Path, canonical: str, *, label: str) -> None:
    path = _under_data_root(data_root, canonical, label=label)
    if os.path.lexists(path):
        raise ValueError(f"{label} must remain absent")


def _regular_file(
    data_root: Path,
    raw_record: Any,
    *,
    label: str,
) -> tuple[Mapping[str, Any], bytes]:
    record = _mapping(raw_record, label)
    if set(record) - {"path", "mode", "size_bytes", "sha256", "state_name"}:
        raise ValueError(f"{label} contains unsupported identity fields")
    mode = record.get("mode")
    size = record.get("size_bytes")
    digest = record.get("sha256")
    if (
        type(mode) is not int
        or mode not in {0o400, 0o444, 0o600, 0o644}
        or type(size) is not int
        or size < 0
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
    ):
        raise ValueError(f"{label} identity is malformed")
    path = _under_data_root(data_root, record.get("path"), label=label)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != mode
            or before.st_size != size
        ):
            raise ValueError(f"{label} type, mode, or size drifted")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    payload = b"".join(chunks)
    fingerprint = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        fingerprint(before) != fingerprint(after)
        or len(payload) != size
        or hashlib.sha256(payload).hexdigest() != digest
    ):
        raise ValueError(f"{label} bytes changed or drifted")
    return (
        {
            "path": record["path"],
            "mode": mode,
            "size_bytes": size,
            "sha256": digest,
        },
        payload,
    )


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
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


def _artifact_inventory(
    data_root: Path,
    specification: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], bytes]:
    root = _real_directory(data_root, specification.get("root"), label="old artifact root")
    expected_top = tuple(specification.get("top_level_entries", ()))
    observed_top = tuple(sorted(entry.name for entry in os.scandir(root)))
    if observed_top != expected_top:
        raise ValueError("old artifact top-level entry inventory drifted")
    records: list[Mapping[str, Any]] = []
    directories: set[str] = set()

    def visit(directory: Path) -> None:
        for entry in os.scandir(directory):
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            metadata = entry.stat(follow_symlinks=False)
            if entry.is_symlink():
                raise ValueError("old artifact tree contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                visit(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("old artifact tree contains a special file")
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    path,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                before = os.fstat(descriptor)
                hasher = hashlib.sha256()
                observed_size = 0
                while chunk := os.read(descriptor, 1024 * 1024):
                    hasher.update(chunk)
                    observed_size += len(chunk)
                after = os.fstat(descriptor)
            except OSError as error:
                raise ValueError("old artifact file is missing or unsafe") from error
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
            if (
                not stat.S_ISREG(before.st_mode)
                or fingerprint(before) != fingerprint(after)
                or observed_size != after.st_size
            ):
                raise ValueError("old artifact file changed while being read")
            records.append(
                {
                    "path": relative,
                    "mode": stat.S_IMODE(after.st_mode),
                    "size_bytes": observed_size,
                    "sha256": hasher.hexdigest(),
                }
            )

    visit(root)
    records.sort(key=lambda item: item["path"])
    required_directories: set[str] = set()
    for record in records:
        parent = PurePosixPath(record["path"]).parent
        while parent.as_posix() != ".":
            required_directories.add(parent.as_posix())
            parent = parent.parent
    if directories != required_directories:
        raise ValueError("old artifact directory inventory contains an extra empty directory")
    payload = canonical_json_bytes(records)
    if (
        len(records) != specification.get("file_count")
        or sum(item["size_bytes"] for item in records)
        != specification.get("total_bytes")
        or len(payload) != specification.get("canonical_inventory_json_size_bytes")
        or hashlib.sha256(payload).hexdigest()
        != specification.get("canonical_inventory_sha256")
    ):
        raise ValueError("old artifact byte/mode inventory drifted")
    return tuple(records), payload


def validate_retained_inventory_repair_failure(
    contract: Any,
    data_root: Path,
) -> Mapping[str, Any]:
    """Verify the failed inventory-repair run byte-for-byte without writes."""
    failure = _parent_failure(contract)
    namespace = failure.get("old_state_namespace")
    state_root = _real_directory(data_root, namespace, label="old state namespace")
    expected_entries = tuple(failure.get("expected_state_root_entries", ()))
    observed_entries = tuple(sorted(entry.name for entry in os.scandir(state_root)))
    if observed_entries != expected_entries:
        raise ValueError("old state namespace entry inventory drifted")

    receipt_records = tuple(
        _mapping(item, "old ordered receipt")
        for item in _sequence(failure.get("ordered_receipts"), "old ordered receipts")
    )
    if tuple(item.get("state_name") for item in receipt_records) != _RETAINED_STATE_NAMES:
        raise ValueError("old ordered receipt prefix must contain exactly states 0..7")
    receipt_root = _real_directory(
        data_root,
        f"{namespace}/ordered-state-receipts",
        label="old ordered receipt directory",
    )
    expected_filenames = tuple(Path(str(item["path"])).name for item in receipt_records)
    observed_filenames = tuple(sorted(entry.name for entry in os.scandir(receipt_root)))
    if observed_filenames != tuple(sorted(expected_filenames)):
        raise ValueError("old ordered receipt set must contain exactly states 0..7")

    previous_sha256: str | None = None
    parsed_receipts = []
    receipt_identities = []
    for ordinal, (name, raw_record) in enumerate(
        zip(_RETAINED_STATE_NAMES, receipt_records, strict=True)
    ):
        expected_path = (
            f"{namespace}/ordered-state-receipts/{ordinal:02d}-{name}.json"
        )
        if raw_record.get("path") != expected_path:
            raise ValueError("old ordered receipt path escaped the state namespace")
        identity, payload = _regular_file(
            data_root,
            raw_record,
            label=f"old ordered receipt {name}",
        )
        record = _strict_json(payload, label=f"old ordered receipt {name}")
        if (
            payload != base_runner.pretty_json_bytes(record)
            or set(record)
            != {
                "schema_version",
                "protocol_id",
                "status",
                "ordinal",
                "stage",
                "previous_receipt_sha256",
                "payload_sha256",
                "payload",
            }
            or record.get("schema_version") != base_runner.SCHEMA_VERSION
            or record.get("protocol_id") != base_runner.PROTOCOL_ID
            or record.get("status")
            != "COMPLETED_GATE_V1_FRESH16_ORDERED_STATE_V1"
            or record.get("ordinal") != ordinal
            or record.get("stage") != name
            or record.get("previous_receipt_sha256") != previous_sha256
            or record.get("payload_sha256")
            != hashlib.sha256(canonical_json_bytes(record.get("payload"))).hexdigest()
        ):
            raise ValueError("old ordered receipt JSON or predecessor chain drifted")
        previous_sha256 = hashlib.sha256(payload).hexdigest()
        parsed_receipts.append(record)
        receipt_identities.append(identity)

    successor_names = tuple(
        _sequence(
            failure.get("expected_absent_successor_state_names"),
            "old absent successor state names",
        )
    )
    if (
        len(successor_names) != 8
        or len(set(successor_names)) != 8
        or any(
            not isinstance(name, str) or _SAFE_STATE_NAME.fullmatch(name) is None
            for name in successor_names
        )
    ):
        raise ValueError("old absent successor state inventory is malformed")
    for ordinal, name in enumerate(successor_names, start=8):
        _assert_absent(
            data_root,
            f"{namespace}/ordered-state-receipts/{ordinal:02d}-{name}.json",
            label=f"old successor receipt {name}",
        )

    seal_identity, seal_payload = _regular_file(
        data_root,
        failure.get("heuristic_local_seal"),
        label="old heuristic local seal",
    )
    seal = _strict_json(seal_payload, label="old heuristic local seal")
    inventory = seal.get("inventory")
    if (
        seal_payload != base_runner.pretty_json_bytes(seal)
        or set(seal)
        != {
            "schema_version",
            "protocol_id",
            "status",
            "inventory",
            "inventory_sha256",
            "label_access_authorized",
        }
        or seal.get("schema_version") != base_runner.SCHEMA_VERSION
        or seal.get("protocol_id") != base_runner.PROTOCOL_ID
        or seal.get("status") != base_runner.HEURISTIC_SEAL_STATUS
        or seal.get("label_access_authorized") is not False
        or not isinstance(inventory, list)
        or len(inventory) != 8
        or seal.get("inventory_sha256")
        != hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
    ):
        raise ValueError("old heuristic local seal schema or inventory drifted")
    heuristic_receipt_payload = _mapping(
        parsed_receipts[-1].get("payload"), "heuristic receipt payload"
    )
    if (
        heuristic_receipt_payload.get("label_blind_file_count") != 8
        or heuristic_receipt_payload.get("label_blind_inventory_sha256")
        != seal["inventory_sha256"]
        or heuristic_receipt_payload.get("label_access_authorized") is not False
    ):
        raise ValueError("heuristic receipt and standalone seal contradict")

    artifact_records, artifact_payload = _artifact_inventory(
        data_root,
        _mapping(failure.get("artifact_tree"), "old artifact tree"),
    )
    artifact_by_path = {item["path"]: item for item in artifact_records}
    for raw in inventory:
        item = _mapping(raw, "sealed label-blind file")
        if set(item) != {"path", "sha256", "size_bytes"}:
            raise ValueError("sealed label-blind file schema drifted")
        artifact = artifact_by_path.get(item.get("path"))
        if (
            artifact is None
            or artifact["sha256"] != item.get("sha256")
            or artifact["size_bytes"] != item.get("size_bytes")
            or artifact["mode"] != 0o444
        ):
            raise ValueError("heuristic seal and artifact bytes contradict")

    for canonical in _sequence(
        failure.get("expected_absent_standalone_files"),
        "old absent standalone files",
    ):
        _assert_absent(data_root, canonical, label="old standalone successor file")

    run = _mapping(failure.get("run_evidence"), "old run evidence")
    if set(run) != {"log", "started", "exit"}:
        raise ValueError("old run evidence must contain log/start/exit only")
    run_identities = {
        name: _regular_file(data_root, run[name], label=f"old run {name}")[0]
        for name in ("log", "started", "exit")
    }
    runtime_identity = _regular_file(
        data_root,
        failure.get("old_runtime_receipt"),
        label="old Docker runtime receipt",
    )[0]
    return {
        "status": "VALID_RETAINED_GATE_V1_FRESH16_INVENTORY_REPAIR_FAILURE",
        "old_state_namespace": str(namespace),
        "ordered_receipt_state_names": list(_RETAINED_STATE_NAMES),
        "ordered_receipt_count": len(receipt_identities),
        "absent_successor_receipt_count": len(successor_names),
        "heuristic_local_seal": dict(seal_identity),
        "sealed_label_blind_file_count": len(inventory),
        "artifact_file_count": len(artifact_records),
        "artifact_total_bytes": sum(item["size_bytes"] for item in artifact_records),
        "artifact_inventory_sha256": hashlib.sha256(artifact_payload).hexdigest(),
        "old_runtime_receipt": dict(runtime_identity),
        "run_evidence": {
            name: dict(record) for name, record in run_identities.items()
        },
        "fresh_label_semantic_decode_count": 0,
        "label_access_claim_count": 0,
        "primary_report_count": 0,
        "remote_mutation_count": 0,
    }


def validate_new_run_roots_absent(contract: Any, data_root: Path) -> Mapping[str, Any]:
    local = _mapping(
        _mapping(getattr(contract, "data", None), "repair contract data").get(
            "local_first_state_machine"
        ),
        "local state machine",
    )
    artifact = local.get("artifact_directory")
    state = f"{local.get('state_directory')}/{local.get('execution_namespace')}"
    _assert_absent(data_root, artifact, label="new artifact root")
    _assert_absent(data_root, state, label="new state root")
    return {
        "new_artifact_root": artifact,
        "new_state_root": state,
        "both_absent": True,
        "file_write_count": 0,
    }


def _validate_hf_owner(api: Any) -> Mapping[str, Any]:
    identity = _mapping(api.whoami(), "Hugging Face whoami")
    if identity.get("name") != "gavinlaw":
        raise ValueError("Hugging Face token owner must be gavinlaw")
    auth = _mapping(identity.get("auth"), "Hugging Face auth identity")
    access_token = _mapping(
        auth.get("accessToken"), "Hugging Face access-token identity"
    )
    if auth.get("type") != "access_token" or access_token.get("role") != "write":
        raise ValueError(
            "Hugging Face token must grant owner-wide write access before "
            "private-repository absence can be trusted"
        )
    return {
        "owner": "gavinlaw",
        "access_token_role": "write",
        "authenticated": True,
        "remote_read_count": 1,
        "remote_mutation_count": 0,
    }


def _assert_hf_repo_absent(
    api: Any,
    destination: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    repo = destination.get("repo")
    repo_type = destination.get("repo_type")
    if (
        not isinstance(repo, str)
        or repo.count("/") != 1
        or repo_type not in {"dataset", "model"}
    ):
        raise ValueError(f"{label} identity is malformed")
    try:
        api.repo_info(repo, repo_type=repo_type)
    except Exception as error:
        if error.__class__.__name__ != "RepositoryNotFoundError":
            raise
    else:
        raise ValueError(f"{label} must remain absent")
    return {
        "repo": repo,
        "repo_type": repo_type,
        "repository_existed": False,
        "remote_read_count": 1,
        "remote_mutation_count": 0,
    }


def validate_destinations_before_delegate(
    contract: Any,
    api: Any,
    *,
    require_new_absent: bool,
) -> Mapping[str, Any]:
    owner = _validate_hf_owner(api)
    failure = _parent_failure(contract)
    old = tuple(
        _mapping(item, "old destination")
        for item in _sequence(failure.get("old_destinations"), "old destinations")
    )
    if len(old) != 2:
        raise ValueError("old destination inventory must contain exactly two repos")
    old_results = [
        _assert_hf_repo_absent(api, item, label=f"abandoned destination {index}")
        for index, item in enumerate(old)
    ]
    new_result = None
    if require_new_absent:
        new_result = _assert_hf_repo_absent(
            api,
            _mapping(getattr(contract, "destination", None), "new destination"),
            label="claim-repair destination",
        )
    return {
        "hf_identity": dict(owner),
        "abandoned_destinations": [dict(item) for item in old_results],
        "new_destination_before_run": (
            dict(new_result) if new_result is not None else None
        ),
        "remote_mutation_count": 0,
    }


def execute_fresh16_claim_serialization_repair(
    *,
    mode: str,
    contract: Any,
    api: Any,
    download_fn: Callable[..., str],
    operation_factory: Callable[..., Any],
    expected_execution_b_git_commit: str,
    data_root: Path,
    fresh_download_parent: Path,
    model_dir: Path,
    snapshot_manifest: Path,
    docker_inspect_receipt: Path | None,
    devices: Sequence[str],
    gpu_uuids: Sequence[str],
    original_failure_validator: Callable[[Any, Path], Mapping[str, Any]] | None = None,
    inventory_failure_validator: Callable[[Any, Path], Mapping[str, Any]] | None = None,
    new_roots_validator: Callable[[Any, Path], Mapping[str, Any]] | None = None,
    destination_validator: Callable[..., Mapping[str, Any]] | None = None,
    execute_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    if mode not in {"run", "validate"}:
        raise ValueError("fresh16 claim repair mode must be run or validate")
    original_failure_validator = original_failure_validator or (
        lambda checked, root: validate_retained_v1_failure(checked.parent, root)
    )
    inventory_failure_validator = (
        inventory_failure_validator or validate_retained_inventory_repair_failure
    )
    new_roots_validator = new_roots_validator or validate_new_run_roots_absent
    destination_validator = destination_validator or validate_destinations_before_delegate
    execute_fn = execute_fn or base_runner.execute_fresh16_evaluation

    original = original_failure_validator(contract, data_root)
    inventory = inventory_failure_validator(contract, data_root)
    roots = new_roots_validator(contract, data_root) if mode == "run" else None
    destinations = destination_validator(
        contract,
        api,
        require_new_absent=mode == "run",
    )
    preflight = {
        "retained_original_v1_failure": dict(original),
        "retained_inventory_repair_failure": dict(inventory),
        "new_run_roots_before_run": dict(roots) if roots is not None else None,
        "remote_destinations": dict(destinations),
        "fresh_semantic_access_before_delegate": 0,
        "label_semantic_access_before_delegate": 0,
        "remote_mutation_before_delegate": 0,
    }
    result = execute_fn(
        mode=mode,
        contract=contract,
        api=api,
        download_fn=download_fn,
        operation_factory=operation_factory,
        expected_execution_b_git_commit=expected_execution_b_git_commit,
        data_root=data_root,
        fresh_download_parent=fresh_download_parent,
        model_dir=model_dir,
        snapshot_manifest=snapshot_manifest,
        docker_inspect_receipt=docker_inspect_receipt,
        devices=devices,
        gpu_uuids=gpu_uuids,
        execution_preflight=preflight,
    )
    return {**result, "claim_serialization_repair_preflight": preflight}


__all__ = [
    "capture_docker_inspect_receipt",
    "execute_fresh16_claim_serialization_repair",
    "materialize_runner_freeze",
    "validate_destinations_before_delegate",
    "validate_execution_b_source",
    "validate_new_run_roots_absent",
    "validate_retained_inventory_repair_failure",
    "validate_retained_v1_failure",
    "validate_source_a",
]
