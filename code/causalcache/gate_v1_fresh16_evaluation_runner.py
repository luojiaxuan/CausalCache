"""Execution-B runner for the sealed fresh-16 primary gate evaluation.

The module intentionally keeps all remote and CUDA work behind injected callables.
Source-A validation imports this file but performs neither remote access nor writes.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import multiprocessing
import os
import platform
import re
import socket
import stat
import statistics
import subprocess
import sys
import tarfile
import tempfile
from types import MappingProxyType
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.gate_v1_data import (
    DEPLOYMENT_BUDGET,
    GateState,
    join_feature_and_label_states,
    select_conditional,
    select_independent,
)
from causalcache.gate_v1_evaluation import (
    HEURISTIC_ORDER,
    _trajectory_deltas,
    _trajectory_equal_mean,
    _selector_records_from_sealed_decisions,
    _validate_frozen_primary_report,
    canonical_report_sha256,
    evaluate_primary_slice_from_sealed_decisions,
    paired_bootstrap_lower,
)
from causalcache.gate_v1_formal_train import load_safetensors_checkpoint
from causalcache.gate_v1_fresh16 import (
    FRESH_IMAGE_COUNT,
    FRESH_STATE_COUNT,
    HEURISTIC_STATUS,
    LEARNED_STATUS,
    Fresh16ImageRequirement,
    Fresh16LabelBlindBundle,
    Fresh16PolicyWorkItem,
    build_fresh16_image_plan,
    build_fresh16_label_blind_bundle,
    combine_policy_score_records,
    decode_policy_work_item_images,
    feature_states_jsonl_bytes,
    join_fresh16_evaluation_states,
    jsonl_score_records_bytes,
    label_states_jsonl_bytes,
    materialize_fresh16_image_features,
    policy_selection_record,
    read_feature_states_jsonl,
    read_label_states_jsonl,
    read_selection_artifact,
    selection_artifact_bytes,
)
from causalcache.gate_v1_fresh16_data import (
    Fresh16LabelSlice,
    Fresh16TrajectorySlice,
    decode_fresh16_labels,
    decode_fresh16_trajectories,
    verify_fresh16_raw_state_transport,
    verify_fresh16_trajectory_transport,
)
from causalcache.gate_v1_fresh16_evaluation_contract import (
    CANONICAL_CONFIG_PATH,
    PAYLOAD_TARGETS,
    PROTOCOL_ID,
    REPORT_TARGETS,
    RUNNER_FREEZE_B_PATH,
    SCHEMA_VERSION,
    Fresh16EvaluationContract,
    canonical_json_bytes,
    pretty_json_bytes,
    sha256_bytes,
)
from causalcache.gate_v1_fresh16_policy_vision import (
    Fresh16GUIOwlV22VisionFeatureRuntime,
)
from causalcache.gate_v1_fresh16_heuristics import (
    policy_vision_similarity_selection,
    recent_selection,
)
from causalcache.gate_v1_provenance import (
    GATE_V1_CONFIG_SHA256,
    ArtifactBinding,
    FrozenEnsembleProvenance,
    HeuristicArtifactProvenance,
    canonical_selection_sha256,
    frozen_ensemble_provenance_from_manifest,
)
from causalcache.gate_v1_training import FittedEnsemble, ensemble_score, model_score
from causalcache.policy.gui_owl_v2_2_vision_runtime import (
    GUI_OWL_V2_2_EAGER_EXPECTED_GPU_NAME,
    _canonical_gpu_uuid,
    _validated_gpu_identity,
)
from causalcache.policy.gui_owl_v2_vision import verify_frozen_vision_runtime
from causalcache.restoration_v2_2_label_table import primary_exact_subset_oracle


SOURCE_VALIDATION_STATUS = "VALID_GATE_V1_FRESH16_EVALUATION_SOURCE_A_V1"
RUNNER_FREEZE_STATUS = "FROZEN_GATE_V1_FRESH16_EVALUATION_EXECUTION_B_V1"
HEURISTIC_SEAL_STATUS = "SEALED_GATE_V1_FRESH16_LABEL_BLIND_PAYLOAD_V1"
LABEL_CLAIM_STATUS = "CLAIMED_GATE_V1_FRESH16_LABEL_ACCESS_V1"
PAYLOAD_PUBLICATION_STATUS = "PUBLISHED_GATE_V1_FRESH16_PAYLOAD_V1"
FINAL_STATUS = "COMPLETED_GATE_V1_FRESH16_PRIMARY_EVALUATION_V1"
VALIDATE_STATUS = "REVALIDATED_GATE_V1_FRESH16_PRIMARY_EVALUATION_V1"

LABEL_BLIND_PAYLOAD_TARGETS = tuple(
    path for path in PAYLOAD_TARGETS if not path.endswith("label-states-v1.jsonl")
)
LABEL_PAYLOAD_PATH = "fresh16-eval/v1/caches/label-states-v1.jsonl"
FEATURE_PAYLOAD_PATH = "fresh16-eval/v1/caches/feature-states-v1.jsonl"
HEURISTIC_PATHS = {
    "dynamic_recent": "fresh16-eval/v1/heuristics/dynamic-recent-v1.json",
    "ocr_rgb_v2": "fresh16-eval/v1/heuristics/ocr-rgb-v2.json",
    "policy_vision_v3": "fresh16-eval/v1/heuristics/policy-vision-v3.json",
}
LEARNED_PATHS = {
    "conditional": "fresh16-eval/v1/learned/conditional-v1.json",
    "independent": "fresh16-eval/v1/learned/independent-v1.json",
}
OCR_SCORE_PATH = "fresh16-eval/v1/scores/ocr-rgb-v2.jsonl"
POLICY_SCORE_PATH = "fresh16-eval/v1/scores/policy-vision-v3.jsonl"
STATE_RECORD_PATH = "fresh16-eval/v1/reports/primary-state-records-v1.jsonl"
PRIMARY_REPORT_PATH = "fresh16-eval/v1/reports/primary-report-v1.json"
RUN_MANIFEST_PATH = "fresh16-eval/v1/manifests/run-manifest-v1.json"
BUNDLE_MANIFEST_PATH = "fresh16-eval/v1/manifests/bundle-manifest-v1.json"

_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ALLOWED_REPO_BASE_FILES = frozenset({".gitattributes", "README.md"})
_HYPER_H200_HOSTNAMES = frozenset(
    {"node-radixark-16-0000", "node-radixark-16-0001"}
)
_HEURISTIC_SEAL_TOKEN = object()
_LABEL_CLAIM_TOKEN = object()


@dataclass(frozen=True)
class SourceIdentity:
    head: str
    remote_main: str
    branch: str
    origin_url: str
    source_inventory: tuple[Mapping[str, Any], ...]
    loaded_module_inventory: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class VerifiedRemoteFile:
    path: str
    local_path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class FormalEnsembles:
    conditional: FittedEnsemble
    conditional_provenance: FrozenEnsembleProvenance
    independent: FittedEnsemble
    independent_provenance: FrozenEnsembleProvenance
    checkpoint_load_count: int


@dataclass(frozen=True)
class LearnedDecisionArtifacts:
    conditional: Mapping[str, Mapping[str, Any]]
    independent: Mapping[str, Mapping[str, Any]]

    def family(self, name: str) -> Mapping[str, Mapping[str, Any]]:
        if name == "conditional":
            return self.conditional
        if name == "independent":
            return self.independent
        raise ValueError("learned decision family is not frozen")


@dataclass(frozen=True)
class FormalDecisionProvenance:
    conditional: FrozenEnsembleProvenance
    independent: FrozenEnsembleProvenance


@dataclass(frozen=True)
class RuntimeIdentity:
    payload: Mapping[str, Any]
    receipt: Mapping[str, Any]
    receipt_sha256: str


@dataclass(frozen=True)
class PolicyWorkerRequest:
    worker_id: str
    work_items: tuple[Fresh16PolicyWorkItem, ...]
    sentinel: Fresh16PolicyWorkItem | None
    image_root: str
    model_dir: str
    snapshot_manifest: str
    device: str
    expected_gpu_uuid: str
    gpu_uuid_type_profile: str
    image_processor_size_profile: str


@dataclass(frozen=True)
class PolicyWorkerResult:
    worker_id: str
    records: tuple[Mapping[str, Any], ...]
    sentinel_record: Mapping[str, Any] | None
    operation_counts: Mapping[str, int]
    runtime_metadata: Mapping[str, Any]


class HeuristicBytesSeal:
    """Opaque binding over the exact eight pre-label payload byte strings."""

    __slots__ = ("_files", "_inventory", "inventory_sha256", "_token")

    def __init__(
        self,
        files: Mapping[str, bytes],
        inventory: tuple[Mapping[str, Any], ...],
        *,
        _token: object,
    ) -> None:
        if _token is not _HEURISTIC_SEAL_TOKEN:
            raise TypeError("heuristic seals must be constructed by seal_label_blind_files")
        self._files = MappingProxyType(dict(files))
        self._inventory = tuple(MappingProxyType(dict(item)) for item in inventory)
        self.inventory_sha256 = sha256_bytes(canonical_json_bytes(inventory))
        self._token = _token

    @property
    def files(self) -> Mapping[str, bytes]:
        return self._files

    @property
    def inventory(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(item) for item in self._inventory)


class LabelAccessClaim:
    """Opaque authorization proving label-blind bytes were fixed first."""

    __slots__ = ("heuristic_inventory_sha256", "_claim", "_token")

    def __init__(
        self,
        heuristic_inventory_sha256: str,
        claim: Mapping[str, Any],
        *,
        _token: object,
    ) -> None:
        if _token is not _LABEL_CLAIM_TOKEN:
            raise TypeError("label claims must be constructed from a heuristic seal")
        self.heuristic_inventory_sha256 = heuristic_inventory_sha256
        self._claim = MappingProxyType(
            json.loads(canonical_json_bytes(dict(claim)))
        )
        self._token = _token

    @property
    def claim(self) -> Mapping[str, Any]:
        return json.loads(canonical_json_bytes(dict(self._claim)))


class DurableStateChain:
    """O_EXCL/identical predecessor-hash chain over the configured 16 states."""

    def __init__(self, root: Path, ordered_states: Sequence[str]) -> None:
        self.root = root / "ordered-state-receipts"
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or self.root.resolve(strict=True) != self.root.absolute():
            raise ValueError("ordered-state receipt root must be a real directory")
        self.ordered_states = tuple(ordered_states)
        if (
            len(self.ordered_states) != 16
            or len(set(self.ordered_states)) != 16
            or any(not re.fullmatch(r"[a-z0-9_]+", item) for item in self.ordered_states)
        ):
            raise ValueError("ordered-state inventory must contain 16 unique safe names")

    def _path(self, index: int, stage: str) -> Path:
        return self.root / f"{index:02d}-{stage}.json"

    def _expected_record(
        self,
        index: int,
        stage: str,
        payload: Mapping[str, Any],
        previous_sha256: str | None,
    ) -> Mapping[str, Any]:
        normalized = json.loads(canonical_json_bytes(payload))
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETED_GATE_V1_FRESH16_ORDERED_STATE_V1",
            "ordinal": index,
            "stage": stage,
            "previous_receipt_sha256": previous_sha256,
            "payload_sha256": sha256_bytes(canonical_json_bytes(normalized)),
            "payload": normalized,
        }

    def verify_prefix(self) -> tuple[Mapping[str, Any], ...]:
        records = []
        previous = None
        missing_seen = False
        for index, stage in enumerate(self.ordered_states):
            path = self._path(index, stage)
            exists = path.exists() or path.is_symlink()
            if not exists:
                missing_seen = True
                continue
            if missing_seen:
                raise ValueError("ordered-state chain contains a gap")
            payload = _regular_file_bytes(path, label=f"state receipt {stage}", mode=0o600)
            record = _strict_json(payload, label=f"state receipt {stage}")
            if (
                payload != pretty_json_bytes(record)
                or record.get("ordinal") != index
                or record.get("stage") != stage
                or record.get("previous_receipt_sha256") != previous
                or record.get("payload_sha256")
                != sha256_bytes(canonical_json_bytes(record.get("payload")))
            ):
                raise ValueError("ordered-state receipt or predecessor chain drifted")
            previous = sha256_bytes(payload)
            records.append(record)
        return tuple(records)

    def append(self, stage: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if stage not in self.ordered_states:
            raise ValueError("ordered-state stage is not configured")
        index = self.ordered_states.index(stage)
        existing = self.verify_prefix()
        if index > len(existing):
            raise ValueError("ordered-state append attempted to skip a predecessor")
        previous_sha = (
            sha256_bytes(
                _regular_file_bytes(
                    self._path(index - 1, self.ordered_states[index - 1]),
                    label="previous state receipt",
                    mode=0o600,
                )
            )
            if index
            else None
        )
        expected = self._expected_record(index, stage, payload, previous_sha)
        if index < len(existing):
            if existing[index] != expected:
                raise ValueError("pre-existing ordered-state receipt differs")
            return existing[index]
        if index != len(existing):
            raise ValueError("ordered-state append index drifted")
        _exclusive_or_identical(
            self._path(index, stage),
            pretty_json_bytes(expected),
            mode=0o600,
        )
        replay = self.verify_prefix()
        if len(replay) != index + 1 or replay[-1] != expected:
            raise ValueError("ordered-state receipt readback drifted")
        return expected

    def verify_complete(self) -> tuple[Mapping[str, Any], ...]:
        result = self.verify_prefix()
        if len(result) != len(self.ordered_states):
            raise ValueError("ordered-state chain is incomplete")
        return result


def _strict_json_value(payload: bytes, *, label: str) -> Any:
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
    return value


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    value = _strict_json_value(payload, label=label)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be one JSON object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


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


def _regular_file_bytes(
    path: Path,
    *,
    label: str,
    mode: int | None = None,
) -> bytes:
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
        if mode is not None and stat.S_IMODE(before.st_mode) != mode:
            raise ValueError(f"{label} mode drifted")
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


def _exclusive_or_identical(path: Path, payload: bytes, *, mode: int) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve(strict=True) != path.parent.absolute():
        raise ValueError(f"file parent traverses a symlink: {path.parent}")
    if path.exists() or path.is_symlink():
        if _regular_file_bytes(path, label=str(path), mode=mode) != payload:
            raise ValueError(f"pre-existing file differs: {path}")
        return False
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive write made no progress")
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    observed = _regular_file_bytes(path, label=str(path), mode=mode)
    if observed != payload:
        raise ValueError(f"exclusive file readback differs: {path}")
    return True


def _git(root: Path, *arguments: str) -> bytes:
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


def _source_paths(contract: Fresh16EvaluationContract) -> tuple[str, ...]:
    raw = contract.source.get("required_source_a_paths")
    if not isinstance(raw, list) or not raw:
        raise ValueError("required Source-A paths are missing")
    paths = tuple(_safe_relative(item, "Source-A path") for item in raw)
    if len(paths) != len(set(paths)) or _runner_freeze_path(contract) in paths:
        raise ValueError("Source-A path inventory is duplicated or includes B")
    return paths


def _runner_freeze_path(contract: Fresh16EvaluationContract) -> str:
    """Return the Source-A-declared B path, preserving the v1 default."""
    raw = contract.source.get("execution_b_runner_freeze")
    if raw is None:
        return RUNNER_FREEZE_B_PATH
    runner = _mapping(raw, "execution-B runner freeze")
    path = _safe_relative(runner.get("path"), "execution-B runner-freeze path")
    if path != RUNNER_FREEZE_B_PATH and not path.startswith(
        "code/configs/causalcache_gate_v1_fresh16_"
    ):
        raise ValueError("fresh16 runner-freeze path escaped the protocol namespace")
    return path


def _source_inventory(
    root: Path,
    commit: str,
    paths: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    records = []
    for relative in paths:
        payload = _regular_file_bytes(root / relative, label=relative)
        if payload != _git(root, "show", f"{commit}:{relative}"):
            raise ValueError(f"source blob differs from commit: {relative}")
        records.append(
            {"path": relative, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        )
    return tuple(records)


def _loaded_module_inventory(root: Path, commit: str) -> tuple[Mapping[str, Any], ...]:
    code_root = root / "code"
    records = []
    for name, module in sorted(sys.modules.items()):
        if not (
            name == "causalcache"
            or name.startswith("causalcache.")
            or name == "scripts"
            or name.startswith("scripts.")
        ):
            continue
        source = getattr(module, "__file__", None)
        if not isinstance(source, str):
            raise ValueError(f"loaded project module lacks source: {name}")
        path = Path(source)
        if not path.is_absolute() or path.suffix != ".py":
            raise ValueError(f"loaded project module is not canonical source: {name}")
        try:
            relative = (Path("code") / path.relative_to(code_root)).as_posix()
        except ValueError as error:
            raise ValueError(f"loaded module escaped repository: {name}") from error
        payload = _regular_file_bytes(path, label=name)
        if payload != _git(root, "show", f"{commit}:{relative}"):
            raise ValueError(f"loaded module differs from commit: {name}")
        records.append(
            {
                "module": name,
                "path": relative,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    if not records:
        raise ValueError("loaded module inventory is empty")
    return tuple(records)


def _live_remote_main(root: Path, remote: str) -> str:
    output = _git(root, "ls-remote", "--exit-code", remote, "refs/heads/main").decode(
        "ascii"
    )
    suffix = "\trefs/heads/main"
    lines = output.splitlines()
    if len(lines) != 1 or not lines[0].endswith(suffix):
        raise ValueError("canonical remote main response drifted")
    commit = lines[0][: -len(suffix)]
    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("canonical remote main is not one commit")
    return commit


def validate_clean_pushed_source(
    contract: Fresh16EvaluationContract,
    *,
    expected_commit: str | None,
    require_live_remote: bool,
) -> SourceIdentity:
    root = contract.repository_root
    source = contract.source
    branch = source["branch"]
    remote = source["origin_name"]
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    tracking = _git(root, "rev-parse", f"{remote}/{branch}").decode().strip()
    current_branch = _git(root, "branch", "--show-current").decode().strip()
    origin_url = _git(root, "remote", "get-url", remote).decode().strip()
    remote_main = _live_remote_main(root, remote) if require_live_remote else tracking
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if (
        _COMMIT.fullmatch(head) is None
        or (expected_commit is not None and head != expected_commit)
        or head != tracking
        or head != remote_main
        or current_branch != branch
        or origin_url != source["origin_url"]
        or dirty
    ):
        raise ValueError("fresh16 source must be clean pushed canonical main")
    return SourceIdentity(
        head=head,
        remote_main=remote_main,
        branch=current_branch,
        origin_url=origin_url,
        source_inventory=_source_inventory(root, head, _source_paths(contract)),
        loaded_module_inventory=_loaded_module_inventory(root, head),
    )


def _source_record(source: SourceIdentity) -> Mapping[str, Any]:
    return {
        "git_commit": source.head,
        "remote_main_git_commit": source.remote_main,
        "branch": source.branch,
        "origin_url": source.origin_url,
        "source_inventory": list(source.source_inventory),
        "source_inventory_sha256": sha256_bytes(
            canonical_json_bytes(source.source_inventory)
        ),
        "loaded_module_inventory": list(source.loaded_module_inventory),
        "loaded_module_inventory_sha256": sha256_bytes(
            canonical_json_bytes(source.loaded_module_inventory)
        ),
    }


def validate_source_a(
    contract: Fresh16EvaluationContract,
    *,
    expected_source_a_git_commit: str | None = None,
) -> Mapping[str, Any]:
    runner_freeze_path = _runner_freeze_path(contract)
    runner = contract.repository_root / runner_freeze_path
    if runner.exists() or runner.is_symlink():
        raise ValueError("execution-B runner freeze must be absent from Source-A")
    source = validate_clean_pushed_source(
        contract,
        expected_commit=expected_source_a_git_commit,
        require_live_remote=False,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": SOURCE_VALIDATION_STATUS,
        "contract_sha256": contract.sha256,
        "source_a": _source_record(source),
        "evaluation_executed": False,
        "execution_authorized": False,
        "remote_mutation_count": 0,
    }


def _runner_freeze_payload(
    contract: Fresh16EvaluationContract,
    source: SourceIdentity,
) -> Mapping[str, Any]:
    runner_freeze_path = _runner_freeze_path(contract)
    prerequisites = list(contract.source["git_prerequisites"])
    paths = list(_source_paths(contract))
    inventory_sha = sha256_bytes(canonical_json_bytes(source.source_inventory))
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": RUNNER_FREEZE_STATUS,
        "contract_sha256": contract.sha256,
        "source_a_git_commit": source.head,
        "execution_b_required_unique_diff": [runner_freeze_path],
        "required_source_a_paths": paths,
        "required_source_a_paths_sha256": sha256_bytes(canonical_json_bytes(paths)),
        "git_prerequisites": prerequisites,
        "git_prerequisites_sha256": sha256_bytes(canonical_json_bytes(prerequisites)),
        "source_blob_inventory": list(source.source_inventory),
        "source_blob_inventory_sha256": inventory_sha,
        "source_a_inventory_sha256": inventory_sha,
        "loaded_module_inventory": list(source.loaded_module_inventory),
        "loaded_module_inventory_sha256": sha256_bytes(
            canonical_json_bytes(source.loaded_module_inventory)
        ),
        "execution_authorized_after_clean_pushed_b_only": True,
    }


def materialize_runner_freeze(
    contract: Fresh16EvaluationContract,
    *,
    expected_source_a_git_commit: str | None = None,
) -> Mapping[str, Any]:
    validation = validate_source_a(
        contract,
        expected_source_a_git_commit=expected_source_a_git_commit,
    )
    record = validation["source_a"]
    source = SourceIdentity(
        head=record["git_commit"],
        remote_main=record["remote_main_git_commit"],
        branch=record["branch"],
        origin_url=record["origin_url"],
        source_inventory=tuple(record["source_inventory"]),
        loaded_module_inventory=tuple(record["loaded_module_inventory"]),
    )
    payload = _runner_freeze_payload(contract, source)
    runner_freeze_path = _runner_freeze_path(contract)
    path = contract.repository_root / runner_freeze_path
    _exclusive_or_identical(path, pretty_json_bytes(payload), mode=0o644)
    root = contract.repository_root
    status = tuple(
        line
        for line in _git(root, "status", "--porcelain=v1", "--untracked-files=all")
        .decode("utf-8")
        .splitlines()
        if line
    )
    if status != (f"?? {runner_freeze_path}",):
        raise ValueError("runner freeze is not the unique Source-A worktree diff")
    return payload


def load_runner_freeze(contract: Fresh16EvaluationContract) -> Mapping[str, Any]:
    runner_freeze_path = _runner_freeze_path(contract)
    payload = _regular_file_bytes(
        contract.repository_root / runner_freeze_path,
        label="execution-B runner freeze",
        mode=0o644,
    )
    value = _strict_json(payload, label="execution-B runner freeze")
    if payload != pretty_json_bytes(value):
        raise ValueError("execution-B runner freeze is not canonical pretty JSON")
    paths = list(_source_paths(contract))
    prerequisites = list(contract.source["git_prerequisites"])
    inventory = value.get("source_blob_inventory")
    modules = value.get("loaded_module_inventory")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "contract_sha256",
        "source_a_git_commit",
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
        "execution_authorized_after_clean_pushed_b_only",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != RUNNER_FREEZE_STATUS
        or value.get("contract_sha256") != contract.sha256
        or value.get("execution_b_required_unique_diff") != [runner_freeze_path]
        or value.get("required_source_a_paths") != paths
        or value.get("required_source_a_paths_sha256")
        != sha256_bytes(canonical_json_bytes(paths))
        or value.get("git_prerequisites") != prerequisites
        or value.get("git_prerequisites_sha256")
        != sha256_bytes(canonical_json_bytes(prerequisites))
        or not isinstance(inventory, list)
        or value.get("source_blob_inventory_sha256")
        != sha256_bytes(canonical_json_bytes(inventory))
        or value.get("source_a_inventory_sha256")
        != sha256_bytes(canonical_json_bytes(inventory))
        or not isinstance(modules, list)
        or value.get("loaded_module_inventory_sha256")
        != sha256_bytes(canonical_json_bytes(modules))
        or value.get("execution_authorized_after_clean_pushed_b_only") is not True
    ):
        raise ValueError("execution-B runner freeze identity drifted")
    return value


def validate_execution_b_source(
    contract: Fresh16EvaluationContract,
    *,
    expected_execution_b_git_commit: str,
) -> SourceIdentity:
    freeze = load_runner_freeze(contract)
    source = validate_clean_pushed_source(
        contract,
        expected_commit=expected_execution_b_git_commit,
        require_live_remote=True,
    )
    source_a = freeze.get("source_a_git_commit")
    if not isinstance(source_a, str) or _COMMIT.fullmatch(source_a) is None:
        raise ValueError("runner freeze Source-A commit is invalid")
    parents = (
        _git(contract.repository_root, "rev-list", "--parents", "-n", "1", source.head)
        .decode("ascii")
        .strip()
        .split()
    )
    if parents != [source.head, source_a]:
        raise ValueError("Execution-B must be the direct single-parent child of Source-A")
    changed = tuple(
        line
        for line in _git(
            contract.repository_root, "diff", "--name-only", source_a, source.head
        )
        .decode("utf-8")
        .splitlines()
        if line
    )
    if changed != (_runner_freeze_path(contract),):
        raise ValueError("Execution-B differs from Source-A outside runner freeze")
    if source.source_inventory != tuple(freeze["source_blob_inventory"]):
        raise ValueError("Execution-B source blobs differ from Source-A freeze")
    if {item["module"]: item for item in source.loaded_module_inventory} != {
        item["module"]: item for item in freeze["loaded_module_inventory"]
    }:
        raise ValueError("Execution-B loaded modules differ from Source-A freeze")
    return source


def capture_docker_inspect_receipt(
    contract: Fresh16EvaluationContract,
    *,
    source: SourceIdentity,
    container_name: str,
    host_data_root: Path,
) -> Mapping[str, Any]:
    specification = _mapping(
        contract.data["runtime_contract"]["docker_inspect_receipt"],
        "Docker inspect receipt specification",
    )
    capture_host = socket.gethostname()
    if (
        contract.data["runtime_contract"].get("host") != "hyper00_or_hyper01_h200"
        or capture_host not in _HYPER_H200_HOSTNAMES
    ):
        raise ValueError("fresh16 runtime receipt must be captured on Hyper00/01")
    if re.fullmatch(
        re.escape(specification["container_name_prefix"]) + r"[0-9]{8}",
        container_name,
    ) is None:
        raise ValueError("fresh16 container name differs from the timestamp form")
    completed = subprocess.run(
        ["docker", "inspect", "--type", "container", container_name],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ValueError("Docker inspect failed for the fresh16 container")
    decoded = _strict_json_value(completed.stdout, label="Docker inspect output")
    if not isinstance(decoded, list) or len(decoded) != 1:
        raise ValueError("Docker inspect must return exactly one container")
    inspect = _mapping(decoded[0], "Docker inspect record")
    host_config = _mapping(inspect.get("HostConfig"), "Docker HostConfig")
    config = _mapping(inspect.get("Config"), "Docker Config")
    state = _mapping(inspect.get("State"), "Docker State")
    container_id = inspect.get("Id")
    image_id = inspect.get("Image")
    hostname = config.get("Hostname")
    requests = host_config.get("DeviceRequests")
    if (
        not isinstance(container_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        or inspect.get("Name") != f"/{container_name}"
        or hostname != container_id[:12]
        or image_id != contract.data["runtime_contract"]["container_image_id"]
        or state.get("Running") is not True
        or host_config.get("Privileged") is not specification["privileged"]
        or host_config.get("Runtime") != specification["runtime"]
        or not isinstance(requests, list)
        or len(requests) != specification["gpu_device_request_count"]
    ):
        raise ValueError("fresh16 Docker runtime boundary drifted")
    request = _mapping(requests[0], "Docker GPU device request")
    device_ids = request.get("DeviceIDs")
    capabilities = request.get("Capabilities")
    if (
        not isinstance(device_ids, list)
        or len(device_ids) != specification["gpu_count"]
        or len(set(device_ids)) != specification["gpu_count"]
        or any(not isinstance(item, str) or not item for item in device_ids)
        or not isinstance(capabilities, list)
        or ["gpu"] not in capabilities
    ):
        raise ValueError("fresh16 Docker GPU request is not exact two-device access")
    mounts = inspect.get("Mounts")
    if not isinstance(mounts, list):
        raise ValueError("fresh16 Docker mounts are missing")
    matches = [
        _mapping(item, "Docker data mount")
        for item in mounts
        if isinstance(item, Mapping)
        and item.get("Destination") == specification["data_mount_destination"]
    ]
    root = host_data_root.resolve(strict=True)
    if host_data_root.is_symlink() or not stat.S_ISDIR(root.lstat().st_mode) or len(matches) != 1:
        raise ValueError("fresh16 Docker data root is unsafe or ambiguous")
    mount = matches[0]
    if (
        mount.get("Type") != "bind"
        or mount.get("RW") is not specification["data_mount_rw"]
        or not isinstance(mount.get("Source"), str)
        or Path(mount["Source"]).resolve(strict=True) != root
    ):
        raise ValueError("fresh16 Docker /data mount drifted")
    receipt = {
        "schema_version": specification["schema_version"],
        "protocol_id": PROTOCOL_ID,
        "status": specification["status"],
        "contract_sha256": contract.sha256,
        "source": _source_record(source),
        "capture_host": capture_host,
        "container": {
            "id": container_id,
            "name": container_name,
            "hostname": hostname,
            "image_id": image_id,
            "running": True,
        },
        "runtime": {
            "privileged": host_config["Privileged"],
            "runtime": host_config["Runtime"],
            "device_request_count": len(requests),
            "device_ids": list(device_ids),
            "gpu_count": len(device_ids),
        },
        "data_mount": {
            "type": "bind",
            "source": str(root),
            "destination": specification["data_mount_destination"],
            "rw": True,
        },
        "evaluation_executed": False,
        "execution_authorized": True,
    }
    receipt_path = _configured_under_data_root(
        host_data_root,
        specification["path"],
        label="Docker inspect receipt",
    )
    _exclusive_or_identical(
        receipt_path,
        pretty_json_bytes(receipt),
        mode=specification["mode"],
    )
    return {**receipt, "receipt_path": specification["path"]}


def validate_execution_runtime(
    contract: Fresh16EvaluationContract,
    *,
    source: SourceIdentity,
    data_root: Path,
    receipt_path: Path,
) -> RuntimeIdentity:
    runtime = _mapping(contract.data["runtime_contract"], "runtime contract")
    specification = _mapping(
        runtime["docker_inspect_receipt"], "Docker inspect receipt specification"
    )
    expected_path = _configured_under_data_root(
        data_root,
        specification["path"],
        label="Docker inspect receipt",
    )
    if receipt_path.absolute() != expected_path:
        raise ValueError("explicit Docker inspect receipt path drifted")
    receipt_payload = _regular_file_bytes(
        expected_path,
        label="Docker inspect receipt",
        mode=specification["mode"],
    )
    receipt = _strict_json(receipt_payload, label="Docker inspect receipt")
    container = _mapping(receipt.get("container"), "Docker receipt container")
    docker_runtime = _mapping(receipt.get("runtime"), "Docker receipt runtime")
    data_mount = _mapping(receipt.get("data_mount"), "Docker receipt data mount")
    expected_top_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "contract_sha256",
        "source",
        "capture_host",
        "container",
        "runtime",
        "data_mount",
        "evaluation_executed",
        "execution_authorized",
    }
    container_id = container.get("id")
    container_name = container.get("name")
    device_ids = docker_runtime.get("device_ids")
    mount_source = data_mount.get("source")
    if (
        receipt_payload != pretty_json_bytes(receipt)
        or set(receipt) != expected_top_keys
        or set(container) != {"id", "name", "hostname", "image_id", "running"}
        or set(docker_runtime)
        != {"privileged", "runtime", "device_request_count", "device_ids", "gpu_count"}
        or set(data_mount) != {"type", "source", "destination", "rw"}
        or receipt.get("schema_version") != specification["schema_version"]
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("status") != specification["status"]
        or receipt.get("contract_sha256") != contract.sha256
        or receipt.get("source") != _source_record(source)
        or receipt.get("evaluation_executed") is not False
        or receipt.get("execution_authorized") is not True
        or receipt.get("capture_host") not in _HYPER_H200_HOSTNAMES
        or runtime.get("host") != "hyper00_or_hyper01_h200"
        or not isinstance(container_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        or not isinstance(container_name, str)
        or re.fullmatch(
            re.escape(specification["container_name_prefix"]) + r"[0-9]{8}",
            container_name,
        )
        is None
        or container.get("hostname") != container_id[:12]
        or container.get("hostname") != socket.gethostname()
        or container.get("image_id") != runtime["container_image_id"]
        or container.get("running") is not True
        or docker_runtime.get("privileged") is not specification["privileged"]
        or docker_runtime.get("runtime") != specification["runtime"]
        or docker_runtime.get("device_request_count")
        != specification["gpu_device_request_count"]
        or docker_runtime.get("gpu_count") != specification["gpu_count"]
        or not isinstance(device_ids, list)
        or len(device_ids) != specification["gpu_count"]
        or len(set(device_ids)) != specification["gpu_count"]
        or any(not isinstance(item, str) or not item for item in device_ids)
        or data_mount.get("type") != "bind"
        or data_mount.get("destination") != specification["data_mount_destination"]
        or data_mount.get("rw") is not specification["data_mount_rw"]
        or not isinstance(mount_source, str)
        or not PurePosixPath(mount_source).is_absolute()
        or PurePosixPath(mount_source) == PurePosixPath("/")
    ):
        raise ValueError("fresh16 Docker inspect receipt identity drifted")
    observed_environment = {
        key: os.environ.get(key) for key in runtime["thread_environment"]
    }
    if (
        observed_environment != dict(runtime["thread_environment"])
        or platform.python_implementation() != runtime["python_implementation"]
        or platform.python_version() != runtime["python_version"]
        or platform.machine() != runtime["machine"]
    ):
        raise ValueError("fresh16 host, Python, or thread environment drifted")
    import importlib.metadata

    import huggingface_hub
    import safetensors
    import torch
    import transformers

    torch.set_num_threads(runtime["finalize_phase"]["torch_num_threads"])
    if torch.get_num_interop_threads() != runtime["finalize_phase"][
        "torch_num_interop_threads"
    ]:
        try:
            torch.set_num_interop_threads(
                runtime["finalize_phase"]["torch_num_interop_threads"]
            )
        except RuntimeError as error:
            raise ValueError("fresh16 torch inter-op thread count cannot be frozen") from error
    torch.use_deterministic_algorithms(True)
    libraries = {
        "torch_version": torch.__version__,
        "safetensors_version": safetensors.__version__,
        "huggingface_hub_version": huggingface_hub.__version__,
        "transformers_version": transformers.__version__,
        "pillow_version": importlib.metadata.version("Pillow"),
    }
    if (
        any(libraries[key] != runtime[key] for key in libraries)
        or torch.get_num_threads() != 1
        or torch.get_num_interop_threads() != 1
        or not torch.are_deterministic_algorithms_enabled()
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 2
    ):
        raise ValueError("fresh16 framework, CPU finalizer, or CUDA boundary drifted")
    payload = {
        "hostname": socket.gethostname(),
        "capture_host": receipt.get("capture_host"),
        "container": dict(container),
        "docker_runtime": dict(docker_runtime),
        "data_mount": dict(data_mount),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        **libraries,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": True,
        "torch_cuda_device_count": torch.cuda.device_count(),
        "thread_environment": observed_environment,
    }
    return RuntimeIdentity(
        payload=payload,
        receipt=MappingProxyType(dict(receipt)),
        receipt_sha256=sha256_bytes(receipt_payload),
    )


def verify_downloaded_file(
    path: Path,
    *,
    expected_path: str,
    expected_sha256: str,
    expected_size_bytes: int,
) -> VerifiedRemoteFile:
    relative = _safe_relative(expected_path, "downloaded file path")
    if _SHA256.fullmatch(expected_sha256) is None or type(expected_size_bytes) is not int:
        raise ValueError("downloaded file binding is malformed")
    payload = _regular_file_bytes(path, label=relative)
    if len(payload) != expected_size_bytes or sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"downloaded immutable bytes drifted: {relative}")
    return VerifiedRemoteFile(
        path=relative,
        local_path=path,
        sha256=expected_sha256,
        size_bytes=expected_size_bytes,
    )


def download_verified_files(
    *,
    download_fn: Callable[..., str],
    repo: str,
    repo_type: str,
    revision: str,
    records: Sequence[Mapping[str, Any]],
    local_root: Path,
) -> Mapping[str, VerifiedRemoteFile]:
    """Download only an explicit immutable file inventory and verify exact bytes."""
    if _COMMIT.fullmatch(revision) is None:
        raise ValueError("download revision must be one immutable HF commit")
    local_root.mkdir(parents=True, exist_ok=True)
    if local_root.is_symlink() or local_root.resolve(strict=True) != local_root.absolute():
        raise ValueError("download root must be a real canonical directory")
    result: dict[str, VerifiedRemoteFile] = {}
    for raw in records:
        record = _mapping(raw, "download record")
        relative = _safe_relative(record.get("path"), "download path")
        if relative in result:
            raise ValueError("download file inventory is duplicated")
        returned = Path(
            download_fn(
                repo_id=repo,
                repo_type=repo_type,
                filename=relative,
                revision=revision,
                local_dir=local_root,
                force_download=True,
            )
        )
        target = local_root / relative
        if not returned.is_absolute() or returned != target:
            raise ValueError("download function returned a noncanonical path")
        current = local_root
        for component in returned.relative_to(local_root).parts[:-1]:
            current /= component
            if not stat.S_ISDIR(current.lstat().st_mode):
                raise ValueError("download path traversed a symlink directory")
        result[relative] = verify_downloaded_file(
            returned,
            expected_path=relative,
            expected_sha256=record["sha256"],
            expected_size_bytes=record["size_bytes"],
        )
    return result


def download_sha_bound_payloads(
    *,
    download_fn: Callable[..., str],
    repo: str,
    repo_type: str,
    revision: str,
    records: Sequence[Mapping[str, Any]],
    local_root: Path,
) -> Mapping[str, bytes]:
    """Download an exact path/SHA inventory whose records may omit byte sizes."""
    if _COMMIT.fullmatch(revision) is None:
        raise ValueError("download revision must be one immutable HF commit")
    local_root.mkdir(parents=True, exist_ok=True)
    if local_root.is_symlink() or local_root.resolve(strict=True) != local_root.absolute():
        raise ValueError("download root must be a real canonical directory")
    result: dict[str, bytes] = {}
    for raw in records:
        record = _mapping(raw, "SHA-bound download record")
        relative = _safe_relative(record.get("path"), "SHA-bound download path")
        if relative in result or _SHA256.fullmatch(str(record.get("sha256"))) is None:
            raise ValueError("SHA-bound download inventory is malformed or duplicated")
        returned = Path(
            download_fn(
                repo_id=repo,
                repo_type=repo_type,
                filename=relative,
                revision=revision,
                local_dir=local_root,
                force_download=True,
            )
        )
        if not returned.is_absolute() or returned != local_root / relative:
            raise ValueError("SHA-bound download returned a noncanonical path")
        payload = _regular_file_bytes(returned, label=relative)
        if sha256_bytes(payload) != record["sha256"] or (
            "size_bytes" in record and len(payload) != record["size_bytes"]
        ):
            raise ValueError(f"SHA-bound downloaded bytes drifted: {relative}")
        result[relative] = payload
    return result


def validate_input_repo(
    api: Any,
    *,
    repo: str,
    repo_type: str,
    revision: str,
    expected_paths: Sequence[str],
    tag: str | None = None,
    annotated_tag_object: str | None = None,
    expected_private: bool,
    allowed_extra_paths: Sequence[str] = tuple(_ALLOWED_REPO_BASE_FILES),
) -> None:
    info = api.repo_info(repo, repo_type=repo_type, revision=revision)
    if (
        getattr(info, "private", None) is not expected_private
        or getattr(info, "sha", None) != revision
    ):
        raise ValueError("immutable input repo privacy or revision drifted")
    remote = set(api.list_repo_files(repo, repo_type=repo_type, revision=revision))
    expected = set(expected_paths)
    allowed = set(allowed_extra_paths)
    if expected - remote or remote - expected - allowed:
        raise ValueError("immutable input repo file inventory drifted")
    if tag is not None:
        tag_info = api.repo_info(repo, repo_type=repo_type, revision=tag)
        if getattr(tag_info, "sha", None) != revision:
            raise ValueError("immutable input tag does not resolve to its frozen revision")
        if annotated_tag_object is not None:
            refs = api.list_repo_refs(repo, repo_type=repo_type)
            matches = [item for item in refs.tags if item.name == tag]
            if (
                len(matches) != 1
                or getattr(matches[0], "target_commit", None) != annotated_tag_object
                or annotated_tag_object == revision
            ):
                raise ValueError("immutable input annotated-tag object drifted")


def _jsonl_lines(payload: bytes, *, count: int, label: str) -> tuple[bytes, ...]:
    if not payload.endswith(b"\n"):
        raise ValueError(f"{label} must be LF-terminated")
    lines = tuple(payload[:-1].split(b"\n"))
    if len(lines) != count or any(not line for line in lines):
        raise ValueError(f"{label} row denominator drifted")
    return lines


def extract_selected_ocr_records(
    payload: bytes,
    requirements: Sequence[Fresh16ImageRequirement],
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    transport_record_count: int = 384,
) -> Mapping[str, Mapping[str, Any]]:
    """Decode exactly the 80 selected OCR rows while scanning other rows as bytes."""
    if len(payload) != expected_size_bytes or sha256_bytes(payload) != expected_sha256:
        raise ValueError("OCR transport byte binding drifted")
    expected = {item.image_member_path: item for item in requirements}
    if len(expected) != FRESH_IMAGE_COUNT:
        raise ValueError("OCR extraction requires exactly 80 unique paths")
    markers = {
        path: b'"image_member_path":' + canonical_json_bytes(path) for path in expected
    }
    matches: dict[str, bytes] = {}
    for line in _jsonl_lines(payload, count=transport_record_count, label="OCR transport"):
        hit = [path for path, marker in markers.items() if marker in line]
        if len(hit) > 1:
            raise ValueError("one OCR row matched multiple selected paths")
        if not hit:
            continue
        path = hit[0]
        if path in matches:
            raise ValueError("selected OCR path occurred more than once")
        matches[path] = line
    if set(matches) != set(expected):
        raise ValueError("selected OCR path coverage drifted")
    result: dict[str, Mapping[str, Any]] = {}
    for path in sorted(matches):
        record = _strict_json(matches[path], label=f"selected OCR row {path}")
        if canonical_json_bytes(record) != matches[path]:
            raise ValueError("selected OCR row is not canonical JSON")
        if (
            record.get("image_member_path") != path
            or record.get("image_sha256") != expected[path].image_sha256
            or record.get("canonical_ocr_record_sha256")
            != expected[path].canonical_ocr_record_sha256
        ):
            raise ValueError("selected OCR identity differs from trajectory plan")
        result[path] = record
    return result


def extract_selected_image_payloads(
    archive_path: Path,
    requirements: Sequence[Fresh16ImageRequirement],
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    output_root: Path | None = None,
) -> Mapping[str, bytes]:
    """Extract and hash only the 80 required tar members, optionally sealing files."""
    archive = _regular_file_bytes(archive_path, label="fresh image archive")
    if len(archive) != expected_size_bytes or sha256_bytes(archive) != expected_sha256:
        raise ValueError("fresh image archive byte binding drifted")
    expected = {item.image_member_path: item for item in requirements}
    if len(expected) != FRESH_IMAGE_COUNT:
        raise ValueError("image extraction requires exactly 80 unique paths")
    result: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        members = {member.name: member for member in bundle.getmembers()}
        if len(members) != len(bundle.getmembers()):
            raise ValueError("image archive contains duplicate member names")
        for relative in sorted(expected):
            member = members.get(relative)
            if member is None or not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"selected image member is missing or unsafe: {relative}")
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"selected image member is unreadable: {relative}")
            payload = stream.read()
            if len(payload) != member.size or sha256_bytes(payload) != expected[relative].image_sha256:
                raise ValueError(f"selected image member identity drifted: {relative}")
            result[relative] = payload
            if output_root is not None:
                _exclusive_or_identical(output_root / relative, payload, mode=0o444)
    if len(result) != FRESH_IMAGE_COUNT:
        raise ValueError("selected image extraction denominator drifted")
    return result


def extract_exact_tar_member(
    archive_payload: bytes,
    *,
    expected_archive_sha256: str,
    expected_archive_size_bytes: int,
    expected_member_count: int,
    member_path: str,
    expected_member_sha256: str,
    expected_member_size_bytes: int,
) -> bytes:
    """Verify one immutable tar and return one regular member by exact identity."""
    relative = _safe_relative(member_path, "tar member path")
    if (
        len(archive_payload) != expected_archive_size_bytes
        or sha256_bytes(archive_payload) != expected_archive_sha256
    ):
        raise ValueError("immutable tar archive binding drifted")
    with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:") as bundle:
        members = bundle.getmembers()
        if len(members) != expected_member_count or len({item.name for item in members}) != len(
            members
        ):
            raise ValueError("immutable tar member inventory drifted")
        matches = [item for item in members if item.name == relative]
        if len(matches) != 1 or not matches[0].isfile() or matches[0].issym() or matches[0].islnk():
            raise ValueError("required tar member is missing or unsafe")
        stream = bundle.extractfile(matches[0])
        if stream is None:
            raise ValueError("required tar member cannot be read")
        payload = stream.read()
    if (
        len(payload) != expected_member_size_bytes
        or sha256_bytes(payload) != expected_member_sha256
    ):
        raise ValueError("required tar member byte identity drifted")
    return payload


def policy_worker_partitions(
    work_items: Sequence[Fresh16PolicyWorkItem],
) -> tuple[tuple[Fresh16PolicyWorkItem, ...], tuple[Fresh16PolicyWorkItem, ...], Fresh16PolicyWorkItem]:
    """Freeze the even/odd 24-state split and one worker-0 n=4 sentinel."""
    items = tuple(work_items)
    if len(items) != FRESH_STATE_COUNT or tuple(item.ordinal for item in items) != tuple(
        range(FRESH_STATE_COUNT)
    ):
        raise ValueError("policy work-item inventory or order drifted")
    even = tuple(item for item in items if item.ordinal % 2 == 0)
    odd = tuple(item for item in items if item.ordinal % 2 == 1)
    sentinels = [item for item in even if len(item.event_images) == 4]
    if len(even) != 24 or len(odd) != 24 or not sentinels:
        raise ValueError("policy worker 24/24 split or n=4 sentinel drifted")
    return even, odd, sentinels[0]


def _payloads_for_item(item: Fresh16PolicyWorkItem, image_root: Path) -> Mapping[str, bytes]:
    identities = (*(identity for _, identity in item.event_images), item.current_image)
    result = {}
    for identity in identities:
        relative = _safe_relative(identity.image_member_path, "policy image path")
        payload = _regular_file_bytes(image_root / relative, label=relative, mode=0o444)
        if sha256_bytes(payload) != identity.image_sha256:
            raise ValueError("policy worker image identity drifted")
        result[relative] = payload
    return result


def _score_policy_item(
    runtime: Fresh16GUIOwlV22VisionFeatureRuntime,
    item: Fresh16PolicyWorkItem,
    *,
    image_root: Path,
    feature_repeats: int,
    worker_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    payloads = _payloads_for_item(item, image_root)
    images = decode_policy_work_item_images(item, payloads)
    output = runtime.score_images(
        images,
        event_step_ids=item.candidate_event_step_ids,
        feature_repeats=feature_repeats,
    )
    if (
        output.get("feature_repeats") != feature_repeats
        or output.get("same_device_replay", {}).get("selection_equal") is not True
        or output.get("same_device_replay", {}).get("ranking_equal") is not True
        or float(output.get("same_device_replay", {}).get("max_abs_score_difference", math.inf))
        > 1e-6
    ):
        raise ValueError("policy same-device replay drifted")
    scores = {
        int(step): float(score) for step, score in output["scores_by_event_step"].items()
    }
    metadata = _mapping(output["runtime_metadata"], "policy runtime metadata")
    record = policy_selection_record(
        item,
        scores,
        worker_id=worker_id,
        device=str(metadata["device"]),
        gpu_uuid=str(metadata["gpu_uuid"]),
    )
    return record, output


def _policy_worker_entry(request: PolicyWorkerRequest) -> PolicyWorkerResult:
    runtime = Fresh16GUIOwlV22VisionFeatureRuntime(
        model_dir=request.model_dir,
        expected_snapshot_manifest=request.snapshot_manifest,
        device=request.device,
        expected_gpu_uuid=request.expected_gpu_uuid,
        gpu_uuid_type_profile=request.gpu_uuid_type_profile,
        image_processor_size_profile=request.image_processor_size_profile,
    )
    records = []
    score_transfer_count = 0
    image_decode_count = 0
    image_root = Path(request.image_root)
    for item in request.work_items:
        record, _ = _score_policy_item(
            runtime,
            item,
            image_root=image_root,
            feature_repeats=2,
            worker_id=request.worker_id,
        )
        records.append(record)
        score_transfer_count += 2 * len(item.event_images)
        image_decode_count += len(item.event_images) + 1
    sentinel_record = None
    if request.sentinel is not None:
        sentinel_record, _ = _score_policy_item(
            runtime,
            request.sentinel,
            image_root=image_root,
            feature_repeats=1,
            worker_id=request.worker_id,
        )
        score_transfer_count += len(request.sentinel.event_images)
        image_decode_count += len(request.sentinel.event_images) + 1
    counts = runtime.operation_counts
    operation_counts = {
        **counts,
        "policy_vision_cosine_scalar_transfer_count": score_transfer_count,
        "policy_vision_image_decode_count": image_decode_count,
    }
    if any(
        value != 0
        for key, value in counts.items()
        if key not in {"image_processor_batch_count", "policy_vision_feature_forward_count"}
    ):
        raise RuntimeError("policy worker observed a forbidden operation")
    return PolicyWorkerResult(
        worker_id=request.worker_id,
        records=tuple(records),
        sentinel_record=sentinel_record,
        operation_counts=operation_counts,
        runtime_metadata=dict(runtime.metadata),
    )


def validate_policy_worker_results(
    results: Sequence[PolicyWorkerResult],
    *,
    work_items: Sequence[Fresh16PolicyWorkItem],
    sentinel: Fresh16PolicyWorkItem,
    tolerance: float = 1e-6,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, tuple[int, ...]], Mapping[str, int]]:
    if len(results) != 2 or {result.worker_id for result in results} != {"even", "odd"}:
        raise ValueError("policy evaluation requires exactly even/odd worker results")
    by_id = {result.worker_id: result for result in results}
    if len(by_id["even"].records) != 24 or len(by_id["odd"].records) != 24:
        raise ValueError("policy workers must each score exactly 24 primary states")
    sentinel_replays = [
        result.sentinel_record for result in results if result.sentinel_record is not None
    ]
    if len(sentinel_replays) != 1:
        raise ValueError("policy workers must produce exactly one cross-device sentinel")
    primary_records = (*by_id["even"].records, *by_id["odd"].records)
    ordered, selections = combine_policy_score_records(primary_records, work_items)
    canonical = ordered[sentinel.ordinal]
    replay = sentinel_replays[0]
    if (
        replay.get("state_id") != canonical.get("state_id")
        or replay.get("worker_id") == canonical.get("worker_id")
        or replay.get("device") == canonical.get("device")
        or replay.get("gpu_uuid") == canonical.get("gpu_uuid")
        or replay.get("ranked_event_step_ids") != canonical.get("ranked_event_step_ids")
        or replay.get("selected_event_step_ids") != canonical.get("selected_event_step_ids")
    ):
        raise ValueError("cross-device n=4 sentinel ranking/selection drifted")
    left = {row["event_step_id"]: float(row["score"]) for row in canonical["scores_by_event_step"]}
    right = {row["event_step_id"]: float(row["score"]) for row in replay["scores_by_event_step"]}
    if set(left) != set(right) or max(abs(left[key] - right[key]) for key in left) > tolerance:
        raise ValueError("cross-device n=4 sentinel score tolerance failed")
    counts = {
        key: sum(int(result.operation_counts.get(key, 0)) for result in results)
        for key in {
            key for result in results for key in result.operation_counts
        }
    }
    required = {
        "image_processor_batch_count": 49,
        "policy_vision_feature_forward_count": 97,
        "policy_vision_cosine_scalar_transfer_count": 292,
        "policy_vision_image_decode_count": 197,
    }
    for key, expected in required.items():
        if counts.get(key) != expected:
            raise ValueError(f"policy operation count drifted: {key}")
    if any(value != 0 for key, value in counts.items() if key not in required):
        raise ValueError("policy forbidden operation count became nonzero")
    normalized_counts = {
        "policy_vision_processor_batch_count": counts[
            "image_processor_batch_count"
        ],
        "policy_vision_feature_forward_count": counts[
            "policy_vision_feature_forward_count"
        ],
        "policy_vision_cosine_scalar_transfer_count": counts[
            "policy_vision_cosine_scalar_transfer_count"
        ],
        "policy_vision_image_decode_count": counts[
            "policy_vision_image_decode_count"
        ],
    }
    return ordered, selections, normalized_counts


def run_policy_vision_process_pool(
    requests: Sequence[PolicyWorkerRequest],
    *,
    work_items: Sequence[Fresh16PolicyWorkItem],
    executor_factory: Callable[..., Any] = ProcessPoolExecutor,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, tuple[int, ...]], Mapping[str, int]]:
    request_tuple = tuple(requests)
    if len(request_tuple) != 2:
        raise ValueError("policy process pool requires exactly two worker requests")
    context = multiprocessing.get_context("spawn")
    with executor_factory(max_workers=2, mp_context=context) as executor:
        futures = [executor.submit(_policy_worker_entry, request) for request in request_tuple]
        results = tuple(future.result() for future in futures)
    _, _, sentinel = policy_worker_partitions(work_items)
    return validate_policy_worker_results(
        results,
        work_items=work_items,
        sentinel=sentinel,
    )


def _verified_model_file(
    payloads: Mapping[str, bytes],
    record: Mapping[str, Any],
) -> bytes:
    path = _safe_relative(record.get("path"), "formal model path")
    try:
        payload = payloads[path]
    except KeyError as error:
        raise ValueError(f"formal model file is missing: {path}") from error
    if (
        ("size_bytes" in record and len(payload) != record["size_bytes"])
        or sha256_bytes(payload) != record["sha256"]
    ):
        raise ValueError(f"formal model bytes drifted: {path}")
    return payload


def load_formal_ensembles(
    contract: Fresh16EvaluationContract,
    payloads: Mapping[str, bytes],
) -> FormalEnsembles:
    """Strictly load two provenance manifests and all ten CPU checkpoints."""
    model = contract.model
    expected_paths = {
        item["path"] for item in (*model["ensemble_manifests"], *model["checkpoints"])
    }
    if set(payloads) != expected_paths:
        raise ValueError("formal model payload inventory drifted")
    provenance_by_family: dict[str, FrozenEnsembleProvenance] = {}
    models_by_family: dict[str, list[Any]] = {"conditional": [], "independent": []}
    for manifest_record in model["ensemble_manifests"]:
        payload = _verified_model_file(payloads, manifest_record)
        value = _strict_json(payload, label=f"{manifest_record['family']} ensemble manifest")
        provenance = frozen_ensemble_provenance_from_manifest(value)
        if (
            provenance.training.family != manifest_record["family"]
            or provenance.sha256 != manifest_record["provenance_sha256"]
        ):
            raise ValueError("formal ensemble manifest provenance drifted")
        provenance_by_family[manifest_record["family"]] = provenance
    for checkpoint in model["checkpoints"]:
        payload = _verified_model_file(payloads, checkpoint)
        provenance = provenance_by_family[checkpoint["family"]]
        bound = provenance.checkpoints[checkpoint["seed"]]
        if (
            bound.seed != checkpoint["seed"]
            or bound.selected_epoch != checkpoint["selected_epoch"]
            or bound.model_state_sha256 != checkpoint["model_state_sha256"]
            or bound.checkpoint_artifact.repository != model["repo"]
            or bound.checkpoint_artifact.revision != model["payload_commit"]
            or bound.checkpoint_artifact.path != checkpoint["path"]
            or bound.checkpoint_artifact.sha256 != checkpoint["sha256"]
        ):
            raise ValueError("formal checkpoint config/provenance binding drifted")
        models_by_family[checkpoint["family"]].append(
            load_safetensors_checkpoint(
                payload,
                family=checkpoint["family"],
                seed=checkpoint["seed"],
                expected_model_state_sha256=checkpoint["model_state_sha256"],
                expected_checkpoint_sha256=checkpoint["sha256"],
            )
        )
    ensembles = {}
    for family in ("conditional", "independent"):
        provenance = provenance_by_family[family]
        training = provenance.training
        ensembles[family] = FittedEnsemble(
            family=family,
            learning_rate=training.learning_rate,
            seeds=tuple(checkpoint.seed for checkpoint in provenance.checkpoints),
            selected_epochs=training.selected_epochs,
            selection_sha256=training.oof_selection_sha256,
            models=tuple(models_by_family[family]),
        )
    return FormalEnsembles(
        conditional=ensembles["conditional"],
        conditional_provenance=provenance_by_family["conditional"],
        independent=ensembles["independent"],
        independent_provenance=provenance_by_family["independent"],
        checkpoint_load_count=10,
    )


class _FeatureOnlyGateView:
    def __init__(self, state: Any) -> None:
        self.source_id = state.source_id
        self.state_id = state.state_id
        self.decision_step_id = state.decision_step_id
        self.candidate_event_step_ids = state.candidate_event_step_ids
        self.q64 = state.q64
        self.candidates = state.candidates

    def candidate(self, event_step_id: int) -> Any:
        for candidate in self.candidates:
            if candidate.event_step_id == event_step_id:
                return candidate
        raise KeyError(event_step_id)


def _conditional_decision_trace(
    state: _FeatureOnlyGateView,
    score: Callable[[Any, int, Sequence[int]], float],
) -> tuple[tuple[int, ...], tuple[Mapping[str, Any], ...]]:
    selected: tuple[int, ...] = ()
    trace = []
    for round_index in range(DEPLOYMENT_BUDGET):
        candidates = tuple(
            event
            for event in state.candidate_event_step_ids
            if event not in selected
        )
        scores = tuple((event, float(score(state, event, selected))) for event in candidates)
        if not scores or any(not math.isfinite(value) for _, value in scores):
            raise ValueError("conditional feature-only decision score is malformed")
        event, predicted = min(scores, key=lambda item: (-item[1], item[0]))
        before = selected
        if predicted > 0.0:
            selected = tuple(sorted((*selected, event)))
            decision = "add"
            selected_event: int | None = event
        else:
            decision = "stop"
            selected_event = None
        trace.append(
            {
                "round": round_index,
                "selected_before": list(before),
                "candidate_predicted_marginal_gains": [
                    {
                        "event_step_id": candidate,
                        "predicted_marginal_gain": value,
                    }
                    for candidate, value in scores
                ],
                "decision": decision,
                "selected_event_step_id": selected_event,
                "selected_after": list(selected),
            }
        )
        if decision == "stop":
            break
    return selected, tuple(trace)


def _learned_decisions_from_feature_states(
    feature_states: Sequence[Any],
    ensembles: FormalEnsembles,
) -> LearnedDecisionArtifacts:
    """Run every frozen ensemble/seed decision on feature-only state views."""
    conditional_ensemble = ensemble_score(ensembles.conditional)
    independent_ensemble = ensemble_score(ensembles.independent)
    conditional_seeds = tuple(
        model_score(model, "conditional") for model in ensembles.conditional.models
    )
    independent_seeds = tuple(
        model_score(model, "independent") for model in ensembles.independent.models
    )
    if (
        len(conditional_seeds) != 5
        or len(independent_seeds) != 5
        or tuple(ensembles.conditional.seeds) != tuple(range(5))
        or tuple(ensembles.independent.seeds) != tuple(range(5))
    ):
        raise ValueError("formal learned decision seed inventory drifted")
    result: dict[str, dict[str, Mapping[str, Any]]] = {
        "conditional": {},
        "independent": {},
    }
    for feature in feature_states:
        view = _FeatureOnlyGateView(feature)
        conditional_selected, conditional_trace = _conditional_decision_trace(
            view, conditional_ensemble
        )
        result["conditional"][feature.state_id] = {
            "ensemble_selected_event_step_ids": list(conditional_selected),
            "ensemble_trace": [dict(item) for item in conditional_trace],
            "seed_selected_event_step_ids": [
                {
                    "seed": seed,
                    "selected_event_step_ids": list(
                        select_conditional(view, score)[0]
                    ),
                }
                for seed, score in enumerate(conditional_seeds)
            ],
        }
        result["independent"][feature.state_id] = {
            "ensemble_selected_event_step_ids": list(
                select_independent(view, independent_ensemble)
            ),
            "seed_selected_event_step_ids": [
                {
                    "seed": seed,
                    "selected_event_step_ids": list(
                        select_independent(view, score)
                    ),
                }
                for seed, score in enumerate(independent_seeds)
            ],
        }
    if any(len(items) != FRESH_STATE_COUNT for items in result.values()):
        raise ValueError("learned decision state denominator drifted")
    return LearnedDecisionArtifacts(
        conditional=MappingProxyType(result["conditional"]),
        independent=MappingProxyType(result["independent"]),
    )


def learned_decision_artifact_bytes(
    family: str,
    decisions: Mapping[str, Mapping[str, Any]],
) -> bytes:
    if family not in {"conditional", "independent"} or len(decisions) != FRESH_STATE_COUNT:
        raise ValueError("learned decision artifact family or denominator drifted")
    selections = {
        state_id: tuple(record["ensemble_selected_event_step_ids"])
        for state_id, record in decisions.items()
    }
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": LEARNED_STATUS,
        "name": family,
        "selection_sha256": canonical_selection_sha256(selections),
        "decision_contract": "FEATURE_ONLY_ALL_ENSEMBLE_AND_SEED_DECISIONS_V1",
        "records": [
            {"state_id": state_id, **dict(decisions[state_id])}
            for state_id in sorted(decisions)
        ],
    }
    return canonical_json_bytes(value) + b"\n"


def _validated_selected_events(
    raw: Any,
    *,
    candidates: Sequence[int],
    label: str,
) -> tuple[int, ...]:
    if not isinstance(raw, list) or any(type(item) is not int for item in raw):
        raise ValueError(f"{label} is malformed")
    selected = tuple(raw)
    if (
        selected != tuple(sorted(selected))
        or len(set(selected)) != len(selected)
        or len(selected) > DEPLOYMENT_BUDGET
        or not set(selected).issubset(candidates)
    ):
        raise ValueError(f"{label} is infeasible")
    return selected


def read_learned_decision_artifact(
    payload: bytes,
    *,
    family: str,
    feature_states: Sequence[Any],
) -> Mapping[str, Mapping[str, Any]]:
    value = _strict_json(payload, label=f"{family} learned decision artifact")
    expected_top = {
        "schema_version",
        "protocol_id",
        "status",
        "name",
        "selection_sha256",
        "decision_contract",
        "records",
    }
    records = value.get("records")
    if (
        payload != canonical_json_bytes(value) + b"\n"
        or set(value) != expected_top
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != LEARNED_STATUS
        or value.get("name") != family
        or value.get("decision_contract")
        != "FEATURE_ONLY_ALL_ENSEMBLE_AND_SEED_DECISIONS_V1"
        or family not in {"conditional", "independent"}
        or not isinstance(records, list)
        or len(records) != FRESH_STATE_COUNT
        or len(feature_states) != FRESH_STATE_COUNT
    ):
        raise ValueError("learned decision artifact schema or denominator drifted")
    features = {state.state_id: state for state in feature_states}
    if len(features) != FRESH_STATE_COUNT:
        raise ValueError("feature state inventory for learned decisions drifted")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        record = _mapping(record, "learned decision state record")
        expected_record = {
            "state_id",
            "ensemble_selected_event_step_ids",
            "seed_selected_event_step_ids",
        }
        if family == "conditional":
            expected_record.add("ensemble_trace")
        if set(record) != expected_record:
            raise ValueError("learned decision state schema drifted")
        state_id = record.get("state_id")
        if not isinstance(state_id, str) or state_id in result or state_id not in features:
            raise ValueError("learned decision state identity drifted")
        candidates = tuple(features[state_id].candidate_event_step_ids)
        ensemble_selected = _validated_selected_events(
            record.get("ensemble_selected_event_step_ids"),
            candidates=candidates,
            label="learned ensemble selection",
        )
        seeds = record.get("seed_selected_event_step_ids")
        if not isinstance(seeds, list) or len(seeds) != 5:
            raise ValueError("learned seed selection denominator drifted")
        for seed, seed_record in enumerate(seeds):
            seed_record = _mapping(seed_record, "learned seed selection")
            if set(seed_record) != {"seed", "selected_event_step_ids"} or seed_record.get(
                "seed"
            ) != seed:
                raise ValueError("learned seed selection identity drifted")
            _validated_selected_events(
                seed_record.get("selected_event_step_ids"),
                candidates=candidates,
                label="learned seed selection",
            )
        if family == "conditional":
            trace = record.get("ensemble_trace")
            if not isinstance(trace, list) or not 1 <= len(trace) <= DEPLOYMENT_BUDGET:
                raise ValueError("conditional learned trace denominator drifted")
            selected: tuple[int, ...] = ()
            stopped = False
            for round_index, raw_round in enumerate(trace):
                round_record = _mapping(raw_round, "conditional learned trace round")
                if set(round_record) != {
                    "round",
                    "selected_before",
                    "candidate_predicted_marginal_gains",
                    "decision",
                    "selected_event_step_id",
                    "selected_after",
                }:
                    raise ValueError("conditional learned trace schema drifted")
                remaining = tuple(event for event in candidates if event not in selected)
                score_rows = round_record.get("candidate_predicted_marginal_gains")
                if not isinstance(score_rows, list) or len(score_rows) != len(remaining):
                    raise ValueError("conditional learned trace score denominator drifted")
                scores = []
                for expected_event, score_row in zip(remaining, score_rows, strict=True):
                    score_row = _mapping(score_row, "conditional learned trace score")
                    score = score_row.get("predicted_marginal_gain")
                    if (
                        set(score_row) != {"event_step_id", "predicted_marginal_gain"}
                        or score_row.get("event_step_id") != expected_event
                        or isinstance(score, bool)
                        or not isinstance(score, (int, float))
                        or not math.isfinite(float(score))
                    ):
                        raise ValueError("conditional learned trace score drifted")
                    scores.append((expected_event, float(score)))
                selected_before = _validated_selected_events(
                    round_record.get("selected_before"),
                    candidates=candidates,
                    label="conditional learned trace selected-before",
                )
                event, predicted = min(scores, key=lambda item: (-item[1], item[0]))
                decision = round_record.get("decision")
                selected_event = round_record.get("selected_event_step_id")
                if (
                    round_record.get("round") != round_index
                    or selected_before != selected
                    or stopped
                ):
                    raise ValueError("conditional learned trace order drifted")
                if predicted > 0.0:
                    expected_after = tuple(sorted((*selected, event)))
                    if decision != "add" or selected_event != event:
                        raise ValueError("conditional learned add decision contradicts scores")
                else:
                    expected_after = selected
                    stopped = True
                    if decision != "stop" or selected_event is not None:
                        raise ValueError("conditional learned stop decision contradicts scores")
                selected_after = _validated_selected_events(
                    round_record.get("selected_after"),
                    candidates=candidates,
                    label="conditional learned trace selected-after",
                )
                if selected_after != expected_after:
                    raise ValueError("conditional learned trace transition drifted")
                selected = selected_after
            if (
                selected != ensemble_selected
                or (stopped and trace[-1]["decision"] != "stop")
                or (not stopped and len(trace) != DEPLOYMENT_BUDGET)
            ):
                raise ValueError("conditional learned trace final selection drifted")
        result[state_id] = record
    if tuple(result) != tuple(sorted(features)):
        raise ValueError("learned decision records are not in canonical state order")
    selections = {
        state_id: tuple(record["ensemble_selected_event_step_ids"])
        for state_id, record in result.items()
    }
    if value.get("selection_sha256") != canonical_selection_sha256(selections):
        raise ValueError("learned decision selection digest drifted")
    return MappingProxyType(result)


def _sealed_decision_inputs(
    learned: LearnedDecisionArtifacts,
) -> Mapping[str, Any]:
    conditional_selections = {
        state_id: tuple(record["ensemble_selected_event_step_ids"])
        for state_id, record in learned.conditional.items()
    }
    conditional_traces = {}
    for state_id, record in learned.conditional.items():
        additions = []
        for round_record in record["ensemble_trace"]:
            if round_record["decision"] == "stop":
                continue
            event = round_record["selected_event_step_id"]
            score_by_event = {
                item["event_step_id"]: item["predicted_marginal_gain"]
                for item in round_record["candidate_predicted_marginal_gains"]
            }
            additions.append(
                {
                    "event_step_id": event,
                    "predicted_marginal_gain": score_by_event[event],
                }
            )
        conditional_traces[state_id] = tuple(additions)
    independent_selections = {
        state_id: tuple(record["ensemble_selected_event_step_ids"])
        for state_id, record in learned.independent.items()
    }

    def seeds(family: Mapping[str, Mapping[str, Any]]) -> tuple[Mapping[str, tuple[int, ...]], ...]:
        return tuple(
            {
                state_id: tuple(
                    record["seed_selected_event_step_ids"][seed][
                        "selected_event_step_ids"
                    ]
                )
                for state_id, record in family.items()
            }
            for seed in range(5)
        )

    return {
        "conditional_selections": conditional_selections,
        "conditional_traces": conditional_traces,
        "independent_selections": independent_selections,
        "conditional_seed_selections": seeds(learned.conditional),
        "independent_seed_selections": seeds(learned.independent),
    }


def learned_selections_from_features(
    bundle: Fresh16LabelBlindBundle,
    ensembles: FormalEnsembles,
) -> LearnedDecisionArtifacts:
    """Seal every model-dependent decision before any label bytes are accessed."""
    return _learned_decisions_from_feature_states(bundle.feature_states, ensembles)


def build_label_blind_payload_files(
    bundle: Fresh16LabelBlindBundle,
    *,
    policy_score_records: Sequence[Mapping[str, Any]],
    policy_selections: Mapping[str, Sequence[int]],
    learned_selections: LearnedDecisionArtifacts,
) -> Mapping[str, bytes]:
    decorated_ocr_records = []
    if len(bundle.ocr_rgb_score_records) != len(bundle.feature_states):
        raise ValueError("OCR score/feature state denominator drifted")
    for ordinal, (record, feature) in enumerate(
        zip(bundle.ocr_rgb_score_records, bundle.feature_states, strict=True)
    ):
        decorated_ocr_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": "FROZEN_GATE_V1_FRESH16_OCR_RGB_SCORES_V1",
                "ordinal": ordinal,
                "decision_step_id": feature.decision_step_id,
                **dict(record),
            }
        )
    files = {
        FEATURE_PAYLOAD_PATH: feature_states_jsonl_bytes(bundle.feature_states),
        HEURISTIC_PATHS["dynamic_recent"]: selection_artifact_bytes(
            "dynamic_recent", bundle.dynamic_recent
        ),
        HEURISTIC_PATHS["ocr_rgb_v2"]: selection_artifact_bytes(
            "ocr_rgb_v2", bundle.ocr_rgb_v2
        ),
        HEURISTIC_PATHS["policy_vision_v3"]: selection_artifact_bytes(
            "policy_vision_v3", policy_selections
        ),
        OCR_SCORE_PATH: jsonl_score_records_bytes(decorated_ocr_records),
        POLICY_SCORE_PATH: jsonl_score_records_bytes(policy_score_records),
        LEARNED_PATHS["conditional"]: learned_decision_artifact_bytes(
            "conditional", learned_selections.conditional
        ),
        LEARNED_PATHS["independent"]: learned_decision_artifact_bytes(
            "independent", learned_selections.independent
        ),
    }
    if set(files) != set(LABEL_BLIND_PAYLOAD_TARGETS):
        raise ValueError("label-blind payload inventory drifted")
    return files


def _file_inventory(files: Mapping[str, bytes]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "path": path,
            "sha256": sha256_bytes(files[path]),
            "size_bytes": len(files[path]),
        }
        for path in sorted(files)
    )


def seal_label_blind_files(files: Mapping[str, bytes]) -> HeuristicBytesSeal:
    if set(files) != set(LABEL_BLIND_PAYLOAD_TARGETS):
        raise ValueError("heuristic seal requires all and only eight label-blind files")
    copied = {path: bytes(payload) for path, payload in files.items()}
    features = read_feature_states_jsonl(copied[FEATURE_PAYLOAD_PATH])
    heuristic_selections = {
        name: read_selection_artifact(copied[path], expected_name=name)
        for name, path in HEURISTIC_PATHS.items()
    }
    replayed_recent = {
        feature.state_id: recent_selection(feature.candidate_event_step_ids)
        for feature in features
    }
    if heuristic_selections["dynamic_recent"] != replayed_recent:
        raise ValueError("dynamic-recent selections contradict feature-state ordering")
    for name, path in LEARNED_PATHS.items():
        read_learned_decision_artifact(
            copied[path], family=name, feature_states=features
        )
    # note (luojiaxuan): Each score vector is ranked again with the frozen tie
    # semantics. This prevents a self-consistent but contradictory score/selection
    # pair from becoming the pre-label seal.
    for path, name in (
        (OCR_SCORE_PATH, "ocr_rgb_v2"),
        (POLICY_SCORE_PATH, "policy_vision_v3"),
    ):
        lines = _jsonl_lines(copied[path], count=48, label=path)
        for ordinal, (line, feature) in enumerate(zip(lines, features, strict=True)):
            record = _strict_json(line, label=path)
            if canonical_json_bytes(record) != line:
                raise ValueError("score record is not canonical JSONL")
            score_rows = record.get("scores_by_event_step")
            if not isinstance(score_rows, list):
                raise ValueError("score record lacks its score vector")
            scores: dict[int, float] = {}
            for row in score_rows:
                if not isinstance(row, Mapping):
                    raise ValueError("score row is malformed")
                step = row.get("event_step_id")
                score = row.get("score")
                if (
                    type(step) is not int
                    or step in scores
                    or isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                ):
                    raise ValueError("score row identity or scalar drifted")
                scores[step] = float(score)
            replay = policy_vision_similarity_selection(
                scores,
                event_step_ids=feature.candidate_event_step_ids,
            )
            expected_worker = "even" if ordinal % 2 == 0 else "odd"
            if (
                record.get("ordinal") != ordinal
                or record.get("source_id") != feature.source_id
                or record.get("state_id") != feature.state_id
                or record.get("decision_step_id") != feature.decision_step_id
                or record.get("candidate_event_step_ids")
                != list(feature.candidate_event_step_ids)
                or record.get("ranked_event_step_ids")
                != list(replay.ranked_event_step_ids)
                or record.get("selected_event_step_ids")
                != list(replay.selected_event_step_ids)
                or heuristic_selections[name][feature.state_id]
                != replay.selected_event_step_ids
            ):
                raise ValueError("score, ranking, selection, or state identity contradicts")
            if path == POLICY_SCORE_PATH and (
                record.get("worker_id") != expected_worker
                or not isinstance(record.get("device"), str)
                or not record["device"]
                or not isinstance(record.get("gpu_uuid"), str)
                or not record["gpu_uuid"]
            ):
                raise ValueError("policy score worker/device provenance drifted")
    inventory = _file_inventory(copied)
    return HeuristicBytesSeal(copied, inventory, _token=_HEURISTIC_SEAL_TOKEN)


def claim_label_access(
    seal: HeuristicBytesSeal,
    *,
    source_git_commit: str,
    contract_sha256: str,
) -> LabelAccessClaim:
    if (
        not isinstance(seal, HeuristicBytesSeal)
        or seal._token is not _HEURISTIC_SEAL_TOKEN
        or _COMMIT.fullmatch(source_git_commit) is None
        or _SHA256.fullmatch(contract_sha256) is None
        or _file_inventory(seal._files) != tuple(dict(item) for item in seal._inventory)
    ):
        raise ValueError("label access requires an intact heuristic byte seal")
    claim = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": LABEL_CLAIM_STATUS,
        "source_git_commit": source_git_commit,
        "contract_sha256": contract_sha256,
        "label_blind_file_count": 8,
        "label_blind_inventory": list(seal.inventory),
        "label_blind_inventory_sha256": seal.inventory_sha256,
        "label_access_authorized": True,
        "legacy_dev5_access_authorized": False,
        "confirm20_access_authorized": False,
        "matched_nll_authorized": False,
        "closed_loop_authorized": False,
    }
    return LabelAccessClaim(
        seal.inventory_sha256,
        claim,
        _token=_LABEL_CLAIM_TOKEN,
    )


def validate_claim_against_payload(
    claim: LabelAccessClaim,
    seal: HeuristicBytesSeal,
    payload_files: Mapping[str, bytes],
) -> None:
    if (
        not isinstance(claim, LabelAccessClaim)
        or claim._token is not _LABEL_CLAIM_TOKEN
        or not isinstance(seal, HeuristicBytesSeal)
        or seal._token is not _HEURISTIC_SEAL_TOKEN
        or claim.heuristic_inventory_sha256 != seal.inventory_sha256
        or claim.claim.get("label_blind_inventory_sha256") != seal.inventory_sha256
        or set(payload_files) != set(PAYLOAD_TARGETS)
    ):
        raise ValueError("label claim/payload binding is malformed")
    observed = {path: payload_files[path] for path in LABEL_BLIND_PAYLOAD_TARGETS}
    if observed != dict(seal.files) or _file_inventory(observed) != seal.inventory:
        raise ValueError("nine-file payload changed after the pre-label heuristic seal")


def decode_labels_after_claim(
    raw_state_payload: bytes,
    claim: LabelAccessClaim,
    *,
    expected_source_ids: Sequence[str],
    expected_source_ids_sha256: str,
) -> Fresh16LabelSlice:
    if (
        not isinstance(claim, LabelAccessClaim)
        or claim._token is not _LABEL_CLAIM_TOKEN
        or claim.claim.get("label_access_authorized") is not True
    ):
        raise ValueError("fresh labels cannot be decoded before a valid label claim")
    transport = verify_fresh16_raw_state_transport(raw_state_payload)
    return decode_fresh16_labels(
        transport,
        expected_source_ids=expected_source_ids,
        expected_source_ids_sha256=expected_source_ids_sha256,
    )


def build_primary_state_records(
    states: Sequence[GateState],
    *,
    learned: LearnedDecisionArtifacts,
    heuristics: Mapping[str, Mapping[str, Sequence[int]]],
) -> bytes:
    decisions = _sealed_decision_inputs(learned)
    records = _selector_records_from_sealed_decisions(
        states,
        heuristics=heuristics,
        **decisions,
    )
    if len(records) != FRESH_STATE_COUNT:
        raise ValueError("primary state trace denominator drifted")
    rows = []
    by_state = {state.state_id: state for state in states}
    for ordinal, state in enumerate(states):
        record = records[state.state_id]
        decision = learned.conditional[state.state_id]
        if tuple(decision["ensemble_selected_event_step_ids"]) != tuple(
            record["conditional"]["selected"]
        ):
            raise ValueError("sealed conditional decision differs from state evaluation")
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": "FROZEN_GATE_V1_FRESH16_PRIMARY_STATE_RECORD_V1",
                "ordinal": ordinal,
                "source_id": state.source_id,
                "state_id": state.state_id,
                "decision_step_id": state.decision_step_id,
                "candidate_event_step_ids": list(state.candidate_event_step_ids),
                "conditional_selection_trace": list(decision["ensemble_trace"]),
                "evaluation": record,
            }
        )
    if set(by_state) != set(records):
        raise ValueError("primary state trace inventory drifted")
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _artifact(
    *, repo: str, revision: str, path: str, payload: bytes
) -> ArtifactBinding:
    return ArtifactBinding(
        repository=repo,
        revision=revision,
        path=path,
        sha256=sha256_bytes(payload),
    )


def _primary_provenance_payload(
    *,
    states: Sequence[GateState],
    formal: FormalDecisionProvenance,
    feature_artifact: ArtifactBinding,
    label_artifact: ArtifactBinding,
    heuristic_artifacts: Sequence[HeuristicArtifactProvenance],
) -> Mapping[str, Any]:
    state_inventory = [
        {
            "source_id": state.source_id,
            "state_id": state.state_id,
            "decision_step_id": state.decision_step_id,
            "candidate_event_step_ids": list(state.candidate_event_step_ids),
        }
        for state in states
    ]
    return {
        "gate_config_sha256": GATE_V1_CONFIG_SHA256,
        "state_inventory_sha256": sha256_bytes(canonical_json_bytes(state_inventory)),
        "conditional_ensemble_sha256": formal.conditional.sha256,
        "conditional_ensemble": formal.conditional.to_payload(),
        "independent_ensemble_sha256": formal.independent.sha256,
        "independent_ensemble": formal.independent.to_payload(),
        "evaluation_feature_artifact": feature_artifact.to_payload(),
        "evaluation_label_artifact": label_artifact.to_payload(),
        "heuristic_artifacts": [item.to_payload() for item in heuristic_artifacts],
    }


_GO_SELECTOR_THRESHOLDS = {
    "ensemble_normalized_recovery_over_exact_minimum": 0.8,
    "ensemble_raw_utility_over_exact_raw_minimum": 0.8,
    "mean_normalized_delta_vs_each_heuristic_minimum": 0.05,
    "strongest_heuristic_positive_trajectory_count_minimum": 12,
    "strongest_heuristic_paired_bootstrap_lower_strictly_greater_than": 0.0,
    "individual_seed_exact_ratio_minimum": 0.75,
    "individual_seed_pass_count_minimum": 4,
    "seed_mean_recovery_population_std_maximum": 0.08,
    "hard_n3_n4_equal_trajectory_raw_utility_ratio_minimum": 0.7,
    "selected_true_nonpositive_addition_rate_maximum": 0.1,
}
_GO_SET_THRESHOLDS = {
    "conditional_minus_independent_mean_normalized_delta_minimum": 0.02,
    "conditional_minus_independent_raw_utility_delta_strictly_greater_than": 0.0,
    "conditional_minus_independent_positive_trajectory_count_minimum": 12,
    "conditional_minus_independent_paired_bootstrap_lower_strictly_greater_than": 0.0,
    "paired_seed_positive_count_minimum": 4,
}


def seal_report_threshold_contract(
    report: Mapping[str, Any],
    contract: Fresh16EvaluationContract,
) -> Mapping[str, Any]:
    """Re-evaluate every GO boolean against the config-bound threshold maps."""
    evaluation = _mapping(contract.data["evaluation_contract"], "evaluation contract")
    if (
        evaluation.get("go_selector_all") != _GO_SELECTOR_THRESHOLDS
        or evaluation.get("go_set_conditioning_all") != _GO_SET_THRESHOLDS
    ):
        raise ValueError("fresh16 GO threshold maps differ from evaluator semantics")
    metrics = _mapping(report.get("metrics"), "primary report metrics")
    exact = float(metrics["exact_normalized_recovery"])
    selector = {
        "ensemble_normalized_recovery_over_exact": (
            float(metrics["ensemble_normalized_recovery"]) / exact
            >= _GO_SELECTOR_THRESHOLDS[
                "ensemble_normalized_recovery_over_exact_minimum"
            ]
        ),
        "ensemble_raw_utility_over_exact_raw": float(
            metrics["ensemble_raw_utility_over_exact_raw"]
        )
        >= _GO_SELECTOR_THRESHOLDS[
            "ensemble_raw_utility_over_exact_raw_minimum"
        ],
        "delta_vs_every_heuristic": all(
            float(value)
            >= _GO_SELECTOR_THRESHOLDS[
                "mean_normalized_delta_vs_each_heuristic_minimum"
            ]
            for value in metrics[
                "conditional_minus_heuristic_mean_normalized_delta"
            ].values()
        ),
        "strongest_positive_trajectory_count": int(
            metrics["strongest_heuristic_positive_trajectory_count"]
        )
        >= _GO_SELECTOR_THRESHOLDS[
            "strongest_heuristic_positive_trajectory_count_minimum"
        ],
        "strongest_bootstrap_lower": float(
            metrics["strongest_heuristic_paired_bootstrap_lower"]
        )
        > _GO_SELECTOR_THRESHOLDS[
            "strongest_heuristic_paired_bootstrap_lower_strictly_greater_than"
        ],
        "individual_seed_pass_count": sum(
            float(value)
            >= _GO_SELECTOR_THRESHOLDS["individual_seed_exact_ratio_minimum"]
            for value in metrics["individual_seed_exact_normalized_ratios"]
        )
        >= _GO_SELECTOR_THRESHOLDS["individual_seed_pass_count_minimum"],
        "seed_population_std": float(metrics["seed_mean_recovery_population_std"])
        <= _GO_SELECTOR_THRESHOLDS[
            "seed_mean_recovery_population_std_maximum"
        ],
        "hard_n3_n4_ratio": float(metrics["hard_n3_n4_raw_utility_ratio"])
        >= _GO_SELECTOR_THRESHOLDS[
            "hard_n3_n4_equal_trajectory_raw_utility_ratio_minimum"
        ],
        "nonpositive_addition_rate": float(
            metrics["selected_true_nonpositive_addition_rate"]
        )
        <= _GO_SELECTOR_THRESHOLDS[
            "selected_true_nonpositive_addition_rate_maximum"
        ],
    }
    set_checks = {
        "mean_normalized_delta": float(
            metrics["conditional_minus_independent_mean_normalized_delta"]
        )
        >= _GO_SET_THRESHOLDS[
            "conditional_minus_independent_mean_normalized_delta_minimum"
        ],
        "raw_utility_delta": float(
            metrics["conditional_minus_independent_raw_utility_delta"]
        )
        > _GO_SET_THRESHOLDS[
            "conditional_minus_independent_raw_utility_delta_strictly_greater_than"
        ],
        "positive_trajectory_count": int(
            metrics["conditional_minus_independent_positive_trajectory_count"]
        )
        >= _GO_SET_THRESHOLDS[
            "conditional_minus_independent_positive_trajectory_count_minimum"
        ],
        "bootstrap_lower": float(
            metrics["conditional_minus_independent_paired_bootstrap_lower"]
        )
        > _GO_SET_THRESHOLDS[
            "conditional_minus_independent_paired_bootstrap_lower_strictly_greater_than"
        ],
        "paired_seed_positive_count": int(metrics["paired_seed_positive_count"])
        >= _GO_SET_THRESHOLDS["paired_seed_positive_count_minimum"],
    }
    if (
        report.get("go_selector_checks") != selector
        or report.get("go_selector") is not all(selector.values())
        or report.get("go_set_conditioning_primary_checks") != set_checks
        or report.get("go_set_conditioning_primary") is not all(set_checks.values())
    ):
        raise ValueError("evaluator GO checks do not replay the frozen threshold maps")
    sealed = dict(report)
    sealed.pop("report_sha256", None)
    thresholds = {
        "go_selector_all": dict(_GO_SELECTOR_THRESHOLDS),
        "go_set_conditioning_all": dict(_GO_SET_THRESHOLDS),
    }
    sealed["frozen_go_thresholds"] = thresholds
    sealed["frozen_go_thresholds_sha256"] = sha256_bytes(
        canonical_json_bytes(thresholds)
    )
    sealed["report_sha256"] = canonical_report_sha256(sealed)
    return sealed


def evaluate_and_build_report_files(
    contract: Fresh16EvaluationContract,
    *,
    states: Sequence[GateState],
    formal_provenance: FormalDecisionProvenance,
    payload_files: Mapping[str, bytes],
    payload_commit: str,
    source: SourceIdentity,
    operation_counts: Mapping[str, int],
    runtime_identity: RuntimeIdentity,
) -> Mapping[str, bytes]:
    if set(payload_files) != set(PAYLOAD_TARGETS) or _COMMIT.fullmatch(payload_commit) is None:
        raise ValueError("primary evaluation requires the complete immutable payload commit")
    validate_operation_counts(contract, operation_counts)
    return _evaluate_and_build_report_files_after_counts(
        contract,
        states=states,
        formal_provenance=formal_provenance,
        payload_files=payload_files,
        payload_commit=payload_commit,
        source=source,
        operation_counts=operation_counts,
        runtime_identity=runtime_identity,
    )


def validate_operation_counts(
    contract: Fresh16EvaluationContract,
    operation_counts: Mapping[str, int],
) -> None:
    expected_counts = _mapping(
        contract.data["execution_planned_operation_contract"],
        "execution operation contract",
    )
    if set(operation_counts) != set(expected_counts) or any(
        type(operation_counts[key]) is not int
        or operation_counts[key] != expected_counts[key]
        for key in expected_counts
    ):
        raise ValueError("fresh16 operation denominator inventory or value drifted")


def _run_runtime_provenance(
    runtime_identity: RuntimeIdentity,
    payload_files: Mapping[str, bytes],
) -> Mapping[str, Any]:
    if not isinstance(runtime_identity, RuntimeIdentity):
        raise ValueError("run manifest requires one validated runtime identity")
    workers: dict[str, tuple[str, str]] = {}
    for line in _jsonl_lines(
        payload_files[POLICY_SCORE_PATH], count=FRESH_STATE_COUNT, label=POLICY_SCORE_PATH
    ):
        record = _strict_json(line, label="policy runtime provenance record")
        worker = record.get("worker_id")
        device = record.get("device")
        gpu_uuid = record.get("gpu_uuid")
        if any(not isinstance(item, str) or not item for item in (worker, device, gpu_uuid)):
            raise ValueError("policy runtime provenance identity is malformed")
        observed = (device, gpu_uuid)
        if worker in workers and workers[worker] != observed:
            raise ValueError("policy worker runtime identity changed within its shard")
        workers[worker] = observed
    if set(workers) != {"even", "odd"} or len({item[0] for item in workers.values()}) != 2 or len(
        {item[1] for item in workers.values()}
    ) != 2:
        raise ValueError("policy runtime provenance is not exact two-device/two-GPU")
    return {
        "docker_inspect_receipt_sha256": runtime_identity.receipt_sha256,
        "docker_inspect_receipt": dict(runtime_identity.receipt),
        "validated_runtime_identity": dict(runtime_identity.payload),
        "validated_runtime_identity_sha256": sha256_bytes(
            canonical_json_bytes(runtime_identity.payload)
        ),
        "policy_workers": [
            {
                "worker_id": worker,
                "device": workers[worker][0],
                "gpu_uuid": workers[worker][1],
            }
            for worker in ("even", "odd")
        ],
    }


def _validate_run_source_and_runtime_identity(
    run: Mapping[str, Any],
    *,
    source: SourceIdentity,
    runtime_identity: RuntimeIdentity,
    payload_files: Mapping[str, bytes],
) -> None:
    if run.get("source") != _source_record(source):
        raise ValueError("run manifest source commit or inventory drifted")
    if run.get("runtime_provenance") != _run_runtime_provenance(
        runtime_identity, payload_files
    ):
        raise ValueError("run manifest runtime provenance drifted")


def _evaluate_and_build_report_files_after_counts(
    contract: Fresh16EvaluationContract,
    *,
    states: Sequence[GateState],
    formal_provenance: FormalDecisionProvenance,
    payload_files: Mapping[str, bytes],
    payload_commit: str,
    source: SourceIdentity,
    operation_counts: Mapping[str, int],
    runtime_identity: RuntimeIdentity,
) -> Mapping[str, bytes]:
    features = read_feature_states_jsonl(payload_files[FEATURE_PAYLOAD_PATH])
    labels = read_label_states_jsonl(payload_files[LABEL_PAYLOAD_PATH])
    replayed_states = join_feature_and_label_states(
        features,
        labels,
        expected_source_ids=tuple(contract.geometry["source_ids"]),
    )
    # note (luojiaxuan): The immutable payload commit, not transient in-memory
    # objects, is the evaluation substrate. The caller copy is accepted only as
    # an additional consistency witness and can never override replayed bytes.
    if tuple(states) != tuple(replayed_states):
        raise ValueError("in-memory states differ from exact payload-byte replay")
    states = replayed_states
    report_bytes, state_records = _build_primary_sealed_outputs(
        contract,
        states=states,
        formal_provenance=formal_provenance,
        payload_files=payload_files,
        payload_commit=payload_commit,
    )
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": FINAL_STATUS,
        "contract_sha256": contract.sha256,
        "source": _source_record(source),
        "runtime_provenance": _run_runtime_provenance(
            runtime_identity, payload_files
        ),
        "payload_commit": payload_commit,
        "payload_inventory": list(_file_inventory(payload_files)),
        "primary_report_sha256": sha256_bytes(report_bytes),
        "primary_state_records_sha256": sha256_bytes(state_records),
        "execution_planned_operation_contract": dict(operation_counts),
        "planned_local_precommit_replay_operation_contract": dict(
            _sealed_output_replay_operation_counts(contract)
        ),
        "gate_trained": True,
        "fresh16_primary_evaluated": True,
        "legacy_dev5_semantic_decode_count": 0,
        "confirm20_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
    }
    run_bytes = canonical_json_bytes(run_manifest) + b"\n"
    partial = {
        STATE_RECORD_PATH: state_records,
        PRIMARY_REPORT_PATH: report_bytes,
        RUN_MANIFEST_PATH: run_bytes,
    }
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_GATE_V1_FRESH16_BUNDLE_MANIFEST_V1",
        "payload_commit": payload_commit,
        "payload_files": list(_file_inventory(payload_files)),
        "report_files_excluding_bundle": list(_file_inventory(partial)),
        "final_target_count": 13,
        "report_commit_embedded": False,
    }
    files = {**partial, BUNDLE_MANIFEST_PATH: canonical_json_bytes(bundle) + b"\n"}
    if set(files) != set(REPORT_TARGETS):
        raise ValueError("primary report output inventory drifted")
    return files


def _build_primary_sealed_outputs(
    contract: Fresh16EvaluationContract,
    *,
    states: Sequence[GateState],
    formal_provenance: FormalDecisionProvenance,
    payload_files: Mapping[str, bytes],
    payload_commit: str,
) -> tuple[bytes, bytes]:
    """Build primary artifacts from sealed decisions without model/selector calls."""
    features = read_feature_states_jsonl(payload_files[FEATURE_PAYLOAD_PATH])
    learned = LearnedDecisionArtifacts(
        conditional=read_learned_decision_artifact(
            payload_files[LEARNED_PATHS["conditional"]],
            family="conditional",
            feature_states=features,
        ),
        independent=read_learned_decision_artifact(
            payload_files[LEARNED_PATHS["independent"]],
            family="independent",
            feature_states=features,
        ),
    )
    heuristics = {
        name: read_selection_artifact(payload_files[path], expected_name=name)
        for name, path in HEURISTIC_PATHS.items()
    }
    destination = contract.destination
    repo = destination["repo"]
    provenance = tuple(
        HeuristicArtifactProvenance(
            name=name,
            artifact=_artifact(
                repo=repo,
                revision=payload_commit,
                path=HEURISTIC_PATHS[name],
                payload=payload_files[HEURISTIC_PATHS[name]],
            ),
            selection_sha256=canonical_selection_sha256(heuristics[name]),
        )
        for name in ("dynamic_recent", "ocr_rgb_v2", "policy_vision_v3")
    )
    feature_artifact = _artifact(
        repo=repo,
        revision=payload_commit,
        path=FEATURE_PAYLOAD_PATH,
        payload=payload_files[FEATURE_PAYLOAD_PATH],
    )
    label_artifact = _artifact(
        repo=repo,
        revision=payload_commit,
        path=LABEL_PAYLOAD_PATH,
        payload=payload_files[LABEL_PAYLOAD_PATH],
    )
    report = evaluate_primary_slice_from_sealed_decisions(
        states,
        expected_source_ids=tuple(contract.geometry["source_ids"]),
        heuristics=heuristics,
        provenance_payload=_primary_provenance_payload(
            states=states,
            formal=formal_provenance,
            feature_artifact=feature_artifact,
            label_artifact=label_artifact,
            heuristic_artifacts=provenance,
        ),
        **_sealed_decision_inputs(learned),
    )
    report = seal_report_threshold_contract(report, contract)
    state_records = build_primary_state_records(
        states,
        learned=learned,
        heuristics=heuristics,
    )
    report_bytes = canonical_json_bytes(report) + b"\n"
    return report_bytes, state_records


def _response_commit(value: Any) -> str:
    for name in ("oid", "commit_id"):
        candidate = getattr(value, name, None)
        if isinstance(candidate, str) and _COMMIT.fullmatch(candidate):
            return candidate
    raise ValueError("HF commit response omitted immutable commit")


def _operation(factory: Callable[..., Any], path: str, payload: bytes) -> Any:
    return factory(path_in_repo=path, path_or_fileobj=io.BytesIO(payload))


def _repo_snapshot(
    api: Any,
    contract: Fresh16EvaluationContract,
    *,
    revision: str = "main",
) -> tuple[str, tuple[str, ...]]:
    destination = contract.destination
    info = api.repo_info(destination["repo"], repo_type="dataset", revision=revision)
    if getattr(info, "private", None) is not True:
        raise ValueError("fresh16 destination must remain private")
    commit = getattr(info, "sha", None)
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise ValueError("fresh16 destination revision is invalid")
    files = tuple(
        sorted(
            api.list_repo_files(
                destination["repo"], repo_type="dataset", revision=commit
            )
        )
    )
    if len(files) != len(set(files)):
        raise ValueError("fresh16 destination file inventory is duplicated")
    return commit, files


def remote_target_partition(remote_files: Sequence[str]) -> tuple[set[str], set[str]]:
    payload = set(PAYLOAD_TARGETS) & set(remote_files)
    reports = set(REPORT_TARGETS) & set(remote_files)
    if payload not in (set(), set(PAYLOAD_TARGETS)) or reports not in (
        set(),
        set(REPORT_TARGETS),
    ):
        raise ValueError("fresh16 destination is partial or conflicting")
    if reports and payload != set(PAYLOAD_TARGETS):
        raise ValueError("fresh16 reports cannot exist without the payload tree")
    unexpected = set(remote_files) - _ALLOWED_REPO_BASE_FILES - set(PAYLOAD_TARGETS) - set(
        REPORT_TARGETS
    )
    if unexpected:
        raise ValueError("fresh16 destination contains unexpected files")
    return payload, reports


def commit_files(
    *,
    api: Any,
    operation_factory: Callable[..., Any],
    contract: Fresh16EvaluationContract,
    files: Mapping[str, bytes],
    parent: str,
    title: str,
) -> str:
    expected = (
        set(PAYLOAD_TARGETS)
        if title == contract.output["payload_commit"]["commit_title"]
        else set(REPORT_TARGETS)
        if title == contract.output["report_commit"]["commit_title"]
        else None
    )
    if expected is None or set(files) != expected:
        raise ValueError("HF commit file inventory/title differs from the frozen stage")
    response = api.create_commit(
        contract.destination["repo"],
        repo_type="dataset",
        revision="main",
        parent_commit=parent,
        commit_message=title,
        operations=[
            _operation(operation_factory, path, files[path]) for path in sorted(files)
        ],
    )
    return _response_commit(response)


def _commit_history(
    api: Any,
    contract: Fresh16EvaluationContract,
    *,
    revision: str,
) -> tuple[tuple[str, str], ...]:
    result = []
    for item in api.list_repo_commits(
        contract.destination["repo"],
        repo_type="dataset",
        revision=revision,
        formatted=False,
    ):
        commit = getattr(item, "commit_id", None)
        title = getattr(item, "title", None)
        if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None or not isinstance(
            title, str
        ):
            raise ValueError("fresh16 commit history is malformed")
        result.append((commit, title))
    if not result or len({item[0] for item in result}) != len(result):
        raise ValueError("fresh16 commit history is empty or duplicated")
    return tuple(result)


def validate_two_commit_chain(
    api: Any,
    contract: Fresh16EvaluationContract,
    *,
    base_commit: str,
    payload_commit: str,
    report_commit: str,
) -> None:
    history = _commit_history(api, contract, revision=report_commit)
    expected = (
        (report_commit, contract.output["report_commit"]["commit_title"]),
        (payload_commit, contract.output["payload_commit"]["commit_title"]),
    )
    if len({base_commit, payload_commit, report_commit}) != 3 or history[:2] != expected:
        raise ValueError("fresh16 payload/report commits are not one direct two-commit chain")
    if len(history) < 3 or history[2][0] != base_commit:
        raise ValueError("fresh16 report chain base drifted")
    payload_revision, payload_files = _repo_snapshot(
        api, contract, revision=payload_commit
    )
    if (
        payload_revision != payload_commit
        or set(payload_files) - _ALLOWED_REPO_BASE_FILES != set(PAYLOAD_TARGETS)
    ):
        raise ValueError("fresh16 payload revision is not the exact nine-file stage")
    main, files = _repo_snapshot(api, contract)
    if main != report_commit or set(files) - _ALLOWED_REPO_BASE_FILES != set(
        (*PAYLOAD_TARGETS, *REPORT_TARGETS)
    ):
        raise ValueError("fresh16 final main tree inventory drifted")


def _tag_snapshot(
    api: Any,
    *,
    repo: str,
    tag: str,
) -> tuple[str, str] | None:
    refs = api.list_repo_refs(repo, repo_type="dataset")
    matches = [item for item in refs.tags if item.name == tag]
    if len(matches) > 1:
        raise ValueError("fresh16 tag is duplicated")
    if not matches:
        return None
    object_id = getattr(matches[0], "target_commit", None)
    resolved = getattr(api.repo_info(repo, repo_type="dataset", revision=tag), "sha", None)
    if (
        not isinstance(object_id, str)
        or _COMMIT.fullmatch(object_id) is None
        or not isinstance(resolved, str)
        or _COMMIT.fullmatch(resolved) is None
        or object_id == resolved
    ):
        raise ValueError("fresh16 tag must be an annotated immutable tag")
    return object_id, resolved


def create_or_validate_tag(
    api: Any,
    contract: Fresh16EvaluationContract,
    *,
    report_commit: str,
) -> tuple[str, int]:
    destination = contract.destination
    existing = _tag_snapshot(api, repo=destination["repo"], tag=destination["tag"])
    if existing is not None:
        if existing[1] != report_commit:
            raise ValueError("fresh16 tag points to another report commit")
        return existing[0], 0
    api.create_tag(
        destination["repo"],
        repo_type="dataset",
        tag=destination["tag"],
        tag_message=destination["tag_message"],
        revision=report_commit,
        exist_ok=False,
    )
    observed = _tag_snapshot(api, repo=destination["repo"], tag=destination["tag"])
    if observed is None or observed[1] != report_commit:
        raise ValueError("fresh16 tag did not bind the report commit")
    return observed[0], 1


def fresh_download_compare(
    *,
    download_fn: Callable[..., str],
    contract: Fresh16EvaluationContract,
    revision: str,
    expected: Mapping[str, bytes],
    fresh_parent: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="gate-v1-fresh16-readback-", dir=fresh_parent) as raw:
        root = Path(raw).resolve()
        for relative in sorted(expected):
            returned = Path(
                download_fn(
                    repo_id=contract.destination["repo"],
                    repo_type="dataset",
                    filename=relative,
                    revision=revision,
                    local_dir=root,
                    force_download=True,
                )
            )
            if returned != root / relative or not returned.is_absolute():
                raise ValueError("fresh16 readback returned a noncanonical path")
            if _regular_file_bytes(returned, label=relative) != expected[relative]:
                raise ValueError(f"fresh16 immutable readback differs: {relative}")


def _metric_replays(
    state: GateState,
    value: Any,
    *,
    label: str,
) -> tuple[int, ...]:
    record = _mapping(value, label)
    if set(record) != {"selected", "raw_utility", "normalized_recovery"}:
        raise ValueError(f"{label} metric schema drifted")
    selected_raw = record["selected"]
    if not isinstance(selected_raw, list) or any(type(item) is not int for item in selected_raw):
        raise ValueError(f"{label} selection is malformed")
    selected = tuple(selected_raw)
    if (
        selected != tuple(sorted(selected))
        or len(set(selected)) != len(selected)
        or len(selected) > 2
        or not set(selected).issubset(state.candidate_event_step_ids)
    ):
        raise ValueError(f"{label} selection is infeasible")
    raw = state.table.utility(selected)
    baseline = state.table.distance(())
    normalized = raw / baseline if baseline > 1e-12 else None
    if record["raw_utility"] != raw or record["normalized_recovery"] != normalized:
        raise ValueError(f"{label} utility does not replay exact labels")
    return selected


def validate_primary_state_evaluations(
    states: Sequence[GateState],
    rows: Sequence[Mapping[str, Any]],
    *,
    learned: Mapping[str, Mapping[str, Sequence[int]]],
    heuristics: Mapping[str, Mapping[str, Sequence[int]]],
) -> Mapping[str, Mapping[str, Any]]:
    if len(states) != len(rows):
        raise ValueError("primary state evaluation row denominator drifted")
    result = {}
    for state, wrapper in zip(states, rows, strict=True):
        record = _mapping(wrapper.get("evaluation"), "primary state evaluation")
        exact = primary_exact_subset_oracle(state.table)
        if (
            record.get("source_id") != state.source_id
            or record.get("event_count") != len(state.candidate_event_step_ids)
            or record.get("baseline") != state.table.distance(())
        ):
            raise ValueError("primary state evaluation identity/baseline drifted")
        exact_selected = _metric_replays(state, record.get("exact"), label="exact")
        if exact_selected != exact.coalition or record["exact"]["raw_utility"] != exact.utility:
            raise ValueError("primary state exact oracle drifted")
        conditional = _metric_replays(
            state, record.get("conditional"), label="conditional"
        )
        independent = _metric_replays(
            state, record.get("independent"), label="independent"
        )
        if (
            conditional != tuple(learned["conditional"][state.state_id])
            or independent != tuple(learned["independent"][state.state_id])
        ):
            raise ValueError("primary learned state selection differs from payload")
        heuristic_records = _mapping(record.get("heuristics"), "state heuristics")
        if tuple(heuristic_records) != HEURISTIC_ORDER:
            raise ValueError("primary state heuristic order drifted")
        for name in HEURISTIC_ORDER:
            selected = _metric_replays(
                state, heuristic_records[name], label=f"heuristic {name}"
            )
            if selected != tuple(heuristics[name][state.state_id]):
                raise ValueError("primary heuristic state selection differs from payload")
        for family in ("conditional", "independent"):
            values = record.get(f"seed_{family}")
            if not isinstance(values, list) or len(values) != 5:
                raise ValueError("primary state seed metric denominator drifted")
            for seed, value in enumerate(values):
                _metric_replays(state, value, label=f"seed {family} {seed}")
        additions = record.get("conditional_selected_addition_count")
        nonpositive = record.get("conditional_true_nonpositive_addition_count")
        trace = wrapper.get("conditional_selection_trace")
        if not isinstance(trace, list):
            raise ValueError("primary conditional selection trace is missing")
        traced_events: list[int] = []
        for item in trace:
            item = _mapping(item, "primary conditional selection trace item")
            event = item.get("event_step_id")
            predicted = item.get("predicted_marginal_gain")
            if (
                set(item) != {"event_step_id", "predicted_marginal_gain"}
                or type(event) is not int
                or event in traced_events
                or event not in state.candidate_event_step_ids
                or isinstance(predicted, bool)
                or not isinstance(predicted, (int, float))
                or not math.isfinite(float(predicted))
                or float(predicted) <= 0.0
            ):
                raise ValueError("primary conditional selection trace item drifted")
            traced_events.append(event)
        if (
            additions != len(conditional)
            or type(nonpositive) is not int
            or not 0 <= nonpositive <= additions
            or len(trace) != additions
            or tuple(sorted(traced_events)) != conditional
        ):
            raise ValueError("primary conditional trace denominator drifted")
        result[state.state_id] = record
    return result


def recompute_primary_metrics_from_state_records(
    states: Sequence[GateState],
    records: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    def metric(method: str, name: str) -> dict[str, Any]:
        return {state.state_id: records[state.state_id][method][name] for state in states}

    conditional_normalized = metric("conditional", "normalized_recovery")
    independent_normalized = metric("independent", "normalized_recovery")
    conditional_mean, excluded_states, excluded_trajectories = _trajectory_equal_mean(
        states, conditional_normalized
    )
    exact_normalized = metric("exact", "normalized_recovery")
    exact_mean, _, _ = _trajectory_equal_mean(states, exact_normalized)
    heuristic_means = {}
    heuristic_deltas = {}
    heuristic_values = {}
    for name in HEURISTIC_ORDER:
        values = {
            state.state_id: records[state.state_id]["heuristics"][name][
                "normalized_recovery"
            ]
            for state in states
        }
        heuristic_values[name] = values
        mean, _, _ = _trajectory_equal_mean(states, values)
        heuristic_means[name] = mean
        heuristic_deltas[name] = conditional_mean - mean
    strongest = max(
        HEURISTIC_ORDER,
        key=lambda name: (heuristic_means[name], -HEURISTIC_ORDER.index(name)),
    )
    strongest_deltas = _trajectory_deltas(
        states, conditional_normalized, heuristic_values[strongest]
    )
    set_deltas = _trajectory_deltas(
        states, conditional_normalized, independent_normalized
    )
    seed_means = []
    seed_exact_ratios = []
    seed_raw_ratios = []
    paired_seed_deltas = []
    exact_raw = metric("exact", "raw_utility")
    exact_raw_mean, _, _ = _trajectory_equal_mean(states, exact_raw)
    for seed in range(5):
        conditional_seed = {
            state.state_id: records[state.state_id]["seed_conditional"][seed][
                "normalized_recovery"
            ]
            for state in states
        }
        independent_seed = {
            state.state_id: records[state.state_id]["seed_independent"][seed][
                "normalized_recovery"
            ]
            for state in states
        }
        mean, _, _ = _trajectory_equal_mean(states, conditional_seed)
        seed_means.append(mean)
        seed_exact_ratios.append(mean / exact_mean)
        seed_raw = {
            state.state_id: records[state.state_id]["seed_conditional"][seed][
                "raw_utility"
            ]
            for state in states
        }
        raw_mean, _, _ = _trajectory_equal_mean(states, seed_raw)
        seed_raw_ratios.append(raw_mean / exact_raw_mean)
        paired_seed_deltas.append(
            statistics.fmean(
                _trajectory_deltas(
                    states, conditional_seed, independent_seed
                ).values()
            )
        )
    hard = [state for state in states if len(state.candidate_event_step_ids) in {3, 4}]
    hard_conditional, _, _ = _trajectory_equal_mean(
        hard, metric("conditional", "raw_utility")
    )
    hard_exact, _, _ = _trajectory_equal_mean(hard, exact_raw)
    conditional_raw_mean, _, _ = _trajectory_equal_mean(
        states, metric("conditional", "raw_utility")
    )
    independent_raw_mean, _, _ = _trajectory_equal_mean(
        states, metric("independent", "raw_utility")
    )
    selected_count = sum(
        record["conditional_selected_addition_count"] for record in records.values()
    )
    nonpositive = sum(
        record["conditional_true_nonpositive_addition_count"]
        for record in records.values()
    )
    return {
        "ensemble_normalized_recovery": conditional_mean,
        "exact_normalized_recovery": exact_mean,
        "ensemble_raw_utility_over_exact_raw": conditional_raw_mean / exact_raw_mean,
        "normalized_excluded_state_count": excluded_states,
        "normalized_excluded_trajectory_count": excluded_trajectories,
        "raw_retained_small_Dempty_state_count": excluded_states,
        "heuristic_mean_normalized_recovery": heuristic_means,
        "conditional_minus_heuristic_mean_normalized_delta": heuristic_deltas,
        "strongest_heuristic": strongest,
        "strongest_heuristic_positive_trajectory_count": sum(
            value > 0.0 for value in strongest_deltas.values()
        ),
        "strongest_heuristic_paired_bootstrap_lower": paired_bootstrap_lower(
            strongest_deltas
        ),
        "individual_seed_mean_normalized_recovery": seed_means,
        "individual_seed_exact_normalized_ratios": seed_exact_ratios,
        "individual_seed_exact_raw_ratios": seed_raw_ratios,
        "individual_seed_ratio_at_least_0_75_count": sum(
            value >= 0.75 for value in seed_exact_ratios
        ),
        "seed_mean_recovery_population_std": statistics.pstdev(seed_means),
        "hard_n3_n4_raw_utility_ratio": hard_conditional / hard_exact,
        "selected_true_nonpositive_addition_rate": (
            nonpositive / selected_count if selected_count else 0.0
        ),
        "conditional_minus_independent_mean_normalized_delta": statistics.fmean(
            set_deltas.values()
        ),
        "conditional_minus_independent_raw_utility_delta": (
            conditional_raw_mean - independent_raw_mean
        ),
        "conditional_minus_independent_positive_trajectory_count": sum(
            value > 0.0 for value in set_deltas.values()
        ),
        "conditional_minus_independent_paired_bootstrap_lower": paired_bootstrap_lower(
            set_deltas
        ),
        "paired_seed_normalized_deltas": paired_seed_deltas,
        "paired_seed_positive_count": sum(value > 0.0 for value in paired_seed_deltas),
    }


_RUN_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "status",
        "contract_sha256",
        "source",
        "runtime_provenance",
        "payload_commit",
        "payload_inventory",
        "primary_report_sha256",
        "primary_state_records_sha256",
        "execution_planned_operation_contract",
        "planned_local_precommit_replay_operation_contract",
        "gate_trained",
        "fresh16_primary_evaluated",
        "legacy_dev5_semantic_decode_count",
        "confirm20_access_count",
        "matched_nll_evaluation_count",
        "closed_loop_episode_count",
    }
)
_BUNDLE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "status",
        "payload_commit",
        "payload_files",
        "report_files_excluding_bundle",
        "final_target_count",
        "report_commit_embedded",
    }
)


def _bind_loaded_formal_provenance(
    report_formal_provenance: FormalDecisionProvenance,
    loaded_formal_provenance: FormalDecisionProvenance | None,
) -> FormalDecisionProvenance:
    if loaded_formal_provenance is None:
        return report_formal_provenance
    if loaded_formal_provenance != report_formal_provenance:
        raise ValueError("primary report ensemble provenance differs from loaded models")
    return loaded_formal_provenance


def _validate_run_manifest_schema(run: Mapping[str, Any]) -> None:
    if (
        set(run) != _RUN_MANIFEST_KEYS
        or run.get("schema_version") != SCHEMA_VERSION
        or run.get("protocol_id") != PROTOCOL_ID
        or run.get("status") != FINAL_STATUS
        or run.get("gate_trained") is not True
        or run.get("fresh16_primary_evaluated") is not True
        or run.get("legacy_dev5_semantic_decode_count") != 0
        or run.get("confirm20_access_count") != 0
        or run.get("matched_nll_evaluation_count") != 0
        or run.get("closed_loop_episode_count") != 0
    ):
        raise ValueError("run manifest exact schema or firewall status drifted")


def _validate_bundle_manifest_schema(bundle: Mapping[str, Any]) -> None:
    if (
        set(bundle) != _BUNDLE_MANIFEST_KEYS
        or bundle.get("schema_version") != SCHEMA_VERSION
        or bundle.get("protocol_id") != PROTOCOL_ID
        or bundle.get("status") != "FROZEN_GATE_V1_FRESH16_BUNDLE_MANIFEST_V1"
        or bundle.get("final_target_count") != 13
        or bundle.get("report_commit_embedded") is not False
    ):
        raise ValueError("bundle manifest exact schema or status drifted")


def replay_completed_files(
    contract: Fresh16EvaluationContract,
    *,
    files: Mapping[str, bytes],
    payload_commit: str,
    report_commit: str | None,
    source: SourceIdentity,
    runtime_identity: RuntimeIdentity,
    formal_provenance: FormalDecisionProvenance | None = None,
) -> Mapping[str, Any]:
    """Replay canonical structure, sealed decisions, true-D metrics, and lineage."""
    if set(files) != set((*PAYLOAD_TARGETS, *REPORT_TARGETS)):
        raise ValueError("completed fresh16 file inventory drifted")
    blind = {path: files[path] for path in LABEL_BLIND_PAYLOAD_TARGETS}
    seal = seal_label_blind_files(blind)
    features = read_feature_states_jsonl(files[FEATURE_PAYLOAD_PATH])
    labels = read_label_states_jsonl(files[LABEL_PAYLOAD_PATH])
    states = join_feature_and_label_states(
        features,
        labels,
        expected_source_ids=tuple(contract.geometry["source_ids"]),
    )
    if len(states) != 48:
        raise ValueError("completed fresh16 state denominator drifted")

    heuristics = {
        name: read_selection_artifact(files[path], expected_name=name)
        for name, path in HEURISTIC_PATHS.items()
    }
    report_payload = files[PRIMARY_REPORT_PATH]
    report = _strict_json(report_payload, label="primary report")
    if canonical_json_bytes(report) + b"\n" != report_payload:
        raise ValueError("primary report is not canonical JSON")
    claimed_report_sha = report.get("report_sha256")
    without_sha = dict(report)
    without_sha.pop("report_sha256", None)
    if (
        claimed_report_sha != canonical_report_sha256(without_sha)
        or seal_report_threshold_contract(report, contract) != report
    ):
        raise ValueError("primary report hash or GO threshold replay drifted")
    _validate_frozen_primary_report(report)
    provenance = _mapping(report.get("provenance"), "primary report provenance")
    state_inventory = [
        {
            "source_id": state.source_id,
            "state_id": state.state_id,
            "decision_step_id": state.decision_step_id,
            "candidate_event_step_ids": list(state.candidate_event_step_ids),
        }
        for state in states
    ]
    if provenance.get("state_inventory_sha256") != sha256_bytes(
        canonical_json_bytes(state_inventory)
    ):
        raise ValueError("primary report state inventory provenance drifted")
    feature_binding = _mapping(
        provenance.get("evaluation_feature_artifact"), "feature binding"
    )
    label_binding = _mapping(
        provenance.get("evaluation_label_artifact"), "label binding"
    )
    expected_binding = lambda path: {
        "repository": contract.destination["repo"],
        "revision": payload_commit,
        "path": path,
        "sha256": sha256_bytes(files[path]),
    }
    if (
        dict(feature_binding) != expected_binding(FEATURE_PAYLOAD_PATH)
        or dict(label_binding) != expected_binding(LABEL_PAYLOAD_PATH)
    ):
        raise ValueError("primary report feature/label artifact binding drifted")
    heuristic_bindings = provenance.get("heuristic_artifacts")
    if not isinstance(heuristic_bindings, list) or len(heuristic_bindings) != 3:
        raise ValueError("primary report heuristic provenance inventory drifted")
    for item, name in zip(
        heuristic_bindings,
        ("dynamic_recent", "ocr_rgb_v2", "policy_vision_v3"),
        strict=True,
    ):
        item = _mapping(item, "heuristic provenance")
        selections = read_selection_artifact(
            files[HEURISTIC_PATHS[name]], expected_name=name
        )
        if (
            item.get("name") != name
            or item.get("artifact") != expected_binding(HEURISTIC_PATHS[name])
            or item.get("selection_sha256")
            != canonical_selection_sha256(selections)
        ):
            raise ValueError("primary report heuristic artifact binding drifted")

    report_formal_provenance = FormalDecisionProvenance(
        conditional=frozen_ensemble_provenance_from_manifest(
            provenance["conditional_ensemble"]
        ),
        independent=frozen_ensemble_provenance_from_manifest(
            provenance["independent_ensemble"]
        ),
    )
    formal_provenance = _bind_loaded_formal_provenance(
        report_formal_provenance, formal_provenance
    )
    replayed_report_payload, replayed_state_payload = _build_primary_sealed_outputs(
        contract,
        states=states,
        formal_provenance=formal_provenance,
        payload_files={path: files[path] for path in PAYLOAD_TARGETS},
        payload_commit=payload_commit,
    )
    if replayed_report_payload != report_payload:
        raise ValueError("sealed-decision replay differs from primary report bytes")

    state_payload = files[STATE_RECORD_PATH]
    if replayed_state_payload != state_payload:
        raise ValueError("sealed-decision replay differs from primary state-record bytes")
    state_lines = _jsonl_lines(state_payload, count=48, label="primary state records")
    parsed_state_rows = []
    for ordinal, (line, state) in enumerate(zip(state_lines, states, strict=True)):
        record = _strict_json(line, label="primary state record")
        if (
            canonical_json_bytes(record) != line
            or record.get("ordinal") != ordinal
            or record.get("source_id") != state.source_id
            or record.get("state_id") != state.state_id
            or record.get("decision_step_id") != state.decision_step_id
            or record.get("candidate_event_step_ids")
            != list(state.candidate_event_step_ids)
        ):
            raise ValueError("primary state record canonical identity drifted")
        parsed_state_rows.append(record)
    if (
        report.get("source_count") != 16
        or report.get("state_count") != 48
    ):
        raise ValueError("primary report denominator drifted")

    run_payload = files[RUN_MANIFEST_PATH]
    run = _strict_json(run_payload, label="run manifest")
    if canonical_json_bytes(run) + b"\n" != run_payload:
        raise ValueError("run manifest is not canonical JSON")
    _validate_run_manifest_schema(run)
    expected_payload_inventory = list(
        _file_inventory({path: files[path] for path in PAYLOAD_TARGETS})
    )
    expected_counts = contract.data["execution_planned_operation_contract"]
    _validate_run_source_and_runtime_identity(
        run,
        source=source,
        runtime_identity=runtime_identity,
        payload_files=files,
    )
    if (
        run.get("contract_sha256") != contract.sha256
        or run.get("payload_commit") != payload_commit
        or run.get("payload_inventory") != expected_payload_inventory
        or run.get("primary_report_sha256") != sha256_bytes(report_payload)
        or run.get("primary_state_records_sha256") != sha256_bytes(state_payload)
        or run.get("execution_planned_operation_contract")
        != expected_counts
        or run.get("planned_local_precommit_replay_operation_contract")
        != _sealed_output_replay_operation_counts(contract)
    ):
        raise ValueError("run manifest lineage, inventory, or denominator drifted")

    bundle_payload = files[BUNDLE_MANIFEST_PATH]
    bundle = _strict_json(bundle_payload, label="bundle manifest")
    if canonical_json_bytes(bundle) + b"\n" != bundle_payload:
        raise ValueError("bundle manifest is not canonical JSON")
    partial_reports = {
        path: files[path]
        for path in (STATE_RECORD_PATH, PRIMARY_REPORT_PATH, RUN_MANIFEST_PATH)
    }
    _validate_bundle_manifest_schema(bundle)
    if (
        bundle.get("payload_commit") != payload_commit
        or bundle.get("payload_files") != expected_payload_inventory
        or bundle.get("report_files_excluding_bundle")
        != list(_file_inventory(partial_reports))
    ):
        raise ValueError("bundle manifest inventory or lineage drifted")
    if report_commit is not None and any(
        report_commit.encode("ascii") in files[path] for path in REPORT_TARGETS
    ):
        raise ValueError("report artifacts illegally embed their own report commit")
    return {
        "state_count": len(states),
        "label_blind_inventory_sha256": seal.inventory_sha256,
        "report_sha256": claimed_report_sha,
        "go_selector": report["go_selector"],
        "go_set_conditioning_primary": report["go_set_conditioning_primary"],
    }


def replay_primary_outputs_with_models(
    contract: Fresh16EvaluationContract,
    *,
    files: Mapping[str, bytes],
    payload_commit: str,
    report_commit: str,
    model_payloads: Mapping[str, bytes],
    source: SourceIdentity,
    runtime_identity: RuntimeIdentity,
) -> Mapping[str, Any]:
    """Load all ten CPU checkpoints and byte-replay every model-dependent output."""
    if set(files) != set((*PAYLOAD_TARGETS, *REPORT_TARGETS)):
        raise ValueError("model-backed replay requires the complete 13-file tree")
    payload_files = {path: files[path] for path in PAYLOAD_TARGETS}
    features = read_feature_states_jsonl(payload_files[FEATURE_PAYLOAD_PATH])
    labels = read_label_states_jsonl(payload_files[LABEL_PAYLOAD_PATH])
    states = join_feature_and_label_states(
        features,
        labels,
        expected_source_ids=tuple(contract.geometry["source_ids"]),
    )
    ensembles = load_formal_ensembles(contract, model_payloads)
    formal_provenance = FormalDecisionProvenance(
        conditional=ensembles.conditional_provenance,
        independent=ensembles.independent_provenance,
    )
    configured_provenance = {
        item["family"]: item["provenance_sha256"]
        for item in contract.model["ensemble_manifests"]
    }
    if (
        formal_provenance.conditional.sha256
        != configured_provenance.get("conditional")
        or formal_provenance.independent.sha256
        != configured_provenance.get("independent")
    ):
        raise ValueError("loaded model provenance differs from the frozen contract")
    learned = _learned_decisions_from_feature_states(features, ensembles)
    for family in ("conditional", "independent"):
        replayed = learned_decision_artifact_bytes(
            family,
            learned.family(family),
        )
        if replayed != payload_files[LEARNED_PATHS[family]]:
            raise ValueError(
                f"immutable model replay differs from learned {family} selection bytes"
            )
    replay = replay_completed_files(
        contract,
        files=files,
        payload_commit=payload_commit,
        report_commit=report_commit,
        source=source,
        runtime_identity=runtime_identity,
        formal_provenance=formal_provenance,
    )
    expected = _mapping(
        contract.data["execution_planned_operation_contract"],
        "planned execution operation contract",
    )
    counts = {
        "formal_model_file_download_count": len(model_payloads),
        "checkpoint_load_count": ensembles.checkpoint_load_count,
        "learned_model_decision_replay_state_count": len(states),
        "exact_subset_oracle_invocation_count": 2 * len(states),
        "bootstrap_interval_count": 2,
        "bootstrap_resamples_per_interval": expected[
            "bootstrap_resamples_per_interval"
        ],
        "bootstrap_resample_draw_count": 2
        * expected["bootstrap_resamples_per_interval"],
        "learned_selection_byte_compare_count": 2,
        "sealed_learned_decision_artifact_validation_count": 4,
        "primary_report_byte_compare_count": 1,
        "primary_state_record_byte_compare_count": 1,
    }
    if (
        len(model_payloads) != 12
        or ensembles.checkpoint_load_count != 10
        or len(states) != FRESH_STATE_COUNT
        or counts["exact_subset_oracle_invocation_count"]
        != expected["exact_subset_oracle_invocation_count"]
        or counts["bootstrap_interval_count"] != expected["bootstrap_interval_count"]
        or counts["bootstrap_resample_draw_count"]
        != expected["bootstrap_resample_draw_count"]
    ):
        raise ValueError("immutable model replay operation denominator drifted")
    return {
        "state_count": len(states),
        "checkpoint_load_count": ensembles.checkpoint_load_count,
        "conditional_selection_sha256": canonical_selection_sha256(
            {
                state_id: tuple(record["ensemble_selected_event_step_ids"])
                for state_id, record in learned.conditional.items()
            }
        ),
        "independent_selection_sha256": canonical_selection_sha256(
            {
                state_id: tuple(record["ensemble_selected_event_step_ids"])
                for state_id, record in learned.independent.items()
            }
        ),
        "operation_counts": counts,
        "go_selector": replay["go_selector"],
        "go_set_conditioning_primary": replay["go_set_conditioning_primary"],
    }


def _sealed_output_replay_operation_counts(
    contract: Fresh16EvaluationContract,
) -> Mapping[str, int]:
    expected = _mapping(
        contract.data["execution_planned_operation_contract"],
        "planned execution operation contract",
    )
    return {
        "formal_model_file_download_count": 0,
        "checkpoint_load_count": 0,
        "learned_model_decision_replay_state_count": 0,
        "sealed_learned_decision_artifact_validation_count": 4,
        "exact_subset_oracle_invocation_count": expected[
            "exact_subset_oracle_invocation_count"
        ],
        "bootstrap_interval_count": expected["bootstrap_interval_count"],
        "bootstrap_resamples_per_interval": expected[
            "bootstrap_resamples_per_interval"
        ],
        "bootstrap_resample_draw_count": expected["bootstrap_resample_draw_count"],
        "primary_report_byte_compare_count": 1,
        "primary_state_record_byte_compare_count": 1,
    }


def validate_completed_remote(
    *,
    api: Any,
    download_fn: Callable[..., str],
    contract: Fresh16EvaluationContract,
    fresh_parent: Path,
    runtime_identity: RuntimeIdentity,
    source: SourceIdentity,
) -> Mapping[str, Any]:
    if not isinstance(runtime_identity, RuntimeIdentity) or not isinstance(
        source, SourceIdentity
    ):
        raise ValueError("completed validation requires the frozen finalize runtime")
    report_commit, files = _repo_snapshot(api, contract)
    remote_target_partition(files)
    if set(files) - _ALLOWED_REPO_BASE_FILES != set((*PAYLOAD_TARGETS, *REPORT_TARGETS)):
        raise ValueError("fresh16 completed remote target inventory drifted")
    history = _commit_history(api, contract, revision=report_commit)
    if len(history) < 3:
        raise ValueError("fresh16 completed remote history is too short")
    payload_commit = history[1][0]
    base_commit = history[2][0]
    validate_two_commit_chain(
        api,
        contract,
        base_commit=base_commit,
        payload_commit=payload_commit,
        report_commit=report_commit,
    )
    tag = _tag_snapshot(
        api,
        repo=contract.destination["repo"],
        tag=contract.destination["tag"],
    )
    if tag is None or tag[1] != report_commit:
        raise ValueError("fresh16 completion tag drifted")
    observed = {}
    with tempfile.TemporaryDirectory(prefix="gate-v1-fresh16-validate-", dir=fresh_parent) as raw:
        root = Path(raw).resolve()
        for relative in (*PAYLOAD_TARGETS, *REPORT_TARGETS):
            returned = Path(
                download_fn(
                    repo_id=contract.destination["repo"],
                    repo_type="dataset",
                    filename=relative,
                    revision=report_commit,
                    local_dir=root,
                    force_download=True,
                )
            )
            if returned != root / relative:
                raise ValueError("fresh16 validation download path drifted")
            observed[relative] = _regular_file_bytes(returned, label=relative)
    payload_observed = {}
    with tempfile.TemporaryDirectory(
        prefix="gate-v1-fresh16-validate-payload-", dir=fresh_parent
    ) as raw:
        root = Path(raw).resolve()
        for relative in PAYLOAD_TARGETS:
            returned = Path(
                download_fn(
                    repo_id=contract.destination["repo"],
                    repo_type="dataset",
                    filename=relative,
                    revision=payload_commit,
                    local_dir=root,
                    force_download=True,
                )
            )
            if returned != root / relative:
                raise ValueError("fresh16 payload validation download path drifted")
            payload_observed[relative] = _regular_file_bytes(returned, label=relative)
    if any(payload_observed[path] != observed[path] for path in PAYLOAD_TARGETS):
        raise ValueError("fresh16 report commit modified immutable payload bytes")
    model_payloads = download_formal_model_payloads(
        api=api,
        download_fn=download_fn,
        contract=contract,
        fresh_parent=fresh_parent,
    )
    model_replay = replay_primary_outputs_with_models(
        contract,
        files=observed,
        payload_commit=payload_commit,
        report_commit=report_commit,
        model_payloads=model_payloads,
        source=source,
        runtime_identity=runtime_identity,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": VALIDATE_STATUS,
        "report_commit": report_commit,
        "payload_commit": payload_commit,
        "annotated_tag_object": tag[0],
        "remote_mutation_count": 0,
        "state_count": model_replay["state_count"],
        "runtime_identity_sha256": sha256_bytes(
            canonical_json_bytes(runtime_identity.payload)
        ),
        "docker_inspect_receipt_sha256": runtime_identity.receipt_sha256,
        "validated_immediate_immutable_replay_operation_contract": model_replay[
            "operation_counts"
        ],
        "immutable_model_replay_passed": True,
        "conditional_selection_sha256": model_replay[
            "conditional_selection_sha256"
        ],
        "independent_selection_sha256": model_replay[
            "independent_selection_sha256"
        ],
        "go_selector": model_replay["go_selector"],
        "go_set_conditioning_primary": model_replay[
            "go_set_conditioning_primary"
        ],
        "gate_trained": True,
        "fresh16_primary_evaluated": True,
        "legacy_dev5_semantic_decode_count": 0,
        "confirm20_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
    }


def _configured_under_data_root(data_root: Path, configured: str, *, label: str) -> Path:
    path = PurePosixPath(configured)
    if not path.is_absolute() or not path.parts[:2] == ("/", "data"):
        raise ValueError(f"{label} must be rooted under /data")
    relative = Path(*path.parts[2:])
    root = data_root.resolve()
    result = (root / relative).absolute()
    if result != root and root not in result.parents:
        raise ValueError(f"{label} escaped the provided data root")
    return result


def _execution_state_namespace(contract: Fresh16EvaluationContract) -> str:
    local = _mapping(
        contract.data["local_first_state_machine"],
        "fresh16 local-first state machine",
    )
    value = local.get("execution_namespace", "gate-v1-fresh16-evaluation-v1")
    if not isinstance(value, str):
        raise ValueError("fresh16 execution namespace must be text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or len(path.parts) != 1
        or path.as_posix() != value
        or not value.startswith("gate-v1-fresh16-evaluation-")
    ):
        raise ValueError("fresh16 execution namespace is not a canonical protocol name")
    return value


def _require_absent_run_roots(*roots: Path) -> None:
    """Reject reused artifact or ordered-state namespaces before first write."""
    for root in roots:
        if os.path.lexists(root):
            raise ValueError(f"fresh16 run root must be absent before execution: {root}")


def _local_files(root: Path, files: Mapping[str, bytes], *, mode: int) -> Mapping[str, bytes]:
    for relative in sorted(files):
        _exclusive_or_identical(root / relative, files[relative], mode=mode)
    replay = {
        relative: _regular_file_bytes(root / relative, label=relative, mode=mode)
        for relative in sorted(files)
    }
    if replay != {relative: files[relative] for relative in sorted(files)}:
        raise ValueError("local immutable artifact replay drifted")
    return replay


def _record_by_kind(records: Sequence[Mapping[str, Any]], kind: str) -> Mapping[str, Any]:
    matches = [record for record in records if record.get("kind") == kind]
    if len(matches) != 1:
        raise ValueError(f"derived input lacks exactly one {kind} record")
    return matches[0]


def formal_model_remote_inventory(
    contract: Fresh16EvaluationContract,
) -> tuple[str, ...]:
    """Derive the exact 16-file model tree from the bound completion summary."""
    lineage = _mapping(contract.data["lineage"], "fresh16 lineage")
    completion_binding = _mapping(
        lineage["formal_gate_completion"], "formal gate completion binding"
    )
    relative = _safe_relative(completion_binding["path"], "formal completion path")
    payload = _regular_file_bytes(
        contract.repository_root / relative,
        label="formal gate completion",
    )
    if (
        len(payload) != completion_binding["size_bytes"]
        or sha256_bytes(payload) != completion_binding["sha256"]
    ):
        raise ValueError("formal gate completion binding drifted")
    completion = _strict_json(payload, label="formal gate completion")
    artifact = _mapping(completion["artifact"], "formal completion artifact")
    canonical_hf = _mapping(artifact["canonical_hf"], "formal canonical HF")
    training = _mapping(completion["training"], "formal completion training")
    completion_auxiliary = (
        _mapping(
            _mapping(training["conditional"], "conditional completion")[
                "oof_report"
            ],
            "conditional OOF report",
        ),
        _mapping(
            _mapping(training["independent"], "independent completion")[
                "oof_report"
            ],
            "independent OOF report",
        ),
        _mapping(canonical_hf["run_manifest"], "formal run manifest"),
        _mapping(canonical_hf["bundle_manifest"], "formal bundle manifest"),
    )
    model = contract.model
    configured_auxiliary_raw = tuple(model.get("remote_tree_auxiliary_files", ()))
    configured_auxiliary = tuple(
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in configured_auxiliary_raw
    )
    expected_auxiliary = tuple(
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in completion_auxiliary
    )
    if configured_auxiliary != expected_auxiliary:
        raise ValueError("formal model auxiliary tree differs from completion lineage")
    downloaded = tuple(
        record["path"]
        for record in (*model["ensemble_manifests"], *model["checkpoints"])
    )
    result = tuple(
        sorted((*downloaded, *(record["path"] for record in configured_auxiliary)))
    )
    if len(result) != 16 or len(set(result)) != 16:
        raise ValueError("formal model exact-16 remote inventory drifted")
    return result


def _ensure_empty_destination_base(
    api: Any,
    contract: Fresh16EvaluationContract,
    *,
    pre_mutation_receipt_path: Path,
) -> tuple[str, int, Mapping[str, Any]]:
    destination = contract.destination
    mutation_count = 0
    try:
        base, files = _repo_snapshot(api, contract)
        receipt = {
            "repository_existed": True,
            "base_commit": base,
            "files": list(files),
            "captured_before_first_mutation": True,
        }
        _exclusive_or_identical(
            pre_mutation_receipt_path,
            pretty_json_bytes(receipt),
            mode=0o600,
        )
    except Exception as error:
        if error.__class__.__name__ != "RepositoryNotFoundError":
            raise
        receipt = {
            "repository_existed": False,
            "base_commit": None,
            "files": [],
            "captured_before_first_mutation": True,
        }
        # note (luojiaxuan): The absence claim is fsynced before create_repo. A
        # crash at the mutation boundary therefore cannot erase what base state
        # the run observed before its first remote write.
        _exclusive_or_identical(
            pre_mutation_receipt_path,
            pretty_json_bytes(receipt),
            mode=0o600,
        )
        api.create_repo(
            destination["repo"],
            repo_type="dataset",
            private=True,
            exist_ok=False,
        )
        mutation_count += 1
        base, files = _repo_snapshot(api, contract)
    payload, reports = remote_target_partition(files)
    if payload or reports:
        raise ValueError("run mode requires an empty destination base")
    post = {
        **receipt,
        "actual_base_commit": base,
        "actual_base_files": list(files),
        "actual_parent_captured_after_optional_create": True,
    }
    return base, mutation_count, post


def _download_primary_inputs(
    *,
    api: Any,
    download_fn: Callable[..., str],
    contract: Fresh16EvaluationContract,
    fresh_parent: Path,
) -> tuple[Mapping[str, bytes], Mapping[str, bytes]]:
    derived = contract.derived
    derived_records = tuple(derived["files"])
    expected_paths, allowed_extra_paths = derived_remote_inventory(contract)
    validate_input_repo(
        api,
        repo=derived["repo"],
        repo_type=derived["repo_type"],
        revision=derived["immutable_revision"],
        expected_paths=expected_paths,
        tag=derived["tag"],
        expected_private=True,
        allowed_extra_paths=allowed_extra_paths,
    )
    with tempfile.TemporaryDirectory(prefix="gate-v1-fresh16-derived-", dir=fresh_parent) as raw:
        verified = download_verified_files(
            download_fn=download_fn,
            repo=derived["repo"],
            repo_type=derived["repo_type"],
            revision=derived["immutable_revision"],
            records=derived_records,
            local_root=Path(raw).resolve(),
        )
        derived_payloads = {
            path: _regular_file_bytes(record.local_path, label=path)
            for path, record in verified.items()
        }

    model_payloads = download_formal_model_payloads(
        api=api,
        download_fn=download_fn,
        contract=contract,
        fresh_parent=fresh_parent,
    )
    return derived_payloads, model_payloads


def derived_remote_inventory(
    contract: Fresh16EvaluationContract,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Bind the shared derived repo without expanding the download allowlist."""
    derived = contract.derived
    records = tuple(derived["files"])
    consumed = tuple(
        _safe_relative(record["path"], "derived consumed path") for record in records
    )
    full_raw = derived.get("remote_tree_full_inventory_paths")
    if full_raw is None:
        return consumed, tuple(sorted(_ALLOWED_REPO_BASE_FILES))
    base_raw = derived.get("remote_tree_base_paths")
    auxiliary_raw = derived.get("remote_tree_auxiliary_paths")
    for value, label in (
        (full_raw, "derived full remote-tree paths"),
        (base_raw, "derived remote-tree base paths"),
        (auxiliary_raw, "derived remote-tree auxiliary paths"),
    ):
        if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
            value, Sequence
        ):
            raise ValueError(f"{label} must be a sequence")
    full = tuple(
        _safe_relative(path, "derived full remote-tree path") for path in full_raw
    )
    base = tuple(
        _safe_relative(path, "derived remote-tree base path") for path in base_raw
    )
    auxiliary = tuple(
        _safe_relative(path, "derived remote-tree auxiliary path")
        for path in auxiliary_raw
    )
    groups = (consumed, base, auxiliary)
    if (
        any(len(group) != len(set(group)) for group in groups)
        or len(full) != len(set(full))
        or set(base) != _ALLOWED_REPO_BASE_FILES
        or any(
            set(left) & set(right)
            for index, left in enumerate(groups)
            for right in groups[index + 1 :]
        )
        or set(full) != set().union(*(set(group) for group in groups))
        or tuple(sorted(full)) != full
    ):
        raise ValueError("derived full remote-tree inventory partition drifted")
    return full, ()


def validate_nonlabel_input_repos(
    api: Any,
    contract: Fresh16EvaluationContract,
) -> Mapping[str, Any]:
    """Run metadata-only input checks before creating a formal run namespace."""
    policy = contract.data["policy_vision_input"]
    snapshot_manifest = _strict_json(
        _regular_file_bytes(
            contract.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json",
            label="GUI-Owl snapshot manifest",
        ),
        label="GUI-Owl snapshot manifest",
    )
    snapshot_files = snapshot_manifest.get("files")
    if (
        not isinstance(snapshot_files, list)
        or len(snapshot_files) != policy["snapshot_file_count"]
    ):
        raise ValueError("GUI-Owl metadata preflight snapshot inventory drifted")
    validate_input_repo(
        api,
        repo=policy["repo"],
        repo_type="model",
        revision=policy["immutable_revision"],
        expected_paths=tuple(item["path"] for item in snapshot_files),
        expected_private=False,
    )

    derived = contract.derived
    expected_derived, allowed_derived = derived_remote_inventory(contract)
    validate_input_repo(
        api,
        repo=derived["repo"],
        repo_type=derived["repo_type"],
        revision=derived["immutable_revision"],
        expected_paths=expected_derived,
        tag=derived["tag"],
        expected_private=True,
        allowed_extra_paths=allowed_derived,
    )

    model = contract.model
    validate_input_repo(
        api,
        repo=model["repo"],
        repo_type=model["repo_type"],
        revision=model["manifest_commit"],
        expected_paths=formal_model_remote_inventory(contract),
        tag=model["tag"],
        annotated_tag_object=model["annotated_tag_object"],
        expected_private=True,
    )
    return {
        "policy_path_count": len(snapshot_files),
        "derived_path_count": len(expected_derived),
        "formal_model_path_count": len(formal_model_remote_inventory(contract)),
        "label_repo_access_count": 0,
        "download_count": 0,
        "semantic_decode_count": 0,
    }


def validate_local_policy_projection(
    model_dir: Path,
    snapshot_manifest: Path,
) -> Mapping[str, Any]:
    """Hash the complete local GUI-Owl projection before creating run state."""
    identity = verify_frozen_vision_runtime(
        model_dir=model_dir,
        expected_snapshot_manifest=snapshot_manifest,
    )
    return {
        "model_dir": identity.model_dir,
        "model_repo": identity.model_repo,
        "model_revision": identity.model_revision,
        "snapshot_manifest_sha256": identity.snapshot_manifest_sha256,
        "verified_model_file_count": identity.verified_model_file_count,
        "verified_model_total_bytes": identity.verified_model_total_bytes,
        "transformers_version": identity.transformers_version,
        "transformers_source_sha256": dict(identity.transformers_source_sha256),
        "model_load_count": 0,
        "model_forward_count": 0,
    }


def validate_policy_gpu_assignments(
    devices: Sequence[str],
    gpu_uuids: Sequence[str],
    *,
    gpu_uuid_type_profile: str,
    torch_module: Any | None = None,
) -> Mapping[str, Any]:
    """Bind logical CUDA indices to the two expected H200 UUIDs pre-state."""
    device_values = tuple(devices)
    uuid_values = tuple(gpu_uuids)
    if device_values != ("cuda:0", "cuda:1"):
        raise ValueError("fresh16 policy devices must be exact logical cuda:0/cuda:1")
    if len(uuid_values) != 2 or len(set(uuid_values)) != 2:
        raise ValueError("fresh16 policy GPU UUIDs must be exact and distinct")
    if torch_module is None:
        import torch as torch_module
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 2:
        raise RuntimeError("fresh16 policy GPU preflight requires exactly two CUDA GPUs")
    records = []
    for device_text, expected_uuid in zip(
        device_values,
        uuid_values,
        strict=True,
    ):
        if _canonical_gpu_uuid(expected_uuid) != expected_uuid:
            raise ValueError("fresh16 expected GPU UUID is not canonical")
        device = torch_module.device(device_text)
        if device.type != "cuda" or device.index not in {0, 1}:
            raise ValueError("fresh16 logical CUDA device identity drifted")
        gpu_name = torch_module.cuda.get_device_name(device)
        if gpu_name != GUI_OWL_V2_2_EAGER_EXPECTED_GPU_NAME:
            raise RuntimeError("fresh16 policy GPU preflight requires H200 GPUs")
        identity = _validated_gpu_identity(
            torch=torch_module,
            device=device,
            expected_gpu_uuid=expected_uuid,
            gpu_uuid_type_profile=gpu_uuid_type_profile,
        )
        if identity.get("logical_device_index") != device.index:
            raise RuntimeError("fresh16 GPU logical index mapping drifted")
        records.append(
            {
                "device": device_text,
                "gpu_name": gpu_name,
                **identity,
            }
        )
    return {
        "assignments": records,
        "cuda_device_count": 2,
        "gpu_uuid_type_profile": gpu_uuid_type_profile,
        "model_load_count": 0,
        "model_forward_count": 0,
    }


def download_formal_model_payloads(
    *,
    api: Any,
    download_fn: Callable[..., str],
    contract: Fresh16EvaluationContract,
    fresh_parent: Path,
) -> Mapping[str, bytes]:
    model = contract.model
    model_records = tuple((*model["ensemble_manifests"], *model["checkpoints"]))
    validate_input_repo(
        api,
        repo=model["repo"],
        repo_type=model["repo_type"],
        revision=model["manifest_commit"],
        expected_paths=formal_model_remote_inventory(contract),
        tag=model["tag"],
        annotated_tag_object=model["annotated_tag_object"],
        expected_private=True,
    )
    with tempfile.TemporaryDirectory(prefix="gate-v1-fresh16-model-", dir=fresh_parent) as raw:
        model_payloads = download_sha_bound_payloads(
            download_fn=download_fn,
            repo=model["repo"],
            repo_type=model["repo_type"],
            revision=model["manifest_commit"],
            records=model_records,
            local_root=Path(raw).resolve(),
        )
    return model_payloads


def download_labels_after_claim(
    *,
    api: Any,
    download_fn: Callable[..., str],
    contract: Fresh16EvaluationContract,
    claim: LabelAccessClaim,
    fresh_parent: Path,
) -> bytes:
    if not isinstance(claim, LabelAccessClaim) or claim._token is not _LABEL_CLAIM_TOKEN:
        raise ValueError("label input download requires the sealed label claim")
    labels = contract.labels
    records = (labels["archive"], labels["sidecar"])
    validate_input_repo(
        api,
        repo=labels["repo"],
        repo_type=labels["repo_type"],
        revision=labels["immutable_revision"],
        expected_paths=tuple(record["path"] for record in records),
        tag=labels["tag"],
        annotated_tag_object=labels["annotated_tag_object"],
        expected_private=True,
    )
    with tempfile.TemporaryDirectory(prefix="gate-v1-fresh16-label-", dir=fresh_parent) as raw:
        payloads = download_sha_bound_payloads(
            download_fn=download_fn,
            repo=labels["repo"],
            repo_type=labels["repo_type"],
            revision=labels["immutable_revision"],
            records=records,
            local_root=Path(raw).resolve(),
        )
    archive_record = labels["archive"]
    raw_record = labels["raw_states_member"]
    prefix = _safe_relative(
        labels.get("archive_member_prefix"),
        "repaired label archive member prefix",
    )
    return extract_exact_tar_member(
        payloads[archive_record["path"]],
        expected_archive_sha256=archive_record["sha256"],
        expected_archive_size_bytes=archive_record["size_bytes"],
        expected_member_count=archive_record["member_count"],
        member_path=f"{prefix}/{raw_record['path']}",
        expected_member_sha256=raw_record["sha256"],
        expected_member_size_bytes=raw_record["size_bytes"],
    )


def execute_fresh16_evaluation(
    *,
    mode: str,
    contract: Fresh16EvaluationContract,
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
    execution_preflight: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Run or replay the one sealed fresh-16 primary evaluation state machine."""
    if mode not in {"run", "validate"}:
        raise ValueError("fresh16 execution mode must be run or validate")
    source = validate_execution_b_source(
        contract,
        expected_execution_b_git_commit=expected_execution_b_git_commit,
    )
    fresh_download_parent = fresh_download_parent.resolve()
    if not fresh_download_parent.is_dir() or fresh_download_parent.is_symlink():
        raise ValueError("fresh16 execution requires one real temp root")
    if docker_inspect_receipt is None:
        raise ValueError("run and validate modes require an explicit Docker inspect receipt")
    runtime_identity = validate_execution_runtime(
        contract,
        source=source,
        data_root=data_root,
        receipt_path=docker_inspect_receipt,
    )
    if mode == "validate":
        return validate_completed_remote(
            api=api,
            download_fn=download_fn,
            contract=contract,
            fresh_parent=fresh_download_parent,
            runtime_identity=runtime_identity,
            source=source,
        )
    if (
        len(tuple(devices)) != 2
        or len(tuple(gpu_uuids)) != 2
        or len(set(devices)) != 2
        or len(set(gpu_uuids)) != 2
    ):
        raise ValueError("fresh16 run mode requires exactly two distinct GPUs")
    policy = contract.data["policy_vision_input"]
    gpu_assignment_preflight = validate_policy_gpu_assignments(
        devices,
        gpu_uuids,
        gpu_uuid_type_profile=policy["gpu_uuid_type_profile"],
    )
    artifact_root = _configured_under_data_root(
        data_root,
        contract.data["local_first_state_machine"]["artifact_directory"],
        label="fresh16 artifact directory",
    )
    state_root = _configured_under_data_root(
        data_root,
        contract.data["local_first_state_machine"]["state_directory"],
        label="fresh16 state directory",
    ) / _execution_state_namespace(contract)
    _require_absent_run_roots(artifact_root, state_root)

    # note (luojiaxuan): The operational repair preflight must finish before a
    # new local namespace is created. These checks use metadata only and never
    # touch the label repo, download payload bytes, or decode fresh semantics.
    try:
        _, existing_files = _repo_snapshot(api, contract)
    except Exception as error:
        if error.__class__.__name__ != "RepositoryNotFoundError":
            raise
    else:
        payload_present, reports_present = remote_target_partition(existing_files)
        if payload_present or reports_present:
            raise ValueError("existing fresh16 targets require immutable validate mode")

    snapshot_payload = _regular_file_bytes(snapshot_manifest, label="GUI-Owl snapshot manifest")
    if sha256_bytes(snapshot_payload) != policy["snapshot_manifest_sha256"]:
        raise ValueError("GUI-Owl snapshot manifest digest drifted")
    snapshot = _strict_json(snapshot_payload, label="GUI-Owl snapshot manifest")
    snapshot_files = snapshot.get("files")
    if (
        snapshot.get("repo") != policy["repo"]
        or snapshot.get("revision") != policy["immutable_revision"]
        or not isinstance(snapshot_files, list)
        or len(snapshot_files) != policy["snapshot_file_count"]
    ):
        raise ValueError("GUI-Owl snapshot manifest identity or inventory drifted")
    metadata_preflight = validate_nonlabel_input_repos(api, contract)
    local_policy_projection = validate_local_policy_projection(
        model_dir,
        snapshot_manifest,
    )

    image_root = artifact_root / "selected-images"
    # note (luojiaxuan): Recheck after remote/model preflight so a competing
    # creator cannot claim either non-resumable root during the validation gap.
    _require_absent_run_roots(artifact_root, state_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    state_chain = DurableStateChain(
        state_root,
        contract.data["local_first_state_machine"]["ordered_states"],
    )
    if state_chain.verify_prefix():
        raise ValueError(
            "incomplete prior ordered-state chain requires manual audit; run is not resumable"
        )
    runtime_receipt_payload = {
        "source_git_commit": source.head,
        "contract_sha256": contract.sha256,
        "snapshot_manifest_sha256": sha256_bytes(snapshot_payload),
        "snapshot_file_count": len(snapshot_files),
        "devices": list(devices),
        "gpu_uuids": list(gpu_uuids),
        "policy_model_revision": policy["immutable_revision"],
        "docker_inspect_receipt_sha256": runtime_identity.receipt_sha256,
        "runtime": dict(runtime_identity.payload),
        "nonlabel_metadata_preflight": dict(metadata_preflight),
        "local_policy_projection_preflight": dict(local_policy_projection),
        "policy_gpu_assignment_preflight": dict(gpu_assignment_preflight),
    }
    if execution_preflight is not None:
        runtime_receipt_payload["execution_preflight"] = dict(
            _mapping(execution_preflight, "fresh16 execution preflight")
        )
    state_chain.append("runtime_receipts", runtime_receipt_payload)
    state_chain.append(
        "global_claim",
        {
            "source_git_commit": source.head,
            "contract_sha256": contract.sha256,
            "fresh16_source_ids_sha256": contract.geometry["source_ids_sha256"],
            "label_access_authorized": False,
            "legacy_dev5_access_authorized": False,
            "confirm20_access_authorized": False,
            "matched_nll_authorized": False,
            "closed_loop_authorized": False,
        },
    )

    derived_payloads, model_payloads = _download_primary_inputs(
        api=api,
        download_fn=download_fn,
        contract=contract,
        fresh_parent=fresh_download_parent,
    )
    derived_records = tuple(contract.derived["files"])
    trajectory_record = _record_by_kind(derived_records, "trajectories")
    trajectories = decode_fresh16_trajectories(
        verify_fresh16_trajectory_transport(
            derived_payloads[trajectory_record["path"]]
        ),
        expected_source_ids=tuple(contract.geometry["source_ids"]),
        expected_source_ids_sha256=contract.geometry["source_ids_sha256"],
    )
    work_items, requirements = build_fresh16_image_plan(trajectories)
    ocr_record = _record_by_kind(derived_records, "ocr")
    selected_ocr = extract_selected_ocr_records(
        derived_payloads[ocr_record["path"]],
        requirements,
        expected_sha256=ocr_record["sha256"],
        expected_size_bytes=ocr_record["size_bytes"],
        transport_record_count=ocr_record["record_count"],
    )
    images_record = _record_by_kind(derived_records, "images")
    # note (luojiaxuan): The verified download is copied into a temporary regular
    # file only because tarfile's production path and the worker path have separate
    # lifetimes; selected bytes are then sealed read-only under the artifact root.
    with tempfile.TemporaryDirectory(prefix="gate-v1-fresh16-image-tar-", dir=fresh_download_parent) as raw:
        archive_path = Path(raw) / "images.tar"
        _exclusive_or_identical(
            archive_path,
            derived_payloads[images_record["path"]],
            mode=0o444,
        )
        selected_images = extract_selected_image_payloads(
            archive_path,
            requirements,
            expected_sha256=images_record["sha256"],
            expected_size_bytes=images_record["size_bytes"],
            output_root=image_root,
        )
    state_chain.append(
        "transport_verification",
        {
            "derived_revision": contract.derived["immutable_revision"],
            "derived_file_inventory": list(_file_inventory(derived_payloads)),
            "fresh16_trajectory_semantic_decode_count": 16,
            "fresh16_ocr_semantic_decode_count": 80,
            "fresh16_unique_image_identity_count": len(requirements),
            "selected_image_payload_count": len(selected_images),
        },
    )
    backend_path = contract.repository_root / "code/configs/restoration_v2_ocr_backend.json"
    backend_payload = _regular_file_bytes(backend_path, label="OCR backend config")
    backend_config = _strict_json(backend_payload, label="OCR backend config")
    image_features = materialize_fresh16_image_features(
        requirements,
        ocr_records_by_path=selected_ocr,
        image_payloads_by_path=selected_images,
        backend_config=backend_config,
        backend_config_sha256=sha256_bytes(backend_payload),
    )
    bundle = build_fresh16_label_blind_bundle(trajectories, image_features)
    state_chain.append(
        "label_blind_cpu_completion",
        {
            "feature_state_count": len(bundle.feature_states),
            "candidate_feature_count": sum(
                len(state.candidate_event_step_ids) for state in bundle.feature_states
            ),
            "feature_bytes_sha256": sha256_bytes(
                feature_states_jsonl_bytes(bundle.feature_states)
            ),
            "dynamic_recent_selection_sha256": canonical_selection_sha256(
                bundle.dynamic_recent
            ),
            "ocr_rgb_selection_sha256": canonical_selection_sha256(
                bundle.ocr_rgb_v2
            ),
            "label_semantic_decode_count": 0,
        },
    )

    ensembles = load_formal_ensembles(contract, model_payloads)
    formal_provenance = FormalDecisionProvenance(
        conditional=ensembles.conditional_provenance,
        independent=ensembles.independent_provenance,
    )
    learned = learned_selections_from_features(bundle, ensembles)
    state_chain.append(
        "checkpoint_replay_completion",
        {
            "model_revision": contract.model["manifest_commit"],
            "model_payload_commit": contract.model["payload_commit"],
            "checkpoint_load_count": ensembles.checkpoint_load_count,
            "conditional_provenance_sha256": ensembles.conditional_provenance.sha256,
            "independent_provenance_sha256": ensembles.independent_provenance.sha256,
            "conditional_selection_sha256": canonical_selection_sha256(
                {
                    state_id: tuple(record["ensemble_selected_event_step_ids"])
                    for state_id, record in learned.conditional.items()
                }
            ),
            "independent_selection_sha256": canonical_selection_sha256(
                {
                    state_id: tuple(record["ensemble_selected_event_step_ids"])
                    for state_id, record in learned.independent.items()
                }
            ),
            "optimizer_step_count": 0,
        },
    )

    even, odd, sentinel = policy_worker_partitions(work_items)
    requests = (
        PolicyWorkerRequest(
            worker_id="even",
            work_items=even,
            sentinel=None,
            image_root=str(image_root),
            model_dir=str(model_dir),
            snapshot_manifest=str(snapshot_manifest),
            device=devices[0],
            expected_gpu_uuid=gpu_uuids[0],
            gpu_uuid_type_profile=policy["gpu_uuid_type_profile"],
            image_processor_size_profile=policy["image_processor_size_profile"],
        ),
        PolicyWorkerRequest(
            worker_id="odd",
            work_items=odd,
            sentinel=sentinel,
            image_root=str(image_root),
            model_dir=str(model_dir),
            snapshot_manifest=str(snapshot_manifest),
            device=devices[1],
            expected_gpu_uuid=gpu_uuids[1],
            gpu_uuid_type_profile=policy["gpu_uuid_type_profile"],
            image_processor_size_profile=policy["image_processor_size_profile"],
        ),
    )
    policy_records, policy_selections, policy_counts = run_policy_vision_process_pool(
        requests,
        work_items=work_items,
    )
    for worker_id in ("even", "odd"):
        worker_records = tuple(
            record for record in policy_records if record["worker_id"] == worker_id
        )
        state_chain.append(
            f"policy_worker_{worker_id}_completion",
            {
                "worker_id": worker_id,
                "primary_state_count": len(worker_records),
                "score_record_sha256": sha256_bytes(
                    canonical_json_bytes(worker_records)
                ),
                "device": worker_records[0]["device"],
                "gpu_uuid": worker_records[0]["gpu_uuid"],
                "feature_repeats": 2,
                "cross_device_sentinel_count": 1 if worker_id == "odd" else 0,
            },
        )
    blind_files = build_label_blind_payload_files(
        bundle,
        policy_score_records=policy_records,
        policy_selections=policy_selections,
        learned_selections=learned,
    )
    local_blind = _local_files(artifact_root, blind_files, mode=0o444)
    seal = seal_label_blind_files(local_blind)
    seal_record = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": HEURISTIC_SEAL_STATUS,
        "inventory": list(seal.inventory),
        "inventory_sha256": seal.inventory_sha256,
        "label_access_authorized": False,
    }
    _exclusive_or_identical(
        state_root / "heuristic-local-seal.json",
        pretty_json_bytes(seal_record),
        mode=0o600,
    )
    state_chain.append(
        "heuristic_local_seal",
        {
            "status": HEURISTIC_SEAL_STATUS,
            "label_blind_file_count": len(seal.files),
            "label_blind_inventory_sha256": seal.inventory_sha256,
            "label_access_authorized": False,
        },
    )
    claim = claim_label_access(
        seal,
        source_git_commit=source.head,
        contract_sha256=contract.sha256,
    )
    _exclusive_or_identical(
        state_root / "label-access-claim.json",
        pretty_json_bytes(claim.claim),
        mode=0o600,
    )
    state_chain.append(
        "label_access_claim",
        {
            "claim_sha256": sha256_bytes(pretty_json_bytes(claim.claim)),
            "label_blind_inventory_sha256": claim.heuristic_inventory_sha256,
            "label_access_authorized": True,
            "legacy_dev5_access_authorized": False,
            "confirm20_access_authorized": False,
        },
    )

    raw_state_payload = download_labels_after_claim(
        api=api,
        download_fn=download_fn,
        contract=contract,
        claim=claim,
        fresh_parent=fresh_download_parent,
    )
    labels = decode_labels_after_claim(
        raw_state_payload,
        claim,
        expected_source_ids=tuple(contract.geometry["source_ids"]),
        expected_source_ids_sha256=contract.geometry["source_ids_sha256"],
    )
    label_payload = label_states_jsonl_bytes(labels.labels)
    state_chain.append(
        "label_cache_completion",
        {
            "label_revision": contract.labels["immutable_revision"],
            "raw_state_transport_sha256": sha256_bytes(raw_state_payload),
            "label_payload_sha256": sha256_bytes(label_payload),
            "fresh16_label_semantic_decode_count": labels.semantic_decode_count,
            "distance_value_count": labels.distance_value_decode_count,
            "run_contract_sha256": labels.run_contract_sha256,
        },
    )
    payload_files = {**dict(seal.files), LABEL_PAYLOAD_PATH: label_payload}
    if set(payload_files) != set(PAYLOAD_TARGETS):
        raise ValueError("fresh16 nine-file payload inventory drifted")
    validate_claim_against_payload(claim, seal, payload_files)
    local_payload = _local_files(artifact_root, payload_files, mode=0o444)
    states = join_fresh16_evaluation_states(bundle.feature_states, labels)

    base_commit, mutations, base_receipt = _ensure_empty_destination_base(
        api,
        contract,
        pre_mutation_receipt_path=state_root / "remote-base-pre-mutation.json",
    )
    _exclusive_or_identical(
        state_root / "remote-base-receipt.json",
        pretty_json_bytes(base_receipt),
        mode=0o600,
    )
    state_chain.append(
        "remote_base_receipt",
        {
            **base_receipt,
            "destination_repo": contract.destination["repo"],
            "payload_parent_commit": base_commit,
        },
    )
    payload_commit = commit_files(
        api=api,
        operation_factory=operation_factory,
        contract=contract,
        files=local_payload,
        parent=base_commit,
        title=contract.output["payload_commit"]["commit_title"],
    )
    mutations += 1
    fresh_download_compare(
        download_fn=download_fn,
        contract=contract,
        revision=payload_commit,
        expected=local_payload,
        fresh_parent=fresh_download_parent,
    )
    state_chain.append(
        "payload_commit_receipt",
        {
            "base_commit": base_commit,
            "payload_commit": payload_commit,
            "payload_file_count": len(local_payload),
            "payload_inventory_sha256": sha256_bytes(
                canonical_json_bytes(_file_inventory(local_payload))
            ),
            "immutable_readback_passed": True,
        },
    )
    operation_counts = {
        **dict(contract.data["execution_planned_operation_contract"]),
        **policy_counts,
    }
    report_files = evaluate_and_build_report_files(
        contract,
        states=states,
        formal_provenance=formal_provenance,
        payload_files=local_payload,
        payload_commit=payload_commit,
        source=source,
        operation_counts=operation_counts,
        runtime_identity=runtime_identity,
    )
    local_reports = _local_files(artifact_root, report_files, mode=0o444)
    # note (luojiaxuan): The report-commit-independent validator must pass over
    # the exact local 13-file candidate before the first report or tag mutation.
    local_replay = replay_completed_files(
        contract,
        files={**local_payload, **local_reports},
        payload_commit=payload_commit,
        report_commit=None,
        source=source,
        runtime_identity=runtime_identity,
    )
    local_replay_counts = _sealed_output_replay_operation_counts(contract)
    state_chain.append(
        "primary_report_completion",
        {
            "payload_commit": payload_commit,
            "primary_report_sha256": sha256_bytes(
                local_reports[PRIMARY_REPORT_PATH]
            ),
            "primary_state_records_sha256": sha256_bytes(
                local_reports[STATE_RECORD_PATH]
            ),
            "report_file_count": len(local_reports),
            "state_record_count": 48,
            "local_precommit_replay_passed": True,
            "local_precommit_replay_report_sha256": local_replay[
                "report_sha256"
            ],
            "validated_local_precommit_replay_operation_contract": local_replay_counts,
        },
    )
    report_commit = commit_files(
        api=api,
        operation_factory=operation_factory,
        contract=contract,
        files=local_reports,
        parent=payload_commit,
        title=contract.output["report_commit"]["commit_title"],
    )
    mutations += 1
    validate_two_commit_chain(
        api,
        contract,
        base_commit=base_commit,
        payload_commit=payload_commit,
        report_commit=report_commit,
    )
    state_chain.append(
        "report_commit_receipt",
        {
            "payload_commit": payload_commit,
            "report_commit": report_commit,
            "direct_parent_verified": True,
            "final_target_count": 13,
        },
    )
    tag_object, tag_mutations = create_or_validate_tag(
        api,
        contract,
        report_commit=report_commit,
    )
    mutations += tag_mutations
    final_files = {**local_payload, **local_reports}
    fresh_download_compare(
        download_fn=download_fn,
        contract=contract,
        revision=report_commit,
        expected=final_files,
        fresh_parent=fresh_download_parent,
    )
    validation = validate_completed_remote(
        api=api,
        download_fn=download_fn,
        contract=contract,
        fresh_parent=fresh_download_parent,
        runtime_identity=runtime_identity,
        source=source,
    )
    state_chain.append(
        "completion_staging",
        {
            "report_commit": report_commit,
            "annotated_tag_object": tag_object,
            "tag": contract.destination["tag"],
            "final_readback_passed": True,
            "validation_status": validation["status"],
            "remote_mutation_count": mutations,
        },
    )
    primary_report = _strict_json(local_reports[PRIMARY_REPORT_PATH], label="primary report")
    completion = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": FINAL_STATUS,
        "source_git_commit": source.head,
        "contract_sha256": contract.sha256,
        "base_commit": base_commit,
        "payload_commit": payload_commit,
        "report_commit": report_commit,
        "annotated_tag_object": tag_object,
        "remote_mutation_count": mutations,
        "payload_file_count": 9,
        "report_file_count": 4,
        "state_count": 48,
        "policy_operation_counts": policy_counts,
        "execution_planned_operation_contract": dict(
            contract.data["execution_planned_operation_contract"]
        ),
        "validated_immediate_immutable_replay_operation_contract": validation[
            "validated_immediate_immutable_replay_operation_contract"
        ],
        "validated_local_precommit_replay_operation_contract": local_replay_counts,
        "immutable_model_replay_passed": validation[
            "immutable_model_replay_passed"
        ],
        "go_selector": primary_report["go_selector"],
        "go_set_conditioning_primary": primary_report[
            "go_set_conditioning_primary"
        ],
        "validation_status": validation["status"],
        "gate_trained": True,
        "legacy_dev5_semantic_decode_count": 0,
        "confirm20_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
    }
    final_receipt = state_chain.append("final_completion", completion)
    state_chain.verify_complete()
    completion["ordered_state_chain_final_receipt_sha256"] = sha256_bytes(
        pretty_json_bytes(final_receipt)
    )
    _exclusive_or_identical(
        state_root / "final-completion.json",
        pretty_json_bytes(completion),
        mode=0o600,
    )
    return completion


__all__ = [
    "BUNDLE_MANIFEST_PATH",
    "DurableStateChain",
    "FEATURE_PAYLOAD_PATH",
    "FINAL_STATUS",
    "FormalDecisionProvenance",
    "FormalEnsembles",
    "HEURISTIC_PATHS",
    "HeuristicBytesSeal",
    "LABEL_BLIND_PAYLOAD_TARGETS",
    "LABEL_PAYLOAD_PATH",
    "LEARNED_PATHS",
    "LabelAccessClaim",
    "LearnedDecisionArtifacts",
    "PolicyWorkerRequest",
    "PolicyWorkerResult",
    "PRIMARY_REPORT_PATH",
    "RUN_MANIFEST_PATH",
    "RuntimeIdentity",
    "SOURCE_VALIDATION_STATUS",
    "STATE_RECORD_PATH",
    "SourceIdentity",
    "VALIDATE_STATUS",
    "VerifiedRemoteFile",
    "build_label_blind_payload_files",
    "build_primary_state_records",
    "capture_docker_inspect_receipt",
    "claim_label_access",
    "commit_files",
    "create_or_validate_tag",
    "decode_labels_after_claim",
    "download_formal_model_payloads",
    "download_labels_after_claim",
    "download_verified_files",
    "derived_remote_inventory",
    "evaluate_and_build_report_files",
    "execute_fresh16_evaluation",
    "extract_selected_image_payloads",
    "extract_selected_ocr_records",
    "fresh_download_compare",
    "formal_model_remote_inventory",
    "learned_decision_artifact_bytes",
    "learned_selections_from_features",
    "load_formal_ensembles",
    "load_runner_freeze",
    "materialize_runner_freeze",
    "policy_worker_partitions",
    "remote_target_partition",
    "replay_completed_files",
    "replay_primary_outputs_with_models",
    "read_learned_decision_artifact",
    "run_policy_vision_process_pool",
    "seal_label_blind_files",
    "validate_clean_pushed_source",
    "validate_completed_remote",
    "validate_execution_b_source",
    "validate_execution_runtime",
    "validate_input_repo",
    "validate_nonlabel_input_repos",
    "validate_local_policy_projection",
    "validate_policy_gpu_assignments",
    "validate_operation_counts",
    "validate_policy_worker_results",
    "validate_source_a",
    "validate_two_commit_chain",
    "verify_downloaded_file",
]
