"""Ledger-neutral CPU validation of the invalid expansion-label v1 payload.

The input must be an ``InvalidForensicEvidence`` object returned by the P0
archive reader after strict validation.  The validator never rewrites either
ledger namespace.  It reconstructs the producer view that existed immediately
before invalidation, proves that the frozen validator still fails only at its
cadence gate, replaces that gate with a structural lifecycle check, and then
independently replays the scientific reduction.
"""

from __future__ import annotations

import copy
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import causalcache.restoration_v2_2_expansion_labels_artifact as frozen_v1
from causalcache.restoration_v2_2_expansion_labels_invalid_forensic import (
    ARCHIVE_STATUS as FORENSIC_ARCHIVE_STATUS,
    ATTEMPT_ROOT_NAMESPACE,
    FORENSIC_MANIFEST_PATH,
    PRODUCER_COMPLETED_STATUS,
    PRODUCER_INVALID_STATUS,
    TERMINAL_LEDGER_NAMESPACE,
    InvalidForensicContract,
    InvalidForensicEvidence,
    canonical_json_bytes as forensic_canonical_json_bytes,
    forensic_contract_from_data,
    sha256_bytes as forensic_sha256_bytes,
    strict_pretty_json_object_bytes,
    validate_invalid_forensic_files,
)
from causalcache.restoration_v2_2_expansion_math_audit import (
    audit_normalized_raw_states,
    compare_with_reducer_output,
)


_CAPTURED_FROZEN_EXTERNAL_REPLAY = frozen_v1.replay_external_input_projections
_CAPTURED_FROZEN_EXTERNAL_REPLAY_CODE = (
    _CAPTURED_FROZEN_EXTERNAL_REPLAY.__code__
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = (
    "causalcache_restoration_v2_2_expansion_exact_labels_"
    "ledger_neutral_scientific_repair_v1"
)
PASS_STATUS = "PASS_LEDGER_NEUTRAL_EXPANSION_LABEL_SCIENTIFIC_REPAIR_V1"
TEST_ONLY_STATUS = (
    "TEST_ONLY_LEDGER_NEUTRAL_EXPANSION_LABEL_SCIENTIFIC_REPAIR_V1"
)
LEGACY_CADENCE_FAILURE = (
    "monitor did not cover the complete worker execution window at the "
    "frozen cadence"
)
EXTERNAL_REPLAY_STATUS = "PASS_INJECTED_EXTERNAL_INPUT_REPLAY"
TEST_EXTERNAL_REPLAY_STATUS = "TEST_ONLY_EMBEDDED_FIXTURE_REPLAY"
FORMAL_EXTERNAL_REPLAY_TIER = "FORMAL_FROZEN_EXTERNAL_INPUT_REPLAY"
TEST_EXTERNAL_REPLAY_TIER = "TEST_ONLY_EMBEDDED_FIXTURE_REPLAY"
FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION = MappingProxyType({
    "module": "causalcache.restoration_v2_2_expansion_labels_artifact",
    "qualname": "replay_external_input_projections",
    "source_path": (
        "code/causalcache/restoration_v2_2_expansion_labels_artifact.py"
    ),
    "source_sha256": (
        "a1ba86e672a51dcde807fdb2d844874696ba0e9a35839f6e631390a7ef0b275d"
    ),
    "source_size_bytes": 119_937,
    "callable_source_sha256": (
        "94887261ce11326e6aaf81a28e791cbee4e6200c6eeb339e4f655f277b891f1c"
    ),
    "callable_source_size_bytes": 7_751,
})
FROZEN_OCR_BACKEND_CONFIG = MappingProxyType({
    "path": "code/configs/restoration_v2_ocr_backend.json",
    "sha256": "51e08a9565f804ecb1ee7f12887cc12742b84c04996fffbbde7c0f304e4bd036",
    "size_bytes": 6_922,
})
FROZEN_P0_CONFIG_SHA256 = (
    "5290a51a250e31be4fcf0a970c77ef31c08c92c5892edd19a12ecb14d9d5a6a2"
)
FROZEN_P0_FORENSIC_TREE_SHA256 = (
    "bc481dd77e26764dad3458e91a3e0247ade6049af7adb1744672aa6f2fb437d1"
)
FROZEN_PARENT_SUBSTRATE = MappingProxyType(
    {
        "repo": (
            "gavinlaw/causalcache-restoration-v2-2-"
            "label-expansion-substrate-mobile"
        ),
        "tag": "v2.2-label-expansion-substrate-v1",
        "path": "raw/v2.2-label-expansion-substrate-v1.tar",
        "immutable_revision": "25ac19cf6ef98adc243d421cd0039ac104ddb539",
        "archive_sha256": (
            "4e77a38be34cb2f3c084a13abd47c0530e6729ff6cca72793977c62fa78ff47d"
        ),
        "archive_size_bytes": 7_475_200,
        "tree_inventory_sha256": (
            "56f291053121ccf813698a48beb6269b1fa1d096b7e974f6eb9424f55bc45543"
        ),
        "outcome": "PASS_V2_2_LABEL_EXPANSION_SUBSTRATE_V1",
        "validated_state_count": 192,
    }
)
FROZEN_DERIVED_ARTIFACT = MappingProxyType(
    {
        "repo": "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile",
        "tag": "restoration-v2-label-expansion-v1.0.0",
        "immutable_revision": "630363a6adb692d72774f16dd0653a50216313ff",
        "payload_prefix": "derived/restoration-v2-label-expansion-v1",
        "artifact_tree_sha256": (
            "9394b369e2b741e6aacf9ece4fc5dae3e6337b25a7e65402307e4e7862b94abc"
        ),
    }
)


ExternalReplayCallback = Callable[
    [Mapping[str, Any]],
    tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]],
]


@dataclass(frozen=True)
class FrozenExternalReplayAttestation:
    implementation: Mapping[str, Any]
    parent_substrate: Mapping[str, Any]
    derived_artifact: Mapping[str, Any]
    ocr_backend_config: Mapping[str, Any]
    parent_substrate_archive_path: str
    derived_artifact_root: str
    ocr_backend_config_path: str


@dataclass(frozen=True)
class TestOnlyExternalReplayAttestation:
    fixture_id: str
    fixture_state_projections_sha256: str


@dataclass(frozen=True)
class FrozenExternalReplayExecutor:
    attestation: FrozenExternalReplayAttestation
    _replay_callable: Callable[..., Any] = field(
        default_factory=(
            lambda _callable=_CAPTURED_FROZEN_EXTERNAL_REPLAY: _callable
        ),
        init=False,
        repr=False,
        compare=False,
    )

    def __call__(
        self, _run_contract: Mapping[str, Any]
    ) -> tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]:
        return self._replay_callable(
            parent_substrate_archive=(
                self.attestation.parent_substrate_archive_path
            ),
            derived_artifact_root=self.attestation.derived_artifact_root,
            ocr_backend_config=self.attestation.ocr_backend_config_path,
        )


@dataclass(frozen=True)
class LedgerNeutralScientificValidation:
    report: Mapping[str, Any]
    run_contract: Mapping[str, Any]
    normalized_state_records: tuple[Mapping[str, Any], ...]
    reduction: Mapping[str, Any]


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


def _timestamp(value: Any, label: str) -> datetime:
    return frozen_v1._timestamp(value, label)


def _exact_microseconds(left: datetime, right: datetime) -> int:
    delta = right - left
    return (
        delta.days * 86_400_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )


def _nearest_rank(values: Sequence[int], quantile: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _inventory(files: Mapping[str, bytes]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "path": name,
            "sha256": forensic_sha256_bytes(files[name]),
            "size_bytes": len(files[name]),
        }
        for name in sorted(files)
    )


def _normalize_forensic_files(
    evidence: InvalidForensicEvidence,
    *,
    forensic_contract: InvalidForensicContract,
) -> tuple[dict[str, bytes], Mapping[str, Any], str]:
    if not isinstance(evidence, InvalidForensicEvidence):
        raise TypeError(
            "forensic evidence must be a P0-validated InvalidForensicEvidence"
        )
    if not isinstance(forensic_contract, InvalidForensicContract):
        raise TypeError("forensic contract must be an InvalidForensicContract")
    rebuilt = validate_invalid_forensic_files(
        evidence.files,
        contract=forensic_contract,
    )
    rebuilt_contract = forensic_contract_from_data(forensic_contract.data)
    if (
        rebuilt_contract.data != forensic_contract.data
        or rebuilt_contract.sha256 != forensic_contract.sha256
    ):
        raise ValueError("supplied P0 forensic contract differs from reconstruction")
    if (
        rebuilt.files != evidence.files
        or rebuilt.manifest != evidence.manifest
        or rebuilt.inventory != evidence.inventory
        or rebuilt.tree_inventory_sha256 != evidence.tree_inventory_sha256
    ):
        raise ValueError("supplied P0 evidence differs from strict reconstruction")
    files = dict(rebuilt.files)
    supplied_manifest = dict(rebuilt.manifest)
    supplied_inventory = tuple(rebuilt.inventory)
    supplied_tree_sha256 = rebuilt.tree_inventory_sha256
    if any(
        not isinstance(name, str) or not isinstance(payload, bytes)
        for name, payload in files.items()
    ):
        raise TypeError("forensic evidence paths and payloads must be strings and bytes")
    try:
        manifest = strict_pretty_json_object_bytes(
            files[FORENSIC_MANIFEST_PATH], label="P0 forensic manifest"
        )
    except KeyError as error:
        raise ValueError("P0 forensic manifest is missing") from error
    if supplied_manifest != manifest:
        raise ValueError("P0 forensic dataclass manifest differs from archived bytes")
    inventory = _inventory(files)
    tree_sha256 = forensic_sha256_bytes(forensic_canonical_json_bytes(inventory))
    if supplied_inventory != inventory:
        raise ValueError("P0 forensic dataclass inventory drifted")
    if supplied_tree_sha256 != tree_sha256:
        raise ValueError("P0 forensic dataclass tree SHA256 drifted")
    if (
        manifest.get("status") != FORENSIC_ARCHIVE_STATUS
        or manifest.get("source_attempt", {}).get("original_attempt_status")
        != PRODUCER_INVALID_STATUS
        or manifest.get("source_attempt", {}).get("root_embedded_snapshot_status")
        != PRODUCER_COMPLETED_STATUS
        or manifest.get("formal_consumption", {}).get("formal_label_loader_eligible")
        is not False
        or manifest.get("negative_declarations", {}).get(
            "producer_invalid_status_reclassified"
        )
        is not False
    ):
        raise ValueError("P0 forensic identity or invalid-status boundary drifted")
    return files, manifest, tree_sha256


def _split_forensic_namespaces(
    files: Mapping[str, bytes],
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    root_prefix = f"{ATTEMPT_ROOT_NAMESPACE}/"
    ledger_prefix = f"{TERMINAL_LEDGER_NAMESPACE}/"
    payload_names = set(files) - {FORENSIC_MANIFEST_PATH}
    if any(
        not (name.startswith(root_prefix) or name.startswith(ledger_prefix))
        for name in payload_names
    ):
        raise ValueError("forensic payload escaped the P0 namespaces")
    root = {
        name[len(root_prefix) :]: payload
        for name, payload in files.items()
        if name.startswith(root_prefix)
    }
    ledgers = {
        name[len(ledger_prefix) :]: payload
        for name, payload in files.items()
        if name.startswith(ledger_prefix)
    }
    if set(root) != set(frozen_v1.expected_file_names()):
        raise ValueError("P0 attempt-root inventory differs from frozen v1 evidence")
    if set(ledgers) != {
        "global_attempt_ledger.json",
        "workers/even.json",
        "workers/odd.json",
    }:
        raise ValueError("P0 terminal-ledger namespace inventory drifted")
    return root, ledgers


def _ledger_chain(
    root: Mapping[str, bytes], terminal_ledgers: Mapping[str, bytes]
) -> Mapping[str, Any]:
    internal_payload = root[frozen_v1.GLOBAL_LEDGER_ARCHIVE_NAME]
    external_payload = terminal_ledgers["global_attempt_ledger.json"]
    internal = strict_pretty_json_object_bytes(
        internal_payload, label="embedded completed global ledger"
    )
    external = strict_pretty_json_object_bytes(
        external_payload, label="terminal invalid global ledger"
    )
    shared_keys = set(internal) - {"status", "ended_at_utc"}
    if (
        internal.get("status") != PRODUCER_COMPLETED_STATUS
        or external.get("status") != PRODUCER_INVALID_STATUS
        or not shared_keys.issubset(external)
        or any(internal[key] != external[key] for key in shared_keys)
        or external.get("failure")
        != {"exception_type": "ValueError", "message": LEGACY_CADENCE_FAILURE}
        or external.get("claimed_ledger_sha256")
        != forensic_sha256_bytes(internal_payload)
        or _timestamp(internal.get("ended_at_utc"), "embedded ledger end")
        > _timestamp(external.get("ended_at_utc"), "terminal ledger end")
    ):
        raise ValueError("P0 completed-to-invalid global ledger chain drifted")
    for worker in ("even", "odd"):
        worker_payloads = (
            root[f"workers/{worker}/{frozen_v1.WORKER_LEDGER_FILENAME}"],
            root[f"{frozen_v1.WORKER_SIBLING_LEDGER_DIRECTORY}/{worker}.json"],
            terminal_ledgers[f"workers/{worker}.json"],
        )
        if len(set(worker_payloads)) != 1:
            raise ValueError(f"P0 {worker} ledger namespaces differ")
    return {
        "embedded_completed_status": internal["status"],
        "terminal_external_status": external["status"],
        "terminal_external_ended_at_utc": external["ended_at_utc"],
        "embedded_completed_sha256": forensic_sha256_bytes(internal_payload),
        "terminal_external_sha256": forensic_sha256_bytes(external_payload),
        "terminal_claims_embedded_completed_sha256": True,
        "producer_ledgers_modified": False,
    }


def _producer_pre_invalidation_view(
    root: Mapping[str, bytes],
    manifest: Mapping[str, Any],
    *,
    terminal_invalid_ended_at_utc: str,
) -> tuple[dict[str, bytes], Mapping[str, Any]]:
    final_log = root[frozen_v1.LOG_PATHS["execution"]]
    if not final_log.endswith(b"\n"):
        raise ValueError("final execution log must be LF-terminated")
    previous_newline = final_log.rfind(b"\n", 0, len(final_log) - 1)
    if previous_newline < 0:
        raise ValueError("final execution log lacks the pre-invalidation prefix")
    prefix = final_log[: previous_newline + 1]
    suffix = final_log[previous_newline + 1 :]
    append = _mapping(
        manifest.get("append_only_execution_log"),
        "P0 append-only execution-log manifest",
    )
    marker = b" attempt invalidated: "
    if suffix.count(b"\n") != 1 or suffix.count(marker) != 1:
        raise ValueError("P0 invalidation suffix must be exactly one line")
    timestamp_bytes, failure_bytes = suffix.split(marker, 1)
    if failure_bytes != LEGACY_CADENCE_FAILURE.encode("utf-8") + b"\n":
        raise ValueError("P0 invalidation suffix failure message drifted")
    try:
        suffix_timestamp = timestamp_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("P0 invalidation suffix timestamp is not UTF-8") from error
    suffix_time = _timestamp(suffix_timestamp, "P0 invalidation suffix timestamp")
    terminal_invalid_end = _timestamp(
        terminal_invalid_ended_at_utc,
        "P0 terminal invalid ledger end",
    )
    if suffix_time < terminal_invalid_end:
        raise ValueError("P0 invalidation suffix predates the terminal invalid ledger")
    suffix_after_ledger_us = _exact_microseconds(
        terminal_invalid_end,
        suffix_time,
    )
    if (
        forensic_sha256_bytes(final_log) != append.get("final_log_sha256")
        or forensic_sha256_bytes(prefix)
        != append.get("pre_failure_prefix_sha256")
        or forensic_sha256_bytes(suffix) != append.get("terminal_suffix_sha256")
        or append.get("terminal_suffix_is_exactly_one_line") is not True
        or append.get("execution_evidence_binds_pre_failure_prefix") is not True
    ):
        raise ValueError("P0 append-only execution-log chain drifted")
    execution = frozen_v1._json(root, frozen_v1.EXECUTION_EVIDENCE_FILENAME)
    if execution.get("execution_log_sha256") != forensic_sha256_bytes(prefix):
        raise ValueError("execution evidence does not bind the pre-invalidation prefix")
    producer = dict(root)
    producer[frozen_v1.LOG_PATHS["execution"]] = prefix
    return producer, {
        "final_log_sha256": forensic_sha256_bytes(final_log),
        "pre_invalidation_prefix_sha256": forensic_sha256_bytes(prefix),
        "invalidation_suffix_sha256": forensic_sha256_bytes(suffix),
        "pre_invalidation_prefix_size_bytes": len(prefix),
        "invalidation_suffix_size_bytes": len(suffix),
        "execution_evidence_binds_pre_invalidation_prefix": True,
        "terminal_invalid_ledger_ended_at_utc": terminal_invalid_ended_at_utc,
        "invalidation_suffix_at_utc": suffix_timestamp,
        "invalidation_suffix_after_terminal_ledger_microseconds": (
            suffix_after_ledger_us
        ),
        "invalidation_suffix_not_before_terminal_invalid_ledger": True,
        "forensic_source_log_bytes_modified": False,
        "in_memory_pre_invalidation_view_reconstructed": True,
    }


def _require_legacy_cadence_failure(files: Mapping[str, bytes]) -> Mapping[str, Any]:
    try:
        frozen_v1.validate_expansion_label_evidence_files(files)
    except ValueError as error:
        if type(error) is not ValueError or str(error) != LEGACY_CADENCE_FAILURE:
            raise ValueError(
                "frozen v1 validator did not reproduce the exact cadence failure"
            ) from error
    else:
        raise ValueError("frozen v1 validator unexpectedly accepted forensic evidence")
    return {
        "exception_type": "ValueError",
        "message": LEGACY_CADENCE_FAILURE,
        "exactly_reproduced": True,
    }


def _validate_execution_hashes(
    files: Mapping[str, bytes], *, run_contract_sha256: str
) -> Mapping[str, Any]:
    execution = frozen_v1._json(files, frozen_v1.EXECUTION_EVIDENCE_FILENAME)
    frozen_v1._exact_keys(
        execution,
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
        "execution evidence",
    )
    expected = {
        "preflight_log_sha256": frozen_v1.sha256_bytes(
            files[frozen_v1.LOG_PATHS["preflight"]]
        ),
        "execution_log_sha256": frozen_v1.sha256_bytes(
            files[frozen_v1.LOG_PATHS["execution"]]
        ),
        "utilization_monitor_log_sha256": frozen_v1.sha256_bytes(
            files[frozen_v1.LOG_PATHS["utilization_monitor"]]
        ),
        "monitor_ready_sha256": frozen_v1.sha256_bytes(
            files[frozen_v1.LOG_PATHS["utilization_monitor_ready"]]
        ),
        "monitor_stop_request_sha256": frozen_v1.sha256_bytes(
            files[frozen_v1.LOG_PATHS["utilization_monitor_stop_request"]]
        ),
        "monitor_summary_sha256": frozen_v1.sha256_bytes(
            files[frozen_v1.MONITOR_SUMMARY_FILENAME]
        ),
    }
    if (
        execution.get("schema_version") != frozen_v1.SCHEMA_VERSION
        or execution.get("protocol_id") != frozen_v1.PROTOCOL_ID
        or execution.get("status")
        != "VALIDATED_EXPANSION_EXACT_LABEL_EXECUTION_EVIDENCE"
        or execution.get("run_contract_sha256") != run_contract_sha256
        or any(execution.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("execution evidence hash binding drifted")
    return expected


def _structural_monitor_validation(
    files: Mapping[str, bytes],
    *,
    run_contract_sha256: str,
    runtime_visible_uuids: tuple[str, str],
    runtime_physical_inventory: tuple[tuple[int, str], tuple[int, str]],
    first_state_claim_time: datetime,
    last_worker_terminal_time: datetime,
) -> Mapping[str, Any]:
    ready = frozen_v1._json(
        files, frozen_v1.LOG_PATHS["utilization_monitor_ready"]
    )
    frozen_v1._exact_keys(
        ready,
        {
            "schema_version",
            "status",
            "run_contract_sha256",
            "monitor_pid",
            "visible_gpu_count",
            "visible_gpu_uuids",
            "started_at_utc",
        },
        "monitor ready record",
    )
    ready_uuids = tuple(_sequence(ready.get("visible_gpu_uuids"), "ready UUIDs"))
    if (
        ready.get("schema_version") != frozen_v1.SCHEMA_VERSION
        or ready.get("status") != "EXPANSION_EXACT_LABEL_MONITOR_READY"
        or ready.get("run_contract_sha256") != run_contract_sha256
        or type(ready.get("monitor_pid")) is not int
        or ready["monitor_pid"] <= 0
        or ready.get("visible_gpu_count") != 2
        or ready_uuids != runtime_visible_uuids
    ):
        raise ValueError("monitor ready identity drifted")
    ready_time = _timestamp(ready.get("started_at_utc"), "monitor ready time")

    stop = frozen_v1._json(
        files, frozen_v1.LOG_PATHS["utilization_monitor_stop_request"]
    )
    frozen_v1._exact_keys(
        stop,
        {"schema_version", "status", "run_contract_sha256", "requested_at_utc"},
        "monitor stop request",
    )
    if (
        stop.get("schema_version") != frozen_v1.SCHEMA_VERSION
        or stop.get("status") != "EXPANSION_EXACT_LABEL_MONITOR_STOP_REQUEST"
        or stop.get("run_contract_sha256") != run_contract_sha256
    ):
        raise ValueError("monitor stop request drifted")
    stop_time = _timestamp(stop.get("requested_at_utc"), "monitor stop time")

    summary = frozen_v1._json(files, frozen_v1.MONITOR_SUMMARY_FILENAME)
    frozen_v1._exact_keys(
        summary,
        {
            "schema_version",
            "status",
            "run_contract_sha256",
            "container_name_prefix",
            "minimum_gpu_utilization_percent",
            "started_before_first_teacher_forward",
            "started_at_utc",
            "stopped_at_utc",
            "sample_count",
            "low_utilization_incident_count",
            "visible_gpu_count",
            "visible_gpu_uuids",
            "stop_request_sha256",
        },
        "monitor summary",
    )
    summary_uuids = tuple(
        _sequence(summary.get("visible_gpu_uuids"), "monitor summary UUIDs")
    )
    if (
        summary.get("schema_version") != frozen_v1.SCHEMA_VERSION
        or summary.get("status") != "GPU_UTILIZATION_MONITOR_SUMMARY"
        or summary.get("run_contract_sha256") != run_contract_sha256
        or summary.get("container_name_prefix") != "sglang-omni-jaxan"
        or summary.get("minimum_gpu_utilization_percent") != 90
        or summary.get("started_before_first_teacher_forward") is not True
        or type(summary.get("sample_count")) is not int
        or summary["sample_count"] <= 0
        or type(summary.get("low_utilization_incident_count")) is not int
        or summary["low_utilization_incident_count"] < 0
        or summary.get("visible_gpu_count") != 2
        or summary_uuids != runtime_visible_uuids
        or summary.get("stop_request_sha256")
        != frozen_v1.sha256_bytes(
            files[frozen_v1.LOG_PATHS["utilization_monitor_stop_request"]]
        )
    ):
        raise ValueError("monitor summary identity drifted")
    summary_start = _timestamp(summary.get("started_at_utc"), "monitor summary start")
    summary_stop = _timestamp(summary.get("stopped_at_utc"), "monitor summary stop")

    sample_times, reconstructed_low_count = frozen_v1._validate_monitor_jsonl(
        files[frozen_v1.LOG_PATHS["utilization_monitor"]],
        runtime_visible_uuids=runtime_visible_uuids,
        runtime_physical_inventory=runtime_physical_inventory,
    )
    if (
        summary.get("sample_count") != len(sample_times)
        or summary.get("low_utilization_incident_count")
        != reconstructed_low_count
    ):
        raise ValueError("monitor summary counts differ from JSONL")
    if any(right <= left for left, right in zip(sample_times, sample_times[1:])):
        raise ValueError("monitor sample timestamps must be strictly increasing")
    if not (
        summary_start == ready_time
        and ready_time <= sample_times[0] <= first_state_claim_time
        and last_worker_terminal_time <= sample_times[-1] <= summary_stop
        and last_worker_terminal_time <= stop_time <= summary_stop
    ):
        raise ValueError("monitor structural lifecycle does not bracket worker execution")

    sample_gap_microseconds = [
        _exact_microseconds(left, right)
        for left, right in zip(sample_times, sample_times[1:])
    ]
    if not sample_gap_microseconds or any(
        value <= 0 for value in sample_gap_microseconds
    ):
        raise ValueError("monitor sample gaps must be positive")
    legacy_limit_us = int(frozen_v1.MONITOR_MAX_SAMPLE_GAP_SECONDS * 1_000_000)
    ready_to_first_sample_us = _exact_microseconds(ready_time, sample_times[0])
    last_sample_to_summary_us = _exact_microseconds(
        sample_times[-1],
        summary_stop,
    )
    coverage_gap_microseconds = [
        ready_to_first_sample_us,
        *sample_gap_microseconds,
        last_sample_to_summary_us,
    ]
    terminal_tail_signed_us = _exact_microseconds(
        sample_times[-1],
        last_worker_terminal_time,
    )
    legacy_components = {
        "first_sample_not_after_first_state_claim": (
            sample_times[0] <= first_state_claim_time
        ),
        "summary_not_before_last_worker_terminal": (
            summary_stop >= last_worker_terminal_time
        ),
        "stop_not_before_last_worker_terminal": (
            stop_time >= last_worker_terminal_time
        ),
        "last_sample_to_terminal_tail_within_limit": (
            terminal_tail_signed_us <= legacy_limit_us
        ),
        "maximum_ready_sample_summary_gap_within_limit": (
            max(coverage_gap_microseconds) <= legacy_limit_us
        ),
    }
    boundary_gap_microseconds = {
        "ready_to_first_sample": ready_to_first_sample_us,
        "first_sample_to_first_state_claim": _exact_microseconds(
            sample_times[0],
            first_state_claim_time,
        ),
        "last_sample_to_last_worker_terminal_signed": terminal_tail_signed_us,
        "last_worker_terminal_to_stop": _exact_microseconds(
            last_worker_terminal_time,
            stop_time,
        ),
        "last_worker_terminal_to_summary": _exact_microseconds(
            last_worker_terminal_time,
            summary_stop,
        ),
        "last_sample_to_summary": last_sample_to_summary_us,
    }
    _validate_execution_hashes(files, run_contract_sha256=run_contract_sha256)
    return {
        "status": "PASS_STRUCTURAL_MONITOR_LIFECYCLE_VALIDATION",
        "acceptance_uses_numeric_gap_threshold": False,
        "legacy_maximum_gap_seconds": frozen_v1.MONITOR_MAX_SAMPLE_GAP_SECONDS,
        "legacy_cadence_formula_exactly_reproduced": True,
        "legacy_cadence_components": legacy_components,
        "legacy_cadence_gate_passed": all(legacy_components.values()),
        "sample_count": len(sample_times),
        "sample_index_range": [0, len(sample_times) - 1],
        "first_sample_at_utc": sample_times[0].isoformat(),
        "last_sample_at_utc": sample_times[-1].isoformat(),
        "low_utilization_incident_count": reconstructed_low_count,
        "boundary_gap_microseconds": boundary_gap_microseconds,
        "coverage_gap_microseconds": coverage_gap_microseconds,
        "coverage_gap_microseconds_sha256": frozen_v1.sha256_bytes(
            frozen_v1.canonical_json_bytes(coverage_gap_microseconds)
        ),
        "sample_gap_microseconds": sample_gap_microseconds,
        "sample_gap_microseconds_sha256": frozen_v1.sha256_bytes(
            frozen_v1.canonical_json_bytes(sample_gap_microseconds)
        ),
        "sample_gap_statistics_seconds": {
            "count": len(sample_gap_microseconds),
            "minimum": min(sample_gap_microseconds) / 1_000_000,
            "maximum": max(sample_gap_microseconds) / 1_000_000,
            "mean": (
                math.fsum(sample_gap_microseconds)
                / len(sample_gap_microseconds)
                / 1_000_000
            ),
            "nearest_rank_p50": (
                _nearest_rank(sample_gap_microseconds, 0.50) / 1_000_000
            ),
            "nearest_rank_p95": (
                _nearest_rank(sample_gap_microseconds, 0.95) / 1_000_000
            ),
            "nearest_rank_p99": (
                _nearest_rank(sample_gap_microseconds, 0.99) / 1_000_000
            ),
            "count_above_1_second": sum(
                value > 1_000_000 for value in sample_gap_microseconds
            ),
            "count_above_2_seconds": sum(
                value > 2_000_000 for value in sample_gap_microseconds
            ),
            "count_above_3_seconds": sum(
                value > 3_000_000 for value in sample_gap_microseconds
            ),
        },
        "ready_before_first_state_claim": True,
        "last_sample_after_last_worker_terminal": True,
        "stop_after_last_worker_terminal": True,
        "summary_after_last_worker_terminal": True,
        "execution_evidence_hashes_verified": True,
    }


def _validate_frozen_scientific_payload(
    files: Mapping[str, bytes],
) -> tuple[
    Mapping[str, Any],
    tuple[Mapping[str, Any], ...],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    manifest = frozen_v1._json(files, frozen_v1.RUN_MANIFEST_FILENAME)
    frozen_v1._exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract",
            "run_contract_sha256",
        },
        "run manifest",
    )
    run_contract = _mapping(manifest.get("run_contract"), "execution run contract")
    run_contract_sha256 = frozen_v1.sha256_bytes(
        frozen_v1.canonical_json_bytes(run_contract)
    )
    if manifest.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError("run manifest SHA256 drifted")
    validated_contract = frozen_v1.validate_execution_run_contract(run_contract)
    frozen_v1._validate_global_ledger(
        frozen_v1._json(files, frozen_v1.GLOBAL_LEDGER_ARCHIVE_NAME),
        run_contract_sha256=run_contract_sha256,
    )

    states = validated_contract["states"]
    by_index: dict[int, Mapping[str, Any]] = {}
    worker_gpu_uuids: list[str] = []
    bindings_by_worker: dict[str, Mapping[str, Any]] = {}
    first_claim_times: list[datetime] = []
    terminal_times: list[datetime] = []
    runtime_visible: tuple[str, str] | None = None
    for spec in frozen_v1.expected_worker_specs():
        (
            records,
            worker_uuid,
            visible,
            bindings,
            first_claim,
            terminal_time,
        ) = frozen_v1._validate_worker_evidence(
            files,
            spec=spec,
            run_contract_sha256=run_contract_sha256,
            states=states,
        )
        worker_gpu_uuids.append(worker_uuid)
        bindings_by_worker[spec.worker_id] = bindings
        first_claim_times.append(first_claim)
        terminal_times.append(terminal_time)
        if runtime_visible is None:
            runtime_visible = visible
        elif visible != runtime_visible:
            raise ValueError("worker visible GPU inventories differ")
        for index, record in zip(spec.state_indices, records, strict=True):
            if index in by_index:
                raise ValueError("state was emitted by multiple workers")
            by_index[index] = record
    if (
        set(by_index) != set(range(frozen_v1.EXPECTED_STATE_COUNT))
        or runtime_visible is None
        or tuple(worker_gpu_uuids) != runtime_visible
    ):
        raise ValueError("worker state schedule or GPU assignment drifted")
    frozen_v1.validate_eager_worker_runtime_pair(bindings_by_worker)

    structural_monitor = _structural_monitor_validation(
        files,
        run_contract_sha256=run_contract_sha256,
        runtime_visible_uuids=runtime_visible,
        runtime_physical_inventory=tuple(
            (
                int(bindings_by_worker[spec.worker_id]["nvidia_smi_index"]),
                str(bindings_by_worker[spec.worker_id]["gpu_uuid"]),
            )
            for spec in frozen_v1.expected_worker_specs()
        ),
        first_state_claim_time=min(first_claim_times),
        last_worker_terminal_time=max(terminal_times),
    )
    ordered = tuple(by_index[index] for index in range(frozen_v1.EXPECTED_STATE_COUNT))
    reduction = frozen_v1.reduce_raw_distance_states(
        ordered,
        expected_states=states,
        run_contract_sha256=run_contract_sha256,
    )
    aggregate = frozen_v1._json(files, frozen_v1.AGGREGATE_FILENAME)
    rebuilt_aggregate = frozen_v1.aggregate_from_reduction(
        reduction,
        run_contract_sha256=run_contract_sha256,
        execution_evidence_sha256=frozen_v1.sha256_bytes(
            files[frozen_v1.EXECUTION_EVIDENCE_FILENAME]
        ),
        started_at_utc=str(aggregate.get("started_at_utc")),
        ended_at_utc=str(aggregate.get("ended_at_utc")),
    )
    if aggregate != rebuilt_aggregate:
        raise ValueError("stored aggregate differs from independent raw reduction")
    math_audit = audit_normalized_raw_states(ordered)
    math_comparison = compare_with_reducer_output(math_audit, reduction)
    return run_contract, ordered, reduction, {
        "structural_monitor": structural_monitor,
        "math_audit": math_comparison,
        "stored_aggregate_exactly_rebuilt": True,
        "stored_pre_monitor_aggregate_outcome": aggregate.get("outcome"),
    }


def _validate_frozen_replay_attestation(
    attestation: FrozenExternalReplayAttestation,
) -> None:
    if type(attestation) is not FrozenExternalReplayAttestation:
        raise TypeError("formal replay requires FrozenExternalReplayAttestation")
    _validate_frozen_external_provenance_globals()
    if (
        dict(attestation.implementation) != FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION
        or dict(attestation.parent_substrate)
        != dict(FROZEN_PARENT_SUBSTRATE)
        or dict(attestation.derived_artifact)
        != dict(FROZEN_DERIVED_ARTIFACT)
        or dict(attestation.ocr_backend_config) != FROZEN_OCR_BACKEND_CONFIG
    ):
        raise ValueError("formal external replay identity or provenance drifted")
    parent_path = Path(attestation.parent_substrate_archive_path)
    derived_root = Path(attestation.derived_artifact_root)
    ocr_path = Path(attestation.ocr_backend_config_path)
    ocr_suffix = Path(FROZEN_OCR_BACKEND_CONFIG["path"]).parts
    if (
        not parent_path.is_absolute()
        or not derived_root.is_absolute()
        or not ocr_path.is_absolute()
        or tuple(ocr_path.parts[-len(ocr_suffix) :]) != ocr_suffix
    ):
        raise ValueError("formal external replay paths are not canonical absolute paths")
    implementation_path = Path(frozen_v1.__file__).resolve()
    implementation_bytes = implementation_path.read_bytes()
    try:
        callable_source = inspect.getsource(
            _CAPTURED_FROZEN_EXTERNAL_REPLAY
        ).encode("utf-8")
    except (OSError, TypeError) as error:
        raise ValueError("frozen external replay callable source is unavailable") from error
    if (
        len(implementation_bytes)
        != FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION["source_size_bytes"]
        or forensic_sha256_bytes(implementation_bytes)
        != FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION["source_sha256"]
        or len(callable_source)
        != FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION["callable_source_size_bytes"]
        or forensic_sha256_bytes(callable_source)
        != FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION["callable_source_sha256"]
        or frozen_v1.replay_external_input_projections
        is not _CAPTURED_FROZEN_EXTERNAL_REPLAY
        or _CAPTURED_FROZEN_EXTERNAL_REPLAY.__code__
        is not _CAPTURED_FROZEN_EXTERNAL_REPLAY_CODE
        or _CAPTURED_FROZEN_EXTERNAL_REPLAY.__module__
        != FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION["module"]
        or _CAPTURED_FROZEN_EXTERNAL_REPLAY.__qualname__
        != FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION["qualname"]
    ):
        raise ValueError("frozen external replay implementation bytes drifted")


def _validate_frozen_external_provenance_globals() -> None:
    if (
        dict(frozen_v1.CANONICAL_PARENT_SUBSTRATE)
        != dict(FROZEN_PARENT_SUBSTRATE)
        or dict(frozen_v1.CANONICAL_DERIVED_ARTIFACT)
        != dict(FROZEN_DERIVED_ARTIFACT)
    ):
        raise ValueError("frozen external replay provenance globals drifted")


def _validate_external_replay(
    run_contract: Mapping[str, Any],
    *,
    callback: ExternalReplayCallback,
    attestation: (
        FrozenExternalReplayAttestation | TestOnlyExternalReplayAttestation
    ),
) -> tuple[Mapping[str, Any], bool]:
    original_contract_bytes = frozen_v1.canonical_json_bytes(run_contract)
    callback_contract = copy.deepcopy(dict(run_contract))
    callback_contract_before = frozen_v1.canonical_json_bytes(callback_contract)

    if type(attestation) is FrozenExternalReplayAttestation:
        _validate_frozen_replay_attestation(attestation)
        if (
            type(callback) is not FrozenExternalReplayExecutor
            or callback.attestation != attestation
            or callback._replay_callable is not _CAPTURED_FROZEN_EXTERNAL_REPLAY
        ):
            raise TypeError(
                "formal replay callback must be the bound frozen replay executor"
            )
        formal_replay = True
    elif type(attestation) is TestOnlyExternalReplayAttestation:
        if (
            not attestation.fixture_id
            or len(attestation.fixture_state_projections_sha256) != 64
        ):
            raise ValueError("test-only replay attestation drifted")
        formal_replay = False
    else:
        raise TypeError("external replay attestation type is not recognized")

    if formal_replay:
        replayed_states, replay_metadata = _CAPTURED_FROZEN_EXTERNAL_REPLAY(
            parent_substrate_archive=(
                attestation.parent_substrate_archive_path
            ),
            derived_artifact_root=attestation.derived_artifact_root,
            ocr_backend_config=attestation.ocr_backend_config_path,
        )
    else:
        replayed_states, replay_metadata = callback(callback_contract)
    if formal_replay:
        _validate_frozen_external_provenance_globals()
    if (
        frozen_v1.canonical_json_bytes(callback_contract)
        != callback_contract_before
    ):
        raise ValueError("external replay callback mutated its run-contract copy")
    if frozen_v1.canonical_json_bytes(run_contract) != original_contract_bytes:
        raise ValueError("external replay callback mutated the original run contract")

    external_replay = frozen_v1.validate_external_input_replay(
        run_contract,
        replayed_states,
        replay_metadata,
    )
    metadata = _mapping(replay_metadata, "external input replay metadata")
    if formal_replay:
        expected_formal_metadata = {
            "parent_substrate_archive_sha256": FROZEN_PARENT_SUBSTRATE[
                "archive_sha256"
            ],
            "parent_substrate_archive_size_bytes": (
                FROZEN_PARENT_SUBSTRATE["archive_size_bytes"]
            ),
            "parent_substrate_tree_inventory_sha256": (
                FROZEN_PARENT_SUBSTRATE["tree_inventory_sha256"]
            ),
            "derived_artifact_repo": FROZEN_DERIVED_ARTIFACT["repo"],
            "derived_artifact_revision": FROZEN_DERIVED_ARTIFACT[
                "immutable_revision"
            ],
            "derived_artifact_tree_sha256": FROZEN_DERIVED_ARTIFACT[
                "artifact_tree_sha256"
            ],
            "ocr_backend_config_sha256": FROZEN_OCR_BACKEND_CONFIG["sha256"],
            "ocr_backend_config_size_bytes": FROZEN_OCR_BACKEND_CONFIG[
                "size_bytes"
            ],
        }
        if any(
            metadata.get(key) != value
            for key, value in expected_formal_metadata.items()
        ):
            raise ValueError("formal external replay provenance metadata drifted")
        return {
            **dict(external_replay),
            "status": EXTERNAL_REPLAY_STATUS,
            "attestation_tier": FORMAL_EXTERNAL_REPLAY_TIER,
            "formal_external_input_replay_verified": True,
            "run_contract_copy_immutable_across_callback": True,
            "implementation": dict(attestation.implementation),
            "parent_substrate": dict(attestation.parent_substrate),
            "derived_artifact": dict(attestation.derived_artifact),
            "ocr_backend_config": dict(attestation.ocr_backend_config),
        }, True

    replay_sha = external_replay["external_state_projections_sha256"]
    if (
        metadata.get("test_only_fixture_replay") is not True
        or metadata.get("fixture_id") != attestation.fixture_id
        or metadata.get("fixture_state_projections_sha256")
        != attestation.fixture_state_projections_sha256
        or replay_sha != attestation.fixture_state_projections_sha256
    ):
        raise ValueError("test-only replay fixture attestation drifted")
    return {
        **dict(external_replay),
        "status": TEST_EXTERNAL_REPLAY_STATUS,
        "attestation_tier": TEST_EXTERNAL_REPLAY_TIER,
        "formal_external_input_replay_verified": False,
        "embedded_echo_is_not_external_replay_evidence": True,
        "run_contract_copy_immutable_across_callback": True,
        "fixture_id": attestation.fixture_id,
    }, False


def validate_ledger_neutral_scientific_payload(
    evidence: InvalidForensicEvidence,
    *,
    forensic_contract: InvalidForensicContract,
    external_input_replay_callback: ExternalReplayCallback,
    external_replay_attestation: (
        FrozenExternalReplayAttestation | TestOnlyExternalReplayAttestation
    ),
) -> LedgerNeutralScientificValidation:
    """Validate P0 forensic bytes without mutating or reclassifying v1."""
    files, manifest, forensic_tree_sha256 = _normalize_forensic_files(
        evidence,
        forensic_contract=forensic_contract,
    )
    if not callable(external_input_replay_callback):
        raise TypeError("external input replay callback must be callable")
    before_inventory = _inventory(files)
    root, terminal_ledgers = _split_forensic_namespaces(files)
    ledger_chain = _ledger_chain(root, terminal_ledgers)
    producer_view, append_only_log = _producer_pre_invalidation_view(
        root,
        manifest,
        terminal_invalid_ended_at_utc=str(
            ledger_chain["terminal_external_ended_at_utc"]
        ),
    )
    legacy_failure = _require_legacy_cadence_failure(producer_view)
    run_contract, records, reduction, scientific = _validate_frozen_scientific_payload(
        producer_view
    )
    formal_replay_requested = (
        type(external_replay_attestation) is FrozenExternalReplayAttestation
    )
    if formal_replay_requested and (
        forensic_contract.sha256 != FROZEN_P0_CONFIG_SHA256
        or forensic_tree_sha256 != FROZEN_P0_FORENSIC_TREE_SHA256
    ):
        raise ValueError(
            "formal scientific repair requires the exact frozen P0 config and tree"
        )
    external_replay, formal_replay = _validate_external_replay(
        run_contract,
        callback=external_input_replay_callback,
        attestation=external_replay_attestation,
    )
    if formal_replay is not formal_replay_requested:
        raise ValueError("external replay tier changed across formal validation")
    if _inventory(files) != before_inventory:
        raise ValueError("P0 forensic evidence changed during scientific validation")
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": PASS_STATUS if formal_replay else TEST_ONLY_STATUS,
        "formal_scientific_repair_pass": formal_replay,
        "original_attempt": {
            **dict(ledger_chain),
            "formal_v1_outcome_remains_invalid": True,
            "formal_v1_pass_claimed": False,
        },
        "append_only_execution_log": dict(append_only_log),
        "legacy_validator_negative_control": dict(legacy_failure),
        "monitor_validation": dict(scientific["structural_monitor"]),
        "raw_distance_reduction": {
            "counts": dict(reduction["counts"]),
            "summary": dict(reduction["summary"]),
            "derived_payload_sha256": reduction["derived_payload_sha256"],
            "stored_aggregate_exactly_rebuilt": scientific[
                "stored_aggregate_exactly_rebuilt"
            ],
            "stored_pre_monitor_aggregate_outcome": scientific[
                "stored_pre_monitor_aggregate_outcome"
            ],
        },
        "independent_stdlib_math_audit": dict(scientific["math_audit"]),
        "external_input_replay": dict(external_replay),
        "forensic_evidence": {
            "strict_p0_revalidation_performed": True,
            "forensic_contract_sha256": forensic_contract.sha256,
            "tree_inventory_sha256": forensic_tree_sha256,
            "formal_required_forensic_contract_sha256": FROZEN_P0_CONFIG_SHA256,
            "formal_required_tree_inventory_sha256": (
                FROZEN_P0_FORENSIC_TREE_SHA256
            ),
            "exact_frozen_p0_source_verified_for_formal_pass": formal_replay,
            "synthetic_p0_source_allowed_only_for_test_status": not formal_replay,
            "bytes_modified": False,
            "ledger_mutation_count": 0,
        },
        "execution_scope": {
            "core_scope_excludes_external_replay_callback": True,
            "core_does_not_claim_global_gpu_or_model_operation_counts": True,
            "formal_zero_operation_attestation_deferred_to_runner": True,
        },
    }
    return LedgerNeutralScientificValidation(
        report=report,
        run_contract=dict(run_contract),
        normalized_state_records=records,
        reduction=reduction,
    )


__all__ = [
    "EXTERNAL_REPLAY_STATUS",
    "FORMAL_EXTERNAL_REPLAY_TIER",
    "FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION",
    "FROZEN_OCR_BACKEND_CONFIG",
    "FROZEN_P0_CONFIG_SHA256",
    "FROZEN_P0_FORENSIC_TREE_SHA256",
    "LEGACY_CADENCE_FAILURE",
    "PASS_STATUS",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "TEST_EXTERNAL_REPLAY_STATUS",
    "TEST_EXTERNAL_REPLAY_TIER",
    "TEST_ONLY_STATUS",
    "ExternalReplayCallback",
    "FrozenExternalReplayAttestation",
    "FrozenExternalReplayExecutor",
    "FROZEN_DERIVED_ARTIFACT",
    "LedgerNeutralScientificValidation",
    "FROZEN_PARENT_SUBSTRATE",
    "TestOnlyExternalReplayAttestation",
    "validate_ledger_neutral_scientific_payload",
]
