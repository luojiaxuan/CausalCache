"""Fail-closed Execution-B orchestration for long-horizon development.

The module owns lifecycle and denominator accounting, while the expensive GUI-Owl
forward pass is injected through :class:`LongHorizonWorkerAdapter`.  This keeps the
formal sharding code testable without importing CUDA in the parent process.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from causalcache.long_horizon_contract import (
    SELECTION_MANIFEST_FREEZE_PATH,
    LongHorizonContract,
)
from causalcache.long_horizon_data import (
    MANIFEST_RELATIVE_PATH as SUBSTRATE_MANIFEST_RELATIVE_PATH,
    TRAJECTORY_JSONL_RELATIVE_PATH,
    canonical_json_bytes,
)
from causalcache.long_horizon_selection import (
    validate_long_horizon_selection_manifest,
)
from causalcache.long_horizon_evaluation import (
    NORMALIZATION_EPSILON,
    StateSelectionMetric,
    build_comparator_pair_union_distance_table,
    build_sparse_reference_distance_table,
    evaluate_pair_union_table,
    exact_at_most_b_oracle,
    paired_trajectory_bootstrap,
    paired_trajectory_deltas,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_long_horizon_execution_b_v1"
RUNNER_STATUS = "FROZEN_CAUSALCACHE_LONG_HORIZON_EXECUTION_B_V1"
WORKER_RESULT_PROTOCOL_ID = "causalcache_long_horizon_worker_result_v1"
WORKER_RECEIPT_PROTOCOL_ID = "causalcache_long_horizon_worker_receipt_v1"
SUMMARY_PROTOCOL_ID = "causalcache_long_horizon_development_summary_v1"
SELECTOR_PREPARATION_MANIFEST_PATH = (
    "derived/long-horizon-development-v1/selector-preparation/manifest.json"
)
SELECTOR_SEAL_PATH = (
    "derived/long-horizon-development-v1/selector-preparation/selector-seal.json"
)
SELECTION_MANIFEST_PATH = SELECTION_MANIFEST_FREEZE_PATH
WORKER_COUNT = 4
TRAJECTORY_COUNT = 24
TRAJECTORIES_PER_WORKER = 6
EXPECTED_RANK_RANGES = ((0, 5), (6, 11), (12, 17), (18, 23))

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40,64}")
_GPU_ID = re.compile(r"(?:0|[1-9][0-9]*)")


class LongHorizonWorkerAdapter(Protocol):
    """One-GPU adapter that evaluates one already-selected development trajectory."""

    def run_trajectory(
        self,
        *,
        trajectory: Mapping[str, Any],
        selector_records: Sequence[Mapping[str, Any]],
        host_gpu_id: str,
        container_cuda_ordinal: int,
        runner_config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class CanonicalLongHorizonRuntimeAdapter:
    """Bridge sealed selector rows to the canonical per-state runtime API."""

    def __init__(
        self,
        *,
        contract: LongHorizonContract,
        runtime: Any,
        distance_backend: Any,
        image_bytes_loader: Any,
        image_decoder: Any,
        selection_seal_sha256: str,
        receipt_store: Any,
    ) -> None:
        self.contract = contract
        self.runtime = runtime
        self.distance_backend = distance_backend
        self.image_bytes_loader = image_bytes_loader
        self.image_decoder = image_decoder
        self.selection_seal_sha256 = selection_seal_sha256
        self.receipt_store = receipt_store

    def run_trajectory(
        self,
        *,
        trajectory: Mapping[str, Any],
        selector_records: Sequence[Mapping[str, Any]],
        host_gpu_id: str,
        container_cuda_ordinal: int,
        runner_config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        # note (luojiaxuan): Importing the CUDA-facing runtime here, after worker
        # assignment, keeps the orchestration parent process CUDA-clean.
        from causalcache.long_horizon_runtime import (
            SealedComparatorPair,
            run_n8_state_with_resume,
            run_n16_pair_with_resume,
        )

        del host_gpu_id, container_cuda_ordinal, runner_config
        maximum_context_tokens = int(
            self.contract.data["policy_context_profile"]["maximum_context_tokens"]
        )
        completion_token_reserve = int(
            self.contract.data["policy_context_profile"][
                "processor_preflight_reserved_completion_tokens"
            ]
        )
        common = {
            "runtime": self.runtime,
            "distance_backend": self.distance_backend,
            "image_bytes_loader": self.image_bytes_loader,
            "image_decoder": self.image_decoder,
            "maximum_context_tokens": maximum_context_tokens,
            "completion_token_reserve": completion_token_reserve,
        }
        n8, _ = run_n8_state_with_resume(
            receipt_store=self.receipt_store,
            trajectory=trajectory,
            **common,
        )
        n16_rows = [
            record
            for record in selector_records
            if record["decision_step_id"] == 18
        ]
        by_key = {
            (int(record["budget_event_capacity"]), str(record["selector_name"])): record
            for record in n16_rows
        }
        if len(by_key) != len(n16_rows):
            raise ValueError("n16 selector seal contains duplicate budget/name records")
        pair_specs = tuple(
            (2, tuple(pair))
            for pair in self.contract.data["reference_contract"]["n16"][
                "budget_2_comparator_pairs"
            ]
        ) + tuple(
            (4, tuple(pair))
            for pair in self.contract.data["reference_contract"]["n16"][
                "budget_4_comparator_pairs"
            ]
        )
        pair_records = []
        state_id = f"{trajectory['source_id']}:decision_step:018"
        for budget, (left_name, right_name) in pair_specs:
            try:
                left = by_key[(budget, left_name)]
                right = by_key[(budget, right_name)]
            except KeyError as error:
                raise ValueError("n16 sealed comparator pair is incomplete") from error
            pair = SealedComparatorPair(
                source_id=str(trajectory["source_id"]),
                state_id=state_id,
                budget_event_capacity=budget,
                left_selector_name=left_name,
                right_selector_name=right_name,
                left_selected_event_step_ids=tuple(left["selected_event_step_ids"]),
                right_selected_event_step_ids=tuple(right["selected_event_step_ids"]),
                selection_seal_sha256=self.selection_seal_sha256,
            )
            record, _ = run_n16_pair_with_resume(
                receipt_store=self.receipt_store,
                trajectory=trajectory,
                pair=pair,
                **common,
            )
            pair_records.append(record)
        return {
            "source_id": trajectory["source_id"],
            "n8": n8,
            "n16_pairs": pair_records,
        }


@dataclass(frozen=True)
class WorkerAssignment:
    worker_index: int
    worker_id: str
    host_gpu_id: str
    container_cuda_ordinal: int
    selection_ranks: tuple[int, ...]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_object_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {label}: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def load_json_object(path: str | Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    payload = Path(path).read_bytes()
    return payload, load_json_object_bytes(payload, label=label)


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


def _relative_path(value: Any, label: str) -> str:
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


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys drifted: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _canonical_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return load_json_object_bytes(canonical_json_bytes(dict(value)), label="mapping")


def inventory_sha256(inventory: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes([dict(record) for record in inventory]))


def _validate_inventory(
    inventory: Any,
    *,
    expected_paths: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    rows = _sequence(inventory, "Source-A inventory")
    records: list[dict[str, Any]] = []
    for raw in rows:
        record = _mapping(raw, "Source-A inventory record")
        _exact_keys(record, {"path", "sha256", "size_bytes"}, "inventory record")
        path = _relative_path(record["path"], "inventory path")
        if (
            _SHA256.fullmatch(str(record["sha256"])) is None
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
        ):
            raise ValueError("Source-A inventory digest or size is invalid")
        records.append(dict(record))
    if [record["path"] for record in records] != sorted(expected_paths):
        raise ValueError("Source-A inventory path roster drifted")
    return tuple(records)


def validate_selection_manifest_payload(
    payload: bytes,
    *,
    contract: LongHorizonContract,
    expected_sha256: str,
) -> dict[str, Any]:
    if sha256_bytes(payload) != expected_sha256:
        raise ValueError("selection manifest SHA256 drifted")
    manifest = load_json_object_bytes(payload, label="long-horizon selection manifest")
    validate_long_horizon_selection_manifest(manifest, contract=contract)
    return manifest


def _expected_state_roster(
    selection_manifest: Mapping[str, Any],
) -> dict[str, tuple[str, int, tuple[int, ...]]]:
    development = _mapping(
        _mapping(selection_manifest["splits"], "selection splits")["development"],
        "development split",
    )
    records = _sequence(development["trajectories"], "development trajectories")
    result: dict[str, tuple[str, int, tuple[int, ...]]] = {}
    for raw in records:
        record = _mapping(raw, "development trajectory")
        source_id = str(record["trajectory_id"])
        for step, count in ((10, 8), (18, 16)):
            state_id = f"{source_id}:decision_step:{step:03d}"
            result[state_id] = (source_id, step, tuple(range(1, count + 1)))
    if len(result) != 48:
        raise ValueError("selection manifest must define exactly 48 development states")
    return result


def validate_selector_seal_payload(
    payload: bytes,
    *,
    expected_sha256: str,
    selection_manifest: Mapping[str, Any],
    contract: LongHorizonContract,
) -> dict[str, Any]:
    """Validate the full B2/B4 selector matrix without reading restoration labels."""
    if sha256_bytes(payload) != expected_sha256:
        raise ValueError("selector seal SHA256 drifted")
    seal = load_json_object_bytes(payload, label="label-blind selector seal")
    required_top = {
        "schema_version",
        "protocol_id",
        "status",
        "selector_names",
        "budgets",
        "selector_names_by_budget",
        "state_count",
        "record_count",
        "records",
    }
    _exact_keys(seal, required_top, "selector seal")
    if (
        seal["schema_version"] != "1.0.0"
        or payload != canonical_json_bytes(seal) + b"\n"
        or seal["protocol_id"] != "causalcache_long_horizon_label_blind_selection"
        or seal["status"] != "SEALED_LONG_HORIZON_LABEL_BLIND_SELECTIONS_V1"
        or seal["state_count"] != 48
    ):
        raise ValueError("selector seal identity or state denominator drifted")
    states = _expected_state_roster(selection_manifest)
    matrix = {
        2: tuple(contract.data["selector_matrix"]["budget_2"]),
        4: tuple(contract.data["selector_matrix"]["budget_4"]),
    }
    expected_keys = {
        (budget, selector, state_id)
        for budget, selectors in matrix.items()
        for selector in selectors
        for state_id in states
    }
    records = _sequence(seal["records"], "selector seal records")
    observed: set[tuple[int, str, str]] = set()
    observed_order: list[tuple[int, str, str]] = []
    record_keys = {
        "selector_name",
        "source_id",
        "state_id",
        "decision_step_id",
        "candidate_event_step_ids",
        "budget_event_capacity",
        "selected_event_step_ids",
    }
    for raw in records:
        record = _mapping(raw, "selector seal record")
        _exact_keys(record, record_keys, "selector seal record")
        budget = record["budget_event_capacity"]
        selector = record["selector_name"]
        state_id = record["state_id"]
        if type(budget) is not int or not isinstance(selector, str) or not isinstance(
            state_id, str
        ):
            raise ValueError("selector seal key types drifted")
        key = (budget, selector, state_id)
        if key in observed:
            raise ValueError("selector seal contains a duplicate budget/selector/state")
        observed.add(key)
        observed_order.append(key)
        try:
            source_id, step, event_ids = states[state_id]
        except KeyError as error:
            raise ValueError("selector seal references a non-development state") from error
        candidates = tuple(record["candidate_event_step_ids"])
        selected = tuple(record["selected_event_step_ids"])
        if (
            record["source_id"] != source_id
            or record["decision_step_id"] != step
            or candidates != event_ids
            or selected != tuple(sorted(selected))
            or len(set(selected)) != len(selected)
            or len(selected) > budget
            or not set(selected).issubset(event_ids)
        ):
            raise ValueError("selector seal record geometry drifted")
    if (
        observed != expected_keys
        or observed_order != sorted(expected_keys)
        or seal["record_count"] != len(expected_keys)
    ):
        raise ValueError("selector seal does not cover the exact B2/B4 matrix")
    expected_names = sorted(set(matrix[2]) | set(matrix[4]))
    expected_names_by_budget = {
        str(budget): sorted(selectors) for budget, selectors in matrix.items()
    }
    if (
        seal["selector_names"] != expected_names
        or seal["budgets"] != [2, 4]
        or seal["selector_names_by_budget"] != expected_names_by_budget
    ):
        raise ValueError("selector seal selector-name roster drifted")
    return seal


def _validate_hf_binding(value: Any, *, label: str) -> Mapping[str, Any]:
    binding = _mapping(value, label)
    _exact_keys(
        binding,
        {"repo", "repo_type", "revision", "path", "sha256", "size_bytes"},
        label,
    )
    if (
        not isinstance(binding["repo"], str)
        or not binding["repo"]
        or binding["repo_type"] not in {"dataset", "model"}
        or _IMMUTABLE_REVISION.fullmatch(str(binding["revision"])) is None
        or _SHA256.fullmatch(str(binding["sha256"])) is None
        or type(binding["size_bytes"]) is not int
        or binding["size_bytes"] <= 0
    ):
        raise ValueError(f"{label} immutable identity drifted")
    _relative_path(binding["path"], f"{label} path")
    return binding


def _validate_preparation_binding(value: Any) -> Mapping[str, Any]:
    binding = _mapping(value, "selector-preparation manifest binding")
    _exact_keys(
        binding,
        {"repo_id", "revision", "path", "sha256", "size_bytes"},
        "selector-preparation manifest binding",
    )
    if (
        not isinstance(binding["repo_id"], str)
        or not binding["repo_id"]
        or _IMMUTABLE_REVISION.fullmatch(str(binding["revision"])) is None
        or binding["path"] != SELECTOR_PREPARATION_MANIFEST_PATH
        or _SHA256.fullmatch(str(binding["sha256"])) is None
        or type(binding["size_bytes"]) is not int
        or binding["size_bytes"] <= 0
    ):
        raise ValueError("selector-preparation immutable identity drifted")
    return binding


def _validate_provenance_files(
    value: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], ...]:
    rows = _sequence(value, label)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        record = _mapping(raw, f"{label} record")
        _exact_keys(
            record,
            {"path", "sha256", "size_bytes"},
            f"{label} record",
        )
        path = _relative_path(record["path"], f"{label} path")
        if (
            path in seen
            or _SHA256.fullmatch(str(record["sha256"])) is None
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
        ):
            raise ValueError(f"{label} contains an invalid or duplicate file")
        seen.add(path)
        records.append(dict(record))
    return tuple(records)


def _require_exact_file_bindings(
    observed: Sequence[Mapping[str, Any]],
    *,
    expected: Sequence[Mapping[str, str]],
    label: str,
    require_order: bool = True,
) -> None:
    expected_by_path = {record["path"]: record["sha256"] for record in expected}
    if len(expected_by_path) != len(expected):
        raise ValueError(f"{label} expected file roster contains a duplicate")
    observed_by_path = {record["path"]: record["sha256"] for record in observed}
    if (
        len(observed_by_path) != len(observed)
        or observed_by_path != expected_by_path
        or (
            require_order
            and [record["path"] for record in observed]
            != [record["path"] for record in expected]
        )
    ):
        raise ValueError(f"{label} immutable file roster drifted")


def validate_selector_preparation_manifest_payload(
    payload: bytes,
    *,
    binding: Mapping[str, Any],
    substrate_binding: Mapping[str, Any],
    seal_binding: Mapping[str, Any],
    contract: LongHorizonContract,
    source_binding: Mapping[str, Any],
    selection_binding: Mapping[str, Any],
    expected_generator_git_revision: str,
) -> dict[str, Any]:
    if (
        sha256_bytes(payload) != binding["sha256"]
        or len(payload) != binding["size_bytes"]
    ):
        raise ValueError("selector-preparation manifest bytes drifted")
    manifest = load_json_object_bytes(
        payload, label="selector-preparation manifest"
    )
    if payload != pretty_json_bytes(manifest):
        raise ValueError("selector-preparation manifest is not canonical pretty JSON")
    expected_top = {
        "schema_version",
        "protocol_id",
        "status",
        "contract",
        "substrate",
        "models",
        "random_seed",
        "selector_names_by_budget",
        "counts",
        "selection_seal",
        "latency",
        "generator",
        "restoration_label_access_count",
        "restoration_distance_access_count",
        "policy_forward_count",
        "optimizer_step_count",
        "unopened_reserve_row_access_count",
        "selection_scoring_completed_before_restoration",
    }
    if (
        set(manifest) != expected_top
        or manifest.get("schema_version") != "1.0.0"
        or manifest.get("protocol_id")
        != "causalcache_long_horizon_label_blind_preparation_v1"
        or manifest.get("status")
        != "SEALED_LONG_HORIZON_LABEL_BLIND_SELECTOR_PREPARATION_V1"
    ):
        raise ValueError("selector-preparation manifest identity drifted")
    config_inventory = next(
        (
            record
            for record in source_binding["required_inventory"]
            if record["path"]
            == "code/configs/causalcache_long_horizon_development_v1.json"
        ),
        None,
    )
    expected_contract = {
        "protocol_id": contract.data["protocol_id"],
        "source_a": {
            "git_commit": source_binding["commit"],
            "config": {
                "path": source_binding["contract_path"],
                "sha256": source_binding["contract_sha256"],
                "size_bytes": (
                    None if config_inventory is None else config_inventory["size_bytes"]
                ),
            },
        },
        "selection_freeze": {
            "git_commit": selection_binding["freeze_commit"],
            "manifest": {
                "path": selection_binding["path"],
                "sha256": selection_binding["sha256"],
                "size_bytes": selection_binding["size_bytes"],
            },
        },
    }
    if config_inventory is None or manifest.get("contract") != expected_contract:
        raise ValueError("selector-preparation Source-A contract binding drifted")
    substrate = _mapping(manifest.get("substrate"), "preparation substrate")
    substrate_manifest = _mapping(
        substrate.get("manifest"), "preparation substrate manifest"
    )
    if (
        substrate.get("repo") != substrate_binding["repo"]
        or substrate.get("repo_type") != substrate_binding["repo_type"]
        or substrate.get("revision") != substrate_binding["revision"]
        or substrate_manifest
        != {
            "repo": substrate_binding["repo"],
            "repo_type": substrate_binding["repo_type"],
            "revision": substrate_binding["revision"],
            "path": substrate_binding["path"],
            "sha256": substrate_binding["sha256"],
            "size_bytes": substrate_binding["size_bytes"],
        }
    ):
        raise ValueError("selector-preparation substrate provenance drifted")
    selection_seal = _mapping(
        manifest.get("selection_seal"), "preparation selection seal"
    )
    _exact_keys(
        selection_seal,
        {"path", "sha256", "size_bytes"},
        "preparation selection seal",
    )
    if (
        seal_binding["path"] != selection_seal.get("path")
        or seal_binding["path"] != SELECTOR_SEAL_PATH
        or selection_seal.get("sha256") != seal_binding["sha256"]
        or selection_seal.get("size_bytes") != seal_binding["size_bytes"]
        or seal_binding["repo"] != binding["repo_id"]
        or seal_binding["revision"] != binding["revision"]
    ):
        raise ValueError("selector-preparation seal provenance drifted")
    models = _mapping(manifest.get("models"), "preparation models")
    formal = _mapping(models.get("formal58"), "preparation formal58 model")
    residual = _mapping(
        models.get("v4_frozen_base_residual"), "preparation v4 model"
    )
    _exact_keys(
        models,
        {"formal58", "v4_frozen_base_residual"},
        "preparation models",
    )
    _exact_keys(
        formal,
        {
            "repo",
            "repo_type",
            "private",
            "revision",
            "files",
            "family_seed_roster",
            "checkpoint_load_count",
        },
        "preparation formal58 model",
    )
    _exact_keys(
        residual,
        {
            "repo",
            "repo_type",
            "private",
            "revision",
            "files",
            "residual_seed_roster",
            "training_allowed",
        },
        "preparation v4 model",
    )
    expected_formal = contract.data["learned_model_artifacts"][
        "formal58_base_and_conditional"
    ]
    expected_residual = contract.data["learned_model_artifacts"][
        "v4_safe_frozen_base_residual"
    ]
    if (
        formal.get("repo") != expected_formal["repo"]
        or formal.get("repo_type") != expected_formal["repo_type"]
        or formal.get("private") is not expected_formal["private"]
        or formal.get("revision") != expected_formal["revision"]
        or formal.get("family_seed_roster")
        != {"conditional": [0, 1, 2, 3, 4], "independent": [0, 1, 2, 3, 4]}
        or formal.get("checkpoint_load_count") != 10
        or residual.get("repo") != expected_residual["repo"]
        or residual.get("repo_type") != expected_residual["repo_type"]
        or residual.get("private") is not expected_residual["private"]
        or residual.get("revision") != expected_residual["revision"]
        or residual.get("residual_seed_roster") != [0, 1, 2, 3, 4]
        or residual.get("training_allowed") is not False
    ):
        raise ValueError("selector-preparation immutable model provenance drifted")
    formal_files = _validate_provenance_files(
        formal["files"], label="preparation formal58 files"
    )
    expected_formal_files = tuple(
        {"path": record["path"], "sha256": record["sha256"]}
        for record in (
            tuple(expected_formal["ensemble_manifests"])
            + tuple(expected_formal["checkpoints"])
        )
    )
    if len(formal_files) != 12 or len(expected_formal_files) != 12:
        raise ValueError("preparation formal58 file denominator drifted")
    _require_exact_file_bindings(
        formal_files,
        expected=expected_formal_files,
        label="preparation formal58 files",
    )

    residual_files = _validate_provenance_files(
        residual["files"], label="preparation v4 files"
    )
    expected_residual_by_path = {
        expected_residual["manifest_path"]: expected_residual["manifest_sha256"],
        expected_residual["label_blind_seal_path"]: expected_residual[
            "label_blind_seal_sha256"
        ],
        **{
            record["path"]: record["sha256"]
            for record in expected_residual["residual_checkpoints"]
        },
    }
    expected_residual_paths = {
        expected_residual["manifest_path"],
        expected_residual["label_blind_seal_path"],
        "residual-model-metadata.json",
        *(record["path"] for record in expected_residual["residual_checkpoints"]),
    }
    observed_residual_by_path = {
        record["path"]: record for record in residual_files
    }
    if (
        len(residual_files) != 8
        or set(observed_residual_by_path) != expected_residual_paths
        or any(
            observed_residual_by_path[path]["sha256"] != digest
            for path, digest in expected_residual_by_path.items()
        )
    ):
        raise ValueError("preparation v4 immutable file roster drifted")

    generator = _mapping(manifest.get("generator"), "preparation generator")
    _exact_keys(
        generator,
        {"git_revision", "source_files"},
        "preparation generator",
    )
    expected_generator_paths = (
        "code/causalcache/long_horizon_prepare.py",
        "code/scripts/build_long_horizon_selector_seal.py",
        "code/scripts/validate_long_horizon_selector_seal.py",
    )
    source_inventory_by_path = {
        record["path"]: record for record in source_binding["required_inventory"]
    }
    if not set(expected_generator_paths).issubset(source_inventory_by_path):
        raise ValueError("Source-A inventory omits selector-preparation sources")
    generator_files = _validate_provenance_files(
        generator["source_files"], label="preparation generator source files"
    )
    expected_generator_files = tuple(
        {
            "path": path,
            "sha256": source_inventory_by_path[path]["sha256"],
        }
        for path in expected_generator_paths
    )
    if generator.get("git_revision") != expected_generator_git_revision:
        raise ValueError("selector-preparation generator revision drifted")
    _require_exact_file_bindings(
        generator_files,
        expected=expected_generator_files,
        label="preparation generator source files",
    )
    if any(
        observed["size_bytes"] != source_inventory_by_path[observed["path"]]["size_bytes"]
        for observed in generator_files
    ):
        raise ValueError("preparation generator source-file size drifted")
    matrix = {
        "2": list(contract.data["selector_matrix"]["budget_2"]),
        "4": list(contract.data["selector_matrix"]["budget_4"]),
    }
    if manifest.get("selector_names_by_budget") != matrix:
        raise ValueError("selector-preparation selector matrix drifted")
    if manifest.get("random_seed") != contract.data["selector_matrix"][
        "random_seed_roster"
    ][0]:
        raise ValueError("selector-preparation random seed drifted")
    if manifest.get("counts") != {
        "state_count": 48,
        "n8_state_count": 24,
        "n16_state_count": 24,
        "selection_record_count": 576,
        "budget_selector_arm_count": 12,
    }:
        raise ValueError("selector-preparation formal counts drifted")
    for key in (
        "restoration_label_access_count",
        "restoration_distance_access_count",
        "policy_forward_count",
        "optimizer_step_count",
        "unopened_reserve_row_access_count",
    ):
        if manifest.get(key) != 0:
            raise PermissionError(f"selector-preparation forbidden access is nonzero: {key}")
    if manifest.get("selection_scoring_completed_before_restoration") is not True:
        raise PermissionError("selector seal was not completed before restoration")
    return manifest


def validate_runner_config(
    value: Mapping[str, Any],
    *,
    contract: LongHorizonContract,
    selection_manifest_payload: bytes,
    substrate_manifest_payload: bytes | None = None,
    selector_preparation_manifest_payload: bytes | None = None,
    selector_seal_payload: bytes | None = None,
) -> dict[str, Any]:
    """Validate Execution-B bindings without opening any reserve row content."""
    config = _mapping(value, "Execution-B runner config")
    _exact_keys(
        config,
        {
            "schema_version",
            "protocol_id",
            "status",
            "source_a",
            "selection_manifest",
            "substrate",
            "selector_preparation_manifest",
            "selector_seal",
            "worker_topology",
            "access_firewall",
            "operation_accounting",
            "authorization",
        },
        "Execution-B runner config",
    )
    if (
        config["schema_version"] != SCHEMA_VERSION
        or config["protocol_id"] != PROTOCOL_ID
        or config["status"] != RUNNER_STATUS
    ):
        raise ValueError("Execution-B runner identity drifted")
    source = _mapping(config["source_a"], "Source-A binding")
    _exact_keys(
        source,
        {
            "branch",
            "commit",
            "remote_ref_commit",
            "contract_path",
            "contract_sha256",
            "required_inventory",
            "required_inventory_sha256",
        },
        "Source-A binding",
    )
    expected_paths = tuple(contract.data["source_freeze"]["required_source_a_paths"])
    inventory = _validate_inventory(
        source["required_inventory"], expected_paths=expected_paths
    )
    expected_branch = contract.data["source_freeze"]["branch"]
    if (
        source["branch"] != expected_branch
        or _GIT_COMMIT.fullmatch(str(source["commit"])) is None
        or source["remote_ref_commit"] != source["commit"]
        or source["contract_path"]
        != "code/configs/causalcache_long_horizon_development_v1.json"
        or source["contract_sha256"] != contract.sha256
        or source["required_inventory_sha256"] != inventory_sha256(inventory)
    ):
        raise ValueError("Source-A Git, contract, or inventory binding drifted")
    selection = _mapping(config["selection_manifest"], "selection binding")
    _exact_keys(
        selection,
        {
            "path",
            "sha256",
            "freeze_commit",
            "freeze_parent_source_a_commit",
            "size_bytes",
        },
        "selection binding",
    )
    if (
        selection["path"] != SELECTION_MANIFEST_PATH
        or _SHA256.fullmatch(str(selection["sha256"])) is None
        or type(selection["size_bytes"]) is not int
        or selection["size_bytes"] <= 0
        or _GIT_COMMIT.fullmatch(str(selection["freeze_commit"])) is None
        or selection["freeze_parent_source_a_commit"] != source["commit"]
    ):
        raise ValueError("selection-freeze commit, path, or SHA256 drifted")
    selection_manifest = validate_selection_manifest_payload(
        selection_manifest_payload,
        contract=contract,
        expected_sha256=selection["sha256"],
    )
    if len(selection_manifest_payload) != selection["size_bytes"]:
        raise ValueError("selection manifest byte size drifted")
    if selection_manifest["generator"]["git_revision"] != source["commit"]:
        raise ValueError("selection manifest was not generated from frozen Source-A")

    substrate = _validate_hf_binding(config["substrate"], label="substrate binding")
    preparation_binding = _validate_preparation_binding(
        config["selector_preparation_manifest"]
    )
    seal_binding = _validate_hf_binding(
        config["selector_seal"], label="selector-seal binding"
    )
    if (
        substrate["repo"] != contract.data["artifact_plan"]["repo"]
        or substrate["repo_type"] != "dataset"
        or substrate["path"] != SUBSTRATE_MANIFEST_RELATIVE_PATH
        or seal_binding["repo"] != contract.data["artifact_plan"]["repo"]
        or seal_binding["repo_type"] != "dataset"
        or seal_binding["path"] != SELECTOR_SEAL_PATH
    ):
        raise ValueError("Execution-B artifact destination binding drifted")
    if substrate_manifest_payload is not None:
        if (
            sha256_bytes(substrate_manifest_payload) != substrate["sha256"]
            or len(substrate_manifest_payload) != substrate["size_bytes"]
        ):
            raise ValueError("substrate manifest SHA256 drifted")
        manifest = load_json_object_bytes(
            substrate_manifest_payload, label="substrate manifest"
        )
        development_ids = [
            record["trajectory_id"]
            for record in selection_manifest["splits"]["development"]["trajectories"]
        ]
        if (
            manifest.get("dataset_repo") != substrate["repo"]
            or manifest.get("development_source_ids") != development_ids
            or manifest.get("unopened_reserve", {}).get(
                "python_row_materialization_count"
            )
            != 0
            or manifest.get("unopened_reserve", {}).get(
                "semantic_content_access_count"
            )
            != 0
            or manifest.get("unopened_reserve", {}).get(
                "transport_row_group_overread_possible"
            )
            is not True
            or manifest.get("unopened_reserve_semantic_content_access_count")
            != 0
            or manifest.get("source_transport_row_group_overread_possible")
            is not True
        ):
            raise ValueError("substrate manifest development/reserve firewall drifted")
    if selector_seal_payload is not None:
        if len(selector_seal_payload) != seal_binding["size_bytes"]:
            raise ValueError("selector seal byte size drifted")
        validate_selector_seal_payload(
            selector_seal_payload,
            expected_sha256=seal_binding["sha256"],
            selection_manifest=selection_manifest,
            contract=contract,
        )
    if selector_preparation_manifest_payload is not None:
        validate_selector_preparation_manifest_payload(
            selector_preparation_manifest_payload,
            binding=preparation_binding,
            substrate_binding=substrate,
            seal_binding=seal_binding,
            contract=contract,
            source_binding=source,
            selection_binding=selection,
            expected_generator_git_revision=selection["freeze_commit"],
        )

    topology = _mapping(config["worker_topology"], "worker topology")
    _exact_keys(
        topology,
        {
            "host_alias",
            "worker_count",
            "gpu_assignments",
            "selection_rank_ranges",
            "trajectories_per_worker",
            "one_explicit_gpu_per_worker",
        },
        "worker topology",
    )
    gpu_assignments = _sequence(topology["gpu_assignments"], "GPU assignments")
    rank_ranges = _sequence(topology["selection_rank_ranges"], "rank ranges")
    observed_ranges = tuple(tuple(item) for item in rank_ranges)
    if (
        topology["host_alias"] != "hyper00"
        or topology["worker_count"] != WORKER_COUNT
        or topology["trajectories_per_worker"] != TRAJECTORIES_PER_WORKER
        or topology["one_explicit_gpu_per_worker"] is not True
        or len(gpu_assignments) != WORKER_COUNT
        or observed_ranges != EXPECTED_RANK_RANGES
    ):
        raise ValueError("Hyper00 four-worker topology drifted")
    normalized_gpu_assignments = []
    for index, raw in enumerate(gpu_assignments):
        assignment = _mapping(raw, "GPU assignment")
        _exact_keys(
            assignment,
            {"worker_index", "host_gpu_id", "container_cuda_ordinal"},
            "GPU assignment",
        )
        if (
            assignment["worker_index"] != index
            or _GPU_ID.fullmatch(str(assignment["host_gpu_id"])) is None
            or type(assignment["container_cuda_ordinal"]) is not int
            or assignment["container_cuda_ordinal"] < 0
        ):
            raise ValueError("GPU host-ID/container-ordinal mapping drifted")
        normalized_gpu_assignments.append(dict(assignment))
    if (
        len({row["host_gpu_id"] for row in normalized_gpu_assignments})
        != WORKER_COUNT
        or len(
            {row["container_cuda_ordinal"] for row in normalized_gpu_assignments}
        )
        != WORKER_COUNT
    ):
        raise ValueError("GPU host IDs and CUDA ordinals must each be unique")

    firewall = _mapping(config["access_firewall"], "execution firewall")
    expected_firewall = {
        "reserve_access_count": 0,
        "top_up_count": 0,
        "replacement_count": 0,
        "post_selection_filter_count": 0,
        "selector_reseal_count": 0,
        "threshold_update_count": 0,
        "old_confirm_access_count": 0,
        "closed_loop_access_count": 0,
        "sealed_test_access_count": 0,
    }
    if dict(firewall) != expected_firewall:
        raise ValueError("Execution-B forbidden access accounting is nonzero or incomplete")
    if config["operation_accounting"] != contract.data["operation_accounting"]:
        raise ValueError("Execution-B operation accounting drifted from Source-A")
    authorization = _mapping(config["authorization"], "Execution-B authorization")
    expected_authorization = {
        "development_policy_access": True,
        "development_restoration_access": True,
        "selector_sets_already_label_blind_sealed": True,
        "reserve_access": False,
        "old_confirm_access": False,
        "closed_loop_access": False,
        "sealed_test_access": False,
    }
    if dict(authorization) != expected_authorization:
        raise ValueError("Execution-B authorization drifted")
    return _canonical_copy(config)


def build_worker_assignments(config: Mapping[str, Any]) -> tuple[WorkerAssignment, ...]:
    topology = _mapping(config["worker_topology"], "worker topology")
    gpu_assignments = tuple(topology["gpu_assignments"])
    ranges = tuple(tuple(value) for value in topology["selection_rank_ranges"])
    if len(gpu_assignments) != WORKER_COUNT or ranges != EXPECTED_RANK_RANGES:
        raise ValueError("worker assignment topology drifted")
    return tuple(
        WorkerAssignment(
            worker_index=index,
            worker_id=f"worker-{index}",
            host_gpu_id=str(gpu_assignments[index]["host_gpu_id"]),
            container_cuda_ordinal=int(
                gpu_assignments[index]["container_cuda_ordinal"]
            ),
            selection_ranks=tuple(range(start, end + 1)),
        )
        for index, (start, end) in enumerate(ranges)
    )


def _parse_canonical_jsonl(payload: bytes, *, label: str) -> tuple[dict[str, Any], ...]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{label} must be non-empty newline-terminated JSONL")
    records = tuple(
        load_json_object_bytes(line, label=label)
        for line in payload.splitlines()
        if line
    )
    replay = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    if replay != payload:
        raise ValueError(f"{label} is not canonical JSONL")
    return records


def load_development_trajectories(
    substrate_root: str | Path,
    *,
    selection_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Load only the development JSONL; reserve has no materialized payload path."""
    root = Path(substrate_root)
    records = _parse_canonical_jsonl(
        root.joinpath(*PurePosixPath(TRAJECTORY_JSONL_RELATIVE_PATH).parts).read_bytes(),
        label="long-horizon development trajectories",
    )
    expected = tuple(
        str(record["trajectory_id"])
        for record in selection_manifest["splits"]["development"]["trajectories"]
    )
    observed = tuple(str(record.get("source_id")) for record in records)
    if len(records) != TRAJECTORY_COUNT or observed != expected:
        raise ValueError("substrate trajectory roster differs from selected development")
    if any(record.get("role") != "development" for record in records):
        raise PermissionError("non-development row appeared in the substrate payload")
    return records


def selector_records_by_source(
    selector_seal: Mapping[str, Any],
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for raw in selector_seal["records"]:
        record = _mapping(raw, "selector seal record")
        grouped[str(record["source_id"])].append(record)
    return {
        source_id: tuple(
            sorted(
                records,
                key=lambda item: (
                    int(item["decision_step_id"]),
                    int(item["budget_event_capacity"]),
                    str(item["selector_name"]),
                ),
            )
        )
        for source_id, records in grouped.items()
    }


def _counter(value: Any, *, label: str) -> dict[str, int]:
    record = _mapping(value, label)
    _exact_keys(record, {"requested", "completed", "missing"}, label)
    if any(type(record[key]) is not int or record[key] < 0 for key in record):
        raise ValueError(f"{label} values must be non-negative integers")
    if record["completed"] + record["missing"] != record["requested"]:
        raise ValueError(f"{label} completed + missing must equal requested")
    return dict(record)


def _operation_counts(value: Any, *, label: str) -> dict[str, dict[str, int]]:
    counts = _mapping(value, label)
    if not counts:
        raise ValueError(f"{label} cannot be empty")
    return {
        str(name): _counter(record, label=f"{label}.{name}")
        for name, record in counts.items()
        if isinstance(name, str) and name
    }


def validate_runtime_trajectory_record(
    value: Mapping[str, Any],
    *,
    source_id: str,
    contract: LongHorizonContract,
) -> dict[str, Any]:
    record = _mapping(value, "runtime trajectory record")
    _exact_keys(record, {"source_id", "n8", "n16_pairs"}, "runtime trajectory record")
    if record["source_id"] != source_id:
        raise ValueError("runtime trajectory source identity drifted")
    n8 = _mapping(record["n8"], "n8 runtime record")
    if not isinstance(n8.get("valid"), bool):
        raise ValueError("n8 runtime record must contain a boolean valid flag")
    if (
        n8.get("state_id") != f"{source_id}:decision_step:010"
        or tuple(n8.get("candidate_event_step_ids", ())) != tuple(range(1, 9))
    ):
        raise ValueError("n8 runtime state/source geometry drifted")
    n8_counts = _operation_counts(n8.get("operation_counts"), label="n8 operations")
    expected_n8 = {
        "processor_preflights": 164,
        "canonical_action_parse_attempts": 2,
        "reference_teacher_forwards": 1,
        "candidate_teacher_forwards": 163,
        "distance_rows": int(
            contract.data["operation_accounting"][
                "n8_budget_4_unique_subset_rows_per_valid_trajectory"
            ]
        ),
        "analytic_self_distance_rows": 1,
    }
    if set(n8_counts) != set(expected_n8):
        raise ValueError("n8 operation-name inventory drifted")
    for name, requested in expected_n8.items():
        if name not in n8_counts or n8_counts[name]["requested"] != requested:
            raise ValueError(f"n8 requested operation count drifted: {name}")
    if n8["valid"] and any(
        counter["missing"] != 0 for counter in n8_counts.values()
    ):
        raise ValueError("valid n8 state must complete every requested operation")

    n16_pairs = _sequence(record["n16_pairs"], "n16 runtime pairs")
    pair_specs = tuple(
        (2, tuple(pair))
        for pair in contract.data["reference_contract"]["n16"][
            "budget_2_comparator_pairs"
        ]
    ) + tuple(
        (4, tuple(pair))
        for pair in contract.data["reference_contract"]["n16"][
            "budget_4_comparator_pairs"
        ]
    )
    if len(n16_pairs) != len(pair_specs):
        raise ValueError("n16 comparator-pair denominator drifted")
    accounting = contract.data["operation_accounting"]
    distance_key = next(
        (
            key
            for key in (
                "n16_maximum_nonself_distance_rows_per_valid_pair",
                "n16_distance_rows_per_valid_pair",
                "n16_distance_arms_per_valid_pair",
            )
            if key in accounting
        ),
        None,
    )
    if distance_key is None:
        raise ValueError("n16 maximum nonself-row accounting is missing")
    expected_distance_rows = int(accounting[distance_key])
    seen_pairs: set[tuple[int, tuple[str, str]]] = set()
    for raw in n16_pairs:
        pair = _mapping(raw, "n16 pair record")
        budget = pair.get("budget_event_capacity")
        selectors = (
            pair.get("left_selector_name"),
            pair.get("right_selector_name"),
        )
        if type(budget) is not int or len(selectors) != 2:
            raise ValueError("n16 pair identity is malformed")
        if (
            pair.get("state_id") != f"{source_id}:decision_step:018"
            or tuple(pair.get("candidate_event_step_ids", ()))
            != tuple(range(1, 17))
            or _SHA256.fullmatch(str(pair.get("selection_seal_sha256"))) is None
        ):
            raise ValueError("n16 runtime state/source/seal geometry drifted")
        identity = (budget, selectors)
        if identity in seen_pairs or identity not in pair_specs:
            raise ValueError("n16 comparator pair is duplicated or unexpected")
        seen_pairs.add(identity)
        if not isinstance(pair.get("valid"), bool):
            raise ValueError("n16 pair must contain a boolean valid flag")
        counts = _operation_counts(pair.get("operation_counts"), label="n16 operations")
        left = tuple(pair.get("left_selected_event_step_ids", ()))
        right = tuple(pair.get("right_selected_event_step_ids", ()))
        reference = tuple(sorted(set(left) | set(right)))
        unique_nonself = []
        for coalition in ((), left, right):
            if coalition != reference and coalition not in unique_nonself:
                unique_nonself.append(coalition)
        requested_distance_rows = len(unique_nonself)
        if requested_distance_rows > expected_distance_rows:
            raise ValueError("n16 unique nonself row count exceeds the frozen maximum")
        expected = {
            "processor_preflights": 1 + requested_distance_rows,
            "canonical_action_parse_attempts": 2,
            "reference_teacher_forwards": 1,
            "candidate_teacher_forwards": requested_distance_rows,
            "distance_rows": requested_distance_rows,
            "analytic_self_distance_rows": 1,
        }
        if set(counts) != set(expected):
            raise ValueError("n16 operation-name inventory drifted")
        for name, requested in expected.items():
            if name not in counts or counts[name]["requested"] != requested:
                raise ValueError(f"n16 requested operation count drifted: {name}")
        if pair["valid"] and any(
            counter["missing"] != 0 for counter in counts.values()
        ):
            raise ValueError("valid n16 pair must complete every requested operation")
    if seen_pairs != set(pair_specs):
        raise ValueError("n16 comparator-pair roster drifted")
    return _canonical_copy(record)


def _persistent_output_root(path: str | Path, *, persistent_root: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute():
        raise ValueError("worker output root must be absolute")
    resolved = root.resolve()
    persistent = Path(persistent_root).resolve()
    if resolved != persistent and persistent not in resolved.parents:
        raise ValueError("worker outputs must remain under the persistent /data root")
    return resolved


def atomic_write(path: Path, payload: bytes) -> None:
    """Durably replace one receipt without exposing partially written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("atomic receipt write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _result_path(output_root: Path, assignment: WorkerAssignment, rank: int) -> Path:
    return output_root / assignment.worker_id / f"trajectory-rank-{rank:02d}.json"


def _receipt_path(output_root: Path, assignment: WorkerAssignment) -> Path:
    return output_root / assignment.worker_id / "receipt.json"


def _validate_worker_result(
    value: Mapping[str, Any],
    *,
    assignment: WorkerAssignment,
    rank: int,
    source_id: str,
    runner_config_sha256: str,
    contract: LongHorizonContract,
) -> dict[str, Any]:
    result = _mapping(value, "worker result")
    _exact_keys(
        result,
        {
            "schema_version",
            "protocol_id",
            "worker_id",
            "worker_index",
            "host_gpu_id",
            "container_cuda_ordinal",
            "selection_rank",
            "source_id",
            "runner_config_sha256",
            "runtime_record",
            "access_accounting",
        },
        "worker result",
    )
    if (
        result["schema_version"] != SCHEMA_VERSION
        or result["protocol_id"] != WORKER_RESULT_PROTOCOL_ID
        or result["worker_id"] != assignment.worker_id
        or result["worker_index"] != assignment.worker_index
        or result["host_gpu_id"] != assignment.host_gpu_id
        or result["container_cuda_ordinal"] != assignment.container_cuda_ordinal
        or result["selection_rank"] != rank
        or result["source_id"] != source_id
        or result["runner_config_sha256"] != runner_config_sha256
        or dict(_mapping(result["access_accounting"], "worker access accounting"))
        != {
            "reserve_access_count": 0,
            "top_up_count": 0,
            "replacement_count": 0,
            "post_selection_filter_count": 0,
        }
    ):
        raise ValueError("worker result identity or firewall drifted")
    validate_runtime_trajectory_record(
        result["runtime_record"], source_id=source_id, contract=contract
    )
    return _canonical_copy(result)


def _load_existing_result(path: Path) -> dict[str, Any]:
    payload, result = load_json_object(path, label="existing worker result")
    if payload != pretty_json_bytes(result):
        raise ValueError("existing worker result is not canonical pretty JSON")
    return result


def _worker_receipt(
    *,
    assignment: WorkerAssignment,
    runner_config_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    files = []
    for rank in assignment.selection_ranks:
        path = _result_path(output_root, assignment, rank)
        files.append(
            {
                "selection_rank": rank,
                "path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": WORKER_RECEIPT_PROTOCOL_ID,
        "status": "COMPLETED_LONG_HORIZON_WORKER_SHARD",
        "worker_id": assignment.worker_id,
        "worker_index": assignment.worker_index,
        "host_gpu_id": assignment.host_gpu_id,
        "container_cuda_ordinal": assignment.container_cuda_ordinal,
        "selection_ranks": list(assignment.selection_ranks),
        "runner_config_sha256": runner_config_sha256,
        "files": files,
    }


def validate_worker_receipt(
    value: Mapping[str, Any],
    *,
    assignment: WorkerAssignment,
    runner_config_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    receipt = _mapping(value, "worker receipt")
    _exact_keys(
        receipt,
        {
            "schema_version",
            "protocol_id",
            "status",
            "worker_id",
            "worker_index",
            "host_gpu_id",
            "container_cuda_ordinal",
            "selection_ranks",
            "runner_config_sha256",
            "files",
        },
        "worker receipt",
    )
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or receipt["protocol_id"] != WORKER_RECEIPT_PROTOCOL_ID
        or receipt["status"] != "COMPLETED_LONG_HORIZON_WORKER_SHARD"
        or receipt["worker_id"] != assignment.worker_id
        or receipt["worker_index"] != assignment.worker_index
        or receipt["host_gpu_id"] != assignment.host_gpu_id
        or receipt["container_cuda_ordinal"] != assignment.container_cuda_ordinal
        or receipt["selection_ranks"] != list(assignment.selection_ranks)
        or receipt["runner_config_sha256"] != runner_config_sha256
    ):
        raise ValueError("worker receipt identity drifted")
    files = _sequence(receipt["files"], "worker receipt files")
    if len(files) != TRAJECTORIES_PER_WORKER:
        raise ValueError("worker receipt file denominator drifted")
    for rank, raw in zip(assignment.selection_ranks, files, strict=True):
        record = _mapping(raw, "worker receipt file")
        expected_path = _result_path(output_root, assignment, rank)
        if (
            record.get("selection_rank") != rank
            or record.get("path") != expected_path.relative_to(output_root).as_posix()
            or record.get("sha256") != sha256_file(expected_path)
            or record.get("size_bytes") != expected_path.stat().st_size
        ):
            raise ValueError("worker receipt file identity drifted")
    return _canonical_copy(receipt)


def run_worker_shard(
    *,
    contract: LongHorizonContract,
    runner_config: Mapping[str, Any],
    runner_config_sha256: str,
    selection_manifest: Mapping[str, Any],
    selector_seal: Mapping[str, Any],
    trajectories: Sequence[Mapping[str, Any]],
    worker_index: int,
    host_gpu_id: str,
    container_cuda_ordinal: int,
    output_root: str | Path,
    adapter: LongHorizonWorkerAdapter,
    persistent_root: str | Path = "/data",
) -> dict[str, Any]:
    """Run or resume one six-trajectory shard with one explicit GPU ID."""
    assignments = build_worker_assignments(runner_config)
    if type(worker_index) is not int or not 0 <= worker_index < len(assignments):
        raise ValueError("worker index is outside the frozen topology")
    assignment = assignments[worker_index]
    if (
        host_gpu_id != assignment.host_gpu_id
        or container_cuda_ordinal != assignment.container_cuda_ordinal
    ):
        raise ValueError(
            "worker GPU host ID or container CUDA ordinal differs from the frozen assignment"
        )
    if len(trajectories) != TRAJECTORY_COUNT:
        raise ValueError("worker requires the fixed 24-trajectory development denominator")
    root = _persistent_output_root(output_root, persistent_root=persistent_root)
    selector_by_source = selector_records_by_source(selector_seal)
    selected_records = selection_manifest["splits"]["development"]["trajectories"]
    expected_source_ids = tuple(str(record["trajectory_id"]) for record in selected_records)
    observed_source_ids = tuple(str(record.get("source_id")) for record in trajectories)
    if observed_source_ids != expected_source_ids:
        raise ValueError("worker trajectory roster differs from development selection")

    receipt_path = _receipt_path(root, assignment)
    if receipt_path.exists():
        payload, receipt = load_json_object(receipt_path, label="existing worker receipt")
        if payload != pretty_json_bytes(receipt):
            raise ValueError("existing worker receipt is not canonical pretty JSON")
        return validate_worker_receipt(
            receipt,
            assignment=assignment,
            runner_config_sha256=runner_config_sha256,
            output_root=root,
        )

    for rank in assignment.selection_ranks:
        trajectory = trajectories[rank]
        source_id = expected_source_ids[rank]
        path = _result_path(root, assignment, rank)
        if path.exists():
            existing = _load_existing_result(path)
            _validate_worker_result(
                existing,
                assignment=assignment,
                rank=rank,
                source_id=source_id,
                runner_config_sha256=runner_config_sha256,
                contract=contract,
            )
            continue
        try:
            source_selector_records = selector_by_source[source_id]
        except KeyError as error:
            raise ValueError("selector seal is missing a development trajectory") from error
        runtime_record = adapter.run_trajectory(
            trajectory=trajectory,
            selector_records=source_selector_records,
            host_gpu_id=assignment.host_gpu_id,
            container_cuda_ordinal=assignment.container_cuda_ordinal,
            runner_config=runner_config,
        )
        validated_runtime = validate_runtime_trajectory_record(
            runtime_record,
            source_id=source_id,
            contract=contract,
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": WORKER_RESULT_PROTOCOL_ID,
            "worker_id": assignment.worker_id,
            "worker_index": assignment.worker_index,
            "host_gpu_id": assignment.host_gpu_id,
            "container_cuda_ordinal": assignment.container_cuda_ordinal,
            "selection_rank": rank,
            "source_id": source_id,
            "runner_config_sha256": runner_config_sha256,
            "runtime_record": validated_runtime,
            "access_accounting": {
                "reserve_access_count": 0,
                "top_up_count": 0,
                "replacement_count": 0,
                "post_selection_filter_count": 0,
            },
        }
        atomic_write(path, pretty_json_bytes(result))
    receipt = _worker_receipt(
        assignment=assignment,
        runner_config_sha256=runner_config_sha256,
        output_root=root,
    )
    atomic_write(receipt_path, pretty_json_bytes(receipt))
    return validate_worker_receipt(
        receipt,
        assignment=assignment,
        runner_config_sha256=runner_config_sha256,
        output_root=root,
    )


def _collect_operation_counts(
    runtime_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for record in runtime_records:
        blocks = [("n8", record["n8"]), *[("n16", row) for row in record["n16_pairs"]]]
        for prefix, block in blocks:
            counts = _operation_counts(block["operation_counts"], label="operations")
            for name, values in counts.items():
                totals[f"{prefix}.{name}"].update(values)
    return {
        name: {key: values[key] for key in ("requested", "completed", "missing")}
        for name, values in sorted(totals.items())
    }


def _coverage_status(
    runtime_records: Sequence[Mapping[str, Any]],
    *,
    contract: LongHorizonContract,
) -> tuple[str, int, dict[str, int]]:
    n8_valid = sum(record["n8"]["valid"] is True for record in runtime_records)
    by_pair: Counter[str] = Counter()
    for record in runtime_records:
        for pair in record["n16_pairs"]:
            key = (
                f"B{pair['budget_event_capacity']}:"
                + "__vs__".join(
                    (pair["left_selector_name"], pair["right_selector_name"])
                )
            )
            by_pair[key] += pair["valid"] is True
    thresholds = contract.data["validity_and_go_contract"][
        "policy_coverage_thresholds"
    ]
    passes = n8_valid >= thresholds["minimum_valid_n8_trajectories"] and all(
        value >= thresholds["minimum_valid_n16_trajectories_per_comparator_pair"]
        for value in by_pair.values()
    )
    status = (
        "GO_LONG_HORIZON_POLICY_COVERAGE"
        if passes
        else thresholds["failure_outcome"]
    )
    return status, n8_valid, dict(sorted(by_pair.items()))


def _seal_index(selector_seal: Mapping[str, Any]) -> Mapping[tuple[int, str, str], tuple[int, ...]]:
    result: dict[tuple[int, str, str], tuple[int, ...]] = {}
    for raw in selector_seal["records"]:
        record = _mapping(raw, "selector seal record")
        key = (
            int(record["budget_event_capacity"]),
            str(record["selector_name"]),
            str(record["state_id"]),
        )
        if key in result:
            raise ValueError("selector seal contains duplicate evaluation keys")
        result[key] = tuple(record["selected_event_step_ids"])
    return result


def _distance_mapping(rows: Any, *, label: str) -> dict[tuple[int, ...], float]:
    result: dict[tuple[int, ...], float] = {}
    for raw in _sequence(rows, label):
        row = _mapping(raw, f"{label} row")
        coalition = tuple(row.get("restored_event_step_ids", ()))
        if coalition in result:
            raise ValueError(f"{label} contains a duplicate coalition")
        distance = row.get("distance")
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            raise ValueError(f"{label} distance must be numeric")
        result[coalition] = float(distance)
    return result


def _metric(
    *,
    selector: str,
    source_id: str,
    state_id: str,
    horizon_n: int,
    budget: int,
    selected: tuple[int, ...],
    distance: float,
    empty_distance: float,
) -> StateSelectionMetric:
    raw = empty_distance - distance
    normalized = raw / empty_distance if empty_distance > NORMALIZATION_EPSILON else 0.0
    return StateSelectionMetric(
        selector_name=selector,
        source_id=source_id,
        state_id=state_id,
        horizon_n=horizon_n,
        budget_event_capacity=budget,
        selected_event_step_ids=selected,
        distance=distance,
        raw_utility=raw,
        normalized_recovery=normalized,
    )


def _paired_condition_report(
    records: Sequence[StateSelectionMetric],
    *,
    left_selector: str,
    right_selector: str,
    budget: int,
    config: Mapping[str, Any],
    selected_source_ids: Sequence[str],
) -> dict[str, Any]:
    observed_normalized = paired_trajectory_deltas(
        records,
        left_selector=left_selector,
        right_selector=right_selector,
        metric="normalized_recovery",
        budget_event_capacity=budget,
    )
    observed_raw = paired_trajectory_deltas(
        records,
        left_selector=left_selector,
        right_selector=right_selector,
        metric="raw_utility",
        budget_event_capacity=budget,
    )
    if set(observed_normalized) != set(observed_raw):
        raise ValueError("paired n8 raw/normalized valid rosters differ")
    selected = tuple(selected_source_ids)
    if (
        len(selected) != 24
        or len(set(selected)) != 24
        or config["n8_fixed_selected_trajectory_denominator"] != len(selected)
        or not set(observed_normalized).issubset(selected)
    ):
        raise ValueError("paired n8 reducer lost the fixed selected denominator")
    normalized = {
        source_id: float(observed_normalized.get(source_id, 0.0))
        for source_id in selected
    }
    raw = {
        source_id: float(observed_raw.get(source_id, 0.0))
        for source_id in selected
    }
    interval = paired_trajectory_bootstrap(
        normalized,
        resamples=int(config["n8_paired_trajectory_bootstrap_iterations"]),
        seed=int(config["n8_paired_trajectory_bootstrap_seed"]),
        confidence=float(config["n8_paired_trajectory_bootstrap_confidence"]),
    )
    return {
        "left_selector": left_selector,
        "right_selector": right_selector,
        "budget_event_capacity": budget,
        "valid_paired_trajectory_count": len(observed_normalized),
        "zero_filled_invalid_trajectory_count": len(selected)
        - len(observed_normalized),
        "fixed_selected_trajectory_denominator": int(
            config["n8_fixed_selected_trajectory_denominator"]
        ),
        "mean_normalized_delta": statistics.fmean(normalized.values()),
        "mean_raw_delta": statistics.fmean(raw.values()),
        "positive_trajectory_count": sum(value > 0.0 for value in normalized.values()),
        "bootstrap": {
            "estimate": interval.estimate,
            "lower": interval.lower,
            "upper": interval.upper,
            "confidence": interval.confidence,
            "resamples": interval.resamples,
            "seed": interval.seed,
            "trajectory_count": interval.trajectory_count,
        },
    }


def _n16_pair_delta(
    pair_metrics: Mapping[tuple[int, str, str], Sequence[StateSelectionMetric]],
    *,
    budget: int,
    left_selector: str,
    right_selector: str,
    selected_source_ids: Sequence[str],
) -> dict[str, Any]:
    key = (budget, left_selector, right_selector)
    reverse = (budget, right_selector, left_selector)
    if key in pair_metrics:
        records = pair_metrics[key]
    elif reverse in pair_metrics:
        records = pair_metrics[reverse]
    else:
        raise ValueError("required n16 comparator pair is absent")
    by_state: dict[tuple[str, str], dict[str, StateSelectionMetric]] = defaultdict(dict)
    for record in records:
        by_state[(record.source_id, record.state_id)][record.selector_name] = record
    observed_normalized: dict[str, float] = {}
    observed_raw: dict[str, float] = {}
    for (source_id, _), values in sorted(by_state.items()):
        if set(values) != {left_selector, right_selector}:
            raise ValueError("n16 pair metric lost one sealed comparator")
        observed_normalized[source_id] = (
            values[left_selector].normalized_recovery
            - values[right_selector].normalized_recovery
        )
        observed_raw[source_id] = (
            values[left_selector].raw_utility - values[right_selector].raw_utility
        )
    if not observed_normalized:
        raise ValueError("required n16 comparator pair has no valid state")
    selected = tuple(selected_source_ids)
    if (
        len(selected) != 24
        or len(set(selected)) != 24
        or not set(observed_normalized).issubset(selected)
    ):
        raise ValueError("n16 pair reducer lost the fixed selected denominator")
    normalized = {
        source_id: float(observed_normalized.get(source_id, 0.0))
        for source_id in selected
    }
    raw = {
        source_id: float(observed_raw.get(source_id, 0.0))
        for source_id in selected
    }
    return {
        "budget_event_capacity": budget,
        "left_selector": left_selector,
        "right_selector": right_selector,
        "valid_paired_trajectory_count": len(observed_normalized),
        "zero_filled_invalid_trajectory_count": len(selected)
        - len(observed_normalized),
        "mean_normalized_delta": statistics.fmean(normalized.values()),
        "mean_raw_delta": statistics.fmean(raw.values()),
        "positive_trajectory_count": sum(value > 0.0 for value in normalized.values()),
    }


def evaluate_scientific_go(
    runtime_records: Sequence[Mapping[str, Any]],
    *,
    selector_seal: Mapping[str, Any],
    contract: LongHorizonContract,
    policy_coverage_passed: bool,
) -> dict[str, Any]:
    """Build canonical tables/oracles and reduce every preregistered GO condition."""
    fixed_reduction = contract.data["validity_and_go_contract"].get(
        "fixed_denominator_reduction"
    )
    expected_fixed_reduction = {
        "selected_trajectory_denominator": 24,
        "invalid_state_contribution": 0.0,
        "n8_paired_delta_zero_fill": True,
        "n16_pair_delta_zero_fill": True,
        "n8_exact_utility_ratio_zero_fill": True,
        "zero_exact_oracle_utility_ratio": 0.0,
        "bootstrap_uses_all_selected_trajectories": True,
        "positive_support_uses_all_selected_trajectories": True,
        "valid_state_missing_or_duplicate_selector_outcome": (
            "INVALID_LONG_HORIZON_SELECTOR_JOIN"
        ),
    }
    if fixed_reduction != expected_fixed_reduction:
        raise ValueError("fixed-denominator zero-fill reduction contract drifted")
    selections = _seal_index(selector_seal)
    selector_seal_sha256 = sha256_bytes(canonical_json_bytes(selector_seal) + b"\n")
    selected_source_ids = tuple(str(record["source_id"]) for record in runtime_records)
    if len(selected_source_ids) != 24 or len(set(selected_source_ids)) != 24:
        raise ValueError("scientific reducer requires the exact 24 selected sources")
    n8_metrics: list[StateSelectionMetric] = []
    oracle_ratios: dict[str, float] = {
        source_id: 0.0 for source_id in selected_source_ids
    }
    n8_oracles: dict[tuple[str, int], Any] = {}
    zero_oracle_utility_count = 0
    n16_pair_metrics: dict[
        tuple[int, str, str], list[StateSelectionMetric]
    ] = defaultdict(list)
    for trajectory_record in runtime_records:
        source_id = str(trajectory_record["source_id"])
        n8 = trajectory_record["n8"]
        if n8["valid"]:
            event_ids = tuple(n8["candidate_event_step_ids"])
            table = build_sparse_reference_distance_table(
                event_ids,
                _distance_mapping(n8["distance_rows"], label="n8 distance rows"),
                maximum_enumerated_budget=4,
                full_reference_distance=n8["full_reference_distance"],
            )
            state_id = str(n8["state_id"])
            for budget_key, budget in (("budget_2", 2), ("budget_4", 4)):
                for selector in contract.data["selector_matrix"][budget_key]:
                    try:
                        selected = selections[(budget, selector, state_id)]
                    except KeyError as error:
                        raise ValueError("n8 selector decision missing from frozen seal") from error
                    n8_metrics.append(
                        _metric(
                            selector=selector,
                            source_id=source_id,
                            state_id=state_id,
                            horizon_n=8,
                            budget=budget,
                            selected=selected,
                            distance=table.distance(selected),
                            empty_distance=table.distance(()),
                        )
                    )
            for budget in (2, 4):
                n8_oracles[(source_id, budget)] = exact_at_most_b_oracle(
                    table, budget_event_capacity=budget
                )
            oracle = n8_oracles[(source_id, 4)]
            independent = next(
                record
                for record in n8_metrics
                if record.source_id == source_id
                and record.state_id == state_id
                and record.selector_name == "restoration_independent_gate"
                and record.budget_event_capacity == 4
            )
            if oracle.raw_utility <= NORMALIZATION_EPSILON:
                zero_oracle_utility_count += 1
                oracle_ratios[source_id] = 0.0
            else:
                oracle_ratios[source_id] = independent.raw_utility / oracle.raw_utility

        for pair in trajectory_record["n16_pairs"]:
            state_id = str(pair["state_id"])
            budget = int(pair["budget_event_capacity"])
            left_name = str(pair["left_selector_name"])
            right_name = str(pair["right_selector_name"])
            try:
                expected_left = selections[(budget, left_name, state_id)]
                expected_right = selections[(budget, right_name, state_id)]
            except KeyError as error:
                raise ValueError("INVALID_LONG_HORIZON_SELECTOR_JOIN") from error
            if (
                pair["selection_seal_sha256"] != selector_seal_sha256
                or tuple(pair["left_selected_event_step_ids"]) != expected_left
                or tuple(pair["right_selected_event_step_ids"]) != expected_right
            ):
                raise ValueError("INVALID_LONG_HORIZON_SELECTOR_JOIN")
            if not pair["valid"]:
                continue
            table = build_comparator_pair_union_distance_table(
                source_id=source_id,
                state_id=state_id,
                event_ids=tuple(pair["candidate_event_step_ids"]),
                selections={
                    left_name: tuple(pair["left_selected_event_step_ids"]),
                    right_name: tuple(pair["right_selected_event_step_ids"]),
                },
                distances=_distance_mapping(
                    pair["distance_rows"], label="n16 pair distance rows"
                ),
                maximum_selector_budget=budget,
            )
            n16_pair_metrics[(budget, left_name, right_name)].extend(
                evaluate_pair_union_table(table)
            )

    validity = contract.data["validity_and_go_contract"]
    set_config = validity["set_aware_vs_independent_development_go"]
    independent_config = validity["independent_long_horizon_development_go"]
    if (
        set_config.get(
            "overall_set_aware_go_requires_at_least_one_selector_to_pass_all_its_conditions"
        )
        is not True
    ):
        raise ValueError("set-aware overall any-selector reduction contract drifted")

    metric_index = {
        (record.source_id, record.budget_event_capacity, record.selector_name): record
        for record in n8_metrics
    }
    secondary_n8: dict[str, Any] = {}
    for budget_key, budget in (("budget_2", 2), ("budget_4", 4)):
        selector_reports: dict[str, Any] = {}
        valid_sources = {
            source_id
            for source_id in selected_source_ids
            if (source_id, budget) in n8_oracles
        }
        for selector in contract.data["selector_matrix"][budget_key]:
            raw_values: list[float] = []
            normalized_values: list[float] = []
            ratio_values: list[float] = []
            exact_matches = 0
            selected_counts: list[int] = []
            for source_id in selected_source_ids:
                state_id = f"{source_id}:decision_step:010"
                selected = selections[(budget, selector, state_id)]
                selected_counts.append(len(selected))
                metric = metric_index.get((source_id, budget, selector))
                oracle = n8_oracles.get((source_id, budget))
                if metric is None or oracle is None:
                    raw_values.append(0.0)
                    normalized_values.append(0.0)
                    ratio_values.append(0.0)
                    continue
                raw_values.append(metric.raw_utility)
                normalized_values.append(metric.normalized_recovery)
                ratio_values.append(
                    0.0
                    if oracle.raw_utility <= NORMALIZATION_EPSILON
                    else metric.raw_utility / oracle.raw_utility
                )
                exact_matches += selected == oracle.coalition
            selector_reports[selector] = {
                "valid_trajectory_count": len(valid_sources),
                "fixed_selected_trajectory_denominator": len(selected_source_ids),
                "zero_filled_invalid_trajectory_count": len(selected_source_ids)
                - len(valid_sources),
                "mean_raw_utility": statistics.fmean(raw_values),
                "mean_normalized_recovery": statistics.fmean(normalized_values),
                "mean_raw_utility_ratio_to_exact": statistics.fmean(ratio_values),
                "exact_coalition_match_count_among_valid": exact_matches,
                "mean_selected_event_count": statistics.fmean(selected_counts),
                "mean_candidate_compression_ratio": statistics.fmean(
                    count / 8.0 for count in selected_counts
                ),
            }
        oracle_raw = [
            n8_oracles[(source_id, budget)].raw_utility
            if (source_id, budget) in n8_oracles
            else 0.0
            for source_id in selected_source_ids
        ]
        secondary_n8[str(budget)] = {
            "exact_at_most_budget_oracle": {
                "valid_trajectory_count": len(valid_sources),
                "fixed_selected_trajectory_denominator": len(selected_source_ids),
                "mean_raw_utility": statistics.fmean(oracle_raw),
            },
            "selectors": selector_reports,
        }
    secondary_n16: dict[str, Any] = {}
    pair_specs = tuple(
        (2, tuple(pair))
        for pair in contract.data["reference_contract"]["n16"][
            "budget_2_comparator_pairs"
        ]
    ) + tuple(
        (4, tuple(pair))
        for pair in contract.data["reference_contract"]["n16"][
            "budget_4_comparator_pairs"
        ]
    )
    for budget, (left, right) in pair_specs:
        key = (budget, left, right)
        reverse = (budget, right, left)
        if key in n16_pair_metrics or reverse in n16_pair_metrics:
            secondary_n16[f"B{budget}:{left}__vs__{right}"] = _n16_pair_delta(
                n16_pair_metrics,
                budget=budget,
                left_selector=left,
                right_selector=right,
                selected_source_ids=selected_source_ids,
            )
    secondary_descriptive = {
        "go_routing_effect": "REPORT_ONLY",
        "n8": secondary_n8,
        "n16_pair_deltas": secondary_n16,
    }
    if not policy_coverage_passed:
        return {
            "n8_valid_metric_trajectory_count": len(
                {record.source_id for record in n8_metrics}
            ),
            "n16_valid_pair_metric_counts": {
                f"B{budget}:{left}__vs__{right}": len(records) // 2
                for (budget, left, right), records in sorted(n16_pair_metrics.items())
            },
            "set_aware_vs_independent": {
                "status": set_config["failure_outcome"],
                "policy_coverage_condition": False,
                "conditions": {},
            },
            "independent_long_horizon": {
                "status": independent_config["failure_outcome"],
                "policy_coverage_condition": False,
                "conditions": {},
            },
            "secondary_descriptive": secondary_descriptive,
        }

    set_conditions: dict[str, Any] = {}
    for selector in set_config["set_aware_selectors"]:
        n8_report = _paired_condition_report(
            n8_metrics,
            left_selector=selector,
            right_selector=set_config["reference_selector"],
            budget=int(set_config["applies_to_budget"]),
            config=set_config,
            selected_source_ids=selected_source_ids,
        )
        n16_report = _n16_pair_delta(
            n16_pair_metrics,
            budget=int(set_config["applies_to_budget"]),
            left_selector=selector,
            right_selector=set_config["reference_selector"],
            selected_source_ids=selected_source_ids,
        )
        checks = {
            "n8_mean_normalized_delta": n8_report["mean_normalized_delta"]
            >= set_config["n8_trajectory_equal_mean_normalized_delta_minimum"],
            "n8_mean_raw_delta": n8_report["mean_raw_delta"] > 0.0,
            "n8_bootstrap_lower": n8_report["bootstrap"]["lower"] > 0.0,
            "n8_positive_trajectory_count": n8_report["positive_trajectory_count"]
            >= set_config["n8_minimum_positive_trajectories"],
            "n16_mean_normalized_delta": n16_report["mean_normalized_delta"] > 0.0,
        }
        set_conditions[selector] = {
            "n8": n8_report,
            "n16": n16_report,
            "checks": checks,
            "all_conditions_pass": all(checks.values()),
        }
    set_pass = any(
        value["all_conditions_pass"] for value in set_conditions.values()
    )

    ratio_mean = statistics.fmean(oracle_ratios.values())
    independent_conditions: dict[str, Any] = {
        "exact_oracle": {
            "valid_trajectory_count": len(
                {record.source_id for record in n8_metrics}
            ),
            "zero_filled_invalid_trajectory_count": 24
            - len({record.source_id for record in n8_metrics}),
            "zero_oracle_utility_count": zero_oracle_utility_count,
            "mean_raw_utility_ratio": ratio_mean,
            "minimum": independent_config[
                "n8_trajectory_equal_mean_raw_utility_ratio_to_exact_minimum"
            ],
            "passes": ratio_mean
            >= independent_config[
                "n8_trajectory_equal_mean_raw_utility_ratio_to_exact_minimum"
            ],
        },
        "comparators": {},
    }
    for comparator in independent_config["n8_comparators"]:
        n8_report = _paired_condition_report(
            n8_metrics,
            left_selector="restoration_independent_gate",
            right_selector=comparator,
            budget=int(independent_config["primary_budget"]),
            config=independent_config,
            selected_source_ids=selected_source_ids,
        )
        n16_report = _n16_pair_delta(
            n16_pair_metrics,
            budget=int(independent_config["primary_budget"]),
            left_selector="restoration_independent_gate",
            right_selector=comparator,
            selected_source_ids=selected_source_ids,
        )
        checks = {
            "n8_mean_normalized_delta": n8_report["mean_normalized_delta"] > 0.0,
            "n8_mean_raw_delta": n8_report["mean_raw_delta"] > 0.0,
            "n8_bootstrap_lower": n8_report["bootstrap"]["lower"] > 0.0,
            "n8_positive_trajectory_count": n8_report["positive_trajectory_count"]
            >= independent_config["n8_per_comparator_minimum_positive_trajectories"],
            "n16_mean_normalized_delta": n16_report["mean_normalized_delta"] > 0.0,
            "n16_mean_raw_delta": n16_report["mean_raw_delta"] > 0.0,
        }
        independent_conditions["comparators"][comparator] = {
            "n8": n8_report,
            "n16": n16_report,
            "checks": checks,
            "all_conditions_pass": all(checks.values()),
        }
    independent_pass = independent_conditions["exact_oracle"]["passes"] and all(
        value["all_conditions_pass"]
        for value in independent_conditions["comparators"].values()
    )
    return {
        "n8_valid_metric_trajectory_count": len(
            {record.source_id for record in n8_metrics}
        ),
        "n16_valid_pair_metric_counts": {
            f"B{budget}:{left}__vs__{right}": len(records) // 2
            for (budget, left, right), records in sorted(n16_pair_metrics.items())
        },
        "set_aware_vs_independent": {
            "status": (
                "GO_SET_AWARE_LONG_HORIZON_DEVELOPMENT_V1"
                if set_pass
                else set_config["failure_outcome"]
            ),
            "policy_coverage_condition": True,
            "conditions": set_conditions,
        },
        "independent_long_horizon": {
            "status": (
                independent_config["pass_outcome"]
                if independent_pass
                else independent_config["failure_outcome"]
            ),
            "policy_coverage_condition": True,
            "conditions": independent_conditions,
        },
        "secondary_descriptive": secondary_descriptive,
    }


def aggregate_worker_outputs(
    *,
    contract: LongHorizonContract,
    runner_config: Mapping[str, Any],
    runner_config_sha256: str,
    selection_manifest: Mapping[str, Any],
    selector_seal: Mapping[str, Any],
    output_root: str | Path,
    persistent_root: str | Path = "/data",
) -> dict[str, Any]:
    """Validate exactly-once rank coverage and emit a lightweight GO envelope."""
    root = _persistent_output_root(output_root, persistent_root=persistent_root)
    assignments = build_worker_assignments(runner_config)
    selected = selection_manifest["splits"]["development"]["trajectories"]
    expected_source_ids = tuple(str(record["trajectory_id"]) for record in selected)
    observed_ranks: list[int] = []
    worker_results: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for assignment in assignments:
        receipt_payload, receipt = load_json_object(
            _receipt_path(root, assignment), label="worker receipt"
        )
        if receipt_payload != pretty_json_bytes(receipt):
            raise ValueError("worker receipt is not canonical pretty JSON")
        receipts.append(
            validate_worker_receipt(
                receipt,
                assignment=assignment,
                runner_config_sha256=runner_config_sha256,
                output_root=root,
            )
        )
        for rank in assignment.selection_ranks:
            result = _load_existing_result(_result_path(root, assignment, rank))
            worker_results.append(
                _validate_worker_result(
                    result,
                    assignment=assignment,
                    rank=rank,
                    source_id=expected_source_ids[rank],
                    runner_config_sha256=runner_config_sha256,
                    contract=contract,
                )
            )
            observed_ranks.append(rank)
    if sorted(observed_ranks) != list(range(TRAJECTORY_COUNT)) or len(
        set(observed_ranks)
    ) != TRAJECTORY_COUNT:
        raise ValueError("worker shards do not cover the 24 selected ranks exactly once")
    runtime_records = tuple(
        result["runtime_record"]
        for result in sorted(worker_results, key=lambda item: item["selection_rank"])
    )
    if tuple(record["source_id"] for record in runtime_records) != expected_source_ids:
        raise ValueError("aggregated runtime roster differs from selection order")
    totals = _collect_operation_counts(runtime_records)
    accounting = contract.data["operation_accounting"]
    expected_requested = {
        "n8.canonical_action_parse_attempts": accounting[
            "n8_total_requested_canonical_action_parse_attempts"
        ],
        "n8.distance_rows": accounting[
            "n8_budget_4_total_requested_unique_subset_rows"
        ],
        "n16.canonical_action_parse_attempts": accounting[
            "n16_total_requested_pair_reference_parse_attempts"
        ],
    }
    for name, expected in expected_requested.items():
        if expected is None or name not in totals or totals[name]["requested"] != expected:
            raise ValueError(f"aggregate requested operation accounting drifted: {name}")
    n16_distance = totals.get("n16.distance_rows")
    maximum_n16_distance = accounting[
        "n16_total_requested_maximum_nonself_distances"
    ]
    if (
        n16_distance is None
        or n16_distance["requested"] > maximum_n16_distance
        or n16_distance["requested"] < 0
    ):
        raise ValueError("aggregate n16 unique nonself distance accounting drifted")
    generation_attempts = (
        totals["n8.canonical_action_parse_attempts"]["requested"]
        + totals["n16.canonical_action_parse_attempts"]["requested"]
    )
    teacher_forwards = sum(
        totals[name]["requested"]
        for name in (
            "n8.reference_teacher_forwards",
            "n8.candidate_teacher_forwards",
            "n16.reference_teacher_forwards",
            "n16.candidate_teacher_forwards",
        )
    )
    gpu_kl_rows = (
        totals["n8.distance_rows"]["requested"]
        + totals["n16.distance_rows"]["requested"]
    )
    maximum_checks = {
        "generation_attempt_count": {
            "observed": generation_attempts,
            "maximum": accounting["maximum_generation_attempt_count"],
        },
        "teacher_forward_count": {
            "observed": teacher_forwards,
            "maximum": accounting["maximum_teacher_forward_count"],
        },
        "gpu_kl_count": {
            "observed": gpu_kl_rows,
            "maximum": accounting["maximum_gpu_kl_count"],
        },
    }
    if any(
        record["observed"] > record["maximum"]
        for record in maximum_checks.values()
    ):
        raise ValueError("aggregate operation count exceeds a frozen maximum")
    if any(
        result["access_accounting"]
        != {
            "reserve_access_count": 0,
            "top_up_count": 0,
            "replacement_count": 0,
            "post_selection_filter_count": 0,
        }
        for result in worker_results
    ):
        raise PermissionError("aggregate contains a forbidden access or denominator edit")
    coverage_status, n8_valid, n16_valid = _coverage_status(
        runtime_records, contract=contract
    )
    scientific = evaluate_scientific_go(
        runtime_records,
        selector_seal=selector_seal,
        contract=contract,
        policy_coverage_passed=(
            coverage_status == "GO_LONG_HORIZON_POLICY_COVERAGE"
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": SUMMARY_PROTOCOL_ID,
        "status": "COMPLETED_LONG_HORIZON_DEVELOPMENT_AGGREGATION",
        "runner_config_sha256": runner_config_sha256,
        "selected_trajectory_denominator": TRAJECTORY_COUNT,
        "completed_trajectory_count": len(runtime_records),
        "selection_ranks": list(range(TRAJECTORY_COUNT)),
        "source_ids": list(expected_source_ids),
        "worker_receipt_sha256": [
            sha256_bytes(pretty_json_bytes(receipt)) for receipt in receipts
        ],
        "operation_counts": totals,
        "operation_maximum_checks": maximum_checks,
        "coverage": {
            "status": coverage_status,
            "valid_n8_trajectory_count": n8_valid,
            "valid_n16_trajectory_count_by_pair": n16_valid,
        },
        "go_statuses": {
            "policy_coverage": coverage_status,
            "set_aware_vs_independent": scientific[
                "set_aware_vs_independent"
            ]["status"],
            "independent_long_horizon": scientific[
                "independent_long_horizon"
            ]["status"],
        },
        "scientific_evaluation": scientific,
        "access_accounting": {
            "reserve_access_count": 0,
            "top_up_count": 0,
            "replacement_count": 0,
            "post_selection_filter_count": 0,
        },
    }


__all__ = [
    "EXPECTED_RANK_RANGES",
    "LongHorizonWorkerAdapter",
    "PROTOCOL_ID",
    "RUNNER_STATUS",
    "SCHEMA_VERSION",
    "SELECTION_MANIFEST_PATH",
    "SUMMARY_PROTOCOL_ID",
    "TRAJECTORY_COUNT",
    "TRAJECTORIES_PER_WORKER",
    "WORKER_COUNT",
    "WorkerAssignment",
    "aggregate_worker_outputs",
    "atomic_write",
    "build_worker_assignments",
    "inventory_sha256",
    "load_development_trajectories",
    "load_json_object",
    "load_json_object_bytes",
    "pretty_json_bytes",
    "run_worker_shard",
    "selector_records_by_source",
    "sha256_bytes",
    "sha256_file",
    "validate_runner_config",
    "validate_runtime_trajectory_record",
    "validate_selection_manifest_payload",
    "validate_selector_seal_payload",
    "validate_worker_receipt",
]
