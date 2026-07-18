"""Execution-B core for the frozen independent-gate confirm-20 experiment.

Remote downloads, Hugging Face commits, and CUDA runtime construction intentionally
remain injectable.  This module owns the irreversible ordering boundary: selector
decisions are made and durably published before reference generation or D(S) access.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from causalcache.gate_v1_contract import canonical_json_bytes
from causalcache.gate_v1_data import (
    CandidateFeatures,
    FeatureState,
    GateState,
    independent_input,
    label_state_from_restoration_record,
)
from causalcache.gate_v1_formal_train import load_safetensors_checkpoint
from causalcache.gate_v1_provenance import (
    FrozenEnsembleProvenance,
    frozen_ensemble_provenance_from_manifest,
)
from causalcache.gate_v1_training import FittedEnsemble, predict_vectors
from causalcache.independent_confirm_artifact import (
    independent_decisions_bytes,
    read_independent_decisions,
)
from causalcache.independent_confirm_data import (
    CONFIRM_BUDGET_EVENT_CAPACITY,
    CONFIRM_CANDIDATE_EVENT_STEP_IDS,
    CONFIRM_DECISION_STEP_ID,
    CONFIRM_SOURCE_IDS,
    CONFIRM_STATE_COUNT,
    Confirm20PolicyWorkItem,
    ValidatedConfirm20Payloads,
)
from causalcache.independent_confirm_evaluation import evaluate_independent_confirm
from causalcache.policy.gui_owl_v2_1 import (
    build_gui_owl_v2_1_mixed_fidelity_messages,
    parse_gui_owl_v2_1_output,
    validate_gui_owl_v2_1_native_messages,
)
from causalcache.restoration_v2_2_label_table import validate_complete_distance_table


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_independent_confirm20_runner_v1"
VALID_STATE_OUTCOME = "VALID_INDEPENDENT_CONFIRM_STATE"
REFERENCE_FAILURE_OUTCOME = "FAILED_INDEPENDENT_CONFIRM_REFERENCE"
WORKER_COUNT = 4
STATES_PER_WORKER = 5
REFERENCE_REPEAT_KL_MAXIMUM = 1e-4

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_PERSISTED_SEAL_TOKEN = object()
_PAYLOAD_COMMIT_TOKEN = object()
_SELECTION_TOKEN = object()


@dataclass(frozen=True)
class LoadedIndependentEnsemble:
    ensemble: FittedEnsemble
    provenance: FrozenEnsembleProvenance
    checkpoint_load_count: int


class IndependentSelectionBundle:
    """Opaque label-blind ensemble and five-seed decisions."""

    __slots__ = (
        "ensemble_selections",
        "seed_selections",
        "score_records",
        "selection_sha256",
        "_payload",
        "_token",
    )

    def __init__(
        self,
        *,
        ensemble_selections: Mapping[str, tuple[int, ...]],
        seed_selections: Sequence[Mapping[str, tuple[int, ...]]],
        score_records: Sequence[Mapping[str, Any]],
        selection_sha256: str,
        payload: bytes,
        _token: object,
    ) -> None:
        if _token is not _SELECTION_TOKEN:
            raise TypeError("independent selections must come from the frozen scorer")
        self.ensemble_selections = MappingProxyType(dict(ensemble_selections))
        self.seed_selections = tuple(
            MappingProxyType(dict(item)) for item in seed_selections
        )
        self.score_records = tuple(
            MappingProxyType(json.loads(canonical_json_bytes(item)))
            for item in score_records
        )
        self.selection_sha256 = selection_sha256
        self._payload = bytes(payload)
        self._token = _token


class PersistedLabelBlindSeal:
    """Opaque receipt proving exact label-blind bytes were durably persisted."""

    __slots__ = (
        "_files",
        "inventory",
        "inventory_sha256",
        "selection_sha256",
        "witness",
        "_token",
    )

    def __init__(
        self,
        *,
        files: Mapping[str, bytes],
        inventory: Sequence[Mapping[str, Any]],
        inventory_sha256: str,
        selection_sha256: str,
        witness: Mapping[str, Any],
        _token: object,
    ) -> None:
        if _token is not _PERSISTED_SEAL_TOKEN:
            raise TypeError("persisted seals cannot be constructed directly")
        self._files = MappingProxyType({path: bytes(value) for path, value in files.items()})
        self.inventory = tuple(
            MappingProxyType(dict(item)) for item in inventory
        )
        self.inventory_sha256 = inventory_sha256
        self.selection_sha256 = selection_sha256
        self.witness = MappingProxyType(json.loads(canonical_json_bytes(witness)))
        self._token = _token

    @property
    def files(self) -> Mapping[str, bytes]:
        return self._files


class PayloadCommitReceipt:
    """Unforgeable authorization issued only after an exact payload commit replay."""

    __slots__ = (
        "payload_commit",
        "inventory_sha256",
        "selection_sha256",
        "_token",
    )

    def __init__(
        self,
        *,
        payload_commit: str,
        inventory_sha256: str,
        selection_sha256: str,
        _token: object,
    ) -> None:
        if _token is not _PAYLOAD_COMMIT_TOKEN:
            raise TypeError("payload commit receipts cannot be constructed directly")
        self.payload_commit = payload_commit
        self.inventory_sha256 = inventory_sha256
        self.selection_sha256 = selection_sha256
        self._token = _token


@dataclass(frozen=True)
class ConfirmWorkerAssignment:
    worker_id: str
    ordinals: tuple[int, ...]
    state_ids: tuple[str, ...]


@dataclass(frozen=True)
class ConfirmWorkerResult:
    worker_id: str
    state_records: tuple[Mapping[str, Any], ...]


class _FeatureOnlyGateView:
    """Expose exactly the feature fields consumed by independent_input."""

    def __init__(self, feature: FeatureState) -> None:
        self.source_id = feature.source_id
        self.state_id = feature.state_id
        self.decision_step_id = feature.decision_step_id
        self.candidate_event_step_ids = feature.candidate_event_step_ids
        self.q64 = feature.q64
        self.candidates = feature.candidates

    def candidate(self, event_step_id: int) -> CandidateFeatures:
        for candidate in self.candidates:
            if candidate.event_step_id == event_step_id:
                return candidate
        raise KeyError(event_step_id)


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
                ValueError(f"non-finite JSON value in {label}: {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(path: Any) -> str:
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError("payload paths must be canonical relative POSIX paths")
    return path


def load_independent_ensemble(
    model_contract: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    *,
    checkpoint_loader: Callable[..., Any] = load_safetensors_checkpoint,
) -> LoadedIndependentEnsemble:
    """Strictly bind and load the existing five independent formal checkpoints."""
    if not isinstance(model_contract, Mapping) or not isinstance(payloads, Mapping):
        raise TypeError("model contract and payloads must be mappings")
    manifest_record = model_contract.get("ensemble_manifest")
    checkpoints = model_contract.get("checkpoints")
    if not isinstance(manifest_record, Mapping) or not isinstance(checkpoints, list):
        raise ValueError("independent model manifest/checkpoint inventory is malformed")
    if len(checkpoints) != 5:
        raise ValueError("independent model requires exactly five checkpoints")
    expected_paths = {
        _safe_relative(manifest_record.get("path")),
        *(_safe_relative(item.get("path")) for item in checkpoints),
    }
    if set(payloads) != expected_paths or any(
        not isinstance(payload, bytes) or not payload for payload in payloads.values()
    ):
        raise ValueError("independent model payload inventory drifted")

    manifest_path = str(manifest_record["path"])
    manifest_payload = payloads[manifest_path]
    if (
        _sha256(manifest_payload) != manifest_record.get("sha256")
        or _SHA256.fullmatch(str(manifest_record.get("provenance_sha256"))) is None
    ):
        raise ValueError("independent ensemble manifest bytes drifted")
    provenance = frozen_ensemble_provenance_from_manifest(
        _strict_json(manifest_payload, label="independent ensemble manifest")
    )
    if (
        provenance.training.family != "independent"
        or provenance.sha256 != manifest_record["provenance_sha256"]
        or tuple(checkpoint.seed for checkpoint in provenance.checkpoints)
        != tuple(range(5))
    ):
        raise ValueError("independent ensemble provenance drifted")

    repository = model_contract.get("repo")
    payload_commit = model_contract.get("payload_commit")
    models: list[Any] = []
    for seed, raw in enumerate(checkpoints):
        if not isinstance(raw, Mapping) or raw.get("seed") != seed:
            raise ValueError("independent checkpoint seed order drifted")
        path = str(raw["path"])
        payload = payloads[path]
        bound = provenance.checkpoints[seed]
        if (
            len(payload) != raw.get("size_bytes")
            or _sha256(payload) != raw.get("sha256")
            or bound.seed != seed
            or bound.selected_epoch != raw.get("selected_epoch")
            or bound.model_state_sha256 != raw.get("model_state_sha256")
            or bound.checkpoint_artifact.repository != repository
            or bound.checkpoint_artifact.revision != payload_commit
            or bound.checkpoint_artifact.path != path
            or bound.checkpoint_artifact.sha256 != raw.get("sha256")
        ):
            raise ValueError(f"independent checkpoint {seed} binding drifted")
        models.append(
            checkpoint_loader(
                payload,
                family="independent",
                seed=seed,
                expected_model_state_sha256=raw["model_state_sha256"],
                expected_checkpoint_sha256=raw["sha256"],
            )
        )
    ensemble = FittedEnsemble(
        family="independent",
        learning_rate=provenance.training.learning_rate,
        seeds=tuple(range(5)),
        selected_epochs=provenance.training.selected_epochs,
        selection_sha256=provenance.training.oof_selection_sha256,
        models=tuple(models),
    )
    return LoadedIndependentEnsemble(
        ensemble=ensemble,
        provenance=provenance,
        checkpoint_load_count=5,
    )


def _validate_feature_roster(features: Sequence[FeatureState]) -> tuple[FeatureState, ...]:
    if isinstance(features, (str, bytes, bytearray, Mapping)):
        raise TypeError("confirm features must be an ordered sequence")
    result = tuple(features)
    if len(result) != CONFIRM_STATE_COUNT or any(
        not isinstance(feature, FeatureState) for feature in result
    ):
        raise ValueError("independent confirm requires exactly 20 FeatureState values")
    if len({feature.source_id for feature in result}) != CONFIRM_STATE_COUNT or len(
        {feature.state_id for feature in result}
    ) != CONFIRM_STATE_COUNT:
        raise ValueError("confirm feature trajectory/state identities must be unique")
    for ordinal, feature in enumerate(result):
        source_id = CONFIRM_SOURCE_IDS[ordinal]
        if (
            feature.source_id != source_id
            or feature.state_id
            != f"{source_id}:decision_step:{CONFIRM_DECISION_STEP_ID:03d}"
            or feature.decision_step_id != CONFIRM_DECISION_STEP_ID
            or feature.candidate_event_step_ids != CONFIRM_CANDIDATE_EVENT_STEP_IDS
            or tuple(item.event_step_id for item in feature.candidates)
            != CONFIRM_CANDIDATE_EVENT_STEP_IDS
        ):
            raise ValueError("confirm feature geometry drifted from fixed n=4")
    return result


def _positive_top_b(
    event_ids: Sequence[int], scores: Sequence[float]
) -> tuple[int, ...]:
    if len(event_ids) != len(scores) or any(not math.isfinite(score) for score in scores):
        raise ValueError("independent score vector is malformed")
    ranked = sorted(
        ((event, score) for event, score in zip(event_ids, scores, strict=True) if score > 0.0),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(sorted(event for event, _ in ranked[:CONFIRM_BUDGET_EVENT_CAPACITY]))


def score_and_select_independent(
    features: Sequence[FeatureState],
    ensemble: FittedEnsemble,
    *,
    predict_fn: Callable[[Any, Sequence[Sequence[float]]], Sequence[float]] = predict_vectors,
) -> IndependentSelectionBundle:
    """Run each seed once per candidate and seal ensemble plus per-seed choices."""
    frozen = _validate_feature_roster(features)
    if (
        not isinstance(ensemble, FittedEnsemble)
        or ensemble.family != "independent"
        or ensemble.seeds != tuple(range(5))
        or len(ensemble.models) != 5
        or len(ensemble.selected_epochs) != 5
    ):
        raise ValueError("selection requires the frozen five-seed independent ensemble")
    keys: list[tuple[FeatureState, int]] = []
    vectors: list[tuple[float, ...]] = []
    for feature in frozen:
        view = _FeatureOnlyGateView(feature)
        for event in feature.candidate_event_step_ids:
            keys.append((feature, event))
            vectors.append(independent_input(view, event))
    per_seed_values: list[tuple[float, ...]] = []
    for model in ensemble.models:
        raw = tuple(float(value) for value in predict_fn(model, tuple(vectors)))
        if len(raw) != len(keys) or any(not math.isfinite(value) for value in raw):
            raise ValueError("independent checkpoint prediction vector drifted")
        per_seed_values.append(raw)

    ensemble_selections: dict[str, tuple[int, ...]] = {}
    seed_selections = [dict() for _ in range(5)]
    ensemble_scores_by_state: dict[str, dict[int, float]] = {}
    seed_scores_by_state: list[dict[str, dict[int, float]]] = [
        {} for _ in range(5)
    ]
    records: list[dict[str, Any]] = []
    for state_index, feature in enumerate(frozen):
        start = state_index * 4
        seed_rows = [values[start : start + 4] for values in per_seed_values]
        ensemble_scores = tuple(
            math.fsum(seed_rows[seed][event_index] for seed in range(5)) / 5
            for event_index in range(4)
        )
        selected = _positive_top_b(feature.candidate_event_step_ids, ensemble_scores)
        ensemble_selections[feature.state_id] = selected
        ensemble_scores_by_state[feature.state_id] = dict(
            zip(feature.candidate_event_step_ids, ensemble_scores, strict=True)
        )
        seed_payload = []
        for seed, scores in enumerate(seed_rows):
            seed_selected = _positive_top_b(feature.candidate_event_step_ids, scores)
            seed_selections[seed][feature.state_id] = seed_selected
            seed_scores_by_state[seed][feature.state_id] = dict(
                zip(feature.candidate_event_step_ids, scores, strict=True)
            )
            seed_payload.append(
                {
                    "seed": seed,
                    "scores_by_event_step": [
                        {"event_step_id": event, "score": score}
                        for event, score in zip(
                            feature.candidate_event_step_ids, scores, strict=True
                        )
                    ],
                    "selected_event_step_ids": list(seed_selected),
                }
            )
        records.append(
            {
                "ordinal": state_index,
                "source_id": feature.source_id,
                "state_id": feature.state_id,
                "candidate_event_step_ids": list(feature.candidate_event_step_ids),
                "ensemble_scores_by_event_step": [
                    {"event_step_id": event, "score": score}
                    for event, score in zip(
                        feature.candidate_event_step_ids, ensemble_scores, strict=True
                    )
                ],
                "ensemble_selected_event_step_ids": list(selected),
                "seed_decisions": seed_payload,
            }
        )
    payload = independent_decisions_bytes(
        ensemble_scores=ensemble_scores_by_state,
        seed_scores=seed_scores_by_state,
    )
    digest = _sha256(payload)
    return IndependentSelectionBundle(
        ensemble_selections=ensemble_selections,
        seed_selections=seed_selections,
        score_records=records,
        selection_sha256=digest,
        payload=payload,
        _token=_SELECTION_TOKEN,
    )


def independent_selection_payload_bytes(bundle: IndependentSelectionBundle) -> bytes:
    if not isinstance(bundle, IndependentSelectionBundle) or bundle._token is not _SELECTION_TOKEN:
        raise TypeError("selection payload requires an intact independent bundle")
    payload = bytes(bundle._payload)
    if _sha256(payload) != bundle.selection_sha256:
        raise ValueError("independent selection bundle changed after scoring")
    return payload


def replay_independent_selection_bundle(payload: bytes) -> IndependentSelectionBundle:
    """Rebuild an opaque selection bundle from exact sealed decision bytes."""
    if not isinstance(payload, bytes):
        raise TypeError("sealed independent decisions must be bytes")
    replay = read_independent_decisions(payload)
    ensemble_scores = replay["ensemble_scores"]
    seed_scores = replay["seed_scores"]
    canonical = independent_decisions_bytes(
        ensemble_scores=ensemble_scores,
        seed_scores=seed_scores,
    )
    if canonical != payload:
        raise ValueError("sealed independent decisions failed canonical byte replay")

    ensemble_selections: dict[str, tuple[int, ...]] = {}
    seed_selections = [dict() for _ in range(5)]
    records: list[dict[str, Any]] = []
    for ordinal, source_id in enumerate(CONFIRM_SOURCE_IDS):
        state_id = f"{source_id}:decision_step:{CONFIRM_DECISION_STEP_ID:03d}"
        ensemble_values = tuple(
            float(ensemble_scores[state_id][event])
            for event in CONFIRM_CANDIDATE_EVENT_STEP_IDS
        )
        ensemble_selected = _positive_top_b(
            CONFIRM_CANDIDATE_EVENT_STEP_IDS, ensemble_values
        )
        ensemble_selections[state_id] = ensemble_selected
        seed_decisions = []
        for seed in range(5):
            values = tuple(
                float(seed_scores[seed][state_id][event])
                for event in CONFIRM_CANDIDATE_EVENT_STEP_IDS
            )
            selected = _positive_top_b(CONFIRM_CANDIDATE_EVENT_STEP_IDS, values)
            seed_selections[seed][state_id] = selected
            seed_decisions.append(
                {
                    "seed": seed,
                    "scores_by_event_step": [
                        {"event_step_id": event, "score": score}
                        for event, score in zip(
                            CONFIRM_CANDIDATE_EVENT_STEP_IDS, values, strict=True
                        )
                    ],
                    "selected_event_step_ids": list(selected),
                }
            )
        records.append(
            {
                "ordinal": ordinal,
                "source_id": source_id,
                "state_id": state_id,
                "candidate_event_step_ids": list(CONFIRM_CANDIDATE_EVENT_STEP_IDS),
                "ensemble_scores_by_event_step": [
                    {"event_step_id": event, "score": score}
                    for event, score in zip(
                        CONFIRM_CANDIDATE_EVENT_STEP_IDS,
                        ensemble_values,
                        strict=True,
                    )
                ],
                "ensemble_selected_event_step_ids": list(ensemble_selected),
                "seed_decisions": seed_decisions,
            }
        )
    return IndependentSelectionBundle(
        ensemble_selections=ensemble_selections,
        seed_selections=seed_selections,
        score_records=records,
        selection_sha256=_sha256(payload),
        payload=payload,
        _token=_SELECTION_TOKEN,
    )


def _payload_inventory(files: Mapping[str, bytes]) -> tuple[dict[str, Any], ...]:
    if not isinstance(files, Mapping) or not files:
        raise ValueError("label-blind payload must contain at least one file")
    records = []
    for path in sorted(files):
        safe = _safe_relative(path)
        payload = files[path]
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("label-blind payload values must be non-empty bytes")
        records.append({"path": safe, "sha256": _sha256(payload), "size_bytes": len(payload)})
    return tuple(records)


def persist_and_seal_label_blind_payload(
    files: Mapping[str, bytes],
    *,
    independent_selections: IndependentSelectionBundle,
    independent_selection_path: str,
    persist_fn: Callable[[Mapping[str, bytes], Sequence[Mapping[str, Any]]], Mapping[str, Any]],
) -> PersistedLabelBlindSeal:
    """Persist exact bytes before issuing any reference/restoration capability."""
    if (
        not isinstance(independent_selections, IndependentSelectionBundle)
        or independent_selections._token is not _SELECTION_TOKEN
    ):
        raise TypeError("label-blind persistence requires sealed independent decisions")
    selection_path = _safe_relative(independent_selection_path)
    copied = {path: bytes(payload) for path, payload in files.items()}
    expected_selection_payload = independent_selection_payload_bytes(
        independent_selections
    )
    if copied.get(selection_path) != expected_selection_payload:
        raise ValueError(
            "label-blind payload does not contain the exact sealed independent decisions"
        )
    inventory = _payload_inventory(copied)
    inventory_sha256 = _sha256(canonical_json_bytes(list(inventory)))
    witness = persist_fn(MappingProxyType(copied), inventory)
    if (
        not isinstance(witness, Mapping)
        or witness.get("durable") is not True
        or witness.get("inventory_sha256") != inventory_sha256
        or witness.get("file_count") != len(inventory)
        or not isinstance(witness.get("persistence_id"), str)
        or not witness["persistence_id"]
    ):
        raise ValueError("durable label-blind persistence witness is malformed")
    return PersistedLabelBlindSeal(
        files=copied,
        inventory=inventory,
        inventory_sha256=inventory_sha256,
        selection_sha256=independent_selections.selection_sha256,
        witness=witness,
        _token=_PERSISTED_SEAL_TOKEN,
    )


def publish_payload_commit(
    seal: PersistedLabelBlindSeal,
    *,
    publish_fn: Callable[[Mapping[str, bytes], Sequence[Mapping[str, Any]]], Mapping[str, Any]],
    verify_fn: Callable[[str, Sequence[Mapping[str, Any]]], bool],
) -> PayloadCommitReceipt:
    """Publish and independently replay the exact persisted, label-blind inventory."""
    if (
        not isinstance(seal, PersistedLabelBlindSeal)
        or seal._token is not _PERSISTED_SEAL_TOKEN
        or _payload_inventory(seal.files) != tuple(dict(item) for item in seal.inventory)
    ):
        raise ValueError("payload publication requires an intact persisted seal")
    inventory = tuple(dict(item) for item in seal.inventory)
    response = publish_fn(seal.files, inventory)
    commit = response.get("payload_commit") if isinstance(response, Mapping) else None
    if (
        not isinstance(commit, str)
        or _COMMIT.fullmatch(commit) is None
        or response.get("inventory_sha256") != seal.inventory_sha256
        or verify_fn(commit, inventory) is not True
    ):
        raise ValueError("payload commit response or fresh replay verification failed")
    return PayloadCommitReceipt(
        payload_commit=commit,
        inventory_sha256=seal.inventory_sha256,
        selection_sha256=seal.selection_sha256,
        _token=_PAYLOAD_COMMIT_TOKEN,
    )


def authorize_adopted_payload_commit(
    seal: PersistedLabelBlindSeal,
    payload_commit: str,
    verify_fn: Callable[[str, Sequence[Mapping[str, Any]]], bool],
) -> PayloadCommitReceipt:
    """Authorize a read-only adopted commit after exact sealed-inventory replay."""
    if (
        not isinstance(seal, PersistedLabelBlindSeal)
        or seal._token is not _PERSISTED_SEAL_TOKEN
        or _payload_inventory(seal.files) != tuple(dict(item) for item in seal.inventory)
    ):
        raise ValueError("payload adoption requires an intact persisted seal")
    inventory = tuple(dict(item) for item in seal.inventory)
    witness = seal.witness
    inventory_sha256 = seal.inventory_sha256
    selection_sha256 = seal.selection_sha256
    if (
        _sha256(canonical_json_bytes(list(inventory))) != inventory_sha256
        or _SHA256.fullmatch(selection_sha256) is None
        or sum(item["sha256"] == selection_sha256 for item in inventory) != 1
        or not isinstance(witness, Mapping)
        or witness.get("durable") is not True
        or witness.get("file_count") != len(inventory)
        or witness.get("inventory_sha256") != inventory_sha256
        or not isinstance(witness.get("persistence_id"), str)
        or not witness["persistence_id"]
        or not isinstance(payload_commit, str)
        or _COMMIT.fullmatch(payload_commit) is None
    ):
        raise ValueError("adopted payload commit or persisted seal binding is invalid")
    verification_inventory = tuple(dict(item) for item in inventory)
    if (
        verify_fn(payload_commit, verification_inventory) is not True
        or verification_inventory != inventory
    ):
        raise ValueError("adopted payload commit fresh replay verification failed")
    return PayloadCommitReceipt(
        payload_commit=payload_commit,
        inventory_sha256=inventory_sha256,
        selection_sha256=selection_sha256,
        _token=_PAYLOAD_COMMIT_TOKEN,
    )


def _require_payload_receipt(receipt: PayloadCommitReceipt) -> None:
    if (
        not isinstance(receipt, PayloadCommitReceipt)
        or receipt._token is not _PAYLOAD_COMMIT_TOKEN
        or _COMMIT.fullmatch(receipt.payload_commit) is None
        or _SHA256.fullmatch(receipt.inventory_sha256) is None
        or _SHA256.fullmatch(receipt.selection_sha256) is None
    ):
        raise PermissionError(
            "reference generation requires a verified label-blind payload commit receipt"
        )


def build_confirm_messages(
    payloads: ValidatedConfirm20Payloads,
    work_item: Confirm20PolicyWorkItem,
    *,
    restored_event_step_ids: Sequence[int],
    image_decoder: Callable[[bytes], Any],
) -> list[dict[str, Any]]:
    """Build one strict n=4 GUI-Owl v2.1 mixed-fidelity confirm prompt."""
    if not isinstance(payloads, ValidatedConfirm20Payloads):
        raise TypeError("confirm messages require validated confirm-20 payloads")
    if not isinstance(work_item, Confirm20PolicyWorkItem):
        raise TypeError("confirm messages require a Confirm20PolicyWorkItem")
    coalition = tuple(restored_event_step_ids)
    if (
        coalition != tuple(sorted(coalition))
        or len(coalition) != len(set(coalition))
        or not set(coalition).issubset(CONFIRM_CANDIDATE_EVENT_STEP_IDS)
    ):
        raise ValueError("confirm restoration coalition is not canonical n=4 geometry")
    matching = [
        trajectory
        for trajectory in payloads.trajectories
        if trajectory.get("source_id") == work_item.source_id
    ]
    if (
        len(matching) != 1
        or work_item.decision_step_id != CONFIRM_DECISION_STEP_ID
        or work_item.candidate_event_step_ids != CONFIRM_CANDIDATE_EVENT_STEP_IDS
        or work_item.state_id
        != f"{work_item.source_id}:decision_step:{CONFIRM_DECISION_STEP_ID:03d}"
    ):
        raise ValueError("confirm work item does not bind one frozen trajectory")
    manifest = {"trajectories": [matching[0]]}
    messages = build_gui_owl_v2_1_mixed_fidelity_messages(
        manifest,
        trajectory_id=work_item.source_id,
        decision_step_id=work_item.decision_step_id,
        restored_event_step_ids=coalition,
        image_bytes_loader=lambda path: payloads.image_payloads_by_path[path],
        image_decoder=image_decoder,
    )
    image_count = validate_gui_owl_v2_1_native_messages(messages)
    if image_count != 1 + len(coalition):
        raise ValueError("confirm prompt image count differs from restored coalition")
    return messages


def bind_confirm_message_builder(
    payloads: ValidatedConfirm20Payloads,
    *,
    image_decoder: Callable[[bytes], Any],
) -> Callable[[Confirm20PolicyWorkItem, tuple[int, ...]], list[dict[str, Any]]]:
    return lambda item, coalition: build_confirm_messages(
        payloads,
        item,
        restored_event_step_ids=coalition,
        image_decoder=image_decoder,
    )


def _coalitions_by_mask(event_ids: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(event for bit, event in enumerate(event_ids) if mask & (1 << bit))
        for mask in range(1 << len(event_ids))
    )


def _coalition_mask(event_ids: Sequence[int], coalition: Sequence[int]) -> int:
    restored = set(coalition)
    return sum(1 << bit for bit, event in enumerate(event_ids) if event in restored)


def _message_inventory(
    messages: Sequence[Mapping[str, Any]],
    *,
    state_id: str,
    coalition: tuple[int, ...],
) -> dict[str, Any]:
    image_count = validate_gui_owl_v2_1_native_messages(messages)
    if image_count != 1 + len(coalition):
        raise ValueError("mixed-fidelity message image inventory drifted")
    text = [
        {"role": message["role"], "text": block["text"]}
        for message in messages
        for block in message["content"]
        if block["type"] == "text"
    ]
    return {
        "state_id": state_id,
        "coalition_mask": _coalition_mask(CONFIRM_CANDIDATE_EVENT_STEP_IDS, coalition),
        "coalition": list(coalition),
        "image_count": image_count,
        "policy_visible_text_sha256": _sha256(canonical_json_bytes(text)),
    }


def _base_operation_counts() -> dict[str, int]:
    return {
        "generation_call_count": 0,
        "reference_teacher_forward_count": 0,
        "reference_repeat_teacher_forward_count": 0,
        "non_reference_coalition_teacher_forward_count": 0,
        "teacher_forward_call_count": 0,
        "teacher_forward_example_count": 0,
        "reference_repeat_kl_measurement_count": 0,
        "coalition_kl_measurement_count": 0,
        "kl_measurement_count": 0,
        "raw_distance_row_count": 0,
        "retry_count": 0,
        "top_up_count": 0,
        "filter_count": 0,
    }


def _generation_evidence(runtime: Any, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        generation = runtime.generate_native_action(messages)
    except ValueError as error:
        return {
            "output_text": getattr(error, "output_text", None),
            "metadata": dict(getattr(error, "metadata", {})),
            "canonical_action": None,
            "parse_error_type": error.__class__.__name__,
            "parse_error_message": str(error),
        }
    output = getattr(generation, "output_text", None)
    metadata = getattr(generation, "metadata", None)
    if not isinstance(output, str) or not isinstance(metadata, Mapping):
        raise RuntimeError("reference generation lost native text or metadata")
    try:
        parsed = parse_gui_owl_v2_1_output(output)
    except (TypeError, ValueError) as error:
        return {
            "output_text": output,
            "metadata": dict(metadata),
            "canonical_action": None,
            "parse_error_type": error.__class__.__name__,
            "parse_error_message": str(error),
        }
    runtime_parsed = getattr(generation, "parsed_output", None)
    runtime_action = getattr(runtime_parsed, "canonical_action", None)
    if runtime_action != parsed.canonical_action:
        raise RuntimeError("runtime and independent parser canonical actions differ")
    return {
        "output_text": output,
        "metadata": dict(metadata),
        "canonical_action": parsed.canonical_action.arguments(),
        "parse_error_type": None,
        "parse_error_message": None,
        "_action": parsed.canonical_action,
    }


def _measurement(measurement: Any, *, label: str) -> tuple[float, Mapping[str, Any]]:
    value = float(getattr(measurement, "value", math.nan))
    audit = getattr(measurement, "audit", None)
    if not math.isfinite(value) or value < 0.0 or not isinstance(audit, Mapping):
        raise RuntimeError(f"{label} KL measurement is invalid")
    if audit.get("full_tensor_host_transfers", 0) != 0:
        raise RuntimeError(f"{label} KL measurement transferred a full logits tensor")
    return value, dict(audit)


def run_confirm_state_once(
    *,
    work_item: Confirm20PolicyWorkItem,
    message_builder: Callable[
        [Confirm20PolicyWorkItem, tuple[int, ...]], Sequence[Mapping[str, Any]]
    ],
    runtime: Any,
    distance_backend: Any,
    payload_commit_receipt: PayloadCommitReceipt,
    run_contract_sha256: str,
) -> dict[str, Any]:
    """Run exactly two generations and the complete n=4 D(S) table, without retry."""
    _require_payload_receipt(payload_commit_receipt)
    if (
        not isinstance(work_item, Confirm20PolicyWorkItem)
        or work_item.candidate_event_step_ids != CONFIRM_CANDIDATE_EVENT_STEP_IDS
        or work_item.decision_step_id != CONFIRM_DECISION_STEP_ID
        or type(work_item.ordinal) is not int
        or not 0 <= work_item.ordinal < CONFIRM_STATE_COUNT
        or _SHA256.fullmatch(run_contract_sha256) is None
    ):
        raise ValueError("confirm state identity or run-contract digest is invalid")
    started = time.perf_counter()
    event_ids = work_item.candidate_event_step_ids
    full = tuple(event_ids)
    full_messages = tuple(message_builder(work_item, full))
    full_inventory = _message_inventory(
        full_messages, state_id=work_item.state_id, coalition=full
    )
    operations = _base_operation_counts()
    generations = []
    for repeat in (1, 2):
        operations["generation_call_count"] += 1
        evidence = _generation_evidence(runtime, full_messages)
        evidence["repeat_index"] = repeat
        generations.append(evidence)

    public_generations = [
        {key: value for key, value in record.items() if key != "_action"}
        for record in generations
    ]
    failure = None
    if any(record["canonical_action"] is None for record in generations):
        failure = {
            "stage": "reference_generation",
            "category": "PARSE_FAILURE",
            "message": "one or more of two reference generations failed strict parsing",
        }
    elif generations[0]["canonical_action"] != generations[1]["canonical_action"]:
        failure = {
            "stage": "reference_generation_repeat_comparison",
            "category": "CANONICAL_ACTION_MISMATCH",
            "message": "two deterministic generations produced different canonical actions",
        }
    state_projection = {
        "ordinal": work_item.ordinal,
        "source_id": work_item.source_id,
        "state_id": work_item.state_id,
        "decision_step_id": work_item.decision_step_id,
        "candidate_event_step_ids": list(event_ids),
    }
    if failure is not None:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "run_contract_sha256": run_contract_sha256,
            "payload_commit": payload_commit_receipt.payload_commit,
            "selection_sha256": payload_commit_receipt.selection_sha256,
            "outcome": REFERENCE_FAILURE_OUTCOME,
            "state": state_projection,
            "failure": failure,
            "reference_generations": public_generations,
            "canonical_action": None,
            "reference_repeat_kl": None,
            "distance_rows": [],
            "operation_counts": operations,
            "duration_seconds": time.perf_counter() - started,
        }

    canonical_action = generations[0]["_action"]
    reference_logits, reference_metadata = runtime.teacher_forced_distance_logits(
        (full_messages,), (canonical_action,)
    )
    operations["reference_teacher_forward_count"] = 1
    operations["teacher_forward_call_count"] += 1
    operations["teacher_forward_example_count"] += 1
    reference_log_probs = distance_backend.prepare_reference(reference_logits)
    del reference_logits

    repeat_logits, repeat_metadata = runtime.teacher_forced_distance_logits(
        (full_messages,), (canonical_action,)
    )
    operations["reference_repeat_teacher_forward_count"] = 1
    operations["teacher_forward_call_count"] += 1
    operations["teacher_forward_example_count"] += 1
    repeat_value, repeat_audit = _measurement(
        distance_backend.measure(reference_log_probs, repeat_logits),
        label="reference repeat",
    )
    del repeat_logits
    operations["reference_repeat_kl_measurement_count"] = 1
    operations["kl_measurement_count"] += 1
    if repeat_value > REFERENCE_REPEAT_KL_MAXIMUM:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "run_contract_sha256": run_contract_sha256,
            "payload_commit": payload_commit_receipt.payload_commit,
            "selection_sha256": payload_commit_receipt.selection_sha256,
            "outcome": REFERENCE_FAILURE_OUTCOME,
            "state": state_projection,
            "failure": {
                "stage": "reference_repeat_teacher_comparison",
                "category": "REFERENCE_REPEAT_KL_EXCEEDED",
                "message": "reference repeat KL exceeds the frozen 1e-4 maximum",
            },
            "reference_generations": public_generations,
            "canonical_action": generations[0]["canonical_action"],
            "reference_teacher_metadata": dict(reference_metadata),
            "reference_repeat_teacher_metadata": dict(repeat_metadata),
            "reference_repeat_kl": repeat_value,
            "reference_repeat_distance_audit": repeat_audit,
            "distance_rows": [],
            "operation_counts": operations,
            "duration_seconds": time.perf_counter() - started,
        }

    distances: dict[frozenset[int], float] = {frozenset(full): 0.0}
    evidence_by_coalition: dict[tuple[int, ...], dict[str, Any]] = {
        full: {
            "prompt_inventory": full_inventory,
            "teacher_metadata": dict(reference_metadata),
            "distance_audit": {
                "reference_identity_distance": True,
                "full_tensor_host_transfers": 0,
            },
        }
    }
    for coalition in _coalitions_by_mask(event_ids):
        if coalition == full:
            continue
        messages = tuple(message_builder(work_item, coalition))
        inventory = _message_inventory(
            messages, state_id=work_item.state_id, coalition=coalition
        )
        logits, metadata = runtime.teacher_forced_distance_logits(
            (messages,), (canonical_action,)
        )
        operations["non_reference_coalition_teacher_forward_count"] += 1
        operations["teacher_forward_call_count"] += 1
        operations["teacher_forward_example_count"] += 1
        value, audit = _measurement(
            distance_backend.measure(reference_log_probs, logits),
            label=f"coalition {coalition}",
        )
        del logits
        operations["coalition_kl_measurement_count"] += 1
        operations["kl_measurement_count"] += 1
        distances[frozenset(coalition)] = value
        evidence_by_coalition[coalition] = {
            "prompt_inventory": inventory,
            "teacher_metadata": dict(metadata),
            "distance_audit": audit,
        }
    del reference_log_probs

    table = validate_complete_distance_table(event_ids, distances)
    rows = []
    for row in table.rows:
        evidence = evidence_by_coalition[row.coalition]
        rows.append(
            {
                "coalition_mask": _coalition_mask(event_ids, row.coalition),
                "coalition": list(row.coalition),
                "distance": row.distance,
                "utility": row.utility,
                **evidence,
            }
        )
    operations["raw_distance_row_count"] = len(rows)
    if operations != {
        "generation_call_count": 2,
        "reference_teacher_forward_count": 1,
        "reference_repeat_teacher_forward_count": 1,
        "non_reference_coalition_teacher_forward_count": 15,
        "teacher_forward_call_count": 17,
        "teacher_forward_example_count": 17,
        "reference_repeat_kl_measurement_count": 1,
        "coalition_kl_measurement_count": 15,
        "kl_measurement_count": 16,
        "raw_distance_row_count": 16,
        "retry_count": 0,
        "top_up_count": 0,
        "filter_count": 0,
    }:
        raise RuntimeError("confirm state operation counts drifted from 2+17 schedule")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_contract_sha256": run_contract_sha256,
        "payload_commit": payload_commit_receipt.payload_commit,
        "selection_sha256": payload_commit_receipt.selection_sha256,
        "outcome": VALID_STATE_OUTCOME,
        "state": state_projection,
        "failure": None,
        "reference_generations": public_generations,
        "canonical_action": generations[0]["canonical_action"],
        "reference_teacher_metadata": dict(reference_metadata),
        "reference_repeat_teacher_metadata": dict(repeat_metadata),
        "reference_repeat_kl": repeat_value,
        "reference_repeat_distance_audit": repeat_audit,
        "distance_rows": rows,
        "operation_counts": operations,
        "duration_seconds": time.perf_counter() - started,
    }


def confirm_worker_assignments(
    work_items: Sequence[Confirm20PolicyWorkItem],
) -> tuple[ConfirmWorkerAssignment, ...]:
    """Freeze the round-robin four-worker, five-state assignment."""
    items = tuple(work_items)
    if (
        len(items) != CONFIRM_STATE_COUNT
        or tuple(item.ordinal for item in items) != tuple(range(CONFIRM_STATE_COUNT))
        or tuple(item.source_id for item in items) != CONFIRM_SOURCE_IDS
        or len({item.state_id for item in items}) != CONFIRM_STATE_COUNT
        or any(item.candidate_event_step_ids != CONFIRM_CANDIDATE_EVENT_STEP_IDS for item in items)
    ):
        raise ValueError("confirm work-item order or denominator drifted")
    result = []
    for worker in range(WORKER_COUNT):
        assigned = tuple(item for item in items if item.ordinal % WORKER_COUNT == worker)
        if len(assigned) != STATES_PER_WORKER:
            raise RuntimeError("confirm worker did not receive exactly five states")
        result.append(
            ConfirmWorkerAssignment(
                worker_id=f"worker-{worker}",
                ordinals=tuple(item.ordinal for item in assigned),
                state_ids=tuple(item.state_id for item in assigned),
            )
        )
    return tuple(result)


def join_confirm_worker_results(
    assignments: Sequence[ConfirmWorkerAssignment],
    results: Sequence[ConfirmWorkerResult],
) -> tuple[Mapping[str, Any], ...]:
    """Barrier: return ordered records only after all four five-state workers finish."""
    frozen_assignments = tuple(assignments)
    frozen_results = tuple(results)
    expected_ids = tuple(f"worker-{index}" for index in range(WORKER_COUNT))
    if (
        len(frozen_assignments) != WORKER_COUNT
        or tuple(item.worker_id for item in frozen_assignments) != expected_ids
        or len(frozen_results) != WORKER_COUNT
        or {item.worker_id for item in frozen_results} != set(expected_ids)
    ):
        raise ValueError("confirm aggregation barrier requires all four workers")
    result_by_id = {result.worker_id: result for result in frozen_results}
    ordered: dict[int, Mapping[str, Any]] = {}
    for assignment in frozen_assignments:
        result = result_by_id[assignment.worker_id]
        if len(result.state_records) != STATES_PER_WORKER:
            raise ValueError("confirm worker result does not contain exactly five states")
        for expected_ordinal, expected_state, record in zip(
            assignment.ordinals,
            assignment.state_ids,
            result.state_records,
            strict=True,
        ):
            state = record.get("state") if isinstance(record, Mapping) else None
            if (
                not isinstance(state, Mapping)
                or state.get("ordinal") != expected_ordinal
                or state.get("state_id") != expected_state
                or expected_ordinal in ordered
            ):
                raise ValueError("confirm worker emitted an unassigned or duplicate state")
            ordered[expected_ordinal] = record
    if set(ordered) != set(range(CONFIRM_STATE_COUNT)):
        raise ValueError("confirm worker barrier did not cover the fixed denominator")
    return tuple(ordered[index] for index in range(CONFIRM_STATE_COUNT))


def _validate_record_operation_counts(record: Mapping[str, Any]) -> None:
    counts = record.get("operation_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("confirm state lacks operation counts")
    success = record.get("outcome") == VALID_STATE_OUTCOME
    failure = record.get("failure")
    repeat_kl_failure = (
        record.get("outcome") == REFERENCE_FAILURE_OUTCOME
        and isinstance(failure, Mapping)
        and failure.get("category") == "REFERENCE_REPEAT_KL_EXCEEDED"
    )
    expected = {
        "generation_call_count": 2,
        "reference_teacher_forward_count": int(success or repeat_kl_failure),
        "reference_repeat_teacher_forward_count": int(success or repeat_kl_failure),
        "non_reference_coalition_teacher_forward_count": 15 if success else 0,
        "teacher_forward_call_count": 17 if success else (2 if repeat_kl_failure else 0),
        "teacher_forward_example_count": 17 if success else (2 if repeat_kl_failure else 0),
        "reference_repeat_kl_measurement_count": int(success or repeat_kl_failure),
        "coalition_kl_measurement_count": 15 if success else 0,
        "kl_measurement_count": 16 if success else int(repeat_kl_failure),
        "raw_distance_row_count": 16 if success else 0,
        "retry_count": 0,
        "top_up_count": 0,
        "filter_count": 0,
    }
    if dict(counts) != expected:
        raise ValueError("confirm state operation counts are not the frozen schedule")


def aggregate_independent_confirm(
    features: Sequence[FeatureState],
    state_records: Sequence[Mapping[str, Any]],
    *,
    independent_selections: IndependentSelectionBundle,
    heuristic_selections: Mapping[str, Mapping[str, Sequence[int]]],
    evaluator: Callable[..., Mapping[str, Any]] = evaluate_independent_confirm,
) -> dict[str, Any]:
    """Locally join fixed-denominator D(S) records and call the frozen evaluator."""
    frozen_features = _validate_feature_roster(features)
    records = tuple(state_records)
    if (
        len(records) != CONFIRM_STATE_COUNT
        or not isinstance(independent_selections, IndependentSelectionBundle)
        or independent_selections._token is not _SELECTION_TOKEN
    ):
        raise ValueError("confirm aggregation requires 20 records and sealed decisions")
    labels = []
    reference_failures = 0
    operation_totals = {key: 0 for key in _base_operation_counts()}
    for ordinal, (feature, record) in enumerate(zip(frozen_features, records, strict=True)):
        state = record.get("state") if isinstance(record, Mapping) else None
        outcome = record.get("outcome") if isinstance(record, Mapping) else None
        if (
            not isinstance(state, Mapping)
            or state.get("ordinal") != ordinal
            or state.get("source_id") != feature.source_id
            or state.get("state_id") != feature.state_id
            or outcome not in {VALID_STATE_OUTCOME, REFERENCE_FAILURE_OUTCOME}
            or record.get("selection_sha256")
            != independent_selections.selection_sha256
        ):
            raise ValueError("confirm result identity/order/outcome drifted")
        _validate_record_operation_counts(record)
        for key, value in record["operation_counts"].items():
            operation_totals[key] += int(value)
        if outcome == REFERENCE_FAILURE_OUTCOME:
            reference_failures += 1
        else:
            labels.append(label_state_from_restoration_record(record, record_schema="expansion"))

    if reference_failures:
        report = evaluator(
            (),
            independent_ensemble_selections=independent_selections.ensemble_selections,
            independent_seed_selections=independent_selections.seed_selections,
            heuristic_selections=heuristic_selections,
            reference_failure_count=reference_failures,
        )
    else:
        label_by_state = {label.state_id: label for label in labels}
        states = []
        for feature in frozen_features:
            label = label_by_state.get(feature.state_id)
            if (
                label is None
                or label.source_id != feature.source_id
                or label.decision_step_id != feature.decision_step_id
                or label.table.event_ids != feature.candidate_event_step_ids
            ):
                raise ValueError("confirm feature/label join geometry drifted")
            states.append(
                GateState(
                    source_id=feature.source_id,
                    state_id=feature.state_id,
                    decision_step_id=feature.decision_step_id,
                    candidate_event_step_ids=feature.candidate_event_step_ids,
                    q64=feature.q64,
                    candidates=feature.candidates,
                    table=label.table,
                )
            )
        report = evaluator(
            tuple(states),
            independent_ensemble_selections=independent_selections.ensemble_selections,
            independent_seed_selections=independent_selections.seed_selections,
            heuristic_selections=heuristic_selections,
            reference_failure_count=0,
        )
    if not isinstance(report, Mapping):
        raise TypeError("independent confirm evaluator must return a mapping")
    return {
        **dict(report),
        "execution": {
            "fixed_state_denominator": CONFIRM_STATE_COUNT,
            "reference_failure_count": reference_failures,
            "operation_counts": operation_totals,
            "retry_count": 0,
            "top_up_count": 0,
            "filter_count": 0,
        },
    }


def aggregate_independent_confirm_workers(
    features: Sequence[FeatureState],
    *,
    assignments: Sequence[ConfirmWorkerAssignment],
    worker_results: Sequence[ConfirmWorkerResult],
    independent_selections: IndependentSelectionBundle,
    heuristic_selections: Mapping[str, Mapping[str, Sequence[int]]],
    evaluator: Callable[..., Mapping[str, Any]] = evaluate_independent_confirm,
) -> dict[str, Any]:
    records = join_confirm_worker_results(assignments, worker_results)
    return aggregate_independent_confirm(
        features,
        records,
        independent_selections=independent_selections,
        heuristic_selections=heuristic_selections,
        evaluator=evaluator,
    )


__all__ = [
    "ConfirmWorkerAssignment",
    "ConfirmWorkerResult",
    "IndependentSelectionBundle",
    "LoadedIndependentEnsemble",
    "PayloadCommitReceipt",
    "PersistedLabelBlindSeal",
    "REFERENCE_FAILURE_OUTCOME",
    "VALID_STATE_OUTCOME",
    "aggregate_independent_confirm",
    "aggregate_independent_confirm_workers",
    "authorize_adopted_payload_commit",
    "bind_confirm_message_builder",
    "build_confirm_messages",
    "confirm_worker_assignments",
    "independent_selection_payload_bytes",
    "join_confirm_worker_results",
    "load_independent_ensemble",
    "persist_and_seal_label_blind_payload",
    "publish_payload_commit",
    "replay_independent_selection_bundle",
    "run_confirm_state_once",
    "score_and_select_independent",
]
