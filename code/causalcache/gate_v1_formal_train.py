"""Train-only orchestration and deterministic artifacts for formal gate v1."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.gate_v1_contract import canonical_json_bytes, sha256_bytes
from causalcache.gate_v1_data import GateState, join_feature_and_label_states
from causalcache.gate_v1_formal_cache import read_feature_cache, read_label_cache
from causalcache.gate_v1_provenance import canonical_model_state_sha256
from causalcache.gate_v1_training import (
    FAMILIES,
    SEEDS,
    FamilySelection,
    FittedEnsemble,
    build_model,
    family_selection_payload,
    family_selection_sha256,
    fit_final_ensemble,
    run_formal_oof,
    validate_formal_training_roster,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_gate_v1_formal_train_v1"
FAMILY_REPORT_STATUS = "COMPLETED_GATE_V1_FORMAL_TRAIN_FAMILY_V1"
RUN_STATUS = "COMPLETED_GATE_V1_FORMAL_TRAIN_V1"
FEATURE_CACHE_SHA256 = (
    "81fded50c4450700220742d3e0be9a5585d1bc51086150515b463bbdf4b4df8e"
)
LABEL_CACHE_SHA256 = (
    "4f9ef172aaa94c3ea8ce53aa43336c9fcb7800c24e181e1462d8239e31053cee"
)
CACHE_BUNDLE_MANIFEST_SHA256 = (
    "15c8bf56ddad4f6f278599c32aaadcd813e0e016db0d523b4892eb47f8d9d144"
)
FORMAL_JOIN_AUDIT_SHA256 = (
    "551e70b7e99f7a761f9c50d2adae3be72b0933044982a015a65e93372e41d77b"
)
FORMAL_SOURCE_IDS_SHA256 = (
    "ccec55afb0882e602f5d2c83ec420682488e2663fe26a7c574a851825c43f225"
)
GATE_PREREGISTRATION_SHA256 = (
    "37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b"
)
CACHE_REPOSITORY = (
    "gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile"
)
CACHE_REVISION = "a61b31bf2e69be00f94469f4a2f2d6b336fcc386"
CACHE_TAG = "gate-v1-formal58-cache-transport-repair-v1"
FEATURE_CACHE_PATH = "formal58-transport-repair/v1/feature-cache-v1.tar"
LABEL_CACHE_PATH = "formal58-transport-repair/v1/label-cache-v1.tar"
CACHE_BUNDLE_MANIFEST_PATH = (
    "formal58-transport-repair/v1/cache-bundle-manifest-v1.json"
)


@dataclass(frozen=True)
class FormalTrainingInputs:
    states: tuple[GateState, ...]
    source_ids: tuple[str, ...]
    folds: tuple[tuple[str, ...], ...]
    feature_cache_sha256: str
    label_cache_sha256: str
    join_audit_sha256: str

    def binding_payload(self) -> dict[str, Any]:
        return {
            "gate_preregistration_sha256": GATE_PREREGISTRATION_SHA256,
            "formal_source_ids_sha256": FORMAL_SOURCE_IDS_SHA256,
            "cache": {
                "repository": CACHE_REPOSITORY,
                "revision": CACHE_REVISION,
                "tag": CACHE_TAG,
                "feature": {
                    "path": FEATURE_CACHE_PATH,
                    "sha256": self.feature_cache_sha256,
                },
                "label": {
                    "path": LABEL_CACHE_PATH,
                    "sha256": self.label_cache_sha256,
                },
                "bundle_manifest": {
                    "path": CACHE_BUNDLE_MANIFEST_PATH,
                    "sha256": CACHE_BUNDLE_MANIFEST_SHA256,
                },
                "join_audit_sha256": self.join_audit_sha256,
            },
            "trajectory_count": len(self.source_ids),
            "state_count": len(self.states),
            "fold_sizes": [len(fold) for fold in self.folds],
        }


@dataclass(frozen=True)
class CheckpointArtifact:
    family: str
    seed: int
    selected_epoch: int
    model_state_sha256: str
    checkpoint_sha256: str
    payload: bytes

    def record_payload(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "seed": self.seed,
            "selected_epoch": self.selected_epoch,
            "format": "safetensors",
            "model_state_sha256": self.model_state_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "size_bytes": len(self.payload),
        }


@dataclass(frozen=True)
class FormalTrainingResult:
    conditional_selection: FamilySelection
    independent_selection: FamilySelection
    conditional_ensemble: FittedEnsemble
    independent_ensemble: FittedEnsemble
    checkpoints: tuple[CheckpointArtifact, ...]
    family_reports: Mapping[str, Mapping[str, Any]]
    operation_counts: Mapping[str, int | str]
    run_manifest: Mapping[str, Any]

    @property
    def run_manifest_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.run_manifest))


def load_repaired_formal_training_inputs(
    feature_archive: bytes,
    label_archive: bytes,
    *,
    expected_source_ids: Sequence[str],
    join_audit_sha256: str,
) -> FormalTrainingInputs:
    """Strictly load the completed repaired caches and replay the formal roster."""
    if not isinstance(feature_archive, bytes) or not isinstance(label_archive, bytes):
        raise TypeError("formal cache archives must be bytes")
    feature_sha256 = sha256_bytes(feature_archive)
    label_sha256 = sha256_bytes(label_archive)
    if feature_sha256 != FEATURE_CACHE_SHA256 or label_sha256 != LABEL_CACHE_SHA256:
        raise ValueError("formal repaired cache transport identity drifted")
    if join_audit_sha256 != FORMAL_JOIN_AUDIT_SHA256:
        raise ValueError("formal repaired cache join-audit binding drifted")
    source_ids = tuple(expected_source_ids)
    if (
        len(source_ids) != 58
        or len(set(source_ids)) != 58
        or sha256_bytes(canonical_json_bytes(list(source_ids)))
        != FORMAL_SOURCE_IDS_SHA256
    ):
        raise ValueError("formal repaired cache source roster drifted")
    features = read_feature_cache(feature_archive)
    labels = read_label_cache(label_archive)
    states = join_feature_and_label_states(
        features,
        labels,
        expected_source_ids=source_ids,
    )
    folds = validate_formal_training_roster(states, source_ids)
    if tuple(len(fold) for fold in folds) != (12, 12, 12, 11, 11):
        raise ValueError("formal repaired cache fold sizes drifted")
    return FormalTrainingInputs(
        states=states,
        source_ids=source_ids,
        folds=folds,
        feature_cache_sha256=feature_sha256,
        label_cache_sha256=label_sha256,
        join_audit_sha256=join_audit_sha256,
    )


def _safetensors_api():
    try:
        import torch
        from safetensors.torch import load, save
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "formal gate checkpoints require PyTorch and safetensors"
        ) from error
    return torch, load, save


def _checkpoint_tensors(model: Any) -> dict[str, Any]:
    torch, _, _ = _safetensors_api()
    state = model.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise ValueError("formal checkpoint state_dict is empty or malformed")
    tensors: dict[str, Any] = {}
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
            raise TypeError("formal checkpoint state entry is malformed")
        value = tensor.detach().cpu().contiguous()
        if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError("formal checkpoint tensors must be finite CPU float32")
        tensors[name] = value
    return tensors


def serialize_safetensors_checkpoint(model: Any) -> bytes:
    """Serialize a model state to deterministic, metadata-free safetensors bytes."""
    _, load, save = _safetensors_api()
    tensors = _checkpoint_tensors(model)
    payload = save(tensors)
    if not isinstance(payload, bytes) or not payload:
        raise RuntimeError("safetensors serializer returned no checkpoint bytes")
    replay = load(payload)
    canonical_replay = {name: replay[name].contiguous() for name in sorted(replay)}
    if set(canonical_replay) != set(tensors) or save(canonical_replay) != payload:
        raise RuntimeError("formal safetensors bytes do not replay canonically")
    for name, tensor in tensors.items():
        if not bool(tensor.equal(canonical_replay[name])):
            raise RuntimeError("formal safetensors tensor replay differs")
    return payload


def load_safetensors_checkpoint(
    payload: bytes,
    *,
    family: str,
    seed: int,
    expected_model_state_sha256: str,
    expected_checkpoint_sha256: str,
) -> Any:
    """Strictly load one formal checkpoint and replay both canonical digests."""
    if not isinstance(payload, bytes) or not payload:
        raise TypeError("formal checkpoint payload must be non-empty bytes")
    if hashlib.sha256(payload).hexdigest() != expected_checkpoint_sha256:
        raise ValueError("formal checkpoint artifact digest drifted")
    _, load, save = _safetensors_api()
    tensors = load(payload)
    canonical = {name: tensors[name].contiguous() for name in sorted(tensors)}
    if save(canonical) != payload:
        raise ValueError("formal checkpoint safetensors encoding is not canonical")
    model = build_model(family, seed=seed)
    expected_names = set(model.state_dict())
    if set(canonical) != expected_names:
        raise ValueError("formal checkpoint tensor inventory drifted")
    model.load_state_dict(canonical, strict=True)
    if canonical_model_state_sha256(model) != expected_model_state_sha256:
        raise ValueError("formal checkpoint model-state digest drifted")
    if serialize_safetensors_checkpoint(model) != payload:
        raise ValueError("formal checkpoint model replay changed artifact bytes")
    return model


def _checkpoint_artifacts(
    ensemble: FittedEnsemble,
) -> tuple[CheckpointArtifact, ...]:
    if (
        ensemble.family not in FAMILIES
        or ensemble.seeds != SEEDS
        or len(ensemble.selected_epochs) != 5
        or len(ensemble.models) != 5
    ):
        raise ValueError("formal final ensemble metadata drifted")
    artifacts: list[CheckpointArtifact] = []
    for seed, epoch, model in zip(
        ensemble.seeds,
        ensemble.selected_epochs,
        ensemble.models,
        strict=True,
    ):
        model_state_sha256 = canonical_model_state_sha256(model)
        payload = serialize_safetensors_checkpoint(model)
        checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
        artifacts.append(
            CheckpointArtifact(
                family=ensemble.family,
                seed=seed,
                selected_epoch=epoch,
                model_state_sha256=model_state_sha256,
                checkpoint_sha256=checkpoint_sha256,
                payload=payload,
            )
        )
    if (
        len({item.model_state_sha256 for item in artifacts}) != 5
        or len({item.checkpoint_sha256 for item in artifacts}) != 5
    ):
        raise ValueError("formal five-seed checkpoints are not distinct")
    return tuple(artifacts)


def _family_operation_counts(selection: FamilySelection) -> dict[str, int | str]:
    if len(selection.grid_trials) != 10 or len(selection.trials) != 5:
        raise ValueError("formal family selection lacks the complete frozen grid")
    oof_epoch_count = sum(trial.epochs_run for trial in selection.grid_trials)
    final_epoch_count = sum(trial.selected_epoch for trial in selection.trials)
    counts: dict[str, int | str] = {
        "oof_trial_count": 10,
        "fold_training_track_count": 50,
        "final_fit_track_count": 5,
        "oof_training_model_initialization_count": 50,
        "final_fit_model_initialization_count": 5,
        "training_model_initialization_count": 55,
        "model_initialization_count": 55,
        "oof_epoch_metric_count": oof_epoch_count,
        "oof_optimizer_step_count": 5 * oof_epoch_count,
        "final_fit_optimizer_step_count": final_epoch_count,
        "optimizer_step_count": 5 * oof_epoch_count + final_epoch_count,
        "checkpoint_count": 5,
        "full_oof_report_count": 1,
        "optimizer_step_formula": (
            "5*sum(grid_trials.epochs_run)+sum(selected_trials.selected_epoch)"
        ),
    }
    expected = 5 * sum(
        trial.epochs_run for trial in selection.grid_trials
    ) + sum(trial.selected_epoch for trial in selection.trials)
    if counts["optimizer_step_count"] != expected:
        raise RuntimeError("formal family optimizer-step count does not replay")
    return counts


def _combined_operation_counts(
    selections: Sequence[FamilySelection],
) -> dict[str, int | str]:
    family_counts = tuple(_family_operation_counts(selection) for selection in selections)
    numeric_keys = tuple(
        key for key, value in family_counts[0].items() if isinstance(value, int)
    )
    result: dict[str, int | str] = {
        key: sum(int(counts[key]) for counts in family_counts) for key in numeric_keys
    }
    result["optimizer_step_formula"] = (
        "5*sum(all_20_grid_trials.epochs_run)"
        "+sum(all_10_selected_trials.selected_epoch)"
    )
    required = {
        "oof_trial_count": 20,
        "fold_training_track_count": 100,
        "final_fit_track_count": 10,
        "oof_training_model_initialization_count": 100,
        "final_fit_model_initialization_count": 10,
        "training_model_initialization_count": 110,
        "model_initialization_count": 110,
        "checkpoint_count": 10,
        "full_oof_report_count": 2,
    }
    if any(result.get(key) != value for key, value in required.items()):
        raise RuntimeError("formal combined operation denominator drifted")
    return result


def _family_report(
    inputs: FormalTrainingInputs,
    selection: FamilySelection,
    ensemble: FittedEnsemble,
    checkpoints: Sequence[CheckpointArtifact],
) -> dict[str, Any]:
    selection_payload = family_selection_payload(selection)
    selection_sha256 = family_selection_sha256(selection)
    if ensemble.selection_sha256 != selection_sha256:
        raise ValueError("formal ensemble selection digest drifted")
    expected_epochs = tuple(trial.selected_epoch for trial in selection.trials)
    family_checkpoints = tuple(
        checkpoint for checkpoint in checkpoints if checkpoint.family == selection.family
    )
    if (
        ensemble.family != selection.family
        or ensemble.learning_rate != selection.learning_rate
        or ensemble.seeds != SEEDS
        or ensemble.selected_epochs != expected_epochs
        or len(family_checkpoints) != 5
        or tuple(item.seed for item in family_checkpoints) != SEEDS
        or tuple(item.selected_epoch for item in family_checkpoints) != expected_epochs
    ):
        raise ValueError("formal family final-fit output differs from OOF selection")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": FAMILY_REPORT_STATUS,
        "family": selection.family,
        "input_binding": inputs.binding_payload(),
        "oof": {
            "complete_grid": selection_payload,
            "selection_sha256": selection_sha256,
        },
        "final_fit": {
            "learning_rate": ensemble.learning_rate,
            "seeds": list(ensemble.seeds),
            "selected_epochs": list(ensemble.selected_epochs),
            "checkpoints": [item.record_payload() for item in family_checkpoints],
        },
        "operation_counts": _family_operation_counts(selection),
        "access_counts": {
            "formal58_trajectory_semantic_decode_count": 58,
            "formal58_feature_state_semantic_decode_count": 174,
            "formal58_label_state_semantic_decode_count": 174,
            "fresh16_semantic_decode_count": 0,
            "legacy_dev5_semantic_decode_count": 0,
            "confirm20_access_count": 0,
            "matched_nll_evaluation_count": 0,
            "closed_loop_episode_count": 0,
        },
    }


def _validate_inputs(inputs: FormalTrainingInputs) -> None:
    if not isinstance(inputs, FormalTrainingInputs):
        raise TypeError("formal training requires validated FormalTrainingInputs")
    if (
        inputs.feature_cache_sha256 != FEATURE_CACHE_SHA256
        or inputs.label_cache_sha256 != LABEL_CACHE_SHA256
        or inputs.join_audit_sha256 != FORMAL_JOIN_AUDIT_SHA256
        or sha256_bytes(canonical_json_bytes(list(inputs.source_ids)))
        != FORMAL_SOURCE_IDS_SHA256
    ):
        raise ValueError("formal training input binding drifted")
    folds = validate_formal_training_roster(inputs.states, inputs.source_ids)
    if folds != inputs.folds:
        raise ValueError("formal training stored folds do not replay")


def run_formal_training(inputs: FormalTrainingInputs) -> FormalTrainingResult:
    """Run both frozen OOF grids and final fits without publishing any artifact."""
    _validate_inputs(inputs)
    conditional_selection = run_formal_oof(
        inputs.states,
        family="conditional",
        expected_source_ids=inputs.source_ids,
    )
    independent_selection = run_formal_oof(
        inputs.states,
        family="independent",
        expected_source_ids=inputs.source_ids,
    )
    conditional_ensemble = fit_final_ensemble(
        inputs.states,
        conditional_selection,
        expected_source_ids=inputs.source_ids,
    )
    independent_ensemble = fit_final_ensemble(
        inputs.states,
        independent_selection,
        expected_source_ids=inputs.source_ids,
    )
    checkpoints = (
        *_checkpoint_artifacts(conditional_ensemble),
        *_checkpoint_artifacts(independent_ensemble),
    )
    if (
        len(checkpoints) != 10
        or len({item.model_state_sha256 for item in checkpoints}) != 10
        or len({item.checkpoint_sha256 for item in checkpoints}) != 10
    ):
        raise ValueError("formal ten-checkpoint inventory is not distinct")
    family_reports = {
        "conditional": _family_report(
            inputs,
            conditional_selection,
            conditional_ensemble,
            checkpoints,
        ),
        "independent": _family_report(
            inputs,
            independent_selection,
            independent_ensemble,
            checkpoints,
        ),
    }
    selections = (conditional_selection, independent_selection)
    operation_counts = _combined_operation_counts(selections)
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": RUN_STATUS,
        "input_binding": inputs.binding_payload(),
        "family_reports": {
            family: {
                "sha256": sha256_bytes(canonical_json_bytes(report)),
                "payload": report,
            }
            for family, report in family_reports.items()
        },
        "checkpoint_inventory": [item.record_payload() for item in checkpoints],
        "operation_counts": operation_counts,
        "training_executed": True,
        "fresh_development_evaluated": False,
        "legacy_development_evaluated": False,
        "confirm_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
    }
    if any(
        not math.isfinite(trial.best_raw_utility_ratio)
        for selection in selections
        for trial in selection.grid_trials
    ):
        raise ValueError("formal OOF report contains a non-finite metric")
    return FormalTrainingResult(
        conditional_selection=conditional_selection,
        independent_selection=independent_selection,
        conditional_ensemble=conditional_ensemble,
        independent_ensemble=independent_ensemble,
        checkpoints=tuple(checkpoints),
        family_reports=family_reports,
        operation_counts=operation_counts,
        run_manifest=run_manifest,
    )


__all__ = [
    "CACHE_BUNDLE_MANIFEST_SHA256",
    "CACHE_REPOSITORY",
    "CACHE_REVISION",
    "CACHE_TAG",
    "CheckpointArtifact",
    "FEATURE_CACHE_SHA256",
    "FORMAL_JOIN_AUDIT_SHA256",
    "FormalTrainingInputs",
    "FormalTrainingResult",
    "LABEL_CACHE_SHA256",
    "load_repaired_formal_training_inputs",
    "load_safetensors_checkpoint",
    "run_formal_training",
    "serialize_safetensors_checkpoint",
]
