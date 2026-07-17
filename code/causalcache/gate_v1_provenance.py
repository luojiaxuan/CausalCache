"""Immutable artifact and model provenance for formal CausalCache gate v1 use."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.gate_v1_contract import canonical_json_bytes


GATE_V1_CONFIG_SHA256 = (
    "37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b"
)
FORMAL_TRAIN_SOURCE_IDS_SHA256 = (
    "ccec55afb0882e602f5d2c83ec420682488e2663fe26a7c574a851825c43f225"
)
FROZEN_SEEDS = (0, 1, 2, 3, 4)
FROZEN_LEARNING_RATES = (0.0003, 0.001)
HEURISTIC_NAMES = ("dynamic_recent", "ocr_rgb_v2", "policy_vision_v3")
ENSEMBLE_PROVENANCE_SCHEMA_VERSION = "0.1.0"
ENSEMBLE_PROVENANCE_PROTOCOL_ID = "causalcache_gate_v1_frozen_ensemble"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40,64}")


def _require_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256 digest")


@dataclass(frozen=True)
class ArtifactBinding:
    """One immutable file/tree binding; mutable branches and local paths are invalid."""

    repository: str
    revision: str
    path: str
    sha256: str

    def validate(self, *, label: str = "artifact") -> None:
        if not isinstance(self.repository, str) or not self.repository.strip():
            raise ValueError(f"{label} repository must be non-empty")
        if (
            not isinstance(self.revision, str)
            or _IMMUTABLE_REVISION.fullmatch(self.revision) is None
        ):
            raise ValueError(f"{label} revision must be an immutable lowercase digest")
        if (
            not isinstance(self.path, str)
            or not self.path
            or self.path.startswith("/")
            or any(part in {"", ".", ".."} for part in self.path.split("/"))
        ):
            raise ValueError(f"{label} path must be a normalized relative artifact path")
        _require_sha256(self.sha256, f"{label} sha256")

    def to_payload(self) -> dict[str, str]:
        self.validate()
        return {
            "repository": self.repository,
            "revision": self.revision,
            "path": self.path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class FrozenTrainingProvenance:
    family: str
    gate_config_sha256: str
    formal_train_source_ids_sha256: str
    learning_rate: float
    selected_epochs: tuple[int, ...]
    oof_selection_sha256: str
    feature_artifact: ArtifactBinding
    label_artifact: ArtifactBinding
    training_report_artifact: ArtifactBinding

    def validate(self) -> None:
        if self.family not in {"conditional", "independent"}:
            raise ValueError("training provenance family is invalid")
        if self.gate_config_sha256 != GATE_V1_CONFIG_SHA256:
            raise ValueError("training provenance gate config is not frozen gate v1")
        if self.formal_train_source_ids_sha256 != FORMAL_TRAIN_SOURCE_IDS_SHA256:
            raise ValueError("training provenance roster is not formal-58")
        if self.learning_rate not in FROZEN_LEARNING_RATES:
            raise ValueError("training provenance learning rate is outside the frozen grid")
        if (
            len(self.selected_epochs) != 5
            or any(
                type(epoch) is not int or not 1 <= epoch <= 500
                for epoch in self.selected_epochs
            )
        ):
            raise ValueError("training provenance requires five valid selected epochs")
        _require_sha256(self.oof_selection_sha256, "OOF selection sha256")
        self.feature_artifact.validate(label="training feature artifact")
        self.label_artifact.validate(label="training label artifact")
        self.training_report_artifact.validate(label="training report artifact")
        if self.feature_artifact == self.label_artifact:
            raise ValueError("training feature and label caches must be physically separated")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "family": self.family,
            "gate_config_sha256": self.gate_config_sha256,
            "formal_train_source_ids_sha256": self.formal_train_source_ids_sha256,
            "learning_rate": self.learning_rate,
            "selected_epochs": list(self.selected_epochs),
            "oof_selection_sha256": self.oof_selection_sha256,
            "feature_artifact": self.feature_artifact.to_payload(),
            "label_artifact": self.label_artifact.to_payload(),
            "training_report_artifact": self.training_report_artifact.to_payload(),
        }


@dataclass(frozen=True)
class SeedCheckpointProvenance:
    seed: int
    selected_epoch: int
    model_state_sha256: str
    checkpoint_artifact: ArtifactBinding

    def validate(self) -> None:
        if type(self.seed) is not int or self.seed not in FROZEN_SEEDS:
            raise ValueError("checkpoint seed is outside the frozen five seeds")
        if type(self.selected_epoch) is not int or not 1 <= self.selected_epoch <= 500:
            raise ValueError("checkpoint selected epoch is invalid")
        _require_sha256(self.model_state_sha256, "model-state sha256")
        self.checkpoint_artifact.validate(label="checkpoint artifact")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "seed": self.seed,
            "selected_epoch": self.selected_epoch,
            "model_state_sha256": self.model_state_sha256,
            "checkpoint_artifact": self.checkpoint_artifact.to_payload(),
        }


@dataclass(frozen=True)
class FrozenEnsembleProvenance:
    training: FrozenTrainingProvenance
    checkpoints: tuple[SeedCheckpointProvenance, ...]

    def validate(self) -> None:
        self.training.validate()
        if len(self.checkpoints) != 5:
            raise ValueError("ensemble provenance requires exactly five checkpoints")
        for checkpoint in self.checkpoints:
            checkpoint.validate()
        if tuple(checkpoint.seed for checkpoint in self.checkpoints) != FROZEN_SEEDS:
            raise ValueError("ensemble checkpoints must be ordered by frozen seed")
        if (
            tuple(checkpoint.selected_epoch for checkpoint in self.checkpoints)
            != self.training.selected_epochs
        ):
            raise ValueError("checkpoint epochs differ from frozen training provenance")
        state_digests = tuple(
            checkpoint.model_state_sha256 for checkpoint in self.checkpoints
        )
        artifact_digests = tuple(
            checkpoint.checkpoint_artifact.sha256 for checkpoint in self.checkpoints
        )
        if len(set(state_digests)) != 5 or len(set(artifact_digests)) != 5:
            raise ValueError("five seed checkpoints must have distinct state and artifact digests")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": ENSEMBLE_PROVENANCE_SCHEMA_VERSION,
            "protocol_id": ENSEMBLE_PROVENANCE_PROTOCOL_ID,
            "training": self.training.to_payload(),
            "checkpoints": [checkpoint.to_payload() for checkpoint in self.checkpoints],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_payload())).hexdigest()


@dataclass(frozen=True)
class HeuristicArtifactProvenance:
    name: str
    artifact: ArtifactBinding
    selection_sha256: str

    def validate(self) -> None:
        if self.name not in HEURISTIC_NAMES:
            raise ValueError("heuristic provenance name is outside the frozen inventory")
        self.artifact.validate(label=f"{self.name} heuristic artifact")
        _require_sha256(self.selection_sha256, f"{self.name} selection sha256")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "name": self.name,
            "artifact": self.artifact.to_payload(),
            "selection_sha256": self.selection_sha256,
        }


def canonical_selection_sha256(
    selections: Mapping[str, Sequence[int]],
) -> str:
    if not isinstance(selections, Mapping):
        raise TypeError("heuristic selections must be a mapping")
    payload: list[dict[str, Any]] = []
    for state_id in sorted(selections):
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("heuristic selection state id is invalid")
        raw = selections[state_id]
        if isinstance(raw, (str, bytes, bytearray, Mapping)):
            raise TypeError("heuristic selection must be an integer sequence")
        selected = tuple(raw)
        if any(type(event) is not int for event in selected):
            raise TypeError("heuristic selection event ids must be integers")
        payload.append({"state_id": state_id, "selected": list(selected)})
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys differ from the frozen manifest schema")


def _artifact_from_payload(value: Any, label: str) -> ArtifactBinding:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    _exact_keys(value, {"repository", "revision", "path", "sha256"}, label)
    artifact = ArtifactBinding(
        repository=value["repository"],
        revision=value["revision"],
        path=value["path"],
        sha256=value["sha256"],
    )
    artifact.validate(label=label)
    return artifact


def artifact_binding_from_manifest(
    value: Mapping[str, Any], *, label: str = "artifact"
) -> ArtifactBinding:
    return _artifact_from_payload(value, label)


def heuristic_artifact_provenance_from_manifest(
    value: Mapping[str, Any],
) -> HeuristicArtifactProvenance:
    if not isinstance(value, Mapping):
        raise TypeError("heuristic provenance must be a mapping")
    _exact_keys(value, {"name", "artifact", "selection_sha256"}, "heuristic provenance")
    provenance = HeuristicArtifactProvenance(
        name=value["name"],
        artifact=_artifact_from_payload(value["artifact"], "heuristic artifact"),
        selection_sha256=value["selection_sha256"],
    )
    provenance.validate()
    return provenance


def frozen_ensemble_provenance_from_manifest(
    value: Mapping[str, Any],
) -> FrozenEnsembleProvenance:
    """Parse one strict JSON-like ensemble manifest and reject all schema drift."""
    if not isinstance(value, Mapping):
        raise TypeError("ensemble provenance manifest must be a mapping")
    _exact_keys(
        value,
        {"schema_version", "protocol_id", "training", "checkpoints"},
        "ensemble provenance manifest",
    )
    if (
        value["schema_version"] != ENSEMBLE_PROVENANCE_SCHEMA_VERSION
        or value["protocol_id"] != ENSEMBLE_PROVENANCE_PROTOCOL_ID
    ):
        raise ValueError("ensemble provenance manifest identity drifted")
    training_value = value["training"]
    if not isinstance(training_value, Mapping):
        raise TypeError("ensemble training provenance must be a mapping")
    _exact_keys(
        training_value,
        {
            "family",
            "gate_config_sha256",
            "formal_train_source_ids_sha256",
            "learning_rate",
            "selected_epochs",
            "oof_selection_sha256",
            "feature_artifact",
            "label_artifact",
            "training_report_artifact",
        },
        "ensemble training provenance",
    )
    selected_epochs = training_value["selected_epochs"]
    if not isinstance(selected_epochs, list):
        raise TypeError("training selected_epochs must be an array")
    training = FrozenTrainingProvenance(
        family=training_value["family"],
        gate_config_sha256=training_value["gate_config_sha256"],
        formal_train_source_ids_sha256=training_value[
            "formal_train_source_ids_sha256"
        ],
        learning_rate=training_value["learning_rate"],
        selected_epochs=tuple(selected_epochs),
        oof_selection_sha256=training_value["oof_selection_sha256"],
        feature_artifact=_artifact_from_payload(
            training_value["feature_artifact"], "training feature artifact"
        ),
        label_artifact=_artifact_from_payload(
            training_value["label_artifact"], "training label artifact"
        ),
        training_report_artifact=_artifact_from_payload(
            training_value["training_report_artifact"],
            "training report artifact",
        ),
    )
    checkpoints_value = value["checkpoints"]
    if not isinstance(checkpoints_value, list):
        raise TypeError("ensemble checkpoints must be an array")
    checkpoints: list[SeedCheckpointProvenance] = []
    for index, checkpoint_value in enumerate(checkpoints_value):
        if not isinstance(checkpoint_value, Mapping):
            raise TypeError("ensemble checkpoint provenance must be a mapping")
        _exact_keys(
            checkpoint_value,
            {"seed", "selected_epoch", "model_state_sha256", "checkpoint_artifact"},
            f"ensemble checkpoint {index}",
        )
        checkpoints.append(
            SeedCheckpointProvenance(
                seed=checkpoint_value["seed"],
                selected_epoch=checkpoint_value["selected_epoch"],
                model_state_sha256=checkpoint_value["model_state_sha256"],
                checkpoint_artifact=_artifact_from_payload(
                    checkpoint_value["checkpoint_artifact"],
                    f"checkpoint {index} artifact",
                ),
            )
        )
    provenance = FrozenEnsembleProvenance(
        training=training,
        checkpoints=tuple(checkpoints),
    )
    provenance.validate()
    return provenance


def canonical_model_state_sha256(model: Any) -> str:
    """Hash tensor names, dtype, shape, and exact numeric values without pickle bytes."""
    state = model.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise ValueError("model state_dict must be a non-empty mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        if not isinstance(name, str) or not name:
            raise ValueError("model state_dict contains an invalid name")
        tensor = state[name]
        try:
            detached = tensor.detach().cpu().contiguous()
            shape = tuple(int(item) for item in detached.shape)
            dtype = str(detached.dtype)
            values = detached.reshape(-1).tolist()
        except (AttributeError, TypeError, ValueError) as error:
            raise TypeError("model state_dict value is not a tensor-like object") from error
        header = canonical_json_bytes({"name": name, "dtype": dtype, "shape": list(shape)})
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        for value in values:
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError("model state contains a non-finite value")
            digest.update(struct.pack(">d", converted))
    return digest.hexdigest()


def validate_frozen_ensemble(ensemble: Any, provenance: FrozenEnsembleProvenance) -> str:
    """Verify an in-memory five-seed ensemble against immutable checkpoint provenance."""
    provenance.validate()
    training = provenance.training
    if (
        getattr(ensemble, "family", None) != training.family
        or getattr(ensemble, "learning_rate", None) != training.learning_rate
        or tuple(getattr(ensemble, "seeds", ())) != FROZEN_SEEDS
        or tuple(getattr(ensemble, "selected_epochs", ())) != training.selected_epochs
        or getattr(ensemble, "selection_sha256", None) != training.oof_selection_sha256
    ):
        raise ValueError("in-memory ensemble metadata differs from frozen training provenance")
    models = tuple(getattr(ensemble, "models", ()))
    if len(models) != 5 or len({id(model) for model in models}) != 5:
        raise ValueError("formal ensemble must contain five distinct in-memory models")
    observed = tuple(canonical_model_state_sha256(model) for model in models)
    expected = tuple(checkpoint.model_state_sha256 for checkpoint in provenance.checkpoints)
    if observed != expected:
        raise ValueError("in-memory model states differ from frozen checkpoints")
    return provenance.sha256


__all__ = [
    "ArtifactBinding",
    "ENSEMBLE_PROVENANCE_PROTOCOL_ID",
    "ENSEMBLE_PROVENANCE_SCHEMA_VERSION",
    "FORMAL_TRAIN_SOURCE_IDS_SHA256",
    "FROZEN_LEARNING_RATES",
    "FROZEN_SEEDS",
    "FrozenEnsembleProvenance",
    "FrozenTrainingProvenance",
    "GATE_V1_CONFIG_SHA256",
    "HEURISTIC_NAMES",
    "HeuristicArtifactProvenance",
    "SeedCheckpointProvenance",
    "artifact_binding_from_manifest",
    "canonical_model_state_sha256",
    "canonical_selection_sha256",
    "frozen_ensemble_provenance_from_manifest",
    "heuristic_artifact_provenance_from_manifest",
    "validate_frozen_ensemble",
]
