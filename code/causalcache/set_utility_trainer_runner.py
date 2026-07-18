"""Generic CPU-only training runner for budget-agnostic set utility.

This module is deliberately source-only infrastructure.  It performs no data
loading, network access, artifact publication, or training at import time.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from causalcache import set_utility_models
from causalcache.set_utility_data import (
    SetUtilityState,
    build_padded_set_utility_batch,
    tensorize_set_utility_batch,
)
from causalcache.set_utility_models import SetUtilityDimensions
from causalcache.set_utility_split_validation import SPLIT_ROLES
from causalcache.set_utility_training import (
    UtilityLossWeights,
    utility_optimization_step,
    utility_predictor_loss,
)


SCHEMA_VERSION = "0.1.0"
PROTOCOL_ID = "causalcache_generic_set_utility_cpu_trainer"
EXECUTION_SCOPE = "source_only_not_formally_authorized"
TRAJECTORY_BALANCED_STRATEGY = "trajectory_uniform_cycle"
OPTIMIZATION_ROLES = ("legacy_train_only", "train")
SELECTION_ROLE = "tune"
MODEL_FAMILIES = (
    "pairwise_additive",
    "deepsets",
    "set_transformer",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value in one hash-stable representation."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite_float(
    value: Any,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{label} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


@dataclass(frozen=True)
class TrainerConfig:
    """Every optimization and early-selection choice required by the runner."""

    seed: int
    epochs: int
    learning_rate: float
    weight_decay: float
    batch_strategy: str
    trajectories_per_batch: int
    states_per_trajectory_per_epoch: int
    loss: UtilityLossWeights
    maximum_gradient_norm: float
    early_stopping_patience: int
    early_stopping_min_delta: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _nonnegative_int(self.seed, "seed"))
        object.__setattr__(self, "epochs", _positive_int(self.epochs, "epochs"))
        object.__setattr__(
            self,
            "learning_rate",
            _finite_float(self.learning_rate, "learning rate", positive=True),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _finite_float(self.weight_decay, "weight decay", nonnegative=True),
        )
        if self.batch_strategy != TRAJECTORY_BALANCED_STRATEGY:
            raise ValueError(
                "batch strategy must be the frozen trajectory-uniform cycle"
            )
        object.__setattr__(
            self,
            "trajectories_per_batch",
            _positive_int(
                self.trajectories_per_batch,
                "trajectories per batch",
            ),
        )
        object.__setattr__(
            self,
            "states_per_trajectory_per_epoch",
            _positive_int(
                self.states_per_trajectory_per_epoch,
                "states per trajectory per epoch",
            ),
        )
        if not isinstance(self.loss, UtilityLossWeights):
            raise TypeError("loss must be UtilityLossWeights")
        object.__setattr__(
            self,
            "maximum_gradient_norm",
            _finite_float(
                self.maximum_gradient_norm,
                "maximum gradient norm",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "early_stopping_patience",
            _positive_int(
                self.early_stopping_patience,
                "early-stopping patience",
            ),
        )
        object.__setattr__(
            self,
            "early_stopping_min_delta",
            _finite_float(
                self.early_stopping_min_delta,
                "early-stopping minimum delta",
                nonnegative=True,
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_strategy": self.batch_strategy,
            "trajectories_per_batch": self.trajectories_per_batch,
            "states_per_trajectory_per_epoch": (
                self.states_per_trajectory_per_epoch
            ),
            "loss": {
                "raw_regression": self.loss.raw_regression,
                "normalized_regression": self.loss.normalized_regression,
                "within_state_ranking": self.loss.within_state_ranking,
                "smooth_l1_beta": self.loss.smooth_l1_beta,
            },
            "maximum_gradient_norm": self.maximum_gradient_norm,
            "early_stopping": {
                "patience": self.early_stopping_patience,
                "minimum_delta": self.early_stopping_min_delta,
                "selection_direction": "lower_tune_objective_is_better",
            },
        }


@dataclass(frozen=True)
class TrainerStateSelection:
    """States admitted to optimization and model selection, respectively."""

    train_states: tuple[SetUtilityState, ...]
    tune_states: tuple[SetUtilityState, ...]


@dataclass(frozen=True)
class SetTransformerArchitectureConfig:
    """Explicit architecture choices used only by the Set Transformer."""

    num_heads: int
    num_layers: int
    dropout: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "num_heads",
            _positive_int(self.num_heads, "set-transformer num heads"),
        )
        object.__setattr__(
            self,
            "num_layers",
            _positive_int(self.num_layers, "set-transformer num layers"),
        )
        dropout = _finite_float(
            self.dropout,
            "set-transformer dropout",
            nonnegative=True,
        )
        if dropout >= 1.0:
            raise ValueError("set-transformer dropout must be smaller than one")
        object.__setattr__(self, "dropout", dropout)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "set_transformer",
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


@dataclass(frozen=True)
class UtilityModelConfig:
    """Strict, serializable and memory-budget-free model configuration."""

    family: str
    hidden_dimension: int
    set_transformer: SetTransformerArchitectureConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("model family must be non-empty text")
        if self.family not in MODEL_FAMILIES:
            raise ValueError(f"unknown set-utility model family: {self.family!r}")
        object.__setattr__(
            self,
            "hidden_dimension",
            _positive_int(self.hidden_dimension, "hidden dimension"),
        )
        if self.family == "set_transformer":
            if not isinstance(
                self.set_transformer,
                SetTransformerArchitectureConfig,
            ):
                raise TypeError(
                    "set_transformer requires an explicit "
                    "SetTransformerArchitectureConfig"
                )
            if self.hidden_dimension % self.set_transformer.num_heads:
                raise ValueError(
                    "set-transformer hidden dimension must be divisible by "
                    "num heads"
                )
        elif self.set_transformer is not None:
            raise ValueError(
                f"{self.family} does not accept set-transformer hyperparameters"
            )

    def to_payload(self) -> dict[str, Any]:
        architecture = (
            {"kind": self.family}
            if self.set_transformer is None
            else self.set_transformer.to_payload()
        )
        return {
            "family": self.family,
            "hidden_dimension": self.hidden_dimension,
            "architecture": architecture,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> UtilityModelConfig:
        """Reconstruct a config while rejecting unknown or implicit fields."""
        if not isinstance(payload, Mapping):
            raise TypeError("model config payload must be a mapping")
        expected_fields = {"family", "hidden_dimension", "architecture"}
        if set(payload) != expected_fields:
            raise ValueError("model config payload fields differ from the schema")
        family = payload["family"]
        architecture = payload["architecture"]
        if not isinstance(architecture, Mapping):
            raise TypeError("model architecture payload must be a mapping")
        if family == "set_transformer":
            expected_architecture = {
                "kind",
                "num_heads",
                "num_layers",
                "dropout",
            }
            if set(architecture) != expected_architecture:
                raise ValueError(
                    "set-transformer architecture fields differ from the schema"
                )
            if architecture["kind"] != "set_transformer":
                raise ValueError("model architecture kind differs from its family")
            transformer = SetTransformerArchitectureConfig(
                num_heads=architecture["num_heads"],
                num_layers=architecture["num_layers"],
                dropout=architecture["dropout"],
            )
        else:
            if set(architecture) != {"kind"}:
                raise ValueError(
                    "simple-model architecture fields differ from the schema"
                )
            if architecture["kind"] != family:
                raise ValueError("model architecture kind differs from its family")
            transformer = None
        return cls(
            family=family,
            hidden_dimension=payload["hidden_dimension"],
            set_transformer=transformer,
        )


@dataclass(frozen=True)
class UtilityModelGeometry:
    dimensions: SetUtilityDimensions
    pair_feature_dimension: int


@dataclass(frozen=True)
class TuneSelection:
    selected_epoch: int
    selected_objective: float
    observed_epoch_count: int
    stopped_early: bool


@dataclass(frozen=True)
class TrainerEpochRecord:
    epoch: int
    batch_count: int
    mean_train_objective: float
    tune_objective: float
    selected_as_best: bool

    def __post_init__(self) -> None:
        _positive_int(self.epoch, "epoch")
        _positive_int(self.batch_count, "batch count")
        _finite_float(self.mean_train_objective, "mean train objective")
        _finite_float(self.tune_objective, "tune objective")
        if type(self.selected_as_best) is not bool:
            raise TypeError("selected_as_best must be boolean")

    def to_payload(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "batch_count": self.batch_count,
            "mean_train_objective": self.mean_train_objective,
            "tune_objective": self.tune_objective,
            "selected_as_best": self.selected_as_best,
        }


@dataclass(frozen=True)
class UtilityTrainingResult:
    family: str
    model: Any
    model_config: UtilityModelConfig
    config: TrainerConfig
    state_selection: TrainerStateSelection
    epoch_records: tuple[TrainerEpochRecord, ...]
    selected_epoch: int
    selected_tune_objective: float
    state_dict: Mapping[str, Any]
    state_dict_manifest: Mapping[str, Any]
    checkpoint_payload: Mapping[str, Any]
    checkpoint_manifest: Mapping[str, Any]


def _validated_states(
    states: Sequence[SetUtilityState],
    *,
    label: str,
) -> tuple[SetUtilityState, ...]:
    if isinstance(states, (str, bytes, bytearray, Mapping)) or not states:
        raise ValueError(f"{label} must contain set-utility states")
    result = tuple(states)
    if any(not isinstance(state, SetUtilityState) for state in result):
        raise TypeError(f"{label} contains an invalid state")
    if len({state.state_id for state in result}) != len(result):
        raise ValueError(f"{label} state ids must be unique")
    return tuple(
        sorted(result, key=lambda state: (state.trajectory_id, state.state_id))
    )


def select_train_and_tune_states(
    states: Sequence[SetUtilityState],
    *,
    role_by_trajectory: Mapping[str, str],
) -> TrainerStateSelection:
    """Admit only train roles to optimization and only tune to selection."""
    selected = _validated_states(states, label="trainer state inventory")
    if not isinstance(role_by_trajectory, Mapping):
        raise TypeError("role_by_trajectory must be a mapping")
    trajectories = {state.trajectory_id for state in selected}
    if set(role_by_trajectory) != trajectories:
        raise ValueError("trajectory-role inventory differs from trainer states")
    for trajectory_id, role in role_by_trajectory.items():
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("trajectory role keys must be non-empty text")
        if role not in SPLIT_ROLES:
            raise ValueError(f"unknown set-utility split role: {role!r}")
    train = tuple(
        state
        for state in selected
        if role_by_trajectory[state.trajectory_id] in OPTIMIZATION_ROLES
    )
    tune = tuple(
        state
        for state in selected
        if role_by_trajectory[state.trajectory_id] == SELECTION_ROLE
    )
    if not train:
        raise ValueError("trainer inventory has no train-role state")
    if not tune:
        raise ValueError("trainer inventory has no tune-role state")
    return TrainerStateSelection(train_states=train, tune_states=tune)


def _derived_seed(*parts: Any) -> int:
    digest = hashlib.sha256(canonical_json_bytes(parts)).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


def _trajectory_state_schedule(
    states: tuple[SetUtilityState, ...],
    *,
    seed: int,
    epoch: int,
    count: int,
) -> tuple[SetUtilityState, ...]:
    result: list[SetUtilityState] = []
    cycle = 0
    while len(result) < count:
        ordered = list(states)
        random.Random(
            _derived_seed(seed, epoch, states[0].trajectory_id, cycle)
        ).shuffle(ordered)
        result.extend(ordered)
        cycle += 1
    return tuple(result[:count])


def trajectory_balanced_epoch_batches(
    states: Sequence[SetUtilityState],
    *,
    config: TrainerConfig,
    epoch: int,
) -> tuple[tuple[SetUtilityState, ...], ...]:
    """Build deterministic batches with equal examples from each trajectory."""
    if not isinstance(config, TrainerConfig):
        raise TypeError("config must be TrainerConfig")
    if type(epoch) is not int or not 1 <= epoch <= config.epochs:
        raise ValueError("epoch must be within the configured one-based range")
    selected = _validated_states(states, label="batch state inventory")
    by_trajectory: dict[str, list[SetUtilityState]] = defaultdict(list)
    for state in selected:
        by_trajectory[state.trajectory_id].append(state)
    scheduled = {
        trajectory_id: _trajectory_state_schedule(
            tuple(trajectory_states),
            seed=config.seed,
            epoch=epoch,
            count=config.states_per_trajectory_per_epoch,
        )
        for trajectory_id, trajectory_states in sorted(by_trajectory.items())
    }

    batches: list[tuple[SetUtilityState, ...]] = []
    for slot in range(config.states_per_trajectory_per_epoch):
        trajectory_ids = sorted(scheduled)
        random.Random(
            _derived_seed(config.seed, epoch, "batch_order", slot)
        ).shuffle(trajectory_ids)
        for start in range(0, len(trajectory_ids), config.trajectories_per_batch):
            batch_trajectories = trajectory_ids[
                start : start + config.trajectories_per_batch
            ]
            batch = tuple(scheduled[item][slot] for item in batch_trajectories)
            if len({state.trajectory_id for state in batch}) != len(batch):
                raise AssertionError("trajectory-balanced batch repeated a trajectory")
            batches.append(batch)
    if not batches:
        raise AssertionError("trajectory-balanced batching produced no batches")
    return tuple(batches)


def infer_utility_model_geometry(
    states: Sequence[SetUtilityState],
    *,
    hidden_dimension: int,
) -> UtilityModelGeometry:
    """Infer feature dimensions while failing closed on cross-state drift."""
    selected = _validated_states(states, label="model geometry states")
    hidden = _positive_int(hidden_dimension, "hidden dimension")
    build_padded_set_utility_batch(selected)
    first = selected[0]
    pair_dimension = (
        0
        if first.pair_features is None
        else len(first.pair_features[0][0])
    )
    return UtilityModelGeometry(
        dimensions=SetUtilityDimensions(
            query=len(first.query_features),
            context=len(first.context_features),
            event=len(first.event_features[0]),
            hidden=hidden,
        ),
        pair_feature_dimension=pair_dimension,
    )


ModelBuilder = Callable[
    [SetUtilityDimensions, int, UtilityModelConfig],
    Any,
]


def _build_pairwise(
    dimensions: SetUtilityDimensions,
    pair_feature_dimension: int,
    model_config: UtilityModelConfig,
) -> Any:
    if model_config.family != "pairwise_additive":
        raise ValueError("pairwise builder received a different model family")
    return set_utility_models.build_pairwise_additive_utility_predictor(
        dimensions,
        pair_feature_dimension=pair_feature_dimension,
    )


def _build_deepsets(
    dimensions: SetUtilityDimensions,
    pair_feature_dimension: int,
    model_config: UtilityModelConfig,
) -> Any:
    del pair_feature_dimension
    if model_config.family != "deepsets":
        raise ValueError("DeepSets builder received a different model family")
    return set_utility_models.build_deepsets_utility_predictor(dimensions)


def _build_set_transformer(
    dimensions: SetUtilityDimensions,
    pair_feature_dimension: int,
    model_config: UtilityModelConfig,
) -> Any:
    del pair_feature_dimension
    if model_config.family != "set_transformer":
        raise ValueError("Set Transformer builder received a different model family")
    architecture = model_config.set_transformer
    if architecture is None:
        raise AssertionError("validated Set Transformer config lost its architecture")
    builder = getattr(
        set_utility_models,
        "build_set_transformer_utility_predictor",
        None,
    )
    if builder is None or not callable(builder):
        raise RuntimeError(
            "set_transformer is registered but its model builder is not implemented"
        )
    return builder(
        dimensions,
        num_heads=architecture.num_heads,
        num_layers=architecture.num_layers,
        dropout=architecture.dropout,
    )


MODEL_FAMILY_REGISTRY: Mapping[str, ModelBuilder] = MappingProxyType(
    {
        "pairwise_additive": _build_pairwise,
        "deepsets": _build_deepsets,
        "set_transformer": _build_set_transformer,
    }
)


def _resolve_model_config(
    *,
    family: str,
    hidden_dimension: int,
    model_config: UtilityModelConfig | None,
) -> UtilityModelConfig:
    if model_config is None:
        if family == "set_transformer":
            raise ValueError(
                "set_transformer requires explicit num_heads, num_layers, and "
                "dropout in model_config"
            )
        return UtilityModelConfig(
            family=family,
            hidden_dimension=hidden_dimension,
        )
    if not isinstance(model_config, UtilityModelConfig):
        raise TypeError("model_config must be UtilityModelConfig")
    if model_config.family != family:
        raise ValueError("model_config family differs from requested model family")
    if model_config.hidden_dimension != hidden_dimension:
        raise ValueError(
            "model_config hidden dimension differs from model geometry"
        )
    return model_config


def build_registered_utility_predictor(
    family: str,
    dimensions: SetUtilityDimensions,
    *,
    pair_feature_dimension: int = 0,
    model_config: UtilityModelConfig | None = None,
    registry: Mapping[str, ModelBuilder] = MODEL_FAMILY_REGISTRY,
) -> Any:
    """Build a registered family without exposing a memory-budget input."""
    if not isinstance(family, str) or not family:
        raise ValueError("model family must be non-empty text")
    if not isinstance(dimensions, SetUtilityDimensions):
        raise TypeError("dimensions must be SetUtilityDimensions")
    pair_dimension = _nonnegative_int(
        pair_feature_dimension,
        "pair feature dimension",
    )
    if not isinstance(registry, Mapping) or family not in registry:
        raise ValueError(f"unknown set-utility model family: {family!r}")
    resolved_config = _resolve_model_config(
        family=family,
        hidden_dimension=dimensions.hidden,
        model_config=model_config,
    )
    builder = registry[family]
    if not callable(builder):
        raise TypeError("model-family registry values must be callable")
    return builder(dimensions, pair_dimension, resolved_config)


def select_tune_epoch(
    tune_objectives: Sequence[float],
    *,
    config: TrainerConfig,
) -> TuneSelection:
    """Select the earliest significant tune improvement and apply patience."""
    if not isinstance(config, TrainerConfig):
        raise TypeError("config must be TrainerConfig")
    if (
        isinstance(tune_objectives, (str, bytes, bytearray, Mapping))
        or not tune_objectives
    ):
        raise ValueError("tune objectives must be a non-empty sequence")
    values = tuple(
        _finite_float(value, "tune objective") for value in tune_objectives
    )
    if len(values) > config.epochs:
        raise ValueError("tune objectives exceed configured epochs")

    best_epoch = 1
    best_value = values[0]
    stale_epochs = 0
    stop_after: int | None = None
    for epoch, value in enumerate(values[1:], start=2):
        if value < best_value - config.early_stopping_min_delta:
            best_epoch = epoch
            best_value = value
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                stop_after = epoch
                break
    observed = len(values) if stop_after is None else stop_after
    return TuneSelection(
        selected_epoch=best_epoch,
        selected_objective=best_value,
        observed_epoch_count=observed,
        stopped_early=stop_after is not None,
    )


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("set-utility trainer runner requires PyTorch") from error
    return torch


def clone_cpu_state_dict(model: Any) -> dict[str, Any]:
    """Detach and clone a model state in canonical key order on CPU."""
    torch = _torch()
    try:
        state_dict = model.state_dict()
    except AttributeError as error:
        raise TypeError("model must expose a PyTorch state_dict") from error
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("model state_dict must be a non-empty mapping")
    result: dict[str, Any] = {}
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not name:
            raise ValueError("state_dict keys must be non-empty text")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("state_dict values must be PyTorch tensors")
        result[name] = tensor.detach().to(device="cpu").contiguous().clone()
    return result


def canonical_state_dict_manifest(
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash exact CPU tensor bytes without serializing a framework checkpoint."""
    torch = _torch()
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a non-empty mapping")
    entries: list[dict[str, Any]] = []
    total_size = 0
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("state_dict keys must be normalized non-empty text")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("state_dict values must be PyTorch tensors")
        if tensor.device.type != "cpu":
            raise ValueError("canonical state_dict tensors must already be on CPU")
        if tensor.layout != torch.strided:
            raise ValueError("canonical state_dict tensors must use strided layout")
        contiguous = tensor.detach().contiguous()
        raw = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        total_size += len(raw)
        entries.append(
            {
                "name": name,
                "dtype": str(contiguous.dtype).removeprefix("torch."),
                "shape": list(contiguous.shape),
                "numel": contiguous.numel(),
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "format": "canonical_cpu_tensor_state_dict_manifest",
        "tensor_count": len(entries),
        "total_size_bytes": total_size,
        "tensors": entries,
    }
    return {
        **body,
        "state_dict_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _state_inventory(states: Sequence[SetUtilityState]) -> dict[str, Any]:
    records = [
        {"trajectory_id": state.trajectory_id, "state_id": state.state_id}
        for state in sorted(
            states,
            key=lambda item: (item.trajectory_id, item.state_id),
        )
    ]
    return {
        "trajectory_count": len({item["trajectory_id"] for item in records}),
        "state_count": len(records),
        "inventory_sha256": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
    }


def _assert_budget_free_payload(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and "budget" in key.casefold():
                raise ValueError(
                    "trainer checkpoint payload cannot contain budget keys"
                )
            _assert_budget_free_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_budget_free_payload(item)


def build_checkpoint_payload(
    *,
    family: str,
    geometry: UtilityModelGeometry,
    model_config: UtilityModelConfig | None = None,
    config: TrainerConfig,
    state_selection: TrainerStateSelection,
    epoch_records: Sequence[TrainerEpochRecord],
    selected_epoch: int,
    selected_tune_objective: float,
    state_dict_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a canonical, publication-neutral description of one checkpoint."""
    if family not in MODEL_FAMILIES:
        raise ValueError("checkpoint model family is not registered")
    if not isinstance(geometry, UtilityModelGeometry):
        raise TypeError("geometry must be UtilityModelGeometry")
    resolved_model_config = _resolve_model_config(
        family=family,
        hidden_dimension=geometry.dimensions.hidden,
        model_config=model_config,
    )
    if not isinstance(config, TrainerConfig):
        raise TypeError("config must be TrainerConfig")
    if not isinstance(state_selection, TrainerStateSelection):
        raise TypeError("state_selection must be TrainerStateSelection")
    records = tuple(epoch_records)
    if not records or any(not isinstance(item, TrainerEpochRecord) for item in records):
        raise ValueError("checkpoint requires validated epoch records")
    if tuple(item.epoch for item in records) != tuple(range(1, len(records) + 1)):
        raise ValueError("epoch records must be contiguous and one-based")
    tune_selection = select_tune_epoch(
        tuple(item.tune_objective for item in records),
        config=config,
    )
    if tune_selection.observed_epoch_count != len(records) or (
        not tune_selection.stopped_early and len(records) != config.epochs
    ):
        raise ValueError("epoch records do not end at the configured stop condition")
    for epoch, record in enumerate(records, start=1):
        prefix = select_tune_epoch(
            tuple(item.tune_objective for item in records[:epoch]),
            config=config,
        )
        if record.selected_as_best != (prefix.selected_epoch == epoch):
            raise ValueError("epoch best-checkpoint flags differ from tune selection")
    if selected_epoch != tune_selection.selected_epoch or not math.isclose(
        float(selected_tune_objective),
        tune_selection.selected_objective,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("checkpoint selection differs from tune-only selection")
    if not isinstance(state_dict_manifest, Mapping):
        raise TypeError("state_dict_manifest must be a mapping")
    state_digest = state_dict_manifest.get("state_dict_sha256")
    if (
        not isinstance(state_digest, str)
        or len(state_digest) != 64
        or any(character not in "0123456789abcdef" for character in state_digest)
    ):
        raise ValueError("state_dict manifest digest is malformed")

    model_config_payload = resolved_model_config.to_payload()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "execution_scope": EXECUTION_SCOPE,
        "model": {
            "family": family,
            "config": model_config_payload,
            "config_sha256": hashlib.sha256(
                canonical_json_bytes(model_config_payload)
            ).hexdigest(),
            "query_dimension": geometry.dimensions.query,
            "context_dimension": geometry.dimensions.context,
            "event_dimension": geometry.dimensions.event,
            "hidden_dimension": geometry.dimensions.hidden,
            "pair_feature_dimension": geometry.pair_feature_dimension,
        },
        "trainer": config.to_payload(),
        "data_roles": {
            "optimization_roles": list(OPTIMIZATION_ROLES),
            "model_selection_role": SELECTION_ROLE,
            "train_inventory": _state_inventory(state_selection.train_states),
            "tune_inventory": _state_inventory(state_selection.tune_states),
            "evaluation_states_consumed": 0,
        },
        "selection": {
            "metric": "tune_weighted_training_objective",
            "direction": "lower_is_better",
            "selected_epoch": selected_epoch,
            "selected_tune_objective": selected_tune_objective,
            "executed_epoch_count": len(records),
            "stopped_early": tune_selection.stopped_early,
        },
        "epoch_records": [item.to_payload() for item in records],
        "state_dict_sha256": state_digest,
    }
    _assert_budget_free_payload(payload)
    return payload


def canonical_checkpoint_manifest(
    checkpoint_payload: Mapping[str, Any],
    state_dict_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind canonical metadata and tensor hashes without reading or writing files."""
    if not isinstance(checkpoint_payload, Mapping) or not checkpoint_payload:
        raise ValueError("checkpoint payload must be a non-empty mapping")
    if not isinstance(state_dict_manifest, Mapping) or not state_dict_manifest:
        raise ValueError("state_dict manifest must be a non-empty mapping")
    payload = dict(checkpoint_payload)
    state_manifest = dict(state_dict_manifest)
    _assert_budget_free_payload(payload)
    if payload.get("state_dict_sha256") != state_manifest.get("state_dict_sha256"):
        raise ValueError("checkpoint payload and state_dict manifest differ")
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "checkpoint_payload": payload,
        "checkpoint_payload_sha256": hashlib.sha256(
            canonical_json_bytes(payload)
        ).hexdigest(),
        "state_dict_manifest": state_manifest,
    }
    return {
        **body,
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _tune_objective_cpu(
    model: Any,
    states: tuple[SetUtilityState, ...],
    *,
    loss: UtilityLossWeights,
) -> float:
    torch = _torch()
    by_trajectory: dict[str, list[SetUtilityState]] = defaultdict(list)
    for state in states:
        by_trajectory[state.trajectory_id].append(state)
    model.eval()
    trajectory_objectives: list[float] = []
    with torch.no_grad():
        for trajectory_id in sorted(by_trajectory):
            batch = tensorize_set_utility_batch(
                build_padded_set_utility_batch(by_trajectory[trajectory_id]),
                device="cpu",
            )
            objective, _ = utility_predictor_loss(model, batch, weights=loss)
            trajectory_objectives.append(
                _finite_float(float(objective), "trajectory tune objective")
            )
    return _finite_float(
        statistics.fmean(trajectory_objectives),
        "trajectory-equal tune objective",
    )


def run_cpu_utility_training(
    states: Sequence[SetUtilityState],
    *,
    role_by_trajectory: Mapping[str, str],
    family: str,
    hidden_dimension: int,
    config: TrainerConfig,
    model_config: UtilityModelConfig | None = None,
    registry: Mapping[str, ModelBuilder] = MODEL_FAMILY_REGISTRY,
) -> UtilityTrainingResult:
    """Train and select one predictor entirely on CPU from caller-supplied states."""
    if not isinstance(config, TrainerConfig):
        raise TypeError("config must be TrainerConfig")
    selected = select_train_and_tune_states(
        states,
        role_by_trajectory=role_by_trajectory,
    )
    geometry = infer_utility_model_geometry(
        (*selected.train_states, *selected.tune_states),
        hidden_dimension=hidden_dimension,
    )
    resolved_model_config = _resolve_model_config(
        family=family,
        hidden_dimension=geometry.dimensions.hidden,
        model_config=model_config,
    )
    torch = _torch()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            model = build_registered_utility_predictor(
                family,
                geometry.dimensions,
                pair_feature_dimension=geometry.pair_feature_dimension,
                model_config=resolved_model_config,
                registry=registry,
            )
            model.to(device="cpu")
            if any(parameter.device.type != "cpu" for parameter in model.parameters()):
                raise RuntimeError("CPU trainer constructed a non-CPU model")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

            records: list[TrainerEpochRecord] = []
            tune_values: list[float] = []
            best_state: dict[str, Any] | None = None
            for epoch in range(1, config.epochs + 1):
                weighted_batch_objective_sum = 0.0
                scheduled_state_count = 0
                batches = trajectory_balanced_epoch_batches(
                    selected.train_states,
                    config=config,
                    epoch=epoch,
                )
                for state_batch in batches:
                    tensor_batch = tensorize_set_utility_batch(
                        build_padded_set_utility_batch(state_batch),
                        device="cpu",
                    )
                    metrics = utility_optimization_step(
                        model,
                        optimizer,
                        tensor_batch,
                        weights=config.loss,
                        maximum_gradient_norm=config.maximum_gradient_norm,
                        objective_scale=(
                            len(state_batch) / config.trajectories_per_batch
                        ),
                    )
                    weighted_batch_objective_sum += (
                        metrics["total"] * len(state_batch)
                    )
                    scheduled_state_count += len(state_batch)
                tune_value = _tune_objective_cpu(
                    model,
                    selected.tune_states,
                    loss=config.loss,
                )
                tune_values.append(tune_value)
                decision = select_tune_epoch(tune_values, config=config)
                improved = decision.selected_epoch == epoch
                if improved:
                    best_state = clone_cpu_state_dict(model)
                records.append(
                    TrainerEpochRecord(
                        epoch=epoch,
                        batch_count=len(batches),
                        mean_train_objective=(
                            weighted_batch_objective_sum / scheduled_state_count
                        ),
                        tune_objective=tune_value,
                        selected_as_best=improved,
                    )
                )
                if decision.stopped_early:
                    break
            selection = select_tune_epoch(tune_values, config=config)
            if best_state is None:
                raise AssertionError("tune selection did not capture a checkpoint")
            model.load_state_dict(best_state, strict=True)
            state_dict = clone_cpu_state_dict(model)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)

    state_manifest = canonical_state_dict_manifest(state_dict)
    checkpoint_payload = build_checkpoint_payload(
        family=family,
        geometry=geometry,
        model_config=resolved_model_config,
        config=config,
        state_selection=selected,
        epoch_records=records,
        selected_epoch=selection.selected_epoch,
        selected_tune_objective=selection.selected_objective,
        state_dict_manifest=state_manifest,
    )
    checkpoint_manifest = canonical_checkpoint_manifest(
        checkpoint_payload,
        state_manifest,
    )
    return UtilityTrainingResult(
        family=family,
        model=model,
        model_config=resolved_model_config,
        config=config,
        state_selection=selected,
        epoch_records=tuple(records),
        selected_epoch=selection.selected_epoch,
        selected_tune_objective=selection.selected_objective,
        state_dict=state_dict,
        state_dict_manifest=state_manifest,
        checkpoint_payload=checkpoint_payload,
        checkpoint_manifest=checkpoint_manifest,
    )


__all__ = [
    "EXECUTION_SCOPE",
    "MODEL_FAMILIES",
    "MODEL_FAMILY_REGISTRY",
    "OPTIMIZATION_ROLES",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "SELECTION_ROLE",
    "SetTransformerArchitectureConfig",
    "TRAJECTORY_BALANCED_STRATEGY",
    "TrainerConfig",
    "TrainerEpochRecord",
    "TrainerStateSelection",
    "TuneSelection",
    "UtilityModelGeometry",
    "UtilityModelConfig",
    "UtilityTrainingResult",
    "build_checkpoint_payload",
    "build_registered_utility_predictor",
    "canonical_checkpoint_manifest",
    "canonical_json_bytes",
    "canonical_state_dict_manifest",
    "clone_cpu_state_dict",
    "infer_utility_model_geometry",
    "run_cpu_utility_training",
    "select_train_and_tune_states",
    "select_tune_epoch",
    "trajectory_balanced_epoch_batches",
]
