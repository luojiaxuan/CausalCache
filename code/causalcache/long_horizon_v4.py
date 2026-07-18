"""CPU-only frozen-base residual inference for long-horizon feature states."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from causalcache.gate_v1_formal_train import load_safetensors_checkpoint
from causalcache.gate_v1_provenance import (
    canonical_model_state_sha256,
    frozen_ensemble_provenance_from_manifest,
)
from causalcache.long_horizon_contract import LongHorizonContract
from causalcache.long_horizon_selectors import (
    ResidualSeedScores,
    feature_view,
)


FROZEN_SEEDS = (0, 1, 2, 3, 4)
BASE_SELECTED_EPOCHS = (60, 51, 6, 56, 54)
BASE_LEARNING_RATE = 0.0003
BASE_SELECTION_SHA256 = (
    "b5889a98adf08c3cf35b647a519b0d08f832879374a9ea8c4b5268c5a1c4b8dd"
)
BASE_CHECKPOINT_SIZE_BYTES = 102_860
BASE_HIDDEN_DIMENSION = 88
BASE_INPUT_DIMENSION = 200
BASE_PARAMETER_COUNT = 25_609
RESIDUAL_SELECTED_EPOCHS = (66, 0, 17, 40, 0)
RESIDUAL_LEARNING_RATE = 0.001
RESIDUAL_INPUT_DIMENSION = 4 * BASE_HIDDEN_DIMENSION + 64
RESIDUAL_HIDDEN_DIMENSION = 64
RESIDUAL_PARAMETER_COUNT = 26_753
RESIDUAL_METADATA_PATH = "residual-model-metadata.json"
SUPPORTED_CANDIDATE_COUNTS = (8, 16)
RESIDUAL_METADATA_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "device",
    "dtype",
    "base_manifest_sha256",
    "base_parameters_trainable",
    "checkpoint_contains_residual_head_only",
    "models",
    "fresh_label_access_count",
    "confirm20_access_count",
    "gpu_operation_count",
}
RESIDUAL_MODEL_METADATA_KEYS = {
    "seed",
    "selected_epoch",
    "learning_rate",
    "base_checkpoint_sha256",
    "base_model_state_sha256",
    "residual_checkpoint_path",
    "residual_checkpoint_sha256",
    "residual_checkpoint_size_bytes",
}


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("long-horizon v4 inference requires PyTorch") from error
    return torch


try:
    import torch as _torch_module
except ModuleNotFoundError:
    _torch_module = None


if _torch_module is not None:

    class FrozenResidualHead(_torch_module.nn.Module):
        """The exact v4 residual-only architecture; it exposes no trainer."""

        def __init__(self) -> None:
            super().__init__()
            self.network = _torch_module.nn.Sequential(
                _torch_module.nn.Linear(
                    RESIDUAL_INPUT_DIMENSION,
                    RESIDUAL_HIDDEN_DIMENSION,
                ),
                _torch_module.nn.GELU(approximate="none"),
                _torch_module.nn.Linear(RESIDUAL_HIDDEN_DIMENSION, 1),
            )

        def forward(self, left: Any, right: Any, q64: Any) -> Any:
            if (
                left.shape != right.shape
                or left.ndim != 2
                or left.shape[1] != BASE_HIDDEN_DIMENSION
                or q64.ndim != 2
                or q64.shape != (left.shape[0], 64)
            ):
                raise ValueError("long-horizon v4 residual tensor geometry drifted")
            features = _torch_module.cat(
                (left, right, left * right, _torch_module.abs(left - right), q64),
                dim=1,
            )
            if features.shape[1] != RESIDUAL_INPUT_DIMENSION:
                raise RuntimeError("long-horizon v4 residual feature size drifted")
            return self.network(features).squeeze(-1)

else:

    class FrozenResidualHead:
        """Import placeholder for source-only machines without PyTorch."""

        def __init__(self) -> None:
            _torch()


@dataclass(frozen=True)
class FrozenV4Seed:
    seed: int
    base_selected_epoch: int
    base_learning_rate: float
    residual_selected_epoch: int
    residual_learning_rate: float
    base_checkpoint_sha256: str
    base_checkpoint_size_bytes: int
    base_model_state_sha256: str
    residual_checkpoint_sha256: str
    residual_checkpoint_size_bytes: int
    residual_model_state_sha256: str
    base_model: Any
    residual_head: Any


@dataclass(frozen=True)
class FrozenV4Ensemble:
    seeds: tuple[FrozenV4Seed, ...]
    formal_manifest_sha256: str
    residual_manifest_sha256: str
    residual_metadata_sha256: str
    device: str = "cpu"
    dtype: str = "float32"
    training_allowed: bool = False

    def predict(self, state: Any) -> tuple[ResidualSeedScores, ...]:
        return predict_residual_seed_scores(self, state)


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
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _canonical_json_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _learned_config(contract: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(contract, LongHorizonContract):
        raise TypeError("v4 loader requires a validated LongHorizonContract")
    data = contract.data
    learned = data.get("learned_model_artifacts")
    if not isinstance(learned, Mapping):
        raise ValueError("long-horizon learned model artifacts are missing")
    formal = learned.get("formal58_base_and_conditional")
    residual = learned.get("v4_safe_frozen_base_residual")
    if not isinstance(formal, Mapping) or not isinstance(residual, Mapping):
        raise ValueError("long-horizon v4 artifact bindings are incomplete")
    if (
        learned.get(
            "execution_b_must_fresh_download_and_verify_exact_rosters_before_selector_scoring"
        )
        is not True
        or learned.get("training_optimizer_or_artifact_substitution_allowed") is not False
    ):
        raise ValueError("long-horizon learned artifact firewall drifted")
    return formal, residual


def frozen_artifact_roster(contract: Any) -> Mapping[str, Any]:
    """Return the exact independent-base and residual records frozen by Source-A."""
    formal, residual = _learned_config(contract)
    manifests = formal.get("ensemble_manifests")
    checkpoints = formal.get("checkpoints")
    residual_checkpoints = residual.get("residual_checkpoints")
    if (
        isinstance(manifests, (str, bytes, bytearray, Mapping))
        or not isinstance(manifests, Sequence)
        or isinstance(checkpoints, (str, bytes, bytearray, Mapping))
        or not isinstance(checkpoints, Sequence)
        or isinstance(residual_checkpoints, (str, bytes, bytearray, Mapping))
        or not isinstance(residual_checkpoints, Sequence)
    ):
        raise ValueError("long-horizon v4 artifact roster is malformed")
    independent_manifests = tuple(
        record
        for record in manifests
        if isinstance(record, Mapping) and record.get("family") == "independent"
    )
    independent_checkpoints = tuple(
        record
        for record in checkpoints
        if isinstance(record, Mapping) and record.get("family") == "independent"
    )
    residual_records = tuple(
        record for record in residual_checkpoints if isinstance(record, Mapping)
    )
    if (
        len(independent_manifests) != 1
        or len(independent_checkpoints) != 5
        or len(residual_records) != 5
        or tuple(record.get("seed") for record in independent_checkpoints)
        != FROZEN_SEEDS
        or tuple(record.get("seed") for record in residual_records) != FROZEN_SEEDS
    ):
        raise ValueError("long-horizon v4 requires ordered seeds 0 through 4")
    if (
        formal.get("repo") != "gavinlaw/causalcache-gate-v1-formal58-selector-mobile"
        or formal.get("revision")
        != "23f6786075c7bff91f93fd7e8a878e070efb72a9"
        or formal.get("private") is not True
        or residual.get("repo")
        != "gavinlaw/causalcache-set-conditioned-v4-frozen-base-residual-exploration-mobile"
        or residual.get("revision")
        != "1641b90a4ebb05037e4710738a78e2c79f81cdc3"
        or residual.get("private") is not True
    ):
        raise ValueError("long-horizon v4 immutable repository binding drifted")
    for record in (*independent_manifests, *independent_checkpoints, *residual_records):
        digest = record.get("sha256")
        path = record.get("path")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in path.split("/")
            or not _is_sha256(digest)
        ):
            raise ValueError("long-horizon v4 path or SHA256 binding is invalid")
    return MappingProxyType(
        {
            "formal": formal,
            "residual": residual,
            "independent_manifest": independent_manifests[0],
            "independent_checkpoints": independent_checkpoints,
            "residual_checkpoints": residual_records,
        }
    )


def _configure_cpu(seed: int) -> Any:
    if type(seed) is not int or seed not in FROZEN_SEEDS:
        raise ValueError("v4 seed is outside the frozen five seeds")
    torch = _torch()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError("v4 CPU inference requires one inter-op thread") from error
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return torch


def build_zero_residual_head(*, seed: int) -> Any:
    """Build an inference-only exact-zero residual head for replay auditing."""
    torch = _configure_cpu(seed)
    head = FrozenResidualHead().to(device="cpu", dtype=torch.float32)
    torch.nn.init.zeros_(head.network[2].weight)
    torch.nn.init.zeros_(head.network[2].bias)
    return _freeze_residual_head(head)


def _validate_base_architecture(model: Any) -> None:
    torch = _torch()
    if not isinstance(model, torch.nn.Sequential) or len(model) != 5:
        raise ValueError("v4 base must be the gate-v1 independent Sequential")
    first, gelu1, second, gelu2, output = tuple(model)
    if (
        not isinstance(first, torch.nn.Linear)
        or first.in_features != BASE_INPUT_DIMENSION
        or first.out_features != BASE_HIDDEN_DIMENSION
        or not isinstance(gelu1, torch.nn.GELU)
        or gelu1.approximate != "none"
        or not isinstance(second, torch.nn.Linear)
        or second.in_features != BASE_HIDDEN_DIMENSION
        or second.out_features != BASE_HIDDEN_DIMENSION
        or not isinstance(gelu2, torch.nn.GELU)
        or gelu2.approximate != "none"
        or not isinstance(output, torch.nn.Linear)
        or output.in_features != BASE_HIDDEN_DIMENSION
        or output.out_features != 1
        or sum(parameter.numel() for parameter in model.parameters())
        != BASE_PARAMETER_COUNT
    ):
        raise ValueError("v4 frozen independent base architecture drifted")


def _validate_residual_architecture(head: Any) -> None:
    torch = _torch()
    if not isinstance(head, FrozenResidualHead):
        raise ValueError("v4 residual head type drifted")
    first, gelu, output = tuple(head.network)
    if (
        not isinstance(first, torch.nn.Linear)
        or first.in_features != RESIDUAL_INPUT_DIMENSION
        or first.out_features != RESIDUAL_HIDDEN_DIMENSION
        or not isinstance(gelu, torch.nn.GELU)
        or gelu.approximate != "none"
        or not isinstance(output, torch.nn.Linear)
        or output.in_features != RESIDUAL_HIDDEN_DIMENSION
        or output.out_features != 1
        or sum(parameter.numel() for parameter in head.parameters())
        != RESIDUAL_PARAMETER_COUNT
    ):
        raise ValueError("v4 residual architecture drifted")


def _freeze_base(model: Any) -> Any:
    _validate_base_architecture(model)
    model.to(device="cpu", dtype=_torch().float32)
    model.eval()
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    _validate_frozen_module(model, label="independent base")
    return model


def _freeze_residual_head(head: Any) -> Any:
    _validate_residual_architecture(head)
    head.to(device="cpu", dtype=_torch().float32)
    head.eval()
    for parameter in head.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    _validate_frozen_module(head, label="residual head")
    return head


def _validate_frozen_module(module: Any, *, label: str) -> None:
    torch = _torch()
    if module.training:
        raise ValueError(f"{label} must remain in eval mode")
    parameters = tuple(module.parameters())
    if not parameters or any(
        parameter.device.type != "cpu"
        or parameter.dtype != torch.float32
        or parameter.requires_grad
        or parameter.grad is not None
        or not bool(torch.isfinite(parameter).all())
        for parameter in parameters
    ):
        raise ValueError(f"{label} is not frozen finite CPU float32")


def _safetensors_api() -> tuple[Any, Any]:
    try:
        from safetensors.torch import load, save
    except ModuleNotFoundError as error:
        raise RuntimeError("long-horizon v4 requires safetensors") from error
    return load, save


def serialize_residual_checkpoint(head: Any) -> bytes:
    """Canonical metadata-free residual-only safetensors serialization."""
    _validate_residual_architecture(head)
    load, save = _safetensors_api()
    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in sorted(head.state_dict().items())
    }
    torch = _torch()
    if any(
        value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
        for value in tensors.values()
    ):
        raise ValueError("v4 residual checkpoint tensors must be finite float32")
    payload = save(tensors)
    replay = load(payload)
    canonical = {name: replay[name].contiguous() for name in sorted(replay)}
    if save(canonical) != payload or any(
        not bool(tensors[name].equal(canonical[name])) for name in tensors
    ):
        raise RuntimeError("v4 residual checkpoint replay is not canonical")
    return payload


def load_residual_checkpoint(
    payload: bytes,
    *,
    seed: int,
    expected_checkpoint_sha256: str,
    expected_size_bytes: int,
) -> Any:
    """Load and byte-replay one residual-only checkpoint without training."""
    if (
        not isinstance(payload, bytes)
        or type(expected_size_bytes) is not int
        or expected_size_bytes <= 0
        or len(payload) != expected_size_bytes
    ):
        raise ValueError("v4 residual checkpoint size drifted")
    if hashlib.sha256(payload).hexdigest() != expected_checkpoint_sha256:
        raise ValueError("v4 residual checkpoint SHA256 drifted")
    load, save = _safetensors_api()
    tensors = load(payload)
    canonical = {name: tensors[name].contiguous() for name in sorted(tensors)}
    if save(canonical) != payload:
        raise ValueError("v4 residual safetensors encoding is not canonical")
    head = FrozenResidualHead()
    expected_names = set(head.state_dict())
    if set(canonical) != expected_names:
        raise ValueError("v4 residual checkpoint tensor inventory drifted")
    head.load_state_dict(canonical, strict=True)
    head = _freeze_residual_head(head)
    if serialize_residual_checkpoint(head) != payload:
        raise ValueError("v4 residual checkpoint replay changed bytes")
    return head


def _file_inventory(value: Any) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            if {"path", "sha256", "size_bytes"}.issubset(node):
                path = node["path"]
                digest = node["sha256"]
                size = node["size_bytes"]
                if (
                    not isinstance(path, str)
                    or not path
                    or path.startswith("/")
                    or ".." in path.split("/")
                    or not _is_sha256(digest)
                    or type(size) is not int
                    or size <= 0
                ):
                    raise ValueError("v4 manifest file inventory record is malformed")
                previous = result.get(path)
                if previous is not None and dict(previous) != dict(node):
                    raise ValueError("v4 manifest has conflicting file inventory records")
                result[path] = node
            for child in node.values():
                visit(child)
        elif isinstance(node, Sequence) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            for child in node:
                visit(child)

    visit(value)
    return result


def _verified_payload(
    payloads: Mapping[str, bytes],
    record: Mapping[str, Any],
    *,
    label: str,
) -> bytes:
    path = record.get("path")
    try:
        payload = payloads[path]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{label} payload is missing: {path}") from error
    if not isinstance(payload, bytes):
        raise TypeError(f"{label} payload must be bytes")
    if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
        raise ValueError(f"{label} SHA256 drifted: {path}")
    if "size_bytes" in record and len(payload) != record["size_bytes"]:
        raise ValueError(f"{label} size drifted: {path}")
    return payload


def _validate_residual_metadata(
    metadata_payload: bytes,
    *,
    base_records: Sequence[Mapping[str, Any]],
    residual_records: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    metadata = _strict_json(metadata_payload, label="v4 residual metadata")
    if _canonical_json_line(metadata) != metadata_payload:
        raise ValueError("v4 residual metadata is not canonical JSONL")
    if (
        set(metadata) != RESIDUAL_METADATA_KEYS
        or metadata.get("schema_version") != "1.0.0"
        or metadata.get("protocol_id")
        != "causalcache_set_conditioned_v4_frozen_base_residual_development_v1"
        or metadata.get("status")
        != "SERIALIZED_SET_CONDITIONED_V4_FROZEN_BASE_RESIDUAL_ENSEMBLE_V1"
        or metadata.get("device") != "cpu"
        or metadata.get("dtype") != "float32"
        or metadata.get("base_manifest_sha256")
        != "38809f1ef4da9636ef6779733cee4926b36b38e3bd86eab199450e685c63da24"
        or metadata.get("base_parameters_trainable") is not False
        or metadata.get("checkpoint_contains_residual_head_only") is not True
        or metadata.get("fresh_label_access_count") != 0
        or metadata.get("confirm20_access_count") != 0
        or metadata.get("gpu_operation_count") != 0
    ):
        raise ValueError("v4 residual metadata runtime or freeze contract drifted")
    models = metadata.get("models")
    if isinstance(models, (str, bytes, bytearray, Mapping)) or not isinstance(
        models, Sequence
    ):
        raise ValueError("v4 residual metadata model roster is malformed")
    if tuple(record.get("seed") for record in models) != FROZEN_SEEDS:
        raise ValueError("v4 residual metadata seed roster drifted")
    for seed, model, base, residual in zip(
        FROZEN_SEEDS,
        models,
        base_records,
        residual_records,
        strict=True,
    ):
        if not isinstance(model, Mapping):
            raise ValueError("v4 residual metadata model record is malformed")
        path = residual["path"]
        file_record = inventory.get(path)
        if (
            set(model) != RESIDUAL_MODEL_METADATA_KEYS
            or model.get("seed") != seed
            or model.get("selected_epoch") != RESIDUAL_SELECTED_EPOCHS[seed]
            or model.get("learning_rate") != RESIDUAL_LEARNING_RATE
            or model.get("base_checkpoint_sha256") != base.get("sha256")
            or model.get("base_model_state_sha256")
            != base.get("model_state_sha256")
            or model.get("residual_checkpoint_path") != path
            or model.get("residual_checkpoint_sha256") != residual.get("sha256")
            or file_record is None
            or model.get("residual_checkpoint_size_bytes")
            != file_record.get("size_bytes")
        ):
            raise ValueError("v4 residual metadata checkpoint binding drifted")
    return metadata


def load_frozen_v4_ensemble(
    contract: Any,
    *,
    formal_payloads: Mapping[str, bytes],
    residual_payloads: Mapping[str, bytes],
    residual_manifest_payload: bytes,
) -> FrozenV4Ensemble:
    """Load Source-A-bound five-seed base/residual artifacts for CPU inference."""
    roster = frozen_artifact_roster(contract)
    formal = roster["formal"]
    residual = roster["residual"]
    manifest_record = roster["independent_manifest"]
    base_records = tuple(roster["independent_checkpoints"])
    residual_records = tuple(roster["residual_checkpoints"])

    expected_formal_paths = {
        manifest_record["path"],
        *(record["path"] for record in base_records),
    }
    if set(formal_payloads) != expected_formal_paths:
        raise ValueError("v4 formal independent payload inventory drifted")
    formal_manifest_payload = _verified_payload(
        formal_payloads,
        manifest_record,
        label="formal independent manifest",
    )
    provenance = frozen_ensemble_provenance_from_manifest(
        _strict_json(formal_manifest_payload, label="formal independent manifest")
    )
    if (
        provenance.training.family != "independent"
        or provenance.sha256 != manifest_record.get("provenance_sha256")
        or provenance.training.learning_rate != BASE_LEARNING_RATE
        or provenance.training.selected_epochs != BASE_SELECTED_EPOCHS
        or provenance.training.oof_selection_sha256 != BASE_SELECTION_SHA256
    ):
        raise ValueError("v4 formal independent training provenance drifted")

    if (
        not isinstance(residual_manifest_payload, bytes)
        or hashlib.sha256(residual_manifest_payload).hexdigest()
        != residual.get("manifest_sha256")
    ):
        raise ValueError("v4 residual artifact manifest SHA256 drifted")
    residual_manifest = _strict_json(
        residual_manifest_payload,
        label="v4 residual artifact manifest",
    )
    inventory = _file_inventory(residual_manifest)
    expected_residual_paths = {
        RESIDUAL_METADATA_PATH,
        *(record["path"] for record in residual_records),
    }
    if set(residual_payloads) != expected_residual_paths:
        raise ValueError("v4 residual payload inventory drifted")
    metadata_record = inventory.get(RESIDUAL_METADATA_PATH)
    if metadata_record is None:
        raise ValueError("v4 residual manifest omits residual model metadata")
    metadata_payload = _verified_payload(
        residual_payloads,
        metadata_record,
        label="v4 residual metadata",
    )
    _validate_residual_metadata(
        metadata_payload,
        base_records=base_records,
        residual_records=residual_records,
        inventory=inventory,
    )

    seeds: list[FrozenV4Seed] = []
    for seed, base_record, residual_record, bound in zip(
        FROZEN_SEEDS,
        base_records,
        residual_records,
        provenance.checkpoints,
        strict=True,
    ):
        if (
            bound.seed != seed
            or bound.selected_epoch != BASE_SELECTED_EPOCHS[seed]
            or bound.checkpoint_artifact.repository != formal.get("repo")
            or bound.checkpoint_artifact.path != base_record.get("path")
            or bound.checkpoint_artifact.sha256 != base_record.get("sha256")
            or bound.model_state_sha256 != base_record.get("model_state_sha256")
        ):
            raise ValueError("v4 base checkpoint provenance binding drifted")
        base_payload = _verified_payload(
            formal_payloads,
            {**base_record, "size_bytes": BASE_CHECKPOINT_SIZE_BYTES},
            label="v4 frozen base checkpoint",
        )
        base_model = load_safetensors_checkpoint(
            base_payload,
            family="independent",
            seed=seed,
            expected_model_state_sha256=bound.model_state_sha256,
            expected_checkpoint_sha256=bound.checkpoint_artifact.sha256,
        )
        base_model = _freeze_base(base_model)
        residual_file_record = inventory.get(residual_record["path"])
        if (
            residual_file_record is None
            or residual_file_record.get("sha256") != residual_record.get("sha256")
        ):
            raise ValueError("v4 residual manifest checkpoint binding drifted")
        residual_payload = _verified_payload(
            residual_payloads,
            residual_file_record,
            label="v4 residual checkpoint",
        )
        residual_head = load_residual_checkpoint(
            residual_payload,
            seed=seed,
            expected_checkpoint_sha256=residual_record["sha256"],
            expected_size_bytes=residual_file_record["size_bytes"],
        )
        seeds.append(
            freeze_v4_seed(
                seed=seed,
                base_model=base_model,
                residual_head=residual_head,
                base_selected_epoch=bound.selected_epoch,
                base_learning_rate=provenance.training.learning_rate,
                residual_selected_epoch=RESIDUAL_SELECTED_EPOCHS[seed],
                residual_learning_rate=RESIDUAL_LEARNING_RATE,
                base_checkpoint_sha256=base_record["sha256"],
                base_checkpoint_size_bytes=len(base_payload),
                base_model_state_sha256=bound.model_state_sha256,
                residual_checkpoint_sha256=residual_record["sha256"],
                residual_checkpoint_size_bytes=len(residual_payload),
            )
        )
    ensemble = FrozenV4Ensemble(
        seeds=tuple(seeds),
        formal_manifest_sha256=manifest_record["sha256"],
        residual_manifest_sha256=residual["manifest_sha256"],
        residual_metadata_sha256=hashlib.sha256(metadata_payload).hexdigest(),
    )
    validate_frozen_v4_ensemble(ensemble)
    return ensemble


def freeze_v4_seed(
    *,
    seed: int,
    base_model: Any,
    residual_head: Any,
    base_selected_epoch: int,
    base_learning_rate: float,
    residual_selected_epoch: int,
    residual_learning_rate: float,
    base_checkpoint_sha256: str,
    base_checkpoint_size_bytes: int,
    base_model_state_sha256: str,
    residual_checkpoint_sha256: str,
    residual_checkpoint_size_bytes: int,
) -> FrozenV4Seed:
    if (
        seed not in FROZEN_SEEDS
        or base_selected_epoch != BASE_SELECTED_EPOCHS[seed]
        or base_learning_rate != BASE_LEARNING_RATE
        or residual_selected_epoch != RESIDUAL_SELECTED_EPOCHS[seed]
        or residual_learning_rate != RESIDUAL_LEARNING_RATE
    ):
        raise ValueError("v4 seed epoch or learning-rate metadata drifted")
    base_model = _freeze_base(base_model)
    residual_head = _freeze_residual_head(residual_head)
    observed_base = canonical_model_state_sha256(base_model)
    observed_residual = canonical_model_state_sha256(residual_head)
    if observed_base != base_model_state_sha256:
        raise ValueError("v4 base model-state SHA256 drifted")
    value = FrozenV4Seed(
        seed=seed,
        base_selected_epoch=base_selected_epoch,
        base_learning_rate=base_learning_rate,
        residual_selected_epoch=residual_selected_epoch,
        residual_learning_rate=residual_learning_rate,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_checkpoint_size_bytes=base_checkpoint_size_bytes,
        base_model_state_sha256=base_model_state_sha256,
        residual_checkpoint_sha256=residual_checkpoint_sha256,
        residual_checkpoint_size_bytes=residual_checkpoint_size_bytes,
        residual_model_state_sha256=observed_residual,
        base_model=base_model,
        residual_head=residual_head,
    )
    validate_frozen_v4_seed(value)
    return value


def validate_frozen_v4_seed(value: FrozenV4Seed) -> None:
    if not isinstance(value, FrozenV4Seed):
        raise TypeError("value must be a FrozenV4Seed")
    if (
        value.seed not in FROZEN_SEEDS
        or value.base_selected_epoch != BASE_SELECTED_EPOCHS[value.seed]
        or value.base_learning_rate != BASE_LEARNING_RATE
        or value.residual_selected_epoch != RESIDUAL_SELECTED_EPOCHS[value.seed]
        or value.residual_learning_rate != RESIDUAL_LEARNING_RATE
        or value.base_checkpoint_size_bytes != BASE_CHECKPOINT_SIZE_BYTES
        or type(value.residual_checkpoint_size_bytes) is not int
        or value.residual_checkpoint_size_bytes <= 0
        or not _is_sha256(value.base_checkpoint_sha256)
        or not _is_sha256(value.base_model_state_sha256)
        or not _is_sha256(value.residual_checkpoint_sha256)
        or not _is_sha256(value.residual_model_state_sha256)
    ):
        raise ValueError("frozen v4 seed metadata drifted")
    _validate_frozen_module(value.base_model, label="independent base")
    _validate_frozen_module(value.residual_head, label="residual head")
    _validate_base_architecture(value.base_model)
    _validate_residual_architecture(value.residual_head)
    if canonical_model_state_sha256(value.base_model) != value.base_model_state_sha256:
        raise ValueError("frozen v4 base state changed")
    if (
        canonical_model_state_sha256(value.residual_head)
        != value.residual_model_state_sha256
    ):
        raise ValueError("frozen v4 residual state changed")


def validate_frozen_v4_ensemble(value: FrozenV4Ensemble) -> None:
    if not isinstance(value, FrozenV4Ensemble):
        raise TypeError("value must be a FrozenV4Ensemble")
    if (
        tuple(seed.seed for seed in value.seeds) != FROZEN_SEEDS
        or len({id(seed.base_model) for seed in value.seeds}) != 5
        or len({id(seed.residual_head) for seed in value.seeds}) != 5
        or value.device != "cpu"
        or value.dtype != "float32"
        or value.training_allowed is not False
        or not _is_sha256(value.formal_manifest_sha256)
        or not _is_sha256(value.residual_manifest_sha256)
        or not _is_sha256(value.residual_metadata_sha256)
    ):
        raise ValueError("frozen v4 ensemble inventory or runtime drifted")
    for seed in value.seeds:
        validate_frozen_v4_seed(seed)


def _independent_input(view: Any, event_step_id: int) -> tuple[float, ...]:
    candidate = view.candidate(event_step_id)
    vector = (
        *view.q64,
        *candidate.h64,
        *(
            left * right
            for left, right in zip(view.q64, candidate.h64, strict=True)
        ),
        *candidate.g8,
    )
    if len(vector) != BASE_INPUT_DIMENSION or any(
        not math.isfinite(float(value)) for value in vector
    ):
        raise ValueError("v4 independent input is not 200 finite values")
    return tuple(float(value) for value in vector)


def _base_forward(
    seed: FrozenV4Seed,
    state: Any,
) -> tuple[Any, Any, Any, tuple[tuple[int, int], ...]]:
    validate_frozen_v4_seed(seed)
    view = feature_view(state)
    event_ids = view.candidate_event_step_ids
    if len(event_ids) not in SUPPORTED_CANDIDATE_COUNTS:
        raise ValueError("v4 long-horizon inference requires n=8 or n=16")
    pairs = tuple(itertools.combinations(event_ids, 2))
    torch = _torch()
    inputs = torch.tensor(
        [_independent_input(view, event) for event in event_ids],
        dtype=torch.float32,
        device="cpu",
    )
    with torch.inference_mode():
        hidden = seed.base_model[0](inputs)
        hidden = seed.base_model[1](hidden)
        hidden = seed.base_model[2](hidden)
        hidden = seed.base_model[3](hidden)
        singleton = seed.base_model[4](hidden).squeeze(-1)
    if (
        hidden.shape != (len(event_ids), BASE_HIDDEN_DIMENSION)
        or singleton.shape != (len(event_ids),)
        or not bool(torch.isfinite(hidden).all())
        or not bool(torch.isfinite(singleton).all())
    ):
        raise ValueError("v4 frozen-base prediction geometry or values drifted")
    return view, hidden, singleton, pairs


def predict_seed_scores(seed: FrozenV4Seed, state: Any) -> ResidualSeedScores:
    """Predict all singleton and chronological pair corrections for one seed."""
    before_base = seed.base_model_state_sha256
    before_residual = seed.residual_model_state_sha256
    view, hidden, singleton, pairs = _base_forward(seed, state)
    event_ids = view.candidate_event_step_ids
    index = {event: position for position, event in enumerate(event_ids)}
    torch = _torch()
    left = torch.tensor([index[pair[0]] for pair in pairs], dtype=torch.long)
    right = torch.tensor([index[pair[1]] for pair in pairs], dtype=torch.long)
    q64 = torch.tensor(
        [view.q64 for _ in pairs],
        dtype=torch.float32,
        device="cpu",
    )
    with torch.inference_mode():
        residual = seed.residual_head(hidden[left], hidden[right], q64)
    if residual.shape != (len(pairs),) or not bool(torch.isfinite(residual).all()):
        raise ValueError("v4 residual prediction geometry or values drifted")
    if seed.residual_selected_epoch == 0 and not bool(residual.equal(torch.zeros_like(residual))):
        raise ValueError("epoch-zero v4 checkpoint is not an exact-zero residual")
    if (
        canonical_model_state_sha256(seed.base_model) != before_base
        or canonical_model_state_sha256(seed.residual_head) != before_residual
    ):
        raise RuntimeError("v4 inference mutated a frozen model")
    return ResidualSeedScores(
        seed=seed.seed,
        singleton_scores=MappingProxyType(
            {
                event: float(singleton[position])
                for position, event in enumerate(event_ids)
            }
        ),
        pair_residual_scores=MappingProxyType(
            {pair: float(residual[position]) for position, pair in enumerate(pairs)}
        ),
    )


def zero_residual_seed_scores(seed: FrozenV4Seed, state: Any) -> ResidualSeedScores:
    """Replay the exact frozen independent base with every pair correction zero."""
    before = seed.base_model_state_sha256
    view, _hidden, singleton, pairs = _base_forward(seed, state)
    if canonical_model_state_sha256(seed.base_model) != before:
        raise RuntimeError("zero-residual replay mutated the independent base")
    return ResidualSeedScores(
        seed=seed.seed,
        singleton_scores=MappingProxyType(
            {
                event: float(singleton[position])
                for position, event in enumerate(view.candidate_event_step_ids)
            }
        ),
        pair_residual_scores=MappingProxyType({pair: 0.0 for pair in pairs}),
    )


def predict_residual_seed_scores(
    ensemble: FrozenV4Ensemble, state: Any
) -> tuple[ResidualSeedScores, ...]:
    """Return the five ordered records consumed by ``residual_b2_selections``."""
    validate_frozen_v4_ensemble(ensemble)
    result = tuple(predict_seed_scores(seed, state) for seed in ensemble.seeds)
    expected_pairs = 28 if len(feature_view(state).candidate_event_step_ids) == 8 else 120
    if (
        tuple(item.seed for item in result) != FROZEN_SEEDS
        or any(len(item.pair_residual_scores) != expected_pairs for item in result)
    ):
        raise RuntimeError("v4 long-horizon prediction inventory drifted")
    return result


__all__ = [
    "BASE_CHECKPOINT_SIZE_BYTES",
    "BASE_LEARNING_RATE",
    "BASE_SELECTED_EPOCHS",
    "FrozenResidualHead",
    "FrozenV4Ensemble",
    "FrozenV4Seed",
    "RESIDUAL_LEARNING_RATE",
    "RESIDUAL_SELECTED_EPOCHS",
    "build_zero_residual_head",
    "freeze_v4_seed",
    "frozen_artifact_roster",
    "load_frozen_v4_ensemble",
    "load_residual_checkpoint",
    "predict_residual_seed_scores",
    "predict_seed_scores",
    "serialize_residual_checkpoint",
    "validate_frozen_v4_ensemble",
    "validate_frozen_v4_seed",
    "zero_residual_seed_scores",
]
