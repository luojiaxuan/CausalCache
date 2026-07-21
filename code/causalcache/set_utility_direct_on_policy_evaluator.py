"""Label-blind unified inference for the two formal marginal selectors."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_evaluation import percentile_type7
from causalcache.set_utility_heldout_inference import BUDGETS
from causalcache.set_utility_direct_marginal_replay import (
    FIXED_ZERO_STOP,
    LEARNED_STOP,
)
from causalcache.set_utility_structured_marginal import (
    StructuredMarginalHeadConfig,
    StructuredTokenConditionalMarginalPredictor,
)
from causalcache.set_utility_token_models import (
    EncodedConditionalMarginalState,
    TokenConditionalMarginalPredictor,
    TokenUtilityModelConfig,
)


HYBRID_THRESHOLD = 0.001
METHODS = (
    "recent",
    "set_transformer",
    "set_transformer_hybrid",
    "structured_deepsets",
    "structured_deepsets_hybrid",
)
FEATURE_KEYS = (
    "candidate_event_step_ids",
    "current_image_key",
    "event_image_keys",
    "event_numeric_features",
    "event_text_keys",
    "instruction_text_key",
    "state_id",
    "trajectory_id",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_inference_state(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only model features; restoration distances are never requested."""
    result = {key: row[key] for key in FEATURE_KEYS}
    candidates = tuple(result["candidate_event_step_ids"])
    if (
        not candidates
        or candidates != tuple(sorted(candidates))
        or len(candidates) != len(set(candidates))
        or any(type(value) is not int or value <= 0 for value in candidates)
    ):
        raise ValueError("inference candidate universe is invalid")
    count = len(candidates)
    if not (
        len(result["event_image_keys"])
        == len(result["event_text_keys"])
        == len(result["event_numeric_features"])
        == count
    ):
        raise ValueError("inference event features are not candidate-aligned")
    if any(
        not isinstance(value, str) or not value
        for value in (
            result["current_image_key"],
            result["instruction_text_key"],
            result["state_id"],
            result["trajectory_id"],
            *result["event_image_keys"],
            *result["event_text_keys"],
        )
    ):
        raise ValueError("inference feature identity is invalid")
    for features in result["event_numeric_features"]:
        if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
            raise ValueError("inference numeric features must be sequences")
        if not features or any(not math.isfinite(float(value)) for value in features):
            raise ValueError("inference numeric features are invalid")
    result["candidate_event_step_ids"] = list(candidates)
    return result


def recent_budget_path(event_ids: Sequence[int]) -> dict[str, list[int]]:
    candidates = tuple(event_ids)
    return {
        str(budget): list(candidates[-min(budget, len(candidates)) :])
        for budget in BUDGETS
    }


def marginal_budget_path(
    event_ids: Sequence[int],
    *,
    score: Callable[[tuple[int, ...]], Sequence[float]],
    stop_semantics: str | None = None,
    recent_fallback_threshold: float | None = None,
) -> dict[str, Any]:
    """Select nested at-most-B prefixes while honoring the model's STOP score."""
    candidates = tuple(event_ids)
    if (
        not candidates
        or candidates != tuple(sorted(candidates))
        or len(candidates) != len(set(candidates))
    ):
        raise ValueError("marginal selector candidate universe is invalid")
    if recent_fallback_threshold is not None and (
        not math.isfinite(float(recent_fallback_threshold))
        or float(recent_fallback_threshold) < 0.0
    ):
        raise ValueError("recent fallback threshold is invalid")
    selected: list[int] = []
    stopped = False
    selections: dict[str, list[int]] = {}
    trace = []
    index_by_event = {event_id: index for index, event_id in enumerate(candidates)}
    for budget in BUDGETS:
        if not stopped and len(selected) < min(budget, len(candidates)):
            values = tuple(float(value) for value in score(tuple(selected)))
            if len(values) != len(candidates) + 1 or any(
                not math.isfinite(value) for value in values
            ):
                raise ValueError("marginal selector scores are invalid")
            if stop_semantics == FIXED_ZERO_STOP and values[0] != 0.0:
                raise ValueError("Set Transformer STOP score must remain zero")
            if stop_semantics not in (None, FIXED_ZERO_STOP, LEARNED_STOP):
                raise ValueError("marginal selector STOP semantics are invalid")
            remaining = tuple(
                event_id for event_id in candidates if event_id not in selected
            )
            best_event, best_marginal = min(
                (
                    (event_id, values[index_by_event[event_id] + 1])
                    for event_id in remaining
                ),
                key=lambda item: (-item[1], item[0]),
            )
            recent_event = max(remaining)
            recent_marginal = values[index_by_event[recent_event] + 1]
            stop_threshold = values[0]
            if recent_fallback_threshold is None:
                if best_marginal <= stop_threshold:
                    chosen_event = None
                    reason = "native_stop"
                    stopped = True
                else:
                    chosen_event = best_event
                    reason = "learned"
            else:
                threshold = float(recent_fallback_threshold)
                if stop_threshold - best_marginal >= threshold:
                    chosen_event = None
                    reason = "confidence_gated_stop"
                    stopped = True
                elif (
                    best_event != recent_event
                    and best_marginal - recent_marginal >= threshold
                    and best_marginal - stop_threshold >= threshold
                ):
                    chosen_event = best_event
                    reason = "confidence_gated_learned_override"
                else:
                    chosen_event = recent_event
                    reason = "recent_fallback"
            trace.append(
                {
                    "base_subset": list(selected),
                    "best_event": best_event,
                    "best_marginal": best_marginal,
                    "chosen_event": chosen_event if chosen_event is not None else "STOP",
                    "learned_advantage_over_recent": best_marginal - recent_marginal,
                    "reason": reason,
                    "recent_event": recent_event,
                    "recent_marginal": recent_marginal,
                    "stop_threshold": stop_threshold,
                }
            )
            if chosen_event is not None:
                selected.append(chosen_event)
                selected.sort()
        selections[str(budget)] = list(selected)
    return {"selections": selections, "trace": trace}


@dataclass(frozen=True)
class FrozenSelectorAdapter:
    """Common source-cache and marginal-scoring API for policy integration."""

    name: str
    model_family: str
    model: Any
    stop_semantics: str

    @property
    def source_encoder(self) -> Any:
        if self.model_family == "set_transformer_direct_marginal":
            return self.model.encoder
        if self.model_family == "deepsets_structured_marginal":
            return self.model.source_encoder
        raise ValueError("unknown frozen selector family")

    def encode_query_source(self, **inputs: Any) -> Any:
        return self.source_encoder.encode_query_source(**inputs)

    def encode_event_sources(self, **inputs: Any) -> Any:
        return self.source_encoder.encode_event_sources(**inputs)

    def condition_encoded_state(self, **inputs: Any) -> EncodedConditionalMarginalState:
        if self.model_family == "set_transformer_direct_marginal":
            return self.model.condition_encoded_state(**inputs)
        encoded = self.model.source_encoder.condition_encoded_state(**inputs)
        return EncodedConditionalMarginalState(
            query=encoded.query,
            events=encoded.events,
            event_mask=encoded.event_mask,
        )

    def score_encoded_candidates(self, encoded: Any, selected_masks: Any) -> Any:
        return self.model.score_encoded_candidates(encoded, selected_masks)


def load_set_transformer_selector(
    *,
    variant: Mapping[str, Any],
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    device: str,
    load_file: Callable[..., Mapping[str, Any]],
) -> FrozenSelectorAdapter:
    model_config = variant.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("Set Transformer variant omits its model config")
    config = TokenUtilityModelConfig(**model_config)
    if config.family != "set_transformer" or not config.preserve_entity_latents:
        raise ValueError("Set Transformer selector architecture drifted")
    if sha256_file(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("Set Transformer checkpoint SHA256 drifted")
    model = TokenConditionalMarginalPredictor(config).to(device)
    model.load_state_dict(load_file(str(checkpoint_path), device=device), strict=True)
    model.eval()
    return FrozenSelectorAdapter(
        name="set_transformer",
        model_family="set_transformer_direct_marginal",
        model=model,
        stop_semantics=FIXED_ZERO_STOP,
    )


def load_structured_deepsets_selector(
    *,
    variant: Mapping[str, Any],
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    device: str,
    load_file: Callable[..., Mapping[str, Any]],
) -> FrozenSelectorAdapter:
    encoder_config = variant.get("encoder")
    head_config = variant.get("head")
    if not isinstance(encoder_config, Mapping) or not isinstance(head_config, Mapping):
        raise ValueError("structured DeepSets variant omits its architecture")
    encoder = TokenUtilityModelConfig(**encoder_config)
    if encoder.family != "deepsets" or encoder.preserve_entity_latents:
        raise ValueError("structured DeepSets selector architecture drifted")
    model = StructuredTokenConditionalMarginalPredictor(
        encoder, StructuredMarginalHeadConfig(**head_config)
    ).to(device)
    if sha256_file(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("structured DeepSets checkpoint SHA256 drifted")
    model.load_state_dict(load_file(str(checkpoint_path), device=device), strict=True)
    model.eval()
    return FrozenSelectorAdapter(
        name="structured_deepsets",
        model_family="deepsets_structured_marginal",
        model=model,
        stop_semantics=LEARNED_STOP,
    )


def validate_expected_rollout(
    records: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Any],
    *,
    expected_checkpoint_sha256: str,
) -> None:
    checkpoint = expected.get("checkpoint")
    expected_records = expected.get("records")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("sha256") != expected_checkpoint_sha256
        or not isinstance(expected_records, Sequence)
    ):
        raise ValueError("expected epoch rollout checkpoint binding drifted")
    by_state = {str(row["state_id"]): row for row in expected_records}
    if len(by_state) != len(expected_records) or set(by_state) != {
        str(row["state_id"]) for row in records
    }:
        raise ValueError("expected epoch rollout state inventory drifted")
    for record in records:
        frozen = by_state[str(record["state_id"])]
        if (
            tuple(frozen.get("candidate_event_ids", ()))
            != tuple(record["candidate_event_ids"])
            or frozen.get("selections") != record["methods"][record["model_name"]]
        ):
            raise ValueError(
                f"native selector differs from frozen epoch rollout: {record['state_id']}"
            )


def latency_summary(
    records: Sequence[Mapping[str, Any]], model_names: Sequence[str]
) -> dict[str, Any]:
    result = {}
    for model_name in model_names:
        components: dict[str, list[float]] = {}
        for record in records:
            for key, raw_value in record["latency_ms"][model_name].items():
                value = float(raw_value)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError("selector latency is invalid")
                components.setdefault(key, []).append(value)
        result[model_name] = {
            key: {
                "p50_ms": percentile_type7(values, 0.5),
                "p95_ms": percentile_type7(values, 0.95),
                "state_count": len(values),
            }
            for key, values in sorted(components.items())
        }
    return result


__all__ = [
    "FEATURE_KEYS",
    "FrozenSelectorAdapter",
    "HYBRID_THRESHOLD",
    "METHODS",
    "latency_summary",
    "load_set_transformer_selector",
    "load_structured_deepsets_selector",
    "marginal_budget_path",
    "project_inference_state",
    "recent_budget_path",
    "sha256_file",
    "validate_expected_rollout",
]
