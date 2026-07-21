"""Generic at-most-budget deployment search for direct marginal selectors."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FIXED_ZERO_STOP = "fixed_zero_gain"
LEARNED_STOP = "learned_threshold"
DIRECT_MARGINAL_STOP_SEMANTICS = (FIXED_ZERO_STOP, LEARNED_STOP)
DIRECT_MARGINAL_REPLAY_SCHEMA_VERSION = "1.0.0"
DIRECT_MARGINAL_REPLAY_STATUS = (
    "FROZEN_DIRECT_MARGINAL_TRAIN_HELDOUT_DEVELOPMENT_REPLAY"
)
SUPPORTED_DIRECT_MARGINAL_REPLAY_ROLES = ("train_heldout", "development")
DIRECT_MARGINAL_MODEL_FAMILIES = (
    "set_transformer_direct_marginal",
    "deepsets_structured_marginal",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed_content_hash(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    payload = json.dumps(
        unsigned,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


@dataclass(frozen=True)
class DirectMarginalReplayBinding:
    """Versioned checkpoint binding for train-heldout or development replay."""

    checkpoint_sha256: str
    content_sha256: str
    model_family: str
    replay_role: str
    schema_version: str
    selector_config_sha256: str
    status: str
    variant: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DirectMarginalReplayBinding":
        expected = {
            "checkpoint_sha256",
            "content_sha256",
            "model_family",
            "replay_role",
            "schema_version",
            "selector_config_sha256",
            "status",
            "variant",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("direct marginal replay binding field inventory drifted")
        if value.get("content_sha256") != _signed_content_hash(value):
            raise ValueError("direct marginal replay binding signature drifted")
        if value.get("schema_version") != DIRECT_MARGINAL_REPLAY_SCHEMA_VERSION:
            raise ValueError("direct marginal replay binding schema drifted")
        if value.get("status") != DIRECT_MARGINAL_REPLAY_STATUS:
            raise ValueError("direct marginal replay binding status drifted")
        if value.get("replay_role") not in SUPPORTED_DIRECT_MARGINAL_REPLAY_ROLES:
            raise ValueError(
                "direct marginal replay role must be train-heldout or development"
            )
        if value.get("model_family") not in DIRECT_MARGINAL_MODEL_FAMILIES:
            raise ValueError("direct marginal replay model family is unsupported")
        variant = value.get("variant")
        if not isinstance(variant, str) or not variant:
            raise ValueError("direct marginal replay variant must be non-empty")
        return cls(
            checkpoint_sha256=_sha256(
                value.get("checkpoint_sha256"), label="checkpoint SHA256"
            ),
            content_sha256=_sha256(
                value.get("content_sha256"), label="binding content SHA256"
            ),
            model_family=str(value["model_family"]),
            replay_role=str(value["replay_role"]),
            schema_version=str(value["schema_version"]),
            selector_config_sha256=_sha256(
                value.get("selector_config_sha256"),
                label="selector config SHA256",
            ),
            status=str(value["status"]),
            variant=variant,
        )


def load_direct_marginal_replay_binding(
    path: Path,
) -> DirectMarginalReplayBinding:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("direct marginal replay binding must be one JSON object")
    return DirectMarginalReplayBinding.from_mapping(value)


def _event_ids(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError("direct marginal event ids must be a sequence")
    result = tuple(values)
    if (
        not result
        or any(type(value) is not int or value <= 0 for value in result)
        or result != tuple(sorted(result))
        or len(result) != len(set(result))
    ):
        raise ValueError(
            "direct marginal event ids must be sorted unique positive integers"
        )
    return result


@dataclass(frozen=True)
class DirectMarginalSelectionPath:
    """One nested at-most-B path with the model's native STOP semantics."""

    budget: int
    predicted_utilities: Mapping[int, float]
    scored_subsets: tuple[tuple[tuple[int, ...], float], ...]
    score_count: int
    selections: Mapping[int, tuple[int, ...]]
    stop_semantics: str
    trace: tuple[Mapping[str, object], ...]

    def selection(self, budget: int) -> tuple[int, ...]:
        if type(budget) is not int or budget not in self.selections:
            raise ValueError("requested budget is absent from the selection path")
        return self.selections[budget]

    def predicted_utility(self, budget: int) -> float:
        if type(budget) is not int or budget not in self.predicted_utilities:
            raise ValueError("requested budget is absent from the selection path")
        return self.predicted_utilities[budget]


def direct_marginal_at_most_budget_path(
    event_ids: Sequence[int],
    *,
    budget: int,
    score: Callable[[tuple[int, ...]], Sequence[float]],
    stop_semantics: str,
) -> DirectMarginalSelectionPath:
    """Select nested B1--B events from ``[STOP, event_1, ..., event_n]`` scores.

    ``fixed_zero_gain`` is the Set Transformer contract: STOP must be exactly
    zero. ``learned_threshold`` is the structured DeepSets contract: the
    learned STOP logit is compared directly with every remaining marginal.
    Equality always stops, which preserves the at-most-budget tie rule.
    """
    events = _event_ids(event_ids)
    if type(budget) is not int or budget not in (1, 2, 3, 4):
        raise ValueError("direct marginal budget must be one of 1, 2, 3, 4")
    if not callable(score):
        raise TypeError("direct marginal scorer must be callable")
    if stop_semantics not in DIRECT_MARGINAL_STOP_SEMANTICS:
        raise ValueError("direct marginal STOP semantics are unsupported")

    selected: list[int] = []
    cumulative = 0.0
    stopped = False
    score_count = 0
    selections: dict[int, tuple[int, ...]] = {}
    utilities: dict[int, float] = {}
    scored_subsets: list[tuple[tuple[int, ...], float]] = [((), cumulative)]
    trace: list[Mapping[str, object]] = []
    event_index = {event_id: index for index, event_id in enumerate(events)}

    for current_budget in range(1, budget + 1):
        if not stopped and len(selected) < min(current_budget, len(events)):
            values = tuple(float(value) for value in score(tuple(selected)))
            if len(values) != len(events) + 1 or any(
                not math.isfinite(value) for value in values
            ):
                raise ValueError(
                    "direct marginal scorer must return finite STOP-plus-event scores"
                )
            stop_score = values[0]
            if stop_semantics == FIXED_ZERO_STOP and stop_score != 0.0:
                raise ValueError("Set Transformer STOP score must remain exactly zero")
            remaining = tuple(
                event_id for event_id in events if event_id not in selected
            )
            best_event, best_marginal = min(
                (
                    (event_id, values[event_index[event_id] + 1])
                    for event_id in remaining
                ),
                key=lambda item: (-item[1], item[0]),
            )
            score_count += len(remaining) + 1
            accepted = best_marginal > stop_score
            trace.append(
                {
                    "accepted": accepted,
                    "base_subset": tuple(selected),
                    "best_event": best_event,
                    "best_marginal": best_marginal,
                    "stop_score": stop_score,
                }
            )
            if accepted:
                selected.append(best_event)
                selected.sort()
                cumulative += best_marginal
                scored_subsets.append((tuple(selected), cumulative))
            else:
                stopped = True
        selections[current_budget] = tuple(selected)
        utilities[current_budget] = cumulative

    return DirectMarginalSelectionPath(
        budget=budget,
        predicted_utilities=utilities,
        scored_subsets=tuple(scored_subsets),
        score_count=score_count,
        selections=selections,
        stop_semantics=stop_semantics,
        trace=tuple(trace),
    )


def load_train_development_direct_marginal_backend(
    *,
    repository_root: Path,
    binding_path: Path,
    selector_config_path: Path,
    checkpoint_path: Path,
    model_dir: Path,
    device: str,
) -> Any:
    """Load a current marginal checkpoint without entering the signed GO path.

    This loader is deliberately separate from
    ``load_go_authorized_live_selector_service``. It accepts only a newly signed
    train-heldout/development binding and therefore cannot reinterpret the old
    320-state native-replay authorization.
    """
    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise ValueError("direct marginal replay requires an explicit CUDA device")
    binding = load_direct_marginal_replay_binding(binding_path.resolve())
    selector_config_path = selector_config_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    if _sha256_file(selector_config_path) != binding.selector_config_sha256:
        raise ValueError("direct marginal replay selector config SHA256 drifted")
    if _sha256_file(checkpoint_path) != binding.checkpoint_sha256:
        raise ValueError("direct marginal replay checkpoint SHA256 drifted")
    config = json.loads(selector_config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("direct marginal replay selector config must be an object")
    variants = config.get("variants")
    if not isinstance(variants, Mapping) or not isinstance(
        variants.get(binding.variant), Mapping
    ):
        raise ValueError("direct marginal replay variant is absent from its config")
    variant = variants[binding.variant]

    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise RuntimeError("direct marginal replay requires safetensors") from error
    from causalcache.policy.gui_owl_variable_history_runtime import (
        GUIOwlVariableHistoryRuntime,
        VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    from causalcache.set_utility_direct_on_policy_evaluator import (
        load_set_transformer_selector,
        load_structured_deepsets_selector,
    )
    from causalcache.set_utility_live_controller import (
        GUIOwlLiveRichSourceEncoder,
        TorchDirectMarginalReplayBackend,
    )

    loader = {
        "set_transformer_direct_marginal": load_set_transformer_selector,
        "deepsets_structured_marginal": load_structured_deepsets_selector,
    }[binding.model_family]
    adapter = loader(
        variant=variant,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=binding.checkpoint_sha256,
        device=device,
        load_file=load_file,
    )
    expected_stop = {
        "set_transformer_direct_marginal": FIXED_ZERO_STOP,
        "deepsets_structured_marginal": LEARNED_STOP,
    }[binding.model_family]
    if (
        adapter.model_family != binding.model_family
        or adapter.stop_semantics != expected_stop
    ):
        raise ValueError("direct marginal adapter semantics drifted from its binding")

    repository_root = repository_root.resolve()
    runtime = GUIOwlVariableHistoryRuntime(
        model_dir=model_dir.resolve(),
        expected_snapshot_manifest=(
            repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device=device,
        target_effective_visual_tokens_per_image=(
            VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
        ),
    )
    backend = TorchDirectMarginalReplayBackend(
        model=adapter,
        source_encoder=GUIOwlLiveRichSourceEncoder(runtime),
        stop_semantics=adapter.stop_semantics,
    )
    backend.replay_binding = binding
    return backend


__all__ = [
    "DIRECT_MARGINAL_REPLAY_SCHEMA_VERSION",
    "DIRECT_MARGINAL_REPLAY_STATUS",
    "DIRECT_MARGINAL_STOP_SEMANTICS",
    "DIRECT_MARGINAL_MODEL_FAMILIES",
    "DirectMarginalReplayBinding",
    "DirectMarginalSelectionPath",
    "FIXED_ZERO_STOP",
    "LEARNED_STOP",
    "SUPPORTED_DIRECT_MARGINAL_REPLAY_ROLES",
    "direct_marginal_at_most_budget_path",
    "load_direct_marginal_replay_binding",
    "load_train_development_direct_marginal_backend",
]
