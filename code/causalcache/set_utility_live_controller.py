"""GO-gated live rich-token selection for an AndroidWorld controller.

The transport contract is stateless: every request carries the complete prior
event history.  This makes an Aries emulator client resumable while an H200
service owns the frozen GUI-Owl encoder and the held-out winner checkpoint.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.low_fidelity_v2 import (
    LowFidelityEventV2,
    serialize_low_fidelity_v2,
)
from causalcache.set_utility_direct_marginal_replay import (
    DIRECT_MARGINAL_STOP_SEMANTICS,
    DirectMarginalSelectionPath,
    direct_marginal_at_most_budget_path,
)
from causalcache.set_utility_heldout_evaluation import COMPLETED_EVALUATION
from causalcache.set_utility_label_producer import (
    MixedFidelityPromptPlan,
    PromptHistoryEvent,
)
from causalcache.set_utility_native_replay import signed_content_hash_is_valid
from causalcache.set_utility_native_replay_evaluation import (
    COMPLETED_NATIVE_REPLAY_EVALUATION,
)
from causalcache.set_utility_processor_prompt import (
    build_set_utility_gui_owl_v2_1_messages,
)
from causalcache.set_utility_search import (
    conditional_greedy_at_most_budget_search,
)
from causalcache.set_utility_variable_history_training import (
    variable_history_event_numeric_features,
)


LIVE_SELECTOR_SCHEMA_VERSION = "1.0.0"
LIVE_SELECTOR_REQUEST_STATUS = "CAUSALCACHE_LIVE_RICH_SELECTOR_REQUEST"
LIVE_SELECTOR_RESPONSE_STATUS = "CAUSALCACHE_LIVE_RICH_SELECTOR_RESPONSE"
LIVE_SELECTOR_AUTHORIZATION_STATUS = "AUTHORIZED_BY_HELDOUT_AND_NATIVE_REPLAY_GO"
SUPPORTED_BUDGETS = (1, 2, 3, 4)
MINIMUM_SUPPORTED_DECISION_STEP = 6
SOURCE_HIDDEN_SIZE = 4096
MAXIMUM_TEXT_TOKENS = 128
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _signed_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def _content_hash_is_valid(value: Mapping[str, Any]) -> bool:
    claimed = value.get("content_sha256")
    if not isinstance(claimed, str):
        return False
    unsigned = dict(value)
    del unsigned["content_sha256"]
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() == claimed


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} field inventory drifted")


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _sha256_text(value: Any, label: str) -> str:
    text = _nonempty_text(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return text


def _ocr_tokens(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be an ordered text array")
    result = tuple(value)
    if any(not isinstance(token, str) or not token for token in result):
        raise ValueError(f"{label} contains an invalid token")
    return result


@dataclass(frozen=True)
class LiveSelectorAuthorization:
    authorization_id: str
    checkpoint_sha256: str
    config_sha256: str
    heldout_result_sha256: str
    model_config_sha256: str
    model_variant: str
    native_replay_result_sha256: str
    selections_sha256: str
    status: str
    winner_model: str

    def __post_init__(self) -> None:
        for name in (
            "authorization_id",
            "checkpoint_sha256",
            "config_sha256",
            "heldout_result_sha256",
            "model_config_sha256",
            "native_replay_result_sha256",
            "selections_sha256",
        ):
            _sha256_text(getattr(self, name), name)
        _nonempty_text(self.model_variant, "model variant")
        if self.winner_model not in {"deepsets", "set_transformer"}:
            raise ValueError("held-out winner model is unsupported")
        if self.status != LIVE_SELECTOR_AUTHORIZATION_STATUS:
            raise ValueError("live selector is not GO-authorized")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authorization_id": self.authorization_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "heldout_result_sha256": self.heldout_result_sha256,
            "model_config_sha256": self.model_config_sha256,
            "model_variant": self.model_variant,
            "native_replay_result_sha256": self.native_replay_result_sha256,
            "selections_sha256": self.selections_sha256,
            "status": self.status,
            "winner_model": self.winner_model,
        }


def authorize_live_selector(
    *,
    heldout_config_path: Path,
    selections_path: Path,
    heldout_result_path: Path,
    native_replay_result_path: Path,
    checkpoint_path: Path,
) -> LiveSelectorAuthorization:
    """Bind the service to signed held-out and native-replay GO results."""
    config = _read_mapping(heldout_config_path)
    selections = _read_mapping(selections_path)
    result = _read_mapping(heldout_result_path)
    native_replay = _read_mapping(native_replay_result_path)
    config_sha = sha256_file(heldout_config_path)
    selections_sha = sha256_file(selections_path)
    result_sha = sha256_file(heldout_result_path)
    native_replay_sha = sha256_file(native_replay_result_path)
    if (
        selections.get("status") != "SEALED_SET_UTILITY_HELDOUT_SELECTIONS"
        or not signed_content_hash_is_valid(selections)
        or selections.get("config_sha256") != config_sha
    ):
        raise ValueError("live selector requires sealed selections for this config")
    heldout_coverage = result.get("coverage")
    heldout_winner = result.get("winner")
    heldout_bindings = result.get("bindings")
    if (
        result.get("status") != COMPLETED_EVALUATION
        or not isinstance(heldout_coverage, Mapping)
        or heldout_coverage.get("complete") is not True
        or heldout_coverage.get("completed_state_count")
        != heldout_coverage.get("expected_state_count")
        or heldout_coverage.get("missing_state_ids") not in ([], ())
        or heldout_coverage.get("skipped_state_count") != 0
        or not isinstance(heldout_winner, Mapping)
        or heldout_winner.get("verdict") != "GO"
        or not signed_content_hash_is_valid(result)
        or not isinstance(heldout_bindings, Mapping)
        or heldout_bindings.get("config_sha256") != config_sha
        or heldout_bindings.get("selections_sha256") != selections_sha
    ):
        raise ValueError("live selector requires a complete signed held-out GO")
    winner = heldout_winner.get("model")
    native_coverage = native_replay.get("coverage")
    native_bindings = native_replay.get("bindings")
    native_latency_gate = native_replay.get("deployment_latency_gate")
    native_behavior_gate = native_replay.get("behavior_recovery_gate")
    closed_loop_authorization = native_replay.get("closed_loop_authorization")
    if (
        native_replay.get("status") != COMPLETED_NATIVE_REPLAY_EVALUATION
        or not signed_content_hash_is_valid(native_replay)
        or not isinstance(native_coverage, Mapping)
        or native_coverage.get("complete") is not True
        or native_coverage.get("completed_state_count")
        != native_coverage.get("expected_state_count")
        or native_coverage.get("missing_state_ids") not in ([], ())
        or native_coverage.get("extra_state_ids") not in ([], ())
        or native_replay.get("winner_model") != winner
        or not isinstance(native_latency_gate, Mapping)
        or native_latency_gate.get("verdict") != "GO"
        or not isinstance(native_behavior_gate, Mapping)
        or native_behavior_gate.get("verdict") != "GO"
        or not isinstance(closed_loop_authorization, Mapping)
        or closed_loop_authorization.get("authorized") is not True
        or closed_loop_authorization.get("verdict") != "GO"
        or tuple(closed_loop_authorization.get("requires", ()))
        != (
            "behavior_recovery_gate.GO",
            "deployment_latency_gate.GO",
        )
        or not isinstance(native_bindings, Mapping)
        or native_bindings.get("config_sha256") != config_sha
        or native_bindings.get("selections_sha256") != selections_sha
        or native_bindings.get("heldout_result_sha256") != result_sha
    ):
        raise ValueError(
            "live selector requires complete native replay behavior and latency GO"
        )
    candidates = config.get("frozen_candidates", {}).get("models", {})
    artifact = selections.get("model_artifacts", {}).get(winner)
    candidate = candidates.get(winner)
    if not isinstance(candidate, Mapping) or not isinstance(artifact, Mapping):
        raise ValueError("held-out winner is absent from frozen model artifacts")
    checkpoint_sha = sha256_file(checkpoint_path)
    if (
        candidate.get("sha256") != checkpoint_sha
        or artifact.get("checkpoint_sha256") != checkpoint_sha
        or candidate.get("variant") != artifact.get("variant")
    ):
        raise ValueError("live selector checkpoint or variant drifted")
    representation = config.get("representation", {})
    model_config_sha = _sha256_text(
        representation.get("model_config_sha256"), "model config SHA256"
    )
    fields = {
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha,
        "heldout_result_sha256": result_sha,
        "model_config_sha256": model_config_sha,
        "model_variant": str(candidate["variant"]),
        "native_replay_result_sha256": native_replay_sha,
        "selections_sha256": selections_sha,
        "status": LIVE_SELECTOR_AUTHORIZATION_STATUS,
        "winner_model": str(winner),
    }
    authorization_id = hashlib.sha256(canonical_json_bytes(fields)).hexdigest()
    return LiveSelectorAuthorization(authorization_id=authorization_id, **fields)


@dataclass(frozen=True)
class PNGPayload:
    sha256: str
    payload: bytes

    def __post_init__(self) -> None:
        _sha256_text(self.sha256, "PNG SHA256")
        if not isinstance(self.payload, bytes) or not self.payload.startswith(
            _PNG_SIGNATURE
        ):
            raise ValueError("live selector image must be encoded PNG bytes")
        if hashlib.sha256(self.payload).hexdigest() != self.sha256:
            raise ValueError("live selector PNG SHA256 drifted")

    @classmethod
    def from_mapping(cls, value: Any, *, label: str) -> "PNGPayload":
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be a mapping")
        _exact_keys(value, {"base64", "media_type", "sha256"}, label)
        if value.get("media_type") != "image/png":
            raise ValueError(f"{label} must use image/png")
        try:
            payload = base64.b64decode(value["base64"], validate=True)
        except (TypeError, ValueError, binascii.Error) as error:
            raise ValueError(f"{label} base64 is invalid") from error
        return cls(
            sha256=_sha256_text(value.get("sha256"), f"{label} SHA256"),
            payload=payload,
        )

    def to_mapping(self) -> dict[str, str]:
        return {
            "base64": base64.b64encode(self.payload).decode("ascii"),
            "media_type": "image/png",
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class LiveRichEvent:
    event_step_id: int
    low_fidelity: LowFidelityEventV2
    post_image: PNGPayload
    post_ocr_tokens: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any, *, index: int) -> "LiveRichEvent":
        if not isinstance(value, Mapping):
            raise ValueError(f"event {index} must be a mapping")
        _exact_keys(
            value,
            {"event_step_id", "low_fidelity_v2", "post_image", "post_ocr_tokens"},
            f"event {index}",
        )
        step_id = value.get("event_step_id")
        if type(step_id) is not int or step_id <= 0:
            raise ValueError(f"event {index} step id is invalid")
        try:
            low = LowFidelityEventV2.from_mapping(value["low_fidelity_v2"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"event {index} low-fidelity summary is invalid") from error
        if low.step_id != step_id:
            raise ValueError(f"event {index} summary step id drifted")
        if low.foreground_app != "unknown" or low.executor_result != "unknown":
            raise ValueError(
                f"event {index} must preserve the training-time unknown fields"
            )
        return cls(
            event_step_id=step_id,
            low_fidelity=low,
            post_image=PNGPayload.from_mapping(
                value["post_image"], label=f"event {index} post image"
            ),
            post_ocr_tokens=_ocr_tokens(
                value["post_ocr_tokens"], f"event {index} post OCR"
            ),
        )

    @property
    def summary_text(self) -> str:
        return serialize_low_fidelity_v2(self.low_fidelity).decode("utf-8").rstrip(
            "\n"
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "event_step_id": self.event_step_id,
            "low_fidelity_v2": self.low_fidelity.to_ordered_dict(),
            "post_image": self.post_image.to_mapping(),
            "post_ocr_tokens": list(self.post_ocr_tokens),
        }

    @classmethod
    def build(
        cls,
        *,
        event_step_id: int,
        low_fidelity_v2: Mapping[str, Any],
        post_image_png: bytes,
        post_ocr_tokens: Sequence[str],
    ) -> "LiveRichEvent":
        return cls.from_mapping(
            {
                "event_step_id": event_step_id,
                "low_fidelity_v2": dict(low_fidelity_v2),
                "post_image": PNGPayload(
                    sha256=hashlib.sha256(post_image_png).hexdigest(),
                    payload=post_image_png,
                ).to_mapping(),
                "post_ocr_tokens": list(post_ocr_tokens),
            },
            index=event_step_id - 1,
        )


@dataclass(frozen=True)
class LiveRichSelectorRequest:
    authorization_id: str
    budget: int
    current_image: PNGPayload
    current_ocr_tokens: tuple[str, ...]
    decision_step_id: int
    events: tuple[LiveRichEvent, ...]
    instruction: str
    request_id: str
    source_id: str
    state_id: str
    content_sha256: str

    @classmethod
    def from_mapping(cls, value: Any) -> "LiveRichSelectorRequest":
        if not isinstance(value, Mapping):
            raise ValueError("live selector request must be a mapping")
        expected = {
            "authorization_id",
            "budget",
            "content_sha256",
            "current_image",
            "current_ocr_tokens",
            "decision_step_id",
            "events",
            "instruction",
            "request_id",
            "schema_version",
            "source_id",
            "state_id",
            "status",
        }
        _exact_keys(value, expected, "live selector request")
        if (
            value.get("schema_version") != LIVE_SELECTOR_SCHEMA_VERSION
            or value.get("status") != LIVE_SELECTOR_REQUEST_STATUS
            or not _content_hash_is_valid(value)
        ):
            raise ValueError("live selector request envelope or content hash drifted")
        decision = value.get("decision_step_id")
        if type(decision) is not int or decision < MINIMUM_SUPPORTED_DECISION_STEP:
            raise ValueError(
                "rich predictor requests begin at the frozen training support step six"
            )
        budget = value.get("budget")
        if type(budget) is not int or budget not in SUPPORTED_BUDGETS:
            raise ValueError("live selector budget must be one of 1,2,3,4")
        raw_events = value.get("events")
        if isinstance(raw_events, (str, bytes, bytearray, Mapping)) or not isinstance(
            raw_events, Sequence
        ):
            raise ValueError("live selector events must be an ordered array")
        events = tuple(
            LiveRichEvent.from_mapping(event, index=index)
            for index, event in enumerate(raw_events)
        )
        expected_ids = tuple(range(1, decision))
        if tuple(event.event_step_id for event in events) != expected_ids:
            raise ValueError(
                "live selector candidates must be every prior eligible event"
            )
        source_id = _nonempty_text(value.get("source_id"), "source id")
        state_id = _nonempty_text(value.get("state_id"), "state id")
        if state_id != f"{source_id}:decision:{decision:03d}":
            raise ValueError("live selector state identity drifted")
        current_image = PNGPayload.from_mapping(
            value.get("current_image"), label="current image"
        )
        current_ocr = _ocr_tokens(value.get("current_ocr_tokens"), "current OCR")
        latest = events[-1]
        if (
            latest.post_image.sha256 != current_image.sha256
            or latest.post_ocr_tokens != current_ocr
        ):
            raise ValueError(
                "latest prior event must remain the current-equivalent candidate"
            )
        return cls(
            authorization_id=_sha256_text(
                value.get("authorization_id"), "authorization id"
            ),
            budget=budget,
            current_image=current_image,
            current_ocr_tokens=current_ocr,
            decision_step_id=decision,
            events=events,
            instruction=_nonempty_text(value.get("instruction"), "instruction"),
            request_id=_nonempty_text(value.get("request_id"), "request id"),
            source_id=source_id,
            state_id=state_id,
            content_sha256=_sha256_text(
                value.get("content_sha256"), "request content SHA256"
            ),
        )

    @classmethod
    def build(
        cls,
        *,
        authorization_id: str,
        budget: int,
        current_image_png: bytes,
        current_ocr_tokens: Sequence[str],
        decision_step_id: int,
        events: Sequence[Mapping[str, Any]],
        instruction: str,
        request_id: str,
        source_id: str,
    ) -> "LiveRichSelectorRequest":
        current = PNGPayload(
            sha256=hashlib.sha256(current_image_png).hexdigest(),
            payload=current_image_png,
        )
        raw = {
            "authorization_id": authorization_id,
            "budget": budget,
            "current_image": current.to_mapping(),
            "current_ocr_tokens": list(current_ocr_tokens),
            "decision_step_id": decision_step_id,
            "events": list(events),
            "instruction": instruction,
            "request_id": request_id,
            "schema_version": LIVE_SELECTOR_SCHEMA_VERSION,
            "source_id": source_id,
            "state_id": f"{source_id}:decision:{decision_step_id:03d}",
            "status": LIVE_SELECTOR_REQUEST_STATUS,
        }
        return cls.from_mapping(_signed_mapping(raw))

    @property
    def candidate_event_ids(self) -> tuple[int, ...]:
        return tuple(event.event_step_id for event in self.events)

    def to_mapping(self) -> dict[str, Any]:
        return _signed_mapping(
            {
                "authorization_id": self.authorization_id,
                "budget": self.budget,
                "current_image": self.current_image.to_mapping(),
                "current_ocr_tokens": list(self.current_ocr_tokens),
                "decision_step_id": self.decision_step_id,
                "events": [event.to_mapping() for event in self.events],
                "instruction": self.instruction,
                "request_id": self.request_id,
                "schema_version": LIVE_SELECTOR_SCHEMA_VERSION,
                "source_id": self.source_id,
                "state_id": self.state_id,
                "status": LIVE_SELECTOR_REQUEST_STATUS,
            }
        )


def build_live_mixed_fidelity_messages(
    request: LiveRichSelectorRequest,
    selected_event_step_ids: Sequence[int],
    *,
    image_decoder: Callable[[bytes], Any],
) -> tuple[Mapping[str, Any], ...]:
    """Render a selected live subset with the exact variable-history prompt."""
    if not isinstance(request, LiveRichSelectorRequest):
        raise TypeError("live prompt requires a validated selector request")
    if not callable(image_decoder):
        raise TypeError("live prompt image decoder must be callable")
    selected = tuple(selected_event_step_ids)
    if (
        any(type(value) is not int for value in selected)
        or selected != tuple(sorted(selected))
        or len(selected) != len(set(selected))
        or len(selected) > request.budget
        or not set(selected).issubset(request.candidate_event_ids)
    ):
        raise ValueError("live prompt subset violates the at-most-B contract")
    selected_set = frozenset(selected)
    current_reference = "live://current-observation"
    payloads = {current_reference: request.current_image.payload}
    history = []
    for event in request.events:
        reference = f"live://event/{event.event_step_id:06d}/post-observation"
        payloads[reference] = event.post_image.payload
        history.append(
            PromptHistoryEvent(
                event_step_id=event.event_step_id,
                low_fidelity_summary=event.low_fidelity,
                is_frozen_candidate=True,
                high_fidelity_observation_ref=(
                    reference if event.event_step_id in selected_set else None
                ),
            )
        )
    plan = MixedFidelityPromptPlan(
        state_id=request.state_id,
        task_instruction=request.instruction,
        restored_event_step_ids=selected,
        history_events=tuple(history),
        current_observation_ref=current_reference,
    )

    def load_image(reference: str) -> bytes:
        try:
            return payloads[reference]
        except KeyError as error:
            raise ValueError("live prompt requested an unknown image") from error

    return tuple(
        build_set_utility_gui_owl_v2_1_messages(
            plan,
            image_bytes_loader=load_image,
            image_decoder=image_decoder,
        )
    )


@dataclass(frozen=True)
class LiveSelectorBackendResult:
    selected_event_step_ids: tuple[int, ...]
    selected_predicted_utility: float
    scored_subsets: tuple[tuple[tuple[int, ...], float], ...]
    latency_ms: Mapping[str, float]
    source_token_counts: Mapping[str, int]


LiveSelectorBackend = Callable[[LiveRichSelectorRequest], LiveSelectorBackendResult]


class LiveRichSelectorService:
    """Transport-neutral idempotent service around one authorized backend."""

    def __init__(
        self,
        *,
        authorization: LiveSelectorAuthorization,
        backend: LiveSelectorBackend,
    ) -> None:
        if not isinstance(authorization, LiveSelectorAuthorization):
            raise TypeError("service authorization is invalid")
        if not callable(backend):
            raise TypeError("live selector backend must be callable")
        self.authorization = authorization
        self.backend = backend
        self._responses: dict[str, dict[str, Any]] = {}
        self._request_ids: dict[str, str] = {}

    def health(self) -> dict[str, Any]:
        return {
            "authorization": self.authorization.to_mapping(),
            "schema_version": LIVE_SELECTOR_SCHEMA_VERSION,
            "status": "READY_CAUSALCACHE_LIVE_RICH_SELECTOR_SERVICE",
        }

    def select(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = LiveRichSelectorRequest.from_mapping(payload)
        if request.authorization_id != self.authorization.authorization_id:
            raise ValueError("request does not bind the authorized held-out winner")
        prior_content = self._request_ids.get(request.request_id)
        if prior_content is not None and prior_content != request.content_sha256:
            raise ValueError("request id was reused for different live state bytes")
        cached = self._responses.get(request.content_sha256)
        if cached is not None:
            return dict(cached)
        started = time.perf_counter()
        result = self.backend(request)
        selected = tuple(result.selected_event_step_ids)
        if (
            selected != tuple(sorted(selected))
            or len(selected) != len(set(selected))
            or len(selected) > request.budget
            or not set(selected).issubset(request.candidate_event_ids)
        ):
            raise ValueError("live selector backend violated the at-most-B contract")
        utility = float(result.selected_predicted_utility)
        if not math.isfinite(utility):
            raise ValueError("live selector backend emitted non-finite utility")
        scored = []
        for subset, value in result.scored_subsets:
            subset = tuple(subset)
            score = float(value)
            if (
                subset != tuple(sorted(subset))
                or len(subset) != len(set(subset))
                or len(subset) > request.budget
                or not set(subset).issubset(request.candidate_event_ids)
                or not math.isfinite(score)
            ):
                raise ValueError("live selector backend score trace is invalid")
            scored.append({"event_step_ids": list(subset), "predicted_utility": score})
        latency = {name: float(value) for name, value in result.latency_ms.items()}
        if not latency or any(
            not name or not math.isfinite(value) or value < 0.0
            for name, value in latency.items()
        ):
            raise ValueError("live selector backend latency audit is invalid")
        counts = {name: int(value) for name, value in result.source_token_counts.items()}
        if any(not name or value < 0 for name, value in counts.items()):
            raise ValueError("live selector source-token audit is invalid")
        response = _signed_mapping(
            {
                "authorization_id": self.authorization.authorization_id,
                "budget": request.budget,
                "candidate_event_step_ids": list(request.candidate_event_ids),
                "checkpoint_sha256": self.authorization.checkpoint_sha256,
                "latency_ms": {
                    **latency,
                    "service_total": (time.perf_counter() - started) * 1000.0,
                },
                "request_content_sha256": request.content_sha256,
                "request_id": request.request_id,
                "schema_version": LIVE_SELECTOR_SCHEMA_VERSION,
                "scored_subsets": scored,
                "selected_event_step_ids": list(selected),
                "selected_predicted_utility": utility,
                "source_token_counts": counts,
                "state_id": request.state_id,
                "status": LIVE_SELECTOR_RESPONSE_STATUS,
                "winner_model": self.authorization.winner_model,
            }
        )
        self._request_ids[request.request_id] = request.content_sha256
        self._responses[request.content_sha256] = response
        return dict(response)


def validate_live_selector_response(
    payload: Mapping[str, Any],
    *,
    request: LiveRichSelectorRequest,
    authorization: LiveSelectorAuthorization,
) -> tuple[int, ...]:
    """Validate the H200 response again on the AndroidWorld client."""
    expected = {
        "authorization_id",
        "budget",
        "candidate_event_step_ids",
        "checkpoint_sha256",
        "content_sha256",
        "latency_ms",
        "request_content_sha256",
        "request_id",
        "schema_version",
        "scored_subsets",
        "selected_event_step_ids",
        "selected_predicted_utility",
        "source_token_counts",
        "state_id",
        "status",
        "winner_model",
    }
    _exact_keys(payload, expected, "live selector response")
    if (
        payload.get("schema_version") != LIVE_SELECTOR_SCHEMA_VERSION
        or payload.get("status") != LIVE_SELECTOR_RESPONSE_STATUS
        or not _content_hash_is_valid(payload)
        or payload.get("authorization_id") != authorization.authorization_id
        or payload.get("checkpoint_sha256") != authorization.checkpoint_sha256
        or payload.get("winner_model") != authorization.winner_model
        or payload.get("request_content_sha256") != request.content_sha256
        or payload.get("request_id") != request.request_id
        or payload.get("state_id") != request.state_id
        or payload.get("budget") != request.budget
        or tuple(payload.get("candidate_event_step_ids", ()))
        != request.candidate_event_ids
    ):
        raise ValueError("live selector response binding drifted")
    selected = tuple(payload.get("selected_event_step_ids", ()))
    if (
        any(type(value) is not int for value in selected)
        or selected != tuple(sorted(selected))
        or len(selected) != len(set(selected))
        or len(selected) > request.budget
        or not set(selected).issubset(request.candidate_event_ids)
    ):
        raise ValueError("live selector response violates at-most-B")
    utility = payload.get("selected_predicted_utility")
    if isinstance(utility, bool) or not isinstance(utility, (int, float)):
        raise ValueError("live selector response utility is invalid")
    if not math.isfinite(float(utility)):
        raise ValueError("live selector response utility is non-finite")
    return selected


def select_live_selector_over_http(
    *,
    endpoint: str,
    request: LiveRichSelectorRequest,
    authorization: LiveSelectorAuthorization,
    timeout_seconds: float,
) -> tuple[tuple[int, ...], dict[str, Any], float]:
    """Call a loopback-tunneled H200 service without a heuristic fallback."""
    if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
        raise ValueError("live selector endpoint must be an HTTP URL")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0.0
    ):
        raise ValueError("live selector timeout must be finite and positive")
    payload = canonical_json_bytes(request.to_mapping())
    http_request = urllib.request.Request(
        endpoint.rstrip("/") + "/select",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(
            http_request, timeout=float(timeout_seconds)
        ) as response:
            body = response.read()
            content_type = response.headers.get_content_type()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError("live rich selector RPC failed; fallback is forbidden") from error
    round_trip_ms = (time.perf_counter() - started) * 1000.0
    if content_type != "application/json":
        raise ValueError("live selector RPC returned a non-JSON content type")
    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("live selector RPC returned invalid JSON") from error
    if not isinstance(result, dict):
        raise ValueError("live selector RPC response must be a JSON object")
    selected = validate_live_selector_response(
        result, request=request, authorization=authorization
    )
    return selected, result, round_trip_ms


def load_go_authorized_live_selector_service(
    *,
    repository_root: Path,
    heldout_config_path: Path,
    selections_path: Path,
    heldout_result_path: Path,
    native_replay_result_path: Path,
    checkpoint_path: Path,
    model_dir: Path,
    device: str,
) -> LiveRichSelectorService:
    """Load the exact H200 source encoder and frozen held-out winner."""
    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise ValueError("live rich selector requires an explicit CUDA device")
    authorization = authorize_live_selector(
        heldout_config_path=heldout_config_path,
        selections_path=selections_path,
        heldout_result_path=heldout_result_path,
        native_replay_result_path=native_replay_result_path,
        checkpoint_path=checkpoint_path,
    )
    heldout = _read_mapping(heldout_config_path)
    relative_model_config = heldout.get("representation", {}).get("model_config")
    if not isinstance(relative_model_config, str) or not relative_model_config:
        raise ValueError("held-out config omits the frozen model config path")
    model_config_path = repository_root.resolve() / relative_model_config
    if sha256_file(model_config_path) != authorization.model_config_sha256:
        raise ValueError("live selector model config SHA256 drifted")
    model_config = _read_mapping(model_config_path)
    try:
        variant = model_config["variants"][authorization.model_variant]
    except (KeyError, TypeError) as error:
        raise ValueError("live selector model variant is absent") from error
    model_arguments = variant.get("model")
    if not isinstance(model_arguments, Mapping):
        raise ValueError("live selector model variant has no architecture")
    if model_arguments.get("family") != authorization.winner_model:
        raise ValueError("live selector model family differs from held-out winner")

    from causalcache.policy.gui_owl_variable_history_runtime import (
        GUIOwlVariableHistoryRuntime,
        VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    from causalcache.set_utility_token_models import (
        TokenSetUtilityPredictor,
        TokenUtilityModelConfig,
    )

    runtime = GUIOwlVariableHistoryRuntime(
        model_dir=model_dir.resolve(),
        expected_snapshot_manifest=(
            repository_root.resolve()
            / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device=device,
        target_effective_visual_tokens_per_image=(
            VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
        ),
    )
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise RuntimeError("live selector requires safetensors") from error
    model = TokenSetUtilityPredictor(
        TokenUtilityModelConfig(**dict(model_arguments))
    ).to(runtime.device)
    model.load_state_dict(
        load_file(str(checkpoint_path.resolve()), device=str(runtime.device)),
        strict=True,
    )
    model.eval()
    chunk_size = heldout.get("selection", {}).get("subset_score_chunk_size")
    backend = TorchLiveRichTokenBackend(
        model=model,
        source_encoder=GUIOwlLiveRichSourceEncoder(runtime),
        subset_score_chunk_size=chunk_size,
    )
    return LiveRichSelectorService(
        authorization=authorization,
        backend=backend,
    )


@dataclass(frozen=True)
class EncodedVisualSource:
    normalized_mean: Any
    token_count: int
    tokens: Any


class GUIOwlLiveRichSourceEncoder:
    """Exact 480-token GUI-Owl source encoder backed by an H200 runtime."""

    def __init__(self, runtime: Any) -> None:
        from causalcache.policy.gui_owl_variable_history_runtime import (
            VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
            VARIABLE_HISTORY_RUNTIME_PROFILE_ID,
        )

        if runtime.metadata.get("runtime_profile_id") != VARIABLE_HISTORY_RUNTIME_PROFILE_ID:
            raise ValueError("live source encoder requires the frozen H200 480-token runtime")
        self.runtime = runtime
        self.target_visual_tokens = VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
        self.torch = runtime.torch
        self._visual: dict[str, EncodedVisualSource] = {}
        self._text: dict[str, Any] = {}

    def reset(self) -> None:
        self._visual.clear()
        self._text.clear()

    def encode_visual(self, image: PNGPayload) -> EncodedVisualSource:
        cached = self._visual.get(image.sha256)
        if cached is not None:
            return cached
        from PIL import Image

        from causalcache.policy.gui_owl_v2_vision import (
            extract_spatial_merger_token_sequences,
        )

        with Image.open(io.BytesIO(image.payload)) as source:
            rgb = source.convert("RGB")
            encoded = self.runtime.processor.image_processor(
                images=[rgb], return_tensors="pt"
            )
        pixel_values = encoded["pixel_values"].to(
            device=self.runtime.device, dtype=self.torch.bfloat16
        )
        grid = encoded["image_grid_thw"].to(self.runtime.device)
        with self.torch.inference_mode():
            batch = extract_spatial_merger_token_sequences(
                model=self.runtime.model,
                pixel_values=pixel_values,
                image_grid_thw=grid,
                runtime_identity=self.runtime.runtime_identity,
            )
        tokens = batch.token_sequences[0]
        count = int(batch.merged_token_counts[0])
        if tuple(tokens.shape) != (count, SOURCE_HIDDEN_SIZE) or count != self.target_visual_tokens:
            raise ValueError("live GUI-Owl visual token geometry drifted")
        mean = tokens.to(dtype=self.torch.float32).mean(dim=0)
        mean = mean / mean.norm(p=2).clamp_min(1e-12)
        result = EncodedVisualSource(mean, count, tokens.contiguous())
        self._visual[image.sha256] = result
        return result

    def encode_text(self, text: str) -> Any:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._text.get(key)
        if cached is not None:
            return cached
        encoded = self.runtime.processor.tokenizer(
            [text],
            add_special_tokens=True,
            max_length=MAXIMUM_TEXT_TOKENS,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        ids = encoded["input_ids"].to(self.runtime.device)
        length = int(encoded["attention_mask"].sum().item())
        with self.torch.inference_mode():
            tokens = self.runtime.model.model.language_model.embed_tokens(ids)[0, :length]
        if tokens.dtype is not self.torch.bfloat16 or tuple(tokens.shape)[1:] != (
            SOURCE_HIDDEN_SIZE,
        ):
            raise ValueError("live GUI-Owl text token geometry drifted")
        result = tokens.contiguous()
        self._text[key] = result
        return result


def _pad_entities(visual: Sequence[Any], text: Sequence[Any], *, torch: Any) -> tuple[Any, ...]:
    if not visual or len(visual) != len(text):
        raise ValueError("live source tensors must be non-empty and aligned")
    visual_length = max(int(row.shape[0]) for row in visual)
    text_length = max(int(row.shape[0]) for row in text)
    hidden = int(visual[0].shape[1])
    device = visual[0].device
    visual_batch = torch.zeros(
        (len(visual), visual_length, hidden), dtype=visual[0].dtype, device=device
    )
    visual_mask = torch.zeros(
        (len(visual), visual_length), dtype=torch.bool, device=device
    )
    text_batch = torch.zeros(
        (len(text), text_length, hidden), dtype=text[0].dtype, device=device
    )
    text_mask = torch.zeros((len(text), text_length), dtype=torch.bool, device=device)
    for index, row in enumerate(visual):
        visual_batch[index, : row.shape[0]] = row
        visual_mask[index, : row.shape[0]] = True
    for index, row in enumerate(text):
        text_batch[index, : row.shape[0]] = row
        text_mask[index, : row.shape[0]] = True
    return visual_batch, visual_mask, text_batch, text_mask


class TorchLiveRichTokenBackend:
    """Run one frozen token predictor with incremental event-source caching."""

    def __init__(
        self,
        *,
        model: Any,
        source_encoder: GUIOwlLiveRichSourceEncoder,
        subset_score_chunk_size: int = 64,
    ) -> None:
        if type(subset_score_chunk_size) is not int or subset_score_chunk_size <= 0:
            raise ValueError("subset score chunk size must be positive")
        self.model = model
        self.source_encoder = source_encoder
        self.torch = source_encoder.torch
        self.device = source_encoder.runtime.device
        self.subset_score_chunk_size = subset_score_chunk_size
        self._event_sources: dict[tuple[str, str], Any] = {}
        self._active_source_id: str | None = None

    def _synchronize(self) -> None:
        self.torch.cuda.synchronize(self.device)

    def _encode_request(
        self, request: LiveRichSelectorRequest
    ) -> tuple[Any, float, Mapping[str, float], Mapping[str, int]]:
        """Encode a live request once for either scalar or marginal search."""
        torch = self.torch
        if self._active_source_id != request.source_id:
            self._event_sources.clear()
            self.source_encoder.reset()
            self._active_source_id = request.source_id
        self._synchronize()
        started = time.perf_counter()
        visual = [
            self.source_encoder.encode_visual(event.post_image)
            for event in request.events
        ]
        text = [
            self.source_encoder.encode_text(event.summary_text)
            for event in request.events
        ]
        query_visual_source = self.source_encoder.encode_visual(request.current_image)
        query_text_source = self.source_encoder.encode_text(request.instruction)
        self._synchronize()
        source_encoding_ms = (time.perf_counter() - started) * 1000.0

        missing = [
            index
            for index, event in enumerate(request.events)
            if (
                event.post_image.sha256,
                hashlib.sha256(event.summary_text.encode("utf-8")).hexdigest(),
            )
            not in self._event_sources
        ]
        event_started = time.perf_counter()
        if missing:
            v, vm, t, tm = _pad_entities(
                [visual[index].tokens for index in missing],
                [text[index] for index in missing],
                torch=torch,
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                encoded = self.model.encode_event_sources(
                    event_visual_tokens=v.unsqueeze(0),
                    event_visual_mask=vm.unsqueeze(0),
                    event_text_tokens=t.unsqueeze(0),
                    event_text_mask=tm.unsqueeze(0),
                    event_mask=torch.ones(
                        (1, len(missing)), dtype=torch.bool, device=self.device
                    ),
                )
            for offset, index in enumerate(missing):
                event = request.events[index]
                key = (
                    event.post_image.sha256,
                    hashlib.sha256(event.summary_text.encode("utf-8")).hexdigest(),
                )
                self._event_sources[key] = encoded[0, offset]
        self._synchronize()
        event_source_encoding_ms = (time.perf_counter() - event_started) * 1000.0

        query_started = time.perf_counter()
        qv, qvm, qt, qtm = _pad_entities(
            [query_visual_source.tokens], [query_text_source], torch=torch
        )
        raw_events = []
        numeric = []
        for event, event_visual in zip(request.events, visual, strict=True):
            key = (
                event.post_image.sha256,
                hashlib.sha256(event.summary_text.encode("utf-8")).hexdigest(),
            )
            raw_events.append(self._event_sources[key])
            cosine = float(
                (event_visual.normalized_mean * query_visual_source.normalized_mean)
                .sum()
                .item()
            )
            numeric.append(
                variable_history_event_numeric_features(
                    event.low_fidelity,
                    decision_step_id=request.decision_step_id,
                    event_ocr_tokens=event.post_ocr_tokens,
                    current_ocr_tokens=request.current_ocr_tokens,
                    event_current_vlm_cosine=cosine,
                )
            )
        event_mask = torch.ones(
            (1, len(raw_events)), dtype=torch.bool, device=self.device
        )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            query = self.model.encode_query_source(
                query_visual_tokens=qv,
                query_visual_mask=qvm,
                query_text_tokens=qt,
                query_text_mask=qtm,
            )
            conditioned = self.model.condition_encoded_state(
                query=query,
                event_sources=torch.stack(raw_events).unsqueeze(0),
                event_numeric_features=torch.tensor(
                    [numeric], dtype=torch.float32, device=self.device
                ),
                event_mask=event_mask,
            )
        self._synchronize()
        query_conditioning_ms = (time.perf_counter() - query_started) * 1000.0
        return (
            conditioned,
            started,
            {
                "event_source_encoding": event_source_encoding_ms,
                "query_conditioning": query_conditioning_ms,
                "raw_source_encoding": source_encoding_ms,
            },
            {
                "candidate_text": sum(int(value.shape[0]) for value in text),
                "candidate_visual": sum(value.token_count for value in visual),
                "query_text": int(query_text_source.shape[0]),
                "query_visual": query_visual_source.token_count,
            },
        )

    def __call__(self, request: LiveRichSelectorRequest) -> LiveSelectorBackendResult:
        torch = self.torch
        conditioned, started, encoding_latency, source_token_counts = (
            self._encode_request(request)
        )
        event_index = {
            event_id: index
            for index, event_id in enumerate(request.candidate_event_ids)
        }

        def score_batch(subsets: tuple[tuple[int, ...], ...]) -> tuple[float, ...]:
            values = []
            for start in range(0, len(subsets), self.subset_score_chunk_size):
                chunk = subsets[start : start + self.subset_score_chunk_size]
                masks = torch.zeros(
                    (1, len(chunk), len(event_index)),
                    dtype=torch.bool,
                    device=self.device,
                )
                for subset_index, subset in enumerate(chunk):
                    for event_id in subset:
                        masks[0, subset_index, event_index[event_id]] = True
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    predicted = self.model.score_encoded_subsets(conditioned, masks)
                values.extend(float(value) for value in predicted[0].tolist())
            return tuple(values)

        search_started = time.perf_counter()
        search = conditional_greedy_at_most_budget_search(
            request.candidate_event_ids,
            budget=request.budget,
            score_batch=score_batch,
        )
        self._synchronize()
        search_ms = (time.perf_counter() - search_started) * 1000.0
        return LiveSelectorBackendResult(
            selected_event_step_ids=search.selected_subset,
            selected_predicted_utility=search.selected_predicted_utility,
            scored_subsets=search.scored_subsets,
            latency_ms={
                **encoding_latency,
                "search": search_ms,
                "selector_total": (time.perf_counter() - started) * 1000.0,
            },
            source_token_counts=source_token_counts,
        )


class TorchDirectMarginalReplayBackend(TorchLiveRichTokenBackend):
    """Use one direct-marginal adapter in the existing mixed-fidelity service."""

    def __init__(
        self,
        *,
        model: Any,
        source_encoder: GUIOwlLiveRichSourceEncoder,
        stop_semantics: str,
        recent_fallback_threshold: float | None = None,
    ) -> None:
        if stop_semantics not in DIRECT_MARGINAL_STOP_SEMANTICS:
            raise ValueError("direct marginal backend STOP semantics are unsupported")
        if recent_fallback_threshold is not None and (
            not math.isfinite(float(recent_fallback_threshold))
            or float(recent_fallback_threshold) < 0.0
        ):
            raise ValueError("direct marginal hybrid threshold is invalid")
        super().__init__(
            model=model,
            source_encoder=source_encoder,
            subset_score_chunk_size=1,
        )
        self.stop_semantics = stop_semantics
        self.recent_fallback_threshold = (
            None
            if recent_fallback_threshold is None
            else float(recent_fallback_threshold)
        )
        self.selection_mode = (
            "direct"
            if self.recent_fallback_threshold is None
            else "confidence_gated_recent"
        )

    def _select_with_audit(
        self, request: LiveRichSelectorRequest
    ) -> tuple[
        DirectMarginalSelectionPath,
        Mapping[str, float],
        Mapping[str, int],
    ]:
        torch = self.torch
        conditioned, started, encoding_latency, source_token_counts = (
            self._encode_request(request)
        )
        event_index = {
            event_id: index
            for index, event_id in enumerate(request.candidate_event_ids)
        }

        def score(selected: tuple[int, ...]) -> tuple[float, ...]:
            mask = torch.zeros(
                (1, len(event_index)), dtype=torch.bool, device=self.device
            )
            for event_id in selected:
                mask[0, event_index[event_id]] = True
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                predicted = self.model.score_encoded_candidates(conditioned, mask)
            values = predicted[0].tolist()
            return tuple(float(value) for value in values)

        search_started = time.perf_counter()
        path = direct_marginal_at_most_budget_path(
            request.candidate_event_ids,
            budget=request.budget,
            score=score,
            stop_semantics=self.stop_semantics,
            recent_fallback_threshold=self.recent_fallback_threshold,
        )
        self._synchronize()
        latency = {
            **encoding_latency,
            "search": (time.perf_counter() - search_started) * 1000.0,
            "selector_total": (time.perf_counter() - started) * 1000.0,
        }
        return path, latency, source_token_counts

    def select_all_budgets(
        self, request: LiveRichSelectorRequest
    ) -> DirectMarginalSelectionPath:
        """Encode once and expose the nested path through the requested budget."""
        path, _, _ = self._select_with_audit(request)
        return path

    def __call__(self, request: LiveRichSelectorRequest) -> LiveSelectorBackendResult:
        path, latency, source_token_counts = self._select_with_audit(request)
        return LiveSelectorBackendResult(
            selected_event_step_ids=path.selection(request.budget),
            selected_predicted_utility=path.predicted_utility(request.budget),
            scored_subsets=path.scored_subsets,
            latency_ms=latency,
            source_token_counts=source_token_counts,
        )


__all__ = [
    "GUIOwlLiveRichSourceEncoder",
    "LIVE_SELECTOR_AUTHORIZATION_STATUS",
    "LIVE_SELECTOR_REQUEST_STATUS",
    "LIVE_SELECTOR_RESPONSE_STATUS",
    "LIVE_SELECTOR_SCHEMA_VERSION",
    "LiveRichEvent",
    "LiveRichSelectorRequest",
    "LiveRichSelectorService",
    "LiveSelectorAuthorization",
    "LiveSelectorBackendResult",
    "PNGPayload",
    "TorchDirectMarginalReplayBackend",
    "TorchLiveRichTokenBackend",
    "authorize_live_selector",
    "build_live_mixed_fidelity_messages",
    "canonical_json_bytes",
    "load_go_authorized_live_selector_service",
    "select_live_selector_over_http",
    "sha256_file",
    "validate_live_selector_response",
]
