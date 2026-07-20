"""Outcome-exposed AndroidWorld adapter for the post-GO utility selector."""

from __future__ import annotations

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

from causalcache.exploratory_closed_loop_roster import (
    MANIFEST_PATH,
    MANIFEST_SHA256,
    SELECTED_INSTANCE_IDENTITY_SHA256,
    validate_frozen_exploratory_closed_loop_roster_files,
)
from causalcache.independent_closed_loop_features import (
    FROZEN_GATE_V1_OCR_BACKEND,
    live_history_event_from_transition,
)
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    GUI_OWL_V2_1_SYSTEM_PROMPT,
    validate_gui_owl_v2_1_native_messages,
)
from causalcache.restoration_v2_text_backend import (
    build_ocr_record,
    canonical_json_bytes as canonical_ocr_json_bytes,
    create_rapidocr_engine,
    prepare_image_bytes,
    validate_backend_config,
)
from causalcache.restoration_v2_baselines import (
    OCR_JACCARD_WEIGHT,
    RGB_HISTOGRAM_WEIGHT,
    joint_rgb_histogram_cosine,
    ocr_token_set_jaccard,
)
from causalcache.set_utility_heldout_inference import (
    ocr_rgb_budget_selections,
    recent_budget_selections,
)
from causalcache.set_utility_live_controller import (
    LIVE_SELECTOR_SCHEMA_VERSION,
    SUPPORTED_BUDGETS,
    LiveRichEvent,
    LiveRichSelectorRequest,
    LiveSelectorAuthorization,
    authorize_live_selector,
    build_live_mixed_fidelity_messages,
    select_live_selector_over_http,
    sha256_file,
    validate_live_selector_response,
)


EARLY_STEP_POLICY_ID = "shared_summary_only_decisions_1_5_v1"
OUTCOME_EXPOSED_EVIDENCE_ROLE = "outcome_exposed_validation12_resource_routing_only"
SELECTOR_READY_STATUS = "READY_CAUSALCACHE_LIVE_RICH_SELECTOR_SERVICE"
ANDROIDWORLD_READY_STATUS = "success"
OCR_CONFIG_SHA256 = FROZEN_GATE_V1_OCR_BACKEND.config_sha256
OCR_MANIFEST_SHA256 = FROZEN_GATE_V1_OCR_BACKEND.manifest_sha256
SUPPORTED_LIVE_ARMS = ("winner", "recent", "ocr_rgb")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return text


def validate_loopback_http_url(value: str, label: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(_nonempty(value, label))
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{label} must be a plain loopback HTTP URL")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"{label} has an invalid port") from error
    if port is None or not 1 <= port <= 65535:
        raise ValueError(f"{label} must contain an explicit port")
    return value.rstrip("/")


def _positive_latency(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def load_outcome_exposed_validation_instance(
    *,
    repository_root: Path,
    task_type: str,
    task_index: int,
) -> dict[str, Any]:
    """Return one mechanically frozen validation-12 identity and nothing else."""
    validation = validate_frozen_exploratory_closed_loop_roster_files(
        repository_root=repository_root
    )
    if task_index != 0:
        raise ValueError("outcome-exposed validation-12 permits only task_index=0")
    manifest_path = repository_root.resolve() / MANIFEST_PATH
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("outcome-exposed validation-12 roster SHA256 drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        record
        for record in manifest["records"]
        if record["instance"]["task_type"] == task_type
        and record["instance"]["task_index"] == task_index
    ]
    if len(records) != 1:
        raise ValueError("requested task is outside outcome-exposed validation-12")
    return {
        "evidence_role": OUTCOME_EXPOSED_EVIDENCE_ROLE,
        "horizon_stratum": records[0]["horizon_stratum"],
        "instance": dict(records[0]["instance"]),
        "roster_manifest_sha256": MANIFEST_SHA256,
        "selected_instance_identity_sha256": SELECTED_INSTANCE_IDENTITY_SHA256,
        "selection_sha256": records[0]["selection_sha256"],
        "suite_seed": validation["suite_seed"],
    }


def encode_deterministic_rgb_png(image: Any) -> bytes:
    """Encode one screenshot as metadata-free RGB PNG for OCR and transport."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("AndroidWorld screenshot must be a Pillow image")
    output = io.BytesIO()
    image.convert("RGB").save(
        output,
        format="PNG",
        optimize=False,
        compress_level=6,
    )
    payload = output.getvalue()
    if not payload.startswith(_PNG_SIGNATURE):
        raise RuntimeError("deterministic screenshot encoder did not emit PNG")
    return payload


def validate_live_ocr_record(
    record: Mapping[str, Any],
    *,
    image_bytes: bytes,
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
) -> dict[str, Any]:
    """Rebuild a live OCR record through the canonical frozen schema."""
    validate_backend_config(backend_config)
    if not isinstance(record, Mapping) or not isinstance(record.get("nodes"), list):
        raise ValueError("live OCR provider returned a malformed record")
    try:
        rebuilt = build_ocr_record(
            image_member_path=record["image_member_path"],
            image_bytes=image_bytes,
            backend_config_sha256=backend_config_sha256,
            boxes=[node["polygon_xy"] for node in record["nodes"]],
            texts=[node["raw_text"] for node in record["nodes"]],
            scores=[float(node["confidence_decimal_string"]) for node in record["nodes"]],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("live OCR record cannot be canonically rebuilt") from error
    if canonical_ocr_json_bytes(dict(record)) != canonical_ocr_json_bytes(rebuilt):
        raise ValueError("live OCR record differs from its canonical rebuild")
    return rebuilt


def run_live_rapidocr_record(
    *,
    engine: Any,
    backend_config: Mapping[str, Any],
    image_member_path: str,
    image_bytes: bytes,
    backend_config_sha256: str,
) -> dict[str, Any]:
    """Apply the frozen OCR parameters to a live encoded RGB screenshot."""
    validate_backend_config(backend_config)
    result = engine(
        image_bytes,
        use_det=True,
        use_cls=False,
        use_rec=True,
        return_word_box=False,
        return_single_char_box=False,
        text_score=0.5,
        box_thresh=0.5,
        unclip_ratio=1.6,
    )
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is not None and hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    return build_ocr_record(
        image_member_path=image_member_path,
        image_bytes=image_bytes,
        backend_config_sha256=backend_config_sha256,
        boxes=boxes,
        texts=texts,
        scores=scores,
    )


@dataclass(frozen=True)
class OnlineObservation:
    image: Any
    image_member_path: str
    image_png: bytes
    metadata: Mapping[str, Any]
    ocr_record: Mapping[str, Any]

    @property
    def ocr_tokens(self) -> tuple[str, ...]:
        return tuple(self.ocr_record["full_spatial_tokens"])

    @property
    def image_sha256(self) -> str:
        return hashlib.sha256(self.image_png).hexdigest()


class PinnedOnlineOCRProvider:
    """Pinned CPU OCR provider; absence or drift is a hard episode failure."""

    def __init__(
        self,
        *,
        backend_config: Mapping[str, Any],
        backend_config_sha256: str,
        engine: Any,
    ) -> None:
        validate_backend_config(backend_config)
        if backend_config_sha256 != OCR_CONFIG_SHA256:
            raise ValueError("live OCR config differs from the frozen backend")
        self.backend_config = dict(backend_config)
        self.backend_config_sha256 = backend_config_sha256
        self.engine = engine

    @classmethod
    def load(
        cls,
        *,
        backend_config_path: Path,
        backend_manifest_path: Path,
        model_dir: Path,
    ) -> "PinnedOnlineOCRProvider":
        if sha256_file(backend_config_path) != OCR_CONFIG_SHA256:
            raise ValueError("live OCR backend config SHA256 drifted")
        if sha256_file(backend_manifest_path) != OCR_MANIFEST_SHA256:
            raise ValueError("live OCR backend manifest SHA256 drifted")
        config = json.loads(backend_config_path.read_text(encoding="utf-8"))
        return cls(
            backend_config=config,
            backend_config_sha256=OCR_CONFIG_SHA256,
            engine=create_rapidocr_engine(config, model_dir),
        )

    def observe(
        self,
        *,
        image: Any,
        image_member_path: str,
        metadata: Mapping[str, Any],
    ) -> OnlineObservation:
        image_png = encode_deterministic_rgb_png(image)
        record = run_live_rapidocr_record(
            engine=self.engine,
            backend_config=self.backend_config,
            image_member_path=image_member_path,
            image_bytes=image_png,
            backend_config_sha256=self.backend_config_sha256,
        )
        validated = validate_live_ocr_record(
            record,
            image_bytes=image_png,
            backend_config=self.backend_config,
            backend_config_sha256=self.backend_config_sha256,
        )
        return OnlineObservation(
            image=image,
            image_member_path=image_member_path,
            image_png=image_png,
            metadata=dict(metadata),
            ocr_record=validated,
        )


def build_shared_early_step_messages(
    *,
    instruction: str,
    history_events: Sequence[LiveRichEvent],
    current_image: Any,
) -> tuple[Mapping[str, Any], ...]:
    """Use one arm-independent summary-only policy for decisions one to five."""
    events = tuple(history_events)
    if len(events) >= 5:
        raise ValueError("shared early-step policy is restricted to decisions one to five")
    if tuple(event.event_step_id for event in events) != tuple(range(1, len(events) + 1)):
        raise ValueError("early-step history is not a complete ordered prefix")
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Please generate the next move from the task, event summaries, restored "
                "post-action states, and current observation.\n\n"
                f"Instruction: {_nonempty(instruction, 'instruction')}"
            ),
        }
    ]
    for event in events:
        content.append(
            {"type": "text", "text": f"Event summary:\n{event.summary_text}\n"}
        )
    content.extend(
        (
            {"type": "text", "text": "Current observation:"},
            {"type": "image", "image": current_image},
            {"type": "text", "text": GUI_OWL_V2_1_FINAL_USER_INSTRUCTION},
        )
    )
    messages = (
        {
            "role": "system",
            "content": [{"type": "text", "text": GUI_OWL_V2_1_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    )
    if validate_gui_owl_v2_1_native_messages(messages) != 1:
        raise RuntimeError("shared early-step policy exposed a history image")
    return messages


@dataclass(frozen=True)
class PreparedPolicyDecision:
    budget: int
    decision_step_id: int
    high_fidelity_history_image_count: int
    messages: tuple[Mapping[str, Any], ...]
    policy_image_count: int
    request_content_sha256: str | None
    rpc_round_trip_ms: float | None
    selected_event_step_ids: tuple[int, ...]
    selection_diagnostics: Mapping[str, Any] | None
    selection_latency_ms: float | None
    selection_method: str
    selector_response: Mapping[str, Any] | None
    selector_used: bool
    warmup_policy_id: str | None


SelectorCall = Callable[
    [LiveRichSelectorRequest],
    tuple[tuple[int, ...], Mapping[str, Any], float],
]


def select_live_baseline(
    request: LiveRichSelectorRequest,
    *,
    arm: str,
) -> tuple[tuple[int, ...], Mapping[str, Any]]:
    """Select a deterministic baseline over the complete live candidate set."""
    if not isinstance(request, LiveRichSelectorRequest):
        raise TypeError("live baseline requires a validated selector request")
    if arm == "recent":
        selected = tuple(
            recent_budget_selections(request.candidate_event_ids)[str(request.budget)]
        )
        return selected, {
            "candidate_event_count": len(request.candidate_event_ids),
            "method": "recent_full_candidate_history",
        }
    if arm != "ocr_rgb":
        raise ValueError("local live baseline must be recent or ocr_rgb")
    current_rgb = prepare_image_bytes(
        request.current_image.payload
    ).resized_rgb_bytes
    score_pairs = []
    for event in request.events:
        ocr_score = ocr_token_set_jaccard(
            event.post_ocr_tokens,
            request.current_ocr_tokens,
        )
        rgb_score = joint_rgb_histogram_cosine(
            prepare_image_bytes(event.post_image.payload).resized_rgb_bytes,
            current_rgb,
        )
        score_pairs.append(
            (
                event.event_step_id,
                OCR_JACCARD_WEIGHT * ocr_score + RGB_HISTOGRAM_WEIGHT * rgb_score,
            )
        )
    selected = tuple(
        ocr_rgb_budget_selections(
            request.candidate_event_ids,
            [score for _, score in score_pairs],
        )[str(request.budget)]
    )
    return selected, {
        "candidate_event_count": len(request.candidate_event_ids),
        "method": "ocr_rgb_full_candidate_history_strictly_positive_top_B",
        "ocr_jaccard_weight": OCR_JACCARD_WEIGHT,
        "rgb_histogram_weight": RGB_HISTOGRAM_WEIGHT,
        "scores_by_event_step": [
            {"event_step_id": event_id, "score": score}
            for event_id, score in score_pairs
        ],
    }


class LiveSetUtilityPolicyAdapter:
    """Join exact online events, remote selection, and the frozen policy prompt."""

    def __init__(
        self,
        *,
        arm: str,
        authorization: LiveSelectorAuthorization,
        budget: int,
        expected_heldout_result_sha256: str,
        expected_native_replay_result_sha256: str,
        image_decoder: Callable[[bytes], Any],
        instruction: str,
        ocr_backend_config: Mapping[str, Any],
        ocr_backend_config_sha256: str,
        selector_call: SelectorCall | None,
        source_id: str,
    ) -> None:
        if budget not in SUPPORTED_BUDGETS:
            raise ValueError("live AndroidWorld budget must be one of 1,2,3,4")
        if arm not in SUPPORTED_LIVE_ARMS:
            raise ValueError("live AndroidWorld arm must be winner, recent, or ocr_rgb")
        if authorization.heldout_result_sha256 != _sha256(
            expected_heldout_result_sha256, "expected held-out result SHA256"
        ):
            raise ValueError("live adapter held-out GO result SHA256 drifted")
        if authorization.native_replay_result_sha256 != _sha256(
            expected_native_replay_result_sha256,
            "expected native replay result SHA256",
        ):
            raise ValueError("live adapter native replay GO result SHA256 drifted")
        if not callable(image_decoder):
            raise TypeError("live image decoder must be callable")
        if arm == "winner" and not callable(selector_call):
            raise TypeError("winner arm requires a live selector call")
        if arm != "winner" and selector_call is not None:
            raise ValueError("local baseline arms must not receive a selector call")
        validate_backend_config(ocr_backend_config)
        if ocr_backend_config_sha256 != OCR_CONFIG_SHA256:
            raise ValueError("live adapter OCR backend SHA256 drifted")
        self.authorization = authorization
        self.arm = arm
        self.budget = budget
        self.image_decoder = image_decoder
        self.instruction = _nonempty(instruction, "instruction")
        self.ocr_backend_config = dict(ocr_backend_config)
        self.ocr_backend_config_sha256 = ocr_backend_config_sha256
        self.selector_call = selector_call
        self.source_id = _nonempty(source_id, "source id")
        self.history: list[LiveRichEvent] = []

    def _validate_current_equivalence(self, current: OnlineObservation) -> None:
        if not self.history:
            return
        latest = self.history[-1]
        if (
            latest.post_image.sha256 != current.image_sha256
            or latest.post_ocr_tokens != current.ocr_tokens
        ):
            raise ValueError("latest live event is not exactly the current observation")

    def prepare(self, current: OnlineObservation) -> PreparedPolicyDecision:
        if not isinstance(current, OnlineObservation):
            raise TypeError("live policy preparation requires OnlineObservation")
        self._validate_current_equivalence(current)
        decision = len(self.history) + 1
        if decision < 6:
            messages = build_shared_early_step_messages(
                instruction=self.instruction,
                history_events=self.history,
                current_image=current.image,
            )
            return PreparedPolicyDecision(
                budget=self.budget,
                decision_step_id=decision,
                high_fidelity_history_image_count=0,
                messages=messages,
                policy_image_count=1,
                request_content_sha256=None,
                rpc_round_trip_ms=None,
                selected_event_step_ids=(),
                selection_diagnostics=None,
                selection_latency_ms=None,
                selection_method="shared_summary_only_early_step",
                selector_response=None,
                selector_used=False,
                warmup_policy_id=EARLY_STEP_POLICY_ID,
            )
        request = LiveRichSelectorRequest.build(
            authorization_id=self.authorization.authorization_id,
            budget=self.budget,
            current_image_png=current.image_png,
            current_ocr_tokens=current.ocr_tokens,
            decision_step_id=decision,
            events=tuple(event.to_mapping() for event in self.history),
            instruction=self.instruction,
            request_id=f"{self.source_id}:request:{decision:03d}",
            source_id=self.source_id,
        )
        response = None
        measured_rtt = None
        diagnostics = None
        selection_latency_ms = None
        if self.arm == "winner":
            if self.selector_call is None:
                raise RuntimeError("winner selector call became unavailable")
            selected, response, round_trip_ms = self.selector_call(request)
            selected = validate_live_selector_response(
                response,
                request=request,
                authorization=self.authorization,
            )
            measured_rtt = float(round_trip_ms)
            if not math.isfinite(measured_rtt) or measured_rtt < 0.0:
                raise ValueError("live selector RPC RTT is invalid")
            selection_latency_ms = measured_rtt
            selection_method = "go_authorized_winner_rpc"
        else:
            selection_started = time.perf_counter()
            selected, diagnostics = select_live_baseline(request, arm=self.arm)
            selection_latency_ms = (time.perf_counter() - selection_started) * 1000.0
            selection_method = str(diagnostics["method"])
        messages = build_live_mixed_fidelity_messages(
            request,
            selected,
            image_decoder=self.image_decoder,
        )
        image_count = validate_gui_owl_v2_1_native_messages(messages)
        if image_count != len(selected) + 1 or len(selected) > self.budget:
            raise RuntimeError("live policy prompt violated equal visual capacity")
        return PreparedPolicyDecision(
            budget=self.budget,
            decision_step_id=decision,
            high_fidelity_history_image_count=len(selected),
            messages=messages,
            policy_image_count=image_count,
            request_content_sha256=request.content_sha256,
            rpc_round_trip_ms=measured_rtt,
            selected_event_step_ids=selected,
            selection_diagnostics=diagnostics,
            selection_latency_ms=selection_latency_ms,
            selection_method=selection_method,
            selector_response=(dict(response) if response is not None else None),
            selector_used=self.arm == "winner",
            warmup_policy_id=None,
        )

    def append_transition(
        self,
        *,
        action: GUIOwlV2Action,
        before: OnlineObservation,
        after: OnlineObservation,
    ) -> LiveRichEvent:
        if not isinstance(action, GUIOwlV2Action):
            raise TypeError("live transition requires a canonical GUI-Owl action")
        step_id = len(self.history) + 1
        projected = live_history_event_from_transition(
            step_id=step_id,
            action=action,
            before_image_bytes=before.image_png,
            after_image_bytes=after.image_png,
            backend_config=self.ocr_backend_config,
            backend_config_sha256=self.ocr_backend_config_sha256,
            before_ocr_record=before.ocr_record,
            after_ocr_record=after.ocr_record,
            ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
        )
        event = LiveRichEvent.build(
            event_step_id=step_id,
            low_fidelity_v2=projected["low_fidelity_v2"],
            post_image_png=after.image_png,
            post_ocr_tokens=projected["post_ocr_spatial_tokens"],
        )
        self.history.append(event)
        return event


def authorization_from_go_artifacts(
    *,
    heldout_config_path: Path,
    selections_path: Path,
    heldout_result_path: Path,
    native_replay_result_path: Path,
    checkpoint_path: Path,
    expected_heldout_result_sha256: str,
    expected_native_replay_result_sha256: str,
) -> LiveSelectorAuthorization:
    if sha256_file(heldout_result_path) != _sha256(
        expected_heldout_result_sha256, "expected held-out result SHA256"
    ):
        raise ValueError("held-out GO result file SHA256 drifted")
    if sha256_file(native_replay_result_path) != _sha256(
        expected_native_replay_result_sha256,
        "expected native replay result SHA256",
    ):
        raise ValueError("native replay GO result file SHA256 drifted")
    return authorize_live_selector(
        heldout_config_path=heldout_config_path,
        selections_path=selections_path,
        heldout_result_path=heldout_result_path,
        native_replay_result_path=native_replay_result_path,
        checkpoint_path=checkpoint_path,
    )


def probe_live_selector_health(
    *,
    endpoint: str,
    authorization: LiveSelectorAuthorization,
    maximum_rtt_ms: float,
    attempts: int = 3,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    endpoint = validate_loopback_http_url(endpoint, "selector endpoint")
    maximum = _positive_latency(maximum_rtt_ms, "maximum selector RTT")
    if type(attempts) is not int or not 1 <= attempts <= 20:
        raise ValueError("selector health attempts must be in [1,20]")
    latencies = []
    for _ in range(attempts):
        started = time.perf_counter()
        try:
            with opener(endpoint + "/health", timeout=10.0) as response:
                payload = response.read()
                content_type = response.headers.get_content_type()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError("live selector health check failed") from error
        elapsed = (time.perf_counter() - started) * 1000.0
        if content_type != "application/json":
            raise ValueError("live selector health returned non-JSON content")
        value = json.loads(payload.decode("utf-8"))
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version") != LIVE_SELECTOR_SCHEMA_VERSION
            or value.get("status") != SELECTOR_READY_STATUS
            or value.get("authorization") != authorization.to_mapping()
        ):
            raise ValueError("live selector health authorization drifted")
        latencies.append(elapsed)
    observed = max(latencies)
    if observed > maximum:
        raise RuntimeError("live selector loopback RTT exceeds the explicit canary gate")
    return {
        "attempt_count": attempts,
        "authorization_id": authorization.authorization_id,
        "maximum_allowed_rtt_ms": maximum,
        "maximum_observed_rtt_ms": observed,
        "status": "PASSED_LIVE_SELECTOR_LOOPBACK_CANARY",
    }


def selector_http_call(
    *,
    endpoint: str,
    authorization: LiveSelectorAuthorization,
    timeout_seconds: float,
) -> SelectorCall:
    endpoint = validate_loopback_http_url(endpoint, "selector endpoint")

    def call(
        request: LiveRichSelectorRequest,
    ) -> tuple[tuple[int, ...], Mapping[str, Any], float]:
        return select_live_selector_over_http(
            endpoint=endpoint,
            request=request,
            authorization=authorization,
            timeout_seconds=timeout_seconds,
        )

    return call


__all__ = [
    "ANDROIDWORLD_READY_STATUS",
    "EARLY_STEP_POLICY_ID",
    "LiveSetUtilityPolicyAdapter",
    "OCR_CONFIG_SHA256",
    "OCR_MANIFEST_SHA256",
    "OUTCOME_EXPOSED_EVIDENCE_ROLE",
    "OnlineObservation",
    "PinnedOnlineOCRProvider",
    "PreparedPolicyDecision",
    "SUPPORTED_LIVE_ARMS",
    "authorization_from_go_artifacts",
    "build_shared_early_step_messages",
    "encode_deterministic_rgb_png",
    "load_outcome_exposed_validation_instance",
    "probe_live_selector_health",
    "run_live_rapidocr_record",
    "selector_http_call",
    "select_live_baseline",
    "validate_loopback_http_url",
    "validate_live_ocr_record",
]
