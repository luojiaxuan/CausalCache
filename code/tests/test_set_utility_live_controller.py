from __future__ import annotations

import hashlib
import json

import pytest

from causalcache.set_utility_heldout_evaluation import COMPLETED_EVALUATION
from causalcache.set_utility_live_controller import (
    LIVE_SELECTOR_AUTHORIZATION_STATUS,
    LiveRichEvent,
    LiveRichSelectorRequest,
    LiveRichSelectorService,
    LiveSelectorAuthorization,
    LiveSelectorBackendResult,
    authorize_live_selector,
    build_live_mixed_fidelity_messages,
    canonical_json_bytes,
    validate_live_selector_response,
)
from causalcache.set_utility_native_replay_evaluation import (
    COMPLETED_NATIVE_REPLAY_EVALUATION,
)


PNG = b"\x89PNG\r\n\x1a\nsynthetic-live-controller-fixture"
AUTHORIZATION_ID = "a" * 64


def _low(step_id: int) -> dict[str, object]:
    return {
        "step_id": step_id,
        "action_type": "click",
        "action_argument": f"coordinate_bin:x{step_id % 10}_y0",
        "foreground_app": "unknown",
        "screen_text_added": [f"screen-{step_id}"],
        "screen_text_removed": [],
        "screen_change": "medium",
        "executor_result": "unknown",
    }


def _event(step_id: int) -> LiveRichEvent:
    return LiveRichEvent.build(
        event_step_id=step_id,
        low_fidelity_v2=_low(step_id),
        post_image_png=PNG,
        post_ocr_tokens=("screen", str(step_id)),
    )


def _request(*, budget: int = 4, request_id: str = "request-6"):
    events = tuple(_event(step) for step in range(1, 6))
    return LiveRichSelectorRequest.build(
        authorization_id=AUTHORIZATION_ID,
        budget=budget,
        current_image_png=PNG,
        current_ocr_tokens=events[-1].post_ocr_tokens,
        decision_step_id=6,
        events=tuple(event.to_mapping() for event in events),
        instruction="Complete the Android task",
        request_id=request_id,
        source_id="episode-1",
    )


def _authorization() -> LiveSelectorAuthorization:
    return LiveSelectorAuthorization(
        authorization_id=AUTHORIZATION_ID,
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        heldout_result_sha256="d" * 64,
        model_config_sha256="e" * 64,
        model_variant="set_transformer_d512_l16_r2_lr3e4",
        native_replay_result_sha256="1" * 64,
        selections_sha256="f" * 64,
        status=LIVE_SELECTOR_AUTHORIZATION_STATUS,
        winner_model="set_transformer",
    )


def _resign(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def test_request_roundtrip_contains_complete_history_including_latest() -> None:
    request = _request()
    assert request.candidate_event_ids == (1, 2, 3, 4, 5)
    assert request.events[-1].post_image.sha256 == request.current_image.sha256
    assert LiveRichSelectorRequest.from_mapping(request.to_mapping()) == request


@pytest.mark.parametrize("mutation", ("missing", "reordered", "latest_mismatch"))
def test_request_rejects_recent_truncation_or_latest_exclusion(mutation: str) -> None:
    payload = _request().to_mapping()
    if mutation == "missing":
        payload["events"] = payload["events"][1:]
    elif mutation == "reordered":
        payload["events"][0], payload["events"][1] = (
            payload["events"][1],
            payload["events"][0],
        )
    else:
        payload["current_ocr_tokens"] = ["different"]
    with pytest.raises(ValueError):
        LiveRichSelectorRequest.from_mapping(_resign(payload))


def test_service_is_at_most_budget_idempotent_and_client_validated() -> None:
    calls = []

    def backend(request: LiveRichSelectorRequest) -> LiveSelectorBackendResult:
        calls.append(request.content_sha256)
        return LiveSelectorBackendResult(
            selected_event_step_ids=(2, 5),
            selected_predicted_utility=1.5,
            scored_subsets=(((), 0.0), ((2,), 1.0), ((2, 5), 1.5)),
            latency_ms={"selector_total": 3.0},
            source_token_counts={"candidate_visual": 2400, "query_visual": 480},
        )

    request = _request(budget=4)
    service = LiveRichSelectorService(
        authorization=_authorization(), backend=backend
    )
    first = service.select(request.to_mapping())
    second = service.select(request.to_mapping())
    assert first == second
    assert calls == [request.content_sha256]
    assert validate_live_selector_response(
        first, request=request, authorization=_authorization()
    ) == (2, 5)
    assert len(first["selected_event_step_ids"]) < request.budget


def test_live_prompt_keeps_all_summaries_and_can_restore_latest() -> None:
    request = _request()
    decoded = []

    def decode(raw: bytes) -> dict[str, object]:
        decoded.append(raw)
        return {"decode_ordinal": len(decoded), "payload": raw}

    messages = build_live_mixed_fidelity_messages(
        request,
        (2, 5),
        image_decoder=decode,
    )
    content = messages[1]["content"]
    images = [block["image"] for block in content if block["type"] == "image"]
    summaries = [
        block["text"]
        for block in content
        if block["type"] == "text" and block["text"].startswith("Event summary:")
    ]
    assert len(summaries) == 5
    assert len(images) == 3
    assert images[-2]["decode_ordinal"] == 2
    assert images[-1]["decode_ordinal"] == 3
    assert decoded[-2:] == [PNG, PNG]


def test_service_rejects_fixed_budget_overfill_and_stale_response() -> None:
    def invalid_backend(_request):
        return LiveSelectorBackendResult(
            selected_event_step_ids=(1, 2),
            selected_predicted_utility=1.0,
            scored_subsets=(((), 0.0), ((1, 2), 1.0)),
            latency_ms={"selector_total": 1.0},
            source_token_counts={},
        )

    request = _request(budget=1)
    service = LiveRichSelectorService(
        authorization=_authorization(), backend=invalid_backend
    )
    with pytest.raises(ValueError, match="at-most-B"):
        service.select(request.to_mapping())

    valid = LiveRichSelectorService(
        authorization=_authorization(),
        backend=lambda _request: LiveSelectorBackendResult(
            selected_event_step_ids=(),
            selected_predicted_utility=0.0,
            scored_subsets=(((), 0.0),),
            latency_ms={"selector_total": 1.0},
            source_token_counts={},
        ),
    ).select(_request().to_mapping())
    other = _request(request_id="request-other")
    with pytest.raises(ValueError, match="binding drifted"):
        validate_live_selector_response(
            valid, request=other, authorization=_authorization()
        )


def test_request_id_cannot_be_reused_for_mutated_state() -> None:
    service = LiveRichSelectorService(
        authorization=_authorization(),
        backend=lambda _request: LiveSelectorBackendResult(
            selected_event_step_ids=(),
            selected_predicted_utility=0.0,
            scored_subsets=(((), 0.0),),
            latency_ms={"selector_total": 1.0},
            source_token_counts={},
        ),
    )
    request = _request(request_id="stable-id")
    service.select(request.to_mapping())
    mutated = request.to_mapping()
    mutated["instruction"] = "A different task"
    with pytest.raises(ValueError, match="reused"):
        service.select(_resign(mutated))


def test_go_authorization_binds_result_selections_and_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "best.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(b"checkpoint").hexdigest()
    config = {
        "frozen_candidates": {
            "models": {
                "set_transformer": {
                    "sha256": checkpoint_sha,
                    "variant": "set_transformer_d512_l16_r2_lr3e4",
                }
            }
        },
        "representation": {"model_config_sha256": "1" * 64},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    selections = _resign(
        {
            "config_sha256": config_sha,
            "model_artifacts": {
                "set_transformer": {
                    "checkpoint_sha256": checkpoint_sha,
                    "variant": "set_transformer_d512_l16_r2_lr3e4",
                }
            },
            "status": "SEALED_SET_UTILITY_HELDOUT_SELECTIONS",
        }
    )
    selections_path = tmp_path / "selections.json"
    selections_path.write_text(json.dumps(selections), encoding="utf-8")
    selections_sha = hashlib.sha256(selections_path.read_bytes()).hexdigest()
    result = _resign(
        {
            "bindings": {
                "config_sha256": config_sha,
                "selections_sha256": selections_sha,
            },
            "coverage": {
                "complete": True,
                "completed_state_count": 805,
                "expected_state_count": 805,
                "missing_state_ids": [],
                "skipped_state_count": 0,
            },
            "status": COMPLETED_EVALUATION,
            "winner": {
                "model": "set_transformer",
                "verdict": "GO",
            },
        }
    )
    result_path = tmp_path / "result.json"
    result["winner"]["verdict"] = "GO"
    result = _resign(result)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    native_replay = _resign(
        {
            "behavior_recovery_gate": {"verdict": "GO"},
            "bindings": {
                "config_sha256": config_sha,
                "heldout_result_sha256": result_sha,
                "selections_sha256": selections_sha,
            },
            "closed_loop_authorization": {
                "authorized": True,
                "requires": [
                    "behavior_recovery_gate.GO",
                    "deployment_latency_gate.GO",
                ],
                "verdict": "GO",
            },
            "coverage": {
                "complete": True,
                "completed_state_count": 320,
                "expected_state_count": 320,
                "extra_state_ids": [],
                "missing_state_ids": [],
            },
            "deployment_latency_gate": {"verdict": "GO"},
            "status": COMPLETED_NATIVE_REPLAY_EVALUATION,
            "winner_model": "set_transformer",
        }
    )
    native_replay_path = tmp_path / "native-replay-result.json"
    native_replay_path.write_text(json.dumps(native_replay), encoding="utf-8")

    authorization = authorize_live_selector(
        heldout_config_path=config_path,
        selections_path=selections_path,
        heldout_result_path=result_path,
        native_replay_result_path=native_replay_path,
        checkpoint_path=checkpoint,
    )
    assert authorization.winner_model == "set_transformer"
    assert authorization.checkpoint_sha256 == checkpoint_sha

    result["winner"]["verdict"] = "NO_GO"
    result_path.write_text(json.dumps(_resign(result)), encoding="utf-8")
    with pytest.raises(ValueError, match="held-out GO"):
        authorize_live_selector(
            heldout_config_path=config_path,
            selections_path=selections_path,
            heldout_result_path=result_path,
            native_replay_result_path=native_replay_path,
            checkpoint_path=checkpoint,
        )

    result["winner"]["verdict"] = "GO"
    result = _resign(result)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == result_sha
    native_replay["behavior_recovery_gate"]["verdict"] = "NO_GO"
    native_replay_path.write_text(json.dumps(_resign(native_replay)), encoding="utf-8")
    with pytest.raises(ValueError, match="native replay"):
        authorize_live_selector(
            heldout_config_path=config_path,
            selections_path=selections_path,
            heldout_result_path=result_path,
            native_replay_result_path=native_replay_path,
            checkpoint_path=checkpoint,
        )
