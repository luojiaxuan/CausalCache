"""Validate frozen restoration-v2 actions through the pinned AndroidWorld executor."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2 import (
    gui_owl_v2_action_to_androidworld,
    parse_gui_owl_v2_output,
)
from scripts.validate_restoration_v2_interfaces import validate_interfaces


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FROZEN_ANDROIDWORLD_REVISION = "11cea575561fb7800b5fb6b6cafa56f7a91de11f"
FROZEN_SERVER_IMAGE = "causalcache-androidworld:11cea575-executor1"
FROZEN_SERVER_IMAGE_ID = (
    "sha256:542e11e5d263ddcd3dffc52c5be2cb2aca0b1f08bbcf2120cecb8150b8d51486"
)
FROZEN_JSON_ACTION_SCHEMA = "android_world.agents.new_json_action"
FROZEN_EXECUTION_HOST = "aries.cs.ucsb.edu"
FROZEN_STACK_CONFIG_SHA256 = "3a14ca4b1464ae4293944ef3387c6f1ad2a3a716dc8af29258a92bd4d6a22481"
FROZEN_INTERFACE_MANIFEST_SHA256 = (
    "5a6166febe7d7575b7dc5dc7637c77cadec48b569a00121cbd5766c67e9c4860"
)
FROZEN_SCIENTIFIC_CONTRACT_SHA256 = (
    "9b9b78d9e1902d6ba7c648c939809c56fe55cccc17de58d4e6eed8d9ddf746cc"
)
FROZEN_ACTION_FIXTURE_SHA256 = (
    "ab6ca84a894a601450424150a510d3e9a2cb4a5522ee0d050ba7a5908af661de"
)
FROZEN_INSPECTOR_EVIDENCE_TYPE = "restoration_v2_live_executor_container_inspection"
FROZEN_LIVE_SOURCE_SHA256 = {
    "/server/android_server.py": "3433a814b6753490c6af0bb24cadb68648dd78b133aac288fdf63b6d5e2a928e",
    "/android_world/agents/new_json_action.py": "14ca00cabf3d5b83e4d55cb683a09a4beccbbc658e21039ca5cf8cef3f543e3f",
    "/android_world/env/actuation.py": "69d1190d1c6dd250d1ca228ea91ffb5fea3fd133b21995b3b298b500612ed925",
    "/android_world/env/interface.py": "1772e5485225f4e52dc68954aa5769d4e49a9e63174f9ec10f3e52560d5646c2",
}
EXPECTED_DISPATCH_CASE_IDS = (
    "click_minimum",
    "click_maximum",
    "tap_alias",
    "long_press",
    "swipe",
    "type_nfkc",
    "system_back",
    "system_home",
    "system_enter",
    "open",
    "open_app_alias",
    "wait",
    "answer",
    "terminate",
)
NEGATIVE_ACTUATION_CONTROL = {"action_type": "click"}
JSON_ACTION_FIELD_ORDER = (
    "action_type",
    "index",
    "x",
    "y",
    "text",
    "direction",
    "goal_status",
    "app_name",
    "keycode",
)
CONTAINER_NAME_PATTERN = re.compile(r"sglang-omni-jaxan-[0-9]{8}")
CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
ATTEMPT_ID_PATTERN = re.compile(r"rv2-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}")


class InvalidDispatchEvidence(ValueError):
    """The runtime or evidence cannot support an executor compatibility claim."""


class ExecutorCompatibilityFailure(ValueError):
    """A frozen valid action failed at the pinned executor endpoint."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def expected_executor_message(payload: Mapping[str, Any]) -> str:
    unknown = set(payload) - set(JSON_ACTION_FIELD_ORDER)
    if unknown:
        raise ValueError(f"payload has fields absent from pinned JSONAction: {sorted(unknown)}")
    properties = [
        f"{key}={payload[key]!r}" for key in JSON_ACTION_FIELD_ORDER if key in payload
    ]
    return f"Action JSONAction({', '.join(properties)}) executed."


def build_url(
    base_url: str,
    path: str,
    params: Mapping[str, Any] | None = None,
) -> str:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    return url


def request_record(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    params: Mapping[str, Any] | None = None,
    body: Mapping[str, Any] | None = None,
    timeout_seconds: float,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if body is not None:
        data = json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif method == "POST":
        data = b""
    request = urllib.request.Request(
        build_url(base_url, path, params),
        data=data,
        headers=headers,
        method=method,
    )
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    try:
        response = urllib.request.urlopen(request, timeout=timeout_seconds)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        raw = response.read()
        content_type = response.headers.get_content_type()
        status_code = int(response.status)
    finished_at = datetime.now(timezone.utc)
    record: dict[str, Any] = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "http_status": status_code,
        "content_type": content_type,
        "response_bytes": len(raw),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "raw_body_utf8": raw.decode("utf-8"),
    }
    if content_type == "application/json":
        decoded = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        record["json_body"] = decoded
    else:
        record["body_text"] = raw.decode("utf-8", errors="replace")
    return record


def request_screenshot_summary(
    base_url: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        build_url(
            base_url,
            "/screenshot",
            {"wait_to_stabilize": "false"},
        ),
        method="GET",
    )
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read()
        content_type = response.headers.get_content_type()
        status_code = int(response.status)
    finished_at = datetime.now(timezone.utc)
    if status_code != 200 or content_type != "application/json":
        raise InvalidDispatchEvidence("live screenshot endpoint did not return HTTP 200 JSON")
    decoded = json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(decoded, dict) or set(decoded) != {"pixels"}:
        raise InvalidDispatchEvidence("live screenshot must contain exactly a pixels field")
    pixels = decoded["pixels"]
    if not isinstance(pixels, list) or not pixels or not isinstance(pixels[0], list):
        raise InvalidDispatchEvidence("live screenshot pixels must be a non-empty image array")
    height = len(pixels)
    width = len(pixels[0])
    if any(not isinstance(row, list) or len(row) != width for row in pixels):
        raise InvalidDispatchEvidence("live screenshot rows have inconsistent width")
    if [width, height] != [1080, 2400]:
        raise InvalidDispatchEvidence("live screenshot does not match the frozen 1080x2400 frame")
    return {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "http_status": status_code,
        "content_type": content_type,
        "response_bytes": len(raw),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "width": width,
        "height": height,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def _require_transport_metadata(
    record: Mapping[str, Any],
    label: str,
    *,
    require_raw_body: bool = True,
) -> None:
    if type(record.get("started_at")) is not str or type(record.get("finished_at")) is not str:
        raise InvalidDispatchEvidence(f"{label} lacks request timestamps")
    try:
        started_at = datetime.fromisoformat(str(record["started_at"]))
        finished_at = datetime.fromisoformat(str(record["finished_at"]))
    except ValueError as error:
        raise InvalidDispatchEvidence(f"{label} has invalid request timestamps") from error
    if started_at.utcoffset() is None or finished_at.utcoffset() is None:
        raise InvalidDispatchEvidence(f"{label} request timestamps must include a timezone")
    if finished_at < started_at:
        raise InvalidDispatchEvidence(f"{label} request timestamps are reversed")
    if type(record.get("response_bytes")) is not int or record["response_bytes"] < 0:
        raise InvalidDispatchEvidence(f"{label} has invalid response byte accounting")
    digest = record.get("response_sha256")
    if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise InvalidDispatchEvidence(f"{label} has invalid response SHA256")
    elapsed = record.get("elapsed_seconds")
    if type(elapsed) not in {int, float} or elapsed < 0:
        raise InvalidDispatchEvidence(f"{label} has invalid request latency")
    if not require_raw_body:
        return
    raw_text = record.get("raw_body_utf8")
    if type(raw_text) is not str:
        raise InvalidDispatchEvidence(f"{label} lacks the raw UTF-8 response body")
    raw = raw_text.encode("utf-8")
    if len(raw) != record["response_bytes"]:
        raise InvalidDispatchEvidence(f"{label} raw body length does not match metadata")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise InvalidDispatchEvidence(f"{label} raw body SHA256 does not match metadata")
    if record.get("content_type") == "application/json":
        decoded = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if record.get("json_body") != decoded:
            raise InvalidDispatchEvidence(f"{label} decoded JSON differs from raw response")
    elif record.get("body_text") != raw_text:
        raise InvalidDispatchEvidence(f"{label} decoded text differs from raw response")


def _require_success_record(
    record: Mapping[str, Any],
    expected_body: Mapping[str, Any],
    label: str,
    *,
    error_type: type[ValueError] = InvalidDispatchEvidence,
) -> None:
    _require_transport_metadata(record, label)
    if record.get("http_status") != 200:
        raise error_type(f"{label} returned non-200 HTTP status")
    if record.get("content_type") != "application/json":
        raise error_type(f"{label} returned non-JSON content")
    if record.get("json_body") != expected_body:
        raise error_type(f"{label} response did not exactly match the pinned response")


def load_dispatch_cases(action_fixture_path: Path) -> list[dict[str, Any]]:
    if hashlib.sha256(action_fixture_path.read_bytes()).hexdigest() != FROZEN_ACTION_FIXTURE_SHA256:
        raise InvalidDispatchEvidence("action fixture differs from the frozen SHA256")
    fixture = read_json(action_fixture_path)
    screen = fixture.get("screen")
    if screen != {"width": 1080, "height": 2400}:
        raise InvalidDispatchEvidence("action fixture screen must remain exactly 1080x2400")
    raw_cases = fixture.get("valid_cases")
    if not isinstance(raw_cases, list):
        raise ValueError("action fixture valid_cases must be a list")
    cases: list[dict[str, Any]] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            raise ValueError("each valid action case must be a JSON object")
        case_id = item.get("id")
        native_output = item.get("native_output")
        canonical_arguments = item.get("canonical_arguments")
        payload = item.get("androidworld_payload")
        if (
            type(case_id) is not str
            or type(native_output) is not str
            or not isinstance(canonical_arguments, dict)
            or not isinstance(payload, dict)
        ):
            raise ValueError(
                "dispatch cases require string id/native output and object canonical/payload fields"
            )
        parsed = parse_gui_owl_v2_output(native_output)
        regenerated_payload = gui_owl_v2_action_to_androidworld(
            parsed.canonical_action,
            screen_width=screen["width"],
            screen_height=screen["height"],
        )
        if parsed.canonical_action.arguments() != canonical_arguments:
            raise InvalidDispatchEvidence(
                f"canonical action drift while regenerating dispatch case: {case_id}"
            )
        if regenerated_payload != payload:
            raise InvalidDispatchEvidence(
                f"parser/bridge payload drift while regenerating dispatch case: {case_id}"
            )
        cases.append(
            {
                "case_id": case_id,
                "native_output_sha256": hashlib.sha256(native_output.encode("utf-8")).hexdigest(),
                "canonical_arguments": canonical_arguments,
                "payload": regenerated_payload,
                "payload_sha256": hashlib.sha256(
                    json.dumps(
                        regenerated_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
    case_ids = tuple(item["case_id"] for item in cases)
    if case_ids != EXPECTED_DISPATCH_CASE_IDS:
        raise ValueError("dispatch case order or coverage differs from the frozen fixture")
    return cases


def validate_negative_control_constructor(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source_root))
    try:
        from android_world.agents import new_json_action
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = previous_dont_write_bytecode
    module_path = Path(new_json_action.__file__).resolve()
    if not module_path.is_relative_to(source_root):
        raise InvalidDispatchEvidence(
            "negative control JSONAction module is outside the pinned source root"
        )
    module_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    expected_module_sha256 = FROZEN_LIVE_SOURCE_SHA256[
        "/android_world/agents/new_json_action.py"
    ]
    if module_sha256 != expected_module_sha256:
        raise InvalidDispatchEvidence(
            "negative control constructor source differs from the live executor source"
        )
    action = new_json_action.JSONAction(**NEGATIVE_ACTUATION_CONTROL)
    if repr(action) != "JSONAction(action_type='click')":
        raise InvalidDispatchEvidence("negative control constructor semantics drifted")
    if action.json_str() != '{"action_type":"click"}':
        raise InvalidDispatchEvidence("negative control JSON serialization semantics drifted")
    return {
        "status": "passed",
        "payload": NEGATIVE_ACTUATION_CONTROL,
        "json_action_repr": repr(action),
        "json_action_json_str": action.json_str(),
        "module_path": str(module_path),
        "module_sha256": module_sha256,
    }


Requester = Callable[..., dict[str, Any]]
ScreenshotRequester = Callable[..., dict[str, Any]]


def dispatch_actions(
    *,
    base_url: str,
    cases: Sequence[Mapping[str, Any]],
    timeout_seconds: float,
    requester: Requester = request_record,
    screenshot_requester: ScreenshotRequester = request_screenshot_summary,
) -> dict[str, Any]:
    if tuple(item["case_id"] for item in cases) != EXPECTED_DISPATCH_CASE_IDS:
        raise ValueError("executor dispatch requires the exact frozen case sequence")
    health_before = requester(
        base_url,
        "/health",
        timeout_seconds=timeout_seconds,
    )
    _require_success_record(health_before, {"status": "success"}, "health-before")
    action_records: list[dict[str, Any]] = []
    negative_control: dict[str, Any] | None = None
    health_after_negative_control: dict[str, Any] | None = None
    cleanup_reset: dict[str, Any] | None = None
    try:
        for index, item in enumerate(cases):
            case_id = str(item["case_id"])
            payload = item["payload"]
            if not isinstance(payload, Mapping):
                raise ValueError(f"dispatch payload is not an object: {case_id}")
            reset_before_case = requester(
                base_url,
                "/reset",
                method="POST",
                params={"go_home": "true"},
                timeout_seconds=timeout_seconds,
            )
            _require_success_record(
                reset_before_case,
                {
                    "status": "success",
                    "message": "Environment reset with go_home=True.",
                },
                f"reset-before-{case_id}",
            )
            live_screenshot = None
            if index == 0:
                live_screenshot = screenshot_requester(
                    base_url,
                    timeout_seconds=timeout_seconds,
                )
            response = requester(
                base_url,
                "/execute_action",
                method="POST",
                body=payload,
                timeout_seconds=timeout_seconds,
            )
            _require_success_record(
                response,
                {
                    "status": "success",
                    "message": expected_executor_message(payload),
                },
                case_id,
                error_type=ExecutorCompatibilityFailure,
            )
            action_records.append(
                {
                    "index": index,
                    "case_id": case_id,
                    "native_output_sha256": item["native_output_sha256"],
                    "canonical_arguments": item["canonical_arguments"],
                    "payload": dict(payload),
                    "payload_sha256": item["payload_sha256"],
                    "reset_before_case": reset_before_case,
                    "live_screenshot_before_first_case": live_screenshot,
                    "response": response,
                }
            )

        reset_before_negative_control = requester(
            base_url,
            "/reset",
            method="POST",
            params={"go_home": "true"},
            timeout_seconds=timeout_seconds,
        )
        _require_success_record(
            reset_before_negative_control,
            {"status": "success", "message": "Environment reset with go_home=True."},
            "reset-before-negative-control",
        )
        negative_response = requester(
            base_url,
            "/execute_action",
            method="POST",
            body=NEGATIVE_ACTUATION_CONTROL,
            timeout_seconds=timeout_seconds,
        )
        if negative_response.get("http_status") != 500:
            raise InvalidDispatchEvidence(
                "negative actuation control did not reach the pinned 500 path"
            )
        negative_control = {
            "purpose": "constructor accepts the action, but actuation must reject missing coordinates",
            "payload": NEGATIVE_ACTUATION_CONTROL,
            "reset_before": reset_before_negative_control,
            "response": negative_response,
        }
        health_after_negative_control = requester(
            base_url,
            "/health",
            timeout_seconds=timeout_seconds,
        )
        _require_success_record(
            health_after_negative_control,
            {"status": "success"},
            "health-after-negative-control",
        )
    finally:
        cleanup_reset = requester(
            base_url,
            "/reset",
            method="POST",
            params={"go_home": "true"},
            timeout_seconds=timeout_seconds,
        )
        _require_success_record(
            cleanup_reset,
            {"status": "success", "message": "Environment reset with go_home=True."},
            "cleanup-reset",
        )

    action_type_counts: dict[str, int] = {}
    for item in action_records:
        action_type = str(item["payload"]["action_type"])
        action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1
    return {
        "status": "passed",
        "validated_case_count": len(action_records),
        "case_ids": [item["case_id"] for item in action_records],
        "androidworld_action_type_counts": action_type_counts,
        "health_before": health_before,
        "action_records": action_records,
        "negative_actuation_control": negative_control,
        "health_after_negative_control": health_after_negative_control,
        "cleanup_reset": cleanup_reset,
    }


def validate_dispatch_evidence(
    summary: Mapping[str, Any],
    *,
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if summary.get("evidence_type") != "restoration_v2_androidworld_executor_dispatch":
        raise InvalidDispatchEvidence("summary has the wrong executor evidence type")
    if summary.get("status") != "passed":
        raise InvalidDispatchEvidence("executor evidence is not marked passed")
    if summary.get("policy_output_generated") is not False:
        raise InvalidDispatchEvidence("executor evidence must not contain policy output")
    if summary.get("restoration_label_generated") is not False:
        raise InvalidDispatchEvidence("executor evidence must not contain restoration labels")
    try:
        dispatch_started = datetime.fromisoformat(str(summary["started_at"]))
        dispatch_finished = datetime.fromisoformat(str(summary["finished_at"]))
    except (KeyError, ValueError) as error:
        raise InvalidDispatchEvidence("executor evidence has invalid dispatch timestamps") from error
    if dispatch_started.utcoffset() is None or dispatch_finished.utcoffset() is None:
        raise InvalidDispatchEvidence("executor dispatch timestamps must include a timezone")
    if dispatch_finished < dispatch_started:
        raise InvalidDispatchEvidence("executor dispatch timestamps are reversed")

    def require_inside_dispatch(record: Mapping[str, Any], label: str) -> None:
        started_at = datetime.fromisoformat(str(record["started_at"]))
        finished_at = datetime.fromisoformat(str(record["finished_at"]))
        if not dispatch_started <= started_at <= finished_at <= dispatch_finished:
            raise InvalidDispatchEvidence(
                f"{label} request lies outside the dispatch interval"
            )
    interface = summary.get("interface_validation")
    if not isinstance(interface, Mapping):
        raise InvalidDispatchEvidence("executor evidence lacks interface validation")
    constructor = interface.get("androidworld_json_action_constructor_validation")
    if not isinstance(constructor, Mapping) or constructor.get("status") != "passed":
        raise InvalidDispatchEvidence("executor evidence lacks a passing constructor validation")
    negative_constructor = summary.get("negative_control_constructor_validation")
    expected_negative_constructor = {
        "status": "passed",
        "payload": NEGATIVE_ACTUATION_CONTROL,
        "json_action_repr": "JSONAction(action_type='click')",
        "json_action_json_str": '{"action_type":"click"}',
        "module_path": constructor.get("module_path"),
        "module_sha256": constructor.get("module_sha256"),
    }
    if negative_constructor != expected_negative_constructor:
        raise InvalidDispatchEvidence(
            "negative actuation control lacks a pinned constructor-acceptance witness"
        )
    if negative_constructor["module_sha256"] != FROZEN_LIVE_SOURCE_SHA256[
        "/android_world/agents/new_json_action.py"
    ]:
        raise InvalidDispatchEvidence(
            "negative control constructor witness differs from the live executor source"
        )
    dispatch = summary.get("androidworld_executor_dispatch_validation")
    if not isinstance(dispatch, Mapping) or dispatch.get("status") != "passed":
        raise InvalidDispatchEvidence("executor dispatch aggregate is absent or non-passing")
    records = dispatch.get("action_records")
    if not isinstance(records, list) or len(records) != len(cases):
        raise InvalidDispatchEvidence("executor evidence has the wrong action denominator")
    expected_case_ids = [str(item["case_id"]) for item in cases]
    if dispatch.get("validated_case_count") != len(cases):
        raise InvalidDispatchEvidence("executor aggregate count differs from raw records")
    if dispatch.get("case_ids") != expected_case_ids:
        raise InvalidDispatchEvidence("executor aggregate case IDs differ from the frozen order")
    if len(set(expected_case_ids)) != len(expected_case_ids):
        raise InvalidDispatchEvidence("frozen executor case IDs are not unique")

    recomputed_counts: dict[str, int] = {}
    for index, (record, case) in enumerate(zip(records, cases, strict=True)):
        if not isinstance(record, Mapping):
            raise InvalidDispatchEvidence("executor action record must be an object")
        payload = case["payload"]
        expected_static = {
            "index": index,
            "case_id": case["case_id"],
            "native_output_sha256": case["native_output_sha256"],
            "canonical_arguments": case["canonical_arguments"],
            "payload": payload,
            "payload_sha256": case["payload_sha256"],
        }
        for key, value in expected_static.items():
            if record.get(key) != value:
                raise InvalidDispatchEvidence(
                    f"executor raw record drifted at {case['case_id']} field {key}"
                )
        reset = record.get("reset_before_case")
        if not isinstance(reset, Mapping):
            raise InvalidDispatchEvidence("each executor case requires its own reset record")
        _require_success_record(
            reset,
            {"status": "success", "message": "Environment reset with go_home=True."},
            f"offline-reset-{case['case_id']}",
        )
        require_inside_dispatch(reset, f"offline-reset-{case['case_id']}")
        screenshot = record.get("live_screenshot_before_first_case")
        if index == 0:
            if (
                not isinstance(screenshot, Mapping)
                or screenshot.get("http_status") != 200
                or screenshot.get("content_type") != "application/json"
                or [screenshot.get("width"), screenshot.get("height")] != [1080, 2400]
            ):
                raise InvalidDispatchEvidence("live executor screenshot is not the frozen 1080x2400 frame")
            _require_transport_metadata(
                screenshot,
                "offline-live-screenshot",
                require_raw_body=False,
            )
            require_inside_dispatch(screenshot, "offline-live-screenshot")
        elif screenshot is not None:
            raise InvalidDispatchEvidence("only the first case may carry the shared screenshot check")
        response = record.get("response")
        if not isinstance(response, Mapping):
            raise InvalidDispatchEvidence("executor raw record lacks a response")
        _require_success_record(
            response,
            {
                "status": "success",
                "message": expected_executor_message(payload),
            },
            f"offline-{case['case_id']}",
            error_type=ExecutorCompatibilityFailure,
        )
        require_inside_dispatch(response, f"offline-{case['case_id']}")
        action_type = str(payload["action_type"])
        recomputed_counts[action_type] = recomputed_counts.get(action_type, 0) + 1
    if dispatch.get("androidworld_action_type_counts") != recomputed_counts:
        raise InvalidDispatchEvidence("executor action-type aggregate differs from raw records")

    health_before = dispatch.get("health_before")
    health_after = dispatch.get("health_after_negative_control")
    cleanup = dispatch.get("cleanup_reset")
    if not all(isinstance(item, Mapping) for item in (health_before, health_after, cleanup)):
        raise InvalidDispatchEvidence("executor evidence lacks health or cleanup records")
    _require_success_record(health_before, {"status": "success"}, "offline-health-before")
    require_inside_dispatch(health_before, "offline-health-before")
    _require_success_record(
        health_after,
        {"status": "success"},
        "offline-health-after-negative-control",
    )
    require_inside_dispatch(
        health_after, "offline-health-after-negative-control"
    )
    _require_success_record(
        cleanup,
        {"status": "success", "message": "Environment reset with go_home=True."},
        "offline-cleanup-reset",
    )
    require_inside_dispatch(cleanup, "offline-cleanup-reset")
    negative = dispatch.get("negative_actuation_control")
    if not isinstance(negative, Mapping) or negative.get("payload") != NEGATIVE_ACTUATION_CONTROL:
        raise InvalidDispatchEvidence("negative actuation control payload is absent or mutated")
    negative_response = negative.get("response")
    if not isinstance(negative_response, Mapping) or negative_response.get("http_status") != 500:
        raise InvalidDispatchEvidence("negative actuation control did not produce HTTP 500")
    _require_transport_metadata(negative_response, "offline-negative-actuation-control")
    require_inside_dispatch(negative_response, "offline-negative-actuation-control")
    negative_reset = negative.get("reset_before")
    if not isinstance(negative_reset, Mapping):
        raise InvalidDispatchEvidence("negative actuation control lacks its own reset")
    _require_success_record(
        negative_reset,
        {"status": "success", "message": "Environment reset with go_home=True."},
        "offline-reset-before-negative-control",
    )
    require_inside_dispatch(negative_reset, "offline-reset-before-negative-control")
    return {
        "status": "PASSED_EXECUTOR_DISPATCH",
        "validated_case_count": len(records),
        "negative_actuation_control_status": 500,
    }


def validate_server_provenance(
    args: argparse.Namespace,
    *,
    runtime_hostname: str,
) -> dict[str, Any]:
    stack_path = args.stack_config.resolve()
    expected_stack_path = REPOSITORY_ROOT / "code/configs/androidworld_stack.json"
    if stack_path != expected_stack_path.resolve():
        raise ValueError("stack config must be the canonical repository-pinned file")
    stack = read_json(stack_path)
    if hashlib.sha256(stack_path.read_bytes()).hexdigest() != FROZEN_STACK_CONFIG_SHA256:
        raise InvalidDispatchEvidence("AndroidWorld stack config differs from the frozen SHA256")
    benchmark = stack.get("benchmark")
    transport = stack.get("server_transport")
    if not isinstance(benchmark, dict) or not isinstance(transport, dict):
        raise ValueError("AndroidWorld stack lacks benchmark or server transport provenance")
    expected_transport = {
        "image": FROZEN_SERVER_IMAGE,
        "image_sha256": FROZEN_SERVER_IMAGE_ID.removeprefix("sha256:"),
        "json_action_schema": FROZEN_JSON_ACTION_SCHEMA,
        "four_coordinate_swipe_smoke": True,
    }
    if transport != expected_transport:
        raise ValueError("AndroidWorld stack server transport differs from the frozen contract")
    if benchmark.get("agent_revision") != FROZEN_ANDROIDWORLD_REVISION:
        raise ValueError("AndroidWorld stack agent revision differs from the frozen contract")
    if args.androidworld_source_revision != FROZEN_ANDROIDWORLD_REVISION:
        raise ValueError("executor run uses a non-frozen AndroidWorld revision")
    if args.server_image != FROZEN_SERVER_IMAGE:
        raise ValueError("executor run uses a non-frozen server image tag")
    if args.server_image_id != FROZEN_SERVER_IMAGE_ID:
        raise ValueError("executor run uses a non-frozen server image ID")
    if args.server_json_action_schema != FROZEN_JSON_ACTION_SCHEMA:
        raise ValueError("executor run uses a non-frozen JSONAction schema")
    if args.host_name != FROZEN_EXECUTION_HOST:
        raise ValueError("executor dispatch must run on the frozen Aries host")
    if CONTAINER_NAME_PATTERN.fullmatch(args.runtime_container_name) is None:
        raise ValueError("runtime container name violates the ownership/timestamp contract")
    if CONTAINER_NAME_PATTERN.fullmatch(args.server_container_name) is None:
        raise ValueError("server container name violates the ownership/timestamp contract")
    if CONTAINER_ID_PATTERN.fullmatch(args.runtime_container_id) is None:
        raise ValueError("runtime container ID must be full lowercase hex")
    if CONTAINER_ID_PATTERN.fullmatch(args.server_container_id) is None:
        raise ValueError("server container ID must be full lowercase hex")
    if runtime_hostname != args.runtime_container_id[:12]:
        raise ValueError("runtime hostname does not match the supplied runtime container ID")
    if SHA256_PATTERN.fullmatch(args.runtime_image_id) is None:
        raise ValueError("runtime image ID must use sha256:<64 lowercase hex>")
    if SHA256_PATTERN.fullmatch(args.runtime_image_repo_digest) is None:
        raise ValueError("runtime image repo digest must use sha256:<64 lowercase hex>")
    parsed_base_url = urllib.parse.urlsplit(args.base_url)
    if (
        parsed_base_url.scheme != "http"
        or parsed_base_url.hostname != "172.17.0.1"
        or parsed_base_url.port not in {5000, 5001, 5002, 5003}
        or parsed_base_url.path not in {"", "/"}
        or parsed_base_url.query
        or parsed_base_url.fragment
        or parsed_base_url.username is not None
        or parsed_base_url.password is not None
    ):
        raise ValueError("base URL must be an exact Aries Docker-bridge worker endpoint")
    if args.server_host_port != parsed_base_url.port:
        raise ValueError("server host port does not match the executor base URL")
    if args.server_container_port != 5000:
        raise ValueError("server container port must remain 5000")
    if args.timeout_seconds <= 0 or args.timeout_seconds > 300:
        raise ValueError("timeout seconds must be in (0, 300]")
    host_inspection = read_json(args.host_inspection)
    host_inspection_raw = args.host_inspection.read_text(encoding="utf-8")
    host_inspection_sha256 = hashlib.sha256(host_inspection_raw.encode("utf-8")).hexdigest()
    inspector_source = REPOSITORY_ROOT / "code/scripts/inspect_restoration_v2_executor_container.py"
    if host_inspection.get("evidence_type") != FROZEN_INSPECTOR_EVIDENCE_TYPE:
        raise InvalidDispatchEvidence("host inspection has the wrong evidence type")
    if host_inspection.get("status") != "passed" or host_inspection.get("host") != args.host_name:
        raise InvalidDispatchEvidence("host inspection is not a passing Aries record")
    if (
        host_inspection.get("attempt_id") != args.attempt_id
        or host_inspection.get("phase") != "before"
    ):
        raise InvalidDispatchEvidence("host inspection does not belong to this pre-run attempt")
    if host_inspection.get("inspector_source_sha256") != hashlib.sha256(
        inspector_source.read_bytes()
    ).hexdigest():
        raise InvalidDispatchEvidence("host inspection was produced by different source bytes")
    inspected_runtime = host_inspection.get("runtime_container")
    inspected_server = host_inspection.get("server_container")
    if not isinstance(inspected_runtime, dict) or not isinstance(inspected_server, dict):
        raise InvalidDispatchEvidence("host inspection lacks container records")
    expected_runtime = {
        "name": args.runtime_container_name,
        "id": args.runtime_container_id,
        "hostname": args.runtime_container_id[:12],
        "config_image": "hongccc/sglang-omni:dev",
        "image_id": args.runtime_image_id,
        "image_repo_digest": f"hongccc/sglang-omni@{args.runtime_image_repo_digest}",
        "mounts": {
            "/data": "/mnt/data6/jiaxuanluo/causalcache",
            "/root/.cache/huggingface": "/mnt/data6/jiaxuanluo/causalcache/.cache/huggingface",
        },
        "running": True,
    }
    expected_server = {
        "name": args.server_container_name,
        "id": args.server_container_id,
        "hostname": args.server_container_id[:12],
        "config_image": args.server_image,
        "image_id": args.server_image_id,
        "host_port": args.server_host_port,
        "container_port": args.server_container_port,
        "running": True,
    }
    if inspected_runtime != expected_runtime or inspected_server != expected_server:
        raise InvalidDispatchEvidence("CLI container provenance differs from live docker inspection")
    if host_inspection.get("live_source_sha256") != FROZEN_LIVE_SOURCE_SHA256:
        raise InvalidDispatchEvidence("live server source hashes differ from the frozen executor")
    health = host_inspection.get("health")
    if not isinstance(health, dict) or health.get("json_body") != {"status": "success"}:
        raise InvalidDispatchEvidence("host inspection lacks an exact live health response")
    _require_transport_metadata(health, "host-inspection-health")
    if health.get("http_status") != 200 or health.get("content_type") != "application/json":
        raise InvalidDispatchEvidence("host inspection health transport is not exact HTTP 200 JSON")
    return {
        "host": args.host_name,
        "base_url": args.base_url.rstrip("/"),
        "runtime_container": {
            "name": args.runtime_container_name,
            "id": args.runtime_container_id,
            "image_id": args.runtime_image_id,
            "image_repo_digest": args.runtime_image_repo_digest,
            "mounts": expected_runtime["mounts"],
        },
        "server_container": {
            "name": args.server_container_name,
            "id": args.server_container_id,
            "image": args.server_image,
            "image_id": args.server_image_id,
            "host_port": args.server_host_port,
            "container_port": args.server_container_port,
            "json_action_schema": args.server_json_action_schema,
            "host_health_url": health["url"],
        },
        "androidworld_source_revision": args.androidworld_source_revision,
        "stack_config": str(stack_path),
        "stack_config_sha256": hashlib.sha256(stack_path.read_bytes()).hexdigest(),
        "host_inspection": str(args.host_inspection.resolve()),
        "host_inspection_sha256": host_inspection_sha256,
        "host_inspection_before": host_inspection,
        "host_inspection_before_raw_utf8": host_inspection_raw,
        "live_source_sha256": FROZEN_LIVE_SOURCE_SHA256,
    }


def run_validation(
    args: argparse.Namespace,
    *,
    requester: Requester = request_record,
    screenshot_requester: ScreenshotRequester = request_screenshot_summary,
    runtime_hostname: str | None = None,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    if ATTEMPT_ID_PATTERN.fullmatch(args.attempt_id) is None:
        raise InvalidDispatchEvidence("attempt ID must use rv2-YYYYMMDDTHHMMSSZ-8hex")
    if args.attempt_id not in args.output_summary.name:
        raise InvalidDispatchEvidence("runner output filename must contain the unique attempt ID")
    if hashlib.sha256(args.interface_manifest.read_bytes()).hexdigest() != FROZEN_INTERFACE_MANIFEST_SHA256:
        raise InvalidDispatchEvidence("interface manifest differs from the frozen SHA256")
    interface_summary = validate_interfaces(
        contract_path=args.contract,
        action_fixture_path=args.action_fixture,
        prompt_fixture_path=args.prompt_fixture,
        interface_manifest_path=args.interface_manifest,
        androidworld_source_root=args.androidworld_source_root,
        androidworld_source_revision=args.androidworld_source_revision,
        container_image_digest=args.runtime_image_repo_digest,
        run_git_commit=args.run_git_commit,
    )
    constructor = interface_summary["androidworld_json_action_constructor_validation"]
    if constructor.get("status") != "passed" or constructor.get("validated_case_count") != 14:
        raise ValueError("executor dispatch requires a passing 14-case constructor preflight")
    negative_constructor = validate_negative_control_constructor(
        args.androidworld_source_root
    )
    origin_main = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "refs/remotes/origin/main"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if origin_main != args.run_git_commit:
        raise InvalidDispatchEvidence("CausalCache HEAD is not synchronized with origin/main")
    server_provenance = validate_server_provenance(
        args,
        runtime_hostname=runtime_hostname or socket.gethostname(),
    )
    cases = load_dispatch_cases(args.action_fixture)
    dispatch = dispatch_actions(
        base_url=args.base_url,
        cases=cases,
        timeout_seconds=args.timeout_seconds,
        requester=requester,
        screenshot_requester=screenshot_requester,
    )
    finished_at = datetime.now(timezone.utc)
    summary = {
        "schema_version": "1.0.0",
        "evidence_type": "restoration_v2_androidworld_executor_dispatch",
        "protocol_id": "causalcache_restoration_v2",
        "step": "androidworld_executor_dispatch_preflight",
        "status": "passed",
        "attempt_id": args.attempt_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "argv": list(argv if argv is not None else sys.argv),
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "gpu_used": False,
            "policy_loaded": False,
        },
        "run_git_commit": args.run_git_commit,
        "source_sha256": {
            "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "host_inspector": hashlib.sha256(
                (
                    REPOSITORY_ROOT
                    / "code/scripts/inspect_restoration_v2_executor_container.py"
                ).read_bytes()
            ).hexdigest(),
            "offline_validator": hashlib.sha256(
                (
                    REPOSITORY_ROOT
                    / "code/scripts/validate_restoration_v2_executor_evidence.py"
                ).read_bytes()
            ).hexdigest(),
        },
        "server_provenance": server_provenance,
        "interface_validation": interface_summary,
        "negative_control_constructor_validation": negative_constructor,
        "androidworld_executor_dispatch_validation": dispatch,
        "policy_output_generated": False,
        "restoration_label_generated": False,
    }
    summary["offline_reduction"] = validate_dispatch_evidence(summary, cases=cases)
    return summary


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as output_file:
        output_file.write(rendered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--action-fixture", required=True, type=Path)
    parser.add_argument("--prompt-fixture", required=True, type=Path)
    parser.add_argument("--interface-manifest", required=True, type=Path)
    parser.add_argument("--stack-config", required=True, type=Path)
    parser.add_argument("--host-inspection", required=True, type=Path)
    parser.add_argument("--androidworld-source-root", required=True, type=Path)
    parser.add_argument("--androidworld-source-revision", required=True)
    parser.add_argument("--run-git-commit", required=True)
    parser.add_argument("--host-name", required=True)
    parser.add_argument("--runtime-container-name", required=True)
    parser.add_argument("--runtime-container-id", required=True)
    parser.add_argument("--runtime-image-id", required=True)
    parser.add_argument("--runtime-image-repo-digest", required=True)
    parser.add_argument("--server-container-name", required=True)
    parser.add_argument("--server-container-id", required=True)
    parser.add_argument("--server-image", required=True)
    parser.add_argument("--server-image-id", required=True)
    parser.add_argument("--server-json-action-schema", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--server-host-port", required=True, type=int)
    parser.add_argument("--server-container-port", required=True, type=int)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    parser.add_argument("--output-summary", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_summary.exists():
        raise FileExistsError(
            f"refusing to overwrite executor attempt output: {args.output_summary}"
        )
    try:
        summary = run_validation(args)
    except Exception as error:
        if isinstance(error, ExecutorCompatibilityFailure):
            verdict = "FAILED_EXECUTOR_COMPATIBILITY"
        else:
            verdict = "INVALID"
        failure = {
            "schema_version": "1.0.0",
            "protocol_id": "causalcache_restoration_v2",
            "step": "androidworld_executor_dispatch_preflight",
            "status": "failed",
            "verdict": verdict,
            "attempt_id": args.attempt_id,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "policy_output_generated": False,
            "restoration_label_generated": False,
        }
        write_summary(args.output_summary, failure)
        raise
    write_summary(args.output_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
