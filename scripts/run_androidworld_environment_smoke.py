"""Exercise the pinned AndroidWorld HTTP environment without a policy model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_url(base_url: str, path: str, params: dict[str, Any] | None = None) -> str:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    return url


def request_raw(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    timeout_seconds: float = 120.0,
) -> tuple[bytes, str]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(
        build_url(base_url, path, params),
        data=data if data is not None else (b"" if method == "POST" else None),
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read(), response.headers.get_content_type()


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    payload, content_type = request_raw(
        base_url,
        path,
        method=method,
        params=params,
        body=body,
        timeout_seconds=timeout_seconds,
    )
    if content_type != "application/json":
        raise ValueError(f"expected application/json, received {content_type}")
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("expected a JSON object")
    return decoded


def summarize_screenshot(payload: bytes, content_type: str) -> dict[str, Any]:
    if content_type != "application/json":
        raise ValueError(f"unexpected screenshot content type: {content_type}")
    if not payload.startswith(b'{"pixels":[') or not payload.endswith(b"]}"):
        raise ValueError("screenshot endpoint did not return a pixels array")
    return {
        "content_type": content_type,
        "response_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    summary: dict[str, Any] = {
        "schema_version": "0.1.0",
        "started_at": started_at,
        "base_url": args.base_url,
        "suite_seed": args.suite_seed,
        "suite_task_combinations": 1,
        "task_type": args.task_type,
        "task_index": args.task_index,
        "executed_action": {"action_type": "navigate_home"},
    }

    health = request_json(args.base_url, "/health")
    if health.get("status") != "success":
        raise ValueError(f"health check failed: {health}")
    summary["health"] = health

    summary["suite_reinitialize"] = request_json(
        args.base_url,
        "/suite/reinitialize",
        params={
            "n_task_combinations": 1,
            "seed": args.suite_seed,
            "task_family": "android_world",
        },
    )
    task_list = request_json(
        args.base_url, "/suite/task_list", params={"max_index": -1}
    )["task_list"]
    if args.task_type not in task_list:
        raise ValueError(f"task type is absent from registry: {args.task_type}")
    summary["registry_task_types"] = len(task_list)

    task_params = {"task_type": args.task_type, "task_idx": args.task_index}
    initialized = False
    try:
        summary["initialize"] = request_json(
            args.base_url, "/task/initialize", method="POST", params=task_params
        )
        initialized = True
        summary["goal"] = request_json(
            args.base_url, "/task/goal", params=task_params
        )["goal"]
        summary["template"] = request_json(
            args.base_url, "/task/template", params=task_params
        )["template"]
        summary["score_before"] = request_json(
            args.base_url, "/task/score", params=task_params
        )["score"]

        before_payload, before_type = request_raw(
            args.base_url,
            "/screenshot",
            params={"wait_to_stabilize": "true"},
        )
        summary["screenshot_before"] = summarize_screenshot(
            before_payload, before_type
        )
        summary["action_response"] = request_json(
            args.base_url,
            "/execute_action",
            method="POST",
            body=summary["executed_action"],
        )
        after_payload, after_type = request_raw(
            args.base_url,
            "/screenshot",
            params={"wait_to_stabilize": "true"},
        )
        summary["screenshot_after"] = summarize_screenshot(after_payload, after_type)
        summary["screenshot_changed"] = (
            summary["screenshot_before"]["sha256"]
            != summary["screenshot_after"]["sha256"]
        )
        summary["score_after"] = request_json(
            args.base_url, "/task/score", params=task_params
        )["score"]
    finally:
        if initialized:
            summary["tear_down"] = request_json(
                args.base_url,
                "/task/tear_down",
                method="POST",
                params=task_params,
            )

    summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--task-type", default="SystemWifiTurnOn")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--suite-seed", type=int, default=271828)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_smoke(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
