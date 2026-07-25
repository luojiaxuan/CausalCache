"""OSWorld 2.0 release, memory split, and shared-policy integration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.osworld import (
    HTTPOSWorldPolicy,
    OSWorldHistoryEvent,
    OSWorldTask,
    build_osworld_policy_request,
    select_osworld_memory,
)


OSWORLD_V2_RELEASE = "osworld-v2-2026.06.24"
OSWORLD_V2_TAG = "v2026.06.24"
OSWORLD_V2_CODE_REVISION = "2b9b7b4eb73243d557bdbf2998fe18d8e18e19c6"
OSWORLD_V2_EXPECTED_TASKS = 108
OSWORLD_V2_MEMORY_PLAN_SCHEMA_VERSION = "causalcache.osworld_v2.memory_split.v1"


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_osworld_v2_checkout(
    root: str | Path,
    *,
    require_task_classes: bool,
    require_assets: bool,
    assets_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed on mixed release components or missing gated substrate."""
    checkout = Path(root).expanduser().resolve()
    revision = _git_revision(checkout)
    if revision != OSWORLD_V2_CODE_REVISION:
        raise ValueError(
            f"OSWorld 2.0 code revision drifted: expected "
            f"{OSWORLD_V2_CODE_REVISION}, got {revision}"
        )
    release_path = (
        checkout
        / "benchmark_releases"
        / f"{OSWORLD_V2_RELEASE}.json"
    )
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if (
        release.get("release") != OSWORLD_V2_RELEASE
        or release.get("osworld_code", {}).get("tag") != OSWORLD_V2_TAG
        or release.get("task_hash_manifest", {}).get("task_count")
        != OSWORLD_V2_EXPECTED_TASKS
    ):
        raise ValueError("OSWorld 2.0 release manifest drifted")
    task_class_root = checkout / "evaluation_examples/task_class"
    task_classes = sorted(task_class_root.glob("task_[0-9][0-9][0-9].py"))
    resolved_assets_root = Path(
        assets_root
        if assets_root is not None
        else checkout / "cache/osworld_v2_assets"
    ).expanduser().resolve()
    if require_task_classes and len(task_classes) != OSWORLD_V2_EXPECTED_TASKS:
        raise RuntimeError(
            "OSWorld 2.0 gated task classes are incomplete; accept and download "
            "xlangai/osworld_v2_tasks@v2026.06.24"
        )
    if require_assets and not resolved_assets_root.is_dir():
        raise RuntimeError(
            "OSWorld 2.0 gated assets are missing; download "
            "xlangai/osworld_v2_assets_gated@v2026.06.24"
        )
    return {
        "release": OSWORLD_V2_RELEASE,
        "code_revision": revision,
        "release_manifest": str(release_path),
        "release_manifest_sha256": hashlib.sha256(
            release_path.read_bytes()
        ).hexdigest(),
        "task_class_count": len(task_classes),
        "task_classes_ready": len(task_classes) == OSWORLD_V2_EXPECTED_TASKS,
        "assets_root": str(resolved_assets_root),
        "assets_ready": resolved_assets_root.is_dir(),
    }


def load_osworld_v2_memory_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != OSWORLD_V2_MEMORY_PLAN_SCHEMA_VERSION:
        raise ValueError("OSWorld 2.0 memory plan schema drifted")
    upstream = plan.get("upstream")
    if (
        not isinstance(upstream, Mapping)
        or upstream.get("release") != OSWORLD_V2_RELEASE
        or upstream.get("code_revision") != OSWORLD_V2_CODE_REVISION
    ):
        raise ValueError("OSWorld 2.0 memory plan release identity drifted")
    records = plan.get("records")
    if not isinstance(records, list) or len(records) != OSWORLD_V2_EXPECTED_TASKS:
        raise ValueError("OSWorld 2.0 memory plan task count drifted")
    task_ids = [record.get("task_id") for record in records]
    if task_ids != [f"{index:03d}" for index in range(1, 109)]:
        raise ValueError("OSWorld 2.0 memory plan task ids drifted")
    return plan


class GUIOwlOSWorldV2Agent:
    """OSWorld 2.0 agent interface backed by the shared frozen policy server."""

    action_space = "pyautogui"
    observation_type = "screenshot"

    def __init__(
        self,
        *,
        policy_endpoint: str,
        memory_arm: str = "recent",
        memory_budget: int = 4,
        screen_size: tuple[int, int] = (1920, 1080),
        timeout_seconds: float = 300.0,
    ) -> None:
        endpoint = policy_endpoint.rstrip("/")
        if not endpoint.endswith("/act"):
            endpoint += "/act"
        self.policy = HTTPOSWorldPolicy(
            endpoint, timeout_seconds=timeout_seconds
        )
        self.memory_arm = memory_arm
        self.memory_budget = memory_budget
        self.screen_size = screen_size
        self.reset()

    def reset(self, _logger: Any = None) -> None:
        del _logger
        self.history: list[OSWorldHistoryEvent] = []
        self.previous_action: dict[str, Any] | None = None
        self.previous_osworld_action: str | None = None
        self.previous_screenshot: bytes | None = None

    def predict(
        self,
        instruction: str,
        obs: Mapping[str, Any],
    ) -> tuple[str, list[str] | None]:
        screenshot = obs.get("screenshot")
        if not isinstance(screenshot, bytes) or not screenshot:
            raise ValueError("OSWorld 2.0 GUI-Owl requires screenshot bytes")
        if self.previous_action is not None:
            self.history.append(
                OSWorldHistoryEvent(
                    step_id=len(self.history) + 1,
                    action=self.previous_action,
                    osworld_action=str(self.previous_osworld_action),
                    result_status="executed",
                    screen_changed=screenshot != self.previous_screenshot,
                    post_screenshot=screenshot,
                )
            )
        task = OSWorldTask(
            domain="tasks",
            task_id=hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16],
            config_path=Path("gated-task-class"),
            config={"instruction": instruction},
        )
        selected = select_osworld_memory(
            self.history,
            arm=self.memory_arm,
            budget=self.memory_budget,
        )
        request = build_osworld_policy_request(
            task=task,
            step_id=len(self.history) + 1,
            screenshot=screenshot,
            history=self.history,
            selected_event_step_ids=selected,
            screen_size=self.screen_size,
        )
        decision = self.policy.act(request)
        self.previous_action = decision.action.to_mapping()
        self.previous_osworld_action = decision.action.to_osworld()
        self.previous_screenshot = screenshot
        raw_output = str(
            decision.raw_response.get("native_output", decision.raw_response)
        )
        return raw_output, [self.previous_osworld_action]
