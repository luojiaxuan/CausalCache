"""OSWorld task, action, policy, and resumable episode adapters."""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


OSWORLD_RUNNER_SCHEMA_VERSION = "causalcache.osworld.runner.v1"
OSWORLD_POLICY_REQUEST_SCHEMA_VERSION = "causalcache.osworld.policy_request.v1"
OSWORLD_POLICY_RESPONSE_SCHEMA_VERSION = "causalcache.osworld.policy_response.v1"
TERMINAL_ACTIONS = frozenset({"wait", "done", "fail"})
SUPPORTED_ACTIONS = frozenset(
    {
        "move",
        "click",
        "double_click",
        "right_click",
        "drag",
        "scroll",
        "type_text",
        "press",
        "hotkey",
        *TERMINAL_ACTIONS,
    }
)
SUPPORTED_MEMORY_ARMS = ("summary", "recent", "full")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_component(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{label} must be one safe path component")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], *, required: set[str], optional: set[str]
) -> None:
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise ValueError(
            f"action keys drifted: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _coordinate(value: Any, *, label: str, upper_bound: int) -> int:
    if type(value) not in (int, float):
        raise TypeError(f"{label} must be numeric")
    parsed = int(round(float(value)))
    if not 0 <= parsed < upper_bound:
        raise ValueError(f"{label}={parsed} falls outside [0, {upper_bound})")
    return parsed


@dataclass(frozen=True)
class DesktopAction:
    """Typed action that can be rendered into OSWorld's PyAutoGUI action space."""

    type: str
    x: int | None = None
    y: int | None = None
    button: str | None = None
    clicks: int | None = None
    duration: float | None = None
    dx: int | None = None
    dy: int | None = None
    text: str | None = None
    key: str | None = None
    keys: tuple[str, ...] | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        screen_size: tuple[int, int] = (1920, 1080),
    ) -> "DesktopAction":
        if not isinstance(value, Mapping):
            raise TypeError("desktop action must be a mapping")
        action_type = value.get("type")
        if action_type not in SUPPORTED_ACTIONS:
            raise ValueError(f"unsupported desktop action type: {action_type!r}")
        width, height = screen_size
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ValueError("screen_size must contain positive integers")

        if action_type in {"wait", "done", "fail"}:
            _require_exact_keys(value, required={"type"}, optional=set())
            return cls(type=action_type)

        if action_type in {"move", "drag"}:
            _require_exact_keys(
                value,
                required={"type", "x", "y"},
                optional={"duration"},
            )
            duration = float(value.get("duration", 0.2 if action_type == "move" else 1.0))
            if not 0.0 <= duration <= 10.0:
                raise ValueError("duration must be within [0, 10]")
            return cls(
                type=action_type,
                x=_coordinate(value["x"], label="x", upper_bound=width),
                y=_coordinate(value["y"], label="y", upper_bound=height),
                duration=duration,
            )

        if action_type in {"click", "double_click", "right_click"}:
            _require_exact_keys(
                value,
                required={"type", "x", "y"},
                optional={"button", "clicks", "duration"},
            )
            default_button = "right" if action_type == "right_click" else "left"
            button = value.get("button", default_button)
            if button not in {"left", "right", "middle"}:
                raise ValueError("button must be left, right, or middle")
            default_clicks = 2 if action_type == "double_click" else 1
            clicks = value.get("clicks", default_clicks)
            if type(clicks) is not int or not 1 <= clicks <= 3:
                raise ValueError("clicks must be an integer within [1, 3]")
            duration = float(value.get("duration", 0.1))
            if not 0.0 <= duration <= 10.0:
                raise ValueError("duration must be within [0, 10]")
            return cls(
                type=action_type,
                x=_coordinate(value["x"], label="x", upper_bound=width),
                y=_coordinate(value["y"], label="y", upper_bound=height),
                button=button,
                clicks=clicks,
                duration=duration,
            )

        if action_type == "scroll":
            _require_exact_keys(
                value,
                required={"type", "dy"},
                optional={"dx", "x", "y"},
            )
            dx = value.get("dx", 0)
            dy = value["dy"]
            if type(dx) is not int or type(dy) is not int:
                raise TypeError("scroll dx and dy must be integers")
            if dx == 0 and dy == 0:
                raise ValueError("scroll cannot have zero displacement")
            x = None if "x" not in value else _coordinate(value["x"], label="x", upper_bound=width)
            y = None if "y" not in value else _coordinate(value["y"], label="y", upper_bound=height)
            if (x is None) != (y is None):
                raise ValueError("scroll x and y must be provided together")
            return cls(type=action_type, x=x, y=y, dx=dx, dy=dy)

        if action_type == "type_text":
            _require_exact_keys(value, required={"type", "text"}, optional=set())
            text = value["text"]
            if not isinstance(text, str) or len(text) > 10000:
                raise ValueError("text must be a string of at most 10000 characters")
            return cls(type=action_type, text=text)

        if action_type == "press":
            _require_exact_keys(value, required={"type", "key"}, optional=set())
            key = value["key"]
            if not isinstance(key, str) or not key or len(key) > 32:
                raise ValueError("key must be a non-empty short string")
            return cls(type=action_type, key=key.casefold())

        _require_exact_keys(value, required={"type", "keys"}, optional=set())
        keys = value["keys"]
        if (
            isinstance(keys, (str, bytes, bytearray))
            or not isinstance(keys, Sequence)
            or not 2 <= len(keys) <= 5
            or any(not isinstance(key, str) or not key or len(key) > 32 for key in keys)
        ):
            raise ValueError("hotkey keys must contain two to five short strings")
        return cls(type=action_type, keys=tuple(key.casefold() for key in keys))

    def to_mapping(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    def to_osworld(self) -> str:
        if self.type in TERMINAL_ACTIONS:
            return self.type.upper()
        if self.type == "move":
            return f"pyautogui.moveTo({self.x}, {self.y}, duration={self.duration!r})"
        if self.type in {"click", "double_click", "right_click"}:
            return (
                f"pyautogui.click({self.x}, {self.y}, clicks={self.clicks}, "
                f"interval={self.duration!r}, button={self.button!r})"
            )
        if self.type == "drag":
            return (
                f"pyautogui.dragTo({self.x}, {self.y}, duration={self.duration!r}, "
                "button='left')"
            )
        if self.type == "scroll":
            prefix = ""
            if self.x is not None:
                prefix = f"pyautogui.moveTo({self.x}, {self.y}); "
            if self.dx:
                return f"{prefix}pyautogui.hscroll({self.dx}); pyautogui.scroll({self.dy})"
            return f"{prefix}pyautogui.scroll({self.dy})"
        if self.type == "type_text":
            return f"pyautogui.write({self.text!r}, interval=0.01)"
        if self.type == "press":
            return f"pyautogui.press({self.key!r})"
        if self.type == "hotkey":
            rendered = ", ".join(repr(key) for key in self.keys or ())
            return f"pyautogui.hotkey({rendered})"
        raise AssertionError(f"unhandled desktop action type: {self.type}")


@dataclass(frozen=True)
class OSWorldTask:
    domain: str
    task_id: str
    config_path: Path
    config: Mapping[str, Any]

    @property
    def instruction(self) -> str:
        value = self.config.get("instruction")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("OSWorld task instruction must be non-empty text")
        return value


def load_osworld_inventory(
    osworld_root: str | Path,
    *,
    meta_path: str | Path = "evaluation_examples/test_all.json",
) -> tuple[tuple[str, str], ...]:
    root = Path(osworld_root).expanduser().resolve()
    meta = Path(meta_path)
    if not meta.is_absolute():
        meta = root / meta
    payload = json.loads(meta.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("OSWorld task meta must be a domain mapping")
    records: list[tuple[str, str]] = []
    for domain in sorted(payload):
        _safe_component(domain, label="OSWorld domain")
        task_ids = payload[domain]
        if isinstance(task_ids, (str, bytes, bytearray)) or not isinstance(task_ids, Sequence):
            raise TypeError(f"OSWorld domain {domain} task ids must be a sequence")
        for task_id in task_ids:
            records.append((domain, _safe_component(task_id, label="OSWorld task id")))
    if not records or len(records) != len(set(records)):
        raise ValueError("OSWorld inventory must be non-empty and unique")
    return tuple(records)


def load_osworld_task(
    osworld_root: str | Path,
    *,
    domain: str,
    task_id: str,
    examples_dir: str | Path = "evaluation_examples/examples",
) -> OSWorldTask:
    root = Path(osworld_root).expanduser().resolve()
    domain = _safe_component(domain, label="OSWorld domain")
    task_id = _safe_component(task_id, label="OSWorld task id")
    base = Path(examples_dir)
    if not base.is_absolute():
        base = root / base
    path = (base / domain / f"{task_id}.json").resolve()
    if base.resolve() not in path.parents:
        raise ValueError("OSWorld task path escaped the examples directory")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("id") != task_id:
        raise ValueError("OSWorld task config id does not match inventory id")
    task = OSWorldTask(domain=domain, task_id=task_id, config_path=path, config=config)
    task.instruction
    return task


def osworld_git_revision(osworld_root: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(osworld_root).expanduser().resolve(),
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if len(revision) != 40:
        raise ValueError("OSWorld checkout does not expose a full Git revision")
    return revision


def import_osworld_desktop_env(osworld_root: str | Path) -> Any:
    root = Path(osworld_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("desktop_env.desktop_env")
    finally:
        if sys.path[0] == str(root):
            sys.path.pop(0)
    origin = Path(module.__file__).resolve()
    if root not in origin.parents:
        raise RuntimeError(f"DesktopEnv imported from unexpected path: {origin}")
    return module.DesktopEnv


class OSWorldEnvironment(Protocol):
    def reset(self, *, task_config: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def step(
        self, action: str, pause: float = 2
    ) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]: ...

    def evaluate(self) -> float: ...

    def close(self) -> None: ...


def create_osworld_environment(
    osworld_root: str | Path,
    *,
    provider_name: str,
    path_to_vm: str | None,
    cache_dir: str | Path,
    screen_size: tuple[int, int],
    headless: bool,
    region: str | None = None,
) -> OSWorldEnvironment:
    desktop_env = import_osworld_desktop_env(osworld_root)
    return desktop_env(
        provider_name=provider_name,
        region=region,
        path_to_vm=path_to_vm,
        action_space="pyautogui",
        cache_dir=str(Path(cache_dir).expanduser().resolve()),
        screen_size=screen_size,
        headless=headless,
        require_a11y_tree=False,
        require_terminal=False,
        os_type="Ubuntu",
    )


@dataclass(frozen=True)
class OSWorldHistoryEvent:
    step_id: int
    action: Mapping[str, Any]
    osworld_action: str
    result_status: str
    screen_changed: bool
    post_screenshot: bytes

    def summary(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": dict(self.action),
            "result_status": self.result_status,
            "screen_changed": self.screen_changed,
            "post_screenshot_sha256": _sha256_bytes(self.post_screenshot),
        }


def select_osworld_memory(
    history: Sequence[OSWorldHistoryEvent], *, arm: str, budget: int
) -> tuple[int, ...]:
    if arm not in SUPPORTED_MEMORY_ARMS:
        raise ValueError(f"unsupported OSWorld memory arm: {arm}")
    if type(budget) is not int or budget < 0:
        raise ValueError("memory budget must be a non-negative integer")
    event_ids = tuple(event.step_id for event in history)
    if arm == "summary" or budget == 0:
        return ()
    if arm == "full":
        return event_ids
    if arm == "recent":
        return event_ids[-min(budget, len(event_ids)) :]
    raise AssertionError(f"unhandled OSWorld memory arm: {arm}")


def build_osworld_policy_request(
    *,
    task: OSWorldTask,
    step_id: int,
    screenshot: bytes,
    history: Sequence[OSWorldHistoryEvent],
    selected_event_step_ids: Sequence[int],
    screen_size: tuple[int, int],
) -> dict[str, Any]:
    selected = set(selected_event_step_ids)
    known = {event.step_id for event in history}
    if not selected.issubset(known):
        raise ValueError("selected OSWorld memory refers to an unknown event")
    return {
        "schema_version": OSWORLD_POLICY_REQUEST_SCHEMA_VERSION,
        "task": {
            "domain": task.domain,
            "task_id": task.task_id,
            "instruction": task.instruction,
        },
        "step_id": step_id,
        "screen_size": list(screen_size),
        "current_screenshot_png_base64": base64.b64encode(screenshot).decode("ascii"),
        "history": [
            {
                **event.summary(),
                "restored_post_screenshot_png_base64": (
                    base64.b64encode(event.post_screenshot).decode("ascii")
                    if event.step_id in selected
                    else None
                ),
            }
            for event in history
        ],
        "selected_event_step_ids": list(selected_event_step_ids),
        "action_schema": {
            "supported_types": sorted(SUPPORTED_ACTIONS),
            "terminal_types": sorted(TERMINAL_ACTIONS),
            "contract": "Return exactly one action mapping in response.action.",
        },
    }


@dataclass(frozen=True)
class PolicyDecision:
    action: DesktopAction
    raw_response: Mapping[str, Any]
    latency_seconds: float


class OSWorldPolicy(Protocol):
    def reset(self, task: OSWorldTask) -> None: ...

    def act(self, request: Mapping[str, Any]) -> PolicyDecision: ...


class HTTPOSWorldPolicy:
    def __init__(self, endpoint: str, *, timeout_seconds: float = 300.0) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("policy endpoint must use http:// or https://")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def reset(self, task: OSWorldTask) -> None:
        return None

    def act(self, request: Mapping[str, Any]) -> PolicyDecision:
        payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
        latency = time.perf_counter() - started
        if not isinstance(raw, Mapping):
            raise TypeError("policy response must be a mapping")
        if raw.get("schema_version") != OSWORLD_POLICY_RESPONSE_SCHEMA_VERSION:
            raise ValueError("policy response schema version drifted")
        screen_size = tuple(request["screen_size"])
        action = DesktopAction.from_mapping(raw.get("action"), screen_size=screen_size)
        return PolicyDecision(action=action, raw_response=raw, latency_seconds=latency)


class ScriptedOSWorldPolicy:
    def __init__(self, actions_by_task: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self.actions_by_task = {
            task_id: tuple(actions) for task_id, actions in actions_by_task.items()
        }
        self.task_id: str | None = None
        self.index = 0

    @classmethod
    def from_json(cls, path: str | Path) -> "ScriptedOSWorldPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            payload = {"*": payload}
        if not isinstance(payload, Mapping):
            raise TypeError("scripted policy JSON must be an action list or task mapping")
        return cls(payload)

    def reset(self, task: OSWorldTask) -> None:
        self.task_id = task.task_id
        self.index = 0

    def act(self, request: Mapping[str, Any]) -> PolicyDecision:
        actions = self.actions_by_task.get(self.task_id or "")
        if actions is None:
            actions = self.actions_by_task.get("*")
        if actions is None or self.index >= len(actions):
            raise RuntimeError(f"scripted policy exhausted for task {self.task_id}")
        raw_action = actions[self.index]
        self.index += 1
        screen_size = tuple(request["screen_size"])
        action = DesktopAction.from_mapping(raw_action, screen_size=screen_size)
        response = {
            "schema_version": OSWORLD_POLICY_RESPONSE_SCHEMA_VERSION,
            "action": action.to_mapping(),
            "source": "scripted",
        }
        return PolicyDecision(action=action, raw_response=response, latency_seconds=0.0)


def _validated_observation(value: Mapping[str, Any]) -> bytes:
    if not isinstance(value, Mapping):
        raise TypeError("OSWorld observation must be a mapping")
    screenshot = value.get("screenshot")
    if not isinstance(screenshot, bytes) or not screenshot:
        raise ValueError("OSWorld observation screenshot must be non-empty bytes")
    return screenshot


def run_osworld_episode(
    *,
    environment: OSWorldEnvironment,
    policy: OSWorldPolicy,
    task: OSWorldTask,
    output_root: str | Path,
    memory_arm: str,
    memory_budget: int,
    max_steps: int,
    pause_seconds: float,
    screen_size: tuple[int, int],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    task_root = Path(output_root).expanduser().resolve() / task.domain / task.task_id
    completion_path = task_root / "result.json"
    if completion_path.exists():
        result = json.loads(completion_path.read_text(encoding="utf-8"))
        if result.get("status") != "COMPLETE_OSWORLD_EPISODE":
            raise ValueError("existing OSWorld result has an invalid completion status")
        return {**result, "resumed_skip": True}

    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt_root = task_root / "attempts" / attempt_id
    attempt_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    history: list[OSWorldHistoryEvent] = []
    steps: list[dict[str, Any]] = []
    policy.reset(task)
    try:
        observation = environment.reset(task_config=task.config)
        screenshot = _validated_observation(observation)
        (attempt_root / "initial.png").write_bytes(screenshot)
        termination_reason = "step_budget_exhausted"
        for step_id in range(1, max_steps + 1):
            selected = select_osworld_memory(
                history, arm=memory_arm, budget=memory_budget
            )
            request = build_osworld_policy_request(
                task=task,
                step_id=step_id,
                screenshot=screenshot,
                history=history,
                selected_event_step_ids=selected,
                screen_size=screen_size,
            )
            decision = policy.act(request)
            osworld_action = decision.action.to_osworld()
            next_observation, reward, done, info = environment.step(
                osworld_action, pause=pause_seconds
            )
            next_screenshot = _validated_observation(next_observation)
            screenshot_name = f"step-{step_id:03d}.png"
            (attempt_root / screenshot_name).write_bytes(next_screenshot)
            result_status = "done" if done else "executed"
            event = OSWorldHistoryEvent(
                step_id=step_id,
                action=decision.action.to_mapping(),
                osworld_action=osworld_action,
                result_status=result_status,
                screen_changed=_sha256_bytes(screenshot) != _sha256_bytes(next_screenshot),
                post_screenshot=next_screenshot,
            )
            history.append(event)
            steps.append(
                {
                    **event.summary(),
                    "osworld_action": osworld_action,
                    "selected_event_step_ids": list(selected),
                    "policy_latency_seconds": decision.latency_seconds,
                    "policy_response": dict(decision.raw_response),
                    "reward": float(reward),
                    "done": bool(done),
                    "info": dict(info),
                    "screenshot_file": f"attempts/{attempt_id}/{screenshot_name}",
                }
            )
            _atomic_json(
                attempt_root / "checkpoint.json",
                {
                    "schema_version": OSWORLD_RUNNER_SCHEMA_VERSION,
                    "status": "RUNNING_OSWORLD_EPISODE",
                    "task_id": task.task_id,
                    "completed_steps": len(steps),
                    "steps": steps,
                    "updated_at": _utc_now(),
                },
            )
            screenshot = next_screenshot
            if done or decision.action.type in {"done", "fail"}:
                termination_reason = (
                    f"policy_{decision.action.type}"
                    if decision.action.type in {"done", "fail"}
                    else "environment_done"
                )
                break
        score = float(environment.evaluate())
        result = {
            "schema_version": OSWORLD_RUNNER_SCHEMA_VERSION,
            "status": "COMPLETE_OSWORLD_EPISODE",
            "task": {
                "domain": task.domain,
                "task_id": task.task_id,
                "config_path": str(task.config_path),
                "instruction": task.instruction,
            },
            "memory": {
                "arm": memory_arm,
                "budget": memory_budget,
                "budget_semantics": "at_most_B_high_fidelity_post_screenshots",
            },
            "max_steps": max_steps,
            "completed_steps": len(steps),
            "termination_reason": termination_reason,
            "score": score,
            "success": score > 0.0,
            "steps": steps,
            "attempt_id": attempt_id,
            "elapsed_seconds": time.perf_counter() - started,
            "completed_at": _utc_now(),
            "provenance": dict(provenance),
            "resumed_skip": False,
        }
        _atomic_json(completion_path, result)
        return result
    except BaseException as error:
        _atomic_json(
            attempt_root / "failure.json",
            {
                "schema_version": OSWORLD_RUNNER_SCHEMA_VERSION,
                "status": "FAILED_OSWORLD_EPISODE",
                "task": {"domain": task.domain, "task_id": task.task_id},
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "completed_steps": len(steps),
                "failed_at": _utc_now(),
                "provenance": dict(provenance),
            },
        )
        raise
