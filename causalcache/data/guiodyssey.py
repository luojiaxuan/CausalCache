"""Deterministic GUIOdyssey conversion for the first mixed-fidelity pilot."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalcache.schema import ActionType, ExecutableAction, ResultStatus


def _coordinate_bin(coordinate: Sequence[int], grid_size: int) -> str:
    if len(coordinate) != 2:
        raise ValueError("coordinates must contain x and y")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    x, y = (int(value) for value in coordinate)
    if not 0 <= x <= 1000 or not 0 <= y <= 1000:
        raise ValueError("GUIOdyssey coordinates must be normalized to [0, 1000]")
    x_bin = min(grid_size - 1, x * grid_size // 1001)
    y_bin = min(grid_size - 1, y * grid_size // 1001)
    return f"x{x_bin}_y{y_bin}"


def _scroll_direction(start_coordinate: Sequence[int], end_coordinate: Sequence[int]) -> str:
    if len(start_coordinate) != 2 or len(end_coordinate) != 2:
        raise ValueError("swipe coordinates must contain x and y")
    start_x, start_y = (int(value) for value in start_coordinate)
    end_x, end_y = (int(value) for value in end_coordinate)
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    if delta_x == 0 and delta_y == 0:
        raise ValueError("swipe must move")
    if abs(delta_y) >= abs(delta_x):
        return "scroll:down" if delta_y < 0 else "scroll:up"
    return "scroll:right" if delta_x < 0 else "scroll:left"


def canonicalize_tool_call(
    tool_call: Mapping[str, Any],
    *,
    grid_size: int = 10,
) -> tuple[ExecutableAction, str]:
    function = tool_call.get("function")
    if not isinstance(function, Mapping):
        raise ValueError("tool call must contain a function object")
    name = str(function["name"])
    arguments = function.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise ValueError("tool call arguments must be an object")

    if name == "tap":
        target = f"coordinate_bin:{_coordinate_bin(arguments['coordinate'], grid_size)}"
        return ExecutableAction(ActionType.TAP, target=target), target
    if name == "long_press":
        target = f"coordinate_bin:{_coordinate_bin(arguments['coordinate'], grid_size)}"
        return ExecutableAction(ActionType.LONG_PRESS, target=target), target
    if name == "type":
        text = str(arguments["text"])
        return ExecutableAction(ActionType.TYPE_TEXT, target="text_field_unknown", text_argument=text), text
    if name == "swipe":
        target = _scroll_direction(arguments["start_coordinate"], arguments["coordinate"])
        return ExecutableAction(ActionType.SWIPE, target=target), target
    if name == "system_button":
        button = str(arguments["button"]).casefold()
        mapping = {
            "home": ActionType.HOME,
            "back": ActionType.BACK,
            "enter": ActionType.ENTER,
        }
        if button not in mapping:
            raise ValueError(f"unsupported system button: {button}")
        return ExecutableAction(mapping[button], target=button), button
    if name == "wait":
        return ExecutableAction(ActionType.WAIT), "wait"
    if name == "stop":
        return ExecutableAction(ActionType.STOP), "stop"
    raise ValueError(f"unsupported GUIOdyssey tool: {name}")


def _image_extension(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    raise ValueError("unsupported embedded image format")


def _content_text(turn: Mapping[str, Any], content_type: str) -> str | None:
    for item in turn.get("content", []):
        if item.get("type") == content_type:
            return str(item["text"])
    return None


def _action_dict(action: ExecutableAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type.value,
        "target": action.target,
        "text_argument": action.text_argument,
        "text_case_sensitive": action.text_case_sensitive,
    }


def _split_tool_calls(
    tool_calls: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], dict[str, Any] | None]:
    executable_calls = []
    terminal_signals = []
    for tool_call in tool_calls:
        function = tool_call.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("tool call must contain a function object")
        if function.get("name") == "terminate":
            arguments = function.get("arguments", {})
            if not isinstance(arguments, Mapping):
                raise ValueError("terminate arguments must be an object")
            terminal_signals.append(dict(arguments))
        else:
            executable_calls.append(tool_call)
    if len(executable_calls) != 1:
        raise ValueError("pilot requires exactly one executable action per observation")
    if len(terminal_signals) > 1:
        raise ValueError("pilot allows at most one terminal signal per observation")
    return executable_calls[0], terminal_signals[0] if terminal_signals else None


def build_pilot_manifest(
    row: Mapping[str, Any],
    *,
    row_index: int,
    upstream_repo: str,
    upstream_revision: str,
    transport_repo: str,
    transport_revision: str,
    transport_file: str,
    transport_file_sha256: str,
    hf_destination: str,
    grid_size: int = 10,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    messages = json.loads(str(row["messages"]))
    metadata = json.loads(str(row["metadata"]))
    images = row["images"]
    user_turns = [turn for turn in messages if turn["role"] == "user"]
    assistant_turns = [turn for turn in messages if turn["role"] == "assistant"]
    if len(images) != len(user_turns) or len(images) != len(assistant_turns):
        raise ValueError("pilot requires one observation and one action per step")

    instruction = _content_text(user_turns[0], "text")
    if instruction is None:
        raise ValueError("first user turn must contain the task instruction")
    source_id = str(metadata["others"]["source_id"])
    image_payloads: dict[str, bytes] = {}
    image_paths: list[str] = []
    for index, image in enumerate(images):
        image_bytes = image.get("bytes")
        if not isinstance(image_bytes, bytes):
            raise ValueError("pilot expects embedded image bytes")
        extension = _image_extension(image_bytes)
        relative_path = f"images/{source_id}/observation-{index:03d}{extension}"
        image_paths.append(relative_path)
        image_payloads[relative_path] = image_bytes

    actions: list[ExecutableAction] = []
    target_summaries: list[str] = []
    raw_tools: list[Mapping[str, Any]] = []
    terminal_signals: list[dict[str, Any] | None] = []
    action_descriptions: list[str | None] = []
    for turn in assistant_turns:
        tool_calls = turn.get("tool_calls", [])
        executable_call, terminal_signal = _split_tool_calls(tool_calls)
        action, target_summary = canonicalize_tool_call(executable_call, grid_size=grid_size)
        actions.append(action)
        target_summaries.append(target_summary)
        raw_tools.append(executable_call)
        terminal_signals.append(terminal_signal)
        action_descriptions.append(_content_text(turn, "action_description"))

    episode_terminal_signals = [signal for signal in terminal_signals if signal is not None]
    if len(episode_terminal_signals) != 1:
        raise ValueError("pilot requires exactly one episode terminal signal")
    episode_status = str(episode_terminal_signals[0].get("status", "unknown"))
    if episode_status not in {ResultStatus.SUCCESS.value, ResultStatus.FAILURE.value}:
        raise ValueError(f"unsupported episode terminal status: {episode_status}")
    if episode_status != ResultStatus.SUCCESS.value:
        raise ValueError("validated pilot requires a successful recorded trajectory")

    steps = []
    for index, action in enumerate(actions):
        steps.append(
            {
                "observation_index": index,
                "observation_path": image_paths[index],
                "action": _action_dict(action),
                "source_tool_call": raw_tools[index],
                "terminal_signal": terminal_signals[index],
                "action_description": action_descriptions[index],
            }
        )

    events = []
    for index in range(len(actions) - 1):
        action = actions[index]
        events.append(
            {
                "step_id": index + 1,
                "observation_before_path": image_paths[index],
                "observation_after_path": image_paths[index + 1],
                "executed_action": _action_dict(action),
                "low_fidelity": {
                    "step_id": index + 1,
                    "action_type": action.action_type.value,
                    "target_text_or_coordinate_bin": target_summaries[index],
                    "deterministic_ui_delta": "not_available",
                    "result_status": ResultStatus.UNKNOWN.value,
                },
            }
        )

    decisions = []
    for index in range(1, len(actions)):
        decisions.append(
            {
                "decision_step_id": index + 1,
                "current_observation_path": image_paths[index],
                "history_event_step_ids": list(range(1, index + 1)),
                "validated_action": _action_dict(actions[index]),
                "validation_source": "recorded_successful_guiodyssey_trajectory",
            }
        )

    manifest = {
        "schema_version": "0.2.0",
        "dataset_repo": hf_destination,
        "source": {
            "upstream_repo": upstream_repo,
            "upstream_revision": upstream_revision,
            "transport_repo": transport_repo,
            "transport_revision": transport_revision,
            "transport_file": transport_file,
            "transport_file_sha256": transport_file_sha256,
            "transport_row_index": row_index,
            "license": "cc-by-4.0",
        },
        "trajectory": {
            "source_id": source_id,
            "instruction": instruction,
            "platform": metadata["platform"],
            "apps": metadata["others"]["apps"],
            "device_name": metadata["others"]["device_name"],
            "resolution": metadata["others"]["resolution"],
            "terminal_status": episode_status,
            "steps": steps,
            "events": events,
            "decisions": decisions,
        },
    }
    return manifest, image_payloads


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    return info


def write_pilot_dataset(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
    image_payloads: Mapping[str, bytes],
) -> None:
    output_path = Path(output_dir)
    data_path = output_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (output_path / "manifest.json").write_bytes(manifest_bytes)

    tar_path = data_path / "guiodyssey-pilot-00000.tar"
    with tarfile.open(tar_path, "w") as archive:
        archive.addfile(_tar_info("manifest.json", len(manifest_bytes)), io.BytesIO(manifest_bytes))
        for name in sorted(image_payloads):
            payload = image_payloads[name]
            archive.addfile(_tar_info(name, len(payload)), io.BytesIO(payload))
