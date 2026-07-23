"""Manifest and metric helpers for the OSWorld mobile-LoRA leakage probe."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.osworld_gui_owl import parse_gui_owl_osworld_action


CONFIG_SCHEMA_VERSION = "causalcache.osworld_lora_leakage_config.v1"
MANIFEST_SCHEMA_VERSION = "causalcache.osworld_lora_leakage_manifest.v1"
PROFILE_SCHEMA_VERSION = "causalcache.osworld_lora_leakage_profile.v1"
SUMMARY_SCHEMA_VERSION = "causalcache.osworld_lora_leakage_summary.v1"
HARD_MOBILE_ACTIONS = frozenset({"swipe", "long_press", "system_button", "open"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("OSWorld LoRA leakage config schema drifted")
    if config["prompt_count"] < 30:
        raise ValueError("leakage probe requires at least 30 prompts")
    if config["memory_budget"] != 4:
        raise ValueError("leakage probe must retain recent at-most-B4")
    profile_ids = [profile["profile_id"] for profile in config["profiles"]]
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("leakage profile ids must be unique")
    return config


def load_manifest(path: Path, *, expected_count: int) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("OSWorld LoRA leakage manifest schema drifted")
    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != expected_count:
        raise ValueError("OSWorld LoRA leakage prompt count drifted")
    ids = [prompt["prompt_id"] for prompt in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("OSWorld LoRA leakage prompt ids must be unique")
    return manifest


def request_from_prompt(prompt: Mapping[str, Any]) -> dict[str, Any]:
    current_path = Path(prompt["current_screenshot_path"])
    current = base64.b64encode(current_path.read_bytes()).decode("ascii")
    history = []
    selected = []
    for event in prompt["history"]:
        step_id = int(event["step_id"])
        restored = base64.b64encode(
            Path(event["restored_screenshot_path"]).read_bytes()
        ).decode("ascii")
        history.append(
            {
                "step_id": step_id,
                "action": event["action"],
                "result_status": event["result_status"],
                "screen_changed": event["screen_changed"],
                "restored_post_screenshot_png_base64": restored,
            }
        )
        selected.append(step_id)
    return {
        "task": {"instruction": prompt["instruction"]},
        "screen_size": prompt["screen_size"],
        "current_screenshot_png_base64": current,
        "selected_event_step_ids": selected,
        "history": history,
    }


def inspect_output(output_text: str, *, screen_size: tuple[int, int]) -> dict[str, Any]:
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", output_text, re.DOTALL)
    tool_name = None
    raw_action = None
    payload_error = None
    if len(matches) == 1:
        try:
            payload = json.loads(matches[0])
            if isinstance(payload, Mapping):
                tool_name = payload.get("name")
                arguments = payload.get("arguments")
                if isinstance(arguments, Mapping):
                    raw_action = arguments.get("action")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            payload_error = f"{error.__class__.__name__}: {error}"
    normalized_action = None
    parser_error = None
    try:
        normalized_action = parse_gui_owl_osworld_action(
            output_text, screen_size=screen_size
        ).type
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        parser_error = f"{error.__class__.__name__}: {error}"
    lexical_mobile = bool(
        re.search(r'"name"\s*:\s*"mobile_use"', output_text)
        or re.search(
            r'"action"\s*:\s*"(?:swipe|long_press|system_button|open)"',
            output_text,
        )
    )
    hard_leakage = (
        tool_name == "mobile_use"
        or raw_action in HARD_MOBILE_ACTIONS
        or lexical_mobile
    )
    return {
        "parser_valid": parser_error is None,
        "parser_error": parser_error,
        "payload_error": payload_error,
        "tool_name": tool_name,
        "raw_action": raw_action,
        "normalized_action": normalized_action,
        "click_alias": raw_action == "click",
        "left_click": raw_action == "left_click",
        "hard_leakage": hard_leakage,
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty leakage profile")
    valid = sum(bool(record["inspection"]["parser_valid"]) for record in records)
    hard = sum(bool(record["inspection"]["hard_leakage"]) for record in records)
    raw_counts = Counter(
        str(record["inspection"]["raw_action"] or "<missing>") for record in records
    )
    normalized_counts = Counter(
        str(record["inspection"]["normalized_action"] or "<invalid>")
        for record in records
    )
    click_total = raw_counts["click"] + raw_counts["left_click"]
    return {
        "prompt_count": len(records),
        "parser_valid_count": valid,
        "parser_valid_rate": valid / len(records),
        "hard_leakage_count": hard,
        "hard_leakage_rate": hard / len(records),
        "click_alias_count": raw_counts["click"],
        "left_click_count": raw_counts["left_click"],
        "click_alias_rate_among_clicks": (
            raw_counts["click"] / click_total if click_total else None
        ),
        "raw_action_counts": dict(sorted(raw_counts.items())),
        "normalized_action_counts": dict(sorted(normalized_counts.items())),
    }


def jensen_shannon_counts(
    left: Mapping[str, int], right: Mapping[str, int]
) -> float:
    keys = sorted(set(left) | set(right))
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total <= 0 or right_total <= 0:
        raise ValueError("action count distributions must be non-empty")
    p = [left.get(key, 0) / left_total for key in keys]
    q = [right.get(key, 0) / right_total for key in keys]
    midpoint = [(a + b) / 2 for a, b in zip(p, q, strict=True)]

    def divergence(values: list[float]) -> float:
        return sum(
            value * math.log(value / center)
            for value, center in zip(values, midpoint, strict=True)
            if value > 0
        )

    return (divergence(p) + divergence(q)) / 2


def classify_profile(
    *, frozen: Mapping[str, Any], adapted: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    drop_points = 100 * (
        float(frozen["parser_valid_rate"]) - float(adapted["parser_valid_rate"])
    )
    hard = int(adapted["hard_leakage_count"])
    if hard > 0 or drop_points > float(thresholds["hard_fail_drop_points"]):
        verdict = "FAIL_HARD_LEAKAGE"
    elif drop_points >= float(thresholds["pass_drop_points"]):
        verdict = "MILD_REQUIRES_ALPHA16"
    else:
        verdict = "PASS_NO_MATERIAL_LEAKAGE"
    return {
        "verdict": verdict,
        "parser_valid_rate_drop_points": drop_points,
        "hard_leakage_count": hard,
        "normalized_action_js_divergence": jensen_shannon_counts(
            frozen["normalized_action_counts"], adapted["normalized_action_counts"]
        ),
    }
