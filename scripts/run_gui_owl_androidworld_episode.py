"""Run one frozen-plan AndroidWorld episode with the pinned GUI-Owl policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl import (
    build_gui_owl_native_messages,
    gui_owl_action_to_androidworld,
)
from causalcache.policy.qwen_runtime import QwenPolicyRuntime
from scripts.run_androidworld_environment_smoke import request_json, request_raw


def select_instance(
    plan: dict[str, Any],
    *,
    task_type: str,
    task_index: int,
) -> dict[str, Any]:
    if plan.get("split") != "validation":
        raise ValueError("GUI-Owl rollout accepts only the frozen validation plan")
    matches = [
        instance
        for instance in plan["instances"]
        if instance["task_type"] == task_type and instance["task_index"] == task_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one validation instance for {task_type}[{task_index}]"
        )
    return matches[0]


def load_screenshot(base_url: str) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from PIL import Image

    payload, content_type = request_raw(
        base_url,
        "/screenshot",
        params={"wait_to_stabilize": "true"},
    )
    if content_type != "application/json":
        raise ValueError(f"unexpected screenshot content type: {content_type}")
    decoded = json.loads(payload)
    if not isinstance(decoded, dict) or "pixels" not in decoded:
        raise ValueError("screenshot response does not contain pixels")
    pixels = np.asarray(decoded["pixels"], dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] not in {3, 4}:
        raise ValueError(f"unexpected screenshot shape: {pixels.shape}")
    image = Image.fromarray(pixels).convert("RGB").copy()
    metadata = {
        "width": image.width,
        "height": image.height,
        "response_bytes": len(payload),
        "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
    }
    return image, metadata


def http_error_record(error: urllib.error.HTTPError) -> dict[str, Any]:
    payload = error.read()
    return {
        "status_code": error.code,
        "reason": str(error.reason),
        "response_body": payload.decode("utf-8", errors="replace"),
    }


def apply_answer_followup_override(
    action: dict[str, Any],
    *,
    previous_executed_action: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    if (
        previous_executed_action is not None
        and previous_executed_action["action_type"] == "answer"
    ):
        return {"action_type": "status", "goal_status": "task_complete"}, True
    return action, False


def run_episode(
    args: argparse.Namespace,
    *,
    runtime: Any | None = None,
) -> dict[str, Any]:
    plan = json.loads(args.validation_plan.read_text(encoding="utf-8"))
    instance = select_instance(
        plan,
        task_type=args.task_type,
        task_index=args.task_index,
    )
    if runtime is None:
        runtime = QwenPolicyRuntime(
            model_dir=args.model_dir,
            device=args.device,
            visual_tokens_per_image=args.visual_tokens_per_image,
        )
    summary: dict[str, Any] = {
        "schema_version": "0.1.0",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "validation_plan": {
            "path": str(args.validation_plan),
            "instance_records_sha256": plan["instance_records_sha256"],
            "suite_seed": plan["suite_seed"],
            "task_combinations": plan["task_combinations"],
        },
        "instance": instance,
        "policy": {
            **runtime.metadata,
            "visual_tokens_per_image": args.visual_tokens_per_image,
            "maximum_visible_images": args.maximum_visible_images,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        "steps": [],
    }
    started = time.monotonic()
    task_params = {"task_type": args.task_type, "task_idx": args.task_index}
    initialized = False
    screenshots: list[Any] = []
    action_outputs: list[str] = []
    executed_actions: list[dict[str, Any]] = []
    parse_successes = 0
    termination_reason = "step_budget_exhausted"

    health = request_json(args.base_url, "/health")
    if health.get("status") != "success":
        raise ValueError(f"health check failed: {health}")
    summary["health"] = health
    summary["suite_reinitialize"] = request_json(
        args.base_url,
        "/suite/reinitialize",
        params={
            "n_task_combinations": plan["task_combinations"],
            "seed": plan["suite_seed"],
            "task_family": "android_world",
        },
    )

    try:
        summary["initialize"] = request_json(
            args.base_url,
            "/task/initialize",
            method="POST",
            params=task_params,
        )
        initialized = True
        server_goal = request_json(args.base_url, "/task/goal", params=task_params)["goal"]
        server_template = request_json(
            args.base_url, "/task/template", params=task_params
        )["template"]
        if server_goal != instance["goal"] or server_template != instance["template"]:
            raise ValueError("live AndroidWorld instance does not match the frozen plan")
        summary["reset"] = request_json(
            args.base_url,
            "/reset",
            method="POST",
            params={"go_home": str(instance["start_on_home_screen"]).lower()},
        )
        summary["score_before"] = request_json(
            args.base_url, "/task/score", params=task_params
        )["score"]
        if summary["score_before"] != 0.0:
            raise ValueError(f"task did not start at zero reward: {summary['score_before']}")

        screenshot, screenshot_metadata = load_screenshot(args.base_url)
        screenshots.append(screenshot)
        summary["screenshots"] = [screenshot_metadata]

        for step_index in range(instance["max_steps"]):
            messages = build_gui_owl_native_messages(
                instruction=instance["goal"],
                screenshots=screenshots,
                action_outputs=action_outputs,
                maximum_visible_images=args.maximum_visible_images,
            )
            generation = runtime.generate_text(
                messages,
                max_new_tokens=args.max_new_tokens,
            )
            step = {"step_index": step_index, "generation": generation}
            summary["steps"].append(step)
            try:
                generated_android_action = gui_owl_action_to_androidworld(
                    generation["output_text"],
                    screen_width=screenshot_metadata["width"],
                    screen_height=screenshot_metadata["height"],
                )
            except (KeyError, TypeError, ValueError) as error:
                step["parse_error"] = str(error)
                termination_reason = "parse_error"
                break

            parse_successes += 1
            android_action, answer_followup_override = apply_answer_followup_override(
                generated_android_action,
                previous_executed_action=(executed_actions[-1] if executed_actions else None),
            )
            if answer_followup_override:
                step["generated_androidworld_action"] = generated_android_action
                step["official_answer_followup_override"] = True
            step["androidworld_action"] = android_action
            try:
                step["execute_response"] = request_json(
                    args.base_url,
                    "/execute_action",
                    method="POST",
                    body=android_action,
                )
            except urllib.error.HTTPError as error:
                step["executor_error"] = http_error_record(error)
                termination_reason = "executor_error"
                break
            action_outputs.append(generation["output_text"])
            executed_actions.append(android_action)
            if android_action["action_type"] == "status":
                termination_reason = "policy_terminated"
                break

            screenshot, screenshot_metadata = load_screenshot(args.base_url)
            screenshots.append(screenshot)
            summary["screenshots"].append(screenshot_metadata)
            if len(screenshots) > args.maximum_visible_images:
                screenshots[-args.maximum_visible_images - 1] = None

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

    summary["termination_reason"] = termination_reason
    summary["run_status"] = "complete"
    summary["model_step_count"] = len(summary["steps"])
    summary["parse_success_count"] = parse_successes
    summary["parse_coverage"] = (
        parse_successes / len(summary["steps"]) if summary["steps"] else 0.0
    )
    summary["environment_success"] = summary["score_after"] == 1.0
    summary["policy_declared_done"] = termination_reason == "policy_terminated"
    summary["official_success"] = (
        summary["environment_success"] and summary["policy_declared_done"]
    )
    summary["terminal_success"] = summary["official_success"]
    summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--validation-plan", type=Path, required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--visual-tokens-per-image", type=int, default=256)
    parser.add_argument("--maximum-visible-images", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_episode(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "task_type": summary["instance"]["task_type"],
                "task_index": summary["instance"]["task_index"],
                "model_step_count": summary["model_step_count"],
                "parse_coverage": summary["parse_coverage"],
                "termination_reason": summary["termination_reason"],
                "environment_success": summary["environment_success"],
                "official_success": summary["official_success"],
                "terminal_success": summary["terminal_success"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
