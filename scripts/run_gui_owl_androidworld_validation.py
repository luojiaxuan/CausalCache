"""Run the frozen GUI-Owl validation plan with resumable emulator workers."""

from __future__ import annotations

import argparse
import json
import math
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causalcache.policy.qwen_runtime import QwenPolicyRuntime
from scripts.run_androidworld_environment_smoke import request_json
from scripts.run_gui_owl_androidworld_episode import run_episode


class SerializedRuntime:
    def __init__(self, runtime: QwenPolicyRuntime) -> None:
        self._runtime = runtime
        self._lock = threading.Lock()
        self.metadata = runtime.metadata

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        with self._lock:
            result = self._runtime.generate_text(
                messages,
                max_new_tokens=max_new_tokens,
            )
        if not isinstance(result, dict):
            raise TypeError("serialized runtime expected a generation dictionary")
        return result


def episode_filename(plan_index: int, instance: dict[str, Any]) -> str:
    return (
        f"{plan_index:03d}-{instance['task_type']}-"
        f"{instance['task_index']}.json"
    )


def build_assignments(
    instances: list[dict[str, Any]],
    base_urls: list[str],
) -> list[list[tuple[int, dict[str, Any]]]]:
    if not base_urls:
        raise ValueError("at least one AndroidWorld base URL is required")
    assignments: list[list[tuple[int, dict[str, Any]]]] = [
        [] for _ in base_urls
    ]
    for plan_index, instance in enumerate(instances):
        assignments[plan_index % len(base_urls)].append((plan_index, instance))
    return assignments


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def aggregate_validation(
    *,
    plan: dict[str, Any],
    episodes: list[dict[str, Any]],
    minimum_parse_coverage: float,
    minimum_official_success: float,
) -> dict[str, Any]:
    if len(episodes) != plan["task_instance_count"]:
        raise ValueError("episode count does not match the frozen validation plan")
    model_steps = sum(int(episode.get("model_step_count", 0)) for episode in episodes)
    parse_successes = sum(
        int(episode.get("parse_success_count", 0)) for episode in episodes
    )
    parse_coverage = parse_successes / model_steps if model_steps else 0.0
    official_successes = sum(bool(episode.get("official_success", False)) for episode in episodes)
    environment_successes = sum(
        bool(episode.get("environment_success", False)) for episode in episodes
    )
    outcomes: Counter[str] = Counter()
    termination_reasons: Counter[str] = Counter()
    for episode in episodes:
        termination_reason = str(episode.get("termination_reason", "exception"))
        termination_reasons[termination_reason] += 1
        if episode.get("run_status") == "exception":
            outcomes["infrastructure_failure"] += 1
        elif termination_reason == "parse_error":
            outcomes["parse_failure"] += 1
        elif termination_reason == "executor_error":
            outcomes["executor_failure"] += 1
        elif episode.get("official_success", False):
            outcomes["official_success"] += 1
        else:
            outcomes["terminal_failure"] += 1
    official_success_rate = official_successes / len(episodes)
    parse_gate_passed = parse_coverage >= minimum_parse_coverage
    success_gate_passed = official_success_rate >= minimum_official_success
    return {
        "schema_version": "0.1.0",
        "split": plan["split"],
        "instance_records_sha256": plan["instance_records_sha256"],
        "task_instance_count": len(episodes),
        "completed_episode_count": sum(
            episode.get("run_status") == "complete" for episode in episodes
        ),
        "exception_episode_count": sum(
            episode.get("run_status") == "exception" for episode in episodes
        ),
        "model_step_count": model_steps,
        "parse_success_count": parse_successes,
        "parse_coverage": parse_coverage,
        "environment_success_count": environment_successes,
        "official_success_count": official_successes,
        "official_success_rate": official_success_rate,
        "outcomes": dict(sorted(outcomes.items())),
        "termination_reasons": dict(sorted(termination_reasons.items())),
        "gates": {
            "minimum_parse_coverage": minimum_parse_coverage,
            "parse_gate_passed": parse_gate_passed,
            "minimum_official_success": minimum_official_success,
            "official_success_gate_passed": success_gate_passed,
            "validation_gate_passed": parse_gate_passed and success_gate_passed,
        },
    }


def aggregate_early_stopped_validation(
    *,
    plan: dict[str, Any],
    episodes: list[dict[str, Any]],
    minimum_parse_coverage: float,
    minimum_official_success: float,
) -> dict[str, Any]:
    planned_count = int(plan["task_instance_count"])
    checkpoint_count = len(episodes)
    if not 0 < checkpoint_count < planned_count:
        raise ValueError("early-stop summary requires a non-empty partial run")
    seen_indices: set[int] = set()
    for episode in episodes:
        plan_index = int(episode["plan_index"])
        if plan_index in seen_indices:
            raise ValueError(f"duplicate validation plan index: {plan_index}")
        if not 0 <= plan_index < planned_count:
            raise ValueError(f"validation plan index is out of range: {plan_index}")
        if episode.get("instance") != plan["instances"][plan_index]:
            raise ValueError(f"episode does not match frozen plan index {plan_index}")
        seen_indices.add(plan_index)

    model_steps = sum(int(episode.get("model_step_count", 0)) for episode in episodes)
    parse_successes = sum(
        int(episode.get("parse_success_count", 0)) for episode in episodes
    )
    parse_coverage = parse_successes / model_steps if model_steps else 0.0
    official_successes = sum(
        bool(episode.get("official_success", False)) for episode in episodes
    )
    unobserved_count = planned_count - checkpoint_count
    maximum_possible_successes = official_successes + unobserved_count
    minimum_required_successes = math.ceil(
        minimum_official_success * planned_count
    )
    if maximum_possible_successes >= minimum_required_successes:
        raise ValueError("success gate is not yet mathematically impossible")

    outcomes: Counter[str] = Counter()
    termination_reasons: Counter[str] = Counter()
    for episode in episodes:
        termination_reason = str(episode.get("termination_reason", "exception"))
        termination_reasons[termination_reason] += 1
        if episode.get("run_status") == "exception":
            outcomes["infrastructure_failure"] += 1
        elif termination_reason == "parse_error":
            outcomes["parse_failure"] += 1
        elif termination_reason == "executor_error":
            outcomes["executor_failure"] += 1
        elif episode.get("official_success", False):
            outcomes["official_success"] += 1
        else:
            outcomes["terminal_failure"] += 1

    parse_gate_passed = parse_coverage >= minimum_parse_coverage
    return {
        "schema_version": "0.1.0",
        "artifact_status": "valid_early_stopped_policy_rejection",
        "split": plan["split"],
        "instance_records_sha256": plan["instance_records_sha256"],
        "plan_instance_count": planned_count,
        "checkpoint_count": checkpoint_count,
        "unobserved_instance_count": unobserved_count,
        "completed_episode_count": sum(
            episode.get("run_status") == "complete" for episode in episodes
        ),
        "exception_episode_count": sum(
            episode.get("run_status") == "exception" for episode in episodes
        ),
        "model_step_count": model_steps,
        "parse_success_count": parse_successes,
        "parse_coverage": parse_coverage,
        "official_success_count": official_successes,
        "official_success_rate_lower_bound": official_successes / planned_count,
        "minimum_required_official_success_count": minimum_required_successes,
        "maximum_possible_official_success_count": maximum_possible_successes,
        "maximum_possible_official_success_rate": (
            maximum_possible_successes / planned_count
        ),
        "outcomes": dict(sorted(outcomes.items())),
        "termination_reasons": dict(sorted(termination_reasons.items())),
        "early_stop_reason": "success_gate_mathematically_impossible",
        "gates": {
            "minimum_parse_coverage": minimum_parse_coverage,
            "parse_gate_passed_on_observed_actions": parse_gate_passed,
            "minimum_official_success": minimum_official_success,
            "official_success_gate_passed": False,
            "validation_gate_passed": False,
        },
    }


def exception_episode(
    *,
    base_url: str,
    instance: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "run_status": "exception",
        "base_url": base_url,
        "instance": instance,
        "model_step_count": 0,
        "parse_success_count": 0,
        "parse_coverage": 0.0,
        "environment_success": False,
        "policy_declared_done": False,
        "official_success": False,
        "terminal_success": False,
        "termination_reason": "exception",
        "failure": {
            "type": error.__class__.__name__,
            "message": str(error),
        },
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads(args.validation_plan.read_text(encoding="utf-8"))
    if plan.get("split") != "validation":
        raise ValueError("full runner accepts only the frozen validation plan")
    if len(set(args.base_url)) != len(args.base_url):
        raise ValueError("AndroidWorld base URLs must be unique")
    for base_url in args.base_url:
        health = request_json(base_url, "/health")
        if health.get("status") != "success":
            raise ValueError(f"health check failed for {base_url}: {health}")

    raw_runtime = QwenPolicyRuntime(
        model_dir=args.model_dir,
        device=args.device,
        visual_tokens_per_image=args.visual_tokens_per_image,
    )
    runtime = SerializedRuntime(raw_runtime)
    assignments = build_assignments(plan["instances"], args.base_url)
    episodes_dir = args.output_dir / "episodes"
    started_at = datetime.now(timezone.utc).isoformat()

    def worker(
        base_url: str,
        assigned: list[tuple[int, dict[str, Any]]],
    ) -> list[Path]:
        paths = []
        for plan_index, instance in assigned:
            path = episodes_dir / episode_filename(plan_index, instance)
            if args.resume and path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("instance") != instance:
                    raise ValueError(f"resume checkpoint does not match plan: {path}")
                paths.append(path)
                continue
            episode_args = argparse.Namespace(
                base_url=base_url,
                model_dir=args.model_dir,
                validation_plan=args.validation_plan,
                task_type=instance["task_type"],
                task_index=instance["task_index"],
                device=args.device,
                visual_tokens_per_image=args.visual_tokens_per_image,
                maximum_visible_images=args.maximum_visible_images,
                max_new_tokens=args.max_new_tokens,
            )
            try:
                episode = run_episode(episode_args, runtime=runtime)
            except Exception as error:
                episode = exception_episode(
                    base_url=base_url,
                    instance=instance,
                    error=error,
                )
            episode["plan_index"] = plan_index
            episode["environment_runtime"] = {
                "base_url": base_url,
                "server_image": args.server_image,
                "server_image_sha256": args.server_image_sha256,
            }
            write_json_atomic(path, episode)
            paths.append(path)
        return paths

    with ThreadPoolExecutor(max_workers=len(args.base_url)) as executor:
        futures = [
            executor.submit(worker, base_url, assigned)
            for base_url, assigned in zip(args.base_url, assignments, strict=True)
        ]
        for future in futures:
            future.result()

    episode_paths = [
        episodes_dir / episode_filename(plan_index, instance)
        for plan_index, instance in enumerate(plan["instances"])
    ]
    episodes = [json.loads(path.read_text(encoding="utf-8")) for path in episode_paths]
    summary = aggregate_validation(
        plan=plan,
        episodes=episodes,
        minimum_parse_coverage=args.minimum_parse_coverage,
        minimum_official_success=args.minimum_official_success,
    )
    summary.update(
        {
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "base_urls": args.base_url,
            "server_image": args.server_image,
            "server_image_sha256": args.server_image_sha256,
            "policy": {
                **runtime.metadata,
                "visual_tokens_per_image": args.visual_tokens_per_image,
                "maximum_visible_images": args.maximum_visible_images,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
            },
            "episode_files": [str(path.relative_to(args.output_dir)) for path in episode_paths],
        }
    )
    write_json_atomic(args.output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--validation-plan", type=Path, required=True)
    parser.add_argument("--device", required=True)
    visual_group = parser.add_mutually_exclusive_group(required=True)
    visual_group.add_argument(
        "--use-model-default-visual-resolution",
        action="store_true",
    )
    visual_group.add_argument("--visual-tokens-per-image", type=int)
    parser.add_argument("--maximum-visible-images", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--minimum-parse-coverage", type=float, default=0.95)
    parser.add_argument("--minimum-official-success", type=float, default=0.5)
    parser.add_argument("--server-image", required=True)
    parser.add_argument("--server-image-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run_validation(parse_args())
    print(
        json.dumps(
            {
                "task_instance_count": summary["task_instance_count"],
                "parse_coverage": summary["parse_coverage"],
                "official_success_rate": summary["official_success_rate"],
                "outcomes": summary["outcomes"],
                "gates": summary["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
