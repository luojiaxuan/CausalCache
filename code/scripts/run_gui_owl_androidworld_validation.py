"""Run the frozen GUI-Owl validation plan with resumable emulator workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
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


def success_gate_is_mathematically_impossible(
    *,
    planned_count: int,
    checkpoint_count: int,
    official_successes: int,
    minimum_official_success: float,
) -> bool:
    if not 0 <= checkpoint_count <= planned_count:
        raise ValueError("checkpoint count must be within the frozen plan")
    if not 0 <= official_successes <= checkpoint_count:
        raise ValueError("official successes must be within completed checkpoints")
    minimum_required_successes = math.ceil(
        minimum_official_success * planned_count
    )
    maximum_possible_successes = official_successes + (
        planned_count - checkpoint_count
    )
    return maximum_possible_successes < minimum_required_successes


def validate_git_checkout(expected_commit: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("run git commit must be a full lowercase SHA")
    repository_root = Path(__file__).resolve().parents[2]
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != expected_commit:
        raise ValueError("run git commit does not match the checkout")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("validation requires a clean git checkout")


def build_run_contract(
    *,
    git_commit: str,
    plan: dict[str, Any],
    runtime_metadata: dict[str, Any],
    maximum_visible_images: int,
    max_new_tokens: int,
    server_image: str,
    server_image_sha256: str,
) -> dict[str, Any]:
    snapshot = runtime_metadata["snapshot"]
    snapshot_sha256 = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "git_commit": git_commit,
        "validation_plan_records_sha256": plan["instance_records_sha256"],
        "model": {
            "repo": snapshot["repo"],
            "revision": snapshot["revision"],
            "snapshot_sha256": snapshot_sha256,
        },
        "runtime": {
            "model_class": runtime_metadata["model_class"],
            "processor_class": runtime_metadata["processor_class"],
            "dtype": runtime_metadata["dtype"],
            "torch_version": runtime_metadata["torch_version"],
            "transformers_version": runtime_metadata["transformers_version"],
            "visual_preprocessing": runtime_metadata["visual_preprocessing"],
        },
        "maximum_visible_images": maximum_visible_images,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "server_image": server_image,
        "server_image_sha256": server_image_sha256,
    }


def validate_resume_checkpoint(
    episode: dict[str, Any],
    *,
    plan_index: int,
    instance: dict[str, Any],
    base_url: str,
    run_contract: dict[str, Any],
) -> None:
    if episode.get("plan_index") != plan_index:
        raise ValueError("resume checkpoint plan index does not match")
    if episode.get("instance") != instance:
        raise ValueError("resume checkpoint instance does not match")
    if episode.get("run_contract") != run_contract:
        raise ValueError("resume checkpoint run contract does not match")
    expected_environment = {
        "base_url": base_url,
        "server_image": run_contract["server_image"],
        "server_image_sha256": run_contract["server_image_sha256"],
    }
    if episode.get("environment_runtime") != expected_environment:
        raise ValueError("resume checkpoint environment runtime does not match")


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
    if not success_gate_is_mathematically_impossible(
        planned_count=planned_count,
        checkpoint_count=checkpoint_count,
        official_successes=official_successes,
        minimum_official_success=minimum_official_success,
    ):
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
    validate_git_checkout(args.run_git_commit)
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
    run_contract = build_run_contract(
        git_commit=args.run_git_commit,
        plan=plan,
        runtime_metadata=runtime.metadata,
        maximum_visible_images=args.maximum_visible_images,
        max_new_tokens=args.max_new_tokens,
        server_image=args.server_image,
        server_image_sha256=args.server_image_sha256,
    )
    assignments = build_assignments(plan["instances"], args.base_url)
    episodes_dir = args.output_dir / "episodes"
    started_at = datetime.now(timezone.utc).isoformat()
    checkpoint_lock = threading.Lock()
    stop_event = threading.Event()

    def completed_episode_paths() -> list[Path]:
        return [
            episodes_dir / episode_filename(plan_index, instance)
            for plan_index, instance in enumerate(plan["instances"])
            if (episodes_dir / episode_filename(plan_index, instance)).exists()
        ]

    def update_early_stop() -> None:
        if not args.early_stop_when_success_is_mathematically_impossible:
            return
        with checkpoint_lock:
            paths = completed_episode_paths()
            episodes = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            official_successes = sum(
                bool(episode.get("official_success", False)) for episode in episodes
            )
            if success_gate_is_mathematically_impossible(
                planned_count=int(plan["task_instance_count"]),
                checkpoint_count=len(episodes),
                official_successes=official_successes,
                minimum_official_success=args.minimum_official_success,
            ):
                stop_event.set()

    initial_episode_paths = completed_episode_paths()
    if initial_episode_paths and not args.resume:
        raise ValueError("output directory already contains validation checkpoints")
    for path in initial_episode_paths:
        episode = json.loads(path.read_text(encoding="utf-8"))
        plan_index = int(episode["plan_index"])
        if not 0 <= plan_index < int(plan["task_instance_count"]):
            raise ValueError(f"resume checkpoint plan index is out of range: {path}")
        expected_path = episodes_dir / episode_filename(
            plan_index, plan["instances"][plan_index]
        )
        if path != expected_path:
            raise ValueError(f"resume checkpoint filename does not match its plan index: {path}")
        validate_resume_checkpoint(
            episode,
            plan_index=plan_index,
            instance=plan["instances"][plan_index],
            base_url=args.base_url[plan_index % len(args.base_url)],
            run_contract=run_contract,
        )
    update_early_stop()

    def worker(
        base_url: str,
        assigned: list[tuple[int, dict[str, Any]]],
    ) -> list[Path]:
        paths = []
        for plan_index, instance in assigned:
            if stop_event.is_set():
                break
            path = episodes_dir / episode_filename(plan_index, instance)
            if args.resume and path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                validate_resume_checkpoint(
                    existing,
                    plan_index=plan_index,
                    instance=instance,
                    base_url=base_url,
                    run_contract=run_contract,
                )
                paths.append(path)
                update_early_stop()
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
            episode["run_contract"] = run_contract
            episode["environment_runtime"] = {
                "base_url": base_url,
                "server_image": args.server_image,
                "server_image_sha256": args.server_image_sha256,
            }
            write_json_atomic(path, episode)
            paths.append(path)
            update_early_stop()
        return paths

    with ThreadPoolExecutor(max_workers=len(args.base_url)) as executor:
        futures = [
            executor.submit(worker, base_url, assigned)
            for base_url, assigned in zip(args.base_url, assignments, strict=True)
        ]
        for future in futures:
            future.result()

    episode_paths = completed_episode_paths()
    episodes = [json.loads(path.read_text(encoding="utf-8")) for path in episode_paths]
    if len(episodes) == int(plan["task_instance_count"]):
        summary = aggregate_validation(
            plan=plan,
            episodes=episodes,
            minimum_parse_coverage=args.minimum_parse_coverage,
            minimum_official_success=args.minimum_official_success,
        )
    elif args.early_stop_when_success_is_mathematically_impossible:
        summary = aggregate_early_stopped_validation(
            plan=plan,
            episodes=episodes,
            minimum_parse_coverage=args.minimum_parse_coverage,
            minimum_official_success=args.minimum_official_success,
        )
    else:
        raise ValueError("validation finished without every frozen-plan checkpoint")
    summary.update(
        {
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "base_urls": args.base_url,
            "server_image": args.server_image,
            "server_image_sha256": args.server_image_sha256,
            "run_contract": run_contract,
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
    parser.add_argument(
        "--early-stop-when-success-is-mathematically-impossible",
        action="store_true",
    )
    parser.add_argument("--server-image", required=True)
    parser.add_argument("--server-image-sha256", required=True)
    parser.add_argument("--run-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run_validation(parse_args())
    result = {
        "parse_coverage": summary["parse_coverage"],
        "outcomes": summary["outcomes"],
        "gates": summary["gates"],
    }
    if "task_instance_count" in summary:
        result.update(
            {
                "task_instance_count": summary["task_instance_count"],
                "official_success_rate": summary["official_success_rate"],
            }
        )
    else:
        result.update(
            {
                "artifact_status": summary["artifact_status"],
                "plan_instance_count": summary["plan_instance_count"],
                "checkpoint_count": summary["checkpoint_count"],
                "maximum_possible_official_success_rate": summary[
                    "maximum_possible_official_success_rate"
                ],
            }
        )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
