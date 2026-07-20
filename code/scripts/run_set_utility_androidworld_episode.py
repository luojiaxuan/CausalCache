#!/usr/bin/env python3
"""Run one post-GO memory arm on outcome-exposed AndroidWorld validation-12."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
import urllib.error
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2 import gui_owl_v2_action_to_androidworld
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21GenerationParseError
from causalcache.set_utility_androidworld import (
    ANDROIDWORLD_READY_STATUS,
    EARLY_STEP_POLICY_ID,
    LiveSetUtilityPolicyAdapter,
    OnlineObservation,
    PinnedOnlineOCRProvider,
    SUPPORTED_LIVE_ARMS,
    authorization_from_go_artifacts,
    load_outcome_exposed_validation_instance,
    probe_live_selector_health,
    selector_http_call,
    validate_loopback_http_url,
)
from scripts.run_androidworld_environment_smoke import request_json
from scripts.run_gui_owl_androidworld_episode import (
    apply_answer_followup_override,
    http_error_record,
    load_screenshot,
)


COMPLETE_EPISODE_STATUS = "COMPLETE_OUTCOME_EXPOSED_VALIDATION12_MEMORY_EPISODE"
TOPOLOGY_ONLY_STATUS = "TOPOLOGY_CHECK_ONLY"


def write_episode_output_atomic(path: Path, value: dict[str, Any]) -> None:
    """Publish one terminal episode record atomically without overwriting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"episode output already exists and was not overwritten: {path}"
            ) from error
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def load_resumable_terminal_output(
    path: Path,
    *,
    arm: str,
    budget: int,
    episode_id: str,
    resume_completed: bool,
    task_index: int,
    task_type: str,
    topology_check_only: bool,
) -> dict[str, Any] | None:
    """Fail on an existing result unless explicit identity-safe reuse is requested."""
    if not path.exists():
        return None
    if not resume_completed:
        raise FileExistsError(
            f"episode output already exists; use --resume-completed to reuse it: {path}"
        )
    if path.is_symlink() or not path.is_file():
        raise ValueError("resumable episode output must be one regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("existing episode output is not valid terminal JSON") from error
    if not isinstance(value, dict):
        raise ValueError("existing episode output must contain one JSON object")
    expected_status = TOPOLOGY_ONLY_STATUS if topology_check_only else COMPLETE_EPISODE_STATUS
    instance = value.get("instance")
    if (
        value.get("run_status") != expected_status
        or value.get("arm") != f"{arm}_B{budget}"
        or value.get("episode_id") != episode_id
        or not isinstance(instance, dict)
        or instance.get("task_type") != task_type
        or instance.get("task_index") != task_index
    ):
        raise ValueError("existing terminal episode identity differs from this invocation")
    if not topology_check_only and "terminal_success" not in value:
        raise ValueError("existing completed episode lacks terminal success evidence")
    return value


class HTTPAndroidWorldEnvironment:
    """One validation episode over a loopback-forwarded AndroidWorld server."""

    def __init__(
        self,
        *,
        base_url: str,
        instance: dict[str, Any],
        observation_namespace: str,
        ocr_provider: PinnedOnlineOCRProvider,
    ) -> None:
        self.base_url = validate_loopback_http_url(
            base_url, "AndroidWorld base URL"
        )
        self.instance = instance
        self.observation_namespace = observation_namespace
        self.ocr_provider = ocr_provider
        self.task_params = {
            "task_type": instance["task_type"],
            "task_idx": instance["task_index"],
        }
        self.observation_index = 0
        self.initialized = False

    def initialize(self) -> dict[str, Any]:
        suite = request_json(
            self.base_url,
            "/suite/reinitialize",
            params={
                "n_task_combinations": 2,
                "seed": 271828,
                "task_family": "android_world",
            },
        )
        initialized = request_json(
            self.base_url,
            "/task/initialize",
            method="POST",
            params=self.task_params,
        )
        self.initialized = True
        goal = request_json(
            self.base_url, "/task/goal", params=self.task_params
        )["goal"]
        template = request_json(
            self.base_url, "/task/template", params=self.task_params
        )["template"]
        if goal != self.instance["goal"] or template != self.instance["template"]:
            raise ValueError("live AndroidWorld task differs from validation-12 identity")
        reset = request_json(
            self.base_url,
            "/reset",
            method="POST",
            params={
                "go_home": str(self.instance["start_on_home_screen"]).lower()
            },
        )
        return {"initialize": initialized, "reset": reset, "suite": suite}

    def score(self) -> float:
        value = request_json(
            self.base_url, "/task/score", params=self.task_params
        )["score"]
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("AndroidWorld score must be finite")
        return score

    def screenshot(self) -> OnlineObservation:
        image, metadata = load_screenshot(self.base_url)
        member = (
            f"live/{self.observation_namespace}/"
            f"observation-{self.observation_index:03d}.png"
        )
        self.observation_index += 1
        return self.ocr_provider.observe(
            image=image,
            image_member_path=member,
            metadata=metadata,
        )

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        return request_json(
            self.base_url,
            "/execute_action",
            method="POST",
            body=action,
        )

    def tear_down(self) -> dict[str, Any] | None:
        if not self.initialized:
            return None
        self.initialized = False
        return request_json(
            self.base_url,
            "/task/tear_down",
            method="POST",
            params=self.task_params,
        )


def probe_androidworld_loopback(
    *,
    base_url: str,
    maximum_rtt_ms: float,
    attempts: int = 3,
    request: Any = request_json,
) -> dict[str, Any]:
    base_url = validate_loopback_http_url(base_url, "AndroidWorld base URL")
    if not 1 <= attempts <= 20:
        raise ValueError("AndroidWorld health attempts must be in [1,20]")
    maximum = float(maximum_rtt_ms)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum AndroidWorld RTT must be finite and positive")
    latencies = []
    for _ in range(attempts):
        started = time.perf_counter()
        health = request(base_url, "/health")
        elapsed = (time.perf_counter() - started) * 1000.0
        if health.get("status") != ANDROIDWORLD_READY_STATUS:
            raise RuntimeError("AndroidWorld loopback health check failed")
        latencies.append(elapsed)
    observed = max(latencies)
    if observed > maximum:
        raise RuntimeError("AndroidWorld relay RTT exceeds the explicit canary gate")
    return {
        "attempt_count": attempts,
        "maximum_allowed_rtt_ms": maximum,
        "maximum_observed_rtt_ms": observed,
        "status": "PASSED_ANDROIDWORLD_LOOPBACK_CANARY",
    }


def _decode_png(payload: bytes) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB").copy()


def _load_policy_runtime(args: argparse.Namespace) -> Any:
    from causalcache.policy.gui_owl_variable_history_runtime import (
        GUIOwlVariableHistoryRuntime,
        VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )

    if VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE != 480:
        raise RuntimeError("live policy visual-token profile drifted")
    return GUIOwlVariableHistoryRuntime(
        model_dir=args.model_dir.resolve(),
        expected_snapshot_manifest=(
            args.repository_root.resolve()
            / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device=args.device,
        target_effective_visual_tokens_per_image=480,
    )


def run_episode(
    args: argparse.Namespace,
    *,
    authorization: Any | None = None,
    environment: Any | None = None,
    ocr_provider: Any | None = None,
    policy_runtime: Any | None = None,
    selector: Any | None = None,
    topology_canary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    roster = load_outcome_exposed_validation_instance(
        repository_root=args.repository_root,
        task_type=args.task_type,
        task_index=args.task_index,
    )
    instance = roster["instance"]
    if authorization is None:
        authorization = authorization_from_go_artifacts(
            heldout_config_path=args.heldout_config,
            selections_path=args.selections,
            heldout_result_path=args.heldout_result,
            native_replay_result_path=args.native_replay_result,
            checkpoint_path=args.checkpoint,
            expected_heldout_result_sha256=args.expected_heldout_result_sha256,
            expected_native_replay_result_sha256=(
                args.expected_native_replay_result_sha256
            ),
        )
    if topology_canary is None:
        topology_canary = {
            "androidworld": probe_androidworld_loopback(
                base_url=args.androidworld_base_url,
                maximum_rtt_ms=args.maximum_androidworld_rtt_ms,
            ),
        }
        if args.arm == "winner":
            if args.selector_endpoint is None or args.maximum_selector_rtt_ms is None:
                raise ValueError("winner arm requires selector endpoint and RTT gate")
            topology_canary["selector"] = probe_live_selector_health(
                endpoint=args.selector_endpoint,
                authorization=authorization,
                maximum_rtt_ms=args.maximum_selector_rtt_ms,
            )
        else:
            topology_canary["selector"] = {
                "arm": args.arm,
                "status": "NOT_USED_BY_LOCAL_BASELINE",
            }
    summary: dict[str, Any] = {
        "arm": f"{args.arm}_B{args.budget}",
        "authorization": authorization.to_mapping(),
        "budget_contract": {
            "current_observation_cost": 0,
            "high_fidelity_history_capacity": args.budget,
            "selection": "at_most_B",
        },
        "early_step_policy_id": EARLY_STEP_POLICY_ID,
        "episode_id": args.episode_id,
        "evidence_role": roster["evidence_role"],
        "instance": instance,
        "roster": {
            key: value for key, value in roster.items() if key != "instance"
        },
        "schema_version": "1.0.0",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": [],
        "topology_canary": topology_canary,
    }
    if args.topology_check_only:
        summary["run_status"] = TOPOLOGY_ONLY_STATUS
        return summary
    if ocr_provider is None:
        ocr_provider = PinnedOnlineOCRProvider.load(
            backend_config_path=args.ocr_backend_config,
            backend_manifest_path=args.ocr_backend_manifest,
            model_dir=args.ocr_model_dir,
        )
    if policy_runtime is None:
        policy_runtime = _load_policy_runtime(args)
    if selector is None and args.arm == "winner":
        if args.selector_endpoint is None:
            raise ValueError("winner arm requires a selector endpoint")
        selector = selector_http_call(
            endpoint=args.selector_endpoint,
            authorization=authorization,
            timeout_seconds=args.selector_timeout_seconds,
        )
    episode_namespace = hashlib.sha256(args.episode_id.encode("utf-8")).hexdigest()[:24]
    source_id = (
        f"{args.episode_id}:{instance['task_type']}:{instance['task_index']}:"
        f"{args.arm}_B{args.budget}"
    )
    adapter = LiveSetUtilityPolicyAdapter(
        arm=args.arm,
        authorization=authorization,
        budget=args.budget,
        expected_heldout_result_sha256=args.expected_heldout_result_sha256,
        expected_native_replay_result_sha256=(
            args.expected_native_replay_result_sha256
        ),
        image_decoder=_decode_png,
        instruction=instance["goal"],
        ocr_backend_config=ocr_provider.backend_config,
        ocr_backend_config_sha256=ocr_provider.backend_config_sha256,
        selector_call=selector,
        source_id=source_id,
    )
    if environment is None:
        environment = HTTPAndroidWorldEnvironment(
            base_url=args.androidworld_base_url,
            instance=instance,
            observation_namespace=episode_namespace,
            ocr_provider=ocr_provider,
        )
    started = time.monotonic()
    initialized = False
    parse_successes = 0
    previous_executed_action = None
    termination_reason = "step_budget_exhausted"
    try:
        summary["environment_initialize"] = environment.initialize()
        initialized = True
        summary["score_before"] = environment.score()
        if summary["score_before"] != 0.0:
            raise ValueError("outcome-exposed episode did not start at zero reward")
        current = environment.screenshot()
        for step_index in range(instance["max_steps"]):
            prepared = adapter.prepare(current)
            step: dict[str, Any] = {
                "budget": prepared.budget,
                "decision_step_id": prepared.decision_step_id,
                "high_fidelity_history_image_count": (
                    prepared.high_fidelity_history_image_count
                ),
                "policy_image_count": prepared.policy_image_count,
                "request_content_sha256": prepared.request_content_sha256,
                "rpc_round_trip_ms": prepared.rpc_round_trip_ms,
                "selected_event_step_ids": list(prepared.selected_event_step_ids),
                "selection_diagnostics": prepared.selection_diagnostics,
                "selection_latency_ms": prepared.selection_latency_ms,
                "selection_method": prepared.selection_method,
                "selector_response": prepared.selector_response,
                "selector_used": prepared.selector_used,
                "step_index": step_index,
                "warmup_policy_id": prepared.warmup_policy_id,
            }
            summary["steps"].append(step)
            try:
                generated = policy_runtime.generate_native_action(prepared.messages)
            except GUIOwlV21GenerationParseError as error:
                step["generation"] = dict(error.metadata)
                step["native_output"] = error.output_text
                step["parse_error"] = {
                    "message": error.parse_error_message,
                    "type": error.parse_error_type,
                }
                termination_reason = "parse_error"
                break
            canonical_action = generated.parsed_output.canonical_action
            step["generation"] = dict(generated.metadata)
            step["native_output"] = generated.output_text
            step["canonical_action"] = canonical_action.arguments()
            parse_successes += 1
            android_action = gui_owl_v2_action_to_androidworld(
                canonical_action,
                screen_width=int(current.metadata["width"]),
                screen_height=int(current.metadata["height"]),
            )
            effective_action, overridden = apply_answer_followup_override(
                android_action,
                previous_executed_action=previous_executed_action,
            )
            if overridden:
                step["generated_androidworld_action"] = android_action
                step["official_answer_followup_override"] = True
            step["androidworld_action"] = effective_action
            try:
                step["execute_response"] = environment.execute(effective_action)
            except urllib.error.HTTPError as error:
                step["executor_error"] = http_error_record(error)
                termination_reason = "executor_error"
                break
            previous_executed_action = effective_action
            if effective_action["action_type"] == "status":
                termination_reason = "policy_terminated"
                break
            after = environment.screenshot()
            event = adapter.append_transition(
                action=canonical_action,
                before=current,
                after=after,
            )
            step["appended_event"] = {
                "event_step_id": event.event_step_id,
                "low_fidelity_v2": event.low_fidelity.to_ordered_dict(),
                "post_image_sha256": event.post_image.sha256,
                "post_ocr_token_count": len(event.post_ocr_tokens),
            }
            current = after
        summary["score_after"] = environment.score()
    finally:
        if initialized:
            summary["tear_down"] = environment.tear_down()
    summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
    summary["environment_success"] = summary["score_after"] == 1.0
    summary["model_step_count"] = len(summary["steps"])
    summary["parse_coverage"] = (
        parse_successes / len(summary["steps"]) if summary["steps"] else 0.0
    )
    summary["policy_declared_done"] = termination_reason == "policy_terminated"
    summary["run_status"] = COMPLETE_EPISODE_STATUS
    summary["terminal_success"] = (
        summary["environment_success"] and summary["policy_declared_done"]
    )
    summary["termination_reason"] = termination_reason
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--arm", choices=SUPPORTED_LIVE_ARMS, required=True)
    parser.add_argument("--androidworld-base-url", required=True)
    parser.add_argument("--selector-endpoint")
    parser.add_argument("--maximum-androidworld-rtt-ms", type=float, required=True)
    parser.add_argument("--maximum-selector-rtt-ms", type=float)
    parser.add_argument("--selector-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--topology-check-only", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--heldout-config", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--heldout-result", type=Path, required=True)
    parser.add_argument("--native-replay-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-heldout-result-sha256", required=True)
    parser.add_argument("--expected-native-replay-result-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ocr-backend-config", type=Path, required=True)
    parser.add_argument("--ocr-backend-manifest", type=Path, required=True)
    parser.add_argument("--ocr-model-dir", type=Path, required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--budget", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_resumable_terminal_output(
        args.output,
        arm=args.arm,
        budget=args.budget,
        episode_id=args.episode_id,
        resume_completed=args.resume_completed,
        task_index=args.task_index,
        task_type=args.task_type,
        topology_check_only=args.topology_check_only,
    )
    if summary is None:
        summary = run_episode(args)
        write_episode_output_atomic(args.output, summary)
    print(
        json.dumps(
            {
                "arm": summary["arm"],
                "run_status": summary["run_status"],
                "task_index": summary["instance"]["task_index"],
                "task_type": summary["instance"]["task_type"],
                "terminal_success": summary.get("terminal_success"),
                "termination_reason": summary.get("termination_reason"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
