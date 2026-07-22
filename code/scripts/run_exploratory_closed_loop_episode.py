#!/usr/bin/env python3
"""Run one exploratory validation-12 closed-loop episode for a local arm."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causalcache.exploratory_closed_loop_memory import (
    EXPLORATORY_MEMORY_BUDGET,
    OCR_RGB_ARM,
    RECENT_ARM,
    SUMMARY_ARM,
    build_live_gui_owl_v2_1_mixed_fidelity_messages,
    select_ocr_rgb_memory,
    select_recent_memory,
    select_summary_memory,
)
from causalcache.independent_closed_loop_features import (
    FROZEN_GATE_V1_OCR_BACKEND,
    live_history_event_from_transition,
    prepare_image_bytes,
)
from causalcache.policy.gui_owl_v2 import gui_owl_v2_action_to_androidworld
from causalcache.policy.gui_owl_v2_1_runtime import (
    GUIOwlV21GenerationParseError,
    GUIOwlV21OfficialToolsRuntime,
)
from causalcache.set_utility_androidworld import (
    EARLY_STEP_POLICY_ID,
    PinnedOnlineOCRProvider,
    build_shared_early_step_messages,
)
from causalcache.set_utility_heldout_evaluation import sha256_file
from causalcache.set_utility_live_controller import LiveRichEvent
from scripts.run_gui_owl_androidworld_episode import (
    apply_answer_followup_override,
    http_error_record,
)
from scripts.run_set_utility_androidworld_episode import (
    HTTPAndroidWorldEnvironment,
    write_episode_output_atomic,
)


LOCAL_ARMS = (SUMMARY_ARM, RECENT_ARM, OCR_RGB_ARM)
CEILING_ARMS = ("summary_B0", "recent_B2", "recent_B4", "recent_B8")
COMPLETE_STATUS = "COMPLETE_EXPLORATORY_VALIDATION12_EPISODE"
EFFECTIVE_VISUAL_TOKENS_PER_IMAGE = 2560


def ceiling_arm_budget(arm: str) -> int:
    if arm not in CEILING_ARMS:
        raise ValueError(f"unsupported memory ceiling arm: {arm}")
    return int(arm.rsplit("_B", 1)[1])


def _load_plan_instance(
    plan_path: Path, *, task_type: str, task_index: int
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("split") not in ("train", "validation"):
        raise ValueError("memory ceiling refuses sealed splits")
    matches = [
        instance
        for instance in plan["instances"]
        if instance["task_type"] == task_type
        and instance["task_index"] == task_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one plan instance for {task_type}[{task_index}]"
        )
    return {
        "instance": matches[0],
        "split": plan["split"],
        "suite_seed": int(plan["suite_seed"]),
        "task_combinations": int(plan["task_combinations"]),
    }


def _decode_png(payload: bytes) -> Any:
    from PIL import Image

    source = Image.open(io.BytesIO(payload))
    try:
        return source.convert("RGB").copy()
    finally:
        source.close()


def _load_instance(manifest_path: Path, *, task_type: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        record
        for record in manifest["records"]
        if record["instance"]["task_type"] == task_type
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one validation-12 record for {task_type}")
    record = matches[0]
    if record["instance"]["task_index"] != 0:
        raise ValueError("validation-12 permits only task_index zero")
    return record


def _select_arm_memory(
    *,
    arm: str,
    history: list[LiveRichEvent],
    current: Any,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    candidates = tuple(event.event_step_id for event in history)[:-1]
    if arm in CEILING_ARMS:
        budget = ceiling_arm_budget(arm)
        selected = (
            () if budget == 0 else candidates[-min(budget, len(candidates)) :]
        )
        return selected, {"method": f"ceiling_recent_latest_min_{budget}"}
    if arm == SUMMARY_ARM:
        return select_summary_memory(candidates), {"method": "summary_empty"}
    if arm == RECENT_ARM:
        return (
            select_recent_memory(candidates),
            {"method": "recent_latest_min_budget"},
        )
    if arm == OCR_RGB_ARM:
        selection = select_ocr_rgb_memory(
            event_step_ids=candidates,
            event_ocr_tokens={
                event.event_step_id: event.post_ocr_tokens
                for event in history[:-1]
            },
            current_ocr_tokens=current.ocr_tokens,
            event_resized_rgb_bytes={
                event.event_step_id: prepare_image_bytes(
                    event.post_image.payload
                ).resized_rgb_bytes
                for event in history[:-1]
            },
            current_resized_rgb_bytes=prepare_image_bytes(
                current.image_png
            ).resized_rgb_bytes,
        )
        return (
            selection.selected_event_step_ids,
            {
                "method": "ocr_rgb_equal_weight_similarity",
                "ranked_event_step_ids": list(selection.ranked_event_step_ids),
                "scores_by_event_step": [
                    [step_id, value]
                    for step_id, value in selection.scores_by_event_step
                ],
            },
        )
    raise ValueError(f"unsupported local exploratory arm: {arm}")


def run_episode(
    args: argparse.Namespace,
    *,
    runtime: Any | None = None,
    ocr_provider: Any | None = None,
) -> dict[str, Any]:
    ceiling_plan = getattr(args, "ceiling_plan", None)
    task_index = int(getattr(args, "task_index", 0))
    shared_early_decisions = int(getattr(args, "shared_early_decisions", 5))
    sample_seed = getattr(args, "sample_seed", None)
    save_images_dir = getattr(args, "save_images_dir", None)
    parse_retries = int(getattr(args, "parse_retries", 0))
    if not 0 <= parse_retries <= 4:
        raise ValueError("parse retries must be within zero to four")
    if not 0 <= shared_early_decisions <= 5:
        raise ValueError("shared early decisions must be within zero to five")
    if ceiling_plan is not None:
        record = _load_plan_instance(
            ceiling_plan, task_type=args.task_type, task_index=task_index
        )
        record["horizon_stratum"] = (
            "long"
            if record["instance"]["max_steps"] >= 23
            else ("medium" if record["instance"]["max_steps"] >= 13 else "short")
        )
    else:
        record = _load_instance(
            args.validation12_manifest, task_type=args.task_type
        )
        if task_index != 0:
            raise ValueError("validation-12 permits only task_index zero")
    instance = dict(record["instance"])
    if ocr_provider is None:
        ocr_provider = PinnedOnlineOCRProvider.load(
            backend_config_path=args.repository_root
            / "code/configs/restoration_v2_ocr_backend.json",
            backend_manifest_path=args.repository_root
            / "data/manifests/restoration_v2_ocr_backend.json",
            model_dir=args.ocr_model_dir,
        )
    if runtime is None:
        runtime = GUIOwlV21OfficialToolsRuntime(
            model_dir=args.model_dir,
            expected_snapshot_manifest=(
                args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
            ),
            device=args.device,
            target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
        )
    if sample_seed is not None:
        runtime.torch.manual_seed(int(sample_seed))
    if save_images_dir is not None:
        save_images_dir = Path(save_images_dir)
        save_images_dir.mkdir(parents=True, exist_ok=True)
    namespace_key = f"{args.arm}:{args.task_type}:{task_index}"
    if sample_seed is not None:
        namespace_key = f"{namespace_key}:{sample_seed}"
    namespace = hashlib.sha256(namespace_key.encode("utf-8")).hexdigest()[:24]
    environment = HTTPAndroidWorldEnvironment(
        base_url=args.base_url,
        instance=instance,
        observation_namespace=namespace,
        ocr_provider=ocr_provider,
        suite_seed=record.get("suite_seed", 271828),
        task_combinations=record.get("task_combinations", 2),
    )

    summary: dict[str, Any] = {
        "arm": args.arm,
        "base_url": args.base_url,
        "budget_contract": {
            "high_fidelity_history_capacity": (
                ceiling_arm_budget(args.arm)
                if args.arm in CEILING_ARMS
                else (0 if args.arm == SUMMARY_ARM else EXPLORATORY_MEMORY_BUDGET)
            ),
            "search": "at_most_budget",
        },
        "horizon_stratum": record["horizon_stratum"],
        "instance": instance,
        "partition": (
            record["split"] if ceiling_plan is not None else "validation"
        ),
        "policy": dict(runtime.metadata),
        "schema_version": "causalcache.exploratory_closed_loop_episode.v1",
        "sample_seed": sample_seed,
        "parse_retries_allowed": parse_retries,
        "shared_early_decisions": shared_early_decisions,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": [],
        "task_index": task_index,
        "task_type": args.task_type,
        "roster_sha256": sha256_file(
            ceiling_plan if ceiling_plan is not None else args.validation12_manifest
        ),
    }

    started = time.monotonic()
    history: list[LiveRichEvent] = []
    parse_attempts = 0
    parse_successes = 0
    previous_executed_action: dict[str, Any] | None = None
    termination_reason = "step_budget_exhausted"
    failure_classification = "terminal_failure"
    infrastructure_failure = False
    normal_environment_chain = False
    score_after = None
    initialized = False
    try:
        summary["environment_initialize"] = environment.initialize()
        initialized = True
        summary["score_before"] = environment.score()
        if summary["score_before"] != 0.0:
            raise ValueError("episode did not start at zero reward")
        current = environment.screenshot()
        for step_index in range(instance["max_steps"]):
            decision_step_id = len(history) + 1
            if len(history) < shared_early_decisions:
                selected: tuple[int, ...] = ()
                selection_diagnostics = {"method": EARLY_STEP_POLICY_ID}
                messages = build_shared_early_step_messages(
                    instruction=instance["goal"],
                    history_events=tuple(history),
                    current_image=current.image,
                )
            else:
                selected, selection_diagnostics = _select_arm_memory(
                    arm=args.arm, history=history, current=current
                )
                messages = build_live_gui_owl_v2_1_mixed_fidelity_messages(
                    instruction=instance["goal"],
                    history_events=[event.to_mapping() for event in history],
                    restored_event_step_ids=selected,
                    selected_post_images_by_event_step={
                        step_id: _decode_png(
                            history[step_id - 1].post_image.payload
                        )
                        for step_id in selected
                    },
                    current_image=current.image,
                )
            step: dict[str, Any] = {
                "decision_step_id": decision_step_id,
                "high_fidelity_history_image_count": len(selected),
                "selected_event_step_ids": list(selected),
                "selection_diagnostics": selection_diagnostics,
                "step_index": step_index,
                "current_image_sha256": current.image_sha256,
            }
            summary["steps"].append(step)
            if save_images_dir is not None:
                image_name = f"step{step_index:03d}.png"
                (save_images_dir / image_name).write_bytes(current.image_png)
                step["saved_image"] = image_name
            generated = None
            for parse_attempt in range(parse_retries + 1):
                parse_attempts += 1
                try:
                    generated = runtime.generate_native_action(list(messages))
                    break
                except GUIOwlV21GenerationParseError as error:
                    step.setdefault("parse_retry_errors", []).append(
                        {
                            "attempt": parse_attempt,
                            "message": error.parse_error_message,
                            "type": error.parse_error_type,
                        }
                    )
                    if parse_attempt == parse_retries:
                        step["native_output"] = error.output_text
                        step["parse_error"] = {
                            "message": error.parse_error_message,
                            "type": error.parse_error_type,
                        }
            if generated is None:
                termination_reason = "parse_error"
                failure_classification = "parse_failure"
                break
            parse_successes += 1
            canonical_action = generated.parsed_output.canonical_action
            step["native_output"] = generated.output_text
            step["canonical_action"] = canonical_action.arguments()
            step["generation"] = {
                key: generated.metadata[key]
                for key in ("prompt_tokens", "generated_tokens")
                if key in generated.metadata
            }
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
                failure_classification = "executor_failure"
                break
            previous_executed_action = effective_action
            if effective_action["action_type"] == "status":
                termination_reason = "policy_terminated"
                break
            after = environment.screenshot()
            projected = live_history_event_from_transition(
                step_id=len(history) + 1,
                action=canonical_action,
                before_image_bytes=current.image_png,
                after_image_bytes=after.image_png,
                backend_config=ocr_provider.backend_config,
                backend_config_sha256=ocr_provider.backend_config_sha256,
                before_ocr_record=current.ocr_record,
                after_ocr_record=after.ocr_record,
                ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
            )
            event = LiveRichEvent.build(
                event_step_id=len(history) + 1,
                low_fidelity_v2=projected["low_fidelity_v2"],
                post_image_png=after.image_png,
                post_ocr_tokens=projected["post_ocr_spatial_tokens"],
            )
            history.append(event)
            step["appended_event"] = {
                "event_step_id": event.event_step_id,
                "post_image_sha256": event.post_image.sha256,
                "post_ocr_token_count": len(event.post_ocr_tokens),
            }
            current = after
        score_after = environment.score()
        normal_environment_chain = True
    except Exception as error:  # noqa: BLE001
        summary["infrastructure_error"] = {
            "message": str(error),
            "type": error.__class__.__name__,
        }
        termination_reason = "infrastructure_exception"
        failure_classification = "infrastructure_failure"
        infrastructure_failure = True
    finally:
        if initialized:
            try:
                summary["tear_down"] = environment.tear_down()
            except Exception as error:  # noqa: BLE001
                summary["tear_down_error"] = str(error)
                if not infrastructure_failure:
                    infrastructure_failure = True
                    failure_classification = "infrastructure_failure"
                    normal_environment_chain = False

    official_success = (
        score_after == 1.0 and termination_reason == "policy_terminated"
    )
    if official_success:
        failure_classification = "official_success"
    summary.update(
        {
            "action_parse_attempt_count": parse_attempts,
            "action_parse_success_count": parse_successes,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "failure_classification": failure_classification,
            "infrastructure_failure": infrastructure_failure,
            "model_step_count": len(summary["steps"]),
            "normal_environment_chain": (
                normal_environment_chain and not infrastructure_failure
            ),
            "official_terminal_success": 1.0 if official_success else 0.0,
            "required_audits_complete": not infrastructure_failure,
            "score_after": score_after,
            "status": COMPLETE_STATUS,
            "termination_reason": termination_reason,
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ocr-model-dir", type=Path, required=True)
    parser.add_argument("--validation12-manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=LOCAL_ARMS, required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"episode output already exists: {args.output}")
    summary = run_episode(args)
    write_episode_output_atomic(args.output, summary)
    print(
        json.dumps(
            {
                "arm": summary["arm"],
                "task_type": summary["task_type"],
                "official_terminal_success": summary["official_terminal_success"],
                "failure_classification": summary["failure_classification"],
                "termination_reason": summary["termination_reason"],
                "model_step_count": summary["model_step_count"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
