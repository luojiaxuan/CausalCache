"""Run the pre-registered ShowUI full-history coverage gate."""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import defaultdict
from pathlib import Path

from causalcache.policy.qwen_runtime import QwenPolicyRuntime, action_dict, load_dataset
from causalcache.policy.showui import build_showui_messages, parse_showui_action
from causalcache.schema import ExecutableAction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-tar", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--coverage-gate", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--visual-tokens-per-image", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _coverage(records: list[dict]) -> dict:
    matches = sum(bool(record["executable_match"]) for record in records)
    parsed = sum(record["parse_error"] is None for record in records)
    return {
        "decisions": len(records),
        "parsed": parsed,
        "matches": matches,
        "coverage": matches / len(records) if records else 0.0,
    }


def main() -> None:
    args = parse_args()
    gate = json.loads(args.coverage_gate.read_text(encoding="utf-8"))
    runtime = QwenPolicyRuntime(
        model_dir=args.model_dir,
        device=args.device,
        visual_tokens_per_image=args.visual_tokens_per_image,
    )
    with tarfile.open(args.dataset_tar) as archive:
        manifest, image_loader = load_dataset(archive)
        decisions = sorted(
            manifest["trajectory"]["decisions"],
            key=lambda decision: int(decision["decision_step_id"]),
        )
        warmup_messages = build_showui_messages(
            manifest,
            decision_step_id=int(decisions[0]["decision_step_id"]),
            restored_event_step_ids=[],
            image_loader=image_loader,
        )
        runtime.warmup(warmup_messages)
        records = []
        for decision in decisions:
            step_id = int(decision["decision_step_id"])
            history_ids = [int(value) for value in decision["history_event_step_ids"]]
            validated_action = ExecutableAction.from_dict(decision["validated_action"])
            messages = build_showui_messages(
                manifest,
                decision_step_id=step_id,
                restored_event_step_ids=history_ids,
                image_loader=image_loader,
            )
            result = runtime.generate(
                messages,
                max_new_tokens=args.max_new_tokens,
                validated_action=validated_action,
                action_parser=parse_showui_action,
            )
            records.append(
                result
                | {
                    "decision_step_id": step_id,
                    "history_events": len(history_ids),
                    "validated_action": action_dict(validated_action),
                }
            )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["validated_action"]["action_type"]].append(record)
    by_action_type = {
        action_type: _coverage(action_records)
        for action_type, action_records in sorted(grouped.items())
    }
    overall = _coverage(records)
    required_types = gate["present_action_types"]
    action_type_gate = all(
        by_action_type.get(action_type, {}).get("matches", 0) >= 1
        for action_type in required_types
    )
    overall_gate = overall["coverage"] >= float(gate["minimum_overall_coverage"])
    summary = {
        "schema_version": "0.1.0",
        "model": runtime.metadata,
        "dataset": {
            "repo": manifest["dataset_repo"],
            "revision": gate["pilot_revision"],
            "source_id": manifest["trajectory"]["source_id"],
            "apps": manifest["trajectory"]["apps"],
        },
        "validation_mode": gate["validation_mode"],
        "visual_tokens_per_image": args.visual_tokens_per_image,
        "overall": overall,
        "by_action_type": by_action_type,
        "gate": {
            "configuration": gate,
            "overall_coverage_pass": overall_gate,
            "action_type_coverage_pass": action_type_gate,
            "passed": overall_gate and action_type_gate,
        },
        "decisions": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
