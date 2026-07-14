"""Measure full-history executable-match coverage on a trajectory pilot."""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import defaultdict
from pathlib import Path

from causalcache.policy import build_policy_messages
from causalcache.policy.qwen_runtime import QwenPolicyRuntime, action_dict, load_dataset
from causalcache.schema import ExecutableAction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-tar", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
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
        warmup_messages = build_policy_messages(
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
            messages = build_policy_messages(
                manifest,
                decision_step_id=step_id,
                restored_event_step_ids=history_ids,
                image_loader=image_loader,
            )
            result = runtime.generate(
                messages,
                max_new_tokens=args.max_new_tokens,
                validated_action=validated_action,
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
    summary = {
        "schema_version": "0.1.0",
        "model": runtime.metadata,
        "dataset": {
            "repo": manifest["dataset_repo"],
            "source_id": manifest["trajectory"]["source_id"],
            "apps": manifest["trajectory"]["apps"],
            "terminal_status": manifest["trajectory"]["terminal_status"],
        },
        "validation_mode": "full_history_executable_match",
        "visual_tokens_per_image": args.visual_tokens_per_image,
        "overall": _coverage(records),
        "by_action_type": {
            action_type: _coverage(action_records)
            for action_type, action_records in sorted(grouped.items())
        },
        "by_history_events": {
            str(record["history_events"]): {
                "matches": int(record["executable_match"]),
                "coverage": float(record["executable_match"]),
            }
            for record in records
        },
        "decisions": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
