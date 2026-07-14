"""Probe OpenCUA logits and one-decision mixed-fidelity behavior."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from causalcache.policy.open_cua import build_open_cua_messages
from causalcache.policy.open_cua_runtime import OpenCUAPolicyRuntime
from causalcache.policy.qwen_runtime import action_dict, load_dataset
from causalcache.schema import ExecutableAction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-tar", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--decision-step-id", type=int, required=True)
    parser.add_argument("--mixed-restored-step-id", type=int, action="append", default=[])
    parser.add_argument("--device", required=True)
    parser.add_argument("--visual-tokens-per-image", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = OpenCUAPolicyRuntime(
        model_dir=args.model_dir,
        device=args.device,
        visual_tokens_per_image=args.visual_tokens_per_image,
    )
    with tarfile.open(args.dataset_tar) as archive:
        manifest, image_loader = load_dataset(archive)
        decision = next(
            item
            for item in manifest["trajectory"]["decisions"]
            if int(item["decision_step_id"]) == args.decision_step_id
        )
        history_ids = [int(step_id) for step_id in decision["history_event_step_ids"]]
        validated_action = ExecutableAction.from_dict(decision["validated_action"])
        variants = {
            "summary_only": [],
            "mixed_fidelity": args.mixed_restored_step_id,
            "full_history": history_ids,
        }
        summary_messages = build_open_cua_messages(
            manifest,
            decision_step_id=args.decision_step_id,
            restored_event_step_ids=[],
            image_loader=image_loader,
        )
        logits_probe = runtime.probe_logits(summary_messages)
        runtime.warmup(summary_messages)
        results = {}
        for variant_name, restored_ids in variants.items():
            messages = build_open_cua_messages(
                manifest,
                decision_step_id=args.decision_step_id,
                restored_event_step_ids=restored_ids,
                image_loader=image_loader,
            )
            results[variant_name] = runtime.generate(
                messages,
                max_new_tokens=args.max_new_tokens,
                validated_action=validated_action,
            ) | {"restored_event_step_ids": list(restored_ids)}

    summary = {
        "schema_version": "0.1.0",
        "model": runtime.metadata,
        "dataset": {
            "repo": manifest["dataset_repo"],
            "source_id": manifest["trajectory"]["source_id"],
        },
        "decision_step_id": args.decision_step_id,
        "validated_action": action_dict(validated_action),
        "visual_tokens_per_image": args.visual_tokens_per_image,
        "logits_probe": logits_probe,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
