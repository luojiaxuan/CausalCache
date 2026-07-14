"""Probe GUI-Owl native one/five-image history and token logits."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from causalcache.policy.gui_owl import (
    build_gui_owl_native_messages,
    parse_gui_owl_action,
    render_gui_owl_action,
)
from causalcache.policy.qwen_runtime import QwenPolicyRuntime, action_dict, load_dataset
from causalcache.schema import ExecutableAction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-tar", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--decision-step-id", type=int, required=True)
    parser.add_argument("--device", required=True)
    visual_group = parser.add_mutually_exclusive_group(required=True)
    visual_group.add_argument(
        "--use-model-default-visual-resolution",
        action="store_true",
    )
    visual_group.add_argument("--visual-tokens-per-image", type=int)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = QwenPolicyRuntime(
        model_dir=args.model_dir,
        device=args.device,
        visual_tokens_per_image=args.visual_tokens_per_image,
    )
    with tarfile.open(args.dataset_tar) as archive:
        manifest, image_loader = load_dataset(archive)
        trajectory = manifest["trajectory"]
        decision = next(
            item
            for item in trajectory["decisions"]
            if int(item["decision_step_id"]) == args.decision_step_id
        )
        current_index = args.decision_step_id - 1
        steps = trajectory["steps"]
        screenshots = [
            image_loader(steps[index]["observation_path"])
            for index in range(current_index + 1)
        ]
        action_outputs = []
        for index in range(current_index):
            step = steps[index]
            function_name = step["source_tool_call"]["function"]["name"]
            description = step.get("action_description") or f"Execute {function_name}"
            action_outputs.append(
                render_gui_owl_action(
                    step["source_tool_call"],
                    description=description,
                )
            )
        single_image_messages = build_gui_owl_native_messages(
            instruction=trajectory["instruction"],
            screenshots=[screenshots[-1]],
            action_outputs=[],
        )
        native_history_messages = build_gui_owl_native_messages(
            instruction=trajectory["instruction"],
            screenshots=screenshots,
            action_outputs=action_outputs,
        )
        variants = {
            "single_image": single_image_messages,
            "native_five_image_history": native_history_messages,
        }
        validated_action = ExecutableAction.from_dict(decision["validated_action"])
        logits_probes = {
            name: runtime.probe_logits(messages)
            for name, messages in variants.items()
        }
        runtime.warmup(single_image_messages)
        results = {
            name: runtime.generate(
                messages,
                max_new_tokens=args.max_new_tokens,
                validated_action=validated_action,
                action_parser=parse_gui_owl_action,
            )
            for name, messages in variants.items()
        }

    summary = {
        "schema_version": "0.1.0",
        "scope": "interface_only_not_policy_coverage",
        "model": runtime.metadata,
        "fixture": {
            "repo": manifest["dataset_repo"],
            "source_id": trajectory["source_id"],
            "decision_step_id": args.decision_step_id,
        },
        "validated_fixture_action": action_dict(validated_action),
        "visual_tokens_per_image": args.visual_tokens_per_image,
        "logits_probes": logits_probes,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
