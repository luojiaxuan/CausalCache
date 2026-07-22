#!/usr/bin/env python3
"""Drive assigned success-collection episodes with one shared sampled runtime."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

from causalcache.set_utility_androidworld import PinnedOnlineOCRProvider
from causalcache.policy.gui_owl_v2_1_sampling_runtime import (
    GUIOwlV21SampledToolsRuntime,
)
from scripts.run_exploratory_closed_loop_episode import (
    EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    run_episode,
    write_episode_output_atomic,
)

COLLECTION_ARM = "summary_B0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ocr-model-dir", type=Path, required=True)
    parser.add_argument("--collection-plan", type=Path, required=True)
    parser.add_argument("--shared-early-decisions", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--parse-retries", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--episode",
        action="append",
        required=True,
        help="SAMPLE_SEED:TASK_TYPE:TASK_INDEX entries executed in order",
    )
    args = parser.parse_args()

    assignments = []
    for entry in args.episode:
        parts = entry.split(":")
        if len(parts) != 3 or not parts[1]:
            raise ValueError(f"invalid collection episode assignment: {entry}")
        assignments.append((int(parts[0]), parts[1], int(parts[2])))

    ocr_provider = PinnedOnlineOCRProvider.load(
        backend_config_path=args.repository_root
        / "code/configs/restoration_v2_ocr_backend.json",
        backend_manifest_path=args.repository_root
        / "data/manifests/restoration_v2_ocr_backend.json",
        model_dir=args.ocr_model_dir,
    )
    runtime = GUIOwlV21SampledToolsRuntime(
        temperature=args.temperature,
        top_p=args.top_p,
        model_dir=args.model_dir,
        expected_snapshot_manifest=(
            args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device=args.device,
        target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    import torch

    completed = 0
    successes = 0
    for sample_seed, task_type, task_index in assignments:
        episode_name = f"s{sample_seed}-{task_type}-{task_index}"
        output = args.output_root / "episodes" / f"{episode_name}.json"
        if output.exists():
            print(json.dumps({"skip_existing": str(output)}), flush=True)
            completed += 1
            continue
        episode_args = SimpleNamespace(
            repository_root=args.repository_root,
            base_url=args.base_url,
            model_dir=args.model_dir,
            ocr_model_dir=args.ocr_model_dir,
            validation12_manifest=None,
            ceiling_plan=args.collection_plan,
            task_index=task_index,
            shared_early_decisions=args.shared_early_decisions,
            sample_seed=sample_seed,
            parse_retries=args.parse_retries,
            save_images_dir=args.output_root / "images" / episode_name,
            arm=COLLECTION_ARM,
            task_type=task_type,
            device=args.device,
            output=output,
        )
        try:
            summary = run_episode(
                episode_args, runtime=runtime, ocr_provider=ocr_provider
            )
            write_episode_output_atomic(output, summary)
            completed += 1
            successes += int(summary["official_terminal_success"])
            print(
                json.dumps(
                    {
                        "sample_seed": sample_seed,
                        "task_type": task_type,
                        "task_index": task_index,
                        "official_terminal_success": summary[
                            "official_terminal_success"
                        ],
                        "failure_classification": summary["failure_classification"],
                        "model_step_count": summary["model_step_count"],
                        "elapsed_seconds": summary["elapsed_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "sample_seed": sample_seed,
                        "task_type": task_type,
                        "task_index": task_index,
                        "worker_episode_error": traceback.format_exc(limit=6),
                    }
                ),
                flush=True,
            )
        finally:
            torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "assigned": len(assignments),
                "completed": completed,
                "successes": successes,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
