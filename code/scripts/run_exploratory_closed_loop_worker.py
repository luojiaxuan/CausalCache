#!/usr/bin/env python3
"""Drive assigned exploratory validation-12 episodes with one shared runtime."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

from causalcache.set_utility_androidworld import PinnedOnlineOCRProvider
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
from scripts.run_exploratory_closed_loop_episode import (
    EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    LOCAL_ARMS,
    run_episode,
    write_episode_output_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ocr-model-dir", type=Path, required=True)
    parser.add_argument("--validation12-manifest", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--episode",
        action="append",
        required=True,
        help="ARM:TASK_TYPE entries executed in order",
    )
    args = parser.parse_args()

    assignments = []
    for entry in args.episode:
        arm, separator, task_type = entry.partition(":")
        if not separator or arm not in LOCAL_ARMS or not task_type:
            raise ValueError(f"invalid episode assignment: {entry}")
        assignments.append((arm, task_type))

    ocr_provider = PinnedOnlineOCRProvider.load(
        backend_config_path=args.repository_root
        / "code/configs/restoration_v2_ocr_backend.json",
        backend_manifest_path=args.repository_root
        / "data/manifests/restoration_v2_ocr_backend.json",
        model_dir=args.ocr_model_dir,
    )
    runtime = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=(
            args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device=args.device,
        target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )

    completed = 0
    for arm, task_type in assignments:
        output = args.output_root / f"{arm}-{task_type}-0.json"
        if output.exists():
            print(json.dumps({"skip_existing": str(output)}), flush=True)
            completed += 1
            continue
        episode_args = SimpleNamespace(
            repository_root=args.repository_root,
            base_url=args.base_url,
            model_dir=args.model_dir,
            ocr_model_dir=args.ocr_model_dir,
            validation12_manifest=args.validation12_manifest,
            arm=arm,
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
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "task_type": task_type,
                        "official_terminal_success": summary[
                            "official_terminal_success"
                        ],
                        "failure_classification": summary["failure_classification"],
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
                        "arm": arm,
                        "task_type": task_type,
                        "worker_episode_error": traceback.format_exc(limit=6),
                    }
                ),
                flush=True,
            )
    print(
        json.dumps(
            {"assigned": len(assignments), "completed": completed},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
