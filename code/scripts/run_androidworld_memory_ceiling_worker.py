#!/usr/bin/env python3
"""Drive assigned AndroidWorld memory-ceiling episodes with one shared runtime."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

from causalcache.set_utility_androidworld import PinnedOnlineOCRProvider
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
from scripts.run_exploratory_closed_loop_episode import (
    CEILING_ARMS,
    EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    run_episode,
    write_episode_output_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ocr-model-dir", type=Path, required=True)
    parser.add_argument("--ceiling-plan", type=Path, required=True)
    parser.add_argument("--shared-early-decisions", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--lora-checkpoint", type=Path, default=None)
    # note (luojiaxuan): rank/alpha 缺省按 adapter 类型解析——full_policy_lora
    # 保持既有 16/32,history_gated_kv 用训练契约的 8/16;显式传参不受影响。
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument(
        "--adapter-type",
        choices=("full_policy_lora", "history_gated_kv"),
        default="full_policy_lora",
    )
    parser.add_argument("--adapter-layer-count", type=int, default=8)
    parser.add_argument("--parse-retries", type=int, default=0)
    parser.add_argument("--retry-temperature", type=float, default=0.7)
    parser.add_argument("--retry-top-p", type=float, default=0.95)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--episode",
        action="append",
        required=True,
        help="ARM:TASK_TYPE:TASK_INDEX entries executed in order",
    )
    args = parser.parse_args()

    history_mode = args.adapter_type == "history_gated_kv"
    lora_rank = (
        args.lora_rank
        if args.lora_rank is not None
        else (8 if history_mode else 16)
    )
    lora_alpha = (
        args.lora_alpha
        if args.lora_alpha is not None
        else (16 if history_mode else 32)
    )
    if history_mode and args.lora_checkpoint is None:
        raise ValueError(
            "--adapter-type history_gated_kv requires --lora-checkpoint"
        )

    assignments = []
    for entry in args.episode:
        parts = entry.split(":")
        if len(parts) != 3 or parts[0] not in CEILING_ARMS or not parts[1]:
            raise ValueError(f"invalid ceiling episode assignment: {entry}")
        assignments.append((parts[0], parts[1], int(parts[2])))

    ocr_provider = PinnedOnlineOCRProvider.load(
        backend_config_path=args.repository_root
        / "code/configs/restoration_v2_ocr_backend.json",
        backend_manifest_path=args.repository_root
        / "data/manifests/restoration_v2_ocr_backend.json",
        model_dir=args.ocr_model_dir,
    )
    if args.parse_retries > 0:
        from causalcache.policy.gui_owl_v2_1_sampling_runtime import (
            GUIOwlV21GreedyWithSampledRetryRuntime,
        )

        runtime_class = GUIOwlV21GreedyWithSampledRetryRuntime
        runtime_kwargs = {
            "temperature": args.retry_temperature,
            "top_p": args.retry_top_p,
        }
    else:
        runtime_class = GUIOwlV21OfficialToolsRuntime
        runtime_kwargs = {}
    runtime = runtime_class(
        **runtime_kwargs,
        model_dir=args.model_dir,
        expected_snapshot_manifest=(
            args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device=args.device,
        target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    import torch

    if args.lora_checkpoint is not None:
        import hashlib

        if history_mode:
            from causalcache.policy.history_gated_lora import (
                inject_history_gated_kv,
                load_history_gated_state_dict,
            )

            wrapped = inject_history_gated_kv(
                runtime.model,
                layer_count=args.adapter_layer_count,
                rank=lora_rank,
                alpha=lora_alpha,
            )
            load_history_gated_state_dict(
                wrapped, torch.load(args.lora_checkpoint, map_location="cpu")
            )
            runtime.enable_history_gated_adapter()
        else:
            from scripts.train_success_sft_lora import (
                inject_lora,
                load_lora_state_dict,
            )

            wrapped = inject_lora(
                runtime.model,
                rank=lora_rank,
                alpha=lora_alpha,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                torch=torch,
            )
            load_lora_state_dict(
                wrapped, torch.load(args.lora_checkpoint, map_location="cpu")
            )
        runtime.metadata = {
            **runtime.metadata,
            "lora_checkpoint": str(args.lora_checkpoint),
            "lora_checkpoint_sha256": hashlib.sha256(
                args.lora_checkpoint.read_bytes()
            ).hexdigest(),
            "lora_module_count": len(wrapped),
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
        }
        if history_mode:
            runtime.metadata = {
                **runtime.metadata,
                "adapter_type": args.adapter_type,
                "adapter_layer_count": args.adapter_layer_count,
                "history_adapter_merge_size": runtime.history_adapter_merge_size,
            }
            print(
                json.dumps(
                    {
                        "adapter_type": args.adapter_type,
                        "lora_modules": len(wrapped),
                    }
                ),
                flush=True,
            )
        else:
            print(json.dumps({"lora_modules": len(wrapped)}), flush=True)

    completed = 0
    for arm, task_type, task_index in assignments:
        output = args.output_root / f"{arm}-{task_type}-{task_index}.json"
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
            ceiling_plan=args.ceiling_plan,
            task_index=task_index,
            shared_early_decisions=args.shared_early_decisions,
            parse_retries=args.parse_retries,
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
                        "arm": arm,
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
            {"assigned": len(assignments), "completed": completed},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
