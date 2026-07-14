"""Run summary-only, mixed-fidelity, and full-history Qwen policy forwards."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path
from typing import Any

from causalcache.policy import build_policy_messages, parse_policy_action
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


def _action_dict(action: ExecutableAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type.value,
        "target": action.target,
        "text_argument": action.text_argument,
        "text_case_sensitive": action.text_case_sensitive,
    }


def _load_dataset(archive: tarfile.TarFile) -> tuple[dict[str, Any], Any]:
    from PIL import Image

    manifest_file = archive.extractfile("manifest.json")
    if manifest_file is None:
        raise ValueError("dataset tar does not contain manifest.json")
    manifest = json.load(manifest_file)

    def load_image(relative_path: str) -> Any:
        member = archive.extractfile(relative_path)
        if member is None:
            raise ValueError(f"missing image: {relative_path}")
        return Image.open(io.BytesIO(member.read())).convert("RGB")

    return manifest, load_image


def main() -> None:
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    if not args.device.startswith("cuda:"):
        raise ValueError("policy smoke requires an explicit cuda device")
    pixels_per_image = args.visual_tokens_per_image * 28 * 28
    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        min_pixels=pixels_per_image,
        max_pixels=pixels_per_image,
        local_files_only=True,
    )
    model_load_start = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(args.device).eval()
    model_load_seconds = time.perf_counter() - model_load_start

    with tarfile.open(args.dataset_tar) as archive:
        manifest, image_loader = _load_dataset(archive)
        decision = next(
            item
            for item in manifest["trajectory"]["decisions"]
            if int(item["decision_step_id"]) == args.decision_step_id
        )
        history_ids = [int(step_id) for step_id in decision["history_event_step_ids"]]
        variants = {
            "summary_only": [],
            "mixed_fidelity": args.mixed_restored_step_id,
            "full_history": history_ids,
        }
        validated_action = ExecutableAction.from_dict(decision["validated_action"])
        warmup_messages = build_policy_messages(
            manifest,
            decision_step_id=args.decision_step_id,
            restored_event_step_ids=[],
            image_loader=image_loader,
        )
        warmup_inputs = processor.apply_chat_template(
            warmup_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(args.device)
        with torch.inference_mode():
            model.generate(**warmup_inputs, do_sample=False, max_new_tokens=1)
        torch.cuda.synchronize(args.device)

        results = {}
        for variant_name, restored_ids in variants.items():
            messages = build_policy_messages(
                manifest,
                decision_step_id=args.decision_step_id,
                restored_event_step_ids=restored_ids,
                image_loader=image_loader,
            )
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            torch.cuda.synchronize(args.device)
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )
            torch.cuda.synchronize(args.device)
            latency_seconds = time.perf_counter() - start
            new_tokens = generated[:, inputs.input_ids.shape[1] :]
            output_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            parsed_action = None
            parse_error = None
            executable_match = False
            try:
                parsed_action = parse_policy_action(output_text)
                executable_match = parsed_action.executable_match(validated_action)
            except (KeyError, TypeError, ValueError) as error:
                parse_error = str(error)
            results[variant_name] = {
                "restored_event_step_ids": list(restored_ids),
                "input_tokens": int(inputs.input_ids.shape[1]),
                "generated_tokens": int(new_tokens.shape[1]),
                "latency_seconds": latency_seconds,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
                "output_text": output_text,
                "parsed_action": _action_dict(parsed_action) if parsed_action is not None else None,
                "parse_error": parse_error,
                "executable_match": executable_match,
            }

    summary = {
        "schema_version": "0.1.0",
        "model": {
            "snapshot": json.loads((args.model_dir / ".snapshot.json").read_text(encoding="utf-8")),
            "dtype": "bfloat16",
            "device": args.device,
            "load_seconds": model_load_seconds,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "processor_class": processor.__class__.__name__,
            "model_class": model.__class__.__name__,
        },
        "dataset": {
            "repo": manifest["dataset_repo"],
            "source_id": manifest["trajectory"]["source_id"],
        },
        "decision_step_id": args.decision_step_id,
        "validated_action": _action_dict(validated_action),
        "visual_tokens_per_image": args.visual_tokens_per_image,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
