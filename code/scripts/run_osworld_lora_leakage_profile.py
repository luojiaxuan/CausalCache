#!/usr/bin/env python3
"""Run one frozen or LoRA profile over the frozen OSWorld leakage prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.osworld_gui_owl import GUIOwlOSWorldRuntime
from causalcache.osworld_lora_leakage import (
    PROFILE_SCHEMA_VERSION,
    inspect_output,
    load_config,
    load_manifest,
    prompt_contract_sha256,
    request_from_prompt,
    sha256_file,
    summarize_records,
)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    config = load_config(args.config.resolve())
    profiles = {profile["profile_id"]: profile for profile in config["profiles"]}
    if args.profile_id not in profiles:
        raise ValueError(f"unknown leakage profile: {args.profile_id}")
    profile = profiles[args.profile_id]
    expects_lora = profile["checkpoint_sha256"] is not None
    if expects_lora != (args.lora_checkpoint is not None):
        raise ValueError("profile and LoRA checkpoint presence disagree")
    if args.lora_checkpoint is not None:
        observed = sha256_file(args.lora_checkpoint.resolve())
        if observed != profile["checkpoint_sha256"]:
            raise ValueError("LoRA checkpoint SHA256 drifted")
    manifest = load_manifest(
        args.manifest.resolve(),
        expected_count=config["prompt_count"],
        expected_prompt_contract_sha256=prompt_contract_sha256(config),
    )
    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir.resolve(),
        expected_snapshot_manifest=(
            repository_root / config["policy_snapshot_manifest"]
        ),
        device=args.device,
        effective_visual_tokens_per_image=config["effective_visual_tokens_per_image"],
        max_new_tokens=config["max_new_tokens"],
    )
    if args.lora_checkpoint is not None:
        import torch

        from scripts.train_success_sft_lora import inject_lora, load_lora_state_dict

        wrapped = inject_lora(
            runtime.model,
            rank=profile["lora_rank"],
            alpha=profile["lora_alpha"],
            target_modules=tuple(profile["target_modules"]),
            torch=torch,
        )
        load_lora_state_dict(
            wrapped, torch.load(args.lora_checkpoint.resolve(), map_location="cpu")
        )
        runtime.metadata = {
            **runtime.metadata,
            "lora_checkpoint": str(args.lora_checkpoint.resolve()),
            "lora_checkpoint_sha256": profile["checkpoint_sha256"],
            "lora_rank": profile["lora_rank"],
            "lora_alpha": profile["lora_alpha"],
            "lora_module_count": len(wrapped),
        }
    profile_root = args.output_root.resolve() / args.profile_id
    records = []
    for prompt in manifest["prompts"]:
        output = profile_root / "records" / f"{prompt['prompt_id']}.json"
        if output.exists():
            record = json.loads(output.read_text(encoding="utf-8"))
        else:
            request = request_from_prompt(prompt)
            output_text, generation = runtime.generate_raw(request)
            record = {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "profile_id": args.profile_id,
                "prompt_id": prompt["prompt_id"],
                "domain": prompt["domain"],
                "task_id": prompt["task_id"],
                "input_mode": prompt["input_mode"],
                "output_text": output_text,
                "inspection": inspect_output(
                    output_text, screen_size=tuple(prompt["screen_size"])
                ),
                "generation": generation,
            }
            _atomic_json(output, record)
        records.append(record)
        print(
            json.dumps(
                {
                    "profile_id": args.profile_id,
                    "prompt_id": prompt["prompt_id"],
                    "parser_valid": record["inspection"]["parser_valid"],
                    "hard_leakage": record["inspection"]["hard_leakage"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "status": "COMPLETE_OSWORLD_LORA_LEAKAGE_PROFILE",
        "profile": profile,
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "runtime": runtime.metadata,
        **summarize_records(records),
    }
    _atomic_json(profile_root / "summary.json", summary)


if __name__ == "__main__":
    main()
