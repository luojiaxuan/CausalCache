#!/usr/bin/env python3
"""Extract full-history V2 HGKV readout plus frozen temporal features."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from causalcache.hgkv_selector_v2 import (
    HGKV_READOUT_DIM,
    V2_FEATURE_DIM,
    temporal_features,
)
from scripts.extract_hgkv_readout_features import (
    build_layer_handles,
    extract_sample_features,
    load_done_keys,
    resolve_head_dim,
    sample_resume_key,
)


def _sample_paths(root: Path, pattern: str) -> list[Path]:
    return sorted(Path(value) for value in glob.glob(str(root / pattern)))


def iter_samples(
    paths: list[Path],
    *,
    shard_index: int,
    shard_count: int,
    shard_by_file: bool,
) -> Iterable[dict[str, Any]]:
    global_index = 0
    for file_index, path in enumerate(paths):
        take_file = not shard_by_file or file_index % shard_count == shard_index
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            take = take_file if shard_by_file else global_index % shard_count == shard_index
            global_index += 1
            if take:
                yield json.loads(line)


def append_temporal_features(
    sample: dict[str, Any], readout: list[float]
) -> list[float]:
    if len(readout) != HGKV_READOUT_DIM:
        raise ValueError(f"HGKV readout dim {len(readout)} != {HGKV_READOUT_DIM}")
    event_step_id = int(sample["singleton_event_step_id"])
    expected = temporal_features(
        event_step_id=event_step_id,
        decision_step=int(sample["decision_step_id"]),
        history_length=int(sample["history_length"]),
        candidate_event_step_ids=sample["candidate_event_step_ids"],
    )
    declared = sample.get("temporal_features")
    if declared is not None:
        if len(declared) != len(expected) or any(
            abs(float(left) - right) > 1e-12
            for left, right in zip(declared, expected)
        ):
            raise ValueError(
                f"{sample['pair_group']} temporal feature declaration drifted"
            )
    result = [*readout, *expected]
    if len(result) != V2_FEATURE_DIM:
        raise AssertionError("V2 feature concatenation produced wrong dimension")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--samples-glob", default="samples-shard*.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--adapter-layer-count", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-by-file", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must be in [0, shard count)")

    from causalcache.policy.gui_owl_v2_1_runtime import (
        GUIOwlV21OfficialToolsRuntime,
    )
    from causalcache.policy.history_gated_lora import (
        inject_history_gated_kv,
        load_history_gated_state_dict,
    )
    from scripts.run_exploratory_closed_loop_episode import (
        EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    from scripts.train_success_sft_lora import (
        encode_sample,
        history_sample_context,
    )

    paths = _sample_paths(args.dataset_root, args.samples_glob)
    if not paths:
        parser.error("--samples-glob matched no files")
    runtime = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=(
            args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device=args.device,
        target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    runtime.model.eval()
    merge_size = int(runtime.processor.image_processor.merge_size)
    wrapped = inject_history_gated_kv(
        runtime.model,
        layer_count=args.adapter_layer_count,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    load_history_gated_state_dict(
        wrapped, torch.load(args.lora_checkpoint, map_location="cpu")
    )
    layers = build_layer_handles(runtime.model, wrapped)
    head_dim = resolve_head_dim(runtime.model)
    n_q_heads = int(layers[0].q_proj.weight.shape[0]) // head_dim
    readout_dim = len(layers) * (head_dim + n_q_heads)
    if readout_dim != HGKV_READOUT_DIM:
        raise RuntimeError(
            f"runtime readout dim {readout_dim} != frozen {HGKV_READOUT_DIM}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_keys = load_done_keys(args.output)
    written = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for sample in iter_samples(
            paths,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            shard_by_file=args.shard_by_file,
        ):
            if sample.get("variant") != "singleton":
                continue
            key = sample_resume_key(sample)
            if key in done_keys:
                continue
            encoded = encode_sample(
                runtime, sample, dataset_root=args.dataset_root, torch=torch
            )
            if encoded is None:
                raise RuntimeError(f"failed to encode V2 singleton {key}")
            context = history_sample_context(encoded, sample, merge_size=merge_size)
            if context is None:
                raise RuntimeError(f"V2 singleton {key} has no history context")
            labels = encoded.pop("labels")
            target_length = int((labels != -100).sum())
            query_pos = int(encoded["input_ids"].shape[1]) - target_length - 1
            readout, scalars = extract_sample_features(
                runtime.model,
                encoded,
                history_mask=context.history_token_mask,
                query_pos=query_pos,
                layers=layers,
                head_dim=head_dim,
            )
            feature = append_temporal_features(sample, readout)
            row = {
                "schema_version": "causalcache.hgkv_selector_v2.feature.v1",
                "episode": sample["episode"],
                "step_index": sample["step_index"],
                "pair_group": sample["pair_group"],
                "variant": sample["variant"],
                "singleton_event_step_id": key[1],
                "restored_event_step_ids": sample["memory_config"][
                    "restored_event_step_ids"
                ],
                "candidate_event_step_ids": sample["candidate_event_step_ids"],
                "history_length": sample["history_length"],
                "layer_indices": [layer.layer_index for layer in layers],
                "hgkv_readout_dim": HGKV_READOUT_DIM,
                "temporal_feature_dim": V2_FEATURE_DIM - HGKV_READOUT_DIM,
                "feature_dim": V2_FEATURE_DIM,
                "feature": feature,
                "scalars": scalars,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            done_keys.add(key)
            written += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "feature_dim": V2_FEATURE_DIM,
                "written": written,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
