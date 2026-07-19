#!/usr/bin/env python3
"""Census full-history prompt lengths without executing the frozen policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from causalcache.set_utility_variable_history import (
    history_bin,
    load_variable_history_config,
)
from causalcache.set_utility_variable_history_inputs import (
    build_variable_history_messages,
    build_variable_history_queries_from_source_row,
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer-batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.tokenizer_batch_size <= 0:
        raise ValueError("tokenizer batch size must be positive")

    try:
        from pyarrow import parquet as pq
        from transformers import AutoProcessor
    except ModuleNotFoundError as error:
        raise RuntimeError("context census requires pyarrow and transformers") from error
    config = load_variable_history_config(args.config)
    reference = config["reference"]
    visual_tokens = int(reference["target_effective_visual_tokens_per_image"])
    context_limit = int(reference["context_limit_tokens"])
    action_reserve = int(reference["reserved_action_tokens"])
    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        min_pixels=visual_tokens * (14 * 2) ** 2,
        max_pixels=visual_tokens * (14 * 2) ** 2,
        local_files_only=True,
    )
    source_manifest = json.loads(
        (args.source_root / "manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("status") != "COMPLETED_VARIABLE_HISTORY_SOURCE":
        raise ValueError("variable-history source is not complete")

    rendered_rows: list[tuple[dict[str, Any], str]] = []
    for shard in source_manifest["shards"]:
        if shard["row_count"] == 0:
            continue
        path = (
            args.source_root
            / "trajectory-shards"
            / f"shard-{shard['logical_shard']:03d}-of-256.parquet"
        )
        table = pq.read_table(
            path,
            columns=[
                "decision_count",
                "history_events_json",
                "role",
                "source_id",
                "task_instruction",
            ],
        )
        for row in table.to_pylist():
            for query in build_variable_history_queries_from_source_row(row):
                messages = build_variable_history_messages(
                    query,
                    query.candidate_event_step_ids,
                    image_bytes_loader=lambda _: b"placeholder",
                    image_decoder=lambda _: "placeholder",
                )
                rendered = processor.apply_chat_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if not isinstance(rendered, str):
                    raise TypeError("chat template did not render one prompt string")
                image_count = len(query.candidate_event_step_ids) + 1
                if rendered.count("<|image_pad|>") != image_count:
                    raise ValueError("rendered image placeholder count drifted")
                rendered_rows.append(
                    (
                        {
                            "candidate_count": len(query.candidate_event_step_ids),
                            "history_bin": history_bin(
                                len(query.candidate_event_step_ids)
                            ),
                            "image_count": image_count,
                            "role": query.role,
                            "state_id": query.state_id,
                            "trajectory_id": query.trajectory_id,
                        },
                        rendered,
                    )
                )

    rows: list[dict[str, Any]] = []
    for start in range(0, len(rendered_rows), args.tokenizer_batch_size):
        batch = rendered_rows[start : start + args.tokenizer_batch_size]
        encoded = processor.tokenizer(
            [rendered for _, rendered in batch],
            add_special_tokens=False,
            padding=False,
        )["input_ids"]
        for (metadata, _), token_ids in zip(batch, encoded, strict=True):
            base_tokens = len(token_ids)
            prompt_tokens = base_tokens + metadata["image_count"] * (
                visual_tokens - 1
            )
            rows.append(
                {
                    **metadata,
                    "base_template_tokens": base_tokens,
                    "fits": prompt_tokens + action_reserve <= context_limit,
                    "prompt_input_tokens": prompt_tokens,
                    "prompt_plus_action_reserve_tokens": prompt_tokens
                    + action_reserve,
                }
            )
    if len(rows) != config["source"]["state_counts"]["total"]:
        raise RuntimeError("context census state count differs from the contract")

    counts = Counter(row["fits"] for row in rows)
    by_bin: dict[str, dict[str, int]] = {}
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row["history_bin"]].append(row["prompt_input_tokens"])
    for name, values in sorted(grouped.items()):
        by_bin[name] = {
            "maximum_prompt_input_tokens": max(values),
            "minimum_prompt_input_tokens": min(values),
            "state_count": len(values),
        }
    state_path = args.output_root / "state-context.jsonl"
    state_payload = b"".join(
        json.dumps(row, allow_nan=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in sorted(rows, key=lambda item: item["state_id"])
    )
    _write_atomic(state_path, state_payload)
    summary = {
        "config_sha256": _sha256_file(args.config),
        "context_limit_tokens": context_limit,
        "fit_state_count": counts[True],
        "history_bins": by_bin,
        "maximum_prompt_input_tokens": max(
            row["prompt_input_tokens"] for row in rows
        ),
        "maximum_prompt_plus_action_reserve_tokens": max(
            row["prompt_plus_action_reserve_tokens"] for row in rows
        ),
        "reserved_action_tokens": action_reserve,
        "schema_version": "1.0.0",
        "source_manifest_sha256": _sha256_file(args.source_root / "manifest.json"),
        "state_context_sha256": hashlib.sha256(state_payload).hexdigest(),
        "state_count": len(rows),
        "status": (
            "PASS_FULL_HISTORY_CONTEXT_CENSUS"
            if counts[False] == 0
            else "BLOCK_FULL_HISTORY_CONTEXT_CENSUS"
        ),
        "target_effective_visual_tokens_per_image": visual_tokens,
        "unfit_state_count": counts[False],
    }
    _write_atomic(args.output_root / "summary.json", _canonical_bytes(summary))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
