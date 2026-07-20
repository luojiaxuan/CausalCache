#!/usr/bin/env python3
"""Recompute full-history context fit from observed image-grid token counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from causalcache.set_utility_variable_history import load_variable_history_config


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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--approximate-state-context", type=Path, required=True)
    parser.add_argument("--token-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_variable_history_config(args.config)
    reference = config["reference"]
    context_limit = int(reference["context_limit_tokens"])
    reserve = int(reference["reserved_action_tokens"])

    counts_by_trajectory: dict[str, tuple[int, ...]] = {}
    shard_ids = set()
    receipt_hashes = []
    source_revisions = set()
    for root in args.token_roots:
        for path in sorted((root / "receipts").glob("shard-*-of-256.json")):
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if receipt.get("status") != "COMPLETED_VARIABLE_HISTORY_TOKEN_SHARD":
                raise ValueError("token receipt is not complete")
            shard = int(receipt["logical_shard"])
            if shard in shard_ids:
                raise ValueError("token roots contain a duplicate logical shard")
            shard_ids.add(shard)
            source_revisions.add(receipt["source_revision"])
            receipt_hashes.append(_sha256_file(path))
            for trajectory_id, values in receipt[
                "visual_token_counts_by_trajectory"
            ].items():
                if trajectory_id in counts_by_trajectory:
                    raise ValueError("token receipts duplicate a trajectory")
                counts = tuple(values)
                if not counts or any(type(value) is not int or value <= 0 for value in counts):
                    raise ValueError("visual token counts are invalid")
                counts_by_trajectory[trajectory_id] = counts
    if shard_ids != set(range(256)) or len(source_revisions) != 1:
        raise ValueError("exact context postflight requires all 256 shards and one revision")

    rows = []
    with args.approximate_state_context.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line:
                continue
            row = json.loads(line)
            counts = counts_by_trajectory[row["trajectory_id"]]
            candidate_count = int(row["candidate_count"])
            if candidate_count >= len(counts):
                raise ValueError("state candidate count exceeds trajectory observations")
            actual_visual_tokens = sum(counts[1 : candidate_count + 1]) + counts[
                candidate_count
            ]
            exact_prompt_tokens = (
                int(row["base_template_tokens"])
                - int(row["image_count"])
                + actual_visual_tokens
            )
            rows.append(
                {
                    "actual_visual_tokens": actual_visual_tokens,
                    "candidate_count": candidate_count,
                    "exact_prompt_input_tokens": exact_prompt_tokens,
                    "exact_prompt_plus_action_reserve_tokens": exact_prompt_tokens
                    + reserve,
                    "fits": exact_prompt_tokens + reserve <= context_limit,
                    "history_bin": row["history_bin"],
                    "role": row["role"],
                    "state_id": row["state_id"],
                    "trajectory_id": row["trajectory_id"],
                }
            )
    if len(rows) != config["source"]["state_counts"]["total"]:
        raise RuntimeError("exact context postflight state count drifted")
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row["history_bin"]].append(row["exact_prompt_input_tokens"])
    fit_counts = Counter(row["fits"] for row in rows)
    failures = [row for row in rows if not row["fits"]]
    detail_payload = b"".join(
        json.dumps(row, allow_nan=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in sorted(rows, key=lambda item: item["state_id"])
    )
    _write_atomic(args.output_root / "state-context-exact.jsonl", detail_payload)
    summary = {
        "config_sha256": _sha256_file(args.config),
        "context_limit_tokens": context_limit,
        "failure_states": failures,
        "fit_state_count": fit_counts[True],
        "history_bins": {
            name: {
                "maximum_exact_prompt_input_tokens": max(values),
                "minimum_exact_prompt_input_tokens": min(values),
                "state_count": len(values),
            }
            for name, values in sorted(grouped.items())
        },
        "maximum_exact_prompt_input_tokens": max(
            row["exact_prompt_input_tokens"] for row in rows
        ),
        "maximum_exact_prompt_plus_action_reserve_tokens": max(
            row["exact_prompt_plus_action_reserve_tokens"] for row in rows
        ),
        "reserved_action_tokens": reserve,
        "schema_version": "1.0.0",
        "source_revision": next(iter(source_revisions)),
        "state_context_exact_sha256": hashlib.sha256(detail_payload).hexdigest(),
        "state_count": len(rows),
        "status": (
            "PASS_FULL_HISTORY_EXACT_GRID_CONTEXT_POSTFLIGHT"
            if fit_counts[False] == 0
            else "BLOCK_FULL_HISTORY_EXACT_GRID_CONTEXT_POSTFLIGHT"
        ),
        "token_receipt_set_sha256": hashlib.sha256(
            "".join(sorted(receipt_hashes)).encode("ascii")
        ).hexdigest(),
        "unfit_state_count": fit_counts[False],
    }
    _write_atomic(args.output_root / "summary.json", _canonical_bytes(summary))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
