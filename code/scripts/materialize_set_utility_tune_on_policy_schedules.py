#!/usr/bin/env python3
"""Build tune-only restoration schedules from frozen learned selectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_inference import (
    BUDGETS,
    canonical_json_bytes,
    recent_budget_selections,
)
from causalcache.set_utility_heldout_evaluation import sha256_file


SCHEDULE_STATUS = "COMPLETED_SET_UTILITY_TUNE_ON_POLICY_SCHEDULE_SHARD"
SUMMARY_STATUS = "COMPLETED_SET_UTILITY_TUNE_ON_POLICY_SCHEDULES"


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _selection_argument(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if (
        not separator
        or re.fullmatch(r"[a-z0-9_]+", name) is None
        or not raw_path
    ):
        raise argparse.ArgumentTypeError("selection must be name=path")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--selection", type=_selection_argument, action="append", required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("tune on-policy schedule output already exists")
    selection_names = [name for name, _ in args.selection]
    if len(selection_names) != len(set(selection_names)) or len(selection_names) < 2:
        raise ValueError("tune schedules require at least two unique learned selectors")

    input_manifest_path = args.input_root / "manifest.json"
    input_manifest = _read_json(input_manifest_path)
    if input_manifest.get("evaluation_labels_included") is not False:
        raise ValueError("tune schedule input crossed the evaluation firewall")
    states = {
        row["state_id"]: row
        for line in (args.input_root / input_manifest["states_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line
        for row in (json.loads(line),)
        if row["role"] == "tune"
    }
    if not states:
        raise ValueError("tune schedule input has no tune states")
    selections = {}
    binding = None
    for name, path in args.selection:
        payload = _read_json(path)
        if payload.get("status") != "COMPLETED_SET_UTILITY_TUNE_SELECTIONS":
            raise ValueError("a tune selection payload is incomplete")
        current_binding = (
            payload["cache_content_sha256"],
            payload["config_sha256"],
            payload["input_content_sha256"],
        )
        if binding is None:
            binding = current_binding
        elif current_binding != binding:
            raise ValueError("tune selection bindings differ")
        records = {row["state_id"]: row for row in payload["records"]}
        if set(records) != set(states):
            raise ValueError("tune selection state inventory drifted")
        selections[name] = (payload, records)
    if binding is None or binding[2] != input_manifest["content_sha256"]:
        raise ValueError("tune selections do not bind the input snapshot")

    by_shard = {index: [] for index in range(256)}
    source_counts = Counter()
    total_coalitions = 0
    for state_id, state in sorted(states.items()):
        events = tuple(state["candidate_event_step_ids"])
        coalitions = [()]
        sources = ["anchor_empty"]

        def add(values: list[int], source: str) -> None:
            coalition = tuple(sorted(values))
            if not set(coalition).issubset(events):
                raise ValueError("tune selector coalition escaped its candidate universe")
            if coalition not in coalitions:
                coalitions.append(coalition)
                sources.append(source)
                source_counts[source] += 1

        expected_recent = recent_budget_selections(events)
        for name in sorted(selections):
            record = selections[name][1][state_id]
            if (
                tuple(record["candidate_event_ids"]) != events
                or record["recent"] != expected_recent
                or record["logical_shard"] != state["logical_shard"]
                or record["trajectory_id"] != state["trajectory_id"]
            ):
                raise ValueError("tune selection state identity drifted")
            for budget in BUDGETS:
                add(record["learned"][str(budget)], f"learned_{name}_b{budget}")
        for budget in BUDGETS:
            add(expected_recent[str(budget)], f"recent_b{budget}")
        add(list(events), "anchor_full")
        source_counts["anchor_empty"] += 1
        total_coalitions += len(coalitions)
        by_shard[state["logical_shard"]].append(
            {
                "candidate_event_ids": list(events),
                "coalitions": [
                    {"event_ids": list(coalition), "source": source}
                    for coalition, source in zip(coalitions, sources, strict=True)
                ],
                "exact": False,
                "logical_shard": state["logical_shard"],
                "role": "tune",
                "state_id": state_id,
                "trajectory_id": state["trajectory_id"],
            }
        )

    selection_bindings = {
        name: {
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "content_sha256": payload["content_sha256"],
            "variant": payload["variant"],
        }
        for name, (payload, _) in sorted(selections.items())
    }
    receipts = []
    for logical_shard in range(256):
        rows = sorted(by_shard[logical_shard], key=lambda row: row["state_id"])
        schedule_payload = b"".join(
            canonical_json_bytes(row) + b"\n" for row in rows
        )
        relative = f"schedule-shards/shard-{logical_shard:03d}-of-256.jsonl"
        _write_atomic(args.output_root / relative, schedule_payload)
        receipt = {
            "coalition_count": sum(len(row["coalitions"]) for row in rows),
            "identity_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {
                        "input_manifest_sha256": sha256_file(input_manifest_path),
                        "logical_shard": logical_shard,
                        "selection_bindings": selection_bindings,
                    }
                )
            ).hexdigest(),
            "logical_shard": logical_shard,
            "role_state_counts": {"tune": len(rows)} if rows else {},
            "schedule_byte_count": len(schedule_payload),
            "schedule_sha256": hashlib.sha256(schedule_payload).hexdigest(),
            "state_count": len(rows),
            "status": SCHEDULE_STATUS,
        }
        _write_atomic(
            args.output_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-256.json",
            canonical_json_bytes(receipt) + b"\n",
        )
        receipts.append(receipt)
    summary = {
        "coalition_count": total_coalitions,
        "input_content_sha256": input_manifest["content_sha256"],
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "schema_version": "1.0.0",
        "selection_bindings": selection_bindings,
        "source_counts": dict(sorted(source_counts.items())),
        "state_count": len(states),
        "status": SUMMARY_STATUS,
    }
    summary["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(summary)
    ).hexdigest()
    _write_atomic(
        args.output_root / "manifest.json",
        canonical_json_bytes(summary) + b"\n",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
