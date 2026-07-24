#!/usr/bin/env python3
"""Build one longest-real-decision full-history state per GUI-Odyssey trajectory."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from causalcache.exploratory_closed_loop_memory import (
    candidate_event_step_ids_from_history,
)
from causalcache.policy.gui_owl_v2_1 import (
    serialize_gui_owl_v2_1_teacher_target,
)
from scripts.build_odyssey_sft_dataset import (
    action_from_annotation,
    load_heldout_episodes,
    load_state_inventory,
)


def _resolution(
    row: dict[str, Any], annotation: dict[str, Any]
) -> tuple[int, int] | None:
    device = annotation.get("device_info")
    if isinstance(device, dict) and isinstance(
        device.get("device_resolution"), list
    ):
        values = device["device_resolution"]
        if len(values) == 2:
            return (int(values[0]), int(values[1]))
    try:
        metadata = json.loads(row.get("raw_metadata", "{}"))
    except (TypeError, json.JSONDecodeError):
        return None
    values = (metadata.get("others") or {}).get("resolution")
    if isinstance(values, list) and len(values) == 2:
        return (int(values[0]), int(values[1]))
    return None


def _history_mappings(
    events_payload: list[dict[str, Any]], decision: int
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for payload in events_payload[: decision - 1]:
        step_id = int(payload["event_step_id"])
        summary = dict(payload["low_fidelity_summary"])
        summary.setdefault("step_id", step_id)
        history.append(
            {
                "event_step_id": step_id,
                "low_fidelity_v2": summary,
            }
        )
    return history


def select_longest_real_state(
    row: dict[str, Any],
    annotation: dict[str, Any],
    decisions: Iterable[int],
    *,
    split: str,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """Select the longest eligible non-terminal decision for one trajectory."""
    source_id = str(row["source_id"])
    events_payload = json.loads(row["history_events_json"])
    if not isinstance(events_payload, list):
        raise ValueError(f"{source_id} history_events_json must be a list")
    steps = annotation.get("steps")
    if not isinstance(steps, list) or not steps:
        return None, {"synthetic_terminal": 0, "ineligible": 1}
    if str(steps[-1].get("action", "")).upper() != "COMPLETE":
        return None, {"synthetic_terminal": 0, "ineligible": 1}
    resolution = _resolution(row, annotation)
    if resolution is None:
        return None, {"synthetic_terminal": 0, "ineligible": 1}
    steps_by_index = {int(step["step"]): step for step in steps}
    eligible: list[dict[str, Any]] = []
    synthetic_terminal = 0
    ineligible = 0
    for decision in sorted(set(int(value) for value in decisions)):
        annotation_step = steps_by_index.get(decision - 1)
        if annotation_step is None:
            ineligible += 1
            continue
        if str(annotation_step.get("action", "")).upper() == "COMPLETE":
            synthetic_terminal += 1
            continue
        action = action_from_annotation(annotation_step, resolution)
        if action is None:
            ineligible += 1
            continue
        try:
            target_text = serialize_gui_owl_v2_1_teacher_target(action)
        except (TypeError, ValueError):
            ineligible += 1
            continue
        history = _history_mappings(events_payload, decision)
        if len(history) < 2:
            ineligible += 1
            continue
        candidate_ids = candidate_event_step_ids_from_history(history)
        if len(candidate_ids) < 2:
            ineligible += 1
            continue
        current_equivalent = int(history[-1]["event_step_id"])
        if current_equivalent in candidate_ids:
            raise AssertionError("current-equivalent event leaked into candidates")
        app = str(
            history[-1]["low_fidelity_v2"].get("foreground_app", "unknown")
        )
        eligible.append(
            {
                "schema_version": "causalcache.hgkv_selector_v2.state.v1",
                "pair_group": f"{source_id}:{decision}",
                "episode": source_id,
                "decision_step": decision,
                "target_action_text": target_text,
                "candidate_event_step_ids": list(candidate_ids),
                "candidate_count": len(candidate_ids),
                "history_length": len(history),
                "current_equivalent_event_step_id": current_equivalent,
                "split": split,
                "action_type": str(annotation_step["action"]).lower(),
                "app": app,
                "synthetic_terminal": False,
                "recent_candidate_cap": None,
            }
        )
    if not eligible:
        return None, {
            "synthetic_terminal": synthetic_terminal,
            "ineligible": ineligible,
        }
    selected = max(
        eligible,
        key=lambda state: (state["candidate_count"], state["decision_step"]),
    )
    return selected, {
        "synthetic_terminal": synthetic_terminal,
        "ineligible": ineligible,
    }


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def inventory_audit(
    rows: Sequence[dict[str, Any]],
    *,
    trajectories_seen: int,
    trajectories_successful: int,
    synthetic_terminal_excluded: int,
    ineligible_decisions: int,
) -> dict[str, Any]:
    counts = [int(row["candidate_count"]) for row in rows]
    history = Counter(int(row["history_length"]) for row in rows)
    actions = Counter(str(row["action_type"]) for row in rows)
    apps = Counter(str(row["app"]) for row in rows)
    if not counts:
        raise ValueError("V2 state inventory is empty")
    if any(row.get("synthetic_terminal") for row in rows):
        raise AssertionError("synthetic terminal state entered V2 inventory")
    if any(row.get("recent_candidate_cap") is not None for row in rows):
        raise AssertionError("recent candidate cap entered V2 inventory")
    return {
        "schema_version": "causalcache.hgkv_selector_v2.inventory_audit.v1",
        "trajectories_seen": trajectories_seen,
        "trajectories_successful": trajectories_successful,
        "states": len(rows),
        "candidate_count": {
            "min": min(counts),
            "median": _percentile(counts, 0.5),
            "p90": _percentile(counts, 0.9),
            "p95": _percentile(counts, 0.95),
            "max": max(counts),
        },
        "candidate_count_ge_4_fraction": (
            sum(value >= 4 for value in counts) / len(counts)
        ),
        "states_with_candidate_count_gt_8": sum(value > 8 for value in counts),
        "action_type_distribution": dict(sorted(actions.items())),
        "app_distribution": dict(sorted(apps.items())),
        "history_length_histogram": {
            str(key): history[key] for key in sorted(history)
        },
        "synthetic_terminal_count": 0,
        "synthetic_terminal_excluded": synthetic_terminal_excluded,
        "recent_8_truncation_count": 0,
        "ineligible_decisions": ineligible_decisions,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: item["pair_group"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--state-context", type=Path, required=True)
    parser.add_argument("--heldout-episodes", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--dev-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    from pyarrow import parquet as pq

    decisions = load_state_inventory(args.state_context)
    heldout = load_heldout_episodes(args.heldout_episodes)
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    seen = 0
    successful = 0
    synthetic_terminal_excluded = 0
    ineligible_decisions = 0
    for shard in sorted(args.source_root.glob("shard-*.parquet")):
        for row in pq.read_table(shard).to_pylist():
            source_id = str(row["source_id"])
            if source_id not in decisions:
                continue
            seen += 1
            annotation_path = args.annotations_root / f"{source_id}.json"
            if not annotation_path.exists():
                continue
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            is_success = (
                isinstance(annotation.get("steps"), list)
                and annotation["steps"]
                and str(annotation["steps"][-1].get("action", "")).upper()
                == "COMPLETE"
            )
            if not is_success:
                continue
            successful += 1
            split = "dev" if source_id in heldout else "train"
            selected, counters = select_longest_real_state(
                row,
                annotation,
                decisions[source_id],
                split=split,
            )
            synthetic_terminal_excluded += counters["synthetic_terminal"]
            ineligible_decisions += counters["ineligible"]
            if selected is None:
                continue
            (dev if split == "dev" else train).append(selected)
    combined = train + dev
    audit = inventory_audit(
        combined,
        trajectories_seen=seen,
        trajectories_successful=successful,
        synthetic_terminal_excluded=synthetic_terminal_excluded,
        ineligible_decisions=ineligible_decisions,
    )
    audit["split_states"] = {"train": len(train), "dev": len(dev)}
    _write_jsonl(args.train_output, train)
    _write_jsonl(args.dev_output, dev)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
