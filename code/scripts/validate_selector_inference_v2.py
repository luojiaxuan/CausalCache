#!/usr/bin/env python3
"""Validate V2 selection coverage, traces, budgets, and exact B1 parity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from causalcache.hgkv_selector_v2 import canonical_event_set
from scripts.build_hgkv_selector_v2_training_data import (
    expand_paths,
    sha256_file,
)


METHODS = ("hgkv_v2_greedy", "hgkv_v2_beam4")
BUDGETS = (1, 2, 4)


def load_states(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_group = str(row["pair_group"])
            if pair_group in states:
                raise ValueError(f"{path}:{line_no} duplicate state")
            states[pair_group] = row
    return states


def validate_selections(
    *,
    states: Mapping[str, Mapping[str, Any]],
    selection_paths: Iterable[Path],
) -> dict[str, Any]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    selected_sizes: dict[tuple[str, int], list[int]] = {}
    for path in selection_paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_group = str(row["pair_group"])
            method = str(row["method"])
            budget = int(row["budget"])
            key = (pair_group, method, budget)
            if pair_group not in states:
                raise ValueError(f"{path}:{line_no} unexpected state")
            if method not in METHODS or budget not in BUDGETS:
                raise ValueError(f"{path}:{line_no} unexpected method/budget")
            if key in rows:
                raise ValueError(f"{path}:{line_no} duplicate selection {key}")
            state = states[pair_group]
            candidates = [
                int(value) for value in state["candidate_event_step_ids"]
            ]
            if row["candidate_event_step_ids"] != candidates:
                raise ValueError(f"{path}:{line_no} inventory drifted")
            selected = canonical_event_set(row["selected_event_step_ids"])
            if not set(selected).issubset(candidates) or len(selected) > budget:
                raise ValueError(f"{path}:{line_no} invalid selected set")
            cumulative = 0.0
            trace_selected: tuple[int, ...] = ()
            for step in row["steps"]:
                if list(trace_selected) != step["prefix"]:
                    raise ValueError(f"{path}:{line_no} trace prefix drifted")
                marginal = float(step["predicted_marginal"])
                cumulative += marginal
                if abs(
                    cumulative - float(step["cumulative_predicted_u"])
                ) > 1e-7:
                    raise ValueError(f"{path}:{line_no} trace sum drifted")
                candidate = step["candidate"]
                if candidate is not None:
                    trace_selected = canonical_event_set(
                        (*trace_selected, int(candidate))
                    )
            declared = float(row["cumulative_predicted_u"])
            if not math.isfinite(declared) or abs(cumulative - declared) > 1e-7:
                raise ValueError(f"{path}:{line_no} final score drifted")
            if trace_selected != selected:
                raise ValueError(f"{path}:{line_no} trace selection drifted")
            if bool(row["stopped_early"]) != (len(selected) < budget):
                raise ValueError(f"{path}:{line_no} STOP flag drifted")
            rows[key] = row
            selected_sizes.setdefault((method, budget), []).append(len(selected))
    expected = {
        (pair_group, method, budget)
        for pair_group in states
        for method in METHODS
        for budget in BUDGETS
    }
    missing = expected - set(rows)
    unexpected = set(rows) - expected
    if missing or unexpected:
        raise ValueError(
            f"selection coverage mismatch missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
    b1_mismatches = 0
    for pair_group in states:
        greedy = rows[(pair_group, "hgkv_v2_greedy", 1)]
        beam = rows[(pair_group, "hgkv_v2_beam4", 1)]
        if (
            greedy["selected_event_step_ids"]
            != beam["selected_event_step_ids"]
            or abs(
                float(greedy["cumulative_predicted_u"])
                - float(beam["cumulative_predicted_u"])
            )
            > 1e-7
        ):
            b1_mismatches += 1
    if b1_mismatches:
        raise ValueError(f"B1 parity failed for {b1_mismatches} states")
    return {
        "states": len(states),
        "rows": len(rows),
        "expected_rows": len(expected),
        "missing_rows": 0,
        "unexpected_rows": 0,
        "b1_parity_mismatches": 0,
        "mean_selected_size": {
            f"{method}/B{budget}": (
                sum(sizes) / len(sizes)
            )
            for (method, budget), sizes in sorted(selected_sizes.items())
        },
        "status": "DONE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", action="append", required=True)
    parser.add_argument("--selections", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state_paths = expand_paths(args.states)
    selection_paths = expand_paths(args.selections)
    if not state_paths or not selection_paths:
        parser.error("state and selection patterns must match files")
    result = validate_selections(
        states=load_states(state_paths),
        selection_paths=selection_paths,
    )
    result["schema_version"] = (
        "causalcache.hgkv_selector_v2.selection_validation.v1"
    )
    result["state_sha256"] = {
        str(path): sha256_file(path) for path in state_paths
    }
    result["selection_sha256"] = {
        str(path): sha256_file(path) for path in selection_paths
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
