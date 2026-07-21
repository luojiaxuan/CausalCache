#!/usr/bin/env python3
"""Reduce long-history oracle wave labels into the frozen diagnostic summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_evaluation import (
    canonical_json_bytes,
    sha256_file,
)
from causalcache.set_utility_long_oracle import (
    WAVES,
    bound_state_ids_from_manifest,
    reduce_long_oracle_metrics,
    select_configured_long_oracle_states,
    tune_headroom_decision,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _wave_root(value: str) -> tuple[int, Path]:
    raw_wave, separator, raw_path = value.partition("=")
    if not separator or not raw_wave or not raw_path:
        raise argparse.ArgumentTypeError("wave root must be wave=path")
    wave = int(raw_wave)
    if wave not in WAVES:
        raise argparse.ArgumentTypeError("wave root index must be in 1..4")
    return wave, Path(raw_path)


def _bound_state_ids(
    repository_root: Path, config: dict[str, Any]
) -> tuple[str, ...]:
    if config.get("selection_mode", "sampled_train") != "bound_state_manifest":
        return ()
    path = repository_root / config["state_manifest"]
    if sha256_file(path) != config["state_manifest_sha256"]:
        raise ValueError("bound state manifest drifted from the frozen binding")
    return bound_state_ids_from_manifest(
        _read_json(path),
        expected_state_count=int(config["expected_state_count"]),
        expected_trajectory_count=int(config["expected_trajectory_count"]),
        expected_history_bin_counts={
            name: int(value)
            for name, value in config["expected_history_bin_counts"].items()
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--wave-root", type=_wave_root, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("long-oracle summary output already exists")
    config = _read_json(args.config)
    config_sha = sha256_file(args.config)
    assignment_path = args.repository_root / config["assignment_manifest"]
    if sha256_file(assignment_path) != config["assignment_manifest_sha256"]:
        raise ValueError("assignment manifest drifted from the frozen binding")
    states = select_configured_long_oracle_states(
        _read_json(assignment_path)["assignments"],
        config=config,
        bound_state_ids=_bound_state_ids(args.repository_root, config),
    )
    roots: dict[int, list[Path]] = defaultdict(list)
    for wave, path in args.wave_root:
        roots[wave].append(path)
    if set(roots) != set(WAVES):
        raise ValueError("reducer requires wave roots for waves one through four")
    wave_terminals: dict[int, dict[str, dict[str, Any]]] = {}
    for wave in WAVES:
        terminals: dict[str, dict[str, Any]] = {}
        for root in roots[wave]:
            states_dir = root / "states"
            if not states_dir.is_dir():
                raise ValueError(f"{states_dir} is not a directory")
            for path in sorted(states_dir.glob("*.json")):
                terminal = _read_json(path)
                state_id = terminal["state_id"]
                if state_id in terminals:
                    raise ValueError(
                        f"duplicate wave {wave} terminal for {state_id}"
                    )
                terminals[state_id] = terminal
        wave_terminals[wave] = terminals
    summary = reduce_long_oracle_metrics(
        states=states,
        wave_terminals=wave_terminals,
        bootstrap_resamples=int(config["bootstrap"]["resamples"]),
        bootstrap_seed=int(config["bootstrap"]["seed"]),
        random_subsets_per_budget=int(config["random_subsets_per_budget"]),
        random_seed=int(config["random_seed"]),
    )
    summary["config_sha256"] = config_sha
    summary["schema_version"] = (
        "causalcache.set_utility_long_oracle_summary.v1"
    )
    summary["status"] = "COMPLETED_SET_UTILITY_LONG_ORACLE_EVALUATION"
    if "tune_headroom_gate" in config:
        gate = config["tune_headroom_gate"]
        summary["tune_headroom_gate"] = tune_headroom_decision(
            summary["comparisons"]["oracle_greedy_minus_recent_macro"],
            confirmed_point_minimum=float(gate["confirmed_point_minimum"]),
            confirmed_lower_minimum=float(gate["confirmed_lower_minimum"]),
            insufficient_upper_maximum=float(gate["insufficient_upper_maximum"]),
        )
    summary["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(summary)
    ).hexdigest()
    payload = canonical_json_bytes(summary) + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(
        args.output.suffix + f".{os.getpid()}.tmp"
    )
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(json.dumps(summary["methods"], sort_keys=True))
    print(json.dumps(summary["comparisons"], sort_keys=True))
    if "tune_headroom_gate" in summary:
        print(json.dumps(summary["tune_headroom_gate"], sort_keys=True))


if __name__ == "__main__":
    main()
