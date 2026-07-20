#!/usr/bin/env python3
"""Materialize one wave of long-history oracle restoration schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_evaluation import (
    canonical_json_bytes,
    sha256_file,
)
from causalcache.set_utility_long_oracle import (
    WAVES,
    additive_top_k,
    greedy_selection_from_distances,
    next_wave_coalitions,
    select_long_oracle_states,
    wave_one_coalitions,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_atomic(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _wave_root(value: str) -> tuple[int, Path]:
    raw_wave, separator, raw_path = value.partition("=")
    if not separator or not raw_wave or not raw_path:
        raise argparse.ArgumentTypeError("wave root must be wave=path")
    wave = int(raw_wave)
    if wave not in WAVES:
        raise argparse.ArgumentTypeError("wave root index must be in 1..4")
    return wave, Path(raw_path)


def _load_wave_terminals(
    roots: dict[int, list[Path]], *, waves: tuple[int, ...]
) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for wave in waves:
        if not roots.get(wave):
            raise ValueError(f"wave {wave} terminals are required but missing")
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
        result[wave] = terminals
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument(
        "--wave-root", type=_wave_root, action="append", default=[]
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.wave not in WAVES:
        raise ValueError("wave must be one of 1, 2, 3, 4")
    if args.output_root.exists():
        raise FileExistsError("long-oracle schedule output already exists")
    config = _read_json(args.config)
    config_sha = sha256_file(args.config)
    assignment_path = args.repository_root / config["assignment_manifest"]
    if sha256_file(assignment_path) != config["assignment_manifest_sha256"]:
        raise ValueError("assignment manifest drifted from the frozen binding")
    assignments = _read_json(assignment_path)["assignments"]
    states = select_long_oracle_states(
        assignments,
        bin_targets={
            name: int(value) for name, value in config["bin_targets"].items()
        },
        maximum_states_per_trajectory=int(
            config["maximum_states_per_trajectory"]
        ),
        salt=config["selection_salt"],
    )

    prior_waves = tuple(range(1, args.wave))
    roots: dict[int, list[Path]] = defaultdict(list)
    for wave, path in args.wave_root:
        roots[wave].append(path)
    if set(roots) != set(prior_waves):
        raise ValueError(
            "wave roots must cover exactly every prior wave of this schedule"
        )
    terminals = _load_wave_terminals(roots, waves=prior_waves)

    schedule_rows: list[dict[str, Any]] = []
    prior_skips: dict[str, str] = {}
    coalition_count = 0
    source_counts: Counter[str] = Counter()
    for state in states:
        candidates = tuple(state["candidate_event_ids"])
        if args.wave == 1:
            coalitions = wave_one_coalitions(
                state_id=state["state_id"],
                candidate_event_ids=candidates,
                random_subsets_per_budget=int(
                    config["random_subsets_per_budget"]
                ),
                random_seed=int(config["random_seed"]),
            )
        else:
            skip_reason = None
            prefix: tuple[int, ...] = ()
            for wave in prior_waves:
                terminal = terminals[wave].get(state["state_id"])
                if terminal is None:
                    skip_reason = f"missing_wave_{wave}"
                    break
                if (
                    terminal.get("status")
                    != "COMPLETED_VARIABLE_HISTORY_LABEL_STATE"
                ):
                    skip_reason = (
                        f"wave_{wave}_"
                        f"{terminal.get('failure_class', 'skipped')}"
                    )
                    break
                prefix = greedy_selection_from_distances(
                    candidate_event_ids=candidates,
                    distance_rows=terminal["distance_rows"],
                    previous_selected=prefix,
                )
            if skip_reason is not None:
                prior_skips[state["state_id"]] = skip_reason
                continue
            additive_subset = (
                additive_top_k(
                    candidate_event_ids=candidates,
                    distance_rows=terminals[1][state["state_id"]][
                        "distance_rows"
                    ],
                    k=args.wave,
                )
                if args.wave >= 3
                else None
            )
            coalitions = next_wave_coalitions(
                candidate_event_ids=candidates,
                previous_selected=prefix,
                additive_subset=additive_subset,
            )
        source_counts.update(row["source"] for row in coalitions)
        coalition_count += len(coalitions)
        schedule_rows.append(
            {
                "candidate_event_ids": list(candidates),
                "coalitions": list(coalitions),
                "exact": False,
                "logical_shard": state["logical_shard"],
                "role": state["role"],
                "state_id": state["state_id"],
                "trajectory_id": state["trajectory_id"],
            }
        )

    by_shard: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(256)
    }
    for row in schedule_rows:
        by_shard[row["logical_shard"]].append(row)
    receipts = []
    shard_bytes_digest = hashlib.sha256()
    for logical_shard in range(256):
        rows = sorted(by_shard[logical_shard], key=lambda row: row["state_id"])
        payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        shard_bytes_digest.update(payload)
        path = (
            args.output_root
            / "schedule-shards"
            / f"shard-{logical_shard:03d}-of-256.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        receipt = {
            "coalition_count": sum(len(row["coalitions"]) for row in rows),
            "identity_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {
                        "config_sha256": config_sha,
                        "logical_shard": logical_shard,
                        "wave": args.wave,
                    }
                )
            ).hexdigest(),
            "logical_shard": logical_shard,
            "role_state_counts": {"train": len(rows)} if rows else {},
            "schedule_byte_count": len(payload),
            "schedule_sha256": hashlib.sha256(payload).hexdigest(),
            "state_count": len(rows),
            "status": "COMPLETED_SET_UTILITY_LONG_ORACLE_SCHEDULE_SHARD",
            "wave": args.wave,
        }
        _write_atomic(
            args.output_root / "receipts" / f"shard-{logical_shard:03d}-of-256.json",
            receipt,
        )
        receipts.append(receipt)
    manifest = {
        "coalition_count": coalition_count,
        "config_sha256": config_sha,
        "prior_wave_skipped_states": dict(sorted(prior_skips.items())),
        "schedule_shards_sha256": shard_bytes_digest.hexdigest(),
        "schema_version": "1.0.0",
        "selected_state_count": len(states),
        "scheduled_state_count": len(schedule_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "status": "COMPLETED_SET_UTILITY_LONG_ORACLE_SCHEDULES",
        "wave": args.wave,
    }
    manifest["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    _write_atomic(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
