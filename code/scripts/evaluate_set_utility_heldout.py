#!/usr/bin/env python3
"""Reduce frozen held-out selector labels into the GO/NO-GO report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_evaluation import (
    canonical_json_bytes,
    evaluate_heldout,
    load_label_terminals,
    sha256_file,
)


def _checkpoint_sizes(values: list[str]) -> dict[str, int]:
    result = {}
    for value in values:
        model, separator, raw_path = value.partition("=")
        if not separator or not model or not raw_path:
            raise ValueError("checkpoint size binding must be MODEL=PATH")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if model in result:
            raise ValueError(f"duplicate checkpoint size binding: {model}")
        result[model] = path.stat().st_size
    return result


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-config", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    config = json.loads(args.heldout_config.read_text(encoding="utf-8"))
    selections = json.loads(args.selections.read_text(encoding="utf-8"))
    config_sha = sha256_file(args.heldout_config)
    if selections.get("config_sha256") != config_sha:
        raise ValueError("sealed selections do not bind the held-out config")
    if selections.get("inventory_sha256") != config["state_inventory"]["sha256"]:
        raise ValueError("sealed selections do not bind the frozen state inventory")
    records = selections.get("records", ())
    state_contract = config["state_inventory"]
    if len(records) != state_contract["union_state_count"]:
        raise ValueError("sealed selections do not cover the frozen evaluation union")
    exact_ids = {
        row["state_id"] for row in records if "exact_oracle" in row["tracks"]
    }
    large_ids = {
        row["state_id"] for row in records if "large_history" in row["tracks"]
    }
    if (
        len(exact_ids) != state_contract["exact_state_count"]
        or len(large_ids) != state_contract["large_history_state_count"]
        or exact_ids | large_ids != {row["state_id"] for row in records}
    ):
        raise ValueError("sealed selection track counts drifted")
    if set(selections.get("model_artifacts", ())) != set(
        config["frozen_candidates"]["models"]
    ):
        raise ValueError("sealed selection model inventory drifted")
    for model, candidate in config["frozen_candidates"]["models"].items():
        artifact = selections.get("model_artifacts", {}).get(model)
        if artifact is None or artifact.get("checkpoint_sha256") != candidate["sha256"]:
            raise ValueError(f"sealed selector checkpoint drifted: {model}")
    metrics = config["metrics"]
    result = evaluate_heldout(
        selections=selections,
        terminals=load_label_terminals(args.label_root),
        budgets=config["selection"]["budgets"],
        normalization_floor=config["representation"]["normalization_floor"],
        bootstrap_resamples=metrics["bootstrap"]["resamples"],
        bootstrap_seed=metrics["bootstrap"]["seed"],
        bootstrap_interval=metrics["bootstrap"]["interval"],
        reference_config_sha256=config["restoration_truth"][
            "reference_config_sha256"
        ],
        checkpoint_sizes=_checkpoint_sizes(args.checkpoint),
    )
    result["bindings"] = {
        "config_sha256": config_sha,
        "label_roots": [str(path.resolve()) for path in args.label_root],
        "selections_sha256": sha256_file(args.selections),
    }
    unsigned = dict(result)
    unsigned.pop("content_sha256", None)
    import hashlib

    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    _write_atomic(args.output, canonical_json_bytes(result, pretty=True))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "winner": result["winner"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
