#!/usr/bin/env python3
"""Materialize the exact optimizer-group inventory before freezing training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from causalcache.set_utility_heldout_evaluation import canonical_json_bytes
from causalcache.set_utility_formal_input_verification import (
    verify_formal_training_input,
)
from causalcache.set_utility_structured_training import (
    apply_split_manifest,
    formal_group_inventory,
    prepare_group_examples,
)
from causalcache.set_utility_train_heldout_contract import sha256_json


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> tuple[dict, ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--input-content-sha256", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-content-sha256", required=True)
    parser.add_argument("--normalization-floor", type=float, default=0.01)
    parser.add_argument("--maximum-base-cardinality", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    input_manifest = verify_formal_training_input(
        input_root, expected_content_sha256=args.input_content_sha256
    )
    states_path = input_root / str(input_manifest.get("states_jsonl", ""))
    split = _read_json(args.split_manifest.resolve())
    unsigned_split = dict(split)
    claimed_split = unsigned_split.pop("content_sha256", None)
    if (
        claimed_split != args.split_content_sha256
        or sha256_json(unsigned_split) != claimed_split
    ):
        raise ValueError("formal inventory split binding drifted")
    states = tuple(row for row in _read_jsonl(states_path) if row.get("role") == "train")
    optimization, heldout = apply_split_manifest(states, split)
    examples = prepare_group_examples(
        optimization,
        normalization_floor=args.normalization_floor,
        maximum_base_cardinality=args.maximum_base_cardinality,
        epsilon=args.epsilon,
    )
    inventory = formal_group_inventory(examples)
    result = {
        "heldout_checkpoint_state_count": len(heldout),
        "input_content_sha256": input_manifest["content_sha256"],
        "optimizer_inventory": inventory,
        "schema_version": "causalcache.formal_optimizer_inventory.v1",
        "split_content_sha256": claimed_split,
        "status": "COMPLETED_SET_UTILITY_FORMAL_OPTIMIZER_INVENTORY",
    }
    result["content_sha256"] = sha256_json(result)
    payload = canonical_json_bytes(result, pretty=True) + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if not args.output.is_file() or args.output.read_bytes() != payload:
            raise ValueError("existing formal optimizer inventory drifted")
    else:
        temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
