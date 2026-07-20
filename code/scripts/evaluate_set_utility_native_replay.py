#!/usr/bin/env python3
"""Reduce frozen native-action replay terminals into a signed report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from causalcache.set_utility_native_replay import canonical_json_bytes
from causalcache.set_utility_native_replay_evaluation import evaluate_native_replay


def _write_atomic(path: Path, value: object) -> None:
    payload = canonical_json_bytes(value, pretty=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--terminal-root", type=Path, action="append", required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--heldout-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate_native_replay(
        schedule_root=args.schedule_root,
        terminal_roots=args.terminal_root,
        selections_path=args.selections,
        heldout_result_path=args.heldout_result,
    )
    _write_atomic(args.output, result)
    print(
        json.dumps(
            {
                "behavior_recovery_gate": result["behavior_recovery_gate"],
                "closed_loop_authorization": result[
                    "closed_loop_authorization"
                ],
                "coverage": result["coverage"],
                "deployment_latency_gate": result["deployment_latency_gate"],
                "output": str(args.output),
                "status": result["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
