#!/usr/bin/env python3
"""Run the frozen and terminal-s60 OSWorld arms concurrently."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causalcache.osworld_transfer import load_transfer_config


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--frozen-policy-endpoint", action="append", required=True)
    parser.add_argument("--terminal-policy-endpoint", action="append", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    repository_root = args.repository_root.resolve()
    config = load_transfer_config(args.config.resolve())
    execution = config["execution"]
    expected_replicas = int(execution["policy_replicas_per_arm"])
    endpoints = {
        "frozen": tuple(args.frozen_policy_endpoint),
        "terminal_s60": tuple(args.terminal_policy_endpoint),
    }
    for arm, values in endpoints.items():
        if len(values) != expected_replicas:
            raise ValueError(
                f"{arm} endpoint count drifted: {len(values)} != {expected_replicas}"
            )

    raw_root = args.raw_root.resolve()
    cache_root = args.cache_root.resolve()
    processes: dict[str, subprocess.Popen[Any]] = {}
    logs: dict[str, Any] = {}
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        for arm in ("frozen", "terminal_s60"):
            arm_root = raw_root / arm
            arm_root.mkdir(parents=True, exist_ok=True)
            (cache_root / arm).mkdir(parents=True, exist_ok=True)
            log = (arm_root / "runner.log").open("a", encoding="utf-8")
            logs[arm] = log
            endpoint_arguments = [
                value
                for endpoint in endpoints[arm]
                for value in ("--policy-endpoint", endpoint)
            ]
            command = [
                sys.executable,
                str(repository_root / "code/scripts/run_osworld_multienv.py"),
                "--repository-root",
                str(repository_root),
                "--osworld-root",
                str(args.osworld_root.resolve()),
                "--config",
                str(repository_root / config["benchmark_config"]),
                "--num-envs",
                str(execution["environments_per_arm"]),
                "--limit",
                str(config["selection"]["task_count"]),
                "--selection-mode",
                config["selection"]["mode"],
                "--output-root",
                str(arm_root),
                "--cache-dir",
                str(cache_root / arm),
                "--max-steps",
                str(execution["max_steps"]),
                "--pause-seconds",
                str(execution["pause_seconds"]),
                *endpoint_arguments,
            ]
            processes[arm] = subprocess.Popen(
                command,
                cwd=repository_root,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return_codes = {arm: process.wait() for arm, process in processes.items()}
    except BaseException:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            process.wait()
        raise
    finally:
        for log in logs.values():
            log.close()

    record = {
        "schema_version": config["schema_version"],
        "status": (
            "COMPLETE_OSWORLD_TRANSFER_EXECUTION"
            if all(code == 0 for code in return_codes.values())
            else "PARTIAL_OSWORLD_TRANSFER_EXECUTION"
        ),
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "return_codes": return_codes,
        "endpoints": endpoints,
    }
    _atomic_json(raw_root / "execution.json", record)
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    if record["status"] != "COMPLETE_OSWORLD_TRANSFER_EXECUTION":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
