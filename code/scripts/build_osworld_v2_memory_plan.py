#!/usr/bin/env python3
"""Build the OSWorld 2.0 phenomenon-defined memory dependency split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from causalcache.osworld_v2 import (
    OSWORLD_V2_CODE_REVISION,
    OSWORLD_V2_EXPECTED_TASKS,
    OSWORLD_V2_MEMORY_PLAN_SCHEMA_VERSION,
    OSWORLD_V2_RELEASE,
    OSWORLD_V2_TAG,
    validate_osworld_v2_checkout,
)


PHENOMENON_FILES = {
    "dynamic_environment": "evaluation_examples/dynamic_environment.json",
    "cross_source_reasoning": "evaluation_examples/cross_source_reasoning.json",
    "implicit_state_inference": (
        "evaluation_examples/implicit_state_inference.json"
    ),
}


def _load_phenomenon(root: Path, name: str, relative_path: str) -> set[str]:
    path = root / relative_path
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {name}:
        raise ValueError(f"OSWorld 2.0 {name} label file schema drifted")
    task_ids = value[name]
    if (
        not isinstance(task_ids, list)
        or any(
            not isinstance(task_id, str)
            or len(task_id) != 3
            or not task_id.isdigit()
            for task_id in task_ids
        )
        or task_ids != sorted(set(task_ids))
    ):
        raise ValueError(f"OSWorld 2.0 {name} task ids drifted")
    return set(task_ids)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_plan(root: Path) -> dict[str, Any]:
    readiness = validate_osworld_v2_checkout(
        root,
        require_task_classes=False,
        require_assets=False,
    )
    labels = {
        name: _load_phenomenon(root, name, relative_path)
        for name, relative_path in PHENOMENON_FILES.items()
    }
    records = []
    for index in range(1, OSWORLD_V2_EXPECTED_TASKS + 1):
        task_id = f"{index:03d}"
        phenomena = sorted(
            name for name, task_ids in labels.items() if task_id in task_ids
        )
        if "implicit_state_inference" in phenomena:
            memory_split = "explicit_hidden_state_memory_core"
        elif phenomena:
            memory_split = "memory_stress_candidate"
        else:
            memory_split = "non_memory_control"
        records.append(
            {
                "task_id": task_id,
                "phenomena": phenomena,
                "memory_split": memory_split,
            }
        )
    memory_core = [
        record["task_id"]
        for record in records
        if record["memory_split"] == "explicit_hidden_state_memory_core"
    ]
    stress = [
        record["task_id"]
        for record in records
        if record["memory_split"]
        in {"explicit_hidden_state_memory_core", "memory_stress_candidate"}
    ]
    controls = [
        record["task_id"]
        for record in records
        if record["memory_split"] == "non_memory_control"
    ]
    return {
        "schema_version": OSWORLD_V2_MEMORY_PLAN_SCHEMA_VERSION,
        "upstream": {
            "repository": "https://github.com/xlang-ai/OSWorld-V2",
            "release": OSWORLD_V2_RELEASE,
            "tag": OSWORLD_V2_TAG,
            "code_revision": OSWORLD_V2_CODE_REVISION,
            "task_count": OSWORLD_V2_EXPECTED_TASKS,
        },
        "construction": {
            "memory_core_rule": "official implicit_state_inference label",
            "memory_stress_rule": (
                "union of official dynamic_environment, "
                "cross_source_reasoning, and implicit_state_inference labels"
            ),
            "control_rule": "complement of the memory-stress union",
            "scope_note": (
                "The split is frozen from task-construction labels before policy "
                "execution; failure traces do not assign memory dependence."
            ),
        },
        "counts": {
            **{name: len(task_ids) for name, task_ids in labels.items()},
            "memory_core": len(memory_core),
            "memory_stress_union": len(stress),
            "non_memory_control": len(controls),
            "all_three_phenomena": len(set.intersection(*labels.values())),
        },
        "splits": {
            "memory_core": memory_core,
            "memory_stress_union": stress,
            "non_memory_control": controls,
        },
        "records": records,
        "records_sha256": _canonical_sha256(records),
        "substrate_readiness": readiness,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(args.osworld_root.expanduser().resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
