#!/usr/bin/env python3
"""Build the revision-locked MobileWorld memory-candidate split from task code."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


SPECIAL_TAGS = frozenset({"agent-mcp", "agent-user-interaction"})


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if len(revision) != 40:
        raise ValueError("MobileWorld checkout does not expose a full revision")
    return revision


def _literal_class_field(node: ast.ClassDef, field: str, *, default: Any) -> Any:
    matches = []
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id == field:
                    matches.append(statement.value)
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == field
        ):
            matches.append(statement.value)
    if not matches:
        return default
    if len(matches) != 1:
        raise ValueError(f"{node.name} defines {field} more than once")
    try:
        return ast.literal_eval(matches[0])
    except (ValueError, TypeError) as error:
        raise ValueError(f"{node.name}.{field} must remain a literal") from error


def scan_mobileworld_tasks(root: Path) -> list[dict[str, Any]]:
    definitions = root / "src/mobile_world/tasks/definitions"
    if not definitions.is_dir():
        raise FileNotFoundError("MobileWorld task definitions directory is missing")
    records = []
    for path in sorted(definitions.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            direct_base_task = any(
                isinstance(base, ast.Name) and base.id == "BaseTask"
                for base in node.bases
            )
            if not direct_base_task:
                continue
            apps = _literal_class_field(node, "app_names", default=None)
            tags = _literal_class_field(node, "task_tags", default=set())
            if (
                not isinstance(apps, (set, list, tuple))
                or not apps
                or any(not isinstance(app, str) or not app for app in apps)
            ):
                raise ValueError(f"{node.name}.app_names must be literal app names")
            if (
                not isinstance(tags, (set, list, tuple))
                or any(not isinstance(tag, str) or not tag for tag in tags)
            ):
                raise ValueError(f"{node.name}.task_tags must be literal tags")
            canonical_apps = sorted(set(apps))
            canonical_tags = sorted(set(tags))
            special = set(canonical_tags) & SPECIAL_TAGS
            if special == {"agent-mcp", "agent-user-interaction"}:
                interface = "mcp_and_user_interaction"
            elif "agent-mcp" in special:
                interface = "mcp"
            elif "agent-user-interaction" in special:
                interface = "user_interaction"
            else:
                interface = "gui_only"
            records.append(
                {
                    "task_name": node.name,
                    "source_path": str(path.relative_to(root)),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "apps": canonical_apps,
                    "app_count": len(canonical_apps),
                    "tags": canonical_tags,
                    "interface": interface,
                    "memory_split": (
                        "cross_app_memory_candidate"
                        if len(canonical_apps) >= 2
                        else "single_app_control"
                    ),
                }
            )
    records.sort(key=lambda record: record["task_name"])
    if len(records) != len({record["task_name"] for record in records}):
        raise ValueError("MobileWorld task class names must be unique")
    return records


def build_plan(
    root: Path,
    *,
    expected_revision: str,
    expected_task_count: int,
) -> dict[str, Any]:
    revision = _git_revision(root)
    if revision != expected_revision:
        raise ValueError(
            f"MobileWorld revision drifted: expected {expected_revision}, got {revision}"
        )
    records = scan_mobileworld_tasks(root)
    if len(records) != expected_task_count:
        raise ValueError(
            f"MobileWorld task count drifted: expected {expected_task_count}, "
            f"got {len(records)}"
        )
    interfaces = Counter(record["interface"] for record in records)
    gui_only = [
        record["task_name"] for record in records if record["interface"] == "gui_only"
    ]
    cross_app_gui = [
        record["task_name"]
        for record in records
        if record["interface"] == "gui_only"
        and record["memory_split"] == "cross_app_memory_candidate"
    ]
    single_app_gui = [
        record["task_name"]
        for record in records
        if record["interface"] == "gui_only"
        and record["memory_split"] == "single_app_control"
    ]
    return {
        "schema_version": "causalcache.mobileworld.memory_split.v1",
        "upstream": {
            "repository": "https://github.com/Tongyi-MAI/MobileWorld",
            "revision": revision,
            "task_count": len(records),
        },
        "construction": {
            "source": "literal BaseTask.app_names and BaseTask.task_tags",
            "memory_candidate_rule": "app_count >= 2",
            "control_rule": "app_count == 1",
            "scope_note": (
                "Cross-app is a preregistered memory-challenge candidate, not a "
                "post-hoc claim that every selected task is memory-dependent."
            ),
            "gui_only_rule": (
                "exclude agent-mcp and agent-user-interaction task tags"
            ),
        },
        "counts": {
            "all_tasks": len(records),
            "interfaces": dict(sorted(interfaces.items())),
            "gui_only_tasks": len(gui_only),
            "gui_only_cross_app_memory_candidates": len(cross_app_gui),
            "gui_only_single_app_controls": len(single_app_gui),
            "all_cross_app_tasks": sum(
                record["memory_split"] == "cross_app_memory_candidate"
                for record in records
            ),
        },
        "benchmark_profiles": {
            "frozen_gui_owl_gui_only": gui_only,
            "memory_candidate": cross_app_gui,
            "single_app_control": single_app_gui,
        },
        "records": records,
        "records_sha256": _canonical_sha256(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobileworld-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-task-count", type=int, default=201)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(
        args.mobileworld_root.expanduser().resolve(),
        expected_revision=args.expected_revision,
        expected_task_count=args.expected_task_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
