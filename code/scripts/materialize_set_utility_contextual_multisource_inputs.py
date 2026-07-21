#!/usr/bin/env python3
"""Materialize versioned contextual inputs from several train label sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from causalcache.set_utility_contextual_multisource_enrichment import (
    ContextualEnrichmentSource,
    materialize_contextual_multisource_enriched_inputs,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or re.fullmatch(r"[a-z0-9_]+", name) is None or not raw_path:
        raise argparse.ArgumentTypeError("source path must be name=path")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-input-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--schedule-root", type=_named_path, action="append", required=True)
    parser.add_argument("--label-root", type=_named_path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--duplicate-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    config = _read_json(args.config)
    declared = config["data_sources"]
    schedule_roots = dict(args.schedule_root)
    if len(schedule_roots) != len(args.schedule_root) or set(schedule_roots) != set(declared):
        raise ValueError("schedule roots do not match the versioned source config")
    label_roots: dict[str, list[Path]] = defaultdict(list)
    for name, path in args.label_root:
        label_roots[name].append(path)
    if set(label_roots) != set(declared):
        raise ValueError("label roots do not match the versioned source config")
    sources = tuple(
        ContextualEnrichmentSource(
            name=name,
            schedule_root=schedule_roots[name].resolve(),
            label_roots=tuple(path.resolve() for path in label_roots[name]),
            expected_schedule_content_sha256=value["schedule_content_sha256"],
            expected_schedule_config_sha256=value["schedule_config_sha256"],
            expected_label_scientific_config_sha256=value[
                "label_scientific_config_sha256"
            ],
            expected_label_execution_config_sha256=value[
                "label_execution_config_sha256"
            ],
        )
        for name, value in sorted(declared.items())
    )
    manifest = materialize_contextual_multisource_enriched_inputs(
        base_input_root=args.base_input_root.resolve(),
        sources=sources,
        output_root=args.output_root.resolve(),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        duplicate_tolerance=args.duplicate_tolerance,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
