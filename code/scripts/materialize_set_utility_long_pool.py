#!/usr/bin/env python3
"""Materialize the policy-blind long-trajectory discovery manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.data.guiodyssey_independent import (
    inspect_candidate,
    source_file_specs,
    verify_local_source_files,
)
from causalcache.set_utility_long_pool import (
    build_discovery_manifest,
    canonical_json_bytes,
    project_candidate,
    validate_discovery_config,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pinned_json(repository_root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = repository_root / str(record["path"])
    if not path.is_file() or _sha256_file(path) != record["sha256"]:
        raise ValueError(f"immutable JSON input drifted: {record['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"immutable JSON input is not an object: {record['path']}")
    return value


def _inspection_config(base: Mapping[str, Any], discovery: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    eligibility = discovery["eligibility_override"]
    result["eligibility"]["minimum_decisions_per_trajectory"] = eligibility[
        "minimum_decisions_per_trajectory"
    ]
    result["eligibility"]["maximum_decisions_per_trajectory"] = eligibility[
        "maximum_decisions_per_trajectory"
    ]
    result["selection"]["trajectory_salt"] = discovery["selection"]["salt"]
    return result


def _iter_rows(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise RuntimeError("long-pool discovery requires pyarrow") from error
    parquet = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet.iter_batches(batch_size=1):
        rows = batch.to_pylist()
        if len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise ValueError("GUIOdyssey parquet row decoding drifted")
        yield row_index, rows[0]
        row_index += 1


def materialize(
    *,
    repository_root: Path,
    config_path: Path,
    source_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    validate_discovery_config(config)
    inputs = config["immutable_inputs"]
    base = _load_pinned_json(repository_root, inputs["base_config"])
    source_manifest = _load_pinned_json(repository_root, inputs["source_file_manifest"])
    prior_artifact = _load_pinned_json(repository_root, inputs["prior_source_pool_artifact"])
    prior_pool = _load_pinned_json(repository_root, inputs["prior_eligible_pool_manifest"])
    if (
        prior_artifact["source_pool"]["rows"]
        != inputs["prior_source_pool_artifact"]["source_row_count"]
        or prior_artifact["source_pool"]["exclusion_counts"][
            "decision_count_above_maximum"
        ]
        != inputs["prior_source_pool_artifact"][
            "previous_decision_count_above_maximum"
        ]
    ):
        raise ValueError("prior source-pool accounting drifted")
    old_records = prior_pool["reconstruction"]["eligible_pool"]
    if len(old_records) != inputs["prior_eligible_pool_manifest"]["eligible_pool_count"]:
        raise ValueError("prior eligible-pool count drifted")
    old_source_ids = {str(record["source_id"]) for record in old_records}
    if len(old_source_ids) != len(old_records):
        raise ValueError("prior eligible-pool source ids are not unique")

    inspection = _inspection_config(base, config)
    specs = source_file_specs(inspection, source_manifest)
    verify_local_source_files(source_root, specs)
    candidates = []
    exclusion_counts: Counter[str] = Counter()
    seen_source_ids: set[str] = set()
    source_row_count = 0
    for spec in specs:
        path = source_root.joinpath(*PurePosixPath(spec.transport_file).parts)
        for row_index, row in _iter_rows(path):
            source_row_count += 1
            result = inspect_candidate(
                row,
                transport_file=spec.transport_file,
                transport_row_index=row_index,
                config=inspection,
            )
            if result.source_id is not None:
                if result.source_id in seen_source_ids:
                    raise ValueError("source_id is duplicated across pinned shards")
                seen_source_ids.add(result.source_id)
            if result.candidate is None:
                if result.exclusion_reason is None:
                    raise RuntimeError("excluded source row has no reason")
                exclusion_counts[result.exclusion_reason.value] += 1
            else:
                candidates.append(
                    project_candidate(
                        result.candidate,
                        salt=config["selection"]["salt"],
                    )
                )
    expected_rows = inputs["prior_source_pool_artifact"]["source_row_count"]
    if source_row_count != expected_rows:
        raise ValueError("pinned GUIOdyssey source row count drifted")
    overlap = old_source_ids & {candidate.source_id for candidate in candidates}
    if overlap:
        raise ValueError("new long pool overlaps the prior eligible pool")
    manifest = build_discovery_manifest(
        candidates=tuple(candidates),
        exclusion_counts=exclusion_counts,
        source_row_count=source_row_count,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        base_config_sha256=inputs["base_config"]["sha256"],
        source_manifest_sha256=inputs["source_file_manifest"]["sha256"],
    )
    manifest["overlap_proof"] = {
        "prior_eligible_pool_count": len(old_source_ids),
        "new_long_pool_count": len(candidates),
        "intersection_count": 0,
        "intersection_source_ids_sha256": hashlib.sha256(
            canonical_json_bytes([])
        ).hexdigest(),
    }
    if output_path.exists():
        raise FileExistsError("long-pool discovery output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize(
        repository_root=args.repository_root.resolve(),
        config_path=args.config.resolve(),
        source_root=args.source_root.resolve(),
        output_path=args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "candidate_count": manifest["selection"]["candidate_count"],
                "length_strata": manifest["pool"]["summary"][
                    "length_stratum_counts"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
