"""Build the frozen independent multi-trajectory GUIOdyssey artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

from causalcache.data.guiodyssey_independent import (
    build_independent_manifest,
    canonical_json_bytes,
    inspect_candidate,
    select_splits,
    sha256_file,
    source_file_specs,
    verify_local_source_files,
    write_independent_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-file-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _source_path(source_root: Path, transport_file: str) -> Path:
    return source_root.joinpath(*PurePosixPath(transport_file).parts)


def _iter_rows(parquet_file: Any) -> Iterator[tuple[int, Mapping[str, Any]]]:
    row_index = 0
    for row_group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_index)
        for row in table.to_pylist():
            yield row_index, row
            row_index += 1
    if row_index != parquet_file.metadata.num_rows:
        raise RuntimeError("Parquet row iteration did not match metadata.num_rows")


def _data_card(manifest: Mapping[str, Any], *, shard: str) -> str:
    source = manifest["source"]
    reference = manifest["splits"]["reference_gate"]
    oracle = manifest["splits"]["oracle_pilot"]
    return f"""---
license: cc-by-4.0
pretty_name: CausalCache GUIOdyssey Independent Mobile
task_categories:
- image-text-to-text
---

# CausalCache GUIOdyssey Independent Mobile

该私有 artifact 是 policy-blind、确定性选择的 independent reference gate 与 oracle pilot 数据。

## Provenance

- Upstream: `{source['upstream_repo']}@{source['upstream_revision']}`
- Transport: `{source['transport_repo']}@{source['transport_revision']}`
- Protocol: `{manifest['protocol_id']}`
- Protocol config SHA256: `{source['protocol_config_sha256']}`
- Source-file manifest SHA256: `{source['source_file_manifest_sha256']}`
- Eligible-pool SHA256: `{manifest['selection']['eligible_pool_sha256']}`

## Frozen splits

- `reference_gate`: {reference['trajectory_count']} trajectories / {reference['decision_count']} decisions
- `oracle_pilot`: {oracle['trajectory_count']} trajectories / {oracle['decision_count']} decisions

## Layout

```text
README.md
manifest.json
{shard}
```

tar shard 内含 manifest 与原始 screenshots。上传 Hugging Face 后，必须先将 immutable revision、
manifest SHA256 与 shard SHA256 写回 Git 的 frozen config，之后才允许 policy inference。
"""


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory must be absent or empty")
    config_bytes = args.config.read_bytes()
    source_manifest_bytes = args.source_file_manifest.read_bytes()
    config = json.loads(config_bytes)
    source_file_manifest = json.loads(source_manifest_bytes)
    specs = source_file_specs(config, source_file_manifest)

    verify_local_source_files(args.source_root, specs)

    import pyarrow.parquet as pq

    candidates = []
    exclusion_counts: Counter[str] = Counter()
    source_row_counts: dict[str, int] = {}
    seen_source_ids: dict[str, tuple[str, int]] = {}
    total_source_rows = 0
    for spec in specs:
        parquet = pq.ParquetFile(_source_path(args.source_root, spec.transport_file))
        source_row_counts[spec.transport_file] = int(parquet.metadata.num_rows)
        total_source_rows += int(parquet.metadata.num_rows)
        for row_index, row in _iter_rows(parquet):
            inspection = inspect_candidate(
                row,
                transport_file=spec.transport_file,
                transport_row_index=row_index,
                config=config,
            )
            if inspection.source_id is not None and bool(
                config["eligibility"]["require_unique_source_id"]
            ):
                previous = seen_source_ids.get(inspection.source_id)
                if previous is not None:
                    raise ValueError(
                        "duplicate source_id across frozen source pool: "
                        f"{inspection.source_id} at {previous} and "
                        f"{(spec.transport_file, row_index)}"
                    )
                seen_source_ids[inspection.source_id] = (spec.transport_file, row_index)
            if inspection.candidate is None:
                if inspection.exclusion_reason is None:
                    raise RuntimeError("excluded row is missing its reason")
                exclusion_counts[inspection.exclusion_reason.value] += 1
            else:
                candidates.append(inspection.candidate)

    if total_source_rows != sum(source_row_counts.values()):
        raise RuntimeError("source row accounting mismatch")
    selected = select_splits(candidates, config=config)
    selected_keys = {
        (candidate.transport_file, candidate.transport_row_index)
        for candidate in selected.reference_gate + selected.oracle_pilot
    }
    keys_by_file: dict[str, set[int]] = defaultdict(set)
    for transport_file, row_index in selected_keys:
        keys_by_file[transport_file].add(row_index)

    selected_rows: dict[tuple[str, int], Mapping[str, Any]] = {}
    for spec in specs:
        wanted = keys_by_file.get(spec.transport_file, set())
        if not wanted:
            continue
        parquet = pq.ParquetFile(_source_path(args.source_root, spec.transport_file))
        for row_index, row in _iter_rows(parquet):
            if row_index in wanted:
                selected_rows[(spec.transport_file, row_index)] = row
        if len([key for key in selected_rows if key[0] == spec.transport_file]) != len(wanted):
            raise RuntimeError(f"failed to reload every selected row from {spec.transport_file}")

    manifest, images = build_independent_manifest(
        selected=selected,
        selected_rows=selected_rows,
        config=config,
        source_specs=specs,
        source_row_counts=source_row_counts,
        exclusion_counts=exclusion_counts,
        total_source_rows=total_source_rows,
        protocol_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        source_file_manifest_sha256=hashlib.sha256(source_manifest_bytes).hexdigest(),
    )
    shard = str(config["artifact"]["shard"])
    tar_path = write_independent_dataset(
        args.output_dir,
        manifest=manifest,
        image_payloads=images,
        shard_relative_path=shard,
    )
    (args.output_dir / "README.md").write_text(
        _data_card(manifest, shard=shard),
        encoding="utf-8",
    )
    manifest_path = args.output_dir / "manifest.json"
    summary = {
        "outcome": "ARTIFACT_BUILT",
        "output_dir": str(args.output_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "shard_sha256": sha256_file(tar_path),
        "source_rows": total_source_rows,
        "eligible_trajectories": len(selected.eligible_candidates),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "splits": manifest["splits"],
        "images": len(images),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
