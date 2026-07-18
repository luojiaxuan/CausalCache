#!/usr/bin/env python3
"""Materialize the offline census from a separately committed execution contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.data.guiodyssey_independent import SourceFileSpec, sha256_file
from causalcache.set_utility_full_pool import (
    build_full_pool_census,
    parse_consumed_ledger,
    validate_complete_source_manifest,
)
from causalcache.set_utility_full_pool_contract import load_execution_contract
from causalcache.set_utility_full_pool_inventory_v1 import canonical_json_bytes


def _source_path(source_root: Path, transport_file: str) -> Path:
    path = source_root.joinpath(*PurePosixPath(transport_file).parts)
    try:
        path.resolve().relative_to(source_root.resolve())
    except ValueError as error:
        raise ValueError("source shard path escaped source_root") from error
    return path


class ParquetRowFactory:
    """Verify and stream one complete source shard at a time."""

    def __init__(self, source_root: Path, *, batch_size: int = 32) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("parquet batch size must be a positive integer")
        self.source_root = source_root.resolve()
        self.batch_size = batch_size

    def __call__(
        self, spec: SourceFileSpec
    ) -> Iterator[tuple[int, Mapping[str, Any]]]:
        path = _source_path(self.source_root, spec.transport_file)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing pinned source file: {spec.transport_file}")
        if path.stat().st_size != spec.size_bytes:
            raise ValueError(f"source file size mismatch: {spec.transport_file}")
        if sha256_file(path) != spec.sha256:
            raise ValueError(f"source file SHA256 mismatch: {spec.transport_file}")
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as error:
            raise RuntimeError("full-pool census requires pyarrow") from error
        parquet = pq.ParquetFile(path)
        row_index = 0
        for batch in parquet.iter_batches(batch_size=self.batch_size):
            for row in batch.to_pylist():
                if not isinstance(row, Mapping):
                    raise TypeError("GUIOdyssey parquet row must decode to a mapping")
                yield row_index, row
                row_index += 1
        if row_index != parquet.metadata.num_rows:
            raise RuntimeError("Parquet row iteration did not match metadata.num_rows")


def materialize(
    *,
    repository_root: Path,
    execution_config_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    contract = load_execution_contract(
        repository_root=repository_root,
        execution_config_path=execution_config_path,
    )
    output_path = contract.output_path
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("full-pool census output already exists")
    base_config = contract.bound_json["base_parser_config"]
    source_manifest = contract.bound_json["p1_inventory_manifest"]
    ledger_manifest = contract.bound_json["consumed_ledger"]
    validate_complete_source_manifest(source_manifest, base_config=base_config)
    consumed = parse_consumed_ledger(
        ledger_manifest,
        manifest_sha256=contract.binding_sha256("consumed_ledger"),
    )
    science = contract.data["science"]
    manifest = build_full_pool_census(
        base_config=base_config,
        source_manifest=source_manifest,
        row_iterator_factory=ParquetRowFactory(
            source_root,
            batch_size=int(contract.data["infrastructure"]["parquet_batch_size"]),
        ),
        base_config_sha256=contract.binding_sha256("base_parser_config"),
        source_manifest_sha256=contract.binding_sha256("p1_inventory_manifest"),
        consumed_ledger=consumed,
        format_safety_maximum_decisions=science[
            "format_safety_maximum_decisions"
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(canonical_json_bytes(manifest) + b"\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize(
        repository_root=args.repository_root.resolve(),
        execution_config_path=args.execution_config,
        source_root=args.source_root.resolve(),
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "source_files": manifest["source"]["file_count"],
                "source_rows": manifest["source"]["total_rows"],
                "candidate_count": manifest["selection"]["candidate_count"],
                "consumed_identity_count": manifest["consumed_ledger"][
                    "consumed_identity_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
