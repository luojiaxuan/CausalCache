#!/usr/bin/env python3
"""Materialize all assignment-defined evaluation features without labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.restoration_v2_text_backend import prepare_image_bytes
from causalcache.set_utility_heldout_snapshot import (
    materialize_full_evaluation_feature_snapshot,
)


SOURCE_COLUMNS = (
    "decision_count",
    "history_events_json",
    "images",
    "ocr_records_json",
    "role",
    "source_id",
    "task_instruction",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--full-visual-token-root", type=Path, required=True)
    parser.add_argument("--frozen-state-inventory", type=Path, required=True)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--partial-from-available-visual-shards", action="store_true")
    args = parser.parse_args()
    try:
        from pyarrow import parquet as pq
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "full evaluation feature snapshot requires pyarrow and safetensors"
        ) from error

    manifest = materialize_full_evaluation_feature_snapshot(
        source_root=args.source_root,
        full_visual_token_root=args.full_visual_token_root,
        frozen_state_inventory_path=args.frozen_state_inventory,
        assignment_manifest_path=args.assignment_manifest,
        output_root=args.output_root,
        read_source_rows=lambda path: pq.read_table(
            path,
            columns=list(SOURCE_COLUMNS),
        ).to_pylist(),
        load_visual_tensors=lambda path: load_file(str(path), device="cpu"),
        prepare_resized_rgb=lambda payload: prepare_image_bytes(
            payload
        ).resized_rgb_bytes,
        partial_from_available_visual_shards=(
            args.partial_from_available_visual_shards
        ),
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
