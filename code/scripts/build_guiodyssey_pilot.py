"""Build a deterministic, shard-oriented GUIOdyssey pilot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from causalcache.data.guiodyssey import build_pilot_manifest, write_pilot_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--upstream-repo", required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--transport-repo", required=True)
    parser.add_argument("--transport-revision", required=True)
    parser.add_argument("--transport-file", required=True)
    parser.add_argument("--transport-file-sha256", required=True)
    parser.add_argument("--hf-destination", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def data_card(manifest: dict) -> str:
    source = manifest["source"]
    trajectory = manifest["trajectory"]
    return f"""---
license: cc-by-4.0
pretty_name: CausalCache GUIOdyssey Pilot Mobile
task_categories:
- image-text-to-text
---

# CausalCache GUIOdyssey Pilot Mobile

该私有 pilot dataset 用于验证 CausalCache 的 mixed-fidelity policy rerun、validated teacher coverage 与 restoration attribution 接口，不是论文主实验全集。

## Source and Provenance

- Upstream: `https://huggingface.co/datasets/{source['upstream_repo']}`
- Upstream revision: `{source['upstream_revision']}`
- Transport: `https://huggingface.co/datasets/{source['transport_repo']}`
- Transport revision: `{source['transport_revision']}`
- Transport file: `{source['transport_file']}`
- Transport file SHA256: `{source['transport_file_sha256']}`
- Transport row: `{source['transport_row_index']}`
- Original trajectory: `{trajectory['source_id']}`
- License: CC BY 4.0

## Layout

```text
README.md
manifest.json
data/guiodyssey-pilot-00000.tar
```

tar shard 内包含 `manifest.json` 和 `{len(trajectory['steps'])}` 张原始 trajectory screenshots。manifest 提供 instruction、action tool call、规范化 executable action、固定五字段 low-fidelity event、decision/history mapping 和 source revisions。

## Generation

从 CausalCache Git 仓库运行 `code/scripts/build_guiodyssey_pilot.py`，并显式传入上述 repo、revision、row index 与输出路径。policy-visible summary 不包含 expert inline reasoning。

## Intended Use

仅用于接口与小规模 attribution pilot。正式结果必须扩大 trajectory/task 覆盖，并报告 validated teacher coverage；不能把该单 trajectory 结果当作 AndroidWorld closed-loop 证据。
"""


def main() -> None:
    args = parse_args()
    parquet = pq.ParquetFile(args.source_parquet)
    if not 0 <= args.row_index < parquet.metadata.num_rows:
        raise ValueError("row-index is outside the source shard")
    table = parquet.read()
    row = table.slice(args.row_index, 1).to_pylist()[0]
    manifest, images = build_pilot_manifest(
        row,
        row_index=args.row_index,
        upstream_repo=args.upstream_repo,
        upstream_revision=args.upstream_revision,
        transport_repo=args.transport_repo,
        transport_revision=args.transport_revision,
        transport_file=args.transport_file,
        transport_file_sha256=args.transport_file_sha256,
        hf_destination=args.hf_destination,
    )
    write_pilot_dataset(args.output_dir, manifest, images)
    (args.output_dir / "README.md").write_text(data_card(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "source_id": manifest["trajectory"]["source_id"],
                "steps": len(manifest["trajectory"]["steps"]),
                "events": len(manifest["trajectory"]["events"]),
                "decisions": len(manifest["trajectory"]["decisions"]),
                "images": len(images),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
