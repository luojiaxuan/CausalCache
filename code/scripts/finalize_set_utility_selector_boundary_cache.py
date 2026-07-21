#!/usr/bin/env python3
"""Validate and publish a complete selector branch boundary cache."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from causalcache.set_utility_selector_boundary_cache import (
    finalize_selector_boundary_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--trainable-layer-count", type=int, default=4)
    parser.add_argument("--context-allowlist", type=Path)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.source_revision) is None:
        raise ValueError("source revision must be a full Git SHA")
    manifest = finalize_selector_boundary_cache(
        input_root=args.input_root,
        source_root=args.source_root,
        cache_root=args.cache_root,
        config_path=args.config,
        source_revision=args.source_revision,
        trainable_layer_count=args.trainable_layer_count,
        context_allowlist_path=args.context_allowlist,
    )
    print(
        json.dumps(
            {
                "content_sha256": manifest["content_sha256"],
                "context_count": manifest["context_count"],
                "shard_count": len(manifest["shards"]),
                "status": manifest["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
