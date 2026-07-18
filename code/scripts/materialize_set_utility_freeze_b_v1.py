#!/usr/bin/env python3
"""Materialize the policy-blind Freeze-B roster and query plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_freeze_b_v1 import materialize_freeze_b_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        default="code/configs/causalcache_set_utility_freeze_b_v1.json",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/set_utility_freeze_b_v1.json",
    )
    args = parser.parse_args()
    manifest, digest = materialize_freeze_b_manifest(
        repository_root=args.repository_root,
        config_relative_path=args.config,
        output_relative_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "trajectory_count": manifest["summary"]["trajectory_count"],
                "query_state_count": manifest["summary"]["query_state_count"],
                "manifest_sha256": digest,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
