"""Package the canonical spatial-reference audit v1 raw evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causalcache.spatial_reference_audit_v1_artifact import (
    package_spatial_reference_audit_evidence,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Package the canonical audit root and sibling attempt ledger as "
            "a new deterministic USTAR archive."
        )
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--global-attempt-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = package_spatial_reference_audit_evidence(
        repository_root=args.repository_root,
        config_path=args.config,
        source_git_commit=args.source_git_commit,
        audit_root=args.audit_root,
        global_attempt_ledger=args.global_attempt_ledger,
        output_archive=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
