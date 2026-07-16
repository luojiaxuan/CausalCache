"""Validate the source-only policy-vision v2 GPU UUID repair contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causalcache.restoration_v2_2_policy_vision_v2_contract import (
    validate_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "configs/"
            "causalcache_restoration_v2_2_policy_vision_baseline_"
            "v2_gpu_uuid_repair.json"
        ),
    )
    parser.add_argument("--repository-root", type=Path, default=Path(".."))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = validate_contract(
        args.contract,
        repository_root=args.repository_root,
        require_output_absent=True,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
