#!/usr/bin/env python3
"""Build the frozen long-horizon label-blind selector seal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "code"))

from causalcache.long_horizon_contract import LongHorizonContract  # noqa: E402
from causalcache.long_horizon_contract import RUNNER_FREEZE_B_PATH  # noqa: E402
from causalcache.long_horizon_prepare import (  # noqa: E402
    build_selector_seal_artifact,
)
from scripts.build_long_horizon_substrate import verify_pushed_checkout  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh-load the verified substrate and frozen model payloads, then "
            "seal all label-blind B=2/B=4 selections."
        )
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--substrate-dir", type=Path, required=True)
    parser.add_argument("--substrate-revision", required=True)
    parser.add_argument("--formal-model-dir", type=Path, required=True)
    parser.add_argument("--v4-model-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify_pushed_checkout(REPOSITORY_ROOT, args.git_revision)
    if (REPOSITORY_ROOT / RUNNER_FREEZE_B_PATH).exists():
        raise ValueError("selector seal must be frozen before the Execution-B runner")
    contract = LongHorizonContract.load(
        args.contract,
        repository_root=REPOSITORY_ROOT,
        require_runner_absent=False,
    )
    result = build_selector_seal_artifact(
        contract=contract,
        substrate_dir=args.substrate_dir,
        substrate_revision=args.substrate_revision,
        formal_model_dir=args.formal_model_dir,
        v4_model_dir=args.v4_model_dir,
        random_seed=args.random_seed,
        git_revision=args.git_revision,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
