"""Run the frozen independent multi-trajectory UI-TARS reference gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from causalcache.reference_gate import run_reference_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-tar", type=Path, required=True)
    parser.add_argument("--hardware-anchor-summary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-git-commit", required=True)
    parser.add_argument("--container-image-digest", required=True)
    return parser.parse_args()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError("output directory must be absent or empty")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    _prepare_output_dir(args.output_dir)
    repo_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    try:
        result = run_reference_gate(
            config_path=args.config.resolve(),
            dataset_tar=args.dataset_tar.resolve(),
            hardware_anchor_summary=args.hardware_anchor_summary.resolve(),
            model_dir=args.model_dir.resolve(),
            device=args.device,
            run_git_commit=args.run_git_commit,
            container_image_digest=args.container_image_digest,
            repo_root=repo_root,
            argv=sys.argv,
        )
        _write_json(args.output_dir / "summary.json", result)
        print(json.dumps({"outcome": result["outcome"]}, indent=2, sort_keys=True))
    except Exception as error:
        failure = {
            "schema_version": "0.1.0",
            "status": "INVALID",
            "failure_class": error.__class__.__name__,
            "message": str(error),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
            "traceback": traceback.format_exc(),
        }
        _write_json(args.output_dir / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
