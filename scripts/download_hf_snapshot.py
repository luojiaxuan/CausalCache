"""Download and verify a pinned public Hugging Face snapshot sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, specification: Mapping[str, Any]) -> str:
    expected_size = int(specification["size"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"size mismatch for {path.name}: {actual_size} != {expected_size}")
    actual_sha256 = file_sha256(path)
    expected_sha256 = specification.get("sha256")
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path.name}")
    return actual_sha256


def download_file(repo: str, revision: str, specification: Mapping[str, Any], output_dir: Path) -> str:
    relative_path = str(specification["path"])
    destination = output_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return verify_file(destination, specification)

    partial = destination.with_suffix(destination.suffix + ".part")
    url = (
        f"https://huggingface.co/{quote(repo, safe='/')}/resolve/"
        f"{quote(revision, safe='')}/{quote(relative_path, safe='/')}?download=true"
    )
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "5",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    actual_sha256 = verify_file(partial, specification)
    partial.replace(destination)
    return actual_sha256


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    repo = str(manifest["repo"])
    revision = str(manifest["revision"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    resolved_files = []
    for index, specification in enumerate(manifest["files"], start=1):
        print(f"[{index}/{len(manifest['files'])}] {specification['path']}", flush=True)
        sha256 = download_file(repo, revision, specification, args.output_dir)
        resolved_files.append(
            {
                "path": specification["path"],
                "size": specification["size"],
                "sha256": sha256,
            }
        )

    resolved = {"repo": repo, "revision": revision, "files": resolved_files}
    snapshot_path = args.output_dir / ".snapshot.json"
    snapshot_path.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(resolved_files), "snapshot": str(snapshot_path)}, indent=2))


if __name__ == "__main__":
    main()
