"""Package external v2.1 processor evidence as a lightweight Git manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causalcache.restoration_v2_1_processor_audit import (
    CANONICAL_EVIDENCE_ARTIFACT_PATH,
    CANONICAL_EVIDENCE_HF_REPO,
    build_processor_evidence_artifact_manifest,
    canonical_json_bytes,
    strict_json_object,
    validate_restoration_v2_1_processor_audit,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bind a private-HF raw processor audit to a small Git manifest."
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--raw-evidence", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-immutable-revision", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--current-git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def package_evidence(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.repository_root).resolve()
    output = Path(args.output).resolve()
    canonical_output = (root / CANONICAL_EVIDENCE_ARTIFACT_PATH).resolve()
    if output != canonical_output:
        raise ValueError(
            f"--output must be the canonical lightweight manifest: {canonical_output}"
        )
    if output.exists():
        raise FileExistsError(f"processor evidence manifest already exists: {output}")
    if args.hf_repo != CANONICAL_EVIDENCE_HF_REPO:
        raise ValueError(f"--hf-repo must be {CANONICAL_EVIDENCE_HF_REPO}")
    raw = strict_json_object(Path(args.raw_evidence), label="raw processor evidence")
    validation = validate_restoration_v2_1_processor_audit(
        raw,
        repository_root=root,
        current_git_commit=args.current_git_commit,
        mode="generation",
    )
    manifest = build_processor_evidence_artifact_manifest(
        raw,
        raw_evidence_path=args.raw_evidence,
        hf_repo=args.hf_repo,
        hf_immutable_revision=args.hf_immutable_revision,
        hf_path=args.hf_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        destination.write(canonical_json_bytes(manifest) + b"\n")
    return {
        "status": "PACKAGED_RESTORATION_V2_1_PROCESSOR_EVIDENCE",
        "output": str(output),
        "hf_artifact": manifest["hf_artifact"],
        "raw_evidence": manifest["raw_evidence"],
        "validated_prompt_count": validation["prompt_count"],
        "evidence_git_commit": validation["evidence_git_commit"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    result = package_evidence(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
