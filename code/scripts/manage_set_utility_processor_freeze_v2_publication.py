#!/usr/bin/env python3
"""Publish or read-only validate the processor-v2 immutable HF artifact."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from causalcache.set_utility_processor_publication_v2 import (
    publish_processor_v2_artifact,
    validate_processor_v2_publication,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--git-summary", type=Path, required=True)
    parser.add_argument("--git-card", type=Path, required=True)
    parser.add_argument("--expected-result-summary-sha256", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-tag", required=True)
    parser.add_argument("--hf-prefix", required=True)
    parser.add_argument("--hf-token-file", type=Path, required=True)
    parser.add_argument("--fresh-download-parent", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    _common(commands.add_parser("publish"))
    _common(commands.add_parser("validate-only"))
    return parser


def _read_token(path: Path) -> str:
    if not path.is_absolute() or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("HF token path must be absolute and support O_NOFOLLOW")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) not in {0o400, 0o600}
        ):
            raise ValueError("HF token file must be regular mode 0400 or 0600")
        blocks: list[bytes] = []
        while block := os.read(descriptor, 64 * 1024):
            blocks.append(block)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError("HF token file is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    payload = b"".join(blocks)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(payload) != after.st_size
    ):
        raise ValueError("HF token file changed during read")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("HF token file must contain UTF-8") from error
    token = text[:-1] if text.endswith("\n") else text
    if (
        not token
        or text not in {token, token + "\n"}
        or any(character.isspace() for character in token)
    ):
        raise ValueError("HF token file must contain exactly one token")
    return token


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    token = _read_token(args.hf_token_file)
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    common = {
        "api": HfApi(token=token),
        "formal_root": args.formal_root,
        "git_summary": args.git_summary,
        "git_card": args.git_card,
        "expected_result_summary_sha256": args.expected_result_summary_sha256,
        "hf_repo": args.hf_repo,
        "hf_tag": args.hf_tag,
        "hf_prefix": args.hf_prefix,
        "fresh_download_parent": args.fresh_download_parent,
        "receipt_path": args.receipt,
    }
    if args.command == "publish":
        result = publish_processor_v2_artifact(
            **common,
            download_fn=partial(hf_hub_download, token=token),
            operation_factory=CommitOperationAdd,
        )
    elif args.command == "validate-only":
        result = validate_processor_v2_publication(**common)
    else:
        raise AssertionError(f"unexpected command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
