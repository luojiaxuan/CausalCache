"""Validate, prepare, or publish the P1 private-HF forensic artifact."""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections.abc import Sequence
from functools import partial
from pathlib import Path

from causalcache.restoration_v2_2_expansion_labels_invalid_forensic import (
    CANONICAL_CONFIG_PATH as P0_CANONICAL_CONFIG_PATH,
    load_frozen_forensic_contract,
)
from causalcache.restoration_v2_2_expansion_labels_invalid_forensic_publication import (
    ALLOWED_REMOTE_STATES,
    CANONICAL_CONFIG_PATH,
    load_frozen_publication_contract,
    prepare_invalid_forensic_publication,
    publish_invalid_forensic_archive,
    validate_clean_pushed_source,
)


def _add_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--publication-config",
        type=Path,
        default=Path(CANONICAL_CONFIG_PATH),
    )
    parser.add_argument(
        "--p0-config",
        type=Path,
        default=Path(P0_CANONICAL_CONFIG_PATH),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-contract")
    _add_contract_arguments(validate)

    prepare = commands.add_parser("prepare")
    _add_contract_arguments(prepare)
    prepare.add_argument("--archive", type=Path, required=True)

    publish = commands.add_parser("publish")
    _add_contract_arguments(publish)
    publish.add_argument("--archive", type=Path, required=True)
    publish.add_argument("--fresh-download-parent", type=Path, required=True)
    publish.add_argument("--hf-token-file", type=Path, required=True)
    return parser


def _load(args: argparse.Namespace):
    publication = load_frozen_publication_contract(
        args.publication_config,
        repository_root=args.repository_root,
    )
    p0 = load_frozen_forensic_contract(
        args.p0_config,
        repository_root=args.repository_root,
    )
    if p0.sha256 != publication.p0_source["config_sha256"]:
        raise ValueError("P0 config SHA256 differs from P1 publication contract")
    return publication, p0


def _prepare(args: argparse.Namespace, publication, p0):
    source = validate_clean_pushed_source(
        repository_root=args.repository_root,
        contract=publication,
    )
    return prepare_invalid_forensic_publication(
        archive_path=args.archive,
        p0_contract=p0,
        contract=publication,
        source_identity=source,
    )


def _read_hf_token(path: Path) -> str:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("HF token path must be a regular non-symlink file")
        if stat.S_IMODE(before.st_mode) not in {0o400, 0o600}:
            raise ValueError("HF token file mode must be 0400 or 0600")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 64 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError("HF token file is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or before.st_mode != after.st_mode
    ):
        raise ValueError("HF token file changed while being read")
    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("HF token file must contain UTF-8") from error
    token = text[:-1] if text.endswith("\n") else text
    if (
        not token
        or text not in {token, token + "\n"}
        or any(character.isspace() for character in token)
    ):
        raise ValueError("HF token file must contain exactly one nonempty token")
    return token


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    publication, p0 = _load(args)
    if args.command == "validate-contract":
        result = {
            "status": "VALID_SOURCE_ONLY_INVALID_FORENSIC_PUBLICATION_CONTRACT",
            "protocol_id": publication.data["protocol_id"],
            "publication_config_sha256": publication.sha256,
            "p0_config_sha256": p0.sha256,
            "destination": dict(publication.destination),
            "allowed_remote_states": list(ALLOWED_REMOTE_STATES),
            "formal_label_loader_eligible": False,
            "network_access_performed": False,
        }
    elif args.command == "prepare":
        prepared = _prepare(args, publication, p0)
        result = {
            "status": "PREPARED_INVALID_FORENSIC_PUBLICATION_BYTES",
            "archive_path": str(prepared.archive_path),
            "archive_sha256": prepared.archive_sha256,
            "archive_size_bytes": len(prepared.archive_bytes),
            "sidecar_sha256": prepared.sidecar_sha256,
            "tree_inventory_sha256": prepared.sidecar["source_archive"][
                "tree_inventory_sha256"
            ],
            "publication_source_git_commit": prepared.source_identity[
                "publication_source_git_commit"
            ],
            "formal_label_loader_eligible": False,
            "network_access_performed": False,
        }
    elif args.command == "publish":
        prepared = _prepare(args, publication, p0)
        # note (luojiaxuan): Importing the HF transport only for the explicit
        # publish command keeps validation and preparation strictly offline.
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

        token = _read_hf_token(args.hf_token_file)
        result = publish_invalid_forensic_archive(
            api=HfApi(token=token),
            download_fn=partial(hf_hub_download, token=token),
            operation_factory=CommitOperationAdd,
            contract=publication,
            p0_contract=p0,
            prepared=prepared,
            fresh_download_parent=args.fresh_download_parent,
        )
    else:
        raise AssertionError(f"unknown publication command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
