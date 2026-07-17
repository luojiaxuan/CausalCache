"""Validate, prepare, or publish the completed repaired label payload."""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections.abc import Sequence
from functools import partial
from pathlib import Path

from causalcache.restoration_v2_2_expansion_labels_scientific_repair_publication import (
    ALLOWED_REMOTE_STATES,
    CANONICAL_CONFIG_PATH,
    load_frozen_publication_contract,
    prepare_scientific_repair_publication,
    publish_scientific_repair_archive,
    validate_clean_pushed_source,
)


def _add_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--publication-config",
        type=Path,
        default=Path(CANONICAL_CONFIG_PATH),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-contract")
    _add_contract_arguments(validate)
    prepare = commands.add_parser("prepare")
    _add_contract_arguments(prepare)
    publish = commands.add_parser("publish")
    _add_contract_arguments(publish)
    publish.add_argument("--fresh-download-parent", type=Path, required=True)
    publish.add_argument("--hf-token-file", type=Path, required=True)
    return parser


def _load(args: argparse.Namespace):
    return load_frozen_publication_contract(
        args.publication_config,
        repository_root=args.repository_root,
    )


def _prepare(args: argparse.Namespace, contract):
    source = validate_clean_pushed_source(
        repository_root=args.repository_root,
        contract=contract,
    )
    return prepare_scientific_repair_publication(
        contract=contract,
        source_identity=source,
    )


def _read_hf_token(path: Path) -> str:
    if not path.is_absolute():
        raise ValueError("HF token path must be absolute")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) not in {0o400, 0o600}
        ):
            raise ValueError("HF token file must be regular mode 0400 or 0600")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 64 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError("HF token file is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    payload = b"".join(chunks)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or before.st_mode != after.st_mode
        or len(payload) != after.st_size
    ):
        raise ValueError("HF token file changed during its read")
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
    contract = _load(args)
    if args.command == "validate-contract":
        result = {
            "schema_version": "1.0.0",
            "status": "VALID_SOURCE_ONLY_REPAIRED_LABEL_PUBLICATION_CONTRACT_V1",
            "protocol_id": contract.data["protocol_id"],
            "publication_config_sha256": contract.sha256,
            "destination": dict(contract.destination),
            "allowed_remote_states": list(ALLOWED_REMOTE_STATES),
            "formal_label_loader_eligible": False,
            "gate_training_unlocked": False,
            "network_access_performed": False,
            "git_remote_verification_performed": False,
            "hf_network_access_performed": False,
        }
    elif args.command == "prepare":
        prepared = _prepare(args, contract)
        result = {
            "schema_version": "1.0.0",
            "status": "PREPARED_REPAIRED_LABEL_PUBLICATION_BYTES_V1",
            "archive_path": str(prepared.archive_path),
            "archive_sha256": prepared.archive_sha256,
            "archive_size_bytes": len(prepared.archive_bytes),
            "sidecar_sha256": prepared.sidecar_sha256,
            "producer_completion_sha256": prepared.producer_completion_sha256,
            "formal_label_loader_eligible": False,
            "gate_training_unlocked": False,
            "network_access_performed": True,
            "git_remote_verification_performed": True,
            "hf_network_access_performed": False,
        }
    elif args.command == "publish":
        prepared = _prepare(args, contract)
        # note (luojiaxuan): HF transport is imported only for the explicit
        # publication command. Prepare still performs its required Git
        # ls-remote check but cannot access or mutate the Hub destination.
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

        token = _read_hf_token(args.hf_token_file)
        result = publish_scientific_repair_archive(
            api=HfApi(token=token),
            download_fn=partial(hf_hub_download, token=token),
            operation_factory=CommitOperationAdd,
            contract=contract,
            prepared=prepared,
            fresh_download_parent=args.fresh_download_parent,
        )
    else:
        raise AssertionError(f"unknown publication command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
