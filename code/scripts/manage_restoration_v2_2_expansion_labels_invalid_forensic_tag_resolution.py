"""Manage the read-only P1 annotated-tag resolution child."""

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
from causalcache.restoration_v2_2_expansion_labels_invalid_forensic_tag_resolution import (
    ALLOWED_REMOTE_API_METHODS,
    CANONICAL_CONFIG_PATH,
    load_frozen_tag_resolution_contract,
    validate_clean_pushed_source,
    validate_parent_publication_state,
    validate_remote_tag_resolution,
)


def _add_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--tag-resolution-config",
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
    validate_contract = commands.add_parser("validate-contract")
    _add_contract_arguments(validate_contract)
    validate_local = commands.add_parser("validate-local-state")
    _add_contract_arguments(validate_local)
    validate_remote = commands.add_parser("validate-remote")
    _add_contract_arguments(validate_remote)
    validate_remote.add_argument(
        "--fresh-download-parent", type=Path, required=True
    )
    validate_remote.add_argument("--hf-token-file", type=Path, required=True)
    return parser


def _load(args: argparse.Namespace):
    contract = load_frozen_tag_resolution_contract(
        args.tag_resolution_config,
        repository_root=args.repository_root,
    )
    p0 = load_frozen_forensic_contract(
        args.p0_config,
        repository_root=args.repository_root,
    )
    if p0.sha256 != contract.p0_source["config_sha256"]:
        raise ValueError("P0 config SHA256 differs from tag-resolution contract")
    return contract, p0


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
        raise ValueError("HF token file changed while being read")
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
        raise ValueError("HF token file must contain exactly one nonempty token")
    return token


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    contract, p0 = _load(args)
    if args.command == "validate-contract":
        result = {
            "status": "VALID_SOURCE_ONLY_READ_ONLY_TAG_RESOLUTION_CONTRACT",
            "protocol_id": contract.data["protocol_id"],
            "tag_resolution_config_sha256": contract.sha256,
            "p0_config_sha256": p0.sha256,
            "allowed_hf_api_methods": list(ALLOWED_REMOTE_API_METHODS),
            "additional_remote_read": "hf_hub_download(force_download=True)",
            "remote_mutation_call_count": 0,
            "formal_label_loader_eligible": False,
            "network_access_performed": False,
        }
    elif args.command == "validate-local-state":
        source = validate_clean_pushed_source(
            repository_root=args.repository_root,
            contract=contract,
        )
        parent = validate_parent_publication_state(contract)
        result = {
            "status": "VALID_PARENT_P1_FAIL_CLOSED_LOCAL_STATE",
            "tag_resolution_config_sha256": contract.sha256,
            "tag_resolution_source": dict(source),
            "parent_claim_sha256": parent.claim_sha256,
            "parent_completion_seal_observed_absent": True,
            "remote_mutation_call_count": 0,
            "formal_label_loader_eligible": False,
            "network_access_performed": False,
        }
    elif args.command == "validate-remote":
        source = validate_clean_pushed_source(
            repository_root=args.repository_root,
            contract=contract,
        )
        # note (luojiaxuan): HF transport and the explicit token are reachable
        # only from this read-only remote-validation command.
        from huggingface_hub import HfApi, hf_hub_download

        token = _read_hf_token(args.hf_token_file)
        result = validate_remote_tag_resolution(
            api=HfApi(token=token),
            download_fn=partial(hf_hub_download, token=token),
            contract=contract,
            p0_contract=p0,
            source_identity=source,
            fresh_download_parent=args.fresh_download_parent,
        )
    else:
        raise AssertionError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
