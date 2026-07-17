"""Bootstrap the formal no-GPU expansion-label scientific repair."""

from __future__ import annotations

import argparse
import importlib.abc
import json
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path


IMPORT_GUARD_MARKER = (
    "causalcache_formal_no_model_framework_import_guard_v1"
)
BLOCKED_IMPORT_ROOTS = (
    "accelerate",
    "bitsandbytes",
    "flash_attn",
    "flax",
    "jax",
    "sglang",
    "tensorflow",
    "torch",
    "torchvision",
    "transformers",
    "triton",
    "vllm",
    "xformers",
)


class NoModelFrameworkImportGuard(importlib.abc.MetaPathFinder):
    causalcache_guard_marker = IMPORT_GUARD_MARKER
    blocked_roots = BLOCKED_IMPORT_ROOTS

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        root = fullname.partition(".")[0]
        if root in self.blocked_roots:
            raise ImportError(
                f"model-framework import forbidden in formal CPU repair: {fullname}"
            )
        return None


def install_import_guard() -> NoModelFrameworkImportGuard:
    imported = sorted(
        name
        for name in sys.modules
        if any(
            name == root or name.startswith(root + ".")
            for root in BLOCKED_IMPORT_ROOTS
        )
    )
    if imported:
        raise RuntimeError("a forbidden model framework was imported before bootstrap")
    existing = [
        finder
        for finder in sys.meta_path
        if getattr(finder, "causalcache_guard_marker", None) == IMPORT_GUARD_MARKER
    ]
    if existing:
        raise RuntimeError("a pre-existing formal import guard is forbidden")
    guard = NoModelFrameworkImportGuard()
    sys.meta_path.insert(0, guard)
    return guard


def remove_import_guard(guard: NoModelFrameworkImportGuard) -> None:
    if guard in sys.meta_path:
        sys.meta_path.remove(guard)


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate-contract", "run", "validate"))
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "code/configs/causalcache_restoration_v2_2_"
            "expansion_labels_scientific_repair_runner_v1.json"
        ),
    )
    parser.add_argument("--source-git-commit")
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--hf-token-file", type=Path)
    parser.add_argument("--fresh-download-parent", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    raw_argv = list(sys.argv) if argv is None else [__name__, *argv]
    args = _parser().parse_args(argv)
    if args.mode != "validate-contract" and __package__:
        raise ValueError("formal mode requires direct bootstrap script execution")
    guard = install_import_guard()
    try:
        from causalcache.restoration_v2_2_expansion_labels_scientific_repair_runner import (
            execute_formal_scientific_repair,
            load_frozen_runner_contract,
            validate_import_guard,
        )

        root = args.repository_root.resolve()
        config = args.contract
        if not config.is_absolute():
            config = root / config
        contract = load_frozen_runner_contract(config, repository_root=root)
        validate_import_guard(contract)
        if args.mode == "validate-contract":
            result = {
                "schema_version": "1.0.0",
                "status": "VALID_SOURCE_ONLY_NO_GPU_SCIENTIFIC_REPAIR_RUNNER_V1",
                "protocol_id": contract.data["protocol_id"],
                "runner_config_sha256": contract.sha256,
                "blocked_import_roots": list(BLOCKED_IMPORT_ROOTS),
                "pil_allowed": True,
                "network_access_count": 0,
                "gpu_or_model_operation_count": 0,
                "formal_run_performed": False,
            }
        else:
            required = {
                "--source-git-commit": args.source_git_commit,
                "--launch-receipt": args.launch_receipt,
                "--hf-token-file": args.hf_token_file,
                "--fresh-download-parent": args.fresh_download_parent,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"formal mode missing required arguments: {missing}")
            token = _read_hf_token(args.hf_token_file.resolve())
            result = execute_formal_scientific_repair(
                contract,
                mode=args.mode,
                expected_source_git_commit=args.source_git_commit,
                receipt_path=args.launch_receipt.resolve(),
                hf_token=token,
                fresh_download_parent=args.fresh_download_parent.resolve(),
                argv=raw_argv,
            )
        validate_import_guard(contract)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        remove_import_guard(guard)


if __name__ == "__main__":
    main()
