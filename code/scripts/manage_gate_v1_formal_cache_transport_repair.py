#!/usr/bin/env python3
"""Execute only the isolated formal-58 transport-identity repair protocol."""

from __future__ import annotations

import argparse
import importlib.abc
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any


FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "accelerate",
        "bitsandbytes",
        "flash_attn",
        "jax",
        "sglang",
        "tensorflow",
        "torch",
        "torchvision",
        "transformers",
        "triton",
        "vllm",
        "xformers",
    }
)


class _ForbiddenImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: object | None = None,
    ) -> None:
        del path, target
        if fullname.partition(".")[0] in FORBIDDEN_IMPORT_ROOTS:
            raise ImportError(f"model/GPU framework import is forbidden: {fullname}")
        return None


# note (luojiaxuan): A direct CLI installs this before all project and Hub
# imports. Unit-test imports must not leak a process-global import hook.
_IMPORT_GUARD = _ForbiddenImportGuard() if __name__ == "__main__" else None
if _IMPORT_GUARD is not None:
    sys.meta_path.insert(0, _IMPORT_GUARD)

from causalcache import gate_v1_formal_cache as cache_core  # noqa: E402
from causalcache.gate_v1_formal_cache_runner import (  # noqa: E402
    execute_formal_cache,
    materialize_runner_freeze,
    validate_source_a,
)
from causalcache.gate_v1_formal_cache_transport_repair_contract import (  # noqa: E402
    CANONICAL_CONFIG_PATH,
    load_frozen_transport_repair_contract,
    validate_failed_v1_attempt,
    validate_transport_repair_source_only_contract,
)
from scripts import manage_gate_v1_formal_cache as base_manager  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def source_parser(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--repository-root", type=Path, required=True)
        child.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
        child.add_argument("--source-a-git-commit")
        return child

    source_parser("validate-source")
    source_parser("materialize-runner-freeze")

    for name in ("run", "validate"):
        child = subparsers.add_parser(name)
        child.add_argument("--repository-root", type=Path, required=True)
        child.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
        child.add_argument("--execution-b-git-commit", required=True)
        child.add_argument("--hf-token-file", type=Path, required=True)
        child.add_argument("--data-root", type=Path, required=True)
        child.add_argument("--fresh-download-parent", type=Path, required=True)
    return parser


def _repair_cache_api() -> SimpleNamespace:
    """Expose the base download schema with repair-only semantic builders."""

    names = (
        "DOWNLOAD_KEYS",
        "EXPANSION_FEATURE_MANIFEST",
        "EXPANSION_FEATURE_OCR",
        "EXPANSION_FEATURE_TRAJECTORIES",
        "EXPANSION_LABEL_ARCHIVE",
        "EXPANSION_LABEL_SIDECAR",
        "FEATURE_SOURCE_KEYS",
        "LABEL_SOURCE_KEYS",
        "LEGACY_FEATURE_MANIFEST",
        "LEGACY_FEATURE_OCR",
        "LEGACY_FEATURE_TRAJECTORIES",
        "LEGACY_LABEL_ARCHIVE",
        "read_feature_cache",
        "read_label_cache",
    )
    values = {name: getattr(cache_core, name) for name in names}
    values.update(
        {
            "build_formal_feature_cache": (
                cache_core.build_formal_feature_cache_transport_repair_v1
            ),
            "build_formal_label_cache": (
                cache_core.build_formal_label_cache_transport_repair_v1
            ),
            "audit_formal_cache_join": (
                cache_core.audit_formal_cache_join_transport_repair_v1
            ),
        }
    )
    return SimpleNamespace(**values)


def _run(args: argparse.Namespace) -> Mapping[str, Any]:
    contract = load_frozen_transport_repair_contract(
        args.contract, repository_root=args.repository_root
    )
    if args.command == "validate-source":
        validation = validate_transport_repair_source_only_contract(
            args.contract, repository_root=args.repository_root
        )
        return {
            **validation,
            "runner": validate_source_a(
                contract, expected_source_a_git_commit=args.source_a_git_commit
            ),
        }
    if args.command == "materialize-runner-freeze":
        validate_transport_repair_source_only_contract(
            args.contract, repository_root=args.repository_root
        )
        return materialize_runner_freeze(
            contract, expected_source_a_git_commit=args.source_a_git_commit
        )

    # note (luojiaxuan): This must precede token/HF/client construction and
    # the runner's new global claim, so v1 cannot be resumed or overwritten.
    failed_v1_attempt = validate_failed_v1_attempt(contract, args.data_root)
    token = base_manager._secure_token(args.hf_token_file)
    hooks = base_manager._hooks(
        contract,
        token,
        args.data_root,
        cache_api=_repair_cache_api(),
    )

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)

    def destination_download(**kwargs: Any) -> str:
        return hf_hub_download(**kwargs, token=token)

    result = execute_formal_cache(
        mode=args.command,
        contract=contract,
        hooks=hooks,
        api=api,
        download_fn=destination_download,
        operation_factory=CommitOperationAdd,
        expected_execution_b_git_commit=args.execution_b_git_commit,
        data_root=args.data_root,
        fresh_download_parent=args.fresh_download_parent,
    )
    return {**result, "failed_v1_attempt": failed_v1_attempt}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
