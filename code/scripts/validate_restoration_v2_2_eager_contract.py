"""Validate the source-only restoration-v2.2 eager full-45 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from causalcache.restoration_v2_2_eager_contract import RestorationV22EagerContract


CANONICAL_REMOTE_URL = "https://github.com/luojiaxuan/CausalCache.git"
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--require-clean-pushed-main",
        action="store_true",
        help=(
            "establish a formal source freeze by requiring clean canonical main, "
            "HEAD == origin/main == advertised remote main, and byte-identical "
            "HEAD blobs for the complete formal source inventory"
        ),
    )
    return parser


def _run_git(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    text: bool = True,
) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )
    if result.returncode != 0:
        stderr = (
            result.stderr
            if text
            else result.stderr.decode("utf-8", errors="replace")
        )
        detail = str(stderr).strip()
        raise ValueError(
            f"Git command failed while validating source freeze: {' '.join(arguments)}"
            + (f": {detail}" if detail else "")
        )
    return result.stdout


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_clean_pushed_formal_inventory(
    *,
    repository_root: str | Path,
    inventory_paths: Sequence[str],
) -> dict[str, object]:
    """Bind every formal source byte to the clean, advertised canonical main."""
    root = Path(repository_root).resolve()
    top_level = str(_run_git(root, ("rev-parse", "--show-toplevel"))).strip()
    if Path(top_level).resolve() != root:
        raise ValueError("repository root differs from the Git top level")

    branch = str(_run_git(root, ("branch", "--show-current"))).strip()
    if branch != "main":
        raise ValueError("formal source freeze requires canonical main")
    status = str(
        _run_git(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    )
    if status:
        raise ValueError(
            "formal source freeze requires a clean worktree including untracked files"
        )

    head = str(_run_git(root, ("rev-parse", "HEAD"))).strip()
    origin_main = str(
        _run_git(root, ("rev-parse", "refs/remotes/origin/main"))
    ).strip()
    advertised = str(
        _run_git(root, ("ls-remote", "--heads", "origin", "refs/heads/main"))
    ).strip()
    remote_url = str(_run_git(root, ("remote", "get-url", "origin"))).strip()
    if GIT_SHA_PATTERN.fullmatch(head) is None:
        raise ValueError("Git HEAD is not a full lowercase commit SHA")
    if head != origin_main or advertised != f"{head}\trefs/heads/main":
        raise ValueError(
            "HEAD, origin/main, and advertised remote main must be identical"
        )
    if remote_url != CANONICAL_REMOTE_URL:
        raise ValueError("origin URL differs from the canonical CausalCache repository")

    inventory = tuple(inventory_paths)
    if not inventory:
        raise ValueError("formal source inventory must not be empty")
    if len(inventory) != len(set(inventory)):
        raise ValueError("formal source inventory contains duplicate paths")

    records: list[dict[str, str]] = []
    for relative in inventory:
        if not isinstance(relative, str) or not relative:
            raise ValueError("formal source inventory contains an invalid path")
        posix = PurePosixPath(relative)
        if (
            posix.is_absolute()
            or "." in posix.parts
            or ".." in posix.parts
            or posix.as_posix() != relative
        ):
            raise ValueError(
                f"formal source inventory path is not canonical: {relative}"
            )
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(
                f"formal source path is missing or escapes the repository: {relative}"
            )
        committed = _run_git(root, ("show", f"HEAD:{relative}"), text=False)
        if not isinstance(committed, bytes) or committed != path.read_bytes():
            raise ValueError(
                f"formal source differs from its committed HEAD blob: {relative}"
            )
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(committed).hexdigest(),
            }
        )

    post_branch = str(_run_git(root, ("branch", "--show-current"))).strip()
    post_status = str(
        _run_git(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    )
    post_head = str(_run_git(root, ("rev-parse", "HEAD"))).strip()
    post_origin_main = str(
        _run_git(root, ("rev-parse", "refs/remotes/origin/main"))
    ).strip()
    post_advertised = str(
        _run_git(root, ("ls-remote", "--heads", "origin", "refs/heads/main"))
    ).strip()
    post_remote_url = str(
        _run_git(root, ("remote", "get-url", "origin"))
    ).strip()
    if (
        post_branch != branch
        or post_status
        or post_head != head
        or post_origin_main != origin_main
        or post_advertised != advertised
        or post_remote_url != remote_url
    ):
        raise ValueError("repository identity drifted during formal source validation")

    return {
        "branch": branch,
        "commit": head,
        "origin_main": origin_main,
        "remote_main": head,
        "remote_url": remote_url,
        "worktree": "clean_including_untracked",
        "formal_inventory_file_count": len(records),
        "formal_inventory_sha256": _canonical_sha256(records),
        "formal_inventory": records,
    }


def validate_from_args(args: argparse.Namespace) -> dict[str, object]:
    contract = RestorationV22EagerContract.load(
        args.config, repository_root=args.repository_root
    )
    validation = contract.validation
    require_freeze = bool(getattr(args, "require_clean_pushed_main", False))
    source_freeze: Mapping[str, object]
    if require_freeze:
        git_identity = validate_clean_pushed_formal_inventory(
            repository_root=args.repository_root,
            inventory_paths=contract.data["scientific_inheritance"][
                "formal_run_source_inventory_paths"
            ],
        )
        source_freeze = {
            "formal_source_freeze_established": True,
            "status": "VALIDATED_CLEAN_PUSHED_MAIN_AND_HEAD_BLOBS",
            "git": git_identity,
        }
    else:
        source_freeze = {
            "formal_source_freeze_established": False,
            "status": "NOT_REQUESTED",
            "git": None,
        }
    return {
        "contract_sha256": contract.source_sha256,
        "contract_valid": True,
        "external_runtime_evidence": {
            "fresh_immutable_parent_evidence_required": contract.data[
                "authorization"
            ][
                "fresh_immutable_parent_evidence_validation_required_before_runtime_import"
            ],
            "status": "NOT_CONSUMED_BY_SOURCE_ONLY_VALIDATOR",
        },
        "fixed_state_denominator": validation["fixed_state_denominator"],
        "maximum_schedule": {
            "generation_call_count": validation["maximum_generation_call_count"],
            "teacher_forward_count": validation["maximum_teacher_forward_count"],
            "kl_measurement_count": validation["maximum_kl_measurement_count"],
        },
        "policy_execution_authorized_by_this_validator": False,
        "gpu_execution_authorized_by_this_validator": False,
        "protocol_id": contract.protocol_id,
        "source_freeze": source_freeze,
        "state_projection_sha256": validation["state_projection_sha256"],
        "worker_count": validation["worker_count"],
        "worker_state_counts": validation["worker_state_counts"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = validate_from_args(_build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
