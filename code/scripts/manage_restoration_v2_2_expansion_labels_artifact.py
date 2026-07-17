"""Manage expansion exact-label runner freeze and immutable raw evidence."""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from causalcache.restoration_v2_2_expansion_labels_artifact import (
    CANONICAL_ARCHIVE_PATH,
    CANONICAL_HF_PATH,
    CANONICAL_HF_REPO,
    CANONICAL_HF_TAG,
    CANONICAL_LEDGER_PATH,
    CANONICAL_OUTPUT_DIR,
    CANONICAL_RUNNER_FREEZE_PATH,
    atomic_exclusive_publish_bytes,
    build_expansion_label_artifact_manifest,
    load_and_validate_runner_freeze,
    materialize_runner_freeze,
    package_raw_expansion_label_evidence,
    pretty_json_bytes,
    read_expansion_label_archive,
    sha256_file,
    validate_hub_download_attestation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("materialize-runner-freeze")
    freeze.add_argument("--repository-root", type=Path, required=True)
    freeze.add_argument("--runner-source-git-commit", required=True)
    freeze.add_argument(
        "--output", type=Path, default=Path(CANONICAL_RUNNER_FREEZE_PATH)
    )

    validate_freeze = commands.add_parser("validate-runner-freeze")
    validate_freeze.add_argument("--repository-root", type=Path, required=True)
    validate_freeze.add_argument(
        "--runner-freeze", type=Path, default=Path(CANONICAL_RUNNER_FREEZE_PATH)
    )

    package = commands.add_parser("package")
    package.add_argument("--raw-output-dir", type=Path, default=CANONICAL_OUTPUT_DIR)
    package.add_argument(
        "--global-attempt-ledger", type=Path, default=CANONICAL_LEDGER_PATH
    )
    package.add_argument("--output", type=Path, default=CANONICAL_ARCHIVE_PATH)
    package.add_argument("--source-git-commit", required=True)
    package.add_argument("--config-sha256", required=True)
    _add_external_replay_arguments(package)

    validate = commands.add_parser("validate")
    validate.add_argument("--evidence", type=Path, required=True)
    validate.add_argument("--source-git-commit", required=True)
    validate.add_argument("--config-sha256", required=True)

    fresh = commands.add_parser("validate-fresh-hf")
    _add_fresh_arguments(fresh)

    manifest = commands.add_parser("create-manifest")
    _add_fresh_arguments(manifest)
    manifest.add_argument("--output", type=Path, required=True)

    upload = commands.add_parser("upload-and-tag")
    upload.add_argument("--source-archive", type=Path, required=True)
    upload.add_argument("--source-git-commit", required=True)
    upload.add_argument("--config-sha256", required=True)
    upload.add_argument("--hf-repo", default=CANONICAL_HF_REPO)
    upload.add_argument("--hf-tag", default=CANONICAL_HF_TAG)
    upload.add_argument("--hf-path", default=CANONICAL_HF_PATH)
    return parser


def _add_fresh_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--hf-repo", default=CANONICAL_HF_REPO)
    parser.add_argument("--hf-tag", default=CANONICAL_HF_TAG)
    parser.add_argument("--hf-path", default=CANONICAL_HF_PATH)
    parser.add_argument("--hf-immutable-revision", required=True)
    _add_external_replay_arguments(parser)


def _add_external_replay_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parent-substrate-archive", type=Path, required=True)
    parser.add_argument("--derived-artifact-root", type=Path, required=True)
    parser.add_argument("--ocr-backend-config", type=Path, required=True)


def _archive_kwargs(
    args: argparse.Namespace,
    *,
    downloaded_archive: Path,
) -> dict[str, object]:
    return {
        "source_archive": args.source_archive,
        "fresh_immutable_archive": downloaded_archive,
        "source_git_commit": args.source_git_commit,
        "config_sha256": args.config_sha256,
        "hf_repo": args.hf_repo,
        "hf_tag": args.hf_tag,
        "hf_path": args.hf_path,
        "hf_immutable_revision": args.hf_immutable_revision,
        "parent_substrate_archive": args.parent_substrate_archive,
        "derived_artifact_root": args.derived_artifact_root,
        "ocr_backend_config": args.ocr_backend_config,
    }


def _info_sha(value: object) -> str:
    if isinstance(value, dict):
        sha = value.get("sha")
    else:
        sha = getattr(value, "sha", None)
    if not isinstance(sha, str):
        raise ValueError("HF dataset-info response lacks a resolved revision")
    return sha


def _info_field(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _validate_dataset_info(
    value: object,
    *,
    hf_repo: str,
    expected_sha: str | None = None,
) -> str:
    sha = _info_sha(value)
    if (
        len(sha) != 40
        or any(character not in "0123456789abcdef" for character in sha)
        or _info_field(value, "id") != hf_repo
        or _info_field(value, "private") is not True
        or (expected_sha is not None and sha != expected_sha)
    ):
        raise ValueError("HF dataset id, privacy, or resolved revision drifted")
    return sha


def _absolute_lexical(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(path: Path) -> None:
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"path component cannot be a symlink: {current}")


def _canonical_download_target(directory: Path, hf_path: str) -> Path:
    relative = PurePosixPath(hf_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("HF path is not canonical relative POSIX")
    return directory.joinpath(*relative.parts)


def download_fresh_hf_archive(
    *,
    hf_repo: str,
    hf_tag: str,
    hf_path: str,
    hf_immutable_revision: str,
    download_dir: str | Path,
    api_factory=None,
    download_fn=None,
) -> tuple[Path, dict[str, object]]:
    if (
        hf_repo != CANONICAL_HF_REPO
        or hf_tag != CANONICAL_HF_TAG
        or hf_path != CANONICAL_HF_PATH
    ):
        raise ValueError("HF download identity differs from frozen destination")
    supplied_directory = _absolute_lexical(download_dir)
    _assert_no_symlink_components(supplied_directory)
    if supplied_directory.exists() and not supplied_directory.is_dir():
        raise FileExistsError("fresh HF download path is not a directory")
    if supplied_directory.exists() and any(supplied_directory.iterdir()):
        raise FileExistsError("fresh HF download directory must be absent or empty")
    supplied_directory.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(supplied_directory)
    directory = supplied_directory
    expected_download = _canonical_download_target(directory, hf_path)
    if expected_download.exists() or expected_download.is_symlink():
        raise FileExistsError("fresh HF archive target already exists")
    if api_factory is None or download_fn is None:
        from huggingface_hub import HfApi, hf_hub_download

        api_factory = api_factory or HfApi
        download_fn = download_fn or hf_hub_download
    api = api_factory()
    pre_tag_sha = _validate_dataset_info(
        api.dataset_info(repo_id=hf_repo, revision=hf_tag, files_metadata=False),
        hf_repo=hf_repo,
        expected_sha=hf_immutable_revision,
    )
    pre_immutable_sha = _validate_dataset_info(
        api.dataset_info(
            repo_id=hf_repo,
            revision=hf_immutable_revision,
            files_metadata=False,
        ),
        hf_repo=hf_repo,
        expected_sha=hf_immutable_revision,
    )
    returned = download_fn(
        repo_id=hf_repo,
        filename=hf_path,
        repo_type="dataset",
        revision=hf_immutable_revision,
        force_download=True,
        local_dir=str(directory),
    )
    downloaded = _absolute_lexical(returned)
    if downloaded != expected_download:
        raise ValueError("forced HF download escaped the exact fresh target")
    _assert_no_symlink_components(downloaded)
    try:
        downloaded_mode = os.lstat(downloaded).st_mode
    except FileNotFoundError as error:
        raise ValueError("forced HF download did not return an archive") from error
    if not stat.S_ISREG(downloaded_mode):
        raise ValueError("forced HF download did not return a regular archive")
    post_tag_sha = _validate_dataset_info(
        api.dataset_info(repo_id=hf_repo, revision=hf_tag, files_metadata=False),
        hf_repo=hf_repo,
        expected_sha=hf_immutable_revision,
    )
    post_immutable_sha = _validate_dataset_info(
        api.dataset_info(
            repo_id=hf_repo,
            revision=hf_immutable_revision,
            files_metadata=False,
        ),
        hf_repo=hf_repo,
        expected_sha=hf_immutable_revision,
    )
    attestation: dict[str, object] = {
        "repo": hf_repo,
        "repo_type": "dataset",
        "repo_id": hf_repo,
        "repo_id_verified": True,
        "repo_private": True,
        "repo_private_verified": True,
        "tag": hf_tag,
        "path": hf_path,
        "tag_query_revision": hf_tag,
        "download_requested_revision": hf_immutable_revision,
        "resolved_revision": hf_immutable_revision,
        "tag_resolved_revision": post_tag_sha,
        "pre_download_tag_resolved_revision": pre_tag_sha,
        "pre_download_immutable_resolved_revision": pre_immutable_sha,
        "post_download_tag_resolved_revision": post_tag_sha,
        "post_download_immutable_resolved_revision": post_immutable_sha,
        "tag_resolution_verified": True,
        "immutable_revision_resolution_verified": True,
        "tag_resolution_stable": pre_tag_sha == post_tag_sha,
        "immutable_revision_resolution_stable": (
            pre_immutable_sha == post_immutable_sha
        ),
        "force_download": True,
        "returned_path_exact": True,
        "no_symlink_components_verified": True,
        "downloaded_path": str(downloaded),
        "downloaded_archive_sha256": sha256_file(downloaded),
        "downloaded_archive_size_bytes": downloaded.stat().st_size,
    }
    validate_hub_download_attestation(
        attestation,
        downloaded_archive=downloaded,
    )
    return downloaded, attestation


def _ref_name(value: object) -> str | None:
    raw = _info_field(value, "name")
    if not isinstance(raw, str):
        return None
    return raw.removeprefix("refs/tags/")


def _commit_oid(value: object) -> str:
    oid = _info_field(value, "oid")
    if (
        not isinstance(oid, str)
        or len(oid) != 40
        or any(character not in "0123456789abcdef" for character in oid)
    ):
        raise ValueError("HF upload response lacks an immutable commit OID")
    return oid


def upload_and_tag_archive(
    *,
    source_archive: str | Path,
    source_git_commit: str,
    config_sha256: str,
    hf_repo: str,
    hf_tag: str,
    hf_path: str,
    api_factory=None,
) -> dict[str, object]:
    if (
        hf_repo != CANONICAL_HF_REPO
        or hf_tag != CANONICAL_HF_TAG
        or hf_path != CANONICAL_HF_PATH
    ):
        raise ValueError("HF upload identity differs from frozen destination")
    supplied_source = Path(source_archive)
    if supplied_source.is_symlink():
        raise ValueError("source archive cannot be a symlink")
    source = supplied_source.resolve()
    if not source.is_file():
        raise ValueError("source archive is missing")
    evidence = read_expansion_label_archive(
        source,
        expected_source_git_commit=source_git_commit,
        expected_config_sha256=config_sha256,
    )
    if api_factory is None:
        from huggingface_hub import HfApi

        api_factory = HfApi
    api = api_factory()
    api.create_repo(
        repo_id=hf_repo,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    pre_info = api.dataset_info(repo_id=hf_repo, files_metadata=False)
    parent_commit = _validate_dataset_info(pre_info, hf_repo=hf_repo)
    refs = api.list_repo_refs(repo_id=hf_repo, repo_type="dataset")
    tags = _info_field(refs, "tags")
    if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes, bytearray)):
        raise ValueError("HF repo refs did not expose a tag inventory")
    if hf_tag in {_ref_name(tag) for tag in tags}:
        raise FileExistsError("frozen HF tag already exists")
    pre_files = api.list_repo_files(
        repo_id=hf_repo,
        repo_type="dataset",
        revision=parent_commit,
    )
    if not isinstance(pre_files, Sequence) or isinstance(
        pre_files, (str, bytes, bytearray)
    ):
        raise ValueError("HF repo did not expose a file inventory")
    if hf_path in pre_files:
        raise FileExistsError("frozen HF archive path already exists")
    upload = api.upload_file(
        path_or_fileobj=str(source),
        path_in_repo=hf_path,
        repo_id=hf_repo,
        repo_type="dataset",
        revision="main",
        parent_commit=parent_commit,
        commit_message="Upload immutable CausalCache expansion exact-label archive",
    )
    immutable_commit = _commit_oid(upload)
    uploaded_info = api.dataset_info(
        repo_id=hf_repo,
        revision=immutable_commit,
        files_metadata=False,
    )
    _validate_dataset_info(
        uploaded_info,
        hf_repo=hf_repo,
        expected_sha=immutable_commit,
    )
    uploaded_files = api.list_repo_files(
        repo_id=hf_repo,
        repo_type="dataset",
        revision=immutable_commit,
    )
    if list(uploaded_files).count(hf_path) != 1:
        raise ValueError("uploaded commit does not contain the exact frozen path")
    api.create_tag(
        repo_id=hf_repo,
        tag=hf_tag,
        revision=immutable_commit,
        repo_type="dataset",
        exist_ok=False,
    )
    tag_sha = _validate_dataset_info(
        api.dataset_info(repo_id=hf_repo, revision=hf_tag, files_metadata=False),
        hf_repo=hf_repo,
        expected_sha=immutable_commit,
    )
    immutable_sha = _validate_dataset_info(
        api.dataset_info(
            repo_id=hf_repo,
            revision=immutable_commit,
            files_metadata=False,
        ),
        hf_repo=hf_repo,
        expected_sha=immutable_commit,
    )
    return {
        "status": "UPLOADED_AND_IMMUTABLY_TAGGED_EXPANSION_EXACT_LABEL_ARCHIVE",
        "repo": hf_repo,
        "repo_type": "dataset",
        "repo_id_verified": True,
        "repo_private_verified": True,
        "create_repo_private": True,
        "create_repo_exist_ok": True,
        "path": hf_path,
        "path_absent_before_upload": True,
        "tag": hf_tag,
        "tag_absent_before_upload": True,
        "overwrite_allowed": False,
        "upload_parent_commit": parent_commit,
        "immutable_revision": immutable_commit,
        "tag_resolved_revision": tag_sha,
        "immutable_revision_resolved_revision": immutable_sha,
        "create_tag_exist_ok": False,
        "source_archive_sha256": sha256_file(source),
        "source_archive_size_bytes": source.stat().st_size,
        "source_tree_inventory_sha256": evidence.tree_inventory_sha256,
        "source_git_commit": evidence.source_git_commit,
        "config_sha256": evidence.config_sha256,
    }


def _download(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    return download_fresh_hf_archive(
        hf_repo=args.hf_repo,
        hf_tag=args.hf_tag,
        hf_path=args.hf_path,
        hf_immutable_revision=args.hf_immutable_revision,
        download_dir=args.download_dir,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "materialize-runner-freeze":
        result = materialize_runner_freeze(
            repository_root=args.repository_root,
            runner_source_git_commit=args.runner_source_git_commit,
            output_path=args.output,
        )
    elif args.command == "validate-runner-freeze":
        _freeze, validation = load_and_validate_runner_freeze(
            args.runner_freeze, repository_root=args.repository_root
        )
        result = {
            "status": "VALID_COMMITTED_PUSHED_EXPANSION_EXACT_LABEL_RUNNER_FREEZE",
            **validation,
        }
    elif args.command == "package":
        result = package_raw_expansion_label_evidence(
            raw_output_dir=args.raw_output_dir,
            global_attempt_ledger=args.global_attempt_ledger,
            output_archive=args.output,
            source_git_commit=args.source_git_commit,
            config_sha256=args.config_sha256,
            parent_substrate_archive=args.parent_substrate_archive,
            derived_artifact_root=args.derived_artifact_root,
            ocr_backend_config=args.ocr_backend_config,
        )
    elif args.command == "validate":
        evidence = read_expansion_label_archive(
            args.evidence,
            expected_source_git_commit=args.source_git_commit,
            expected_config_sha256=args.config_sha256,
        )
        result = {
            "status": "VALID_V2_2_EXPANSION_EXACT_LABEL_RAW_EVIDENCE",
            "source_git_commit": evidence.source_git_commit,
            "execution_git_commit": evidence.execution_git_commit,
            "outcome": evidence.outcome,
            "counts": dict(evidence.reduction["counts"]),
            "derived_payload_sha256": evidence.reduction[
                "derived_payload_sha256"
            ],
            "tree_inventory_sha256": evidence.tree_inventory_sha256,
            "file_count": len(evidence.inventory),
        }
    elif args.command == "validate-fresh-hf":
        downloaded, attestation = _download(args)
        manifest = build_expansion_label_artifact_manifest(
            **_archive_kwargs(args, downloaded_archive=downloaded),
            hub_download_attestation=attestation,
        )
        result = {
            "status": "VALID_FRESH_IMMUTABLE_EXPANSION_EXACT_LABEL_HF_EVIDENCE",
            "outcome": manifest["result"]["outcome"],
            "raw_tree_inventory_sha256": manifest["raw_archive"][
                "tree_inventory_sha256"
            ],
            "hf_artifact": manifest["hf_artifact"],
        }
    elif args.command == "create-manifest":
        downloaded, attestation = _download(args)
        manifest = build_expansion_label_artifact_manifest(
            **_archive_kwargs(args, downloaded_archive=downloaded),
            hub_download_attestation=attestation,
        )
        atomic_exclusive_publish_bytes(args.output, pretty_json_bytes(manifest))
        result = {
            "status": "CREATED_EXPANSION_EXACT_LABEL_ARTIFACT_MANIFEST",
            "output": str(args.output.resolve()),
            "outcome": manifest["result"]["outcome"],
            "raw_archive": manifest["raw_archive"],
            "hf_artifact": manifest["hf_artifact"],
        }
    elif args.command == "upload-and-tag":
        result = upload_and_tag_archive(
            source_archive=args.source_archive,
            source_git_commit=args.source_git_commit,
            config_sha256=args.config_sha256,
            hf_repo=args.hf_repo,
            hf_tag=args.hf_tag,
            hf_path=args.hf_path,
        )
    else:
        raise AssertionError(f"unknown artifact command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
