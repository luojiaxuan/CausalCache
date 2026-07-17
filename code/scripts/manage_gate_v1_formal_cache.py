#!/usr/bin/env python3
"""Validate, freeze, materialize, or replay the formal-58 cache protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
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
EXECUTION_ZERO_KEYS = (
    "development_semantic_decode_count",
    "confirm_semantic_decode_count",
    "training_example_count",
    "optimizer_step_count",
    "model_load_count",
    "model_forward_count",
    "oracle_metric_count",
    "oof_metric_count",
    "checkpoint_count",
    "matched_nll_evaluation_count",
    "closed_loop_episode_count",
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


# note (luojiaxuan): A direct CLI process installs the guard before project/HF
# imports. Library/test imports must not leak a process-global import hook.
_IMPORT_GUARD = _ForbiddenImportGuard() if __name__ == "__main__" else None
if _IMPORT_GUARD is not None:
    sys.meta_path.insert(0, _IMPORT_GUARD)

from causalcache import gate_v1_formal_cache as cache_core  # noqa: E402
from causalcache.gate_v1_formal_cache_contract import (  # noqa: E402
    CANONICAL_CONFIG_PATH,
    load_frozen_formal_cache_contract,
    load_strict_json_object,
    validate_source_only_contract,
)
from causalcache.gate_v1_formal_cache_runner import (  # noqa: E402
    CacheArtifact,
    FormalCacheHooks,
    execute_formal_cache,
    materialize_runner_freeze,
    validate_source_a,
)


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


def _secure_token(path: Path) -> str:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) not in {
            0o400,
            0o600,
        }:
            raise ValueError("HF token must be a mode-0400/0600 regular file")
        payload = b""
        while chunk := os.read(descriptor, 4096):
            payload += chunk
        after = os.fstat(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("HF token changed while being read")
    try:
        token = payload.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError("HF token is not ASCII") from error
    if not token or any(character.isspace() for character in token):
        raise ValueError("HF token is empty or malformed")
    return token


def _file_bytes(path: Path, *, expected_mode: int | None = None) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"download is not a regular file: {path}")
        if expected_mode is not None and stat.S_IMODE(metadata.st_mode) != expected_mode:
            raise ValueError(f"file mode drifted: {path}")
        payload = b""
        while chunk := os.read(descriptor, 1024 * 1024):
            payload += chunk
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fingerprint = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if fingerprint(metadata) != fingerprint(after) or len(payload) != metadata.st_size:
        raise ValueError(f"download changed while being read: {path}")
    return payload


def _strict_json_bytes(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value in {label}: {item}")
        ),
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _verify_label_publication_completion(contract: Any) -> Mapping[str, Any]:
    binding = contract.data["publication_completion_binding"]["completion_seal"]
    path = Path(binding["path"])
    payload = _file_bytes(path, expected_mode=binding["mode"])
    if (
        len(payload) != binding["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != binding["sha256"]
    ):
        raise ValueError("repaired-label publication completion identity drifted")
    value = _strict_json_bytes(payload, label="repaired-label publication completion")
    if value.get("status") != binding["status"]:
        raise ValueError("repaired-label publication completion status drifted")
    return {
        "completion_validation_count": 1,
        "completion_sha256": binding["sha256"],
        "completion_status": binding["status"],
    }


def _observed_execution_counts(
    feature_audit: Mapping[str, Any],
    label_audit: Mapping[str, Any],
    join_audit: Mapping[str, Any],
) -> Mapping[str, int]:
    feature_counts = feature_audit.get("counts")
    label_counts = label_audit.get("counts")
    join_counts = join_audit.get("counts")
    if not all(
        isinstance(value, Mapping)
        for value in (feature_counts, label_counts, join_counts)
    ):
        raise ValueError("formal core count audits are missing")
    result = {
        "trajectory_semantic_decode_count": feature_counts.get(
            "trajectory_semantic_decode_count"
        ),
        "feature_state_count": feature_counts.get("feature_state_count"),
        "ocr_semantic_decode_count": feature_counts.get(
            "ocr_semantic_decode_count"
        ),
        "label_state_semantic_decode_count": label_counts.get(
            "label_state_semantic_decode_count"
        ),
        "distance_value_decode_count": label_counts.get(
            "distance_value_decode_count"
        ),
        "join_validation_count": join_audit.get("joined_state_count"),
        "candidate_feature_count": feature_counts.get("candidate_feature_count"),
        "conditional_edge_count": label_counts.get("conditional_edge_count"),
        "independent_target_count": label_counts.get("independent_target_count"),
    }
    consistency = {
        "trajectory_count": result["trajectory_semantic_decode_count"],
        "state_count": result["feature_state_count"],
        "candidate_feature_count": result["candidate_feature_count"],
        "distance_value_count": result["distance_value_decode_count"],
        "conditional_edge_count": result["conditional_edge_count"],
        "independent_target_count": result["independent_target_count"],
    }
    if (
        dict(join_counts) != consistency
        or result["label_state_semantic_decode_count"]
        != result["feature_state_count"]
        or result["join_validation_count"] != result["feature_state_count"]
        or any(type(value) is not int or value < 0 for value in result.values())
    ):
        raise ValueError("formal core feature/label/join count audits disagree")
    audit_maps = []
    for source in (feature_audit, label_audit, join_audit):
        for name in ("access_audit", "operation_audit"):
            value = source.get(name)
            if not isinstance(value, Mapping):
                raise ValueError(f"formal core {name} is missing")
            if any(
                not isinstance(key, str)
                or not key.endswith("_count")
                or type(count) is not int
                or count != 0
                for key, count in value.items()
            ):
                raise ValueError(f"formal core {name} contains a nonzero/invalid count")
            audit_maps.append(value)
    for key in EXECUTION_ZERO_KEYS:
        observed = [mapping[key] for mapping in audit_maps if key in mapping]
        if not observed or any(value != 0 for value in observed):
            raise ValueError(f"formal core prohibited operation audit drifted: {key}")
        result[key] = 0
    return result


def _rosters(contract: Any) -> Mapping[str, list[str]]:
    root = contract.repository_root
    legacy = load_strict_json_object(
        root / "data/manifests/restoration_v2_selection.json"
    )
    expansion = load_strict_json_object(
        root / "code/configs/causalcache_restoration_v2_2_label_expansion_v1.json"
    )
    legacy_ids = [
        item["source_id"]
        for item in legacy["roles"]["v2_label_train"]["trajectories"]
    ]
    expansion_ids = list(
        expansion["selection"]["gate_train_expansion"]["source_ids"]
    )
    return {
        "legacy_train": legacy_ids,
        "fresh_train_expansion": expansion_ids,
        "formal_train": legacy_ids + expansion_ids,
    }


def _input_specs(
    contract: Any,
    *,
    cache_api: Any = cache_core,
) -> Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]]:
    result: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    key_by_artifact_and_suffix = {
        ("legacy_derived_features", "manifest.json"): cache_api.LEGACY_FEATURE_MANIFEST,
        ("legacy_derived_features", "trajectories-00000-of-00001.jsonl"): cache_api.LEGACY_FEATURE_TRAJECTORIES,
        ("legacy_derived_features", "ocr-records-00000-of-00001.jsonl"): cache_api.LEGACY_FEATURE_OCR,
        ("expansion_derived_features", "manifest.json"): cache_api.EXPANSION_FEATURE_MANIFEST,
        ("expansion_derived_features", "trajectories-00000-of-00001.jsonl"): cache_api.EXPANSION_FEATURE_TRAJECTORIES,
        ("expansion_derived_features", "ocr-records-00000-of-00001.jsonl"): cache_api.EXPANSION_FEATURE_OCR,
    }
    for artifact_name, raw_artifact in contract.inputs.items():
        artifact = dict(raw_artifact)
        for raw_file in artifact["files"]:
            file_record = dict(raw_file)
            suffix = Path(file_record["path"]).name
            key = key_by_artifact_and_suffix.get((artifact_name, suffix))
            if artifact_name == "legacy_restoration_labels" and suffix.endswith(".tar"):
                key = cache_api.LEGACY_LABEL_ARCHIVE
            if (
                artifact_name == "repaired_expansion_restoration_labels"
                and suffix.endswith(".tar")
            ):
                key = cache_api.EXPANSION_LABEL_ARCHIVE
            if (
                artifact_name == "repaired_expansion_restoration_labels"
                and suffix.endswith(".json")
            ):
                key = cache_api.EXPANSION_LABEL_SIDECAR
            if key is not None:
                if key in result:
                    raise ValueError(f"duplicate formal input binding: {key}")
                result[key] = (artifact, file_record)
    if set(result) != set(cache_api.DOWNLOAD_KEYS):
        raise ValueError("formal input binding inventory drifted")
    return result


def _hooks(
    contract: Any,
    token: str,
    data_root: Path,
    *,
    cache_api: Any = cache_core,
) -> FormalCacheHooks:
    from huggingface_hub import hf_hub_download

    specs = _input_specs(contract, cache_api=cache_api)
    rosters = _rosters(contract)
    formal_config = contract.data
    saved: dict[str, CacheArtifact] = {}
    provenance: dict[str, Any] = {}

    def downloads(keys: Sequence[str]) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
        payloads: dict[str, bytes] = {}
        bindings: dict[str, dict[str, Any]] = {}
        for key in keys:
            artifact, file_record = specs[key]
            local_dir = data_root / "inputs" / key
            returned = Path(
                hf_hub_download(
                    repo_id=artifact["repo"],
                    filename=file_record["path"],
                    repo_type="dataset",
                    revision=artifact["immutable_revision"],
                    token=token,
                    local_dir=local_dir,
                    force_download=True,
                )
            )
            payloads[key] = _file_bytes(returned)
            bindings[key] = {
                "sha256": file_record["sha256"],
                "size_bytes": file_record["size_bytes"],
            }
        return payloads, bindings

    def artifact_audit(value: Any) -> dict[str, Any]:
        return {
            "kind": value.kind,
            "counts": dict(value.counts),
            "access_audit": dict(value.access_audit),
            "operation_audit": dict(value.operation_audit),
        }

    def build_feature(_contract: Any, _root: Path) -> CacheArtifact:
        payloads, bindings = downloads(cache_api.FEATURE_SOURCE_KEYS)
        value = cache_api.build_formal_feature_cache(
            payloads,
            transport_bindings=bindings,
            frozen_rosters=rosters,
            frozen_config=formal_config,
        )
        result = CacheArtifact(value.archive, artifact_audit(value))
        saved["feature"] = result
        return result

    def read_feature(_contract: Any, payload: bytes) -> Mapping[str, Any]:
        cache_api.read_feature_cache(payload)
        return saved["feature"].audit

    def build_label(_contract: Any, _root: Path) -> CacheArtifact:
        completion_audit = _verify_label_publication_completion(contract)
        repaired = contract.inputs["repaired_expansion_restoration_labels"]
        sidecars = [item for item in repaired["files"] if item["path"].endswith(".json")]
        archives = [item for item in repaired["files"] if item["path"].endswith(".tar")]
        if len(sidecars) != 1 or len(archives) != 1:
            raise ValueError("repaired-label archive/sidecar binding drifted")
        sidecar_record = sidecars[0]
        payloads, bindings = downloads(cache_api.LABEL_SOURCE_KEYS)
        sidecar_payload = payloads[cache_api.EXPANSION_LABEL_SIDECAR]
        sidecar = _strict_json_bytes(
            sidecar_payload, label="repaired-label publication sidecar"
        )
        destination = sidecar.get("destination")
        scientific = sidecar.get("scientific_repair")
        if (
            sidecar.get("status")
            != "IMMUTABLE_REPAIRED_EXPANSION_LABELS_PUBLICATION_MANIFEST_V1"
            or not isinstance(destination, Mapping)
            or destination.get("repo") != repaired["repo"]
            or destination.get("repo_type") != repaired["repo_type"]
            or destination.get("private") is not True
            or destination.get("tag") != repaired["tag"]
            or destination.get("archive_path") != archives[0]["path"]
            or destination.get("sidecar_path") != sidecar_record["path"]
            or not isinstance(scientific, Mapping)
            or scientific.get("archive_sha256") != archives[0]["sha256"]
            or scientific.get("archive_size_bytes") != archives[0]["size_bytes"]
            or scientific.get("formal_scientific_repair_pass") is not True
        ):
            raise ValueError("repaired-label sidecar scientific binding drifted")
        provenance.update(
            {
                **completion_audit,
                "sidecar_validation_count": 1,
                "sidecar_sha256": sidecar_record["sha256"],
                "immutable_revision": repaired["immutable_revision"],
                "annotated_tag_object": repaired["annotated_tag_object"],
            }
        )
        value = cache_api.build_formal_label_cache(
            payloads,
            transport_bindings=bindings,
            frozen_rosters=rosters,
            frozen_config=formal_config,
        )
        audit = artifact_audit(value)
        audit["repaired_label_publication"] = dict(provenance)
        result = CacheArtifact(value.archive, audit)
        saved["label"] = result
        return result

    def read_label(_contract: Any, payload: bytes) -> Mapping[str, Any]:
        cache_api.read_label_cache(payload)
        return saved["label"].audit

    def join_only(
        _contract: Any,
        _feature_audit: Mapping[str, Any],
        _label_audit: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        audit = cache_api.audit_formal_cache_join(
            saved["feature"].payload,
            saved["label"].payload,
            frozen_rosters=rosters,
            frozen_config=formal_config,
        )
        audit_record = asdict(audit)
        observed_counts = _observed_execution_counts(
            saved["feature"].audit,
            saved["label"].audit,
            audit_record,
        )
        return {
            **audit_record,
            "execution_operation_counts": dict(observed_counts),
            "repaired_label_publication": dict(provenance),
        }

    return FormalCacheHooks(
        build_feature=build_feature,
        read_feature=read_feature,
        build_label=build_label,
        read_label=read_label,
        join_only_audit=join_only,
    )


def _run(args: argparse.Namespace) -> Mapping[str, Any]:
    contract = load_frozen_formal_cache_contract(
        args.contract, repository_root=args.repository_root
    )
    if args.command == "validate-source":
        validate_source_only_contract(
            args.contract, repository_root=args.repository_root
        )
        return validate_source_a(
            contract, expected_source_a_git_commit=args.source_a_git_commit
        )
    if args.command == "materialize-runner-freeze":
        validate_source_only_contract(
            args.contract, repository_root=args.repository_root
        )
        return materialize_runner_freeze(
            contract, expected_source_a_git_commit=args.source_a_git_commit
        )

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    token = _secure_token(args.hf_token_file)
    hooks = _hooks(contract, token, args.data_root)
    api = HfApi(token=token)

    def destination_download(**kwargs: Any) -> str:
        return hf_hub_download(**kwargs, token=token)

    return execute_formal_cache(
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
