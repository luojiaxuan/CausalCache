"""Mechanical validator for the frozen validation-12 exploratory roster."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROTOCOL_ID = "causalcache_exploratory_closed_loop_validation12_v1"
PLAN_PATH = "code/configs/androidworld_validation_plan.json"
PLAN_SHA256 = "6429599d9d351c396cf0da12939ed82fb84c9acb9c07b051ae92f9d2832e3b42"
PLAN_INSTANCE_RECORDS_SHA256 = (
    "8204e7f832f1d70becd51f299977f0e4a322a080bbfab64010345c48f901b0e8"
)
MANIFEST_PATH = "data/manifests/exploratory_closed_loop_validation12_v1.json"
MANIFEST_SHA256 = (
    "8e68e6eb1adde5627e83ce07cf4e8d7cef4843250a2e1b58baa247af57a07a42"
)
SELECTED_INSTANCE_IDENTITY_SHA256 = (
    "e0ae4ad79b33f1bce44158be5ec22f55e83fc054462407d56b2d3b9c5036199b"
)
SPLIT = "validation"
SUITE_SEED = 271828
ELIGIBLE_TASK_INDEX = 0
TAKE_PER_STRATUM = 4
STRATA = ("short", "medium", "long")
EXPECTED_POOL_SIZES = {"short": 15, "medium": 8, "long": 8}
RANK_KEY_DESCRIPTION = (
    "sha256(protocol_id+NUL+split+NUL+suite_seed_decimal+NUL+task_type+NUL+task_index_decimal)"
)
IDENTITY_FIELDS = (
    "horizon_stratum",
    "max_steps",
    "task_index",
    "task_type",
)
PLAN_TOP_LEVEL_KEYS = {
    "schema_version",
    "split",
    "source",
    "registry_sha256",
    "suite_seed",
    "task_combinations",
    "task_type_count",
    "task_instance_count",
    "instance_records_sha256",
    "instances",
}
INSTANCE_KEYS = {
    "task_type",
    "task_index",
    "goal",
    "template",
    "complexity",
    "max_steps",
    "start_on_home_screen",
}
MANIFEST_TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "source_validation_plan",
    "selection",
    "records",
    "selected_instance_identity_sha256",
}
RECORD_KEYS = {"horizon_stratum", "instance", "selection_sha256"}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def horizon_stratum(max_steps: int) -> str:
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be one positive integer")
    if max_steps <= 12:
        return "short"
    if max_steps <= 22:
        return "medium"
    return "long"


def selection_sha256(*, task_type: str, task_index: int) -> str:
    if not isinstance(task_type, str) or not task_type or "\0" in task_type:
        raise ValueError("task_type must be non-empty text")
    if type(task_index) is not int or task_index < 0:
        raise ValueError("task_index must be one nonnegative integer")
    payload = "\0".join(
        (
            PROTOCOL_ID,
            SPLIT,
            str(SUITE_SEED),
            task_type,
            str(task_index),
        )
    ).encode("utf-8")
    return sha256_bytes(payload)


def selected_instance_identity_projection(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    projection: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        record = _mapping(raw, f"record {index}")
        instance = _mapping(record.get("instance"), f"record {index} instance")
        projection.append(
            {
                "horizon_stratum": record.get("horizon_stratum"),
                "max_steps": instance.get("max_steps"),
                "task_index": instance.get("task_index"),
                "task_type": instance.get("task_type"),
            }
        )
    return projection


def selected_instance_identity_sha256(
    records: Sequence[Mapping[str, Any]],
) -> str:
    return sha256_bytes(
        canonical_json_bytes(selected_instance_identity_projection(records))
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a sequence")
    return value


def _equal(actual: Any, expected: Any, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(
            f"{label} drifted: expected {expected!r}, got {actual!r}"
        )


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} key inventory drifted: "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _strict_json_bytes(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {label}: {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    return _mapping(value, label)


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
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
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    fingerprint = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if fingerprint(before) != fingerprint(after) or len(payload) != after.st_size:
        raise ValueError(f"{label} changed while being read")
    return payload


def _validate_instance(raw: Any, *, label: str) -> dict[str, Any]:
    instance = _mapping(raw, label)
    _exact_keys(instance, INSTANCE_KEYS, label)
    if not isinstance(instance["task_type"], str) or not instance["task_type"]:
        raise ValueError(f"{label} task_type must be non-empty text")
    if type(instance["task_index"]) is not int or instance["task_index"] < 0:
        raise ValueError(f"{label} task_index must be nonnegative integer")
    for key in ("goal", "template"):
        if not isinstance(instance[key], str):
            raise ValueError(f"{label} {key} must be text")
    complexity = instance["complexity"]
    if type(complexity) not in (int, float) or complexity <= 0:
        raise ValueError(f"{label} complexity must be positive numeric")
    horizon_stratum(instance["max_steps"])
    if instance["max_steps"] != int(10 * float(complexity)):
        raise ValueError(f"{label} max_steps does not match complexity")
    if type(instance["start_on_home_screen"]) is not bool:
        raise ValueError(f"{label} start_on_home_screen must be boolean")
    return dict(instance)


def _validate_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    _exact_keys(plan, PLAN_TOP_LEVEL_KEYS, "validation plan")
    _equal(plan["schema_version"], "0.1.0", "plan schema version")
    _equal(plan["split"], SPLIT, "plan split")
    _equal(
        dict(_mapping(plan["source"], "plan source")),
        {
            "agent_path": "Mobile-Agent-v3.5/android_world_v3.5/android_world/agents/gui_owl.py",
            "agent_repo": "X-PLUG/MobileAgent",
            "agent_revision": "11cea575561fb7800b5fb6b6cafa56f7a91de11f",
            "canonical_repo": "google-research/android_world",
            "canonical_revision": "3e50888527ef9f29b9157ecd537e408008bb1c85",
        },
        "plan source",
    )
    _equal(
        plan["registry_sha256"],
        "185ae2019706693bd32ecc25ffd0c8f87be87331cae6f7d7e31c91674c962b89",
        "plan registry SHA256",
    )
    _equal(plan["suite_seed"], SUITE_SEED, "plan suite seed")
    _equal(plan["task_combinations"], 2, "plan task combinations")
    _equal(plan["task_type_count"], 31, "plan task type count")
    _equal(plan["task_instance_count"], 62, "plan instance count")
    _equal(
        plan["instance_records_sha256"],
        PLAN_INSTANCE_RECORDS_SHA256,
        "plan instance records SHA256",
    )
    raw_instances = _sequence(plan["instances"], "plan instances")
    instances = [
        _validate_instance(raw, label=f"plan instance {index}")
        for index, raw in enumerate(raw_instances)
    ]
    if len(instances) != 62:
        raise ValueError("plan instance denominator drifted")
    if sha256_bytes(canonical_json_bytes(instances)) != PLAN_INSTANCE_RECORDS_SHA256:
        raise ValueError("plan instance payload does not match frozen SHA256")
    identities = [(item["task_type"], item["task_index"]) for item in instances]
    if len(set(identities)) != len(identities):
        raise ValueError("plan task identity is duplicated")
    task_counts = Counter(item["task_type"] for item in instances)
    if len(task_counts) != 31 or set(task_counts.values()) != {2}:
        raise ValueError("plan must contain two instances for each of 31 task types")
    return instances


def _expected_records(
    instances: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pools: dict[str, list[tuple[str, dict[str, Any]]]] = {
        name: [] for name in STRATA
    }
    for instance in instances:
        if instance["task_index"] != ELIGIBLE_TASK_INDEX:
            continue
        stratum = horizon_stratum(instance["max_steps"])
        digest = selection_sha256(
            task_type=instance["task_type"], task_index=instance["task_index"]
        )
        pools[stratum].append((digest, dict(instance)))
    pool_sizes = {name: len(pools[name]) for name in STRATA}
    if pool_sizes != EXPECTED_POOL_SIZES:
        raise ValueError(
            f"eligible stratum pool sizes drifted: expected {EXPECTED_POOL_SIZES}, "
            f"got {pool_sizes}"
        )
    records: list[dict[str, Any]] = []
    for stratum in STRATA:
        ranked = sorted(pools[stratum], key=lambda item: item[0])
        for digest, instance in ranked[:TAKE_PER_STRATUM]:
            records.append(
                {
                    "horizon_stratum": stratum,
                    "instance": instance,
                    "selection_sha256": digest,
                }
            )
    return records, pool_sizes


def _validate_manifest_header(manifest: Mapping[str, Any]) -> None:
    _exact_keys(manifest, MANIFEST_TOP_LEVEL_KEYS, "roster manifest")
    _equal(manifest["schema_version"], "1.0.0", "manifest schema version")
    _equal(manifest["protocol_id"], PROTOCOL_ID, "manifest protocol id")
    _equal(
        manifest["status"],
        "FROZEN_EXPLORATORY_CLOSED_LOOP_VALIDATION12_ROSTER_V1",
        "manifest status",
    )
    source = _mapping(manifest["source_validation_plan"], "source validation plan")
    _equal(
        dict(source),
        {
            "path": PLAN_PATH,
            "sha256": PLAN_SHA256,
            "instance_records_sha256": PLAN_INSTANCE_RECORDS_SHA256,
            "split": SPLIT,
            "suite_seed": SUITE_SEED,
        },
        "manifest source validation plan binding",
    )
    selection = _mapping(manifest["selection"], "manifest selection")
    _equal(
        dict(selection),
        {
            "arm_count": 5,
            "eligible_task_index": ELIGIBLE_TASK_INDEX,
            "episode_count": 60,
            "rank_key": RANK_KEY_DESCRIPTION,
            "strata": {
                "long": "max_steps_gte_23",
                "medium": "13_lte_max_steps_lte_22",
                "short": "max_steps_lte_12",
            },
            "take_per_stratum": TAKE_PER_STRATUM,
            "template_count": 12,
        },
        "manifest selection contract",
    )


def validate_exploratory_closed_loop_roster(
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    instances = _validate_plan(_mapping(plan, "validation plan"))
    _validate_manifest_header(_mapping(manifest, "roster manifest"))
    raw_records = _sequence(manifest["records"], "manifest records")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_records):
        record = _mapping(raw, f"manifest record {index}")
        _exact_keys(record, RECORD_KEYS, f"manifest record {index}")
        records.append(
            {
                "horizon_stratum": record["horizon_stratum"],
                "instance": _validate_instance(
                    record["instance"], label=f"manifest record {index} instance"
                ),
                "selection_sha256": record["selection_sha256"],
            }
        )
    if len(records) != len(STRATA) * TAKE_PER_STRATUM:
        raise ValueError("selected record denominator drifted")
    expected, pool_sizes = _expected_records(instances)
    if records != expected:
        raise ValueError("selected records differ from mechanical stratum top-4")
    identity_projection = selected_instance_identity_projection(records)
    if any(set(item) != set(IDENTITY_FIELDS) for item in identity_projection):
        raise ValueError("selected identity projection field inventory drifted")
    identity_sha = sha256_bytes(canonical_json_bytes(identity_projection))
    _equal(
        identity_sha,
        SELECTED_INSTANCE_IDENTITY_SHA256,
        "recomputed selected instance identity SHA256",
    )
    _equal(
        manifest["selected_instance_identity_sha256"],
        SELECTED_INSTANCE_IDENTITY_SHA256,
        "manifest selected instance identity SHA256",
    )
    return {
        "status": "VALID_EXPLORATORY_CLOSED_LOOP_VALIDATION12_ROSTER_V1",
        "protocol_id": PROTOCOL_ID,
        "split": SPLIT,
        "suite_seed": SUITE_SEED,
        "eligible_task_index": ELIGIBLE_TASK_INDEX,
        "eligible_stratum_pool_sizes": pool_sizes,
        "take_per_stratum": TAKE_PER_STRATUM,
        "selected_record_count": len(records),
        "selection_sha256s": [record["selection_sha256"] for record in records],
        "identity_projection_fields": list(IDENTITY_FIELDS),
        "selected_instance_identity_sha256": identity_sha,
    }


def validate_frozen_exploratory_closed_loop_roster_files(
    *,
    repository_root: str | Path,
    plan_path: str | Path = PLAN_PATH,
    manifest_path: str | Path = MANIFEST_PATH,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_source = Path(plan_path)
    if not plan_source.is_absolute():
        plan_source = root / plan_source
    manifest_source = Path(manifest_path)
    if not manifest_source.is_absolute():
        manifest_source = root / manifest_source
    plan_payload = _regular_file_bytes(plan_source, label="validation plan")
    manifest_payload = _regular_file_bytes(manifest_source, label="roster manifest")
    _equal(sha256_bytes(plan_payload), PLAN_SHA256, "validation plan file SHA256")
    _equal(
        sha256_bytes(manifest_payload), MANIFEST_SHA256, "roster manifest file SHA256"
    )
    result = validate_exploratory_closed_loop_roster(
        plan=_strict_json_bytes(plan_payload, label="validation plan"),
        manifest=_strict_json_bytes(manifest_payload, label="roster manifest"),
    )
    return {
        **result,
        "plan_path": PLAN_PATH,
        "plan_sha256": PLAN_SHA256,
        "manifest_path": MANIFEST_PATH,
        "manifest_sha256": MANIFEST_SHA256,
    }


__all__ = [
    "ELIGIBLE_TASK_INDEX",
    "EXPECTED_POOL_SIZES",
    "IDENTITY_FIELDS",
    "MANIFEST_PATH",
    "MANIFEST_SHA256",
    "PLAN_INSTANCE_RECORDS_SHA256",
    "PLAN_PATH",
    "PLAN_SHA256",
    "PROTOCOL_ID",
    "RANK_KEY_DESCRIPTION",
    "SELECTED_INSTANCE_IDENTITY_SHA256",
    "STRATA",
    "SUITE_SEED",
    "TAKE_PER_STRATUM",
    "canonical_json_bytes",
    "horizon_stratum",
    "selected_instance_identity_projection",
    "selected_instance_identity_sha256",
    "selection_sha256",
    "sha256_bytes",
    "validate_exploratory_closed_loop_roster",
    "validate_frozen_exploratory_closed_loop_roster_files",
]
