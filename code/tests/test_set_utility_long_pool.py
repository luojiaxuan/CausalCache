from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from causalcache.data.guiodyssey_independent import Candidate
from causalcache.set_utility_long_pool import (
    FROZEN_CONFIG_SHA256,
    PROTOCOL_ID,
    build_discovery_manifest,
    canonical_json_bytes,
    instruction_app_group_sha256,
    length_stratum,
    long_selection_sha256,
    normalize_instruction,
    project_candidate,
    validate_discovery_config,
    validate_discovery_source_only,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "code/configs/causalcache_set_utility_long_pool_discovery_v1.json"


def _candidate(source_id: str, decision_count: int, row: int) -> Candidate:
    return Candidate(
        source_id=source_id,
        transport_file="data/shard.parquet",
        transport_row_index=row,
        selection_sha256="0" * 64,
        decision_count=decision_count,
        normalized_app_labels=("settings",),
        action_type_counts=(("tap", decision_count),),
    )


def test_canonical_config_is_source_only_and_valid() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_discovery_config(config)
    assert config["protocol_id"] == PROTOCOL_ID
    assert set(config["operation_limits"].values()) == {0}
    assert config["selection"]["role_assignment_allowed"] is False
    assert config["selection"]["query_state_selection_allowed"] is False
    result = validate_discovery_source_only(repository_root=ROOT)
    assert result["config_sha256"] == FROZEN_CONFIG_SHA256
    assert result["config_read_count"] == 1
    assert result["data_read_count"] == 0


def test_config_rejects_role_assignment_or_model_work() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    changed = copy.deepcopy(config)
    changed["selection"]["role_assignment_allowed"] = True
    with pytest.raises(ValueError, match="selection contract"):
        validate_discovery_config(changed)
    changed = copy.deepcopy(config)
    changed["operation_limits"]["policy_forward_count"] = 1
    with pytest.raises(ValueError, match="prohibit"):
        validate_discovery_config(changed)


@pytest.mark.parametrize(
    ("decision_count", "expected"),
    ((13, "long_13_16"), (16, "long_13_16"), (17, "long_17_24"), (64, "long_25_64")),
)
def test_length_strata(decision_count: int, expected: str) -> None:
    assert length_stratum(decision_count) == expected


@pytest.mark.parametrize("decision_count", (12, 65))
def test_length_strata_reject_old_or_unbounded_rows(decision_count: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        length_stratum(decision_count)


def test_projection_uses_new_salt_and_does_not_assign_a_role() -> None:
    projected = project_candidate(
        _candidate("trajectory-a", 18, 7),
        instruction_app_group_sha256_value="1" * 64,
        salt="new-study",
    )
    assert projected.selection_sha256 == long_selection_sha256(
        "trajectory-a", salt="new-study"
    )
    assert projected.length_stratum == "long_17_24"
    assert projected.instruction_app_group_sha256 == "1" * 64
    assert "role" not in projected.manifest_record()


def test_instruction_app_group_hash_is_normalized_and_content_free() -> None:
    left = instruction_app_group_sha256(
        "  Open\u3000SETTINGS and enable Wi-Fi  ",
        normalized_app_labels=("settings",),
    )
    right = instruction_app_group_sha256(
        "open settings AND enable wi-fi",
        normalized_app_labels=("settings",),
    )
    assert left == right
    assert normalize_instruction(" A\n B ") == "a b"
    assert len(left) == 64


def test_discovery_manifest_is_a_census_not_a_split() -> None:
    candidates = tuple(
        project_candidate(
            _candidate(f"trajectory-{index}", count, index),
            instruction_app_group_sha256_value=f"{index}" * 64,
            salt="study",
        )
        for index, count in enumerate((13, 16, 17, 25), start=1)
    )
    manifest = build_discovery_manifest(
        candidates=candidates,
        exclusion_counts={"decision_count_below_minimum": 6},
        source_row_count=10,
        config_sha256="a" * 64,
        base_config_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
    )
    assert manifest["selection"]["candidate_count"] == 4
    assert manifest["selection"]["role_assignment_count"] == 0
    assert manifest["selection"]["query_state_selection_count"] == 0
    assert "splits" not in manifest
    assert len(manifest["pool"]["trajectories"]) == 4
    assert manifest["pool"]["summary"]["decision_count_histogram"] == {
        "13": 1,
        "16": 1,
        "17": 1,
        "25": 1,
    }
    assert manifest["pool"]["summary"]["distinct_instruction_app_group_count"] == 4
    assert canonical_json_bytes(manifest) == canonical_json_bytes(manifest)


def test_manifest_rejects_duplicate_source_ids() -> None:
    repeated = project_candidate(
        _candidate("same", 13, 1),
        instruction_app_group_sha256_value="1" * 64,
        salt="study",
    )
    with pytest.raises(ValueError, match="unique"):
        build_discovery_manifest(
            candidates=(repeated, repeated),
            exclusion_counts={},
            source_row_count=2,
            config_sha256="a" * 64,
            base_config_sha256="b" * 64,
            source_manifest_sha256="c" * 64,
        )
