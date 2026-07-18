from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from causalcache.set_utility_freeze_b_v1 import (
    STATUS,
    canonical_pretty_json_bytes,
    derive_freeze_b_manifest,
    load_freeze_b_config,
    validate_freeze_b_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "code/configs/causalcache_set_utility_freeze_b_v1.json"
CENSUS_PATH = ROOT / "data/manifests/set_utility_full_pool_census_v2.json"
MANIFEST_PATH = ROOT / "data/manifests/set_utility_freeze_b_v1.json"


def _config() -> dict:
    return load_freeze_b_config(ROOT)


def _census() -> dict:
    return json.loads(CENSUS_PATH.read_text(encoding="utf-8"))


def test_freeze_b_source_config_is_valid_and_still_forbids_training() -> None:
    result = validate_freeze_b_config(_config(), repository_root=ROOT)
    assert result["status"] == "VALID_SET_UTILITY_FREEZE_B_V1_SOURCE"
    assert result["trajectory_count"] == 1200
    assert result["query_state_count"] == 2400
    assert result["maximum_raw_label_rows"] == 328800
    assert result["training_authorized"] is False
    assert result["label_generation_authorized"] is False


def test_freeze_b_roster_is_deterministic_balanced_and_group_disjoint() -> None:
    config = _config()
    census = _census()
    first = derive_freeze_b_manifest(config, census, repository_root=ROOT)
    second = derive_freeze_b_manifest(config, census, repository_root=ROOT)
    assert first == second
    assert first["status"] == STATUS
    assert first["summary"]["trajectory_count"] == 1200
    assert first["summary"]["query_state_count"] == 2400
    assert first["summary"]["role_trajectory_counts"] == {
        "evaluation": 100,
        "train": 1000,
        "tune": 100,
    }
    assert first["summary"]["stratum_trajectory_counts"] == {
        "decisions_10_17": 400,
        "decisions_18_plus": 400,
        "decisions_6_9": 400,
    }
    assignments = first["assignments"]
    assert len({row["source_id"] for row in assignments}) == 1200
    assert (
        len({row["instruction_app_group_sha256"] for row in assignments})
        == 1200
    )
    assert first["split_audit"]["overlap_counts"] == {
        "forbidden_consumed_group": 0,
        "forbidden_source": 0,
        "instruction_app_group_partition": 0,
        "legacy_group_partition": 0,
        "legacy_role": 0,
        "source_identity": 0,
        "trajectory_role": 0,
    }


def test_committed_freeze_b_manifest_replays_exactly() -> None:
    expected = derive_freeze_b_manifest(_config(), _census(), repository_root=ROOT)
    assert MANIFEST_PATH.read_bytes() == canonical_pretty_json_bytes(expected)


def test_freeze_b_queries_are_exactly_two_distinct_policy_blind_states() -> None:
    manifest = derive_freeze_b_manifest(_config(), _census(), repository_root=ROOT)
    by_source = Counter(row["source_id"] for row in manifest["query_states"])
    assert set(by_source.values()) == {2}
    assert manifest["summary"]["initial_candidate_count_histogram"]["4"] == 400
    assert manifest["summary"]["initial_candidate_count_histogram"]["8"] == 400
    assert manifest["summary"]["initial_candidate_count_histogram"]["16"] >= 400
    for row in manifest["query_states"]:
        assert row["processor_candidate_freeze_status"] == (
            "PENDING_SEPARATE_EXECUTION"
        )
        assert row["current_equivalent_event_step_id"] == row["decision_step_id"] - 1
        assert row["initial_candidate_event_step_ids"] == sorted(
            row["initial_candidate_event_step_ids"]
        )
        assert row["current_equivalent_event_step_id"] not in row[
            "initial_candidate_event_step_ids"
        ]
        assert 4 <= row["initial_candidate_count"] <= 16
    assert first_nonzero_forbidden_operation(manifest) is None


def first_nonzero_forbidden_operation(manifest: dict) -> str | None:
    allowed = {
        "p0_manifest_read_count",
        "trajectory_role_assignment_count",
        "query_state_selection_count",
    }
    for key, value in manifest["operation_counts"].items():
        if key not in allowed and value != 0:
            return key
    return None


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["authorization"].__setitem__(
                "train_predictor", True
            ),
            "authorization drifted",
        ),
        (
            lambda value: value["selection"][
                "role_stratum_trajectory_counts"
            ]["evaluation"].__setitem__("decisions_6_9", 34),
            "role totals drifted|balanced",
        ),
        (
            lambda value: value["training"].__setitem__("seeds", [17]),
            "training seeds drifted",
        ),
        (
            lambda value: value["features"].__setitem__(
                "schema_id", "lightweight_only"
            ),
            "feature schema drifted",
        ),
    ],
)
def test_freeze_b_config_drift_fails_closed(mutation, match: str) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    with pytest.raises(ValueError, match=match):
        validate_freeze_b_config(config, repository_root=ROOT)


def test_freeze_b_config_file_is_canonical_pretty_json() -> None:
    parsed = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = json.dumps(
        parsed,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    assert CONFIG_PATH.read_text(encoding="utf-8") == expected
