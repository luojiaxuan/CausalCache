from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from causalcache.set_utility_freeze_b_v1 import canonical_pretty_json_bytes
from causalcache.set_utility_freeze_b_v2 import (
    DEFAULT_OUTPUT_PATH,
    STATUS,
    derive_freeze_b_v2_manifest,
    load_freeze_b_v2_config,
    validate_freeze_b_v2_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "code/configs/causalcache_set_utility_freeze_b_v2_terminal_index_repair.json"
)
P0_PATH = ROOT / "data/manifests/set_utility_full_pool_census_v2.json"
MANIFEST_PATH = ROOT / DEFAULT_OUTPUT_PATH


def _config() -> dict:
    return load_freeze_b_v2_config(ROOT)


def _p0() -> dict:
    return json.loads(P0_PATH.read_text(encoding="utf-8"))


def test_terminal_index_repair_source_is_strictly_policy_blind() -> None:
    result = validate_freeze_b_v2_config(_config(), repository_root=ROOT)
    assert result == {
        "status": "VALID_SET_UTILITY_FREEZE_B_V2_TERMINAL_INDEX_REPAIR_SOURCE",
        "config_sha256": result["config_sha256"],
        "parent_config_sha256": (
            "df8c00bfddda7589e3c9b58cc36f5cbe305ad3fee9c6a8bc4272888a6c648440"
        ),
        "trajectory_count": 1200,
        "query_state_count": 2400,
        "raw_source_access_authorized": False,
        "processor_authorized": False,
        "label_generation_authorized": False,
        "training_authorized": False,
    }


def test_repaired_roster_and_queries_are_deterministic_and_disjoint() -> None:
    first = derive_freeze_b_v2_manifest(_config(), _p0(), repository_root=ROOT)
    second = derive_freeze_b_v2_manifest(_config(), _p0(), repository_root=ROOT)
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
    assert first["selection"]["too_short_for_two_states_by_stratum"] == {}
    assert first["summary"]["initial_candidate_count_histogram"] == {
        "4": 400,
        "5": 56,
        "6": 66,
        "7": 106,
        "8": 572,
        "9": 82,
        "10": 63,
        "11": 54,
        "12": 49,
        "13": 59,
        "14": 42,
        "15": 32,
        "16": 819,
    }
    assert first["summary"]["initial_exact_schedule_raw_label_rows"] == 171730
    assert first["initial_exact_schedule"]["operations"][
        "total_model_operations"
    ] == 181330
    assignments = first["assignments"]
    assert len({record["source_id"] for record in assignments}) == 1200
    assert len(
        {record["instruction_app_group_sha256"] for record in assignments}
    ) == 1200
    assert first["split_audit"]["overlap_counts"] == {
        "forbidden_consumed_group": 0,
        "forbidden_source": 0,
        "instruction_app_group_partition": 0,
        "legacy_group_partition": 0,
        "legacy_role": 0,
        "source_identity": 0,
        "trajectory_role": 0,
    }


def test_terminal_query_uses_decision_count_plus_one() -> None:
    manifest = derive_freeze_b_v2_manifest(_config(), _p0(), repository_root=ROOT)
    assignment_by_source = {
        record["source_id"]: record for record in manifest["assignments"]
    }
    by_source = Counter(record["source_id"] for record in manifest["query_states"])
    assert set(by_source.values()) == {2}
    for record in manifest["query_states"]:
        assignment = assignment_by_source[record["source_id"]]
        assert 2 <= record["decision_step_id"] <= assignment["decision_count"] + 1
        assert record["current_equivalent_event_step_id"] == (
            record["decision_step_id"] - 1
        )
        assert record["current_equivalent_event_step_id"] not in record[
            "initial_candidate_event_step_ids"
        ]
        if record["query_kind"] == "terminal":
            assert record["decision_step_id"] == assignment["decision_count"] + 1
            assert record["decision_step_id"] == assignment[
                "terminal_decision_step_id"
            ]


def test_parent_v1_terminal_bug_is_preserved_as_evidence_not_reused() -> None:
    parent = json.loads(
        (ROOT / "data/manifests/set_utility_freeze_b_v1.json").read_text(
            encoding="utf-8"
        )
    )
    repaired = derive_freeze_b_v2_manifest(
        _config(), _p0(), repository_root=ROOT
    )
    parent_assignment = {
        record["source_id"]: record for record in parent["assignments"]
    }
    for record in parent["query_states"]:
        if record["query_kind"] == "terminal":
            assert record["decision_step_id"] == parent_assignment[
                record["source_id"]
            ]["decision_count"]
    assert repaired["repair"]["parent_v1_status"] == (
        "INVALID_QUERY_PLAN_TERMINAL_OFF_BY_ONE"
    )
    assert repaired["repair"][
        "parent_v1_query_states_must_not_be_used_for_processor_or_labels"
    ] is True


def test_committed_repair_manifest_replays_exactly() -> None:
    if not MANIFEST_PATH.exists():
        pytest.skip("repair manifest materializes after the source contract is valid")
    expected = derive_freeze_b_v2_manifest(
        _config(), _p0(), repository_root=ROOT
    )
    assert MANIFEST_PATH.read_bytes() == canonical_pretty_json_bytes(expected)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["repair"].__setitem__(
                "terminal_decision_step_formula", "decision_count"
            ),
            "repair semantics drifted",
        ),
        (
            lambda value: value["authorization"].__setitem__(
                "read_raw_source_shards", True
            ),
            "authorization drifted",
        ),
        (
            lambda value: value["output"].__setitem__(
                "contains_raw_instruction", True
            ),
            "output contract drifted",
        ),
    ],
)
def test_terminal_index_repair_config_drift_fails_closed(
    mutation,
    match: str,
) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    with pytest.raises(ValueError, match=match):
        validate_freeze_b_v2_config(config, repository_root=ROOT)


def test_terminal_index_repair_config_is_canonical_pretty_json() -> None:
    parsed = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = json.dumps(
        parsed,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    assert CONFIG_PATH.read_text(encoding="utf-8") == expected
