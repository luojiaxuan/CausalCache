from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalcache.set_utility_budget_deferral_truth_v1 import (
    INCOMPLETE_RESULT_STATUS,
    RESULT_SCHEMA,
    SCHEDULE_SCHEMA,
    SCHEDULE_STATUS,
    SEAL_SCHEMA,
    SEAL_STATUS,
    SELECTION_SCHEMA,
    SELECTION_STATUS,
    _runner_state_identity,
    materialize_budget_deferral_truth_schedule,
    reduce_budget_deferral_truth,
)
from causalcache.set_utility_heldout_evaluation import (
    COMPLETED_LABEL,
    SKIPPED_LABEL,
    canonical_json_bytes,
    sha256_file,
)
from causalcache.set_utility_train_heldout_contract import sha256_json
from causalcache.set_utility_variable_history import history_bin


def _write_json(path: Path, value: dict, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value, pretty=pretty) + b"\n")


def _write_signed(path: Path, value: dict, *, pretty: bool = False) -> dict:
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    result = {**unsigned, "content_sha256": sha256_json(unsigned)}
    _write_json(path, result, pretty=pretty)
    return result


def _fixture(tmp_path: Path) -> dict:
    config_path = tmp_path / "config.json"
    config = {
        "deployment": {
            "budgets": [1, 2, 3, 4],
            "post_evaluation_route_or_threshold_changes_allowed": False,
            "selection_by_budget": {
                "1": "recent",
                "2": "recent",
                "3": "deepsets_direct_conditional_marginal",
                "4": "deepsets_direct_conditional_marginal",
            },
            "selection_cardinality": "at_most_B",
        },
        "firewall": {
            "evaluation_features_allowed_after_this_freeze": True,
            "evaluation_truth_allowed_only_after_signed_selection_seal": True,
            "evaluation_truth_for_training_or_calibration": False,
        },
        "frozen_candidate_evaluation": {
            "history_bin_counts": {
                "long": 1,
                "medium": 1,
                "short": 1,
                "very_long": 1,
            },
            "historical_access_boundary": {
                "previously_consumed_trajectory_count": 2,
                "state_new_count": 2,
                "trajectory_new_state_count": 1,
                "trajectory_new_trajectory_count": 1,
            },
            "label_blind_selection_must_be_sealed_before_truth_access": True,
            "role": "evaluation",
            "state_count": 4,
            "state_inventory_sha256": "1" * 64,
        },
        "name": "causalcache_set_utility_budget_deferral_v1",
        "predictor": {
            "checkpoint_sha256": "2" * 64,
            "model_family": "deepsets_structured_marginal",
        },
        "schema_version": "1.0.0",
        "status": "FROZEN_BEFORE_CANDIDATE_EVALUATION",
    }
    _write_json(config_path, config, pretty=True)
    counts = (5, 9, 17, 33)
    trajectories = ("trajectory-a", "trajectory-a", "trajectory-b", "trajectory-c")
    slice_tags = (
        ["all"],
        ["all", "state_new"],
        ["all", "state_new", "trajectory_new"],
        ["all"],
    )
    logical_shards = (0, 0, 1, 2)
    records = []
    for index, (count, trajectory_id, slices, logical_shard) in enumerate(
        zip(counts, trajectories, slice_tags, logical_shards, strict=True)
    ):
        candidates = list(range(1, count + 1))
        recent = {str(budget): candidates[-budget:] for budget in range(1, 5)}
        direct = {
            "1": recent["1"],
            "2": recent["2"],
            "3": candidates[:3],
            "4": candidates[:4],
        }
        records.append(
            {
                "candidate_event_ids": candidates,
                "history_bin": history_bin(count),
                "latency_ms": {"selector": 1.0 + index},
                "logical_shard": logical_shard,
                "methods": {
                    "recent": recent,
                    "structured_deepsets_budget_deferral": direct,
                },
                "slices": slices,
                "state_id": f"{trajectory_id}:decision:{count + 1:03d}",
                "trajectory_id": trajectory_id,
            }
        )
    denominator = {
        "history_bin_counts": config["frozen_candidate_evaluation"][
            "history_bin_counts"
        ],
        "slice_counts": {
            "all": {"state_count": 4, "trajectory_count": 3},
            "state_new": {"state_count": 2, "trajectory_count": 2},
            "trajectory_new": {"state_count": 1, "trajectory_count": 1},
        },
        "state_count": 4,
        "trajectory_count": 3,
    }
    bindings = {
        "cache_content_sha256": "3" * 64,
        "cache_manifest_file_sha256": "4" * 64,
        "checkpoint_sha256": "2" * 64,
        "config_content_sha256": sha256_json(config),
        "config_file_sha256": sha256_file(config_path),
        "input_content_sha256": "5" * 64,
        "input_manifest_file_sha256": "6" * 64,
        "input_states_sha256": "7" * 64,
        "predictor_config_file_sha256": "9" * 64,
        "state_inventory_content_sha256": "1" * 64,
        "state_inventory_file_sha256": "8" * 64,
    }
    selection_path = tmp_path / "selection.json"
    selection = _write_signed(
        selection_path,
        {
            "bindings": bindings,
            "denominator": denominator,
            "method_semantics": {
                "recent": "deterministic_most_recent_at_most_B",
                "structured_deepsets_budget_deferral": (
                    "B1_B2_recent_B3_B4_direct_conditional_marginal"
                ),
            },
            "methods": ["recent", "structured_deepsets_budget_deferral"],
            "records": records,
            "schema_version": SELECTION_SCHEMA,
            "status": SELECTION_STATUS,
            "truth_accessed": False,
        },
    )
    seal_path = tmp_path / "selection-seal.json"
    seal = _write_signed(
        seal_path,
        {
            "bindings": bindings,
            "denominator": denominator,
            "schema_version": SEAL_SCHEMA,
            "selection_content_sha256": selection["content_sha256"],
            "selection_file_sha256": sha256_file(selection_path),
            "status": SEAL_STATUS,
            "truth_accessed": False,
        },
    )
    source_path = tmp_path / "source-manifest.json"
    trajectory_by_shard = {0: ["trajectory-a"], 1: ["trajectory-b"], 2: ["trajectory-c"]}
    source = {
        "shards": [
            {
                "logical_shard": logical,
                "sha256": f"{logical + 100:064x}",
                "trajectory_ids": trajectory_by_shard.get(logical, []),
            }
            for logical in range(256)
        ],
        "status": "COMPLETED_VARIABLE_HISTORY_SOURCE",
    }
    source["content_sha256"] = sha256_json(source)
    _write_json(source_path, source, pretty=True)
    return {
        "config_path": config_path,
        "records": records,
        "seal": seal,
        "seal_path": seal_path,
        "selection": selection,
        "selection_path": selection_path,
        "source": source,
        "source_path": source_path,
    }


def _materialize(fixture: dict, tmp_path: Path) -> Path:
    schedule_root = tmp_path / "schedule"
    summary = materialize_budget_deferral_truth_schedule(
        config_path=fixture["config_path"],
        selection_path=fixture["selection_path"],
        selection_seal_path=fixture["seal_path"],
        output_root=schedule_root,
        workers=8,
    )
    assert summary["schema_version"] == SCHEDULE_SCHEMA
    assert summary["status"] == SCHEDULE_STATUS
    assert summary["state_count"] == 4
    return schedule_root


def _write_terminals(
    fixture: dict,
    schedule_root: Path,
    label_root: Path,
    *,
    skipped_ids: set[str] = frozenset(),
    missing_ids: set[str] = frozenset(),
    full_anchor_distance: float = 0.0,
) -> None:
    records = {row["state_id"]: row for row in fixture["records"]}
    source_shas = {
        row["logical_shard"]: row["sha256"] for row in fixture["source"]["shards"]
    }
    for logical_shard in range(256):
        schedule_path = (
            schedule_root
            / "schedule-shards"
            / f"shard-{logical_shard:03d}-of-256.jsonl"
        )
        receipt_path = (
            schedule_root / "receipts" / f"shard-{logical_shard:03d}-of-256.json"
        )
        for line in schedule_path.read_text(encoding="utf-8").splitlines():
            schedule = json.loads(line)
            state_id = schedule["state_id"]
            if state_id in missing_ids:
                continue
            identity = _runner_state_identity(
                schedule=schedule,
                execution_config_sha256="a" * 64,
                scientific_config_sha256="b" * 64,
                schedule_receipt_sha256=sha256_file(receipt_path),
                source_revision="c" * 40,
                source_shard_sha256=source_shas[logical_shard],
            )
            common = {
                "execution_config_sha256": "a" * 64,
                "role": "evaluation",
                "scientific_config_sha256": "b" * 64,
                "source_revision": "c" * 40,
                "state_id": state_id,
                "state_identity_sha256": identity,
                "trajectory_id": records[state_id]["trajectory_id"],
            }
            if state_id in skipped_ids:
                terminal = {
                    **common,
                    "failure_class": "ReferenceActionMismatch",
                    "status": SKIPPED_LABEL,
                }
            else:
                distance_rows = []
                for coalition in schedule["coalitions"]:
                    subset = coalition["event_ids"]
                    sources = set(coalition["sources"])
                    if "anchor:full" in sources:
                        distance = full_anchor_distance
                    elif "anchor:empty" in sources:
                        distance = 1.0
                    elif "structured_deepsets_budget_deferral:B4" in sources:
                        distance = 0.2
                    elif "structured_deepsets_budget_deferral:B3" in sources:
                        distance = 0.4
                    elif "recent:B4" in sources:
                        distance = 0.6
                    elif "recent:B3" in sources:
                        distance = 0.7
                    elif "recent:B2" in sources:
                        distance = 0.8
                    else:
                        distance = 0.9
                    distance_rows.append(
                        {
                            "coalition_event_step_ids": subset,
                            "distance": distance,
                        }
                    )
                terminal = {
                    **common,
                    "candidate_event_step_ids": schedule["candidate_event_ids"],
                    "distance_rows": distance_rows,
                    "status": COMPLETED_LABEL,
                }
            path = label_root / "states" / f"{state_id.replace(':', '_')}.json"
            _write_json(path, terminal, pretty=True)


def test_schedule_is_selection_only_deduplicated_and_runner_compatible(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    schedule_root = _materialize(fixture, tmp_path)
    first = json.loads(
        next(
            line
            for line in (
                schedule_root / "schedule-shards" / "shard-000-of-256.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        )
    )
    subsets = [tuple(row["event_ids"]) for row in first["coalitions"]]
    assert len(subsets) == len(set(subsets)) == 8
    assert subsets[0] == ()
    assert subsets[-1] == tuple(first["candidate_event_ids"])
    assert first["role"] == "evaluation"
    assert {source for row in first["coalitions"] for source in row["sources"]} == {
        "anchor:empty",
        "anchor:full",
        "recent:B1",
        "recent:B2",
        "recent:B3",
        "recent:B4",
        "structured_deepsets_budget_deferral:B3",
        "structured_deepsets_budget_deferral:B4",
    }


def test_schedule_rejects_unsealed_or_truth_accessing_selection(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    selection = dict(fixture["selection"])
    selection.pop("content_sha256")
    selection["truth_accessed"] = True
    _write_signed(fixture["selection_path"], selection)
    with pytest.raises(ValueError, match="selection seal drifted"):
        materialize_budget_deferral_truth_schedule(
            config_path=fixture["config_path"],
            selection_path=fixture["selection_path"],
            selection_seal_path=fixture["seal_path"],
            output_root=tmp_path / "schedule",
        )


def test_reduce_reports_all_three_slice_coverage_and_paired_macro(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    schedule_root = _materialize(fixture, tmp_path)
    label_root = tmp_path / "labels"
    skipped_id = fixture["records"][1]["state_id"]
    _write_terminals(
        fixture,
        schedule_root,
        label_root,
        skipped_ids={skipped_id},
    )
    result = reduce_budget_deferral_truth(
        config_path=fixture["config_path"],
        selection_path=fixture["selection_path"],
        selection_seal_path=fixture["seal_path"],
        schedule_root=schedule_root,
        source_manifest_path=fixture["source_path"],
        label_roots=[label_root],
        output_path=tmp_path / "result.json",
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )
    assert result["schema_version"] == RESULT_SCHEMA
    assert result["status"] == INCOMPLETE_RESULT_STATUS
    assert result["slices"]["all"]["coverage"]["completed_state_count"] == 3
    assert result["slices"]["all"]["coverage"]["skipped_state_count"] == 1
    assert result["slices"]["state_new"]["coverage"]["expected_state_count"] == 2
    assert result["slices"]["trajectory_new"]["coverage"]["complete"] is True
    paired = result["slices"]["all"]["metrics"]["paired_delta"][
        "structured_deepsets_budget_deferral_minus_recent"
    ]["macro_B1_B4"]
    assert paired["point_estimate"] == pytest.approx(0.175)
    assert paired["cluster_count"] == 3
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    claimed = payload.pop("content_sha256")
    assert claimed == sha256_json(payload)


def test_reduce_rejects_nonzero_full_anchor(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    schedule_root = _materialize(fixture, tmp_path)
    label_root = tmp_path / "labels"
    _write_terminals(
        fixture,
        schedule_root,
        label_root,
        full_anchor_distance=0.01,
    )
    with pytest.raises(ValueError, match="full-history anchor is nonzero"):
        reduce_budget_deferral_truth(
            config_path=fixture["config_path"],
            selection_path=fixture["selection_path"],
            selection_seal_path=fixture["seal_path"],
            schedule_root=schedule_root,
            source_manifest_path=fixture["source_path"],
            label_roots=[label_root],
            output_path=tmp_path / "result.json",
            bootstrap_resamples=10,
        )


def test_reduce_reports_missing_terminals_without_dropping_the_denominator(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    schedule_root = _materialize(fixture, tmp_path)
    label_root = tmp_path / "labels"
    missing_id = fixture["records"][3]["state_id"]
    _write_terminals(
        fixture,
        schedule_root,
        label_root,
        missing_ids={missing_id},
    )
    result = reduce_budget_deferral_truth(
        config_path=fixture["config_path"],
        selection_path=fixture["selection_path"],
        selection_seal_path=fixture["seal_path"],
        schedule_root=schedule_root,
        source_manifest_path=fixture["source_path"],
        label_roots=[label_root],
        output_path=tmp_path / "result.json",
        bootstrap_resamples=10,
    )
    coverage = result["slices"]["all"]["coverage"]
    assert coverage["expected_state_count"] == 4
    assert coverage["completed_state_count"] == 3
    assert coverage["missing_state_ids"] == [missing_id]
    assert coverage["missing_state_count"] == 1
