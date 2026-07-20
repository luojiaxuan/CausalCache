from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import serialize_gui_owl_v2_1_teacher_target
from causalcache.set_utility_native_replay import (
    NATIVE_REPLAY_SHARD_STATUS,
    NATIVE_REPLAY_STATUS,
    canonical_json_bytes,
    compare_action_components,
    signed_content_hash_is_valid,
)
from causalcache.set_utility_native_replay_evaluation import (
    COMPLETED_NATIVE_REPLAY_EVALUATION,
    INCOMPLETE_NATIVE_REPLAY_EVALUATION,
    evaluate_native_replay,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value, pretty=True))


def _signed(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)[:-1]
    ).hexdigest()
    return result


def _action_for(ordinal: int) -> GUIOwlV2Action:
    if ordinal % 2 == 0:
        return GUIOwlV2Action(action="click", coordinate=(20, 30))
    return GUIOwlV2Action(action="type", text="ABC")


def _different_action(reference: GUIOwlV2Action) -> GUIOwlV2Action:
    if reference.action == "click":
        return GUIOwlV2Action(action="click", coordinate=(40, 50))
    return GUIOwlV2Action(action="type", text="wrong")


def _coalitions() -> list[dict[str, object]]:
    return [
        {
            "event_ids": [1],
            "sources": [f"winner:B{budget}" for budget in range(1, 5)],
        },
        {
            "event_ids": [2],
            "sources": [f"ocr_rgb:B{budget}" for budget in range(1, 5)],
        },
        {
            "event_ids": [5],
            "sources": [f"recent:B{budget}" for budget in range(1, 5)],
        },
    ]


def _build_artifacts(
    tmp_path: Path,
    *,
    include_text_actions: bool = True,
    selector_p95_ms: float = 5.0,
    winner_exact: bool = True,
) -> dict[str, Path]:
    schedule_root = tmp_path / "schedule"
    terminal_root = tmp_path / "terminals"
    selections_path = tmp_path / "selections.json"
    heldout_path = tmp_path / "heldout.json"
    config_sha = "a" * 64
    inventory_sha = "b" * 64
    source_sha = "c" * 64
    reference_inventory_sha = "d" * 64
    winner_model = "set_transformer"
    selection_records = []
    schedules_by_shard: dict[int, list[dict[str, object]]] = {
        index: [] for index in range(256)
    }
    for ordinal in range(320):
        trajectory_id = f"trajectory-{ordinal // 5:03d}"
        state_id = f"{trajectory_id}:decision:{ordinal % 5 + 6:03d}"
        logical_shard = ordinal % 256
        candidates = [1, 2, 3, 4, 5]
        methods = {
            winner_model: {str(budget): [1] for budget in range(1, 5)},
            "recent": {str(budget): [5] for budget in range(1, 5)},
            "ocr_rgb": {str(budget): [2] for budget in range(1, 5)},
        }
        selection_records.append(
            {
                "candidate_event_ids": candidates,
                "logical_shard": logical_shard,
                "methods": methods,
                "state_id": state_id,
                "tracks": ["exact_oracle"],
                "trajectory_id": trajectory_id,
            }
        )
        reference = (
            _action_for(ordinal)
            if include_text_actions
            else GUIOwlV2Action(action="click", coordinate=(20, 30))
        )
        schedule = {
            "candidate_event_ids": candidates,
            "coalitions": _coalitions(),
            "logical_shard": logical_shard,
            "reference": {
                "canonical_action": reference.arguments(),
                "serialized_action": serialize_gui_owl_v2_1_teacher_target(reference),
                "terminal_sha256": hashlib.sha256(state_id.encode()).hexdigest(),
            },
            "role": "evaluation",
            "state_id": state_id,
            "trajectory_id": trajectory_id,
            "winner_model": winner_model,
        }
        schedules_by_shard[logical_shard].append(schedule)

    selections = _signed(
        {
            "config_sha256": config_sha,
            "inventory_sha256": inventory_sha,
            "model_artifacts": {winner_model: {"checkpoint_sha256": "e" * 64}},
            "records": selection_records,
            "schema_version": "1.0.0",
            "status": "SEALED_SET_UTILITY_HELDOUT_SELECTIONS",
        }
    )
    _write_json(selections_path, selections)
    selections_sha = hashlib.sha256(selections_path.read_bytes()).hexdigest()
    heldout = _signed(
        {
            "bindings": {
                "config_sha256": config_sha,
                "selections_sha256": selections_sha,
            },
            "coverage": {
                "complete": True,
                "completed_state_count": 805,
                "expected_state_count": 805,
                "missing_state_ids": [],
                "skipped_state_count": 0,
            },
            "method_summaries": {
                winner_model: {
                    "latency": {
                        "selector_total_from_cached_source_tokens": {
                            "p95_ms": selector_p95_ms
                        }
                    }
                }
            },
            "schema_version": "1.0.0",
            "status": "COMPLETED_SET_UTILITY_HELDOUT_EVALUATION",
            "winner": {"model": winner_model, "verdict": "GO"},
        }
    )
    _write_json(heldout_path, heldout)
    heldout_sha = hashlib.sha256(heldout_path.read_bytes()).hexdigest()

    bindings = {
        "config_sha256": config_sha,
        "heldout_result_sha256": heldout_sha,
        "inventory_sha256": inventory_sha,
        "reference_terminal_inventory_sha256": reference_inventory_sha,
        "selections_sha256": selections_sha,
        "source_manifest_sha256": source_sha,
    }
    shard_records = []
    total_coalitions = 0
    for logical_shard in range(256):
        rows = schedules_by_shard[logical_shard]
        schedule_path = (
            schedule_root
            / "schedule-shards"
            / f"shard-{logical_shard:03d}-of-256.jsonl"
        )
        schedule_path.parent.mkdir(parents=True, exist_ok=True)
        schedule_bytes = b"".join(canonical_json_bytes(row) for row in rows)
        schedule_path.write_bytes(schedule_bytes)
        schedule_sha = hashlib.sha256(schedule_bytes).hexdigest()
        coalition_count = sum(len(row["coalitions"]) for row in rows)
        total_coalitions += coalition_count
        receipt = {
            **bindings,
            "coalition_count": coalition_count,
            "logical_shard": logical_shard,
            "schedule_sha256": schedule_sha,
            "state_count": len(rows),
            "status": NATIVE_REPLAY_SHARD_STATUS,
        }
        receipt_path = (
            schedule_root / "receipts" / f"shard-{logical_shard:03d}-of-256.json"
        )
        _write_json(receipt_path, receipt)
        shard_records.append(
            {
                "coalition_count": coalition_count,
                "logical_shard": logical_shard,
                "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                "schedule_sha256": schedule_sha,
                "state_count": len(rows),
            }
        )
    manifest = _signed(
        {
            **bindings,
            "budgets": [1, 2, 3, 4],
            "coalition_count_after_deduplication": total_coalitions,
            "logical_shard_count": 256,
            "methods": ["winner", "recent", "ocr_rgb"],
            "schema_version": "1.0.0",
            "shards": shard_records,
            "state_count": 320,
            "status": NATIVE_REPLAY_STATUS,
            "winner_model": winner_model,
        }
    )
    _write_json(schedule_root / "manifest.json", manifest)

    for ordinal, selection in enumerate(selection_records):
        state_id = selection["state_id"]
        schedule = next(
            row
            for row in schedules_by_shard[selection["logical_shard"]]
            if row["state_id"] == state_id
        )
        reference = (
            _action_for(ordinal)
            if include_text_actions
            else GUIOwlV2Action(action="click", coordinate=(20, 30))
        )
        state_identity = hashlib.sha256(("identity:" + state_id).encode()).hexdigest()
        records = []
        for coalition_ordinal, coalition in enumerate(schedule["coalitions"]):
            event_id = coalition["event_ids"][0]
            identity = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "coalition": coalition,
                        "ordinal": coalition_ordinal,
                        "state_identity_sha256": state_identity,
                    }
                )
            ).hexdigest()
            latency_seconds = {1: 0.1, 2: 0.3, 5: 0.2}[event_id]
            if event_id == 2:
                record = {
                    "coalition_event_step_ids": coalition["event_ids"],
                    "coalition_identity_sha256": identity,
                    "failure": {"message": "bad output", "type": "ValueError"},
                    "generation_metadata": {"latency_seconds": latency_seconds},
                    "native_output": "not a native action",
                    "sources": coalition["sources"],
                    "status": "FAILED_SET_UTILITY_NATIVE_REPLAY_PARSE",
                }
            else:
                prediction = (
                    reference
                    if event_id == 1 and winner_exact
                    else _different_action(reference)
                )
                record = {
                    "coalition_event_step_ids": coalition["event_ids"],
                    "coalition_identity_sha256": identity,
                    "comparison": compare_action_components(reference, prediction),
                    "generation_metadata": {"latency_seconds": latency_seconds},
                    "native_output": serialize_gui_owl_v2_1_teacher_target(prediction),
                    "predicted_action": prediction.arguments(),
                    "sources": coalition["sources"],
                    "status": "COMPLETED_SET_UTILITY_NATIVE_REPLAY_COALITION",
                }
            records.append(record)
        terminal = {
            "candidate_event_step_ids": selection["candidate_event_ids"],
            "records": records,
            "reference": schedule["reference"],
            "role": "evaluation",
            "source_revision": "f" * 40,
            "state_id": state_id,
            "state_identity_sha256": state_identity,
            "status": "COMPLETED_SET_UTILITY_NATIVE_REPLAY_STATE",
            "trajectory_id": selection["trajectory_id"],
            "winner_model": winner_model,
            "worker_index": 0,
        }
        _write_json(
            terminal_root / "states" / f"{state_id.replace(':', '_')}.json",
            terminal,
        )
    return {
        "heldout": heldout_path,
        "schedule": schedule_root,
        "selections": selections_path,
        "terminals": terminal_root,
    }


def _evaluate(paths: dict[str, Path]) -> dict[str, object]:
    return evaluate_native_replay(
        schedule_root=paths["schedule"],
        terminal_roots=(paths["terminals"],),
        selections_path=paths["selections"],
        heldout_result_path=paths["heldout"],
    )


def test_native_replay_reducer_reports_component_metrics_and_latency_gate(
    tmp_path: Path,
) -> None:
    result = _evaluate(_build_artifacts(tmp_path))
    assert result["status"] == COMPLETED_NATIVE_REPLAY_EVALUATION
    assert result["coverage"]["completed_state_count"] == 320
    assert result["method_summaries"]["winner"]["by_budget"]["1"][
        "canonical_action_exact"
    ]["trajectory_equal_rate"] == 1.0
    assert result["method_summaries"]["recent"]["by_budget"]["4"][
        "action_type_exact"
    ]["trajectory_equal_rate"] == 1.0
    assert result["method_summaries"]["recent"]["by_budget"]["1"][
        "target_exact"
    ]["denominator"] == 160
    assert result["method_summaries"]["recent"]["by_budget"]["1"][
        "text_nfkc_exact"
    ]["denominator"] == 160
    assert result["method_summaries"]["ocr_rgb"]["by_budget"]["2"][
        "parse_coverage"
    ]["trajectory_equal_rate"] == 0.0
    assert result["comparisons"]["winner_minus_recent"]["by_budget"]["3"][
        "canonical_action_exact"
    ]["trajectory_equal_delta"] == 1.0
    assert result["all_native_replay_generation_latency_ms"]["p95_ms"] == 300.0
    assert result["native_full_policy_generation_latency_ms"]["p95_ms"] == 100.0
    assert result["deployment_latency_gate"]["ratio"] == pytest.approx(0.05)
    assert result["deployment_latency_gate"]["verdict"] == "GO"
    assert result["behavior_recovery_gate"]["verdict"] == "GO"
    assert result["behavior_recovery_gate"]["requirements"][
        "primary_bootstrap_lower_strictly_positive_vs_recent_and_ocr_rgb"
    ] is True
    assert result["closed_loop_authorization"] == {
        "authorized": True,
        "requires": [
            "behavior_recovery_gate.GO",
            "deployment_latency_gate.GO",
        ],
        "verdict": "GO",
    }
    assert signed_content_hash_is_valid(result)


def test_native_replay_reducer_fails_closed_on_missing_exact_state(
    tmp_path: Path,
) -> None:
    paths = _build_artifacts(tmp_path)
    next((paths["terminals"] / "states").glob("*.json")).unlink()
    result = _evaluate(paths)
    assert result["status"] == INCOMPLETE_NATIVE_REPLAY_EVALUATION
    assert result["coverage"]["complete"] is False
    assert result["coverage"]["completed_state_count"] == 319
    assert len(result["coverage"]["missing_state_ids"]) == 1
    assert result["deployment_latency_gate"]["verdict"] == "NO_GO"
    assert result["behavior_recovery_gate"]["verdict"] == "NO_GO"
    assert result["closed_loop_authorization"]["authorized"] is False
    assert result["method_summaries"] == {}


def test_native_replay_reducer_applies_strict_latency_ratio_gate(
    tmp_path: Path,
) -> None:
    result = _evaluate(_build_artifacts(tmp_path, selector_p95_ms=11.0))
    assert result["coverage"]["complete"] is True
    assert result["deployment_latency_gate"]["ratio"] == pytest.approx(0.11)
    assert result["deployment_latency_gate"]["verdict"] == "NO_GO"
    assert result["behavior_recovery_gate"]["verdict"] == "GO"
    assert result["closed_loop_authorization"]["verdict"] == "NO_GO"


def test_native_replay_reducer_requires_scientific_behavior_recovery_go(
    tmp_path: Path,
) -> None:
    result = _evaluate(_build_artifacts(tmp_path, winner_exact=False))
    assert result["coverage"]["complete"] is True
    assert result["deployment_latency_gate"]["verdict"] == "GO"
    behavior = result["behavior_recovery_gate"]
    assert behavior["verdict"] == "NO_GO"
    assert behavior["comparisons"]["winner_minus_recent"][
        "primary_paired_trajectory_bootstrap"
    ]["lower_strictly_positive"] is False
    assert result["closed_loop_authorization"]["authorized"] is False


def test_native_replay_reducer_fails_closed_when_applicable_metric_is_unjudgable(
    tmp_path: Path,
) -> None:
    result = _evaluate(_build_artifacts(tmp_path, include_text_actions=False))
    behavior = result["behavior_recovery_gate"]
    assert behavior["requirements"]["all_B1_B4_action_metrics_judgable"] is False
    assert behavior["comparisons"]["winner_minus_recent"]["by_budget"]["1"][
        "text_nfkc_exact"
    ]["judgable"] is False
    assert behavior["verdict"] == "NO_GO"
    assert result["closed_loop_authorization"]["authorized"] is False


def test_native_replay_reducer_rejects_signed_input_drift(tmp_path: Path) -> None:
    paths = _build_artifacts(tmp_path)
    heldout = json.loads(paths["heldout"].read_text(encoding="utf-8"))
    heldout["winner"]["model"] = "deepsets"
    _write_json(paths["heldout"], heldout)
    with pytest.raises(ValueError, match="does not authorize|binding"):
        _evaluate(paths)
