from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalcache.set_utility_heldout_evaluation import canonical_json_bytes, sha256_file
from causalcache.set_utility_heldout_truth_schedule import (
    FORMAL_MANIFEST_STATUS,
    FORMAL_RECEIPT_STATUS,
    LABEL_STATUS,
)
from causalcache.set_utility_post_selection_truth import (
    METHODS,
    RESULT_SCHEMA,
    SCHEDULE_SCHEMA,
    _merge_distance,
    plan_post_selection_truth,
    reduce_post_selection_truth,
)
from causalcache.set_utility_train_heldout_contract import sha256_json
from causalcache.set_utility_variable_history import history_bin


def _write_compact(path: Path, value: dict) -> dict:
    signed = {**value, "content_sha256": sha256_json(value)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(signed) + b"\n")
    return signed


def _fixture(tmp_path: Path) -> dict:
    input_root = tmp_path / "input"
    input_root.mkdir()
    states = []
    checkpoint_states = []
    selector_records = []
    candidate_counts = (5, 9, 17, 33)
    for index in range(256):
        count = candidate_counts[index // 64]
        candidates = list(range(1, count + 1))
        state_id = f"trajectory-{index % 64:03d}:decision:{index // 64 + 6:03d}"
        trajectory_id = f"trajectory-{index % 64:03d}"
        recent = {
            str(budget): candidates[-budget:] for budget in range(1, 5)
        }
        first = {str(budget): candidates[:budget] for budget in range(1, 5)}
        distances = {(): 1.0, tuple(candidates): 0.0}
        for budget in range(1, 5):
            distances[tuple(recent[str(budget)])] = 1.0 - 0.2 * budget
            if not (index == 0 and budget == 2):
                distances[tuple(first[str(budget)])] = 1.0 - (0.2 * budget + 0.1)
        states.append(
            {
                "candidate_event_step_ids": candidates,
                "distance_rows": [
                    {
                        "coalition_event_step_ids": list(subset),
                        "distance": distance,
                    }
                    for subset, distance in sorted(
                        distances.items(), key=lambda item: (len(item[0]), item[0])
                    )
                ],
                "logical_shard": index,
                "role": "train",
                "state_id": state_id,
                "trajectory_id": trajectory_id,
            }
        )
        name = history_bin(count)
        checkpoint_states.append(
            {
                "candidate_count": count,
                "history_bin": name,
                "state_id": state_id,
                "trajectory_id": trajectory_id,
            }
        )
        selector_records.append(
            {
                "candidate_event_ids": candidates,
                "history_bin": name,
                "latency_ms": {method: 1.0 for method in METHODS},
                "methods": {
                    "recent": recent,
                    "set_transformer": first,
                    "set_transformer_hybrid": first,
                    "structured_deepsets": recent,
                    "structured_deepsets_hybrid": first,
                },
                "state_id": state_id,
                "trajectory_id": trajectory_id,
            }
        )
    states_path = input_root / "states.jsonl"
    states_path.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in states)
    )
    input_content = "a" * 64
    input_manifest = {
        "content_sha256": input_content,
        "evaluation_labels_included": False,
        "state_count": len(states),
        "states_jsonl": "states.jsonl",
        "states_sha256": sha256_file(states_path),
    }
    input_manifest_path = input_root / "manifest.json"
    input_manifest_path.write_bytes(canonical_json_bytes(input_manifest) + b"\n")
    heldout_path = tmp_path / "heldout.json"
    heldout = _write_compact(
        heldout_path,
        {
            "checkpoint_state_ids": [row["state_id"] for row in checkpoint_states],
            "checkpoint_states": checkpoint_states,
            "firewall": {
                "allowed_role": "train",
                "evaluation_access": False,
                "tune_access": False,
            },
            "schema_version": "1.0.0",
        },
    )
    selector_path = tmp_path / "selector.json"
    selector = _write_compact(
        selector_path,
        {
            "bindings": {
                "cache_content_sha256": "b" * 64,
                "heldout_manifest_content_sha256": heldout["content_sha256"],
                "heldout_manifest_file_sha256": sha256_file(heldout_path),
                "input_content_sha256": input_content,
                "input_states_sha256": sha256_file(states_path),
                "set_transformer_config_sha256": "c" * 64,
                "source_manifest_file_sha256": "4" * 64,
                "structured_deepsets_config_sha256": "d" * 64,
            },
            "checkpoints": {
                "set_transformer": {
                    "model_family": "set_transformer_direct_marginal",
                    "sha256": "e" * 64,
                    "variant": "set",
                },
                "structured_deepsets": {
                    "model_family": "deepsets_structured_marginal",
                    "sha256": "f" * 64,
                    "variant": "deep",
                },
            },
            "hybrid_contract": {"advantage_threshold": 0.001},
            "latency_summary_ms": {},
            "methods": list(METHODS),
            "records": selector_records,
            "schema_version": "causalcache.direct_on_policy_unified_selections.v1",
            "status": "COMPLETED_DIRECT_ON_POLICY_UNIFIED_SELECTIONS",
            "truth_labels_accessed": False,
        },
    )
    common = {
        "execution_config_sha256": "1" * 64,
        "heldout_manifest_content_sha256": heldout["content_sha256"],
        "heldout_manifest_file_sha256": sha256_file(heldout_path),
        "input_content_sha256": input_content,
        "input_manifest_file_sha256": sha256_file(input_manifest_path),
        "scientific_config_sha256": "2" * 64,
        "source_content_sha256": "3" * 64,
        "source_manifest_file_sha256": "4" * 64,
        "source_revision": "5" * 40,
    }
    root_a = _truth_root(
        tmp_path / "root-a",
        state=states[0],
        rows=[([1, 2], 0.5), (states[0]["candidate_event_step_ids"], 0.0)],
        common=common,
        schedule_family="unrelated_family_a",
    )
    root_b = _truth_root(
        tmp_path / "root-b",
        state=states[1],
        rows=[([], 1.0), (states[1]["candidate_event_step_ids"], 0.0)],
        common=common,
        schedule_family="unrelated_family_b",
    )
    return {
        "heldout_path": heldout_path,
        "input_root": input_root,
        "root_a": root_a,
        "root_b": root_b,
        "selector": selector,
        "selector_path": selector_path,
    }


def _truth_root(
    root: Path,
    *,
    state: dict,
    rows: list[tuple[list[int], float]],
    common: dict,
    schedule_family: str,
) -> Path:
    states_root = root / "states"
    states_root.mkdir(parents=True)
    state_identity = "6" * 64
    terminal = {
        "candidate_event_step_ids": state["candidate_event_step_ids"],
        "distance_rows": [
            {"coalition_event_step_ids": subset, "distance": distance}
            for subset, distance in rows
        ],
        "execution_config_sha256": common["execution_config_sha256"],
        "role": "train",
        "scientific_config_sha256": common["scientific_config_sha256"],
        "source_revision": common["source_revision"],
        "state_id": state["state_id"],
        "state_identity_sha256": state_identity,
        "status": LABEL_STATUS,
        "trajectory_id": state["trajectory_id"],
    }
    terminal_path = states_root / f"{state['state_id'].replace(':', '_')}.json"
    terminal_path.write_bytes(canonical_json_bytes(terminal) + b"\n")
    manifest_path = root / "formal-truth-manifest.json"
    manifest = _write_compact(
        manifest_path,
        {
            **common,
            "coalition_count": len(rows),
            "model_schedules": {
                "schedule": {
                    "content_sha256": "7" * 64,
                    "file_sha256": "8" * 64,
                    "model_family": schedule_family,
                }
            },
            "schema_version": "causalcache.formal_heldout_truth_manifest.v1",
            "state_count": 1,
            "states": [
                {
                    "candidate_event_ids": state["candidate_event_step_ids"],
                    "coalition_count": len(rows),
                    "logical_shard": state["logical_shard"],
                    "state_id": state["state_id"],
                    "state_identity_sha256": state_identity,
                    "terminal_file_sha256": sha256_file(terminal_path),
                    "terminal_relative_path": terminal_path.relative_to(root).as_posix(),
                    "trajectory_id": state["trajectory_id"],
                }
            ],
            "status": FORMAL_MANIFEST_STATUS,
            "truth_schedule_summary_content_sha256": "9" * 64,
            "truth_schedule_summary_file_sha256": "0" * 64,
        },
    )
    _write_compact(
        root / "formal-truth-receipt.json",
        {
            "coalition_count": len(rows),
            "formal_manifest_content_sha256": manifest["content_sha256"],
            "formal_manifest_file_sha256": sha256_file(manifest_path),
            "heldout_manifest_content_sha256": common[
                "heldout_manifest_content_sha256"
            ],
            "input_content_sha256": common["input_content_sha256"],
            "schema_version": "causalcache.formal_heldout_truth_receipt.v1",
            "source_manifest_file_sha256": common["source_manifest_file_sha256"],
            "state_count": 1,
            "status": FORMAL_RECEIPT_STATUS,
            "truth_schedule_summary_content_sha256": "9" * 64,
            "truth_schedule_summary_file_sha256": "0" * 64,
        },
    )
    return root


def test_plan_unions_roots_across_schedule_families_and_only_requests_missing(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "schedule.json"
    schedule = plan_post_selection_truth(
        selector_output_path=fixture["selector_path"],
        input_root=fixture["input_root"],
        heldout_manifest_path=fixture["heldout_path"],
        truth_roots=[fixture["root_b"]],
        output_path=output,
    )
    assert schedule["schema_version"] == SCHEDULE_SCHEMA
    assert schedule["state_count"] == 256
    assert schedule["missing_coalition_count"] == 1
    assert schedule["epoch_checkpoints"] == [
        {"checkpoint_sha256": "e" * 64, "epoch": 1}
    ]
    assert schedule["records"][0]["missing_coalitions"] == [
        {
            "event_ids": [1, 2],
            "source": "set_transformer:B2+set_transformer_hybrid:B2+structured_deepsets_hybrid:B2",
        }
    ]
    complete = plan_post_selection_truth(
        selector_output_path=fixture["selector_path"],
        input_root=fixture["input_root"],
        heldout_manifest_path=fixture["heldout_path"],
        truth_roots=[fixture["root_a"], fixture["root_b"]],
        output_path=tmp_path / "complete.json",
    )
    assert complete["missing_coalition_count"] == 0
    assert complete["status"] == "COMPLETE_SET_TRANSFORMER_CONTROL_TRUTH"


def test_reduce_is_trajectory_equal_and_emits_paired_bootstrap(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "result.json"
    result = reduce_post_selection_truth(
        selector_output_path=fixture["selector_path"],
        input_root=fixture["input_root"],
        heldout_manifest_path=fixture["heldout_path"],
        truth_roots=[fixture["root_a"], fixture["root_b"]],
        output_path=output,
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )
    assert result["schema_version"] == RESULT_SCHEMA
    assert result["methods"]["summary_only"][
        "primary_trajectory_equal_B1_B4_macro"
    ] == 0.0
    assert result["methods"]["recent"][
        "primary_trajectory_equal_B1_B4_macro"
    ] == pytest.approx(0.5)
    assert result["methods"]["set_transformer"][
        "primary_trajectory_equal_B1_B4_macro"
    ] == pytest.approx(0.6)
    comparison = result["paired_deltas"]["set_transformer_minus_recent"]
    assert comparison["primary_macro_B1_B4"]["observed_delta"] == pytest.approx(
        0.1
    )
    assert comparison["long_plus_macro_B1_B4"]["trajectory_count"] == 64
    assert result["paired_deltas"]["recent_minus_summary_only"][
        "primary_macro_B1_B4"
    ]["observed_delta"] == pytest.approx(0.5)
    unsigned = dict(json.loads(output.read_text()))
    claimed = unsigned.pop("content_sha256")
    assert claimed == sha256_json(unsigned)


def test_tampered_terminal_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    terminal = next((fixture["root_a"] / "states").glob("*.json"))
    terminal.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="terminal file hash drifted"):
        plan_post_selection_truth(
            selector_output_path=fixture["selector_path"],
            input_root=fixture["input_root"],
            heldout_manifest_path=fixture["heldout_path"],
            truth_roots=[fixture["root_a"]],
            output_path=tmp_path / "schedule.json",
        )


def test_truth_roots_must_share_scientific_identity(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = fixture["root_b"] / "formal-truth-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("content_sha256")
    manifest["scientific_config_sha256"] = "a" * 64
    _write_compact(manifest_path, manifest)
    receipt_path = fixture["root_b"] / "formal-truth-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("content_sha256")
    receipt["formal_manifest_content_sha256"] = json.loads(
        manifest_path.read_text()
    )["content_sha256"]
    receipt["formal_manifest_file_sha256"] = sha256_file(manifest_path)
    _write_compact(receipt_path, receipt)
    with pytest.raises(ValueError, match="disagree on shared bindings"):
        plan_post_selection_truth(
            selector_output_path=fixture["selector_path"],
            input_root=fixture["input_root"],
            heldout_manifest_path=fixture["heldout_path"],
            truth_roots=[fixture["root_a"], fixture["root_b"]],
            output_path=tmp_path / "schedule.json",
        )


def test_merge_distance_uses_frozen_duplicate_tolerance() -> None:
    table = {(1,): 0.5}
    _merge_distance(table, (1,), 0.5 + 0.9e-6)
    assert table[(1,)] == 0.5

    with pytest.raises(ValueError, match="coalition distance"):
        _merge_distance(table, (1,), 0.5 + 1.1e-6)
