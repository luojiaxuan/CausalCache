from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from causalcache.set_utility_train_heldout_contract import (
    COMPLETE_TRUTH_STATUS,
    INCOMPLETE_TRUTH_STATUS,
    build_train_input_state_inventory,
    materialize_train_heldout_contract,
    reduce_epoch_truth,
    select_checkpoint_from_epoch_truth,
    sha256_json,
)
from causalcache.set_utility_variable_history import states_from_assignments
from scripts.materialize_set_utility_train_heldout_contract import main


def _assignments() -> list[dict[str, object]]:
    decision_counts = (40, 40, 40, 24, 24, 16, 16, 8)
    result = [
        {
            "decision_count": decision_count,
            "role": "train",
            "trajectory_id": f"train-{index:02d}",
        }
        for index, decision_count in enumerate(decision_counts)
    ]
    result.extend(
        [
            {
                "decision_count": 40,
                "role": "tune",
                "trajectory_id": "forbidden-tune",
            },
            {
                "decision_count": 40,
                "role": "evaluation",
                "trajectory_id": "forbidden-evaluation",
            },
        ]
    )
    return result


def _input_inventory() -> dict[str, object]:
    states = states_from_assignments(_assignments())
    dropped = {
        "train-00:decision:006",
        "train-07:decision:009",
    }
    records = [
        {
            "candidate_event_step_ids": list(state.candidate_event_ids),
            "role": state.role,
            "state_id": state.state_id,
            "trajectory_id": state.trajectory_id,
        }
        for state in states
        if state.state_id not in dropped
    ]
    role_counts: dict[str, int] = {}
    for row in records:
        role = row["role"]
        role_counts[role] = role_counts.get(role, 0) + 1
    manifest = {
        "content_sha256": "d" * 64,
        "evaluation_labels_included": False,
        "role_counts": role_counts,
        "state_count": len(records),
        "states_jsonl": "states.jsonl",
        "states_sha256": "e" * 64,
        "status": "COMPLETED_SET_UTILITY_CONTEXTUAL_INPUT_SNAPSHOT",
    }
    return build_train_input_state_inventory(
        manifest,
        records,
        input_manifest_sha256="f" * 64,
    )


def _config(
    source_sha256: str,
    inventory_content_sha256: str,
) -> dict[str, object]:
    return {
        "checkpoint_selection": {
            "budgets": [1, 2, 3, 4],
            "higher_is_better": True,
            "minimum_delta": 0.0001,
            "patience": 2,
            "require_complete_truth_each_epoch": True,
        },
        "firewall": {
            "all_heldout_trajectory_states_excluded_from_optimizer": True,
            "evaluation_access": False,
            "tune_access": False,
        },
        "source": {
            "allowed_role": "train",
            "assignment_manifest": "data/assignments.json",
            "assignment_manifest_sha256": source_sha256,
            "expected_assignment_derived_train_state_count": 176,
            "expected_assignment_only_exclusion_count": 2,
            "expected_train_state_count": 174,
            "expected_train_trajectory_count": 8,
            "train_state_inventory": "data/inventory.json",
            "train_state_inventory_content_sha256": inventory_content_sha256,
            "training_input_content_sha256": "d" * 64,
            "training_input_manifest_sha256": "f" * 64,
            "training_states_sha256": "e" * 64,
        },
        "split": {
            "checkpoint_state_targets_by_history_bin": {
                "long": 2,
                "medium": 2,
                "short": 2,
                "very_long": 2,
            },
            "heldout_trajectory_count": 3,
            "selection_salt": "unit-test-heldout-salt",
        },
    }


def _contract() -> dict[str, object]:
    source_sha = "a" * 64
    inventory = _input_inventory()
    return materialize_train_heldout_contract(
        _assignments(),
        train_state_inventory=inventory,
        config=_config(source_sha, inventory["content_sha256"]),
        assignment_manifest_sha256=source_sha,
        config_sha256="b" * 64,
    )


def _complete_truth(
    contract: dict[str, object],
    *,
    epoch: int,
    checkpoint_sha256: str,
    offset: float = 0.0,
) -> dict[str, object]:
    records = [
        {
            "budget": budget,
            "normalized_recovery": offset + budget / 10.0,
            "state_id": row["state_id"],
        }
        for row in contract["checkpoint_states"]
        for budget in (1, 2, 3, 4)
    ]
    return reduce_epoch_truth(
        records,
        checkpoint_states=contract["checkpoint_states"],
        epoch=epoch,
        checkpoint_sha256=checkpoint_sha256,
        contract_content_sha256=contract["content_sha256"],
    )


def test_train_input_inventory_exports_no_tune_or_evaluation_identity() -> None:
    inventory = _input_inventory()
    assert inventory["census"]["source_role_counts"] == {
        "train": 174,
        "tune": 36,
        "evaluation": 36,
    }
    assert inventory["census"]["train_state_count"] == 174
    assert all(
        row["state_id"].startswith("train-") for row in inventory["train_states"]
    )
    exported = json.dumps(inventory, sort_keys=True)
    assert "forbidden-tune" not in exported
    assert "forbidden-evaluation" not in exported


def test_contract_is_deterministic_balanced_and_trajectory_disjoint() -> None:
    first = _contract()
    second = _contract()

    assert first == second
    assert first["census"]["checkpoint_history_bin_counts"] == {
        "short": 2,
        "medium": 2,
        "long": 2,
        "very_long": 2,
    }
    assert first["census"]["source_train_state_count"] == 174
    assert first["census"]["assignment_derived_train_state_count"] == 176
    assert first["census"]["assignment_only_exclusion_count"] == 2
    assert first["source"]["assignment_only_exclusions"] == [
        {
            "reason": "ABSENT_FROM_FROZEN_STRUCTURED_TRAINING_INPUT",
            "state_id": "train-00:decision:006",
            "trajectory_id": "train-00",
        },
        {
            "reason": "ABSENT_FROM_FROZEN_STRUCTURED_TRAINING_INPUT",
            "state_id": "train-07:decision:009",
            "trajectory_id": "train-07",
        },
    ]
    assert first["source"]["assignment_only_exclusions_sha256"] == sha256_json(
        first["source"]["assignment_only_exclusions"]
    )
    assert len(first["heldout_trajectory_ids"]) == 3
    assert len(first["optimizer_trajectory_ids"]) == 5
    assert set(first["heldout_trajectory_ids"]).isdisjoint(
        first["optimizer_trajectory_ids"]
    )
    assert set(first["heldout_all_state_ids"]).isdisjoint(
        first["optimizer_state_ids"]
    )
    all_identifiers = json.dumps(first, sort_keys=True)
    assert "forbidden-tune" not in all_identifiers
    assert "forbidden-evaluation" not in all_identifiers
    for name in (
        "checkpoint_state_ids",
        "heldout_all_state_ids",
        "heldout_trajectory_ids",
        "optimizer_state_ids",
        "optimizer_trajectory_ids",
    ):
        assert first["allowlist_sha256s"][f"{name}_sha256"] == sha256_json(
            first[name]
        )


def test_contract_rejects_source_drift_and_inventory_drift() -> None:
    inventory = _input_inventory()
    config = _config("a" * 64, inventory["content_sha256"])
    with pytest.raises(ValueError, match="drifted"):
        materialize_train_heldout_contract(
            _assignments(),
            train_state_inventory=inventory,
            config=config,
            assignment_manifest_sha256="c" * 64,
            config_sha256="b" * 64,
        )
    config["source"]["expected_train_trajectory_count"] = 9
    with pytest.raises(ValueError, match="inventory"):
        materialize_train_heldout_contract(
            _assignments(),
            train_state_inventory=inventory,
            config=config,
            assignment_manifest_sha256="a" * 64,
            config_sha256="b" * 64,
        )


def test_epoch_truth_is_fail_closed_until_every_state_budget_pair_exists() -> None:
    contract = _contract()
    records = [
        {
            "budget": budget,
            "normalized_recovery": budget / 10.0,
            "state_id": row["state_id"],
        }
        for row in contract["checkpoint_states"]
        for budget in (1, 2, 3, 4)
    ]
    incomplete = reduce_epoch_truth(
        records[:-1],
        checkpoint_states=contract["checkpoint_states"],
        epoch=1,
        checkpoint_sha256="c" * 64,
        contract_content_sha256=contract["content_sha256"],
    )
    assert incomplete["status"] == INCOMPLETE_TRUTH_STATUS
    assert incomplete["missing_pair_count"] == 1
    decision = select_checkpoint_from_epoch_truth(
        [incomplete], patience=2, minimum_delta=0.0001
    )
    assert decision["decision_ready"] is False
    assert decision["observed_complete_epoch_count"] == 0
    assert decision["selected_epoch"] is None

    complete = reduce_epoch_truth(
        records,
        checkpoint_states=contract["checkpoint_states"],
        epoch=1,
        checkpoint_sha256="c" * 64,
        contract_content_sha256=contract["content_sha256"],
    )
    assert complete["status"] == COMPLETE_TRUTH_STATUS
    assert complete["truth_complete"] is True
    assert complete["observed_row_count"] == 8 * 4
    assert complete[
        "primary_trajectory_equal_true_B1_B4_recovery_macro"
    ] == pytest.approx(0.25)
    assert complete[
        "long_plus_very_long_trajectory_equal_true_B1_B4_recovery_macro"
    ] == pytest.approx(0.25)


def test_checkpoint_selection_uses_primary_then_long_tie_break_and_patience() -> None:
    contract = _contract()
    epochs = [
        _complete_truth(contract, epoch=1, checkpoint_sha256="1" * 64),
        _complete_truth(
            contract,
            epoch=2,
            checkpoint_sha256="2" * 64,
            offset=0.00005,
        ),
        _complete_truth(
            contract,
            epoch=3,
            checkpoint_sha256="3" * 64,
            offset=0.01,
        ),
        _complete_truth(
            contract,
            epoch=4,
            checkpoint_sha256="4" * 64,
            offset=0.009,
        ),
        _complete_truth(
            contract,
            epoch=5,
            checkpoint_sha256="5" * 64,
            offset=0.008,
        ),
    ]
    decision = select_checkpoint_from_epoch_truth(
        epochs, patience=2, minimum_delta=0.0001
    )
    assert decision["decision_ready"] is True
    assert decision["selected_epoch"] == 3
    assert decision["selected_checkpoint_sha256"] == "3" * 64
    assert decision["stopped_early"] is True
    assert decision["stop_after_epoch"] == 5

    same_primary = [dict(epochs[0]), dict(epochs[1])]
    primary_key = "primary_trajectory_equal_true_B1_B4_recovery_macro"
    long_key = "long_plus_very_long_trajectory_equal_true_B1_B4_recovery_macro"
    same_primary[1][primary_key] = same_primary[0][primary_key]
    same_primary[1][long_key] = same_primary[0][long_key] + 0.01
    tie_break = select_checkpoint_from_epoch_truth(
        same_primary, patience=2, minimum_delta=0.0001
    )
    assert tie_break["selected_epoch"] == 2


def test_materialize_cli_writes_one_bound_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repo"
    source_path = repository_root / "data" / "assignments.json"
    source_path.parent.mkdir(parents=True)
    source_payload = {"assignments": _assignments()}
    source_path.write_text(
        json.dumps(source_payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    inventory = _input_inventory()
    inventory_path = repository_root / "data" / "inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, sort_keys=True) + "\n", encoding="utf-8"
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            _config(source_sha, inventory["content_sha256"]), sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "contract.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_set_utility_train_heldout_contract.py",
            "--repository-root",
            str(repository_root),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    value = json.loads(output_path.read_text(encoding="utf-8"))
    assert value["source"]["assignment_manifest_sha256"] == source_sha
    assert value["census"]["heldout_trajectory_count"] == 3
    with pytest.raises(FileExistsError):
        main()
