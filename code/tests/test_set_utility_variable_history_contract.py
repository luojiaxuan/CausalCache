from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/configs/causalcache_set_utility_variable_history_v1.json"


def _contract() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_full_history_candidate_contract_has_no_fixed_n_or_recent_truncation() -> None:
    contract = _contract()
    state = contract["state"]
    assert state["candidate_rule"] == (
        "all_event_ids_1_through_decision_step_minus_1"
    )
    assert state["candidate_count_is_data_property"] is True
    assert state["candidate_count_input_to_model"] is False
    assert state["include_current_equivalent_event"] is True
    assert state["recent_candidate_truncation_allowed"] is False
    assert state["history_bins"] == {
        "short": [5, 8],
        "medium": [9, 16],
        "long": [17, 32],
        "very_long": [33, 45],
    }


def test_reference_cannot_restore_context_fit_by_dropping_old_events() -> None:
    reference = _contract()["reference"]
    assert reference["target_effective_visual_tokens_per_image"] == 512
    assert reference["full_reference_coalition"] == "C_t"
    assert reference["full_reference_distance"] == 0.0
    assert reference["full_reference_must_fit_before_rollout"] is True
    assert reference["drop_oldest_candidate_allowed"] is False
    assert reference["context_infeasibility_action"] == (
        "BLOCK_ROLLOUT_AND_CHANGE_REFERENCE_PROFILE"
    )


def test_broad_sampler_and_evaluation_tracks_are_frozen() -> None:
    contract = _contract()
    broad = contract["broad_training_labels"]
    assert broad["target_unique_subsets_per_state"] == 40
    assert broad["small_history_exact_maximum_n"] == 5
    assert sum(
        component["count"] for component in broad["components"].values()
    ) == 40
    assert broad["similarity"]["label_blind"] is True
    evaluation = contract["evaluation"]
    assert evaluation["exact_oracle_track"]["state_count"] == 320
    assert evaluation["large_history_track"]["state_count"] == 720
    assert evaluation["budgets"] == [1, 2, 3, 4]
    assert evaluation["budget_input_to_predictor"] is False
    assert evaluation["candidate_count_input_to_predictor"] is False


def test_resumption_is_finer_than_one_state() -> None:
    resume = _contract()["resumability"]
    assert resume["atomic_unit"] == "coalition_microbatch"
    assert resume["partial_state_progress_persisted"] is True
    assert resume["completed_coalitions_skipped_on_resume"] is True
    assert resume["scientific_config_must_match"] is True
    assert resume["dynamic_logical_shards"] == 256
