from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from causalcache.sealed_matrix_v1 import (
    ACTION_INCOMPATIBLE_TASKS,
    ARMS,
    FULL_LAYER_SHA256,
    HISTORY_GATED_SHA256,
    POLICIES,
    Attempt,
    aggregate_matrix,
    load_roster,
    normalize_attempt,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ROSTER_PATH = REPO_ROOT / "data/manifests/androidworld_full_suite_plan_v1.json"


def _policy_payload(policy: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "target_effective_visual_tokens_per_image": 2560,
    }
    if policy == "full_layer":
        payload["lora_checkpoint_sha256"] = FULL_LAYER_SHA256
    elif policy == "history_gated":
        payload.update(
            {
                "adapter_type": "history_gated_kv",
                "lora_checkpoint_sha256": HISTORY_GATED_SHA256,
            }
        )
    return payload


def _row(
    roster_row: dict[str, object],
    *,
    policy: str,
    arm: str,
    success: float = 0.0,
    infrastructure_failure: bool = False,
    model_step_count: int = 1,
) -> dict[str, object]:
    classification = (
        "infrastructure_failure"
        if infrastructure_failure
        else ("official_success" if success else "terminal_failure")
    )
    return {
        "allow_sealed_split": True,
        "arm": arm,
        "failure_classification": classification,
        "infrastructure_failure": infrastructure_failure,
        "instance": deepcopy(roster_row),
        "model_step_count": model_step_count,
        "official_terminal_success": success,
        "parse_retries_allowed": 1,
        "partition": "full",
        "policy": _policy_payload(policy),
        "required_audits_complete": not infrastructure_failure,
        "score_after": 1.0 if success else (None if infrastructure_failure else 0.0),
        "shared_early_decisions": 2,
        "started_at": "2026-07-24T00:00:00+00:00",
        "task_index": roster_row["task_index"],
        "task_type": roster_row["task_type"],
        "termination_reason": (
            "infrastructure_exception"
            if infrastructure_failure
            else ("policy_terminated" if success else "step_budget_exhausted")
        ),
    }


def _attempt(
    roster_row: dict[str, object],
    *,
    policy: str,
    arm: str,
    success: float = 0.0,
    infrastructure_failure: bool = False,
    model_step_count: int = 1,
    suffix: str = "",
) -> Attempt:
    roster = {
        (str(roster_row["task_type"]), int(roster_row["task_index"])): roster_row
    }
    return normalize_attempt(
        _row(
            roster_row,
            policy=policy,
            arm=arm,
            success=success,
            infrastructure_failure=infrastructure_failure,
            model_step_count=model_step_count,
        ),
        source="fixture",
        path=Path(f"/fixture/{policy}-{arm}{suffix}.json"),
        file_sha256=f"{policy}-{arm}{suffix}",
        roster=roster,
    )


def test_roster_freezes_two_instances_and_nine_action_incompatible_tasks() -> None:
    roster = load_roster(ROSTER_PATH)
    assert len(roster) == 232
    assert len({task_type for task_type, _ in roster}) == 116
    assert len(ACTION_INCOMPATIBLE_TASKS) == 9
    assert len([name for name in ACTION_INCOMPATIBLE_TASKS if "Verify" in name]) == 7


def test_void_attempt_does_not_override_formal_attempt() -> None:
    full_roster = load_roster(ROSTER_PATH)
    roster_row = next(iter(full_roster.values()))
    attempts = [
        _attempt(
            roster_row,
            policy="frozen",
            arm="summary_B0",
            infrastructure_failure=True,
            model_step_count=0,
            suffix="-void",
        ),
        _attempt(
            roster_row,
            policy="frozen",
            arm="summary_B0",
            success=1.0,
            suffix="-formal",
        ),
        _attempt(
            roster_row,
            policy="frozen",
            arm="recent_B4",
            infrastructure_failure=True,
            model_step_count=3,
            suffix="-started-infra",
        ),
    ]
    report = aggregate_matrix(
        attempts,
        roster=full_roster,
        bootstrap_resamples=10,
    )
    assert report["audit"]["formal_cell_count"] == 1
    assert report["audit"]["started_infrastructure_attempt_count"] == 1
    assert report["audit"]["void_attempt_count"] == 1
    assert report["status"] == "INCOMPLETE_SEALED_ZERO_SHOT_MATRIX_V1"


def test_complete_synthetic_matrix_uses_template_macro_pairs() -> None:
    full_roster = load_roster(ROSTER_PATH)
    attempts = []
    for roster_row in full_roster.values():
        task_index = int(roster_row["task_index"])
        for policy in POLICIES:
            for arm in ARMS:
                success = float(
                    policy == "history_gated"
                    and arm == "recent_B4"
                    and task_index == 0
                )
                attempts.append(
                    _attempt(
                        roster_row,
                        policy=policy,
                        arm=arm,
                        success=success,
                    )
                )
    report = aggregate_matrix(
        attempts,
        roster=full_roster,
        bootstrap_resamples=100,
    )
    assert report["status"] == "COMPLETE_SEALED_ZERO_SHOT_MATRIX_V1"
    headline = report["views"]["headline_all_116"]
    action = report["views"]["action_compatible_107"]
    assert headline["template_count"] == 116
    assert headline["instance_count"] == 232
    assert action["template_count"] == 107
    assert action["instance_count"] == 214
    assert (
        headline["cell_metrics"]["history_gated"]["recent_B4"][
            "template_macro_success"
        ]
        == pytest.approx(0.5)
    )
    comparison = headline["paired_comparisons"]["fixed_arm_policies"][
        "recent_B4:history_gated-minus-frozen"
    ]
    assert comparison["paired_template_macro_difference"] == pytest.approx(0.5)
    assert comparison["bootstrap_95_ci"] == pytest.approx([0.5, 0.5])


def test_duplicate_formal_attempts_fail_closed() -> None:
    full_roster = load_roster(ROSTER_PATH)
    roster_row = next(iter(full_roster.values()))
    attempts = [
        _attempt(
            roster_row,
            policy="frozen",
            arm="summary_B0",
            suffix="-a",
        ),
        _attempt(
            roster_row,
            policy="frozen",
            arm="summary_B0",
            suffix="-b",
        ),
    ]
    report = aggregate_matrix(
        attempts,
        roster=full_roster,
        bootstrap_resamples=10,
    )
    assert report["audit"]["unauthorized_noninfra_duplicate_count"] == 1
    assert report["audit"]["formal_cell_count"] == 0


def test_policy_identity_and_success_formula_fail_closed() -> None:
    full_roster = load_roster(ROSTER_PATH)
    roster_row = next(iter(full_roster.values()))
    roster = {
        (str(roster_row["task_type"]), int(roster_row["task_index"])): roster_row
    }
    bad_policy = _row(
        roster_row,
        policy="frozen",
        arm="summary_B0",
    )
    bad_policy["policy"] = {
        "adapter_type": "history_gated_kv",
        "lora_checkpoint_sha256": FULL_LAYER_SHA256,
        "target_effective_visual_tokens_per_image": 2560,
    }
    with pytest.raises(ValueError, match="unregistered matrix policy"):
        normalize_attempt(
            bad_policy,
            source="fixture",
            path=Path("/fixture/bad-policy.json"),
            file_sha256="bad-policy",
            roster=roster,
        )

    bad_success = _row(
        roster_row,
        policy="frozen",
        arm="summary_B0",
        success=1.0,
    )
    bad_success["termination_reason"] = "step_budget_exhausted"
    with pytest.raises(ValueError, match="score_after"):
        normalize_attempt(
            bad_success,
            source="fixture",
            path=Path("/fixture/bad-success.json"),
            file_sha256="bad-success",
            roster=roster,
        )


def test_policy_agnostic_nine_cell_void_enables_only_hard_delete_view() -> None:
    full_roster = load_roster(ROSTER_PATH)
    deleted_key = next(iter(full_roster))
    attempts = []
    for roster_key, roster_row in full_roster.items():
        for policy in POLICIES:
            for arm in ARMS:
                if roster_key == deleted_key:
                    attempts.append(
                        _attempt(
                            roster_row,
                            policy=policy,
                            arm=arm,
                            infrastructure_failure=True,
                            model_step_count=0,
                        )
                    )
                else:
                    attempts.append(
                        _attempt(
                            roster_row,
                            policy=policy,
                            arm=arm,
                        )
                    )
    report = aggregate_matrix(
        attempts,
        roster=full_roster,
        bootstrap_resamples=10,
    )
    assert report["status"] == "INCOMPLETE_SEALED_ZERO_SHOT_MATRIX_V1"
    assert not report["views"]["headline_all_116"]["ready"]
    assert report["views"]["hard_delete_env_init_only"]["ready"]
    assert report["views"]["hard_delete_env_init_only"]["template_count"] == 116
    assert report["views"]["hard_delete_env_init_only"]["instance_count"] == 231
