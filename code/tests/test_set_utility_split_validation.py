from __future__ import annotations

import pytest

from causalcache.set_utility_split_validation import (
    AUDIT_STATUS,
    PASSED_CHECKS,
    SPLIT_ROLES,
    SetUtilitySplitAssignment,
    validate_group_aware_split_assignments,
)


def _group(character: str) -> str:
    return character * 64


def _assignment(
    trajectory_id: str,
    source_id: str,
    group: str,
    role: str,
) -> SetUtilitySplitAssignment:
    return SetUtilitySplitAssignment(
        trajectory_id=trajectory_id,
        source_id=source_id,
        instruction_app_group_sha256=_group(group),
        role=role,  # type: ignore[arg-type]
    )


def _valid_assignments() -> list[SetUtilitySplitAssignment]:
    return [
        _assignment("train-1", "source-train-1", "a", "train"),
        _assignment("train-2", "source-train-2", "a", "train"),
        _assignment("tune-1", "source-tune-1", "b", "tune"),
        _assignment("eval-1", "source-eval-1", "c", "evaluation"),
        _assignment(
            "phase2-cal-1",
            "source-phase2-cal-1",
            "d",
            "phase2_calibration",
        ),
        _assignment(
            "phase2-eval-1",
            "source-phase2-eval-1",
            "e",
            "phase2_evaluation",
        ),
        _assignment("legacy-1", "formal58-1", "a", "legacy_train_only"),
    ]


def test_valid_split_emits_deterministic_overlap_summary_and_hash() -> None:
    assignments = _valid_assignments()
    first = validate_group_aware_split_assignments(
        assignments,
        legacy_train_only_source_ids={"formal58-1"},
        forbidden_source_ids={"old-dev5-1", "fresh16-1", "confirm20-1"},
        legacy_train_only_group_sha256s={_group("a")},
        forbidden_consumed_group_sha256s={_group("f")},
    )
    second = validate_group_aware_split_assignments(
        list(reversed(assignments)),
        legacy_train_only_source_ids={"formal58-1"},
        forbidden_source_ids={"confirm20-1", "fresh16-1", "old-dev5-1"},
        legacy_train_only_group_sha256s={_group("a")},
        forbidden_consumed_group_sha256s={_group("f")},
    )

    assert first.assignments == second.assignments
    assert first.audit.summary == second.audit.summary
    assert first.audit.summary_sha256 == second.audit.summary_sha256
    assert len(first.audit.summary_sha256) == 64
    assert first.audit.summary["audit_status"] == AUDIT_STATUS
    assert first.audit.summary["assignment_count"] == 7
    assert first.audit.summary["trajectory_count"] == 7
    assert first.audit.summary["source_count"] == 7
    assert first.audit.summary["instruction_app_group_count"] == 5
    assert first.audit.summary["role_trajectory_counts"] == {
        "train": 2,
        "tune": 1,
        "evaluation": 1,
        "phase2_calibration": 1,
        "phase2_evaluation": 1,
        "legacy_train_only": 1,
    }
    assert first.audit.summary["effective_partition_group_counts"] == {
        "train": 1,
        "tune": 1,
        "evaluation": 1,
        "phase2_calibration": 1,
        "phase2_evaluation": 1,
    }
    assert set(first.audit.summary["passed_checks"]) == set(PASSED_CHECKS)
    assert all(value == 0 for value in first.audit.summary["overlap_counts"].values())
    serialized = repr(first.audit.summary)
    for raw_identity in (
        "train-1",
        "source-train-1",
        "formal58-1",
        "old-dev5-1",
        _group("a"),
    ):
        assert raw_identity not in serialized


def test_same_trajectory_cannot_cross_roles() -> None:
    assignments = [
        _assignment("shared", "source-a", "a", "train"),
        _assignment("shared", "source-b", "b", "tune"),
    ]
    with pytest.raises(ValueError, match="trajectory_id cannot cross roles"):
        validate_group_aware_split_assignments(
            assignments,
            legacy_train_only_source_ids=set(),
            forbidden_source_ids=set(),
            legacy_train_only_group_sha256s=set(),
            forbidden_consumed_group_sha256s=set(),
        )


def test_one_source_cannot_alias_two_trajectories() -> None:
    assignments = [
        _assignment("trajectory-a", "shared-source", "a", "train"),
        _assignment("trajectory-b", "shared-source", "a", "train"),
    ]
    with pytest.raises(ValueError, match="source_id must identify exactly one"):
        validate_group_aware_split_assignments(
            assignments,
            legacy_train_only_source_ids=set(),
            forbidden_source_ids=set(),
            legacy_train_only_group_sha256s=set(),
            forbidden_consumed_group_sha256s=set(),
        )


@pytest.mark.parametrize(
    ("left_role", "right_role"),
    (
        ("train", "tune"),
        ("tune", "evaluation"),
        ("phase2_calibration", "phase2_evaluation"),
        ("legacy_train_only", "tune"),
        ("legacy_train_only", "phase2_evaluation"),
    ),
)
def test_instruction_app_group_cannot_cross_development_partitions(
    left_role: str,
    right_role: str,
) -> None:
    left_source = "formal58-1" if left_role == "legacy_train_only" else "source-a"
    assignments = [
        _assignment("trajectory-a", left_source, "a", left_role),
        _assignment("trajectory-b", "source-b", "a", right_role),
    ]
    legacy = {"formal58-1"} if left_role == "legacy_train_only" else set()
    with pytest.raises(ValueError, match="group cannot cross development partitions"):
        validate_group_aware_split_assignments(
            assignments,
            legacy_train_only_source_ids=legacy,
            forbidden_source_ids=set(),
            legacy_train_only_group_sha256s=(
                {_group("a")} if left_role == "legacy_train_only" else set()
            ),
            forbidden_consumed_group_sha256s=set(),
        )


@pytest.mark.parametrize("invalid_role", ("tune", "evaluation", "train"))
def test_declared_formal58_source_is_locked_to_legacy_train_only(
    invalid_role: str,
) -> None:
    with pytest.raises(ValueError, match="legacy formal-train sources"):
        validate_group_aware_split_assignments(
            [_assignment("legacy-1", "formal58-1", "a", invalid_role)],
            legacy_train_only_source_ids={"formal58-1"},
            forbidden_source_ids=set(),
            legacy_train_only_group_sha256s={_group("a")},
            forbidden_consumed_group_sha256s=set(),
        )


def test_legacy_role_cannot_hide_an_undeclared_source() -> None:
    with pytest.raises(ValueError, match="legacy formal-train sources"):
        validate_group_aware_split_assignments(
            [_assignment("legacy-1", "unknown-source", "a", "legacy_train_only")],
            legacy_train_only_source_ids={"formal58-1"},
            forbidden_source_ids=set(),
            legacy_train_only_group_sha256s={_group("a")},
            forbidden_consumed_group_sha256s=set(),
        )


@pytest.mark.parametrize("forbidden", ("old-dev5-1", "fresh16-1", "confirm20-1"))
def test_explicit_old_development_and_confirm_identities_fail_closed(
    forbidden: str,
) -> None:
    with pytest.raises(ValueError, match="explicitly forbidden source identity"):
        validate_group_aware_split_assignments(
            [_assignment("trajectory-1", forbidden, "a", "train")],
            legacy_train_only_source_ids=set(),
            forbidden_source_ids={"old-dev5-1", "fresh16-1", "confirm20-1"},
            legacy_train_only_group_sha256s=set(),
            forbidden_consumed_group_sha256s=set(),
        )


def test_consumed_group_firewall_rejects_new_source_alias() -> None:
    with pytest.raises(ValueError, match="forbidden consumed group"):
        validate_group_aware_split_assignments(
            [_assignment("new-trajectory", "new-source", "f", "train")],
            legacy_train_only_source_ids=set(),
            forbidden_source_ids=set(),
            legacy_train_only_group_sha256s=set(),
            forbidden_consumed_group_sha256s={_group("f")},
        )


def test_legacy_group_is_locked_to_effective_train_partition() -> None:
    with pytest.raises(ValueError, match="legacy train groups"):
        validate_group_aware_split_assignments(
            [_assignment("new-trajectory", "new-source", "a", "evaluation")],
            legacy_train_only_source_ids=set(),
            forbidden_source_ids=set(),
            legacy_train_only_group_sha256s={_group("a")},
            forbidden_consumed_group_sha256s=set(),
        )

    result = validate_group_aware_split_assignments(
        [_assignment("new-trajectory", "new-source", "a", "train")],
        legacy_train_only_source_ids=set(),
        forbidden_source_ids=set(),
        legacy_train_only_group_sha256s={_group("a")},
        forbidden_consumed_group_sha256s=set(),
    )
    assert result.audit.summary["overlap_counts"]["legacy_group_partition"] == 0


def test_source_firewall_sets_are_required_to_be_explicit_and_disjoint() -> None:
    assignment = _assignment("trajectory-1", "source-1", "a", "train")
    with pytest.raises(TypeError, match="explicit set"):
        validate_group_aware_split_assignments(
            [assignment],
            legacy_train_only_source_ids=[],  # type: ignore[arg-type]
            forbidden_source_ids=set(),
            legacy_train_only_group_sha256s=set(),
            forbidden_consumed_group_sha256s=set(),
        )

    with pytest.raises(TypeError, match="explicit set of group SHA256"):
        validate_group_aware_split_assignments(
            [assignment],
            legacy_train_only_source_ids=set(),
            forbidden_source_ids=set(),
            legacy_train_only_group_sha256s=[],  # type: ignore[arg-type]
            forbidden_consumed_group_sha256s=set(),
        )
    with pytest.raises(ValueError, match="group sets must be disjoint"):
        validate_group_aware_split_assignments(
            [assignment],
            legacy_train_only_source_ids=set(),
            forbidden_source_ids=set(),
            legacy_train_only_group_sha256s={_group("f")},
            forbidden_consumed_group_sha256s={_group("f")},
        )
    with pytest.raises(ValueError, match="must be disjoint"):
        validate_group_aware_split_assignments(
            [assignment],
            legacy_train_only_source_ids={"shared"},
            forbidden_source_ids={"shared"},
            legacy_train_only_group_sha256s=set(),
            forbidden_consumed_group_sha256s=set(),
        )


def test_empty_duplicate_and_malformed_assignments_fail_closed() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_group_aware_split_assignments(
            [],
            legacy_train_only_source_ids=set(),
            forbidden_source_ids=set(),
            legacy_train_only_group_sha256s=set(),
            forbidden_consumed_group_sha256s=set(),
        )

    assignment = _assignment("trajectory-1", "source-1", "a", "train")
    with pytest.raises(ValueError, match="duplicate trajectory assignment"):
        validate_group_aware_split_assignments(
            [assignment, assignment],
            legacy_train_only_source_ids=set(),
            forbidden_source_ids=set(),
            legacy_train_only_group_sha256s=set(),
            forbidden_consumed_group_sha256s=set(),
        )

    with pytest.raises(ValueError, match="lowercase SHA256"):
        SetUtilitySplitAssignment(
            trajectory_id="trajectory-1",
            source_id="source-1",
            instruction_app_group_sha256="A" * 64,
            role="train",
        )
    with pytest.raises(ValueError, match="role must be one of"):
        _assignment("trajectory-1", "source-1", "a", "validation")
    assert set(SPLIT_ROLES) == {
        "train",
        "tune",
        "evaluation",
        "phase2_calibration",
        "phase2_evaluation",
        "legacy_train_only",
    }
