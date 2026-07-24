from __future__ import annotations

from scripts.plan_sealed_matrix_relaunch_v1 import WorkerSpec, assign_workers


def test_lpt_assignment_is_policy_isolated_and_complete() -> None:
    roster = {
        ("short", 0): {"max_steps": 10},
        ("medium", 0): {"max_steps": 20},
        ("long", 0): {"max_steps": 40},
    }
    missing = [
        {
            "arm": "summary_B0",
            "policy": policy,
            "task_index": 0,
            "task_type": task_type,
        }
        for policy in ("frozen", "history_gated")
        for task_type in ("short", "medium", "long")
    ]
    assignments = assign_workers(
        missing,
        workers=[
            WorkerSpec("frozen", 2, 28300, -1),
            WorkerSpec("frozen", 3, 28301, -1),
            WorkerSpec("history_gated", 4, 28302, -1),
            WorkerSpec("history_gated", 5, 28303, -1),
        ],
        roster=roster,
    )

    assert sum(len(worker["cells"]) for worker in assignments) == len(missing)
    for worker in assignments:
        assert {
            row["policy"] for row in worker["cells"]
        } == {worker["policy"]}
    for policy in ("frozen", "history_gated"):
        loads = sorted(
            worker["estimated_max_steps"]
            for worker in assignments
            if worker["policy"] == policy
        )
        assert loads == [30, 40]
