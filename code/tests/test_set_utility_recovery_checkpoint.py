from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from causalcache.set_utility_recovery_checkpoint import (
    RecoveryCheckpointManager,
    trajectory_equal_budget_recovery,
)


def _records(
    values: dict[tuple[str, str], float],
) -> list[dict[str, object]]:
    return [
        {
            "budget": budget,
            "normalized_recovery": value,
            "state_id": state,
            "trajectory_id": trajectory,
        }
        for (trajectory, state), value in values.items()
        for budget in (1, 2, 3, 4)
    ]


def test_recovery_is_trajectory_equal_not_state_equal() -> None:
    summary = trajectory_equal_budget_recovery(
        _records({("a", "a1"): 0.0, ("a", "a2"): 1.0, ("b", "b1"): 1.0})
    )
    assert summary.macro_b1_b4 == pytest.approx(0.75)
    assert summary.budget_means == pytest.approx({1: 0.75, 2: 0.75, 3: 0.75, 4: 0.75})
    assert summary.state_count == 3
    assert summary.trajectory_count == 2


def test_recovery_requires_every_state_budget() -> None:
    records = _records({("a", "a1"): 0.5})
    records.pop()
    with pytest.raises(ValueError, match="omit state/budget"):
        trajectory_equal_budget_recovery(records)


def test_manager_saves_every_epoch_and_selects_macro_recovery(tmp_path: Path) -> None:
    def saver(model: bytes, destination: Path) -> None:
        destination.write_bytes(model)

    manager = RecoveryCheckpointManager(
        tmp_path / "run",
        identity={"config_sha256": "a" * 64},
        selection_split="train_trajectory_holdout",
        saver=saver,
    )
    for epoch, value in ((1, 0.4), (2, 0.6), (3, 0.6)):
        manager.save_epoch(f"epoch-{epoch}".encode(), epoch=epoch)
        manager.record_recovery(
            epoch=epoch,
            records=_records({("a", "a1"): value, ("b", "b1"): value}),
        )
    manifest = manager.manifest
    assert [row["epoch"] for row in manifest["epochs"]] == [1, 2, 3]
    assert manifest["best"]["epoch"] == 2
    for row in manifest["epochs"]:
        checkpoint = tmp_path / "run" / row["checkpoint"]["path"]
        assert checkpoint.is_file()
        assert row["checkpoint"]["sha256"] == hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()
    on_disk = json.loads(
        (tmp_path / "run" / "recovery-checkpoints.json").read_text(encoding="utf-8")
    )
    assert on_disk == manifest


def test_after_epoch_preserves_checkpoint_when_evaluator_fails(tmp_path: Path) -> None:
    def saver(model: bytes, destination: Path) -> None:
        destination.write_bytes(model)

    manager = RecoveryCheckpointManager(
        tmp_path / "run",
        identity={"run": "failure-safe"},
        selection_split="train_trajectory_holdout",
        saver=saver,
    )

    def fail(_: bytes):
        raise RuntimeError("evaluation failed")

    with pytest.raises(RuntimeError, match="evaluation failed"):
        manager.after_epoch(b"weights", epoch=1, evaluator=fail)
    manifest = manager.manifest
    assert manifest["epochs"][0]["status"] == "AWAITING_TRUE_RECOVERY"
    assert (tmp_path / "run" / "checkpoints" / "epoch-0001.safetensors").is_file()


def test_manager_rejects_final_evaluation_for_checkpoint_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot select"):
        RecoveryCheckpointManager(
            tmp_path / "run",
            identity={"run": "leak"},
            selection_split="evaluation",
            saver=lambda _model, _path: None,
        )


def test_manager_rejects_manifest_content_drift(tmp_path: Path) -> None:
    root = tmp_path / "run"
    RecoveryCheckpointManager(
        root,
        identity={"run": "bound"},
        selection_split="train_trajectory_holdout",
        saver=lambda _model, path: path.write_bytes(b"weights"),
    )
    manifest_path = root / "recovery-checkpoints.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection_metric"] = "training_loss"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity drifted"):
        RecoveryCheckpointManager(
            root,
            identity={"run": "bound"},
            selection_split="train_trajectory_holdout",
            saver=lambda _model, path: path.write_bytes(b"weights"),
        )
