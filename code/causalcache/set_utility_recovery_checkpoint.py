"""Epoch checkpointing selected by true trajectory-equal B1--B4 recovery."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


BUDGETS = (1, 2, 3, 4)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(value) + b"\n"
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BudgetRecoverySummary:
    """Trajectory-equal recovery used as the only checkpoint objective."""

    budget_means: Mapping[int, float]
    macro_b1_b4: float
    state_count: int
    trajectory_count: int

    def __post_init__(self) -> None:
        if tuple(sorted(self.budget_means)) != BUDGETS:
            raise ValueError("recovery summary must contain exactly B1--B4")
        values = (*self.budget_means.values(), self.macro_b1_b4)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("recovery summary values must be finite")
        if type(self.state_count) is not int or self.state_count <= 0:
            raise ValueError("state_count must be a positive integer")
        if type(self.trajectory_count) is not int or self.trajectory_count <= 0:
            raise ValueError("trajectory_count must be a positive integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget_means": {
                str(budget): float(self.budget_means[budget]) for budget in BUDGETS
            },
            "macro_B1_B4_trajectory_equal_normalized_recovery": float(
                self.macro_b1_b4
            ),
            "state_count": self.state_count,
            "trajectory_count": self.trajectory_count,
        }


def trajectory_equal_budget_recovery(
    records: Iterable[Mapping[str, Any]],
) -> BudgetRecoverySummary:
    """Aggregate true selected-subset recovery without treating states as IID."""
    rows = tuple(records)
    if not rows:
        raise ValueError("checkpoint recovery records are empty")
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    state_keys = set()
    for index, row in enumerate(rows):
        trajectory_id = row.get("trajectory_id")
        state_id = row.get("state_id")
        budget = row.get("budget")
        recovery = row.get("normalized_recovery")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError(f"recovery row {index} has no trajectory identity")
        if not isinstance(state_id, str) or not state_id:
            raise ValueError(f"recovery row {index} has no state identity")
        if budget not in BUDGETS:
            raise ValueError(f"recovery row {index} has an invalid budget")
        if not isinstance(recovery, (int, float)) or isinstance(recovery, bool):
            raise TypeError(f"recovery row {index} must contain a numeric recovery")
        value = float(recovery)
        if not math.isfinite(value):
            raise ValueError(f"recovery row {index} is non-finite")
        key = (trajectory_id, state_id, budget)
        if key in state_keys:
            raise ValueError(f"duplicate recovery state/budget row: {key}")
        state_keys.add(key)
        grouped[(trajectory_id, budget)].append(value)
    states = sorted({(trajectory, state) for trajectory, state, _ in state_keys})
    missing_state_budgets = [
        (trajectory, state, budget)
        for trajectory, state in states
        for budget in BUDGETS
        if (trajectory, state, budget) not in state_keys
    ]
    if missing_state_budgets:
        raise ValueError(
            "recovery records omit state/budget cells: "
            f"{missing_state_budgets}"
        )
    trajectories = sorted({trajectory for trajectory, _ in grouped})
    missing = [
        (trajectory, budget)
        for trajectory in trajectories
        for budget in BUDGETS
        if (trajectory, budget) not in grouped
    ]
    if missing:
        raise ValueError(f"recovery records omit trajectory/budget cells: {missing}")
    budget_means = {}
    for budget in BUDGETS:
        trajectory_values = [
            sum(grouped[(trajectory, budget)]) / len(grouped[(trajectory, budget)])
            for trajectory in trajectories
        ]
        budget_means[budget] = sum(trajectory_values) / len(trajectory_values)
    return BudgetRecoverySummary(
        budget_means=budget_means,
        macro_b1_b4=sum(budget_means.values()) / len(BUDGETS),
        state_count=len(states),
        trajectory_count=len(trajectories),
    )


def atomic_safetensors_epoch_saver(model: Any, destination: Path) -> None:
    """Persist one immutable CPU checkpoint without optimizer-only tensors."""
    try:
        from safetensors.torch import save_file
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("epoch checkpointing requires safetensors") from error
    state = {
        key: value.detach().to(device="cpu").contiguous()
        for key, value in model.state_dict().items()
    }
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    save_file(state, str(temporary))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


class RecoveryCheckpointManager:
    """Save every epoch and select only by true B1--B4 recovery."""

    def __init__(
        self,
        output_root: Path,
        *,
        identity: Mapping[str, Any],
        selection_split: str,
        saver: Callable[[Any, Path], None] = atomic_safetensors_epoch_saver,
    ) -> None:
        if not isinstance(selection_split, str) or not selection_split:
            raise ValueError("checkpoint selection split must be named")
        if selection_split in {"evaluation", "test", "sealed_evaluation"}:
            raise ValueError(
                "final evaluation/test labels cannot select training checkpoints"
            )
        self.output_root = Path(output_root)
        self.checkpoint_root = self.output_root / "checkpoints"
        self.manifest_path = self.output_root / "recovery-checkpoints.json"
        self.identity = json.loads(json.dumps(identity, sort_keys=True))
        self.selection_split = selection_split
        self.saver = saver
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            unsigned = dict(manifest)
            claimed_sha256 = unsigned.pop("content_sha256", None)
            if (
                manifest.get("identity") != self.identity
                or manifest.get("selection_split") != selection_split
                or manifest.get("schema_version")
                != "causalcache.recovery_epoch_checkpoints.v1"
                or claimed_sha256
                != hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
            ):
                raise ValueError("recovery checkpoint resume identity drifted")
            self._manifest = manifest
        else:
            if self.output_root.exists() and any(self.output_root.iterdir()):
                raise FileExistsError(
                    "checkpoint output exists without a recovery manifest"
                )
            self.checkpoint_root.mkdir(parents=True, exist_ok=True)
            self._manifest = {
                "best": None,
                "epochs": [],
                "identity": self.identity,
                "schema_version": "causalcache.recovery_epoch_checkpoints.v1",
                "selection_metric": (
                    "macro_B1_B4_trajectory_equal_normalized_recovery"
                ),
                "selection_split": selection_split,
            }
            self._publish()

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._manifest))

    def _publish(self) -> None:
        unsigned = dict(self._manifest)
        unsigned.pop("content_sha256", None)
        self._manifest["content_sha256"] = hashlib.sha256(
            _canonical_json_bytes(unsigned)
        ).hexdigest()
        _atomic_json(self.manifest_path, self._manifest)

    def save_epoch(self, model: Any, *, epoch: int) -> dict[str, Any]:
        """Save before evaluation so an evaluator failure cannot lose the epoch."""
        if type(epoch) is not int or epoch <= 0:
            raise ValueError("epoch must be a positive integer")
        if any(int(row["epoch"]) == epoch for row in self._manifest["epochs"]):
            raise ValueError(f"epoch {epoch} is already registered")
        destination = self.checkpoint_root / f"epoch-{epoch:04d}.safetensors"
        if destination.exists():
            raise FileExistsError(destination)
        self.saver(model, destination)
        if not destination.is_file():
            raise RuntimeError("epoch saver did not create the checkpoint")
        checkpoint = {
            "byte_count": destination.stat().st_size,
            "path": str(destination.relative_to(self.output_root)),
            "sha256": _sha256_file(destination),
        }
        self._manifest["epochs"].append(
            {
                "checkpoint": checkpoint,
                "epoch": epoch,
                "recovery": None,
                "status": "AWAITING_TRUE_RECOVERY",
            }
        )
        self._manifest["epochs"].sort(key=lambda row: int(row["epoch"]))
        self._publish()
        return checkpoint

    def record_recovery(
        self,
        *,
        epoch: int,
        records: Iterable[Mapping[str, Any]],
    ) -> BudgetRecoverySummary:
        matches = [row for row in self._manifest["epochs"] if row["epoch"] == epoch]
        if len(matches) != 1:
            raise ValueError("recovery requires exactly one saved epoch")
        entry = matches[0]
        if entry["recovery"] is not None:
            raise ValueError(f"epoch {epoch} recovery is already registered")
        summary = trajectory_equal_budget_recovery(records)
        entry["recovery"] = summary.as_dict()
        entry["status"] = "TRUE_RECOVERY_RECORDED"
        candidates = [
            row for row in self._manifest["epochs"] if row["recovery"] is not None
        ]
        best = min(
            candidates,
            key=lambda row: (
                -row["recovery"][
                    "macro_B1_B4_trajectory_equal_normalized_recovery"
                ],
                row["epoch"],
            ),
        )
        self._manifest["best"] = {
            "checkpoint": best["checkpoint"],
            "epoch": best["epoch"],
            "recovery": best["recovery"],
        }
        self._publish()
        return summary

    def after_epoch(
        self,
        model: Any,
        *,
        epoch: int,
        evaluator: Callable[[Any], Iterable[Mapping[str, Any]]],
    ) -> BudgetRecoverySummary:
        """Minimal trainer hook: persist, run truth selection, then update best."""
        self.save_epoch(model, epoch=epoch)
        return self.record_recovery(epoch=epoch, records=evaluator(model))


__all__ = [
    "BUDGETS",
    "BudgetRecoverySummary",
    "RecoveryCheckpointManager",
    "atomic_safetensors_epoch_saver",
    "trajectory_equal_budget_recovery",
]
