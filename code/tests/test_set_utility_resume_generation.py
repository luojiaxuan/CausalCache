from __future__ import annotations

from pathlib import Path

import pytest

from causalcache.set_utility_resume_generation import (
    has_published_resume_generation,
    load_resume_rank_snapshot,
    publish_resume_generation,
    read_latest_resume_generation,
    restore_rng_states,
    resume_generation_path,
    write_resume_rank_snapshot,
)


torch = pytest.importorskip("torch")


def _snapshot(*, identity: dict[str, object], weight: float) -> dict[str, object]:
    return {
        "identity": identity,
        "model": {"weight": torch.tensor([weight])},
        "optimizer": {
            "param_groups": [{"lr": 0.1, "params": [0]}],
            "state": {0: {"exp_avg": torch.ones(1), "step": torch.tensor(3.0)}},
        },
        "python_rng_state": (3, (), None),
        "scheduler": {"_step_count": 4, "last_epoch": 3},
        "torch_rng_state": torch.get_rng_state(),
    }


def _generation(
    root: Path,
    *,
    epoch: int,
    identity: dict[str, object],
    weights: tuple[float, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        write_resume_rank_snapshot(
            root,
            epoch=epoch,
            rank=rank,
            world_size=len(weights),
            snapshot=_snapshot(identity=identity, weight=weight),
            torch=torch,
        )
        for rank, weight in enumerate(weights)
    )


def test_partial_next_generation_is_ignored_and_previous_epoch_replays(
    tmp_path: Path,
) -> None:
    identity = {"config_sha256": "a" * 64, "world_size": 2}
    epoch_one = _generation(
        tmp_path, epoch=1, identity=identity, weights=(1.0, 1.0)
    )
    published = publish_resume_generation(
        tmp_path, epoch_one, epoch=1, world_size=2
    )
    assert published["epoch"] == 1
    with pytest.raises(ValueError, match="immutable"):
        write_resume_rank_snapshot(
            tmp_path,
            epoch=1,
            rank=0,
            world_size=2,
            snapshot=_snapshot(identity=identity, weight=7.0),
            torch=torch,
        )

    write_resume_rank_snapshot(
        tmp_path,
        epoch=2,
        rank=0,
        world_size=2,
        snapshot=_snapshot(identity=identity, weight=2.0),
        torch=torch,
    )
    with pytest.raises(ValueError, match="inventory is incomplete"):
        publish_resume_generation(
            tmp_path,
            (
                write_resume_rank_snapshot(
                    tmp_path,
                    epoch=2,
                    rank=0,
                    world_size=2,
                    snapshot=_snapshot(identity=identity, weight=2.0),
                    torch=torch,
                ),
            ),
            epoch=2,
            world_size=2,
        )

    latest = read_latest_resume_generation(tmp_path, expected_world_size=2)
    assert latest["epoch"] == 1
    snapshot, pointer = load_resume_rank_snapshot(
        tmp_path,
        rank=1,
        expected_world_size=2,
        expected_identity=identity,
        map_location="cpu",
        torch=torch,
    )
    assert snapshot["epoch"] == 1
    assert pointer["content_sha256"] == published["content_sha256"]
    assert resume_generation_path(tmp_path, epoch=2, rank=0).is_file()


def test_rank_drift_never_publishes_generation(tmp_path: Path) -> None:
    identity = {"config_sha256": "b" * 64, "world_size": 2}
    drifted = _generation(
        tmp_path, epoch=1, identity=identity, weights=(1.0, 9.0)
    )
    with pytest.raises(ValueError, match="rank drift"):
        publish_resume_generation(tmp_path, drifted, epoch=1, world_size=2)
    assert has_published_resume_generation(tmp_path) is False


def test_restore_rng_states_normalizes_loaded_tensors_to_cpu_byte() -> None:
    class FakeCuda:
        def __init__(self) -> None:
            self.state = None

        def set_rng_state(self, state):
            self.state = state

    class FakeTorch:
        uint8 = torch.uint8

        def __init__(self) -> None:
            self.cuda = FakeCuda()
            self.state = None

        @staticmethod
        def is_tensor(value):
            return torch.is_tensor(value)

        def set_rng_state(self, state):
            self.state = state

    fake = FakeTorch()
    restore_rng_states(
        {
            "torch_rng_state": torch.tensor([1, 2], dtype=torch.int64),
            "cuda_rng_state": torch.tensor([3, 4], dtype=torch.int64),
        },
        torch=fake,
    )
    assert fake.state.device.type == "cpu"
    assert fake.state.dtype == torch.uint8
    assert fake.cuda.state.device.type == "cpu"
    assert fake.cuda.state.dtype == torch.uint8
