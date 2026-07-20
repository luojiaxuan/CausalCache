from types import SimpleNamespace

from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21GenerationParseError
from scripts.run_set_utility_variable_history_labels import (
    _cuda_device,
    _microbatches,
    _run_state,
    _state_lane,
    _state_identity,
)


def test_label_device_must_be_explicit_cuda_index() -> None:
    assert _cuda_device("cuda:3") == "cuda:3"
    import pytest

    with pytest.raises(ValueError, match="explicit cuda:N"):
        _cuda_device("cuda")


def _schedule() -> dict[str, object]:
    return {
        "candidate_event_ids": [1, 2, 3, 4, 5, 6],
        "coalitions": [
            {"event_ids": [], "source": "empty"},
            *(
                {"event_ids": [event], "source": "singleton"}
                for event in range(1, 7)
            ),
            {"event_ids": [1, 6], "source": "pair"},
            {"event_ids": [1, 2, 3, 4, 5, 6], "source": "full"},
        ],
        "state_id": "trajectory:decision:007",
    }


def test_microbatches_group_by_cardinality_and_skip_full_identity() -> None:
    batches = _microbatches(_schedule(), microbatch_size=4)
    assert batches == (
        ((),),
        ((1,), (2,), (3,), (4,)),
        ((5,), (6,)),
        ((1, 6),),
    )


def test_state_identity_binds_revision_and_schedule() -> None:
    values = {
        "source_revision": "a" * 40,
        "scientific_config_sha": "b" * 64,
        "execution_config_sha": "c" * 64,
        "source_shard_sha": "d" * 64,
        "schedule_receipt_sha": "e" * 64,
        "schedule": _schedule(),
    }
    first = _state_identity(**values)
    second = _state_identity(**values)
    assert first == second
    assert first != _state_identity(**{**values, "source_revision": "f" * 40})


def test_state_lanes_are_deterministic_and_disjoint() -> None:
    state_ids = [f"trajectory:decision:{index:03d}" for index in range(32)]
    first = {_state_lane(state_id, 2) for state_id in state_ids}
    second = [_state_lane(state_id, 2) for state_id in state_ids]
    assert first == {0, 1}
    assert second == [_state_lane(state_id, 2) for state_id in state_ids]


def test_reference_parse_failure_is_an_atomic_state_skip(tmp_path, monkeypatch) -> None:
    state_id = "trajectory:decision:007"
    query = SimpleNamespace(
        candidate_event_step_ids=(1, 2, 3, 4, 5, 6),
        role="train",
        state_id=state_id,
        trajectory_id="trajectory",
    )
    trajectory = SimpleNamespace(image_payloads={})

    class Runtime:
        def generate_native_action(self, messages):
            del messages
            raise GUIOwlV21GenerationParseError(
                output_text="malformed",
                metadata={},
                parse_error=ValueError("invalid reference tool call"),
            )

    monkeypatch.setattr(
        "scripts.run_set_utility_variable_history_labels.build_variable_history_messages",
        lambda *args, **kwargs: (),
    )
    result = _run_state(
        query=query,
        trajectory=trajectory,
        schedule=_schedule(),
        output_root=tmp_path,
        runtime=Runtime(),
        base_decoder=lambda payload: payload,
        kl_kernel=None,
        state_identity="a" * 64,
        source_revision="b" * 40,
        scientific_config_sha="c" * 64,
        execution_config_sha="d" * 64,
        worker_index=3,
        maximum_reference_repeat_kl=1e-6,
        teacher_microbatch_size=4,
    )

    assert result["status"] == "SKIPPED_VARIABLE_HISTORY_LABEL_STATE"
    assert result["failure_class"] == "GUIOwlV21GenerationParseError"
    terminal = tmp_path / "states" / "trajectory_decision_007.json"
    assert terminal.is_file()
    assert _run_state(
        query=query,
        trajectory=trajectory,
        schedule=_schedule(),
        output_root=tmp_path,
        runtime=Runtime(),
        base_decoder=lambda payload: payload,
        kl_kernel=None,
        state_identity="a" * 64,
        source_revision="b" * 40,
        scientific_config_sha="c" * 64,
        execution_config_sha="d" * 64,
        worker_index=3,
        maximum_reference_repeat_kl=1e-6,
        teacher_microbatch_size=4,
    ) == result
