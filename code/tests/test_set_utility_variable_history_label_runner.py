from scripts.run_set_utility_variable_history_labels import (
    _microbatches,
    _state_identity,
)


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
