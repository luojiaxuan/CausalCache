from types import SimpleNamespace

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import serialize_gui_owl_v2_1_teacher_target
from scripts.run_set_utility_native_replay import _run_state, _state_identity


def test_native_replay_state_is_resumable_and_records_components(
    tmp_path, monkeypatch
) -> None:
    reference = GUIOwlV2Action(action="click", coordinate=(20, 30))
    schedule = {
        "candidate_event_ids": [1, 2, 3, 4, 5],
        "coalitions": [
            {
                "event_ids": [1, 3],
                "sources": ["winner:B2", "ocr_rgb:B2"],
            }
        ],
        "reference": {
            "canonical_action": reference.arguments(),
            "serialized_action": serialize_gui_owl_v2_1_teacher_target(reference),
            "terminal_sha256": "a" * 64,
        },
        "role": "evaluation",
        "state_id": "trajectory:decision:006",
        "trajectory_id": "trajectory",
        "winner_model": "set_transformer",
    }
    query = SimpleNamespace(
        candidate_event_step_ids=(1, 2, 3, 4, 5),
        role="evaluation",
        state_id="trajectory:decision:006",
        trajectory_id="trajectory",
    )
    trajectory = SimpleNamespace(image_payloads={})
    calls = []

    class Runtime:
        def generate_native_action(self, messages):
            calls.append(messages)
            return SimpleNamespace(
                metadata={"latency_ms": 12.0},
                output_text=serialize_gui_owl_v2_1_teacher_target(reference),
                parsed_output=SimpleNamespace(canonical_action=reference),
            )

    monkeypatch.setattr(
        "scripts.run_set_utility_native_replay.build_variable_history_messages",
        lambda query, coalition, **kwargs: (query.state_id, tuple(coalition)),
    )
    result = _run_state(
        query=query,
        trajectory=trajectory,
        schedule=schedule,
        output_root=tmp_path,
        runtime=Runtime(),
        base_decoder=lambda payload: payload,
        state_identity="b" * 64,
        source_revision="c" * 40,
        worker_index=2,
    )
    assert len(calls) == 1
    assert result["status"] == "COMPLETED_SET_UTILITY_NATIVE_REPLAY_STATE"
    assert result["records"][0]["comparison"]["canonical_action_exact"] is True
    assert result["records"][0]["sources"] == ["winner:B2", "ocr_rgb:B2"]

    resumed = _run_state(
        query=query,
        trajectory=trajectory,
        schedule=schedule,
        output_root=tmp_path,
        runtime=Runtime(),
        base_decoder=lambda payload: payload,
        state_identity="b" * 64,
        source_revision="c" * 40,
        worker_index=2,
    )
    assert resumed == result
    assert len(calls) == 1


def test_native_replay_state_identity_binds_schedule() -> None:
    values = {
        "source_revision": "a" * 40,
        "source_shard_sha256": "b" * 64,
        "schedule_manifest_sha256": "c" * 64,
        "schedule_receipt_sha256": "d" * 64,
        "schedule": {"state_id": "state", "coalitions": []},
    }
    identity = _state_identity(**values)
    assert identity == _state_identity(**values)
    assert identity != _state_identity(
        **{**values, "schedule": {"state_id": "state", "coalitions": [[]]}}
    )
