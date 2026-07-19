from __future__ import annotations

from dataclasses import dataclass

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import ParsedGUIOwlV21Output
from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v1 import (
    AUTO_FROZEN_ENCODED_CONDITION,
    PreparedExactInputV1,
)
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21GenerationResult
from causalcache.set_utility_gui_owl_v2_1_action_stability_adapter_v1 import (
    run_fresh_encode_condition_v1,
    run_frozen_encoded_condition_v1,
)
from causalcache.set_utility_gui_owl_v2_1_throughput_adapter import (
    GUIOwlV21ThroughputReferenceInput,
)


def _result(action: GUIOwlV2Action, sequence: str, decoded: str) -> GUIOwlV21GenerationResult:
    return GUIOwlV21GenerationResult(
        output_text="opaque",
        parsed_output=ParsedGUIOwlV21Output(
            canonical_action=action,
            raw_native_output="opaque",
            raw_tool_call_json="opaque",
            source_action_name=action.action,
        ),
        metadata={
            "decoded_output_utf8_sha256": decoded * 64,
            "generated_token_ids_sha256": sequence * 64,
        },
    )


@dataclass
class _FakePrepared:
    unchanged: bool = True


class _FakeRuntime:
    def __init__(self, results):
        self.results = list(results)
        self.fresh_calls = 0
        self.prepared_calls = 0
        self.prepare_calls = 0

    def generate_native_action(self, messages):
        self.fresh_calls += 1
        return self.results.pop(0)

    def prepare_exact_input_v1(self, messages):
        self.prepare_calls += 1
        return _FakePrepared()

    def prepared_input_unchanged_v1(self, prepared):
        return prepared.unchanged

    def generate_from_prepared_v1(self, prepared):
        self.prepared_calls += 1
        return self.results.pop(0)


def test_fresh_condition_projects_three_equality_levels_without_hashes() -> None:
    runtime = _FakeRuntime(
        [
            _result(GUIOwlV2Action(action="wait"), "a", "b"),
            _result(GUIOwlV2Action(action="wait"), "c", "b"),
        ]
    )
    payload = run_fresh_encode_condition_v1(
        runtime,
        GUIOwlV21ThroughputReferenceInput(messages=({"role": "user"},)),
    )
    assert runtime.fresh_calls == 2
    assert payload["encode_call_count"] == 2
    assert payload["generation_call_count"] == 2
    assert payload["exact_generated_sequence_equal"] is False
    assert payload["decoded_output_equal"] is True
    assert payload["canonical_action_equal"] is True
    assert "sha256" not in repr(payload).lower()


def test_frozen_condition_encodes_once_and_checks_input_three_times() -> None:
    runtime = _FakeRuntime(
        [
            _result(GUIOwlV2Action(action="wait"), "a", "b"),
            _result(GUIOwlV2Action(action="click", coordinate=(1, 2)), "c", "d"),
        ]
    )
    payload = run_frozen_encoded_condition_v1(
        runtime,
        GUIOwlV21ThroughputReferenceInput(messages=({"role": "user"},)),
        condition_id=AUTO_FROZEN_ENCODED_CONDITION,
    )
    assert runtime.prepare_calls == 1
    assert runtime.prepared_calls == 2
    assert payload["encode_call_count"] == 1
    assert payload["encoded_input_unchanged_before"] is True
    assert payload["encoded_input_unchanged_between"] is True
    assert payload["encoded_input_unchanged_after"] is True
    assert payload["canonical_action_equal"] is False


def test_failed_generation_exposes_only_safe_class() -> None:
    class Broken(_FakeRuntime):
        def generate_native_action(self, messages):
            self.fresh_calls += 1
            raise RuntimeError("sensitive output")

    runtime = Broken([])
    payload = run_fresh_encode_condition_v1(
        runtime,
        GUIOwlV21ThroughputReferenceInput(messages=({"role": "user"},)),
    )
    assert payload["failure_class"] == "RuntimeError"
    assert payload["generation_call_count"] == 1
    assert payload["generation_completed_count"] == 0
    assert "sensitive" not in repr(payload)
