from __future__ import annotations

from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v1 import (
    GUIOwlV21AutoActionStabilityRuntimeV1,
    GUIOwlV21EagerActionStabilityRuntimeV1,
    PreparedExactInputV1,
    validate_auto_attention_v1,
)
from causalcache.policy.gui_owl_v2_1_throughput_runtime import (
    GUIOwlV21ThroughputRuntime,
)
from causalcache.policy.gui_owl_v2_2_eager_runtime import GUIOwlV22EagerRuntime


class _Tensor:
    def __init__(self, value: int):
        self.value = value
        self.shape = (1,)
        self.dtype = "fake"
        self.device = "cuda:0"

    def clone(self):
        return _Tensor(self.value)

    def equal(self, other):
        return isinstance(other, _Tensor) and self.value == other.value


def test_concrete_profiles_preserve_the_two_frozen_parent_runtimes() -> None:
    assert issubclass(GUIOwlV21AutoActionStabilityRuntimeV1, GUIOwlV21ThroughputRuntime)
    assert issubclass(GUIOwlV21EagerActionStabilityRuntimeV1, GUIOwlV22EagerRuntime)


def test_prepared_input_detects_tensor_mutation_and_wrong_owner() -> None:
    runtime = object.__new__(GUIOwlV21AutoActionStabilityRuntimeV1)
    prepared = PreparedExactInputV1(
        owner_identity=id(runtime),
        model_inputs={"input_ids": _Tensor(1)},
        image_counts=(1,),
        snapshots={"input_ids": _Tensor(1)},
    )
    assert runtime.prepared_input_unchanged_v1(prepared) is True
    prepared.model_inputs["input_ids"].value = 2
    assert runtime.prepared_input_unchanged_v1(prepared) is False

    other = object.__new__(GUIOwlV21AutoActionStabilityRuntimeV1)
    try:
        other.prepared_input_unchanged_v1(prepared)
    except ValueError as error:
        assert "different runtime" in str(error)
    else:
        raise AssertionError("wrong-owner prepared input was accepted")


def test_auto_profile_requires_resolved_sdpa_at_all_three_levels() -> None:
    expected = {"text": "sdpa", "top": "sdpa", "vision": "sdpa"}
    assert validate_auto_attention_v1(expected) == expected
    try:
        validate_auto_attention_v1({"text": "sdpa", "top": "eager", "vision": "sdpa"})
    except RuntimeError as error:
        assert "resolve to SDPA" in str(error)
    else:
        raise AssertionError("mixed automatic attention profile was accepted")
