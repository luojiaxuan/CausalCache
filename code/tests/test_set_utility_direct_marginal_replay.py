from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from causalcache.set_utility_direct_marginal_replay import (
    DIRECT_MARGINAL_REPLAY_SCHEMA_VERSION,
    DIRECT_MARGINAL_REPLAY_STATUS,
    FIXED_ZERO_STOP,
    LEARNED_STOP,
    DirectMarginalReplayBinding,
    direct_marginal_at_most_budget_path,
)
from causalcache.set_utility_live_controller import TorchDirectMarginalReplayBackend


def _signed_binding(**updates: object) -> dict[str, object]:
    value = {
        "checkpoint_sha256": "a" * 64,
        "model_family": "set_transformer_direct_marginal",
        "replay_role": "train_heldout",
        "schema_version": DIRECT_MARGINAL_REPLAY_SCHEMA_VERSION,
        "selector_config_sha256": "b" * 64,
        "status": DIRECT_MARGINAL_REPLAY_STATUS,
        "variant": "set_transformer_direct_d256_l16_r2_s2",
        **updates,
    }
    value["content_sha256"] = hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return value


def test_fixed_zero_stop_returns_nested_at_most_budget_path() -> None:
    def score(selected: tuple[int, ...]) -> tuple[float, ...]:
        return {
            (): (0.0, 0.2, 0.8, 0.1),
            (2,): (0.0, -0.1, 0.0, 0.4),
            (2, 3): (0.0, -0.3, 0.0, 0.0),
        }[selected]

    path = direct_marginal_at_most_budget_path(
        (1, 2, 3), budget=4, score=score, stop_semantics=FIXED_ZERO_STOP
    )

    assert path.selections == {
        1: (2,),
        2: (2, 3),
        3: (2, 3),
        4: (2, 3),
    }
    assert path.predicted_utilities == pytest.approx(
        {1: 0.8, 2: 1.2, 3: 1.2, 4: 1.2}
    )
    assert tuple(subset for subset, _ in path.scored_subsets) == (
        (),
        (2,),
        (2, 3),
    )
    assert tuple(value for _, value in path.scored_subsets) == pytest.approx(
        (0.0, 0.8, 1.2)
    )
    assert path.score_count == 9
    assert path.trace[-1]["reason"] == "native_stop"
    assert path.recent_fallback_threshold is None
    assert path.recent_fallback_budgets == ()
    assert path.budget_routes == {
        1: "direct_learned",
        2: "direct_learned",
        3: "direct_learned",
        4: "direct_learned",
    }


def test_learned_stop_is_not_replaced_by_zero_gain() -> None:
    path = direct_marginal_at_most_budget_path(
        (1, 2),
        budget=2,
        score=lambda _selected: (0.4, 0.3, 0.2),
        stop_semantics=LEARNED_STOP,
    )

    assert path.selections == {1: (), 2: ()}
    assert path.score_count == 3
    assert path.trace[0]["base_subset"] == []
    assert path.trace[0]["best_event"] == 1
    assert path.trace[0]["best_marginal"] == pytest.approx(0.3)
    assert path.trace[0]["stop_threshold"] == pytest.approx(0.4)
    assert path.trace[0]["reason"] == "native_stop"


def test_learned_stop_can_accept_signed_marginal_below_zero() -> None:
    path = direct_marginal_at_most_budget_path(
        (1, 2),
        budget=1,
        score=lambda _selected: (-0.5, -0.25, -0.75),
        stop_semantics=LEARNED_STOP,
    )

    assert path.selection(1) == (1,)
    assert path.predicted_utility(1) == -0.25


def test_equal_candidate_and_stop_prefers_stop() -> None:
    path = direct_marginal_at_most_budget_path(
        (2, 5),
        budget=1,
        score=lambda _selected: (0.0, 0.0, 0.0),
        stop_semantics=FIXED_ZERO_STOP,
    )
    assert path.selection(1) == ()


def test_fixed_zero_stop_rejects_model_semantics_drift() -> None:
    with pytest.raises(ValueError, match="must remain zero"):
        direct_marginal_at_most_budget_path(
            (1,),
            budget=1,
            score=lambda _selected: (0.1, 1.0),
            stop_semantics=FIXED_ZERO_STOP,
        )


def test_versioned_binding_is_separate_from_old_go_authorization() -> None:
    binding = DirectMarginalReplayBinding.from_mapping(_signed_binding())
    assert binding.replay_role == "train_heldout"

    old_status = _signed_binding(status="AUTHORIZED_BY_HELDOUT_AND_NATIVE_REPLAY_GO")
    with pytest.raises(ValueError, match="status drifted"):
        DirectMarginalReplayBinding.from_mapping(old_status)


def test_binding_rejects_post_signature_mutation() -> None:
    value = _signed_binding()
    value["variant"] = "mutated"
    with pytest.raises(ValueError, match="signature drifted"):
        DirectMarginalReplayBinding.from_mapping(value)


class _FakeMask:
    def __init__(self) -> None:
        self.selected: set[int] = set()

    def __setitem__(self, key: tuple[int, int], value: bool) -> None:
        if value:
            self.selected.add(key[1])


class _FakeTorch:
    bool = bool
    bfloat16 = "bfloat16"

    @staticmethod
    def zeros(*_args: object, **_kwargs: object) -> _FakeMask:
        return _FakeMask()

    @staticmethod
    def inference_mode() -> object:
        return nullcontext()

    @staticmethod
    def autocast(**_kwargs: object) -> object:
        return nullcontext()


class _FakeScores:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values

    def __getitem__(self, index: int) -> "_FakeScores":
        assert index == 0
        return self

    def tolist(self) -> list[float]:
        return list(self.values)


class _FakeMarginalModel:
    def __init__(
        self, scores: dict[tuple[int, ...], tuple[float, ...]]
    ) -> None:
        self.scores = scores

    def score_encoded_candidates(
        self, _encoded: object, selected_mask: _FakeMask
    ) -> _FakeScores:
        selected = tuple(index + 1 for index in sorted(selected_mask.selected))
        return _FakeScores(self.scores[selected])


class _BackendWithoutCudaEncoding(TorchDirectMarginalReplayBackend):
    def __init__(
        self,
        scores: dict[tuple[int, ...], tuple[float, ...]],
        *,
        stop_semantics: str = LEARNED_STOP,
        recent_fallback_threshold: float | None = None,
        recent_fallback_budgets: tuple[int, ...] = (),
    ) -> None:
        self.model = _FakeMarginalModel(scores)
        self.torch = _FakeTorch()
        self.device = "cuda:0"
        self.encode_calls = 0
        self.stop_semantics = stop_semantics
        self.recent_fallback_threshold = recent_fallback_threshold
        self.recent_fallback_budgets = recent_fallback_budgets
        self.selection_mode = (
            "recent_budget_deferral"
            if recent_fallback_budgets
            else (
                "direct"
                if recent_fallback_threshold is None
                else "confidence_gated_recent"
            )
        )

    def _encode_request(self, _request: object):
        self.encode_calls += 1
        return object(), time.perf_counter(), {"raw_source_encoding": 0.0}, {}

    def _synchronize(self) -> None:
        return None


def test_live_backend_hybrid_mode_requires_explicit_valid_threshold() -> None:
    source_encoder = SimpleNamespace(
        runtime=SimpleNamespace(device="cuda:0"),
        torch=_FakeTorch(),
    )
    direct = TorchDirectMarginalReplayBackend(
        model=_FakeMarginalModel({}),
        source_encoder=source_encoder,
        stop_semantics=FIXED_ZERO_STOP,
    )
    hybrid = TorchDirectMarginalReplayBackend(
        model=_FakeMarginalModel({}),
        source_encoder=source_encoder,
        stop_semantics=FIXED_ZERO_STOP,
        recent_fallback_threshold=0.001,
    )
    deferred = TorchDirectMarginalReplayBackend(
        model=_FakeMarginalModel({}),
        source_encoder=source_encoder,
        stop_semantics=LEARNED_STOP,
        recent_fallback_budgets=(1, 2),
    )
    assert direct.selection_mode == "direct"
    assert direct.recent_fallback_threshold is None
    assert hybrid.selection_mode == "confidence_gated_recent"
    assert hybrid.recent_fallback_threshold == pytest.approx(0.001)
    assert deferred.selection_mode == "recent_budget_deferral"
    assert deferred.recent_fallback_budgets == (1, 2)
    with pytest.raises(ValueError, match="threshold is invalid"):
        TorchDirectMarginalReplayBackend(
            model=_FakeMarginalModel({}),
            source_encoder=source_encoder,
            stop_semantics=FIXED_ZERO_STOP,
            recent_fallback_threshold=float("nan"),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        TorchDirectMarginalReplayBackend(
            model=_FakeMarginalModel({}),
            source_encoder=source_encoder,
            stop_semantics=LEARNED_STOP,
            recent_fallback_threshold=0.001,
            recent_fallback_budgets=(1, 2),
        )


def test_live_backend_exposes_nested_path_and_learned_stop() -> None:
    request = SimpleNamespace(candidate_event_ids=(1, 2), budget=2)
    backend = _BackendWithoutCudaEncoding(
        {
            (): (0.25, 0.5, 0.1),
            (1,): (0.3, 0.0, 0.2),
        }
    )
    path = backend.select_all_budgets(request)
    assert path.selections == {1: (1,), 2: (1,)}

    result = backend(request)
    assert result.selected_event_step_ids == (1,)
    assert result.selected_predicted_utility == 0.5
    assert result.scored_subsets == (((), 0.0), ((1,), 0.5))
    assert result.latency_ms["selector_total"] >= 0.0


def test_live_backend_defaults_to_direct_without_confidence_fallback() -> None:
    request = SimpleNamespace(candidate_event_ids=(1, 2), budget=1)
    backend = _BackendWithoutCudaEncoding(
        {(): (0.0, 0.5005, 0.5)},
        stop_semantics=FIXED_ZERO_STOP,
    )

    path = backend.select_all_budgets(request)
    assert path.selection(1) == (1,)
    assert path.recent_fallback_threshold is None
    assert path.trace[0]["reason"] == "learned"
    assert backend.selection_mode == "direct"


def test_hybrid_falls_back_when_learned_advantage_is_below_threshold() -> None:
    request = SimpleNamespace(candidate_event_ids=(1, 2), budget=1)
    backend = _BackendWithoutCudaEncoding(
        {(): (0.0, 0.5005, 0.5)},
        stop_semantics=FIXED_ZERO_STOP,
        recent_fallback_threshold=0.001,
    )

    path = backend.select_all_budgets(request)
    assert path.selection(1) == (2,)
    assert path.predicted_utility(1) == pytest.approx(0.5)
    assert path.trace[0]["reason"] == "recent_fallback"
    assert backend.selection_mode == "confidence_gated_recent"


def test_hybrid_uses_confident_learned_override_and_remains_nested() -> None:
    request = SimpleNamespace(candidate_event_ids=(1, 2, 3), budget=4)
    backend = _BackendWithoutCudaEncoding(
        {
            (): (0.0, 0.5005, 0.1, 0.5),
            (3,): (0.0, 0.8, 0.7, 0.0),
            (1, 3): (0.0, 0.0, -0.2, 0.0),
        },
        stop_semantics=FIXED_ZERO_STOP,
        recent_fallback_threshold=0.001,
    )

    path = backend.select_all_budgets(request)
    assert path.selections == {
        1: (3,),
        2: (1, 3),
        3: (1, 3),
        4: (1, 3),
    }
    assert [row["reason"] for row in path.trace] == [
        "recent_fallback",
        "confidence_gated_learned_override",
        "confidence_gated_stop",
    ]


def test_hybrid_preserves_structured_deepsets_learned_stop() -> None:
    request = SimpleNamespace(candidate_event_ids=(1, 2), budget=4)
    backend = _BackendWithoutCudaEncoding(
        {(): (0.4, 0.3, 0.2)},
        stop_semantics=LEARNED_STOP,
        recent_fallback_threshold=0.001,
    )

    path = backend.select_all_budgets(request)
    assert path.selections == {1: (), 2: (), 3: (), 4: ()}
    assert path.trace[0]["stop_threshold"] == pytest.approx(0.4)
    assert path.trace[0]["reason"] == "confidence_gated_stop"


def test_budget_deferral_skips_learned_search_for_b1_b2() -> None:
    request = SimpleNamespace(candidate_event_ids=(1, 2, 3, 4), budget=2)
    backend = _BackendWithoutCudaEncoding(
        {},
        stop_semantics=LEARNED_STOP,
        recent_fallback_budgets=(1, 2),
    )

    path = backend.select_all_budgets(request)
    assert path.selections == {1: (4,), 2: (3, 4)}
    assert path.budget_routes == {
        1: "recent_budget_deferral",
        2: "recent_budget_deferral",
    }
    assert path.score_count == 0
    assert path.predicted_utilities == {1: 0.0, 2: 0.0}
    assert all(row["predicted_utility_scored"] is False for row in path.trace)
    assert backend.encode_calls == 0


def test_budget_deferral_can_be_non_nested_across_budgets() -> None:
    request = SimpleNamespace(candidate_event_ids=(1, 2, 3, 4), budget=4)
    backend = _BackendWithoutCudaEncoding(
        {
            (): (0.0, 0.9, 0.2, 0.1, 0.05),
            (1,): (0.0, 0.0, 0.8, 0.2, 0.1),
            (1, 2): (0.0, 0.0, 0.0, 0.7, 0.2),
            (1, 2, 3): (0.0, 0.0, 0.0, 0.0, -0.1),
        },
        stop_semantics=FIXED_ZERO_STOP,
        recent_fallback_budgets=(1, 2),
    )

    path = backend.select_all_budgets(request)
    assert path.selections == {
        1: (4,),
        2: (3, 4),
        3: (1, 2, 3),
        4: (1, 2, 3),
    }
    assert not set(path.selection(2)).issubset(path.selection(3))
    assert all(len(path.selection(budget)) <= budget for budget in (1, 2, 3, 4))
    assert path.budget_routes == {
        1: "recent_budget_deferral",
        2: "recent_budget_deferral",
        3: "direct_learned",
        4: "direct_learned",
    }
    assert backend.encode_calls == 1


def test_budget_deferral_keeps_deepsets_learned_stop_for_b3_b4() -> None:
    request = SimpleNamespace(candidate_event_ids=(1, 2, 3, 4), budget=4)
    backend = _BackendWithoutCudaEncoding(
        {
            (): (0.1, 0.8, 0.2, 0.1, 0.05),
            (1,): (0.2, 0.0, 0.7, 0.1, 0.05),
            (1, 2): (0.6, 0.0, 0.0, 0.5, 0.4),
        },
        stop_semantics=LEARNED_STOP,
        recent_fallback_budgets=(1, 2),
    )

    path = backend.select_all_budgets(request)
    assert path.selection(1) == (4,)
    assert path.selection(2) == (3, 4)
    assert path.selection(3) == (1, 2)
    assert path.selection(4) == (1, 2)
    assert any(
        row.get("reason") == "native_stop"
        and row.get("stop_threshold") == pytest.approx(0.6)
        for row in path.trace
    )
