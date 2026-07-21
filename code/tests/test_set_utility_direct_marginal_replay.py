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
    assert path.trace[-1]["accepted"] is False


def test_learned_stop_is_not_replaced_by_zero_gain() -> None:
    path = direct_marginal_at_most_budget_path(
        (1, 2),
        budget=2,
        score=lambda _selected: (0.4, 0.3, 0.2),
        stop_semantics=LEARNED_STOP,
    )

    assert path.selections == {1: (), 2: ()}
    assert path.score_count == 3
    assert path.trace == (
        {
            "accepted": False,
            "base_subset": (),
            "best_event": 1,
            "best_marginal": 0.3,
            "stop_score": 0.4,
        },
    )


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
    with pytest.raises(ValueError, match="exactly zero"):
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
    def score_encoded_candidates(
        self, _encoded: object, selected_mask: _FakeMask
    ) -> _FakeScores:
        if not selected_mask.selected:
            return _FakeScores((0.25, 0.5, 0.1))
        return _FakeScores((0.3, 0.0, 0.2))


class _BackendWithoutCudaEncoding(TorchDirectMarginalReplayBackend):
    def __init__(self) -> None:
        self.model = _FakeMarginalModel()
        self.torch = _FakeTorch()
        self.device = "cuda:0"
        self.stop_semantics = LEARNED_STOP

    def _encode_request(self, _request: object):
        return object(), time.perf_counter(), {"raw_source_encoding": 0.0}, {}

    def _synchronize(self) -> None:
        return None


def test_live_backend_exposes_nested_path_and_learned_stop() -> None:
    request = SimpleNamespace(candidate_event_ids=(1, 2), budget=2)
    backend = _BackendWithoutCudaEncoding()
    path = backend.select_all_budgets(request)
    assert path.selections == {1: (1,), 2: (1,)}

    result = backend(request)
    assert result.selected_event_step_ids == (1,)
    assert result.selected_predicted_utility == 0.5
    assert result.scored_subsets == (((), 0.0), ((1,), 0.5))
    assert result.latency_ms["selector_total"] >= 0.0
