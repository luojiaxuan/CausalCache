from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import scripts.run_set_conditioned_v4_exploration as v4_runner

from causalcache.gate_v1_data import CandidateFeatures, FeatureState, GateState
from causalcache.gate_v1_provenance import canonical_selection_sha256
from causalcache.restoration_v2_2_label_table import validate_complete_distance_table
from causalcache.set_conditioned_v4_exploration import (
    BASE_MANIFEST_SHA256,
    METHODS,
    OOFTrial,
    ResidualStatePrediction,
    _historical_decision_details,
    base_selection_from_scores,
    canonical_json_line,
    choose_oof_selection,
    claim_fresh_label_access,
    frozen_base_residual_interpretation,
    load_frozen_base_ensemble,
    prediction_artifact_bytes,
    read_prediction_artifact,
    seal_label_blind_outputs,
    sha256_bytes,
    state_prediction_record,
    verify_evaluation_report,
    verify_fresh_label_access_claim,
    verify_label_blind_seal,
    zero_residual_prediction,
)
from causalcache.set_conditioned_v3_parser_repair_contract import (
    EXACT_HISTORICAL_PROTOCOL_ID,
)


def _state(source: str, event_count: int, *, ordinal: int = 0) -> GateState:
    event_ids = tuple(range(1, event_count + 1))
    baseline = 20.0
    distances = {
        coalition: baseline - float(sum(coalition))
        for size in range(event_count + 1)
        for coalition in itertools.combinations(event_ids, size)
    }
    table = validate_complete_distance_table(event_ids, distances)
    decision_step = 10 + ordinal * 3 + event_count
    return GateState(
        source_id=source,
        state_id=f"{source}:decision_step:{decision_step:03d}",
        decision_step_id=decision_step,
        candidate_event_step_ids=event_ids,
        q64=(0.0,) * 64,
        candidates=tuple(
            CandidateFeatures(
                event_step_id=event,
                h64=(0.0,) * 64,
                g8=(0.0,) * 8,
            )
            for event in event_ids
        ),
        table=table,
    )


def _feature(state: GateState) -> FeatureState:
    return FeatureState(
        source_id=state.source_id,
        state_id=state.state_id,
        decision_step_id=state.decision_step_id,
        candidate_event_step_ids=state.candidate_event_step_ids,
        q64=state.q64,
        candidates=state.candidates,
    )


def _prediction(
    event_ids: tuple[int, ...],
    *,
    singleton_scores: tuple[float, ...],
    pair_residual: float,
) -> ResidualStatePrediction:
    return ResidualStatePrediction(
        singleton_scores=dict(zip(event_ids, singleton_scores, strict=True)),
        pair_residual_scores={
            pair: pair_residual for pair in itertools.combinations(event_ids, 2)
        },
    )


class SetConditionedV4ExplorationTest(unittest.TestCase):
    def test_zero_residual_exactly_replays_positive_top_b(self) -> None:
        scores = {1: -2.0, 2: 0.0, 3: 1.5, 4: 0.5}
        prediction = zero_residual_prediction(scores)
        self.assertEqual(base_selection_from_scores(scores), (3, 4))
        self.assertTrue(
            all(value == 0.0 for value in prediction.pair_residual_scores.values())
        )

    def test_four_of_five_guard_accepts_and_falls_back(self) -> None:
        feature = _feature(_state("source-a", 2))
        pair = _prediction(
            (1, 2), singleton_scores=(1.0, -1.0), pair_residual=2.0
        )
        base = _prediction(
            (1, 2), singleton_scores=(1.0, -1.0), pair_residual=-2.0
        )
        accepted = state_prediction_record(feature, (pair, pair, pair, pair, base))
        self.assertEqual(accepted["frozen_base_selected"], [1])
        self.assertEqual(accepted["v4_unguarded_selected"], [1, 2])
        self.assertEqual(accepted["v4_safe_selected"], [1, 2])
        self.assertEqual(accepted["safe_decision"]["pair_candidate_vote_count"], 4)
        self.assertTrue(accepted["safe_decision"]["used_pair_candidate"])

        strong_pair = _prediction(
            (1, 2), singleton_scores=(1.0, -1.0), pair_residual=3.0
        )
        weak_base = _prediction(
            (1, 2), singleton_scores=(1.0, -1.0), pair_residual=-1.0
        )
        fallback = state_prediction_record(
            feature,
            (strong_pair, strong_pair, strong_pair, weak_base, weak_base),
        )
        self.assertEqual(fallback["v4_unguarded_selected"], [1, 2])
        self.assertEqual(fallback["v4_safe_selected"], [1])
        self.assertEqual(fallback["safe_decision"]["pair_candidate_vote_count"], 3)
        self.assertTrue(fallback["safe_decision"]["used_fallback"])

    def test_feature_only_artifact_has_48_replayable_rows(self) -> None:
        states = tuple(
            _state(f"source-{source:02d}", event_count, ordinal=event_count)
            for source in range(16)
            for event_count in (2, 3, 4)
        )
        features = tuple(map(_feature, states))
        predictions = tuple(
            (
                _prediction(
                    feature.candidate_event_step_ids,
                    singleton_scores=tuple(
                        0.1 * event for event in feature.candidate_event_step_ids
                    ),
                    pair_residual=0.0,
                ),
            )
            * 5
            for feature in features
        )
        payload = prediction_artifact_bytes(
            features,
            predictions,
            base_inventory=({"seed": 0},),
            residual_inventory=({"seed": 0},),
        )
        self.assertNotIn(b"distance", payload)
        replay = read_prediction_artifact(payload, features=features)
        self.assertEqual(len(replay), 48)
        self.assertTrue(
            all(
                row["frozen_base_selected"] == row["zero_residual_selected"]
                for row in replay.values()
            )
        )

    def test_frozen_manifest_byte_drift_fails_before_checkpoint_load(self) -> None:
        with self.assertRaisesRegex(ValueError, "manifest SHA256 drifted"):
            load_frozen_base_ensemble(b"{}\n", {})
        self.assertEqual(
            len(BASE_MANIFEST_SHA256),
            64,
        )

    def test_exact_historical_artifact_reorder_joins_seed_rows_by_state_id(self) -> None:
        features = tuple(
            _feature(_state(f"source-{source:02d}", count, ordinal=count))
            for source in range(16)
            for count in (2, 3, 4)
        )
        ensemble = {feature.state_id: () for feature in features}
        records = [
            {
                "state_id": feature.state_id,
                "ensemble_selected_event_step_ids": [],
                "seed_selected_event_step_ids": [
                    {"seed": seed, "selected_event_step_ids": []}
                    for seed in range(5)
                ],
            }
            for feature in reversed(features)
        ]
        payload = canonical_json_line(
            {
                "schema_version": "1.0.0",
                "protocol_id": EXACT_HISTORICAL_PROTOCOL_ID,
                "status": "FROZEN_GATE_V1_FRESH16_LEARNED_SELECTIONS_V1",
                "name": "independent",
                "selection_sha256": canonical_selection_sha256(ensemble),
                "decision_contract": (
                    "FEATURE_ONLY_ALL_ENSEMBLE_AND_SEED_DECISIONS_V1"
                ),
                "records": records,
            }
        )
        with patch(
            "causalcache.set_conditioned_v3_parser_repair."
            "HISTORICAL_INDEPENDENT_SHA256",
            sha256_bytes(payload),
        ):
            observed, by_seed = _historical_decision_details(
                payload,
                features=features,
            )
        self.assertEqual(observed, ensemble)
        self.assertEqual(len(by_seed), 5)
        self.assertTrue(
            all(rows == ensemble for rows in by_seed)
        )

    def test_oof_learning_rate_tie_chooses_lower_rate(self) -> None:
        trials = []
        for learning_rate, score in ((0.0003, 0.8), (0.001, 0.80005)):
            for seed in range(5):
                trials.append(
                    OOFTrial(
                        learning_rate=learning_rate,
                        seed=seed,
                        selected_epoch=0,
                        selected_score=score,
                        selected_raw_ratio=score,
                        selected_normalized_ratio=score,
                        optimization_epochs_run=0,
                        score_by_epoch=(score,),
                        raw_ratio_by_epoch=(score,),
                        normalized_ratio_by_epoch=(score,),
                        selected_predictions={},
                    )
                )
        self.assertEqual(choose_oof_selection(trials).learning_rate, 0.0003)

    def test_five_checks_route_promising_only_when_all_pass(self) -> None:
        def method(normalized: float, raw: float) -> dict[str, float]:
            return {
                "mean_normalized_recovery": normalized,
                "mean_raw_utility": raw,
            }

        metrics = {
            "overall": {
                "methods": {
                    "v4_safe": method(0.73, 0.05),
                    "frozen_base": method(0.71, 0.04),
                }
            },
            "by_event_count": {
                "4": {
                    "methods": {
                        "v4_safe": method(0.75, 0.05),
                        "frozen_base": method(0.72, 0.04),
                    }
                }
            },
        }
        bootstraps = {
            "v4_safe_minus_frozen_base": {
                "normalized": {
                    "lower": 0.001,
                    "positive_trajectory_count": 8,
                }
            }
        }
        route = frozen_base_residual_interpretation(metrics, bootstraps)
        self.assertEqual(len(route["checks"]), 5)
        self.assertTrue(all(route["checks"].values()))
        self.assertTrue(route["value"].startswith("PROMISING_"))
        bootstraps["v4_safe_minus_frozen_base"]["normalized"]["lower"] = 0.0
        stopped = frozen_base_residual_interpretation(metrics, bootstraps)
        self.assertEqual(
            stopped["value"],
            "NO_DEVELOPMENT_EVIDENCE_FOR_FROZEN_BASE_RESIDUAL",
        )
        self.assertFalse(stopped["may_authorize_confirm20"])

    def test_seal_claim_and_report_only_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            seal = seal_label_blind_outputs(
                output,
                {"artifact.bin": b"artifact"},
                training_report={"path": "train.json", "sha256": "0" * 64},
                base_invariant_report={
                    "path": "base.json",
                    "sha256": "1" * 64,
                },
                frozen_base_input_inventory=(),
                source_a_git_commit="1" * 40,
                execution_b_git_commit="2" * 40,
                runner_freeze_sha256="3" * 64,
                contract_sha256="4" * 64,
            )
            self.assertEqual(seal["fresh_label_access_count"], 0)
            self.assertEqual(verify_label_blind_seal(output)["confirm20_access_count"], 0)
            claim = claim_fresh_label_access(output)
            self.assertEqual(claim["fresh_label_access_claim_count"], 1)
            self.assertEqual(
                verify_fresh_label_access_claim(output)["confirm20_access_count"],
                0,
            )
            with self.assertRaisesRegex(ValueError, "one-shot v4 output"):
                claim_fresh_label_access(output)

            report = {
                "schema_version": "1.0.0",
                "protocol_id": (
                    "causalcache_set_conditioned_v4_"
                    "frozen_base_residual_development_v1"
                ),
                "status": (
                    "COMPLETED_SET_CONDITIONED_V4_"
                    "FRESH16_CONSUMED_DEVELOPMENT_V1"
                ),
                "label_blind_seal_sha256": "5" * 64,
                "label_access_claim_sha256": "6" * 64,
                "fresh_feature_sha256": (
                    "deacc63480ed706d44e7690d974250860"
                    "d1bc3e5967d4c2a3315090fe4b93939"
                ),
                "fresh_label_sha256": (
                    "832096b98011264a6b7fee74b2e787988"
                    "76ea0d5ffc5c9c9baf7642be8e92382"
                ),
                "historical_v1_independent_sha256": (
                    "2359c0f4b03e7009291c8912bea7193a"
                    "4dfebd5f01d4876efd312bc595404dd2"
                ),
                "trajectory_count": 16,
                "state_count": 48,
                "v4_fresh_label_access_claim_count": 1,
                "v4_fresh_label_semantic_decode_attempt_count": 1,
                "accounting_scope": "v3_to_v4_lineage",
                "lineage_totals_are_project_global": False,
                "v3_to_v4_lineage_claim_count": 2,
                "v3_to_v4_lineage_semantic_decode_attempt_count": 3,
                "confirm20_access_count": 0,
                "legacy_dev5_access_count": 0,
                "matched_nll_evaluation_count": 0,
                "closed_loop_episode_count": 0,
                "raw_gui_access_count": 0,
                "policy_forward_count": 0,
                "gpu_operation_count": 0,
                "development_interpretation": {
                    "value": (
                        "NO_DEVELOPMENT_EVIDENCE_FOR_"
                        "FROZEN_BASE_RESIDUAL"
                    ),
                    "primary_contrast": "v4_safe_minus_frozen_base",
                    "may_authorize_confirm20": False,
                    "may_change_v1_or_v3_verdict": False,
                    "may_support_another_v5_retune_on_fresh16": False,
                },
            }
            report["report_sha256"] = sha256_bytes(
                canonical_json_line(report)[:-1]
            )
            (output / v4_runner.DEVELOPMENT_REPORT_NAME).write_bytes(
                canonical_json_line(report)
            )
            self.assertEqual(verify_evaluation_report(output)["state_count"], 48)

            report["legacy_dev5_access_count"] = 1
            report.pop("report_sha256")
            report["report_sha256"] = sha256_bytes(
                canonical_json_line(report)[:-1]
            )
            (output / v4_runner.DEVELOPMENT_REPORT_NAME).write_bytes(
                canonical_json_line(report)
            )
            with self.assertRaisesRegex(ValueError, "report replay drifted"):
                verify_evaluation_report(output)

    def test_train_lineage_failure_precedes_every_experiment_input_read(self) -> None:
        calls = {"read": 0, "train": 0}
        args = SimpleNamespace(
            command="train-seal",
            input_dir=Path("input"),
            output_dir=Path("output"),
            repository_root=Path("repo"),
            contract="contract.json",
            source_a_git_commit="1" * 40,
            execution_b_git_commit="2" * 40,
            execution_b_freeze=v4_runner.EXECUTION_B_RUNNER_FREEZE_PATH,
        )
        with (
            patch.object(
                v4_runner,
                "_validate_execution",
                side_effect=ValueError("bad Execution-B"),
            ),
            patch.object(
                v4_runner,
                "_read",
                side_effect=lambda _path: calls.__setitem__(
                    "read", calls["read"] + 1
                ),
            ),
            patch.object(
                v4_runner,
                "train_and_seal",
                side_effect=lambda **_kwargs: calls.__setitem__(
                    "train", calls["train"] + 1
                ),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "bad Execution-B"):
                v4_runner._run(args)
        self.assertEqual(calls, {"read": 0, "train": 0})

    def test_evaluate_seal_mismatch_precedes_claim_or_input_read(self) -> None:
        calls = {"claim": 0, "read": 0, "evaluate": 0}
        validation = {
            "execution_b_git_commit": "2" * 40,
            "runner_freeze_sha256": "3" * 64,
            "contract_sha256": "4" * 64,
        }
        args = SimpleNamespace(
            command="evaluate",
            input_dir=Path("input"),
            output_dir=Path("output"),
            repository_root=Path("repo"),
            contract="contract.json",
            source_a_git_commit="1" * 40,
            execution_b_git_commit="2" * 40,
            execution_b_freeze=v4_runner.EXECUTION_B_RUNNER_FREEZE_PATH,
        )
        with (
            patch.object(v4_runner, "_validate_execution", return_value=validation),
            patch.object(
                v4_runner,
                "verify_label_blind_seal",
                return_value={
                    "source_a_git_commit": "1" * 40,
                    "execution_b_git_commit": "2" * 40,
                    "runner_freeze_sha256": "f" * 64,
                    "contract_sha256": "4" * 64,
                },
            ),
            patch.object(
                v4_runner,
                "claim_fresh_label_access",
                side_effect=lambda _path: calls.__setitem__(
                    "claim", calls["claim"] + 1
                ),
            ),
            patch.object(
                v4_runner,
                "_read",
                side_effect=lambda _path: calls.__setitem__(
                    "read", calls["read"] + 1
                ),
            ),
            patch.object(
                v4_runner,
                "evaluate_sealed_fresh16",
                side_effect=lambda **_kwargs: calls.__setitem__(
                    "evaluate", calls["evaluate"] + 1
                ),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "seal differs"):
                v4_runner._run(args)
        self.assertEqual(calls, {"claim": 0, "read": 0, "evaluate": 0})

    def test_evaluate_prevalidates_feature_before_claim_then_reads_labels(self) -> None:
        order: list[str] = []
        validation = {
            "execution_b_git_commit": "2" * 40,
            "runner_freeze_sha256": "3" * 64,
            "contract_sha256": "4" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = SimpleNamespace(
                command="evaluate",
                input_dir=Path("input"),
                output_dir=output,
                repository_root=Path("repo"),
                contract="contract.json",
                source_a_git_commit="1" * 40,
                execution_b_git_commit="2" * 40,
                execution_b_freeze=v4_runner.EXECUTION_B_RUNNER_FREEZE_PATH,
            )

            def read(path: Path) -> bytes:
                order.append(f"read:{path.name}")
                return path.name.encode()

            def evaluate(**kwargs: object) -> dict[str, bool]:
                self.assertEqual(kwargs["prevalidated_features"], ("feature",))
                order.append("evaluate")
                return {"complete": True}

            with (
                patch.object(
                    v4_runner,
                    "_validate_execution",
                    return_value=validation,
                ),
                patch.object(
                    v4_runner,
                    "verify_label_blind_seal",
                    return_value={
                        "source_a_git_commit": "1" * 40,
                        "execution_b_git_commit": "2" * 40,
                        "runner_freeze_sha256": "3" * 64,
                        "contract_sha256": "4" * 64,
                    },
                ),
                patch.object(v4_runner, "_read", side_effect=read),
                patch.object(
                    v4_runner,
                    "load_fresh_features",
                    side_effect=lambda _payload: (
                        order.append("validate_feature") or ("feature",)
                    ),
                ),
                patch.object(
                    v4_runner,
                    "claim_fresh_label_access",
                    side_effect=lambda _path: order.append("claim"),
                ),
                patch.object(
                    v4_runner,
                    "verify_fresh_label_access_claim",
                    side_effect=lambda _path: order.append("verify_claim"),
                ),
                patch.object(
                    v4_runner,
                    "evaluate_sealed_fresh16",
                    side_effect=evaluate,
                ),
            ):
                self.assertEqual(v4_runner._run(args), {"complete": True})
        self.assertEqual(
            order,
            [
                "read:fresh16-feature-states.jsonl",
                "validate_feature",
                "claim",
                "verify_claim",
                "read:fresh16-label-states.jsonl",
                "read:fresh16-historical-independent.json",
                "evaluate",
            ],
        )

    def test_validate_with_report_does_not_touch_seal_or_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / v4_runner.DEVELOPMENT_REPORT_NAME).write_bytes(b"placeholder")
            args = SimpleNamespace(command="validate", output_dir=output)
            with (
                patch.object(
                    v4_runner,
                    "verify_evaluation_report",
                    return_value={"status": "verified"},
                ) as report,
                patch.object(
                    v4_runner,
                    "verify_label_blind_seal",
                    side_effect=AssertionError("seal reread"),
                ),
                patch.object(
                    v4_runner,
                    "_read",
                    side_effect=AssertionError("input reread"),
                ),
            ):
                self.assertEqual(
                    v4_runner._run(args),
                    {"report": {"status": "verified"}},
                )
            report.assert_called_once_with(output)

    def test_method_inventory_is_frozen(self) -> None:
        self.assertEqual(
            METHODS,
            (
                "exact",
                "frozen_base",
                "v4_unguarded",
                "v4_safe",
                "historical_v1_independent",
            ),
        )


if __name__ == "__main__":
    unittest.main()
