from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import itertools
import json
import math
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import causalcache.gate_v1_evaluation as gate_v1_evaluation
from causalcache.gate_v1_contract import derive_rosters, load_strict_json_object
from causalcache.gate_v1_data import (
    CONDITIONAL_INPUT_DIMENSION,
    INDEPENDENT_INPUT_DIMENSION,
    CandidateFeatures,
    FeatureState,
    GateState,
    LabelState,
    build_training_batch,
    feature_state_from_derived,
    join_feature_and_label_states,
    label_state_from_restoration_record,
    select_conditional,
    select_independent,
    signed_hash64,
)
from causalcache.gate_v1_evaluation import (
    canonical_report_sha256,
    evaluate_combined21_compatibility,
    evaluate_primary_slice,
    type7_quantile,
    validate_evaluation_roster,
)
from causalcache.gate_v1_provenance import (
    FORMAL_TRAIN_SOURCE_IDS_SHA256,
    GATE_V1_CONFIG_SHA256,
    ArtifactBinding,
    FrozenEnsembleProvenance,
    FrozenTrainingProvenance,
    HeuristicArtifactProvenance,
    SeedCheckpointProvenance,
    canonical_model_state_sha256,
    canonical_selection_sha256,
    frozen_ensemble_provenance_from_manifest,
    validate_frozen_ensemble,
)
from causalcache.gate_v1_training import (
    FittedEnsemble,
    OOFTrial,
    choose_learning_rate,
    run_synthetic_two_step_smoke,
    validate_formal_training_roster,
)
from causalcache.restoration_v2_2_label_table import validate_complete_distance_table
from scripts.run_gate_v1_trainer_smoke import synthetic_states


def _table(event_count: int, *, baseline: float = 1.0):
    event_ids = tuple(range(1, event_count + 1))
    weights = {event_id: 0.4 / event_id for event_id in event_ids}
    distances = {}
    for size in range(event_count + 1):
        for coalition in itertools.combinations(event_ids, size):
            utility = sum(weights[event_id] for event_id in coalition)
            distances[coalition] = baseline - utility
    return validate_complete_distance_table(event_ids, distances)


def _negative_table(event_count: int):
    event_ids = tuple(range(1, event_count + 1))
    weights = {event_id: 0.3 / event_id for event_id in event_ids}
    weights[event_ids[-1]] = -0.05
    distances = {}
    for size in range(event_count + 1):
        for coalition in itertools.combinations(event_ids, size):
            distances[coalition] = 1.0 - sum(weights[event] for event in coalition)
    return validate_complete_distance_table(event_ids, distances)


def _small_baseline_table(event_count: int):
    event_ids = tuple(range(1, event_count + 1))
    penalties = {event_id: 0.02 / event_id for event_id in event_ids}
    distances = {
        coalition: 1e-13 + sum(penalties[event] for event in coalition)
        for size in range(event_count + 1)
        for coalition in itertools.combinations(event_ids, size)
    }
    return validate_complete_distance_table(event_ids, distances)


def _state(
    source: str,
    event_count: int,
    *,
    negative: bool = False,
    small_baseline: bool = False,
) -> GateState:
    event_ids = tuple(range(1, event_count + 1))
    q64 = tuple(1.0 / 8.0 for _ in range(64))
    candidates = tuple(
        CandidateFeatures(
            event_step_id=event_id,
            h64=tuple(1.0 if index == event_id - 1 else 0.0 for index in range(64)),
            g8=(0.25, 0.5, 0.0, 0.0, 0.0, 0.1, 1.0 / 3.0, 0.5),
        )
        for event_id in event_ids
    )
    return GateState(
        source_id=source,
        state_id=f"{source}:decision_step:{event_count + 2:03d}",
        decision_step_id=event_count + 2,
        candidate_event_step_ids=event_ids,
        q64=q64,
        candidates=candidates,
        table=(
            _small_baseline_table(event_count)
            if small_baseline
            else _negative_table(event_count)
            if negative
            else _table(event_count)
        ),
    )


def _roster(count: int, *, negative: bool = False) -> tuple[GateState, ...]:
    return tuple(
        _state(f"source-{source_index:02d}", event_count, negative=negative)
        for source_index in range(count)
        for event_count in (2, 3, 4)
    )


def _roster_ids(source_ids: tuple[str, ...]) -> tuple[GateState, ...]:
    return tuple(
        _state(source_id, event_count)
        for source_id in source_ids
        for event_count in (2, 3, 4)
    )


ROOT = Path(__file__).resolve().parents[2]
ROSTERS = derive_rosters(
    load_strict_json_object(ROOT / "data/manifests/restoration_v2_selection.json"),
    load_strict_json_object(
        ROOT / "code/configs/causalcache_restoration_v2_2_label_expansion_v1.json"
    ),
)


class _FakeTensor:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = values
        self.shape = (len(values),)
        self.dtype = "torch.float32"

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def reshape(self, *_shape):
        return self

    def tolist(self):
        return list(self._values)


class _FakeModel:
    def __init__(self, marker: float) -> None:
        self.marker = marker

    def state_dict(self):
        return {"weight": _FakeTensor((self.marker, self.marker + 0.25))}


def _artifact(name: str) -> ArtifactBinding:
    return ArtifactBinding(
        repository="unit-test/causalcache",
        revision=hashlib.sha1(f"revision:{name}".encode()).hexdigest(),
        path=f"gate-v1/{name}.json",
        sha256=hashlib.sha256(f"artifact:{name}".encode()).hexdigest(),
    )


def _fake_frozen_ensemble(
    family: str,
    *,
    feature_artifact: ArtifactBinding,
    label_artifact: ArtifactBinding,
) -> tuple[FittedEnsemble, FrozenEnsembleProvenance]:
    offset = 0.0 if family == "conditional" else 100.0
    models = tuple(_FakeModel(offset + seed + 1.0) for seed in range(5))
    selected_epochs = (3, 4, 5, 6, 7)
    selection_sha256 = hashlib.sha256(f"{family}:selection".encode()).hexdigest()
    ensemble = FittedEnsemble(
        family=family,
        learning_rate=0.0003,
        seeds=tuple(range(5)),
        selected_epochs=selected_epochs,
        selection_sha256=selection_sha256,
        models=models,
    )
    training = FrozenTrainingProvenance(
        family=family,
        gate_config_sha256=GATE_V1_CONFIG_SHA256,
        formal_train_source_ids_sha256=FORMAL_TRAIN_SOURCE_IDS_SHA256,
        learning_rate=0.0003,
        selected_epochs=selected_epochs,
        oof_selection_sha256=selection_sha256,
        feature_artifact=feature_artifact,
        label_artifact=label_artifact,
        training_report_artifact=_artifact(f"{family}-training-report"),
    )
    checkpoints = tuple(
        SeedCheckpointProvenance(
            seed=seed,
            selected_epoch=selected_epochs[seed],
            model_state_sha256=canonical_model_state_sha256(model),
            checkpoint_artifact=_artifact(f"{family}-checkpoint-{seed}"),
        )
        for seed, model in enumerate(models)
    )
    return ensemble, FrozenEnsembleProvenance(
        training=training,
        checkpoints=checkpoints,
    )


class GateV1DataTest(unittest.TestCase):
    def test_signed_hash_replays_digest_bin_sign_and_normalization(self) -> None:
        observed = signed_hash64((("instruction", "  Alpha\tBETA  "),))
        expected = [0.0] * 64
        for token in ("alpha", "beta"):
            digest = hashlib.sha256(f"instruction\0{token}".encode()).digest()
            expected[int.from_bytes(digest[:8], "big") % 64] += (
                1.0 if digest[8] % 2 == 0 else -1.0
            )
        norm = math.sqrt(sum(value * value for value in expected))
        expected = tuple(value / norm for value in expected)
        self.assertEqual(observed, expected)

    def test_feature_builder_accepts_canonical_json_field_order(self) -> None:
        low = {
            "step_id": 1,
            "action_type": "click",
            "action_argument": "coordinate_bin:x1_y2",
            "foreground_app": "unknown",
            "screen_text_added": ["next"],
            "screen_text_removed": [],
            "screen_change": "low",
            "executor_result": "unknown",
        }
        trajectory = {
            "source_id": "s",
            "role": "forbidden-but-ignored",
            "instruction": "Open Settings",
            "events": [
                {
                    "step_id": step,
                    "low_fidelity_v2": {**low, "step_id": step},
                    "observation_after_path": f"obs-{step}",
                }
                for step in range(1, 4)
            ],
        }
        decision = {
            "state_id": "s:decision_step:004",
            "decision_step_id": 4,
            "history_event_step_ids": [1, 2, 3],
            "candidate_event_step_ids": [1, 2],
            "current_observation_path": "obs-3",
        }
        ocr = {
            f"obs-{step}": {"full_spatial_tokens": ["token", str(step)]}
            for step in range(1, 4)
        }
        trajectory = json.loads(json.dumps(trajectory, sort_keys=True))
        self.assertEqual(
            tuple(trajectory["events"][0]["low_fidelity_v2"]),
            tuple(sorted(low)),
        )
        feature = feature_state_from_derived(
            trajectory, decision, ocr_records_by_path=ocr
        )
        self.assertEqual(feature.candidate_event_step_ids, (1, 2))
        self.assertEqual(len(feature.q64), 64)
        self.assertEqual(len(feature.candidates[0].h64), 64)
        self.assertEqual(len(feature.candidates[0].g8), 8)
        for mutation in ("missing", "extra"):
            malformed = copy.deepcopy(trajectory)
            candidate = malformed["events"][0]["low_fidelity_v2"]
            if mutation == "missing":
                candidate.pop("action_type")
            else:
                candidate["unexpected"] = "forbidden"
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                ValueError,
                "field inventory",
            ):
                feature_state_from_derived(
                    malformed,
                    decision,
                    ocr_records_by_path=ocr,
                )

    def test_join_and_hierarchical_batches_are_exact(self) -> None:
        states = _roster(2, negative=True)
        features = tuple(
            FeatureState(
                source_id=state.source_id,
                state_id=state.state_id,
                decision_step_id=state.decision_step_id,
                candidate_event_step_ids=state.candidate_event_step_ids,
                q64=state.q64,
                candidates=state.candidates,
            )
            for state in states
        )
        labels = tuple(
            LabelState(
                source_id=state.source_id,
                state_id=state.state_id,
                decision_step_id=state.decision_step_id,
                table=state.table,
            )
            for state in states
        )
        joined = join_feature_and_label_states(
            features, labels, expected_source_ids=("source-00", "source-01")
        )
        conditional = build_training_batch(joined, family="conditional")
        independent = build_training_batch(joined, family="independent")
        self.assertEqual(len(conditional.examples), 58)
        self.assertEqual(len(independent.examples), 18)
        self.assertEqual(conditional.input_dimension, CONDITIONAL_INPUT_DIMENSION)
        self.assertEqual(independent.input_dimension, INDEPENDENT_INPUT_DIMENSION)
        self.assertAlmostEqual(conditional.regression_weight_sum, 1.0, places=12)
        self.assertAlmostEqual(independent.regression_weight_sum, 1.0, places=12)
        self.assertTrue(any(row.raw_target < 0.0 for row in conditional.examples))
        self.assertTrue(any(row.raw_target < 0.0 for row in independent.examples))

    def test_small_Dempty_excludes_only_regression_and_keeps_raw_ranking(self) -> None:
        states = tuple(
            _state("small", event_count, small_baseline=True)
            for event_count in (2, 3, 4)
        ) + tuple(_state("eligible", event_count) for event_count in (2, 3, 4))
        for family in ("conditional", "independent"):
            batch = build_training_batch(states, family=family)
            small_indices = {
                index
                for index, example in enumerate(batch.examples)
                if example.source_id == "small"
            }
            self.assertTrue(small_indices)
            self.assertTrue(
                all(
                    batch.examples[index].normalized_target is None
                    and batch.examples[index].regression_weight == 0.0
                    for index in small_indices
                )
            )
            self.assertTrue(
                any(
                    pair.left_index in small_indices
                    and pair.right_index in small_indices
                    and pair.weight > 0.0
                    for pair in batch.ranking_pairs
                )
            )
            self.assertAlmostEqual(batch.regression_weight_sum, 1.0, places=12)

    def test_synthetic_fixture_contains_conditional_interaction_targets(self) -> None:
        batch = build_training_batch(synthetic_states(), family="conditional")
        targets: dict[tuple[str, int], set[float]] = {}
        for example in batch.examples:
            targets.setdefault((example.state_id, example.event_step_id), set()).add(
                example.raw_target
            )
        self.assertTrue(any(len(values) > 1 for values in targets.values()))

    def test_label_projection_reads_only_identity_geometry_and_distances(self) -> None:
        table = _table(3)
        record = {
            "state": {
                "trajectory_id": "source",
                "state_id": "source:decision_step:005",
                "decision_step_id": 5,
                "candidate_event_step_ids": [1, 2, 3],
                "role": "must-not-enter-label-state",
            },
            "distance_rows": [
                {
                    "coalition_event_step_ids": list(row.coalition),
                    "distance_kl": row.distance,
                    "teacher_metadata": "opaque-and-ignored",
                }
                for row in table.rows
            ],
        }
        projected = label_state_from_restoration_record(record)
        self.assertEqual(projected.source_id, "source")
        self.assertEqual(projected.table, table)
        self.assertFalse(hasattr(projected, "role"))

    def test_label_projection_explicitly_selects_expansion_source_id(self) -> None:
        table = _table(2)
        record = {
            "state": {
                "source_id": "expansion-source",
                "state_id": "expansion-source:decision_step:004",
                "decision_step_id": 4,
                "candidate_event_step_ids": [1, 2],
            },
            "distance_rows": [
                {
                    "coalition": list(row.coalition),
                    "distance": row.distance,
                }
                for row in table.rows
            ],
        }
        projected = label_state_from_restoration_record(
            record,
            record_schema="expansion",
        )
        self.assertEqual(projected.source_id, "expansion-source")
        self.assertEqual(projected.table, table)
        with self.assertRaisesRegex(ValueError, "identity"):
            label_state_from_restoration_record(record)

    def test_label_projection_never_falls_back_between_identity_schemas(self) -> None:
        table = _table(2)
        rows = [
            {
                "coalition_event_step_ids": list(row.coalition),
                "distance_kl": row.distance,
            }
            for row in table.rows
        ]
        legacy_only = {
            "state": {
                "trajectory_id": "legacy-source",
                "state_id": "legacy-source:decision_step:004",
                "decision_step_id": 4,
                "candidate_event_step_ids": [1, 2],
            },
            "distance_rows": rows,
        }
        with self.assertRaisesRegex(ValueError, "identity"):
            label_state_from_restoration_record(
                legacy_only,
                record_schema="expansion",
            )
        for invalid_record_schema in ("automatic", None, []):
            with self.subTest(
                invalid_record_schema=invalid_record_schema
            ), self.assertRaisesRegex(ValueError, "unsupported"):
                label_state_from_restoration_record(
                    legacy_only,
                    record_schema=invalid_record_schema,
                )

    def test_label_projection_rejects_dual_identity_schemas(self) -> None:
        table = _table(2)
        record = {
            "state": {
                "trajectory_id": "legacy-source",
                "source_id": "expansion-source",
                "state_id": "legacy-source:decision_step:004",
                "decision_step_id": 4,
                "candidate_event_step_ids": [1, 2],
            },
            "distance_rows": [
                {
                    "coalition_event_step_ids": list(row.coalition),
                    "distance_kl": row.distance,
                }
                for row in table.rows
            ],
        }
        for record_schema in ("legacy", "expansion"):
            with self.subTest(record_schema=record_schema), self.assertRaisesRegex(
                ValueError,
                "mixes identity schemas",
            ):
                label_state_from_restoration_record(
                    record,
                    record_schema=record_schema,
                )
        record["state"]["source_id"] = "legacy-source"
        with self.assertRaisesRegex(ValueError, "mixes identity schemas"):
            label_state_from_restoration_record(record)
        record["distance_rows"] = [
            {
                "coalition": list(row.coalition),
                "distance": row.distance,
            }
            for row in table.rows
        ]
        with self.assertRaisesRegex(ValueError, "mixes identity schemas"):
            label_state_from_restoration_record(
                record,
                record_schema="expansion",
            )

    def test_label_projection_rejects_mixed_distance_row_schemas(self) -> None:
        table = _table(2)
        record = {
            "state": {
                "trajectory_id": "source",
                "source_id": "source",
                "state_id": "source:decision_step:004",
                "decision_step_id": 4,
                "candidate_event_step_ids": [1, 2],
            },
            "distance_rows": [
                {
                    "coalition_event_step_ids": list(row.coalition),
                    "distance_kl": row.distance,
                    "coalition": list(row.coalition),
                }
                for row in table.rows
            ],
        }
        with self.assertRaisesRegex(ValueError, "mixes"):
            label_state_from_restoration_record(record)
        record["distance_rows"] = [
            {
                "coalition": list(row.coalition),
                "distance": row.distance,
                "distance_kl": row.distance,
            }
            for row in table.rows
        ]
        with self.assertRaisesRegex(ValueError, "mixes"):
            label_state_from_restoration_record(
                record,
                record_schema="expansion",
            )
        record["distance_rows"] = [
            {
                "coalition_event_step_ids": list(row.coalition),
                "distance_kl": row.distance,
            }
            for row in table.rows
        ]
        with self.assertRaisesRegex(ValueError, "mixes"):
            label_state_from_restoration_record(
                record,
                record_schema="expansion",
            )

    def test_selectors_rescore_stop_and_tie_break(self) -> None:
        state = _state("source", 4)
        calls = []

        def conditional(_state, event, coalition):
            calls.append((event, coalition))
            if not coalition:
                return {1: 2.0, 2: 2.0, 3: 1.0, 4: 0.0}[event]
            return 3.0 if event == 3 else -1.0

        selected, trace = select_conditional(state, conditional)
        self.assertEqual(selected, (1, 3))
        self.assertEqual(trace[0][0], 1)
        self.assertTrue(any(coalition == (1,) for _, coalition in calls))
        self.assertEqual(
            select_conditional(state, lambda *_: 0.0)[0],
            (),
        )
        independent_calls = []
        selected_static = select_independent(
            state,
            lambda _state, event, coalition: independent_calls.append(
                (event, coalition)
            )
            or float(event),
        )
        self.assertEqual(selected_static, (3, 4))
        self.assertEqual(len(independent_calls), 4)
        self.assertTrue(all(not coalition for _, coalition in independent_calls))


class GateV1TrainingScheduleTest(unittest.TestCase):
    def test_formal_roster_replays_exact_frozen_folds_before_torch(self) -> None:
        source_ids = ROSTERS["formal_train"]
        folds = validate_formal_training_roster(
            _roster_ids(source_ids), source_ids
        )
        self.assertEqual(tuple(len(fold) for fold in folds), (12, 12, 12, 11, 11))
        with self.assertRaises(ValueError):
            validate_formal_training_roster(
                _roster_ids(tuple(reversed(source_ids))),
                source_ids,
            )

    def test_formal_roster_rejects_state_shuffle_duplicate_and_geometry_drift(self) -> None:
        source_ids = ROSTERS["formal_train"]
        states = _roster_ids(source_ids)
        shuffled = tuple(
            state
            for source_id in source_ids
            for state in reversed(
                tuple(item for item in states if item.source_id == source_id)
            )
        )
        with self.assertRaises(ValueError):
            validate_formal_training_roster(shuffled, source_ids)
        duplicate = list(states)
        duplicate[1] = replace(duplicate[1], state_id=duplicate[0].state_id)
        with self.assertRaises(ValueError):
            validate_formal_training_roster(tuple(duplicate), source_ids)
        drifted = list(states)
        drifted[0] = replace(drifted[0], candidate_event_step_ids=(2, 3))
        with self.assertRaises(ValueError):
            validate_formal_training_roster(tuple(drifted), source_ids)

    def test_cli_fixture_is_synthetic_bounded_and_has_negative_targets(self) -> None:
        states = synthetic_states()
        self.assertEqual(len(states), 6)
        self.assertEqual(
            tuple(dict.fromkeys(state.source_id for state in states)),
            ("synthetic-00", "synthetic-01"),
        )
        for family in ("conditional", "independent"):
            batch = build_training_batch(states, family=family)
            self.assertTrue(any(row.raw_target < 0.0 for row in batch.examples))

    def test_learning_rate_selection_uses_five_seed_mean_and_low_lr_tie(self) -> None:
        trials = []
        for learning_rate, value in ((0.0003, 0.8), (0.001, 0.80005)):
            metric_history = (0.0, 0.0, value) + (value,) * 50
            trials.extend(
                OOFTrial(
                    family="conditional",
                    learning_rate=learning_rate,
                    seed=seed,
                    selected_epoch=3,
                    best_raw_utility_ratio=value,
                    epochs_run=len(metric_history),
                    metric_by_epoch=metric_history,
                )
                for seed in range(5)
            )
        selected = choose_learning_rate(trials, family="conditional")
        self.assertEqual(selected.learning_rate, 0.0003)
        self.assertEqual(tuple(trial.seed for trial in selected.trials), tuple(range(5)))
        self.assertEqual(len(selected.grid_trials), 10)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional locally")
    def test_synthetic_two_step_smoke(self) -> None:
        result = run_synthetic_two_step_smoke(synthetic_states())
        self.assertEqual(
            result["status"], "VALID_GATE_V1_SYNTHETIC_TWO_STEP_SMOKE"
        )
        self.assertEqual(result["paper_metric_count"], 0)
        self.assertEqual(result["persistent_checkpoint_count"], 0)
        self.assertTrue(result["families"]["conditional"]["interaction_target_exercised"])


class GateV1ProvenanceTest(unittest.TestCase):
    def test_artifacts_require_immutable_revision(self) -> None:
        artifact = replace(_artifact("valid"), revision="main")
        with self.assertRaises(ValueError):
            artifact.validate()

    def test_frozen_ensemble_rejects_duplicate_model_objects(self) -> None:
        feature = _artifact("train-feature")
        label = _artifact("train-label")
        ensemble, provenance = _fake_frozen_ensemble(
            "conditional",
            feature_artifact=feature,
            label_artifact=label,
        )
        self.assertEqual(validate_frozen_ensemble(ensemble, provenance), provenance.sha256)
        duplicate = replace(
            ensemble,
            models=(ensemble.models[0],) * 5,
        )
        with self.assertRaises(ValueError):
            validate_frozen_ensemble(duplicate, provenance)

    def test_ensemble_manifest_roundtrip_and_schema_drift_fail_closed(self) -> None:
        ensemble, provenance = _fake_frozen_ensemble(
            "conditional",
            feature_artifact=_artifact("manifest-feature"),
            label_artifact=_artifact("manifest-label"),
        )
        parsed = frozen_ensemble_provenance_from_manifest(provenance.to_payload())
        self.assertEqual(parsed, provenance)
        self.assertEqual(validate_frozen_ensemble(ensemble, parsed), provenance.sha256)
        tampered = copy.deepcopy(provenance.to_payload())
        tampered["unexpected"] = 1
        with self.assertRaises(ValueError):
            frozen_ensemble_provenance_from_manifest(tampered)

    def test_formal_api_has_no_callable_or_bootstrap_override(self) -> None:
        parameters = inspect.signature(evaluate_primary_slice).parameters
        self.assertNotIn("conditional_scores", parameters)
        self.assertNotIn("independent_scores", parameters)
        self.assertNotIn("bootstrap_resamples", parameters)
        combined_parameters = inspect.signature(
            evaluate_combined21_compatibility
        ).parameters
        self.assertNotIn("conditional_scores", combined_parameters)
        self.assertNotIn("independent_scores", combined_parameters)


class GateV1EvaluationTest(unittest.TestCase):
    @staticmethod
    def _conditional_score(_state, event, _coalition):
        return {1: 10.0, 2: 9.0, 3: 1.0, 4: 0.5}.get(event, -1.0)

    @staticmethod
    def _independent_score(_state, event, _coalition):
        return float(event)

    def test_fresh16_go_statistics_and_combined_guard(self) -> None:
        primary_ids = ROSTERS["fresh_development"]
        primary_states = _roster_ids(primary_ids)
        heuristics = {
            name: {state.state_id: () for state in primary_states}
            for name in ("dynamic_recent", "ocr_rgb_v2", "policy_vision_v3")
        }
        heuristic_provenance = tuple(
            HeuristicArtifactProvenance(
                name=name,
                artifact=_artifact(f"{name}-fresh16"),
                selection_sha256=canonical_selection_sha256(heuristics[name]),
            )
            for name in ("dynamic_recent", "ocr_rgb_v2", "policy_vision_v3")
        )
        train_feature = _artifact("formal58-feature")
        train_label = _artifact("formal58-label")
        conditional_ensemble, conditional_provenance = _fake_frozen_ensemble(
            "conditional",
            feature_artifact=train_feature,
            label_artifact=train_label,
        )
        independent_ensemble, independent_provenance = _fake_frozen_ensemble(
            "independent",
            feature_artifact=train_feature,
            label_artifact=train_label,
        )
        with patch(
            "causalcache.gate_v1_evaluation.model_score",
            side_effect=lambda _model, family: (
                self._conditional_score
                if family == "conditional"
                else self._independent_score
            ),
        ):
            primary = evaluate_primary_slice(
                primary_states,
                expected_source_ids=primary_ids,
                conditional_ensemble=conditional_ensemble,
                conditional_provenance=conditional_provenance,
                independent_ensemble=independent_ensemble,
                independent_provenance=independent_provenance,
                heuristics=heuristics,
                evaluation_feature_artifact=_artifact("fresh16-feature"),
                evaluation_label_artifact=_artifact("fresh16-label"),
                heuristic_provenance=heuristic_provenance,
            )
        self.assertTrue(primary["go_selector"])
        self.assertTrue(primary["go_set_conditioning_primary"])
        self.assertEqual(primary["bootstrap"]["resamples"], 10_000)
        self.assertEqual(
            primary["provenance"]["conditional_ensemble_sha256"],
            conditional_provenance.sha256,
        )
        unbound = dict(primary)
        digest = unbound.pop("report_sha256")
        self.assertEqual(digest, canonical_report_sha256(unbound))
        nested_tamper = copy.deepcopy(primary)
        nested_tamper.pop("report_sha256")
        nested_tamper["provenance"]["conditional_ensemble"]["checkpoints"][0][
            "model_state_sha256"
        ] = "f" * 64
        nested_tamper["report_sha256"] = canonical_report_sha256(nested_tamper)
        with self.assertRaises(ValueError):
            gate_v1_evaluation._validate_frozen_primary_report(nested_tamper)

        combined_ids = ROSTERS["combined_development"]
        combined_states = _roster_ids(combined_ids)
        with patch(
            "causalcache.gate_v1_evaluation.model_score",
            side_effect=lambda _model, family: (
                self._conditional_score
                if family == "conditional"
                else self._independent_score
            ),
        ):
            combined = evaluate_combined21_compatibility(
                primary,
                combined_states,
                expected_source_ids=combined_ids,
                conditional_ensemble=conditional_ensemble,
                conditional_provenance=conditional_provenance,
                independent_ensemble=independent_ensemble,
                independent_provenance=independent_provenance,
                evaluation_feature_artifact=_artifact("combined21-feature"),
                evaluation_label_artifact=_artifact("combined21-label"),
            )
        self.assertTrue(combined["go_to_post_go_contract"])
        self.assertFalse(combined["confirm_access_authorized"])
        self.assertEqual(combined["combined21_normalized_excluded_state_count"], 0)

        alternate_checkpoints = list(independent_provenance.checkpoints)
        alternate_checkpoints[0] = replace(
            alternate_checkpoints[0],
            checkpoint_artifact=_artifact("independent-checkpoint-0-alternate"),
        )
        alternate_provenance = replace(
            independent_provenance,
            checkpoints=tuple(alternate_checkpoints),
        )
        with self.assertRaises(ValueError):
            evaluate_combined21_compatibility(
                primary,
                combined_states,
                expected_source_ids=combined_ids,
                conditional_ensemble=conditional_ensemble,
                conditional_provenance=conditional_provenance,
                independent_ensemble=independent_ensemble,
                independent_provenance=alternate_provenance,
                evaluation_feature_artifact=_artifact("combined21-feature"),
                evaluation_label_artifact=_artifact("combined21-label"),
            )

    def test_tampered_primary_cannot_unlock_compatibility(self) -> None:
        report = {
            "status": "FROZEN_GATE_V1_FRESH16_PRIMARY_EVALUATION",
            "go_selector": True,
            "go_set_conditioning_primary": True,
            "combined21_compatibility_evaluated": False,
        }
        report["report_sha256"] = canonical_report_sha256(report)
        report["go_selector"] = False
        with self.assertRaises(ValueError):
            gate_v1_evaluation._validate_frozen_primary_report(report)

    def test_evaluation_roster_rejects_noncanonical_flat_state_order(self) -> None:
        source_ids = ROSTERS["fresh_development"]
        states = _roster_ids(source_ids)
        shuffled = tuple(
            state
            for source_id in source_ids
            for state in reversed(
                tuple(item for item in states if item.source_id == source_id)
            )
        )
        with self.assertRaises(ValueError):
            validate_evaluation_roster(
                shuffled,
                source_ids,
                frozen_source_ids_sha256=(
                    gate_v1_evaluation.FRESH_DEVELOPMENT_IDS_SHA256
                ),
            )

    def test_type7_quantile(self) -> None:
        self.assertEqual(type7_quantile([0.0, 10.0], 0.25), 2.5)


if __name__ == "__main__":
    unittest.main()
