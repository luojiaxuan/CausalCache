from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from causalcache import gate_v1_formal_train as formal
from causalcache import gate_v1_training as training
from causalcache.gate_v1_contract import derive_rosters
from causalcache.gate_v1_training import (
    FittedEnsemble,
    OOFTrial,
    choose_learning_rate,
)


ROOT = Path(__file__).resolve().parents[2]


def _formal_source_ids() -> tuple[str, ...]:
    legacy = json.loads(
        (ROOT / "data/manifests/restoration_v2_selection.json").read_text()
    )
    expansion = json.loads(
        (
            ROOT
            / "code/configs/causalcache_restoration_v2_2_label_expansion_v1.json"
        ).read_text()
    )
    return derive_rosters(legacy, expansion)["formal_train"]


def _selection(family: str, *, selected_learning_rate: float = 0.0003):
    trials = []
    for learning_rate in (0.0003, 0.001):
        value = 0.8 if learning_rate == selected_learning_rate else 0.7
        history = (value,) * 51
        trials.extend(
            OOFTrial(
                family=family,
                learning_rate=learning_rate,
                seed=seed,
                selected_epoch=1,
                best_raw_utility_ratio=value,
                epochs_run=51,
                metric_by_epoch=history,
            )
            for seed in range(5)
        )
    return choose_learning_rate(trials, family=family)


def _ensemble(family: str, selection) -> FittedEnsemble:
    return FittedEnsemble(
        family=family,
        learning_rate=selection.learning_rate,
        seeds=tuple(range(5)),
        selected_epochs=tuple(trial.selected_epoch for trial in selection.trials),
        selection_sha256=training.family_selection_sha256(selection),
        models=tuple(object() for _ in range(5)),
    )


def _checkpoints(family: str) -> tuple[formal.CheckpointArtifact, ...]:
    return tuple(
        formal.CheckpointArtifact(
            family=family,
            seed=seed,
            selected_epoch=1,
            model_state_sha256=hashlib.sha256(
                f"{family}:state:{seed}".encode()
            ).hexdigest(),
            checkpoint_sha256=hashlib.sha256(
                f"{family}:checkpoint:{seed}".encode()
            ).hexdigest(),
            payload=f"{family}:{seed}".encode(),
        )
        for seed in range(5)
    )


class FormalTrainInputTest(unittest.TestCase):
    def test_repaired_cache_loader_binds_bytes_roster_and_join_audit(self) -> None:
        feature = b"feature-cache"
        label = b"label-cache"
        source_ids = _formal_source_ids()
        folds = tuple(
            tuple(source_ids[index::5]) for index in range(5)
        )
        states = tuple(object() for _ in range(174))
        with (
            patch.object(
                formal, "FEATURE_CACHE_SHA256", hashlib.sha256(feature).hexdigest()
            ),
            patch.object(
                formal, "LABEL_CACHE_SHA256", hashlib.sha256(label).hexdigest()
            ),
            patch.object(formal, "read_feature_cache", return_value=("feature",)),
            patch.object(formal, "read_label_cache", return_value=("label",)),
            patch.object(
                formal,
                "join_feature_and_label_states",
                return_value=states,
            ) as join,
            patch.object(
                formal,
                "validate_formal_training_roster",
                return_value=folds,
            ),
        ):
            inputs = formal.load_repaired_formal_training_inputs(
                feature,
                label,
                expected_source_ids=source_ids,
                join_audit_sha256=formal.FORMAL_JOIN_AUDIT_SHA256,
            )
        self.assertEqual(inputs.states, states)
        self.assertEqual(inputs.source_ids, source_ids)
        self.assertEqual(tuple(len(fold) for fold in inputs.folds), (12, 12, 12, 11, 11))
        join.assert_called_once_with(
            ("feature",),
            ("label",),
            expected_source_ids=source_ids,
        )

    def test_repaired_cache_loader_rejects_before_semantic_decode(self) -> None:
        with (
            patch.object(formal, "read_feature_cache") as read_feature,
            patch.object(formal, "read_label_cache") as read_label,
            self.assertRaises(ValueError),
        ):
            formal.load_repaired_formal_training_inputs(
                b"wrong-feature",
                b"wrong-label",
                expected_source_ids=_formal_source_ids(),
                join_audit_sha256=formal.FORMAL_JOIN_AUDIT_SHA256,
            )
        read_feature.assert_not_called()
        read_label.assert_not_called()


class FormalTrainScheduleTest(unittest.TestCase):
    def test_operation_counts_replay_complete_grid_and_formula(self) -> None:
        conditional = _selection("conditional")
        independent = _selection("independent")
        counts = formal._combined_operation_counts((conditional, independent))
        self.assertEqual(counts["oof_trial_count"], 20)
        self.assertEqual(counts["fold_training_track_count"], 100)
        self.assertEqual(counts["final_fit_track_count"], 10)
        self.assertEqual(counts["training_model_initialization_count"], 110)
        self.assertEqual(counts["model_initialization_count"], 110)
        self.assertEqual(counts["checkpoint_count"], 10)
        self.assertEqual(counts["full_oof_report_count"], 2)
        self.assertEqual(counts["oof_epoch_metric_count"], 1020)
        self.assertEqual(counts["oof_optimizer_step_count"], 5100)
        self.assertEqual(counts["final_fit_optimizer_step_count"], 10)
        self.assertEqual(counts["optimizer_step_count"], 5110)

    def test_orchestrator_runs_families_in_order_and_builds_full_reports(self) -> None:
        source_ids = _formal_source_ids()
        folds = tuple(tuple(source_ids[index::5]) for index in range(5))
        inputs = formal.FormalTrainingInputs(
            states=tuple(object() for _ in range(174)),
            source_ids=source_ids,
            folds=folds,
            feature_cache_sha256=formal.FEATURE_CACHE_SHA256,
            label_cache_sha256=formal.LABEL_CACHE_SHA256,
            join_audit_sha256=formal.FORMAL_JOIN_AUDIT_SHA256,
        )
        conditional = _selection("conditional")
        independent = _selection("independent")
        conditional_ensemble = _ensemble("conditional", conditional)
        independent_ensemble = _ensemble("independent", independent)
        checkpoint_groups = (
            _checkpoints("conditional"),
            _checkpoints("independent"),
        )
        with (
            patch.object(
                formal,
                "validate_formal_training_roster",
                return_value=folds,
            ),
            patch.object(
                formal,
                "run_formal_oof",
                side_effect=(conditional, independent),
            ) as run_oof,
            patch.object(
                formal,
                "fit_final_ensemble",
                side_effect=(conditional_ensemble, independent_ensemble),
            ) as fit,
            patch.object(
                formal,
                "_checkpoint_artifacts",
                side_effect=checkpoint_groups,
            ),
        ):
            result = formal.run_formal_training(inputs)
        self.assertEqual(
            [call.kwargs["family"] for call in run_oof.call_args_list],
            ["conditional", "independent"],
        )
        self.assertEqual(fit.call_count, 2)
        self.assertEqual(len(result.checkpoints), 10)
        conditional_grid = result.family_reports["conditional"]["oof"][
            "complete_grid"
        ]["grid_trials"]
        self.assertEqual(len(conditional_grid), 10)
        self.assertEqual(result.operation_counts["fold_training_track_count"], 100)
        self.assertEqual(result.run_manifest["confirm_access_count"], 0)
        self.assertEqual(len(result.run_manifest_sha256), 64)


class DeterministicRuntimeTest(unittest.TestCase):
    def test_configure_cpu_sets_both_thread_pools(self) -> None:
        calls = []
        fake = SimpleNamespace(
            set_num_threads=lambda value: calls.append(("intra", value)),
            set_num_interop_threads=lambda value: calls.append(("interop", value)),
            get_num_interop_threads=lambda: 1,
            manual_seed=lambda value: calls.append(("seed", value)),
            use_deterministic_algorithms=lambda value: calls.append(
                ("deterministic", value)
            ),
        )
        with patch.object(training, "_torch", return_value=fake):
            self.assertIs(training.configure_deterministic_cpu(3), fake)
        self.assertEqual(
            calls,
            [
                ("intra", 1),
                ("interop", 1),
                ("seed", 3),
                ("deterministic", True),
            ],
        )

    def test_optimizer_flags_are_explicit(self) -> None:
        captured = {}

        def adamw(parameters, **kwargs):
            captured["parameters"] = parameters
            captured.update(kwargs)
            return "optimizer"

        fake = SimpleNamespace(optim=SimpleNamespace(AdamW=adamw))
        model = SimpleNamespace(parameters=lambda: ("parameter",))
        with patch.object(training, "_torch", return_value=fake):
            self.assertEqual(
                training.build_optimizer(model, learning_rate=0.0003),
                "optimizer",
            )
        self.assertFalse(captured["foreach"])
        self.assertFalse(captured["fused"])
        self.assertFalse(captured["maximize"])
        self.assertFalse(captured["capturable"])
        self.assertFalse(captured["differentiable"])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
        "PyTorch and safetensors are optional locally",
    )
    def test_safetensors_checkpoint_exact_model_replay(self) -> None:
        model = training.build_model("conditional", seed=2)
        self.assertEqual(model[1].approximate, "none")
        self.assertEqual(model[3].approximate, "none")
        payload = formal.serialize_safetensors_checkpoint(model)
        model_state_sha256 = formal.canonical_model_state_sha256(model)
        loaded = formal.load_safetensors_checkpoint(
            payload,
            family="conditional",
            seed=2,
            expected_model_state_sha256=model_state_sha256,
            expected_checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
        )
        self.assertEqual(
            formal.canonical_model_state_sha256(loaded),
            model_state_sha256,
        )
        self.assertEqual(formal.serialize_safetensors_checkpoint(loaded), payload)


if __name__ == "__main__":
    unittest.main()
