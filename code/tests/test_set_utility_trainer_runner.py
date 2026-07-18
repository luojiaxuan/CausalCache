from __future__ import annotations

import importlib.util
import inspect
import itertools
import json
import unittest
from collections import Counter

from causalcache.set_utility_data import SetUtilityState, SubsetUtilityTarget
from causalcache.set_utility_models import SetUtilityDimensions
from causalcache.set_utility_trainer_runner import (
    EXECUTION_SCOPE,
    MODEL_FAMILIES,
    MODEL_FAMILY_REGISTRY,
    SetTransformerArchitectureConfig,
    TRAJECTORY_BALANCED_STRATEGY,
    TrainerConfig,
    TrainerEpochRecord,
    UtilityModelGeometry,
    UtilityModelConfig,
    build_checkpoint_payload,
    build_registered_utility_predictor,
    canonical_checkpoint_manifest,
    canonical_json_bytes,
    canonical_state_dict_manifest,
    infer_utility_model_geometry,
    run_cpu_utility_training,
    select_train_and_tune_states,
    select_tune_epoch,
    trajectory_balanced_epoch_batches,
)
from causalcache.set_utility_training import UtilityLossWeights


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _state(trajectory_id: str, state_id: str, event_count: int = 2) -> SetUtilityState:
    event_ids = tuple(range(1, event_count + 1))
    return SetUtilityState(
        trajectory_id=trajectory_id,
        state_id=state_id,
        event_ids=event_ids,
        maximum_labeled_cardinality=2,
        query_features=(0.1, 0.2),
        context_features=(0.3,),
        event_features=tuple((float(item), 1.0) for item in event_ids),
        pair_features=tuple(
            tuple((float(left + right),) for right in event_ids)
            for left in event_ids
        ),
        targets=tuple(
            SubsetUtilityTarget(
                subset=subset,
                raw_utility=float(sum(subset)),
            )
            for cardinality in range(min(event_count, 2) + 1)
            for subset in itertools.combinations(event_ids, cardinality)
        ),
        normalization_scale=4.0,
    )


def _config(**overrides: object) -> TrainerConfig:
    values: dict[str, object] = {
        "seed": 17,
        "epochs": 6,
        "learning_rate": 3e-4,
        "weight_decay": 1e-2,
        "batch_strategy": TRAJECTORY_BALANCED_STRATEGY,
        "trajectories_per_batch": 2,
        "states_per_trajectory_per_epoch": 3,
        "loss": UtilityLossWeights(1.0, 0.5, 0.25, 1.0),
        "maximum_gradient_norm": 1.0,
        "early_stopping_patience": 2,
        "early_stopping_min_delta": 0.1,
    }
    values.update(overrides)
    return TrainerConfig(**values)  # type: ignore[arg-type]


def _set_transformer_config(
    *,
    hidden_dimension: int = 8,
    num_heads: int = 2,
    num_layers: int = 3,
    dropout: float = 0.125,
) -> UtilityModelConfig:
    return UtilityModelConfig(
        family="set_transformer",
        hidden_dimension=hidden_dimension,
        set_transformer=SetTransformerArchitectureConfig(
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        ),
    )


def _contains_budget_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            "budget" in str(key).casefold() or _contains_budget_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_budget_key(item) for item in value)
    return False


class TrainerConfigTest(unittest.TestCase):
    def test_config_is_explicit_validated_and_budget_free(self) -> None:
        config = _config()
        self.assertEqual(config.seed, 17)
        self.assertEqual(config.loss.raw_regression, 1.0)
        expected_fields = {
            "seed",
            "epochs",
            "learning_rate",
            "weight_decay",
            "batch_strategy",
            "trajectories_per_batch",
            "states_per_trajectory_per_epoch",
            "loss",
            "maximum_gradient_norm",
            "early_stopping_patience",
            "early_stopping_min_delta",
        }
        self.assertEqual(set(TrainerConfig.__dataclass_fields__), expected_fields)
        self.assertFalse(_contains_budget_key(config.to_payload()))

        for override in (
            {"seed": -1},
            {"epochs": 0},
            {"learning_rate": 0.0},
            {"weight_decay": -0.1},
            {"batch_strategy": "random_rows"},
            {"trajectories_per_batch": 0},
            {"states_per_trajectory_per_epoch": 0},
            {"maximum_gradient_norm": 0.0},
            {"early_stopping_patience": 0},
            {"early_stopping_min_delta": -0.1},
        ):
            with self.subTest(override=override), self.assertRaises(
                (TypeError, ValueError)
            ):
                _config(**override)

    def test_training_interfaces_never_accept_a_budget(self) -> None:
        for function in (
            build_registered_utility_predictor,
            infer_utility_model_geometry,
            run_cpu_utility_training,
            select_train_and_tune_states,
            select_tune_epoch,
            trajectory_balanced_epoch_batches,
        ):
            self.assertNotIn("budget", inspect.signature(function).parameters)


class UtilityModelConfigTest(unittest.TestCase):
    def test_set_transformer_architecture_is_explicit_and_serializable(self) -> None:
        config = _set_transformer_config()
        self.assertEqual(
            config.to_payload(),
            {
                "family": "set_transformer",
                "hidden_dimension": 8,
                "architecture": {
                    "kind": "set_transformer",
                    "num_heads": 2,
                    "num_layers": 3,
                    "dropout": 0.125,
                },
            },
        )
        self.assertFalse(_contains_budget_key(config.to_payload()))
        self.assertEqual(
            json.loads(canonical_json_bytes(config.to_payload())),
            config.to_payload(),
        )
        self.assertEqual(
            UtilityModelConfig.from_payload(config.to_payload()),
            config,
        )

    def test_set_transformer_rejects_implicit_or_invalid_architecture(self) -> None:
        with self.assertRaisesRegex(TypeError, "explicit"):
            UtilityModelConfig(
                family="set_transformer",
                hidden_dimension=8,
            )
        for architecture in (
            {"num_heads": 0, "num_layers": 2, "dropout": 0.0},
            {"num_heads": 2, "num_layers": 0, "dropout": 0.0},
            {"num_heads": 2, "num_layers": 2, "dropout": -0.1},
            {"num_heads": 2, "num_layers": 2, "dropout": 1.0},
        ):
            with self.subTest(architecture=architecture), self.assertRaises(
                (TypeError, ValueError)
            ):
                SetTransformerArchitectureConfig(**architecture)
        with self.assertRaisesRegex(ValueError, "divisible"):
            _set_transformer_config(hidden_dimension=10, num_heads=4)

    def test_simple_families_reject_set_transformer_hyperparameters(self) -> None:
        architecture = SetTransformerArchitectureConfig(
            num_heads=2,
            num_layers=2,
            dropout=0.0,
        )
        for family in ("pairwise_additive", "deepsets"):
            with self.subTest(family=family), self.assertRaisesRegex(
                ValueError,
                "does not accept",
            ):
                UtilityModelConfig(
                    family=family,
                    hidden_dimension=8,
                    set_transformer=architecture,
                )
            config = UtilityModelConfig(family=family, hidden_dimension=8)
            self.assertEqual(
                config.to_payload()["architecture"],
                {"kind": family},
            )

    def test_payload_parser_rejects_unknown_or_cross_family_fields(self) -> None:
        for payload in (
            {
                "family": "deepsets",
                "hidden_dimension": 8,
                "architecture": {"kind": "deepsets"},
                "num_heads": 2,
            },
            {
                "family": "deepsets",
                "hidden_dimension": 8,
                "architecture": {
                    "kind": "deepsets",
                    "num_heads": 2,
                },
            },
            {
                "family": "set_transformer",
                "hidden_dimension": 8,
                "architecture": {
                    "kind": "set_transformer",
                    "num_heads": 2,
                    "num_layers": 2,
                },
            },
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                UtilityModelConfig.from_payload(payload)


class TrajectoryBalancedBatchTest(unittest.TestCase):
    def test_every_trajectory_contributes_equally_and_deterministically(self) -> None:
        states = (
            _state("trajectory-a", "a-1"),
            _state("trajectory-a", "a-2"),
            _state("trajectory-a", "a-3"),
            _state("trajectory-b", "b-1"),
            _state("trajectory-c", "c-1"),
            _state("trajectory-c", "c-2"),
        )
        config = _config()
        first = trajectory_balanced_epoch_batches(states, config=config, epoch=1)
        second = trajectory_balanced_epoch_batches(
            tuple(reversed(states)), config=config, epoch=1
        )
        first_ids = tuple(tuple(item.state_id for item in batch) for batch in first)
        second_ids = tuple(tuple(item.state_id for item in batch) for batch in second)
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(first), 6)
        self.assertTrue(all(1 <= len(batch) <= 2 for batch in first))
        self.assertTrue(
            all(
                len({state.trajectory_id for state in batch}) == len(batch)
                for batch in first
            )
        )
        counts = Counter(
            state.trajectory_id for batch in first for state in batch
        )
        self.assertEqual(
            counts,
            Counter({"trajectory-a": 3, "trajectory-b": 3, "trajectory-c": 3}),
        )

    def test_batching_rejects_epoch_and_inventory_drift(self) -> None:
        state = _state("trajectory-a", "state-a")
        with self.assertRaisesRegex(ValueError, "one-based"):
            trajectory_balanced_epoch_batches((state,), config=_config(), epoch=0)
        with self.assertRaisesRegex(ValueError, "unique"):
            trajectory_balanced_epoch_batches(
                (state, state), config=_config(), epoch=1
            )


class RoleFirewallTest(unittest.TestCase):
    def test_optimization_and_selection_roles_are_disjoint(self) -> None:
        states = (
            _state("new-train", "new-train-state"),
            _state("legacy", "legacy-state"),
            _state("tune", "tune-state"),
            _state("evaluation", "evaluation-state"),
            _state("phase2", "phase2-state"),
        )
        selected = select_train_and_tune_states(
            states,
            role_by_trajectory={
                "new-train": "train",
                "legacy": "legacy_train_only",
                "tune": "tune",
                "evaluation": "evaluation",
                "phase2": "phase2_evaluation",
            },
        )
        self.assertEqual(
            {state.state_id for state in selected.train_states},
            {"new-train-state", "legacy-state"},
        )
        self.assertEqual(
            tuple(state.state_id for state in selected.tune_states),
            ("tune-state",),
        )
        all_consumed = {
            state.state_id
            for state in (*selected.train_states, *selected.tune_states)
        }
        self.assertNotIn("evaluation-state", all_consumed)
        self.assertNotIn("phase2-state", all_consumed)

    def test_missing_roles_and_tune_fail_closed(self) -> None:
        states = (_state("train", "train-state"), _state("eval", "eval-state"))
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            select_train_and_tune_states(
                states,
                role_by_trajectory={"train": "train"},
            )
        with self.assertRaisesRegex(ValueError, "no tune-role"):
            select_train_and_tune_states(
                states,
                role_by_trajectory={"train": "train", "eval": "evaluation"},
            )


class ModelRegistryTest(unittest.TestCase):
    def test_registry_reserves_all_three_model_families(self) -> None:
        self.assertEqual(
            set(MODEL_FAMILIES),
            {"pairwise_additive", "deepsets", "set_transformer"},
        )
        self.assertEqual(set(MODEL_FAMILY_REGISTRY), set(MODEL_FAMILIES))

    def test_extensible_registry_builds_without_a_family_specific_runner(self) -> None:
        sentinel = object()
        observed: list[
            tuple[SetUtilityDimensions, int, UtilityModelConfig]
        ] = []

        def builder(
            dimensions: SetUtilityDimensions,
            pair_dimension: int,
            model_config: UtilityModelConfig,
        ) -> object:
            observed.append((dimensions, pair_dimension, model_config))
            return sentinel

        dimensions = SetUtilityDimensions(query=2, context=1, event=2, hidden=8)
        model_config = _set_transformer_config()
        result = build_registered_utility_predictor(
            "set_transformer",
            dimensions,
            pair_feature_dimension=1,
            model_config=model_config,
            registry={"set_transformer": builder},
        )
        self.assertIs(result, sentinel)
        self.assertEqual(observed, [(dimensions, 1, model_config)])

    def test_set_transformer_builder_respects_optional_torch_boundary(self) -> None:
        dimensions = SetUtilityDimensions(query=2, context=1, event=2, hidden=8)
        model_config = _set_transformer_config()
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            build_registered_utility_predictor("set_transformer", dimensions)
        if TORCH_AVAILABLE:
            model = build_registered_utility_predictor(
                "set_transformer",
                dimensions,
                model_config=model_config,
            )
            self.assertEqual(model.num_heads, 2)
            self.assertEqual(model.num_layers, 3)
            self.assertEqual(model.dropout, 0.125)
        else:
            with self.assertRaises(RuntimeError):
                build_registered_utility_predictor(
                    "set_transformer",
                    dimensions,
                    model_config=model_config,
                )
        with self.assertRaisesRegex(ValueError, "unknown"):
            build_registered_utility_predictor("unknown", dimensions)

    def test_builder_rejects_model_config_family_or_geometry_drift(self) -> None:
        dimensions = SetUtilityDimensions(query=2, context=1, event=2, hidden=8)
        with self.assertRaisesRegex(ValueError, "family differs"):
            build_registered_utility_predictor(
                "deepsets",
                dimensions,
                model_config=UtilityModelConfig(
                    family="pairwise_additive",
                    hidden_dimension=8,
                ),
            )
        with self.assertRaisesRegex(ValueError, "hidden dimension differs"):
            build_registered_utility_predictor(
                "deepsets",
                dimensions,
                model_config=UtilityModelConfig(
                    family="deepsets",
                    hidden_dimension=16,
                ),
            )

    def test_geometry_is_inferred_from_validated_states(self) -> None:
        geometry = infer_utility_model_geometry(
            (_state("a", "a-1"), _state("b", "b-1")),
            hidden_dimension=16,
        )
        self.assertEqual(
            geometry,
            UtilityModelGeometry(
                dimensions=SetUtilityDimensions(
                    query=2, context=1, event=2, hidden=16
                ),
                pair_feature_dimension=1,
            ),
        )


class TuneAndCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _config()
        self.states = select_train_and_tune_states(
            (_state("train", "train-state"), _state("tune", "tune-state")),
            role_by_trajectory={"train": "train", "tune": "tune"},
        )
        self.geometry = infer_utility_model_geometry(
            (*self.states.train_states, *self.states.tune_states),
            hidden_dimension=8,
        )

    def test_tune_selection_uses_min_delta_patience_and_earliest_tie(self) -> None:
        selection = select_tune_epoch(
            (3.0, 2.8, 2.79, 2.78),
            config=self.config,
        )
        self.assertEqual(selection.selected_epoch, 2)
        self.assertEqual(selection.selected_objective, 2.8)
        self.assertEqual(selection.observed_epoch_count, 4)
        self.assertTrue(selection.stopped_early)

    def test_checkpoint_helpers_are_canonical_and_budget_free(self) -> None:
        records = (
            TrainerEpochRecord(1, 1, 4.0, 3.0, True),
            TrainerEpochRecord(2, 1, 3.0, 2.8, True),
            TrainerEpochRecord(3, 1, 2.0, 2.79, False),
            TrainerEpochRecord(4, 1, 1.0, 2.78, False),
        )
        state_manifest = {
            "schema_version": "0.1.0",
            "format": "canonical_cpu_tensor_state_dict_manifest",
            "tensor_count": 1,
            "total_size_bytes": 4,
            "tensors": [],
            "state_dict_sha256": "a" * 64,
        }
        payload = build_checkpoint_payload(
            family="deepsets",
            geometry=self.geometry,
            model_config=UtilityModelConfig(
                family="deepsets",
                hidden_dimension=8,
            ),
            config=self.config,
            state_selection=self.states,
            epoch_records=records,
            selected_epoch=2,
            selected_tune_objective=2.8,
            state_dict_manifest=state_manifest,
        )
        self.assertEqual(payload["execution_scope"], EXECUTION_SCOPE)
        self.assertEqual(
            payload["model"]["config"],
            {
                "family": "deepsets",
                "hidden_dimension": 8,
                "architecture": {"kind": "deepsets"},
            },
        )
        self.assertEqual(len(payload["model"]["config_sha256"]), 64)
        self.assertEqual(payload["data_roles"]["evaluation_states_consumed"], 0)
        self.assertFalse(_contains_budget_key(payload))
        self.assertEqual(
            canonical_json_bytes(payload),
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode(),
        )
        first = canonical_checkpoint_manifest(payload, state_manifest)
        second = canonical_checkpoint_manifest(
            dict(reversed(tuple(payload.items()))),
            state_manifest,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first["manifest_sha256"]), 64)

        bad = dict(payload)
        bad["state_dict_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "differ"):
            canonical_checkpoint_manifest(bad, state_manifest)

    def test_checkpoint_rejects_non_tune_selected_epoch(self) -> None:
        records = (
            TrainerEpochRecord(1, 1, 4.0, 3.0, True),
            TrainerEpochRecord(2, 1, 3.0, 2.8, True),
            TrainerEpochRecord(3, 1, 2.0, 2.79, False),
            TrainerEpochRecord(4, 1, 1.0, 2.78, False),
        )
        with self.assertRaisesRegex(ValueError, "tune-only"):
            build_checkpoint_payload(
                family="deepsets",
                geometry=self.geometry,
                model_config=UtilityModelConfig(
                    family="deepsets",
                    hidden_dimension=8,
                ),
                config=self.config,
                state_selection=self.states,
                epoch_records=records,
                selected_epoch=1,
                selected_tune_objective=3.0,
                state_dict_manifest={"state_dict_sha256": "a" * 64},
            )

    def test_checkpoint_cannot_hide_set_transformer_defaults(self) -> None:
        records = tuple(
            TrainerEpochRecord(
                epoch,
                1,
                float(5 - epoch),
                1.0,
                epoch == 1,
            )
            for epoch in range(1, self.config.epochs + 1)
        )
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            build_checkpoint_payload(
                family="set_transformer",
                geometry=self.geometry,
                config=self.config,
                state_selection=self.states,
                epoch_records=records,
                selected_epoch=1,
                selected_tune_objective=1.0,
                state_dict_manifest={"state_dict_sha256": "a" * 64},
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional locally")
class CanonicalStateDictTorchTest(unittest.TestCase):
    def test_tensor_order_is_canonical_and_bytes_are_content_addressed(self) -> None:
        import torch

        left = {
            "z": torch.tensor([1.0, 2.0]),
            "a": torch.tensor([[3]], dtype=torch.int64),
        }
        right = {"a": left["a"].clone(), "z": left["z"].clone()}
        first = canonical_state_dict_manifest(left)
        second = canonical_state_dict_manifest(right)
        self.assertEqual(first, second)
        self.assertEqual([item["name"] for item in first["tensors"]], ["a", "z"])
        right["z"][0] = 9.0
        self.assertNotEqual(
            first["state_dict_sha256"],
            canonical_state_dict_manifest(right)["state_dict_sha256"],
        )


class SourceOnlyBoundaryTest(unittest.TestCase):
    def test_runner_source_has_no_network_hf_or_file_io_stack(self) -> None:
        import causalcache.set_utility_trainer_runner as runner

        source = inspect.getsource(runner)
        for forbidden in (
            "huggingface_hub",
            "requests",
            "urllib",
            "socket",
            "subprocess",
            "open(",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
