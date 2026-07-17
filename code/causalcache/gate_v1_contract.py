"""Fail-closed source contract for the first learned CausalCache gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.1.0"
PROTOCOL_ID = "causalcache_gate_v1_preregistration"
STATUS = "VALID_CAUSALCACHE_GATE_V1_PREREGISTRATION"
CANONICAL_CONFIG_PATH = "code/configs/causalcache_gate_v1_preregistration.json"
LEGACY_SELECTION_PATH = "data/manifests/restoration_v2_selection.json"
LEGACY_SELECTION_SHA256 = (
    "292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b"
)
EXPANSION_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_label_expansion_v1.json"
)
EXPANSION_CONFIG_SHA256 = (
    "e74904fb75cd72429e6adb51ab7500adcaa1915fb6cebc8567a2945bc9373947"
)
OOF_SALT = "causalcache-gate-v1-oof"

EXPECTED_SECTION_SHA256 = {
    "access_firewall": "7ad4a8806162507e75faa8607a5bee119058f2163e816dc6a299ce7bc6a58740",
    "label_contract": "542cba96cc7bd1149ef4d249ca40d61cb1ff099489af2a2b15053780c8a08f5c",
    "feature_contract": "6459120a1f28efe0aec9462d17e58882c94cba84cc291e9b9d771d8a579fb82c",
    "model_contract": "a6cf42e7ad9a3c5b59c8aa14a4acca79500167f57e846597462f24031a3627a9",
    "weighting_contract": "755826abeff37eb8c5f1c7a5f9b1f5a27139a7898c23e77df4240bf16a1c43b8",
    "loss_contract": "b25a4df8b2824a788a0d2c6dc021b841a30a1edb31af2079723e70ff5a9284d1",
    "optimization_contract": "693571166cf4fec33971213507b45e087ce393718e9469785973ac76cde67514",
    "oof_contract": "b1cac5b740aed26a0c8de31416cd19b68f6e381ddf3e0844f52d3f5bd981bc3e",
    "inference_contract": "4e70c96b5f0a8608e47dc9649cb6ac5ffbe5e305cc646db4fe07b3cdefbab374",
    "current_trainer_smoke": "c60229cfda20ff9bc70007eb2a915dfb790ad3ac8f0064b24f113bac374f9fa7",
    "evaluation_contract": "3923965f8564334ee19e7d3503ec985aecd56130a0bd938e44590fb9e28b7911",
    "prohibited_work": "679dfecccd3e439911f4eb16760e788d49fc807c2822b910151219c0fb2acc92",
}

EXPECTED_ROSTER_DIGESTS = {
    "legacy_train": "97c026f4e6e48b0f748c8495c967c0c796af256ee78fdf2e81b530f54be9cb35",
    "fresh_train_expansion": "11904e1d102b69561b402bea4e42044be9572884f6168597c5f07c3ca346cb34",
    "formal_train": "ccec55afb0882e602f5d2c83ec420682488e2663fe26a7c574a851825c43f225",
    "legacy_development": "0e3d67d79ac45392122fd9a712af43ea544445420e7fdb06df22bad7d84b05e6",
    "fresh_development": "1c37cbf6b67b0ddee61b3efe27d33b30c8fbfb471ced12f12e49624a10b82454",
    "combined_development": "ac75b14490254f59b93ebbc66e15d11201a3d241f3cc59e5c34f4cf0d3b59b0c",
    "sealed_confirm": "c84ba8b9b705abc7ba7d6d7230b868fd4600e3a835b1c763a8aaf1f93feb052e",
}

EXPECTED_FOLD_DIGESTS = (
    "9915d8f296dae456ece574c71eebd87c4350e9b9e3b9e8d9f4d0e9327a8b2fc1",
    "70888ea8de1555f4deab1cf13eff7a5636bd57f20b6661a3e97bf455492c2d56",
    "ea6d3a3176e32af059ffa2318229d5a947ada59dc312c1871e9899f655c4d1a3",
    "bfb489b086d8840802771b8461d1e14ad966c65adc5105edec21be7f72045ba1",
    "61194d993373471bff009bc9f588ca97d73d5fe6c927ee9233ce708e6540d467",
)
EXPECTED_FOLD_SIZES = (12, 12, 12, 11, 11)
EXPECTED_ASSIGNMENT_DIGEST = (
    "df2703ee1af10f9a5fec6504ec9d17f26d13ed249f2b79083fcb0db62f0276bd"
)

TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "preregistration_status",
    "lineage",
    "rosters",
    *EXPECTED_SECTION_SHA256,
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_strict_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a sequence")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys drifted: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} drifted: expected {expected!r}, got {actual!r}")


def _id_digest(source_ids: Sequence[str]) -> str:
    return sha256_bytes(canonical_json_bytes(list(source_ids)))


def _source_ids_from_legacy_role(
    selection: Mapping[str, Any], role: str
) -> tuple[str, ...]:
    try:
        records = selection["roles"][role]["trajectories"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"legacy role {role} is missing") from exc
    sequence = _sequence(records, f"legacy role {role} trajectories")
    ids = tuple(str(_mapping(record, "trajectory")["source_id"]) for record in sequence)
    if len(ids) != len(set(ids)):
        raise ValueError(f"legacy role {role} contains duplicate source IDs")
    return ids


def _source_ids_from_expansion(
    expansion: Mapping[str, Any], split: str
) -> tuple[str, ...]:
    try:
        section = _mapping(expansion["selection"][split], f"expansion {split}")
        values = _sequence(section["source_ids"], f"expansion {split} source IDs")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"expansion split {split} is missing") from exc
    ids = tuple(str(value) for value in values)
    if len(ids) != len(set(ids)):
        raise ValueError(f"expansion split {split} contains duplicate source IDs")
    _equal(section["trajectory_count"], len(ids), f"{split} trajectory count")
    _equal(section["source_ids_sha256"], _id_digest(ids), f"{split} ID digest")
    return ids


def derive_rosters(
    legacy: Mapping[str, Any], expansion: Mapping[str, Any]
) -> dict[str, tuple[str, ...]]:
    legacy_train = _source_ids_from_legacy_role(legacy, "v2_label_train")
    legacy_development = _source_ids_from_legacy_role(legacy, "v2_development")
    sealed_confirm = _source_ids_from_legacy_role(legacy, "v2_confirm_primary")
    fresh_train = _source_ids_from_expansion(expansion, "gate_train_expansion")
    fresh_development = _source_ids_from_expansion(
        expansion, "gate_development_expansion"
    )
    return {
        "legacy_train": legacy_train,
        "fresh_train_expansion": fresh_train,
        "formal_train": legacy_train + fresh_train,
        "legacy_development": legacy_development,
        "fresh_development": fresh_development,
        "combined_development": legacy_development + fresh_development,
        "sealed_confirm": sealed_confirm,
    }


def derive_oof_folds(
    formal_train_ids: Sequence[str], *, fold_count: int = 5
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, ...], ...]]:
    ordered = sorted(
        formal_train_ids,
        key=lambda source_id: (
            sha256_bytes(f"{OOF_SALT}\0{source_id}".encode("utf-8")),
            source_id,
        ),
    )
    assignments = tuple(
        (source_id, rank % fold_count) for rank, source_id in enumerate(ordered)
    )
    folds = tuple(
        tuple(source_id for source_id, fold in assignments if fold == fold_index)
        for fold_index in range(fold_count)
    )
    return assignments, folds


def mlp_parameter_count(input_dimension: int, hidden_dimension: int) -> int:
    return (
        input_dimension * hidden_dimension
        + hidden_dimension
        + hidden_dimension * hidden_dimension
        + hidden_dimension
        + hidden_dimension
        + 1
    )


def hierarchical_weight_audit(trajectory_count: int) -> dict[str, Any]:
    if type(trajectory_count) is not int or trajectory_count <= 0:
        raise ValueError("trajectory_count must be a positive integer")
    total = Fraction(0, 1)
    edge_count = 0
    cells = []
    for candidate_count in (2, 3, 4):
        for prefix_cardinality in (0, 1):
            coalition_count = math.comb(candidate_count, prefix_cardinality)
            candidates_per_coalition = candidate_count - prefix_cardinality
            cell_edge_count = coalition_count * candidates_per_coalition
            edge_weight = Fraction(
                1,
                trajectory_count
                * 3
                * 2
                * coalition_count
                * candidates_per_coalition,
            )
            total += trajectory_count * cell_edge_count * edge_weight
            edge_count += trajectory_count * cell_edge_count
            cells.append(
                {
                    "candidate_event_count": candidate_count,
                    "prefix_cardinality": prefix_cardinality,
                    "coalition_count_per_trajectory_state": coalition_count,
                    "candidates_per_coalition": candidates_per_coalition,
                    "edge_count": trajectory_count * cell_edge_count,
                    "edge_weight_numerator": edge_weight.numerator,
                    "edge_weight_denominator": edge_weight.denominator,
                }
            )
    return {
        "trajectory_count": trajectory_count,
        "edge_count": edge_count,
        "weight_sum_numerator": total.numerator,
        "weight_sum_denominator": total.denominator,
        "cells": cells,
    }


def independent_weight_audit(trajectory_count: int) -> dict[str, Any]:
    if type(trajectory_count) is not int or trajectory_count <= 0:
        raise ValueError("trajectory_count must be a positive integer")
    total = Fraction(0, 1)
    target_count = 0
    cells = []
    for candidate_count in (2, 3, 4):
        target_weight = Fraction(1, trajectory_count * 3 * candidate_count)
        cell_target_count = trajectory_count * candidate_count
        total += cell_target_count * target_weight
        target_count += cell_target_count
        cells.append(
            {
                "candidate_event_count": candidate_count,
                "target_count": cell_target_count,
                "target_weight_numerator": target_weight.numerator,
                "target_weight_denominator": target_weight.denominator,
            }
        )
    return {
        "trajectory_count": trajectory_count,
        "target_count": target_count,
        "weight_sum_numerator": total.numerator,
        "weight_sum_denominator": total.denominator,
        "cells": cells,
    }


def _validate_static_sections(config: Mapping[str, Any]) -> None:
    for section, expected_digest in EXPECTED_SECTION_SHA256.items():
        value = config.get(section)
        if not isinstance(value, Mapping):
            raise ValueError(f"{section} must be a mapping")
        observed = sha256_bytes(canonical_json_bytes(value))
        _equal(observed, expected_digest, f"{section} canonical digest")


def _validate_lineage(
    config: Mapping[str, Any], *, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    lineage = _mapping(config["lineage"], "lineage")
    _exact_keys(
        lineage,
        {"legacy_selection", "label_expansion_split", "legacy_labels"},
        "lineage",
    )
    legacy_binding = _mapping(lineage["legacy_selection"], "legacy selection")
    expansion_binding = _mapping(
        lineage["label_expansion_split"], "label expansion split"
    )
    _equal(
        legacy_binding,
        {"path": LEGACY_SELECTION_PATH, "sha256": LEGACY_SELECTION_SHA256},
        "legacy selection binding",
    )
    _equal(
        expansion_binding,
        {"path": EXPANSION_CONFIG_PATH, "sha256": EXPANSION_CONFIG_SHA256},
        "label expansion binding",
    )
    _equal(
        lineage["legacy_labels"],
        {
            "repo": "gavinlaw/causalcache-restoration-labels-mobile",
            "revision": "8f6baae5c0b23b08915fa1b0fb848dd519b4c8db",
            "path": "raw/v2.2-eager-train-dev-exact-v2.tar",
            "sha256": "99120d5444d31962d9f4254c3e40bc5f06d3e4d3d90a74b1749e7ccd7aefb29e",
        },
        "legacy label binding",
    )
    legacy_path = repository_root / LEGACY_SELECTION_PATH
    expansion_path = repository_root / EXPANSION_CONFIG_PATH
    for path, expected, label in (
        (legacy_path, LEGACY_SELECTION_SHA256, "legacy selection"),
        (expansion_path, EXPANSION_CONFIG_SHA256, "label expansion split"),
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{label} must be a regular non-symlink file")
        _equal(sha256_file(path), expected, f"{label} file SHA256")
    return load_strict_json_object(legacy_path), load_strict_json_object(expansion_path)


def _validate_rosters(
    config: Mapping[str, Any], derived: Mapping[str, tuple[str, ...]]
) -> None:
    rosters = _mapping(config["rosters"], "rosters")
    _exact_keys(
        rosters,
        {
            *EXPECTED_ROSTER_DIGESTS,
            "pairwise_disjoint",
        },
        "rosters",
    )
    expected_counts = {
        "legacy_train": (10, 30, 290),
        "fresh_train_expansion": (48, 144, 1392),
        "formal_train": (58, 174, 1682),
        "legacy_development": (5, 15, 145),
        "fresh_development": (16, 48, 464),
        "combined_development": (21, 63, 609),
    }
    for name, expected_digest in EXPECTED_ROSTER_DIGESTS.items():
        section = _mapping(rosters[name], f"roster {name}")
        ids = derived[name]
        _equal(len(ids), section["trajectory_count"], f"{name} trajectory count")
        _equal(_id_digest(ids), expected_digest, f"{name} derived digest")
        _equal(section["source_ids_sha256"], expected_digest, f"{name} config digest")
        if name in expected_counts:
            trajectories, states, edges = expected_counts[name]
            _equal(
                (
                    section["trajectory_count"],
                    section["state_count"],
                    section["conditional_edge_count"],
                ),
                (trajectories, states, edges),
                f"{name} geometry counts",
            )
        else:
            _equal(set(section), {"trajectory_count", "source_ids_sha256"}, name)
    _equal(
        rosters["formal_train"]["composition"],
        ["legacy_train", "fresh_train_expansion"],
        "formal train composition",
    )
    _equal(
        rosters["combined_development"]["composition"],
        ["legacy_development", "fresh_development"],
        "development composition",
    )
    _equal(
        rosters["pairwise_disjoint"],
        ["formal_train", "combined_development", "sealed_confirm"],
        "disjoint roster declaration",
    )
    disjoint_names = tuple(rosters["pairwise_disjoint"])
    for left_index, left_name in enumerate(disjoint_names):
        for right_name in disjoint_names[left_index + 1 :]:
            if set(derived[left_name]) & set(derived[right_name]):
                raise ValueError(f"{left_name} and {right_name} overlap")


def _validate_models(config: Mapping[str, Any]) -> None:
    models = _mapping(config["model_contract"], "model contract")
    for name, expected_dimension, expected_hidden, expected_parameters in (
        ("conditional", 330, 64, 25409),
        ("independent", 200, 88, 25609),
    ):
        model = _mapping(models[name], f"{name} model")
        components = _sequence(model["input_components"], f"{name} components")
        component_dimension = sum(int(_sequence(item, "component")[1]) for item in components)
        _equal(component_dimension, expected_dimension, f"{name} component dimension")
        _equal(model["input_dimension"], expected_dimension, f"{name} input dimension")
        _equal(model["hidden_dimension"], expected_hidden, f"{name} hidden dimension")
        computed = mlp_parameter_count(expected_dimension, expected_hidden)
        _equal(computed, expected_parameters, f"{name} computed parameter count")
        _equal(model["parameter_count"], computed, f"{name} parameter count")
    features = _mapping(config["feature_contract"], "feature contract")
    _equal(features["q64"]["dimension"], 64, "q64 dimension")
    _equal(features["h64"]["dimension"], 64, "h64 dimension")
    _equal(features["g8"]["dimension"], 8, "g8 dimension")
    _equal(
        len(features["h64"]["low_fidelity_fields_in_exact_order"]),
        8,
        "low-fidelity field count",
    )


def _validate_weighting(
    config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    weights = _mapping(config["weighting_contract"], "weighting contract")
    expected_per_trajectory = {
        "n2_k0": 2,
        "n2_k1": 2,
        "n3_k0": 3,
        "n3_k1": 6,
        "n4_k0": 4,
        "n4_k1": 12,
        "total": 29,
    }
    _equal(
        weights["per_trajectory_edge_counts"],
        expected_per_trajectory,
        "per-trajectory edge counts",
    )
    audits = {
        name: hierarchical_weight_audit(count)
        for name, count in {
            "legacy_train": 10,
            "fresh_train_expansion": 48,
            "formal_train": 58,
            "legacy_development": 5,
            "fresh_development": 16,
            "combined_development": 21,
        }.items()
    }
    expected_edges = {
        "legacy_train": 290,
        "fresh_train_expansion": 1392,
        "formal_train": 1682,
        "legacy_development": 145,
        "fresh_development": 464,
        "combined_development": 609,
    }
    for name, audit in audits.items():
        _equal(audit["edge_count"], expected_edges[name], f"{name} weighted edge count")
        _equal(
            (audit["weight_sum_numerator"], audit["weight_sum_denominator"]),
            (1, 1),
            f"{name} weight sum",
        )
    independent_audits = {
        name: independent_weight_audit(count)
        for name, count in {
            "legacy_train": 10,
            "fresh_train_expansion": 48,
            "formal_train": 58,
            "legacy_development": 5,
            "fresh_development": 16,
            "combined_development": 21,
        }.items()
    }
    for name, audit in independent_audits.items():
        _equal(
            audit["target_count"],
            audit["trajectory_count"] * 9,
            f"{name} independent target count",
        )
        _equal(
            (audit["weight_sum_numerator"], audit["weight_sum_denominator"]),
            (1, 1),
            f"{name} independent weight sum",
        )
    return audits, independent_audits


def _validate_oof(
    config: Mapping[str, Any], formal_train_ids: Sequence[str]
) -> tuple[tuple[str, ...], ...]:
    oof = _mapping(config["oof_contract"], "OOF contract")
    assignments, folds = derive_oof_folds(formal_train_ids)
    _equal(tuple(len(fold) for fold in folds), EXPECTED_FOLD_SIZES, "OOF fold sizes")
    observed_digests = tuple(_id_digest(fold) for fold in folds)
    _equal(observed_digests, EXPECTED_FOLD_DIGESTS, "OOF fold digests")
    assignment_payload = [
        {"source_id": source_id, "fold": fold} for source_id, fold in assignments
    ]
    observed_assignment_digest = sha256_bytes(canonical_json_bytes(assignment_payload))
    _equal(observed_assignment_digest, EXPECTED_ASSIGNMENT_DIGEST, "OOF assignment digest")
    _equal(tuple(oof["fold_sizes"]), EXPECTED_FOLD_SIZES, "config OOF fold sizes")
    _equal(
        tuple(oof["fold_source_ids_sha256"]),
        EXPECTED_FOLD_DIGESTS,
        "config OOF fold digests",
    )
    _equal(
        oof["assignment_table_sha256"],
        EXPECTED_ASSIGNMENT_DIGEST,
        "config OOF assignment digest",
    )
    if set(source_id for fold in folds for source_id in fold) != set(formal_train_ids):
        raise ValueError("OOF folds do not cover exactly the formal train roster")
    return folds


def _validate_thresholds(config: Mapping[str, Any]) -> None:
    evaluation = _mapping(config["evaluation_contract"], "evaluation contract")
    selector = _mapping(evaluation["go_selector_all"], "GO_SELECTOR")
    conditioning = _mapping(
        evaluation["go_set_conditioning_all"], "GO_SET_CONDITIONING"
    )
    _equal(
        selector,
        {
            "ensemble_normalized_recovery_over_exact_minimum": 0.8,
            "ensemble_raw_utility_over_exact_raw_minimum": 0.8,
            "mean_normalized_delta_vs_each_heuristic_minimum": 0.05,
            "strongest_heuristic_positive_trajectory_count_minimum": 12,
            "strongest_heuristic_paired_bootstrap_lower_strictly_greater_than": 0.0,
            "individual_seed_exact_ratio_minimum": 0.75,
            "individual_seed_pass_count_minimum": 4,
            "seed_mean_recovery_population_std_maximum": 0.08,
            "hard_n3_n4_equal_trajectory_raw_utility_ratio_minimum": 0.7,
            "selected_true_nonpositive_addition_rate_maximum": 0.1,
        },
        "GO_SELECTOR thresholds",
    )
    _equal(
        conditioning,
        {
            "conditional_minus_independent_mean_normalized_delta_minimum": 0.02,
            "conditional_minus_independent_raw_utility_delta_strictly_greater_than": 0.0,
            "conditional_minus_independent_positive_trajectory_count_minimum": 12,
            "conditional_minus_independent_paired_bootstrap_lower_strictly_greater_than": 0.0,
            "paired_seed_positive_count_minimum": 4,
            "combined21_conditional_minus_independent_mean_normalized_delta_minimum": 0.0,
        },
        "GO_SET_CONDITIONING thresholds",
    )
    _equal(
        evaluation["bootstrap"],
        {
            "unit": "trajectory",
            "resamples": 10000,
            "seed": 271828,
            "confidence": 0.9,
            "interval": "percentile",
            "quantile": "Hyndman_Fan_type_7",
        },
        "bootstrap contract",
    )


def validate_gate_v1_config(
    config: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    _exact_keys(config, TOP_LEVEL_KEYS, "gate-v1 config")
    _equal(config["schema_version"], SCHEMA_VERSION, "schema version")
    _equal(config["protocol_id"], PROTOCOL_ID, "protocol ID")
    _equal(
        config["preregistration_status"],
        "frozen_before_any_label_expansion_policy_output",
        "preregistration status",
    )
    _validate_static_sections(config)
    legacy, expansion = _validate_lineage(config, repository_root=root)
    rosters = derive_rosters(legacy, expansion)
    _validate_rosters(config, rosters)
    _validate_models(config)
    weight_audits, independent_weight_audits = _validate_weighting(config)
    folds = _validate_oof(config, rosters["formal_train"])
    _validate_thresholds(config)
    return {
        "status": STATUS,
        "protocol_id": PROTOCOL_ID,
        "formal_train_trajectory_count": len(rosters["formal_train"]),
        "formal_train_conditional_edge_count": weight_audits["formal_train"][
            "edge_count"
        ],
        "combined_development_trajectory_count": len(
            rosters["combined_development"]
        ),
        "combined_development_conditional_edge_count": weight_audits[
            "combined_development"
        ]["edge_count"],
        "fresh_development_trajectory_count": len(rosters["fresh_development"]),
        "fresh_development_conditional_edge_count": weight_audits[
            "fresh_development"
        ]["edge_count"],
        "formal_train_independent_target_count": independent_weight_audits[
            "formal_train"
        ]["target_count"],
        "combined_development_independent_target_count": independent_weight_audits[
            "combined_development"
        ]["target_count"],
        "fresh_development_independent_target_count": independent_weight_audits[
            "fresh_development"
        ]["target_count"],
        "sealed_confirm_trajectory_count": len(rosters["sealed_confirm"]),
        "oof_fold_sizes": [len(fold) for fold in folds],
        "oof_fold_source_ids_sha256": [_id_digest(fold) for fold in folds],
        "conditional_parameter_count": mlp_parameter_count(330, 64),
        "independent_parameter_count": mlp_parameter_count(200, 88),
        "hierarchical_weight_sums": {
            name: f"{audit['weight_sum_numerator']}/{audit['weight_sum_denominator']}"
            for name, audit in weight_audits.items()
        },
        "independent_weight_sums": {
            name: f"{audit['weight_sum_numerator']}/{audit['weight_sum_denominator']}"
            for name, audit in independent_weight_audits.items()
        },
        "fresh_development_is_only_formal_go_slice": True,
        "confirm_access_authorized": False,
        "matched_nll_authorized": False,
        "closed_loop_authorized": False,
    }


def validate_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    resolved = resolved.resolve()
    canonical = (root / CANONICAL_CONFIG_PATH).resolve()
    if resolved != canonical or not resolved.is_file() or resolved.is_symlink():
        raise ValueError("gate-v1 config must be the canonical regular non-symlink file")
    config = load_strict_json_object(resolved)
    result = validate_gate_v1_config(config, repository_root=root)
    result["config_path"] = CANONICAL_CONFIG_PATH
    result["config_sha256"] = sha256_file(resolved)
    return result
