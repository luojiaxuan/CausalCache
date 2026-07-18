from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from causalcache.long_horizon_contract import LongHorizonContract
from causalcache.long_horizon_execution import (
    EXPECTED_RANK_RANGES,
    PROTOCOL_ID,
    RUNNER_STATUS,
    SCHEMA_VERSION,
    SELECTOR_PREPARATION_MANIFEST_PATH,
    SELECTOR_SEAL_PATH,
    CanonicalLongHorizonRuntimeAdapter,
    aggregate_worker_outputs,
    build_worker_assignments,
    canonical_json_bytes,
    evaluate_scientific_go,
    inventory_sha256,
    pretty_json_bytes,
    run_worker_shard,
    sha256_bytes,
    validate_runtime_trajectory_record,
    validate_runner_config,
    validate_selector_seal_payload,
)
from causalcache.long_horizon_data import (
    MANIFEST_RELATIVE_PATH as SUBSTRATE_MANIFEST_RELATIVE_PATH,
)
from causalcache.long_horizon_prepare import (
    BUILD_CLI_PATH,
    MANIFEST_RELATIVE_PATH,
    MODULE_PATH,
    SELECTION_RELATIVE_PATH,
    VALIDATOR_CLI_PATH,
    LoadedPreparationInputs,
    ScoredSelectionSeal,
    _manifest as build_preparation_manifest,
)


PAIR_SPECS = (
    (2, ("restoration_independent_gate", "recent")),
    (2, ("restoration_independent_gate", "ocr_rgb_v2")),
    (2, ("v1_conditional", "restoration_independent_gate")),
    (2, ("v4_safe_frozen_base_residual", "restoration_independent_gate")),
    (4, ("restoration_independent_gate", "recent")),
    (4, ("restoration_independent_gate", "ocr_rgb_v2")),
)


def _contract(tmp_path: Path) -> LongHorizonContract:
    budget_2 = [
        "restoration_independent_gate",
        "recent",
        "ocr_rgb_v2",
        "v1_conditional",
        "v4_safe_frozen_base_residual",
        "random",
        "summary_only",
    ]
    budget_4 = [
        "restoration_independent_gate",
        "recent",
        "ocr_rgb_v2",
        "random",
        "summary_only",
    ]
    source_a_paths = sorted(
        (
            "code/configs/causalcache_long_horizon_development_v1.json",
            MODULE_PATH,
            BUILD_CLI_PATH,
            VALIDATOR_CLI_PATH,
        )
    )
    formal_manifests = [
        {
            "family": family,
            "path": f"formal/{family}-ensemble-manifest.json",
            "sha256": f"{index + 1:064x}",
        }
        for index, family in enumerate(("conditional", "independent"))
    ]
    formal_checkpoints = [
        {
            "family": family,
            "seed": seed,
            "path": f"formal/{family}-seed-{seed}.safetensors",
            "sha256": f"{index + 3:064x}",
        }
        for index, (family, seed) in enumerate(
            (family, seed)
            for family in ("conditional", "independent")
            for seed in range(5)
        )
    ]
    residual_checkpoints = [
        {
            "seed": seed,
            "path": f"residual-checkpoints/seed-{seed}.safetensors",
            "sha256": f"{index + 20:064x}",
        }
        for index, seed in enumerate(range(5))
    ]
    data = {
        "protocol_id": "causalcache_long_horizon_development_v1",
        "source_freeze": {
            "branch": "luojiaxuan/independent-gate-long-horizon-development",
            "required_source_a_paths": source_a_paths,
        },
        "selector_matrix": {
            "budget_2": budget_2,
            "budget_4": budget_4,
            "random_seed_roster": [271828],
        },
        "reference_contract": {
            "n16": {
                "budget_2_comparator_pairs": [list(pair) for _, pair in PAIR_SPECS[:4]],
                "budget_4_comparator_pairs": [list(pair) for _, pair in PAIR_SPECS[4:]],
            }
        },
        "operation_accounting": {
            "n8_total_requested_canonical_action_parse_attempts": 48,
            "n8_budget_4_unique_subset_rows_per_valid_trajectory": 163,
            "n8_budget_4_total_requested_unique_subset_rows": 3912,
            "n16_total_requested_pair_reference_parse_attempts": 288,
            "n16_maximum_nonself_distance_rows_per_valid_pair": 3,
            "n16_total_requested_maximum_nonself_distances": 432,
            "maximum_generation_attempt_count": 336,
            "maximum_teacher_forward_count": 4512,
            "maximum_gpu_kl_count": 4344,
        },
        "validity_and_go_contract": {
            "policy_coverage_thresholds": {
                "minimum_valid_n8_trajectories": 18,
                "minimum_valid_n16_trajectories_per_comparator_pair": 18,
                "failure_outcome": "NO_GO_INSUFFICIENT_LONG_HORIZON_POLICY_COVERAGE",
            },
            "fixed_denominator_reduction": {
                "selected_trajectory_denominator": 24,
                "invalid_state_contribution": 0.0,
                "n8_paired_delta_zero_fill": True,
                "n16_pair_delta_zero_fill": True,
                "n8_exact_utility_ratio_zero_fill": True,
                "zero_exact_oracle_utility_ratio": 0.0,
                "bootstrap_uses_all_selected_trajectories": True,
                "positive_support_uses_all_selected_trajectories": True,
                "valid_state_missing_or_duplicate_selector_outcome": "INVALID_LONG_HORIZON_SELECTOR_JOIN",
            },
            "set_aware_vs_independent_development_go": {
                "applies_to_budget": 2,
                "set_aware_selectors": [
                    "v1_conditional",
                    "v4_safe_frozen_base_residual",
                ],
                "reference_selector": "restoration_independent_gate",
                "n8_trajectory_equal_mean_normalized_delta_minimum": 0.01,
                "n8_paired_trajectory_bootstrap_confidence": 0.9,
                "n8_paired_trajectory_bootstrap_iterations": 100,
                "n8_paired_trajectory_bootstrap_seed": 20260718,
                "n8_minimum_positive_trajectories": 14,
                "n8_fixed_selected_trajectory_denominator": 24,
                "overall_set_aware_go_requires_at_least_one_selector_to_pass_all_its_conditions": True,
                "failure_outcome": "NO_GO_SET_AWARE_LONG_HORIZON_DEVELOPMENT_V1",
            },
            "independent_long_horizon_development_go": {
                "primary_budget": 4,
                "n8_trajectory_equal_mean_raw_utility_ratio_to_exact_minimum": 0.85,
                "n8_comparators": ["recent", "ocr_rgb_v2"],
                "n8_paired_trajectory_bootstrap_confidence": 0.9,
                "n8_paired_trajectory_bootstrap_iterations": 100,
                "n8_paired_trajectory_bootstrap_seed": 20260718,
                "n8_per_comparator_minimum_positive_trajectories": 14,
                "n8_fixed_selected_trajectory_denominator": 24,
                "pass_outcome": "AUTHORIZE_NEW_CLOSED_LOOP_SOURCE_A_ONLY",
                "failure_outcome": "NO_GO_LONG_HORIZON_INDEPENDENT_DEVELOPMENT_V1",
            },
        },
        "policy_context_profile": {
            "maximum_context_tokens": 32768,
            "processor_preflight_reserved_completion_tokens": 256,
        },
        "artifact_plan": {
            "repo": "gavinlaw/causalcache-long-horizon-development-mobile"
        },
        "learned_model_artifacts": {
            "formal58_base_and_conditional": {
                "repo": "owner/formal58",
                "repo_type": "model",
                "private": True,
                "revision": "1" * 40,
                "ensemble_manifests": formal_manifests,
                "checkpoints": formal_checkpoints,
            },
            "v4_safe_frozen_base_residual": {
                "repo": "owner/v4",
                "repo_type": "model",
                "private": True,
                "revision": "2" * 40,
                "manifest_path": "manifest.json",
                "manifest_sha256": "3" * 64,
                "label_blind_seal_path": "label-blind-seal.json",
                "label_blind_seal_sha256": "4" * 64,
                "residual_checkpoints": residual_checkpoints,
            },
        },
    }
    return LongHorizonContract(
        data=data,
        sha256="a" * 64,
        repository_root=tmp_path,
        source_path=tmp_path / "contract.json",
    )


def _runner() -> dict:
    return {
        "worker_topology": {
            "host_alias": "hyper00",
            "worker_count": 4,
            "gpu_assignments": [
                {
                    "worker_index": index,
                    "host_gpu_id": str(index),
                    "container_cuda_ordinal": index,
                }
                for index in range(4)
            ],
            "selection_rank_ranges": [list(item) for item in EXPECTED_RANK_RANGES],
            "trajectories_per_worker": 6,
            "one_explicit_gpu_per_worker": True,
        }
    }


def _source_inventory(contract: LongHorizonContract) -> list[dict]:
    records = []
    for index, path in enumerate(contract.data["source_freeze"]["required_source_a_paths"]):
        records.append(
            {
                "path": path,
                "sha256": (
                    contract.sha256
                    if path
                    == "code/configs/causalcache_long_horizon_development_v1.json"
                    else f"{index + 40:064x}"
                ),
                "size_bytes": 1000 + index,
            }
        )
    return records


def _model_provenance(contract: LongHorizonContract) -> dict:
    formal = contract.data["learned_model_artifacts"][
        "formal58_base_and_conditional"
    ]
    residual = contract.data["learned_model_artifacts"][
        "v4_safe_frozen_base_residual"
    ]
    formal_records = tuple(formal["ensemble_manifests"]) + tuple(
        formal["checkpoints"]
    )
    residual_hashes = {
        record["path"]: record["sha256"]
        for record in residual["residual_checkpoints"]
    }
    residual_hashes[residual["manifest_path"]] = residual["manifest_sha256"]
    residual_hashes[residual["label_blind_seal_path"]] = residual[
        "label_blind_seal_sha256"
    ]
    residual_hashes["residual-model-metadata.json"] = "5" * 64
    return {
        "formal58": {
            "repo": formal["repo"],
            "repo_type": formal["repo_type"],
            "private": formal["private"],
            "revision": formal["revision"],
            "files": [
                {
                    "path": record["path"],
                    "sha256": record["sha256"],
                    "size_bytes": 2000 + index,
                }
                for index, record in enumerate(formal_records)
            ],
            "family_seed_roster": {
                "conditional": [0, 1, 2, 3, 4],
                "independent": [0, 1, 2, 3, 4],
            },
            "checkpoint_load_count": 10,
        },
        "v4_frozen_base_residual": {
            "repo": residual["repo"],
            "repo_type": residual["repo_type"],
            "private": residual["private"],
            "revision": residual["revision"],
            "files": [
                {
                    "path": path,
                    "sha256": residual_hashes[path],
                    "size_bytes": 3000 + index,
                }
                for index, path in enumerate(
                    (
                        residual["manifest_path"],
                        residual["label_blind_seal_path"],
                        *sorted(
                            {
                                "residual-model-metadata.json",
                                *(record["path"] for record in residual["residual_checkpoints"]),
                            }
                        ),
                    )
                )
            ],
            "residual_seed_roster": [0, 1, 2, 3, 4],
            "training_allowed": False,
        },
    }


def _runner_validation_fixture(tmp_path: Path) -> tuple[dict, bytes, bytes, bytes]:
    contract = _contract(tmp_path)
    source_commit = "a" * 40
    selection_commit = "b" * 40
    substrate_revision = "c" * 40
    preparation_revision = "d" * 40
    inventory = _source_inventory(contract)

    selection = _selection()
    selection["generator"] = {"git_revision": source_commit}
    selection_payload = pretty_json_bytes(selection)
    selector_seal = _selector_seal()
    selector_seal_payload = canonical_json_bytes(selector_seal) + b"\n"
    selector_seal_sha256 = sha256_bytes(selector_seal_payload)
    substrate_binding = {
        "repo": contract.data["artifact_plan"]["repo"],
        "repo_type": "dataset",
        "revision": substrate_revision,
        "path": SUBSTRATE_MANIFEST_RELATIVE_PATH,
        "sha256": "5" * 64,
        "size_bytes": 5000,
    }
    seal_binding = {
        "repo": contract.data["artifact_plan"]["repo"],
        "repo_type": "dataset",
        "revision": preparation_revision,
        "path": SELECTOR_SEAL_PATH,
        "sha256": selector_seal_sha256,
        "size_bytes": len(selector_seal_payload),
    }
    inventory_by_path = {record["path"]: record for record in inventory}
    states = tuple(
        SimpleNamespace(candidate_event_step_ids=tuple(range(1, count + 1)))
        for _source_index in range(24)
        for count in (8, 16)
    )
    loaded = LoadedPreparationInputs(
        states=states,
        ocr_rgb_by_state=MappingProxyType({}),
        formal_ensembles=object(),
        v4_ensemble=object(),
        contract_provenance=MappingProxyType(
            {
                "protocol_id": contract.data["protocol_id"],
                "source_a": {
                    "git_commit": source_commit,
                    "config": {
                        "path": "code/configs/causalcache_long_horizon_development_v1.json",
                        "sha256": contract.sha256,
                        "size_bytes": inventory_by_path[
                            "code/configs/causalcache_long_horizon_development_v1.json"
                        ]["size_bytes"],
                    },
                },
                "selection_freeze": {
                    "git_commit": selection_commit,
                    "manifest": {
                        "path": "data/manifests/causalcache_long_horizon_development_v1_selection.json",
                        "sha256": sha256_bytes(selection_payload),
                        "size_bytes": len(selection_payload),
                    },
                },
            }
        ),
        substrate_provenance=MappingProxyType(
            {
                "repo": substrate_binding["repo"],
                "repo_type": substrate_binding["repo_type"],
                "revision": substrate_binding["revision"],
                "manifest": dict(substrate_binding),
            }
        ),
        model_provenance=MappingProxyType(_model_provenance(contract)),
    )
    scored = ScoredSelectionSeal(
        seal=SimpleNamespace(
            sha256=selector_seal_sha256,
            payload_bytes=selector_seal_payload,
            records=tuple(selector_seal["records"]),
        ),
        latency_by_arm=(),
    )
    generator = {
        "git_revision": selection_commit,
        "source_files": [
            dict(inventory_by_path[path])
            for path in (MODULE_PATH, BUILD_CLI_PATH, VALIDATOR_CLI_PATH)
        ],
    }
    preparation_manifest = build_preparation_manifest(
        loaded=loaded,
        scored=scored,
        random_seed=271828,
        selector_matrix={
            2: contract.data["selector_matrix"]["budget_2"],
            4: contract.data["selector_matrix"]["budget_4"],
        },
        generator=generator,
    )
    preparation_payload = pretty_json_bytes(preparation_manifest)
    runner = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": RUNNER_STATUS,
        "source_a": {
            "branch": contract.data["source_freeze"]["branch"],
            "commit": source_commit,
            "remote_ref_commit": source_commit,
            "contract_path": "code/configs/causalcache_long_horizon_development_v1.json",
            "contract_sha256": contract.sha256,
            "required_inventory": inventory,
            "required_inventory_sha256": inventory_sha256(inventory),
        },
        "selection_manifest": {
            "path": "data/manifests/causalcache_long_horizon_development_v1_selection.json",
            "sha256": sha256_bytes(selection_payload),
            "freeze_commit": selection_commit,
            "freeze_parent_source_a_commit": source_commit,
            "size_bytes": len(selection_payload),
        },
        "substrate": substrate_binding,
        "selector_preparation_manifest": {
            "repo_id": contract.data["artifact_plan"]["repo"],
            "revision": preparation_revision,
            "path": SELECTOR_PREPARATION_MANIFEST_PATH,
            "sha256": sha256_bytes(preparation_payload),
            "size_bytes": len(preparation_payload),
        },
        "selector_seal": seal_binding,
        "worker_topology": _runner()["worker_topology"],
        "access_firewall": {
            "reserve_access_count": 0,
            "top_up_count": 0,
            "replacement_count": 0,
            "post_selection_filter_count": 0,
            "selector_reseal_count": 0,
            "threshold_update_count": 0,
            "old_confirm_access_count": 0,
            "closed_loop_access_count": 0,
            "sealed_test_access_count": 0,
        },
        "operation_accounting": contract.data["operation_accounting"],
        "authorization": {
            "development_policy_access": True,
            "development_restoration_access": True,
            "selector_sets_already_label_blind_sealed": True,
            "reserve_access": False,
            "old_confirm_access": False,
            "closed_loop_access": False,
            "sealed_test_access": False,
        },
    }
    return runner, selection_payload, selector_seal_payload, preparation_payload


def _selection() -> dict:
    return {
        "splits": {
            "development": {
                "trajectories": [
                    {"trajectory_id": f"source-{rank:02d}"} for rank in range(24)
                ]
            }
        }
    }


def _trajectories() -> tuple[dict, ...]:
    return tuple(
        {"source_id": f"source-{rank:02d}", "role": "development"}
        for rank in range(24)
    )


def _selector_seal() -> dict:
    matrix = {
        2: _contract(Path(".")).data["selector_matrix"]["budget_2"],
        4: _contract(Path(".")).data["selector_matrix"]["budget_4"],
    }
    records = []
    for rank in range(24):
        source_id = f"source-{rank:02d}"
        for step, count in ((10, 8), (18, 16)):
            state_id = f"{source_id}:decision_step:{step:03d}"
            for budget, selectors in matrix.items():
                for selector in selectors:
                    records.append(
                        {
                            "source_id": source_id,
                            "state_id": state_id,
                            "decision_step_id": step,
                            "candidate_event_step_ids": list(range(1, count + 1)),
                            "budget_event_capacity": budget,
                            "selector_name": selector,
                            "selected_event_step_ids": (
                                []
                                if selector == "summary_only"
                                else [3, 4]
                                if selector in {"recent", "ocr_rgb_v2"}
                                else [5, 6]
                                if selector
                                in {"v1_conditional", "v4_safe_frozen_base_residual"}
                                else [1, 2]
                            ),
                        }
                    )
    records.sort(
        key=lambda item: (
            item["budget_event_capacity"],
            item["selector_name"],
            item["state_id"],
        )
    )
    return {
        "schema_version": "1.0.0",
        "protocol_id": "causalcache_long_horizon_label_blind_selection",
        "status": "SEALED_LONG_HORIZON_LABEL_BLIND_SELECTIONS_V1",
        "selector_names": sorted(set(matrix[2]) | set(matrix[4])),
        "budgets": [2, 4],
        "selector_names_by_budget": {
            str(budget): sorted(selectors) for budget, selectors in matrix.items()
        },
        "state_count": 48,
        "record_count": len(records),
        "records": records,
    }


def _count(requested: int, completed: int | None = None) -> dict:
    done = requested if completed is None else completed
    return {"requested": requested, "completed": done, "missing": requested - done}


def _runtime_record(source_id: str, *, valid: bool = True) -> dict:
    n8_completed = 163 if valid else 7
    n8 = {
        "valid": valid,
        "state_id": f"{source_id}:decision_step:010",
        "candidate_event_step_ids": list(range(1, 9)),
        "distance_rows": [
            {
                "restored_event_step_ids": list(coalition),
                "distance": 100.0 - sum(coalition),
            }
            for size in range(5)
            for coalition in itertools.combinations(range(1, 9), size)
        ],
        "full_reference_distance": 0.0,
        "operation_counts": {
            "processor_preflights": _count(164),
            "canonical_action_parse_attempts": _count(2),
            "reference_teacher_forwards": _count(1),
            "candidate_teacher_forwards": _count(163, n8_completed),
            "distance_rows": _count(163, n8_completed),
            "analytic_self_distance_rows": _count(1),
        },
    }
    pairs = []
    selector_seal_sha256 = sha256_bytes(
        canonical_json_bytes(_selector_seal()) + b"\n"
    )
    selected_by_name = {
        "restoration_independent_gate": [1, 2],
        "recent": [3, 4],
        "ocr_rgb_v2": [3, 4],
        "v1_conditional": [5, 6],
        "v4_safe_frozen_base_residual": [5, 6],
    }
    for budget, (left, right) in PAIR_SPECS:
        left_selected = selected_by_name[left]
        right_selected = selected_by_name[right]
        reference = sorted(set(left_selected) | set(right_selected))
        pairs.append(
            {
                "valid": valid,
                "budget_event_capacity": budget,
                "left_selector_name": left,
                "right_selector_name": right,
                "left_selected_event_step_ids": left_selected,
                "right_selected_event_step_ids": right_selected,
                "selection_seal_sha256": selector_seal_sha256,
                "state_id": f"{source_id}:decision_step:018",
                "candidate_event_step_ids": list(range(1, 17)),
                "distance_rows": [
                    {"restored_event_step_ids": [], "distance": 100.0},
                    {"restored_event_step_ids": left_selected, "distance": 90.0},
                    {"restored_event_step_ids": right_selected, "distance": 80.0},
                    {
                        "restored_event_step_ids": reference,
                        "distance": 0.0,
                    },
                ],
                "operation_counts": {
                    "processor_preflights": _count(4),
                    "canonical_action_parse_attempts": _count(2),
                    "reference_teacher_forwards": _count(1),
                    "candidate_teacher_forwards": _count(3, 3 if valid else 1),
                    "distance_rows": _count(3, 3 if valid else 1),
                    "analytic_self_distance_rows": _count(1),
                },
            }
        )
    return {"source_id": source_id, "n8": n8, "n16_pairs": pairs}


class FakeAdapter:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on_call = fail_on_call

    def run_trajectory(
        self,
        *,
        trajectory,
        selector_records,
        host_gpu_id,
        container_cuda_ordinal,
        runner_config,
    ):
        del selector_records, host_gpu_id, container_cuda_ordinal, runner_config
        self.calls.append(trajectory["source_id"])
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("injected interruption")
        return _runtime_record(trajectory["source_id"])


def _run_one(tmp_path: Path, worker_index: int, adapter: FakeAdapter) -> dict:
    return run_worker_shard(
        contract=_contract(tmp_path),
        runner_config=_runner(),
        runner_config_sha256="b" * 64,
        selection_manifest=_selection(),
        selector_seal=_selector_seal(),
        trajectories=_trajectories(),
        worker_index=worker_index,
        host_gpu_id=str(worker_index),
        container_cuda_ordinal=worker_index,
        output_root=tmp_path / "run",
        adapter=adapter,
        persistent_root=tmp_path,
    )


def test_build_worker_assignments_is_exact_four_by_six() -> None:
    assignments = build_worker_assignments(_runner())
    assert [item.host_gpu_id for item in assignments] == ["0", "1", "2", "3"]
    assert [item.container_cuda_ordinal for item in assignments] == [0, 1, 2, 3]
    assert [item.selection_ranks for item in assignments] == [
        tuple(range(0, 6)),
        tuple(range(6, 12)),
        tuple(range(12, 18)),
        tuple(range(18, 24)),
    ]


def test_runner_accepts_prepare_generated_manifest_and_rejects_model_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, selection_payload, seal_payload, preparation_payload = (
        _runner_validation_fixture(tmp_path)
    )
    assert MANIFEST_RELATIVE_PATH == SELECTOR_PREPARATION_MANIFEST_PATH
    assert SELECTION_RELATIVE_PATH == SELECTOR_SEAL_PATH
    monkeypatch.setattr(
        "causalcache.long_horizon_execution.validate_long_horizon_selection_manifest",
        lambda *_args, **_kwargs: None,
    )
    validated = validate_runner_config(
        runner,
        contract=_contract(tmp_path),
        selection_manifest_payload=selection_payload,
        selector_preparation_manifest_payload=preparation_payload,
        selector_seal_payload=seal_payload,
    )
    assert validated["selector_seal"]["path"] == SELECTION_RELATIVE_PATH
    v4_files = json.loads(preparation_payload)["models"][
        "v4_frozen_base_residual"
    ]["files"]
    assert len(v4_files) == 8
    assert {record["path"] for record in v4_files} == {
        "manifest.json",
        "label-blind-seal.json",
        "residual-model-metadata.json",
        *(f"residual-checkpoints/seed-{seed}.safetensors" for seed in range(5)),
    }

    mutated_manifest = json.loads(preparation_payload)
    mutated_manifest["models"]["formal58"]["files"][0]["sha256"] = "f" * 64
    mutated_payload = pretty_json_bytes(mutated_manifest)
    mutated_runner = copy.deepcopy(runner)
    mutated_runner["selector_preparation_manifest"]["sha256"] = sha256_bytes(
        mutated_payload
    )
    mutated_runner["selector_preparation_manifest"]["size_bytes"] = len(
        mutated_payload
    )
    with pytest.raises(ValueError, match="formal58 files immutable file roster"):
        validate_runner_config(
            mutated_runner,
            contract=_contract(tmp_path),
            selection_manifest_payload=selection_payload,
            selector_preparation_manifest_payload=mutated_payload,
            selector_seal_payload=seal_payload,
        )


@pytest.mark.parametrize(
    ("path", "mutation"),
    [
        ("manifest.json", "drop"),
        ("label-blind-seal.json", "drop"),
        ("manifest.json", "digest"),
        ("label-blind-seal.json", "digest"),
    ],
)
def test_runner_rejects_missing_or_misbound_v4_manifest_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    mutation: str,
) -> None:
    runner, selection_payload, seal_payload, preparation_payload = (
        _runner_validation_fixture(tmp_path)
    )
    monkeypatch.setattr(
        "causalcache.long_horizon_execution.validate_long_horizon_selection_manifest",
        lambda *_args, **_kwargs: None,
    )
    manifest = json.loads(preparation_payload)
    files = manifest["models"]["v4_frozen_base_residual"]["files"]
    record = next(record for record in files if record["path"] == path)
    if mutation == "drop":
        files.remove(record)
    else:
        record["sha256"] = "f" * 64
    mutated_payload = pretty_json_bytes(manifest)
    runner["selector_preparation_manifest"]["sha256"] = sha256_bytes(
        mutated_payload
    )
    runner["selector_preparation_manifest"]["size_bytes"] = len(mutated_payload)
    with pytest.raises(ValueError, match="preparation v4 immutable file roster"):
        validate_runner_config(
            runner,
            contract=_contract(tmp_path),
            selection_manifest_payload=selection_payload,
            selector_preparation_manifest_payload=mutated_payload,
            selector_seal_payload=seal_payload,
        )


def test_worker_is_receipt_resumable_without_reexecution(tmp_path: Path) -> None:
    first = FakeAdapter()
    receipt = _run_one(tmp_path, 0, first)
    assert len(first.calls) == 6
    assert receipt["selection_ranks"] == list(range(6))
    second = FakeAdapter()
    replay = _run_one(tmp_path, 0, second)
    assert replay == receipt
    assert second.calls == []


def test_worker_resumes_after_partial_interruption(tmp_path: Path) -> None:
    interrupted = FakeAdapter(fail_on_call=2)
    with pytest.raises(RuntimeError, match="injected interruption"):
        _run_one(tmp_path, 0, interrupted)
    assert interrupted.calls == ["source-00", "source-01"]
    resumed = FakeAdapter()
    _run_one(tmp_path, 0, resumed)
    assert resumed.calls == [f"source-{rank:02d}" for rank in range(1, 6)]


def test_worker_rejects_gpu_different_from_frozen_assignment(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GPU host ID"):
        run_worker_shard(
            contract=_contract(tmp_path),
            runner_config=_runner(),
            runner_config_sha256="b" * 64,
            selection_manifest=_selection(),
            selector_seal=_selector_seal(),
            trajectories=_trajectories(),
            worker_index=0,
            host_gpu_id="3",
            container_cuda_ordinal=0,
            output_root=tmp_path / "run",
            adapter=FakeAdapter(),
            persistent_root=tmp_path,
        )


def test_worker_rejects_output_outside_persistent_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-run"
    with pytest.raises(ValueError, match="persistent /data"):
        run_worker_shard(
            contract=_contract(tmp_path),
            runner_config=_runner(),
            runner_config_sha256="b" * 64,
            selection_manifest=_selection(),
            selector_seal=_selector_seal(),
            trajectories=_trajectories(),
            worker_index=0,
            host_gpu_id="0",
            container_cuda_ordinal=0,
            output_root=outside,
            adapter=FakeAdapter(),
            persistent_root=tmp_path,
        )


def test_runtime_accounting_conserves_requests_and_preserves_partial_invalid(
    tmp_path: Path,
) -> None:
    record = _runtime_record("source-00", valid=False)
    validated = validate_runtime_trajectory_record(
        record, source_id="source-00", contract=_contract(tmp_path)
    )
    assert validated["n8"]["operation_counts"]["distance_rows"] == _count(163, 7)
    broken = copy.deepcopy(record)
    broken["n8"]["operation_counts"]["distance_rows"]["missing"] = 0
    with pytest.raises(ValueError, match=r"completed \+ missing"):
        validate_runtime_trajectory_record(
            broken, source_id="source-00", contract=_contract(tmp_path)
        )


def test_valid_runtime_requires_every_operation_to_complete(tmp_path: Path) -> None:
    record = _runtime_record("source-00")
    record["n8"]["operation_counts"]["processor_preflights"] = _count(164, 163)
    with pytest.raises(ValueError, match="valid n8 state must complete every"):
        validate_runtime_trajectory_record(
            record, source_id="source-00", contract=_contract(tmp_path)
        )

    record = _runtime_record("source-00")
    record["n16_pairs"][0]["operation_counts"][
        "canonical_action_parse_attempts"
    ] = _count(2, 1)
    with pytest.raises(ValueError, match="valid n16 pair must complete every"):
        validate_runtime_trajectory_record(
            record, source_id="source-00", contract=_contract(tmp_path)
        )


def test_aggregate_requires_exactly_once_24_and_conserves_operations(
    tmp_path: Path,
) -> None:
    for worker in range(4):
        _run_one(tmp_path, worker, FakeAdapter())
    summary = aggregate_worker_outputs(
        contract=_contract(tmp_path),
        runner_config=_runner(),
        runner_config_sha256="b" * 64,
        selection_manifest=_selection(),
        selector_seal=_selector_seal(),
        output_root=tmp_path / "run",
        persistent_root=tmp_path,
    )
    assert summary["selected_trajectory_denominator"] == 24
    assert summary["completed_trajectory_count"] == 24
    assert summary["coverage"]["status"] == "GO_LONG_HORIZON_POLICY_COVERAGE"
    assert summary["operation_counts"]["n8.distance_rows"]["requested"] == 3912
    assert summary["operation_counts"]["n16.distance_rows"]["requested"] == 432
    assert summary["access_accounting"]["top_up_count"] == 0
    descriptive = summary["scientific_evaluation"]["secondary_descriptive"]
    assert set(descriptive["n8"]) == {"2", "4"}
    assert "random" in descriptive["n8"]["4"]["selectors"]
    assert "summary_only" in descriptive["n8"]["2"]["selectors"]
    assert len(descriptive["n16_pair_deltas"]) == 6


def test_aggregate_rejects_missing_worker_result(tmp_path: Path) -> None:
    for worker in range(4):
        _run_one(tmp_path, worker, FakeAdapter())
    path = tmp_path / "run/worker-3/trajectory-rank-23.json"
    path.unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        aggregate_worker_outputs(
            contract=_contract(tmp_path),
            runner_config=_runner(),
            runner_config_sha256="b" * 64,
            selection_manifest=_selection(),
            selector_seal=_selector_seal(),
            output_root=tmp_path / "run",
            persistent_root=tmp_path,
        )


def test_selector_seal_requires_budget_selector_state_matrix(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    selection = _selection()
    records = []
    matrix = {
        2: contract.data["selector_matrix"]["budget_2"],
        4: contract.data["selector_matrix"]["budget_4"],
    }
    for rank in range(24):
        source_id = f"source-{rank:02d}"
        for step, count in ((10, 8), (18, 16)):
            state_id = f"{source_id}:decision_step:{step:03d}"
            for budget, selectors in matrix.items():
                for selector in selectors:
                    records.append(
                        {
                            "selector_name": selector,
                            "source_id": source_id,
                            "state_id": state_id,
                            "decision_step_id": step,
                            "candidate_event_step_ids": list(range(1, count + 1)),
                            "budget_event_capacity": budget,
                            "selected_event_step_ids": [],
                        }
                    )
    seal = {
        "schema_version": "1.0.0",
        "protocol_id": "causalcache_long_horizon_label_blind_selection",
        "status": "SEALED_LONG_HORIZON_LABEL_BLIND_SELECTIONS_V1",
        "selector_names": sorted(set(matrix[2]) | set(matrix[4])),
        "budgets": [2, 4],
        "selector_names_by_budget": {
            str(budget): sorted(selectors) for budget, selectors in matrix.items()
        },
        "state_count": 48,
        "record_count": len(records),
        "records": records,
    }
    seal["records"].sort(
        key=lambda item: (
            item["budget_event_capacity"],
            item["selector_name"],
            item["state_id"],
        )
    )
    payload = (json.dumps(
        seal, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ) + "\n").encode()
    validated = validate_selector_seal_payload(
        payload,
        expected_sha256=sha256_bytes(payload),
        selection_manifest=selection,
        contract=contract,
    )
    assert validated["record_count"] == 576
    seal["records"].pop()
    seal["record_count"] -= 1
    bad = (json.dumps(seal, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(ValueError, match="exact B2/B4 matrix"):
        validate_selector_seal_payload(
            bad,
            expected_sha256=sha256_bytes(bad),
            selection_manifest=selection,
            contract=contract,
        )


def test_canonical_adapter_constructs_only_frozen_n16_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, str, str]] = []

    def fake_n8(**kwargs):
        return {"valid": True}, False

    def fake_n16(*, pair, **kwargs):
        del kwargs
        calls.append(
            (
                pair.budget_event_capacity,
                pair.left_selector_name,
                pair.right_selector_name,
            )
        )
        return {"valid": True}, False

    monkeypatch.setattr(
        "causalcache.long_horizon_runtime.run_n8_state_with_resume", fake_n8
    )
    monkeypatch.setattr(
        "causalcache.long_horizon_runtime.run_n16_pair_with_resume", fake_n16
    )
    adapter = CanonicalLongHorizonRuntimeAdapter(
        contract=_contract(tmp_path),
        runtime=object(),
        distance_backend=object(),
        image_bytes_loader=lambda _: b"",
        image_decoder=lambda _: object(),
        selection_seal_sha256="c" * 64,
        receipt_store=object(),
    )
    trajectory = {"source_id": "source-00"}
    selector_records = [
        {
            "decision_step_id": 18,
            "budget_event_capacity": budget,
            "selector_name": selector,
            "selected_event_step_ids": [1, 2],
        }
        for budget, selectors in (
            (2, _contract(tmp_path).data["selector_matrix"]["budget_2"]),
            (4, _contract(tmp_path).data["selector_matrix"]["budget_4"]),
        )
        for selector in selectors
    ]
    result = adapter.run_trajectory(
        trajectory=trajectory,
        selector_records=selector_records,
        host_gpu_id="0",
        container_cuda_ordinal=0,
        runner_config=_runner(),
    )
    assert result["source_id"] == "source-00"
    assert calls == [(budget, *pair) for budget, pair in PAIR_SPECS]


def test_scientific_reducer_zero_fills_six_invalid_trajectories(tmp_path: Path) -> None:
    records = [
        _runtime_record(f"source-{rank:02d}", valid=rank < 18)
        for rank in range(24)
    ]
    result = evaluate_scientific_go(
        records,
        selector_seal=_selector_seal(),
        contract=_contract(tmp_path),
        policy_coverage_passed=True,
    )
    report = result["set_aware_vs_independent"]["conditions"]["v1_conditional"]
    assert report["n8"]["valid_paired_trajectory_count"] == 18
    assert report["n8"]["zero_filled_invalid_trajectory_count"] == 6
    assert report["n8"]["bootstrap"]["trajectory_count"] == 24
    assert report["n16"]["valid_paired_trajectory_count"] == 18
    assert report["n16"]["zero_filled_invalid_trajectory_count"] == 6
    oracle = result["independent_long_horizon"]["conditions"]["exact_oracle"]
    assert oracle["valid_trajectory_count"] == 18
    assert oracle["zero_filled_invalid_trajectory_count"] == 6


def test_set_aware_overall_go_requires_any_one_named_selector(tmp_path: Path) -> None:
    seal = copy.deepcopy(_selector_seal())
    for record in seal["records"]:
        if record["selector_name"] == "v4_safe_frozen_base_residual":
            record["selected_event_step_ids"] = [1]
    seal_sha256 = sha256_bytes(canonical_json_bytes(seal) + b"\n")
    records = [_runtime_record(f"source-{rank:02d}") for rank in range(24)]
    for trajectory in records:
        for pair in trajectory["n16_pairs"]:
            pair["selection_seal_sha256"] = seal_sha256
            if pair["left_selector_name"] == "v1_conditional":
                pair["distance_rows"] = [
                    {"restored_event_step_ids": [], "distance": 100.0},
                    {"restored_event_step_ids": [5, 6], "distance": 70.0},
                    {"restored_event_step_ids": [1, 2], "distance": 80.0},
                    {"restored_event_step_ids": [1, 2, 5, 6], "distance": 0.0},
                ]
            if pair["left_selector_name"] == "v4_safe_frozen_base_residual":
                pair["left_selected_event_step_ids"] = [1]
                pair["distance_rows"] = [
                    {"restored_event_step_ids": [], "distance": 100.0},
                    {"restored_event_step_ids": [1], "distance": 90.0},
                    {"restored_event_step_ids": [1, 2], "distance": 0.0},
                ]
    result = evaluate_scientific_go(
        records,
        selector_seal=seal,
        contract=_contract(tmp_path),
        policy_coverage_passed=True,
    )
    set_aware = result["set_aware_vs_independent"]
    assert set_aware["conditions"]["v1_conditional"]["all_conditions_pass"]
    assert not set_aware["conditions"]["v4_safe_frozen_base_residual"][
        "all_conditions_pass"
    ]
    assert set_aware["status"] == "GO_SET_AWARE_LONG_HORIZON_DEVELOPMENT_V1"


def test_scientific_reducer_rejects_n16_set_not_in_seal(tmp_path: Path) -> None:
    records = [_runtime_record(f"source-{rank:02d}") for rank in range(24)]
    records[0]["n16_pairs"][0]["left_selected_event_step_ids"] = [8]
    with pytest.raises(ValueError, match="INVALID_LONG_HORIZON_SELECTOR_JOIN"):
        evaluate_scientific_go(
            records,
            selector_seal=_selector_seal(),
            contract=_contract(tmp_path),
            policy_coverage_passed=True,
        )
