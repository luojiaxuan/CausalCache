from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from causalcache.independent_confirm_failure_decomposition_contract import (
    CANONICAL_CONFIG_PATH,
    CANONICAL_RUNNER_FREEZE_PATH,
    CHILD_REPO,
    CHILD_TAG,
    CONFIG_SHA256,
    EXPECTED_PARENT_TARGETS,
    PARENT_REPORT_COMMIT,
    PARENT_TAG_OBJECT,
    load_frozen_failure_decomposition_contract,
    load_runner_freeze,
    pretty_json_bytes,
    runner_freeze_bytes,
    sha256_bytes,
    validate_contract_data,
)
from causalcache.independent_confirm_failure_decomposition_contract import _git_blob


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def contract():
    return load_frozen_failure_decomposition_contract(repository_root=ROOT)


def test_canonical_config_identity_and_consumed_confirm_audit_scope(contract) -> None:
    assert contract.sha256 == CONFIG_SHA256
    assert contract.source_path == ROOT / CANONICAL_CONFIG_PATH
    assert contract.data["audit_scope"] == {
        "confirm_consumed": True,
        "formal_run_not_yet_executed": True,
        "not_blind_holdout": True,
        "pre_source_a_exploratory_numeric_read": True,
        "user_case_split_predated_exploratory_read": True,
    }


def test_git_blob_preserves_exact_trailing_lf(tmp_path) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "test@example.com"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Test"), cwd=tmp_path, check=True)
    payload = b"exact-bytes-with-trailing-lf\n"
    (tmp_path / "bound.txt").write_bytes(payload)
    subprocess.run(("git", "add", "bound.txt"), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True)
    assert _git_blob(tmp_path, "HEAD:bound.txt") == payload


def test_parent_identity_and_exact_three_reader_are_frozen(contract) -> None:
    parent = contract.data["parent_confirm"]
    assert parent["hf"]["report_commit"] == PARENT_REPORT_COMMIT
    assert parent["hf"]["annotated_tag_object"] == PARENT_TAG_OBJECT
    assert parent["locked_result"]["status"] == "NO_GO_INDEPENDENT_CONFIRM"
    assert parent["locked_result"]["closed_loop_executed"] is False
    assert tuple(contract.data["input_contract"]["exact_force_download_targets"]) == (
        EXPECTED_PARENT_TARGETS
    )
    assert contract.data["input_contract"]["parent_remote_mutation_call_count"] == 0


def test_j_projection_and_case_a_b_inconclusive_routing_are_frozen(contract) -> None:
    analysis = contract.data["analysis_contract"]
    assert analysis["j_selector"] == {
        "definition": "0.5_times_empty_marginal_plus_mean_of_other_singleton_base_conditional_marginals",
        "event_score": "0.5_times_D_empty_minus_D_j_plus_mean_i_not_j_of_D_i_minus_D_ij",
        "forced_fill": False,
        "strictly_positive_score_required": True,
        "tie_break": "lower_event_step_id",
        "top_b": True,
    }
    routing = contract.data["routing_contract"]
    assert routing["case_a"]["maximum_raw_mean_delta_inclusive"] == 0.0
    assert routing["case_b"]["minimum_positive_trajectory_count"] == 12
    assert routing["case_b"]["all_required"] == [
        "J_minus_O_raw_mean_strictly_positive",
        "J_minus_O_paired_bootstrap_90_lower_strictly_positive",
        "J_minus_O_positive_trajectory_count_at_least_12_of_20",
        "sealed_I_minus_O_raw_mean_strictly_negative",
    ]
    assert routing["inconclusive"]["action"] == (
        "NO_INDEPENDENT_V2_RESCUE_FROM_THIS_DIAGNOSTIC"
    )


def test_cpu_only_source_a_keeps_every_authorization_and_operation_locked(
    contract,
) -> None:
    assert contract.data["runtime_contract"] == {
        "cpu_only": True,
        "gpu_count": 0,
        "gpu_preflight_required": False,
        "model_runtime_allowed": False,
        "network_scope": "parent_read_only_then_new_child_publication_only",
        "torch_import_required": False,
    }
    assert all(value is False for value in contract.data["authorization"].values())
    assert all(
        value == 0
        for value in contract.data["source_only_operation_contract"].values()
    )
    assert contract.data["routing_contract"]["never_authorizes"] == [
        "confirm_reclassification",
        "closed_loop",
        "matched_nll",
        "sealed_androidworld_test",
        "gate_training_on_confirm20",
    ]


def test_child_hf_and_source_a_b_inventory_are_minimal(contract) -> None:
    destination = contract.data["destination"]
    assert destination["repo"] == CHILD_REPO
    assert destination["tag"] == CHILD_TAG
    assert destination["private"] is True
    assert destination["single_exact_report_commit"] is True
    source = contract.data["source_freeze"]
    assert source["execution_b_runner_freeze"] == {
        "path": CANONICAL_RUNNER_FREEZE_PATH,
        "must_be_absent_during_source_only_validation": True,
        "only_allowed_execution_b_source_tree_diff": True,
        "direct_single_parent_child_of_source_a": True,
        "bind_source_a_inventory_sha256": True,
        "bind_loaded_module_inventory_sha256": True,
        "bind_required_source_a_paths": True,
        "bind_git_prerequisites": True,
    }
    assert CANONICAL_RUNNER_FREEZE_PATH not in source["required_source_a_paths"]
    assert len(source["required_source_a_paths"]) == len(
        set(source["required_source_a_paths"])
    )
    assert len(source["git_prerequisites"]) == 6


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("audit_scope", "not_blind_holdout"), False),
        (("analysis_contract", "j_selector", "forced_fill"), True),
        (("routing_contract", "case_a", "maximum_raw_mean_delta_inclusive"), 0.01),
        (("routing_contract", "case_b", "minimum_positive_trajectory_count"), 11),
        (("runtime_contract", "gpu_count"), 1),
        (("authorization", "closed_loop_authorized"), True),
        (("input_contract", "parent_remote_mutation_call_count"), 1),
        (("parent_confirm", "hf", "report_commit"), "0" * 40),
    ],
)
def test_contract_rejects_audit_science_or_authorization_drift(
    contract, path, replacement
) -> None:
    changed = copy.deepcopy(contract.data)
    cursor = changed
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(ValueError):
        validate_contract_data(changed)


def _fake_source_validation(contract) -> dict:
    source_inventory = [
        {"path": path, "size_bytes": 1, "sha256": "1" * 64}
        for path in contract.source["required_source_a_paths"]
    ]
    loaded = [
        {
            "module": module,
            "path": "code/" + module.replace(".", "/") + ".py",
            "size_bytes": 1,
            "sha256": "2" * 64,
        }
        for module in contract.source["required_execution_modules"]
    ]
    return {
        "source_a": {
            "git_commit": "a" * 40,
            "origin_main_git_commit": "a" * 40,
            "branch": "main",
            "origin_url": "https://github.com/luojiaxuan/CausalCache.git",
            "source_inventory": source_inventory,
            "source_inventory_sha256": sha256_bytes(
                json.dumps(
                    source_inventory,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
            "loaded_module_inventory": loaded,
            "loaded_module_inventory_sha256": sha256_bytes(
                json.dumps(
                    loaded,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
        }
    }


def test_runner_freeze_is_deterministic_and_binds_source_parent_and_child(
    contract, tmp_path
) -> None:
    validation = _fake_source_validation(contract)
    first = runner_freeze_bytes(contract, validation)
    assert first == runner_freeze_bytes(contract, validation)
    clone = replace(contract, repository_root=tmp_path)
    runner = tmp_path / CANONICAL_RUNNER_FREEZE_PATH
    runner.parent.mkdir(parents=True)
    runner.write_bytes(first)
    frozen = load_runner_freeze(clone)
    assert frozen["source_a_git_commit"] == "a" * 40
    assert frozen["execution_b_required_unique_diff"] == [
        CANONICAL_RUNNER_FREEZE_PATH
    ]
    assert frozen["parent_report_commit"] == PARENT_REPORT_COMMIT
    assert frozen["child_repo"] == CHILD_REPO
    assert frozen["diagnostic_only"] is True
    assert frozen["closed_loop_authorized"] is False


def test_runner_freeze_rejects_post_freeze_authorization_drift(
    contract, tmp_path
) -> None:
    payload = runner_freeze_bytes(contract, _fake_source_validation(contract))
    value = json.loads(payload)
    value["closed_loop_authorized"] = True
    clone = replace(contract, repository_root=tmp_path)
    runner = tmp_path / CANONICAL_RUNNER_FREEZE_PATH
    runner.parent.mkdir(parents=True)
    runner.write_bytes(pretty_json_bytes(value))
    with pytest.raises(ValueError):
        load_runner_freeze(clone)
