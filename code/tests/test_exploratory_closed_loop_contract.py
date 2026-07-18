from __future__ import annotations

import copy
import hashlib
import json
import shutil
import socket
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from unittest import mock

import pytest

import causalcache.exploratory_closed_loop_contract as contract_module
from causalcache.exploratory_closed_loop_contract import (
    ARMS,
    CANONICAL_CONFIG_PATH,
    FROZEN_CONFIG_SHA256,
    PARENT_NO_GO,
    ROSTER_MANIFEST_PATH,
    ROSTER_MANIFEST_SHA256,
    ROUTES,
    VALIDATION_STATUS,
    load_frozen_exploratory_closed_loop_contract,
    validate_exploratory_closed_loop_contract,
    validate_source_only_contract,
)
from scripts.validate_exploratory_closed_loop_contract import _parser


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _copy_canonical_config(destination_root: Path) -> Path:
    target = destination_root / CANONICAL_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG, target)
    return target


def test_frozen_contract_binds_roster_parent_models_and_ocr() -> None:
    contract = load_frozen_exploratory_closed_loop_contract(
        repository_root=ROOT
    )
    assert contract.sha256 == FROZEN_CONFIG_SHA256
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == FROZEN_CONFIG_SHA256
    assert contract.roster["manifest"] == {
        "path": ROSTER_MANIFEST_PATH,
        "sha256": ROSTER_MANIFEST_SHA256,
        "status": "FROZEN_EXPLORATORY_CLOSED_LOOP_VALIDATION12_ROSTER_V1",
        "selected_instance_identity_sha256": (
            "e0ae4ad79b33f1bce44158be5ec22f55e83fc054462407d56b2d3b9c5036199b"
        ),
    }
    assert contract.data["parent_no_go"]["result_summary"][
        "scientific_outcome"
    ] == PARENT_NO_GO
    assert contract.data["policy"]["snapshot_manifest"]["sha256"] == (
        "50b675ec31c5c46dbb0d44c137a808fffb9d054916d39b596648d4eb9df7cbc3"
    )
    assert contract.data["ocr"]["backend_manifest"]["sha256"] == (
        "107478672438b52e9c2ccc9ef8de5d13d4e16329ab1afbe95772e2d3df5bd2a9"
    )
    checkpoints = contract.data["gate"]["checkpoints"]
    assert [(item["family"], item["seed"]) for item in checkpoints] == [
        *(("conditional", seed) for seed in range(5)),
        *(("independent", seed) for seed in range(5)),
    ]


def test_contract_freezes_five_arms_reselection_runtime_and_denominator() -> None:
    contract = load_frozen_exploratory_closed_loop_contract(
        repository_root=ROOT
    )
    assert tuple(contract.arms["ordered_arms"]) == ARMS
    assert contract.arms["all_candidates_rescored_at_every_policy_decision"]
    assert not contract.arms["current_equivalent_newest_post_state_is_candidate"]
    assert contract.arms["maximum_high_fidelity_events"] == 2
    assert contract.roster["template_count"] == 12
    assert contract.roster["arm_count"] == 5
    assert contract.roster["episode_count"] == 60
    assert contract.runtime["policy_host"] == "aries"
    assert contract.runtime["emulator_host"] == "aries"
    assert contract.runtime["topology"] == (
        "same_host_policy_and_androidworld_emulator"
    )
    assert contract.runtime["maximum_worker_count"] == 4
    assert not contract.runtime["hyper_policy_allowed_in_v1"]


def test_validity_itt_and_routes_are_exact_and_followups_stay_locked() -> None:
    contract = load_frozen_exploratory_closed_loop_contract(
        repository_root=ROOT
    )
    routing = contract.data["validity_and_routing"]
    validity = routing["validity"]
    assert validity["exact_episode_record_count"] == 60
    assert validity["paired_arm_inventory_complete"]
    assert validity["missing_episode_record_outcome"] == ROUTES[3]
    assert validity[
        "recorded_parse_selector_executor_or_infrastructure_failure_itt_success"
    ] == 0.0
    assert (
        routing["advance_route"],
        routing["stop_route"],
        routing["fallback_route"],
        routing["invalid_route"],
    ) == ROUTES
    assert routing["advance_requires"] == {
        "valid_execution": True,
        "independent_vs_recent_overall_wins_minus_losses_minimum": 2,
        "independent_vs_recent_long_wins_minus_losses_minimum": 1,
        "independent_vs_ocr_overall_wins_minus_losses_minimum": 0,
        "independent_vs_ocr_long_wins_minus_losses_minimum": 0,
        "independent_vs_summary_overall_wins_minus_losses_minimum": 0,
        "memory_binding_decision_count_strictly_positive": True,
        "independent_recent_selector_disagreement_count_strictly_positive": True,
    }
    assert not routing["conditional_success_may_rescue_independent_route"]
    assert contract.locks["train60_protocol_status"] == (
        "LOCKED_REQUIRES_ADVANCE_AND_NEW_VERSIONED_CONTRACT"
    )
    assert contract.locks["sealed_test75_status"] == "LOCKED"
    assert contract.locks["matched_nll_status"] == "LOCKED"
    assert contract.locks["paper_primary_table_status"] == "LOCKED"


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["scientific_scope"].update(
            {"offline_confirm_reclassified_as_pass": True}
        ),
        lambda value: value["parent_no_go"]["result_summary"].update(
            {"scientific_outcome": "GO_TO_PAIRED_CLOSED_LOOP"}
        ),
        lambda value: value["roster"]["manifest"].update({"sha256": "0" * 64}),
        lambda value: value["roster"].update({"episode_count": 59}),
        lambda value: value["policy"].update(
            {"revision": "0" * 40}
        ),
        lambda value: value["gate"]["checkpoints"].pop(),
        lambda value: value["ocr"].update(
            {"feature_or_ocr_fallback_allowed": True}
        ),
        lambda value: value["memory_and_arms"]["ordered_arms"].reverse(),
        lambda value: value["memory_and_arms"].update(
            {"all_candidates_rescored_at_every_policy_decision": False}
        ),
        lambda value: value["online_rollout"].update(
            {"fixed_trajectory_replay_across_arms_allowed": True}
        ),
        lambda value: value["evaluation"].update(
            {"statistical_significance_or_long_horizon_claim_allowed": True}
        ),
        lambda value: value["validity_and_routing"].update(
            {"advance_route": "ADVANCE_NOW"}
        ),
        lambda value: value["runtime"].update({"policy_host": "hyper00"}),
        lambda value: value["locked_followups"].update(
            {"sealed_test75_status": "OPEN"}
        ),
        lambda value: value["source_only_operation_contract"].update(
            {"network_call_count": 1}
        ),
        lambda value: value["authorization"].update(
            {"exploratory_closed_loop_execution_authorized": True}
        ),
    ),
)
def test_contract_rejects_any_scientific_or_authority_drift(mutation) -> None:
    changed = copy.deepcopy(_config())
    mutation(changed)
    with pytest.raises((PermissionError, ValueError), match="drifted|zero|false"):
        validate_exploratory_closed_loop_contract(changed)


def test_contract_rejects_unknown_top_level_key() -> None:
    changed = _config()
    changed["unbound_override"] = False
    with pytest.raises(ValueError, match="top-level key inventory drifted"):
        validate_exploratory_closed_loop_contract(changed)


def test_source_only_validator_reads_only_contract_and_has_no_side_effects() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture_root = Path(directory)
        canonical = _copy_canonical_config(fixture_root)
        write_methods = (
            "write_bytes",
            "write_text",
            "touch",
            "mkdir",
            "rename",
            "replace",
            "unlink",
        )
        patches = [
            mock.patch.object(Path, name, side_effect=AssertionError(f"write: {name}"))
            for name in write_methods
        ]
        patches.extend(
            (
                mock.patch.object(
                    socket, "socket", side_effect=AssertionError("network")
                ),
                mock.patch.object(
                    urllib.request,
                    "urlopen",
                    side_effect=AssertionError("network"),
                ),
                mock.patch.object(
                    subprocess, "run", side_effect=AssertionError("subprocess")
                ),
                mock.patch.object(
                    contract_module,
                    "_regular_file_bytes",
                    wraps=contract_module._regular_file_bytes,
                ),
            )
        )
        for patch in patches:
            patch.start()
        try:
            result = validate_source_only_contract(repository_root=fixture_root)
            reader = patches[-1].target._regular_file_bytes
        finally:
            for patch in reversed(patches):
                patch.stop()

    assert reader.call_count == 1
    assert reader.call_args.args[0] == canonical.resolve()
    assert result["status"] == VALIDATION_STATUS
    assert result["only_read_path"] == CANONICAL_CONFIG_PATH
    assert result["bound_identity_sections"] == [
        "parent_no_go",
        "roster",
        "policy",
        "gate",
        "ocr",
    ]
    assert result["artifact_access_count"] == 0
    assert not result["bound_artifacts_opened_or_verified"]
    assert not result["execution_authorized"]
    assert all(value == 0 for value in result["source_only_operation_counts"].values())


def test_loader_rejects_noncanonical_path_and_byte_drift(tmp_path: Path) -> None:
    _copy_canonical_config(tmp_path)
    alternative = tmp_path / "alternate.json"
    shutil.copyfile(CONFIG, alternative)
    with pytest.raises(ValueError, match="path is not canonical"):
        load_frozen_exploratory_closed_loop_contract(
            alternative, repository_root=tmp_path
        )

    canonical = tmp_path / CANONICAL_CONFIG_PATH
    canonical.write_bytes(canonical.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="contract bytes drifted"):
        load_frozen_exploratory_closed_loop_contract(repository_root=tmp_path)


def test_loader_rejects_canonical_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    shutil.copyfile(CONFIG, target)
    canonical = tmp_path / CANONICAL_CONFIG_PATH
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.symlink_to(target)
    with pytest.raises(ValueError, match="path is not canonical"):
        load_frozen_exploratory_closed_loop_contract(repository_root=tmp_path)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key.*schema_version"):
        contract_module._strict_json_bytes(
            b'{"schema_version":"1","schema_version":"2"}', label="fixture"
        )
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        contract_module._strict_json_bytes(b'{"value":NaN}', label="fixture")


def test_cli_surface_cannot_accept_artifacts_devices_or_outputs() -> None:
    parser = _parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert options == {"-h", "--help", "--contract", "--repository-root"}
    for forbidden in (
        "--data",
        "--manifest",
        "--checkpoint",
        "--model",
        "--device",
        "--server",
        "--hf-token-file",
        "--output",
    ):
        assert forbidden not in options
