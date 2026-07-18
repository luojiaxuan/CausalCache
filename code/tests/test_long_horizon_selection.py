from __future__ import annotations

import copy
import hashlib
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from causalcache.data.guiodyssey_independent import (
    Candidate,
    ExclusionReason,
    InspectionResult,
    SourceFileSpec,
    trajectory_selection_sha256,
)
from causalcache.long_horizon_contract import LongHorizonContract
from causalcache.long_horizon_selection import (
    LongHorizonPool,
    build_long_horizon_selection_manifest,
    derive_structural_inspection_config,
    development_input_records,
    reconstruct_long_horizon_pool,
    validate_long_horizon_selection_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


def _contract() -> LongHorizonContract:
    return LongHorizonContract.load(repository_root=ROOT)


def _candidate(index: int, *, source_id: str | None = None) -> Candidate:
    source_id = source_id or f"long-horizon-synthetic-{index:03d}"
    decision_count = 18 + index % 9
    return Candidate(
        source_id=source_id,
        transport_file=f"mobile/use/train/shard-{index % 16:05d}-of-00610.parquet",
        transport_row_index=index // 16,
        selection_sha256=trajectory_selection_sha256(
            source_id,
            salt="causalcache-long-horizon-development-v1",
        ),
        decision_count=decision_count,
        normalized_app_labels=(f"app-{index % 4}",),
        action_type_counts=(("swipe", 2), ("tap", decision_count - 3), ("type_text", 1)),
    )


def _pool(*, count: int = 42) -> LongHorizonPool:
    return LongHorizonPool(
        candidates=tuple(reversed([_candidate(index) for index in range(count)])),
        total_source_rows=212,
        source_row_counts={f"shard-{index:02d}": 1 for index in range(16)},
        exclusion_counts={"decision_count_below_minimum": 170},
    )


def _generator() -> dict:
    return {
        "git_revision": "a" * 40,
        "module_path": "code/causalcache/long_horizon_selection.py",
        "module_sha256": "b" * 64,
        "cli_path": "code/scripts/build_long_horizon_selection.py",
        "cli_sha256": "c" * 64,
        "contract_validator_path": "code/scripts/validate_long_horizon_contract.py",
        "contract_validator_sha256": "d" * 64,
    }


def _build(pool: LongHorizonPool | None = None) -> dict:
    contract = _contract()
    return build_long_horizon_selection_manifest(
        pool=pool or _pool(),
        contract=contract,
        parent_config_sha256=contract.source_pool["parent_protocol_config"]["sha256"],
        source_file_manifest_sha256=contract.source_pool["source_file_manifest"]["sha256"],
        generator=_generator(),
    )


def test_selection_is_exact_42_to_24_and_18_with_canonical_records() -> None:
    manifest = _build()
    assert manifest["reconstruction"]["eligible_trajectory_count"] == 42
    development = manifest["splits"]["development"]
    reserve = manifest["splits"]["unopened_reserve"]
    assert development["trajectory_count"] == 24
    assert reserve["trajectory_count"] == 18
    assert development["state_count"] == 48
    assert reserve["state_count"] == 0
    assert reserve["states"] == []
    required = {
        "trajectory_id",
        "shard_path",
        "row_index",
        "decision_count",
        "selection_rank",
        "role",
        "selection_sha256",
    }
    assert required.issubset(development["trajectories"][0])
    assert required == set(reserve["trajectories"][0])
    assert "action_type_counts" in development["trajectories"][0]
    assert "action_type_counts" not in reserve["trajectories"][0]
    assert manifest["historical_role_firewall"]["intersection_count"] == 0


def test_state_geometry_is_strict_n8_and_n16_without_padding() -> None:
    states = _build()["splits"]["development"]["states"]
    assert len(states) == 48
    for index in range(0, len(states), 2):
        n8, n16 = states[index : index + 2]
        assert n8["decision_step_id"] == 10
        assert n8["candidate_event_step_ids"] == list(range(1, 9))
        assert n8["history_event_step_ids"] == list(range(1, 10))
        assert n16["decision_step_id"] == 18
        assert n16["candidate_event_step_ids"] == list(range(1, 17))
        assert n16["history_event_step_ids"] == list(range(1, 18))
        assert n8["trajectory_id"] == n16["trajectory_id"]


def test_selection_is_deterministic_under_input_order() -> None:
    first = _build(_pool())
    pool = _pool()
    forward = LongHorizonPool(
        candidates=tuple(reversed(pool.candidates)),
        total_source_rows=pool.total_source_rows,
        source_row_counts=pool.source_row_counts,
        exclusion_counts=pool.exclusion_counts,
    )
    second = _build(forward)
    assert first == second


@pytest.mark.parametrize("count", [41, 43])
def test_selection_rejects_any_pool_count_other_than_exact_42(count: int) -> None:
    with pytest.raises(ValueError, match="exactly 42"):
        _build(_pool(count=count))


def test_selection_rejects_ids_from_historical_roles() -> None:
    contract = _contract()
    historical_id = contract.data["historical_role_firewall"]["manifests"][0]
    # note (luojiaxuan): Parse the ID from the bound manifest; do not copy it here.
    from causalcache.long_horizon_contract import historical_role_inventory

    parsed_id = historical_role_inventory(
        contract.data, repository_root=ROOT
    )["union_source_ids"][0]
    candidates = list(_pool().candidates)
    candidates[0] = _candidate(100, source_id=parsed_id)
    pool = LongHorizonPool(
        candidates=tuple(candidates),
        total_source_rows=212,
        source_row_counts=_pool().source_row_counts,
        exclusion_counts=_pool().exclusion_counts,
    )
    assert historical_id["extractor"] == "splits_source_ids"
    with pytest.raises(ValueError, match="overlap historical roles"):
        _build(pool)


def test_validator_rejects_padding_reserve_semantics_and_split_locator_overlap() -> None:
    contract = _contract()
    manifest = _build()

    padded = copy.deepcopy(manifest)
    padded["semantics"]["padding_count"] = 1
    with pytest.raises(ValueError, match="forbidden selection"):
        validate_long_horizon_selection_manifest(padded, contract=contract)

    reserve_semantics = copy.deepcopy(manifest)
    reserve_semantics["splits"]["unopened_reserve"]["trajectories"][0][
        "normalized_app_labels"
    ] = ["leak"]
    with pytest.raises(ValueError, match="schema drifted"):
        validate_long_horizon_selection_manifest(reserve_semantics, contract=contract)

    overlap = copy.deepcopy(manifest)
    first = overlap["splits"]["development"]["trajectories"][0]
    reserve = overlap["splits"]["unopened_reserve"]["trajectories"][0]
    reserve["shard_path"] = first["shard_path"]
    reserve["row_index"] = first["row_index"]
    with pytest.raises(ValueError):
        validate_long_horizon_selection_manifest(overlap, contract=contract)


def test_only_development_records_are_exposed_to_downstream_builder() -> None:
    records = development_input_records(_build(), contract=_contract())
    assert len(records) == 24
    assert {record["role"] for record in records} == {"development"}
    assert all("action_type_counts" in record for record in records)


def test_n8_exact_counts_and_n16_pair_union_is_not_search_or_oracle() -> None:
    manifest = _build()
    n8 = manifest["reference_plan"]["n8"]
    n16 = manifest["reference_plan"]["n16"]
    assert n8["budget_2_subset_count"] == 37
    assert n8["budget_4_subset_count"] == 163
    assert n16["reference_type"] == "frozen_selector_pair_union"
    assert n16["enumeration_or_shortlist_allowed"] is False
    assert n16["global_oracle_or_search_claim_allowed"] is False
    assert n16["budget_2_comparator_pairs"] == [
        ["restoration_independent_gate", "recent"],
        ["restoration_independent_gate", "ocr_rgb_v2"],
        ["v1_conditional", "restoration_independent_gate"],
        ["v4_safe_frozen_base_residual", "restoration_independent_gate"],
    ]
    assert manifest["policy_context_profile"]["maximum_context_tokens"] == 32768
    assert manifest["policy_context_profile"][
        "n8_full_reference_total_images_including_current"
    ] == 9
    assert manifest["policy_context_profile"]["n16_full_reference_allowed"] is False
    assert manifest["policy_context_profile"][
        "n16_budget_4_pair_union_maximum_total_images_including_current"
    ] == 9
    assert manifest["operation_plan"][
        "n8_budget_4_total_requested_unique_subset_rows"
    ] == 3912
    assert manifest["execution_b_plan"]["worker_selection_rank_ranges"] == [
        [0, 5],
        [6, 11],
        [12, 17],
        [18, 23],
    ]
    assert manifest["artifact_plan"]["repo"] == (
        "gavinlaw/causalcache-long-horizon-development-mobile"
    )
    learned = manifest["learned_model_artifacts"]
    assert len(learned["formal58_base_and_conditional"]["checkpoints"]) == 10
    assert len(
        learned["v4_safe_frozen_base_residual"]["residual_checkpoints"]
    ) == 5
    independent_go = manifest["validity_and_go_plan"][
        "independent_long_horizon_development_go"
    ]
    assert independent_go[
        "n8_trajectory_equal_mean_raw_utility_ratio_to_exact_minimum"
    ] == 0.85
    assert independent_go["n8_per_comparator_minimum_positive_trajectories"] == 14
    assert independent_go["budget_2_is_secondary_report_only"] is True
    assert independent_go["pass_outcome"] == (
        "AUTHORIZE_NEW_CLOSED_LOOP_SOURCE_A_ONLY"
    )
    assert independent_go["old_closed_loop_or_test_direct_unlock_allowed"] is False


def test_structural_inspection_changes_only_bounds_and_salt() -> None:
    contract = _contract()
    import json

    parent = json.loads(
        (ROOT / contract.source_pool["parent_protocol_config"]["path"]).read_text()
    )
    derived = derive_structural_inspection_config(
        parent_config=parent,
        contract=contract,
    )
    expected = copy.deepcopy(parent)
    expected["eligibility"]["minimum_decisions_per_trajectory"] = 18
    expected["eligibility"]["maximum_decisions_per_trajectory"] = 60
    expected["selection"]["trajectory_salt"] = (
        "causalcache-long-horizon-development-v1"
    )
    assert derived == expected


def test_raw_parquet_reconstruction_scans_every_row_with_derived_contract() -> None:
    contract = _contract()
    import json

    parent = json.loads(
        (ROOT / contract.source_pool["parent_protocol_config"]["path"]).read_text()
    )
    source_manifest = json.loads(
        (ROOT / contract.source_pool["source_file_manifest"]["path"]).read_text()
    )
    candidate = _candidate(0)
    rows = [{"row": 0}, {"row": 1}]

    class Table:
        def to_pylist(self):
            return rows

    class Parquet:
        num_row_groups = 1
        metadata = types.SimpleNamespace(num_rows=2)

        def read_row_group(self, index):
            assert index == 0
            return Table()

    parquet_module = types.ModuleType("pyarrow.parquet")
    parquet_module.ParquetFile = lambda _: Parquet()
    pyarrow_module = types.ModuleType("pyarrow")
    pyarrow_module.__path__ = []
    pyarrow_module.parquet = parquet_module

    def inspect(row, *, transport_file, transport_row_index, config):
        assert config["eligibility"]["minimum_decisions_per_trajectory"] == 18
        assert config["eligibility"]["maximum_decisions_per_trajectory"] == 60
        assert config["selection"]["trajectory_salt"] == (
            "causalcache-long-horizon-development-v1"
        )
        if row["row"] == 0:
            return InspectionResult(
                source_id=candidate.source_id,
                candidate=candidate,
                exclusion_reason=None,
            )
        return InspectionResult(
            source_id="excluded-row",
            candidate=None,
            exclusion_reason=ExclusionReason.DECISION_COUNT_BELOW_MINIMUM,
        )

    spec = SourceFileSpec(
        "mobile/use/train/shard-00000-of-00610.parquet",
        1,
        hashlib.sha256(b"x").hexdigest(),
    )
    with (
        patch.dict(
            sys.modules,
            {"pyarrow": pyarrow_module, "pyarrow.parquet": parquet_module},
        ),
        patch(
            "causalcache.long_horizon_selection.source_file_specs",
            return_value=(spec,),
        ),
        patch("causalcache.long_horizon_selection.verify_local_source_files"),
        patch(
            "causalcache.long_horizon_selection.inspect_candidate",
            side_effect=inspect,
        ) as inspected,
    ):
        pool = reconstruct_long_horizon_pool(
            source_root=Path("/does/not/matter"),
            parent_config=parent,
            source_file_manifest=source_manifest,
            contract=contract,
        )
    assert inspected.call_count == 2
    assert pool.candidates == (candidate,)
    assert pool.total_source_rows == 2
    assert pool.exclusion_counts == {"decision_count_below_minimum": 1}
