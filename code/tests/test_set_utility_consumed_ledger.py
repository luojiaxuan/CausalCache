from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from causalcache.set_utility_consumed_ledger import (
    CANONICAL_LEDGER_MANIFEST_PATH,
    EXPECTED_UNION_COUNT,
    STATUS,
    InputBinding,
    _load_canonical_input,
    build_canonical_consumed_identity_ledger,
    derive_consumed_identity_ledger,
    ledger_manifest_bytes,
    sha256_bytes,
    write_ledger_manifest_exclusive,
)
from causalcache.set_utility_consumed_ledger_contract import PROTOCOL_ID
from causalcache.set_utility_full_pool import parse_consumed_ledger
from scripts.materialize_set_utility_consumed_identity_ledger import materialize


ROOT = Path(__file__).resolve().parents[2]
LEGACY_PATH = ROOT / "data/manifests/restoration_v2_selection.json"
EXPANSION_PATH = (
    ROOT / "data/manifests/restoration_v2_2_label_expansion_selection.json"
)
REFERENCE_PATH = ROOT / "data/manifests/independent_reference_gate_v1_artifact.json"
PREREGISTRATION_PATH = ROOT / "code/configs/causalcache_gate_v1_preregistration.json"
CANONICAL_MANIFEST_PATH = ROOT / CANONICAL_LEDGER_MANIFEST_PATH


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_evidence() -> tuple[dict, dict, dict, dict, tuple[InputBinding, ...]]:
    canonical = build_canonical_consumed_identity_ledger(repository_root=ROOT)
    bindings = tuple(
        InputBinding(path=record["path"], sha256=record["sha256"])
        for record in canonical.payload["inputs"]
    )
    return (
        _read(PREREGISTRATION_PATH),
        _read(LEGACY_PATH),
        _read(EXPANSION_PATH),
        _read(REFERENCE_PATH),
        bindings,
    )


def _derive(
    *,
    preregistration: dict | None = None,
    legacy: dict | None = None,
    expansion: dict | None = None,
    reference: dict | None = None,
    inputs: tuple[InputBinding, ...] | None = None,
):
    base_preregistration, base_legacy, base_expansion, base_reference, base_inputs = (
        _canonical_evidence()
    )
    return derive_consumed_identity_ledger(
        preregistration=preregistration or base_preregistration,
        legacy_selection=legacy or base_legacy,
        expansion_selection=expansion or base_expansion,
        reference_artifact=reference or base_reference,
        inputs=inputs or base_inputs,
    )


def test_canonical_ledger_mechanically_recovers_58_49_and_107() -> None:
    ledger = build_canonical_consumed_identity_ledger(repository_root=ROOT)
    assert ledger.payload["status"] == STATUS
    assert len(ledger.legacy_train_only_source_ids) == 58
    assert len(ledger.forbidden_consumed_source_ids) == 49
    assert (
        ledger.legacy_train_only_source_ids
        & ledger.forbidden_consumed_source_ids
        == set()
    )
    assert ledger.payload["consumed_union"]["source_id_count"] == EXPECTED_UNION_COUNT
    assert ledger.payload["assignment_count"] == EXPECTED_UNION_COUNT
    assert len(ledger.payload["assignments"]) == EXPECTED_UNION_COUNT

    role_counts = {
        record["ledger_role"]: record["source_id_count"]
        for record in ledger.payload["role_inventories"]
    }
    assert role_counts == {
        "legacy_label_train": 10,
        "expansion_train": 48,
        "reference8": 8,
        "old_development5": 5,
        "fresh16": 16,
        "confirm20": 20,
    }
    partition_counts = Counter(
        record["role"] for record in ledger.payload["assignments"]
    )
    assert partition_counts == {"legacy_train_only": 58, "forbidden_consumed": 49}


def test_reference8_is_extracted_and_cross_checked_not_copied_into_code() -> None:
    ledger = build_canonical_consumed_identity_ledger(repository_root=ROOT)
    reference_manifest = _read(REFERENCE_PATH)
    expected = set(reference_manifest["splits"]["reference_gate"]["source_ids"])
    reference_inventory = next(
        record
        for record in ledger.payload["role_inventories"]
        if record["ledger_role"] == "reference8"
    )
    emitted = set(reference_inventory["source_ids"])
    assert emitted == expected
    assert len(emitted) == 8


def test_payload_matches_shared_contract_and_full_pool_parser_directly() -> None:
    ledger = build_canonical_consumed_identity_ledger(repository_root=ROOT)
    assert ledger.payload["protocol_id"] == PROTOCOL_ID
    parsed = parse_consumed_ledger(
        ledger.payload,
        manifest_sha256=ledger.summary_sha256,
    )
    assert len(parsed.source_to_role) == 107
    assert parsed.assignment_inventory_sha256 == (
        ledger.payload["assignment_inventory_sha256"]
    )
    assert parsed.role_by_source == {
        record["source_id"]: record["role"]
        for record in ledger.payload["assignments"]
    }


def test_canonical_git_manifest_is_byte_exact_and_full_pool_parseable() -> None:
    ledger = build_canonical_consumed_identity_ledger(repository_root=ROOT)
    payload = CANONICAL_MANIFEST_PATH.read_bytes()
    assert payload == ledger_manifest_bytes(ledger)
    parsed_manifest = json.loads(payload)
    parsed = parse_consumed_ledger(
        parsed_manifest,
        manifest_sha256=sha256_bytes(payload),
    )
    assert len(parsed.source_to_role) == 107
    assert parsed.assignment_inventory_sha256 == (
        ledger.payload["assignment_inventory_sha256"]
    )


def test_exclusive_writer_is_reproducible_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    ledger = build_canonical_consumed_identity_ledger(repository_root=ROOT)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_sha256 = write_ledger_manifest_exclusive(ledger, first)
    second_sha256 = write_ledger_manifest_exclusive(ledger, second)
    assert first.read_bytes() == second.read_bytes() == ledger_manifest_bytes(ledger)
    assert first_sha256 == second_sha256 == sha256_bytes(first.read_bytes())
    before = first.read_bytes()
    with pytest.raises(FileExistsError):
        write_ledger_manifest_exclusive(ledger, first)
    assert first.read_bytes() == before


def test_materialize_cli_core_reports_exact_counts_and_no_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "ledger.json"
    summary = materialize(repository_root=ROOT, output_path=output)
    assert summary["assignment_count"] == 107
    assert summary["partition_counts"] == {
        "legacy_train_only": 58,
        "forbidden_consumed": 49,
    }
    assert summary["file_sha256"] == sha256_bytes(output.read_bytes())
    with pytest.raises(FileExistsError):
        materialize(repository_root=ROOT, output_path=output)


def test_canonical_input_and_inventory_hashes_are_stable() -> None:
    first = build_canonical_consumed_identity_ledger(repository_root=ROOT)
    second = build_canonical_consumed_identity_ledger(repository_root=ROOT)
    assert first.payload == second.payload
    assert first.summary_sha256 == second.summary_sha256

    assert {record["path"]: record["sha256"] for record in first.payload["inputs"]} == {
        "code/configs/causalcache_gate_v1_fresh16_evaluation_v1.json": (
            "c98647aecf6b07e0ccccf1601b5e21289b595b7a4a45cc7ab9a542b1bd7dff2e"
        ),
        "code/configs/causalcache_gate_v1_preregistration.json": (
            "37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b"
        ),
        "code/configs/causalcache_set_utility_long_pool_discovery_v1.json": (
            "d10484f53f579bf26012ba6fdd3e7e701c90d7e760db6667ebf7e74c75a98957"
        ),
        "data/manifests/independent_reference_gate_v1_artifact.json": (
            "833139954e332c00a2ed06fe2c3e71e701ca9a4a381f4c41680ec57d64a827ff"
        ),
        "data/manifests/restoration_v2_2_label_expansion_selection.json": (
            "4aec4deffc6405c3d06ca3001d082e4fbd85ee44f55785708a0cee573edcd169"
        ),
        "data/manifests/restoration_v2_selection.json": (
            "292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b"
        ),
    }
    role_hashes = {
        record["ledger_role"]: record["source_ids_sha256"]
        for record in first.payload["role_inventories"]
    }
    assert role_hashes == {
        "legacy_label_train": (
            "97c026f4e6e48b0f748c8495c967c0c796af256ee78fdf2e81b530f54be9cb35"
        ),
        "expansion_train": (
            "11904e1d102b69561b402bea4e42044be9572884f6168597c5f07c3ca346cb34"
        ),
        "reference8": (
            "6522e2f9c7eb08765606250fbb480b4f46e009a892b3693a573ea07c9c10791d"
        ),
        "old_development5": (
            "0e3d67d79ac45392122fd9a712af43ea544445420e7fdb06df22bad7d84b05e6"
        ),
        "fresh16": (
            "1c37cbf6b67b0ddee61b3efe27d33b30c8fbfb471ced12f12e49624a10b82454"
        ),
        "confirm20": (
            "c84ba8b9b705abc7ba7d6d7230b868fd4600e3a835b1c763a8aaf1f93feb052e"
        ),
    }
    assert first.payload["partitions"]["legacy_train_only"][
        "source_ids_sha256"
    ] == "ccec55afb0882e602f5d2c83ec420682488e2663fe26a7c574a851825c43f225"
    assert first.payload["partitions"]["forbidden_consumed"][
        "source_ids_sha256"
    ] == "84c5aa49c806e87d8715078f8908263ca1de0eecd037e0c8f76c6778ef818ec9"
    assert first.payload["consumed_union"]["source_set_sha256"] == (
        "54be8ee52248e47adcfbaea0c03a5d19793486519d6cf124474fb468f628b7b1"
    )
    assert first.payload["assignment_inventory_sha256"] == (
        "a0d5af88fe1b007ef077d0f6e18e4fa7aac11a6bbd222cdf95778001bc97a98f"
    )


@pytest.mark.parametrize(
    ("origin", "section", "new_role"),
    (
        ("legacy", "roles", "unexpected_legacy_role"),
        ("expansion", "splits", "unexpected_expansion_role"),
        ("reference", "splits", "unexpected_reference_split"),
    ),
)
def test_unknown_roles_fail_closed(origin: str, section: str, new_role: str) -> None:
    preregistration, legacy, expansion, reference, inputs = _canonical_evidence()
    target = {"legacy": legacy, "expansion": expansion, "reference": reference}[origin]
    target[section][new_role] = {}
    with pytest.raises(ValueError, match="unknown or missing"):
        derive_consumed_identity_ledger(
            preregistration=preregistration,
            legacy_selection=legacy,
            expansion_selection=expansion,
            reference_artifact=reference,
            inputs=inputs,
        )


def test_missing_source_fails_closed_on_frozen_count() -> None:
    preregistration, legacy, expansion, reference, inputs = _canonical_evidence()
    expansion["splits"]["gate_train_expansion"]["trajectories"].pop()
    with pytest.raises(ValueError, match="count drifted"):
        derive_consumed_identity_ledger(
            preregistration=preregistration,
            legacy_selection=legacy,
            expansion_selection=expansion,
            reference_artifact=reference,
            inputs=inputs,
        )


def test_duplicate_source_inside_one_role_fails_closed() -> None:
    preregistration, legacy, expansion, reference, inputs = _canonical_evidence()
    trajectories = legacy["roles"]["v2_label_train"]["trajectories"]
    trajectories[1]["source_id"] = trajectories[0]["source_id"]
    with pytest.raises(ValueError, match="duplicate source IDs"):
        derive_consumed_identity_ledger(
            preregistration=preregistration,
            legacy_selection=legacy,
            expansion_selection=expansion,
            reference_artifact=reference,
            inputs=inputs,
        )


def test_cross_role_overlap_fails_closed() -> None:
    preregistration, legacy, expansion, reference, inputs = _canonical_evidence()
    old_development_id = legacy["roles"]["v2_development"]["trajectories"][0][
        "source_id"
    ]
    legacy["roles"]["v1_reference_contract_audit_only"]["trajectories"][0][
        "source_id"
    ] = old_development_id
    reference["splits"]["reference_gate"]["source_ids"][0] = old_development_id
    with pytest.raises(ValueError, match="roles overlap"):
        derive_consumed_identity_ledger(
            preregistration=preregistration,
            legacy_selection=legacy,
            expansion_selection=expansion,
            reference_artifact=reference,
            inputs=inputs,
        )


def test_reference_evidence_disagreement_fails_closed() -> None:
    preregistration, legacy, expansion, reference, inputs = _canonical_evidence()
    reference["splits"]["reference_gate"]["source_ids"] = list(
        reversed(reference["splits"]["reference_gate"]["source_ids"])
    )
    with pytest.raises(ValueError, match="reference8 identity evidence disagrees"):
        derive_consumed_identity_ledger(
            preregistration=preregistration,
            legacy_selection=legacy,
            expansion_selection=expansion,
            reference_artifact=reference,
            inputs=inputs,
        )


def test_preregistered_role_hash_drift_fails_closed() -> None:
    preregistration, legacy, expansion, reference, inputs = _canonical_evidence()
    preregistration["rosters"]["fresh_development"]["source_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source IDs drifted"):
        derive_consumed_identity_ledger(
            preregistration=preregistration,
            legacy_selection=legacy,
            expansion_selection=expansion,
            reference_artifact=reference,
            inputs=inputs,
        )


def test_malformed_source_id_and_duplicate_input_path_fail_closed() -> None:
    preregistration, legacy, expansion, reference, inputs = _canonical_evidence()
    malformed = copy.deepcopy(expansion)
    malformed["splits"]["gate_development_expansion"]["trajectories"][0][
        "source_id"
    ] = "bad source id"
    with pytest.raises(ValueError, match="normalized source ID"):
        derive_consumed_identity_ledger(
            preregistration=preregistration,
            legacy_selection=legacy,
            expansion_selection=malformed,
            reference_artifact=reference,
            inputs=inputs,
        )

    with pytest.raises(ValueError, match="duplicate paths"):
        derive_consumed_identity_ledger(
            preregistration=preregistration,
            legacy_selection=legacy,
            expansion_selection=expansion,
            reference_artifact=reference,
            inputs=(*inputs, inputs[0]),
        )

    with pytest.raises(ValueError, match="input inventory drifted"):
        derive_consumed_identity_ledger(
            preregistration=preregistration,
            legacy_selection=legacy,
            expansion_selection=expansion,
            reference_artifact=reference,
            inputs=inputs[:-1],
        )


def test_byte_hash_drift_is_rejected_before_json_is_used() -> None:
    with pytest.raises(ValueError, match="SHA256 drifted"):
        _load_canonical_input(
            ROOT,
            InputBinding(
                path="data/manifests/restoration_v2_selection.json",
                sha256="0" * 64,
            ),
        )
