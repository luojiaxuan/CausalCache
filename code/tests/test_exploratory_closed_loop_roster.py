from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

import causalcache.exploratory_closed_loop_roster as roster_module
from causalcache.exploratory_closed_loop_roster import (
    EXPECTED_POOL_SIZES,
    IDENTITY_FIELDS,
    MANIFEST_PATH,
    MANIFEST_SHA256,
    PLAN_PATH,
    PLAN_SHA256,
    PROTOCOL_ID,
    RANK_KEY_DESCRIPTION,
    SELECTED_INSTANCE_IDENTITY_SHA256,
    canonical_json_bytes,
    horizon_stratum,
    selected_instance_identity_projection,
    selected_instance_identity_sha256,
    selection_sha256,
    validate_exploratory_closed_loop_roster,
    validate_frozen_exploratory_closed_loop_roster_files,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / PLAN_PATH
MANIFEST = ROOT / MANIFEST_PATH


def _values() -> tuple[dict, dict]:
    return (
        json.loads(PLAN.read_text(encoding="utf-8")),
        json.loads(MANIFEST.read_text(encoding="utf-8")),
    )


def _rebind_plan_instances(plan: dict) -> None:
    plan["task_instance_count"] = len(plan["instances"])
    plan["task_type_count"] = len(
        {instance["task_type"] for instance in plan["instances"]}
    )
    plan["instance_records_sha256"] = hashlib.sha256(
        canonical_json_bytes(plan["instances"])
    ).hexdigest()


def test_frozen_files_recompute_exact_roster_and_byte_identities() -> None:
    result = validate_frozen_exploratory_closed_loop_roster_files(
        repository_root=ROOT
    )
    assert result["status"] == (
        "VALID_EXPLORATORY_CLOSED_LOOP_VALIDATION12_ROSTER_V1"
    )
    assert result["protocol_id"] == PROTOCOL_ID
    assert result["eligible_task_index"] == 0
    assert result["eligible_stratum_pool_sizes"] == EXPECTED_POOL_SIZES
    assert result["take_per_stratum"] == 4
    assert result["selected_record_count"] == 12
    assert result["identity_projection_fields"] == list(IDENTITY_FIELDS)
    assert (
        result["selected_instance_identity_sha256"]
        == SELECTED_INSTANCE_IDENTITY_SHA256
    )
    assert result["plan_sha256"] == PLAN_SHA256
    assert result["manifest_sha256"] == MANIFEST_SHA256
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == PLAN_SHA256
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == MANIFEST_SHA256


def test_selection_digest_uses_exact_nul_delimited_utf8_key() -> None:
    expected_payload = "\0".join(
        (
            PROTOCOL_ID,
            "validation",
            "271828",
            "SimpleSmsSendClipboardContent",
            "0",
        )
    ).encode("utf-8")
    expected = hashlib.sha256(expected_payload).hexdigest()
    assert expected == (
        "0e2779869f580ecc39229d2e69200a90650df7a885a6a46d7c220547ee9e12e2"
    )
    assert selection_sha256(
        task_type="SimpleSmsSendClipboardContent", task_index=0
    ) == expected
    _, manifest = _values()
    assert manifest["selection"]["rank_key"] == RANK_KEY_DESCRIPTION
    assert all(
        record["selection_sha256"]
        == selection_sha256(
            task_type=record["instance"]["task_type"],
            task_index=record["instance"]["task_index"],
        )
        for record in manifest["records"]
    )


def test_identity_projection_has_exact_fields_order_and_canonical_hash() -> None:
    _, manifest = _values()
    projection = selected_instance_identity_projection(manifest["records"])
    assert len(projection) == 12
    assert all(set(item) == set(IDENTITY_FIELDS) for item in projection)
    assert projection[0] == {
        "horizon_stratum": "short",
        "max_steps": 12,
        "task_index": 0,
        "task_type": "SimpleSmsSendClipboardContent",
    }
    payload = json.dumps(
        projection,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert not payload.endswith(b"\n")
    assert canonical_json_bytes(projection) == payload
    assert hashlib.sha256(payload).hexdigest() == (
        SELECTED_INSTANCE_IDENTITY_SHA256
    )
    assert selected_instance_identity_sha256(manifest["records"]) == (
        SELECTED_INSTANCE_IDENTITY_SHA256
    )


def test_stratum_boundaries_are_exact_and_type_strict() -> None:
    assert horizon_stratum(1) == "short"
    assert horizon_stratum(12) == "short"
    assert horizon_stratum(13) == "medium"
    assert horizon_stratum(22) == "medium"
    assert horizon_stratum(23) == "long"
    with pytest.raises(ValueError, match="positive integer"):
        horizon_stratum(True)
    with pytest.raises(ValueError, match="positive integer"):
        horizon_stratum(0)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda plan, manifest: manifest.update({"unexpected": False}),
        lambda plan, manifest: manifest["selection"].update(
            {"rank_key": "sha256(task_type)"}
        ),
        lambda plan, manifest: manifest["selection"].update(
            {"eligible_task_index": 1}
        ),
        lambda plan, manifest: manifest["selection"].update(
            {"take_per_stratum": 3}
        ),
        lambda plan, manifest: manifest["source_validation_plan"].update(
            {"sha256": "0" * 64}
        ),
        lambda plan, manifest: manifest["records"][0].update(
            {"selection_sha256": "0" * 64}
        ),
        lambda plan, manifest: manifest["records"].__setitem__(
            slice(0, 2), reversed(manifest["records"][:2])
        ),
        lambda plan, manifest: manifest["records"][0]["instance"].update(
            {"goal": "post-hoc replacement"}
        ),
        lambda plan, manifest: manifest["records"][0].update(
            {"horizon_stratum": "long"}
        ),
        lambda plan, manifest: manifest.update(
            {"selected_instance_identity_sha256": "0" * 64}
        ),
    ),
)
def test_manifest_mutations_fail_closed(mutation) -> None:
    plan, manifest = _values()
    mutation(plan, manifest)
    with pytest.raises(ValueError, match="drifted|differ|key inventory"):
        validate_exploratory_closed_loop_roster(plan=plan, manifest=manifest)


def test_plan_pool_size_drift_fails_even_with_rebound_internal_metadata() -> None:
    plan, manifest = _values()
    plan["instances"] = [
        instance
        for instance in plan["instances"]
        if not (
            instance["task_index"] == 0
            and horizon_stratum(instance["max_steps"]) == "short"
        )
    ]
    _rebind_plan_instances(plan)
    with pytest.raises(ValueError, match="plan instance count"):
        validate_exploratory_closed_loop_roster(plan=plan, manifest=manifest)


def test_mechanical_pool_derivation_rejects_non_15_8_8_inventory() -> None:
    plan, _ = _values()
    instances = [
        instance
        for instance in plan["instances"]
        if not (
            instance["task_index"] == 0
            and instance["task_type"] == "MarkorDeleteNote"
        )
    ]
    with pytest.raises(ValueError, match="eligible stratum pool sizes drifted"):
        roster_module._expected_records(instances)


def test_plan_instance_drift_fails_against_frozen_record_digest() -> None:
    plan, manifest = _values()
    plan["instances"][0]["goal"] = "drifted"
    with pytest.raises(ValueError, match="plan instance payload"):
        validate_exploratory_closed_loop_roster(plan=plan, manifest=manifest)


def test_selected_records_are_grouped_and_ranked_within_each_stratum() -> None:
    _, manifest = _values()
    observed_strata = [record["horizon_stratum"] for record in manifest["records"]]
    assert observed_strata == ["short"] * 4 + ["medium"] * 4 + ["long"] * 4
    for stratum in ("short", "medium", "long"):
        digests = [
            record["selection_sha256"]
            for record in manifest["records"]
            if record["horizon_stratum"] == stratum
        ]
        assert digests == sorted(digests)


def test_frozen_file_validator_rejects_manifest_byte_drift(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    manifest = tmp_path / "manifest.json"
    shutil.copyfile(PLAN, plan)
    shutil.copyfile(MANIFEST, manifest)
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="roster manifest file SHA256 drifted"):
        validate_frozen_exploratory_closed_loop_roster_files(
            repository_root=tmp_path,
            plan_path=plan,
            manifest_path=manifest,
        )


def test_frozen_file_validator_rejects_plan_byte_drift(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    manifest = tmp_path / "manifest.json"
    shutil.copyfile(PLAN, plan)
    shutil.copyfile(MANIFEST, manifest)
    plan.write_bytes(plan.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="validation plan file SHA256 drifted"):
        validate_frozen_exploratory_closed_loop_roster_files(
            repository_root=tmp_path,
            plan_path=plan,
            manifest_path=manifest,
        )


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key.*schema_version"):
        roster_module._strict_json_bytes(
            b'{"schema_version":"1","schema_version":"2"}', label="fixture"
        )
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        roster_module._strict_json_bytes(b'{"value":NaN}', label="fixture")


def test_selection_digest_rejects_nul_ambiguous_identity() -> None:
    with pytest.raises(ValueError, match="task_type"):
        selection_sha256(task_type="Task\0Injected", task_index=0)
