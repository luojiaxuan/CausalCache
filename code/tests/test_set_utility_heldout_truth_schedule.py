from __future__ import annotations

import hashlib
import json

import pytest

from causalcache.set_utility_heldout_evaluation import canonical_json_bytes, sha256_file
from causalcache.set_utility_heldout_truth_schedule import (
    EXPECTED_SHARD_COUNT,
    FORMAL_MANIFEST_STATUS,
    SCHEDULE_STATUS,
    materialize_heldout_truth_schedules,
    seal_formal_truth_root,
)
from causalcache.set_utility_train_heldout_contract import sha256_json
from causalcache.set_utility_variable_history import history_bin


def _write_signed(path, value: dict[str, object]) -> str:
    unsigned = dict(value)
    digest = hashlib.sha256(
        canonical_json_bytes(unsigned, pretty=True)
    ).hexdigest()
    signed = {**unsigned, "content_sha256": digest}
    path.write_bytes(canonical_json_bytes(signed, pretty=True) + b"\n")
    return digest


def _fixture(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    states = []
    checkpoint_states = []
    source_shards = []
    for index in range(EXPECTED_SHARD_COUNT):
        trajectory_id = f"t{index:03d}"
        state_id = f"{trajectory_id}:decision:006"
        candidates = [1, 2, 3, 4, 5]
        states.append(
            {
                "candidate_event_step_ids": candidates,
                "logical_shard": index,
                "role": "train",
                "state_id": state_id,
                "trajectory_id": trajectory_id,
            }
        )
        checkpoint_states.append(
            {
                "candidate_count": len(candidates),
                "history_bin": history_bin(len(candidates)),
                "state_id": state_id,
                "trajectory_id": trajectory_id,
            }
        )
        source_shards.append(
            {
                "logical_shard": index,
                "sha256": hashlib.sha256(f"shard-{index}".encode()).hexdigest(),
                "trajectory_ids": [trajectory_id],
            }
        )
    states_path = input_root / "states.jsonl"
    states_path.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in states)
    )
    input_sha = "a" * 64
    (input_root / "manifest.json").write_bytes(
        canonical_json_bytes(
            {
                "content_sha256": input_sha,
                "evaluation_labels_included": False,
                "state_count": len(states),
                "states_jsonl": "states.jsonl",
                "states_sha256": sha256_file(states_path),
            },
            pretty=True,
        )
        + b"\n"
    )
    heldout = {
        "checkpoint_state_ids": [row["state_id"] for row in checkpoint_states],
        "checkpoint_states": checkpoint_states,
        "firewall": {
            "allowed_role": "train",
            "evaluation_access": False,
            "tune_access": False,
        },
    }
    heldout["content_sha256"] = sha256_json(heldout)
    heldout_path = tmp_path / "heldout.json"
    heldout_path.write_bytes(canonical_json_bytes(heldout) + b"\n")
    source_path = tmp_path / "source.json"
    source_path.write_bytes(
        canonical_json_bytes(
            {
                "content_sha256": "b" * 64,
                "shards": source_shards,
                "status": "COMPLETED_VARIABLE_HISTORY_SOURCE",
            },
            pretty=True,
        )
        + b"\n"
    )
    source_sha = sha256_file(source_path)
    schedules = {}
    for model_name in ("deepsets", "set_transformer"):
        records = []
        for index, state in enumerate(states):
            missing = []
            if index % 2 == 0:
                missing.append(
                    {
                        "event_ids": [1],
                        "source": f"{model_name}_epoch_rollout",
                    }
                )
            if model_name == "set_transformer" and index % 3 == 0:
                missing.append(
                    {
                        "event_ids": [1, 5],
                        "source": "set_transformer_interaction",
                    }
                )
            records.append(
                {
                    "candidate_event_ids": state["candidate_event_step_ids"],
                    "desired_coalitions": [],
                    "history_bin": "short",
                    "logical_shard": state["logical_shard"],
                    "missing_coalitions": missing,
                    "state_id": state["state_id"],
                    "trajectory_id": state["trajectory_id"],
                }
            )
        schedule = {
            "epoch_checkpoints": [
                {
                    "checkpoint_sha256": (
                        "c" * 64 if model_name == "deepsets" else "d" * 64
                    ),
                    "epoch": 1,
                }
            ],
            "epoch_count": 1,
            "epochs": [1],
            "missing_coalition_count": sum(
                len(row["missing_coalitions"]) for row in records
            ),
            "records": records,
            "schema_version": "causalcache.structured_truth_schedule.v1",
            "state_count": len(records),
            "model_family": (
                "deepsets_structured_marginal"
                if model_name == "deepsets"
                else "set_transformer_direct_marginal"
            ),
            "status": (
                "PENDING_STRUCTURED_HELDOUT_TRUTH"
                if model_name == "deepsets"
                else "PENDING_SET_TRANSFORMER_CONTROL_TRUTH"
            ),
        }
        path = tmp_path / f"{model_name}.json"
        _write_signed(path, schedule)
        schedules[model_name] = path
    return {
        "expected_heldout_sha": heldout["content_sha256"],
        "expected_input_sha": input_sha,
        "heldout_path": heldout_path,
        "input_root": input_root,
        "schedules": schedules,
        "source_path": source_path,
        "source_sha": source_sha,
        "states": states,
    }


def _materialize(fixture, output_root):
    return materialize_heldout_truth_schedules(
        input_root=fixture["input_root"],
        expected_input_content_sha256=fixture["expected_input_sha"],
        heldout_manifest_path=fixture["heldout_path"],
        expected_heldout_content_sha256=fixture["expected_heldout_sha"],
        source_manifest_path=fixture["source_path"],
        expected_source_manifest_sha256=fixture["source_sha"],
        truth_schedule_paths=fixture["schedules"],
        output_root=output_root,
        workers=8,
    )


def _write_truth_terminals(schedule_root, truth_root):
    source_revision = "e" * 40
    scientific_sha = "f" * 64
    execution_sha = "1" * 64
    states_root = truth_root / "states"
    states_root.mkdir(parents=True)
    for schedule_path in sorted((schedule_root / "schedule-shards").glob("*.jsonl")):
        logical_shard = int(schedule_path.name.split("-")[1])
        receipt_path = (
            schedule_root / "receipts" / f"shard-{logical_shard:03d}-of-256.json"
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for line in schedule_path.read_text(encoding="utf-8").splitlines():
            schedule = json.loads(line)
            identity_payload = {
                "candidate_event_ids": schedule["candidate_event_ids"],
                "coalitions": schedule["coalitions"],
                "execution_config_sha256": execution_sha,
                "scientific_config_sha256": scientific_sha,
                "schedule_receipt_sha256": sha256_file(receipt_path),
                "source_revision": source_revision,
                "source_shard_sha256": receipt["source_shard_sha256"],
                "state_id": schedule["state_id"],
            }
            state_identity = hashlib.sha256(
                canonical_json_bytes(identity_payload, pretty=True) + b"\n"
            ).hexdigest()
            candidates = tuple(schedule["candidate_event_ids"])
            terminal = {
                "candidate_event_step_ids": list(candidates),
                "distance_rows": [
                    {
                        "coalition_event_step_ids": coalition["event_ids"],
                        "distance": (
                            0.0
                            if tuple(coalition["event_ids"]) == candidates
                            else float(ordinal + 1) / 10.0
                        ),
                    }
                    for ordinal, coalition in enumerate(schedule["coalitions"])
                ],
                "execution_config_sha256": execution_sha,
                "role": "train",
                "scientific_config_sha256": scientific_sha,
                "source_revision": source_revision,
                "state_id": schedule["state_id"],
                "state_identity_sha256": state_identity,
                "status": "COMPLETED_VARIABLE_HISTORY_LABEL_STATE",
                "trajectory_id": schedule["trajectory_id"],
            }
            path = states_root / f"{schedule['state_id'].replace(':', '_')}.json"
            path.write_bytes(canonical_json_bytes(terminal, pretty=True) + b"\n")


def test_shared_truth_schedule_unions_models_and_materializes_all_shards(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "output"
    kwargs = {
        "input_root": fixture["input_root"],
        "expected_input_content_sha256": fixture["expected_input_sha"],
        "heldout_manifest_path": fixture["heldout_path"],
        "expected_heldout_content_sha256": fixture["expected_heldout_sha"],
        "source_manifest_path": fixture["source_path"],
        "expected_source_manifest_sha256": fixture["source_sha"],
        "truth_schedule_paths": fixture["schedules"],
        "output_root": output_root,
        "workers": 8,
    }
    result = materialize_heldout_truth_schedules(**kwargs)
    assert result["status"] == SCHEDULE_STATUS
    assert result["requested_missing_coalition_count"] == 128 + 86
    assert result["scheduled_state_count"] == 171
    assert len(list((output_root / "schedule-shards").glob("*.jsonl"))) == 256
    assert len(list((output_root / "receipts").glob("*.json"))) == 256

    rows = [
        json.loads(line)
        for path in sorted((output_root / "schedule-shards").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    state_zero = next(row for row in rows if row["state_id"].startswith("t000:"))
    coalitions = {tuple(row["event_ids"]): row for row in state_zero["coalitions"]}
    assert set(coalitions) == {(1,), (1, 5), (1, 2, 3, 4, 5)}
    assert coalitions[(1,)]["source"] == (
        "deepsets:deepsets_epoch_rollout+"
        "set_transformer:set_transformer_epoch_rollout"
    )
    assert materialize_heldout_truth_schedules(**kwargs) == result


def test_shared_truth_schedule_rejects_schedule_and_state_drift(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    schedule_path = fixture["schedules"]["deepsets"]
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["records"][0]["candidate_event_ids"] = [1, 2, 3, 4]
    unsigned = dict(schedule)
    unsigned.pop("content_sha256")
    _write_signed(schedule_path, unsigned)
    with pytest.raises(ValueError, match="identity drifted"):
        materialize_heldout_truth_schedules(
            input_root=fixture["input_root"],
            expected_input_content_sha256=fixture["expected_input_sha"],
            heldout_manifest_path=fixture["heldout_path"],
            expected_heldout_content_sha256=fixture["expected_heldout_sha"],
            source_manifest_path=fixture["source_path"],
            expected_source_manifest_sha256=fixture["source_sha"],
            truth_schedule_paths=fixture["schedules"],
            output_root=tmp_path / "bad",
            workers=1,
        )


def test_shared_truth_schedule_rejects_source_manifest_drift(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="source manifest file drifted"):
        materialize_heldout_truth_schedules(
            input_root=fixture["input_root"],
            expected_input_content_sha256=fixture["expected_input_sha"],
            heldout_manifest_path=fixture["heldout_path"],
            expected_heldout_content_sha256=fixture["expected_heldout_sha"],
            source_manifest_path=fixture["source_path"],
            expected_source_manifest_sha256="9" * 64,
            truth_schedule_paths=fixture["schedules"],
            output_root=tmp_path / "bad-source",
            workers=1,
        )


def test_shared_truth_schedule_requires_same_epoch_two_model_union(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["schedules"]["set_transformer"]
    schedule = json.loads(path.read_text(encoding="utf-8"))
    schedule.pop("content_sha256")
    schedule["epoch_count"] = 2
    schedule["epochs"] = [1, 2]
    schedule["epoch_checkpoints"].append(
        {"checkpoint_sha256": "2" * 64, "epoch": 2}
    )
    _write_signed(path, schedule)
    with pytest.raises(ValueError, match="same epochs"):
        _materialize(fixture, tmp_path / "bad-epochs")


def test_formal_truth_seal_and_allowlisted_loader_reject_tampering(tmp_path) -> None:
    from scripts.train_set_utility_structured_marginal import (
        _load_supplemental_truth,
    )

    fixture = _fixture(tmp_path)
    schedule_root = tmp_path / "schedule"
    _materialize(fixture, schedule_root)
    truth_root = tmp_path / "truth"
    _write_truth_terminals(schedule_root, truth_root)
    manifest = seal_formal_truth_root(
        schedule_root=schedule_root, truth_root=truth_root
    )
    assert manifest["status"] == FORMAL_MANIFEST_STATUS
    assert manifest["state_count"] == 171
    assert manifest["input_content_sha256"] == fixture["expected_input_sha"]
    assert manifest["heldout_manifest_content_sha256"] == fixture[
        "expected_heldout_sha"
    ]
    assert manifest["source_manifest_file_sha256"] == fixture["source_sha"]
    assert seal_formal_truth_root(
        schedule_root=schedule_root, truth_root=truth_root
    ) == manifest

    states_by_id = {row["state_id"]: row for row in fixture["states"]}
    loaded = _load_supplemental_truth(
        ((truth_root, manifest["content_sha256"]),),
        allowed_state_ids=set(states_by_id),
        states_by_id=states_by_id,
        expected_model_family="deepsets_structured_marginal",
        expected_epoch_checkpoints={1: "c" * 64},
        expected_input_content_sha256=fixture["expected_input_sha"],
        expected_heldout_manifest_content_sha256=fixture["expected_heldout_sha"],
        expected_source_manifest_file_sha256=fixture["source_sha"],
    )
    assert len(loaded) == 171
    assert loaded["t000:decision:006"][(1, 5)] > 0.0
    provenance = {
        "expected_input_content_sha256": fixture["expected_input_sha"],
        "expected_heldout_manifest_content_sha256": fixture[
            "expected_heldout_sha"
        ],
        "expected_source_manifest_file_sha256": fixture["source_sha"],
    }
    for key in provenance:
        mismatched = {**provenance, key: "8" * 64}
        with pytest.raises(ValueError, match="manifest/receipt binding"):
            _load_supplemental_truth(
                ((truth_root, manifest["content_sha256"]),),
                allowed_state_ids=set(states_by_id),
                states_by_id=states_by_id,
                expected_model_family="deepsets_structured_marginal",
                expected_epoch_checkpoints={1: "c" * 64},
                **mismatched,
            )
    with pytest.raises(ValueError, match="manifest/receipt binding"):
        _load_supplemental_truth(
            ((truth_root, "9" * 64),),
            allowed_state_ids=set(states_by_id),
            states_by_id=states_by_id,
            expected_model_family="deepsets_structured_marginal",
            expected_epoch_checkpoints={1: "c" * 64},
            expected_input_content_sha256=fixture["expected_input_sha"],
            expected_heldout_manifest_content_sha256=fixture["expected_heldout_sha"],
            expected_source_manifest_file_sha256=fixture["source_sha"],
        )
    with pytest.raises(ValueError, match="only stale epoch schedules"):
        _load_supplemental_truth(
            ((truth_root, manifest["content_sha256"]),),
            allowed_state_ids=set(states_by_id),
            states_by_id=states_by_id,
            expected_model_family="deepsets_structured_marginal",
            expected_epoch_checkpoints={1: "c" * 64, 2: "2" * 64},
            expected_input_content_sha256=fixture["expected_input_sha"],
            expected_heldout_manifest_content_sha256=fixture["expected_heldout_sha"],
            expected_source_manifest_file_sha256=fixture["source_sha"],
        )

    terminal_path = next((truth_root / "states").glob("*.json"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["distance_rows"][0]["distance"] += 1.0
    terminal_path.write_bytes(canonical_json_bytes(terminal, pretty=True) + b"\n")
    with pytest.raises(ValueError, match="terminal file hash"):
        _load_supplemental_truth(
            ((truth_root, manifest["content_sha256"]),),
            allowed_state_ids=set(states_by_id),
            states_by_id=states_by_id,
            expected_model_family="deepsets_structured_marginal",
            expected_epoch_checkpoints={1: "c" * 64},
            expected_input_content_sha256=fixture["expected_input_sha"],
            expected_heldout_manifest_content_sha256=fixture["expected_heldout_sha"],
            expected_source_manifest_file_sha256=fixture["source_sha"],
        )


def test_formal_truth_seal_rejects_skipped_or_wrong_identity(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    schedule_root = tmp_path / "schedule"
    _materialize(fixture, schedule_root)
    truth_root = tmp_path / "truth"
    _write_truth_terminals(schedule_root, truth_root)
    terminal_path = next((truth_root / "states").glob("*.json"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["status"] = "SKIPPED_VARIABLE_HISTORY_LABEL_STATE"
    terminal_path.write_bytes(canonical_json_bytes(terminal, pretty=True) + b"\n")
    with pytest.raises(ValueError, match="identity or status"):
        seal_formal_truth_root(schedule_root=schedule_root, truth_root=truth_root)
