from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from causalcache.data.guiodyssey_independent import SourceFileSpec
from causalcache.set_utility_consumed_ledger import EXPECTED_PARTITION_COUNTS
from causalcache.set_utility_full_pool import (
    FORMAT_SAFETY_EXCLUSION,
    PROTOCOL_ID,
    UNCONSUMED_MARKER,
    build_full_pool_census,
    build_full_pool_inspection_config,
    candidate_capacity_stratum,
    parse_consumed_ledger,
    supported_candidate_counts,
    validate_complete_source_manifest,
)
from causalcache.set_utility_full_pool_inventory_v1 import (
    COMPLETE_STATUS,
    EXPECTED_DIRECTORY,
    EXPECTED_REPO_ID,
    EXPECTED_REVISION,
    FROZEN_CONFIG_SHA256,
    PROTOCOL_ID as INVENTORY_PROTOCOL_ID,
    canonical_json_bytes,
    sha256_bytes,
)
from scripts.materialize_set_utility_full_pool import ParquetRowFactory


ROOT = Path(__file__).resolve().parents[2]
PNG = b"\x89PNG\r\n\x1a\n" + b"full-pool-fixture"
LEDGER_PATH = ROOT / "data/manifests/set_utility_consumed_identity_ledger_v1.json"


def _base_config() -> dict:
    return json.loads(
        (ROOT / "code/configs/independent_reference_gate_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _path(index: int) -> str:
    return f"mobile/use/train/shard-{index:05d}-of-00610.parquet"


def _source_manifest() -> dict:
    records = [
        {
            "path": _path(index),
            "size_bytes": index + 1,
            "lfs_sha256": f"{index % 16:x}" * 64,
        }
        for index in range(610)
    ]
    return {
        "schema_version": "1.0.0",
        "protocol_id": INVENTORY_PROTOCOL_ID,
        "status": COMPLETE_STATUS,
        "source": {
            "repo_id": EXPECTED_REPO_ID,
            "repo_type": "dataset",
            "revision": EXPECTED_REVISION,
            "target_directory": EXPECTED_DIRECTORY,
            "config_sha256": FROZEN_CONFIG_SHA256,
        },
        "inventory": {
            "file_count": 610,
            "total_size_bytes": sum(range(1, 611)),
            "files_sha256": sha256_bytes(canonical_json_bytes(records)),
            "files": records,
        },
        "authorization_scope": {
            "remote_metadata_inventory_only": True,
            "file_download_or_row_decode": False,
            "semantic_census_or_role_assignment": False,
            "restoration_labels_or_training": False,
            "model_or_gpu_operations": False,
            "hugging_face_mutation": False,
        },
    }


def _tap() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "tap",
            "arguments": {"coordinate": [200, 300], "clicks": 1},
        },
    }


def _row(
    source_id: str,
    *,
    decision_count: int,
    instruction: str,
    apps: tuple[str, ...] = ("Settings",),
) -> dict:
    messages = []
    action_count = decision_count + 1
    for index in range(action_count):
        user_content = [{"type": "image", "index": index}]
        if index == 0:
            user_content.append({"type": "text", "text": instruction})
        messages.append({"role": "user", "content": user_content})
        tool_calls = [_tap()]
        if index == action_count - 1:
            tool_calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": "terminate",
                        "arguments": {"status": "success"},
                    },
                }
            )
        messages.append({"role": "assistant", "content": [], "tool_calls": tool_calls})
    return {
        "messages": json.dumps(messages),
        "metadata": json.dumps(
            {
                "platform": "mobile",
                "others": {"source_id": source_id, "apps": list(apps)},
            }
        ),
        "images": [{"bytes": PNG, "path": None} for _ in range(action_count)],
    }


def _consumed_binding():
    payload = LEDGER_PATH.read_bytes()
    return parse_consumed_ledger(
        json.loads(payload), manifest_sha256=hashlib.sha256(payload).hexdigest()
    )


def _build(
    rows_by_file: dict[str, list[dict]],
    *,
    ceiling: int | None = None,
    include_consumed: bool = True,
    consumed_decision_overrides: dict[str, int] | None = None,
) -> dict:
    base = _base_config()
    source = _source_manifest()
    consumed = _consumed_binding()
    combined = {path: list(rows) for path, rows in rows_by_file.items()}
    if include_consumed:
        consumed_rows = []
        overrides = consumed_decision_overrides or {}
        for source_id, partition in consumed.source_to_role:
            consumed_rows.append(
                _row(
                    source_id,
                    decision_count=overrides.get(source_id, 6),
                    instruction=f"Historical fixture {partition} {source_id}",
                    apps=("Settings", partition),
                )
            )
        combined.setdefault(_path(0), [])[:0] = consumed_rows

    def row_factory(spec: SourceFileSpec):
        return tuple(enumerate(combined.get(spec.transport_file, ())))

    return build_full_pool_census(
        base_config=base,
        source_manifest=source,
        row_iterator_factory=row_factory,
        base_config_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        consumed_ledger=consumed,
        format_safety_maximum_decisions=ceiling,
    )


def test_complete_source_manifest_requires_exact_canonical_p1_shape() -> None:
    manifest = _source_manifest()
    specs = validate_complete_source_manifest(manifest, base_config=_base_config())
    assert len(specs) == 610
    assert specs[0].transport_file == _path(0)
    assert specs[-1].transport_file == _path(609)

    missing = copy.deepcopy(manifest)
    missing["inventory"]["files"].pop()
    missing["inventory"]["file_count"] = 609
    with pytest.raises(ValueError, match="exactly 610"):
        validate_complete_source_manifest(missing, base_config=_base_config())

    changed = copy.deepcopy(manifest)
    changed["inventory"]["files"][0]["size_bytes"] = 9
    with pytest.raises(ValueError, match="files_sha256 mismatch"):
        validate_complete_source_manifest(changed, base_config=_base_config())

    wrong_protocol = copy.deepcopy(manifest)
    wrong_protocol["protocol_id"] = "invented_flat_manifest"
    with pytest.raises(ValueError, match="identity drifted"):
        validate_complete_source_manifest(wrong_protocol, base_config=_base_config())


def test_unbounded_parser_adapter_does_not_reuse_the_old_maximum() -> None:
    base = _base_config()
    config = build_full_pool_inspection_config(
        base, transport_files=tuple(_path(index) for index in range(610))
    )
    assert config["eligibility"]["minimum_decisions_per_trajectory"] == 6
    assert config["eligibility"]["maximum_decisions_per_trajectory"] > 64
    assert base["eligibility"]["maximum_decisions_per_trajectory"] == 12


@pytest.mark.parametrize(
    ("decision_count", "stratum", "candidate_counts"),
    (
        (6, "decisions_6_9", (4,)),
        (9, "decisions_6_9", (4,)),
        (10, "decisions_10_17", (4, 8)),
        (17, "decisions_10_17", (4, 8)),
        (18, "decisions_18_plus", (4, 8, 16)),
        (100, "decisions_18_plus", (4, 8, 16)),
    ),
)
def test_candidate_capacity_strata(
    decision_count: int, stratum: str, candidate_counts: tuple[int, ...]
) -> None:
    assert candidate_capacity_stratum(decision_count) == stratum
    assert supported_candidate_counts(decision_count) == candidate_counts


def test_census_excludes_all_consumed_and_emits_group_firewall() -> None:
    consumed = _consumed_binding()
    short_id, short_partition = consumed.source_to_role[0]
    manifest = _build(
        {
            _path(309): [
                _row(
                    "new-trajectory-10",
                    decision_count=10,
                    instruction="Open the browser",
                    apps=("Firefox",),
                )
            ],
            _path(609): [
                _row(
                    "new-trajectory-18",
                    decision_count=18,
                    instruction="Send a private fixture message",
                    apps=("Gmail",),
                )
            ],
        },
        consumed_decision_overrides={short_id: 5},
    )
    assert manifest["protocol_id"] == PROTOCOL_ID
    assert manifest["selection"]["candidate_count"] == 2
    assert {row["source_id"] for row in manifest["pool"]["trajectories"]} == {
        "new-trajectory-10",
        "new-trajectory-18",
    }
    assert all(
        row["consumed_role_marker"] == UNCONSUMED_MARKER
        for row in manifest["pool"]["trajectories"]
    )
    assert manifest["consumed_ledger"]["observed_partition_counts"] == (
        EXPECTED_PARTITION_COUNTS
    )
    candidate_exclusions = manifest["source"][
        "consumed_candidate_exclusion_counts_by_partition"
    ]
    assert candidate_exclusions[short_partition] == (
        EXPECTED_PARTITION_COUNTS[short_partition] - 1
    )
    assert manifest["source"]["consumed_parser_exclusion_counts_by_partition"] == {
        short_partition: 1
    }
    audit = manifest["consumed_group_audit"]
    assert audit["record_count"] == 107
    assert audit["contains_raw_instruction"] is False
    assert set(audit["records"][0]) == {
        "source_id",
        "partition",
        "instruction_app_group_sha256",
    }
    serialized = json.dumps(manifest, sort_keys=True)
    assert "Historical fixture" not in serialized
    assert "Open the browser" not in serialized
    assert set(manifest["operation_counts"].values()) == {0}


def test_default_accepts_long_rows_but_explicit_format_ceiling_is_named() -> None:
    row = _row("new-long-trajectory", decision_count=65, instruction="Long fixture")
    unbounded = _build({_path(17): [row]})
    assert unbounded["selection"]["candidate_count"] == 1
    bounded = _build({_path(17): [row]}, ceiling=64)
    assert bounded["selection"]["candidate_count"] == 0
    assert bounded["source"]["parser_exclusion_counts"] == {
        FORMAT_SAFETY_EXCLUSION: 1
    }


def test_duplicate_source_identity_across_shards_fails_closed() -> None:
    duplicate = _row("new-duplicate", decision_count=6, instruction="Duplicate")
    with pytest.raises(ValueError, match="duplicate source_id"):
        _build({_path(1): [duplicate], _path(609): [duplicate]})


def test_all_consumed_identities_must_exist_in_complete_scan() -> None:
    with pytest.raises(ValueError, match="absent from the complete source scan"):
        _build({}, include_consumed=False)


def test_parquet_row_factory_verifies_bytes_then_streams_fixture(tmp_path: Path) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    pyarrow = pytest.importorskip("pyarrow")
    relative = _path(0)
    path = tmp_path.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True)
    parquet.write_table(
        pyarrow.Table.from_pylist(
            [{"messages": "[]", "metadata": "{}", "images": []}]
        ),
        path,
    )
    payload = path.read_bytes()
    spec = SourceFileSpec(relative, len(payload), hashlib.sha256(payload).hexdigest())
    assert list(ParquetRowFactory(tmp_path, batch_size=1)(spec)) == [
        (0, {"messages": "[]", "metadata": "{}", "images": []})
    ]
    changed = SourceFileSpec(relative, len(payload) + 1, spec.sha256)
    with pytest.raises(ValueError, match="size mismatch"):
        list(ParquetRowFactory(tmp_path)(changed))
