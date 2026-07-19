from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from causalcache.data.guiodyssey_independent import (
    SourceFileSpec,
    normalize_app_label,
    trajectory_selection_sha256,
)
from causalcache.set_utility_full_pool import (
    SELECTION_SALT,
    build_full_pool_inspection_config,
)
from causalcache.set_utility_long_pool import instruction_app_group_sha256
from causalcache.set_utility_processor_substrate import (
    SourceProvenance,
    ValidatedOcrBatch,
    build_selected_pilot,
    build_selected_row_read_plan,
    build_trajectory_processor_substrate,
    load_selected_rows_once,
)


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = ROOT / "code/configs/independent_reference_gate_v1.json"
PNG = b"\x89PNG\r\n\x1a\n" + b"fixture"


def _row(source_id: str, *, instruction: str, decision_count: int = 6) -> dict:
    action_count = decision_count + 1
    messages: list[dict] = []
    for index in range(action_count):
        user_content = [{"type": "image", "index": index}]
        if index == 0:
            user_content.append({"type": "text", "text": instruction})
        messages.append({"role": "user", "content": user_content})
        tool_calls = [{"function": {"name": "wait", "arguments": {}}}]
        if index == action_count - 1:
            tool_calls.append(
                {
                    "function": {
                        "name": "terminate",
                        "arguments": {"status": "success"},
                    }
                }
            )
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "action_description", "text": f"Wait {index}."}
                ],
                "tool_calls": tool_calls,
            }
        )
    return {
        "images": [
            {"bytes": PNG + bytes([index]), "path": None}
            for index in range(action_count)
        ],
        "messages": json.dumps(messages),
        "metadata": json.dumps(
            {
                "platform": "mobile",
                "others": {
                    "source_id": source_id,
                    "apps": ["Fixture App"],
                    "device_name": "fixture-device",
                    "resolution": [720, 1280],
                },
            }
        ),
    }


def _assignment(
    source_id: str,
    *,
    transport_file: str,
    transport_row_index: int,
    instruction: str,
    decision_count: int = 6,
) -> dict:
    normalized_apps = (normalize_app_label("Fixture App"),)
    return {
        "candidate_capacity_stratum": "decisions_6_9",
        "decision_count": decision_count,
        "instruction_app_group_sha256": instruction_app_group_sha256(
            instruction,
            normalized_app_labels=normalized_apps,
        ),
        "p0_selection_sha256": trajectory_selection_sha256(
            source_id,
            salt=SELECTION_SALT,
        ),
        "role": "train",
        "source_id": source_id,
        "terminal_decision_step_id": decision_count + 1,
        "trajectory_id": source_id,
        "transport_file": transport_file,
        "transport_row_index": transport_row_index,
    }


def _inspection_config(transport_files: tuple[str, ...]) -> dict:
    base = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    return build_full_pool_inspection_config(
        base,
        transport_files=transport_files,
    )


def _source_files() -> tuple[SourceFileSpec, ...]:
    return (
        SourceFileSpec("mobile/use/train/a.parquet", 100, "a" * 64),
        SourceFileSpec("mobile/use/train/b.parquet", 200, "b" * 64),
    )


def test_committed_v2_assignments_group_selected_rows_without_decoding() -> None:
    manifest = json.loads(
        (
            ROOT
            / "data/manifests/set_utility_freeze_b_v2_terminal_index_repair.json"
        ).read_text(encoding="utf-8")
    )
    source = json.loads(
        (ROOT / "data/manifests/set_utility_full_pool_inventory_v1.json").read_text(
            encoding="utf-8"
        )
    )
    census = json.loads(
        (ROOT / "data/manifests/set_utility_full_pool_census_v2.json").read_text(
            encoding="utf-8"
        )
    )
    source_files = tuple(
        SourceFileSpec(record["path"], record["size_bytes"], record["lfs_sha256"])
        for record in source["inventory"]["files"]
    )
    plan = build_selected_row_read_plan(
        manifest["assignments"],
        source_files=source_files,
        source_row_counts=census["source"]["row_counts_by_file"],
    )

    expected_shards = {
        record["transport_file"] for record in manifest["assignments"]
    }
    assert plan.assignment_count == 1200
    assert len(plan.shards) == len(expected_shards)
    assert {shard.source_file.transport_file for shard in plan.shards} == expected_shards
    assert sum(len(shard.assignments) for shard in plan.shards) == 1200
    assert sum(
        record["decision_count"] + 1 for record in manifest["assignments"]
    ) == 18_792


def test_selected_loader_reads_each_selected_shard_once_and_sequentially() -> None:
    files = _source_files()
    assignments = (
        _assignment(
            "source-a0",
            transport_file=files[0].transport_file,
            transport_row_index=0,
            instruction="First task",
        ),
        _assignment(
            "source-a2",
            transport_file=files[0].transport_file,
            transport_row_index=2,
            instruction="Second task",
        ),
        _assignment(
            "source-b1",
            transport_file=files[1].transport_file,
            transport_row_index=1,
            instruction="Third task",
        ),
    )
    plan = build_selected_row_read_plan(
        assignments,
        source_files=files,
        source_row_counts={files[0].transport_file: 3, files[1].transport_file: 2},
    )
    rows_by_path = {
        files[0].transport_file: (
            _row("source-a0", instruction="First task"),
            {},
            _row("source-a2", instruction="Second task"),
        ),
        files[1].transport_file: (
            {},
            _row("source-b1", instruction="Third task"),
        ),
    }
    calls: Counter[str] = Counter()

    def rows(spec: SourceFileSpec):
        calls[spec.transport_file] += 1
        return enumerate(rows_by_path[spec.transport_file])

    loaded = load_selected_rows_once(
        plan,
        row_iterator_factory=rows,
        inspection_config=_inspection_config(
            tuple(file.transport_file for file in files)
        ),
    )

    assert calls == Counter({files[0].transport_file: 1, files[1].transport_file: 1})
    assert tuple(item.assignment.source_id for item in loaded) == (
        "source-a0",
        "source-a2",
        "source-b1",
    )


def test_selected_loader_fails_on_transport_or_raw_identity_drift() -> None:
    source_file = _source_files()[0]
    assignment = _assignment(
        "expected-source",
        transport_file=source_file.transport_file,
        transport_row_index=0,
        instruction="Expected task",
    )
    plan = build_selected_row_read_plan(
        (assignment,),
        source_files=(source_file,),
        source_row_counts={source_file.transport_file: 1},
    )
    with pytest.raises(ValueError, match="raw-row/P0 assignment identity drifted"):
        load_selected_rows_once(
            plan,
            row_iterator_factory=lambda spec: enumerate(
                (_row("different-source", instruction="Expected task"),)
            ),
            inspection_config=_inspection_config((source_file.transport_file,)),
        )

    duplicate = copy.deepcopy(assignment)
    duplicate["source_id"] = "another-source"
    duplicate["trajectory_id"] = "another-source"
    duplicate["p0_selection_sha256"] = trajectory_selection_sha256(
        "another-source", salt=SELECTION_SALT
    )
    with pytest.raises(ValueError, match="transport rows are not unique"):
        build_selected_row_read_plan(
            (assignment, duplicate),
            source_files=(source_file,),
            source_row_counts={source_file.transport_file: 1},
        )


def _query(
    source_id: str,
    *,
    kind: str,
    decision_step_id: int,
) -> dict:
    current = decision_step_id - 1
    candidates = tuple(range(1, current))[-16:]
    return {
        "candidate_capacity_stratum": "decisions_6_9",
        "current_equivalent_event_step_id": current,
        "decision_step_id": decision_step_id,
        "initial_candidate_count": len(candidates),
        "initial_candidate_event_step_ids": list(candidates),
        "maximum_labeled_cardinality": 2,
        "processor_candidate_freeze_status": "PENDING_SEPARATE_EXECUTION",
        "query_kind": kind,
        "role": "train",
        "source_id": source_id,
        "state_id": f"{source_id}:decision:{decision_step_id:03d}",
        "trajectory_id": source_id,
    }


def test_arbitrary_terminal_builds_full_history_with_injected_ocr_only() -> None:
    source_file = _source_files()[0]
    assignment = _assignment(
        "terminal-source",
        transport_file=source_file.transport_file,
        transport_row_index=0,
        instruction="Complete all six waits",
    )
    plan = build_selected_row_read_plan(
        (assignment,),
        source_files=(source_file,),
        source_row_counts={source_file.transport_file: 1},
    )
    loaded = load_selected_rows_once(
        plan,
        row_iterator_factory=lambda spec: enumerate(
            (_row("terminal-source", instruction="Complete all six waits"),)
        ),
        inspection_config=_inspection_config((source_file.transport_file,)),
    )[0]
    pilot = build_selected_pilot(
        loaded,
        provenance=SourceProvenance(
            upstream_repo="cua-lite/GUIOdyssey",
            upstream_revision="e" * 40,
            transport_repo="cua-lite/GUIOdyssey",
            transport_revision="e" * 40,
            derived_repo="owner/set-utility-substrate",
        ),
    )
    ocr_calls: list[tuple[str, ...]] = []
    derived_calls: list[int] = []

    def fake_ocr(images: dict[str, bytes]) -> ValidatedOcrBatch:
        ocr_calls.append(tuple(images))
        return ValidatedOcrBatch(
            records_by_path={path: {"image_member_path": path} for path in images},
            prepared_by_path={path: object() for path in images},
        )

    def fake_derived(event: dict, **kwargs) -> dict:
        step_id = event["step_id"]
        derived_calls.append(step_id)
        assert set(kwargs["image_payloads"]) == set(kwargs["ocr_records_by_path"])
        return {
            "step_id": step_id,
            "low_fidelity_v2": {
                "step_id": step_id,
                "action_type": "wait",
                "action_argument": "wait",
                "foreground_app": "unknown",
                "screen_text_added": [f"screen-{step_id}"],
                "screen_text_removed": [],
                "screen_change": "low",
                "executor_result": "unknown",
            },
            "high_fidelity_v2": {
                "content_type": "image",
                "image_role": "post_action_state",
                "image_member_path": event["observation_after_path"],
            },
        }

    substrate = build_trajectory_processor_substrate(
        pilot,
        query_states=(
            _query("terminal-source", kind="stratum_anchor", decision_step_id=6),
            _query("terminal-source", kind="terminal", decision_step_id=7),
        ),
        ocr_batch_builder=fake_ocr,
        derived_event_builder=fake_derived,
    )

    assert len(ocr_calls) == 1
    assert len(ocr_calls[0]) == 7
    assert derived_calls == [1, 2, 3, 4, 5, 6]
    terminal = next(query for query in substrate.query_states if query.query_kind == "terminal")
    anchor = next(
        query for query in substrate.query_states if query.query_kind == "stratum_anchor"
    )
    assert terminal.decision_step_id == assignment["decision_count"] + 1
    assert len(substrate.utility_history_for_query(terminal)) == 6
    assert tuple(
        event.event_step_id for event in substrate.utility_history_for_query(anchor)
    ) == (1, 2, 3, 4, 5)
    payload = substrate.to_payload()
    assert payload["terminal_decision_step_id"] == 7
    assert payload["query_states"][1]["current_observation_ref"].endswith(
        "observation-006.png"
    )
    assert "utility" not in json.dumps(payload).casefold()
    with pytest.raises(ValueError, match="terminal query is not decision_count"):
        build_trajectory_processor_substrate(
            pilot,
            query_states=(
                _query(
                    "terminal-source",
                    kind="stratum_anchor",
                    decision_step_id=6,
                ),
                _query("terminal-source", kind="terminal", decision_step_id=6),
            ),
            ocr_batch_builder=fake_ocr,
            derived_event_builder=fake_derived,
        )


def test_query_topology_rejects_missing_anchor_future_or_cardinality_drift() -> None:
    source_file = _source_files()[0]
    assignment = _assignment(
        "query-contract-source",
        transport_file=source_file.transport_file,
        transport_row_index=0,
        instruction="Complete all six waits",
    )
    plan = build_selected_row_read_plan(
        (assignment,),
        source_files=(source_file,),
        source_row_counts={source_file.transport_file: 1},
    )
    loaded = load_selected_rows_once(
        plan,
        row_iterator_factory=lambda spec: enumerate(
            (_row("query-contract-source", instruction="Complete all six waits"),)
        ),
        inspection_config=_inspection_config((source_file.transport_file,)),
    )[0]
    pilot = build_selected_pilot(
        loaded,
        provenance=SourceProvenance(
            upstream_repo="cua-lite/GUIOdyssey",
            upstream_revision="e" * 40,
            transport_repo="cua-lite/GUIOdyssey",
            transport_revision="e" * 40,
            derived_repo="owner/set-utility-substrate",
        ),
    )

    def fake_ocr(images: dict[str, bytes]) -> ValidatedOcrBatch:
        return ValidatedOcrBatch(
            records_by_path={path: {"image_member_path": path} for path in images},
            prepared_by_path={path: object() for path in images},
        )

    def fake_derived(event: dict, **kwargs) -> dict:
        step_id = event["step_id"]
        return {
            "step_id": step_id,
            "low_fidelity_v2": {
                "step_id": step_id,
                "action_type": "wait",
                "action_argument": "wait",
                "foreground_app": "unknown",
                "screen_text_added": [],
                "screen_text_removed": [],
                "screen_change": "low",
                "executor_result": "unknown",
            },
            "high_fidelity_v2": {
                "content_type": "image",
                "image_role": "post_action_state",
                "image_member_path": event["observation_after_path"],
            },
        }

    terminal = _query(
        "query-contract-source", kind="terminal", decision_step_id=7
    )
    with pytest.raises(ValueError, match="exactly one anchor and one terminal"):
        build_trajectory_processor_substrate(
            pilot,
            query_states=(terminal, copy.deepcopy(terminal)),
            ocr_batch_builder=fake_ocr,
            derived_event_builder=fake_derived,
        )

    late_anchor = _query(
        "query-contract-source", kind="stratum_anchor", decision_step_id=7
    )
    with pytest.raises(ValueError, match="stratum-anchor decision step drifted"):
        build_trajectory_processor_substrate(
            pilot,
            query_states=(late_anchor, terminal),
            ocr_batch_builder=fake_ocr,
            derived_event_builder=fake_derived,
        )

    bad_cardinality = _query(
        "query-contract-source", kind="stratum_anchor", decision_step_id=6
    )
    bad_cardinality["maximum_labeled_cardinality"] = 3
    with pytest.raises(ValueError, match="cardinality must equal two"):
        build_trajectory_processor_substrate(
            pilot,
            query_states=(bad_cardinality, terminal),
            ocr_batch_builder=fake_ocr,
            derived_event_builder=fake_derived,
        )
