from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from causalcache.set_utility_contextual_inputs import (
    CONTEXTUAL_INPUT_STATUS,
    CONTEXTUAL_REQUIREMENT_STATUS,
)
from causalcache.set_utility_contextual_multisource_enrichment import (
    ContextualEnrichmentSource,
    materialize_contextual_multisource_enriched_inputs,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes


SCIENCE = "s" * 64
EXECUTION = "e" * 64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _content_manifest(value: dict) -> dict:
    result = dict(value)
    result["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def _fixture(root: Path) -> tuple[Path, tuple[ContextualEnrichmentSource, ...]]:
    base = root / "base"
    requirement = b'{"context_key":"x"}\n'
    requirement_path = base / "requirement-shards/shard-000-of-256.jsonl"
    requirement_path.parent.mkdir(parents=True)
    requirement_path.write_bytes(requirement)
    state = {
        "candidate_event_step_ids": [1, 2, 3],
        "distance_rows": [
            {"coalition_event_step_ids": [], "distance": 1.0},
            {"coalition_event_step_ids": [1, 2, 3], "distance": 0.0},
        ],
        "maximum_labeled_cardinality": 0,
        "role": "train",
        "state_id": "trajectory:decision:004",
        "trajectory_id": "trajectory",
    }
    state_payload = canonical_json_bytes(state) + b"\n"
    (base / "states.jsonl").write_bytes(state_payload)
    base_content = hashlib.sha256(
        state_payload + hashlib.sha256(requirement).hexdigest().encode("ascii")
    ).hexdigest()
    _write_json(
        base / "manifest.json",
        {
            "content_sha256": base_content,
            "evaluation_labels_included": False,
            "requirement_shards": [
                {
                    "path": "requirement-shards/shard-000-of-256.jsonl",
                    "sha256": hashlib.sha256(requirement).hexdigest(),
                    "status": CONTEXTUAL_REQUIREMENT_STATUS,
                }
            ],
            "state_count": 1,
            "states_jsonl": "states.jsonl",
            "states_sha256": hashlib.sha256(state_payload).hexdigest(),
            "status": CONTEXTUAL_INPUT_STATUS,
        },
    )
    sources = []
    definitions = (
        ("decision_v2", "COMPLETED_SET_UTILITY_TRAIN_ON_POLICY_SCHEDULES", [2], 0.4),
        ("long_oracle", "COMPLETED_SET_UTILITY_LONG_ORACLE_SCHEDULES", [3], 0.3),
    )
    for name, status, novel_subset, novel_distance in definitions:
        schedule_root = root / name / "schedule"
        schedule_row = {
            "candidate_event_ids": [1, 2, 3],
            "coalitions": [
                {"event_ids": [], "source": "empty"},
                {"event_ids": [1], "source": "shared"},
                {"event_ids": novel_subset, "source": "novel"},
                {"event_ids": [1, 2, 3], "source": "full"},
            ],
            "role": "train",
            "state_id": state["state_id"],
            "trajectory_id": state["trajectory_id"],
        }
        shard = schedule_root / "schedule-shards/shard-000-of-256.jsonl"
        shard.parent.mkdir(parents=True)
        shard.write_bytes(canonical_json_bytes(schedule_row) + b"\n")
        manifest = _content_manifest(
            {
                "config_sha256": name.ljust(64, "x"),
                "input_content_sha256": base_content if name == "decision_v2" else None,
                "scheduled_state_count": 1,
                "status": status,
            }
        )
        _write_json(schedule_root / "manifest.json", manifest)
        label_root = root / name / "labels"
        _write_json(
            label_root / "states/trajectory_decision_004.json",
            {
                "candidate_event_step_ids": [1, 2, 3],
                "distance_rows": [
                    {
                        "coalition_event_step_ids": [],
                        "distance": 1.0 + (1e-8 if name == "long_oracle" else 0.0),
                    },
                    {"coalition_event_step_ids": [1], "distance": 0.8},
                    {
                        "coalition_event_step_ids": novel_subset,
                        "distance": novel_distance,
                    },
                    {"coalition_event_step_ids": [1, 2, 3], "distance": 0.0},
                ],
                "execution_config_sha256": EXECUTION,
                "role": "train",
                "scientific_config_sha256": SCIENCE,
                "source_revision": "a" * 40,
                "state_id": state["state_id"],
                "status": "COMPLETED_VARIABLE_HISTORY_LABEL_STATE",
                "trajectory_id": state["trajectory_id"],
            },
        )
        (label_root / "states/._ignored.json").write_bytes(b"not-json")
        sources.append(
            ContextualEnrichmentSource(
                name=name,
                schedule_root=schedule_root,
                label_roots=(label_root,),
                expected_schedule_content_sha256=manifest["content_sha256"],
                expected_schedule_config_sha256=manifest["config_sha256"],
                expected_label_scientific_config_sha256=SCIENCE,
                expected_label_execution_config_sha256=EXECUTION,
            )
        )
    return base, tuple(sources)


def test_multisource_merge_deduplicates_and_preserves_firewall() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        base, sources = _fixture(root)
        manifest = materialize_contextual_multisource_enriched_inputs(
            base_input_root=base,
            sources=sources,
            output_root=root / "output",
            config_sha256="c" * 64,
        )
        state = json.loads((root / "output/states.jsonl").read_text())
        table = {
            tuple(row["coalition_event_step_ids"]): row["distance"]
            for row in state["distance_rows"]
        }
        assert table == {
            (): 1.0,
            (1,): 0.8,
            (2,): 0.4,
            (3,): 0.3,
            (1, 2, 3): 0.0,
        }
        assert manifest["evaluation_labels_included"] is False
        assert manifest["enrichment"]["added_distance_row_count"] == 3
        assert manifest["enrichment"]["duplicate_distance_row_count"] == 5
        assert manifest["enrichment"]["duplicate_max_abs_delta"] == pytest.approx(1e-8)
        assert set(manifest["enrichment"]["source_bindings"]) == {
            "decision_v2",
            "long_oracle",
        }


def test_multisource_merge_rejects_runtime_binding_drift() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        base, sources = _fixture(root)
        broken = ContextualEnrichmentSource(
            **{
                **sources[0].__dict__,
                "expected_label_execution_config_sha256": "z" * 64,
            }
        )
        with pytest.raises(ValueError, match="runtime binding drifted"):
            materialize_contextual_multisource_enriched_inputs(
                base_input_root=base,
                sources=(broken, sources[1]),
                output_root=root / "output",
                config_sha256="c" * 64,
            )
