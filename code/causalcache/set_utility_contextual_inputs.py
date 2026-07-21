"""Label-blind contextual entity inputs for utility predictor v3."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_evaluation import canonical_json_bytes, sha256_file
from causalcache.set_utility_heldout_snapshot import FULL_EVALUATION_OUTPUT_STATUS


CONTEXTUAL_PROFILE_ID = "gui_owl_final_lm_hidden_image_plus_preceding_text64_v1"
CONTEXTUAL_INPUT_STATUS = "COMPLETED_SET_UTILITY_CONTEXTUAL_INPUT_SNAPSHOT"
CONTEXTUAL_REQUIREMENT_STATUS = "COMPLETED_SET_UTILITY_CONTEXTUAL_REQUIREMENT_SHARD"
SUPPORTED_INPUT_STATUS = "COMPLETED_MERGED_VARIABLE_HISTORY_TRAINING_INPUT_SNAPSHOT"
TRAIN_TUNE_ROLE_SCOPE = "train_tune"
EVALUATION_ROLE_SCOPE = "evaluation"
_VISUAL_KEY = re.compile(r"(?P<trajectory>[0-9]+):observation:(?P<step>[0-9]{3})")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def contextual_input_lineage_sha256s(manifest: Mapping[str, Any]) -> frozenset[str]:
    values = {
        value
        for value in (
            manifest.get("content_sha256"),
            manifest.get("parent_content_sha256"),
        )
        if isinstance(value, str)
    }
    ancestors = manifest.get("ancestor_content_sha256s", ())
    if not isinstance(ancestors, (list, tuple)) or any(
        not isinstance(value, str) for value in ancestors
    ):
        raise ValueError("contextual input ancestor lineage is invalid")
    values.update(ancestors)
    return frozenset(values)


def contextual_prompt_text(
    *, role: str, instruction: str, event_text: str | None = None
) -> str:
    if role not in {"query", "event"}:
        raise ValueError("contextual entity role must be query or event")
    if not isinstance(instruction, str) or not instruction:
        raise ValueError("contextual entity instruction must be non-empty")
    if role == "query":
        if event_text is not None:
            raise ValueError("query contextual entity cannot contain event text")
        body = "Entity role: current GUI query observation."
    else:
        if not isinstance(event_text, str) or not event_text:
            raise ValueError("event contextual entity requires low-fidelity text")
        body = f"Entity role: historical GUI event.\nEvent summary:\n{event_text}"
    return f"Task instruction:\n{instruction}\n\n{body}"


def contextual_key(
    *, role: str, prompt_text: str, source_visual_key: str
) -> str:
    payload = {
        "profile": CONTEXTUAL_PROFILE_ID,
        "prompt_text": prompt_text,
        "role": role,
        "source_visual_key": source_visual_key,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _visual_identity(value: Any, *, trajectory_id: str) -> tuple[str, int]:
    if not isinstance(value, str):
        raise ValueError("contextual source visual key must be a string")
    match = _VISUAL_KEY.fullmatch(value)
    if match is None or match.group("trajectory") != trajectory_id:
        raise ValueError("contextual source visual identity drifted")
    return value, int(match.group("step"))


def materialize_contextual_inputs(
    *,
    input_root: Path,
    output_root: Path,
    role_scope: str = TRAIN_TUNE_ROLE_SCOPE,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("contextual input output already exists")
    input_manifest_path = input_root / "manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if role_scope == TRAIN_TUNE_ROLE_SCOPE:
        expected_status = SUPPORTED_INPUT_STATUS
        allowed_roles = {"train", "tune"}
    elif role_scope == EVALUATION_ROLE_SCOPE:
        expected_status = FULL_EVALUATION_OUTPUT_STATUS
        allowed_roles = {"evaluation"}
    else:
        raise ValueError("contextual input role scope is invalid")
    if (
        input_manifest.get("status") != expected_status
        or input_manifest.get("evaluation_labels_included") is not False
    ):
        raise ValueError("contextual input snapshot status or label firewall drifted")
    if role_scope == EVALUATION_ROLE_SCOPE and (
        input_manifest.get("evaluation_labels_loaded") is not False
        or input_manifest.get("distance_rows_included") is not False
        or input_manifest.get("label_file_read_count") != 0
        or input_manifest.get("evaluation_scope")
        != "all_assignment_evaluation_states"
    ):
        raise ValueError("evaluation contextual input crossed the truth firewall")
    states_path = input_root / input_manifest["states_jsonl"]
    if sha256_file(states_path) != input_manifest["states_sha256"]:
        raise ValueError("contextual source states drifted")
    states = tuple(
        json.loads(line)
        for line in states_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    if len(states) != input_manifest["state_count"]:
        raise ValueError("contextual source state count drifted")

    requirements: dict[str, dict[str, Any]] = {}
    transformed = []
    for state in states:
        if state.get("role") not in allowed_roles:
            raise ValueError("contextual snapshot crossed its frozen role firewall")
        if "distance_rows" in state or "distance" in state or "utility" in state:
            raise ValueError("contextual snapshot contains forbidden truth fields")
        trajectory_id = state["trajectory_id"]
        logical_shard = state["logical_shard"]
        if type(logical_shard) is not int or not 0 <= logical_shard < 256:
            raise ValueError("contextual state logical shard drifted")
        query_visual, query_step = _visual_identity(
            state["current_image_key"], trajectory_id=trajectory_id
        )
        query_text = contextual_prompt_text(
            role="query", instruction=state["instruction"]
        )
        query_key = contextual_key(
            role="query", prompt_text=query_text, source_visual_key=query_visual
        )
        query_requirement = {
            "context_key": query_key,
            "entity_role": "query",
            "logical_shard": logical_shard,
            "observation_step": query_step,
            "prompt_text": query_text,
            "source_visual_key": query_visual,
            "trajectory_id": trajectory_id,
        }
        prior = requirements.setdefault(query_key, query_requirement)
        if prior != query_requirement:
            raise ValueError("contextual query-key collision")

        event_keys = []
        for visual_key, event_text in zip(
            state["event_image_keys"], state["event_texts"], strict=True
        ):
            source_visual, event_step = _visual_identity(
                visual_key, trajectory_id=trajectory_id
            )
            prompt_text = contextual_prompt_text(
                role="event",
                instruction=state["instruction"],
                event_text=event_text,
            )
            key = contextual_key(
                role="event",
                prompt_text=prompt_text,
                source_visual_key=source_visual,
            )
            requirement = {
                "context_key": key,
                "entity_role": "event",
                "logical_shard": logical_shard,
                "observation_step": event_step,
                "prompt_text": prompt_text,
                "source_visual_key": source_visual,
                "trajectory_id": trajectory_id,
            }
            prior = requirements.setdefault(key, requirement)
            if prior != requirement:
                raise ValueError("contextual event-key collision")
            event_keys.append(key)
        if len(event_keys) != len(state["candidate_event_step_ids"]):
            raise ValueError("contextual event inventory drifted")
        transformed.append(
            {
                **state,
                "contextual_profile_id": CONTEXTUAL_PROFILE_ID,
                "current_image_key": query_key,
                "event_image_keys": event_keys,
                "event_text_keys": event_keys,
                "instruction_text_key": query_key,
                "source_current_image_key": query_visual,
                "source_event_image_keys": list(state["event_image_keys"]),
            }
        )

    by_shard = {index: [] for index in range(256)}
    for row in requirements.values():
        by_shard[row["logical_shard"]].append(row)
    requirement_receipts = []
    for logical_shard in range(256):
        rows = sorted(by_shard[logical_shard], key=lambda row: row["context_key"])
        payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        relative = f"requirement-shards/shard-{logical_shard:03d}-of-256.jsonl"
        _write_atomic(output_root / relative, payload)
        requirement_receipts.append(
            {
                "byte_count": len(payload),
                "context_count": len(rows),
                "logical_shard": logical_shard,
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "status": CONTEXTUAL_REQUIREMENT_STATUS,
            }
        )
    state_payload = b"".join(
        canonical_json_bytes(row) + b"\n"
        for row in sorted(transformed, key=lambda row: row["state_id"])
    )
    _write_atomic(output_root / "states.jsonl", state_payload)
    role_counts = Counter(row["role"] for row in transformed)
    entity_role_counts = Counter(row["entity_role"] for row in requirements.values())
    manifest = {
        "content_sha256": hashlib.sha256(
            state_payload
            + "".join(row["sha256"] for row in requirement_receipts).encode("ascii")
        ).hexdigest(),
        "context_count": len(requirements),
        "contextual_profile_id": CONTEXTUAL_PROFILE_ID,
        "entity_role_counts": dict(sorted(entity_role_counts.items())),
        "evaluation_labels_included": False,
        "input_content_sha256": input_manifest["content_sha256"],
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "requirement_shards": requirement_receipts,
        "role_counts": dict(sorted(role_counts.items())),
        "role_scope": role_scope,
        "schema_version": "1.0.0",
        "state_count": len(transformed),
        "states_jsonl": "states.jsonl",
        "states_sha256": hashlib.sha256(state_payload).hexdigest(),
        "status": CONTEXTUAL_INPUT_STATUS,
        "trajectory_count": len({row["trajectory_id"] for row in transformed}),
    }
    if role_scope == EVALUATION_ROLE_SCOPE:
        exposure_path = input_root / input_manifest["exposure_slices_json"]
        if sha256_file(exposure_path) != input_manifest["exposure_slices_sha256"]:
            raise ValueError("evaluation contextual exposure slice bytes drifted")
        exposure_payload = exposure_path.read_bytes()
        _write_atomic(output_root / "exposure-slices.json", exposure_payload)
        manifest.update(
            {
                "distance_rows_included": False,
                "evaluation_labels_loaded": False,
                "evaluation_scope": "all_assignment_evaluation_states",
                "exposure_slices_json": "exposure-slices.json",
                "exposure_slices_sha256": input_manifest[
                    "exposure_slices_sha256"
                ],
                "historical_exposure_counts": input_manifest[
                    "historical_exposure_counts"
                ],
                "label_file_read_count": 0,
            }
        )
    _write_atomic(
        output_root / "manifest.json", canonical_json_bytes(manifest, pretty=True)
    )
    return manifest


__all__ = [
    "CONTEXTUAL_INPUT_STATUS",
    "CONTEXTUAL_PROFILE_ID",
    "CONTEXTUAL_REQUIREMENT_STATUS",
    "EVALUATION_ROLE_SCOPE",
    "TRAIN_TUNE_ROLE_SCOPE",
    "contextual_input_lineage_sha256s",
    "contextual_key",
    "contextual_prompt_text",
    "materialize_contextual_inputs",
]
