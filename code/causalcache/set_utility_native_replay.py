"""Post-GO native-action replay contracts for frozen held-out selectors."""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import parse_gui_owl_v2_1_output
from causalcache.set_utility_evaluation_schedule import (
    EXACT_TRACK,
    EvaluationScheduleConfig,
    SelectorState,
    evaluation_tracks_from_inventory,
    selector_states_from_payload,
    source_trajectory_shards,
)
from causalcache.set_utility_heldout_evaluation import COMPLETED_EVALUATION


NATIVE_REPLAY_SCHEMA_VERSION = "1.0.0"
NATIVE_REPLAY_METHODS = ("winner", "recent", "ocr_rgb")
NATIVE_REPLAY_STATUS = "FROZEN_SET_UTILITY_NATIVE_REPLAY_SCHEDULE"
NATIVE_REPLAY_SHARD_STATUS = "FROZEN_SET_UTILITY_NATIVE_REPLAY_SCHEDULE_SHARD"
COMPLETED_LABEL_STATUS = "COMPLETED_VARIABLE_HISTORY_LABEL_STATE"
_TARGET_ACTIONS = frozenset(("click", "long_press", "swipe", "system_button"))


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _target(action: GUIOwlV2Action) -> tuple[Any, ...] | None:
    if action.action not in _TARGET_ACTIONS:
        return None
    if action.action in {"click", "long_press"}:
        return ("point", action.coordinate)
    if action.action == "swipe":
        return ("swipe", action.coordinate, action.coordinate2)
    return ("system_button", action.button)


def compare_action_components(
    reference: GUIOwlV2Action,
    prediction: GUIOwlV2Action,
) -> dict[str, Any]:
    """Compare canonical, action-type, target, and NFKC-exact text components."""
    if not isinstance(reference, GUIOwlV2Action) or not isinstance(
        prediction, GUIOwlV2Action
    ):
        raise TypeError("action comparison requires canonical GUIOwlV2Action values")
    reference_target = _target(reference)
    predicted_target = _target(prediction)
    text_applicable = reference.text is not None
    text_exact = None
    if text_applicable:
        predicted_text = (
            unicodedata.normalize("NFKC", prediction.text)
            if prediction.text is not None
            else None
        )
        text_exact = predicted_text == unicodedata.normalize("NFKC", reference.text)
    return {
        "canonical_action_exact": prediction == reference,
        "action_type_exact": prediction.action == reference.action,
        "target_applicable": reference_target is not None,
        "target_exact": (
            predicted_target == reference_target
            if reference_target is not None
            else None
        ),
        "text_applicable": text_applicable,
        "text_nfkc_exact": text_exact,
    }


def signed_content_hash_is_valid(payload: Mapping[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    if not isinstance(claimed, str):
        return False
    unsigned = dict(payload)
    del unsigned["content_sha256"]
    return hashlib.sha256(canonical_json_bytes(unsigned)[:-1]).hexdigest() == claimed


def _canonical_subset(
    value: Any,
    *,
    candidates: tuple[int, ...],
    budget: int,
    label: str,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be an event-id array")
    subset = tuple(value)
    if (
        any(type(item) is not int for item in subset)
        or subset != tuple(sorted(subset))
        or len(subset) != len(set(subset))
        or len(subset) > budget
        or not set(subset).issubset(candidates)
    ):
        raise ValueError(f"{label} is not a canonical at-most-{budget} subset")
    return subset


def build_native_replay_state(
    state: SelectorState,
    *,
    winner_model: str,
    budgets: Sequence[int],
    reference_terminal: Mapping[str, Any],
    reference_terminal_sha256: str,
) -> dict[str, Any]:
    """Build one exact-track replay state and deduplicate selected coalitions."""
    if EXACT_TRACK not in state.tracks:
        raise ValueError("native replay v1 accepts exact-track states only")
    if winner_model not in state.methods:
        raise ValueError("held-out winner is absent from selector methods")
    normalized_budgets = tuple(budgets)
    if normalized_budgets != (1, 2, 3, 4):
        raise ValueError("native replay budgets must be exactly 1,2,3,4")
    if reference_terminal.get("status") != COMPLETED_LABEL_STATUS:
        raise ValueError("native replay reference terminal is not completed")
    if (
        reference_terminal.get("state_id") != state.state_id
        or reference_terminal.get("trajectory_id") != state.trajectory_id
        or tuple(reference_terminal.get("candidate_event_step_ids", ()))
        != state.candidate_event_ids
    ):
        raise ValueError("native replay reference terminal identity drifted")
    reference = reference_terminal.get("reference")
    if not isinstance(reference, Mapping):
        raise ValueError("native replay terminal omits its full-history reference")
    serialized_action = reference.get("serialized_action")
    if not isinstance(serialized_action, str):
        raise ValueError("native replay terminal omits its serialized reference action")
    canonical_action = parse_gui_owl_v2_1_output(serialized_action).canonical_action

    tags: dict[tuple[int, ...], set[str]] = {}
    method_mapping = {
        "winner": winner_model,
        "recent": "recent",
        "ocr_rgb": "ocr_rgb",
    }
    for replay_method, selector_method in method_mapping.items():
        if selector_method not in state.methods:
            raise ValueError(f"native replay method is absent: {selector_method}")
        for budget in normalized_budgets:
            subset = _canonical_subset(
                state.methods[selector_method][budget],
                candidates=state.candidate_event_ids,
                budget=budget,
                label=f"{selector_method} B{budget}",
            )
            tags.setdefault(subset, set()).add(f"{replay_method}:B{budget}")
    coalitions = [
        {"event_ids": list(subset), "sources": sorted(sources)}
        for subset, sources in sorted(
            tags.items(), key=lambda item: (len(item[0]), item[0])
        )
    ]
    return {
        "candidate_event_ids": list(state.candidate_event_ids),
        "coalitions": coalitions,
        "logical_shard": state.logical_shard,
        "reference": {
            "canonical_action": canonical_action.arguments(),
            "serialized_action": serialized_action,
            "terminal_sha256": reference_terminal_sha256,
        },
        "role": "evaluation",
        "state_id": state.state_id,
        "trajectory_id": state.trajectory_id,
        "winner_model": winner_model,
    }


def _terminal_files(
    label_roots: Sequence[Path], state_ids: Sequence[str]
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for root in label_roots:
        states_root = root / "states"
        if not states_root.is_dir():
            raise FileNotFoundError(states_root)
        for state_id in state_ids:
            path = states_root / f"{state_id.replace(':', '_')}.json"
            if not path.is_file():
                continue
            if state_id in files:
                raise ValueError(f"duplicate label terminal state: {state_id}")
            files[state_id] = path
    return files


def materialize_native_replay_schedule(
    *,
    config_path: Path,
    inventory_path: Path,
    selections_path: Path,
    heldout_result_path: Path,
    source_manifest_path: Path,
    label_roots: Sequence[Path],
    output_root: Path,
) -> dict[str, Any]:
    """Freeze the 320-state post-GO action-replay schedule."""
    if output_root.exists():
        raise FileExistsError(output_root)
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = EvaluationScheduleConfig.from_mapping(config_payload)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    tracks = evaluation_tracks_from_inventory(inventory, config=config)
    selections = json.loads(selections_path.read_text(encoding="utf-8"))
    states = selector_states_from_payload(selections, config=config, tracks=tracks)
    result = json.loads(heldout_result_path.read_text(encoding="utf-8"))
    config_sha = sha256_file(config_path)
    selections_sha = sha256_file(selections_path)
    if (
        selections.get("config_sha256") != config_sha
        or not signed_content_hash_is_valid(selections)
    ):
        raise ValueError("sealed held-out selections are not valid for this config")
    if (
        result.get("status") != COMPLETED_EVALUATION
        or result.get("coverage", {}).get("complete") is not True
        or result.get("winner", {}).get("verdict") != "GO"
        or not signed_content_hash_is_valid(result)
    ):
        raise ValueError("held-out result does not authorize native replay")
    if (
        result.get("bindings", {}).get("config_sha256") != config_sha
        or result.get("bindings", {}).get("selections_sha256") != selections_sha
    ):
        raise ValueError("held-out result input bindings drifted")
    winner_model = result["winner"].get("model")
    if not isinstance(winner_model, str) or winner_model not in selections.get(
        "model_artifacts", {}
    ):
        raise ValueError("held-out winner model is invalid")

    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    _, trajectory_shards = source_trajectory_shards(
        source, logical_shard_count=config.logical_shard_count
    )
    exact_states = tuple(state for state in states if EXACT_TRACK in state.tracks)
    if len(exact_states) != config.exact_state_count or len(exact_states) != 320:
        raise ValueError("native replay stage must contain the frozen exact-track 320")
    terminals = _terminal_files(
        label_roots, tuple(state.state_id for state in exact_states)
    )
    rows_by_shard: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(config.logical_shard_count)
    }
    terminal_hashes = {}
    for state in exact_states:
        if trajectory_shards.get(state.trajectory_id) != state.logical_shard:
            raise ValueError("native replay state/source logical shard drifted")
        terminal_path = terminals.get(state.state_id)
        if terminal_path is None:
            raise ValueError(
                f"native replay reference terminal is missing: {state.state_id}"
            )
        terminal_sha = sha256_file(terminal_path)
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        if terminal.get("scientific_config_sha256") != config_payload[
            "restoration_truth"
        ]["reference_config_sha256"]:
            raise ValueError("native replay terminal reference profile drifted")
        terminal_hashes[state.state_id] = terminal_sha
        rows_by_shard[state.logical_shard].append(
            build_native_replay_state(
                state,
                winner_model=winner_model,
                budgets=config.budgets,
                reference_terminal=terminal,
                reference_terminal_sha256=terminal_sha,
            )
        )

    bindings = {
        "config_sha256": config_sha,
        "heldout_result_sha256": sha256_file(heldout_result_path),
        "inventory_sha256": sha256_file(inventory_path),
        "reference_terminal_inventory_sha256": hashlib.sha256(
            canonical_json_bytes(terminal_hashes)[:-1]
        ).hexdigest(),
        "selections_sha256": selections_sha,
        "source_manifest_sha256": sha256_file(source_manifest_path),
    }
    output_root.mkdir(parents=True)
    total_coalitions = 0
    shard_records = []
    for logical_shard in range(config.logical_shard_count):
        rows = sorted(rows_by_shard[logical_shard], key=lambda item: item["state_id"])
        payload = b"".join(canonical_json_bytes(row) for row in rows)
        schedule_path = (
            output_root
            / "schedule-shards"
            / f"shard-{logical_shard:03d}-of-{config.logical_shard_count}.jsonl"
        )
        _write_atomic(schedule_path, payload)
        coalition_count = sum(len(row["coalitions"]) for row in rows)
        total_coalitions += coalition_count
        receipt = {
            **bindings,
            "coalition_count": coalition_count,
            "logical_shard": logical_shard,
            "schedule_sha256": hashlib.sha256(payload).hexdigest(),
            "state_count": len(rows),
            "status": NATIVE_REPLAY_SHARD_STATUS,
        }
        receipt_payload = canonical_json_bytes(receipt, pretty=True)
        _write_atomic(
            output_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-{config.logical_shard_count}.json",
            receipt_payload,
        )
        shard_records.append(
            {
                "coalition_count": coalition_count,
                "logical_shard": logical_shard,
                "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
                "schedule_sha256": receipt["schedule_sha256"],
                "state_count": len(rows),
            }
        )
    manifest = {
        **bindings,
        "budgets": list(config.budgets),
        "coalition_count_after_deduplication": total_coalitions,
        "logical_shard_count": config.logical_shard_count,
        "methods": list(NATIVE_REPLAY_METHODS),
        "schema_version": NATIVE_REPLAY_SCHEMA_VERSION,
        "shards": shard_records,
        "state_count": len(exact_states),
        "status": NATIVE_REPLAY_STATUS,
        "winner_model": winner_model,
    }
    manifest["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)[:-1]
    ).hexdigest()
    _write_atomic(
        output_root / "manifest.json", canonical_json_bytes(manifest, pretty=True)
    )
    return manifest


__all__ = [
    "COMPLETED_LABEL_STATUS",
    "NATIVE_REPLAY_METHODS",
    "NATIVE_REPLAY_SCHEMA_VERSION",
    "NATIVE_REPLAY_STATUS",
    "build_native_replay_state",
    "canonical_json_bytes",
    "compare_action_components",
    "materialize_native_replay_schedule",
    "sha256_file",
    "signed_content_hash_is_valid",
]
