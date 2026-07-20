"""Label-blind held-out feature snapshots for variable-history predictors."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.low_fidelity_v2 import LowFidelityEventV2, serialize_low_fidelity_v2
from causalcache.restoration_v2_baselines import (
    OCR_JACCARD_WEIGHT,
    RGB_HISTOGRAM_WEIGHT,
    joint_rgb_histogram,
    normalized_ocr_token_set,
)
from causalcache.set_utility_variable_history_inputs import (
    build_variable_history_queries_from_source_row,
)
from causalcache.set_utility_variable_history_training import (
    VARIABLE_HISTORY_NUMERIC_FEATURE_NAMES,
    variable_history_event_numeric_features,
)


EXPECTED_EXACT_STATE_COUNT = 320
EXPECTED_LARGE_HISTORY_STATE_COUNT = 720
EXPECTED_UNION_STATE_COUNT = 805
SOURCE_STATUS = "COMPLETED_VARIABLE_HISTORY_SOURCE"
INVENTORY_STATUS = "FROZEN_VARIABLE_HISTORY_STATE_INVENTORY"
OUTPUT_STATUS = "COMPLETED_SET_UTILITY_HELDOUT_FEATURE_SNAPSHOT"
PARTITION_OUTPUT_STATUS = "COMPLETED_SET_UTILITY_HELDOUT_FEATURE_PARTITION"
VISUAL_SHARD_STATUS = "COMPLETED_VARIABLE_HISTORY_TOKEN_SHARD"


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _sha256_file(path: Path) -> str:
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


def _text_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _visual_key(trajectory_id: str, observation_step: int) -> str:
    return f"{trajectory_id}:observation:{observation_step:03d}"


def _trajectory_id_from_state(state_id: str) -> str:
    trajectory_id, separator, decision = state_id.rpartition(":decision:")
    if (
        not separator
        or not trajectory_id
        or len(decision) != 3
        or not decision.isdigit()
        or int(decision) < 6
    ):
        raise ValueError("held-out state identity is invalid")
    return trajectory_id


def _track_membership(
    inventory: Mapping[str, Any],
    *,
    expected_exact_state_count: int,
    expected_large_history_state_count: int,
    expected_union_state_count: int,
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    if inventory.get("status") != INVENTORY_STATUS:
        raise ValueError("held-out state inventory is not frozen")
    tracks = inventory.get("evaluation_tracks")
    if not isinstance(tracks, Mapping):
        raise ValueError("held-out state inventory omits evaluation tracks")
    exact = tuple(tracks.get("exact_state_ids", ()))
    large = tuple(tracks.get("large_history_state_ids", ()))
    if (
        len(exact) != expected_exact_state_count
        or len(large) != expected_large_history_state_count
        or len(exact) != len(set(exact))
        or len(large) != len(set(large))
        or any(not isinstance(state_id, str) for state_id in (*exact, *large))
    ):
        raise ValueError("held-out evaluation track counts or identities drifted")
    union = set(exact) | set(large)
    if len(union) != expected_union_state_count:
        raise ValueError("held-out evaluation track union count drifted")
    exact_set = set(exact)
    large_set = set(large)
    membership = {
        state_id: tuple(
            name
            for name, selected in (
                ("exact_oracle", state_id in exact_set),
                ("large_history", state_id in large_set),
            )
            if selected
        )
        for state_id in sorted(union)
    }
    counts = {
        "exact_oracle": len(exact),
        "large_history": len(large),
        "overlap": len(exact_set & large_set),
        "union": len(union),
    }
    return membership, counts


def _row_count(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(value)


def _cosine_from_means(means: Any, event_step: int, current_step: int) -> float:
    left = means[event_step]
    right = means[current_step]
    try:
        value = float((left * right).sum().item())
    except (AttributeError, TypeError):
        value = math.fsum(
            float(a) * float(b) for a, b in zip(left, right, strict=True)
        )
    if not math.isfinite(value) or not -1.000001 <= value <= 1.000001:
        raise ValueError("held-out visual mean cosine is invalid")
    return min(1.0, max(-1.0, value))


def _histogram_cosine(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("held-out RGB histograms have incompatible geometry")
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("held-out RGB histogram norm must be positive")
    return min(1.0, max(0.0, dot / (left_norm * right_norm)))


def _set_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _feature_row(
    *,
    source: Mapping[str, Any],
    query: Any,
    logical_shard: int,
    membership: Sequence[str],
    means: Any,
    ocr_token_sets_by_step: Sequence[frozenset[str]],
    rgb_histograms_by_step: Sequence[Sequence[int]],
) -> tuple[dict[str, Any], dict[str, str]]:
    trajectory_id = str(source["source_id"])
    if query.role != "evaluation" or source["role"] != "evaluation":
        raise ValueError("held-out feature snapshot received a non-evaluation row")
    candidates = tuple(query.candidate_event_step_ids)
    current_step = query.decision_step_id - 1
    if candidates != tuple(range(1, current_step + 1)):
        raise ValueError("held-out candidates are not the complete prior history")
    if _row_count(means) != int(source["decision_count"]) + 1:
        raise ValueError("held-out visual means do not cover the source trajectory")

    history = {
        int(event["event_step_id"]): event
        for event in json.loads(source["history_events_json"])
    }
    if tuple(sorted(history)) != tuple(range(1, int(source["decision_count"]) + 1)):
        raise ValueError("held-out source history is incomplete")
    ocr = json.loads(source["ocr_records_json"])
    current_reference = (
        f"images/{trajectory_id}/observation-{current_step:03d}.png"
    )
    if query.current_observation_ref != current_reference:
        raise ValueError("held-out current observation reference drifted")
    try:
        current_ocr = ocr[current_reference]["full_spatial_tokens"]
    except (KeyError, TypeError) as error:
        raise ValueError("held-out current OCR record is incomplete") from error

    instruction = str(source["task_instruction"])
    instruction_key = _text_key(instruction)
    texts = {instruction_key: instruction}
    event_texts = []
    event_text_keys = []
    event_numeric_features = []
    ocr_rgb_scores = []
    for step in candidates:
        event = history[step]
        expected_reference = (
            f"images/{trajectory_id}/observation-{step:03d}.png"
        )
        if event["high_fidelity_observation_ref"] != expected_reference:
            raise ValueError("held-out event observation reference drifted")
        low = LowFidelityEventV2.from_mapping(event["low_fidelity_summary"])
        text = serialize_low_fidelity_v2(low).decode("utf-8").rstrip("\n")
        key = _text_key(text)
        prior = texts.setdefault(key, text)
        if prior != text:
            raise ValueError("held-out feature text SHA256 collision")
        event_texts.append(text)
        event_text_keys.append(key)
        try:
            event_ocr = ocr[expected_reference]["full_spatial_tokens"]
        except (KeyError, TypeError) as error:
            raise ValueError("held-out event OCR record is incomplete") from error
        numeric = variable_history_event_numeric_features(
            low,
            decision_step_id=query.decision_step_id,
            event_ocr_tokens=event_ocr,
            current_ocr_tokens=current_ocr,
            event_current_vlm_cosine=_cosine_from_means(
                means,
                step,
                current_step,
            ),
        )
        if len(numeric) != 11:
            raise RuntimeError("held-out numeric feature width drifted")
        event_numeric_features.append(list(numeric))
        score = (
            OCR_JACCARD_WEIGHT
            * _set_jaccard(
                ocr_token_sets_by_step[step],
                ocr_token_sets_by_step[current_step],
            )
            + RGB_HISTOGRAM_WEIGHT
            * _histogram_cosine(
                rgb_histograms_by_step[step],
                rgb_histograms_by_step[current_step],
            )
        )
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError("held-out OCR/RGB score is invalid")
        ocr_rgb_scores.append(score)

    row = {
        "candidate_event_step_ids": list(candidates),
        "current_image_key": _visual_key(trajectory_id, current_step),
        "decision_step_id": query.decision_step_id,
        "event_image_keys": [
            _visual_key(trajectory_id, step) for step in candidates
        ],
        "event_numeric_features": event_numeric_features,
        "event_text_keys": event_text_keys,
        "event_texts": event_texts,
        "instruction": instruction,
        "instruction_text_key": instruction_key,
        "logical_shard": logical_shard,
        "ocr_rgb_scores": ocr_rgb_scores,
        "role": "evaluation",
        "state_id": query.state_id,
        "track_membership": list(membership),
        "trajectory_id": trajectory_id,
    }
    if "distance_rows" in row:
        raise RuntimeError("held-out feature row contains forbidden labels")
    return row, texts


def materialize_heldout_feature_snapshot(
    *,
    source_root: Path,
    full_visual_token_root: Path,
    frozen_state_inventory_path: Path,
    output_root: Path,
    read_source_rows: Callable[[Path], Sequence[Mapping[str, Any]]],
    load_visual_tensors: Callable[[Path], Mapping[str, Any]],
    prepare_resized_rgb: Callable[[bytes], bytes],
    expected_exact_state_count: int = EXPECTED_EXACT_STATE_COUNT,
    expected_large_history_state_count: int = EXPECTED_LARGE_HISTORY_STATE_COUNT,
    expected_union_state_count: int = EXPECTED_UNION_STATE_COUNT,
    partial_from_available_visual_shards: bool = False,
) -> dict[str, Any]:
    """Materialize only label-blind inputs for the frozen held-out state union."""
    source_root = source_root.resolve()
    full_visual_token_root = full_visual_token_root.resolve()
    frozen_state_inventory_path = frozen_state_inventory_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError("held-out feature snapshot output already exists")

    inventory = _read_json(frozen_state_inventory_path)
    membership, track_counts = _track_membership(
        inventory,
        expected_exact_state_count=expected_exact_state_count,
        expected_large_history_state_count=expected_large_history_state_count,
        expected_union_state_count=expected_union_state_count,
    )
    target_trajectories = {
        _trajectory_id_from_state(state_id) for state_id in membership
    }

    source_manifest_path = source_root / "manifest.json"
    source_manifest = _read_json(source_manifest_path)
    finalized_visual_manifest_path = full_visual_token_root / "manifest.json"
    finalized_visual_manifest = (
        _read_json(finalized_visual_manifest_path)
        if finalized_visual_manifest_path.exists()
        else None
    )
    finalized_visual_shards = {
        int(row["logical_shard"]): row
        for row in (
            finalized_visual_manifest.get("shards", ())
            if finalized_visual_manifest is not None
            else ()
        )
        if row.get("kind") == "visual" and "logical_shard" in row
    }

    def visual_shard_is_available(logical_shard: int) -> bool:
        original = (
            full_visual_token_root
            / "token-shards"
            / f"shard-{logical_shard:03d}-of-256.safetensors"
        )
        receipt = (
            full_visual_token_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-256.json"
        )
        finalized = (
            full_visual_token_root
            / "visual-shards"
            / f"shard-{logical_shard:03d}-of-256.safetensors"
        )
        return (original.exists() and receipt.exists()) or (
            finalized.exists() and logical_shard in finalized_visual_shards
        )
    if (
        source_manifest.get("status") != SOURCE_STATUS
        or source_manifest.get("logical_shard_count") != 256
        or source_manifest.get("assignment_manifest_sha256")
        != inventory.get("assignment_manifest_sha256")
        or source_manifest.get("config_sha256") != inventory.get("config_sha256")
    ):
        raise ValueError("held-out source and frozen inventory identities differ")

    selected_source_shards = []
    covered_trajectories: set[str] = set()
    for shard in source_manifest.get("shards", ()):
        shard_trajectories = set(shard.get("trajectory_ids", ()))
        selected = shard_trajectories & target_trajectories
        if not selected:
            continue
        logical_shard = int(shard["logical_shard"])
        if (
            partial_from_available_visual_shards
            and not visual_shard_is_available(logical_shard)
        ):
            continue
        if covered_trajectories & selected:
            raise ValueError("held-out trajectory appears in multiple source shards")
        covered_trajectories.update(selected)
        source_path = (
            source_root
            / "trajectory-shards"
            / f"shard-{logical_shard:03d}-of-256.parquet"
        )
        if (
            source_path.stat().st_size != shard["byte_count"]
            or _sha256_file(source_path) != shard["sha256"]
        ):
            raise ValueError("held-out source shard bytes drifted")
        selected_source_shards.append((logical_shard, shard, source_path))
    if not covered_trajectories or (
        not partial_from_available_visual_shards
        and covered_trajectories != target_trajectories
    ):
        raise ValueError("held-out source manifest does not cover every target trajectory")
    active_membership = {
        state_id: tags
        for state_id, tags in membership.items()
        if _trajectory_id_from_state(state_id) in covered_trajectories
    }
    active_track_counts = {
        "exact_oracle": sum(
            "exact_oracle" in tags for tags in active_membership.values()
        ),
        "large_history": sum(
            "large_history" in tags for tags in active_membership.values()
        ),
        "overlap": sum(len(tags) == 2 for tags in active_membership.values()),
        "union": len(active_membership),
    }

    output_rows: dict[str, dict[str, Any]] = {}
    texts: dict[str, str] = {}
    visual_bindings = []
    source_bindings = []
    for logical_shard, source_shard, source_path in sorted(selected_source_shards):
        original_token_path = (
            full_visual_token_root
            / "token-shards"
            / f"shard-{logical_shard:03d}-of-256.safetensors"
        )
        receipt_path = (
            full_visual_token_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-256.json"
        )
        if original_token_path.exists() and receipt_path.exists():
            token_path = original_token_path
            receipt = _read_json(receipt_path)
            binding_source = "token_shard_receipt"
            binding_sha256 = _sha256_file(receipt_path)
        else:
            token_path = (
                full_visual_token_root
                / "visual-shards"
                / f"shard-{logical_shard:03d}-of-256.safetensors"
            )
            try:
                receipt = finalized_visual_shards[logical_shard]
            except KeyError as error:
                raise FileNotFoundError(
                    "held-out visual token shard is absent from both supported layouts"
                ) from error
            binding_source = "finalized_cache_manifest"
            binding_sha256 = _sha256_file(finalized_visual_manifest_path)
        token_sha256 = _sha256_file(token_path)
        if binding_source == "token_shard_receipt":
            valid_status = receipt.get("status") == VISUAL_SHARD_STATUS
        else:
            valid_status = finalized_visual_manifest.get("status") == (
                "COMPLETED_VARIABLE_HISTORY_TOKEN_CACHE"
            )
        if (
            not valid_status
            or receipt.get("logical_shard") != logical_shard
            or receipt.get("byte_count") != token_path.stat().st_size
            or receipt.get("sha256") != token_sha256
        ):
            raise ValueError("held-out full visual token shard receipt drifted")
        tensors = load_visual_tensors(token_path)
        source_rows = read_source_rows(source_path)
        for source in source_rows:
            trajectory_id = str(source["source_id"])
            if trajectory_id not in target_trajectories:
                continue
            if source["role"] != "evaluation":
                raise ValueError("held-out trajectory is not in the evaluation split")
            means_key = f"means__{trajectory_id}"
            if means_key not in tensors:
                raise ValueError("held-out visual token shard omits trajectory means")
            queries = {
                query.state_id: query
                for query in build_variable_history_queries_from_source_row(source)
            }
            images = source.get("images")
            if (
                not isinstance(images, Sequence)
                or isinstance(images, (str, bytes, bytearray, Mapping))
                or len(images) != int(source["decision_count"]) + 1
            ):
                raise ValueError("held-out source image inventory is incomplete")
            resized_rgb_by_step = []
            for image in images:
                payload = image.get("bytes") if isinstance(image, Mapping) else None
                if not isinstance(payload, bytes) or not payload:
                    raise ValueError("held-out source image payload is invalid")
                resized_rgb_by_step.append(prepare_resized_rgb(payload))
            rgb_histograms_by_step = tuple(
                joint_rgb_histogram(payload) for payload in resized_rgb_by_step
            )
            ocr_records = json.loads(source["ocr_records_json"])
            ocr_token_sets_by_step = tuple(
                normalized_ocr_token_set(
                    ocr_records[
                        f"images/{trajectory_id}/observation-{step:03d}.png"
                    ]["full_spatial_tokens"]
                )
                for step in range(int(source["decision_count"]) + 1)
            )
            for state_id in sorted(
                state_id
                for state_id in active_membership
                if _trajectory_id_from_state(state_id) == trajectory_id
            ):
                try:
                    query = queries[state_id]
                except KeyError as error:
                    raise ValueError(
                        "held-out frozen state is absent from its source trajectory"
                    ) from error
                row, row_texts = _feature_row(
                    source=source,
                    query=query,
                    logical_shard=logical_shard,
                    membership=active_membership[state_id],
                    means=tensors[means_key],
                    ocr_token_sets_by_step=ocr_token_sets_by_step,
                    rgb_histograms_by_step=rgb_histograms_by_step,
                )
                if state_id in output_rows:
                    raise ValueError("held-out state was materialized more than once")
                output_rows[state_id] = row
                for key, value in row_texts.items():
                    prior = texts.setdefault(key, value)
                    if prior != value:
                        raise ValueError("held-out feature text SHA256 collision")
        source_bindings.append(
            {
                "byte_count": source_shard["byte_count"],
                "logical_shard": logical_shard,
                "sha256": source_shard["sha256"],
            }
        )
        visual_bindings.append(
            {
                "byte_count": token_path.stat().st_size,
                "binding_sha256": binding_sha256,
                "binding_source": binding_source,
                "logical_shard": logical_shard,
                "sha256": token_sha256,
            }
        )

    if set(output_rows) != set(active_membership):
        raise RuntimeError("held-out feature snapshot did not materialize its frozen union")
    if any("distance_rows" in row for row in output_rows.values()):
        raise RuntimeError("held-out feature snapshot contains forbidden labels")
    state_payload = b"".join(
        _canonical_json(output_rows[state_id]) for state_id in sorted(output_rows)
    )
    text_payload = b"".join(
        _canonical_json({"key": key, "text": texts[key]}) for key in sorted(texts)
    )
    source_binding_sha256 = hashlib.sha256(
        b"".join(_canonical_json(value) for value in source_bindings)
    ).hexdigest()
    visual_binding_sha256 = hashlib.sha256(
        b"".join(_canonical_json(value) for value in visual_bindings)
    ).hexdigest()
    manifest = {
        "content_sha256": hashlib.sha256(state_payload + text_payload).hexdigest(),
        "distance_rows_included": False,
        "evaluation_labels_included": False,
        "evaluation_labels_loaded": False,
        "frozen_state_inventory_sha256": _sha256_file(
            frozen_state_inventory_path
        ),
        "input_bindings": {
            "assignment_manifest_sha256": source_manifest[
                "assignment_manifest_sha256"
            ],
            "config_sha256": source_manifest["config_sha256"],
            "source_manifest_sha256": _sha256_file(source_manifest_path),
            "source_selected_shards_sha256": source_binding_sha256,
            "state_identity_sha256": inventory["state_identity_sha256"],
            "visual_selected_shards_sha256": visual_binding_sha256,
        },
        "label_file_read_count": 0,
        "numeric_feature_names": list(VARIABLE_HISTORY_NUMERIC_FEATURE_NAMES),
        "ocr_rgb_profile": {
            "formula": "0.5*ocr_jaccard+0.5*joint_rgb_histogram_cosine",
            "rgb_histogram_bins_per_channel": 16,
            "rgb_resize": "256x256_bilinear_rgb",
            "scores_aligned_to": "candidate_event_step_ids",
        },
        "role_counts": {"evaluation": len(output_rows)},
        "schema_version": "1.0.0",
        "source_shards": source_bindings,
        "state_count": len(output_rows),
        "states_jsonl": "states.jsonl",
        "states_sha256": hashlib.sha256(state_payload).hexdigest(),
        "status": (
            PARTITION_OUTPUT_STATUS
            if partial_from_available_visual_shards
            else OUTPUT_STATUS
        ),
        "text_count": len(texts),
        "texts_jsonl": "texts.jsonl",
        "texts_sha256": hashlib.sha256(text_payload).hexdigest(),
        "track_counts": active_track_counts,
        "trajectory_count": len(covered_trajectories),
        "visual_shards": visual_bindings,
        "visual_token_profile": "full_480_target_actual_grid_bf16_4096",
    }
    output_root.mkdir(parents=True)
    _write_atomic(output_root / "states.jsonl", state_payload)
    _write_atomic(output_root / "texts.jsonl", text_payload)
    _write_atomic(output_root / "manifest.json", _canonical_json(manifest))
    return manifest


def merge_heldout_feature_snapshots(
    *, input_roots: Sequence[Path], output_root: Path
) -> dict[str, Any]:
    """Merge disjoint label-free host partitions into the frozen 805-state union."""
    roots = tuple(Path(root).resolve() for root in input_roots)
    output_root = output_root.resolve()
    if len(roots) < 2 or output_root.exists():
        raise ValueError("held-out snapshot merge requires multiple fresh partitions")
    manifests = [_read_json(root / "manifest.json") for root in roots]
    if any(
        manifest.get("status") != PARTITION_OUTPUT_STATUS
        or manifest.get("evaluation_labels_loaded") is not False
        or manifest.get("label_file_read_count") != 0
        for manifest in manifests
    ):
        raise ValueError("held-out feature partition status or label firewall drifted")
    stable_keys = (
        "frozen_state_inventory_sha256",
        "numeric_feature_names",
        "ocr_rgb_profile",
        "visual_token_profile",
    )
    for key in stable_keys:
        if len({_canonical_json(manifest[key]) for manifest in manifests}) != 1:
            raise ValueError(f"held-out feature partition binding drifted: {key}")
    input_binding_keys = (
        "assignment_manifest_sha256",
        "config_sha256",
        "source_manifest_sha256",
        "state_identity_sha256",
    )
    for key in input_binding_keys:
        if len(
            {manifest["input_bindings"][key] for manifest in manifests}
        ) != 1:
            raise ValueError(f"held-out feature source binding drifted: {key}")

    rows: dict[str, dict[str, Any]] = {}
    texts: dict[str, str] = {}
    for root, manifest in zip(roots, manifests, strict=True):
        state_path = root / manifest["states_jsonl"]
        text_path = root / manifest["texts_jsonl"]
        if (
            _sha256_file(state_path) != manifest["states_sha256"]
            or _sha256_file(text_path) != manifest["texts_sha256"]
        ):
            raise ValueError("held-out feature partition bytes drifted")
        for line in state_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row["state_id"] in rows or "distance_rows" in row:
                raise ValueError("held-out feature partitions overlap or contain labels")
            rows[row["state_id"]] = row
        for line in text_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            prior = texts.setdefault(row["key"], row["text"])
            if prior != row["text"]:
                raise ValueError("held-out feature partitions contain a text collision")
    if len(rows) != EXPECTED_UNION_STATE_COUNT:
        raise ValueError("merged held-out feature snapshot does not contain 805 states")
    track_counts = {
        "exact_oracle": sum(
            "exact_oracle" in row["track_membership"] for row in rows.values()
        ),
        "large_history": sum(
            "large_history" in row["track_membership"] for row in rows.values()
        ),
        "overlap": sum(len(row["track_membership"]) == 2 for row in rows.values()),
        "union": len(rows),
    }
    if track_counts != {
        "exact_oracle": EXPECTED_EXACT_STATE_COUNT,
        "large_history": EXPECTED_LARGE_HISTORY_STATE_COUNT,
        "overlap": 235,
        "union": EXPECTED_UNION_STATE_COUNT,
    }:
        raise ValueError("merged held-out feature track counts drifted")
    state_payload = b"".join(
        _canonical_json(rows[state_id]) for state_id in sorted(rows)
    )
    text_payload = b"".join(
        _canonical_json({"key": key, "text": texts[key]}) for key in sorted(texts)
    )
    source_shards = sorted(
        (row for manifest in manifests for row in manifest["source_shards"]),
        key=lambda row: row["logical_shard"],
    )
    visual_shards = sorted(
        (row for manifest in manifests for row in manifest["visual_shards"]),
        key=lambda row: row["logical_shard"],
    )
    if (
        len({row["logical_shard"] for row in source_shards}) != len(source_shards)
        or len({row["logical_shard"] for row in visual_shards}) != len(visual_shards)
    ):
        raise ValueError("held-out feature partition shard bindings overlap")
    source_binding_sha256 = hashlib.sha256(
        b"".join(_canonical_json(value) for value in source_shards)
    ).hexdigest()
    visual_binding_sha256 = hashlib.sha256(
        b"".join(_canonical_json(value) for value in visual_shards)
    ).hexdigest()
    base = manifests[0]
    manifest = {
        "content_sha256": hashlib.sha256(state_payload + text_payload).hexdigest(),
        "distance_rows_included": False,
        "evaluation_labels_included": False,
        "evaluation_labels_loaded": False,
        "frozen_state_inventory_sha256": base["frozen_state_inventory_sha256"],
        "input_bindings": {
            **{
                key: base["input_bindings"][key] for key in input_binding_keys
            },
            "source_selected_shards_sha256": source_binding_sha256,
            "visual_selected_shards_sha256": visual_binding_sha256,
        },
        "label_file_read_count": 0,
        "merged_partition_content_sha256s": sorted(
            manifest["content_sha256"] for manifest in manifests
        ),
        "numeric_feature_names": base["numeric_feature_names"],
        "ocr_rgb_profile": base["ocr_rgb_profile"],
        "role_counts": {"evaluation": len(rows)},
        "schema_version": "1.0.0",
        "source_shards": source_shards,
        "state_count": len(rows),
        "states_jsonl": "states.jsonl",
        "states_sha256": hashlib.sha256(state_payload).hexdigest(),
        "status": OUTPUT_STATUS,
        "text_count": len(texts),
        "texts_jsonl": "texts.jsonl",
        "texts_sha256": hashlib.sha256(text_payload).hexdigest(),
        "track_counts": track_counts,
        "trajectory_count": len({row["trajectory_id"] for row in rows.values()}),
        "visual_shards": visual_shards,
        "visual_token_profile": base["visual_token_profile"],
    }
    output_root.mkdir(parents=True)
    _write_atomic(output_root / "states.jsonl", state_payload)
    _write_atomic(output_root / "texts.jsonl", text_payload)
    _write_atomic(output_root / "manifest.json", _canonical_json(manifest))
    return manifest


__all__ = [
    "EXPECTED_EXACT_STATE_COUNT",
    "EXPECTED_LARGE_HISTORY_STATE_COUNT",
    "EXPECTED_UNION_STATE_COUNT",
    "OUTPUT_STATUS",
    "PARTITION_OUTPUT_STATUS",
    "materialize_heldout_feature_snapshot",
    "merge_heldout_feature_snapshots",
]
