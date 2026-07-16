"""Policy-vision feature scoring and post-selection restoration evaluation."""

from __future__ import annotations

import hashlib
import io
import math
import statistics
import tarfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.restoration_v2_2_geometry_stats import (
    StateMetricRow,
    paired_trajectory_bootstrap,
)
from causalcache.restoration_v2_2_ocr_rgb import COMPARATOR_METHODS
from causalcache.restoration_v2_2_ocr_rgb_contract import (
    canonical_json_bytes,
    sha256_file,
    strict_json_object_bytes,
)
from causalcache.restoration_v2_2_selector_geometry import (
    SelectorGeometryState,
    load_selector_geometry_states,
)
from causalcache.restoration_v2_baselines import (
    CANDIDATE_EVENT_STEP_IDS,
    SCORE_TIE_ABS_TOL,
    SCORE_TIE_REL_TOL,
    select_top_two,
)


SCHEMA_VERSION = "1.0.0"
STATUS = "COMPLETED_RESTORATION_V2_2_POLICY_VISION_BASELINE_V1"
PRIMARY_EVENT_IDS = CANDIDATE_EVENT_STEP_IDS
PRIMARY_BUDGET = 2
PRIMARY_DECISION_STEP = 6
NORMALIZATION_EPSILON = 1e-12
EXPECTED_STATE_COUNT = 15
EXPECTED_TRAIN_COUNT = 10
EXPECTED_DEVELOPMENT_COUNT = 5
EXPECTED_UNIQUE_IMAGE_COUNT = 75
EXPECTED_CANDIDATE_SCORE_COUNT = 60
EXPECTED_LABEL_STATE_COUNT = 45
EXPECTED_LABEL_DISTANCE_ROW_COUNT = 420
EXPECTED_PRIMARY_DISTANCE_ROW_COUNT = 240
OCR_RGB_METHOD = "ocr_rgb"
POLICY_VISION_METHOD = "policy_vision"
OUTLIER_TRAJECTORY_ID = "0141544666483837"


@dataclass(frozen=True)
class PolicyVisionImageIdentity:
    """One immutable raw screenshot identity passed to a feature worker."""

    image_member_path: str
    image_sha256: str


@dataclass(frozen=True)
class PolicyVisionWorkItem:
    """A feature-only state plan that deliberately contains no D(S) values."""

    primary_ordinal: int
    state_index: int
    role: str
    trajectory_id: str
    state_id: str
    event_images: tuple[tuple[int, PolicyVisionImageIdentity], ...]
    current_image: PolicyVisionImageIdentity

    def event_image_map(self) -> dict[int, PolicyVisionImageIdentity]:
        return dict(self.event_images)


def _safe_member_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a safe relative path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or str(parsed) != value
        or any(part in {".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"{label} must be a safe relative path")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _canonical_jsonl_records(
    path: Path,
    *,
    expected_count: int,
    label: str,
) -> tuple[dict[str, Any], ...]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{label} must be non-empty newline-terminated JSONL")
    lines = payload.splitlines(keepends=True)
    if len(lines) != expected_count:
        raise ValueError(f"{label} row count drifted")
    records = []
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            raise ValueError(f"{label}[{index}] is not newline terminated")
        record = strict_json_object_bytes(line[:-1], label=f"{label}[{index}]")
        if canonical_json_bytes(record) + b"\n" != line:
            raise ValueError(f"{label}[{index}] is not canonical JSONL")
        records.append(record)
    return tuple(records)


def load_identity_witness(
    contract: Any,
) -> tuple[tuple[PolicyVisionWorkItem, ...], dict[str, dict[str, Any]]]:
    """Load only state/image identities from the committed valid OCR sibling."""
    witness = contract.data["immutable_inputs"]["ocr_rgb_identity_witness"]
    path = (
        contract.repository_root
        / witness["directory"]
        / "state_scores.jsonl"
    )
    records = _canonical_jsonl_records(
        path,
        expected_count=EXPECTED_STATE_COUNT,
        label="OCR/RGB identity witness",
    )
    work_items: list[PolicyVisionWorkItem] = []
    witness_by_state: dict[str, dict[str, Any]] = {}
    image_paths: set[str] = set()
    image_hashes: dict[str, str] = {}
    for ordinal, record in enumerate(records):
        state = record.get("state")
        candidates = record.get("candidate_scores")
        current = record.get("current_image")
        if (
            record.get("protocol_id") != witness["protocol_id"]
            or not isinstance(state, Mapping)
            or not isinstance(candidates, list)
            or not isinstance(current, Mapping)
        ):
            raise ValueError("OCR/RGB identity witness schema drifted")
        if (
            state.get("role") not in ("v2_label_train", "v2_development")
            or state.get("decision_step_id") != PRIMARY_DECISION_STEP
            or state.get("candidate_event_step_ids") != list(PRIMARY_EVENT_IDS)
            or state.get("budget_event_capacity") != PRIMARY_BUDGET
            or type(state.get("index")) is not int
        ):
            raise ValueError("OCR/RGB identity witness primary scope drifted")
        trajectory_id = state.get("trajectory_id")
        state_id = state.get("state_id")
        if (
            not isinstance(trajectory_id, str)
            or state_id != f"{trajectory_id}:decision_step:006"
            or state_id in witness_by_state
        ):
            raise ValueError("OCR/RGB identity witness state identity drifted")
        event_images: list[tuple[int, PolicyVisionImageIdentity]] = []
        if len(candidates) != len(PRIMARY_EVENT_IDS):
            raise ValueError("OCR/RGB identity witness candidate denominator drifted")
        for expected_step, candidate in zip(
            PRIMARY_EVENT_IDS,
            candidates,
            strict=True,
        ):
            if (
                not isinstance(candidate, Mapping)
                or candidate.get("event_step_id") != expected_step
            ):
                raise ValueError("OCR/RGB identity witness candidate order drifted")
            identity = PolicyVisionImageIdentity(
                image_member_path=_safe_member_path(
                    candidate.get("post_image_member_path"),
                    "candidate image",
                ),
                image_sha256=str(candidate.get("post_image_sha256")),
            )
            event_images.append((expected_step, identity))
        current_identity = PolicyVisionImageIdentity(
            image_member_path=_safe_member_path(
                current.get("image_member_path"),
                "current image",
            ),
            image_sha256=str(current.get("image_sha256")),
        )
        for identity in (
            *(item[1] for item in event_images),
            current_identity,
        ):
            if len(identity.image_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in identity.image_sha256
            ):
                raise ValueError("OCR/RGB identity witness image SHA256 drifted")
            previous = image_hashes.setdefault(
                identity.image_member_path,
                identity.image_sha256,
            )
            if previous != identity.image_sha256:
                raise ValueError("one image path has inconsistent SHA256 identities")
            image_paths.add(identity.image_member_path)
        work_items.append(
            PolicyVisionWorkItem(
                primary_ordinal=ordinal,
                state_index=state["index"],
                role=state["role"],
                trajectory_id=trajectory_id,
                state_id=state_id,
                event_images=tuple(event_images),
                current_image=current_identity,
            )
        )
        witness_by_state[state_id] = record
    if len(image_paths) != EXPECTED_UNIQUE_IMAGE_COUNT:
        raise ValueError("policy-vision identity witness must contain 75 unique images")
    if Counter(item.role for item in work_items) != {
        "v2_label_train": EXPECTED_TRAIN_COUNT,
        "v2_development": EXPECTED_DEVELOPMENT_COUNT,
    }:
        raise ValueError("policy-vision identity witness role denominator drifted")
    if [item.state_index for item in work_items] != list(range(2, 45, 3)):
        raise ValueError("policy-vision primary state-index inventory drifted")
    return tuple(work_items), witness_by_state


def feature_worker_shards(
    work_items: Sequence[PolicyVisionWorkItem],
) -> tuple[tuple[PolicyVisionWorkItem, ...], tuple[PolicyVisionWorkItem, ...]]:
    """Freeze two disjoint no-DDP shards by the original primary state parity."""
    if len(work_items) != EXPECTED_STATE_COUNT:
        raise ValueError("policy-vision sharding requires exactly 15 work items")
    worker_zero = tuple(item for item in work_items if item.state_index % 2 == 0)
    worker_one = tuple(item for item in work_items if item.state_index % 2 == 1)
    if len(worker_zero) != 8 or len(worker_one) != 7:
        raise ValueError("policy-vision worker shard sizes must be 8 and 7")
    if set(worker_zero).intersection(worker_one) or set(
        (*worker_zero, *worker_one)
    ) != set(work_items):
        raise ValueError("policy-vision worker shards must be an exact disjoint union")
    return worker_zero, worker_one


def _expected_worker_fields(item: PolicyVisionWorkItem) -> tuple[str, str]:
    if item.state_index % 2 == 0:
        return "even", "cuda:0"
    return "odd", "cuda:1"


def validate_feature_worker_provenance(
    feature_records: Sequence[Mapping[str, Any]],
    work_items: Sequence[PolicyVisionWorkItem],
) -> None:
    """Bind every recorded state back to its frozen parity worker."""
    item_by_state = {item.state_id: item for item in work_items}
    if len(item_by_state) != EXPECTED_STATE_COUNT:
        raise ValueError("policy-vision provenance requires 15 unique work items")
    bindings: dict[str, set[tuple[str, str]]] = {"even": set(), "odd": set()}
    seen: set[str] = set()
    for record in feature_records:
        state = record.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("policy-vision provenance record lacks state metadata")
        state_id = state.get("state_id")
        if not isinstance(state_id, str) or state_id not in item_by_state:
            raise ValueError("policy-vision provenance state identity drifted")
        if state_id in seen:
            raise ValueError("policy-vision provenance contains a duplicate state")
        seen.add(state_id)
        item = item_by_state[state_id]
        expected_state = {
            "index": item.state_index,
            "role": item.role,
            "trajectory_id": item.trajectory_id,
            "state_id": item.state_id,
        }
        worker_id, device = _expected_worker_fields(item)
        gpu_uuid = record.get("gpu_uuid")
        if (
            dict(state) != expected_state
            or record.get("worker_id") != worker_id
            or record.get("device") != device
            or not isinstance(gpu_uuid, str)
            or not gpu_uuid
        ):
            raise ValueError("policy-vision row-to-worker provenance drifted")
        bindings[worker_id].add((device, gpu_uuid))
    if seen != set(item_by_state):
        raise ValueError("policy-vision provenance state inventory drifted")
    if any(len(values) != 1 for values in bindings.values()):
        raise ValueError("policy-vision parity worker binding is not unique")
    if len({next(iter(values))[1] for values in bindings.values()}) != 2:
        raise ValueError("policy-vision parity workers must use distinct GPU UUIDs")


def extract_worker_image_payloads(
    *,
    image_tar: Path,
    work_items: Sequence[PolicyVisionWorkItem],
    expected_tar_member_count: int,
) -> dict[str, bytes]:
    """Extract only one worker's allowlisted screenshots and verify every digest."""
    required: dict[str, str] = {}
    for item in work_items:
        identities = (*item.event_image_map().values(), item.current_image)
        for identity in identities:
            previous = required.setdefault(
                identity.image_member_path,
                identity.image_sha256,
            )
            if previous != identity.image_sha256:
                raise ValueError("worker image requirements contain inconsistent hashes")
    payloads: dict[str, bytes] = {}
    with tarfile.open(image_tar, mode="r:") as archive:
        members = archive.getmembers()
        if len(members) != expected_tar_member_count:
            raise ValueError("derived image tar member denominator drifted")
        seen: set[str] = set()
        for member in members:
            name = _safe_member_path(member.name, "derived image member")
            if name in seen or not member.isfile() or member.issym() or member.islnk():
                raise ValueError("derived image tar contains a duplicate/non-regular member")
            seen.add(name)
            if name not in required:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read selected image member: {name}")
            payload = handle.read()
            if hashlib.sha256(payload).hexdigest() != required[name]:
                raise ValueError(f"selected image SHA256 drifted: {name}")
            payloads[name] = payload
    if set(payloads) != set(required):
        raise ValueError("selected worker image payload coverage drifted")
    return payloads


def decode_rgb_images(
    item: PolicyVisionWorkItem,
    payloads: Mapping[str, bytes],
) -> tuple[Any, ...]:
    """Decode one state in the frozen event-1..4/current order."""
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise RuntimeError("policy-vision image decoding requires Pillow") from error
    identities = (
        *(item.event_image_map()[step] for step in PRIMARY_EVENT_IDS),
        item.current_image,
    )
    images = []
    try:
        for identity in identities:
            payload = payloads.get(identity.image_member_path)
            if payload is None:
                raise ValueError("policy-vision payload mapping lacks a required image")
            source = Image.open(io.BytesIO(payload))
            source.load()
            converted = source.convert("RGB")
            if converted is not source:
                source.close()
            images.append(converted)
        if len(images) != 5 or any(image.mode != "RGB" for image in images):
            raise ValueError("policy-vision state must decode to exactly five RGB images")
        return tuple(images)
    except BaseException:
        for image in images:
            image.close()
        raise


def close_images(images: Sequence[Any]) -> None:
    for image in images:
        image.close()


def _feature_state_record(
    *,
    item: PolicyVisionWorkItem,
    worker_id: str,
    result: Mapping[str, Any],
    gpu_uuid: str,
    device: str,
) -> dict[str, Any]:
    scores = result.get("scores_by_event_step")
    grids = result.get("image_grid_thw")
    merged_counts = result.get("merged_token_counts")
    if (
        not isinstance(scores, Mapping)
        or set(scores) != {str(step) for step in PRIMARY_EVENT_IDS}
        or not isinstance(grids, list)
        or len(grids) != 5
        or not isinstance(merged_counts, list)
        or len(merged_counts) != 5
    ):
        raise ValueError("policy-vision runtime state result schema drifted")
    selection = select_top_two(
        {
            step: _finite_float(scores[str(step)], "runtime policy-vision score")
            for step in PRIMARY_EVENT_IDS
        }
    )
    if (
        result.get("ranked_event_step_ids")
        != list(selection.ranked_event_step_ids)
        or result.get("selected_event_step_ids")
        != list(selection.selected_event_step_ids)
    ):
        raise ValueError("policy-vision runtime selection differs from frozen ranking")
    replay = result.get("same_device_replay")
    if (
        result.get("feature_repeats") != 2
        or not isinstance(replay, Mapping)
        or replay.get("performed") is not True
        or replay.get("ranking_equal") is not True
        or replay.get("selection_equal") is not True
        or _finite_float(
            replay.get("max_abs_score_difference"),
            "same-device score replay difference",
        )
        > 1e-6
    ):
        raise ValueError("policy-vision same-device replay failed")
    event_images = item.event_image_map()
    return {
        "state": {
            "index": item.state_index,
            "role": item.role,
            "trajectory_id": item.trajectory_id,
            "state_id": item.state_id,
        },
        "worker_id": worker_id,
        "device": device,
        "gpu_uuid": gpu_uuid,
        "candidate_scores": [
            {
                "event_step_id": step,
                "post_image_member_path": event_images[step].image_member_path,
                "post_image_sha256": event_images[step].image_sha256,
                "policy_vision_cosine": float(scores[str(step)]),
                "image_grid_thw": grids[step - 1],
                "merged_token_count": int(merged_counts[step - 1]),
            }
            for step in PRIMARY_EVENT_IDS
        ],
        "current_image_member_path": item.current_image.image_member_path,
        "current_image_sha256": item.current_image.image_sha256,
        "current_image_grid_thw": grids[4],
        "current_merged_token_count": int(merged_counts[4]),
        "ranked_event_step_ids": list(selection.ranked_event_step_ids),
        "selected_coalition": list(selection.selected_event_step_ids),
        "normalized_embedding_norm_range": result.get(
            "normalized_embedding_norm_range"
        ),
        "score_replay": {
            "absolute_tolerance": 1e-6,
            "ranking_equal": True,
            "selection_equal": True,
            "max_abs_score_difference": float(
                replay["max_abs_score_difference"]
            ),
            "repeat_results": result.get("repeat_results"),
        },
    }


def run_policy_vision_worker(
    *,
    worker_id: str,
    device: str,
    expected_gpu_uuid: str,
    canonical_items: Sequence[PolicyVisionWorkItem],
    sentinel_item: PolicyVisionWorkItem | None,
    derived_root: Path,
    derived_payload_prefix: str,
    expected_tar_member_count: int,
    model_dir: Path,
    snapshot_manifest: Path,
) -> dict[str, Any]:
    """Run one isolated feature worker; this function never receives D(S)."""
    from causalcache.policy.gui_owl_v2_2_vision_runtime import (
        GUIOwlV22VisionFeatureRuntime,
        _canonical_gpu_uuid,
    )

    if worker_id not in {"even", "odd"}:
        raise ValueError("policy-vision worker id must be even or odd")
    if not canonical_items:
        raise ValueError("policy-vision worker shard cannot be empty")
    expected_parity = 0 if worker_id == "even" else 1
    if any(item.state_index % 2 != expected_parity for item in canonical_items):
        raise ValueError("policy-vision worker received the wrong parity shard")
    processing_items = tuple(canonical_items) + (
        (() if sentinel_item is None else (sentinel_item,))
    )
    image_tar = (
        derived_root
        / derived_payload_prefix
        / "images-00000-of-00001.tar"
    )
    payloads = extract_worker_image_payloads(
        image_tar=image_tar,
        work_items=processing_items,
        expected_tar_member_count=expected_tar_member_count,
    )
    started = time.perf_counter()
    canonical_expected_gpu_uuid = _canonical_gpu_uuid(expected_gpu_uuid)
    runtime = GUIOwlV22VisionFeatureRuntime(
        model_dir=model_dir,
        expected_snapshot_manifest=snapshot_manifest,
        device=device,
        expected_gpu_uuid=canonical_expected_gpu_uuid,
    )
    gpu_uuid = runtime.metadata.get("gpu_uuid")
    if gpu_uuid != canonical_expected_gpu_uuid:
        raise RuntimeError("policy-vision runtime GPU UUID differs from worker binding")
    records = []
    for item in canonical_items:
        images = decode_rgb_images(item, payloads)
        try:
            result = runtime.score_five_images(images, feature_repeats=2)
        finally:
            close_images(images)
        records.append(
            _feature_state_record(
                item=item,
                worker_id=worker_id,
                result=result,
                gpu_uuid=gpu_uuid,
                device=device,
            )
        )
    sentinel_record = None
    if sentinel_item is not None:
        images = decode_rgb_images(sentinel_item, payloads)
        try:
            sentinel_result = runtime.score_five_images(
                images,
                feature_repeats=1,
            )
        finally:
            close_images(images)
        sentinel_record = {
            "state": {
                "index": sentinel_item.state_index,
                "state_id": sentinel_item.state_id,
            },
            "worker_id": worker_id,
            "device": device,
            "gpu_uuid": gpu_uuid,
            "scores_by_event_step": sentinel_result["scores_by_event_step"],
            "ranked_event_step_ids": sentinel_result["ranked_event_step_ids"],
            "selected_coalition": sentinel_result["selected_event_step_ids"],
            "image_grid_thw": sentinel_result["image_grid_thw"],
            "merged_token_counts": sentinel_result["merged_token_counts"],
            "normalized_embedding_norm_range": sentinel_result[
                "normalized_embedding_norm_range"
            ],
        }
    elapsed = time.perf_counter() - started
    memory = {
        "max_memory_allocated_bytes": int(
            runtime.torch.cuda.max_memory_allocated(runtime.device)
        ),
        "max_memory_reserved_bytes": int(
            runtime.torch.cuda.max_memory_reserved(runtime.device)
        ),
    }
    return {
        "worker_id": worker_id,
        "device": device,
        "gpu_uuid": gpu_uuid,
        "canonical_state_records": records,
        "cross_device_sentinel_record": sentinel_record,
        "runtime_metadata": runtime.metadata,
        "runtime_operation_counts": runtime.operation_counts,
        "worker_input_counts": {
            "canonical_state_count": len(canonical_items),
            "sentinel_state_count": int(sentinel_item is not None),
            "image_payload_count": len(payloads),
        },
        "peak_cuda_memory": memory,
        "elapsed_seconds": elapsed,
    }


def merge_policy_vision_workers(
    *,
    worker_outputs: Sequence[Mapping[str, Any]],
    work_items: Sequence[PolicyVisionWorkItem],
    operation_ceiling: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Validate worker isolation, all replays, and the exact execution ledger."""
    if len(worker_outputs) != 2:
        raise ValueError("policy-vision execution requires exactly two workers")
    by_id = {output.get("worker_id"): output for output in worker_outputs}
    if set(by_id) != {"even", "odd"}:
        raise ValueError("policy-vision worker identity coverage drifted")
    expected_shards = feature_worker_shards(work_items)
    expected_ids = {
        "even": {item.state_id for item in expected_shards[0]},
        "odd": {item.state_id for item in expected_shards[1]},
    }
    item_by_state = {item.state_id: item for item in work_items}
    top_level_bindings: dict[str, tuple[str, str]] = {}
    for worker_id, expected_device in (("even", "cuda:0"), ("odd", "cuda:1")):
        output = by_id[worker_id]
        gpu_uuid = output.get("gpu_uuid")
        if (
            output.get("device") != expected_device
            or not isinstance(gpu_uuid, str)
            or not gpu_uuid
        ):
            raise ValueError("policy-vision top-level worker binding drifted")
        top_level_bindings[worker_id] = (expected_device, gpu_uuid)
    if len({binding[1] for binding in top_level_bindings.values()}) != 2:
        raise ValueError("policy-vision workers must bind distinct GPU UUIDs")
    canonical: dict[str, dict[str, Any]] = {}
    for worker_id in ("even", "odd"):
        output = by_id[worker_id]
        rows = output.get("canonical_state_records")
        if not isinstance(rows, list):
            raise ValueError("policy-vision worker state records must be a list")
        observed = {
            row.get("state", {}).get("state_id")
            for row in rows
            if isinstance(row, Mapping)
        }
        if observed != expected_ids[worker_id]:
            raise ValueError("policy-vision worker shard state inventory drifted")
        for row in rows:
            state_id = row["state"]["state_id"]
            if state_id in canonical:
                raise ValueError("policy-vision workers emitted a duplicate state")
            item = item_by_state[state_id]
            expected_state = {
                "index": item.state_index,
                "role": item.role,
                "trajectory_id": item.trajectory_id,
                "state_id": item.state_id,
            }
            expected_device, expected_gpu_uuid = top_level_bindings[worker_id]
            candidates = row.get("candidate_scores")
            replay = row.get("score_replay")
            if (
                row.get("state") != expected_state
                or row.get("worker_id") != worker_id
                or row.get("device") != expected_device
                or row.get("gpu_uuid") != expected_gpu_uuid
                or not isinstance(candidates, list)
                or len(candidates) != len(PRIMARY_EVENT_IDS)
                or not isinstance(replay, Mapping)
                or replay.get("ranking_equal") is not True
                or replay.get("selection_equal") is not True
                or _finite_float(
                    replay.get("max_abs_score_difference"),
                    "same-device replay difference",
                )
                > 1e-6
            ):
                raise ValueError("policy-vision canonical replay evidence drifted")
            expected_images = item.event_image_map()
            for step, candidate in zip(
                PRIMARY_EVENT_IDS,
                candidates,
                strict=True,
            ):
                identity = expected_images[step]
                if (
                    not isinstance(candidate, Mapping)
                    or candidate.get("event_step_id") != step
                    or candidate.get("post_image_member_path")
                    != identity.image_member_path
                    or candidate.get("post_image_sha256") != identity.image_sha256
                ):
                    raise ValueError(
                        "policy-vision canonical candidate identity drifted"
                    )
            if (
                row.get("current_image_member_path")
                != item.current_image.image_member_path
                or row.get("current_image_sha256")
                != item.current_image.image_sha256
            ):
                raise ValueError("policy-vision canonical current identity drifted")
            _canonical_feature_scores(row)
            canonical[state_id] = dict(row)
    if set(canonical) != {item.state_id for item in work_items}:
        raise ValueError("policy-vision canonical worker union drifted")

    sentinel_item = work_items[0]
    source = canonical[sentinel_item.state_id]
    sentinel = by_id["odd"].get("cross_device_sentinel_record")
    odd_device, odd_gpu_uuid = top_level_bindings["odd"]
    if (
        not isinstance(sentinel, Mapping)
        or sentinel.get("state")
        != {"index": sentinel_item.state_index, "state_id": sentinel_item.state_id}
        or sentinel.get("worker_id") != "odd"
        or sentinel.get("device") != odd_device
        or sentinel.get("gpu_uuid") != odd_gpu_uuid
        or by_id["even"].get("cross_device_sentinel_record") is not None
    ):
        raise ValueError("policy-vision cross-device sentinel identity drifted")
    source_scores = {
        str(row["event_step_id"]): float(row["policy_vision_cosine"])
        for row in source["candidate_scores"]
    }
    sentinel_scores = sentinel.get("scores_by_event_step")
    if not isinstance(sentinel_scores, Mapping) or set(sentinel_scores) != set(
        source_scores
    ):
        raise ValueError("policy-vision cross-device sentinel score schema drifted")
    cross_differences = {
        step: abs(source_scores[step] - float(sentinel_scores[step]))
        for step in sorted(source_scores)
    }
    cross_max = max(cross_differences.values())
    if (
        cross_max > 1e-6
        or sentinel.get("ranked_event_step_ids")
        != source["ranked_event_step_ids"]
        or sentinel.get("selected_coalition") != source["selected_coalition"]
    ):
        raise ValueError("policy-vision cross-device sentinel replay failed")

    runtime_counts = Counter()
    for output in worker_outputs:
        counts = output.get("runtime_operation_counts")
        if not isinstance(counts, Mapping):
            raise ValueError("policy-vision runtime operation ledger is absent")
        for key, value in counts.items():
            if type(value) is not int or value < 0:
                raise ValueError("policy-vision runtime operation count is invalid")
            runtime_counts[key] += value
    expected_runtime_counts = {
        "image_processor_batch_count": operation_ceiling[
            "image_processor_batch_count"
        ],
        "policy_vision_feature_forward_count": operation_ceiling[
            "policy_vision_feature_forward_count"
        ],
        "top_model_forward_count": operation_ceiling["policy_forward_count"],
        "language_model_forward_count": operation_ceiling[
            "language_model_forward_count"
        ],
        "lm_head_forward_count": operation_ceiling["lm_head_forward_count"],
        "generation_count": operation_ceiling["generation_count"],
    }
    if dict(runtime_counts) != expected_runtime_counts:
        raise ValueError(
            "policy-vision runtime operation ledger differs from contract: "
            + repr(dict(runtime_counts))
        )
    ordered = tuple(canonical[item.state_id] for item in work_items)
    replay = {
        "same_device": {
            "state_count": len(ordered),
            "all_rankings_equal": True,
            "all_selections_equal": True,
            "maximum_absolute_score_difference": max(
                float(row["score_replay"]["max_abs_score_difference"])
                for row in ordered
            ),
            "absolute_tolerance": 1e-6,
        },
        "cross_device_sentinel": {
            "primary_ordinal": sentinel_item.primary_ordinal,
            "state_index": sentinel_item.state_index,
            "state_id": sentinel_item.state_id,
            "source_worker": "even",
            "verification_worker": "odd",
            "per_event_absolute_score_difference": cross_differences,
            "maximum_absolute_score_difference": cross_max,
            "ranking_equal": True,
            "selection_equal": True,
            "absolute_tolerance": 1e-6,
        },
        "runtime_operation_counts": dict(runtime_counts),
        "workers": [
            {
                key: by_id[worker_id][key]
                for key in (
                    "worker_id",
                    "device",
                    "gpu_uuid",
                    "runtime_metadata",
                    "worker_input_counts",
                    "peak_cuda_memory",
                    "elapsed_seconds",
                )
            }
            for worker_id in ("even", "odd")
        ],
    }
    return ordered, replay


def _distance_table(state: SelectorGeometryState) -> dict[tuple[int, ...], float]:
    result = {tuple(row.coalition): float(row.distance) for row in state.table.rows}
    if len(result) != 16:
        raise ValueError("primary policy-vision evaluation requires a complete 2^4 table")
    return result


def load_primary_label_states(
    labels_archive: Path,
    work_items: Sequence[PolicyVisionWorkItem],
) -> dict[str, SelectorGeometryState]:
    states = load_selector_geometry_states(labels_archive)
    if len(states) != EXPECTED_LABEL_STATE_COUNT or sum(
        len(state.table.rows) for state in states
    ) != EXPECTED_LABEL_DISTANCE_ROW_COUNT:
        raise ValueError("raw restoration-label denominator drifted")
    primary = {
        state.state_id: state
        for state in states
        if state.decision_step_id == PRIMARY_DECISION_STEP
        and state.candidate_event_step_ids == PRIMARY_EVENT_IDS
    }
    expected_ids = {item.state_id for item in work_items}
    if set(primary) != expected_ids or sum(
        len(state.table.rows) for state in primary.values()
    ) != EXPECTED_PRIMARY_DISTANCE_ROW_COUNT:
        raise ValueError("raw restoration-label primary inventory drifted")
    return primary


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = frozenset(left)
    right_set = frozenset(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def _canonical_feature_scores(
    feature_record: Mapping[str, Any],
) -> tuple[dict[int, float], tuple[int, ...], tuple[int, ...]]:
    rows = feature_record.get("candidate_scores")
    if not isinstance(rows, list) or len(rows) != len(PRIMARY_EVENT_IDS):
        raise ValueError("feature record must contain four candidate scores")
    scores: dict[int, float] = {}
    for expected_step, row in zip(PRIMARY_EVENT_IDS, rows, strict=True):
        if not isinstance(row, Mapping) or row.get("event_step_id") != expected_step:
            raise ValueError("feature candidate score order drifted")
        scores[expected_step] = _finite_float(
            row.get("policy_vision_cosine"),
            "policy-vision cosine",
        )
    selection = select_top_two(scores)
    ranked = tuple(selection.ranked_event_step_ids)
    selected = tuple(selection.selected_event_step_ids)
    if feature_record.get("ranked_event_step_ids") != list(ranked):
        raise ValueError("feature record ranking differs from frozen tie-break")
    if feature_record.get("selected_coalition") != list(selected):
        raise ValueError("feature record coalition differs from frozen top-two")
    return scores, ranked, selected


def evaluate_feature_record(
    *,
    feature_record: Mapping[str, Any],
    work_item: PolicyVisionWorkItem,
    witness_record: Mapping[str, Any],
    label_state: SelectorGeometryState,
    protocol_id: str,
) -> dict[str, Any]:
    """Join frozen feature selection with D(S) only after selection is complete."""
    feature_state = feature_record.get("state")
    expected_feature_state = {
        "index": work_item.state_index,
        "role": work_item.role,
        "trajectory_id": work_item.trajectory_id,
        "state_id": work_item.state_id,
    }
    expected_worker_id, expected_device = _expected_worker_fields(work_item)
    feature_gpu_uuid = feature_record.get("gpu_uuid")
    if (
        not isinstance(feature_state, Mapping)
        or dict(feature_state) != expected_feature_state
        or feature_record.get("worker_id") != expected_worker_id
        or feature_record.get("device") != expected_device
        or not isinstance(feature_gpu_uuid, str)
        or not feature_gpu_uuid
    ):
        raise ValueError("feature record state identity drifted")
    if (
        label_state.role != work_item.role
        or label_state.trajectory_id != work_item.trajectory_id
        or label_state.state_id != work_item.state_id
    ):
        raise ValueError("feature plan and restoration labels disagree")
    scores, ranked, selected = _canonical_feature_scores(feature_record)
    distances = _distance_table(label_state)
    baseline_distance = distances[()]
    selected_distance = distances[selected]
    utility = baseline_distance - selected_distance
    recovery = (
        utility / baseline_distance
        if baseline_distance > NORMALIZATION_EPSILON
        else None
    )
    exact_subset, exact_distance = min(
        (
            (coalition, distance)
            for coalition, distance in distances.items()
            if len(coalition) <= PRIMARY_BUDGET
        ),
        key=lambda item: (item[1], len(item[0]), item[0]),
    )
    exact_cardinality, exact_cardinality_distance = min(
        (
            (coalition, distance)
            for coalition, distance in distances.items()
            if len(coalition) == PRIMARY_BUDGET
        ),
        key=lambda item: (item[1], item[0]),
    )
    exact_recovery = (
        (baseline_distance - exact_distance) / baseline_distance
        if baseline_distance > NORMALIZATION_EPSILON
        else None
    )
    exact_cardinality_recovery = (
        (baseline_distance - exact_cardinality_distance) / baseline_distance
        if baseline_distance > NORMALIZATION_EPSILON
        else None
    )
    witness_state = witness_record.get("state")
    if witness_state != {
        "budget_event_capacity": PRIMARY_BUDGET,
        "candidate_event_step_ids": list(PRIMARY_EVENT_IDS),
        "decision_step_id": PRIMARY_DECISION_STEP,
        "index": work_item.state_index,
        "role": work_item.role,
        "state_id": work_item.state_id,
        "trajectory_id": work_item.trajectory_id,
    }:
        raise ValueError("OCR/RGB witness state fields drifted")
    if (
        witness_record.get("exact_subset_coalition") != list(exact_subset)
        or witness_record.get("exact_cardinality_oracle_coalition")
        != list(exact_cardinality)
        or not math.isclose(
            _finite_float(
                witness_record.get("baseline_summary_only_distance"),
                "witness baseline distance",
            ),
            baseline_distance,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("OCR/RGB witness and raw restoration table disagree")
    witness_candidates = witness_record["candidate_scores"]
    candidate_rows = []
    for step_id, witness_candidate in zip(
        PRIMARY_EVENT_IDS,
        witness_candidates,
        strict=True,
    ):
        identity = work_item.event_image_map()[step_id]
        candidate_rows.append(
            {
                "event_step_id": step_id,
                "post_image_member_path": identity.image_member_path,
                "post_image_sha256": identity.image_sha256,
                "policy_vision_cosine": scores[step_id],
                "image_grid_thw": feature_record["candidate_scores"][step_id - 1][
                    "image_grid_thw"
                ],
                "merged_token_count": feature_record["candidate_scores"][step_id - 1][
                    "merged_token_count"
                ],
                "ocr_rgb_combined_similarity_witness": _finite_float(
                    witness_candidate.get("combined_similarity"),
                    "OCR/RGB combined similarity witness",
                ),
            }
        )
    geometry = witness_record.get("geometry_comparators")
    if not isinstance(geometry, Mapping) or set(geometry) != set(COMPARATOR_METHODS):
        raise ValueError("OCR/RGB geometry-comparator witness drifted")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "state": dict(witness_state),
        "worker": {
            "worker_id": feature_record.get("worker_id"),
            "device": feature_record.get("device"),
            "gpu_uuid": feature_record.get("gpu_uuid"),
        },
        "current_image": {
            "image_member_path": work_item.current_image.image_member_path,
            "image_sha256": work_item.current_image.image_sha256,
            "image_grid_thw": feature_record.get("current_image_grid_thw"),
            "merged_token_count": feature_record.get("current_merged_token_count"),
        },
        "candidate_scores": candidate_rows,
        "ranked_event_step_ids": list(ranked),
        "selected_coalition": list(selected),
        "score_replay": feature_record.get("score_replay"),
        "baseline_summary_only_distance": baseline_distance,
        "selected_distance": selected_distance,
        "actual_utility": utility,
        "normalized_recovery": recovery,
        "exact_subset_coalition": list(exact_subset),
        "exact_subset_distance": exact_distance,
        "exact_subset_normalized_recovery": exact_recovery,
        "absolute_regret_to_exact_subset": (
            baseline_distance - exact_distance - utility
        ),
        "normalized_recovery_regret_to_exact_subset": (
            None
            if recovery is None or exact_recovery is None
            else exact_recovery - recovery
        ),
        "exact_coalition_match": selected == exact_subset,
        "jaccard_to_exact_coalition": _jaccard(selected, exact_subset),
        "exact_cardinality_oracle_coalition": list(exact_cardinality),
        "exact_cardinality_oracle_distance": exact_cardinality_distance,
        "exact_cardinality_oracle_normalized_recovery": (
            exact_cardinality_recovery
        ),
        "absolute_regret_to_exact_cardinality_oracle": (
            baseline_distance - exact_cardinality_distance - utility
        ),
        "normalized_recovery_regret_to_exact_cardinality_oracle": (
            None
            if recovery is None or exact_cardinality_recovery is None
            else exact_cardinality_recovery - recovery
        ),
        "exact_cardinality_coalition_match": selected == exact_cardinality,
        "jaccard_to_exact_cardinality_coalition": _jaccard(
            selected,
            exact_cardinality,
        ),
        "ocr_rgb_comparator": {
            "selected_coalition": witness_record.get("selected_coalition"),
            "normalized_recovery": witness_record.get("normalized_recovery"),
        },
        "geometry_comparators": {
            method: dict(geometry[method]) for method in COMPARATOR_METHODS
        },
    }


def evaluate_feature_records(
    *,
    feature_records: Sequence[Mapping[str, Any]],
    work_items: Sequence[PolicyVisionWorkItem],
    witness_by_state: Mapping[str, Mapping[str, Any]],
    primary_label_states: Mapping[str, SelectorGeometryState],
    protocol_id: str,
) -> tuple[dict[str, Any], ...]:
    if len(feature_records) != EXPECTED_STATE_COUNT:
        raise ValueError("policy-vision evaluation requires exactly 15 feature records")
    feature_by_state: dict[str, Mapping[str, Any]] = {}
    for record in feature_records:
        state = record.get("state")
        if not isinstance(state, Mapping) or not isinstance(state.get("state_id"), str):
            raise ValueError("feature record lacks a state identity")
        state_id = state["state_id"]
        if state_id in feature_by_state:
            raise ValueError("feature records contain duplicate states")
        feature_by_state[state_id] = record
    validate_feature_worker_provenance(feature_records, work_items)
    expected_ids = {item.state_id for item in work_items}
    if (
        set(feature_by_state) != expected_ids
        or set(witness_by_state) != expected_ids
        or set(primary_label_states) != expected_ids
    ):
        raise ValueError("feature/witness/label state inventories differ")
    rows = [
        evaluate_feature_record(
            feature_record=feature_by_state[item.state_id],
            work_item=item,
            witness_record=witness_by_state[item.state_id],
            label_state=primary_label_states[item.state_id],
            protocol_id=protocol_id,
        )
        for item in work_items
    ]
    if sum(len(row["candidate_scores"]) for row in rows) != (
        EXPECTED_CANDIDATE_SCORE_COUNT
    ):
        raise RuntimeError("policy-vision evaluated score denominator drifted")
    return tuple(rows)


def validate_canonical_preprocessing_geometry(
    records: Sequence[Mapping[str, Any]],
    preprocessing_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the formal 75-image processor geometry before any aggregate is written."""
    if len(records) != EXPECTED_STATE_COUNT:
        raise ValueError("processor geometry validation requires exactly 15 states")
    grid_histogram: Counter[str] = Counter()
    state_rows: Counter[str] = Counter()
    raw_total = 0
    merged_total = 0
    for record in records:
        image_rows = [
            *record["candidate_scores"],
            record["current_image"],
        ]
        if len(image_rows) != 5:
            raise ValueError("processor geometry state must contain five images")
        state_raw = 0
        for image_row in image_rows:
            grid = image_row.get("image_grid_thw")
            merged = image_row.get("merged_token_count")
            if (
                not isinstance(grid, list)
                or len(grid) != 3
                or any(type(value) is not int or value <= 0 for value in grid)
                or type(merged) is not int
                or merged <= 0
            ):
                raise ValueError("processor geometry row schema drifted")
            temporal, height, width = grid
            raw = temporal * height * width
            if height % 2 or width % 2 or merged != raw // 4:
                raise ValueError("processor merged-token geometry drifted")
            grid_histogram[f"{temporal},{height},{width}"] += 1
            state_raw += raw
            raw_total += raw
            merged_total += merged
        state_rows[str(state_raw)] += 1
    observed = {
        "grid_thw_histogram_over_75_canonical_images": dict(
            sorted(grid_histogram.items())
        ),
        "state_pixel_value_row_histogram_over_15_canonical_batches": dict(
            sorted(state_rows.items())
        ),
        "canonical_raw_patch_rows": raw_total,
        "canonical_merged_visual_tokens": merged_total,
    }
    expected = {
        key: preprocessing_contract[key]
        for key in observed
    }
    if observed != expected:
        raise ValueError(
            "formal policy-vision processor geometry differs from source freeze: "
            + repr(observed)
        )
    return observed


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric sequence")
    return math.fsum(values) / len(values)


def _comparison_value(record: Mapping[str, Any], method: str) -> float:
    if method == OCR_RGB_METHOD:
        return _finite_float(
            record["ocr_rgb_comparator"]["normalized_recovery"],
            "OCR/RGB comparator recovery",
        )
    return _finite_float(
        record["geometry_comparators"][method]["normalized_recovery"],
        f"{method} comparator recovery",
    )


def summarize_policy_vision_records(
    records: Sequence[Mapping[str, Any]],
    *,
    operation_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute pre-registered split, paired, saturation, and outlier reports."""
    if len(records) != EXPECTED_STATE_COUNT:
        raise ValueError("policy-vision summary requires exactly 15 states")
    comparator_methods = (*COMPARATOR_METHODS, OCR_RGB_METHOD)
    by_role: dict[str, Any] = {}
    for role in ("v2_label_train", "v2_development", "overall_stratified"):
        selected = [
            record
            for record in records
            if role == "overall_stratified" or record["state"]["role"] == role
        ]
        recoveries = [
            _finite_float(record["normalized_recovery"], "normalized recovery")
            for record in selected
        ]
        utilities = [
            _finite_float(record["actual_utility"], "actual utility")
            for record in selected
        ]
        baselines = [
            _finite_float(
                record["baseline_summary_only_distance"],
                "baseline distance",
            )
            for record in selected
        ]
        exact_recoveries = [
            _finite_float(
                record["exact_subset_normalized_recovery"],
                "exact recovery",
            )
            for record in selected
        ]
        role_record = {
            "state_count": len(selected),
            "trajectory_count": len(
                {record["state"]["trajectory_id"] for record in selected}
            ),
            "mean_actual_utility": _mean(utilities),
            "mean_normalized_recovery": _mean(recoveries),
            "median_normalized_recovery": statistics.median(recoveries),
            "minimum_normalized_recovery": min(recoveries),
            "maximum_normalized_recovery": max(recoveries),
            "negative_normalized_recovery_count": sum(
                value < 0.0 for value in recoveries
            ),
            "mean_baseline_summary_only_distance": _mean(baselines),
            "ratio_of_sums_utility_to_baseline_distance": (
                math.fsum(utilities) / math.fsum(baselines)
                if math.fsum(baselines) > NORMALIZATION_EPSILON
                else None
            ),
            "recovery_ratio_of_means_to_exact_subset": (
                _mean(recoveries) / _mean(exact_recoveries)
                if abs(_mean(exact_recoveries)) > NORMALIZATION_EPSILON
                else None
            ),
            "mean_selected_cardinality": 2.0,
            "exact_coalition_match_rate": _mean(
                [float(record["exact_coalition_match"]) for record in selected]
            ),
            "mean_jaccard_to_exact_coalition": _mean(
                [float(record["jaccard_to_exact_coalition"]) for record in selected]
            ),
            "exact_cardinality_coalition_match_rate": _mean(
                [
                    float(record["exact_cardinality_coalition_match"])
                    for record in selected
                ]
            ),
            "mean_jaccard_to_exact_cardinality_coalition": _mean(
                [
                    float(record["jaccard_to_exact_cardinality_coalition"])
                    for record in selected
                ]
            ),
            "selected_cardinality_histogram": {"2": len(selected)},
            "mean_recovery_delta_by_comparator": {
                method: _mean(
                    [
                        float(record["normalized_recovery"])
                        - _comparison_value(record, method)
                        for record in selected
                    ]
                )
                for method in comparator_methods
            },
        }
        by_role[role] = role_record

    metric_rows = []
    for record in records:
        metrics = {
            POLICY_VISION_METHOD: float(record["normalized_recovery"]),
            **{
                method: _comparison_value(record, method)
                for method in comparator_methods
            },
        }
        metric_rows.append(
            StateMetricRow(
                role=record["state"]["role"],
                trajectory_id=record["state"]["trajectory_id"],
                state_id=record["state"]["state_id"],
                metrics=metrics,
            )
        )

    def bootstrap(method: str) -> dict[str, Any]:
        report = paired_trajectory_bootstrap(
            metric_rows,
            POLICY_VISION_METHOD,
            method,
            resamples=10_000,
            seed=271_828,
            confidence=0.9,
            tie_epsilon=1e-12,
        )
        payload = asdict(report)
        for split in ("train", "development", "overall_stratified"):
            outcome = payload[split]["win_tie_loss"]
            outcome["count"] = (
                outcome["wins"] + outcome["ties"] + outcome["losses"]
            )
        return payload

    development = sorted(
        (
            record
            for record in records
            if record["state"]["role"] == "v2_development"
        ),
        key=lambda record: record["state"]["trajectory_id"],
    )
    development_deltas = {
        method: [
            {
                "trajectory_id": record["state"]["trajectory_id"],
                "policy_vision_normalized_recovery": float(
                    record["normalized_recovery"]
                ),
                "comparator_normalized_recovery": _comparison_value(
                    record,
                    method,
                ),
                "paired_delta": float(record["normalized_recovery"])
                - _comparison_value(record, method),
            }
            for record in development
        ]
        for method in comparator_methods
    }

    all_scores = [
        _finite_float(candidate["policy_vision_cosine"], "policy-vision cosine")
        for record in records
        for candidate in record["candidate_scores"]
    ]
    within_ranges = []
    top_two_margins = []
    near_tie_count = 0
    selected_frequency: Counter[int] = Counter()
    coalition_overlap = Counter()
    grid_histogram: Counter[str] = Counter()
    merged_token_total = 0
    raw_patch_total = 0
    for record in records:
        scores = {
            candidate["event_step_id"]: float(candidate["policy_vision_cosine"])
            for candidate in record["candidate_scores"]
        }
        ranked = record["ranked_event_step_ids"]
        within_ranges.append(max(scores.values()) - min(scores.values()))
        margin = scores[ranked[1]] - scores[ranked[2]]
        top_two_margins.append(margin)
        if math.isclose(
            scores[ranked[1]],
            scores[ranked[2]],
            rel_tol=SCORE_TIE_REL_TOL,
            abs_tol=SCORE_TIE_ABS_TOL,
        ):
            near_tie_count += 1
        selected = tuple(record["selected_coalition"])
        selected_frequency.update(selected)
        recent = tuple(
            record["geometry_comparators"]["dynamic_recent"][
                "selected_coalition"
            ]
        )
        ocr = tuple(record["ocr_rgb_comparator"]["selected_coalition"])
        coalition_overlap["exact_recent"] += selected == recent
        coalition_overlap["exact_ocr_rgb"] += selected == ocr
        for candidate in record["candidate_scores"]:
            grid = candidate["image_grid_thw"]
            grid_histogram[str(grid)] += 1
            merged = int(candidate["merged_token_count"])
            merged_token_total += merged
            raw_patch_total += merged * 4
        current_grid = record["current_image"]["image_grid_thw"]
        grid_histogram[str(current_grid)] += 1
        current_merged = int(record["current_image"]["merged_token_count"])
        merged_token_total += current_merged
        raw_patch_total += current_merged * 4

    outliers = [
        record
        for record in records
        if record["state"]["trajectory_id"] == OUTLIER_TRAJECTORY_ID
    ]
    if len(outliers) != 1:
        raise ValueError("pre-registered OCR/RGB outlier is absent or duplicated")
    outlier = outliers[0]
    return {
        "scope": "primary_n4_b2_only",
        "state_count": EXPECTED_STATE_COUNT,
        "trajectory_count": EXPECTED_STATE_COUNT,
        "candidate_comparison_count": EXPECTED_CANDIDATE_SCORE_COUNT,
        "unique_image_count": EXPECTED_UNIQUE_IMAGE_COUNT,
        "by_role": by_role,
        "paired_trajectory_bootstrap": {
            method: bootstrap(method) for method in comparator_methods
        },
        "development_paired_deltas": development_deltas,
        "feature_diagnostics": {
            "cosine_count": len(all_scores),
            "cosine_minimum": min(all_scores),
            "cosine_maximum": max(all_scores),
            "cosine_mean": _mean(all_scores),
            "within_state_cosine_range_mean": _mean(within_ranges),
            "within_state_cosine_range_minimum": min(within_ranges),
            "within_state_cosine_range_maximum": max(within_ranges),
            "top2_minus_top3_margin_mean": _mean(top_two_margins),
            "top2_minus_top3_margin_minimum": min(top_two_margins),
            "near_tie_state_count": near_tie_count,
            "selected_event_frequency": {
                str(step): selected_frequency[step] for step in PRIMARY_EVENT_IDS
            },
            "exact_coalition_overlap_with_recent_count": coalition_overlap[
                "exact_recent"
            ],
            "exact_coalition_overlap_with_ocr_rgb_count": coalition_overlap[
                "exact_ocr_rgb"
            ],
            "image_grid_thw_histogram": dict(sorted(grid_histogram.items())),
            "raw_patch_total": raw_patch_total,
            "merged_token_total": merged_token_total,
        },
        "pre_registered_outlier": {
            "trajectory_id": OUTLIER_TRAJECTORY_ID,
            "selected_coalition": outlier["selected_coalition"],
            "ocr_rgb_selected_coalition": outlier["ocr_rgb_comparator"][
                "selected_coalition"
            ],
            "baseline_summary_only_distance": outlier[
                "baseline_summary_only_distance"
            ],
            "selected_distance": outlier["selected_distance"],
            "normalized_recovery": outlier["normalized_recovery"],
            "exact_subset_coalition": outlier["exact_subset_coalition"],
            "exact_subset_distance": outlier["exact_subset_distance"],
            "exact_subset_normalized_recovery": outlier[
                "exact_subset_normalized_recovery"
            ],
            "negative_recovery_preserved_without_clamp": (
                float(outlier["normalized_recovery"]) < 0.0
            ),
        },
        "operation_counts": dict(operation_counts),
    }


def verify_derived_projection(contract: Any, derived_root: Path) -> None:
    """Verify the exact-six immutable HF projection without parsing unused rows."""
    derived = contract.data["immutable_inputs"]["derived_dataset"]
    expected = {
        record["path"]: (record["size_bytes"], record["sha256"])
        for record in derived["exact_files"]
    }
    observed: dict[str, Path] = {}
    for path in sorted(derived_root.rglob("*")):
        if path.is_symlink():
            raise ValueError("derived projection cannot contain symlinks")
        if path.is_file():
            observed[path.relative_to(derived_root).as_posix()] = path
    if set(observed) != set(expected):
        raise ValueError("derived projection exact-six inventory drifted")
    for relative, path in observed.items():
        size, digest = expected[relative]
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"derived immutable file identity drifted: {relative}")
