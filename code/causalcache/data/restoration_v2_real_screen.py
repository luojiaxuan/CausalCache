"""Policy-blind real-screen OCR golden materialization for restoration-v2."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

from causalcache.data.guiodyssey import build_pilot_manifest
from causalcache.data.guiodyssey_independent import (
    source_file_specs,
    verify_local_source_files,
)
from causalcache.data.restoration_v2_selection import (
    validate_state_content_witnesses,
)
from causalcache.restoration_v2_text_backend import (
    prepare_selected_guiodyssey_image,
    sha256_bytes,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_restoration_v2"
ARTIFACT_ID = "causalcache-restoration-v2-real-screen-golden-v1"
PAYLOAD_PREFIX = "golden/real-screen-v1"
IMAGE_TAR_RELATIVE_PATH = f"{PAYLOAD_PREFIX}/images-00000-of-00001.tar"
OCR_JSONL_RELATIVE_PATH = f"{PAYLOAD_PREFIX}/ocr-records-00000-of-00001.jsonl"
MANIFEST_RELATIVE_PATH = f"{PAYLOAD_PREFIX}/manifest.json"
GITATTRIBUTES_RELATIVE_PATH = ".gitattributes"
ARTIFACT_RELATIVE_PATHS = (
    GITATTRIBUTES_RELATIVE_PATH,
    "README.md",
    IMAGE_TAR_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    OCR_JSONL_RELATIVE_PATH,
)

ELIGIBLE_ROLE_ORDER = ("v2_label_train", "v2_development")
ELIGIBLE_IMAGE_ROLE_ORDER = ("candidate_post_state", "current_observation")
EXPECTED_STATE_COUNT = 45
EXPECTED_OCCURRENCE_COUNT = 180
EXPECTED_CANDIDATE_OCCURRENCE_COUNT = 135
EXPECTED_CURRENT_OCCURRENCE_COUNT = 45
EXPECTED_UNIQUE_IMAGE_COUNT = 75
EXPECTED_CONFIRM_UNIQUE_IMAGE_COUNT = 97
EXPECTED_CONFIRM_INTERSECTION_COUNT = 0
EXPECTED_ORIENTATION_COUNTS = {"portrait": 55, "landscape": 20, "square": 0}

EXPECTED_SELECTED_IMAGES = {
    "portrait": (
        {
            "image_sha256": (
                "02a92f2d9749446e03533884f0dccce0fadff860a07e3192414fe60f6af63c3f"
            ),
            "image_member_path": "images/0131649930078879/observation-005.png",
            "width": 720,
            "height": 1280,
        },
        {
            "image_sha256": (
                "0e3304250dcf14ea3fb2c19cc4e14a2716b7cc77d00966c3cbc6d9e1cde4031a"
            ),
            "image_member_path": "images/0217738978329323/observation-004.png",
            "width": 1440,
            "height": 3120,
        },
        {
            "image_sha256": (
                "112d550f8bc65da5541015d7ff68677c410865f57a629144e9214ba6bb1e3cd1"
            ),
            "image_member_path": "images/0217738978329323/observation-003.png",
            "width": 1440,
            "height": 3120,
        },
    ),
    "landscape": (
        {
            "image_sha256": (
                "11cbacfa5532c1839970ea53db979df9bcc6a12ab91839ac284ca7af903b054b"
            ),
            "image_member_path": "images/0214300008821039/observation-003.png",
            "width": 2560,
            "height": 1600,
        },
        {
            "image_sha256": (
                "1c68bb93daceb610845ba368ba3015e7c3a86336c9271f062c6b9e61d7a7ef5c"
            ),
            "image_member_path": "images/0119685762769531/observation-003.png",
            "width": 2560,
            "height": 1600,
        },
        {
            "image_sha256": (
                "250c54400ebcffa05d2085c6ba2127d226732aea577792fa2dc96bf0b4d38127"
            ),
            "image_member_path": "images/0214993880872733/observation-004.png",
            "width": 2560,
            "height": 1600,
        },
    ),
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _safe_member_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("image member path must be a safe relative path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or str(parsed) != value or any(
        part in {".", ".."} for part in parsed.parts
    ):
        raise ValueError("image member path must be a safe relative path")
    return value


def _occurrence_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    event_step_id = record.get("event_step_id")
    return (
        ELIGIBLE_ROLE_ORDER.index(str(record["role"])),
        str(record["state_id"]),
        ELIGIBLE_IMAGE_ROLE_ORDER.index(str(record["image_role"])),
        -1 if event_step_id is None else int(event_step_id),
        str(record["image_member_path"]),
        str(record["image_sha256"]),
    )


def _state_occurrences(
    state: Mapping[str, Any],
    *,
    role: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in state["candidate_event_post_states"]:
        records.append(
            {
                "role": role,
                "state_id": str(state["state_id"]),
                "source_id": str(state["source_id"]),
                "decision_step_id": int(state["decision_step_id"]),
                "image_role": "candidate_post_state",
                "event_step_id": int(candidate["event_step_id"]),
                "image_member_path": _safe_member_path(
                    candidate["post_state_member_path"]
                ),
                "image_sha256": _require_sha256(
                    candidate["post_state_sha256"],
                    "candidate post-state SHA256",
                ),
            }
        )
    current = state["current_observation"]
    records.append(
        {
            "role": role,
            "state_id": str(state["state_id"]),
            "source_id": str(state["source_id"]),
            "decision_step_id": int(state["decision_step_id"]),
            "image_role": "current_observation",
            "event_step_id": None,
            "image_member_path": _safe_member_path(current["member_path"]),
            "image_sha256": _require_sha256(
                current["sha256"],
                "current-observation SHA256",
            ),
        }
    )
    return records


def collect_real_screen_occurrences(
    selection_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[str]]:
    validate_state_content_witnesses(selection_manifest)
    roles = selection_manifest.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("selection manifest roles must be an object")
    eligible: list[dict[str, Any]] = []
    state_count = 0
    for role in ELIGIBLE_ROLE_ORDER:
        states = roles[role]["states"]
        state_count += len(states)
        for state in states:
            eligible.extend(_state_occurrences(state, role=role))
    if state_count != EXPECTED_STATE_COUNT:
        raise ValueError("real-screen eligible state count drifted")
    eligible.sort(key=_occurrence_sort_key)
    if len(eligible) != EXPECTED_OCCURRENCE_COUNT:
        raise ValueError("real-screen eligible occurrence count drifted")
    role_counts = {
        image_role: sum(
            record["image_role"] == image_role for record in eligible
        )
        for image_role in ELIGIBLE_IMAGE_ROLE_ORDER
    }
    if role_counts != {
        "candidate_post_state": EXPECTED_CANDIDATE_OCCURRENCE_COUNT,
        "current_observation": EXPECTED_CURRENT_OCCURRENCE_COUNT,
    }:
        raise ValueError("real-screen eligible image-role counts drifted")

    eligible_shas = {record["image_sha256"] for record in eligible}
    if len(eligible_shas) != EXPECTED_UNIQUE_IMAGE_COUNT:
        raise ValueError("real-screen eligible unique-image count drifted")

    confirm_shas: set[str] = set()
    for state in roles["v2_confirm_primary"]["states"]:
        for record in _state_occurrences(state, role="v2_confirm_primary"):
            confirm_shas.add(record["image_sha256"])
    if len(confirm_shas) != EXPECTED_CONFIRM_UNIQUE_IMAGE_COUNT:
        raise ValueError("real-screen confirm unique-image count drifted")
    if len(eligible_shas.intersection(confirm_shas)) != (
        EXPECTED_CONFIRM_INTERSECTION_COUNT
    ):
        raise ValueError("real-screen eligible/confirm SHA overlap drifted")
    return eligible, confirm_shas


def _orientation(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if height > width:
        return "portrait"
    if width > height:
        return "landscape"
    raise ValueError("square images are invalid for the frozen orientation strata")


def build_candidate_pool(
    selection_manifest: Mapping[str, Any],
    *,
    image_payloads: Mapping[str, bytes],
    backend_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    occurrences, _ = collect_real_screen_occurrences(selection_manifest)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for occurrence in occurrences:
        grouped[occurrence["image_sha256"]].append(occurrence)
    if len(grouped) != EXPECTED_UNIQUE_IMAGE_COUNT:
        raise ValueError("candidate pool unique-image count drifted")

    expected_paths = {
        record["image_member_path"]
        for records in grouped.values()
        for record in records
    }
    if set(image_payloads) != expected_paths:
        raise ValueError("loaded image payload paths differ from the eligible pool")

    pool: list[dict[str, Any]] = []
    for image_sha256, records in grouped.items():
        paths = sorted({record["image_member_path"] for record in records})
        member_path = paths[0]
        for path in paths:
            payload = image_payloads[path]
            if (
                not isinstance(payload, bytes)
                or sha256_bytes(payload) != image_sha256
            ):
                raise ValueError("eligible image payload SHA256 drifted")
        payload = image_payloads[member_path]
        prepared = prepare_selected_guiodyssey_image(payload, backend_config)
        eligible_occurrences = []
        for record in sorted(records, key=_occurrence_sort_key):
            eligible_occurrences.append(
                {
                    "role": record["role"],
                    "state_id": record["state_id"],
                    "source_id": record["source_id"],
                    "decision_step_id": record["decision_step_id"],
                    "image_role": record["image_role"],
                    "event_step_id": record["event_step_id"],
                    "image_member_path": record["image_member_path"],
                    "image_sha256": record["image_sha256"],
                }
            )
        pool.append(
            {
                "image_sha256": image_sha256,
                "image_member_path": member_path,
                "image_size_bytes": len(payload),
                "source_format": prepared.source_format,
                "source_mode": prepared.source_mode,
                "width": prepared.width,
                "height": prepared.height,
                "exif_present": prepared.exif_present,
                "alpha_extrema": (
                    list(prepared.alpha_extrema)
                    if prepared.alpha_extrema is not None
                    else None
                ),
                "rgb_bytes_sha256": prepared.rgb_bytes_sha256,
                "resized_rgb_256x256_sha256": prepared.resized_rgb_bytes_sha256,
                "orientation": _orientation(prepared.width, prepared.height),
                "eligible_occurrences": eligible_occurrences,
            }
        )
    pool.sort(key=lambda record: (record["image_sha256"], record["image_member_path"]))
    for index, record in enumerate(pool):
        record["candidate_pool_index"] = index
    validate_candidate_pool(pool)
    return pool


def validate_candidate_pool(pool: Sequence[Mapping[str, Any]]) -> None:
    if len(pool) != EXPECTED_UNIQUE_IMAGE_COUNT:
        raise ValueError("candidate pool must contain exactly 75 images")
    expected_indices = list(range(EXPECTED_UNIQUE_IMAGE_COUNT))
    if [record.get("candidate_pool_index") for record in pool] != expected_indices:
        raise ValueError("candidate pool indices are not canonical")
    keys = [
        (str(record["image_sha256"]), str(record["image_member_path"]))
        for record in pool
    ]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ValueError("candidate pool keys are not unique and sorted")
    orientation_counts = {
        orientation: sum(record["orientation"] == orientation for record in pool)
        for orientation in ("portrait", "landscape")
    }
    orientation_counts["square"] = 0
    if orientation_counts != EXPECTED_ORIENTATION_COUNTS:
        raise ValueError("candidate pool orientation counts drifted")
    if sum(len(record["eligible_occurrences"]) for record in pool) != (
        EXPECTED_OCCURRENCE_COUNT
    ):
        raise ValueError("candidate pool occurrence accounting drifted")
    for record in pool:
        image_sha256 = _require_sha256(
            record.get("image_sha256"), "candidate image SHA256"
        )
        image_member_path = _safe_member_path(record.get("image_member_path"))
        if record.get("orientation") != _orientation(
            int(record["width"]), int(record["height"])
        ):
            raise ValueError("candidate image orientation drifted")
        occurrences = record.get("eligible_occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            raise ValueError("candidate image must retain eligible occurrences")
        if occurrences != sorted(occurrences, key=_occurrence_sort_key):
            raise ValueError("candidate image occurrences are not canonical")
        occurrence_paths = []
        for occurrence in occurrences:
            if occurrence.get("image_sha256") != image_sha256:
                raise ValueError("candidate occurrence image SHA256 drifted")
            occurrence_paths.append(
                _safe_member_path(occurrence.get("image_member_path"))
            )
        if image_member_path != min(occurrence_paths):
            raise ValueError("candidate representative path is not canonical")


def select_real_screen_golden(
    pool: Sequence[Mapping[str, Any]],
    *,
    backend_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    validate_candidate_pool(pool)
    contract = backend_config["golden_contract"]
    rule = contract["real_screen_selection"]
    if tuple(rule["eligible_state_roles"]) != ELIGIBLE_ROLE_ORDER:
        raise ValueError("real-screen eligible state roles drifted")
    if tuple(rule["eligible_image_roles"]) != ELIGIBLE_IMAGE_ROLE_ORDER:
        raise ValueError("real-screen eligible image roles drifted")
    if rule["deduplicate_by"] != "image_sha256":
        raise ValueError("real-screen deduplication rule drifted")
    if tuple(rule["orientation_strata"]) != ("portrait", "landscape"):
        raise ValueError("real-screen orientation strata drifted")
    if int(rule["count_per_stratum"]) != 3:
        raise ValueError("real-screen count per stratum drifted")
    if tuple(rule["within_stratum_order"]) != (
        "image_sha256",
        "image_member_path",
    ):
        raise ValueError("real-screen within-stratum order drifted")
    if contract["confirm_images_may_be_used_for_golden_selection"] is not False:
        raise ValueError("confirm images cannot enter real-screen golden selection")

    selected: list[dict[str, Any]] = []
    for orientation in rule["orientation_strata"]:
        candidates = [
            dict(record) for record in pool if record["orientation"] == orientation
        ]
        candidates.sort(
            key=lambda record: (record["image_sha256"], record["image_member_path"])
        )
        if len(candidates) < int(rule["count_per_stratum"]):
            raise ValueError(rule["insufficient_stratum_outcome"])
        for rank, record in enumerate(candidates[: int(rule["count_per_stratum"])]):
            record["orientation_rank"] = rank
            record["golden_case_id"] = f"{orientation}-{rank:03d}"
            record["image_tar_member"] = f"images/{record['image_sha256']}.png"
            selected.append(record)

    observed = {
        orientation: tuple(
            {
                "image_sha256": record["image_sha256"],
                "image_member_path": record["image_member_path"],
                "width": record["width"],
                "height": record["height"],
            }
            for record in selected
            if record["orientation"] == orientation
        )
        for orientation in ("portrait", "landscape")
    }
    if observed != EXPECTED_SELECTED_IMAGES:
        raise ValueError("real-screen frozen selected image keys drifted")
    return selected


def _iter_parquet_rows(parquet_file: Any) -> Iterator[tuple[int, Mapping[str, Any]]]:
    row_index = 0
    for row_group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_index)
        for row in table.to_pylist():
            yield row_index, row
            row_index += 1
    if row_index != parquet_file.metadata.num_rows:
        raise RuntimeError("Parquet row iteration did not match metadata.num_rows")


def load_screening_image_payloads(
    selection_manifest: Mapping[str, Any],
    *,
    source_root: str | Path,
    source_file_manifest: Mapping[str, Any],
    v1_config: Mapping[str, Any],
    derived_repo: str,
) -> dict[str, bytes]:
    occurrences, _ = collect_real_screen_occurrences(selection_manifest)
    needed_paths_by_source: dict[str, set[str]] = defaultdict(set)
    expected_sha_by_path: dict[str, str] = {}
    for record in occurrences:
        path = record["image_member_path"]
        needed_paths_by_source[record["source_id"]].add(path)
        previous = expected_sha_by_path.setdefault(path, record["image_sha256"])
        if previous != record["image_sha256"]:
            raise ValueError("one source member path maps to multiple image SHAs")

    specs = source_file_specs(v1_config, source_file_manifest)
    root = Path(source_root)
    verify_local_source_files(root, specs)
    spec_by_path = {spec.transport_file: spec for spec in specs}
    trajectory_records = {}
    for role in ELIGIBLE_ROLE_ORDER:
        for record in selection_manifest["roles"][role]["trajectories"]:
            source_id = str(record["source_id"])
            if source_id in trajectory_records:
                raise ValueError("screening trajectory source IDs must be unique")
            trajectory_records[source_id] = record
    if set(trajectory_records) != set(needed_paths_by_source):
        raise ValueError("screening trajectory records differ from image occurrences")

    wanted_by_file: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for record in trajectory_records.values():
        transport_file = str(record["transport_file"])
        row_index = int(record["transport_row_index"])
        if row_index in wanted_by_file[transport_file]:
            raise ValueError("duplicate selected Parquet row")
        wanted_by_file[transport_file][row_index] = record

    import pyarrow.parquet as pq

    selected_rows: dict[str, Mapping[str, Any]] = {}
    for transport_file, wanted in wanted_by_file.items():
        spec = spec_by_path.get(transport_file)
        if spec is None:
            raise ValueError("selected transport file is absent from source manifest")
        source_path = root.joinpath(*PurePosixPath(transport_file).parts)
        parquet = pq.ParquetFile(source_path)
        for row_index, row in _iter_parquet_rows(parquet):
            record = wanted.get(row_index)
            if record is not None:
                selected_rows[str(record["source_id"])] = row
    if set(selected_rows) != set(trajectory_records):
        raise ValueError("failed to reload every screening source row")

    payloads: dict[str, bytes] = {}
    for source_id in sorted(selected_rows):
        record = trajectory_records[source_id]
        spec = spec_by_path[str(record["transport_file"])]
        pilot, images = build_pilot_manifest(
            selected_rows[source_id],
            row_index=int(record["transport_row_index"]),
            upstream_repo=str(v1_config["source_pool"]["upstream_repo"]),
            upstream_revision=str(v1_config["source_pool"]["upstream_revision"]),
            transport_repo=str(v1_config["source_pool"]["transport_repo"]),
            transport_revision=str(v1_config["source_pool"]["transport_revision"]),
            transport_file=str(record["transport_file"]),
            transport_file_sha256=spec.sha256,
            hf_destination=derived_repo,
            grid_size=int(v1_config["policy"]["coordinate_grid_size"]),
        )
        trajectory = pilot["trajectory"]
        if trajectory["source_id"] != source_id:
            raise ValueError("screening source ID changed while reloading images")
        if len(trajectory["decisions"]) != int(record["decision_count"]):
            raise ValueError("screening decision count changed while reloading images")
        for member_path in sorted(needed_paths_by_source[source_id]):
            payload = images.get(member_path)
            if payload is None:
                raise ValueError("screening source row is missing an eligible image")
            if sha256_bytes(payload) != expected_sha_by_path[member_path]:
                raise ValueError("screening image differs from selection witness")
            payloads[member_path] = payload
    if set(payloads) != set(expected_sha_by_path):
        raise ValueError("screening image payload inventory is incomplete")
    return payloads


def build_ocr_jsonl(
    selected: Sequence[Mapping[str, Any]],
    *,
    ocr_records_by_sha256: Mapping[str, Mapping[str, Any]],
) -> bytes:
    lines = []
    for record in selected:
        image_sha256 = str(record["image_sha256"])
        ocr_record = ocr_records_by_sha256.get(image_sha256)
        if not isinstance(ocr_record, Mapping):
            raise ValueError("selected image is missing its OCR record")
        if ocr_record.get("image_sha256") != image_sha256:
            raise ValueError("selected OCR record image SHA256 drifted")
        if ocr_record.get("image_member_path") != record["image_member_path"]:
            raise ValueError("selected OCR record source member path drifted")
        lines.append(
            canonical_json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol_id": PROTOCOL_ID,
                    "golden_case_id": record["golden_case_id"],
                    "candidate_pool_index": record["candidate_pool_index"],
                    "orientation": record["orientation"],
                    "orientation_rank": record["orientation_rank"],
                    "image_sha256": image_sha256,
                    "image_member_path": record["image_member_path"],
                    "image_tar_member": record["image_tar_member"],
                    "ocr_record": dict(ocr_record),
                }
            )
        )
    return b"\n".join(lines) + b"\n"


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    info.type = tarfile.REGTYPE
    return info


def build_selected_image_tar_bytes(
    selected: Sequence[Mapping[str, Any]],
    *,
    image_payloads: Mapping[str, bytes],
) -> tuple[bytes, list[tuple[str, bytes]]]:
    members = []
    for record in selected:
        source_path = str(record["image_member_path"])
        payload = image_payloads[source_path]
        if sha256_bytes(payload) != record["image_sha256"]:
            raise ValueError("selected image payload SHA256 drifted before tar write")
        members.append((str(record["image_tar_member"]), payload))
    if [name for name, _ in members] != sorted(name for name, _ in members):
        raise ValueError("selected image tar members are not lexicographically ordered")
    if len({name for name, _ in members}) != len(members):
        raise ValueError("selected image tar members must be unique")
    buffer = io.BytesIO()
    with tarfile.open(
        fileobj=buffer,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for name, payload in members:
            archive.addfile(_tar_info(name, len(payload)), io.BytesIO(payload))
    return buffer.getvalue(), members


def write_selected_image_tar(
    path: str | Path,
    selected: Sequence[Mapping[str, Any]],
    *,
    image_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"real-screen image tar already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tar_bytes, members = build_selected_image_tar_bytes(
        selected,
        image_payloads=image_payloads,
    )
    with output.open("xb") as handle:
        handle.write(tar_bytes)
    return {
        "path": IMAGE_TAR_RELATIVE_PATH,
        "size_bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "member_count": len(members),
        "members": [
            {
                "path": name,
                "size_bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
            for name, payload in members
        ],
    }


def build_payload_manifest(
    *,
    dataset_repo: str,
    source_contract_sha256: str,
    inputs: Mapping[str, Any],
    generator: Mapping[str, Any],
    runtime: Mapping[str, Any],
    backend_config: Mapping[str, Any],
    candidate_pool: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    ocr_records_by_sha256: Mapping[str, Mapping[str, Any]],
    image_tar_file: Mapping[str, Any],
    ocr_jsonl_file: Mapping[str, Any],
) -> dict[str, Any]:
    validate_candidate_pool(candidate_pool)
    expected_selected = select_real_screen_golden(
        candidate_pool,
        backend_config=backend_config,
    )
    if list(selected) != expected_selected:
        raise ValueError("real-screen selected records drifted before manifest build")
    source_contract_sha256 = _require_sha256(
        source_contract_sha256,
        "real-screen source contract SHA256",
    )
    payload_files = [dict(image_tar_file), dict(ocr_jsonl_file)]
    if [record.get("path") for record in payload_files] != [
        IMAGE_TAR_RELATIVE_PATH,
        OCR_JSONL_RELATIVE_PATH,
    ]:
        raise ValueError("real-screen payload file paths drifted")
    selected_images = []
    for record in selected:
        ocr_record = ocr_records_by_sha256[record["image_sha256"]]
        selected_images.append(
            {
                "golden_case_id": record["golden_case_id"],
                "candidate_pool_index": record["candidate_pool_index"],
                "orientation": record["orientation"],
                "orientation_rank": record["orientation_rank"],
                "image_sha256": record["image_sha256"],
                "image_member_path": record["image_member_path"],
                "image_tar_member": record["image_tar_member"],
                "width": record["width"],
                "height": record["height"],
                "ocr_node_count": len(ocr_record["nodes"]),
                "full_spatial_tokens_sha256": ocr_record[
                    "full_spatial_tokens_sha256"
                ],
                "canonical_ocr_record_sha256": ocr_record[
                    "canonical_ocr_record_sha256"
                ],
            }
        )
    file_index = [
        {
            "path": record["path"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in payload_files
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "artifact_id": ARTIFACT_ID,
        "status": "MATERIALIZED_POLICY_BLIND_REAL_SCREEN_GOLDEN",
        "dataset_repo": dataset_repo,
        "payload_prefix": PAYLOAD_PREFIX,
        "policy_output_used": False,
        "restoration_output_used": False,
        "confirm_images_used": False,
        "source_contract_sha256": source_contract_sha256,
        "inputs": dict(inputs),
        "generator": dict(generator),
        "runtime": dict(runtime),
        "selection": {
            "rule": dict(backend_config["golden_contract"]["real_screen_selection"]),
            "same_sha_representative_path": "minimum_image_member_path",
            "orientation_definition": {
                "portrait": "height>width",
                "landscape": "width>height",
                "square": "INVALID_DERIVED_ARTIFACT_BEFORE_POLICY_OUTPUT",
            },
            "eligible_state_count": EXPECTED_STATE_COUNT,
            "eligible_occurrence_count": EXPECTED_OCCURRENCE_COUNT,
            "candidate_post_state_occurrence_count": (
                EXPECTED_CANDIDATE_OCCURRENCE_COUNT
            ),
            "current_observation_occurrence_count": (
                EXPECTED_CURRENT_OCCURRENCE_COUNT
            ),
            "unique_image_count": EXPECTED_UNIQUE_IMAGE_COUNT,
            "orientation_counts": dict(EXPECTED_ORIENTATION_COUNTS),
            "confirm_unique_image_count": EXPECTED_CONFIRM_UNIQUE_IMAGE_COUNT,
            "confirm_sha_intersection_count": EXPECTED_CONFIRM_INTERSECTION_COUNT,
            "selected_image_count": len(selected),
            "selected_images": selected_images,
        },
        "candidate_pool": [dict(record) for record in candidate_pool],
        "files": payload_files,
        "payload_index_sha256": sha256_bytes(canonical_json_bytes(file_index)),
    }


def artifact_readme_bytes() -> bytes:
    return (
        "---\n"
        "license: cc-by-4.0\n"
        "task_categories:\n"
        "- image-to-text\n"
        "---\n\n"
        "# CausalCache restoration-v2 real-screen OCR golden\n\n"
        "This private dataset prefix freezes the policy-blind six-image OCR "
        "behavioral golden for CausalCache restoration-v2. It contains raw "
        "GUIOdyssey PNG bytes, deterministic OCR JSONL, and a provenance "
        "manifest. It does not contain policy outputs, restoration outputs, "
        "or confirm images. The full derived dataset remains a separate "
        "pre-output dependency.\n"
    ).encode("utf-8")


def artifact_gitattributes_bytes() -> bytes:
    return b"*.tar filter=lfs diff=lfs merge=lfs -text\n"


def artifact_tree_identity(root: str | Path) -> dict[str, Any]:
    artifact_root = Path(root)
    observed = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.rglob("*")
        if path.is_file()
    }
    if observed != set(ARTIFACT_RELATIVE_PATHS):
        raise ValueError("real-screen artifact file inventory drifted")
    files = []
    for relative in ARTIFACT_RELATIVE_PATHS:
        path = artifact_root.joinpath(*PurePosixPath(relative).parts)
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "files": files,
        "artifact_tree_sha256": sha256_bytes(canonical_json_bytes(files)),
    }
