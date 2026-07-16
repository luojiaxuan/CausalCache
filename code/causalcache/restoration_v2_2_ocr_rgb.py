"""CPU-only OCR/RGB baseline reduction over the frozen primary geometry slice."""

from __future__ import annotations

import itertools
import math
import re
import sys
import tarfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.data.guiodyssey_restoration_v2 import validate_ocr_record
from causalcache.restoration_v2_2_ocr_rgb_contract import (
    ALLOWED_ROLES,
    EXPECTED_OPERATION_CEILING,
    PROTOCOL_ID,
    RestorationV22OcrRgbContract,
    canonical_json_bytes,
    sha256_file,
    strict_json_object_bytes,
)
from causalcache.restoration_v2_2_geometry_stats import (
    StateMetricRow,
    paired_trajectory_bootstrap,
)
from causalcache.restoration_v2_2_selector_geometry import (
    SelectorGeometryState,
    load_selector_geometry_states,
)
from causalcache.restoration_v2_baselines import (
    joint_rgb_histogram_cosine,
    ocr_token_set_jaccard,
    select_top_two,
)
from causalcache.restoration_v2_text_backend import (
    PreparedImage,
    load_backend_config,
)


SCHEMA_VERSION = "1.0.0"
STATUS = "COMPLETED_RESTORATION_V2_2_OCR_RGB_BASELINE_V1"
PRIMARY_EVENT_IDS = (1, 2, 3, 4)
PRIMARY_BUDGET = 2
PRIMARY_DECISION_STEP = 6
CURRENT_EQUIVALENT_EVENT_STEP = 5
NORMALIZATION_EPSILON = 1e-12
GEOMETRY_PRIMARY_MARKERS = (
    b'"budget_event_capacity":2',
    b'"candidate_event_count":4',
)
COMPARATOR_METHODS = (
    "exact_subset",
    "exact_cardinality_oracle",
    "true_conditional_greedy",
    "budget_conditioned_independent",
    "full_shapley_independent",
    "dynamic_recent",
    "analytic_exact_cardinality_random",
)


@dataclass(frozen=True)
class ImageRequirement:
    image_member_path: str
    image_sha256: str
    canonical_ocr_record_sha256: str


@dataclass(frozen=True)
class SelectedImageFeature:
    image_member_path: str
    image_sha256: str
    canonical_ocr_record_sha256: str
    full_spatial_tokens: tuple[str, ...]
    resized_rgb_bytes: bytes


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


def _jsonl_lines(path: Path, *, expected_count: int, label: str) -> tuple[bytes, ...]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{label} must be non-empty newline-terminated JSONL")
    lines = tuple(payload.splitlines(keepends=True))
    if len(lines) != expected_count or any(not line.endswith(b"\n") for line in lines):
        raise ValueError(f"{label} line denominator drifted")
    return lines


def _parse_canonical_line(line: bytes, *, label: str) -> dict[str, Any]:
    if not line.endswith(b"\n"):
        raise ValueError(f"{label} is not newline terminated")
    record = strict_json_object_bytes(line[:-1], label=label)
    if canonical_json_bytes(record) + b"\n" != line:
        raise ValueError(f"{label} is not canonical JSONL")
    return record


def select_jsonl_records_by_identity(
    path: Path,
    *,
    identity_field: str,
    allowed_identities: Iterable[str],
    expected_total_line_count: int,
    label: str,
) -> dict[str, dict[str, Any]]:
    """Parse only allowlisted records while treating every other line as opaque bytes."""
    allowed = frozenset(allowed_identities)
    if not allowed or any(not isinstance(value, str) or not value for value in allowed):
        raise ValueError("allowed identities must be non-empty strings")
    field = re.escape(identity_field.encode("ascii"))
    identity_pattern = re.compile(rb'"' + field + rb'":"([^"\\]+)"')
    selected: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(
        _jsonl_lines(path, expected_count=expected_total_line_count, label=label)
    ):
        matches = identity_pattern.findall(line)
        if len(matches) != 1:
            raise ValueError(f"{label} line {index} has invalid identity encoding")
        try:
            identity = matches[0].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{label} line {index} identity is not UTF-8") from error
        if identity not in allowed:
            continue
        if identity in selected:
            raise ValueError(f"{label} contains duplicate selected identity: {identity}")
        record = _parse_canonical_line(line, label=f"{label}[{index}]")
        if record.get(identity_field) != identity:
            raise ValueError(f"{label} selected identity changed after JSON parsing")
        selected[identity] = record
    if set(selected) != set(allowed):
        raise ValueError(
            f"{label} selected identity coverage drifted: "
            f"missing={sorted(set(allowed) - set(selected))}"
        )
    return selected


def load_primary_geometry_records(
    contract: RestorationV22OcrRgbContract,
) -> tuple[dict[str, Any], ...]:
    geometry = contract.data["immutable_inputs"]["selector_geometry_result"]
    path = contract.repository_root / geometry["directory"] / (
        "state_budget_records.jsonl"
    )
    records = []
    for index, line in enumerate(
        _jsonl_lines(path, expected_count=180, label="geometry state records")
    ):
        if not all(marker in line for marker in GEOMETRY_PRIMARY_MARKERS):
            continue
        record = _parse_canonical_line(line, label=f"geometry primary[{index}]")
        state = record.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("geometry primary record lacks state metadata")
        if (
            record.get("budget_event_capacity") != PRIMARY_BUDGET
            or state.get("candidate_event_count") != len(PRIMARY_EVENT_IDS)
            or state.get("candidate_event_step_ids") != list(PRIMARY_EVENT_IDS)
            or state.get("decision_step_id") != PRIMARY_DECISION_STEP
            or state.get("role") not in ALLOWED_ROLES
        ):
            raise ValueError("geometry primary scope drifted")
        trajectory_id = state.get("trajectory_id")
        state_id = state.get("state_id")
        if (
            not isinstance(trajectory_id, str)
            or state_id != f"{trajectory_id}:decision_step:006"
        ):
            raise ValueError("geometry primary trajectory/state identity drifted")
        records.append(record)
    if len(records) != 15:
        raise ValueError("geometry primary slice must contain exactly 15 states")
    state_ids = [record["state"]["state_id"] for record in records]
    trajectories = [record["state"]["trajectory_id"] for record in records]
    if len(set(state_ids)) != 15 or len(set(trajectories)) != 15:
        raise ValueError("geometry primary state and trajectory ids must be unique")
    if Counter(record["state"]["role"] for record in records) != {
        "v2_label_train": 10,
        "v2_development": 5,
    }:
        raise ValueError("geometry primary role denominator drifted")
    return tuple(sorted(records, key=lambda item: item["state"]["index"]))


def validate_derived_projection(
    contract: RestorationV22OcrRgbContract,
    derived_root: Path,
) -> None:
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
        raise ValueError(
            "derived projection exact-six inventory drifted: "
            f"missing={sorted(set(expected) - set(observed))}, "
            f"extra={sorted(set(observed) - set(expected))}"
        )
    for relative, path in observed.items():
        size, digest = expected[relative]
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"derived immutable file identity drifted: {relative}")


def _selected_trajectory_inputs(
    geometry_records: Sequence[Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, tuple[dict[int, ImageRequirement], ImageRequirement]],
    frozenset[str],
]:
    result: dict[str, tuple[dict[int, ImageRequirement], ImageRequirement]] = {}
    image_paths: set[str] = set()
    for geometry in geometry_records:
        state = geometry["state"]
        trajectory_id = state["trajectory_id"]
        trajectory = trajectories[trajectory_id]
        if trajectory.get("role") != state["role"] or trajectory.get("source_id") != (
            trajectory_id
        ):
            raise ValueError("derived trajectory role or source identity drifted")
        if trajectory.get("role") not in ALLOWED_ROLES:
            raise ValueError("non-train/development trajectory reached semantic parsing")
        decisions = trajectory.get("decisions")
        events = trajectory.get("events")
        if not isinstance(decisions, list) or not isinstance(events, list):
            raise ValueError("selected derived trajectory schema drifted")
        selected_decisions = [
            decision
            for decision in decisions
            if isinstance(decision, Mapping)
            and decision.get("decision_step_id") == PRIMARY_DECISION_STEP
        ]
        if len(selected_decisions) != 1:
            raise ValueError("selected trajectory must contain one decision step 6")
        decision = selected_decisions[0]
        if (
            decision.get("state_id") != state["state_id"]
            or decision.get("candidate_event_step_ids") != list(PRIMARY_EVENT_IDS)
            or decision.get("current_equivalent_event_step_id")
            != CURRENT_EQUIVALENT_EVENT_STEP
        ):
            raise ValueError("selected trajectory decision-step contract drifted")
        event_by_step = {
            event.get("step_id"): event
            for event in events
            if isinstance(event, Mapping) and type(event.get("step_id")) is int
        }
        if not set((*PRIMARY_EVENT_IDS, CURRENT_EQUIVALENT_EVENT_STEP)).issubset(
            event_by_step
        ):
            raise ValueError("selected trajectory event coverage drifted")

        event_requirements: dict[int, ImageRequirement] = {}
        for step_id in PRIMARY_EVENT_IDS:
            event = event_by_step[step_id]
            high = event.get("high_fidelity_v2")
            refs = event.get("ocr_record_refs")
            if not isinstance(high, Mapping) or not isinstance(refs, Mapping):
                raise ValueError("candidate event lacks high-fidelity/OCR identity")
            path = _safe_member_path(
                event.get("observation_after_path"),
                "candidate post image",
            )
            digest = event.get("observation_after_sha256")
            if (
                high.get("image_member_path") != path
                or high.get("image_sha256") != digest
                or high.get("image_role") != "post_action_state"
                or high.get("include_before_image") is not False
                or high.get("include_additional_action_text") is not False
            ):
                raise ValueError("candidate post-state high-fidelity identity drifted")
            requirement = ImageRequirement(
                image_member_path=path,
                image_sha256=str(digest),
                canonical_ocr_record_sha256=str(
                    refs.get("after_canonical_ocr_record_sha256")
                ),
            )
            event_requirements[step_id] = requirement
            image_paths.add(path)

        current_event = event_by_step[CURRENT_EQUIVALENT_EVENT_STEP]
        current_refs = current_event.get("ocr_record_refs")
        if not isinstance(current_refs, Mapping):
            raise ValueError("current-equivalent event lacks OCR identity")
        current_path = _safe_member_path(
            decision.get("current_observation_path"),
            "current observation",
        )
        current_digest = decision.get("current_observation_sha256")
        if (
            current_event.get("observation_after_path") != current_path
            or current_event.get("observation_after_sha256") != current_digest
        ):
            raise ValueError("decision current image is not event-5 post-state")
        current_requirement = ImageRequirement(
            image_member_path=current_path,
            image_sha256=str(current_digest),
            canonical_ocr_record_sha256=str(
                current_refs.get("after_canonical_ocr_record_sha256")
            ),
        )
        image_paths.add(current_path)
        result[state["state_id"]] = (event_requirements, current_requirement)
    if len(result) != 15 or len(image_paths) != 75:
        raise ValueError("selected trajectory inputs must bind 15 states and 75 images")
    return result, frozenset(image_paths)


def _selected_image_payloads(
    image_tar: Path,
    selected_paths: frozenset[str],
    *,
    expected_member_count: int,
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    with tarfile.open(image_tar, mode="r:") as archive:
        members = archive.getmembers()
        if len(members) != expected_member_count:
            raise ValueError("derived image tar member denominator drifted")
        names: set[str] = set()
        for member in members:
            name = _safe_member_path(member.name, "tar member")
            if name in names or not member.isfile() or member.issym() or member.islnk():
                raise ValueError("derived image tar contains duplicate or non-regular member")
            names.add(name)
            if name not in selected_paths:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read selected image member: {name}")
            payloads[name] = handle.read()
    if set(payloads) != set(selected_paths):
        raise ValueError("selected image payload coverage drifted")
    return payloads


def _validate_runtime(contract: RestorationV22OcrRgbContract) -> None:
    runtime = contract.data["immutable_inputs"]["baseline_implementation"]["runtime"]
    if f"{sys.version_info.major}.{sys.version_info.minor}" != runtime[
        "python_major_minor"
    ]:
        raise ValueError("formal OCR/RGB runtime Python major/minor drifted")
    try:
        import PIL
    except ImportError as error:
        raise RuntimeError("formal OCR/RGB runtime requires pinned Pillow") from error
    if PIL.__version__ != runtime["pillow_version"]:
        raise ValueError("formal OCR/RGB Pillow version drifted")


def _materialize_selected_features(
    contract: RestorationV22OcrRgbContract,
    derived_root: Path,
    requirements: Mapping[
        str,
        tuple[Mapping[int, ImageRequirement], ImageRequirement],
    ],
    selected_paths: frozenset[str],
) -> dict[str, SelectedImageFeature]:
    derived = contract.data["immutable_inputs"]["derived_dataset"]
    prefix = derived["payload_prefix"]
    ocr_path = derived_root / prefix / "ocr-records-00000-of-00001.jsonl"
    tar_path = derived_root / prefix / "images-00000-of-00001.tar"
    ocr_records = select_jsonl_records_by_identity(
        ocr_path,
        identity_field="image_member_path",
        allowed_identities=selected_paths,
        expected_total_line_count=derived["counts"]["ocr_record_count"],
        label="derived OCR records",
    )
    payloads = _selected_image_payloads(
        tar_path,
        selected_paths,
        expected_member_count=derived["counts"]["image_member_count"],
    )
    expected_by_path: dict[str, ImageRequirement] = {}
    for event_requirements, current in requirements.values():
        for requirement in (*event_requirements.values(), current):
            previous = expected_by_path.setdefault(
                requirement.image_member_path,
                requirement,
            )
            if previous != requirement:
                raise ValueError("selected image requirement identity is inconsistent")
    if set(expected_by_path) != set(selected_paths):
        raise ValueError("selected image requirements drifted")

    backend_path = contract.repository_root / "code/configs/restoration_v2_ocr_backend.json"
    backend_config = load_backend_config(backend_path)
    backend_sha = contract.data["immutable_inputs"]["baseline_implementation"][
        "runtime"
    ]["ocr_backend_config_sha256"]
    result: dict[str, SelectedImageFeature] = {}
    for path in sorted(selected_paths):
        requirement = expected_by_path[path]
        record = ocr_records[path]
        payload = payloads[path]
        prepared: PreparedImage = validate_ocr_record(
            record,
            image_bytes=payload,
            backend_config=backend_config,
            backend_config_sha256=backend_sha,
        )
        if (
            record.get("image_sha256") != requirement.image_sha256
            or record.get("canonical_ocr_record_sha256")
            != requirement.canonical_ocr_record_sha256
        ):
            raise ValueError("selected image trajectory/OCR identity drifted")
        tokens = record.get("full_spatial_tokens")
        if not isinstance(tokens, list) or any(not isinstance(token, str) for token in tokens):
            raise ValueError("selected OCR full spatial tokens are invalid")
        result[path] = SelectedImageFeature(
            image_member_path=path,
            image_sha256=requirement.image_sha256,
            canonical_ocr_record_sha256=requirement.canonical_ocr_record_sha256,
            full_spatial_tokens=tuple(tokens),
            resized_rgb_bytes=prepared.resized_rgb_bytes,
        )
    if len(result) != 75:
        raise ValueError("selected feature materialization must produce 75 images")
    return result


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = frozenset(left)
    right_set = frozenset(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def score_ocr_rgb_state(
    *,
    geometry_record: Mapping[str, Any],
    event_features: Mapping[int, SelectedImageFeature],
    current_feature: SelectedImageFeature,
    distance_by_coalition: Mapping[tuple[int, ...], float],
) -> dict[str, Any]:
    """Score one synthetic or formal primary state without loading any model."""
    if set(event_features) != set(PRIMARY_EVENT_IDS):
        raise ValueError("state scoring requires event features 1..4 exactly")
    expected_coalitions = {
        coalition
        for cardinality in range(len(PRIMARY_EVENT_IDS) + 1)
        for coalition in itertools.combinations(PRIMARY_EVENT_IDS, cardinality)
    }
    if set(distance_by_coalition) != expected_coalitions:
        raise ValueError("state scoring requires the complete 2^4 distance table")
    distances = {
        coalition: _finite_float(value, f"distance {coalition}")
        for coalition, value in distance_by_coalition.items()
    }
    if any(value < 0.0 for value in distances.values()):
        raise ValueError("distance table values must be non-negative")
    state = geometry_record.get("state")
    methods = geometry_record.get("methods")
    if not isinstance(state, Mapping) or not isinstance(methods, Mapping):
        raise ValueError("geometry record lacks state or method metadata")
    if (
        state.get("candidate_event_step_ids") != list(PRIMARY_EVENT_IDS)
        or geometry_record.get("budget_event_capacity") != PRIMARY_BUDGET
    ):
        raise ValueError("geometry record is outside primary n4B2")
    baseline_distance = distances[()]
    if not math.isclose(
        _finite_float(
            geometry_record.get("baseline_summary_only_distance"),
            "geometry baseline distance",
        ),
        baseline_distance,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("geometry and label empty-coalition distances differ")
    feasible = [
        (coalition, distance)
        for coalition, distance in distances.items()
        if len(coalition) <= PRIMARY_BUDGET
    ]
    exact_coalition, exact_distance = min(
        feasible,
        key=lambda item: (item[1], len(item[0]), item[0]),
    )
    geometry_exact = methods.get("exact_subset")
    if not isinstance(geometry_exact, Mapping) or geometry_exact.get(
        "selected_coalition"
    ) != list(exact_coalition):
        raise ValueError("geometry exact subset differs from the raw distance table")

    event_tokens = {
        step_id: event_features[step_id].full_spatial_tokens
        for step_id in PRIMARY_EVENT_IDS
    }
    event_rgb = {
        step_id: event_features[step_id].resized_rgb_bytes
        for step_id in PRIMARY_EVENT_IDS
    }
    combined_scores: dict[int, float] = {}
    candidate_rows = []
    for step_id in PRIMARY_EVENT_IDS:
        text_score = ocr_token_set_jaccard(
            event_tokens[step_id],
            current_feature.full_spatial_tokens,
        )
        rgb_score = joint_rgb_histogram_cosine(
            event_rgb[step_id],
            current_feature.resized_rgb_bytes,
        )
        expected_combined = 0.5 * text_score + 0.5 * rgb_score
        combined_scores[step_id] = expected_combined
        feature = event_features[step_id]
        candidate_rows.append(
            {
                "event_step_id": step_id,
                "post_image_member_path": feature.image_member_path,
                "post_image_sha256": feature.image_sha256,
                "post_canonical_ocr_record_sha256": (
                    feature.canonical_ocr_record_sha256
                ),
                "ocr_jaccard": text_score,
                "rgb_histogram_cosine": rgb_score,
                "combined_similarity": expected_combined,
            }
        )
    selection = select_top_two(combined_scores)
    selected = tuple(selection.selected_event_step_ids)
    selected_distance = distances[selected]
    utility = baseline_distance - selected_distance
    recovery = (
        utility / baseline_distance
        if baseline_distance > NORMALIZATION_EPSILON
        else None
    )
    exact_utility = baseline_distance - exact_distance
    exact_recovery = (
        exact_utility / baseline_distance
        if baseline_distance > NORMALIZATION_EPSILON
        else None
    )
    exact_cardinality_coalition, exact_cardinality_distance = min(
        (
            (coalition, distance)
            for coalition, distance in distances.items()
            if len(coalition) == PRIMARY_BUDGET
        ),
        key=lambda item: (item[1], item[0]),
    )
    geometry_exact_cardinality = methods.get("exact_cardinality_oracle")
    if not isinstance(
        geometry_exact_cardinality, Mapping
    ) or geometry_exact_cardinality.get("selected_coalition") != list(
        exact_cardinality_coalition
    ):
        raise ValueError(
            "geometry exact-cardinality oracle differs from the raw distance table"
        )
    exact_cardinality_utility = baseline_distance - exact_cardinality_distance
    exact_cardinality_recovery = (
        exact_cardinality_utility / baseline_distance
        if baseline_distance > NORMALIZATION_EPSILON
        else None
    )
    comparators = {}
    for method_name in COMPARATOR_METHODS:
        method = methods.get(method_name)
        if not isinstance(method, Mapping):
            raise ValueError(f"geometry comparator is absent: {method_name}")
        comparators[method_name] = {
            "selected_coalition": method.get("selected_coalition"),
            "normalized_recovery": method.get("normalized_recovery"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "state": {
            "index": state.get("index"),
            "role": state.get("role"),
            "trajectory_id": state.get("trajectory_id"),
            "state_id": state.get("state_id"),
            "decision_step_id": state.get("decision_step_id"),
            "candidate_event_step_ids": list(PRIMARY_EVENT_IDS),
            "budget_event_capacity": PRIMARY_BUDGET,
        },
        "current_image": {
            "image_member_path": current_feature.image_member_path,
            "image_sha256": current_feature.image_sha256,
            "canonical_ocr_record_sha256": (
                current_feature.canonical_ocr_record_sha256
            ),
        },
        "candidate_scores": candidate_rows,
        "ranked_event_step_ids": list(selection.ranked_event_step_ids),
        "selected_coalition": list(selected),
        "baseline_summary_only_distance": baseline_distance,
        "selected_distance": selected_distance,
        "actual_utility": utility,
        "normalized_recovery": recovery,
        "exact_subset_coalition": list(exact_coalition),
        "exact_subset_distance": exact_distance,
        "exact_subset_normalized_recovery": exact_recovery,
        "absolute_regret_to_exact_subset": exact_utility - utility,
        "normalized_recovery_regret_to_exact_subset": (
            None if recovery is None or exact_recovery is None else exact_recovery - recovery
        ),
        "exact_coalition_match": selected == exact_coalition,
        "jaccard_to_exact_coalition": _jaccard(selected, exact_coalition),
        "exact_cardinality_oracle_coalition": list(exact_cardinality_coalition),
        "exact_cardinality_oracle_distance": exact_cardinality_distance,
        "exact_cardinality_oracle_normalized_recovery": (
            exact_cardinality_recovery
        ),
        "absolute_regret_to_exact_cardinality_oracle": (
            exact_cardinality_utility - utility
        ),
        "normalized_recovery_regret_to_exact_cardinality_oracle": (
            None
            if recovery is None or exact_cardinality_recovery is None
            else exact_cardinality_recovery - recovery
        ),
        "exact_cardinality_coalition_match": (
            selected == exact_cardinality_coalition
        ),
        "jaccard_to_exact_cardinality_coalition": _jaccard(
            selected,
            exact_cardinality_coalition,
        ),
        "geometry_comparators": comparators,
    }


def build_state_score_records(
    *,
    contract: RestorationV22OcrRgbContract,
    labels_archive: Path,
    derived_root: Path,
) -> tuple[dict[str, Any], ...]:
    contract.validate_bound_sources()
    validate_derived_projection(contract, derived_root)
    _validate_runtime(contract)
    geometry_records = load_primary_geometry_records(contract)
    trajectory_ids = {
        record["state"]["trajectory_id"] for record in geometry_records
    }
    derived = contract.data["immutable_inputs"]["derived_dataset"]
    trajectory_path = (
        derived_root
        / derived["payload_prefix"]
        / "trajectories-00000-of-00001.jsonl"
    )
    trajectories = select_jsonl_records_by_identity(
        trajectory_path,
        identity_field="source_id",
        allowed_identities=trajectory_ids,
        expected_total_line_count=derived["counts"]["trajectory_count"],
        label="derived trajectories",
    )
    requirements, selected_paths = _selected_trajectory_inputs(
        geometry_records,
        trajectories,
    )
    features = _materialize_selected_features(
        contract,
        derived_root,
        requirements,
        selected_paths,
    )

    label_states = load_selector_geometry_states(labels_archive)
    if len(label_states) != 45 or sum(
        len(state.table.rows) for state in label_states
    ) != 420:
        raise ValueError("validated raw-label state or distance-row denominator drifted")
    primary_label_states = {
        state.state_id: state
        for state in label_states
        if state.decision_step_id == PRIMARY_DECISION_STEP
        and state.candidate_event_step_ids == PRIMARY_EVENT_IDS
    }
    expected_state_ids = {
        record["state"]["state_id"] for record in geometry_records
    }
    if set(primary_label_states) != expected_state_ids:
        raise ValueError("geometry and raw-label primary state inventories differ")
    if sum(len(state.table.rows) for state in primary_label_states.values()) != 240:
        raise ValueError("primary raw-label distance-row denominator drifted")

    results = []
    for geometry in geometry_records:
        state_id = geometry["state"]["state_id"]
        label_state: SelectorGeometryState = primary_label_states[state_id]
        if (
            label_state.role != geometry["state"]["role"]
            or label_state.trajectory_id != geometry["state"]["trajectory_id"]
            or label_state.state_id != state_id
            or label_state.decision_step_id != PRIMARY_DECISION_STEP
            or label_state.candidate_event_step_ids != PRIMARY_EVENT_IDS
        ):
            raise ValueError("geometry and raw-label primary state witnesses differ")
        event_requirements, current_requirement = requirements[state_id]
        event_features = {
            step_id: features[requirement.image_member_path]
            for step_id, requirement in event_requirements.items()
        }
        current_feature = features[current_requirement.image_member_path]
        distances = {
            tuple(row.coalition): row.distance for row in label_state.table.rows
        }
        results.append(
            score_ocr_rgb_state(
                geometry_record=geometry,
                event_features=event_features,
                current_feature=current_feature,
                distance_by_coalition=distances,
            )
        )
    if len(results) != 15 or sum(
        len(record["candidate_scores"]) for record in results
    ) != 60:
        raise RuntimeError("formal OCR/RGB reducer denominator drifted")
    if Counter(record["state"]["role"] for record in results) != {
        "v2_label_train": 10,
        "v2_development": 5,
    }:
        raise RuntimeError("formal OCR/RGB reducer role denominator drifted")
    return tuple(results)


def summarize_state_scores(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(records) != 15:
        raise ValueError("formal OCR/RGB summary requires exactly 15 states")

    def mean(values: Sequence[float]) -> float:
        if not values:
            raise ValueError("cannot average an empty metric sequence")
        return math.fsum(values) / len(values)

    by_role: dict[str, Any] = {}
    for role in (*ALLOWED_ROLES, "overall_stratified"):
        selected = [
            record
            for record in records
            if role == "overall_stratified" or record["state"]["role"] == role
        ]
        recoveries = [
            _finite_float(record["normalized_recovery"], "normalized recovery")
            for record in selected
        ]
        exact_recoveries = [
            _finite_float(
                record["exact_subset_normalized_recovery"],
                "exact normalized recovery",
            )
            for record in selected
        ]
        by_role[role] = {
            "state_count": len(selected),
            "trajectory_count": len(
                {record["state"]["trajectory_id"] for record in selected}
            ),
            "mean_actual_utility": mean(
                [float(record["actual_utility"]) for record in selected]
            ),
            "mean_normalized_recovery": mean(recoveries),
            "recovery_ratio_of_means_to_exact_subset": (
                mean(recoveries) / mean(exact_recoveries)
                if abs(mean(exact_recoveries)) > NORMALIZATION_EPSILON
                else None
            ),
            "mean_selected_cardinality": 2.0,
            "exact_coalition_match_rate": mean(
                [float(record["exact_coalition_match"]) for record in selected]
            ),
            "mean_jaccard_to_exact_coalition": mean(
                [float(record["jaccard_to_exact_coalition"]) for record in selected]
            ),
            "exact_cardinality_coalition_match_rate": mean(
                [
                    float(record["exact_cardinality_coalition_match"])
                    for record in selected
                ]
            ),
            "mean_jaccard_to_exact_cardinality_coalition": mean(
                [
                    float(record["jaccard_to_exact_cardinality_coalition"])
                    for record in selected
                ]
            ),
            "selected_cardinality_histogram": {
                "2": len(selected),
            },
            "selection_metric_semantics": "deterministic_selected_coalition",
            "mean_recovery_delta_to_exact_subset": mean(
                [
                    float(record["normalized_recovery"])
                    - float(record["exact_subset_normalized_recovery"])
                    for record in selected
                ]
            ),
            "mean_recovery_delta_to_exact_cardinality_oracle": mean(
                [
                    float(record["normalized_recovery"])
                    - float(
                        record["exact_cardinality_oracle_normalized_recovery"]
                    )
                    for record in selected
                ]
            ),
            "mean_recovery_delta_to_recent": mean(
                [
                    float(record["normalized_recovery"])
                    - float(
                        record["geometry_comparators"]["dynamic_recent"][
                            "normalized_recovery"
                        ]
                    )
                    for record in selected
                ]
            ),
            "mean_recovery_delta_to_budget_conditioned_independent": mean(
                [
                    float(record["normalized_recovery"])
                    - float(
                        record["geometry_comparators"][
                            "budget_conditioned_independent"
                        ]["normalized_recovery"]
                    )
                    for record in selected
                ]
            ),
            "mean_recovery_delta_to_random_expectation": mean(
                [
                    float(record["normalized_recovery"])
                    - float(
                        record["geometry_comparators"][
                            "analytic_exact_cardinality_random"
                        ]["normalized_recovery"]
                    )
                    for record in selected
                ]
            ),
        }

    metric_rows = []
    for record in records:
        metrics = {"ocr_rgb": float(record["normalized_recovery"])}
        for method_name in COMPARATOR_METHODS:
            metrics[method_name] = float(
                record["geometry_comparators"][method_name]["normalized_recovery"]
            )
        metric_rows.append(
            StateMetricRow(
                role=record["state"]["role"],
                trajectory_id=record["state"]["trajectory_id"],
                state_id=record["state"]["state_id"],
                metrics=metrics,
            )
        )

    def serialize_bootstrap(method_name: str) -> dict[str, Any]:
        report = paired_trajectory_bootstrap(
            metric_rows,
            "ocr_rgb",
            method_name,
            resamples=10_000,
            seed=271_828,
            confidence=0.9,
            tie_epsilon=1e-12,
        )
        payload = asdict(report)
        for split in ("train", "development", "overall_stratified"):
            outcomes = payload[split]["win_tie_loss"]
            outcomes["count"] = (
                outcomes["wins"] + outcomes["ties"] + outcomes["losses"]
            )
        return payload

    development_records = sorted(
        (
            record
            for record in records
            if record["state"]["role"] == "v2_development"
        ),
        key=lambda record: record["state"]["trajectory_id"],
    )
    development_paired_deltas = {
        method_name: [
            {
                "trajectory_id": record["state"]["trajectory_id"],
                "ocr_rgb_normalized_recovery": float(
                    record["normalized_recovery"]
                ),
                "comparator_normalized_recovery": float(
                    record["geometry_comparators"][method_name][
                        "normalized_recovery"
                    ]
                ),
                "paired_delta": float(record["normalized_recovery"])
                - float(
                    record["geometry_comparators"][method_name][
                        "normalized_recovery"
                    ]
                ),
            }
            for record in development_records
        ]
        for method_name in COMPARATOR_METHODS
    }

    return {
        "scope": "primary_n4_b2_only",
        "state_count": 15,
        "trajectory_count": 15,
        "candidate_comparison_count": 60,
        "unique_image_count": 75,
        "by_role": by_role,
        "paired_trajectory_bootstrap": {
            method_name: serialize_bootstrap(method_name)
            for method_name in COMPARATOR_METHODS
        },
        "development_paired_deltas": development_paired_deltas,
        "operation_counts": dict(EXPECTED_OPERATION_CEILING),
    }
