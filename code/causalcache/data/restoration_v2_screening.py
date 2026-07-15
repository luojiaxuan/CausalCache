"""Fail-closed loader for the frozen restoration-v2 screening substrate."""

from __future__ import annotations

import json
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from causalcache.data.guiodyssey_restoration_v2 import (
    EXPECTED_FORMAL_COUNTS,
    IMAGE_TAR_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    TRAJECTORY_JSONL_RELATIVE_PATH,
    canonical_json_bytes,
    load_json_object,
    parse_canonical_jsonl,
    sha256_bytes,
    sha256_file,
    validate_artifact,
)
from causalcache.data.restoration_v2_selection import (
    validate_selection_manifest,
    validate_state_content_witnesses,
)
from causalcache.policy.gui_owl_v2 import (
    GUI_OWL_V2_DECISION_STEPS,
    build_gui_owl_v2_mixed_fidelity_messages,
)
from causalcache.restoration_v2_text_backend import load_backend_config


SCREENING_ROLES = ("v2_label_train", "v2_development")
CONFIRM_ROLE = "v2_confirm_primary"
EXPECTED_SCREENING_TRAJECTORIES = 15
EXPECTED_SCREENING_STATES = 45
EXPECTED_ROLE_TRAJECTORY_COUNTS = {
    "v2_label_train": 10,
    "v2_development": 5,
    CONFIRM_ROLE: 20,
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ScreeningState:
    """One member of the immutable 45-state development denominator."""

    index: int
    role: str
    trajectory_id: str
    decision_step_id: int
    candidate_event_step_ids: tuple[int, ...]

    @property
    def state_id(self) -> str:
        return f"{self.trajectory_id}:decision_step:{self.decision_step_id:03d}"


@dataclass(frozen=True)
class ValidatedScreeningArtifact:
    """Read-only screening view that cannot address confirm trajectories."""

    artifact_root: Path
    artifact_tree_sha256: str
    artifact_manifest_sha256: str
    screening_manifest_sha256: str
    states: tuple[ScreeningState, ...]
    validation: Mapping[str, Any]
    _screening_manifest_json: bytes = field(repr=False)
    _screening_image_payloads: Mapping[str, bytes] = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.states) != EXPECTED_SCREENING_STATES:
            raise ValueError("validated screening artifact must contain exactly 45 states")
        if any(state.role not in SCREENING_ROLES for state in self.states):
            raise ValueError("validated screening artifact cannot expose confirm states")

    def image_bytes(self, member_path: str) -> bytes:
        """Return immutable bytes only for a screening-role image member."""
        try:
            return self._screening_image_payloads[member_path]
        except KeyError as error:
            raise ValueError(
                "image member is not part of the frozen screening-role inventory"
            ) from error

    def build_messages(
        self,
        state: ScreeningState,
        *,
        restored_event_step_ids: Sequence[int],
        image_decoder: Callable[[bytes], Any],
    ) -> list[dict[str, Any]]:
        """Build native messages for one exact screening state."""
        if state not in self.states:
            raise ValueError("state is not a member of the frozen screening denominator")
        if state.role not in SCREENING_ROLES:
            raise ValueError("confirm-role message construction is forbidden during screening")
        manifest = json.loads(self._screening_manifest_json)
        return build_gui_owl_v2_mixed_fidelity_messages(
            manifest,
            trajectory_id=state.trajectory_id,
            decision_step_id=state.decision_step_id,
            restored_event_step_ids=restored_event_step_ids,
            image_bytes_loader=self.image_bytes,
            image_decoder=image_decoder,
        )


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _safe_member_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("image member path must be a non-empty string")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError("image member path must be a safe relative POSIX path")
    if not value.startswith("images/"):
        raise ValueError("screening image members must belong to images/")
    return value


def _read_safe_image_tar(
    path: Path,
    *,
    expected_count: int,
    expected_inventory_sha256: str,
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    records: list[dict[str, Any]] = []
    with tarfile.open(path, mode="r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("screening image tar inventory must be sorted and unique")
        for member in members:
            name = _safe_member_path(member.name)
            if not member.isfile() or member.type != tarfile.REGTYPE:
                raise ValueError("screening image tar allows only regular files")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("screening image tar member is unreadable")
            payload = handle.read()
            if not payload or len(payload) != member.size:
                raise ValueError("screening image tar member size drifted")
            digest = sha256_bytes(payload)
            payloads[name] = payload
            records.append(
                {
                    "path": name,
                    "size_bytes": len(payload),
                    "sha256": digest,
                }
            )
    if len(records) != expected_count:
        raise ValueError("screening image tar member count drifted")
    if sha256_bytes(canonical_json_bytes(records)) != expected_inventory_sha256:
        raise ValueError("screening image bytes or member inventory drifted")
    return payloads


def _role_source_ids(scientific_config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    try:
        roles = scientific_config["data"]["roles"]
    except (KeyError, TypeError) as error:
        raise ValueError("scientific config is missing data.roles") from error
    result: dict[str, tuple[str, ...]] = {}
    for role in SCREENING_ROLES:
        record = roles.get(role) if isinstance(roles, Mapping) else None
        values = record.get("source_ids") if isinstance(record, Mapping) else None
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(f"scientific config {role} source_ids are invalid")
        expected_count = EXPECTED_ROLE_TRAJECTORY_COUNTS[role]
        if len(values) != expected_count or len(set(values)) != expected_count:
            raise ValueError(f"scientific config {role} source_ids drifted")
        steps = record.get("state_decision_step_ids")
        if steps != list(GUI_OWL_V2_DECISION_STEPS):
            raise ValueError(f"scientific config {role} decision steps drifted")
        result[role] = tuple(values)
    if set(result[SCREENING_ROLES[0]]) & set(result[SCREENING_ROLES[1]]):
        raise ValueError("scientific screening role source IDs overlap")
    return result


def _screening_trajectories_and_states(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    artifact_manifest: Mapping[str, Any],
    scientific_config: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[ScreeningState, ...]]:
    expected_screening_ids = _role_source_ids(scientific_config)
    role_source_ids = artifact_manifest.get("role_source_ids")
    if not isinstance(role_source_ids, Mapping):
        raise ValueError("derived artifact role_source_ids are missing")
    for role, expected_count in EXPECTED_ROLE_TRAJECTORY_COUNTS.items():
        values = role_source_ids.get(role)
        if not isinstance(values, list) or len(values) != expected_count:
            raise ValueError(f"derived artifact {role} source IDs drifted")
        if len(set(values)) != expected_count:
            raise ValueError(f"derived artifact {role} source IDs are not unique")
    for role in SCREENING_ROLES:
        if tuple(role_source_ids[role]) != expected_screening_ids[role]:
            raise ValueError(f"derived artifact {role} differs from the frozen config")
    selection_roles = selection_manifest.get("roles")
    if not isinstance(selection_roles, Mapping):
        raise ValueError("selection manifest roles are missing")
    for role in (*SCREENING_ROLES, CONFIRM_ROLE):
        selection_role = selection_roles.get(role)
        if not isinstance(selection_role, Mapping):
            raise ValueError(f"selection manifest {role} role is missing")
        selected_trajectories = selection_role.get("trajectories")
        if not isinstance(selected_trajectories, list):
            raise ValueError(f"selection manifest {role} trajectories are invalid")
        selected_ids = [record.get("source_id") for record in selected_trajectories]
        if selected_ids != role_source_ids[role]:
            raise ValueError(f"derived artifact {role} differs from frozen selection IDs")
    all_role_ids = [
        source_id
        for role in (*SCREENING_ROLES, CONFIRM_ROLE)
        for source_id in role_source_ids[role]
    ]
    if len(set(all_role_ids)) != len(all_role_ids):
        raise ValueError("derived artifact role source IDs overlap")

    expected_order = [
        (role, source_id)
        for role in (*SCREENING_ROLES, CONFIRM_ROLE)
        for source_id in role_source_ids[role]
    ]
    observed_order = [
        (record.get("role"), record.get("source_id")) for record in trajectories
    ]
    if observed_order != expected_order:
        raise ValueError("derived trajectory role/source order drifted")

    screening_records: list[dict[str, Any]] = []
    states: list[ScreeningState] = []
    for record in trajectories:
        role = record.get("role")
        if role == CONFIRM_ROLE:
            continue
        if role not in SCREENING_ROLES:
            raise ValueError("unknown trajectory role in derived artifact")
        events = record.get("events")
        decisions = record.get("decisions")
        if not isinstance(events, list) or len(events) != 5:
            raise ValueError("screening trajectory must contain exactly five events")
        if not isinstance(decisions, list) or len(decisions) != 3:
            raise ValueError("screening trajectory must contain exactly three states")
        decision_steps = tuple(
            decision.get("decision_step_id")
            for decision in decisions
            if isinstance(decision, Mapping)
        )
        if decision_steps != GUI_OWL_V2_DECISION_STEPS:
            raise ValueError("screening trajectory decision-step order drifted")
        selected_states = selection_roles[role].get("states")
        if not isinstance(selected_states, list):
            raise ValueError(f"selection manifest {role} states are invalid")
        expected_selected_states = [
            state
            for state in selected_states
            if isinstance(state, Mapping)
            and state.get("source_id") == record.get("source_id")
        ]
        if len(expected_selected_states) != len(decisions):
            raise ValueError("derived screening states differ from frozen selection")
        copied = json.loads(canonical_json_bytes(record))
        screening_records.append(copied)
        for decision, selected_state in zip(
            decisions,
            expected_selected_states,
            strict=True,
        ):
            candidate_ids = decision.get("candidate_event_step_ids")
            if not isinstance(candidate_ids, list) or any(
                type(value) is not int for value in candidate_ids
            ):
                raise ValueError("screening state candidate event IDs are invalid")
            state = ScreeningState(
                index=len(states),
                role=role,
                trajectory_id=str(record["source_id"]),
                decision_step_id=int(decision["decision_step_id"]),
                candidate_event_step_ids=tuple(candidate_ids),
            )
            if decision.get("state_id") != state.state_id:
                raise ValueError("derived decision state_id drifted")
            selected_projection = {
                key: selected_state.get(key)
                for key in (
                    "state_id",
                    "source_id",
                    "decision_step_id",
                    "history_event_step_ids",
                    "candidate_event_step_ids",
                    "current_equivalent_event_step_id",
                )
            }
            derived_projection = {
                "state_id": decision.get("state_id"),
                "source_id": record.get("source_id"),
                "decision_step_id": decision.get("decision_step_id"),
                "history_event_step_ids": decision.get("history_event_step_ids"),
                "candidate_event_step_ids": candidate_ids,
                "current_equivalent_event_step_id": decision.get(
                    "current_equivalent_event_step_id"
                ),
            }
            if derived_projection != selected_projection:
                raise ValueError("derived screening state differs from frozen selection")
            states.append(state)
    if len(screening_records) != EXPECTED_SCREENING_TRAJECTORIES:
        raise ValueError("screening trajectory denominator must contain exactly 15 records")
    if len(states) != EXPECTED_SCREENING_STATES:
        raise ValueError("screening state denominator must contain exactly 45 states")
    return tuple(screening_records), tuple(states)


def _screening_image_inventory(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    all_image_payloads: Mapping[str, bytes],
) -> Mapping[str, bytes]:
    required: dict[str, str] = {}

    def bind(path_value: object, sha_value: object) -> None:
        path = _safe_member_path(path_value)
        digest = _require_sha256(sha_value, name=f"image SHA256 for {path}")
        previous = required.setdefault(path, digest)
        if previous != digest:
            raise ValueError("screening image path has conflicting SHA256 witnesses")

    for trajectory in trajectories:
        for event in trajectory["events"]:
            bind(event.get("observation_before_path"), event.get("observation_before_sha256"))
            bind(event.get("observation_after_path"), event.get("observation_after_sha256"))
        for decision in trajectory["decisions"]:
            bind(
                decision.get("current_observation_path"),
                decision.get("current_observation_sha256"),
            )
    selected: dict[str, bytes] = {}
    for path, expected_sha256 in sorted(required.items()):
        try:
            payload = all_image_payloads[path]
        except KeyError as error:
            raise ValueError("screening trajectory references a missing image") from error
        if sha256_bytes(payload) != expected_sha256:
            raise ValueError("screening trajectory image SHA256 witness drifted")
        selected[path] = payload
    if len(selected) != EXPECTED_SCREENING_TRAJECTORIES * 6:
        raise ValueError("screening image inventory must contain exactly 90 members")
    return MappingProxyType(selected)


def load_validated_screening_artifact(
    *,
    artifact_root: str | Path,
    backend_config_path: str | Path,
    scientific_config_path: str | Path,
    selection_manifest_path: str | Path,
    expected_artifact_tree_sha256: str,
) -> ValidatedScreeningArtifact:
    """Validate the immutable derived artifact and expose only screening roles.

    The existing public validator runs first and remains the authoritative
    artifact/schema validator. This loader then independently reads only
    canonical trajectory JSONL and safe regular tar members, binding every
    loaded byte string back to the validated manifest inventory.
    """
    expected_tree = _require_sha256(
        expected_artifact_tree_sha256,
        name="expected_artifact_tree_sha256",
    )
    root = Path(artifact_root)
    backend_path = Path(backend_config_path)
    scientific_path = Path(scientific_config_path)
    selection_path = Path(selection_manifest_path)
    backend_config = load_backend_config(backend_path)
    backend_sha256 = sha256_file(backend_path)
    _, scientific_config = load_json_object(scientific_path)
    _, selection_manifest = load_json_object(selection_path)
    validate_selection_manifest(selection_manifest, v2_contract=scientific_config)
    validate_state_content_witnesses(selection_manifest)

    validation = validate_artifact(
        output_dir=root,
        backend_config=backend_config,
        backend_config_sha256=backend_sha256,
        selection_manifest=selection_manifest,
        require_formal=False,
        require_ocr_replay=False,
    )
    if validation.get("outcome") != (
        "PASSED_GUIODYSSEY_RESTORATION_V2_ARTIFACT_VALIDATION"
    ):
        raise ValueError("derived artifact public validation did not pass")
    if validation.get("artifact_tree_sha256") != expected_tree:
        raise ValueError("derived artifact tree differs from the immutable execution binding")
    if validation.get("formal_counts_enforced") is not True:
        raise ValueError("screening requires the formal derived artifact")
    if validation.get("counts") != EXPECTED_FORMAL_COUNTS:
        raise ValueError("derived artifact formal counts drifted")
    if validation.get("ocr_replay_performed") is not False:
        raise ValueError("screening loader must not rerun OCR")
    if validation.get("policy_loaded") is not False:
        raise ValueError("derived artifact validation unexpectedly loaded a policy")

    _, artifact_manifest = load_json_object(root / MANIFEST_RELATIVE_PATH)
    if artifact_manifest.get("formal_counts_enforced") is not True:
        raise ValueError("derived artifact manifest is not formal")
    if artifact_manifest.get("counts") != EXPECTED_FORMAL_COUNTS:
        raise ValueError("derived artifact manifest count fields drifted")
    trajectories = parse_canonical_jsonl(
        (root / TRAJECTORY_JSONL_RELATIVE_PATH).read_bytes(),
        label="restoration-v2 screening trajectories",
    )
    screening_trajectories, states = _screening_trajectories_and_states(
        trajectories,
        artifact_manifest=artifact_manifest,
        scientific_config=scientific_config,
        selection_manifest=selection_manifest,
    )
    all_image_payloads = _read_safe_image_tar(
        root / IMAGE_TAR_RELATIVE_PATH,
        expected_count=EXPECTED_FORMAL_COUNTS["image_member_count"],
        expected_inventory_sha256=_require_sha256(
            artifact_manifest["inventories"]["image_members_sha256"],
            name="artifact image_members_sha256",
        ),
    )
    screening_images = _screening_image_inventory(
        screening_trajectories,
        all_image_payloads=all_image_payloads,
    )
    screening_manifest_json = canonical_json_bytes(
        {"trajectories": list(screening_trajectories)}
    )
    return ValidatedScreeningArtifact(
        artifact_root=root.resolve(),
        artifact_tree_sha256=expected_tree,
        artifact_manifest_sha256=sha256_file(root / MANIFEST_RELATIVE_PATH),
        screening_manifest_sha256=sha256_bytes(screening_manifest_json),
        states=states,
        validation=MappingProxyType(dict(validation)),
        _screening_manifest_json=screening_manifest_json,
        _screening_image_payloads=screening_images,
    )
