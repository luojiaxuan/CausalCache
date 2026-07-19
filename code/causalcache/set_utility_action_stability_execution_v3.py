"""Execution envelope and one-state runner for strict-determinism D2."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.set_utility_action_stability_contract_v1 import (
    canonical_json_bytes,
    canonical_pretty_json_bytes,
    sha256_bytes,
)
from causalcache.set_utility_action_stability_contract_v3 import (
    CANONICAL_CONFIG_PATH,
    CANONICAL_EXECUTION_CONFIG_PATH,
    D1B_AGGREGATE_PATH,
    D1B_AGGREGATE_SHA256,
    load_action_stability_source_v3_contract,
)
from causalcache.set_utility_action_stability_diagnostic_v1 import (
    WORKER_INDEX_BY_STATE,
    _validate_metric_safe_tree,
)
from causalcache.set_utility_action_stability_diagnostic_v2 import (
    STATE_IDS as D1B_STATE_IDS,
)
from causalcache.set_utility_action_stability_diagnostic_v3 import (
    STATE_IDS,
    STRICT_DETERMINISM_FROZEN_ENCODED_CONDITION,
    STRICT_DETERMINISM_PROFILE,
    run_strict_determinism_condition_v3,
    validate_strict_partial_v3,
)
from causalcache.set_utility_action_stability_execution_v2 import (
    ActionStabilityStateLaunchV2,
    PARENT_CANDIDATE_SCHEDULE_SHA256,
    PARENT_EXECUTION_ENVELOPE_SHA256,
    PARENT_MODEL_INVENTORY_SHA256,
    PARENT_PROCESSOR_INVENTORY_SHA256,
    load_parent_artifact_binding_v2,
)
from causalcache.set_utility_throughput_pilot_contract_v1 import (
    CANONICAL_CONFIG_PATH as PARENT_SOURCE_CONFIG_PATH,
)
from causalcache.set_utility_throughput_pilot_execution_v1 import (
    load_worker_semantic_inputs_v1,
)


SCHEMA_VERSION = "1.0.0"
ENVELOPE_PROTOCOL_ID = "causalcache_set_utility_action_stability_execution_envelope_v3"
ENVELOPE_STATUS = "AUTHORIZED_THREE_PROCESS_STRICT_DETERMINISM_D2_EXECUTION"
ENVELOPE_VALIDATION_STATUS = "VALID_STRICT_DETERMINISM_D2_EXECUTION_ENVELOPE"
EXECUTION_PROTOCOL_ID = "causalcache_set_utility_action_stability_execution_v3"
TERMINAL_PROTOCOL_ID = "causalcache_set_utility_action_stability_state_terminal_v3"
CLAIMED_STATUS = "CLAIMED_NO_RETRY_STRICT_DETERMINISM_D2_STATE_PROCESS"
COMPLETED_STATUS = "COMPLETED_STRICT_DETERMINISM_D2_STATE_PROCESS"
FAILED_STATUS = "FAILED_STRICT_DETERMINISM_D2_STATE_PROCESS"

SOURCE_BRANCH = "main"
SOURCE_REMOTE = "origin"
PREFLIGHT_MAX_AGE_SECONDS = 15 * 60
HISTORICAL_D1_AGGREGATE_PATH = (
    "data/results/set_utility_action_stability_diagnostic_v1/aggregate.json"
)
HISTORICAL_D1_AGGREGATE_SHA256 = (
    "2debde6de3f552e9551d0ee37d25b82fa2ca85dfb42746d389dde39eff577ba2"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{16,}")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
_SAFE_FAILURE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")


@dataclass(frozen=True, slots=True)
class ActionStabilityStateLaunchV3:
    repository_root: Path
    execution_envelope_path: Path
    parent_envelope_path: Path
    processor_root: Path
    model_dir: Path
    run_root: Path
    state_id: str
    state_index: int
    gpu_uuid: str
    execution_envelope_sha256: str
    source_config_sha256: str
    source_inventory_sha256: str

    def __post_init__(self) -> None:
        for value in (
            self.repository_root,
            self.execution_envelope_path,
            self.parent_envelope_path,
            self.processor_root,
            self.model_dir,
            self.run_root,
        ):
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError("D2 launch paths must be absolute")
        if self.state_id not in STATE_IDS or self.state_index != STATE_IDS.index(
            self.state_id
        ):
            raise ValueError("D2 launch state identity drifted")
        if _GPU_UUID.fullmatch(self.gpu_uuid) is None:
            raise ValueError("D2 launch GPU UUID is invalid")
        for value in (
            self.execution_envelope_sha256,
            self.source_config_sha256,
            self.source_inventory_sha256,
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError("D2 launch SHA256 identity is invalid")


def _timestamp(value: str, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"D2 {label} timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"D2 {label} timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise ValueError(f"D2 {label} timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _absolute(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"D2 {label} must be absolute")
    return path


def _strict_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"D2 {label} must be one regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"D2 {label} must be strict JSON") from error
    if not isinstance(value, dict) or canonical_pretty_json_bytes(value) != raw:
        raise ValueError(f"D2 {label} must be canonical pretty JSON")
    return value, sha256_bytes(raw)


def canonical_run_layout(run_root: str | Path) -> dict[str, Any]:
    root = _absolute(run_root, label="run root")
    states = []
    for index, state_id in enumerate(STATE_IDS):
        state_root = root / "states" / f"state-{index:02d}"
        states.append(
            {
                "attempt_path": str(state_root / "attempt.json"),
                "stderr_log_path": str(state_root / "stderr.log"),
                "stdout_log_path": str(state_root / "stdout.log"),
                "terminal_path": str(state_root / "terminal.json"),
            }
        )
    return {
        "aggregate_path": str(root / "aggregate.json"),
        "envelope_path": str(root / "execution-envelope.json"),
        "run_root": str(root),
        "states": states,
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_clean_pushed_source_v3(
    *, repository_root: str | Path, expected_git_revision: str
) -> dict[str, str]:
    root = Path(repository_root).resolve()
    head = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    remote = _git(root, "rev-parse", f"refs/remotes/{SOURCE_REMOTE}/{SOURCE_BRANCH}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if (
        head != expected_git_revision
        or _COMMIT.fullmatch(head) is None
        or branch != SOURCE_BRANCH
        or remote != head
        or dirty
    ):
        raise ValueError("D2 envelope requires clean pushed Source-A")
    return {"branch": branch, "git_revision": head, "remote_revision": remote}


def build_action_stability_envelope_v3(
    *,
    repository_root: str | Path,
    source_git_revision: str,
    run_root: str | Path,
    python_executable: str,
    parent_envelope_path: str | Path,
    processor_root: str | Path,
    model_dir: str | Path,
    preflight_path: str | Path,
    preflight_completed_at_utc: str,
    host_alias: str,
    hostname: str,
    container_id: str,
    container_image_reference: str,
    container_image_digest: str,
    driver_version: str,
    selected_gpus: Sequence[Mapping[str, Any]],
    software_versions: Mapping[str, str],
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    contract = load_action_stability_source_v3_contract(repository_root=root)
    if _COMMIT.fullmatch(source_git_revision) is None:
        raise ValueError("D2 source revision is invalid")
    if host_alias not in {"hyper00", "hyper01"} or not hostname:
        raise ValueError("D2 requires one Hyper H200 host")
    if _CONTAINER_ID.fullmatch(container_id) is None:
        raise ValueError("D2 container id is invalid")
    if not container_image_reference or not container_image_digest.startswith("sha256:"):
        raise ValueError("D2 image identity is invalid")
    python = _absolute(python_executable, label="Python executable")
    parent = _absolute(parent_envelope_path, label="parent envelope")
    processor = _absolute(processor_root, label="processor root")
    model = _absolute(model_dir, label="model directory")
    preflight = _absolute(preflight_path, label="fleet preflight")
    if not python.is_file() or not parent.is_file() or not preflight.is_file():
        raise ValueError("D2 runtime or preflight artifact is missing")
    if sha256_bytes(parent.read_bytes()) != PARENT_EXECUTION_ENVELOPE_SHA256:
        raise ValueError("D2 parent throughput envelope drifted")
    if not processor.is_dir() or not model.is_dir():
        raise ValueError("D2 processor/model roots are missing")
    if len(selected_gpus) != len(STATE_IDS):
        raise ValueError("D2 requires exactly three H200 GPUs")
    gpus = []
    for visible_index, raw in enumerate(selected_gpus):
        gpu = dict(raw)
        if (
            set(gpu) != {"host_index", "name", "total_memory_bytes", "uuid"}
            or gpu["name"] != "NVIDIA H200"
            or type(gpu["host_index"]) is not int
            or type(gpu["total_memory_bytes"]) is not int
            or _GPU_UUID.fullmatch(str(gpu["uuid"])) is None
        ):
            raise ValueError("D2 selected GPU binding drifted")
        gpus.append({**gpu, "visible_index": visible_index})
    if len({gpu["uuid"] for gpu in gpus}) != len(gpus):
        raise ValueError("D2 selected GPU UUIDs must be unique")
    completed = _timestamp(preflight_completed_at_utc, label="preflight completion")
    now = _timestamp(utc_now(), label="materialization")
    if now < completed or (now - completed).total_seconds() > PREFLIGHT_MAX_AGE_SECONDS:
        raise ValueError("D2 fleet preflight is stale")
    source = contract.data["source"]
    config_path = root.joinpath(*PurePosixPath(CANONICAL_CONFIG_PATH).parts)
    config_raw = config_path.read_bytes()
    if sha256_bytes(config_raw) != contract.config_sha256:
        raise ValueError("D2 source config binding drifted")
    layout = canonical_run_layout(run_root)
    state_processes = []
    for index, state_id in enumerate(STATE_IDS):
        state_layout = layout["states"][index]
        state_processes.append(
            {
                "argv": [
                    python_executable,
                    str(root / "code/scripts/run_set_utility_action_stability_state_v3.py"),
                    "--execution-envelope",
                    layout["envelope_path"],
                    "--state-id",
                    state_id,
                ],
                "cuda_visible_devices": gpus[index]["uuid"],
                "device": "cuda:0",
                "gpu_uuid": gpus[index]["uuid"],
                "output": state_layout,
                "process_scope": "exactly_one_state_fresh_os_process",
                "state_id": state_id,
                "state_index": index,
            }
        )
    return {
        "artifacts": {
            "d1b_aggregate": {
                "path": str(root / D1B_AGGREGATE_PATH),
                "sha256": D1B_AGGREGATE_SHA256,
            },
            "model_dir": str(model),
            "model_inventory_sha256": PARENT_MODEL_INVENTORY_SHA256,
            "parent_envelope": {
                "path": str(parent),
                "sha256": PARENT_EXECUTION_ENVELOPE_SHA256,
            },
            "processor_inventory_sha256": PARENT_PROCESSOR_INVENTORY_SHA256,
            "processor_root": str(processor),
        },
        "authorization": {
            "generate_restoration_labels": False,
            "run_closed_loop": False,
            "run_three_process_strict_determinism_d2": True,
            "source_a_alone_authorizes_execution": False,
            "train_predictor": False,
        },
        "container": {
            "id": container_id,
            "image_digest": container_image_digest,
            "image_reference": container_image_reference,
        },
        "execution": {
            "aggregate_argv": [
                python_executable,
                str(root / "code/scripts/aggregate_set_utility_action_stability_v3.py"),
                "--execution-envelope",
                layout["envelope_path"],
            ],
            "condition_id": STRICT_DETERMINISM_FROZEN_ENCODED_CONDITION,
            "encode_call_ceiling": len(STATE_IDS),
            "generation_call_ceiling": len(STATE_IDS) * 2,
            "layout": layout,
            "no_retry": True,
            "no_top_up": True,
            "state_processes": state_processes,
        },
        "gpus": gpus,
        "host": {
            "alias": host_alias,
            "driver_version": driver_version,
            "hostname": hostname,
        },
        "materialized_at_utc": utc_now(),
        "preflight": {
            "completed_at_utc": preflight_completed_at_utc,
            "max_age_seconds": PREFLIGHT_MAX_AGE_SECONDS,
            "path": str(preflight),
            "sha256": sha256_bytes(preflight.read_bytes()),
        },
        "protocol_id": ENVELOPE_PROTOCOL_ID,
        "run_root": str(_absolute(run_root, label="run root")),
        "runtime": {
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "profile": STRICT_DETERMINISM_PROFILE,
            "python_executable": python_executable,
            "software_versions": dict(software_versions),
        },
        "schema_version": SCHEMA_VERSION,
        "source": {
            "branch": SOURCE_BRANCH,
            "config_path": CANONICAL_CONFIG_PATH,
            "config_sha256": contract.config_sha256,
            "execution_config_path": CANONICAL_EXECUTION_CONFIG_PATH,
            "git_revision": source_git_revision,
            "inventory_sha256": source["inventory_sha256"],
            "remote": SOURCE_REMOTE,
        },
        "status": ENVELOPE_STATUS,
    }


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise OSError("D2 exclusive write made no progress")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_canonical_envelope_pair_v3(
    envelope: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    payload = canonical_pretty_json_bytes(envelope)
    git_path = root.joinpath(*PurePosixPath(CANONICAL_EXECUTION_CONFIG_PATH).parts)
    data_path = Path(str(envelope["execution"]["layout"]["envelope_path"]))
    _write_exclusive(git_path, payload, mode=0o644)
    try:
        _write_exclusive(data_path, payload)
    except Exception:
        git_path.unlink(missing_ok=True)
        raise
    return {"bytes": len(payload), "path": str(data_path), "sha256": sha256_bytes(payload)}


def _validate_committed_b(
    root: Path, *, source_revision: str, expected_bytes: bytes
) -> str:
    head = _git(root, "rev-parse", "HEAD")
    parents = _git(root, "rev-list", "--parents", "-n", "1", head).split()
    changed = _git(root, "diff", "--name-only", "--no-renames", source_revision, head).splitlines()
    remote = _git(root, "rev-parse", f"refs/remotes/{SOURCE_REMOTE}/{SOURCE_BRANCH}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    committed = subprocess.run(
        ["git", "show", f"{head}:{CANONICAL_EXECUTION_CONFIG_PATH}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if (
        parents != [head, source_revision]
        or changed != [CANONICAL_EXECUTION_CONFIG_PATH]
        or remote != head
        or dirty
        or committed != expected_bytes
    ):
        raise ValueError("D2 execution requires pushed direct-child Execution-B")
    return head


def load_action_stability_envelope_v3(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
    verify_repository: bool = True,
    require_fresh_preflight: bool = True,
    verify_current_container: bool = True,
) -> dict[str, Any]:
    envelope_path = _absolute(path, label="execution envelope")
    envelope, envelope_sha = _strict_canonical(envelope_path, label="execution envelope")
    root = Path(repository_root).resolve() if repository_root else Path(__file__).resolve().parents[2]
    contract = load_action_stability_source_v3_contract(repository_root=root)
    source = envelope.get("source")
    execution = envelope.get("execution")
    artifacts = envelope.get("artifacts")
    if (
        envelope.get("protocol_id") != ENVELOPE_PROTOCOL_ID
        or envelope.get("status") != ENVELOPE_STATUS
        or not isinstance(source, Mapping)
        or not isinstance(execution, Mapping)
        or not isinstance(artifacts, Mapping)
        or source.get("config_sha256") != contract.config_sha256
        or source.get("inventory_sha256") != contract.data["source"]["inventory_sha256"]
        or source.get("config_path") != CANONICAL_CONFIG_PATH
        or source.get("execution_config_path") != CANONICAL_EXECUTION_CONFIG_PATH
        or execution.get("condition_id") != STRICT_DETERMINISM_FROZEN_ENCODED_CONDITION
        or execution.get("no_retry") is not True
        or execution.get("no_top_up") is not True
        or [item.get("state_id") for item in execution.get("state_processes", [])]
        != list(STATE_IDS)
        or artifacts.get("d1b_aggregate", {}).get("sha256") != D1B_AGGREGATE_SHA256
    ):
        raise ValueError("D2 execution envelope identity drifted")
    if envelope_path != Path(execution["layout"]["envelope_path"]):
        raise ValueError("D2 envelope path is not canonical")
    if require_fresh_preflight:
        completed = _timestamp(envelope["preflight"]["completed_at_utc"], label="preflight completion")
        age = (datetime.now(timezone.utc) - completed).total_seconds()
        if age < 0 or age > PREFLIGHT_MAX_AGE_SECONDS:
            raise ValueError("D2 fleet preflight is stale at execution")
    if verify_current_container:
        observed = os.environ.get("HOSTNAME", "")
        if not envelope["container"]["id"].startswith(observed):
            raise ValueError("D2 current container identity drifted")
    b_revision = None
    if verify_repository:
        b_revision = _validate_committed_b(
            root,
            source_revision=str(source["git_revision"]),
            expected_bytes=envelope_path.read_bytes(),
        )
    return {
        "artifacts": dict(artifacts),
        "envelope_path": str(envelope_path),
        "execution": dict(execution),
        "execution_b_revision": b_revision,
        "gpus": list(envelope["gpus"]),
        "repository_root": str(root),
        "run_root": envelope["run_root"],
        "source": dict(source),
        "validation": {
            "envelope_sha256": envelope_sha,
            "source_config_sha256": contract.config_sha256,
            "source_inventory_sha256": contract.data["source"]["inventory_sha256"],
            "status": ENVELOPE_VALIDATION_STATUS,
            "this_envelope_authorizes_gpu_execution": True,
        },
    }


def launch_from_projection_v3(
    projection: Mapping[str, Any], *, state_id: str, visible_gpu_uuid: str | None
) -> ActionStabilityStateLaunchV3:
    processes = projection["execution"]["state_processes"]
    matches = [item for item in processes if item.get("state_id") == state_id]
    if len(matches) != 1 or visible_gpu_uuid != matches[0].get("gpu_uuid"):
        raise PermissionError("D2 state/GPU process binding drifted")
    if projection["validation"].get("this_envelope_authorizes_gpu_execution") is not True:
        raise PermissionError("D2 envelope is not authorizing")
    artifacts = projection["artifacts"]
    source = projection["source"]
    validation = projection["validation"]
    return ActionStabilityStateLaunchV3(
        repository_root=Path(projection["repository_root"]),
        execution_envelope_path=Path(projection["envelope_path"]),
        parent_envelope_path=Path(artifacts["parent_envelope"]["path"]),
        processor_root=Path(artifacts["processor_root"]),
        model_dir=Path(artifacts["model_dir"]),
        run_root=Path(projection["run_root"]),
        state_id=state_id,
        state_index=STATE_IDS.index(state_id),
        gpu_uuid=str(visible_gpu_uuid),
        execution_envelope_sha256=validation["envelope_sha256"],
        source_config_sha256=validation["source_config_sha256"],
        source_inventory_sha256=validation["source_inventory_sha256"],
    )


def _legacy_artifact_launch(launch: ActionStabilityStateLaunchV3) -> ActionStabilityStateLaunchV2:
    # note (luojiaxuan): This object is used only by the already-audited parent
    # artifact/semantic loader.  D2 authority comes exclusively from the v3
    # envelope above; no D1b attempt, terminal, or execution authorization is
    # reused or mutated.
    d1b_index = D1B_STATE_IDS.index(launch.state_id)
    d1b_wave = 0 if d1b_index in (0, 3, 4, 5) else 1
    d1b_waves = (
        (D1B_STATE_IDS[0], D1B_STATE_IDS[3], D1B_STATE_IDS[4], D1B_STATE_IDS[5]),
        (D1B_STATE_IDS[1], D1B_STATE_IDS[2]),
    )
    return ActionStabilityStateLaunchV2(
        repository_root=launch.repository_root,
        source_config_path=(
            "code/configs/causalcache_set_utility_action_stability_diagnostic_v2.json"
        ),
        execution_envelope_path=launch.execution_envelope_path,
        parent_envelope_path=launch.parent_envelope_path,
        historical_d1_aggregate_path=launch.repository_root / HISTORICAL_D1_AGGREGATE_PATH,
        processor_root=launch.processor_root,
        model_dir=launch.model_dir,
        output_root=launch.run_root,
        state_id=launch.state_id,
        state_index=d1b_index,
        wave_index=d1b_wave,
        wave_slot=d1b_waves[d1b_wave].index(launch.state_id),
        device="cuda:0",
        gpu_uuid=launch.gpu_uuid,
        execution_envelope_sha256=launch.execution_envelope_sha256,
        source_config_sha256="0" * 64,
        source_inventory_sha256="0" * 64,
        historical_d1_aggregate_sha256=HISTORICAL_D1_AGGREGATE_SHA256,
        validation_status="VALID_SET_UTILITY_ACTION_STABILITY_EXECUTION_ENVELOPE_V2",
        gpu_execution_authorized=True,
    )


def _runtime_bundle_v3(launch: ActionStabilityStateLaunchV3, validated: Any, parent_contract: Any) -> tuple[object, Callable[[object], Callable[[object], object]], str]:
    from PIL import Image

    from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v3 import (
        GUIOwlV21StrictDeterminismActionStabilityRuntimeV3,
    )
    from causalcache.policy.gui_owl_v2_runtime import (
        FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    from causalcache.set_utility_gui_owl_v2_1_throughput_adapter import (
        build_gui_owl_v2_1_throughput_reference_input,
    )

    inputs = parent_contract.data["inputs"]
    model_binding = inputs["model_snapshot_manifest"]
    runtime = GUIOwlV21StrictDeterminismActionStabilityRuntimeV3(
        model_dir=launch.model_dir,
        expected_snapshot_manifest=launch.repository_root / model_binding["path"],
        device="cuda:0",
        target_effective_visual_tokens_per_image=(
            FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
        ),
    )

    def decode_rgb(payload: bytes) -> Any:
        source = Image.open(io.BytesIO(payload))
        try:
            return source.convert("RGB")
        finally:
            source.close()

    def builder_for(joined: object) -> Callable[[object], object]:
        return lambda plan: build_gui_owl_v2_1_throughput_reference_input(
            joined, plan, image_decoder=decode_rgb
        )

    identity = {
        "device": "cuda:0",
        "model_inventory_sha256": validated.model_inventory_sha256,
        "profile": STRICT_DETERMINISM_PROFILE,
        "runtime_metadata_sha256": sha256_bytes(canonical_json_bytes(dict(runtime.metadata))),
        "state_id": launch.state_id,
    }
    return runtime, builder_for, sha256_bytes(canonical_json_bytes(identity))


def _process_identity(launch: ActionStabilityStateLaunchV3) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "monotonic_ns": time.monotonic_ns(),
                "pid": os.getpid(),
                "state_id": launch.state_id,
                "wall_time_ns": time.time_ns(),
            }
        )
    )


def _safe_failure(error: BaseException) -> str:
    name = error.__class__.__name__
    return name if _SAFE_FAILURE.fullmatch(name) else "UnexpectedException"


def run_action_stability_state_v3(launch: ActionStabilityStateLaunchV3) -> dict[str, Any]:
    contract = load_action_stability_source_v3_contract(
        repository_root=launch.repository_root
    )
    if (
        contract.config_sha256 != launch.source_config_sha256
        or contract.data["source"]["inventory_sha256"] != launch.source_inventory_sha256
    ):
        raise PermissionError("D2 Source-A/envelope identity drifted")
    layout = canonical_run_layout(launch.run_root)["states"][launch.state_index]
    attempt_path = Path(layout["attempt_path"])
    terminal_path = Path(layout["terminal_path"])
    process_sha = _process_identity(launch)
    attempt = {
        "candidate_schedule_sha256": PARENT_CANDIDATE_SCHEDULE_SHA256,
        "condition_id": STRICT_DETERMINISM_FROZEN_ENCODED_CONDITION,
        "execution_envelope_sha256": launch.execution_envelope_sha256,
        "metric_safe": True,
        "model_inventory_sha256": PARENT_MODEL_INVENTORY_SHA256,
        "process_identity_sha256": process_sha,
        "process_scope": "exactly_one_state_fresh_os_process",
        "processor_inventory_sha256": PARENT_PROCESSOR_INVENTORY_SHA256,
        "profile": STRICT_DETERMINISM_PROFILE,
        "protocol_id": EXECUTION_PROTOCOL_ID,
        "retry_count": 0,
        "schema_version": SCHEMA_VERSION,
        "source_config_sha256": launch.source_config_sha256,
        "source_inventory_sha256": launch.source_inventory_sha256,
        "state_id": launch.state_id,
        "state_index": launch.state_index,
        "status": CLAIMED_STATUS,
    }
    attempt_bytes = canonical_pretty_json_bytes(attempt)
    _write_exclusive(attempt_path, attempt_bytes)
    partial = None
    runtime_sha = None
    failure_class = None
    try:
        legacy = _legacy_artifact_launch(launch)
        parent_contract, validated = load_parent_artifact_binding_v2(legacy)
        states = load_worker_semantic_inputs_v1(validated, parent_contract)
        matches = [item for item in states if item.state_id == launch.state_id]
        if len(matches) != 1:
            raise ValueError("D2 semantic parent state is missing")
        runtime, builder_for, runtime_sha = _runtime_bundle_v3(
            launch, validated, parent_contract
        )
        raw = run_strict_determinism_condition_v3(
            matches[0].query,
            reference_input_builder=builder_for(matches[0].joined_input),
            runtime=runtime,
        )
        partial = validate_strict_partial_v3(raw)
    except Exception as error:
        failure_class = _safe_failure(error)
    conditions = [] if partial is None else partial["conditions"]
    terminal = {
        "attempt_sha256": sha256_bytes(attempt_bytes),
        "counts": {
            "encode_call_count": sum(item["encode_call_count"] for item in conditions),
            "generation_call_count": sum(item["generation_call_count"] for item in conditions),
            "retry_count": 0,
        },
        "execution_envelope_sha256": launch.execution_envelope_sha256,
        "failure_class": failure_class,
        "metric_safe": True,
        "partial_result": partial,
        "process_identity_sha256": process_sha,
        "profile": STRICT_DETERMINISM_PROFILE,
        "protocol_id": TERMINAL_PROTOCOL_ID,
        "runtime_identity_sha256": runtime_sha,
        "schema_version": SCHEMA_VERSION,
        "source_config_sha256": launch.source_config_sha256,
        "source_inventory_sha256": launch.source_inventory_sha256,
        "state_id": launch.state_id,
        "state_index": launch.state_index,
        "status": COMPLETED_STATUS if failure_class is None else FAILED_STATUS,
    }
    _validate_metric_safe_tree(terminal)
    _write_exclusive(terminal_path, canonical_pretty_json_bytes(terminal))
    return terminal


__all__ = [
    "COMPLETED_STATUS",
    "ENVELOPE_VALIDATION_STATUS",
    "FAILED_STATUS",
    "TERMINAL_PROTOCOL_ID",
    "build_action_stability_envelope_v3",
    "canonical_run_layout",
    "launch_from_projection_v3",
    "load_action_stability_envelope_v3",
    "run_action_stability_state_v3",
    "validate_clean_pushed_source_v3",
    "write_canonical_envelope_pair_v3",
]
