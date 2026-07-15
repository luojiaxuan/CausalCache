"""Package and independently validate restoration-v2 executor-dispatch evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.validate_restoration_v2_executor_dispatch import (
    ATTEMPT_ID_PATTERN,
    FROZEN_ACTION_FIXTURE_SHA256,
    FROZEN_ANDROIDWORLD_REVISION,
    FROZEN_EXECUTION_HOST,
    FROZEN_INSPECTOR_EVIDENCE_TYPE,
    FROZEN_INTERFACE_MANIFEST_SHA256,
    FROZEN_JSON_ACTION_SCHEMA,
    FROZEN_LIVE_SOURCE_SHA256,
    FROZEN_SCIENTIFIC_CONTRACT_SHA256,
    FROZEN_SERVER_IMAGE,
    FROZEN_SERVER_IMAGE_ID,
    FROZEN_STACK_CONFIG_SHA256,
    REPOSITORY_ROOT,
    _reject_constant,
    _require_transport_metadata,
    _unique_object,
    load_dispatch_cases,
    read_json,
    validate_dispatch_evidence,
)


FROZEN_RUNTIME_IMAGE_ID = (
    "sha256:81b5df11b32ad8460be270a67066196cb7c6d4fb92cb5d05a44fb06d1ec88d21"
)
FROZEN_RUNTIME_REPO_DIGEST = (
    "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
)
MAX_INSPECTION_GAP_SECONDS = 600.0


def _source_sha256() -> dict[str, str]:
    paths = {
        "runner": REPOSITORY_ROOT
        / "code/scripts/validate_restoration_v2_executor_dispatch.py",
        "host_inspector": REPOSITORY_ROOT
        / "code/scripts/inspect_restoration_v2_executor_container.py",
        "offline_validator": REPOSITORY_ROOT
        / "code/scripts/validate_restoration_v2_executor_evidence.py",
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _validate_run_commit_sources(
    run_commit: str,
    recorded_sources: Mapping[str, str],
) -> None:
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "cat-file",
                "-e",
                f"{run_commit}^{{commit}}",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("executor run Git commit does not exist in this repository") from error
    relative_paths = {
        "runner": "code/scripts/validate_restoration_v2_executor_dispatch.py",
        "host_inspector": "code/scripts/inspect_restoration_v2_executor_container.py",
        "offline_validator": "code/scripts/validate_restoration_v2_executor_evidence.py",
    }
    if set(recorded_sources) != set(relative_paths):
        raise ValueError("executor evidence source hash inventory drifted")
    for name, relative_path in relative_paths.items():
        try:
            raw = subprocess.run(
                [
                    "git",
                    "-C",
                    str(REPOSITORY_ROOT),
                    "show",
                    f"{run_commit}:{relative_path}",
                ],
                check=True,
                capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"executor run Git commit lacks frozen source: {relative_path}"
            ) from error
        if hashlib.sha256(raw).hexdigest() != recorded_sources[name]:
            raise ValueError(
                f"executor source bytes do not belong to run Git commit: {relative_path}"
            )


def _parse_time(value: Any, label: str) -> datetime:
    if type(value) is not str:
        raise ValueError(f"{label} timestamp must be a string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{label} timestamp must include a timezone")
    return parsed


def _parse_embedded_json(raw: Any, expected_sha256: Any, label: str) -> dict[str, Any]:
    if type(raw) is not str:
        raise ValueError(f"{label} raw JSON must be embedded as UTF-8 text")
    encoded = raw.encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ValueError(f"{label} embedded bytes do not match the recorded SHA256")
    value = json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} embedded JSON must be an object")
    return value


def _validate_host_inspection(
    record: Mapping[str, Any],
    *,
    attempt_id: str,
    phase: str,
    expected_sources: Mapping[str, str],
) -> None:
    if record.get("evidence_type") != FROZEN_INSPECTOR_EVIDENCE_TYPE:
        raise ValueError(f"{phase} inspection has the wrong evidence type")
    if record.get("status") != "passed" or record.get("host") != FROZEN_EXECUTION_HOST:
        raise ValueError(f"{phase} inspection is not a passing Aries record")
    if record.get("attempt_id") != attempt_id or record.get("phase") != phase:
        raise ValueError(f"{phase} inspection belongs to a different attempt or phase")
    if record.get("inspector_source_sha256") != expected_sources["host_inspector"]:
        raise ValueError(f"{phase} inspection source hash differs from the frozen inspector")
    if record.get("live_source_sha256") != FROZEN_LIVE_SOURCE_SHA256:
        raise ValueError(f"{phase} inspection live executor source hashes drifted")
    runtime = record.get("runtime_container")
    server = record.get("server_container")
    if not isinstance(runtime, Mapping) or not isinstance(server, Mapping):
        raise ValueError(f"{phase} inspection lacks runtime/server container records")
    if (
        runtime.get("running") is not True
        or runtime.get("image_id") != FROZEN_RUNTIME_IMAGE_ID
        or runtime.get("image_repo_digest")
        != f"hongccc/sglang-omni@{FROZEN_RUNTIME_REPO_DIGEST}"
    ):
        raise ValueError(f"{phase} inspection runtime container identity drifted")
    if (
        server.get("running") is not True
        or server.get("config_image") != FROZEN_SERVER_IMAGE
        or server.get("image_id") != FROZEN_SERVER_IMAGE_ID
        or server.get("container_port") != 5000
    ):
        raise ValueError(f"{phase} inspection executor container identity drifted")
    health = record.get("health")
    if not isinstance(health, Mapping):
        raise ValueError(f"{phase} inspection lacks raw health evidence")
    _require_transport_metadata(health, f"offline-{phase}-inspection-health")
    if (
        health.get("http_status") != 200
        or health.get("content_type") != "application/json"
        or health.get("json_body") != {"status": "success"}
    ):
        raise ValueError(f"{phase} inspection health response is not exact")
    started = _parse_time(record.get("started_at"), f"{phase}-inspection-started")
    finished = _parse_time(record.get("finished_at"), f"{phase}-inspection-finished")
    if finished < started:
        raise ValueError(f"{phase} inspection has reversed timestamps")
    health_started = _parse_time(
        health.get("started_at"), f"{phase}-inspection-health-started"
    )
    health_finished = _parse_time(
        health.get("finished_at"), f"{phase}-inspection-health-finished"
    )
    if not started <= health_started <= health_finished <= finished:
        raise ValueError(f"{phase} inspection health request lies outside its interval")


def _validate_provenance(summary: Mapping[str, Any]) -> None:
    expected_sources = _source_sha256()
    if summary.get("source_sha256") != expected_sources:
        raise ValueError("executor evidence source hashes differ from the frozen validators")
    attempt_id = summary.get("attempt_id")
    if type(attempt_id) is not str or ATTEMPT_ID_PATTERN.fullmatch(attempt_id) is None:
        raise ValueError("executor evidence has an invalid attempt ID")
    run_commit = summary.get("run_git_commit")
    if type(run_commit) is not str or re.fullmatch(r"[0-9a-f]{40}", run_commit) is None:
        raise ValueError("executor evidence lacks a full run Git commit")
    _validate_run_commit_sources(run_commit, expected_sources)

    interface = summary.get("interface_validation")
    if not isinstance(interface, Mapping):
        raise ValueError("executor evidence lacks interface provenance")
    if interface.get("scientific_contract_sha256") != FROZEN_SCIENTIFIC_CONTRACT_SHA256:
        raise ValueError("executor evidence scientific contract SHA256 drifted")
    action = interface.get("action_fixture")
    manifest = interface.get("interface_manifest")
    constructor = interface.get("androidworld_json_action_constructor_validation")
    if not isinstance(action, Mapping) or not isinstance(manifest, Mapping) or not isinstance(
        constructor, Mapping
    ):
        raise ValueError("executor evidence lacks action/manifest/constructor provenance")
    if (
        action.get("sha256") != FROZEN_ACTION_FIXTURE_SHA256
        or action.get("valid_case_count") != 14
        or action.get("invalid_case_count") != 23
        or action.get("coordinate_scalar_checks") != 6000
    ):
        raise ValueError("executor evidence action fixture provenance drifted")
    if manifest.get("sha256") != FROZEN_INTERFACE_MANIFEST_SHA256:
        raise ValueError("executor evidence interface manifest SHA256 drifted")
    if (
        constructor.get("status") != "passed"
        or constructor.get("validated_case_count") != 14
        or constructor.get("source_revision") != FROZEN_ANDROIDWORLD_REVISION
        or constructor.get("run_git_commit") != run_commit
        or constructor.get("container_image_digest") != FROZEN_RUNTIME_REPO_DIGEST
    ):
        raise ValueError("executor evidence constructor provenance or denominator drifted")

    provenance = summary.get("server_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("executor evidence lacks server provenance")
    if (
        provenance.get("host") != FROZEN_EXECUTION_HOST
        or provenance.get("androidworld_source_revision") != FROZEN_ANDROIDWORLD_REVISION
        or provenance.get("stack_config_sha256") != FROZEN_STACK_CONFIG_SHA256
        or provenance.get("live_source_sha256") != FROZEN_LIVE_SOURCE_SHA256
    ):
        raise ValueError("executor evidence server provenance drifted")
    runtime = provenance.get("runtime_container")
    server = provenance.get("server_container")
    if not isinstance(runtime, Mapping) or not isinstance(server, Mapping):
        raise ValueError("executor evidence lacks bound runtime/server identity")
    if (
        runtime.get("image_id") != FROZEN_RUNTIME_IMAGE_ID
        or runtime.get("image_repo_digest") != FROZEN_RUNTIME_REPO_DIGEST
        or server.get("image") != FROZEN_SERVER_IMAGE
        or server.get("image_id") != FROZEN_SERVER_IMAGE_ID
        or server.get("container_port") != 5000
    ):
        raise ValueError("executor evidence runtime/server image identity drifted")
    expected_base_url = f"http://172.17.0.1:{server.get('host_port')}"
    if provenance.get("base_url") != expected_base_url:
        raise ValueError("executor evidence base URL is not derived from the inspected host port")

    before = _parse_embedded_json(
        provenance.get("host_inspection_before_raw_utf8"),
        provenance.get("host_inspection_sha256"),
        "before inspection",
    )
    if provenance.get("host_inspection_before") != before:
        raise ValueError("parsed before inspection differs from embedded raw bytes")
    after = _parse_embedded_json(
        provenance.get("host_inspection_after_raw_utf8"),
        provenance.get("host_inspection_after_sha256"),
        "after inspection",
    )
    if provenance.get("host_inspection_after") != after:
        raise ValueError("parsed after inspection differs from embedded raw bytes")
    _validate_host_inspection(
        before,
        attempt_id=attempt_id,
        phase="before",
        expected_sources=expected_sources,
    )
    _validate_host_inspection(
        after,
        attempt_id=attempt_id,
        phase="after",
        expected_sources=expected_sources,
    )
    for key in ("runtime_container", "server_container", "live_source_sha256"):
        if before.get(key) != after.get(key):
            raise ValueError(f"live executor {key} changed across the dispatch attempt")
    inspected_runtime = before["runtime_container"]
    inspected_server = before["server_container"]
    expected_runtime = {
        "name": inspected_runtime.get("name"),
        "id": inspected_runtime.get("id"),
        "image_id": inspected_runtime.get("image_id"),
        "image_repo_digest": FROZEN_RUNTIME_REPO_DIGEST,
        "mounts": inspected_runtime.get("mounts"),
    }
    expected_server = {
        "name": inspected_server.get("name"),
        "id": inspected_server.get("id"),
        "image": inspected_server.get("config_image"),
        "image_id": inspected_server.get("image_id"),
        "host_port": inspected_server.get("host_port"),
        "container_port": inspected_server.get("container_port"),
        "json_action_schema": FROZEN_JSON_ACTION_SCHEMA,
        "host_health_url": f"http://127.0.0.1:{inspected_server.get('host_port')}/health",
    }
    if runtime != expected_runtime or server != expected_server:
        raise ValueError(
            "top-level runtime/server provenance differs from embedded live inspection"
        )
    for inspection in (before, after):
        if inspection["health"].get("url") != expected_server["host_health_url"]:
            raise ValueError("embedded health URL differs from inspected server port")

    pre_finished = _parse_time(before.get("finished_at"), "before-inspection-finished")
    run_started = _parse_time(summary.get("started_at"), "dispatch-started")
    run_finished = _parse_time(summary.get("finished_at"), "dispatch-finished")
    post_started = _parse_time(after.get("started_at"), "after-inspection-started")
    if not pre_finished <= run_started <= run_finished <= post_started:
        raise ValueError("inspection and dispatch timestamps are not properly ordered")
    if (run_started - pre_finished).total_seconds() > MAX_INSPECTION_GAP_SECONDS:
        raise ValueError("before inspection is stale relative to dispatch")
    if (post_started - run_finished).total_seconds() > MAX_INSPECTION_GAP_SECONDS:
        raise ValueError("after inspection is stale relative to dispatch")

    packaging = summary.get("canonical_packaging")
    if not isinstance(packaging, Mapping):
        raise ValueError("executor evidence was not packaged with post-run inspection")
    if (
        packaging.get("attempt_id") != attempt_id
        or packaging.get("offline_validator_sha256") != expected_sources["offline_validator"]
    ):
        raise ValueError("executor evidence packaging provenance drifted")
    if packaging.get("after_inspection_sha256") != provenance.get(
        "host_inspection_after_sha256"
    ):
        raise ValueError("packaging after-inspection SHA256 differs from embedded evidence")
    raw_summary = _parse_embedded_json(
        packaging.get("raw_summary_raw_utf8"),
        packaging.get("raw_summary_sha256"),
        "raw dispatch summary",
    )
    reconstructed_raw = copy.deepcopy(summary)
    reconstructed_raw.pop("canonical_packaging", None)
    reconstructed_provenance = reconstructed_raw.get("server_provenance")
    if not isinstance(reconstructed_provenance, dict):
        raise ValueError("canonical evidence lacks reconstructable raw server provenance")
    for key in (
        "host_inspection_after",
        "host_inspection_after_raw_utf8",
        "host_inspection_after_sha256",
    ):
        reconstructed_provenance.pop(key, None)
    if raw_summary != reconstructed_raw:
        raise ValueError("embedded raw dispatch summary differs from canonical reconstruction")
    packaged_at = _parse_time(packaging.get("packaged_at"), "packaged-at")
    post_finished = _parse_time(after.get("finished_at"), "after-inspection-finished")
    if packaged_at < post_finished:
        raise ValueError("canonical evidence was packaged before post-inspection finished")


def validate_summary(summary: Mapping[str, Any]) -> dict[str, object]:
    _validate_provenance(summary)
    action_fixture_path = REPOSITORY_ROOT / "data/fixtures/gui_owl_v2_action_roundtrip.json"
    cases = load_dispatch_cases(action_fixture_path)
    reduced = validate_dispatch_evidence(summary, cases=cases)
    if summary.get("offline_reduction") != reduced:
        raise ValueError("stored executor verdict differs from independently recomputed raw records")
    return {
        "status": reduced["status"],
        "validated_case_count": reduced["validated_case_count"],
        "negative_actuation_control_status": reduced[
            "negative_actuation_control_status"
        ],
        "attempt_id": summary["attempt_id"],
        "run_git_commit": summary["run_git_commit"],
    }


def package(raw_summary_path: Path, after_inspection_path: Path) -> dict[str, Any]:
    raw_summary_raw = raw_summary_path.read_text(encoding="utf-8")
    raw_summary = read_json(raw_summary_path)
    after_raw = after_inspection_path.read_text(encoding="utf-8")
    after = read_json(after_inspection_path)
    attempt_id = raw_summary.get("attempt_id")
    if type(attempt_id) is not str or ATTEMPT_ID_PATTERN.fullmatch(attempt_id) is None:
        raise ValueError("raw executor summary has an invalid attempt ID")
    if after.get("attempt_id") != attempt_id or after.get("phase") != "after":
        raise ValueError("after inspection belongs to a different attempt or phase")
    summary = copy.deepcopy(raw_summary)
    provenance = summary.get("server_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("raw executor summary lacks mutable server provenance")
    provenance["host_inspection_after"] = after
    provenance["host_inspection_after_raw_utf8"] = after_raw
    provenance["host_inspection_after_sha256"] = hashlib.sha256(
        after_raw.encode("utf-8")
    ).hexdigest()
    validator_sha256 = _source_sha256()["offline_validator"]
    summary["canonical_packaging"] = {
        "attempt_id": attempt_id,
        "packaged_at": datetime.now(timezone.utc).isoformat(),
        "raw_summary_sha256": hashlib.sha256(raw_summary_raw.encode("utf-8")).hexdigest(),
        "raw_summary_raw_utf8": raw_summary_raw,
        "after_inspection_sha256": hashlib.sha256(
            after_inspection_path.read_bytes()
        ).hexdigest(),
        "offline_validator_sha256": validator_sha256,
    }
    summary["offline_reduction"] = validate_dispatch_evidence(
        summary,
        cases=load_dispatch_cases(
            REPOSITORY_ROOT / "data/fixtures/gui_owl_v2_action_roundtrip.json"
        ),
    )
    validate_summary(summary)
    return summary


def validate_file(summary_path: Path) -> dict[str, object]:
    summary = read_json(summary_path)
    attempt_id = summary.get("attempt_id")
    if type(attempt_id) is not str or attempt_id not in summary_path.name:
        raise ValueError("canonical summary filename must retain its attempt ID")
    result = validate_summary(summary)
    return {
        **result,
        "summary_path": str(summary_path),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        **{f"{key}_sha256": value for key, value in _source_sha256().items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--raw-summary", required=True, type=Path)
    package_parser.add_argument("--host-inspection-after", required=True, type=Path)
    package_parser.add_argument("--output-summary", required=True, type=Path)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "package":
        if args.output_summary.exists():
            raise FileExistsError(
                f"refusing to overwrite canonical executor evidence: {args.output_summary}"
            )
        summary = package(args.raw_summary, args.host_inspection_after)
        if summary["attempt_id"] not in args.output_summary.name:
            raise ValueError("canonical output filename must contain the attempt ID")
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        with args.output_summary.open("x", encoding="utf-8") as output_file:
            output_file.write(rendered)
        print(rendered, end="")
        return
    result = validate_file(args.summary)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
