"""Validate frozen restoration-v2 action, summary, and prompt interfaces."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from causalcache.low_fidelity_v2 import (
    EXECUTOR_RESULT_VALUES,
    LOW_FIDELITY_V2_ACTIONS,
    LOW_FIDELITY_V2_KEYS,
    MAXIMUM_SCREEN_TEXT_TOKENS,
    SCREEN_CHANGE_THRESHOLDS,
    LowFidelityEventV2,
    screen_change_from_mean_absolute_rgb_difference,
    serialize_low_fidelity_v2,
    sha256_bytes,
)
from causalcache.policy.gui_owl_v2 import (
    CANONICAL_GUI_OWL_V2_ACTIONS,
    GUI_OWL_V2_ACTION_ALIASES,
    GUI_OWL_V2_ARGUMENT_KEY_ORDER,
    GUI_OWL_V2_DECISION_STEPS,
    GUI_OWL_V2_PARAMETERS_BY_ACTION,
    GUI_OWL_V2_SYSTEM_BUTTONS,
    GUI_OWL_V2_TEACHER_CARRIER,
    build_gui_owl_v2_mixed_fidelity_messages,
    gui_owl_v2_action_to_androidworld,
    normalized_coordinate_to_pixel,
    parse_gui_owl_v2_output,
    serialize_gui_owl_v2_teacher_target,
    serialize_gui_owl_v2_tool_call,
)
from causalcache.restoration_v2_contract import RestorationV2Contract


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FROZEN_ANDROIDWORLD_JSON_ACTION_REVISION = "11cea575561fb7800b5fb6b6cafa56f7a91de11f"


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _read_json(path: Path) -> tuple[Mapping[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(
        raw,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"fixture must be a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_action_fixture(
    fixture: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if fixture.get("schema_version") != "1.0.0":
        raise ValueError("unexpected action fixture schema_version")
    if fixture.get("contract_id") != "gui_owl_androidworld_v2":
        raise ValueError("unexpected action fixture contract_id")
    if fixture.get("coverage_scope") != "all_actions_aliases_buttons_and_coordinate_scalar_domain":
        raise ValueError("unexpected action fixture coverage_scope")
    screen = fixture["screen"]
    parsed_cases = []
    canonical_actions = set()
    source_actions = set()
    buttons = set()
    payloads = []
    for case in fixture["valid_cases"]:
        parsed = parse_gui_owl_v2_output(case["native_output"])
        if parsed.canonical_action.arguments() != case["canonical_arguments"]:
            raise ValueError(f"canonical action mismatch: {case['id']}")
        if serialize_gui_owl_v2_tool_call(parsed.canonical_action) != case["canonical_tool_call"]:
            raise ValueError(f"canonical tool-call mismatch: {case['id']}")
        reparsed = parse_gui_owl_v2_output(
            serialize_gui_owl_v2_teacher_target(parsed.canonical_action)
        )
        if reparsed.canonical_action != parsed.canonical_action:
            raise ValueError(f"canonical teacher target is not round-trippable: {case['id']}")
        payload = gui_owl_v2_action_to_androidworld(
            parsed.canonical_action,
            screen_width=int(screen["width"]),
            screen_height=int(screen["height"]),
        )
        if payload != case["androidworld_payload"]:
            raise ValueError(f"AndroidWorld bridge mismatch: {case['id']}")
        canonical_actions.add(parsed.canonical_action.action)
        source_actions.add(parsed.source_action_name)
        if parsed.canonical_action.button is not None:
            buttons.add(parsed.canonical_action.button)
        payloads.append({"case_id": case["id"], "payload": payload})
        parsed_cases.append({"case_id": case["id"], "canonical": parsed.canonical_action})

    if canonical_actions != set(CANONICAL_GUI_OWL_V2_ACTIONS):
        raise ValueError("valid action fixture does not cover every canonical action")
    if not set(GUI_OWL_V2_ACTION_ALIASES).issubset(source_actions):
        raise ValueError("valid action fixture does not cover every accepted alias")
    if buttons != set(GUI_OWL_V2_SYSTEM_BUTTONS):
        raise ValueError("valid action fixture does not cover every system button")

    for case in fixture["invalid_cases"]:
        try:
            parse_gui_owl_v2_output(case["native_output"])
        except ValueError as error:
            if case["error_contains"] not in str(error):
                raise ValueError(f"unexpected rejection for invalid case: {case['id']}") from error
        else:
            raise ValueError(f"invalid action fixture case was accepted: {case['id']}")

    coordinate_checks = 0
    for extent in (1, 2, 432, 768, int(screen["width"]), int(screen["height"])):
        pixels = [normalized_coordinate_to_pixel(value, extent) for value in range(1000)]
        expected_pixels = [
            (2 * value * (extent - 1) + 999) // 1998 for value in range(1000)
        ]
        if pixels != expected_pixels:
            raise ValueError(f"coordinate scaling formula drifted for extent={extent}")
        if pixels[0] != 0 or pixels[-1] != extent - 1 or pixels != sorted(pixels):
            raise ValueError(f"coordinate scaling invariant failed for extent={extent}")
        if any(pixel < 0 or pixel >= extent for pixel in pixels):
            raise ValueError(f"coordinate scaling left pixel extent={extent}")
        coordinate_checks += len(pixels)
    return payloads, {
        "valid_case_count": len(parsed_cases),
        "invalid_case_count": len(fixture["invalid_cases"]),
        "coordinate_scalar_checks": coordinate_checks,
        "canonical_actions": sorted(canonical_actions),
        "accepted_aliases": sorted(set(GUI_OWL_V2_ACTION_ALIASES)),
        "system_buttons": sorted(buttons),
    }


def _validate_implementation_against_contract(data: Mapping[str, Any]) -> dict[str, Any]:
    reference = data["reference_definition"]
    action = data["action_contract"]
    low = data["low_fidelity_event"]
    high = data["high_fidelity_event"]
    roles = data["data"]["roles"]
    if tuple(action["canonical_prompt_actions"]) != CANONICAL_GUI_OWL_V2_ACTIONS:
        raise ValueError("implementation canonical actions drifted from the scientific contract")
    if action["accepted_model_aliases"] != GUI_OWL_V2_ACTION_ALIASES:
        raise ValueError("implementation aliases drifted from the scientific contract")
    if tuple(action["system_buttons"]) != GUI_OWL_V2_SYSTEM_BUTTONS:
        raise ValueError("implementation system buttons drifted from the scientific contract")
    parameters = {
        name: (["status=success"] if name == "terminate" else list(values))
        for name, values in GUI_OWL_V2_PARAMETERS_BY_ACTION.items()
    }
    if action["parameter_contract"] != parameters:
        raise ValueError("implementation parameter matrix drifted from the scientific contract")
    if reference["teacher_forced_action_carrier"] != GUI_OWL_V2_TEACHER_CARRIER:
        raise ValueError("teacher-forced carrier drifted from the scientific contract")
    if tuple(reference["canonical_tool_call_serialization"]["argument_key_order"]) != GUI_OWL_V2_ARGUMENT_KEY_ORDER:
        raise ValueError("canonical argument key order drifted from the scientific contract")
    if tuple(low["serialized_key_order"]) != LOW_FIDELITY_V2_KEYS:
        raise ValueError("low-fidelity key order drifted from the scientific contract")
    if LOW_FIDELITY_V2_ACTIONS != CANONICAL_GUI_OWL_V2_ACTIONS:
        raise ValueError("low-fidelity action inventory must use canonical policy actions")
    text_delta = low["screen_text_delta"]
    if (
        int(text_delta["maximum_added_tokens"]) != MAXIMUM_SCREEN_TEXT_TOKENS
        or int(text_delta["maximum_removed_tokens"]) != MAXIMUM_SCREEN_TEXT_TOKENS
    ):
        raise ValueError("screen-text cap drifted from the scientific contract")
    if SCREEN_CHANGE_THRESHOLDS != (0.005, 0.05, 0.20):
        raise ValueError("screen-change thresholds drifted from the scientific contract")
    threshold_categories = tuple(
        screen_change_from_mean_absolute_rgb_difference(value)
        for value in (0.005, 0.0050001, 0.05, 0.050001, 0.20, 0.200001)
    )
    if threshold_categories != ("none", "low", "low", "medium", "medium", "high"):
        raise ValueError("screen-change boundary behavior drifted from the scientific contract")
    if tuple(low["executor_result_values"]) != EXECUTOR_RESULT_VALUES:
        raise ValueError("executor-result inventory drifted from the scientific contract")
    if not (
        high["images_per_event"] == 1
        and high["image"] == "post_action_state"
        and high["include_before_image"] is False
        and high["include_additional_action_text"] is False
    ):
        raise ValueError("high-fidelity image intervention drifted from the scientific contract")
    label_steps = tuple(roles["v2_label_train"]["state_decision_step_ids"])
    development_steps = tuple(roles["v2_development"]["state_decision_step_ids"])
    if label_steps != GUI_OWL_V2_DECISION_STEPS or development_steps != GUI_OWL_V2_DECISION_STEPS:
        raise ValueError("development decision steps drifted from the scientific contract")
    if roles["v2_confirm_primary"]["decision_step_id"] != GUI_OWL_V2_DECISION_STEPS[-1]:
        raise ValueError("confirm decision step drifted from the scientific contract")
    return {
        "canonical_action_count": len(CANONICAL_GUI_OWL_V2_ACTIONS),
        "low_fidelity_field_count": len(LOW_FIDELITY_V2_KEYS),
        "decision_steps": list(GUI_OWL_V2_DECISION_STEPS),
        "screen_text_token_cap": MAXIMUM_SCREEN_TEXT_TOKENS,
    }


def _validate_prompt_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    if fixture.get("schema_version") != "1.0.0":
        raise ValueError("unexpected prompt fixture schema_version")
    if fixture.get("protocol_id") != "causalcache_restoration_v2":
        raise ValueError("unexpected prompt fixture protocol_id")
    trajectory = fixture["trajectories"][0]
    for event in trajectory["events"]:
        record = LowFidelityEventV2.from_mapping(event["low_fidelity_v2"])
        serialized = serialize_low_fidelity_v2(record)
        if serialized.decode("utf-8") != event["low_fidelity_v2_serialized"]:
            raise ValueError(f"low-fidelity serialization mismatch at step {event['step_id']}")
        if sha256_bytes(serialized) != event["low_fidelity_v2_sha256"]:
            raise ValueError(f"low-fidelity SHA256 mismatch at step {event['step_id']}")

    coalition_count = 0
    confirm_coalition_count = 0
    event_by_step = {int(event["step_id"]): event for event in trajectory["events"]}
    decisions = {int(decision["decision_step_id"]): decision for decision in trajectory["decisions"]}
    if set(decisions) != {4, 5, 6}:
        raise ValueError("prompt fixture must cover exactly decision steps 4, 5, and 6")
    for decision_step_id in (4, 5, 6):
        decision = decisions[decision_step_id]
        candidates = tuple(int(value) for value in decision["candidate_event_step_ids"])
        reference_text = None
        for size in range(len(candidates) + 1):
            for coalition in itertools.combinations(candidates, size):
                messages = build_gui_owl_v2_mixed_fidelity_messages(
                    fixture,
                    trajectory_id=trajectory["source_id"],
                    decision_step_id=decision_step_id,
                    restored_event_step_ids=coalition,
                    image_bytes_loader=lambda path: path.encode("utf-8"),
                    image_decoder=lambda raw: raw.decode("utf-8"),
                )
                if [message["role"] for message in messages] != ["system", "user"]:
                    raise ValueError("v2 prompt must contain exactly system and user messages")
                text_items = [
                    item["text"]
                    for message in messages
                    for item in message["content"]
                    if item["type"] == "text"
                ]
                if reference_text is None:
                    reference_text = text_items
                if text_items != reference_text:
                    raise ValueError(
                        f"policy-visible text changed for step={decision_step_id}, coalition={coalition}"
                    )
                expected_content_types = ["text"]
                for step_id in decision["history_event_step_ids"]:
                    expected_content_types.append("text")
                    if step_id in coalition:
                        expected_content_types.append("image")
                expected_content_types.extend(("text", "image", "text"))
                if [item["type"] for item in messages[1]["content"]] != expected_content_types:
                    raise ValueError("restored post-state image must immediately follow its event summary")
                image_items = [
                    item["image"]
                    for message in messages
                    for item in message["content"]
                    if item["type"] == "image"
                ]
                expected = [
                    event_by_step[step_id]["observation_after_path"] for step_id in coalition
                ]
                expected.append(decision["current_observation_path"])
                if image_items != expected:
                    raise ValueError(
                        f"post-state-only image contract failed for step={decision_step_id}, coalition={coalition}"
                    )
                coalition_count += 1
                if decision_step_id == 6:
                    confirm_coalition_count += 1
    return {
        "event_count": len(trajectory["events"]),
        "coalition_count": coalition_count,
        "confirm_coalition_count": confirm_coalition_count,
        "decision_steps": [4, 5, 6],
        "reference_image_count": 5,
        "summary_only_image_count": 1,
    }


def _validate_androidworld_payloads(
    payloads: list[dict[str, Any]],
    source_root: Path | None,
    *,
    source_revision: str | None,
    container_image_digest: str | None,
    run_git_commit: str | None,
) -> dict[str, Any]:
    provenance = (source_root, source_revision, container_image_digest, run_git_commit)
    if all(value is None for value in provenance):
        return {"status": "not_run", "validated_case_count": 0}
    if any(value is None for value in provenance):
        raise ValueError("AndroidWorld integration requires source root/revision, image digest, and run Git commit")
    assert source_root is not None
    assert source_revision is not None
    assert container_image_digest is not None
    assert run_git_commit is not None
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError("AndroidWorld source revision must be a full lowercase Git SHA")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", container_image_digest) is None:
        raise ValueError("container image digest must use sha256:<64 lowercase hex>")
    if re.fullmatch(r"[0-9a-f]{40}", run_git_commit) is None:
        raise ValueError("run Git commit must be a full lowercase Git SHA")
    if not source_root.is_dir():
        raise FileNotFoundError(f"AndroidWorld source root does not exist: {source_root}")
    source_root = source_root.resolve()
    source_head = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_revision != FROZEN_ANDROIDWORLD_JSON_ACTION_REVISION:
        raise ValueError("AndroidWorld source revision is not the frozen MobileAgent revision")
    if source_head != source_revision:
        raise ValueError("AndroidWorld source checkout does not match the pinned revision")
    source_status = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if source_status:
        raise ValueError("AndroidWorld source checkout must be clean")
    repository_head = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if repository_head != run_git_commit:
        raise ValueError("run Git commit does not match the CausalCache checkout")
    repository_status = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if repository_status:
        raise ValueError("CausalCache checkout must be clean for AndroidWorld integration")
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source_root))
    try:
        from android_world.agents import new_json_action
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = previous_dont_write_bytecode
    module_path = Path(new_json_action.__file__).resolve()
    if not module_path.is_relative_to(source_root):
        raise ValueError("imported AndroidWorld JSONAction module is outside the pinned source root")
    for item in payloads:
        try:
            new_json_action.JSONAction(**item["payload"])
        except Exception as error:
            raise ValueError(
                f"pinned AndroidWorld JSONAction rejected fixture case: {item['case_id']}"
            ) from error
    source_status_after = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if source_status_after:
        raise ValueError("AndroidWorld integration modified the pinned source checkout")
    return {
        "status": "passed",
        "validated_case_count": len(payloads),
        "source_root": str(source_root),
        "source_revision": source_revision,
        "module_path": str(module_path),
        "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "container_image_digest": container_image_digest,
        "run_git_commit": run_git_commit,
    }


def _validate_interface_manifest(
    manifest_path: Path,
    *,
    scientific_contract_sha256: str,
    action_summary: Mapping[str, Any],
    prompt_summary: Mapping[str, Any],
) -> dict[str, Any]:
    manifest, manifest_sha256 = _read_json(manifest_path)
    if manifest.get("protocol_id") != "causalcache_restoration_v2":
        raise ValueError("unexpected restoration-v2 interface manifest protocol_id")
    if manifest.get("schema_version") != "1.0.0":
        raise ValueError("unexpected restoration-v2 interface manifest schema_version")
    if manifest.get("interface_version") != "restoration-v2-interface-1.0.0":
        raise ValueError("unexpected restoration-v2 interface version")
    if manifest.get("scientific_contract_sha256") != scientific_contract_sha256:
        raise ValueError("interface manifest scientific contract SHA256 mismatch")
    if manifest.get("materialized_before_any_v2_policy_output") is not True:
        raise ValueError("interface manifest must be materialized before v2 policy output")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("interface manifest files must be a non-empty list")
    paths = [str(record["path"]) for record in records]
    expected_paths = {
        "code/causalcache/policy/gui_owl_v2.py",
        "code/causalcache/low_fidelity_v2.py",
        "code/scripts/validate_restoration_v2_interfaces.py",
        "data/fixtures/gui_owl_v2_action_roundtrip.json",
        "data/fixtures/restoration_v2_prompt_low_fidelity.json",
        "docs/restoration_v2_interfaces.md",
    }
    if set(paths) != expected_paths:
        raise ValueError("interface manifest must pin the complete frozen source set")
    if len(paths) != len(set(paths)):
        raise ValueError("interface manifest file paths must be unique")
    for record in records:
        relative_path = Path(str(record["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("interface manifest paths must be safe repository-relative paths")
        path = REPOSITORY_ROOT / relative_path
        raw = path.read_bytes()
        if len(raw) != int(record["size_bytes"]):
            raise ValueError(f"interface source size mismatch: {relative_path}")
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"interface source SHA256 mismatch: {relative_path}")
    local = manifest.get("local_cpu_validation")
    expected_local = {
        "valid_action_cases": action_summary["valid_case_count"],
        "invalid_action_cases": action_summary["invalid_case_count"],
        "coordinate_scalar_checks": action_summary["coordinate_scalar_checks"],
        "prompt_coalitions": prompt_summary["coalition_count"],
        "confirm_prompt_coalitions": prompt_summary["confirm_coalition_count"],
        "status": "passed",
    }
    if local != expected_local:
        raise ValueError("interface manifest local CPU validation summary mismatch")
    if manifest.get("policy_output_generated_by_this_validation") is not False:
        raise ValueError("interface manifest must not claim a v2 policy output")
    if manifest.get("androidworld_json_action_constructor_validation") != {
        "status": "pending",
        "reason": "requires pinned AndroidWorld source integration preflight",
    }:
        raise ValueError("frozen interface manifest must keep executor integration pending")
    if manifest.get("androidworld_executor_dispatch_validation") != {
        "status": "pending",
        "reason": "JSONAction construction alone does not execute device-side effects",
    }:
        raise ValueError("frozen interface manifest must keep executor dispatch pending")
    return {
        "path": str(manifest_path),
        "sha256": manifest_sha256,
        "interface_version": manifest["interface_version"],
        "source_file_count": len(records),
    }


def validate_interfaces(
    *,
    contract_path: Path,
    action_fixture_path: Path,
    prompt_fixture_path: Path,
    interface_manifest_path: Path,
    androidworld_source_root: Path | None = None,
    androidworld_source_revision: str | None = None,
    container_image_digest: str | None = None,
    run_git_commit: str | None = None,
) -> dict[str, Any]:
    contract = RestorationV2Contract.load(contract_path)
    expected_paths = {
        "action": REPOSITORY_ROOT / "data/fixtures/gui_owl_v2_action_roundtrip.json",
        "prompt": REPOSITORY_ROOT / "data/fixtures/restoration_v2_prompt_low_fidelity.json",
        "manifest": REPOSITORY_ROOT / "data/manifests/restoration_v2_interfaces.json",
    }
    actual_paths = {
        "action": action_fixture_path.resolve(),
        "prompt": prompt_fixture_path.resolve(),
        "manifest": interface_manifest_path.resolve(),
    }
    for name, expected in expected_paths.items():
        if actual_paths[name] != expected.resolve():
            raise ValueError(f"{name} input must be the canonical repository-pinned file")
    contract_bindings = _validate_implementation_against_contract(contract.data)
    action_fixture, action_sha256 = _read_json(action_fixture_path)
    prompt_fixture, prompt_sha256 = _read_json(prompt_fixture_path)
    for name, fixture in (("action", action_fixture), ("prompt", prompt_fixture)):
        if fixture.get("scientific_contract_sha256") != contract.source_sha256:
            raise ValueError(f"{name} fixture scientific contract SHA256 mismatch")
    payloads, action_summary = _validate_action_fixture(action_fixture)
    prompt_summary = _validate_prompt_fixture(prompt_fixture)
    interface_manifest = _validate_interface_manifest(
        interface_manifest_path,
        scientific_contract_sha256=contract.source_sha256,
        action_summary=action_summary,
        prompt_summary=prompt_summary,
    )
    androidworld = _validate_androidworld_payloads(
        payloads,
        androidworld_source_root,
        source_revision=androidworld_source_revision,
        container_image_digest=container_image_digest,
        run_git_commit=run_git_commit,
    )
    return {
        "schema_version": "1.0.0",
        "protocol_id": contract.protocol_id,
        "scientific_contract_sha256": contract.source_sha256,
        "action_fixture": {
            "path": str(action_fixture_path),
            "sha256": action_sha256,
            **action_summary,
        },
        "prompt_low_fidelity_fixture": {
            "path": str(prompt_fixture_path),
            "sha256": prompt_sha256,
            **prompt_summary,
        },
        "interface_manifest": interface_manifest,
        "implementation_contract_bindings": contract_bindings,
        "androidworld_json_action_constructor_validation": androidworld,
        "policy_output_generated_by_this_validation": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--action-fixture", required=True, type=Path)
    parser.add_argument("--prompt-fixture", required=True, type=Path)
    parser.add_argument("--interface-manifest", required=True, type=Path)
    parser.add_argument("--androidworld-source-root", type=Path)
    parser.add_argument("--androidworld-source-revision")
    parser.add_argument("--container-image-digest")
    parser.add_argument("--run-git-commit")
    parser.add_argument("--output-summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_interfaces(
        contract_path=args.contract,
        action_fixture_path=args.action_fixture,
        prompt_fixture_path=args.prompt_fixture,
        interface_manifest_path=args.interface_manifest,
        androidworld_source_root=args.androidworld_source_root,
        androidworld_source_revision=args.androidworld_source_revision,
        container_image_digest=args.container_image_digest,
        run_git_commit=args.run_git_commit,
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_summary is not None:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
