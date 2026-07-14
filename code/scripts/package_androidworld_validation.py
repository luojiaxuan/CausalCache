"""Package one validated AndroidWorld run as a deterministic HF payload."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


POLICY_SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
FULL_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
TRACE_FILENAME = "validation-00000-of-00001.jsonl.gz"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _load_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    payload = path.read_bytes()
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value, payload


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_probability(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be a finite probability")
    return result


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} does not match validated episodes")


def _validate_policy_slug(policy_slug: str) -> None:
    if POLICY_SLUG_PATTERN.fullmatch(policy_slug) is None:
        raise ValueError(
            "policy slug must be lowercase kebab/dot syntax without path separators"
        )


def _validate_hf_destination(hf_repo: str, hf_tag: str) -> None:
    repo_parts = hf_repo.split("/")
    if (
        len(repo_parts) != 2
        or any(not part or part in {".", ".."} for part in repo_parts)
        or any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) is None
            for part in repo_parts
        )
    ):
        raise ValueError("HF repo must have safe owner/name syntax")
    if (
        not hf_tag
        or hf_tag in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", hf_tag) is None
    ):
        raise ValueError("HF tag must use safe tag syntax")


def _validate_output_target(output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise ValueError("output directory must not be a symlink")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError("output path exists and is not a directory")
        if any(output_dir.iterdir()):
            raise ValueError("refusing to write into a non-empty output directory")


def _validate_plan(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    if plan.get("split") != "validation":
        raise ValueError("packager accepts only a frozen validation plan")
    instances = plan.get("instances")
    if not isinstance(instances, list) or not all(
        isinstance(instance, dict) for instance in instances
    ):
        raise ValueError("validation plan instances must be a list of objects")
    plan_count = _require_exact_int(
        plan.get("task_instance_count"), "plan task_instance_count", minimum=1
    )
    if plan_count != len(instances):
        raise ValueError("validation plan task count does not match instances")
    records_sha256 = _sha256(_canonical_json_bytes(instances))
    if plan.get("instance_records_sha256") != records_sha256:
        raise ValueError("validation plan instance_records_sha256 does not match")
    return instances, records_sha256


def _expected_episode_name(plan_index: int, instance: dict[str, Any]) -> str:
    task_type = instance.get("task_type")
    task_index = instance.get("task_index")
    if not isinstance(task_type, str) or not task_type:
        raise ValueError(f"plan instance {plan_index} has an invalid task_type")
    _require_exact_int(task_index, f"plan instance {plan_index} task_index")
    filename = f"{plan_index:03d}-{task_type}-{task_index}.json"
    if PurePosixPath(filename).name != filename or "\\" in filename:
        raise ValueError(f"plan instance {plan_index} yields an unsafe filename")
    return filename


def _load_and_validate_episodes(
    *,
    validation_run_dir: Path,
    summary: dict[str, Any],
    instances: list[dict[str, Any]],
    run_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    episode_files = summary.get("episode_files")
    if not isinstance(episode_files, list) or not all(
        isinstance(path, str) for path in episode_files
    ):
        raise ValueError("summary episode_files must be a list of paths")
    if len(set(episode_files)) != len(episode_files):
        raise ValueError("summary episode_files contains duplicates")

    episodes_dir = validation_run_dir / "episodes"
    if episodes_dir.is_symlink() or not episodes_dir.is_dir():
        raise ValueError("validation run must contain a regular episodes directory")
    entries = sorted(episodes_dir.iterdir())
    if any(
        path.is_symlink() or not path.is_file() or path.suffix != ".json"
        for path in entries
    ):
        raise ValueError("episodes directory must contain only regular JSON files")
    actual_relative_paths = [f"episodes/{path.name}" for path in entries]
    if set(episode_files) != set(actual_relative_paths):
        raise ValueError("summary episode set does not match the episodes directory")

    episodes_by_index: dict[int, tuple[dict[str, Any], str]] = {}
    for relative_path in episode_files:
        posix_path = PurePosixPath(relative_path)
        if len(posix_path.parts) != 2 or posix_path.parts[0] != "episodes":
            raise ValueError("summary contains an unsafe episode path")
        episode, _ = _load_json_object(
            validation_run_dir / Path(*posix_path.parts), "episode"
        )
        plan_index = _require_exact_int(episode.get("plan_index"), "episode plan_index")
        if plan_index >= len(instances):
            raise ValueError("episode plan_index is outside the frozen plan")
        if plan_index in episodes_by_index:
            raise ValueError(f"duplicate episode plan_index: {plan_index}")
        if episode.get("instance") != instances[plan_index]:
            raise ValueError(f"episode instance does not match plan_index {plan_index}")
        expected_name = _expected_episode_name(plan_index, instances[plan_index])
        if posix_path.name != expected_name:
            raise ValueError(f"episode filename does not match plan_index {plan_index}")
        if episode.get("run_contract") != run_contract:
            raise ValueError(f"episode run_contract does not match summary: {plan_index}")
        episodes_by_index[plan_index] = (episode, relative_path)

    ordered_indices = sorted(episodes_by_index)
    expected_order = [episodes_by_index[index][1] for index in ordered_indices]
    if episode_files != expected_order:
        raise ValueError("summary episode_files must be ordered by plan_index")
    return [episodes_by_index[index][0] for index in ordered_indices]


def _episode_aggregates(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    completed_count = 0
    exception_count = 0
    model_step_count = 0
    parse_success_count = 0
    environment_success_count = 0
    official_success_count = 0
    outcomes: Counter[str] = Counter()
    termination_reasons: Counter[str] = Counter()

    for episode in episodes:
        run_status = episode.get("run_status")
        if run_status == "complete":
            completed_count += 1
        elif run_status == "exception":
            exception_count += 1
        else:
            raise ValueError("episode run_status must be complete or exception")
        episode_model_steps = _require_exact_int(
            episode.get("model_step_count"), "episode model_step_count"
        )
        episode_parse_successes = _require_exact_int(
            episode.get("parse_success_count"), "episode parse_success_count"
        )
        if episode_parse_successes > episode_model_steps:
            raise ValueError("episode parse successes exceed model steps")
        model_step_count += episode_model_steps
        parse_success_count += episode_parse_successes
        if type(episode.get("environment_success")) is not bool:
            raise ValueError("episode environment_success must be boolean")
        if type(episode.get("official_success")) is not bool:
            raise ValueError("episode official_success must be boolean")
        environment_success_count += int(episode["environment_success"])
        official_success_count += int(episode["official_success"])
        termination_reason = episode.get("termination_reason")
        if not isinstance(termination_reason, str) or not termination_reason:
            raise ValueError("episode termination_reason must be a non-empty string")
        termination_reasons[termination_reason] += 1
        if run_status == "exception":
            outcomes["infrastructure_failure"] += 1
        elif termination_reason == "parse_error":
            outcomes["parse_failure"] += 1
        elif termination_reason == "executor_error":
            outcomes["executor_failure"] += 1
        elif episode["official_success"]:
            outcomes["official_success"] += 1
        else:
            outcomes["terminal_failure"] += 1

    parse_coverage = (
        parse_success_count / model_step_count if model_step_count else 0.0
    )
    return {
        "completed_episode_count": completed_count,
        "exception_episode_count": exception_count,
        "model_step_count": model_step_count,
        "parse_success_count": parse_success_count,
        "parse_coverage": parse_coverage,
        "environment_success_count": environment_success_count,
        "official_success_count": official_success_count,
        "outcomes": dict(sorted(outcomes.items())),
        "termination_reasons": dict(sorted(termination_reasons.items())),
    }


def _validate_summary_counts(
    *,
    summary: dict[str, Any],
    episodes: list[dict[str, Any]],
    plan_count: int,
) -> str:
    aggregates = _episode_aggregates(episodes)
    for key in (
        "completed_episode_count",
        "exception_episode_count",
        "model_step_count",
        "parse_success_count",
        "parse_coverage",
        "official_success_count",
        "outcomes",
        "termination_reasons",
    ):
        _require_equal(summary.get(key), aggregates[key], f"summary {key}")

    gates = summary.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("summary gates must be an object")
    minimum_parse_coverage = _require_probability(
        gates.get("minimum_parse_coverage"), "minimum_parse_coverage"
    )
    minimum_official_success = _require_probability(
        gates.get("minimum_official_success"), "minimum_official_success"
    )
    parse_gate_passed = aggregates["parse_coverage"] >= minimum_parse_coverage
    episode_count = len(episodes)

    if "artifact_status" not in summary:
        if episode_count != plan_count:
            raise ValueError("complete summary must contain every plan checkpoint")
        if [episode["plan_index"] for episode in episodes] != list(range(plan_count)):
            raise ValueError("complete summary must cover every plan_index")
        _require_equal(
            summary.get("task_instance_count"), plan_count, "summary task_instance_count"
        )
        _require_equal(
            summary.get("environment_success_count"),
            aggregates["environment_success_count"],
            "summary environment_success_count",
        )
        official_success_rate = aggregates["official_success_count"] / plan_count
        _require_equal(
            summary.get("official_success_rate"),
            official_success_rate,
            "summary official_success_rate",
        )
        _require_equal(
            gates.get("parse_gate_passed"), parse_gate_passed, "summary parse gate"
        )
        success_gate_passed = official_success_rate >= minimum_official_success
        _require_equal(
            gates.get("official_success_gate_passed"),
            success_gate_passed,
            "summary official success gate",
        )
        _require_equal(
            gates.get("validation_gate_passed"),
            parse_gate_passed and success_gate_passed,
            "summary validation gate",
        )
        return "complete"

    if summary.get("artifact_status") != "valid_early_stopped_policy_rejection":
        raise ValueError("unsupported summary artifact_status")
    if not 0 < episode_count < plan_count:
        raise ValueError("early-stop summary requires a non-empty partial run")
    _require_equal(
        summary.get("plan_instance_count"), plan_count, "summary plan_instance_count"
    )
    _require_equal(
        summary.get("checkpoint_count"), episode_count, "summary checkpoint_count"
    )
    unobserved_count = plan_count - episode_count
    _require_equal(
        summary.get("unobserved_instance_count"),
        unobserved_count,
        "summary unobserved_instance_count",
    )
    _require_equal(
        summary.get("early_stop_reason"),
        "success_gate_mathematically_impossible",
        "summary early_stop_reason",
    )
    minimum_required = math.ceil(minimum_official_success * plan_count)
    maximum_possible = aggregates["official_success_count"] + unobserved_count
    if maximum_possible >= minimum_required:
        raise ValueError("early-stop summary is not mathematically decisive")
    _require_equal(
        summary.get("minimum_required_official_success_count"),
        minimum_required,
        "summary minimum required successes",
    )
    _require_equal(
        summary.get("maximum_possible_official_success_count"),
        maximum_possible,
        "summary maximum possible successes",
    )
    _require_equal(
        summary.get("maximum_possible_official_success_rate"),
        maximum_possible / plan_count,
        "summary maximum possible success rate",
    )
    _require_equal(
        summary.get("official_success_rate_lower_bound"),
        aggregates["official_success_count"] / plan_count,
        "summary official success lower bound",
    )
    _require_equal(
        gates.get("parse_gate_passed_on_observed_actions"),
        parse_gate_passed,
        "summary observed parse gate",
    )
    _require_equal(
        gates.get("official_success_gate_passed"), False, "summary success gate"
    )
    _require_equal(
        gates.get("validation_gate_passed"), False, "summary validation gate"
    )
    return "early_stopped_policy_rejection"


def _deterministic_gzip_jsonl(episodes: list[dict[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=buffer,
        mtime=0,
    ) as archive:
        for episode in episodes:
            archive.write(_canonical_json_bytes(episode))
            archive.write(b"\n")
    return buffer.getvalue()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def package_validation(
    *,
    validation_run_dir: Path,
    validation_plan: Path,
    output_dir: Path,
    policy_slug: str,
    source_run_git_commit: str,
    hf_repo: str,
    hf_tag: str,
) -> dict[str, Any]:
    _validate_policy_slug(policy_slug)
    if FULL_GIT_SHA_PATTERN.fullmatch(source_run_git_commit) is None:
        raise ValueError("source run git commit must be a full lowercase SHA")
    _validate_hf_destination(hf_repo, hf_tag)
    _validate_output_target(output_dir)

    plan, _ = _load_json_object(validation_plan, "validation plan")
    instances, plan_records_sha256 = _validate_plan(plan)
    summary, summary_bytes = _load_json_object(
        validation_run_dir / "summary.json", "validation summary"
    )
    if summary.get("split") != "validation":
        raise ValueError("summary split does not match the validation plan")
    if summary.get("instance_records_sha256") != plan_records_sha256:
        raise ValueError("summary does not match frozen plan records")
    run_contract = summary.get("run_contract")
    if not isinstance(run_contract, dict):
        raise ValueError("summary must contain a run_contract object")
    if run_contract.get("git_commit") != source_run_git_commit:
        raise ValueError("source run git commit does not match summary run_contract")
    if run_contract.get("validation_plan_records_sha256") != plan_records_sha256:
        raise ValueError("summary run_contract does not match frozen plan records")

    episodes = _load_and_validate_episodes(
        validation_run_dir=validation_run_dir,
        summary=summary,
        instances=instances,
        run_contract=run_contract,
    )
    run_kind = _validate_summary_counts(
        summary=summary,
        episodes=episodes,
        plan_count=len(instances),
    )
    shard_bytes = _deterministic_gzip_jsonl(episodes)

    shard_relative = Path("data") / policy_slug / TRACE_FILENAME
    summary_relative = Path("runs") / policy_slug / "summary.json"
    manifest_relative = Path("runs") / policy_slug / "payload_manifest.json"
    manifest = {
        "schema_version": "0.1.0",
        "artifact_type": "androidworld_closed_loop_validation_payload",
        "policy_slug": policy_slug,
        "source_run_git_commit": source_run_git_commit,
        "run_kind": run_kind,
        "record_count": len(episodes),
        "validation_plan": {
            "split": "validation",
            "instance_records_sha256": plan_records_sha256,
            "task_instance_count": len(instances),
        },
        "hf_destination": {
            "repo": hf_repo,
            "repo_type": "dataset",
            "tag": hf_tag,
        },
        "layout": {
            "trace_shard": shard_relative.as_posix(),
            "summary": summary_relative.as_posix(),
            "manifest": manifest_relative.as_posix(),
        },
        "files": {
            "trace_shard": {
                "path": shard_relative.as_posix(),
                "sha256": _sha256(shard_bytes),
                "bytes": len(shard_bytes),
                "record_count": len(episodes),
            },
            "summary": {
                "path": summary_relative.as_posix(),
                "sha256": _sha256(summary_bytes),
                "bytes": len(summary_bytes),
            },
        },
    }
    manifest_bytes = _canonical_json_bytes(manifest) + b"\n"

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        _write_bytes_atomic(staging_dir / shard_relative, shard_bytes)
        _write_bytes_atomic(staging_dir / summary_relative, summary_bytes)
        _write_bytes_atomic(staging_dir / manifest_relative, manifest_bytes)
        if output_dir.exists():
            _validate_output_target(output_dir)
            output_dir.rmdir()
        os.replace(staging_dir, output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-run-dir", type=Path, required=True)
    parser.add_argument("--validation-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-slug", required=True)
    parser.add_argument("--source-run-git-commit", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-tag", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = package_validation(
        validation_run_dir=args.validation_run_dir,
        validation_plan=args.validation_plan,
        output_dir=args.output_dir,
        policy_slug=args.policy_slug,
        source_run_git_commit=args.source_run_git_commit,
        hf_repo=args.hf_repo,
        hf_tag=args.hf_tag,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
