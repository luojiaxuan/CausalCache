"""Fail-closed orchestration for the frozen real-policy go/no-go diagnostic."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causalcache.attribution import (
    estimate_budget_conditioned_restoration,
    exact_permutation_restoration,
)
from causalcache.diagnostic import (
    DiagnosticOutcomeRow,
    canonical_json_action,
    classify_diagnostic_outcome,
    evaluate_exact_coalition_table,
    has_positive_non_recent_gain,
    normalized_recovery,
    rgb_histogram_cosine,
)
from causalcache.policy import build_policy_messages
from causalcache.policy.qwen_runtime import (
    QwenPolicyRuntime,
    action_dict,
    full_vocab_action_path_kl,
    load_dataset,
)
from causalcache.schema import ExecutableAction
from causalcache.selection import jaccard, select_positive_value_knapsack, spearman


Coalition = frozenset[int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def enumerate_bounded_coalitions(
    event_ids: Sequence[int],
    *,
    maximum_restored_events: int,
) -> tuple[Coalition, ...]:
    ordered_ids = tuple(sorted(int(event_id) for event_id in event_ids))
    if len(ordered_ids) != len(set(ordered_ids)) or not ordered_ids:
        raise ValueError("event ids must be unique and non-empty")
    if maximum_restored_events < 0:
        raise ValueError("maximum_restored_events cannot be negative")
    result: list[Coalition] = []
    for size in range(min(len(ordered_ids), maximum_restored_events) + 1):
        result.extend(frozenset(values) for values in itertools.combinations(ordered_ids, size))
    return tuple(result)


def coalition_key(coalition: Coalition) -> str:
    return ",".join(str(event_id) for event_id in sorted(coalition))


def validate_visual_accounting(
    metadata: Mapping[str, Any],
    *,
    restored_event_count: int,
    intervention: Mapping[str, Any],
) -> dict[str, int]:
    if restored_event_count < 0:
        raise ValueError("restored_event_count cannot be negative")
    images_per_event = int(intervention["images_per_restored_event"])
    per_image = int(intervention["effective_visual_tokens_per_image"])
    event_cost = int(intervention["event_cost_visual_tokens"])
    if images_per_event * per_image != event_cost:
        raise ValueError("configured event cost does not match effective image accounting")
    expected_images = 1 + images_per_event * restored_event_count
    expected_visual_tokens = expected_images * per_image
    actual_images = int(metadata["image_count"])
    actual_visual_tokens = int(metadata["effective_visual_tokens"])
    if actual_images != expected_images:
        raise ValueError(
            f"image-count mismatch: {actual_images} != {expected_images}"
        )
    if actual_visual_tokens != expected_visual_tokens:
        raise ValueError(
            "effective visual-token mismatch: "
            f"{actual_visual_tokens} != {expected_visual_tokens}"
        )
    return {
        "image_count": actual_images,
        "effective_visual_tokens": actual_visual_tokens,
        "incremental_memory_visual_tokens": restored_event_count * event_cost,
    }


def validate_run_contract(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    dataset_tar: Path,
    model_dir: Path,
    run_git_commit: str,
    repo_root: Path,
) -> dict[str, Any]:
    if config.get("name") != "exploratory_oracle_diagnostic_v1":
        raise ValueError("unexpected diagnostic config name")
    if config.get("status") != "preregistered_before_restoration_forward":
        raise ValueError("diagnostic config is not preregistered")
    if config["scope"].get("can_produce_paper_go") is not False:
        raise ValueError("selection-biased diagnostic must not produce a paper GO")

    actual_commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != run_git_commit:
        raise ValueError(f"run Git commit mismatch: {actual_commit} != {run_git_commit}")
    worktree_status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if worktree_status:
        raise ValueError("formal diagnostic requires a clean Git worktree")

    observed_dataset_sha = sha256_file(dataset_tar)
    expected_dataset_sha = str(config["dataset"]["shard_sha256"])
    if observed_dataset_sha != expected_dataset_sha:
        raise ValueError("dataset shard SHA256 does not match the frozen config")

    snapshot_path = model_dir / ".snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("repo") != config["policy"]["repo"]:
        raise ValueError("model snapshot repo does not match the frozen config")
    if snapshot.get("revision") != config["policy"]["revision"]:
        raise ValueError("model snapshot revision does not match the frozen config")

    intervention = config["intervention"]
    if (
        int(intervention["effective_visual_tokens_per_image"])
        * int(intervention["images_per_restored_event"])
        != int(intervention["event_cost_visual_tokens"])
    ):
        raise ValueError("frozen visual cost fields are internally inconsistent")
    budgets = tuple(int(value) for value in intervention["budgets_visual_tokens"])
    maximum_events = int(config["coalitions"]["maximum_restored_events"])
    event_cost = int(intervention["event_cost_visual_tokens"])
    if tuple(budget // event_cost for budget in budgets) != (1, maximum_events):
        raise ValueError("frozen budgets no longer encode the registered 1/2-event pilot")

    return {
        "git_commit": actual_commit,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_json_sha256(config),
        "dataset_path": str(dataset_tar),
        "dataset_sha256": observed_dataset_sha,
        "model_dir": str(model_dir),
        "model_snapshot": snapshot,
    }


def _distance_for_index_coalition(
    coalition: Coalition,
    *,
    index_to_step: Mapping[int, int],
    distances: Mapping[Coalition, float],
) -> float:
    step_coalition = frozenset(index_to_step[index] for index in coalition)
    return float(distances[step_coalition])


def _selected_steps(
    selected_indices: Coalition,
    *,
    index_to_step: Mapping[int, int],
) -> Coalition:
    return frozenset(index_to_step[index] for index in selected_indices)


def _selection_metric(
    coalition: Coalition,
    *,
    distances: Mapping[Coalition, float],
    event_cost: int,
    epsilon: float,
) -> dict[str, Any]:
    summary_distance = float(distances[frozenset()])
    distance = float(distances[coalition])
    return {
        "coalition": sorted(coalition),
        "cost": len(coalition) * event_cost,
        "distance": distance,
        "utility": summary_distance - distance,
        "normalized_recovery": normalized_recovery(
            summary_distance=summary_distance,
            distance=distance,
            epsilon=epsilon,
        ),
    }


def _attribution_for_budget(
    *,
    distances: Mapping[Coalition, float],
    event_ids: Sequence[int],
    event_cost: int,
    budget: int,
    epsilon: float,
    sample_counts: Sequence[int],
    seeds: Sequence[int],
) -> dict[str, Any]:
    ordered_steps = tuple(sorted(event_ids))
    index_to_step = {index: step_id for index, step_id in enumerate(ordered_steps)}
    costs = {index: event_cost for index in index_to_step}

    def indexed_distance(coalition: Coalition) -> float:
        return _distance_for_index_coalition(
            coalition,
            index_to_step=index_to_step,
            distances=distances,
        )

    exact_index_scores = exact_permutation_restoration(
        indexed_distance,
        costs,
        budget=budget,
    )
    exact_scores = {
        index_to_step[index]: float(score) for index, score in exact_index_scores.items()
    }
    exact_selected_indices = select_positive_value_knapsack(
        exact_index_scores,
        costs,
        budget=budget,
    )
    exact_selected = _selected_steps(
        exact_selected_indices,
        index_to_step=index_to_step,
    )
    exact_metric = _selection_metric(
        exact_selected,
        distances=distances,
        event_cost=event_cost,
        epsilon=epsilon,
    )

    sweeps = []
    for sample_count in sample_counts:
        for seed in seeds:
            estimates = estimate_budget_conditioned_restoration(
                indexed_distance,
                costs,
                budget=budget,
                sample_count=int(sample_count),
                seed=int(seed),
            )
            estimated_index_scores = {
                index: estimate.mean for index, estimate in estimates.items()
            }
            estimated_scores = {
                index_to_step[index]: float(score)
                for index, score in estimated_index_scores.items()
            }
            selected_indices = select_positive_value_knapsack(
                estimated_index_scores,
                costs,
                budget=budget,
            )
            selected = _selected_steps(selected_indices, index_to_step=index_to_step)
            metric = _selection_metric(
                selected,
                distances=distances,
                event_cost=event_cost,
                epsilon=epsilon,
            )
            exact_utility = float(exact_metric["utility"])
            if abs(exact_utility) > epsilon:
                utility_ratio: float | None = float(metric["utility"]) / exact_utility
            elif abs(float(metric["utility"])) <= epsilon:
                utility_ratio = 1.0
            else:
                utility_ratio = None
            sweeps.append(
                {
                    "sample_count": int(sample_count),
                    "seed": int(seed),
                    "scores": estimated_scores,
                    "standard_errors": {
                        index_to_step[index]: float(estimate.standard_error)
                        for index, estimate in estimates.items()
                    },
                    "selected": metric,
                    "spearman_vs_exact": spearman(
                        exact_index_scores,
                        estimated_index_scores,
                    ),
                    "top_budget_jaccard_vs_exact": jaccard(
                        selected_indices,
                        exact_selected_indices,
                    ),
                    "exact_selector_utility_ratio": utility_ratio,
                }
            )

    return {
        "exact_scores": exact_scores,
        "exact_selected": exact_metric,
        "sample_sweep": sweeps,
    }


def _gpu_metadata(runtime: QwenPolicyRuntime) -> dict[str, Any]:
    device_index = runtime.torch.device(runtime.device).index
    if device_index is None:
        raise ValueError("runtime device must include an explicit CUDA index")
    properties = runtime.torch.cuda.get_device_properties(device_index)
    return {
        "logical_device": runtime.device,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
    }


def _state_similarity_scores(
    manifest: Mapping[str, Any],
    *,
    decision: Mapping[str, Any],
    image_loader: Callable[[str], Any],
) -> dict[int, float]:
    events = {
        int(event["step_id"]): event for event in manifest["trajectory"]["events"]
    }
    current_image = image_loader(str(decision["current_observation_path"]))
    return {
        int(step_id): rgb_histogram_cosine(
            image_loader(str(events[int(step_id)]["observation_after_path"])),
            current_image,
        )
        for step_id in decision["history_event_step_ids"]
    }


def _run_state(
    *,
    runtime: QwenPolicyRuntime,
    manifest: Mapping[str, Any],
    image_loader: Callable[[str], Any],
    state_spec: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], DiagnosticOutcomeRow]:
    step_id = int(state_spec["decision_step_id"])
    decisions = {
        int(decision["decision_step_id"]): decision
        for decision in manifest["trajectory"]["decisions"]
    }
    if step_id not in decisions:
        raise ValueError(f"configured decision step {step_id} is absent from the dataset")
    decision = decisions[step_id]
    history_ids = tuple(int(value) for value in decision["history_event_step_ids"])
    if len(history_ids) != int(state_spec["history_events"]):
        raise ValueError("configured history length does not match the dataset")
    validated_action = ExecutableAction.from_dict(decision["validated_action"])
    intervention = config["intervention"]
    event_cost = int(intervention["event_cost_visual_tokens"])
    max_events = int(config["coalitions"]["maximum_restored_events"])

    full_messages = build_policy_messages(
        manifest,
        decision_step_id=step_id,
        restored_event_step_ids=history_ids,
        image_loader=image_loader,
    )
    full_generation = runtime.generate(
        full_messages,
        max_new_tokens=int(config["policy"]["decoding"]["max_new_tokens"]),
        validated_action=validated_action,
    )
    if not full_generation["executable_match"]:
        raise ValueError(f"full-history action at decision {step_id} is no longer validated")
    validate_visual_accounting(
        full_generation,
        restored_event_count=len(history_ids),
        intervention=intervention,
    )
    canonical_action = canonical_json_action(str(full_generation["output_text"]))
    action_token_ids = runtime.tokenize_canonical_action(canonical_action)

    reference_log_probs, reference_metadata = runtime.teacher_forced_action_log_probs(
        full_messages,
        action_token_ids,
    )
    validate_visual_accounting(
        reference_metadata,
        restored_event_count=len(history_ids),
        intervention=intervention,
    )
    repeat_log_probs, repeat_metadata = runtime.teacher_forced_action_log_probs(
        full_messages,
        action_token_ids,
    )
    validate_visual_accounting(
        repeat_metadata,
        restored_event_count=len(history_ids),
        intervention=intervention,
    )
    repeat_kl = full_vocab_action_path_kl(reference_log_probs, repeat_log_probs)
    absolute_floor = float(config["distance"]["sensitivity_epsilon"]["absolute_floor"])
    noise_multiplier = float(
        config["distance"]["sensitivity_epsilon"]["repeat_noise_multiplier"]
    )
    epsilon = max(absolute_floor, noise_multiplier * float(repeat_kl["mean"]))

    coalitions = enumerate_bounded_coalitions(
        history_ids,
        maximum_restored_events=max_events,
    )
    distance_records: dict[Coalition, dict[str, Any]] = {}
    for coalition in coalitions:
        messages = build_policy_messages(
            manifest,
            decision_step_id=step_id,
            restored_event_step_ids=sorted(coalition),
            image_loader=image_loader,
        )
        candidate_log_probs, metadata = runtime.teacher_forced_action_log_probs(
            messages,
            action_token_ids,
        )
        accounting = validate_visual_accounting(
            metadata,
            restored_event_count=len(coalition),
            intervention=intervention,
        )
        kl = full_vocab_action_path_kl(reference_log_probs, candidate_log_probs)
        distance_records[coalition] = {
            "distance_mean": float(kl["mean"]),
            "distance_sum": float(kl["sum"]),
            "per_token_kl": [float(value) for value in kl["per_token"]],
            "accounting": accounting,
            "forward": metadata,
        }
        del candidate_log_probs
    del reference_log_probs, repeat_log_probs

    distances = {
        coalition: float(record["distance_mean"])
        for coalition, record in distance_records.items()
    }
    summary_distance = distances[frozenset()]
    similarity_scores = _state_similarity_scores(
        manifest,
        decision=decision,
        image_loader=image_loader,
    )
    event_costs = {event_id: event_cost for event_id in history_ids}
    budget_results: dict[str, Any] = {}
    selected_for_generation: set[Coalition] = {frozenset(), frozenset(history_ids)}
    for budget_value in intervention["budgets_visual_tokens"]:
        budget = int(budget_value)
        feasible_distances = {
            coalition: distance
            for coalition, distance in distances.items()
            if len(coalition) * event_cost <= budget
        }
        exact = evaluate_exact_coalition_table(
            feasible_distances,
            event_costs,
            budget=budget,
            similarity_scores=similarity_scores,
            epsilon=epsilon,
        )
        attribution = _attribution_for_budget(
            distances=distances,
            event_ids=history_ids,
            event_cost=event_cost,
            budget=budget,
            epsilon=epsilon,
            sample_counts=config["attribution"]["samples_sweep"],
            seeds=config["attribution"]["seeds"],
        )
        exact_payload = asdict(exact)
        budget_results[str(budget)] = {
            "baselines": exact_payload,
            "attribution": attribution,
        }
        selected_for_generation.update(
            {
                frozenset(exact.recent.coalition),
                frozenset(exact.similarity.coalition),
                frozenset(exact.oracle.coalition),
                frozenset(attribution["exact_selected"]["coalition"]),
            }
        )

    selected_generations: dict[str, Any] = {}
    for coalition in sorted(selected_for_generation, key=lambda value: (len(value), tuple(value))):
        if coalition == frozenset(history_ids):
            result = full_generation
        else:
            messages = build_policy_messages(
                manifest,
                decision_step_id=step_id,
                restored_event_step_ids=sorted(coalition),
                image_loader=image_loader,
            )
            result = runtime.generate(
                messages,
                max_new_tokens=int(config["policy"]["decoding"]["max_new_tokens"]),
                validated_action=validated_action,
            )
        validate_visual_accounting(
            result,
            restored_event_count=len(coalition),
            intervention=intervention,
        )
        selected_generations[coalition_key(coalition)] = {
            "coalition": sorted(coalition),
            "result": result,
        }

    primary_budget = budget_results["1024"]
    primary_baselines = primary_budget["baselines"]
    primary_attribution = primary_budget["attribution"]
    outcome_row = DiagnosticOutcomeRow(
        decision_step_id=step_id,
        memory_sensitive=summary_distance > epsilon,
        budget_1024_oracle_utility=float(primary_baselines["oracle"]["utility"]),
        epsilon=epsilon,
        budget_1024_normalized_oracle_recovery=float(
            primary_baselines["oracle"]["normalized_recovery"]
        ),
        positive_non_recent_exact_gain=has_positive_non_recent_gain(
            primary_attribution["exact_scores"],
            recent_coalition=primary_baselines["recent"]["coalition"],
            epsilon=epsilon,
        ),
    )
    return (
        {
            "decision_step_id": step_id,
            "history_event_step_ids": list(history_ids),
            "validated_action": action_dict(validated_action),
            "canonical_action": canonical_action,
            "canonical_action_token_ids": action_token_ids,
            "full_generation": full_generation,
            "reference_forward": reference_metadata,
            "repeat_forward": repeat_metadata,
            "repeat_kl": repeat_kl,
            "epsilon": epsilon,
            "memory_sensitive": summary_distance > epsilon,
            "similarity_scores": similarity_scores,
            "distance_table": {
                coalition_key(coalition): {
                    "coalition": sorted(coalition),
                    **record,
                }
                for coalition, record in sorted(
                    distance_records.items(),
                    key=lambda item: (len(item[0]), tuple(item[0])),
                )
            },
            "budgets": budget_results,
            "selected_generations": selected_generations,
        },
        outcome_row,
    )


def run_diagnostic(
    *,
    config_path: Path,
    dataset_tar: Path,
    model_dir: Path,
    device: str,
    run_git_commit: str,
    container_image_digest: str,
    repo_root: Path,
    argv: Sequence[str],
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_contract = validate_run_contract(
        config=config,
        config_path=config_path,
        dataset_tar=dataset_tar,
        model_dir=model_dir,
        run_git_commit=run_git_commit,
        repo_root=repo_root,
    )
    if not container_image_digest.startswith("sha256:"):
        raise ValueError("container image digest must be an explicit sha256 digest")

    runtime = QwenPolicyRuntime(
        model_dir=model_dir,
        device=device,
        visual_tokens_per_image=int(config["intervention"]["visual_tokens_per_image"]),
    )
    import tarfile

    with tarfile.open(dataset_tar) as archive:
        manifest, image_loader = load_dataset(archive)
        if manifest.get("dataset_repo") != config["dataset"]["repo"]:
            raise ValueError("dataset manifest repo does not match the frozen config")
        if manifest["trajectory"].get("source_id") != config["dataset"]["source_id"]:
            raise ValueError("dataset source id does not match the frozen config")
        first_state = config["states"][0]
        warmup_messages = build_policy_messages(
            manifest,
            decision_step_id=int(first_state["decision_step_id"]),
            restored_event_step_ids=[],
            image_loader=image_loader,
        )
        runtime.warmup(warmup_messages)
        states = []
        outcome_rows = []
        for state_spec in config["states"]:
            state_result, outcome_row = _run_state(
                runtime=runtime,
                manifest=manifest,
                image_loader=image_loader,
                state_spec=state_spec,
                config=config,
            )
            states.append(state_result)
            outcome_rows.append(outcome_row)

    expected_reference_forwards = int(
        config["coalitions"]["expected_unique_forward_count_excluding_generation"]
    )
    observed_reference_forwards = sum(len(state["distance_table"]) + 1 for state in states)
    if observed_reference_forwards != expected_reference_forwards:
        raise RuntimeError(
            "teacher-forced coalition/reference count changed from the frozen contract"
        )
    outcome = classify_diagnostic_outcome(outcome_rows, config=config)
    if outcome.status == "INVALID":
        raise RuntimeError(f"diagnostic reducer rejected the run: {outcome.reasons}")

    completed_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "0.1.0",
        "diagnostic_name": config["name"],
        "outcome": asdict(outcome),
        "scope": config["scope"],
        "run": {
            **run_contract,
            "argv": list(argv),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "container_image_digest": container_image_digest,
            "started_at": started_at,
            "completed_at": completed_at,
            "wall_time_seconds": time.perf_counter() - wall_start,
            "teacher_forced_reference_and_coalition_forwards": observed_reference_forwards,
            "teacher_forced_repeat_forwards": len(states),
        },
        "runtime": runtime.metadata,
        "gpu": _gpu_metadata(runtime),
        "dataset": {
            "repo": config["dataset"]["repo"],
            "revision": config["dataset"]["revision"],
            "source_id": config["dataset"]["source_id"],
            "shard_sha256": config["dataset"]["shard_sha256"],
        },
        "states": states,
    }
