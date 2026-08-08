"""Exact-budget FindingDory memory-restoration oracle utilities."""

from __future__ import annotations

import ast
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


SUMMARY_CONTENT_KEYS = (
    "objects",
    "receptacles",
    "rooms",
    "interactions",
    "fine_attributes",
    "state_changes",
)


def parse_answer_groups(raw: str) -> tuple[tuple[int, ...], ...]:
    value = ast.literal_eval(raw)
    if not isinstance(value, list) or not value:
        raise ValueError("FindingDory answer must be a non-empty list")
    groups: list[tuple[int, ...]] = []
    for group in value:
        if not isinstance(group, list) or not group:
            raise ValueError("each FindingDory answer group must be non-empty")
        frames = tuple(int(frame) for frame in group)
        if any(frame < 0 for frame in frames):
            raise ValueError("FindingDory frame ids must be non-negative")
        groups.append(frames)
    return tuple(groups)


def _contiguous_runs(frames: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    ordered = sorted(set(int(frame) for frame in frames))
    if not ordered:
        return ()
    runs: list[list[int]] = [[ordered[0]]]
    for frame in ordered[1:]:
        if frame == runs[-1][-1] + 1:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return tuple(tuple(run) for run in runs)


def oracle_evidence_frame(
    answer_groups: Sequence[Sequence[int]],
    *,
    maximum_history_frame: int,
) -> int:
    """Choose the deterministic center of the longest valid historical run."""
    valid = [
        int(frame)
        for group in answer_groups
        for frame in group
        if 0 <= int(frame) <= maximum_history_frame
    ]
    runs = _contiguous_runs(valid)
    if not runs:
        raise ValueError("task has no valid goal frame in the selectable history")
    # note (luojiaxuan): Longest-run then earliest-run tie breaking avoids model-output
    # selection and favors a viewpoint away from ambiguous interval boundaries.
    run = min(runs, key=lambda item: (-len(item), item[0]))
    return run[(len(run) - 1) // 2]


def exact_budget_selections(
    answer_groups: Sequence[Sequence[int]],
    *,
    budget: int,
    current_frame: int,
) -> dict[str, tuple[int, ...]]:
    if budget <= 0 or budget >= current_frame:
        raise ValueError("budget must fit strictly inside the historical frame range")
    history = tuple(range(current_frame))
    recent = history[-budget:]
    evidence = oracle_evidence_frame(
        answer_groups,
        maximum_history_frame=current_frame - 1,
    )
    oracle = [evidence]
    if len(oracle) < budget:
        for frame in reversed(history):
            if frame not in oracle:
                oracle.append(frame)
            if len(oracle) == budget:
                break
    result = {
        "recent": tuple(sorted(recent)),
        "oracle": tuple(sorted(oracle)),
    }
    if any(len(selection) != budget for selection in result.values()):
        raise RuntimeError("selection violated exact-B")
    return result


def parse_predicted_frame(raw: str) -> int | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, Mapping):
            continue
        candidates = value.get("frame_indices", value.get("frame_id"))
        if isinstance(candidates, list) and candidates:
            candidates = candidates[0]
        if isinstance(candidates, bool):
            return None
        if isinstance(candidates, int):
            return candidates
        if isinstance(candidates, str) and re.fullmatch(r"\s*\d+\s*", candidates):
            return int(candidates)
    return None


def _clean_summary_value(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\d+(?::\d+)?", "", value).strip()
    if isinstance(value, list):
        return [_clean_summary_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _clean_summary_value(item)
            for key, item in value.items()
            if not re.search(r"frame|time|index", str(key), flags=re.IGNORECASE)
        }
    if value is None or isinstance(value, bool):
        return value
    return _clean_summary_value(str(value))


def normalize_content_summary(raw: str) -> dict[str, Any]:
    """Remove the legacy overlaid frame/time namespace from VLM captions."""
    decoder = json.JSONDecoder()
    parsed = None
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            parsed = value
            break
    if parsed is None:
        result = {"unstructured_caption": _clean_summary_value(raw)}
    else:
        result = {
            key: _clean_summary_value(parsed[key])
            for key in SUMMARY_CONTENT_KEYS
            if key in parsed
        }
        if not result:
            result = {"unstructured_caption": _clean_summary_value(raw)}
    if re.search(r"\d", json.dumps(result, ensure_ascii=False)):
        raise RuntimeError("normalized content summary retained a numeric namespace")
    return result


def frame_is_valid(frame: int | None, answer_groups: Sequence[Sequence[int]]) -> bool:
    return frame is not None and any(frame in group for group in answer_groups)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(float(value) for value in values)
    location = probability * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_bootstrap_ci(
    episode_deltas: Mapping[str, Sequence[float]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    episodes = sorted(episode_deltas)
    if not episodes:
        raise ValueError("bootstrap requires episodes")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        values = [
            value
            for _ in episodes
            for value in episode_deltas[rng.choice(episodes)]
        ]
        draws.append(math.fsum(values) / len(values))
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


def _mcnemar_exact_p(oracle_only: int, recent_only: int) -> float:
    discordant = oracle_only + recent_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(oracle_only, recent_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def reduce_paired_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
    minimum_oracle_success: float,
    minimum_delta: float,
    minimum_coverage: float,
) -> dict[str, Any]:
    by_pair: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["episode_id"]), str(row["task_id"]), int(row["budget"]))
        arm = str(row["arm"])
        if arm in by_pair[key]:
            raise ValueError(f"duplicate arm for pair: {key} {arm}")
        by_pair[key][arm] = row
    budgets = sorted({key[2] for key in by_pair})
    report: dict[str, Any] = {}
    for budget in budgets:
        pairs = [
            (key, arms)
            for key, arms in by_pair.items()
            if key[2] == budget
        ]
        if any(set(arms) != {"recent", "oracle"} for _, arms in pairs):
            raise ValueError(f"budget {budget} has incomplete paired arms")
        recent = [bool(arms["recent"]["success"]) for _, arms in pairs]
        oracle = [bool(arms["oracle"]["success"]) for _, arms in pairs]
        deltas = [float(right) - float(left) for left, right in zip(recent, oracle, strict=True)]
        episode_deltas: dict[str, list[float]] = defaultdict(list)
        for (episode_id, _, _), delta in zip((key for key, _ in pairs), deltas, strict=True):
            episode_deltas[episode_id].append(delta)
        ci_low, ci_high = _cluster_bootstrap_ci(
            episode_deltas,
            samples=bootstrap_samples,
            seed=seed + budget,
        )
        oracle_only = sum(right and not left for left, right in zip(recent, oracle, strict=True))
        recent_only = sum(left and not right for left, right in zip(recent, oracle, strict=True))
        coverage = sum(
            arms["recent"]["selected_frames"] != arms["oracle"]["selected_frames"]
            for _, arms in pairs
        ) / len(pairs)
        oracle_success = sum(oracle) / len(oracle)
        delta = math.fsum(deltas) / len(deltas)
        gate = (
            oracle_success >= minimum_oracle_success
            and delta >= minimum_delta
            and ci_low > 0.0
            and coverage >= minimum_coverage
        )
        report[str(budget)] = {
            "paired_tasks": len(pairs),
            "episodes": len(episode_deltas),
            "recent_hl_sr": sum(recent) / len(recent),
            "oracle_hl_sr": oracle_success,
            "oracle_minus_recent": delta,
            "cluster_bootstrap_ci95": [ci_low, ci_high],
            "oracle_only_successes": oracle_only,
            "recent_only_successes": recent_only,
            "mcnemar_exact_p": _mcnemar_exact_p(oracle_only, recent_only),
            "selection_difference_coverage": coverage,
            "gate_pass": gate,
        }
    return {"budgets": report, "any_budget_gate_pass": any(item["gate_pass"] for item in report.values())}


__all__ = [
    "exact_budget_selections",
    "frame_is_valid",
    "normalize_content_summary",
    "oracle_evidence_frame",
    "parse_answer_groups",
    "parse_predicted_frame",
    "reduce_paired_results",
]
