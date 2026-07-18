"""Read-only horizon-incidence statistics for archived AndroidWorld traces."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


HORIZON_BINS = ("n_le_4", "n_5_8", "n_9_15", "n_ge_16")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def candidate_count_for_decision_index(decision_index: int) -> int:
    """Return the recoverable history count before one zero-based decision."""
    if type(decision_index) is not int or decision_index < 0:
        raise ValueError("decision_index must be a non-negative integer")
    return max(0, decision_index - 1)


def horizon_bin(candidate_count: int) -> str:
    if type(candidate_count) is not int or candidate_count < 0:
        raise ValueError("candidate_count must be a non-negative integer")
    if candidate_count <= 4:
        return "n_le_4"
    if candidate_count <= 8:
        return "n_5_8"
    if candidate_count <= 15:
        return "n_9_15"
    return "n_ge_16"


def read_deterministic_gzip_jsonl(payload: bytes) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("trace shard must contain non-empty bytes")
    try:
        decoded = gzip.decompress(payload).decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("trace shard is not UTF-8 gzip JSONL") from error
    if not decoded.endswith("\n"):
        raise ValueError("trace JSONL must end with a newline")
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(decoded.splitlines(), start=1):
        try:
            record = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid trace JSON at line {line_number}") from error
        if not isinstance(record, Mapping):
            raise ValueError("trace records must be JSON objects")
        records.append(record)
    return tuple(records)


def _steps(record: Mapping[str, Any]) -> Sequence[Any]:
    steps = record.get("steps")
    if steps is None:
        if (
            record.get("run_status") == "exception"
            and record.get("model_step_count") == 0
        ):
            return ()
        raise ValueError(
            "only zero-step exception traces may omit the ordered steps array"
        )
    if isinstance(steps, (str, bytes, bytearray, Mapping)) or not isinstance(
        steps, Sequence
    ):
        raise ValueError("every trace record must contain an ordered steps array")
    return steps


def summarize_horizon_incidence(
    records: Sequence[Mapping[str, Any]],
    *,
    artifact_label: str,
) -> dict[str, Any]:
    if not isinstance(artifact_label, str) or not artifact_label:
        raise ValueError("artifact_label must be non-empty text")
    if not records:
        raise ValueError("horizon incidence requires at least one trace record")
    plan_indices: list[int] = []
    decision_histogram: Counter[str] = Counter()
    candidate_histogram: Counter[int] = Counter()
    episode_rows = []
    nonempty = 0
    reach8 = 0
    reach16 = 0
    total_decisions = 0
    for ordinal, record in enumerate(records):
        plan_index = record.get("plan_index")
        if type(plan_index) is not int or plan_index < 0:
            raise ValueError("trace plan_index must be a non-negative integer")
        plan_indices.append(plan_index)
        steps = _steps(record)
        if record.get("model_step_count") != len(steps):
            raise ValueError("trace model_step_count differs from the steps array")
        counts = tuple(
            candidate_count_for_decision_index(index) for index in range(len(steps))
        )
        for count in counts:
            candidate_histogram[count] += 1
            decision_histogram[horizon_bin(count)] += 1
        maximum = max(counts, default=None)
        if counts:
            nonempty += 1
            reach8 += int(maximum is not None and maximum >= 8)
            reach16 += int(maximum is not None and maximum >= 16)
        total_decisions += len(counts)
        episode_rows.append(
            {
                "record_ordinal": ordinal,
                "plan_index": plan_index,
                "decision_count": len(counts),
                "maximum_candidate_count": maximum,
                "reached_n8": bool(maximum is not None and maximum >= 8),
                "reached_n16": bool(maximum is not None and maximum >= 16),
            }
        )
    if len(set(plan_indices)) != len(plan_indices) or plan_indices != sorted(plan_indices):
        raise ValueError("trace plan_index values must be unique and sorted")
    if sum(decision_histogram.values()) != total_decisions:
        raise RuntimeError("horizon-bin accounting drifted")

    def fraction(count: int) -> float:
        return count / total_decisions if total_decisions else 0.0

    n8_decisions = sum(
        count for candidate, count in candidate_histogram.items() if candidate >= 8
    )
    n16_decisions = sum(
        count for candidate, count in candidate_histogram.items() if candidate >= 16
    )
    result = {
        "artifact_label": artifact_label,
        "record_count": len(records),
        "nonempty_episode_count": nonempty,
        "zero_step_episode_count": len(records) - nonempty,
        "decision_count": total_decisions,
        "candidate_count_definition": "max(0, zero_based_decision_index - 1)",
        "horizon_bin_counts": {
            name: int(decision_histogram[name]) for name in HORIZON_BINS
        },
        "candidate_count_histogram": {
            str(candidate): int(count)
            for candidate, count in sorted(candidate_histogram.items())
        },
        "n_ge_8": {
            "decision_count": n8_decisions,
            "decision_fraction": fraction(n8_decisions),
            "episode_count_among_nonempty": reach8,
            "episode_fraction_among_nonempty": (
                reach8 / nonempty if nonempty else 0.0
            ),
        },
        "n_ge_16": {
            "decision_count": n16_decisions,
            "decision_fraction": fraction(n16_decisions),
            "episode_count_among_nonempty": reach16,
            "episode_fraction_among_nonempty": (
                reach16 / nonempty if nonempty else 0.0
            ),
        },
        "episode_records": episode_rows,
        "interpretation_boundary": (
            "stack_specific_early_stopped_trace_incidence_not_full_benchmark_prevalence"
        ),
    }
    for section in ("n_ge_8", "n_ge_16"):
        for field in (
            "decision_fraction",
            "episode_fraction_among_nonempty",
        ):
            if not math.isfinite(float(result[section][field])):
                raise RuntimeError("horizon incidence produced a non-finite fraction")
    return result


def summarize_gzip_trace_shard(
    payload: bytes,
    *,
    artifact_label: str,
) -> dict[str, Any]:
    summary = summarize_horizon_incidence(
        read_deterministic_gzip_jsonl(payload),
        artifact_label=artifact_label,
    )
    return {
        **summary,
        "trace_shard_size_bytes": len(payload),
        "trace_shard_sha256": sha256_bytes(payload),
    }


__all__ = [
    "HORIZON_BINS",
    "candidate_count_for_decision_index",
    "horizon_bin",
    "read_deterministic_gzip_jsonl",
    "sha256_bytes",
    "summarize_gzip_trace_shard",
    "summarize_horizon_incidence",
]
