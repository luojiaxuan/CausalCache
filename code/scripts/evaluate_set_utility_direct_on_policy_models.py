#!/usr/bin/env python3
"""Run both selected direct-marginal models on the frozen train holdout."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalcache.set_utility_direct_on_policy_evaluator import (
    HYBRID_THRESHOLD,
    METHODS,
    FrozenSelectorAdapter,
    latency_summary,
    load_set_transformer_selector,
    load_structured_deepsets_selector,
    marginal_budget_path,
    project_inference_state,
    recent_budget_path,
    sha256_file,
    validate_expected_rollout,
)
from causalcache.set_utility_formal_input_verification import (
    verify_formal_contextual_cache,
    verify_formal_training_input,
)
from causalcache.set_utility_set_transformer_control import validate_control_config
from causalcache.set_utility_train_heldout_contract import sha256_json
from scripts.run_set_utility_tune_selectors import _pad_entities
from scripts.train_set_utility_structured_marginal import (
    _load_split_manifest,
    _read_signed_json,
    _slice_encoded,
    _trim_padded_candidate_scores,
    _validate_config as validate_structured_config,
)
from scripts.train_set_utility_token_predictor import (
    _TokenCache,
    _batch_stream,
    _cache_covers_input,
    _configure_attention_backend,
    _read_json,
)


STATUS = "COMPLETED_DIRECT_ON_POLICY_UNIFIED_SELECTIONS"
SCHEMA_VERSION = "causalcache.direct_on_policy_unified_selections.v1"


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_inference_states(
    states_path: Path,
    *,
    split_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    requested = set(split_manifest["checkpoint_state_ids"])
    metadata = {
        row["state_id"]: row for row in split_manifest["checkpoint_states"]
    }
    heldout_trajectories = set(split_manifest["heldout_trajectory_ids"])
    result = []
    with states_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            state_id = raw.get("state_id")
            if state_id not in requested:
                continue
            if raw.get("role") != "train":
                raise ValueError("checkpoint inference state escaped the train role")
            state = project_inference_state(raw)
            declared = metadata[state_id]
            if (
                state["trajectory_id"] not in heldout_trajectories
                or state["trajectory_id"] != declared["trajectory_id"]
                or len(state["candidate_event_step_ids"])
                != int(declared["candidate_count"])
            ):
                raise ValueError("checkpoint inference state identity drifted")
            state["history_bin"] = declared["history_bin"]
            result.append(state)
    by_state = {state["state_id"]: state for state in result}
    if len(by_state) != len(result) or set(by_state) != requested:
        raise ValueError("checkpoint inference denominator is incomplete")
    return tuple(sorted(result, key=lambda row: (row["trajectory_id"], row["state_id"])))


def _milliseconds(call: Any, *, torch: Any) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - started) * 1000.0


def _model_path(
    adapter: FrozenSelectorAdapter,
    encoded: Any,
    event_ids: tuple[int, ...],
    *,
    threshold: float | None,
    device: str,
    torch: Any,
) -> dict[str, Any]:
    by_event = {event_id: index for index, event_id in enumerate(event_ids)}
    encoded_event_width = int(encoded.event_mask.shape[1])
    if encoded_event_width < len(event_ids):
        raise ValueError("encoded selector state is narrower than its candidates")

    def score(selected: tuple[int, ...]) -> tuple[float, ...]:
        mask = torch.zeros(
            (1, encoded_event_width), dtype=torch.bool, device=device
        )
        for event_id in selected:
            mask[0, by_event[event_id]] = True
        with torch.autocast(
            device_type=device.split(":", 1)[0], dtype=torch.bfloat16
        ):
            values = adapter.score_encoded_candidates(encoded, mask)[0]
        values = _trim_padded_candidate_scores(values, len(event_ids))
        result = tuple(float(value) for value in values.tolist())
        if len(result) != len(event_ids) + 1:
            raise ValueError("selector score geometry drifted")
        return result

    return marginal_budget_path(
        event_ids,
        score=score,
        stop_semantics=adapter.stop_semantics,
        recent_fallback_threshold=threshold,
    )


def _run_state(
    adapter: FrozenSelectorAdapter,
    state: Mapping[str, Any],
    *,
    cache: _TokenCache,
    event_sources: dict[tuple[str, int], Any],
    event_identities: dict[tuple[str, int], tuple[str, str]],
    device: str,
    torch: Any,
) -> tuple[dict[str, list[int]], dict[str, float]]:
    events = tuple(state["candidate_event_step_ids"])
    trajectory_id = str(state["trajectory_id"])

    def prepare_sources() -> tuple[list[int], tuple[Any, ...]]:
        new_indices = []
        for index, event_id in enumerate(events):
            key = (trajectory_id, event_id)
            identity = (
                state["event_image_keys"][index],
                state["event_text_keys"][index],
            )
            previous = event_identities.get(key)
            if previous is not None and previous != identity:
                raise ValueError("cached event source identity drifted")
            if key not in event_sources:
                event_identities[key] = identity
                new_indices.append(index)
        new_entities: tuple[Any, ...] = ()
        if new_indices:
            new_entities = _pad_entities(
                [cache.visual(state["event_image_keys"][index]) for index in new_indices],
                [cache.text(state["event_text_keys"][index]) for index in new_indices],
                device=device,
                torch=torch,
            )
        query_entities = _pad_entities(
            [cache.visual(state["current_image_key"])],
            [cache.text(state["instruction_text_key"])],
            device=device,
            torch=torch,
        )
        return new_indices, (*new_entities, *query_entities)

    (new_indices, entities), source_io_ms = _milliseconds(
        prepare_sources, torch=torch
    )
    event_encoding_ms = 0.0
    entity_offset = 0
    if new_indices:
        event_visual, event_visual_mask, event_text, event_text_mask = entities[:4]
        entity_offset = 4

        def encode_events() -> Any:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return adapter.encode_event_sources(
                    event_visual_tokens=event_visual.unsqueeze(0),
                    event_visual_mask=event_visual_mask.unsqueeze(0),
                    event_text_tokens=event_text.unsqueeze(0),
                    event_text_mask=event_text_mask.unsqueeze(0),
                    event_mask=torch.ones(
                        (1, len(new_indices)), dtype=torch.bool, device=device
                    ),
                )

        encoded_new, event_encoding_ms = _milliseconds(encode_events, torch=torch)
        for offset, index in enumerate(new_indices):
            event_sources[(trajectory_id, events[index])] = encoded_new[0, offset]
    query_visual, query_visual_mask, query_text, query_text_mask = entities[
        entity_offset : entity_offset + 4
    ]

    def encode_query() -> Any:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return adapter.encode_query_source(
                query_visual_tokens=query_visual,
                query_visual_mask=query_visual_mask,
                query_text_tokens=query_text,
                query_text_mask=query_text_mask,
            )

    query, query_encoding_ms = _milliseconds(encode_query, torch=torch)
    raw_events = torch.stack(
        [event_sources[(trajectory_id, event_id)] for event_id in events]
    ).unsqueeze(0)
    event_mask = torch.ones((1, len(events)), dtype=torch.bool, device=device)
    numeric = torch.tensor(
        [state["event_numeric_features"]], dtype=torch.float32, device=device
    )

    def condition() -> Any:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return adapter.condition_encoded_state(
                query=query,
                event_sources=raw_events,
                event_numeric_features=numeric,
                event_mask=event_mask,
            )

    encoded, conditioning_ms = _milliseconds(condition, torch=torch)
    learned, learned_search_ms = _milliseconds(
        lambda: _model_path(
            adapter,
            encoded,
            events,
            threshold=None,
            device=device,
            torch=torch,
        ),
        torch=torch,
    )
    hybrid, hybrid_search_ms = _milliseconds(
        lambda: _model_path(
            adapter,
            encoded,
            events,
            threshold=HYBRID_THRESHOLD,
            device=device,
            torch=torch,
        ),
        torch=torch,
    )
    shared = source_io_ms + event_encoding_ms + query_encoding_ms + conditioning_ms
    latency = {
        "cold_standalone": shared + learned_search_ms,
        "event_source_encoding": event_encoding_ms,
        "hybrid_cold_standalone": shared + hybrid_search_ms,
        "hybrid_search": hybrid_search_ms,
        "hybrid_warm_selector_from_cached_sources": (
            conditioning_ms + hybrid_search_ms
        ),
        "learned_search": learned_search_ms,
        "query_source_encoding": query_encoding_ms,
        "source_token_io_and_transfer": source_io_ms,
        "state_conditioning": conditioning_ms,
        "warm_selector_from_cached_sources": conditioning_ms + learned_search_ms,
    }
    return {"hybrid": hybrid, "learned": learned}, latency


def _latency_and_hybrid(
    adapter: FrozenSelectorAdapter,
    states: Sequence[Mapping[str, Any]],
    *,
    cache: _TokenCache,
    warmup_iterations: int,
    device: str,
    torch: Any,
) -> dict[str, dict[str, Any]]:
    if warmup_iterations <= 0:
        raise ValueError("warmup iteration count must be positive")
    for _ in range(warmup_iterations):
        _run_state(
            adapter,
            states[0],
            cache=cache,
            event_sources={},
            event_identities={},
            device=device,
            torch=torch,
        )
    event_sources: dict[tuple[str, int], Any] = {}
    event_identities: dict[tuple[str, int], tuple[str, str]] = {}
    result = {}
    for state in states:
        paths, latency = _run_state(
            adapter,
            state,
            cache=cache,
            event_sources=event_sources,
            event_identities=event_identities,
            device=device,
            torch=torch,
        )
        result[str(state["state_id"])] = {"paths": paths, "latency_ms": latency}
    return result


def _native_rollout(
    adapter: FrozenSelectorAdapter,
    states: Sequence[Mapping[str, Any]],
    *,
    cache: _TokenCache,
    batch_size: int,
    device: str,
    torch: Any,
) -> tuple[dict[str, Any], ...]:
    # note (luojiaxuan): The shared training collator expects target-shaped fields;
    # this synthetic zero anchor preserves its batching path without exposing labels.
    label_blind_states = tuple(
        {
            **state,
            "distance_rows": [
                {"coalition_event_step_ids": [], "distance": 0.0}
            ],
        }
        for state in states
    )
    records = []
    adapter.model.eval()
    with torch.inference_mode():
        for selected, batch in _batch_stream(
            label_blind_states,
            batch_size=batch_size,
            cache=cache,
            device=device,
            torch=torch,
            normalization_floor=0.01,
        ):
            model_inputs = dict(batch["model"])
            model_inputs.pop("subset_masks")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                encoded_batch = adapter.model.encode_state_once(**model_inputs)
            for index, state in enumerate(selected):
                encoded = _slice_encoded(encoded_batch, index)
                event_ids = tuple(state["candidate_event_step_ids"])
                path = _model_path(
                    adapter,
                    encoded,
                    event_ids,
                    threshold=None,
                    device=device,
                    torch=torch,
                )
                records.append(
                    {
                        "candidate_event_ids": list(event_ids),
                        "selections": path["selections"],
                        "state_id": state["state_id"],
                        "trace": path["trace"],
                        "trajectory_id": state["trajectory_id"],
                    }
                )
    return tuple(sorted(records, key=lambda row: row["state_id"]))


def _trace_at_base(
    trace: Sequence[Mapping[str, Any]], base: Sequence[int]
) -> Mapping[str, Any] | None:
    target = tuple(base)
    for row in trace:
        if tuple(row.get("base_subset", ())) == target:
            return row
    return None


def _score_delta(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> float | None:
    if left is None or right is None:
        return None
    left_scores = {
        str(row["action"]): float(row["score"])
        for row in left.get("scored_actions", ())
    }
    right_scores = {
        str(row["action"]): float(row["score"])
        for row in right.get("scored_actions", ())
    }
    if not left_scores or set(left_scores) != set(right_scores):
        return None
    return max(abs(left_scores[key] - right_scores[key]) for key in left_scores)


def _mismatch_summary(
    states: Sequence[Mapping[str, Any]],
    *,
    incremental_by_state: Mapping[str, Mapping[str, Any]],
    frozen_by_state: Mapping[str, Mapping[str, Any]],
    recomputed_native_by_state: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    details = []
    state_hashes = []
    native_reproduction_mismatches = []
    for state in states:
        state_id = str(state["state_id"])
        incremental = incremental_by_state[state_id]["paths"]["learned"]
        frozen = frozen_by_state[state_id]
        native = recomputed_native_by_state[state_id]
        state_hash = sha256_json(
            {
                "candidate_event_ids": state["candidate_event_step_ids"],
                "state_id": state_id,
                "trajectory_id": state["trajectory_id"],
            }
        )
        if native["selections"] != frozen["selections"]:
            native_reproduction_mismatches.append(
                {"state_id": state_id, "state_identity_sha256": state_hash}
            )
        if incremental["selections"] == frozen["selections"]:
            continue
        first_budget = next(
            budget
            for budget in range(1, 5)
            if incremental["selections"][str(budget)]
            != frozen["selections"][str(budget)]
        )
        previous = (
            [] if first_budget == 1 else frozen["selections"][str(first_budget - 1)]
        )
        incremental_trace = _trace_at_base(incremental["trace"], previous)
        native_trace = _trace_at_base(native["trace"], previous)
        state_hashes.append(state_hash)
        details.append(
            {
                "first_differing_budget": first_budget,
                "frozen_native_selection": frozen["selections"][str(first_budget)],
                "incremental_selection": incremental["selections"][str(first_budget)],
                "incremental_top_decision_margin": (
                    None
                    if incremental_trace is None
                    else incremental_trace.get("top_decision_margin")
                ),
                "max_absolute_score_delta_vs_recomputed_native": _score_delta(
                    incremental_trace, native_trace
                ),
                "recomputed_native_matches_frozen_at_budget": (
                    native["selections"][str(first_budget)]
                    == frozen["selections"][str(first_budget)]
                ),
                "recomputed_native_top_decision_margin": (
                    None
                    if native_trace is None
                    else native_trace.get("top_decision_margin")
                ),
                "shared_base_subset": list(previous),
                "state_id": state_id,
                "state_identity_sha256": state_hash,
            }
        )
    return {
        "compared_state_count": len(states),
        "mismatch_count": len(details),
        "mismatch_state_identity_set_sha256": sha256_json(sorted(state_hashes)),
        "mismatches": details,
        "recomputed_native_vs_frozen_mismatch_count": len(
            native_reproduction_mismatches
        ),
        "recomputed_native_vs_frozen_mismatches": native_reproduction_mismatches,
        "semantics": "deployment_incremental_cached_sources_vs_frozen_epoch_rollout",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--set-config", type=Path, required=True)
    parser.add_argument("--set-variant", required=True)
    parser.add_argument("--set-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-set-checkpoint-sha256", required=True)
    parser.add_argument("--expected-set-rollout", type=Path)
    parser.add_argument("--deepsets-config", type=Path, required=True)
    parser.add_argument("--deepsets-variant", required=True)
    parser.add_argument("--deepsets-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-deepsets-checkpoint-sha256", required=True)
    parser.add_argument("--expected-deepsets-rollout", type=Path)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("unified selector output already exists")
    if not args.device.startswith("cuda:"):
        raise ValueError("unified selector evaluation requires an explicit CUDA device")
    try:
        import torch
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise RuntimeError("unified selector evaluation requires PyTorch") from error
    if not torch.cuda.is_available():
        raise RuntimeError("unified selector evaluation requires CUDA")

    set_config_path = args.set_config.resolve()
    deepsets_config_path = args.deepsets_config.resolve()
    set_config = _read_json(set_config_path)
    deepsets_config = _read_json(deepsets_config_path)
    set_variant = validate_control_config(set_config, args.set_variant)
    deepsets_variant = validate_structured_config(
        deepsets_config, args.deepsets_variant
    )
    split_path = args.split_manifest.resolve()
    split_manifest = _load_split_manifest(split_path, set_config)
    deepsets_split = _load_split_manifest(split_path, deepsets_config)
    if split_manifest is None or deepsets_split != split_manifest:
        raise ValueError("selector train-heldout manifests differ")
    if set_config["input"] != deepsets_config["input"]:
        raise ValueError("selector formal input bindings differ")
    input_root = args.input_root.resolve()
    cache_root = args.cache_root.resolve()
    input_manifest = verify_formal_training_input(
        input_root,
        expected_content_sha256=set_config["input"][
            "training_input_content_sha256"
        ],
    )
    cache_manifest = verify_formal_contextual_cache(
        cache_root,
        expected_content_sha256=set_config["input"][
            "contextual_cache_content_sha256"
        ],
    )
    if not _cache_covers_input(input_manifest, cache_manifest):
        raise ValueError("contextual cache does not cover the formal input")
    states = _read_inference_states(
        input_root / input_manifest["states_jsonl"],
        split_manifest=split_manifest,
    )

    torch.set_float32_matmul_precision("high")
    attention_backend = _configure_attention_backend(
        torch, set_config["training"]
    )
    if deepsets_config["training"].get("attention_backend") != set_config[
        "training"
    ].get("attention_backend"):
        raise ValueError("selector attention backend profiles differ")
    torch.manual_seed(int(set_config["training"]["seed"]))
    torch.cuda.manual_seed_all(int(set_config["training"]["seed"]))
    set_checkpoint = args.set_checkpoint.resolve()
    deepsets_checkpoint = args.deepsets_checkpoint.resolve()
    adapters = {
        "set_transformer": load_set_transformer_selector(
            variant=set_variant,
            checkpoint_path=set_checkpoint,
            expected_checkpoint_sha256=args.expected_set_checkpoint_sha256,
            device=args.device,
            load_file=load_file,
        ),
        "structured_deepsets": load_structured_deepsets_selector(
            variant=deepsets_variant,
            checkpoint_path=deepsets_checkpoint,
            expected_checkpoint_sha256=args.expected_deepsets_checkpoint_sha256,
            device=args.device,
            load_file=load_file,
        ),
    }
    cache = _TokenCache(cache_root, cache_manifest, device="cpu", mode="lazy_cpu")
    native = {
        name: _native_rollout(
            adapter,
            states,
            cache=cache,
            batch_size=int(
                set_variant["heldout_batch_size"]
                if name == "set_transformer"
                else deepsets_variant["heldout_batch_size"]
            ),
            device=args.device,
            torch=torch,
        )
        for name, adapter in adapters.items()
    }
    expected_paths = {
        "set_transformer": args.expected_set_rollout,
        "structured_deepsets": args.expected_deepsets_rollout,
    }
    expected_rollout_hashes = {}
    frozen_by_model = {}
    native_reproduction_mismatch_ids = {}
    for name, path in expected_paths.items():
        if path is None:
            expected_rollout_hashes[name] = None
            frozen_by_model[name] = {
                row["state_id"]: row for row in native[name]
            }
            native_reproduction_mismatch_ids[name] = ()
            continue
        expected = _read_signed_json(path.resolve())
        native_for_validation = [
            {
                "candidate_event_ids": row["candidate_event_ids"],
                "methods": {name: row["selections"]},
                "model_name": name,
                "state_id": row["state_id"],
            }
            for row in native[name]
        ]
        native_reproduction_mismatch_ids[name] = validate_expected_rollout(
            native_for_validation,
            expected,
            expected_checkpoint_sha256=(
                args.expected_set_checkpoint_sha256
                if name == "set_transformer"
                else args.expected_deepsets_checkpoint_sha256
            ),
            require_exact_selections=False,
        )
        expected_rollout_hashes[name] = expected["content_sha256"]
        frozen_by_model[name] = {
            row["state_id"]: row for row in expected["records"]
        }

    measured = {
        name: _latency_and_hybrid(
            adapter,
            states,
            cache=cache,
            warmup_iterations=args.warmup_iterations,
            device=args.device,
            torch=torch,
        )
        for name, adapter in adapters.items()
    }
    native_by_state = {
        name: {row["state_id"]: row for row in rows}
        for name, rows in native.items()
    }
    diagnostics = {
        name: _mismatch_summary(
            states,
            incremental_by_state=measured[name],
            frozen_by_state=frozen_by_model[name],
            recomputed_native_by_state=native_by_state[name],
        )
        for name in adapters
    }
    for name, mismatches in native_reproduction_mismatch_ids.items():
        observed = tuple(
            row["state_id"]
            for row in diagnostics[name][
                "recomputed_native_vs_frozen_mismatches"
            ]
        )
        if set(mismatches) != set(observed):
            raise RuntimeError("native reproduction diagnostics disagree")
    records = []
    for state in states:
        state_id = str(state["state_id"])
        records.append(
            {
                "candidate_event_ids": list(state["candidate_event_step_ids"]),
                "history_bin": state["history_bin"],
                "latency_ms": {
                    name: measured[name][state_id]["latency_ms"]
                    for name in adapters
                },
                "methods": {
                    "recent": recent_budget_path(state["candidate_event_step_ids"]),
                    "set_transformer": measured["set_transformer"][state_id][
                        "paths"
                    ]["learned"]["selections"],
                    "set_transformer_hybrid": measured["set_transformer"][
                        state_id
                    ]["paths"]["hybrid"]["selections"],
                    "structured_deepsets": measured["structured_deepsets"][
                        state_id
                    ]["paths"]["learned"]["selections"],
                    "structured_deepsets_hybrid": measured[
                        "structured_deepsets"
                    ][state_id]["paths"]["hybrid"]["selections"],
                },
                "state_id": state_id,
                "trajectory_id": state["trajectory_id"],
            }
        )
    unsigned = {
        "attention_backend": attention_backend,
        "bindings": {
            "cache_content_sha256": cache_manifest["content_sha256"],
            "heldout_manifest_content_sha256": split_manifest["content_sha256"],
            "heldout_manifest_file_sha256": sha256_file(split_path),
            "input_content_sha256": input_manifest["content_sha256"],
            "input_states_sha256": input_manifest["states_sha256"],
            "set_transformer_config_sha256": sha256_file(set_config_path),
            "source_manifest_file_sha256": set_config["input"][
                "source_manifest_file_sha256"
            ],
            "structured_deepsets_config_sha256": sha256_file(
                deepsets_config_path
            ),
        },
        "checkpoints": {
            "set_transformer": {
                "byte_count": set_checkpoint.stat().st_size,
                "model_family": adapters["set_transformer"].model_family,
                "native_rollout_content_sha256": expected_rollout_hashes[
                    "set_transformer"
                ],
                "sha256": args.expected_set_checkpoint_sha256,
                "stop_semantics": adapters["set_transformer"].stop_semantics,
                "variant": args.set_variant,
            },
            "structured_deepsets": {
                "byte_count": deepsets_checkpoint.stat().st_size,
                "model_family": adapters["structured_deepsets"].model_family,
                "native_rollout_content_sha256": expected_rollout_hashes[
                    "structured_deepsets"
                ],
                "sha256": args.expected_deepsets_checkpoint_sha256,
                "stop_semantics": adapters["structured_deepsets"].stop_semantics,
                "variant": args.deepsets_variant,
            },
        },
        "denominator": {
            "history_bin_counts": split_manifest["census"][
                "checkpoint_history_bin_counts"
            ],
            "state_count": len(states),
            "trajectory_count": len({state["trajectory_id"] for state in states}),
        },
        "incremental_vs_frozen_native_diagnostics": diagnostics,
        "device": args.device,
        "hybrid_contract": {
            "minimum_learned_advantage_over_recent": HYBRID_THRESHOLD,
            "selection": (
                "stop_only_if_stop_beats_best_event_by_threshold;"
                "use_non_recent_best_only_if_it_beats_recent_and_stop_by_threshold;"
                "otherwise_use_recent"
            ),
        },
        "latency_contract": {
            "event_sources_cached_by_trajectory_and_event": True,
            "measurement": "warm_single_state_online_inference",
            "warm_selector_from_cached_sources": "state_conditioning+learned_search",
            "cold_standalone": (
                "source_token_io_and_transfer+event_source_encoding+"
                "query_source_encoding+state_conditioning+learned_search"
            ),
            "warmup_iterations": args.warmup_iterations,
        },
        "latency_summary_ms": latency_summary(
            records, ("set_transformer", "structured_deepsets")
        ),
        "method_semantics": {
            "recent": "deterministic_most_recent_at_most_B",
            "set_transformer": "deployment_incremental_cached_source_encoding",
            "set_transformer_hybrid": (
                "deployment_incremental_cached_source_encoding_with_recent_fallback"
            ),
            "structured_deepsets": (
                "deployment_incremental_cached_source_encoding"
            ),
            "structured_deepsets_hybrid": (
                "deployment_incremental_cached_source_encoding_with_recent_fallback"
            ),
        },
        "methods": list(METHODS),
        "records": records,
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "truth_labels_accessed": False,
    }
    result = {**unsigned, "content_sha256": sha256_json(unsigned)}
    _write_atomic(args.output.resolve(), result)
    print(json.dumps({key: result[key] for key in ("content_sha256", "status")}))


if __name__ == "__main__":
    main()
