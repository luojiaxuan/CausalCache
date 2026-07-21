#!/usr/bin/env python3
"""Run one frozen direct conditional-marginal checkpoint over tune states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_inference import (
    BUDGETS,
    canonical_json_bytes,
    recent_budget_selections,
)
from causalcache.set_utility_token_models import (
    TokenConditionalMarginalPredictor,
    TokenUtilityModelConfig,
)
from scripts.run_set_utility_tune_selectors import (
    _TokenCache,
    _binding_covers_input,
    _milliseconds,
    _pad_entities,
    _read_json,
    _sha256_file,
    _write_atomic,
)


def _direct_budget_path(
    model: Any,
    encoded: Any,
    event_ids: tuple[int, ...],
    *,
    device: str,
    torch: Any,
) -> tuple[dict[str, list[int]], dict[str, float], int, list[dict[str, Any]]]:
    selected: list[int] = []
    cumulative = 0.0
    stopped = False
    selections = {}
    utilities = {}
    trace = []
    score_count = 0
    event_index = {event_id: index for index, event_id in enumerate(event_ids)}
    for budget in BUDGETS:
        if not stopped and len(selected) < min(budget, len(event_ids)):
            selected_mask = torch.zeros(
                (1, len(event_ids)), dtype=torch.bool, device=device
            )
            for event_id in selected:
                selected_mask[0, event_index[event_id]] = True
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores = model.score_encoded_candidates(encoded, selected_mask)[0]
            values = [float(value) for value in scores.tolist()]
            if any(not math.isfinite(value) for value in values):
                raise RuntimeError("direct selector emitted a non-finite marginal")
            valid_actions = [(0, 0.0)] + [
                (index + 1, values[index + 1])
                for index, event_id in enumerate(event_ids)
                if event_id not in selected
            ]
            score_count += len(valid_actions)
            best_action, best_gain = min(
                valid_actions,
                key=lambda item: (-item[1], item[0]),
            )
            ranked = sorted(
                (
                    {
                        "action": "STOP" if action == 0 else event_ids[action - 1],
                        "predicted_marginal": gain,
                    }
                    for action, gain in valid_actions
                ),
                key=lambda item: (
                    -item["predicted_marginal"],
                    -1 if item["action"] == "STOP" else item["action"],
                ),
            )
            trace.append(
                {
                    "base_subset": sorted(selected),
                    "ranked_actions": ranked[: min(8, len(ranked))],
                }
            )
            if best_action == 0:
                stopped = True
            else:
                selected.append(event_ids[best_action - 1])
                selected.sort()
                cumulative += best_gain
        selections[str(budget)] = list(selected)
        utilities[str(budget)] = cumulative
    return selections, utilities, score_count, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--state-id-file", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("direct tune selector output already exists")
    try:
        import torch
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise RuntimeError("direct tune selector requires PyTorch") from error
    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise RuntimeError("direct tune selector requires an explicit CUDA device")

    input_root = args.input_root.resolve()
    input_manifest = _read_json(input_root / "manifest.json")
    cache_root = args.cache_root.resolve()
    cache_manifest = _read_json(cache_root / "manifest.json")
    config_path = args.config.resolve()
    config = _read_json(config_path)
    variant = config["variants"].get(args.variant)
    if not isinstance(variant, dict):
        raise ValueError("unknown direct tune-selector variant")
    if config.get("deployment", {}).get("selection") != (
        "iterative_conditional_marginal_with_explicit_stop"
    ):
        raise ValueError("direct selector deployment contract drifted")
    summary = _read_json(args.training_summary)
    if (
        input_manifest.get("evaluation_labels_included") is not False
        or cache_manifest.get("evaluation_labels_included") is not False
        or not _binding_covers_input(
            input_manifest, cache_manifest.get("input_content_sha256")
        )
        or summary.get("evaluation_records_loaded") is not False
        or summary.get("tune_labels_entered_training_or_selection") is not False
        or not _binding_covers_input(
            input_manifest, summary.get("input_content_sha256")
        )
        or summary.get("cache_content_sha256") != cache_manifest.get("content_sha256")
        or summary.get("config_sha256") != _sha256_file(config_path)
        or summary.get("variant") != args.variant
        or summary.get("best_checkpoint", {}).get("sha256")
        != _sha256_file(args.checkpoint)
    ):
        raise ValueError("direct tune selector input/checkpoint firewall drifted")
    states = tuple(
        row
        for line in (input_root / input_manifest["states_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line
        for row in (json.loads(line),)
        if row["role"] == "tune"
    )
    if args.state_id_file is not None:
        requested = tuple(
            line
            for line in args.state_id_file.read_text(encoding="utf-8").splitlines()
            if line
        )
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("state-id filter must be non-empty and unique")
        requested_set = set(requested)
        if not requested_set.issubset({state["state_id"] for state in states}):
            raise ValueError("state-id filter escapes the tune inventory")
        states = tuple(state for state in states if state["state_id"] in requested_set)
    if not states:
        raise ValueError("direct selector received no tune states")

    seed = int(summary["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    model = TokenConditionalMarginalPredictor(
        TokenUtilityModelConfig(**variant["model"])
    ).to(args.device)
    model.load_state_dict(
        load_file(str(args.checkpoint), device=args.device), strict=True
    )
    model.eval()
    cache = _TokenCache(cache_root, cache_manifest, device="cpu")
    event_sources = {}
    records = []
    with torch.inference_mode():
        for state in sorted(
            states, key=lambda row: (row["trajectory_id"], row["state_id"])
        ):
            events = tuple(state["candidate_event_step_ids"])
            new_indices = [
                index
                for index, event_id in enumerate(events)
                if (state["trajectory_id"], event_id) not in event_sources
            ]
            event_encoding_ms = 0.0
            if new_indices:
                visual, visual_mask, text, text_mask = _pad_entities(
                    [
                        cache.visual(state["event_image_keys"][index])
                        for index in new_indices
                    ],
                    [
                        cache.text(state["event_text_keys"][index])
                        for index in new_indices
                    ],
                    device=args.device,
                    torch=torch,
                )

                def encode_new_events() -> Any:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        return model.encoder.encode_event_sources(
                            event_visual_tokens=visual.unsqueeze(0),
                            event_visual_mask=visual_mask.unsqueeze(0),
                            event_text_tokens=text.unsqueeze(0),
                            event_text_mask=text_mask.unsqueeze(0),
                            event_mask=torch.ones(
                                (1, len(new_indices)),
                                dtype=torch.bool,
                                device=args.device,
                            ),
                        )

                encoded_new, event_encoding_ms = _milliseconds(
                    encode_new_events, torch=torch
                )
                for offset, index in enumerate(new_indices):
                    event_sources[(state["trajectory_id"], events[index])] = (
                        encoded_new[0, offset]
                    )

            query_visual, query_visual_mask, query_text, query_text_mask = (
                _pad_entities(
                    [cache.visual(state["current_image_key"])],
                    [cache.text(state["instruction_text_key"])],
                    device=args.device,
                    torch=torch,
                )
            )

            def encode_query() -> Any:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return model.encoder.encode_query_source(
                        query_visual_tokens=query_visual,
                        query_visual_mask=query_visual_mask,
                        query_text_tokens=query_text,
                        query_text_mask=query_text_mask,
                    )

            query, query_encoding_ms = _milliseconds(encode_query, torch=torch)
            raw_events = torch.stack(
                [
                    event_sources[(state["trajectory_id"], event_id)]
                    for event_id in events
                ]
            ).unsqueeze(0)
            event_mask = torch.ones(
                (1, len(events)), dtype=torch.bool, device=args.device
            )
            numeric = torch.tensor(
                [state["event_numeric_features"]],
                dtype=torch.float32,
                device=args.device,
            )

            def condition() -> Any:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return model.condition_encoded_state(
                        query=query,
                        event_sources=raw_events,
                        event_numeric_features=numeric,
                        event_mask=event_mask,
                    )

            encoded, conditioning_ms = _milliseconds(condition, torch=torch)
            torch.cuda.synchronize()
            started = time.perf_counter()
            learned, utilities, score_count, trace = _direct_budget_path(
                model,
                encoded,
                events,
                device=args.device,
                torch=torch,
            )
            torch.cuda.synchronize()
            search_ms = (time.perf_counter() - started) * 1000.0
            records.append(
                {
                    "candidate_event_ids": list(events),
                    "direct_steps": trace,
                    "latency_ms": {
                        "conditioning": conditioning_ms,
                        "event_source_encoding": event_encoding_ms,
                        "query_source_encoding": query_encoding_ms,
                        "search": search_ms,
                    },
                    "learned": learned,
                    "logical_shard": state["logical_shard"],
                    "predicted_utilities": utilities,
                    "recent": recent_budget_selections(events),
                    "search": {
                        "method": "iterative_conditional_marginal_with_explicit_stop"
                    },
                    "state_id": state["state_id"],
                    "subset_score_count": score_count,
                    "trajectory_id": state["trajectory_id"],
                }
            )
    result = {
        "cache_content_sha256": cache_manifest["content_sha256"],
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "config_sha256": _sha256_file(config_path),
        "input_content_sha256": input_manifest["content_sha256"],
        "records": sorted(records, key=lambda row: row["state_id"]),
        "role": "tune",
        "schema_version": "causalcache.direct_marginal_tune_selections.v1",
        "status": "COMPLETED_SET_UTILITY_TUNE_SELECTIONS",
        "variant": args.variant,
    }
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_atomic(args.output, result)
    print(
        json.dumps(
            {
                "content_sha256": result["content_sha256"],
                "state_count": len(records),
                "status": result["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
