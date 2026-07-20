#!/usr/bin/env python3
"""Run one frozen rich-token model over the label-blind held-out snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_inference import (
    canonical_json_bytes,
    conditional_greedy_budget_path,
    ocr_rgb_budget_selections,
    random_budget_selections,
    recent_budget_selections,
)
from causalcache.set_utility_token_models import (
    TokenSetUtilityPredictor,
    TokenUtilityModelConfig,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class _HeldoutTokenCache:
    def __init__(self, root: Path, manifest: dict[str, Any], *, device: str) -> None:
        from safetensors import safe_open

        self.tensors: dict[str, Any] = {}
        grouped: dict[Path, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for cache_key, record in manifest["tensor_inventory"].items():
            grouped[root / record["partition"] / record["shard"]].append(
                (cache_key, record)
            )
        for path in sorted(grouped):
            with safe_open(str(path), framework="pt", device=device) as handle:
                for cache_key, record in grouped[path]:
                    tensor = handle.get_tensor(record["tensor"])
                    if (
                        list(tensor.shape) != record["shape"]
                        or str(tensor.dtype) != record["dtype"]
                    ):
                        raise ValueError("held-out cached token metadata drifted")
                    self.tensors[cache_key] = tensor
        if len(self.tensors) != len(manifest["tensor_inventory"]):
            raise RuntimeError("held-out token cache preload is incomplete")

    def visual(self, key: str) -> Any:
        return self.tensors[f"visual:{key}"]

    def text(self, key: str) -> Any:
        return self.tensors[f"text:{key}"]


def _pad_entities(visual: list[Any], text: list[Any], *, torch: Any) -> tuple[Any, ...]:
    if not visual or len(visual) != len(text):
        raise ValueError("entity source tensors must be non-empty and aligned")
    visual_length = max(row.shape[0] for row in visual)
    text_length = max(row.shape[0] for row in text)
    hidden = visual[0].shape[1]
    device = visual[0].device
    visual_batch = torch.zeros(
        (len(visual), visual_length, hidden),
        dtype=visual[0].dtype,
        device=device,
    )
    visual_mask = torch.zeros(
        (len(visual), visual_length), dtype=torch.bool, device=device
    )
    text_batch = torch.zeros(
        (len(text), text_length, hidden), dtype=text[0].dtype, device=device
    )
    text_mask = torch.zeros(
        (len(text), text_length), dtype=torch.bool, device=device
    )
    for index, row in enumerate(visual):
        visual_batch[index, : row.shape[0]] = row
        visual_mask[index, : row.shape[0]] = True
    for index, row in enumerate(text):
        text_batch[index, : row.shape[0]] = row
        text_mask[index, : row.shape[0]] = True
    return visual_batch, visual_mask, text_batch, text_mask


def _milliseconds(call: Any, *, torch: Any) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--heldout-config", type=Path, required=True)
    parser.add_argument("--model-name", choices=("deepsets", "set_transformer"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("held-out model selection output already exists")
    try:
        import torch
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise RuntimeError("held-out selector inference requires torch and safetensors") from error
    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise RuntimeError("held-out selector inference requires an explicit CUDA device")

    repository_root = args.repository_root.resolve()
    heldout_path = args.heldout_config.resolve()
    heldout = _read_json(heldout_path)
    input_root = args.input_root.resolve()
    input_manifest_path = input_root / "manifest.json"
    input_manifest = _read_json(input_manifest_path)
    allowed_input_statuses = {
        "COMPLETED_SET_UTILITY_HELDOUT_FEATURE_SNAPSHOT",
        "COMPLETED_SET_UTILITY_HELDOUT_FEATURE_PARTITION",
    }
    if (
        input_manifest.get("status") not in allowed_input_statuses
        or input_manifest.get("evaluation_labels_loaded") is not False
        or input_manifest.get("label_file_read_count") != 0
        or int(input_manifest.get("state_count", 0)) <= 0
    ):
        raise ValueError("held-out feature snapshot violates the label firewall")
    inventory_path = repository_root / heldout["state_inventory"]["path"]
    if _sha256_file(inventory_path) != heldout["state_inventory"]["sha256"]:
        raise ValueError("held-out state inventory hash drifted")
    model_config_path = repository_root / heldout["representation"]["model_config"]
    if _sha256_file(model_config_path) != heldout["representation"]["model_config_sha256"]:
        raise ValueError("held-out model config hash drifted")
    model_config = _read_json(model_config_path)
    candidate = heldout["frozen_candidates"]["models"][args.model_name]
    if _sha256_file(args.checkpoint) != candidate["sha256"]:
        raise ValueError("held-out checkpoint hash drifted")
    variant = model_config["variants"][candidate["variant"]]

    cache_root = args.cache_root.resolve()
    cache_manifest = _read_json(cache_root / "manifest.json")
    if (
        cache_manifest.get("input_content_sha256") != input_manifest["content_sha256"]
        or cache_manifest.get("evaluation_labels_included") is not False
    ):
        raise ValueError("held-out token cache identity or label firewall drifted")
    states = _read_jsonl(input_root / input_manifest["states_jsonl"])
    if (
        len(states) != input_manifest["state_count"]
        or any(row["role"] != "evaluation" for row in states)
    ):
        raise ValueError("held-out selector state inventory drifted")

    torch.manual_seed(20260720)
    torch.cuda.manual_seed_all(20260720)
    torch.set_float32_matmul_precision("high")
    model = TokenSetUtilityPredictor(
        TokenUtilityModelConfig(**variant["model"])
    ).to(args.device)
    model.load_state_dict(load_file(str(args.checkpoint), device=args.device), strict=True)
    model.eval()
    cache = _HeldoutTokenCache(cache_root, cache_manifest, device=args.device)
    event_sources: dict[tuple[str, int], Any] = {}
    records = []
    random_seed = int(heldout["baselines"]["random"]["seed"])

    with torch.inference_mode():
        for state in sorted(
            states,
            key=lambda row: (row["trajectory_id"], row["decision_step_id"]),
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
                    [cache.visual(state["event_image_keys"][index]) for index in new_indices],
                    [cache.text(state["event_text_keys"][index]) for index in new_indices],
                    torch=torch,
                )

                def encode_new_events() -> Any:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        return model.encode_event_sources(
                            event_visual_tokens=visual.unsqueeze(0),
                            event_visual_mask=visual_mask.unsqueeze(0),
                            event_text_tokens=text.unsqueeze(0),
                            event_text_mask=text_mask.unsqueeze(0),
                            event_mask=torch.ones(
                                (1, len(new_indices)), dtype=torch.bool, device=args.device
                            ),
                        )

                encoded_new, event_encoding_ms = _milliseconds(
                    encode_new_events, torch=torch
                )
                for offset, index in enumerate(new_indices):
                    event_sources[(state["trajectory_id"], events[index])] = encoded_new[0, offset]

            query_visual, query_visual_mask, query_text, query_text_mask = _pad_entities(
                [cache.visual(state["current_image_key"])],
                [cache.text(state["instruction_text_key"])],
                torch=torch,
            )

            def encode_query() -> Any:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return model.encode_query_source(
                        query_visual_tokens=query_visual,
                        query_visual_mask=query_visual_mask,
                        query_text_tokens=query_text,
                        query_text_mask=query_text_mask,
                    )

            query, query_encoding_ms = _milliseconds(encode_query, torch=torch)
            raw_events = torch.stack(
                [event_sources[(state["trajectory_id"], event_id)] for event_id in events]
            ).unsqueeze(0)
            event_mask = torch.ones((1, len(events)), dtype=torch.bool, device=args.device)
            numeric = torch.tensor(
                [state["event_numeric_features"]], dtype=torch.float32, device=args.device
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
            event_index = {event_id: index for index, event_id in enumerate(events)}

            def score_batch(subsets: tuple[tuple[int, ...], ...]) -> tuple[float, ...]:
                values = []
                chunk_size = int(heldout["selection"]["subset_score_chunk_size"])
                for start in range(0, len(subsets), chunk_size):
                    chunk = subsets[start : start + chunk_size]
                    masks = torch.zeros(
                        (1, len(chunk), len(events)), dtype=torch.bool, device=args.device
                    )
                    for subset_index, subset in enumerate(chunk):
                        for event_id in subset:
                            masks[0, subset_index, event_index[event_id]] = True
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predicted = model.score_encoded_subsets(encoded, masks)
                    values.extend(float(value) for value in predicted[0].tolist())
                return tuple(values)

            torch.cuda.synchronize()
            search_started = time.perf_counter()
            learned, predicted, score_count = conditional_greedy_budget_path(
                events, score_batch=score_batch
            )
            torch.cuda.synchronize()
            search_ms = (time.perf_counter() - search_started) * 1000.0
            if any(not math.isfinite(value) for value in predicted.values()):
                raise RuntimeError("held-out model emitted non-finite utility")
            methods = {
                args.model_name: learned,
                "ocr_rgb": ocr_rgb_budget_selections(events, state["ocr_rgb_scores"]),
                "random": random_budget_selections(
                    events, state_id=state["state_id"], seed=random_seed
                ),
                "recent": recent_budget_selections(events),
            }
            records.append(
                {
                    "candidate_event_ids": list(events),
                    "latency_ms": {
                        "encoded_conditional_search": search_ms,
                        "event_source_encoding": event_encoding_ms,
                        "query_conditioning": conditioning_ms,
                        "query_source_encoding": query_encoding_ms,
                        "selector_total_from_cached_source_tokens": (
                            query_encoding_ms + conditioning_ms + search_ms
                        ),
                    },
                    "logical_shard": state["logical_shard"],
                    "methods": methods,
                    "predicted_utilities": predicted,
                    "state_id": state["state_id"],
                    "subset_score_count": score_count,
                    "tracks": state["track_membership"],
                    "trajectory_id": state["trajectory_id"],
                }
            )
    records.sort(key=lambda row: row["state_id"])
    result = {
        "cache_content_sha256": cache_manifest["content_sha256"],
        "checkpoint_sha256": candidate["sha256"],
        "config_sha256": _sha256_file(heldout_path),
        "input_content_sha256": input_manifest["content_sha256"],
        "inventory_sha256": heldout["state_inventory"]["sha256"],
        "model_name": args.model_name,
        "records": records,
        "schema_version": "1.0.0",
        "status": (
            "COMPLETED_SET_UTILITY_HELDOUT_MODEL_SELECTION_PARTITION"
            if input_manifest["status"]
            == "COMPLETED_SET_UTILITY_HELDOUT_FEATURE_PARTITION"
            else "COMPLETED_SET_UTILITY_HELDOUT_MODEL_SELECTIONS"
        ),
        "variant": candidate["variant"],
    }
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_atomic(args.output, result)
    print(json.dumps({key: result[key] for key in ("content_sha256", "model_name", "status")}, sort_keys=True))


if __name__ == "__main__":
    main()
