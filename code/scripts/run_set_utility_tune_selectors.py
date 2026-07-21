#!/usr/bin/env python3
"""Run a frozen contextual utility predictor over tune states only."""

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
    beam_budget_path,
    canonical_json_bytes,
    conditional_greedy_budget_path,
    recent_budget_selections,
)
from causalcache.set_utility_contextual_inputs import (
    contextual_input_lineage_sha256s,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_covers_input(input_manifest: dict[str, Any], bound_sha256: Any) -> bool:
    return isinstance(bound_sha256, str) and bound_sha256 in (
        contextual_input_lineage_sha256s(input_manifest)
    )


def _write_atomic(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class _TokenCache:
    def __init__(self, root: Path, manifest: dict[str, Any], *, device: str) -> None:
        from safetensors import safe_open

        if device != "cpu":
            raise ValueError("tune selector token cache must use lazy CPU mmap")
        grouped: dict[Path, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for key, record in manifest["tensor_inventory"].items():
            grouped[root / record["partition"] / record["shard"]].append(
                (key, record)
            )
        self.records = {}
        self.handles = {}
        for path in sorted(grouped):
            try:
                handle = safe_open(str(path), framework="pt", device="cpu")
            except Exception as error:
                raise RuntimeError(f"failed to open contextual cache shard: {path}") from error
            self.handles[path] = handle
            for key, record in grouped[path]:
                self.records[key] = (handle, record, path)
        if len(self.records) != len(manifest["tensor_inventory"]):
            raise RuntimeError("contextual token cache inventory is incomplete")

    def _get(self, key: str) -> Any:
        handle, record, path = self.records[key]
        try:
            tensor = handle.get_tensor(record["tensor"])
        except Exception as error:
            raise RuntimeError(f"failed to read contextual cache shard: {path}") from error
        if list(tensor.shape) != record["shape"] or str(tensor.dtype) != record["dtype"]:
            raise ValueError("contextual token cache metadata drifted")
        return tensor

    def visual(self, key: str) -> Any:
        return self._get(f"visual:{key}")

    def text(self, key: str) -> Any:
        return self._get(f"text:{key}")


def _pad_entities(
    visual: list[Any], text: list[Any], *, device: str, torch: Any
) -> tuple[Any, ...]:
    if not visual or len(visual) != len(text):
        raise ValueError("entity source tensors must be non-empty and aligned")
    visual_length = max(row.shape[0] for row in visual)
    text_length = max(row.shape[0] for row in text)
    hidden = visual[0].shape[1]
    source_device = visual[0].device
    visual_batch = torch.zeros(
        (len(visual), visual_length, hidden),
        dtype=visual[0].dtype,
        device=source_device,
    )
    visual_mask = torch.zeros(
        (len(visual), visual_length), dtype=torch.bool, device=source_device
    )
    text_batch = torch.zeros(
        (len(text), text_length, hidden), dtype=text[0].dtype, device=source_device
    )
    text_mask = torch.zeros(
        (len(text), text_length), dtype=torch.bool, device=source_device
    )
    for index, row in enumerate(visual):
        visual_batch[index, : row.shape[0]] = row
        visual_mask[index, : row.shape[0]] = True
    for index, row in enumerate(text):
        text_batch[index, : row.shape[0]] = row
        text_mask[index, : row.shape[0]] = True
    return tuple(
        value.to(device, non_blocking=False)
        for value in (visual_batch, visual_mask, text_batch, text_mask)
    )


def _milliseconds(call: Any, *, torch: Any) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-config", type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--role", choices=("train", "tune"), default="tune")
    parser.add_argument("--conditional-candidates-per-step", type=int, default=0)
    parser.add_argument("--state-id-file", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("tune selector output already exists")
    if args.conditional_candidates_per_step < 0:
        raise ValueError("conditional candidates per step cannot be negative")
    try:
        import torch
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise RuntimeError("tune selector requires PyTorch and safetensors") from error
    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise RuntimeError("tune selector requires an explicit CUDA device")

    input_root = args.input_root.resolve()
    input_manifest = _read_json(input_root / "manifest.json")
    cache_root = args.cache_root.resolve()
    cache_manifest = _read_json(cache_root / "manifest.json")
    config_path = args.config.resolve()
    config = _read_json(config_path)
    if args.variant not in config["variants"]:
        raise ValueError("unknown contextual tune-selector variant")
    variant = config["variants"][args.variant]
    selection_config_path = (
        config_path
        if args.selection_config is None
        else args.selection_config.resolve()
    )
    selection_source = (
        config
        if selection_config_path == config_path
        else _read_json(selection_config_path)
    )
    selection_config = selection_source.get("selection", {})
    search_method = str(selection_config.get("search", "conditional_greedy"))
    if search_method not in {"conditional_greedy", "beam"}:
        raise ValueError("unknown contextual selector search method")
    beam_width = int(selection_config.get("beam_width", 0))
    if search_method == "beam" and beam_width <= 0:
        raise ValueError("beam search requires a positive committed width")
    summary = _read_json(args.training_summary)
    if (
        input_manifest.get("evaluation_labels_included") is not False
        or cache_manifest.get("evaluation_labels_included") is not False
        or not _binding_covers_input(
            input_manifest, cache_manifest.get("input_content_sha256")
        )
        or summary.get("evaluation_records_loaded") is not False
        or not _binding_covers_input(
            input_manifest, summary.get("input_content_sha256")
        )
        or summary.get("cache_content_sha256") != cache_manifest.get("content_sha256")
        or summary.get("config_sha256") != _sha256_file(config_path)
        or summary.get("variant") != args.variant
        or summary.get("best_checkpoint", {}).get("sha256")
        != _sha256_file(args.checkpoint)
    ):
        raise ValueError("tune selector input/checkpoint firewall drifted")
    states = tuple(
        row
        for line in (input_root / input_manifest["states_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line
        for row in (json.loads(line),)
        if row["role"] == args.role
    )
    if not states or any(row["role"] != args.role for row in states):
        raise ValueError("selector state inventory drifted")
    if args.state_id_file is not None:
        requested = tuple(
            line.strip()
            for line in args.state_id_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("state-id filter must be non-empty and unique")
        available = {row["state_id"] for row in states}
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(
                f"state-id filter escapes the {args.role} inventory: {missing[:3]}"
            )
        requested_set = set(requested)
        states = tuple(row for row in states if row["state_id"] in requested_set)

    torch.manual_seed(int(summary["seed"]))
    torch.cuda.manual_seed_all(int(summary["seed"]))
    torch.set_float32_matmul_precision("high")
    model = TokenSetUtilityPredictor(
        TokenUtilityModelConfig(**variant["model"])
    ).to(args.device)
    model.load_state_dict(load_file(str(args.checkpoint), device=args.device), strict=True)
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
                    [cache.visual(state["event_image_keys"][index]) for index in new_indices],
                    [cache.text(state["event_text_keys"][index]) for index in new_indices],
                    device=args.device,
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
            event_index = {event_id: index for index, event_id in enumerate(events)}
            conditional_steps = []

            def score_batch(subsets: tuple[tuple[int, ...], ...]) -> tuple[float, ...]:
                masks = torch.zeros(
                    (1, len(subsets), len(events)),
                    dtype=torch.bool,
                    device=args.device,
                )
                for subset_index, subset in enumerate(subsets):
                    for event_id in subset:
                        masks[0, subset_index, event_index[event_id]] = True
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predictions = model.score_encoded_subsets(encoded, masks)
                values = tuple(float(value) for value in predictions[0].tolist())
                if args.conditional_candidates_per_step and len(subsets) > 1:
                    ranked = sorted(
                        zip(subsets, values, strict=True),
                        key=lambda item: (-item[1], item[0]),
                    )[: args.conditional_candidates_per_step]
                    shared = sorted(set.intersection(*(set(row) for row in subsets)))
                    conditional_steps.append(
                        {
                            "base_subset": shared,
                            "ranked_candidates": [
                                {
                                    "predicted_utility": utility,
                                    "subset": list(subset),
                                }
                                for subset, utility in ranked
                            ],
                        }
                    )
                return values

            torch.cuda.synchronize()
            search_started = time.perf_counter()
            if search_method == "beam":
                learned, utilities, score_count, beam_steps = beam_budget_path(
                    events,
                    score_batch=score_batch,
                    width=beam_width,
                )
            else:
                learned, utilities, score_count = conditional_greedy_budget_path(
                    events, score_batch=score_batch
                )
                beam_steps = ()
            torch.cuda.synchronize()
            search_ms = (time.perf_counter() - search_started) * 1000.0
            if any(not math.isfinite(value) for value in utilities.values()):
                raise RuntimeError("tune selector emitted non-finite utility")
            record = {
                "candidate_event_ids": list(events),
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
                    "beam_width": beam_width if search_method == "beam" else None,
                    "method": search_method,
                },
                "state_id": state["state_id"],
                "subset_score_count": score_count,
                "trajectory_id": state["trajectory_id"],
            }
            if conditional_steps:
                record["conditional_steps"] = conditional_steps
            if beam_steps:
                record["beam_steps"] = list(beam_steps)
            records.append(record)
    status = (
        "COMPLETED_SET_UTILITY_TUNE_SELECTIONS"
        if args.role == "tune"
        else "COMPLETED_SET_UTILITY_TRAIN_SELECTIONS"
    )
    result = {
        "cache_content_sha256": cache_manifest["content_sha256"],
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "config_sha256": _sha256_file(config_path),
        "input_content_sha256": input_manifest["content_sha256"],
        "records": sorted(records, key=lambda row: row["state_id"]),
        "schema_version": "1.0.0",
        "role": args.role,
        "status": status,
        "search_config_sha256": _sha256_file(selection_config_path),
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
                "variant": args.variant,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
