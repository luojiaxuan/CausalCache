#!/usr/bin/env python3
"""Run and seal the one frozen budget-deferral candidate on evaluation states."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from causalcache.set_utility_budget_deferral_evaluation import (
    budget_deferral_methods,
    build_selection_payload,
    evaluation_slice_memberships,
    project_label_blind_evaluation_state,
    sha256_file,
    validate_budget_deferral_config,
    write_selection_and_seal,
)
from causalcache.set_utility_direct_on_policy_evaluator import (
    load_structured_deepsets_selector,
)
from causalcache.set_utility_formal_input_verification import (
    verify_formal_contextual_cache,
    verify_formal_training_input,
)
from causalcache.set_utility_train_heldout_contract import sha256_json
from scripts.evaluate_set_utility_direct_on_policy_models import _run_state
from scripts.train_set_utility_structured_marginal import (
    _validate_config as validate_structured_config,
)
from scripts.train_set_utility_token_predictor import (
    _TokenCache,
    _cache_covers_input,
    _configure_attention_backend,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_states(
    path: Path, *, expected_count: int
) -> tuple[dict[str, Any], ...]:
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.append(project_label_blind_evaluation_state(json.loads(line)))
    by_state = {state["state_id"]: state for state in result}
    if len(result) != expected_count or len(by_state) != len(result):
        raise ValueError("evaluation contextual input denominator drifted")
    return tuple(
        sorted(result, key=lambda row: (row["trajectory_id"], row["state_id"]))
    )


def _label_firewall(
    input_manifest: Mapping[str, Any], cache_manifest: Mapping[str, Any]
) -> None:
    if (
        input_manifest.get("evaluation_labels_included") is not False
        or input_manifest.get("evaluation_labels_loaded", False) is not False
        or input_manifest.get("distance_rows_included", False) is not False
        or int(input_manifest.get("label_file_read_count", 0)) != 0
        or cache_manifest.get("evaluation_labels_included") is not False
    ):
        raise ValueError("evaluation input/cache violates the label firewall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-config", type=Path, required=True)
    parser.add_argument("--predictor-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-input-content-sha256", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expected-cache-content-sha256", required=True)
    parser.add_argument("--state-inventory", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--seal-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:"):
        raise ValueError("evaluation selector requires an explicit CUDA device")
    if args.warmup_iterations <= 0:
        raise ValueError("warmup iterations must be positive")
    try:
        import torch
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise RuntimeError("evaluation selector requires PyTorch and safetensors") from error
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation selector requires CUDA")

    contract_path = args.contract_config.resolve()
    contract = validate_budget_deferral_config(_read_json(contract_path))
    predictor_path = args.predictor_config.resolve()
    predictor_config = _read_json(predictor_path)
    predictor = contract["predictor"]
    variant = validate_structured_config(predictor_config, predictor["variant"])

    input_root = args.input_root.resolve()
    cache_root = args.cache_root.resolve()
    input_manifest = verify_formal_training_input(
        input_root,
        expected_content_sha256=args.expected_input_content_sha256,
    )
    cache_manifest = verify_formal_contextual_cache(
        cache_root,
        expected_content_sha256=args.expected_cache_content_sha256,
    )
    _label_firewall(input_manifest, cache_manifest)
    if not _cache_covers_input(input_manifest, cache_manifest):
        raise ValueError("evaluation contextual cache does not cover its input")
    expected_count = int(contract["frozen_candidate_evaluation"]["state_count"])
    if (
        input_manifest.get("state_count") != expected_count
        or input_manifest.get("role_counts") != {"evaluation": expected_count}
        or cache_manifest.get("input_content_sha256")
        != input_manifest.get("content_sha256")
    ):
        raise ValueError("evaluation input/cache denominator binding drifted")
    states = _read_states(
        input_root / input_manifest["states_jsonl"], expected_count=expected_count
    )
    expected_history = contract["frozen_candidate_evaluation"][
        "history_bin_counts"
    ]
    if Counter(state["history_bin"] for state in states) != Counter(expected_history):
        raise ValueError("evaluation history-bin census drifted")
    inventory_path = args.state_inventory.resolve()
    inventory = _read_json(inventory_path)
    memberships, slice_counts = evaluation_slice_memberships(
        states, inventory=inventory, config=contract
    )

    torch.set_float32_matmul_precision("high")
    _configure_attention_backend(torch, predictor_config["training"])
    seed = int(predictor_config["training"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    checkpoint_path = args.checkpoint.resolve()
    adapter = load_structured_deepsets_selector(
        variant=variant,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=predictor["checkpoint_sha256"],
        device=args.device,
        load_file=load_file,
    )
    cache = _TokenCache(cache_root, cache_manifest, device="cpu", mode="lazy_cpu")
    with torch.inference_mode():
        for _ in range(args.warmup_iterations):
            _run_state(
                adapter,
                states[0],
                cache=cache,
                event_sources={},
                event_identities={},
                device=args.device,
                torch=torch,
            )
        event_sources: dict[tuple[str, int], Any] = {}
        event_identities: dict[tuple[str, int], tuple[str, str]] = {}
        records = []
        for state in states:
            paths, latency = _run_state(
                adapter,
                state,
                cache=cache,
                event_sources=event_sources,
                event_identities=event_identities,
                device=args.device,
                torch=torch,
            )
            methods = budget_deferral_methods(
                state["candidate_event_step_ids"],
                learned_selections=paths["learned"]["selections"],
            )
            records.append(
                {
                    "candidate_event_ids": list(
                        state["candidate_event_step_ids"]
                    ),
                    "history_bin": state["history_bin"],
                    "latency_ms": {
                        key: latency[key]
                        for key in (
                            "cold_standalone",
                            "event_source_encoding",
                            "learned_search",
                            "query_source_encoding",
                            "source_token_io_and_transfer",
                            "state_conditioning",
                            "warm_selector_from_cached_sources",
                        )
                    },
                    "logical_shard": state["logical_shard"],
                    "methods": methods,
                    "slices": list(memberships[state["state_id"]]),
                    "state_id": state["state_id"],
                    "trajectory_id": state["trajectory_id"],
                }
            )

    bindings = {
        "cache_content_sha256": cache_manifest["content_sha256"],
        "cache_manifest_file_sha256": sha256_file(cache_root / "manifest.json"),
        "checkpoint_sha256": predictor["checkpoint_sha256"],
        "config_content_sha256": sha256_json(contract),
        "config_file_sha256": sha256_file(contract_path),
        "input_content_sha256": input_manifest["content_sha256"],
        "input_manifest_file_sha256": sha256_file(input_root / "manifest.json"),
        "input_states_sha256": input_manifest["states_sha256"],
        "predictor_config_file_sha256": sha256_file(predictor_path),
        "state_inventory_content_sha256": inventory["state_identity_sha256"],
        "state_inventory_file_sha256": sha256_file(inventory_path),
    }
    selection = build_selection_payload(
        records, bindings=bindings, slice_counts=slice_counts
    )
    seal = write_selection_and_seal(
        selection,
        selection_path=args.selection_output.resolve(),
        seal_path=args.seal_output.resolve(),
    )
    print(
        json.dumps(
            {
                "selection_content_sha256": selection["content_sha256"],
                "selection_file_sha256": seal["selection_file_sha256"],
                "state_count": selection["denominator"]["state_count"],
                "status": seal["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
