#!/usr/bin/env python3
"""Train and select the structured direct-marginal student without proxy gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalcache.set_utility_recovery_checkpoint import RecoveryCheckpointManager
from causalcache.set_utility_resume_generation import (
    has_published_resume_generation,
    load_resume_generation_collective,
    restore_rng_states,
    save_resume_generation_collective,
)
from causalcache.set_utility_formal_input_verification import (
    coordinate_formal_cache_verification,
    verify_formal_contextual_cache,
    verify_formal_training_input,
)
from causalcache.set_utility_heldout_truth_schedule import (
    FORMAL_MANIFEST_STATUS,
    FORMAL_RECEIPT_STATUS,
    LABEL_STATUS,
)
from causalcache.set_utility_train_heldout_contract import (
    reduce_epoch_truth,
    select_checkpoint_from_epoch_truth,
)
from causalcache.set_utility_structured_marginal import (
    StructuredMarginalHeadConfig,
    StructuredMarginalLossConfig,
    StructuredTokenConditionalMarginalPredictor,
    structured_conditional_marginal_loss,
    structured_greedy_budget_path,
)
from causalcache.set_utility_structured_training import (
    EpochRecovery,
    apply_split_manifest,
    balanced_group_epoch,
    balancing_inventory,
    build_epoch_truth_plan,
    conditional_groups,
    evaluate_epoch_truth,
    fixed_trajectory_split,
    formal_group_inventory,
    merge_epoch_truth_plans,
    pack_group_examples,
    prepare_group_examples,
)
from causalcache.set_utility_token_models import (
    EncodedConditionalMarginalState,
    TokenUtilityModelConfig,
)
from scripts.train_set_utility_direct_marginal_v3 import (
    _attach_conditional_supervision,
)
from scripts.train_set_utility_token_predictor import (
    _TokenCache,
    _batch_stream,
    _cache_covers_input,
    _configure_attention_backend,
    _distributed_epoch_shard,
    _distributed_metric_average,
    _read_json,
    _read_jsonl,
    _seed_training_runtime,
)


STRUCTURED_CONTROL_STATUS = "FROZEN_STRUCTURED_DIRECT_MARGINAL_CONTROL"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    value["content_sha256"] = hashlib.sha256(
        _canonical_json_bytes(unsigned)
    ).hexdigest()
    return value


def _read_signed_json(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    unsigned = dict(value)
    claimed = unsigned.pop("content_sha256", None)
    if claimed != hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest():
        raise ValueError(f"signed JSON content hash drifted: {path}")
    return value


def _load_split_manifest(path: Path | None, config: Mapping[str, Any]) -> dict[str, Any] | None:
    expected = config["input"].get("train_heldout_manifest_content_sha256")
    if expected is None:
        expected = config["input"].get("train_heldout_manifest_sha256")
    if path is None:
        if expected is not None:
            raise ValueError("committed training config requires a split manifest")
        return None
    manifest = _read_json(path)
    unsigned = dict(manifest)
    claimed = unsigned.pop("content_sha256", None)
    from causalcache.set_utility_train_heldout_contract import sha256_json

    if expected is None or claimed != expected or sha256_json(unsigned) != claimed:
        raise ValueError("train-heldout split manifest binding drifted")
    firewall = manifest.get("firewall", {})
    checkpoint = manifest.get("checkpoint_selection", {})
    selection = config["checkpoint_selection"]
    if (
        firewall.get("allowed_role", "train") != "train"
        or firewall.get("tune_access") is not False
        or firewall.get("evaluation_access") is not False
        or firewall.get("all_heldout_trajectory_states_excluded_from_optimizer")
        is not True
    ):
        raise ValueError("train-heldout manifest allows a non-train role")
    if (
        tuple(checkpoint.get("budgets", ())) != (1, 2, 3, 4)
        or checkpoint.get("require_complete_truth_each_epoch") is not True
        or int(checkpoint.get("patience", -1)) != int(selection["patience"])
        or float(checkpoint.get("minimum_delta", -1.0))
        != float(selection["minimum_delta"])
        or float(checkpoint.get("tie_breaker_minimum_delta", 0.0))
        != float(selection.get("tie_breaker_minimum_delta", 0.0))
    ):
        raise ValueError("train-heldout checkpoint-selection contract drifted")
    return manifest


def _load_supplemental_truth(
    sources: Sequence[tuple[Path, str]],
    *,
    allowed_state_ids: set[str],
    states_by_id: Mapping[str, Mapping[str, Any]],
    expected_model_family: str,
    expected_epoch_checkpoints: Mapping[int, str],
    expected_input_content_sha256: str,
    expected_heldout_manifest_content_sha256: str,
    expected_source_manifest_file_sha256: str,
) -> dict[str, dict[tuple[int, ...], float]]:
    """Load only explicitly allowlisted, sealed, checkpoint-bound truth roots."""
    result: dict[str, dict[tuple[int, ...], float]] = defaultdict(dict)
    if set(states_by_id) != allowed_state_ids:
        raise ValueError("supplemental truth state inventory drifted")
    expected_checkpoints = {
        int(epoch): _require_sha256(checkpoint, label="expected checkpoint SHA256")
        for epoch, checkpoint in expected_epoch_checkpoints.items()
    }
    expected_input = _require_sha256(
        expected_input_content_sha256,
        label="expected formal truth input content SHA256",
    )
    expected_heldout = _require_sha256(
        expected_heldout_manifest_content_sha256,
        label="expected formal truth heldout manifest content SHA256",
    )
    expected_source = _require_sha256(
        expected_source_manifest_file_sha256,
        label="expected formal truth source manifest file SHA256",
    )
    if sources and not expected_checkpoints:
        raise ValueError("formal truth cannot precede a saved model checkpoint")
    shared_label_identity: tuple[str, str, str] | None = None
    saw_current_epoch_inventory = False
    seen_roots: set[Path] = set()
    seen_manifests: set[str] = set()
    for root, raw_expected_manifest_sha in sources:
        root = root.resolve()
        expected_manifest_sha = _require_sha256(
            raw_expected_manifest_sha, label="allowlisted formal manifest SHA256"
        )
        if root in seen_roots or expected_manifest_sha in seen_manifests:
            raise ValueError("formal truth allowlist contains a duplicate source")
        seen_roots.add(root)
        seen_manifests.add(expected_manifest_sha)
        manifest_path = root / "formal-truth-manifest.json"
        receipt_path = root / "formal-truth-receipt.json"
        manifest = _read_compact_signed_json(
            manifest_path,
            schema_version="causalcache.formal_heldout_truth_manifest.v1",
        )
        receipt = _read_compact_signed_json(
            receipt_path,
            schema_version="causalcache.formal_heldout_truth_receipt.v1",
        )
        if (
            manifest.get("content_sha256") != expected_manifest_sha
            or manifest.get("status") != FORMAL_MANIFEST_STATUS
            or receipt.get("status") != FORMAL_RECEIPT_STATUS
            or receipt.get("formal_manifest_content_sha256")
            != expected_manifest_sha
            or receipt.get("formal_manifest_file_sha256")
            != _sha256_file(manifest_path)
            or receipt.get("truth_schedule_summary_content_sha256")
            != manifest.get("truth_schedule_summary_content_sha256")
            or receipt.get("truth_schedule_summary_file_sha256")
            != manifest.get("truth_schedule_summary_file_sha256")
            or receipt.get("state_count") != manifest.get("state_count")
            or receipt.get("coalition_count") != manifest.get("coalition_count")
            or manifest.get("input_content_sha256") != expected_input
            or receipt.get("input_content_sha256") != expected_input
            or manifest.get("heldout_manifest_content_sha256")
            != expected_heldout
            or receipt.get("heldout_manifest_content_sha256")
            != expected_heldout
            or manifest.get("source_manifest_file_sha256") != expected_source
            or receipt.get("source_manifest_file_sha256") != expected_source
        ):
            raise ValueError("formal truth manifest/receipt binding drifted")
        model_schedules = manifest.get("model_schedules")
        if not isinstance(model_schedules, Mapping):
            raise ValueError("formal truth omits model schedule provenance")
        matches = [
            binding
            for binding in model_schedules.values()
            if isinstance(binding, Mapping)
            and binding.get("model_family") == expected_model_family
        ]
        if len(matches) != 1:
            raise ValueError("formal truth does not uniquely bind this model family")
        model_binding = matches[0]
        _require_sha256(
            model_binding.get("content_sha256"),
            label="formal truth model schedule content SHA256",
        )
        _require_sha256(
            model_binding.get("file_sha256"),
            label="formal truth model schedule file SHA256",
        )
        _require_sha256(
            manifest.get("truth_schedule_summary_content_sha256"),
            label="formal truth schedule summary content SHA256",
        )
        _require_sha256(
            manifest.get("truth_schedule_summary_file_sha256"),
            label="formal truth schedule summary file SHA256",
        )
        epochs = model_binding.get("epochs")
        epoch_bindings = model_binding.get("epoch_checkpoints")
        if (
            not isinstance(epochs, list)
            or not epochs
            or epochs != list(range(1, len(epochs) + 1))
            or not isinstance(epoch_bindings, list)
            or len(epoch_bindings) != len(epochs)
        ):
            raise ValueError("formal truth model epoch inventory drifted")
        for epoch, binding in zip(epochs, epoch_bindings, strict=True):
            if not isinstance(binding, Mapping) or binding.get("epoch") != epoch:
                raise ValueError("formal truth checkpoint epoch drifted")
            checkpoint = _require_sha256(
                binding.get("checkpoint_sha256"),
                label="formal truth checkpoint SHA256",
            )
            if expected_checkpoints.get(epoch) != checkpoint:
                raise ValueError("formal truth checkpoint allowlist drifted")
        if set(epochs) == set(expected_checkpoints):
            saw_current_epoch_inventory = True

        source_revision = manifest.get("source_revision")
        scientific_sha = _require_sha256(
            manifest.get("scientific_config_sha256"),
            label="formal truth scientific config SHA256",
        )
        execution_sha = _require_sha256(
            manifest.get("execution_config_sha256"),
            label="formal truth execution config SHA256",
        )
        if not isinstance(source_revision, str) or re.fullmatch(
            r"[0-9a-f]{40}", source_revision
        ) is None:
            raise ValueError("formal truth source revision drifted")
        label_identity = (source_revision, scientific_sha, execution_sha)
        if shared_label_identity is None:
            shared_label_identity = label_identity
        elif label_identity != shared_label_identity:
            raise ValueError("formal truth roots disagree on source/config revisions")
        bindings = manifest.get("states")
        if (
            not isinstance(bindings, list)
            or manifest.get("state_count") != len(bindings)
            or receipt.get("state_count") != len(bindings)
        ):
            raise ValueError("formal truth manifest state inventory drifted")
        expected_paths: set[Path] = set()
        observed_rows = 0
        seen_states: set[str] = set()
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise ValueError("formal truth state binding is invalid")
            state_id = binding.get("state_id")
            relative = binding.get("terminal_relative_path")
            state = states_by_id.get(str(state_id))
            if (
                state_id not in allowed_state_ids
                or state_id in seen_states
                or not isinstance(relative, str)
                or state is None
                or binding.get("trajectory_id") != state.get("trajectory_id")
                or tuple(binding.get("candidate_event_ids", ()))
                != tuple(state.get("candidate_event_step_ids", ()))
            ):
                raise ValueError("formal truth state binding escapes fixed heldout states")
            seen_states.add(str(state_id))
            path = (root / relative).resolve()
            if root not in path.parents or path.parent != (root / "states").resolve():
                raise ValueError("formal truth terminal path escapes its root")
            expected_paths.add(path)
            if (
                not path.is_file()
                or _sha256_file(path) != binding.get("terminal_file_sha256")
            ):
                raise ValueError("formal truth terminal file hash drifted")
            terminal = _read_json(path)
            if (
                terminal.get("status") != LABEL_STATUS
                or terminal.get("role") != "train"
                or terminal.get("state_id") != state_id
                or terminal.get("trajectory_id") != state.get("trajectory_id")
                or tuple(terminal.get("candidate_event_step_ids", ()))
                != tuple(state.get("candidate_event_step_ids", ()))
                or terminal.get("state_identity_sha256")
                != binding.get("state_identity_sha256")
                or terminal.get("source_revision") != source_revision
                or terminal.get("scientific_config_sha256") != scientific_sha
                or terminal.get("execution_config_sha256") != execution_sha
            ):
                raise ValueError("formal truth terminal provenance drifted")
            rows = terminal.get("distance_rows")
            if not isinstance(rows, list) or len(rows) != binding.get("coalition_count"):
                raise ValueError("formal truth terminal coalition count drifted")
            observed_rows += len(rows)
            candidates = tuple(state["candidate_event_step_ids"])
            for ordinal, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    raise ValueError("formal truth terminal distance row is invalid")
                subset = _truth_subset(
                    row.get("coalition_event_step_ids"),
                    candidates=candidates,
                    label=f"formal truth {state_id} row {ordinal}",
                )
                distance = row.get("distance")
                if (
                    isinstance(distance, bool)
                    or not isinstance(distance, (int, float))
                    or not math.isfinite(float(distance))
                    or float(distance) < 0.0
                ):
                    raise ValueError("formal truth distance is invalid")
                previous = result[str(state_id)].get(subset)
                if previous is not None and previous != float(distance):
                    raise ValueError("supplemental truth roots conflict")
                result[str(state_id)][subset] = float(distance)
        actual_paths = {path.resolve() for path in (root / "states").glob("*.json")}
        if actual_paths != expected_paths or observed_rows != manifest.get(
            "coalition_count"
        ):
            raise ValueError("formal truth root no longer matches its sealed inventory")
    if sources and not saw_current_epoch_inventory:
        raise ValueError("formal truth allowlist contains only stale epoch schedules")
    return dict(result)


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _read_compact_signed_json(path: Path, *, schema_version: str) -> dict[str, Any]:
    from causalcache.set_utility_train_heldout_contract import sha256_json

    value = _read_json(path)
    unsigned = dict(value)
    claimed = unsigned.pop("content_sha256", None)
    if claimed != sha256_json(unsigned) or value.get("schema_version") != schema_version:
        raise ValueError(f"formal truth signed metadata drifted: {path}")
    return value


def _truth_subset(
    value: Any, *, candidates: tuple[int, ...], label: str
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an event-id array")
    subset = tuple(value)
    if (
        any(type(event) is not int for event in subset)
        or subset != tuple(sorted(subset))
        or len(subset) != len(set(subset))
        or not set(subset).issubset(candidates)
    ):
        raise ValueError(f"{label} is not canonical")
    return subset


def _truth_sources(values: Sequence[str]) -> tuple[tuple[Path, str], ...]:
    sources = []
    for value in values:
        expected_sha, separator, raw_root = value.partition("=")
        if not separator or not raw_root:
            raise ValueError(
                "--truth-source requires MANIFEST_CONTENT_SHA256=ROOT"
            )
        sources.append((Path(raw_root).resolve(), _require_sha256(
            expected_sha, label="truth-source manifest SHA256"
        )))
    return tuple(sources)


def _plan_checkpoint_bindings(
    plans: Sequence[Mapping[str, Any]],
) -> dict[int, str]:
    result: dict[int, str] = {}
    for plan in plans:
        epoch = int(plan["epoch"])
        checkpoint = plan.get("checkpoint")
        if epoch in result or not isinstance(checkpoint, Mapping):
            raise ValueError("saved epoch checkpoint inventory drifted")
        result[epoch] = _require_sha256(
            checkpoint.get("sha256"), label="saved epoch checkpoint SHA256"
        )
    if result and sorted(result) != list(range(1, max(result) + 1)):
        raise ValueError("saved epoch checkpoint inventory is not contiguous")
    return result


def _validate_config(config: Mapping[str, Any], variant_name: str) -> dict[str, Any]:
    if config.get("schema_version") != "1.0.0":
        raise ValueError("structured training config schema drifted")
    if config.get("status") != STRUCTURED_CONTROL_STATUS:
        raise ValueError("structured training config is not frozen for execution")
    variant = config.get("variants", {}).get(variant_name)
    if not isinstance(variant, dict):
        raise ValueError("unknown structured training variant")
    training = config.get("training")
    selection = config.get("checkpoint_selection")
    if not isinstance(training, dict) or not isinstance(selection, dict):
        raise ValueError("structured training config omits training/selection")
    if tuple(selection.get("budgets", ())) != (1, 2, 3, 4):
        raise ValueError("structured checkpoint selection requires B1--B4")
    if selection.get("metric") != (
        "trajectory_equal_true_B1_B4_normalized_recovery_macro"
    ):
        raise ValueError("structured checkpoint selection metric drifted")
    if config.get("firewall", {}).get("allowed_input_role") != "train":
        raise ValueError("structured training firewall must allow only train")
    if config.get("firewall", {}).get("final_evaluation_access") is not False:
        raise ValueError("structured training must forbid final evaluation")
    input_contract = config.get("input", {})
    for key in (
        "training_input_content_sha256",
        "required_ancestor_input_content_sha256",
        "contextual_cache_content_sha256",
        "train_heldout_manifest_content_sha256",
        "source_manifest_file_sha256",
    ):
        value = input_contract.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"structured training input {key} is not frozen")
    if input_contract["training_input_content_sha256"] == input_contract[
        "required_ancestor_input_content_sha256"
    ]:
        raise ValueError("structured training still points to the ancestor input")
    if input_contract.get("merged_on_policy_labels_required") is not True:
        raise ValueError("structured training requires merged on-policy labels")
    if input_contract.get("evaluation_access") is not False:
        raise ValueError("structured training input firewall is open")
    merged = input_contract.get("merged_on_policy_contract")
    if not isinstance(merged, Mapping):
        raise ValueError("structured merged on-policy contract is absent")
    if merged.get("source_name") != "direct_on_policy_v1":
        raise ValueError("structured merged on-policy source drifted")
    _require_sha256(
        merged.get("schedule_content_sha256"),
        label="structured on-policy schedule content SHA256",
    )
    for key in (
        "added_distance_row_count",
        "state_count",
        "total_complete_group_count",
        "train_state_count",
    ):
        if type(merged.get(key)) is not int or int(merged[key]) <= 0:
            raise ValueError(f"structured merged on-policy {key} is invalid")
    exact_inventory = training.get("exact_optimizer_inventory")
    if not isinstance(exact_inventory, Mapping):
        raise ValueError("structured exact optimizer inventory is not frozen")
    _require_sha256(
        exact_inventory.get("content_sha256"),
        label="structured optimizer inventory content SHA256",
    )
    if (
        type(exact_inventory.get("optimizer_state_count")) is not int
        or type(exact_inventory.get("complete_group_count")) is not int
    ):
        raise ValueError("structured exact optimizer inventory counts are invalid")
    axes = tuple(training.get("balancing", {}).get("axes", ()))
    required_axes = {"class", "cardinality", "history", "stop_all_negative"}
    if set(axes) != required_axes:
        raise ValueError("structured balancing must cover all frozen axes")
    maximum_groups = training["balancing"].get(
        "maximum_groups_per_encoded_state"
    )
    if type(maximum_groups) is not int or maximum_groups <= 0:
        raise ValueError("structured balancing must cap groups per encoded state")
    if training["balancing"].get("loss_aggregation") != (
        "global_equal_weight_per_candidate_complete_group_across_"
        "packing_accumulation_and_ddp"
    ):
        raise ValueError("structured group-loss aggregation drifted")
    barrier = training.get("epoch_barrier", {})
    if barrier != {
        "mode": "checkpoint_rollout_truth_resume_before_next_optimizer_epoch",
        "posthoc_union_role": "debug_only",
        "require_candidate_complete_truth": True,
        "waiting_status": "WAITING_FOR_HELDOUT_TRUTH",
    }:
        raise ValueError("structured formal epoch barrier drifted")
    if selection.get("incomplete_epoch_action") != (
        "wait_without_selecting_or_advancing_patience"
    ):
        raise ValueError("structured incomplete-epoch action drifted")
    return variant


def _load_training_inputs(
    *,
    input_root: Path,
    cache_root: Path,
    config: Mapping[str, Any],
    split_manifest: Mapping[str, Any] | None,
    cache_receipt_mode: str = "auto",
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    binding = config["input"]
    input_manifest = verify_formal_training_input(
        input_root,
        expected_content_sha256=binding["training_input_content_sha256"],
    )
    cache_manifest = verify_formal_contextual_cache(
        cache_root,
        expected_content_sha256=binding["contextual_cache_content_sha256"],
        receipt_mode=cache_receipt_mode,
    )
    states_path = input_root / str(input_manifest.get("states_jsonl", ""))
    merged = binding["merged_on_policy_contract"]
    enrichment = input_manifest.get("enrichment")
    source_bindings = (
        enrichment.get("source_bindings")
        if isinstance(enrichment, Mapping)
        else None
    )
    source_binding = (
        source_bindings.get(merged["source_name"])
        if isinstance(source_bindings, Mapping)
        else None
    )
    if (
        input_manifest.get("evaluation_labels_included") is not False
        or cache_manifest.get("evaluation_labels_included") is not False
        or not _cache_covers_input(input_manifest, cache_manifest)
        or input_manifest.get("content_sha256")
        != binding["training_input_content_sha256"]
        or cache_manifest.get("content_sha256")
        != binding["contextual_cache_content_sha256"]
        or not states_path.is_file()
        or _sha256_file(states_path) != input_manifest.get("states_sha256")
        or input_manifest.get("parent_content_sha256")
        != binding["required_ancestor_input_content_sha256"]
        or not isinstance(source_binding, Mapping)
        or source_binding.get("schedule_content_sha256")
        != merged["schedule_content_sha256"]
        or source_binding.get("state_count") != merged["state_count"]
        or source_binding.get("added_distance_row_count")
        != merged["added_distance_row_count"]
    ):
        raise ValueError("structured training input/cache identity or firewall drifted")
    states = tuple(
        row
        for row in _read_jsonl(input_root / input_manifest["states_jsonl"])
        if row["role"] == "train"
    )
    if not states:
        raise ValueError("structured training input contains no train states")
    if len(states) != int(merged["train_state_count"]):
        raise ValueError("structured train state count drifted")
    total_complete_groups = sum(
        len(
            conditional_groups(
                state,
                normalization_floor=float(config["training"]["normalization_floor"]),
                maximum_base_cardinality=int(
                    config["training"]["maximum_base_cardinality"]
                ),
            )
        )
        for state in states
    )
    if total_complete_groups != int(merged["total_complete_group_count"]):
        raise ValueError("structured total complete-group count drifted")
    if split_manifest is None:
        split = config["training"]["fallback_split"]
        optimization, heldout = fixed_trajectory_split(
            states,
            heldout_fraction=float(split["fraction"]),
            salt=str(split["salt"]),
        )
    else:
        optimization, heldout = apply_split_manifest(states, split_manifest)
    return input_manifest, cache_manifest, optimization, heldout


def _slice_encoded(
    encoded: EncodedConditionalMarginalState, index: int
) -> EncodedConditionalMarginalState:
    return EncodedConditionalMarginalState(
        query=encoded.query[index : index + 1],
        events=encoded.events[index : index + 1],
        event_mask=encoded.event_mask[index : index + 1],
    )


def _selected_mask_for_encoded_state(
    encoded: EncodedConditionalMarginalState,
    event_ids: Sequence[int],
    chosen: Sequence[int],
    *,
    torch: Any,
) -> Any:
    selected_mask = torch.zeros_like(encoded.event_mask, dtype=torch.bool)
    by_event = {event_id: position for position, event_id in enumerate(event_ids)}
    for event_id in chosen:
        selected_mask[0, by_event[event_id]] = True
    return selected_mask


def _trim_padded_candidate_scores(values: Any, event_count: int) -> Any:
    if values.ndim != 1 or values.shape[0] < event_count + 1:
        raise ValueError("heldout candidate score geometry drifted")
    return values[: event_count + 1]


def _rollout_heldout(
    model: Any,
    states: Sequence[dict[str, Any]],
    *,
    cache: Any,
    batch_size: int,
    device: str,
    normalization_floor: float,
    torch: Any,
) -> tuple[dict[str, Any], ...]:
    records = []
    model.eval()
    with torch.inference_mode():
        for selected, batch in _batch_stream(
            tuple(states),
            batch_size=batch_size,
            cache=cache,
            device=device,
            torch=torch,
            normalization_floor=normalization_floor,
        ):
            model_inputs = dict(batch["model"])
            model_inputs.pop("subset_masks")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                encoded_batch = model.encode_state_once(**model_inputs)
            for index, state in enumerate(selected):
                encoded = _slice_encoded(encoded_batch, index)
                event_ids = tuple(state["candidate_event_step_ids"])

                def score(chosen: tuple[int, ...]) -> tuple[float, ...]:
                    selected_mask = _selected_mask_for_encoded_state(
                        encoded,
                        event_ids,
                        chosen,
                        torch=torch,
                    )
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        values = model.score_encoded_candidates(encoded, selected_mask)[0]
                    values = _trim_padded_candidate_scores(values, len(event_ids))
                    return tuple(float(value) for value in values.tolist())

                path = structured_greedy_budget_path(score, event_ids)
                records.append(
                    {
                        "candidate_event_ids": list(event_ids),
                        "queried_bases": [row["base_subset"] for row in path["trace"]],
                        "selections": path["selections"],
                        "state_id": state["state_id"],
                        "trace": path["trace"],
                        "trajectory_id": state["trajectory_id"],
                    }
                )
    return tuple(sorted(records, key=lambda row: row["state_id"]))


def _epoch_plan_path(output_root: Path, epoch: int) -> Path:
    return output_root / "heldout-rollouts" / f"epoch-{epoch:04d}.json"


def _read_epoch_plans(output_root: Path) -> tuple[dict[str, Any], ...]:
    paths = sorted((output_root / "heldout-rollouts").glob("epoch-*.json"))
    result = []
    for path in paths:
        value = _read_json(path)
        unsigned = dict(value)
        claimed = unsigned.pop("content_sha256", None)
        if claimed != hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest():
            raise ValueError(f"epoch rollout content hash drifted: {path}")
        result.append(value)
    return tuple(result)


def _publish_truth_schedule(
    output_root: Path, plans: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    schedule = merge_epoch_truth_plans(plans)
    schedule["model_family"] = "deepsets_structured_marginal"
    schedule["epoch_checkpoints"] = [
        {
            "checkpoint_sha256": _require_sha256(
                plan["checkpoint"]["sha256"],
                label="truth schedule checkpoint SHA256",
            ),
            "epoch": int(plan["epoch"]),
        }
        for plan in sorted(plans, key=lambda value: int(value["epoch"]))
    ]
    schedule["status"] = (
        "COMPLETE_STRUCTURED_HELDOUT_TRUTH"
        if schedule["missing_coalition_count"] == 0
        else "PENDING_STRUCTURED_HELDOUT_TRUTH"
    )
    _write_atomic(output_root / "heldout-truth-schedule.json", _signed(schedule))
    return schedule


def _selection_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    selection = config["checkpoint_selection"]
    return {
        "minimum_delta": float(selection["minimum_delta"]),
        "patience": int(selection["patience"]),
        "tie_breaker_minimum_delta": float(
            selection.get("tie_breaker_minimum_delta", 0.0)
        ),
    }


def _save_or_verify_epoch_checkpoint(
    manager: RecoveryCheckpointManager,
    model: Any,
    *,
    epoch: int,
    output_root: Path,
) -> dict[str, Any]:
    matches = [
        row for row in manager.manifest["epochs"] if int(row["epoch"]) == epoch
    ]
    if not matches:
        return manager.save_epoch(model, epoch=epoch)
    if len(matches) != 1:
        raise ValueError("resume found a duplicated epoch checkpoint")
    checkpoint = dict(matches[0]["checkpoint"])
    registered = output_root / checkpoint["path"]
    if (
        not registered.is_file()
        or registered.stat().st_size != int(checkpoint["byte_count"])
        or _sha256_file(registered) != checkpoint["sha256"]
    ):
        raise ValueError("registered epoch checkpoint bytes drifted")
    replay = registered.with_name(f".{registered.name}.{os.getpid()}.resume-verify")
    try:
        manager.saver(model, replay)
        if _sha256_file(replay) != checkpoint["sha256"]:
            raise ValueError("resumed epoch replay differs from the saved checkpoint")
    finally:
        replay.unlink(missing_ok=True)
    return checkpoint


def _read_existing_epoch_plan(
    output_root: Path, *, epoch: int, checkpoint: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _epoch_plan_path(output_root, epoch)
    if not path.exists():
        return None
    value = _read_json(path)
    unsigned = dict(value)
    claimed = unsigned.pop("content_sha256", None)
    if (
        claimed != hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
        or int(value.get("epoch", -1)) != epoch
        or value.get("checkpoint") != checkpoint
    ):
        raise ValueError("resumed epoch plan identity drifted")
    return value


def _annotate_epoch_plan(
    plan: dict[str, Any],
    *,
    split_manifest: Mapping[str, Any],
    states_by_id: Mapping[str, Mapping[str, Any]],
    normalization_floor: float,
    supplemental: Mapping[str, Mapping[tuple[int, ...], float]],
) -> dict[str, Any]:
    recovery, selected_missing = evaluate_epoch_truth(
        plan,
        states_by_id=states_by_id,
        normalization_floor=normalization_floor,
        supplemental=supplemental,
    )
    contract_epoch_truth = _contract_epoch_truth(
        plan,
        split_manifest=split_manifest,
        states_by_id=states_by_id,
        normalization_floor=normalization_floor,
        supplemental=supplemental,
    )
    plan["contract_epoch_truth"] = contract_epoch_truth
    plan["recovery"] = _epoch_recovery_dict(recovery) if plan["truth_complete"] else None
    plan["selected_recovery_diagnostic"] = _epoch_recovery_dict(recovery)
    plan["selected_missing"] = list(selected_missing)
    plan["status"] = (
        "COMPLETE_STRUCTURED_EPOCH_TRUE_RECOVERY"
        if contract_epoch_truth["truth_complete"] is True
        else "WAITING_FOR_HELDOUT_TRUTH"
    )
    return plan


def _refresh_epoch_plans(
    output_root: Path,
    plans: Sequence[Mapping[str, Any]],
    *,
    split_manifest: Mapping[str, Any],
    states_by_id: Mapping[str, Mapping[str, Any]],
    normalization_floor: float,
    supplemental: Mapping[str, Mapping[tuple[int, ...], float]],
) -> tuple[dict[str, Any], ...]:
    """Re-evaluate saved rollouts before any resumed optimization step."""
    rebuilt = []
    for original in plans:
        plan = build_epoch_truth_plan(
            epoch=int(original["epoch"]),
            checkpoint=original["checkpoint"],
            rollout_records=original["records"],
            states_by_id=states_by_id,
            supplemental=supplemental,
        )
        _annotate_epoch_plan(
            plan,
            split_manifest=split_manifest,
            states_by_id=states_by_id,
            normalization_floor=normalization_floor,
            supplemental=supplemental,
        )
        _write_atomic(
            _epoch_plan_path(output_root, int(plan["epoch"])), _signed(plan)
        )
        rebuilt.append(plan)
    return tuple(rebuilt)


def _attach_structured_group_weights(
    batch: dict[str, Any],
    selected: Sequence[Mapping[str, Any]],
    *,
    torch: Any,
) -> dict[str, Any]:
    group_mask = batch["conditional_group_mask"]
    weights = torch.zeros(
        group_mask.shape,
        dtype=torch.float32,
        device=group_mask.device,
    )
    for batch_index, state in enumerate(selected):
        groups = tuple(state["conditional_groups"])
        raw_weights = tuple(state.get("conditional_group_weights", (1.0,) * len(groups)))
        if len(raw_weights) != len(groups) or any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in raw_weights
        ):
            raise ValueError("structured packed group weights are invalid")
        weights[batch_index, : len(groups)] = torch.tensor(
            raw_weights, dtype=torch.float32, device=group_mask.device
        )
    if bool((weights[~group_mask] != 0.0).any()) or bool(
        (weights.sum(dim=1) <= 0.0).any()
    ):
        raise ValueError("structured packed group weights disagree with group mask")
    batch["conditional_group_weights"] = weights
    return batch


def _structured_group_sum_backward_scale(
    *, global_group_weight: float, world_size: int
) -> float:
    if (
        not math.isfinite(global_group_weight)
        or global_group_weight <= 0.0
        or world_size <= 0
    ):
        raise ValueError("structured global group denominator is invalid")
    return world_size / global_group_weight


def _training_loss(
    predictions: Any,
    batch: Mapping[str, Any],
    *,
    loss_config: StructuredMarginalLossConfig,
    reduction: str = "mean",
) -> tuple[Any, dict[str, float]]:
    normalized = predictions / batch["scales"][:, None, None]
    return structured_conditional_marginal_loss(
        normalized,
        normalized_targets=batch["conditional_normalized_targets"],
        action_mask=batch["conditional_action_mask"],
        group_mask=batch["conditional_group_mask"],
        selected_masks=batch["model"]["selected_masks"],
        config=loss_config,
        group_weights=batch.get("conditional_group_weights"),
        reduction=reduction,
    )


def _initialize_model(
    *,
    variant: Mapping[str, Any],
    initialization: Mapping[str, Any],
    checkpoint_path: Path | None,
    device: str,
    torch: Any,
) -> Any:
    model = StructuredTokenConditionalMarginalPredictor(
        TokenUtilityModelConfig(**variant["encoder"]),
        StructuredMarginalHeadConfig(**variant["head"]),
    ).to(device)
    mode = initialization["mode"]
    expected_sha = initialization.get("checkpoint_sha256")
    if mode == "fresh":
        if checkpoint_path is not None or expected_sha is not None:
            raise ValueError("fresh initialization cannot bind a checkpoint")
        return model
    if checkpoint_path is None or _sha256_file(checkpoint_path) != expected_sha:
        raise ValueError("structured initialization checkpoint drifted")
    from safetensors.torch import load_file

    state = load_file(str(checkpoint_path), device=device)
    if mode == "shared_direct_encoder":
        model.load_shared_encoder_from_direct_state(state)
    elif mode == "structured_full":
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError("unknown structured initialization mode")
    return model


def _fit(args: argparse.Namespace) -> None:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("structured training requires PyTorch") from error
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("structured training requires CUDA")
    config_path = args.config.resolve()
    config = _read_json(config_path)
    variant = _validate_config(config, args.variant)
    split_manifest = _load_split_manifest(args.split_manifest, config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    allowed_world_sizes = tuple(int(value) for value in variant["allowed_world_sizes"])
    if world_size not in allowed_world_sizes:
        raise ValueError("DDP world size is outside the committed variant")
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    input_root = args.input_root.resolve()
    cache_root = args.cache_root.resolve()
    cache_receipt_mode = coordinate_formal_cache_verification(
        cache_root,
        expected_content_sha256=config["input"]["contextual_cache_content_sha256"],
        rank=rank,
        distributed=distributed,
        torch=torch,
    )
    input_manifest, cache_manifest, optimization_states, heldout_states = (
        _load_training_inputs(
            input_root=input_root,
            cache_root=cache_root,
            config=config,
            split_manifest=split_manifest,
            cache_receipt_mode=cache_receipt_mode,
        )
    )
    training = config["training"]
    selection = config["checkpoint_selection"]
    examples = prepare_group_examples(
        optimization_states,
        normalization_floor=float(training["normalization_floor"]),
        maximum_base_cardinality=int(training["maximum_base_cardinality"]),
        epsilon=float(training["balancing"]["epsilon"]),
    )
    observed_optimizer_inventory = formal_group_inventory(examples)
    if observed_optimizer_inventory != training["exact_optimizer_inventory"]:
        raise ValueError("structured exact optimizer inventory drifted")
    minimums = training["minimum_inventory"]
    if (
        len(examples) < int(minimums["complete_group_count"])
        or len({row["state_id"] for row in examples})
        < int(minimums["optimizer_state_count"])
        or len(heldout_states) != int(minimums["heldout_state_count"])
    ):
        raise ValueError("structured training inventory is below committed minimums")
    seed = int(training["seed"])
    _seed_training_runtime(torch, seed)
    torch.set_float32_matmul_precision("high")
    attention_backend = _configure_attention_backend(torch, training)
    checkpoint_model = _initialize_model(
        variant=variant,
        initialization=config["initialization"],
        checkpoint_path=args.initial_checkpoint.resolve()
        if args.initial_checkpoint is not None
        else None,
        device=device,
        torch=torch,
    )
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            checkpoint_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed(seed + rank)
    else:
        model = checkpoint_model
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(variant["learning_rate"]),
        weight_decay=float(variant["weight_decay"]),
    )
    epochs = int(training["epochs"])
    batch_size = int(variant["per_device_batch_size"])
    accumulation = int(variant["gradient_accumulation_steps"])
    balancing = training["balancing"]
    sample_count = int(balancing["samples_per_epoch"])
    warmup_epochs = max(1, round(epochs * float(training["warmup_ratio"])))
    minimum_lr_ratio = float(training["minimum_lr_ratio"])

    def lr_scale(epoch_index: int) -> float:
        if epoch_index < warmup_epochs:
            return (epoch_index + 1) / warmup_epochs
        progress = (epoch_index - warmup_epochs) / max(
            1, epochs - warmup_epochs
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    identity = {
        "cache_content_sha256": cache_manifest["content_sha256"],
        "config_sha256": _sha256_file(config_path),
        "input_content_sha256": input_manifest["content_sha256"],
        "optimizer_inventory_content_sha256": observed_optimizer_inventory[
            "content_sha256"
        ],
        "split_manifest_sha256": config["input"].get(
            "train_heldout_manifest_content_sha256",
            config["input"].get("train_heldout_manifest_sha256"),
        ),
        "variant": args.variant,
        "world_size": world_size,
    }
    output_root = args.output_root.resolve()
    manager = None
    if rank == 0:
        manager = RecoveryCheckpointManager(
            output_root,
            identity=identity,
            selection_split="train_trajectory_holdout",
        )
        if not args.resume and manager.manifest["epochs"]:
            raise FileExistsError(
                "structured checkpoint history exists without --resume"
            )
    if distributed:
        torch.distributed.barrier()
    (output_root / "heldout-rollouts").mkdir(parents=True, exist_ok=True)
    cache = _TokenCache(
        cache_root,
        cache_manifest,
        device="cpu",
        mode=str(variant["cache_mode"]),
    )
    loss_config = StructuredMarginalLossConfig(**training["loss"])
    heldout_by_id = {row["state_id"]: row for row in heldout_states}
    start_epoch = 1
    if args.resume:
        snapshot, _ = load_resume_generation_collective(
            output_root,
            rank=rank,
            world_size=world_size,
            identity=identity,
            map_location=device,
            distributed=distributed,
            torch=torch,
        )
        checkpoint_model.load_state_dict(snapshot["model"], strict=True)
        optimizer.load_state_dict(snapshot["optimizer"])
        scheduler.load_state_dict(snapshot["scheduler"])
        random.setstate(snapshot["python_rng_state"])
        restore_rng_states(snapshot, torch=torch)
        start_epoch = int(snapshot["epoch"]) + 1
    elif has_published_resume_generation(output_root):
        raise FileExistsError("structured resume generation exists without --resume")

    truth_plans = (
        tuple(
            plan
            for plan in _read_epoch_plans(output_root)
            if int(plan["epoch"]) < start_epoch
        )
        if args.resume
        else ()
    )
    supplemental = _load_supplemental_truth(
        _truth_sources(args.truth_source),
        allowed_state_ids=set(heldout_by_id),
        states_by_id=heldout_by_id,
        expected_model_family="deepsets_structured_marginal",
        expected_epoch_checkpoints=_plan_checkpoint_bindings(truth_plans),
        expected_input_content_sha256=input_manifest["content_sha256"],
        expected_heldout_manifest_content_sha256=split_manifest["content_sha256"],
        expected_source_manifest_file_sha256=config["input"][
            "source_manifest_file_sha256"
        ],
    )

    training_history = []
    progress_path = output_root / "training-progress.json"
    if args.resume and progress_path.exists():
        previous_progress = _read_signed_json(progress_path)
        if previous_progress.get("identity") != identity:
            raise ValueError("structured progress resume identity drifted")
        training_history = [
            dict(row)
            for row in previous_progress.get("history", ())
            if int(row["epoch"]) < start_epoch
        ]
        if [int(row["epoch"]) for row in training_history] != list(
            range(1, start_epoch)
        ):
            raise ValueError("structured progress epoch history drifted")
    started = time.time()
    resume_action = "continue"
    if args.resume and rank == 0:
        assert split_manifest is not None
        all_saved_plans = _read_epoch_plans(output_root)
        future_plan_epochs = [
            int(plan["epoch"])
            for plan in all_saved_plans
            if int(plan["epoch"]) >= start_epoch
        ]
        if future_plan_epochs not in ([], [start_epoch]):
            raise ValueError("structured resume found invalid future rollout epochs")
        saved_plans = tuple(
            plan
            for plan in all_saved_plans
            if int(plan["epoch"]) < start_epoch
        )
        if (
            len(saved_plans) != start_epoch - 1
            or int(saved_plans[-1]["epoch"]) != start_epoch - 1
        ):
            raise ValueError("structured resume checkpoint/rollout epochs drifted")
        saved_plans = _refresh_epoch_plans(
            output_root,
            saved_plans,
            split_manifest=split_manifest,
            states_by_id=heldout_by_id,
            normalization_floor=float(selection["normalization_floor"]),
            supplemental=supplemental,
        )
        incomplete_epochs = [
            int(plan["epoch"])
            for plan in saved_plans
            if plan["contract_epoch_truth"]["truth_complete"] is not True
        ]
        if incomplete_epochs and incomplete_epochs != [start_epoch - 1]:
            raise ValueError("formal resume found truth missing before the last epoch")
        schedule = _publish_truth_schedule(output_root, saved_plans)
        selection_trace = _contract_checkpoint_selection(saved_plans, config=config)
        resume_action = _formal_epoch_action(selection_trace)
        if training_history:
            training_history[-1]["heldout_recovery"] = saved_plans[-1].get(
                "recovery"
            )
            training_history[-1]["selection_trace"] = selection_trace
            training_history[-1]["truth_missing_coalition_count"] = schedule[
                "missing_coalition_count"
            ]
        progress = {
            "attention_backend": attention_backend,
            "balancing_inventory": balancing_inventory(examples),
            "elapsed_seconds_this_invocation": time.time() - started,
            "history": training_history,
            "identity": identity,
            "schema_version": "causalcache.structured_training_progress.v1",
            "status": (
                "WAITING_FOR_HELDOUT_TRUTH"
                if resume_action == "wait_for_heldout_truth"
                else "COMPLETED_TRUE_RECOVERY_EARLY_STOP"
                if resume_action == "early_stop"
                else "RESUMED_AFTER_COMPLETE_HELDOUT_TRUTH"
            ),
        }
        _write_atomic(progress_path, _signed(progress))
    if distributed:
        payload = [resume_action]
        torch.distributed.broadcast_object_list(payload, src=0)
        resume_action = str(payload[0])
        torch.distributed.barrier()

    termination_reason = (
        "waiting_for_heldout_truth"
        if resume_action == "wait_for_heldout_truth"
        else "true_recovery_early_stopping"
        if resume_action == "early_stop"
        else "maximum_epochs"
    )
    epoch_range = (
        range(start_epoch, epochs + 1) if resume_action == "continue" else ()
    )
    for epoch in epoch_range:
        model.train()
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        global_examples = balanced_group_epoch(
            examples,
            seed=seed + epoch,
            sample_count=sample_count,
            axes=tuple(balancing["axes"]),
            power=float(balancing["power"]),
            maximum_weight=float(balancing["maximum_weight"]),
        )
        global_examples = pack_group_examples(
            global_examples,
            maximum_groups_per_state=int(
                balancing["maximum_groups_per_encoded_state"]
            ),
            seed=seed + 100_000 + epoch,
        )
        if distributed:
            epoch_examples, _ = _distributed_epoch_shard(
                global_examples,
                per_device_batch_size=batch_size,
                gradient_accumulation_steps=accumulation,
                rank=rank,
                world_size=world_size,
            )
        else:
            padding = (-len(global_examples)) % (batch_size * accumulation)
            epoch_examples = global_examples + tuple(
                global_examples[index % len(global_examples)] for index in range(padding)
            )
        optimizer.zero_grad(set_to_none=True)
        totals = defaultdict(float)
        denominator = 0.0
        interaction_count = 0.0
        interaction_group_weight = 0.0
        interaction_sign_correct_weight = 0.0
        interaction_sign_weight = 0.0
        batches = iter(
            _batch_stream(
                epoch_examples,
                batch_size=batch_size,
                cache=cache,
                device=device,
                torch=torch,
                normalization_floor=float(training["normalization_floor"]),
            )
        )
        while True:
            raw_window = tuple(islice(batches, accumulation))
            if not raw_window:
                break
            window = []
            for selected, batch in raw_window:
                batch = _attach_conditional_supervision(batch, selected, torch=torch)
                batch = _attach_structured_group_weights(
                    batch, selected, torch=torch
                )
                window.append((selected, batch))
            local_window_weight = sum(
                float(batch["conditional_group_weights"].sum())
                for _, batch in window
            )
            group_denominator = torch.tensor(
                local_window_weight, dtype=torch.float64, device=device
            )
            if distributed:
                torch.distributed.all_reduce(
                    group_denominator, op=torch.distributed.ReduceOp.SUM
                )
            backward_scale = _structured_group_sum_backward_scale(
                global_group_weight=float(group_denominator.item()),
                world_size=world_size,
            )
            for window_index, (_, batch) in enumerate(window):
                synchronize = window_index + 1 == len(window)
                sync_context = (
                    nullcontext()
                    if not distributed or synchronize
                    else model.no_sync()
                )
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predictions = model(**batch["model"])
                        loss_sum, metrics = _training_loss(
                            predictions,
                            batch,
                            loss_config=loss_config,
                            reduction="sum",
                        )
                    (loss_sum * backward_scale).backward()
                group_weight = float(batch["conditional_group_weights"].sum())
                denominator += group_weight
                interaction_count += metrics["interaction_count"]
                interaction_group_weight += metrics["interaction_group_weight"]
                interaction_sign_correct_weight += (
                    metrics["interaction_sign_accuracy"]
                    * metrics["interaction_sign_weight"]
                )
                interaction_sign_weight += metrics["interaction_sign_weight"]
                for key, value in metrics.items():
                    if key not in {
                        "conditional_group_weight",
                        "interaction_count",
                        "interaction_group_weight",
                        "interaction_sign_accuracy",
                        "interaction_sign_weight",
                    }:
                        totals[key] += value * group_weight
            torch.nn.utils.clip_grad_norm_(
                parameters, float(training["maximum_gradient_norm"])
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        train_metrics = (
            _distributed_metric_average(totals, denominator, device=device, torch=torch)
            if distributed
            else {key: value / denominator for key, value in sorted(totals.items())}
        )
        epoch_counts = torch.tensor(
            [
                denominator,
                interaction_count,
                interaction_group_weight,
                interaction_sign_correct_weight,
                interaction_sign_weight,
            ],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            torch.distributed.all_reduce(
                epoch_counts, op=torch.distributed.ReduceOp.SUM
            )
        train_metrics["conditional_group_weight"] = float(epoch_counts[0].item())
        train_metrics["interaction_count"] = float(epoch_counts[1].item())
        train_metrics["interaction_group_weight"] = float(epoch_counts[2].item())
        train_metrics["interaction_sign_accuracy"] = (
            float(epoch_counts[3].item() / epoch_counts[4].item())
            if epoch_counts[4].item() > 0.0
            else 0.0
        )
        if not all(math.isfinite(value) for value in train_metrics.values()):
            raise RuntimeError("structured training produced non-finite metrics")
        scheduler.step()

        epoch_action = "continue"
        if rank == 0:
            assert manager is not None
            checkpoint = _save_or_verify_epoch_checkpoint(
                manager,
                checkpoint_model,
                epoch=epoch,
                output_root=output_root,
            )
            plan = _read_existing_epoch_plan(
                output_root, epoch=epoch, checkpoint=checkpoint
            )
            if plan is None:
                rollout_records = _rollout_heldout(
                    checkpoint_model,
                    heldout_states,
                    cache=cache,
                    batch_size=int(variant["heldout_batch_size"]),
                    device=device,
                    normalization_floor=float(training["normalization_floor"]),
                    torch=torch,
                )
            else:
                rollout_records = plan["records"]
            plan = build_epoch_truth_plan(
                epoch=epoch,
                checkpoint=checkpoint,
                rollout_records=rollout_records,
                states_by_id=heldout_by_id,
                supplemental=supplemental,
            )
            assert split_manifest is not None
            _annotate_epoch_plan(
                plan,
                split_manifest=split_manifest,
                states_by_id=heldout_by_id,
                normalization_floor=float(selection["normalization_floor"]),
                supplemental=supplemental,
            )
            _write_atomic(_epoch_plan_path(output_root, epoch), _signed(plan))
            plans = _read_epoch_plans(output_root)
            schedule = _publish_truth_schedule(output_root, plans)
            selection_trace = _contract_checkpoint_selection(plans, config=config)
            epoch_action = _formal_epoch_action(selection_trace)
            epoch_record = {
                "epoch": epoch,
                "heldout_recovery": plan.get("recovery"),
                "learning_rate": epoch_learning_rate,
                "selection_trace": selection_trace,
                "train": train_metrics,
                "truth_missing_coalition_count": schedule[
                    "missing_coalition_count"
                ],
            }
            training_history.append(epoch_record)
            progress = {
                "attention_backend": attention_backend,
                "balancing_inventory": balancing_inventory(examples),
                "elapsed_seconds_this_invocation": time.time() - started,
                "history": training_history,
                "identity": identity,
                "schema_version": "causalcache.structured_training_progress.v1",
                "status": (
                    "WAITING_FOR_HELDOUT_TRUTH"
                    if epoch_action == "wait_for_heldout_truth"
                    else "COMPLETED_TRUE_RECOVERY_EARLY_STOP"
                    if epoch_action == "early_stop"
                    else "RUNNING_STRUCTURED_TRAINING"
                ),
            }
            _write_atomic(progress_path, _signed(progress))
            print(json.dumps(epoch_record, sort_keys=True), flush=True)
        if distributed:
            payload = [epoch_action]
            torch.distributed.broadcast_object_list(payload, src=0)
            epoch_action = str(payload[0])
            torch.distributed.barrier()
        save_resume_generation_collective(
            output_root,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
            model=checkpoint_model,
            optimizer=optimizer,
            scheduler=scheduler,
            identity=identity,
            distributed=distributed,
            torch=torch,
        )
        if epoch_action == "wait_for_heldout_truth":
            termination_reason = "waiting_for_heldout_truth"
            break
        if epoch_action == "early_stop":
            termination_reason = "true_recovery_early_stopping"
            break

    if rank == 0:
        plans = _read_epoch_plans(output_root)
        selection_trace = _contract_checkpoint_selection(plans, config=config)
        summary = {
            "attention_backend": attention_backend,
            "balancing_inventory": balancing_inventory(examples),
            "checkpoint_count": len(plans),
            "elapsed_seconds_this_invocation": time.time() - started,
            "evaluation_or_test_records_loaded": False,
            "heldout_state_count": len(heldout_states),
            "identity": identity,
            "optimization_group_count": len(examples),
            "schema_version": "causalcache.structured_training.v1",
            "selection": selection_trace,
            "selected_checkpoint": _selected_checkpoint(plans, selection_trace),
            "status": (
                "WAITING_FOR_HELDOUT_TRUTH"
                if termination_reason == "waiting_for_heldout_truth"
                else "COMPLETED_STRUCTURED_TRAINING_SELECTED_BY_TRUE_RECOVERY"
            ),
            "termination_reason": termination_reason,
        }
        _write_atomic(output_root / "summary.json", _signed(summary))
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _recovery_records(
    plan: Mapping[str, Any],
    *,
    states_by_id: Mapping[str, Mapping[str, Any]],
    normalization_floor: float,
    supplemental: Mapping[str, Mapping[tuple[int, ...], float]],
    require_complete: bool = True,
) -> tuple[dict[str, Any], ...]:
    from causalcache.set_utility_structured_training import distance_table

    rows = []
    for record in plan["records"]:
        state = states_by_id[record["state_id"]]
        truth = distance_table(state, supplemental)
        denominator = max(truth[()], normalization_floor)
        for budget in (1, 2, 3, 4):
            subset = tuple(record["selections"][str(budget)])
            if subset not in truth:
                if require_complete:
                    raise ValueError("recovery record requires complete selected truth")
                continue
            rows.append(
                {
                    "budget": budget,
                    "history_bin": record["history_bin"],
                    "normalized_recovery": (truth[()] - truth[subset]) / denominator,
                    "state_id": state["state_id"],
                    "trajectory_id": state["trajectory_id"],
                }
            )
    return tuple(rows)


def _contract_epoch_truth(
    plan: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any],
    states_by_id: Mapping[str, Mapping[str, Any]],
    normalization_floor: float,
    supplemental: Mapping[str, Mapping[tuple[int, ...], float]],
) -> dict[str, Any]:
    """Reduce one epoch only after its actual queried bases are truth-complete."""
    selected_records = _recovery_records(
        plan,
        states_by_id=states_by_id,
        normalization_floor=normalization_floor,
        supplemental=supplemental,
        require_complete=False,
    )
    eligible_records = selected_records if plan["truth_complete"] else ()
    result = reduce_epoch_truth(
        eligible_records,
        checkpoint_states=split_manifest["checkpoint_states"],
        epoch=int(plan["epoch"]),
        checkpoint_sha256=plan["checkpoint"]["sha256"],
        contract_content_sha256=split_manifest["content_sha256"],
    )
    result["candidate_complete_truth"] = bool(plan["truth_complete"])
    result["selected_truth_row_count"] = len(selected_records)
    return result


def _contract_checkpoint_selection(
    plans: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    if not plans:
        raise ValueError("structured checkpoint selection has no epoch plans")
    epoch_truth = []
    for plan in plans:
        value = plan.get("contract_epoch_truth")
        if not isinstance(value, Mapping):
            raise ValueError("epoch plan has no frozen-contract truth reduction")
        epoch_truth.append(value)
    return select_checkpoint_from_epoch_truth(
        epoch_truth,
        **_selection_parameters(config),
    )


def _formal_epoch_action(selection: Mapping[str, Any]) -> str:
    """Map frozen true-recovery state to the only legal formal action."""
    if selection.get("decision_ready") is not True:
        if selection.get("status") != "WAITING_FOR_COMPLETE_TRAIN_HELDOUT_EPOCH_TRUTH":
            raise ValueError("incomplete epoch selection has an invalid status")
        return "wait_for_heldout_truth"
    if selection.get("stopped_early") is True:
        return "early_stop"
    if selection.get("status") != "CONTINUE_TRAIN_HELDOUT_CHECKPOINT_SELECTION":
        raise ValueError("complete epoch selection has an invalid status")
    return "continue"


def _epoch_recovery_dict(value: EpochRecovery | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "budget_means": {
            str(budget): value.recovery.budget_means[budget]
            for budget in (1, 2, 3, 4)
        },
        "epoch": value.epoch,
        "long_plus_macro": value.long_plus_macro,
        "macro_B1_B4": value.recovery.macro_b1_b4,
        "state_count": value.recovery.state_count,
        "trajectory_count": value.recovery.trajectory_count,
    }


def _selected_checkpoint(
    plans: Sequence[Mapping[str, Any]], selection: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if selection is None or selection.get("decision_ready") is not True:
        return None
    epoch = int(selection["selected_epoch"])
    matches = [plan["checkpoint"] for plan in plans if int(plan["epoch"]) == epoch]
    if len(matches) != 1:
        raise ValueError("selected recovery epoch has no unique checkpoint")
    return dict(matches[0])


def _finalize(args: argparse.Namespace) -> None:
    config = _read_json(args.config.resolve())
    _validate_config(config, args.variant)
    split_manifest = _load_split_manifest(args.split_manifest, config)
    input_manifest, _, _, heldout_states = _load_training_inputs(
        input_root=args.input_root.resolve(),
        cache_root=args.cache_root.resolve(),
        config=config,
        split_manifest=split_manifest,
    )
    output_root = args.output_root.resolve()
    heldout_by_id = {row["state_id"]: row for row in heldout_states}
    original_plans = _read_epoch_plans(output_root)
    if not original_plans:
        raise ValueError("delayed selection found no saved epoch rollouts")
    supplemental = _load_supplemental_truth(
        _truth_sources(args.truth_source),
        allowed_state_ids=set(heldout_by_id),
        states_by_id=heldout_by_id,
        expected_model_family="deepsets_structured_marginal",
        expected_epoch_checkpoints=_plan_checkpoint_bindings(original_plans),
        expected_input_content_sha256=input_manifest["content_sha256"],
        expected_heldout_manifest_content_sha256=split_manifest["content_sha256"],
        expected_source_manifest_file_sha256=config["input"][
            "source_manifest_file_sha256"
        ],
    )
    rebuilt = []
    if split_manifest is None:
        raise ValueError("delayed selection requires the frozen train-heldout manifest")
    for original in original_plans:
        plan = build_epoch_truth_plan(
            epoch=int(original["epoch"]),
            checkpoint=original["checkpoint"],
            rollout_records=original["records"],
            states_by_id=heldout_by_id,
            supplemental=supplemental,
        )
        recovery, selected_missing = evaluate_epoch_truth(
            plan,
            states_by_id=heldout_by_id,
            normalization_floor=float(
                config["checkpoint_selection"]["normalization_floor"]
            ),
            supplemental=supplemental,
        )
        contract_epoch_truth = _contract_epoch_truth(
            plan,
            split_manifest=split_manifest,
            states_by_id=heldout_by_id,
            normalization_floor=float(
                config["checkpoint_selection"]["normalization_floor"]
            ),
            supplemental=supplemental,
        )
        plan["contract_epoch_truth"] = contract_epoch_truth
        plan["recovery"] = _epoch_recovery_dict(recovery) if plan["truth_complete"] else None
        plan["selected_recovery_diagnostic"] = _epoch_recovery_dict(recovery)
        plan["selected_missing"] = list(selected_missing)
        plan["status"] = (
            "COMPLETE_STRUCTURED_EPOCH_TRUE_RECOVERY"
            if contract_epoch_truth["truth_complete"] is True
            else "PENDING_STRUCTURED_EPOCH_TRUE_RECOVERY"
        )
        rebuilt.append(plan)
    schedule = _publish_truth_schedule(output_root, rebuilt)
    checkpoint_manifest = _read_json(output_root / "recovery-checkpoints.json")
    identity = checkpoint_manifest["identity"]
    if (
        identity.get("config_sha256") != _sha256_file(args.config.resolve())
        or identity.get("input_content_sha256") != input_manifest["content_sha256"]
        or identity.get("variant") != args.variant
    ):
        raise ValueError("delayed selection checkpoint identity drifted")
    selection = _contract_checkpoint_selection(rebuilt, config=config)
    complete = bool(
        selection["decision_ready"]
        and schedule["missing_coalition_count"] == 0
    )
    result = {
        "checkpoint_count": len(rebuilt),
        "input_content_sha256": input_manifest["content_sha256"],
        "missing_coalition_count": schedule["missing_coalition_count"],
        "schema_version": "causalcache.structured_delayed_selection.v1",
        "selection": selection,
        "selected_checkpoint": _selected_checkpoint(rebuilt, selection),
        "status": (
            "COMPLETED_STRUCTURED_DELAYED_TRUE_RECOVERY_SELECTION"
            if complete
            else "PENDING_STRUCTURED_DELAYED_TRUE_RECOVERY"
        ),
    }
    _write_atomic(output_root / "delayed-selection.json", _signed(result))
    for plan in rebuilt:
        _write_atomic(
            _epoch_plan_path(output_root, int(plan["epoch"])), _signed(plan)
        )
    print(json.dumps({"status": result["status"], "selection": selection}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "finalize"):
        child = subparsers.add_parser(command)
        child.add_argument("--input-root", type=Path, required=True)
        child.add_argument("--cache-root", type=Path, required=True)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--variant", required=True)
        child.add_argument("--split-manifest", type=Path)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument(
            "--truth-source",
            action="append",
            default=[],
            metavar="MANIFEST_CONTENT_SHA256=ROOT",
            help="Repeat only for explicitly allowlisted sealed truth roots.",
        )
    fit = subparsers.choices["fit"]
    fit.add_argument("--initial-checkpoint", type=Path)
    fit.add_argument("--device", required=True)
    fit.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "fit":
        _fit(args)
    else:
        _finalize(args)


if __name__ == "__main__":
    main()
