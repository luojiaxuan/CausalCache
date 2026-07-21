#!/usr/bin/env python3
"""Train the selector-only GUI-Owl top-layer LoRA under a truth barrier."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalcache.set_utility_formal_input_verification import (
    verify_formal_training_input,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes
from causalcache.set_utility_lora_selector import (
    FullSequenceBoundaryEntity,
    SelectorSideLoRAMarginalPredictor,
)
from causalcache.set_utility_recovery_checkpoint import RecoveryCheckpointManager
from causalcache.set_utility_resume_generation import (
    has_published_resume_generation,
    load_resume_generation_collective,
    restore_rng_states,
    save_resume_generation_collective,
)
from causalcache.set_utility_selector_boundary_cache import (
    SELECTOR_BOUNDARY_CACHE_STATUS,
)
from causalcache.set_utility_set_transformer_control import (
    CONTINUE_TRAINING,
    EARLY_STOP,
    WAITING_FOR_TRUTH,
    balanced_control_epoch,
    build_control_epoch_truth,
    control_balancing_inventory,
    control_epoch_barrier,
    group_equal_direct_loss,
    merge_control_truth_schedule,
    pack_control_examples,
    prepare_control_examples,
)
from causalcache.set_utility_structured_marginal import (
    structured_greedy_budget_path,
)
from causalcache.set_utility_structured_training import (
    apply_split_manifest,
    conditional_groups,
    formal_group_inventory,
)
from causalcache.set_utility_token_models import (
    EncodedConditionalMarginalState,
    TokenConditionalMarginalPredictor,
    TokenUtilityModelConfig,
)
from scripts.train_set_utility_direct_marginal_v3 import (
    _attach_conditional_supervision,
)
from scripts.train_set_utility_set_transformer_control import (
    _group_sum_backward_scale,
    _optimizer_steps,
    _partition_resume_plans,
)
from scripts.train_set_utility_structured_marginal import (
    _epoch_plan_path,
    _load_split_manifest,
    _load_supplemental_truth,
    _plan_checkpoint_bindings,
    _read_epoch_plans,
    _read_existing_epoch_plan,
    _read_signed_json,
    _save_or_verify_epoch_checkpoint,
    _selected_checkpoint,
    _sha256_file,
    _signed,
    _truth_sources,
    _write_atomic,
)
from scripts.train_set_utility_token_predictor import (
    _configure_attention_backend,
    _distributed_epoch_shard,
    _distributed_metric_average,
    _read_json,
    _read_jsonl,
    _seed_training_runtime,
    _targets,
)

PHASE_LORA_ONLY = "lora_only"
PHASE_JOINT = "joint"
PHASES = (PHASE_LORA_ONLY, PHASE_JOINT)
SELECTOR_CHECKPOINT_PREFIXES = ("predictor.",)
SELECTOR_CHECKPOINT_MARKERS = (".lora_a.weight", ".lora_b.weight")


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _verify_snapshot_manifest(path: Path, *, expected_sha256: str) -> str:
    expected = _require_sha256(
        expected_sha256, label="GUI-Owl snapshot manifest SHA256"
    )
    resolved = path.resolve()
    if not resolved.is_file() or _sha256_file(resolved) != expected:
        raise ValueError("GUI-Owl snapshot manifest file SHA256 drifted")
    return expected


def _verify_boundary_manifest(
    root: Path,
    *,
    expected_content_sha256: str,
    expected_contextual_input_sha256: str,
    expected_extraction_config_sha256: str,
) -> dict[str, Any]:
    manifest_path = root.resolve() / "manifest.json"
    manifest = _read_json(manifest_path)
    unsigned = dict(manifest)
    unsigned["content_sha256"] = ""
    observed_content = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    expected_content = _require_sha256(
        expected_content_sha256, label="selector boundary content SHA256"
    )
    expected_context = _require_sha256(
        expected_contextual_input_sha256,
        label="selector boundary contextual-input SHA256",
    )
    expected_extraction_config = _require_sha256(
        expected_extraction_config_sha256,
        label="selector boundary extraction-config SHA256",
    )
    if (
        manifest.get("status") != SELECTOR_BOUNDARY_CACHE_STATUS
        or manifest.get("evaluation_labels_included") is not False
        or manifest.get("content_sha256") != expected_content
        or observed_content != expected_content
        or manifest.get("contextual_input_content_sha256") != expected_context
        or manifest.get("config_sha256") != expected_extraction_config
    ):
        raise ValueError("selector boundary cache identity or firewall drifted")
    shards = manifest.get("shards")
    inventory = manifest.get("tensor_inventory")
    if not isinstance(shards, list) or not shards or not isinstance(inventory, Mapping):
        raise ValueError("selector boundary cache inventory is invalid")
    declared_paths = set()
    for row in shards:
        if not isinstance(row, Mapping):
            raise ValueError("selector boundary shard receipt is invalid")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("selector boundary shard path is invalid")
        path = (root.resolve() / relative).resolve()
        if root.resolve() not in path.parents or path in declared_paths:
            raise ValueError("selector boundary shard path escapes or repeats")
        declared_paths.add(path)
        _require_sha256(row.get("sha256"), label="selector boundary shard SHA256")
        if not path.is_file() or path.stat().st_size != row.get("byte_count"):
            raise ValueError("selector boundary shard stat drifted")
    if any(not isinstance(row, Mapping) for row in inventory.values()):
        raise ValueError("selector boundary tensor inventory row is invalid")
    inventory_paths = {
        (root.resolve() / str(row["partition"]) / str(row["shard"])).resolve()
        for row in inventory.values()
    }
    if inventory_paths != declared_paths:
        raise ValueError("selector boundary shard/tensor inventory drifted")
    return manifest


def _compact_selector_state_dict(model: Any) -> dict[str, Any]:
    state = model.state_dict()
    selected = {
        key: value
        for key, value in state.items()
        if key.startswith(SELECTOR_CHECKPOINT_PREFIXES)
        or any(marker in key for marker in SELECTOR_CHECKPOINT_MARKERS)
    }
    expected = len(model.predictor.state_dict()) + len(model.named_lora_parameters())
    if len(selected) != expected:
        raise RuntimeError("selector compact checkpoint inventory drifted")
    return selected


def _load_compact_selector_state(model: Any, state: Mapping[str, Any]) -> None:
    expected = set(_compact_selector_state_dict(model))
    if set(state) != expected:
        missing = sorted(expected - set(state))[:8]
        unexpected = sorted(set(state) - expected)[:8]
        raise ValueError(
            "selector compact checkpoint inventory drifted: "
            f"missing={missing}, unexpected={unexpected}"
        )
    incompatible = model.load_state_dict(dict(state), strict=False)
    if incompatible.unexpected_keys:
        raise ValueError("selector compact checkpoint has unexpected tensors")
    allowed_missing = set(model.state_dict()) - expected
    if set(incompatible.missing_keys) != allowed_missing:
        raise ValueError("selector compact checkpoint omitted a trainable tensor")


def _compact_safetensors_saver(model: Any, destination: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("selector checkpointing requires safetensors") from error
    state = {
        key: value.detach().to(device="cpu").contiguous()
        for key, value in _compact_selector_state_dict(model).items()
    }
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    save_file(state, str(temporary))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


class _CompactStateView:
    def __init__(self, model: Any) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        return _compact_selector_state_dict(self.model)


class _SelectorBoundaryCache:
    """Lazy CPU reader for allowlisted full-sequence selector entities."""

    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, Any],
        *,
        maximum_resident_entities: int = 16,
        maximum_open_shards: int = 64,
    ) -> None:
        try:
            from safetensors import safe_open
        except ModuleNotFoundError as error:  # pragma: no cover
            raise RuntimeError("selector boundary cache needs safetensors") from error
        if maximum_resident_entities < 0:
            raise ValueError("resident selector entity count cannot be negative")
        if maximum_open_shards <= 0:
            raise ValueError("open selector shard count must be positive")
        self.root = root.resolve()
        self.maximum_resident_entities = maximum_resident_entities
        self.maximum_open_shards = maximum_open_shards
        self._resident: OrderedDict[str, FullSequenceBoundaryEntity] = OrderedDict()
        self._records: dict[str, dict[str, Any]] = {}
        self._handles: OrderedDict[Path, Any] = OrderedDict()
        self._safe_open = safe_open
        inventory = manifest.get("tensor_inventory")
        if not isinstance(inventory, Mapping):
            raise ValueError("selector boundary cache has no tensor inventory")
        by_path: dict[Path, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
        for cache_key, record in inventory.items():
            if not isinstance(cache_key, str) or ":" not in cache_key:
                raise ValueError("selector boundary tensor key drifted")
            role, context_key = cache_key.split(":", 1)
            path = self.root / str(record["partition"]) / str(record["shard"])
            by_path[path].append((f"{role}:{context_key}", record))
        for path in sorted(by_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            for cache_key, record in by_path[path]:
                self._records[cache_key] = {
                    "path": path,
                    "record": dict(record),
                }
        if len(self._records) != len(inventory):
            raise ValueError("selector boundary cache preload omitted an inventory row")

    def context_keys(self) -> frozenset[str]:
        roles_by_context: dict[str, set[str]] = defaultdict(set)
        for key in self._records:
            role, context_key = key.split(":", 1)
            roles_by_context[context_key].add(role)
        required = {
            "boundary_hidden",
            "input_ids",
            "attention_mask",
            "position_ids",
            "visual_indices",
            "text_indices",
        }
        if any(roles != required for roles in roles_by_context.values()):
            raise ValueError("selector boundary context tensor roles drifted")
        return frozenset(roles_by_context)

    def _handle(self, path: Path) -> Any:
        handle = self._handles.pop(path, None)
        if handle is None:
            handle = self._safe_open(str(path), framework="pt", device="cpu")
        self._handles[path] = handle
        while len(self._handles) > self.maximum_open_shards:
            _, evicted = self._handles.popitem(last=False)
            del evicted
        return handle

    def _tensor(self, role: str, key: str) -> Any:
        record = self._records.get(f"{role}:{key}")
        if record is None:
            raise KeyError(f"selector boundary context is missing: {key}")
        tensor = self._handle(record["path"]).get_tensor(record["record"]["tensor"])
        expected = record["record"]
        if (
            list(tensor.shape) != expected["shape"]
            or str(tensor.dtype) != expected["dtype"]
        ):
            raise ValueError("selector boundary tensor metadata drifted")
        return tensor

    def entity(self, key: str) -> FullSequenceBoundaryEntity:
        resident = self._resident.pop(key, None)
        if resident is not None:
            self._resident[key] = resident
            return resident
        entity = FullSequenceBoundaryEntity(
            boundary_hidden_state=self._tensor("boundary_hidden", key),
            input_ids=self._tensor("input_ids", key),
            attention_mask=self._tensor("attention_mask", key),
            position_ids=self._tensor("position_ids", key),
            visual_positions=self._tensor("visual_indices", key),
            text_positions=self._tensor("text_indices", key),
        )
        if self.maximum_resident_entities:
            self._resident[key] = entity
            while len(self._resident) > self.maximum_resident_entities:
                self._resident.popitem(last=False)
        return entity


def _required_context_keys(states: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    values = set()
    for state in states:
        if state["current_image_key"] != state["instruction_text_key"]:
            raise ValueError("selector query visual/text context keys differ")
        if list(state["event_image_keys"]) != list(state["event_text_keys"]):
            raise ValueError("selector event visual/text context keys differ")
        values.add(state["current_image_key"])
        values.update(state["event_image_keys"])
    return frozenset(values)


def _selector_collate(
    states: Sequence[dict[str, Any]],
    *,
    cache: _SelectorBoundaryCache,
    device: Any,
    normalization_floor: float,
    torch: Any,
) -> dict[str, Any]:
    if not states:
        raise ValueError("cannot collate an empty selector batch")
    event_counts = [len(state["candidate_event_step_ids"]) for state in states]
    if any(count <= 0 for count in event_counts):
        raise ValueError("selector state has no candidate events")
    maximum_events = max(event_counts)
    numeric_width = len(states[0]["event_numeric_features"][0])
    event_mask = torch.zeros(
        (len(states), maximum_events), dtype=torch.bool, device=device
    )
    numeric = torch.zeros(
        (len(states), maximum_events, numeric_width),
        dtype=torch.float32,
        device=device,
    )
    query_entities = []
    event_entities = []
    target_rows = []
    for batch_index, state in enumerate(states):
        count = event_counts[batch_index]
        if not (
            len(state["event_image_keys"])
            == len(state["event_text_keys"])
            == len(state["event_numeric_features"])
            == count
        ):
            raise ValueError("selector event features do not align")
        query_entities.append(cache.entity(state["current_image_key"]))
        events = [cache.entity(key) for key in state["event_image_keys"]]
        event_entities.append(tuple(events + [None] * (maximum_events - count)))
        event_mask[batch_index, :count] = True
        row_numeric = torch.tensor(
            state["event_numeric_features"], dtype=torch.float32, device=device
        )
        if tuple(row_numeric.shape) != (count, numeric_width):
            raise ValueError("selector numeric feature geometry drifted")
        numeric[batch_index, :count] = row_numeric
        target_rows.append(
            _targets(state, torch, normalization_floor=normalization_floor)
        )
    maximum_subsets = max(row[0].shape[0] for row in target_rows)
    subset_masks = torch.zeros(
        (len(states), maximum_subsets, maximum_events),
        dtype=torch.bool,
        device=device,
    )
    raw_targets = torch.zeros(
        (len(states), maximum_subsets), dtype=torch.float32, device=device
    )
    for batch_index, target in enumerate(target_rows):
        count = event_counts[batch_index]
        subset_masks[batch_index, : target[0].shape[0], :count] = target[0].to(
            device=device
        )
        raw_targets[batch_index, : target[1].shape[0]] = target[1].to(device=device)
    return {
        "model": {
            "query_entities": tuple(query_entities),
            "event_entities": tuple(event_entities),
            "event_numeric_features": numeric,
            "event_mask": event_mask,
            "subset_masks": subset_masks,
        },
        "raw_targets": raw_targets,
        "scales": torch.tensor(
            [row[3][0] for row in target_rows],
            dtype=torch.float32,
            device=device,
        ),
        "state_ids": tuple(state["state_id"] for state in states),
        "trajectory_ids": tuple(state["trajectory_id"] for state in states),
    }


def _selector_batch_stream(
    states: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    cache: _SelectorBoundaryCache,
    device: Any,
    normalization_floor: float,
    torch: Any,
):
    for offset in range(0, len(states), batch_size):
        selected = list(states[offset : offset + batch_size])
        yield selected, _selector_collate(
            selected,
            cache=cache,
            device=device,
            normalization_floor=normalization_floor,
            torch=torch,
        )


def _heldout_loss_states(
    states: Sequence[dict[str, Any]],
    *,
    normalization_floor: float,
    maximum_base_cardinality: int,
) -> tuple[dict[str, Any], ...]:
    prepared = []
    for state in states:
        groups = conditional_groups(
            state,
            normalization_floor=normalization_floor,
            maximum_base_cardinality=maximum_base_cardinality,
        )
        if groups:
            value = dict(state)
            value["conditional_groups"] = groups
            prepared.append(value)
    if not prepared:
        raise ValueError("selector LoRA heldout split has no complete loss groups")
    return tuple(prepared)


def _evaluate_holdout_loss(
    model: SelectorSideLoRAMarginalPredictor,
    states: Sequence[dict[str, Any]],
    *,
    cache: _SelectorBoundaryCache,
    batch_size: int,
    device: Any,
    normalization_floor: float,
    loss_config: Mapping[str, Any],
    torch: Any,
) -> tuple[dict[str, float], float]:
    totals = defaultdict(float)
    denominator = 0.0
    model.eval()
    with torch.inference_mode():
        for selected, batch in _selector_batch_stream(
            states,
            batch_size=batch_size,
            cache=cache,
            device=device,
            normalization_floor=normalization_floor,
            torch=torch,
        ):
            batch = _attach_conditional_supervision(batch, selected, torch=torch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(**batch["model"])
                _, metrics = group_equal_direct_loss(
                    predictions,
                    batch,
                    loss_config=loss_config,
                    reduction="mean",
                    torch=torch,
                )
            weight = float(batch["conditional_group_mask"].sum())
            denominator += weight
            for key, value in metrics.items():
                if key != "conditional_group_count":
                    totals[key] += value * weight
    if denominator <= 0.0:
        raise RuntimeError("selector LoRA heldout loss denominator is empty")
    return dict(totals), denominator


def _slice_encoded(
    encoded: EncodedConditionalMarginalState, index: int
) -> EncodedConditionalMarginalState:
    return EncodedConditionalMarginalState(
        query=encoded.query[index : index + 1],
        events=encoded.events[index : index + 1],
        event_mask=encoded.event_mask[index : index + 1],
    )


def _rollout_heldout(
    model: SelectorSideLoRAMarginalPredictor,
    states: Sequence[dict[str, Any]],
    *,
    cache: _SelectorBoundaryCache,
    device: Any,
    normalization_floor: float,
    entity_microbatch_size: int,
    torch: Any,
) -> tuple[dict[str, Any], ...]:
    records = []
    model.eval()
    with torch.inference_mode():
        for state in sorted(states, key=lambda row: row["state_id"]):
            batch = _selector_collate(
                [state],
                cache=cache,
                device=device,
                normalization_floor=normalization_floor,
                torch=torch,
            )
            model_inputs = dict(batch["model"])
            model_inputs.pop("subset_masks")
            tokens = model.encode_predictor_tokens(
                query_entities=model_inputs["query_entities"],
                event_entities=model_inputs["event_entities"],
                event_mask=model_inputs["event_mask"],
                entity_microbatch_size=entity_microbatch_size,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                encoded = model.predictor.encode_state_once(
                    **tokens.as_kwargs(),
                    event_numeric_features=model_inputs["event_numeric_features"],
                    event_mask=model_inputs["event_mask"],
                )
            encoded = _slice_encoded(encoded, 0)
            event_ids = tuple(state["candidate_event_step_ids"])
            by_event = {event_id: index for index, event_id in enumerate(event_ids)}

            def score(chosen: tuple[int, ...]) -> tuple[float, ...]:
                selected_mask = torch.zeros_like(encoded.event_mask, dtype=torch.bool)
                for event_id in chosen:
                    selected_mask[0, by_event[event_id]] = True
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    values = model.predictor.score_encoded_candidates(
                        encoded, selected_mask
                    )[0, : len(event_ids) + 1]
                result = tuple(float(value) for value in values.tolist())
                if any(not math.isfinite(value) for value in result):
                    raise RuntimeError(
                        "selector LoRA rollout emitted non-finite scores"
                    )
                return result

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
    return tuple(records)


def _selector_model_family(phase: str) -> str:
    if phase not in PHASES:
        raise ValueError("unknown selector LoRA phase")
    return f"selector_lora_v1_{phase}"


def _validate_parent_phase_summary(
    summary: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    expected_identity: Mapping[str, Any],
) -> None:
    selected = summary.get("selected_checkpoint")
    identity = summary.get("identity")
    if (
        summary.get("status") != "COMPLETED_SELECTOR_LORA_SELECTED_BY_TRUE_RECOVERY"
        or summary.get("phase") != PHASE_LORA_ONLY
        or not isinstance(selected, Mapping)
        or selected.get("sha256") != checkpoint_sha256
        or not isinstance(identity, Mapping)
    ):
        raise ValueError("joint selector LoRA parent selection binding drifted")
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"joint selector LoRA parent identity drifted at {key}")


def _set_phase_training_mode(
    model: SelectorSideLoRAMarginalPredictor, *, phase: str
) -> None:
    """Enable only the stochastic modules that the current phase optimizes."""
    if phase not in PHASES:
        raise ValueError("unknown selector LoRA phase")
    model.train()
    # note (luojiaxuan): Frozen Qwen layers must stay in inference mode while
    # gradients flow through their LoRA branches. In phase one the frozen head
    # must also keep dropout disabled, otherwise the LoRA learns through a
    # stochastic head that differs from the heldout rollout path.
    model.top_language_model.eval()
    model.predictor.train(phase == PHASE_JOINT)


def _publish_schedule(
    output_root: Path,
    plans: Sequence[Mapping[str, Any]],
    *,
    phase: str,
) -> dict[str, Any]:
    schedule = merge_control_truth_schedule(plans)
    schedule["model_family"] = _selector_model_family(phase)
    schedule["status"] = (
        "COMPLETE_SELECTOR_LORA_HELDOUT_TRUTH"
        if schedule["missing_coalition_count"] == 0
        else "PENDING_SELECTOR_LORA_HELDOUT_TRUTH"
    )
    _write_atomic(output_root / "heldout-truth-schedule.json", _signed(schedule))
    return schedule


def _selection(
    plans: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    selection = config["checkpoint_selection"]
    return control_epoch_barrier(
        plans,
        patience=int(selection["patience"]),
        minimum_delta=float(selection["minimum_delta"]),
        tie_breaker_minimum_delta=float(
            selection.get("tie_breaker_minimum_delta", 0.0)
        ),
    )


def _verify_epoch_checkpoint(output_root: Path, plan: Mapping[str, Any]) -> None:
    checkpoint = plan["checkpoint"]
    path = output_root / checkpoint["path"]
    if (
        not path.is_file()
        or path.stat().st_size != int(checkpoint["byte_count"])
        or _sha256_file(path) != checkpoint["sha256"]
    ):
        raise ValueError("selector LoRA epoch checkpoint drifted")


def _rebuild_plans(
    plans: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    states_by_id: Mapping[str, Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    normalization_floor: float,
    supplemental: Mapping[str, Mapping[tuple[int, ...], float]],
) -> tuple[dict[str, Any], ...]:
    rebuilt = []
    for plan in plans:
        _verify_epoch_checkpoint(output_root, plan)
        value = build_control_epoch_truth(
            epoch=int(plan["epoch"]),
            checkpoint=plan["checkpoint"],
            rollout_records=plan["records"],
            states_by_id=states_by_id,
            split_manifest=split_manifest,
            normalization_floor=normalization_floor,
            supplemental=supplemental,
        )
        _write_atomic(_epoch_plan_path(output_root, int(plan["epoch"])), _signed(value))
        rebuilt.append(value)
    return tuple(rebuilt)


def _load_top_language_model(
    *,
    model_dir: Path,
    snapshot_manifest: Path,
    device: Any,
    layer_count: int,
    torch: Any,
) -> Any:
    from transformers import AutoModelForImageTextToText

    from causalcache.policy.gui_owl_v2_vision import verify_frozen_vision_runtime
    from causalcache.set_utility_selector_branch import qwen3vl_language_model

    identity = verify_frozen_vision_runtime(
        model_dir=model_dir, expected_snapshot_manifest=snapshot_manifest
    )
    policy = AutoModelForImageTextToText.from_pretrained(
        identity.model_dir,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    policy.eval().requires_grad_(False)
    language_model = qwen3vl_language_model(policy)
    if layer_count != 4:
        raise ValueError("selector LoRA v1 requires exactly four top layers")
    language_model.layers = torch.nn.ModuleList(
        tuple(language_model.layers[-layer_count:])
    )
    policy.model.language_model = None
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return language_model


def _load_training_state(
    *,
    input_root: Path,
    boundary_root: Path,
    config: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    input_manifest = verify_formal_training_input(
        input_root,
        expected_content_sha256=config["input"]["training_input_content_sha256"],
    )
    boundary_manifest = _verify_boundary_manifest(
        boundary_root,
        expected_content_sha256=config["input"][
            "selector_boundary_cache_content_sha256"
        ],
        expected_contextual_input_sha256=config["input"][
            "contextual_input_content_sha256"
        ],
        expected_extraction_config_sha256=config["input"][
            "boundary_extraction_config_sha256"
        ],
    )
    states = tuple(
        row
        for row in _read_jsonl(input_root / input_manifest["states_jsonl"])
        if row["role"] == "train"
    )
    optimization, heldout = apply_split_manifest(states, split_manifest)
    if set(_required_context_keys((*optimization, *heldout))) != {
        key.split(":", 1)[1] for key in boundary_manifest["tensor_inventory"]
    }:
        raise ValueError(
            "selector boundary cache does not exactly cover training states"
        )
    return input_manifest, boundary_manifest, optimization, heldout


def _validate_config(
    config: Mapping[str, Any], *, phase: str, variant: str
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError("selector LoRA phase is invalid")
    selected = config.get("variants", {}).get(variant)
    if not isinstance(selected, Mapping):
        raise ValueError("selector LoRA variant is unknown")
    lora = config.get("selector_lora", {})
    phase_config = config.get("phases", {}).get(phase)
    model = selected.get("model")
    input_contract = config.get("input")
    if not isinstance(input_contract, Mapping):
        raise ValueError("selector LoRA input contract is missing")
    for key in (
        "training_input_content_sha256",
        "contextual_input_content_sha256",
        "selector_boundary_cache_content_sha256",
        "initial_checkpoint_sha256",
        "source_manifest_file_sha256",
        "train_heldout_manifest_content_sha256",
        "teacher_snapshot_manifest_sha256",
        "boundary_extraction_config_sha256",
    ):
        _require_sha256(input_contract.get(key), label=f"selector LoRA {key}")
    if (
        lora.get("trainable_layer_count") != 4
        or tuple(lora.get("target_modules", ()))
        != ("q_proj", "k_proj", "v_proj", "o_proj")
        or type(lora.get("rank")) is not int
        or int(lora["rank"]) <= 0
        or float(lora.get("alpha", 0.0)) <= 0.0
        or not isinstance(phase_config, Mapping)
        or not 1 <= int(phase_config.get("epochs", 0)) <= 4
        or model.get("family") != "set_transformer"
        or model.get("preserve_entity_latents") is not True
    ):
        raise ValueError("selector LoRA architecture or phase contract drifted")
    if phase == PHASE_LORA_ONLY and int(phase_config["epochs"]) != 1:
        raise ValueError("selector LoRA-only phase must contain exactly one epoch")
    allowed_world_sizes = tuple(int(value) for value in selected["allowed_world_sizes"])
    if not allowed_world_sizes or any(value <= 0 for value in allowed_world_sizes):
        raise ValueError("selector LoRA allowed world sizes are invalid")
    return dict(selected)


def _fit(args: argparse.Namespace) -> None:
    try:
        import torch
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise RuntimeError("selector LoRA training requires PyTorch") from error
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("selector LoRA training requires CUDA")
    config_path = args.config.resolve()
    config = _read_json(config_path)
    variant = _validate_config(config, phase=args.phase, variant=args.variant)
    split_manifest = _load_split_manifest(args.split_manifest.resolve(), config)
    if split_manifest is None:
        raise ValueError("selector LoRA training requires the frozen split manifest")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if world_size not in tuple(int(value) for value in variant["allowed_world_sizes"]):
        raise ValueError("selector LoRA DDP world size is outside the config")
    if distributed:
        if args.device != "cuda":
            raise RuntimeError("distributed selector LoRA training needs --device cuda")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    input_root = args.input_root.resolve()
    boundary_root = args.boundary_cache_root.resolve()
    input_manifest, boundary_manifest, optimization_states, heldout_states = (
        _load_training_state(
            input_root=input_root,
            boundary_root=boundary_root,
            config=config,
            split_manifest=split_manifest,
        )
    )
    training = config["training"]
    examples = prepare_control_examples(
        optimization_states,
        normalization_floor=float(training["normalization_floor"]),
        maximum_base_cardinality=int(training["maximum_base_cardinality"]),
    )
    heldout_loss_states = _heldout_loss_states(
        heldout_states,
        normalization_floor=float(training["normalization_floor"]),
        maximum_base_cardinality=int(training["maximum_base_cardinality"]),
    )
    observed_inventory = formal_group_inventory(examples)
    expected_inventory = training.get("exact_optimizer_inventory")
    if expected_inventory is not None and observed_inventory != expected_inventory:
        raise ValueError("selector LoRA exact optimizer inventory drifted")
    seed = int(training["seed"])
    _seed_training_runtime(torch, seed)
    torch.set_float32_matmul_precision("high")
    attention_backend = _configure_attention_backend(torch, training)

    initial_checkpoint = args.initial_checkpoint.resolve()
    if _sha256_file(initial_checkpoint) != config["input"]["initial_checkpoint_sha256"]:
        raise ValueError("selector LoRA initial Set checkpoint drifted")
    predictor = TokenConditionalMarginalPredictor(
        TokenUtilityModelConfig(**variant["model"])
    ).to(device)
    predictor.load_state_dict(load_file(str(initial_checkpoint)), strict=True)
    snapshot_manifest = args.snapshot_manifest.resolve()
    snapshot_manifest_sha256 = _verify_snapshot_manifest(
        snapshot_manifest,
        expected_sha256=config["input"]["teacher_snapshot_manifest_sha256"],
    )
    top_language_model = _load_top_language_model(
        model_dir=args.model_dir.resolve(),
        snapshot_manifest=snapshot_manifest,
        device=device,
        layer_count=int(config["selector_lora"]["trainable_layer_count"]),
        torch=torch,
    )
    checkpoint_model = SelectorSideLoRAMarginalPredictor(
        top_language_model=top_language_model,
        predictor=predictor,
        lora_rank=int(config["selector_lora"]["rank"]),
        lora_alpha=float(config["selector_lora"]["alpha"]),
        entity_microbatch_size=int(variant["entity_microbatch_size"]),
    ).to(device)
    parent_checkpoint_sha256 = None
    parent_summary_sha256 = None
    if args.phase == PHASE_JOINT:
        if (
            args.parent_selector_checkpoint is None
            or args.parent_selector_summary is None
        ):
            raise ValueError(
                "joint selector LoRA phase requires the selected LoRA-only checkpoint "
                "and summary"
            )
        parent_path = args.parent_selector_checkpoint.resolve()
        parent_checkpoint_sha256 = _sha256_file(parent_path)
        parent_summary_path = args.parent_selector_summary.resolve()
        parent_summary = _read_signed_json(parent_summary_path)
        parent_summary_sha256 = _sha256_file(parent_summary_path)
        _validate_parent_phase_summary(
            parent_summary,
            checkpoint_sha256=parent_checkpoint_sha256,
            expected_identity={
                "boundary_cache_content_sha256": boundary_manifest["content_sha256"],
                "boundary_extraction_config_sha256": boundary_manifest[
                    "config_sha256"
                ],
                "config_sha256": _sha256_file(config_path),
                "initial_checkpoint_sha256": config["input"][
                    "initial_checkpoint_sha256"
                ],
                "input_content_sha256": input_manifest["content_sha256"],
                "model_family": _selector_model_family(PHASE_LORA_ONLY),
                "optimizer_inventory_content_sha256": observed_inventory[
                    "content_sha256"
                ],
                "parent_selector_checkpoint_sha256": None,
                "parent_selector_summary_sha256": None,
                "phase": PHASE_LORA_ONLY,
                "split_manifest_content_sha256": split_manifest["content_sha256"],
                "teacher_snapshot_manifest_sha256": snapshot_manifest_sha256,
                "variant": args.variant,
            },
        )
        _load_compact_selector_state(
            checkpoint_model, load_file(str(parent_path), device=str(device))
        )
        checkpoint_model.set_head_trainable(True)
    else:
        if (
            args.parent_selector_checkpoint is not None
            or args.parent_selector_summary is not None
        ):
            raise ValueError("LoRA-only phase cannot accept a parent selector artifact")
        checkpoint_model.set_head_trainable(False)

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
    lora_parameters = [value for _, value in checkpoint_model.named_lora_parameters()]
    if args.phase == PHASE_LORA_ONLY:
        if {
            id(parameter)
            for parameter in checkpoint_model.parameters()
            if parameter.requires_grad
        } != set(map(id, lora_parameters)):
            raise RuntimeError("LoRA-only phase exposed a non-LoRA trainable parameter")
        parameter_groups = [
            {
                "name": "selector_lora",
                "params": lora_parameters,
                "lr": float(variant["lora_learning_rate"]),
                "weight_decay": float(variant["weight_decay"]),
            }
        ]
    else:
        parameter_groups = list(
            checkpoint_model.optimizer_parameter_groups(
                lora_learning_rate=float(variant["lora_learning_rate"]),
                head_learning_rate=float(variant["head_learning_rate"]),
                weight_decay=float(variant["weight_decay"]),
            )
        )
    parameters = tuple(
        parameter for group in parameter_groups for parameter in group["params"]
    )
    optimizer = torch.optim.AdamW(parameter_groups)
    epochs = int(config["phases"][args.phase]["epochs"])
    batch_size = int(variant["per_device_batch_size"])
    accumulation = int(variant["gradient_accumulation_steps"])
    balancing = training["balancing"]

    def epoch_examples(epoch: int) -> tuple[dict[str, Any], ...]:
        sampled = balanced_control_epoch(
            examples,
            seed=seed + epoch,
            sample_count=int(balancing["samples_per_epoch"]),
            power=float(balancing["power"]),
            maximum_weight=float(balancing["maximum_weight"]),
        )
        return pack_control_examples(
            sampled,
            maximum_groups_per_state=int(balancing["maximum_groups_per_encoded_state"]),
            seed=seed + 100_000 + epoch,
        )

    counts = [len(epoch_examples(epoch)) for epoch in range(1, epochs + 1)]
    steps = [
        _optimizer_steps(
            count,
            batch_size=batch_size,
            accumulation=accumulation,
            world_size=world_size,
        )
        for count in counts
    ]
    total_steps = sum(steps)
    warmup_steps = max(1, round(total_steps * float(training["warmup_ratio"])))
    minimum_lr_ratio = float(training["minimum_lr_ratio"])

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    identity = {
        "boundary_cache_content_sha256": boundary_manifest["content_sha256"],
        "boundary_extraction_config_sha256": boundary_manifest["config_sha256"],
        "config_sha256": _sha256_file(config_path),
        "initial_checkpoint_sha256": config["input"]["initial_checkpoint_sha256"],
        "input_content_sha256": input_manifest["content_sha256"],
        "model_family": _selector_model_family(args.phase),
        "optimizer_inventory_content_sha256": observed_inventory["content_sha256"],
        "parent_selector_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_selector_summary_sha256": parent_summary_sha256,
        "phase": args.phase,
        "split_manifest_content_sha256": split_manifest["content_sha256"],
        "teacher_snapshot_manifest_sha256": snapshot_manifest_sha256,
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
            saver=_compact_safetensors_saver,
        )
        if not args.resume and manager.manifest["epochs"]:
            raise FileExistsError("selector LoRA checkpoint history needs --resume")
    if distributed:
        torch.distributed.barrier()
    (output_root / "heldout-rollouts").mkdir(parents=True, exist_ok=True)
    cache = _SelectorBoundaryCache(
        boundary_root,
        boundary_manifest,
        maximum_resident_entities=int(variant.get("resident_entity_cache_size", 16)),
        maximum_open_shards=int(variant.get("maximum_open_boundary_shards", 64)),
    )
    if not _required_context_keys((*optimization_states, *heldout_states)).issubset(
        cache.context_keys()
    ):
        raise ValueError("selector boundary cache cannot serve the frozen split")
    heldout_by_id = {row["state_id"]: row for row in heldout_states}
    truth_sources = _truth_sources(args.truth_source)
    supplemental: dict[str, dict[tuple[int, ...], float]] = {}
    start_epoch = 1
    history = []
    barrier_action = CONTINUE_TRAINING
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
        _load_compact_selector_state(checkpoint_model, snapshot["model"])
        optimizer.load_state_dict(snapshot["optimizer"])
        scheduler.load_state_dict(snapshot["scheduler"])
        random.setstate(snapshot["python_rng_state"])
        restore_rng_states(snapshot, torch=torch)
        start_epoch = int(snapshot["epoch"]) + 1
        completed, _ = _partition_resume_plans(
            _read_epoch_plans(output_root), snapshot_epoch=int(snapshot["epoch"])
        )
        supplemental = _load_supplemental_truth(
            truth_sources,
            allowed_state_ids=set(heldout_by_id),
            states_by_id=heldout_by_id,
            expected_model_family=_selector_model_family(args.phase),
            expected_epoch_checkpoints=_plan_checkpoint_bindings(completed),
            expected_input_content_sha256=input_manifest["content_sha256"],
            expected_heldout_manifest_content_sha256=split_manifest["content_sha256"],
            expected_source_manifest_file_sha256=config["input"][
                "source_manifest_file_sha256"
            ],
        )
        if rank == 0:
            progress = _read_signed_json(output_root / "training-progress.json")
            if progress.get("identity") != identity:
                raise ValueError("selector LoRA resume progress identity drifted")
            history = [
                dict(row)
                for row in progress.get("history", ())
                if int(row["epoch"]) <= int(snapshot["epoch"])
            ]
            if [int(row["epoch"]) for row in history] != list(
                range(1, int(snapshot["epoch"]) + 1)
            ):
                raise ValueError("selector LoRA resume progress epochs drifted")
            rebuilt = _rebuild_plans(
                completed,
                output_root=output_root,
                states_by_id=heldout_by_id,
                split_manifest=split_manifest,
                normalization_floor=float(
                    config["checkpoint_selection"]["normalization_floor"]
                ),
                supplemental=supplemental,
            )
            schedule = _publish_schedule(output_root, rebuilt, phase=args.phase)
            barrier_action, selection = _selection(rebuilt, config)
            history[-1]["selection"] = selection
            history[-1]["truth_missing_coalition_count"] = schedule[
                "missing_coalition_count"
            ]
            _write_atomic(
                output_root / "training-progress.json",
                _signed(
                    {
                        "history": history,
                        "identity": identity,
                        "schema_version": "causalcache.selector_lora_progress.v1",
                        "status": barrier_action,
                    }
                ),
            )
        if distributed:
            payload = [barrier_action]
            torch.distributed.broadcast_object_list(payload, src=0)
            barrier_action = str(payload[0])
    elif has_published_resume_generation(output_root):
        raise FileExistsError("selector LoRA resume generation exists without --resume")

    termination_reason = "maximum_epochs"
    if barrier_action == WAITING_FOR_TRUTH:
        termination_reason = "heldout_truth_barrier"
        start_epoch = epochs + 1
    elif barrier_action == EARLY_STOP:
        termination_reason = "true_recovery_early_stopping"
        start_epoch = epochs + 1
    started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        _set_phase_training_mode(checkpoint_model, phase=args.phase)
        global_examples = epoch_examples(epoch)
        if distributed:
            selected_examples, _ = _distributed_epoch_shard(
                global_examples,
                per_device_batch_size=batch_size,
                gradient_accumulation_steps=accumulation,
                rank=rank,
                world_size=world_size,
            )
        else:
            padding = (-len(global_examples)) % (batch_size * accumulation)
            selected_examples = global_examples + tuple(
                global_examples[index % len(global_examples)]
                for index in range(padding)
            )
        optimizer.zero_grad(set_to_none=True)
        totals = defaultdict(float)
        denominator = 0.0
        batches = iter(
            _selector_batch_stream(
                selected_examples,
                batch_size=batch_size,
                cache=cache,
                device=device,
                normalization_floor=float(training["normalization_floor"]),
                torch=torch,
            )
        )
        while True:
            raw_window = tuple(islice(batches, accumulation))
            if not raw_window:
                break
            window = tuple(
                (
                    selected,
                    _attach_conditional_supervision(batch, selected, torch=torch),
                )
                for selected, batch in raw_window
            )
            group_count = torch.tensor(
                sum(
                    float(batch["conditional_group_mask"].sum()) for _, batch in window
                ),
                dtype=torch.float64,
                device=device,
            )
            if distributed:
                torch.distributed.all_reduce(group_count)
            backward_scale = _group_sum_backward_scale(
                global_group_count=float(group_count.item()), world_size=world_size
            )
            for window_index, (_, batch) in enumerate(window):
                synchronize = window_index + 1 == len(window)
                sync_context = (
                    nullcontext() if not distributed or synchronize else model.no_sync()
                )
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predictions = model(**batch["model"])
                        loss_sum, metrics = group_equal_direct_loss(
                            predictions,
                            batch,
                            loss_config=training["loss"],
                            reduction="sum",
                            torch=torch,
                        )
                    (loss_sum * backward_scale).backward()
                weight = float(batch["conditional_group_mask"].sum())
                denominator += weight
                for key, value in metrics.items():
                    if key != "conditional_group_count":
                        totals[key] += value * weight
            torch.nn.utils.clip_grad_norm_(
                parameters, float(training["maximum_gradient_norm"])
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        train_metrics = (
            _distributed_metric_average(totals, denominator, device=device, torch=torch)
            if distributed
            else {key: value / denominator for key, value in sorted(totals.items())}
        )
        global_group_count = torch.tensor(
            denominator, dtype=torch.float64, device=device
        )
        if distributed:
            torch.distributed.all_reduce(global_group_count)
        train_metrics["conditional_group_count"] = float(global_group_count.item())
        if not all(math.isfinite(value) for value in train_metrics.values()):
            raise RuntimeError("selector LoRA produced non-finite train metrics")

        holdout_totals, holdout_denominator = _evaluate_holdout_loss(
            checkpoint_model,
            tuple(heldout_loss_states[rank::world_size]),
            cache=cache,
            batch_size=int(variant.get("heldout_loss_batch_size", 1)),
            device=device,
            normalization_floor=float(training["normalization_floor"]),
            loss_config=training["loss"],
            torch=torch,
        )
        holdout_metrics = (
            _distributed_metric_average(
                holdout_totals,
                holdout_denominator,
                device=device,
                torch=torch,
            )
            if distributed
            else {
                key: value / holdout_denominator
                for key, value in sorted(holdout_totals.items())
            }
        )
        heldout_group_count = torch.tensor(
            holdout_denominator, dtype=torch.float64, device=device
        )
        if distributed:
            torch.distributed.all_reduce(heldout_group_count)
        holdout_metrics["conditional_group_count"] = float(heldout_group_count.item())
        if not all(math.isfinite(value) for value in holdout_metrics.values()):
            raise RuntimeError("selector LoRA produced non-finite heldout metrics")

        barrier_action = CONTINUE_TRAINING
        checkpoint = None
        if rank == 0:
            assert manager is not None
            checkpoint = _save_or_verify_epoch_checkpoint(
                manager,
                checkpoint_model,
                epoch=epoch,
                output_root=output_root,
            )
        if distributed:
            checkpoint_payload = [checkpoint]
            torch.distributed.broadcast_object_list(checkpoint_payload, src=0)
            checkpoint = checkpoint_payload[0]
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("selector LoRA checkpoint publication failed")
        local_heldout = tuple(heldout_states[rank::world_size])
        local_records = _rollout_heldout(
            checkpoint_model,
            local_heldout,
            cache=cache,
            device=device,
            normalization_floor=float(training["normalization_floor"]),
            entity_microbatch_size=int(variant["entity_microbatch_size"]),
            torch=torch,
        )
        if distributed:
            gathered_records: list[tuple[dict[str, Any], ...] | None] = [
                None
            ] * world_size
            torch.distributed.all_gather_object(gathered_records, local_records)
            records = tuple(
                row
                for shard in gathered_records
                for row in (() if shard is None else shard)
            )
        else:
            records = local_records
        if rank == 0:
            records = tuple(sorted(records, key=lambda row: row["state_id"]))
            if len(records) != len(heldout_states) or len(
                {row["state_id"] for row in records}
            ) != len(records):
                raise RuntimeError("selector LoRA heldout rollout inventory drifted")
            existing = _read_existing_epoch_plan(
                output_root, epoch=epoch, checkpoint=checkpoint
            )
            records = existing["records"] if existing is not None else records
            plan = build_control_epoch_truth(
                epoch=epoch,
                checkpoint=checkpoint,
                rollout_records=records,
                states_by_id=heldout_by_id,
                split_manifest=split_manifest,
                normalization_floor=float(
                    config["checkpoint_selection"]["normalization_floor"]
                ),
                supplemental=supplemental,
            )
            _write_atomic(_epoch_plan_path(output_root, epoch), _signed(plan))
            plans = _read_epoch_plans(output_root)
            schedule = _publish_schedule(output_root, plans, phase=args.phase)
            barrier_action, selection = _selection(plans, config)
            record = {
                "barrier_action": barrier_action,
                "epoch": epoch,
                "learning_rates": scheduler.get_last_lr(),
                "selection": selection,
                "train": train_metrics,
                "train_heldout_loss": holdout_metrics,
                "truth_missing_coalition_count": schedule["missing_coalition_count"],
            }
            history.append(record)
            _write_atomic(
                output_root / "training-progress.json",
                _signed(
                    {
                        "history": history,
                        "identity": identity,
                        "schema_version": "causalcache.selector_lora_progress.v1",
                        "status": barrier_action,
                    }
                ),
            )
            print(json.dumps(record, sort_keys=True), flush=True)
        if distributed:
            payload = [barrier_action]
            torch.distributed.broadcast_object_list(payload, src=0)
            barrier_action = str(payload[0])
            torch.distributed.barrier()
        save_resume_generation_collective(
            output_root,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
            model=_CompactStateView(checkpoint_model),
            optimizer=optimizer,
            scheduler=scheduler,
            identity=identity,
            distributed=distributed,
            torch=torch,
        )
        if barrier_action == WAITING_FOR_TRUTH:
            termination_reason = "heldout_truth_barrier"
            break
        if barrier_action == EARLY_STOP:
            termination_reason = "true_recovery_early_stopping"
            break

    if rank == 0:
        plans = _read_epoch_plans(output_root)
        barrier_action, selection = _selection(plans, config)
        summary = {
            "attention_backend": attention_backend,
            "balancing_inventory": control_balancing_inventory(examples),
            "boundary_cache_content_sha256": boundary_manifest["content_sha256"],
            "checkpoint_count": len(plans),
            "elapsed_seconds_this_invocation": time.time() - started,
            "evaluation_or_test_records_loaded": False,
            "heldout_loss_state_count": len(heldout_loss_states),
            "heldout_state_count": len(heldout_states),
            "identity": identity,
            "optimization_group_count": len(examples),
            "phase": args.phase,
            "schema_version": "causalcache.selector_lora_training.v1",
            "selection": selection,
            "selected_checkpoint": _selected_checkpoint(plans, selection),
            "status": (
                WAITING_FOR_TRUTH
                if barrier_action == WAITING_FOR_TRUTH
                else "COMPLETED_SELECTOR_LORA_SELECTED_BY_TRUE_RECOVERY"
            ),
            "termination_reason": termination_reason,
        }
        _write_atomic(output_root / "summary.json", _signed(summary))
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--boundary-cache-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-selector-checkpoint", type=Path)
    parser.add_argument("--parent-selector-summary", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--truth-source", action="append", default=[])
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", action="store_true")
    _fit(parser.parse_args())


if __name__ == "__main__":
    main()
