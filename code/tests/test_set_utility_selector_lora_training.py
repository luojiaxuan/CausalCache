from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.train_set_utility_selector_lora_v1 import (
    PHASE_JOINT,
    PHASE_LORA_ONLY,
    _SelectorBoundaryCache,
    _compact_selector_state_dict,
    _load_compact_selector_state,
    _required_context_keys,
    _selector_collate,
    _selector_model_family,
    _set_phase_training_mode,
    _validate_config,
    _validate_parent_phase_summary,
    _verify_boundary_manifest,
    _verify_snapshot_manifest,
)
from scripts.train_set_utility_structured_marginal import _load_split_manifest
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes
from causalcache.set_utility_selector_boundary_cache import (
    SELECTOR_BOUNDARY_CACHE_STATUS,
)
from causalcache.set_utility_heldout_truth_schedule import _validate_signed_schedule


class _FakePredictor:
    def state_dict(self):
        return {"weight": object(), "bias": object()}


class _FakeSelector:
    def __init__(self) -> None:
        self.predictor = _FakePredictor()
        self.loaded = None

    def state_dict(self):
        return {
            "predictor.weight": object(),
            "predictor.bias": object(),
            "top_language_model.layers.0.self_attn.q_proj.base.weight": object(),
            "top_language_model.layers.0.self_attn.q_proj.lora_a.weight": object(),
            "top_language_model.layers.0.self_attn.q_proj.lora_b.weight": object(),
        }

    def named_lora_parameters(self):
        return (("a", object()), ("b", object()))

    def load_state_dict(self, state, *, strict):
        assert strict is False
        self.loaded = dict(state)
        return SimpleNamespace(
            unexpected_keys=[],
            missing_keys=["top_language_model.layers.0.self_attn.q_proj.base.weight"],
        )


def _state() -> dict:
    return {
        "candidate_event_step_ids": [1, 2],
        "current_image_key": "query",
        "distance_rows": [
            {"coalition_event_step_ids": [], "distance": 2.0},
            {"coalition_event_step_ids": [1], "distance": 1.0},
            {"coalition_event_step_ids": [2], "distance": 1.5},
        ],
        "event_image_keys": ["event-1", "event-2"],
        "event_numeric_features": [[0.0, 1.0], [1.0, 0.0]],
        "event_text_keys": ["event-1", "event-2"],
        "instruction_text_key": "query",
        "state_id": "trajectory:decision:003",
        "trajectory_id": "trajectory",
    }


def _config() -> dict:
    return {
        "input": {
            "contextual_input_content_sha256": "a" * 64,
            "boundary_extraction_config_sha256": "0" * 64,
            "initial_checkpoint_sha256": "b" * 64,
            "selector_boundary_cache_content_sha256": "c" * 64,
            "source_manifest_file_sha256": "d" * 64,
            "train_heldout_manifest_content_sha256": "e" * 64,
            "teacher_snapshot_manifest_sha256": "1" * 64,
            "training_input_content_sha256": "f" * 64,
        },
        "selector_lora": {
            "alpha": 16,
            "rank": 8,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "trainable_layer_count": 4,
        },
        "phases": {
            PHASE_LORA_ONLY: {"epochs": 1},
            PHASE_JOINT: {"epochs": 4},
        },
        "training": {
            "exact_optimizer_inventory_content_sha256": "2" * 64,
        },
        "variants": {
            "set": {
                "allowed_world_sizes": [1, 2, 4, 6, 8],
                "model": {
                    "family": "set_transformer",
                    "preserve_entity_latents": True,
                },
            }
        },
    }


def test_compact_selector_checkpoint_contains_only_head_and_lora() -> None:
    model = _FakeSelector()
    state = _compact_selector_state_dict(model)
    assert set(state) == {
        "predictor.weight",
        "predictor.bias",
        "top_language_model.layers.0.self_attn.q_proj.lora_a.weight",
        "top_language_model.layers.0.self_attn.q_proj.lora_b.weight",
    }
    _load_compact_selector_state(model, state)
    assert model.loaded == state
    with pytest.raises(ValueError, match="inventory"):
        _load_compact_selector_state(model, {"predictor.weight": object()})


def test_selector_context_contract_requires_shared_entity_keys() -> None:
    assert _required_context_keys((_state(),)) == {
        "query",
        "event-1",
        "event-2",
    }
    bad_query = _state()
    bad_query["instruction_text_key"] = "other-query"
    with pytest.raises(ValueError, match="query visual/text"):
        _required_context_keys((bad_query,))
    bad_event = _state()
    bad_event["event_text_keys"] = ["event-1", "other-event"]
    with pytest.raises(ValueError, match="event visual/text"):
        _required_context_keys((bad_event,))


def test_selector_phase_contract_and_model_family_are_versioned() -> None:
    config = _config()
    assert _validate_config(config, phase=PHASE_LORA_ONLY, variant="set")
    assert _validate_config(config, phase=PHASE_JOINT, variant="set")
    assert _selector_model_family(PHASE_LORA_ONLY) == ("selector_lora_v1_lora_only")
    assert _selector_model_family(PHASE_JOINT) == "selector_lora_v1_joint"
    config["phases"][PHASE_LORA_ONLY]["epochs"] = 2
    with pytest.raises(ValueError, match="exactly one epoch"):
        _validate_config(config, phase=PHASE_LORA_ONLY, variant="set")


def test_committed_executable_config_matches_frozen_split() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (
            repository_root
            / "code/configs/causalcache_set_utility_selector_lora_training_v1.json"
        ).read_text(encoding="utf-8")
    )
    variant = "set_transformer_selector_lora_top4_r8"
    assert _validate_config(config, phase=PHASE_LORA_ONLY, variant=variant)
    assert _validate_config(config, phase=PHASE_JOINT, variant=variant)
    assert config["input"]["selector_boundary_cache_content_sha256"] != "0" * 64
    split = _load_split_manifest(
        repository_root / "data/manifests/set_utility_train_heldout_v2.json",
        config,
    )
    assert split["checkpoint_selection"]["patience"] == 3


def test_phase_one_keeps_frozen_head_and_backbone_in_eval_mode() -> None:
    torch = pytest.importorskip("torch")

    class _ModeProbe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.top_language_model = torch.nn.Sequential(torch.nn.Dropout(0.5))
            self.predictor = torch.nn.Sequential(torch.nn.Dropout(0.5))

    model = _ModeProbe()
    _set_phase_training_mode(model, phase=PHASE_LORA_ONLY)
    assert model.training is True
    assert model.top_language_model.training is False
    assert model.predictor.training is False
    _set_phase_training_mode(model, phase=PHASE_JOINT)
    assert model.top_language_model.training is False
    assert model.predictor.training is True


def test_joint_phase_parent_is_bound_to_phase_one_identity() -> None:
    expected_identity = {
        "boundary_extraction_config_sha256": "c" * 64,
        "config_sha256": "a" * 64,
        "model_family": "selector_lora_v1_lora_only",
        "phase": PHASE_LORA_ONLY,
        "teacher_snapshot_manifest_sha256": "d" * 64,
    }
    summary = {
        "identity": expected_identity,
        "phase": PHASE_LORA_ONLY,
        "selected_checkpoint": {"sha256": "b" * 64},
        "status": "COMPLETED_SELECTOR_LORA_SELECTED_BY_TRUE_RECOVERY",
    }
    _validate_parent_phase_summary(
        summary,
        checkpoint_sha256="b" * 64,
        expected_identity=expected_identity,
    )
    drifted = {**summary, "identity": {**expected_identity, "phase": PHASE_JOINT}}
    with pytest.raises(ValueError, match="identity drifted"):
        _validate_parent_phase_summary(
            drifted,
            checkpoint_sha256="b" * 64,
            expected_identity=expected_identity,
        )
    for key in (
        "boundary_extraction_config_sha256",
        "teacher_snapshot_manifest_sha256",
    ):
        drifted = {
            **summary,
            "identity": {**expected_identity, key: "e" * 64},
        }
        with pytest.raises(ValueError, match=f"identity drifted at {key}"):
            _validate_parent_phase_summary(
                drifted,
                checkpoint_sha256="b" * 64,
                expected_identity=expected_identity,
            )


def test_snapshot_manifest_file_sha256_is_bound(tmp_path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_bytes(b"frozen snapshot\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _verify_snapshot_manifest(path, expected_sha256=digest) == digest
    with pytest.raises(ValueError, match="file SHA256 drifted"):
        _verify_snapshot_manifest(path, expected_sha256="f" * 64)
    path.unlink()
    with pytest.raises(ValueError, match="file SHA256 drifted"):
        _verify_snapshot_manifest(path, expected_sha256=digest)


def test_boundary_manifest_content_and_context_hashes_fail_closed(tmp_path) -> None:
    shard = tmp_path / "boundary-shards" / "chunk.safetensors"
    shard.parent.mkdir()
    shard.write_bytes(b"sealed")
    contextual_sha = "c" * 64
    extraction_config_sha = "b" * 64
    manifest = {
        "config_sha256": extraction_config_sha,
        "content_sha256": "",
        "contextual_input_content_sha256": contextual_sha,
        "evaluation_labels_included": False,
        "shards": [
            {
                "byte_count": shard.stat().st_size,
                "path": "boundary-shards/chunk.safetensors",
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
        ],
        "status": SELECTOR_BOUNDARY_CACHE_STATUS,
        "tensor_inventory": {
            "boundary_hidden:k": {
                "partition": "boundary-shards",
                "shard": "chunk.safetensors",
            }
        },
    }
    manifest["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        _verify_boundary_manifest(
            tmp_path,
            expected_content_sha256=manifest["content_sha256"],
            expected_contextual_input_sha256=contextual_sha,
            expected_extraction_config_sha256=extraction_config_sha,
        )
        == manifest
    )
    with pytest.raises(ValueError, match="identity"):
        _verify_boundary_manifest(
            tmp_path,
            expected_content_sha256="d" * 64,
            expected_contextual_input_sha256=contextual_sha,
            expected_extraction_config_sha256=extraction_config_sha,
        )
    with pytest.raises(ValueError, match="identity"):
        _verify_boundary_manifest(
            tmp_path,
            expected_content_sha256=manifest["content_sha256"],
            expected_contextual_input_sha256="e" * 64,
            expected_extraction_config_sha256=extraction_config_sha,
        )
    with pytest.raises(ValueError, match="identity"):
        _verify_boundary_manifest(
            tmp_path,
            expected_content_sha256=manifest["content_sha256"],
            expected_contextual_input_sha256=contextual_sha,
            expected_extraction_config_sha256="a" * 64,
        )
    shard.write_bytes(b"changed-size")
    with pytest.raises(ValueError, match="stat"):
        _verify_boundary_manifest(
            tmp_path,
            expected_content_sha256=manifest["content_sha256"],
            expected_contextual_input_sha256=contextual_sha,
            expected_extraction_config_sha256=extraction_config_sha,
        )


def test_existing_truth_materializer_accepts_selector_lora_schedule(tmp_path) -> None:
    records = [
        {
            "missing_coalitions": [[1]] if index == 0 else [],
            "state_id": f"state-{index:03d}",
        }
        for index in range(256)
    ]
    schedule = {
        "epoch_checkpoints": [{"checkpoint_sha256": "a" * 64, "epoch": 1}],
        "epoch_count": 1,
        "epochs": [1],
        "missing_coalition_count": 1,
        "model_family": "selector_lora_v1_joint",
        "records": records,
        "schema_version": "causalcache.structured_truth_schedule.v1",
        "state_count": 256,
        "status": "PENDING_SELECTOR_LORA_HELDOUT_TRUTH",
    }
    digest = hashlib.sha256(canonical_json_bytes(schedule, pretty=True)).hexdigest()
    path = tmp_path / "selector-lora-schedule.json"
    path.write_bytes(
        canonical_json_bytes({**schedule, "content_sha256": digest}, pretty=True)
        + b"\n"
    )
    observed, observed_digest = _validate_signed_schedule(path)
    assert observed == {**schedule, "content_sha256": digest}
    assert observed_digest == digest


def test_selector_boundary_training_cache_loads_entities_lazily(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    inventory = {}
    for index in range(2):
        key = f"{index:064x}"
        tensors = {
            f"boundary_hidden__{key}": torch.zeros(5, 8, dtype=torch.bfloat16),
            f"input_ids__{key}": torch.arange(5, dtype=torch.int64),
            f"attention_mask__{key}": torch.ones(5, dtype=torch.int64),
            f"position_ids__{key}": torch.arange(5, dtype=torch.int64).repeat(3, 1),
            f"visual_indices__{key}": torch.tensor([3, 4], dtype=torch.int64),
            f"text_indices__{key}": torch.tensor([1, 2], dtype=torch.int64),
        }
        shard = f"chunk-{index}.safetensors"
        save_file(tensors, str(tmp_path / shard))
        for role, tensor in (
            ("boundary_hidden", tensors[f"boundary_hidden__{key}"]),
            ("input_ids", tensors[f"input_ids__{key}"]),
            ("attention_mask", tensors[f"attention_mask__{key}"]),
            ("position_ids", tensors[f"position_ids__{key}"]),
            ("visual_indices", tensors[f"visual_indices__{key}"]),
            ("text_indices", tensors[f"text_indices__{key}"]),
        ):
            inventory[f"{role}:{key}"] = {
                "dtype": str(tensor.dtype),
                "partition": ".",
                "shape": list(tensor.shape),
                "shard": shard,
                "tensor": f"{role}__{key}",
            }
    cache = _SelectorBoundaryCache(
        tmp_path,
        {"tensor_inventory": inventory},
        maximum_resident_entities=1,
        maximum_open_shards=1,
    )
    keys = sorted(cache.context_keys())
    assert len(keys) == 2
    assert cache.entity(keys[0]).boundary_hidden_state.dtype == torch.bfloat16
    assert cache.entity(keys[1]).visual_positions.tolist() == [3, 4]
    assert len(cache._handles) == 1
    assert len(cache._resident) == 1


def test_selector_collate_uses_variable_event_padding_and_boundary_entities() -> None:
    torch = pytest.importorskip("torch")

    class Cache:
        def entity(self, key):
            return f"entity:{key}"

    second = _state()
    second["candidate_event_step_ids"] = [1]
    second["event_image_keys"] = ["event-1"]
    second["event_text_keys"] = ["event-1"]
    second["event_numeric_features"] = [[0.5, 0.5]]
    second["distance_rows"] = [
        {"coalition_event_step_ids": [], "distance": 1.0},
        {"coalition_event_step_ids": [1], "distance": 0.0},
    ]
    second["state_id"] = "trajectory:decision:002"
    batch = _selector_collate(
        [_state(), second],
        cache=Cache(),
        device="cpu",
        normalization_floor=0.01,
        torch=torch,
    )
    assert batch["model"]["event_mask"].tolist() == [
        [True, True],
        [True, False],
    ]
    assert batch["model"]["event_entities"] == (
        ("entity:event-1", "entity:event-2"),
        ("entity:event-1", None),
    )
    assert tuple(batch["model"]["event_numeric_features"].shape) == (2, 2, 2)
    assert tuple(batch["model"]["subset_masks"].shape) == (2, 3, 2)
