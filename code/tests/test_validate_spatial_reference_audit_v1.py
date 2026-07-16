from __future__ import annotations

import copy
import hashlib
import unittest
from pathlib import Path

from causalcache.data.guiodyssey_restoration_v2 import canonical_json_bytes
from causalcache.spatial_reference_audit_v1 import (
    load_and_validate_config,
    profile_by_id,
)
from scripts.validate_spatial_reference_audit_v1 import (
    _validate_generation,
    _validate_host,
    _validate_runtime,
    _validate_shared_repeat,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_and_validate_config(
    ROOT / "code/configs/spatial_reference_audit_v1.json"
)


def _score(token_id: int, target_logit: float) -> dict[str, object]:
    return {
        "target_token_id": token_id,
        "target_logit": target_logit,
        "target_log_probability": -1.0,
        "target_rank_strict": 1,
        "top1_token_id": token_id,
        "top2_token_id": 9,
        "top1_logit": 4.0,
        "top2_logit": 3.0,
        "top1_minus_top2_margin": 1.0,
    }


def _shape(*, aligned: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "image_count": 1,
        "image_grid_thw": [[1, 80, 128]],
        "effective_visual_tokens": 2560,
        "policy_visible_text_tokens": 100,
        "prompt_input_tokens": 2660,
    }
    if aligned:
        result["extended_prompt_aligned_inputs"] = ["attention_mask", "input_ids"]
    return result


class SpatialReferenceAuditValidatorTest(unittest.TestCase):
    def test_generation_uses_actual_ids_and_rejects_hash_mutation(self) -> None:
        token_ids = [1, 2, 3]
        generation = {
            "repeat_index": 1,
            "output_text": "<tool_call>x</tool_call>",
            "generated_token_ids": token_ids,
            "canonical_action": {"action": "click", "coordinate": [1, 2]},
            "parse_error_type": None,
            "parse_error_message": None,
            "metadata": {
                **_shape(aligned=False),
                "generated_token_ids": token_ids,
                "generated_token_ids_sha256": hashlib.sha256(
                    canonical_json_bytes(token_ids)
                ).hexdigest(),
                "generated_tokens": 3,
                "decoded_output_retokenized_ids": token_ids,
                "decoded_output_retokenization_matches_actual_ids": True,
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
                "return_dict_in_generate": False,
                "full_logit_tensor_host_transfers": 0,
            },
        }
        _validate_generation(generation, repeat_index=1)
        drifted = copy.deepcopy(generation)
        drifted["metadata"]["generated_token_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            _validate_generation(drifted, repeat_index=1)

    def test_shared_prefix_pair_margin_is_from_one_vector(self) -> None:
        value = {
            **_shape(aligned=True),
            "repeat_index": 1,
            "teacher_token_ids": [[1, 2], [1, 3]],
            "first_divergent_token_index": 1,
            "first_divergent_token_ids": [2, 3],
            "shared_prefix_token_ids": [1],
            "raw_candidate_summaries": [_score(2, 2.5), _score(3, 2.0)],
            "generation_aligned_candidate_summaries": [
                _score(2, 2.5),
                _score(3, 2.0),
            ],
            "raw_first_minus_second_candidate_logit": 0.5,
            "generation_aligned_first_minus_second_candidate_logit": 0.5,
            "full_vocabulary_logits_host_transfers": 0,
            "interpretation": (
                "single_shared_prefix_post_hoc_teacher_forced_competing_token_"
                "margin_not_original_generation_time_margin"
            ),
        }
        _validate_shared_repeat(
            value,
            repeat_index=1,
            token_rows=[[1, 2], [1, 3]],
            divergence=1,
        )
        drifted = copy.deepcopy(value)
        drifted["generation_aligned_first_minus_second_candidate_logit"] = 0.4
        with self.assertRaisesRegex(ValueError, "pair margin"):
            _validate_shared_repeat(
                drifted,
                repeat_index=1,
                token_rows=[[1, 2], [1, 3]],
                divergence=1,
            )

    def test_eager_runtime_and_live_host_are_fail_closed(self) -> None:
        profile = profile_by_id(CONFIG, "bf16_eager_control")
        runtime = {
            "audit_profile_id": profile.profile_id,
            "dtype": "torch.bfloat16",
            "device": "cuda:0",
            "model_repo": "mPLUG/GUI-Owl-1.5-8B-Instruct",
            "model_revision": "06d5faecff74840bab2be2425e9c42667a5d04fc",
            "snapshot_manifest_sha256": CONFIG["inputs"]["model_snapshot_manifest"][
                "sha256"
            ],
            "verified_model_file_count": 14,
            "verified_model_total_bytes": 17545907171,
            "python_version": CONFIG["runtime_constraints"]["python_version"],
            "torch_version": CONFIG["runtime_constraints"]["torch_version"],
            "torch_cuda_version": CONFIG["runtime_constraints"][
                "torch_cuda_version"
            ],
            "cudnn_version": CONFIG["runtime_constraints"]["cudnn_version"],
            "transformers_version": "5.6.0",
            "frozen": True,
            "single_device": True,
            "seed": 0,
            "deterministic_algorithms_requested": False,
            "deterministic_algorithms_enabled": False,
            "deterministic_warn_only_enabled": False,
            "strict_cuda_determinism_claimed": False,
            "scientific_environment_variables_set_by_runtime": False,
            "scientific_environment_audit": {
                "audited_names": CONFIG["runtime_constraints"][
                    "audited_scientific_environment_variables"
                ],
                "present_names": [],
                "all_absent": True,
            },
            "instrumented_generation_output_scores": False,
            "eager_numerical_controls_requested": True,
            "requested_attention_implementation": "eager",
            "observed_attention_implementation": {
                "model": "eager",
                "text_config": "eager",
                "vision_config": "eager",
            },
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "numerical_control_claim": (
                "eager_fixed_seed_tf32_disabled_numerical_control_not_strict_cuda_determinism"
            ),
        }
        _validate_runtime(runtime, profile=profile, config=CONFIG, repository_root=ROOT)
        drifted = copy.deepcopy(runtime)
        drifted["observed_attention_implementation"]["vision_config"] = "sdpa"
        with self.assertRaisesRegex(ValueError, "eager attention"):
            _validate_runtime(
                drifted,
                profile=profile,
                config=CONFIG,
                repository_root=ROOT,
            )

        host = {
            "alias": "hyper01",
            "hostname": "node-radixark-16-0001",
            "container_hostname": "a" * 12,
            "container_id": "a" * 64,
            "container_image_digest": CONFIG["runtime_constraints"][
                "container_image_digest"
            ],
            "device": "cuda:0",
            "visible_gpu_count": 1,
            "cuda_visible_ordinal": 0,
            "nvidia_smi_gpu_index": 6,
            "gpu_name": "NVIDIA H200",
            "gpu_uuid": "GPU-test",
            "gpu_pci_bus_id": "00000000:01:00.0",
            "nvidia_driver_version": CONFIG["runtime_constraints"][
                "nvidia_driver_version"
            ],
            "software": {
                key: CONFIG["runtime_constraints"][key]
                for key in (
                    "python_version",
                    "torch_version",
                    "torch_cuda_version",
                    "cudnn_version",
                    "transformers_version",
                )
            },
            "scientific_environment_audit": {
                "audited_names": CONFIG["runtime_constraints"][
                    "audited_scientific_environment_variables"
                ],
                "present_names": [],
                "all_absent": True,
            },
            "nvidia_smi_query": (
                "nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version "
                "--format=csv,noheader,nounits"
            ),
        }
        _validate_host(host, config=CONFIG)
        host["visible_gpu_count"] = 2
        with self.assertRaisesRegex(ValueError, "host/container/GPU"):
            _validate_host(host, config=CONFIG)


if __name__ == "__main__":
    unittest.main()
