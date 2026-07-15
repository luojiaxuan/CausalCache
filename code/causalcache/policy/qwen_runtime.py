"""Reusable single-GPU Qwen-family runtime for CausalCache policy experiments."""

from __future__ import annotations

import io
import json
import math
import tarfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from causalcache.policy.prompt import parse_policy_action
from causalcache.schema import ExecutableAction


def _validated_token_ids(token_ids: Sequence[int], *, name: str) -> list[int]:
    result: list[int] = []
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"{name} must contain non-negative integer token ids")
        result.append(token_id)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _teacher_forced_action_layout(
    prompt_token_ids: Sequence[int],
    action_token_ids: Sequence[int],
) -> dict[str, list[int]]:
    """Describe the causal-LM shift used for teacher-forced action scoring."""
    prompt_ids = _validated_token_ids(prompt_token_ids, name="prompt_token_ids")
    action_ids = _validated_token_ids(action_token_ids, name="action_token_ids")
    model_input_ids = prompt_ids + action_ids[:-1]
    first_action_logit = len(prompt_ids) - 1
    return {
        "model_input_ids": model_input_ids,
        "action_token_ids": action_ids,
        "action_logit_positions": list(
            range(first_action_logit, first_action_logit + len(action_ids))
        ),
    }


def _teacher_forced_sequence_fill_value(key: str) -> int:
    values = {
        "attention_mask": 1,
        "mm_token_type_ids": 0,
    }
    if key not in values:
        raise ValueError(f"unsupported sequence-aligned model input: {key}")
    return values[key]


def _sequence_action_path_kl(
    reference: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
) -> list[float]:
    reference_rows = [[float(value) for value in row] for row in reference]
    candidate_rows = [[float(value) for value in row] for row in candidate]
    if not reference_rows or len(reference_rows) != len(candidate_rows):
        raise ValueError("reference and candidate must have the same non-empty shape")
    vocabulary_size = len(reference_rows[0])
    if vocabulary_size == 0:
        raise ValueError("action-path log-probabilities need a non-empty vocabulary")
    for reference_row, candidate_row in zip(reference_rows, candidate_rows, strict=True):
        if len(reference_row) != vocabulary_size or len(candidate_row) != vocabulary_size:
            raise ValueError("reference and candidate must have the same rectangular shape")
        if not all(math.isfinite(value) for value in reference_row + candidate_row):
            raise ValueError("action-path log-probabilities must be finite")
    return [
        sum(
            math.exp(reference_value) * (reference_value - candidate_value)
            for reference_value, candidate_value in zip(
                reference_row,
                candidate_row,
                strict=True,
            )
        )
        for reference_row, candidate_row in zip(
            reference_rows,
            candidate_rows,
            strict=True,
        )
    ]


def full_vocab_action_path_kl(reference: Any, candidate: Any) -> dict[str, Any]:
    """Compute full-vocabulary KL at every teacher-forced action position."""
    if hasattr(reference, "detach") or hasattr(candidate, "detach"):
        try:
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError("tensor KL computation requires PyTorch") from error
        if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
            raise TypeError("reference and candidate must use the same tensor representation")
        if reference.ndim != 2 or candidate.ndim != 2 or reference.shape != candidate.shape:
            raise ValueError("reference and candidate must have the same rank-2 shape")
        if reference.shape[0] == 0 or reference.shape[1] == 0:
            raise ValueError("action-path log-probabilities must have non-empty dimensions")
        reference_tensor = reference.detach().to(device="cpu", dtype=torch.float32)
        candidate_tensor = candidate.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(reference_tensor).all() or not torch.isfinite(
            candidate_tensor
        ).all():
            raise ValueError("action-path log-probabilities must be finite")
        per_token_tensor = torch.sum(
            reference_tensor.exp() * (reference_tensor - candidate_tensor),
            dim=-1,
        )
        per_token = [float(value) for value in per_token_tensor.tolist()]
    else:
        per_token = _sequence_action_path_kl(reference, candidate)

    negative_tolerance = 1e-5
    if min(per_token) < -negative_tolerance:
        raise ValueError("action-path KL is materially negative; inputs may not be log-probabilities")
    nonnegative = [max(0.0, value) for value in per_token]
    total = float(sum(nonnegative))
    return {
        "per_token": nonnegative,
        "sum": total,
        "mean": total / len(nonnegative),
    }


def visual_patch_factor(model_dir: Path) -> int:
    config = json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
    patch_size = int(config.get("patch_size", 14))
    merge_size = int(config.get("merge_size", 2))
    if patch_size <= 0 or merge_size <= 0:
        raise ValueError("visual patch and merge sizes must be positive")
    return patch_size * merge_size


def effective_visual_tokens(
    image_grid_thw: Sequence[Sequence[int]],
    *,
    merge_size: int,
) -> int:
    if merge_size <= 0:
        raise ValueError("visual merge size must be positive")
    divisor = merge_size * merge_size
    total = 0
    for grid in image_grid_thw:
        if len(grid) != 3 or any(int(value) <= 0 for value in grid):
            raise ValueError("each visual grid must contain positive t, h, and w")
        temporal, height, width = (int(value) for value in grid)
        total += temporal * height * width // divisor
    return total


def action_dict(action: ExecutableAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type.value,
        "target": action.target,
        "text_argument": action.text_argument,
        "text_case_sensitive": action.text_case_sensitive,
    }


def load_dataset(archive: tarfile.TarFile) -> tuple[dict[str, Any], Any]:
    from PIL import Image

    manifest_file = archive.extractfile("manifest.json")
    if manifest_file is None:
        raise ValueError("dataset tar does not contain manifest.json")
    manifest = json.load(manifest_file)

    def load_image(relative_path: str) -> Any:
        member = archive.extractfile(relative_path)
        if member is None:
            raise ValueError(f"missing image: {relative_path}")
        return Image.open(io.BytesIO(member.read())).convert("RGB")

    return manifest, load_image


class QwenPolicyRuntime:
    def __init__(
        self,
        *,
        model_dir: Path,
        device: str,
        visual_tokens_per_image: int | None,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not device.startswith("cuda:"):
            raise ValueError("Qwen policy runtime requires an explicit cuda device")
        patch_factor = visual_patch_factor(model_dir)
        if visual_tokens_per_image is not None and visual_tokens_per_image <= 0:
            raise ValueError("visual token target must be positive")
        processor_kwargs: dict[str, Any] = {}
        if visual_tokens_per_image is not None:
            pixels_per_image = visual_tokens_per_image * patch_factor * patch_factor
            processor_kwargs = {
                "min_pixels": pixels_per_image,
                "max_pixels": pixels_per_image,
            }
        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_dir,
            **processor_kwargs,
            local_files_only=True,
        )
        self.merge_size = int(self.processor.image_processor.merge_size)
        load_start = time.perf_counter()
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(device).eval()
        self.load_seconds = time.perf_counter() - load_start
        self.metadata = {
            "snapshot": json.loads((model_dir / ".snapshot.json").read_text(encoding="utf-8")),
            "dtype": "bfloat16",
            "device": device,
            "load_seconds": self.load_seconds,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "processor_class": self.processor.__class__.__name__,
            "model_class": self.model.__class__.__name__,
            "visual_patch_factor": patch_factor,
            "visual_preprocessing": {
                "mode": (
                    "model_default"
                    if visual_tokens_per_image is None
                    else "fixed_token_target"
                ),
                "target_tokens_per_image": visual_tokens_per_image,
                "merge_size": self.merge_size,
            },
        }

    def _encode(self, messages: list[dict[str, Any]]) -> Any:
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

    def tokenize_canonical_action(self, canonical_action: str) -> list[int]:
        if not isinstance(canonical_action, str) or not canonical_action.strip():
            raise ValueError("canonical action must be a non-empty string")
        token_ids = _validated_token_ids(
            self.processor.tokenizer.encode(
                canonical_action,
                add_special_tokens=False,
            ),
            name="canonical_action_token_ids",
        )
        special_ids = set(
            int(value) for value in self.processor.tokenizer.all_special_ids
        )
        if special_ids.intersection(token_ids):
            raise ValueError("canonical action must not contain special tokens")
        return token_ids

    def teacher_forced_action_log_probs(
        self,
        messages: list[dict[str, Any]],
        action_token_ids: Sequence[int],
    ) -> tuple[Any, dict[str, Any]]:
        """Return CPU float32 full-vocabulary log probabilities on the action path."""
        action_ids = _validated_token_ids(
            action_token_ids,
            name="action_token_ids",
        )
        special_ids = set(
            int(value) for value in self.processor.tokenizer.all_special_ids
        )
        if special_ids.intersection(action_ids):
            raise ValueError("action_token_ids must not contain special tokens")

        inputs = self._encode(messages)
        prompt_input_ids = inputs["input_ids"]
        if prompt_input_ids.ndim != 2 or prompt_input_ids.shape[0] != 1:
            raise ValueError("teacher forcing requires a single rank-2 prompt input")
        prompt_ids = [int(value) for value in prompt_input_ids[0].detach().cpu().tolist()]
        layout = _teacher_forced_action_layout(prompt_ids, action_ids)
        model_inputs = dict(inputs)
        forced_prefix = action_ids[:-1]
        sequence_aligned_keys = {
            key
            for key, value in model_inputs.items()
            if key != "input_ids"
            and hasattr(value, "shape")
            and tuple(value.shape) == tuple(prompt_input_ids.shape)
        }
        for key in sequence_aligned_keys:
            _teacher_forced_sequence_fill_value(key)
        if forced_prefix:
            prefix_tensor = self.torch.tensor(
                [forced_prefix],
                dtype=prompt_input_ids.dtype,
                device=prompt_input_ids.device,
            )
            model_inputs["input_ids"] = self.torch.cat(
                [prompt_input_ids, prefix_tensor],
                dim=1,
            )
            for key in sorted(sequence_aligned_keys):
                sequence_tensor = model_inputs[key]
                forced_values = self.torch.full(
                    (1, len(forced_prefix)),
                    _teacher_forced_sequence_fill_value(key),
                    dtype=sequence_tensor.dtype,
                    device=sequence_tensor.device,
                )
                model_inputs[key] = self.torch.cat(
                    [sequence_tensor, forced_values],
                    dim=1,
                )

        actual_model_input_ids = [
            int(value)
            for value in model_inputs["input_ids"][0].detach().cpu().tolist()
        ]
        if actual_model_input_ids != layout["model_input_ids"]:
            raise RuntimeError("teacher-forced model input does not match the causal shift layout")
        action_length = len(action_ids)
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(
                **model_inputs,
                use_cache=False,
                return_dict=True,
                logits_to_keep=action_length,
            )
            logits = outputs.logits
            if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != action_length:
                raise RuntimeError(
                    "teacher-forced logits must have shape [1, action_tokens, vocabulary]"
                )
            vocabulary_size = int(logits.shape[2])
            if any(token_id >= vocabulary_size for token_id in action_ids):
                raise ValueError("action token id exceeds the model vocabulary")
            if not self.torch.isfinite(logits).all():
                raise RuntimeError("teacher-forced action logits contain non-finite values")
            log_probs = self.torch.log_softmax(logits[0].float(), dim=-1)
            if not self.torch.isfinite(log_probs).all():
                raise RuntimeError("teacher-forced action log-probabilities are non-finite")
            cpu_log_probs = log_probs.detach().to(device="cpu", dtype=self.torch.float32)
        self.torch.cuda.synchronize(self.device)
        latency_seconds = time.perf_counter() - start

        image_grid = (
            model_inputs["image_grid_thw"].detach().cpu().tolist()
            if "image_grid_thw" in model_inputs
            else []
        )
        metadata = {
            "prompt_input_tokens": len(prompt_ids),
            "input_tokens": int(model_inputs["input_ids"].shape[1]),
            "teacher_forced_prefix_tokens": len(forced_prefix),
            "extended_sequence_aligned_inputs": sorted(sequence_aligned_keys),
            "action_tokens": action_length,
            "vocabulary_size": vocabulary_size,
            "action_logit_positions": layout["action_logit_positions"],
            "image_count": len(image_grid),
            "image_grid_thw": image_grid,
            "effective_visual_tokens": effective_visual_tokens(
                image_grid,
                merge_size=self.merge_size,
            ),
            "logits_to_keep": action_length,
            "latency_seconds": latency_seconds,
            "peak_gpu_memory_bytes": int(
                self.torch.cuda.max_memory_allocated(self.device)
            ),
            "finite_log_probs": True,
        }
        del outputs, logits, log_probs
        self.torch.cuda.empty_cache()
        return cpu_log_probs, metadata

    def probe_logits(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = self._encode(messages)
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(
                **inputs,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
        self.torch.cuda.synchronize(self.device)
        logits = outputs.logits
        result = {
            "input_tokens": int(inputs.input_ids.shape[1]),
            "image_count": int(inputs.image_grid_thw.shape[0]),
            "image_grid_thw": inputs.image_grid_thw.detach().cpu().tolist(),
            "effective_visual_tokens": effective_visual_tokens(
                inputs.image_grid_thw.detach().cpu().tolist(),
                merge_size=self.merge_size,
            ),
            "logits_shape": list(logits.shape),
            "logits_to_keep": 1,
            "latency_seconds": time.perf_counter() - start,
            "peak_gpu_memory_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
            "finite_last_token_logits": bool(self.torch.isfinite(logits[:, -1]).all().item()),
        }
        del outputs, logits
        self.torch.cuda.empty_cache()
        return result

    def warmup(self, messages: list[dict[str, Any]]) -> None:
        inputs = self._encode(messages)
        with self.torch.inference_mode():
            self.model.generate(**inputs, do_sample=False, max_new_tokens=1)
        self.torch.cuda.synchronize(self.device)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        validated_action: ExecutableAction,
        action_parser: Callable[[str, Any], ExecutableAction] | None = None,
    ) -> dict[str, Any]:
        result, inputs = self.generate_text(
            messages,
            max_new_tokens=max_new_tokens,
            return_inputs=True,
        )
        output_text = result["output_text"]
        parsed_action = None
        parse_error = None
        executable_match = False
        try:
            parsed_action = (
                parse_policy_action(output_text)
                if action_parser is None
                else action_parser(output_text, inputs)
            )
            executable_match = parsed_action.executable_match(validated_action)
        except (KeyError, TypeError, ValueError) as error:
            parse_error = str(error)
        result.update(
            {
                "parsed_action": action_dict(parsed_action) if parsed_action is not None else None,
                "parse_error": parse_error,
                "executable_match": executable_match,
            }
        )
        return result

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        return_inputs: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], Any]:
        """Generate raw policy text for closed-loop execution or downstream parsing."""
        inputs = self._encode(messages)
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        self.torch.cuda.synchronize(self.device)
        latency_seconds = time.perf_counter() - start
        new_tokens = generated[:, inputs.input_ids.shape[1] :]
        output_text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        result = {
            "input_tokens": int(inputs.input_ids.shape[1]),
            "image_count": int(inputs.image_grid_thw.shape[0]),
            "image_grid_thw": inputs.image_grid_thw.detach().cpu().tolist(),
            "effective_visual_tokens": effective_visual_tokens(
                inputs.image_grid_thw.detach().cpu().tolist(),
                merge_size=self.merge_size,
            ),
            "generated_tokens": int(new_tokens.shape[1]),
            "latency_seconds": latency_seconds,
            "peak_gpu_memory_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
            "output_text": output_text,
        }
        if return_inputs:
            return result, inputs
        return result
