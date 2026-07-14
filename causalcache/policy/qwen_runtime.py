"""Reusable single-GPU Qwen-family runtime for CausalCache policy experiments."""

from __future__ import annotations

import io
import json
import tarfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from causalcache.policy.prompt import parse_policy_action
from causalcache.schema import ExecutableAction


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
        visual_tokens_per_image: int,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not device.startswith("cuda:"):
            raise ValueError("Qwen policy runtime requires an explicit cuda device")
        pixels_per_image = visual_tokens_per_image * 28 * 28
        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_dir,
            min_pixels=pixels_per_image,
            max_pixels=pixels_per_image,
            local_files_only=True,
        )
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
        }

    def _encode(self, messages: list[dict[str, Any]]) -> Any:
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

    def probe_logits(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = self._encode(messages)
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(**inputs, use_cache=False, return_dict=True)
        self.torch.cuda.synchronize(self.device)
        logits = outputs.logits
        result = {
            "input_tokens": int(inputs.input_ids.shape[1]),
            "image_count": int(inputs.image_grid_thw.shape[0]),
            "logits_shape": list(logits.shape),
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
        return {
            "input_tokens": int(inputs.input_ids.shape[1]),
            "image_count": int(inputs.image_grid_thw.shape[0]),
            "generated_tokens": int(new_tokens.shape[1]),
            "latency_seconds": latency_seconds,
            "peak_gpu_memory_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
            "output_text": output_text,
            "parsed_action": action_dict(parsed_action) if parsed_action is not None else None,
            "parse_error": parse_error,
            "executable_match": executable_match,
        }
