"""Pinned single-GPU runtime for the OpenCUA remote architecture."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from causalcache.policy.open_cua import parse_open_cua_action
from causalcache.policy.qwen_runtime import action_dict
from causalcache.schema import ExecutableAction


class OpenCUAPolicyRuntime:
    def __init__(
        self,
        *,
        model_dir: Path,
        device: str,
        visual_tokens_per_image: int,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoModel, AutoProcessor

        if not device.startswith("cuda:"):
            raise ValueError("OpenCUA policy runtime requires an explicit cuda device")
        if transformers.__version__ != "4.53.0":
            raise RuntimeError(
                "OpenCUA pinned remote code requires transformers==4.53.0; "
                f"found {transformers.__version__}"
            )
        pixels_per_image = visual_tokens_per_image * 28 * 28
        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_dir,
            min_pixels=pixels_per_image,
            max_pixels=pixels_per_image,
            local_files_only=True,
            trust_remote_code=True,
        )
        load_start = time.perf_counter()
        self.model = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
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
            "trust_remote_code": True,
        }

    def encode(self, messages: list[dict[str, Any]]) -> Any:
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

    def probe_logits(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = self.encode(messages)
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(**inputs, use_cache=False, return_dict=True)
        self.torch.cuda.synchronize(self.device)
        logits = outputs.logits
        result = {
            "input_tokens": int(inputs.input_ids.shape[1]),
            "logits_shape": list(logits.shape),
            "latency_seconds": time.perf_counter() - start,
            "peak_gpu_memory_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
            "finite_last_token_logits": bool(self.torch.isfinite(logits[:, -1]).all().item()),
        }
        del outputs, logits
        self.torch.cuda.empty_cache()
        return result

    def warmup(self, messages: list[dict[str, Any]]) -> None:
        inputs = self.encode(messages)
        with self.torch.inference_mode():
            self.model.generate(**inputs, do_sample=False, max_new_tokens=1)
        self.torch.cuda.synchronize(self.device)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        validated_action: ExecutableAction,
    ) -> dict[str, Any]:
        inputs = self.encode(messages)
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
        output_text = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        parsed_action = None
        parse_error = None
        executable_match = False
        try:
            parsed_action = parse_open_cua_action(output_text, inputs)
            executable_match = parsed_action.executable_match(validated_action)
        except (KeyError, TypeError, ValueError, SyntaxError) as error:
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
