from __future__ import annotations

import struct
import tempfile
import types
import unittest
from pathlib import Path

from causalcache.policy.gui_owl_v2 import (
    GUI_OWL_V2_SYSTEM_PROMPT,
    GUI_OWL_V2_TEACHER_CARRIER,
)
from causalcache.policy.gui_owl_v2_runtime import (
    GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
)
from scripts.audit_gui_owl_v2_processor import (
    _write_exclusive,
    deterministic_rgb_png,
    run_processor_audit,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _FakeTensor:
    def __init__(
        self,
        data: object,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: str = "torch.int64",
    ) -> None:
        self.data = data
        self.shape = shape if shape is not None else self._shape(data)
        self.dtype = dtype
        self.device = "cpu"
        self.requires_grad = False

    @classmethod
    def _shape(cls, value: object) -> tuple[int, ...]:
        if not isinstance(value, list):
            return ()
        if not value:
            return (0,)
        return (len(value), *cls._shape(value[0]))

    def detach(self) -> _FakeTensor:
        return self

    def to(self, *, device: str) -> _FakeTensor:
        if device != "cpu":
            raise AssertionError("processor audit should only materialize metadata on CPU")
        return self

    def tolist(self) -> object:
        return self.data


class _FakeTokenizer:
    def __init__(self, *, merge_joint_boundary: bool = False) -> None:
        self.merge_joint_boundary = merge_joint_boundary

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("audit tokenization must disable added special tokens")
        if text == GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX:
            return [501, 502]
        if text.startswith(GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX):
            suffix = text[len(GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX) :]
            values = [501, 502, *[ord(character) + 1 for character in suffix]]
            if (
                self.merge_joint_boundary
                and text.startswith(
                    GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX
                    + GUI_OWL_V2_TEACHER_CARRIER
                )
                and "<tool_call>" in text
            ):
                return values[:-1]
            return values
        return [ord(character) + 1 for character in text]


class _FakeImage:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.closed = False

    @property
    def label(self) -> str:
        return "portrait" if self.height > self.width else "landscape"

    def load(self) -> None:
        return None

    def convert(self, mode: str) -> _FakeImage:
        if mode != "RGB":
            raise AssertionError("audit fixture must remain RGB")
        return self

    def close(self) -> None:
        self.closed = True


class _FakeImageModule:
    @staticmethod
    def open(buffer: object) -> _FakeImage:
        raw = buffer.getvalue()  # type: ignore[attr-defined]
        width, height = struct.unpack(">II", raw[16:24])
        return _FakeImage(width, height)


class _FakeProcessor:
    def __init__(
        self,
        *,
        merge_joint_boundary: bool = False,
        bad_batch_shape: bool = False,
        bad_grid_count: bool = False,
        bad_pixel_dtype: bool = False,
        min_pixels: int,
        max_pixels: int,
    ) -> None:
        self.tokenizer = _FakeTokenizer(merge_joint_boundary=merge_joint_boundary)
        self.image_processor = types.SimpleNamespace(
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            merge_size=2,
        )
        self.bad_batch_shape = bad_batch_shape
        self.bad_grid_count = bad_grid_count
        self.bad_pixel_dtype = bad_pixel_dtype
        self.template_calls: list[tuple[object, dict[str, object]]] = []

    def apply_chat_template(self, conversations: object, **kwargs: object) -> dict[str, object]:
        self.template_calls.append((conversations, kwargs))
        samples = list(conversations)  # type: ignore[arg-type]
        grids: list[list[int]] = []
        effective_per_sample: list[int] = []
        for messages in samples:
            images = [
                item["image"]
                for item in messages[1]["content"]
                if item["type"] == "image"
            ]
            sample_grids = [
                [1, 150, 68] if image.label == "portrait" else [1, 68, 150]
                for image in images
            ]
            grids.extend(sample_grids)
            effective_per_sample.append(
                sum(t * h * w // 4 for t, h, w in sample_grids)
            )
        if len(set(effective_per_sample)) != 1 and len(samples) > 1:
            raise AssertionError("fake batch fixture must have equal sequence lengths")
        sequence_length = effective_per_sample[0] + 8
        prefix = self.tokenizer.encode(
            GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
            add_special_tokens=False,
        )
        rows = [[7] * (sequence_length - len(prefix)) + prefix for _ in samples]
        masks = [[1] * sequence_length for _ in samples]
        if self.bad_batch_shape and len(samples) == 2:
            rows = rows[:1]
            masks = masks[:1]
        if self.bad_grid_count:
            grids = grids[:-1]
        raw_patches = sum(t * h * w for t, h, w in grids)
        return {
            "input_ids": _FakeTensor(rows),
            "attention_mask": _FakeTensor(masks),
            "pixel_values": _FakeTensor(
                None,
                shape=(raw_patches, 1536),
                dtype=("torch.float16" if self.bad_pixel_dtype else "torch.float32"),
            ),
            "image_grid_thw": _FakeTensor(grids),
        }


class _FakeAutoProcessor:
    processor: _FakeProcessor
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs: object) -> _FakeProcessor:
        cls.calls.append((model_dir, kwargs))
        return cls.processor


def _identity(model_dir: Path) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_dir=str(model_dir.resolve()),
        model_repo="mPLUG/GUI-Owl-1.5-8B-Instruct",
        model_revision="06d5faecff74840bab2be2425e9c42667a5d04fc",
        snapshot_manifest_sha256="1" * 64,
        verified_model_file_count=14,
        verified_model_total_bytes=17_545_907_171,
        transformers_version="5.6.0",
        transformers_source_sha256=(("processing_qwen3_vl.py", "2" * 64),),
    )


def _runtime_verifier(**kwargs: object) -> types.SimpleNamespace:
    return _identity(Path(kwargs["model_dir"]))  # type: ignore[arg-type]


def _processor(**overrides: object) -> _FakeProcessor:
    target_pixels = 2560 * (16 * 2) ** 2
    values: dict[str, object] = {
        "min_pixels": target_pixels,
        "max_pixels": target_pixels,
    }
    values.update(overrides)
    return _FakeProcessor(**values)  # type: ignore[arg-type]


class GUIOwlV2ProcessorAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeAutoProcessor.calls = []

    def _run(self, processor: _FakeProcessor) -> dict[str, object]:
        _FakeAutoProcessor.processor = processor
        return run_processor_audit(
            model_dir=Path("/pinned/gui-owl"),
            expected_snapshot_manifest=Path("/git/snapshot.json"),
            runtime_verifier=_runtime_verifier,
            auto_processor_cls=_FakeAutoProcessor,
            image_module=_FakeImageModule,
        )

    def test_policy_output_free_audit_covers_one_five_and_nested_batch_two(self) -> None:
        processor = _processor()
        result = self._run(processor)

        self.assertEqual(result["status"], "passed")
        self.assertIs(result["auto_processor_only"], True)
        self.assertIs(result["auto_processor_loaded"], True)
        self.assertIs(result["model_weights_loaded"], False)
        self.assertEqual(
            result["pretrained_loader_calls"],
            ["AutoProcessor.from_pretrained"],
        )
        self.assertIs(result["model_weight_files_sha256_verified"], True)
        self.assertIs(result["model_weights_materialized_as_tensors"], False)
        self.assertIs(result["policy_model_loaded"], False)
        self.assertIs(result["policy_loaded"], False)
        self.assertIs(result["policy_forward_executed"], False)
        self.assertIs(result["policy_generate_executed"], False)
        self.assertIs(result["policy_output"], False)
        self.assertIs(result["policy_output_generated"], False)
        self.assertEqual(
            result["single_conversation_one_image"]["samples"][0]["image_count"],  # type: ignore[index]
            1,
        )
        self.assertEqual(
            result["single_conversation_five_images"]["samples"][0]["image_count"],  # type: ignore[index]
            5,
        )
        batch = result["nested_batch_two_equal_shape"]
        self.assertEqual(batch["batch_size"], 2)  # type: ignore[index]
        self.assertIs(batch["equal_image_count"], True)  # type: ignore[index]
        self.assertIs(batch["equal_sequence_length"], True)  # type: ignore[index]
        self.assertIs(batch["no_padding"], True)  # type: ignore[index]
        self.assertTrue(batch["attention_mask_all_one"])  # type: ignore[index]
        self.assertEqual(len(processor.template_calls), 3)
        for conversations, kwargs in processor.template_calls:
            for messages in conversations:
                self.assertEqual(
                    messages[0],
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": GUI_OWL_V2_SYSTEM_PROMPT}
                        ],
                    },
                )
            self.assertEqual(
                kwargs,
                {
                    "tokenize": True,
                    "add_generation_prompt": True,
                    "return_dict": True,
                    "return_tensors": "pt",
                    "padding": False,
                },
            )
        self.assertEqual(
            _FakeAutoProcessor.calls,
            [
                (
                    str(Path("/pinned/gui-owl").resolve()),
                    {
                        "min_pixels": 2_621_440,
                        "max_pixels": 2_621_440,
                        "local_files_only": True,
                    },
                )
            ],
        )

    def test_script_has_no_policy_model_loader_forward_or_generate_call(self) -> None:
        source = (
            REPOSITORY_ROOT / "code/scripts/audit_gui_owl_v2_processor.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("AutoModel", source)
        self.assertNotIn(".forward(", source)
        self.assertNotIn(".generate(", source)
        self._run(_processor())

    def test_joint_tokenizer_boundary_failure_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "assistant-prefix/carrier/tool boundary"):
            self._run(_processor(merge_joint_boundary=True))

    def test_nested_batch_shape_and_grid_count_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch shape"):
            self._run(_processor(bad_batch_shape=True))
        with self.assertRaisesRegex(ValueError, "image_grid_thw"):
            self._run(_processor(bad_grid_count=True))

    def test_actual_min_and_max_pixels_must_equal_frozen_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "min/max target"):
            self._run(_processor(max_pixels=2_621_439))

    def test_processor_tensor_dtype_drift_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "pixel_values dtype drifted"):
            self._run(_processor(bad_pixel_dtype=True))

    def test_deterministic_png_and_exclusive_json(self) -> None:
        first = deterministic_rgb_png(width=108, height=240, seed=37)
        second = deterministic_rgb_png(width=108, height=240, seed=37)
        landscape = deterministic_rgb_png(width=240, height=108, seed=91)
        self.assertEqual(first, second)
        self.assertNotEqual(first, landscape)
        self.assertEqual(first[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", first[16:24]), (108, 240))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            _write_exclusive(output, {"status": "passed"})
            with self.assertRaises(FileExistsError):
                _write_exclusive(output, {"status": "passed"})


if __name__ == "__main__":
    unittest.main()
