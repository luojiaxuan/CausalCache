from __future__ import annotations

import types
import unittest
from collections.abc import Sequence

from causalcache.policy.gui_owl_v2 import (
    GUI_OWL_V2_SYSTEM_PROMPT,
    GUI_OWL_V2_TEACHER_CARRIER,
    GUIOwlV2Action,
    serialize_gui_owl_v2_tool_call,
)
from causalcache.policy.gui_owl_v2_runtime import (
    FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS,
    GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
    GUIOwlV2GenerationParseError,
    GUIOwlV2Runtime,
    build_gui_owl_v2_teacher_tokens,
    gui_owl_v2_teacher_layout,
    prompt_aligned_input_keys,
)


class _FakeTensor:
    def __init__(
        self,
        data: object,
        *,
        dtype: str,
        device: str = "cuda:0",
        name: str = "tensor",
        shape: tuple[int, ...] | None = None,
    ) -> None:
        self.data = data
        self.dtype = dtype
        self.device = device
        self.name = name
        self.requires_grad = False
        self.cpu_transfer_attempts = 0
        self.shape = shape if shape is not None else self._shape(data)
        self.ndim = len(self.shape)

    @classmethod
    def _shape(cls, data: object) -> tuple[int, ...]:
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
            return ()
        values = list(data)
        if not values:
            return (0,)
        return (len(values), *cls._shape(values[0]))

    def detach(self) -> _FakeTensor:
        return self

    def to(self, device: object = None, dtype: object = None, **kwargs: object) -> _FakeTensor:
        requested_device = kwargs.get("device", device)
        if requested_device == "cpu":
            self.cpu_transfer_attempts += 1
            if self.name == "logits":
                raise AssertionError("full logits must never move to CPU")
        return self

    def tolist(self) -> object:
        if self.name == "logits":
            raise AssertionError("full logits must never be materialized as a host list")
        return self.data

    def all(self) -> _FakeAssertion:
        return _FakeAssertion(True)

    def __getitem__(self, index: object) -> _FakeTensor:
        if not isinstance(index, tuple) or len(index) != 2:
            raise AssertionError(f"unexpected fake tensor index: {index!r}")
        row_index, column_index = index
        if row_index != slice(None):
            raise AssertionError(f"unexpected row index: {row_index!r}")
        rows = [list(row)[column_index] for row in self.data]  # type: ignore[union-attr]
        return _FakeTensor(
            rows,
            dtype=self.dtype,
            device=self.device,
            name=f"{self.name}_slice",
        )


class _FakeAssertion:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> _FakeAssertion:
        return self


class _InferenceMode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class _FakeTorch:
    bfloat16 = "torch.bfloat16"

    def __init__(self) -> None:
        self.async_assertions: list[tuple[bool, str]] = []
        self.cuda = _FakeCuda()

    def inference_mode(self) -> _InferenceMode:
        return _InferenceMode()

    def tensor(self, values: object, *, dtype: str, device: str) -> _FakeTensor:
        return _FakeTensor(values, dtype=dtype, device=device, name="suffix")

    def full(
        self,
        shape: tuple[int, int],
        fill_value: int,
        *,
        dtype: str,
        device: str,
    ) -> _FakeTensor:
        rows = [[fill_value] * shape[1] for _ in range(shape[0])]
        return _FakeTensor(rows, dtype=dtype, device=device, name="fill")

    def cat(self, values: Sequence[_FakeTensor], *, dim: int) -> _FakeTensor:
        if dim != 1 or len(values) != 2:
            raise AssertionError("fake torch supports only the runtime's rank-2 dim-1 cat")
        left, right = values
        rows = [
            [*left_row, *right_row]
            for left_row, right_row in zip(  # type: ignore[arg-type]
                left.data,
                right.data,
                strict=True,
            )
        ]
        return _FakeTensor(
            rows,
            dtype=left.dtype,
            device=left.device,
            name=f"extended_{left.name}",
        )

    def isfinite(self, value: _FakeTensor) -> _FakeAssertion:
        if value.name != "logits":
            raise AssertionError("only logits finite validation is expected")
        return _FakeAssertion(True)

    def _assert_async(self, value: _FakeAssertion, message: str) -> None:
        self.async_assertions.append((value.value, message))
        if not value.value:
            raise RuntimeError(message)


class _FakeCuda:
    def __init__(self) -> None:
        self.reset_calls: list[object] = []
        self.synchronize_calls: list[object] = []

    def reset_peak_memory_stats(self, device: object) -> None:
        self.reset_calls.append(device)

    def synchronize(self, device: object) -> None:
        self.synchronize_calls.append(device)

    def max_memory_allocated(self, device: object) -> int:
        return 123_456

    def max_memory_reserved(self, device: object) -> int:
        return 234_567


class _FakeBatchFeature(dict[str, _FakeTensor]):
    def __init__(self, values: dict[str, _FakeTensor]) -> None:
        super().__init__(values)
        self.to_calls: list[object] = []

    def to(self, device: object) -> _FakeBatchFeature:
        self.to_calls.append(device)
        return self


class _CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("runtime tokenization must disable added special tokens")
        if text == GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX:
            return [501, 502]
        if text.startswith(GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX):
            suffix = text[len(GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX) :]
            return [501, 502, *[ord(character) + 1 for character in suffix]]
        return [ord(character) + 1 for character in text]


class _BoundaryMergingTokenizer(_CharacterTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        values = super().encode(text, add_special_tokens=add_special_tokens)
        if text.startswith(GUI_OWL_V2_TEACHER_CARRIER) and "<tool_call>" in text:
            return values[1:]
        return values


class _AssistantBoundaryMergingTokenizer(_CharacterTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        values = super().encode(text, add_special_tokens=add_special_tokens)
        if text == GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX + GUI_OWL_V2_TEACHER_CARRIER:
            return values[1:]
        return values


class _FakeProcessor:
    def __init__(
        self,
        *,
        unknown_prompt_tensor: bool = False,
        bad_assistant_prefix: bool = False,
    ) -> None:
        self.tokenizer = _CharacterTokenizer()
        self.unknown_prompt_tensor = unknown_prompt_tensor
        self.bad_assistant_prefix = bad_assistant_prefix
        self.template_calls: list[tuple[object, dict[str, object]]] = []
        self.decode_calls: list[tuple[object, dict[str, object]]] = []

    def apply_chat_template(self, conversations: object, **kwargs: object) -> _FakeBatchFeature:
        self.template_calls.append((conversations, kwargs))
        batch_size = len(conversations)  # type: ignore[arg-type]
        assistant_prefix = self.tokenizer.encode(
            GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
            add_special_tokens=False,
        )
        if self.bad_assistant_prefix:
            assistant_prefix = [501, 999]
        values = {
            "input_ids": _FakeTensor(
                [[10, 11, 12, 13, *assistant_prefix] for _ in range(batch_size)],
                dtype="torch.int64",
                name="input_ids",
            ),
            "attention_mask": _FakeTensor(
                [[1] * (4 + len(assistant_prefix)) for _ in range(batch_size)],
                dtype="torch.int64",
                name="attention_mask",
            ),
            "mm_token_type_ids": _FakeTensor(
                [[0] * (4 + len(assistant_prefix)) for _ in range(batch_size)],
                dtype="torch.int64",
                name="mm_token_type_ids",
            ),
            "image_grid_thw": _FakeTensor(
                [[1, 2, 2] for _ in range(batch_size)],
                dtype="torch.int64",
                name="image_grid_thw",
            ),
        }
        if self.unknown_prompt_tensor:
            values["position_ids"] = _FakeTensor(
                [
                    list(range(4 + len(assistant_prefix)))
                    for _ in range(batch_size)
                ],
                dtype="torch.int64",
                name="position_ids",
            )
        return _FakeBatchFeature(values)

    def batch_decode(self, values: object, **kwargs: object) -> list[str]:
        self.decode_calls.append((values, kwargs))
        return [
            "Action: Tap the visible target.\n"
            '<tool_call>\n{"name":"mobile_use","arguments":{"action":"click",'
            '"coordinate":[123,456]}}\n</tool_call>'
        ]


class _FakeModel:
    def __init__(self) -> None:
        self.forward_calls: list[dict[str, object]] = []
        self.generate_calls: list[dict[str, object]] = []
        self.last_logits: _FakeTensor | None = None

    def __call__(self, **kwargs: object) -> object:
        self.forward_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        action_tokens = kwargs["logits_to_keep"]
        assert isinstance(input_ids, _FakeTensor)
        assert isinstance(action_tokens, int)
        logits = _FakeTensor(
            None,
            dtype="torch.bfloat16",
            name="logits",
            shape=(input_ids.shape[0], action_tokens, 128),
        )
        self.last_logits = logits
        return types.SimpleNamespace(logits=logits)

    def generate(self, **kwargs: object) -> _FakeTensor:
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, _FakeTensor)
        rows = [[*row, 90, 91, 92] for row in input_ids.data]  # type: ignore[union-attr]
        return _FakeTensor(rows, dtype="torch.int64", name="generated")


def _native_messages(image: str) -> list[dict[str, object]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": GUI_OWL_V2_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Instruction and event summaries"},
                {"type": "image", "image": image},
                {"type": "text", "text": "Return the next action."},
            ],
        },
    ]


def _fake_runtime(
    *,
    unknown_prompt_tensor: bool = False,
    bad_assistant_prefix: bool = False,
) -> GUIOwlV2Runtime:
    runtime = object.__new__(GUIOwlV2Runtime)
    runtime.torch = _FakeTorch()
    runtime.device = "cuda:0"
    runtime.processor = _FakeProcessor(
        unknown_prompt_tensor=unknown_prompt_tensor,
        bad_assistant_prefix=bad_assistant_prefix,
    )
    runtime.model = _FakeModel()
    return runtime


class GUIOwlV2TeacherTokenTest(unittest.TestCase):
    def test_causal_shift_excludes_carrier_from_distance_span(self) -> None:
        action = GUIOwlV2Action(action="click", coordinate=(12, 34))
        tokens = build_gui_owl_v2_teacher_tokens(_CharacterTokenizer(), action)
        layout = gui_owl_v2_teacher_layout(
            prompt_input_tokens=7,
            teacher_tokens=tokens,
        )
        self.assertEqual(tokens.carrier_text, GUI_OWL_V2_TEACHER_CARRIER)
        self.assertEqual(tokens.distance_text, serialize_gui_owl_v2_tool_call(action))
        self.assertEqual(
            layout.forced_suffix_token_ids,
            (*tokens.carrier_token_ids, *tokens.distance_token_ids[:-1]),
        )
        self.assertEqual(
            layout.distance_logit_positions[0],
            7 + len(tokens.carrier_token_ids) - 1,
        )
        self.assertEqual(
            len(layout.distance_logit_positions),
            len(tokens.distance_token_ids),
        )
        carrier_logit_positions = set(
            range(7 - 1, 7 + len(tokens.carrier_token_ids) - 1)
        )
        self.assertTrue(carrier_logit_positions.isdisjoint(layout.distance_logit_positions))

    def test_tokenizer_boundary_merge_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "boundary"):
            build_gui_owl_v2_teacher_tokens(
                _BoundaryMergingTokenizer(),
                GUIOwlV2Action(action="wait"),
            )
        with self.assertRaisesRegex(ValueError, "assistant-prefix/carrier"):
            build_gui_owl_v2_teacher_tokens(
                _AssistantBoundaryMergingTokenizer(),
                GUIOwlV2Action(action="wait"),
            )


class GUIOwlV2RuntimeFakeModelTest(unittest.TestCase):
    def test_exact_shape_batch_two_returns_gpu_bf16_distance_logits(self) -> None:
        runtime = _fake_runtime()
        messages = (_native_messages("first"), _native_messages("second"))
        actions = (
            GUIOwlV2Action(action="click", coordinate=(1, 2)),
            GUIOwlV2Action(action="click", coordinate=(1, 2)),
        )
        logits, metadata = runtime.teacher_forced_distance_logits(messages, actions)

        first_tokens = build_gui_owl_v2_teacher_tokens(runtime.processor.tokenizer, actions[0])
        second_tokens = build_gui_owl_v2_teacher_tokens(runtime.processor.tokenizer, actions[1])
        self.assertEqual(
            len(first_tokens.distance_token_ids),
            len(second_tokens.distance_token_ids),
        )
        self.assertEqual(
            tuple(logits.shape),
            (2, len(first_tokens.distance_token_ids), 128),
        )
        self.assertEqual(logits.device, "cuda:0")
        self.assertEqual(logits.dtype, "torch.bfloat16")
        self.assertEqual(logits.cpu_transfer_attempts, 0)
        self.assertEqual(metadata["batch_size"], 2)
        self.assertEqual(metadata["full_logit_tensor_host_transfers"], 0)
        self.assertEqual(
            metadata["extended_prompt_aligned_inputs"],
            ["attention_mask", "mm_token_type_ids"],
        )
        self.assertEqual([sample["image_count"] for sample in metadata["samples"]], [1, 1])
        self.assertEqual(
            [sample["effective_visual_tokens"] for sample in metadata["samples"]],
            [1, 1],
        )
        self.assertEqual(
            [sample["policy_visible_text_tokens"] for sample in metadata["samples"]],
            [5, 5],
        )
        self.assertEqual(metadata["peak_gpu_memory_allocated_bytes"], 123_456)
        self.assertEqual(metadata["peak_gpu_memory_reserved_bytes"], 234_567)
        self.assertGreaterEqual(metadata["latency_seconds"], 0.0)

        template_conversations, template_kwargs = runtime.processor.template_calls[0]
        self.assertEqual(template_conversations, list(messages))
        self.assertEqual(
            template_kwargs,
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
                "padding": False,
            },
        )
        self.assertNotIn("tools", template_kwargs)
        forward = runtime.model.forward_calls[0]
        self.assertEqual(forward["logits_to_keep"], len(first_tokens.distance_token_ids))
        self.assertIs(forward["use_cache"], False)
        self.assertIs(forward["return_dict"], True)
        expected_first_input = [
            10,
            11,
            12,
            13,
            501,
            502,
            *first_tokens.carrier_token_ids,
            *first_tokens.distance_token_ids[:-1],
        ]
        self.assertEqual(forward["input_ids"].data[0], expected_first_input)
        self.assertEqual(
            forward["attention_mask"].data[0],
            [1] * len(expected_first_input),
        )
        self.assertEqual(
            forward["mm_token_type_ids"].data[0],
            [0] * len(expected_first_input),
        )
        self.assertEqual(runtime.torch.async_assertions, [])
        self.assertEqual(runtime.torch.cuda.reset_calls, ["cuda:0"])
        self.assertEqual(runtime.torch.cuda.synchronize_calls, ["cuda:0", "cuda:0"])

    def test_batch_requires_one_shared_canonical_reference_action(self) -> None:
        runtime = _fake_runtime()
        with self.assertRaisesRegex(ValueError, "one canonical reference action"):
            runtime.teacher_forced_distance_logits(
                (_native_messages("first"), _native_messages("second")),
                (
                    GUIOwlV2Action(action="click", coordinate=(1, 2)),
                    GUIOwlV2Action(action="click", coordinate=(3, 4)),
                ),
            )
        self.assertEqual(runtime.model.forward_calls, [])

    def test_unknown_prompt_aligned_tensor_is_rejected_before_forward(self) -> None:
        runtime = _fake_runtime(unknown_prompt_tensor=True)
        with self.assertRaisesRegex(ValueError, "unknown prompt-aligned.*position_ids"):
            runtime.teacher_forced_distance_logits(
                (_native_messages("first"), _native_messages("second")),
                (GUIOwlV2Action(action="wait"), GUIOwlV2Action(action="wait")),
            )
        self.assertEqual(runtime.model.forward_calls, [])

    def test_processor_prompt_must_end_with_frozen_assistant_prefix(self) -> None:
        runtime = _fake_runtime(bad_assistant_prefix=True)
        with self.assertRaisesRegex(ValueError, "frozen assistant prefix"):
            runtime.prepare_native_message_shape(_native_messages("current"))
        self.assertEqual(runtime.model.forward_calls, [])

    def test_prompt_aligned_helper_rejects_unknown_rank_three_field(self) -> None:
        inputs = {
            "input_ids": _FakeTensor([[1, 2]], dtype="torch.int64"),
            "attention_mask": _FakeTensor([[1, 1]], dtype="torch.int64"),
            "unknown": _FakeTensor(
                None,
                dtype="torch.float32",
                shape=(1, 2, 8),
            ),
        }
        with self.assertRaisesRegex(ValueError, "unknown prompt-aligned"):
            prompt_aligned_input_keys(inputs)

    def test_generation_uses_frozen_deterministic_kwargs_and_strict_parser(self) -> None:
        runtime = _fake_runtime()
        result = runtime.generate_native_action(_native_messages("current"))
        self.assertEqual(result.parsed_output.canonical_action.action, "click")
        self.assertEqual(result.parsed_output.canonical_action.coordinate, (123, 456))
        self.assertEqual(result.metadata["generated_tokens"], 3)
        generate = runtime.model.generate_calls[0]
        self.assertIs(generate["do_sample"], False)
        self.assertEqual(generate["max_new_tokens"], FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS)
        self.assertNotIn("temperature", generate)
        self.assertNotIn("top_p", generate)
        self.assertEqual(result.metadata["peak_gpu_memory_allocated_bytes"], 123_456)
        self.assertEqual(result.metadata["peak_gpu_memory_reserved_bytes"], 234_567)
        self.assertEqual(runtime.torch.cuda.synchronize_calls, ["cuda:0", "cuda:0"])
        _, decode_kwargs = runtime.processor.decode_calls[0]
        self.assertEqual(
            decode_kwargs,
            {
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            },
        )

    def test_generation_parse_failure_preserves_raw_output_and_metadata(self) -> None:
        runtime = _fake_runtime()
        raw_output = "Action: malformed\n<tool_call>not-json</tool_call>"
        runtime.processor.batch_decode = lambda values, **kwargs: [raw_output]
        with self.assertRaises(GUIOwlV2GenerationParseError) as caught:
            runtime.generate_native_action(_native_messages("current"))
        self.assertEqual(caught.exception.output_text, raw_output)
        self.assertEqual(caught.exception.metadata["do_sample"], False)
        self.assertEqual(caught.exception.parse_error_type, "ValueError")

    def test_shape_preparation_does_not_execute_policy_forward(self) -> None:
        runtime = _fake_runtime()
        shape = runtime.prepare_native_message_shape(_native_messages("current"))
        self.assertEqual(shape["image_count"], 1)
        self.assertEqual(shape["sequence_length"], 6)
        self.assertEqual(shape["assistant_generation_prefix_tokens"], 2)
        self.assertIs(shape["policy_forward_executed"], False)
        self.assertEqual(runtime.model.forward_calls, [])
        self.assertEqual(runtime.model.generate_calls, [])

    def test_context_limit_comes_from_verified_text_config(self) -> None:
        runtime = _fake_runtime()
        runtime.model.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(max_position_embeddings=32_768)
        )
        self.assertEqual(runtime.maximum_context_tokens(), 32_768)
        runtime.model.config.text_config.max_position_embeddings = None
        with self.assertRaisesRegex(ValueError, "max_position_embeddings"):
            runtime.maximum_context_tokens()


if __name__ == "__main__":
    unittest.main()
