from __future__ import annotations

import hashlib
import json
import tempfile
import types
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    GUI_OWL_V2_1_MOBILE_USE_TOOL,
    GUI_OWL_V2_1_SYSTEM_PROMPT,
    serialize_gui_owl_v2_1_teacher_target,
)
from causalcache.policy.gui_owl_v2_1_runtime import (
    GUI_OWL_V2_1_EXPECTED_ASSISTANT_PREFIX_TOKEN_IDS,
    GUI_OWL_V2_1_EXPECTED_CHAT_TEMPLATE_FILE_SHA256,
    GUI_OWL_V2_1_EXPECTED_CHAT_TEMPLATE_TEXT_SHA256,
    GUI_OWL_V2_1_EXPECTED_PAD_TOKEN_ID,
    GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS,
    GUI_OWL_V2_1_EXPECTED_TOOL_CALL_CLOSE_TOKEN_ID,
    GUI_OWL_V2_1_EXPECTED_TOOL_CALL_OPEN_TOKEN_ID,
    GUIOwlV21GenerationParseError,
    GUIOwlV21ChatTemplateIdentity,
    GUIOwlV21OfficialToolsRuntime,
    build_gui_owl_v2_1_teacher_tokens,
    gui_owl_v2_1_teacher_layout,
    validate_gui_owl_v2_1_chat_template,
    validate_gui_owl_v2_1_generation_tokens,
)
from causalcache.policy.gui_owl_v2_runtime import (
    FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS,
    GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
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
        self.masked_columns: dict[int, float] = {}
        self.shape = shape if shape is not None else self._shape(data)
        self.ndim = len(self.shape)

    @classmethod
    def _shape(cls, data: object) -> tuple[int, ...]:
        if not isinstance(data, Sequence) or isinstance(
            data,
            (str, bytes, bytearray),
        ):
            return ()
        values = list(data)
        if not values:
            return (0,)
        return (len(values), *cls._shape(values[0]))

    def detach(self) -> _FakeTensor:
        return self

    def to(
        self,
        device: object = None,
        dtype: object = None,
        **kwargs: object,
    ) -> _FakeTensor:
        del dtype
        requested_device = kwargs.get("device", device)
        if requested_device == "cpu":
            self.cpu_transfer_attempts += 1
            if self.name == "logits":
                raise AssertionError("full logits must never move to CPU")
        return self

    def tolist(self) -> object:
        if self.name == "logits":
            raise AssertionError("full logits must never be materialized on the host")
        return self.data

    def clone(self) -> _FakeTensor:
        cloned = _FakeTensor(
            self.data,
            dtype=self.dtype,
            device=self.device,
            name=f"cloned_{self.name}",
            shape=self.shape,
        )
        cloned.masked_columns = dict(self.masked_columns)
        return cloned

    def __setitem__(self, index: object, value: object) -> None:
        if (
            not isinstance(index, tuple)
            or len(index) != 2
            or index[0] is not Ellipsis
            or not isinstance(index[1], list)
            or type(value) is not float
        ):
            raise AssertionError(f"unexpected fake tensor assignment: {index!r}")
        for token_id in index[1]:
            self.masked_columns[int(token_id)] = value

    def __getitem__(self, index: object) -> _FakeTensor:
        if isinstance(index, int):
            row = self.data[index]  # type: ignore[index]
            return _FakeTensor(
                row,
                dtype=self.dtype,
                device=self.device,
                name=f"{self.name}_row",
            )
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


class _InferenceMode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class _FakeCuda:
    def __init__(self) -> None:
        self.reset_calls: list[object] = []
        self.synchronize_calls: list[object] = []

    def reset_peak_memory_stats(self, device: object) -> None:
        self.reset_calls.append(device)

    def synchronize(self, device: object) -> None:
        self.synchronize_calls.append(device)

    def max_memory_allocated(self, device: object) -> int:
        del device
        return 123_456

    def max_memory_reserved(self, device: object) -> int:
        del device
        return 234_567


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()

    def inference_mode(self) -> _InferenceMode:
        return _InferenceMode()

    def finfo(self, dtype: object) -> object:
        if dtype != "torch.bfloat16":
            raise AssertionError(f"unexpected fake finfo dtype: {dtype}")
        return types.SimpleNamespace(min=-3.3895313892515355e38)

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
            raise AssertionError("fake torch supports only rank-2 dim-1 cat")
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


class _FakeBatchFeature(dict[str, _FakeTensor]):
    def to(self, device: object) -> _FakeBatchFeature:
        del device
        return self


class _Tokenizer:
    eos_token_id = GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS[0]
    pad_token_id = GUI_OWL_V2_1_EXPECTED_PAD_TOKEN_ID
    chat_template = "fake official tools chat template"

    def __init__(self) -> None:
        self.all_special_ids = [
            *GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS,
            GUI_OWL_V2_1_EXPECTED_ASSISTANT_PREFIX_TOKEN_IDS[0],
        ]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("runtime tokenization must disable added special tokens")
        pieces = (
            (
                GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
                GUI_OWL_V2_1_EXPECTED_ASSISTANT_PREFIX_TOKEN_IDS,
            ),
            ("<tool_call>", (GUI_OWL_V2_1_EXPECTED_TOOL_CALL_OPEN_TOKEN_ID,)),
            ("</tool_call>", (GUI_OWL_V2_1_EXPECTED_TOOL_CALL_CLOSE_TOKEN_ID,)),
        )
        result: list[int] = []
        offset = 0
        while offset < len(text):
            matched = False
            for piece, token_ids in pieces:
                if text.startswith(piece, offset):
                    result.extend(token_ids)
                    offset += len(piece)
                    matched = True
                    break
            if not matched:
                result.append(ord(text[offset]) + 1_000)
                offset += 1
        return result

    def decode(self, token_ids: object, *, skip_special_tokens: bool) -> str:
        values = list(token_ids)
        if values == [GUI_OWL_V2_1_EXPECTED_TOOL_CALL_CLOSE_TOKEN_ID]:
            return "</tool_call>"
        if skip_special_tokens:
            values = [value for value in values if value not in self.all_special_ids]
        return "".join(chr(value - 1_000) for value in values if value >= 1_000)


class _BoundaryMergingTokenizer(_Tokenizer):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        values = super().encode(text, add_special_tokens=add_special_tokens)
        if text.startswith(GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX + "<tool_call>"):
            return values[1:]
        return values


class _TeacherEOSInjectingTokenizer(_Tokenizer):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        values = super().encode(text, add_special_tokens=add_special_tokens)
        if "<tool_call>\n{" in text and '"mobile_use"' in text:
            distance_start = (
                len(GUI_OWL_V2_1_EXPECTED_ASSISTANT_PREFIX_TOKEN_IDS)
                if text.startswith(GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX)
                else 0
            )
            values[distance_start + 1] = (
                GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS[0]
            )
        return values


class _FakeProcessor:
    def __init__(self, *, output_text: str | None = None) -> None:
        self.tokenizer = _Tokenizer()
        self.output_text = output_text or serialize_gui_owl_v2_1_teacher_target(
            GUIOwlV2Action(action="click", coordinate=(123, 456))
        )
        self.template_calls: list[tuple[object, dict[str, object]]] = []
        self.decode_calls: list[tuple[object, dict[str, object]]] = []

    def apply_chat_template(
        self,
        conversations: object,
        **kwargs: object,
    ) -> _FakeBatchFeature:
        self.template_calls.append((conversations, kwargs))
        batch_size = len(conversations)  # type: ignore[arg-type]
        prefix = list(GUI_OWL_V2_1_EXPECTED_ASSISTANT_PREFIX_TOKEN_IDS)
        prompt = [
            10,
            GUI_OWL_V2_1_EXPECTED_TOOL_CALL_OPEN_TOKEN_ID,
            GUI_OWL_V2_1_EXPECTED_TOOL_CALL_CLOSE_TOKEN_ID,
            11,
            *prefix,
        ]
        return _FakeBatchFeature(
            {
                "input_ids": _FakeTensor(
                    [list(prompt) for _ in range(batch_size)],
                    dtype="torch.int64",
                    name="input_ids",
                ),
                "attention_mask": _FakeTensor(
                    [[1] * len(prompt) for _ in range(batch_size)],
                    dtype="torch.int64",
                    name="attention_mask",
                ),
                "mm_token_type_ids": _FakeTensor(
                    [[0] * len(prompt) for _ in range(batch_size)],
                    dtype="torch.int64",
                    name="mm_token_type_ids",
                ),
                "image_grid_thw": _FakeTensor(
                    [[1, 2, 2] for _ in range(batch_size)],
                    dtype="torch.int64",
                    name="image_grid_thw",
                ),
            }
        )

    def batch_decode(self, values: object, **kwargs: object) -> list[str]:
        self.decode_calls.append((values, kwargs))
        return [self.output_text]


class _FakeModel:
    def __init__(self, *, generated_suffix: list[int] | None = None) -> None:
        self.generation_config = types.SimpleNamespace(
            eos_token_id=list(GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS),
            pad_token_id=GUI_OWL_V2_1_EXPECTED_PAD_TOKEN_ID,
            forced_eos_token_id=None,
            stop_strings=None,
        )
        self.generated_suffix = generated_suffix or [
            90,
            91,
            GUI_OWL_V2_1_EXPECTED_TOOL_CALL_CLOSE_TOKEN_ID,
        ]
        self.forward_calls: list[dict[str, object]] = []
        self.generate_calls: list[dict[str, object]] = []
        self.raw_logits: _FakeTensor | None = None

    def __call__(self, **kwargs: object) -> object:
        self.forward_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        action_tokens = kwargs["logits_to_keep"]
        assert isinstance(input_ids, _FakeTensor)
        assert isinstance(action_tokens, int)
        self.raw_logits = _FakeTensor(
            None,
            dtype="torch.bfloat16",
            name="logits",
            shape=(input_ids.shape[0], action_tokens, 151_669),
        )
        return types.SimpleNamespace(logits=self.raw_logits)

    def generate(self, **kwargs: object) -> _FakeTensor:
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, _FakeTensor)
        rows = [
            [*row, *self.generated_suffix]
            for row in input_ids.data  # type: ignore[union-attr]
        ]
        return _FakeTensor(rows, dtype="torch.int64", name="generated")


def _messages(image: str) -> list[dict[str, object]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": GUI_OWL_V2_1_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Instruction and summaries"},
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
                },
            ],
        },
    ]


def _runtime(
    *,
    output_text: str | None = None,
    generated_suffix: list[int] | None = None,
) -> GUIOwlV21OfficialToolsRuntime:
    runtime = object.__new__(GUIOwlV21OfficialToolsRuntime)
    runtime.torch = _FakeTorch()
    runtime.device = "cuda:0"
    runtime.processor = _FakeProcessor(output_text=output_text)
    runtime.model = _FakeModel(generated_suffix=generated_suffix)
    runtime.chat_template_identity = GUIOwlV21ChatTemplateIdentity(
        file_sha256=GUI_OWL_V2_1_EXPECTED_CHAT_TEMPLATE_FILE_SHA256,
        text_sha256=GUI_OWL_V2_1_EXPECTED_CHAT_TEMPLATE_TEXT_SHA256,
    )
    runtime.generation_tokens = validate_gui_owl_v2_1_generation_tokens(
        runtime.processor.tokenizer,
        runtime.model.generation_config,
    )
    runtime.metadata = runtime._interface_metadata()
    return runtime


class GUIOwlV21TeacherTokenTest(unittest.TestCase):
    def test_direct_official_span_has_no_action_carrier(self) -> None:
        action = GUIOwlV2Action(action="click", coordinate=(12, 34))
        tokens = build_gui_owl_v2_1_teacher_tokens(_Tokenizer(), action)
        layout = gui_owl_v2_1_teacher_layout(
            prompt_input_tokens=7,
            teacher_tokens=tokens,
        )
        self.assertEqual(
            tokens.distance_text,
            serialize_gui_owl_v2_1_teacher_target(action),
        )
        self.assertNotIn("Action:", tokens.distance_text)
        self.assertEqual(
            tokens.distance_token_ids[0],
            GUI_OWL_V2_1_EXPECTED_TOOL_CALL_OPEN_TOKEN_ID,
        )
        self.assertEqual(
            tokens.distance_token_ids[-1],
            GUI_OWL_V2_1_EXPECTED_TOOL_CALL_CLOSE_TOKEN_ID,
        )
        self.assertEqual(layout.distance_logit_positions[0], 6)
        self.assertEqual(
            layout.distance_logit_positions[-1],
            6 + len(tokens.distance_token_ids) - 1,
        )
        self.assertEqual(
            layout.forced_suffix_token_ids,
            tokens.distance_token_ids[:-1],
        )
        self.assertEqual(layout.model_input_tokens, 7 + len(tokens.distance_token_ids) - 1)

    def test_assistant_tool_boundary_merge_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "boundary"):
            build_gui_owl_v2_1_teacher_tokens(
                _BoundaryMergingTokenizer(),
                GUIOwlV2Action(action="wait"),
            )

    def test_teacher_target_cannot_contain_a_suppressed_standard_eos(self) -> None:
        with self.assertRaisesRegex(ValueError, "suppressed standard EOS"):
            build_gui_owl_v2_1_teacher_tokens(
                _TeacherEOSInjectingTokenizer(),
                GUIOwlV2Action(action="wait"),
            )

    def test_generation_token_contract_rejects_eos_and_special_tag_drift(self) -> None:
        tokenizer = _Tokenizer()
        config = _FakeModel().generation_config
        contract = validate_gui_owl_v2_1_generation_tokens(tokenizer, config)
        self.assertEqual(
            contract.tool_call_close_token_id,
            GUI_OWL_V2_1_EXPECTED_TOOL_CALL_CLOSE_TOKEN_ID,
        )
        config.eos_token_id = [151645]
        with self.assertRaisesRegex(ValueError, "standard EOS"):
            validate_gui_owl_v2_1_generation_tokens(tokenizer, config)
        config.eos_token_id = list(GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS)
        tokenizer.all_special_ids.append(GUI_OWL_V2_1_EXPECTED_TOOL_CALL_CLOSE_TOKEN_ID)
        with self.assertRaisesRegex(ValueError, "tool tags"):
            validate_gui_owl_v2_1_generation_tokens(tokenizer, config)

    def test_chat_template_file_and_in_memory_value_are_both_bound(self) -> None:
        template = "official {{ tools }} template"
        payload = json.dumps({"chat_template": template}).encode("utf-8")
        file_sha = hashlib.sha256(payload).hexdigest()
        text_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat_template.json"
            path.write_bytes(payload)
            tokenizer = types.SimpleNamespace(chat_template=template)
            with (
                mock.patch(
                    "causalcache.policy.gui_owl_v2_1_runtime."
                    "GUI_OWL_V2_1_EXPECTED_CHAT_TEMPLATE_FILE_SHA256",
                    file_sha,
                ),
                mock.patch(
                    "causalcache.policy.gui_owl_v2_1_runtime."
                    "GUI_OWL_V2_1_EXPECTED_CHAT_TEMPLATE_TEXT_SHA256",
                    text_sha,
                ),
            ):
                identity = validate_gui_owl_v2_1_chat_template(
                    model_dir=directory,
                    tokenizer=tokenizer,
                )
                self.assertEqual(identity.file_sha256, file_sha)
                tokenizer.chat_template += " drift"
                with self.assertRaisesRegex(ValueError, "in-memory"):
                    validate_gui_owl_v2_1_chat_template(
                        model_dir=directory,
                        tokenizer=tokenizer,
                    )


class GUIOwlV21RuntimeFakeModelTest(unittest.TestCase):
    def test_generation_uses_official_tools_and_model_emitted_close_as_sole_eos(self) -> None:
        runtime = _runtime()
        result = runtime.generate_native_action(_messages("current"))
        self.assertEqual(result.parsed_output.canonical_action.action, "click")
        self.assertEqual(result.metadata["termination_reason"], "model_emitted_tool_call_close")
        self.assertIs(result.metadata["model_emitted_tool_call_close"], True)
        self.assertIs(result.metadata["host_injected_tool_call_closer"], False)
        self.assertEqual(
            result.metadata["final_generated_token_id"],
            GUI_OWL_V2_1_EXPECTED_TOOL_CALL_CLOSE_TOKEN_ID,
        )
        self.assertEqual(len(result.metadata["generated_token_ids_sha256"]), 64)
        self.assertEqual(len(result.metadata["decoded_output_utf8_sha256"]), 64)
        conversations, template_kwargs = runtime.processor.template_calls[0]
        self.assertEqual(conversations, [_messages("current")])
        self.assertEqual(template_kwargs["tools"], [GUI_OWL_V2_1_MOBILE_USE_TOOL])
        self.assertNotIn("Action:", str(template_kwargs["tools"]))
        self.assertEqual(
            {key: value for key, value in template_kwargs.items() if key != "tools"},
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
                "padding": False,
            },
        )
        generate = runtime.model.generate_calls[0]
        self.assertIs(generate["do_sample"], False)
        self.assertEqual(generate["max_new_tokens"], FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS)
        self.assertEqual(
            generate["eos_token_id"],
            GUI_OWL_V2_1_EXPECTED_TOOL_CALL_CLOSE_TOKEN_ID,
        )
        self.assertEqual(
            generate["suppress_tokens"],
            list(GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS),
        )
        self.assertEqual(generate["pad_token_id"], GUI_OWL_V2_1_EXPECTED_PAD_TOKEN_ID)
        self.assertEqual(generate["num_beams"], 1)
        self.assertEqual(generate["num_return_sequences"], 1)
        self.assertIs(generate["return_dict_in_generate"], False)
        self.assertEqual(result.metadata["num_beams"], 1)
        self.assertEqual(result.metadata["num_return_sequences"], 1)
        self.assertNotIn("temperature", generate)
        self.assertNotIn("top_p", generate)
        _, decode_kwargs = runtime.processor.decode_calls[0]
        self.assertEqual(
            decode_kwargs,
            {
                "skip_special_tokens": False,
                "clean_up_tokenization_spaces": False,
            },
        )

    def test_teacher_uses_same_tools_prompt_and_predicts_close_in_distance_span(self) -> None:
        runtime = _runtime()
        messages = (_messages("first"), _messages("second"))
        action = GUIOwlV2Action(action="wait")
        logits, metadata = runtime.teacher_forced_distance_logits(
            messages,
            (action, action),
        )
        tokens = build_gui_owl_v2_1_teacher_tokens(runtime.processor.tokenizer, action)
        self.assertEqual(
            tuple(logits.shape),
            (2, len(tokens.distance_token_ids), 151_669),
        )
        self.assertEqual(metadata["teacher_carrier"], None)
        self.assertEqual(metadata["samples"][0]["teacher_carrier_tokens"], 0)
        self.assertEqual(
            metadata["distance_span"],
            "official_tool_call_open_through_close_inclusive",
        )
        self.assertIs(metadata["teacher_target_ends_with_model_generation_eos"], True)
        self.assertEqual(metadata["logits_to_keep"], len(tokens.distance_token_ids))
        self.assertEqual(
            metadata["teacher_standard_eos_suppressed_token_ids"],
            list(GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS),
        )
        self.assertIs(
            metadata["teacher_target_disjoint_from_suppressed_standard_eos"],
            True,
        )
        self.assertEqual(
            metadata["teacher_standard_eos_mask_application"],
            "same_mask_on_every_reference_and_candidate_action_path_position_"
            "before_float32_log_softmax",
        )
        self.assertTrue(
            all(
                logits.masked_columns[token_id]
                == metadata["teacher_standard_eos_suppression_value"]
                for token_id in GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS
            )
        )
        self.assertEqual(runtime.model.raw_logits.masked_columns, {})
        self.assertIs(metadata["teacher_raw_logits_mutated"], False)
        _, template_kwargs = runtime.processor.template_calls[0]
        self.assertEqual(template_kwargs["tools"], [GUI_OWL_V2_1_MOBILE_USE_TOOL])
        forward = runtime.model.forward_calls[0]
        prompt = [
            10,
            GUI_OWL_V2_1_EXPECTED_TOOL_CALL_OPEN_TOKEN_ID,
            GUI_OWL_V2_1_EXPECTED_TOOL_CALL_CLOSE_TOKEN_ID,
            11,
            *GUI_OWL_V2_1_EXPECTED_ASSISTANT_PREFIX_TOKEN_IDS,
        ]
        self.assertEqual(
            forward["input_ids"].data[0],
            [*prompt, *tokens.distance_token_ids[:-1]],
        )
        self.assertNotIn(
            GUI_OWL_V2_1_EXPECTED_TOOL_CALL_CLOSE_TOKEN_ID,
            tokens.distance_token_ids[:-1][-1:],
        )
        self.assertEqual(
            metadata["samples"][0]["distance_logit_positions"][-1],
            len(prompt) - 1 + len(tokens.distance_token_ids) - 1,
        )

    def test_strict_parse_failure_preserves_model_close_and_does_not_recover(self) -> None:
        runtime = _runtime(output_text="<tool_call>\nnot-json\n</tool_call>")
        with self.assertRaises(GUIOwlV21GenerationParseError) as caught:
            runtime.generate_native_action(_messages("current"))
        self.assertEqual(
            caught.exception.output_text,
            "<tool_call>\nnot-json\n</tool_call>",
        )
        self.assertIs(caught.exception.metadata["model_emitted_tool_call_close"], True)
        self.assertIs(caught.exception.metadata["output_recovery_or_normalization"], False)

    def test_max_tokens_without_model_close_is_a_parse_failure_not_completion(self) -> None:
        runtime = _runtime(
            output_text='<tool_call>\n{"name": "mobile_use"',
            generated_suffix=[90] * FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS,
        )
        with self.assertRaises(GUIOwlV21GenerationParseError) as caught:
            runtime.generate_native_action(_messages("current"))
        self.assertEqual(
            caught.exception.metadata["termination_reason"],
            "max_new_tokens_without_tool_call_close",
        )
        self.assertIs(caught.exception.metadata["model_emitted_tool_call_close"], False)
        self.assertIs(caught.exception.metadata["host_injected_tool_call_closer"], False)

    def test_unexpected_short_termination_and_suppressed_eos_are_runtime_errors(self) -> None:
        short = _runtime(generated_suffix=[90, 91])
        with self.assertRaisesRegex(RuntimeError, "before close or max_new_tokens"):
            short.generate_native_action(_messages("current"))
        suppressed = _runtime(
            generated_suffix=[GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS[0]]
        )
        with self.assertRaisesRegex(RuntimeError, "suppressed standard EOS"):
            suppressed.generate_native_action(_messages("current"))

    def test_shape_preparation_records_interface_without_policy_forward(self) -> None:
        runtime = _runtime()
        shape = runtime.prepare_native_message_shape(_messages("current"))
        self.assertEqual(shape["image_count"], 1)
        self.assertEqual(shape["assistant_generation_prefix_tokens"], 3)
        self.assertEqual(shape["official_tools_argument_count"], 1)
        self.assertIs(shape["policy_forward_executed"], False)
        self.assertEqual(runtime.model.forward_calls, [])
        self.assertEqual(runtime.model.generate_calls, [])


if __name__ == "__main__":
    unittest.main()
