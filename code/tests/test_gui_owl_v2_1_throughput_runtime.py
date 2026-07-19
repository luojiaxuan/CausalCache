from __future__ import annotations

import inspect
import types
import unittest
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from typing import TypeVar
from unittest import mock

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import serialize_gui_owl_v2_1_teacher_target
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
from causalcache.policy.gui_owl_v2_1_throughput_runtime import (
    GUI_OWL_V2_1_THROUGHPUT_CUDA_PEAK_MEASUREMENT_OWNERS,
    GUIOwlV21ThroughputRuntime,
)


_BASE_MODULE = "causalcache.policy.gui_owl_v2_1_runtime"
_THROUGHPUT_MODULE = "causalcache.policy.gui_owl_v2_1_throughput_runtime"
_T = TypeVar("_T")


class _FakeTensor:
    def __init__(
        self,
        data: object,
        *,
        dtype: str,
        device: str = "cuda:0",
        shape: tuple[int, ...] | None = None,
    ) -> None:
        self.data = data
        self.dtype = dtype
        self.device = device
        self.shape = shape if shape is not None else self._shape(data)
        self.ndim = len(self.shape)
        self.requires_grad = False

    @classmethod
    def _shape(cls, data: object) -> tuple[int, ...]:
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
            return ()
        values = list(data)
        if not values:
            return (0,)
        return (len(values), *cls._shape(values[0]))

    def __getitem__(self, index: object) -> _FakeTensor:
        if isinstance(index, int):
            return _FakeTensor(
                self.data[index],  # type: ignore[index]
                dtype=self.dtype,
                device=self.device,
            )
        if not isinstance(index, tuple) or len(index) != 2:
            raise AssertionError(f"unexpected fake tensor index: {index!r}")
        rows, columns = index
        if rows != slice(None):
            raise AssertionError(f"unexpected fake tensor rows: {rows!r}")
        return _FakeTensor(
            [row[columns] for row in self.data],  # type: ignore[union-attr]
            dtype=self.dtype,
            device=self.device,
        )

    def detach(self) -> _FakeTensor:
        return self

    def to(self, *, device: str) -> _FakeTensor:
        if device != "cpu":
            raise AssertionError(f"unexpected fake tensor device: {device}")
        return self

    def tolist(self) -> object:
        return self.data


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


class _InferenceMode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        del args
        return None


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()

    def inference_mode(self) -> _InferenceMode:
        return _InferenceMode()


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = object()
        self.decode_calls: list[tuple[object, dict[str, object]]] = []
        self.output_text = serialize_gui_owl_v2_1_teacher_target(
            GUIOwlV2Action(action="wait")
        )

    def batch_decode(self, values: object, **kwargs: object) -> list[str]:
        self.decode_calls.append((values, kwargs))
        return [self.output_text]


class _FakeModel:
    def __init__(self, *, close_token_id: int) -> None:
        self.close_token_id = close_token_id
        self.forward_calls: list[dict[str, object]] = []
        self.generate_calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.forward_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        action_tokens = kwargs["logits_to_keep"]
        assert isinstance(input_ids, _FakeTensor)
        assert isinstance(action_tokens, int)
        return types.SimpleNamespace(
            logits=_FakeTensor(
                None,
                dtype="torch.bfloat16",
                shape=(input_ids.shape[0], action_tokens, 128),
            )
        )

    def generate(self, **kwargs: object) -> _FakeTensor:
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, _FakeTensor)
        return _FakeTensor(
            [
                [*row, 42, self.close_token_id]
                for row in input_ids.data  # type: ignore[union-attr]
            ],
            dtype="torch.int64",
        )


class _RuntimeHarness:
    close_token_id = 99

    def __init__(self) -> None:
        self.torch = _FakeTorch()
        self.device = "cuda:0"
        self.encode_calls: list[object] = []
        self.processor = _FakeProcessor()
        self.model = _FakeModel(close_token_id=self.close_token_id)
        self.generation_tokens = types.SimpleNamespace(
            tool_call_close_token_id=self.close_token_id,
            standard_eos_token_ids=(1, 2),
            pad_token_id=0,
        )

    def _encode_exact_batch(
        self,
        messages_batch: object,
    ) -> tuple[dict[str, object], tuple[int, ...]]:
        self.encode_calls.append(messages_batch)
        batch_size = len(messages_batch)  # type: ignore[arg-type]
        return (
            {
                "input_ids": _FakeTensor(
                    [[10, 11, 12] for _ in range(batch_size)],
                    dtype="torch.int64",
                )
            },
            (1,) * batch_size,
        )

    def _image_metadata(
        self,
        model_inputs: object,
        image_counts: tuple[int, ...],
    ) -> tuple[dict[str, object], ...]:
        del model_inputs
        return tuple({"image_count": count} for count in image_counts)

    def _interface_metadata(self) -> dict[str, object]:
        return {"interface": "frozen"}


class _BaseHarness(_RuntimeHarness, GUIOwlV21OfficialToolsRuntime):
    pass


class _ThroughputHarness(_RuntimeHarness, GUIOwlV21ThroughputRuntime):
    pass


def _fixed_clock_call(callable_: Callable[[], _T]) -> _T:
    with mock.patch(
        f"{_THROUGHPUT_MODULE}.time.perf_counter",
        side_effect=(10.0, 10.25),
    ):
        return callable_()


@contextmanager
def _patched_teacher_dependencies() -> Iterator[None]:
    teacher_tokens = types.SimpleNamespace(
        distance_text="target",
        distance_token_ids=(70, 71),
    )
    layout = types.SimpleNamespace(
        forced_suffix_token_ids=(70,),
        distance_action_tokens=2,
        model_input_tokens=4,
        distance_logit_positions=(2, 3),
    )

    def extend(
        model_inputs: dict[str, object],
        suffixes: object,
        *,
        tensor_module: object,
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        del suffixes, tensor_module
        return model_inputs, ("input_ids",)

    def mask(
        raw_logits: object,
        *,
        standard_eos_token_ids: object,
        tensor_module: object,
    ) -> tuple[object, float]:
        del standard_eos_token_ids, tensor_module
        return raw_logits, -123.0

    with ExitStack() as stack:
        for module in (_BASE_MODULE, _THROUGHPUT_MODULE):
            stack.enter_context(
                mock.patch(
                    f"{module}.build_gui_owl_v2_1_teacher_tokens",
                    return_value=teacher_tokens,
                )
            )
            stack.enter_context(
                mock.patch(
                    f"{module}.gui_owl_v2_1_teacher_layout",
                    return_value=layout,
                )
            )
            stack.enter_context(
                mock.patch(
                    f"{module}.extend_gui_owl_v2_teacher_inputs",
                    side_effect=extend,
                )
            )
            stack.enter_context(
                mock.patch(
                    f"{module}.mask_gui_owl_v2_1_teacher_standard_eos_logits",
                    side_effect=mask,
                )
            )
        yield


class GUIOwlV21ThroughputRuntimeTest(unittest.TestCase):
    def test_owner_is_keyword_only_and_versioned_runtime_subclasses_pinned_base(self) -> None:
        self.assertTrue(issubclass(GUIOwlV21ThroughputRuntime, GUIOwlV21OfficialToolsRuntime))
        self.assertEqual(
            GUI_OWL_V2_1_THROUGHPUT_CUDA_PEAK_MEASUREMENT_OWNERS,
            ("runtime", "caller"),
        )
        for method_name in ("generate_native_action", "teacher_forced_distance_logits"):
            parameter = inspect.signature(
                getattr(GUIOwlV21ThroughputRuntime, method_name)
            ).parameters["cuda_peak_measurement_owner"]
            self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertEqual(parameter.default, "runtime")

    def test_generation_default_and_caller_owned_paths_match_pinned_base(self) -> None:
        base = _BaseHarness()
        default = _ThroughputHarness()
        caller = _ThroughputHarness()

        base_result = _fixed_clock_call(
            lambda: base.generate_native_action(({"role": "user"},))
        )
        default_result = _fixed_clock_call(
            lambda: default.generate_native_action(({"role": "user"},))
        )
        caller_result = _fixed_clock_call(
            lambda: caller.generate_native_action(
                ({"role": "user"},),
                cuda_peak_measurement_owner="caller",
            )
        )

        self.assertEqual(base.torch.cuda.reset_calls, ["cuda:0"])
        self.assertEqual(default.torch.cuda.reset_calls, ["cuda:0"])
        self.assertEqual(caller.torch.cuda.reset_calls, [])
        self.assertEqual(base_result, default_result)
        self.assertEqual(default_result, caller_result)
        for runtime in (base, default, caller):
            self.assertEqual(runtime.torch.cuda.synchronize_calls, ["cuda:0"] * 2)
            self.assertEqual(len(runtime.model.generate_calls), 1)
            self.assertEqual(len(runtime.processor.decode_calls), 1)

    def test_teacher_default_and_caller_owned_paths_match_pinned_base(self) -> None:
        base = _BaseHarness()
        default = _ThroughputHarness()
        caller = _ThroughputHarness()
        action = GUIOwlV2Action(action="wait")
        messages = (({"role": "user"},), ({"role": "user"},))
        with _patched_teacher_dependencies():
            base_logits, base_metadata = _fixed_clock_call(
                lambda: base.teacher_forced_distance_logits(messages, (action, action))
            )
            default_logits, default_metadata = _fixed_clock_call(
                lambda: default.teacher_forced_distance_logits(
                    messages,
                    (action, action),
                )
            )
            caller_logits, caller_metadata = _fixed_clock_call(
                lambda: caller.teacher_forced_distance_logits(
                    messages,
                    (action, action),
                    cuda_peak_measurement_owner="caller",
                )
            )

        self.assertEqual(base.torch.cuda.reset_calls, ["cuda:0"])
        self.assertEqual(default.torch.cuda.reset_calls, ["cuda:0"])
        self.assertEqual(caller.torch.cuda.reset_calls, [])
        self.assertEqual(base_metadata, default_metadata)
        self.assertEqual(default_metadata, caller_metadata)
        self.assertEqual(
            (base_logits.shape, base_logits.dtype, base_logits.device),
            (default_logits.shape, default_logits.dtype, default_logits.device),
        )
        self.assertEqual(
            (default_logits.shape, default_logits.dtype, default_logits.device),
            (caller_logits.shape, caller_logits.dtype, caller_logits.device),
        )
        for runtime in (base, default, caller):
            self.assertEqual(runtime.torch.cuda.synchronize_calls, ["cuda:0"] * 2)
            self.assertEqual(len(runtime.model.forward_calls), 1)

    def test_unknown_measurement_owner_fails_without_reset(self) -> None:
        runtime = _ThroughputHarness()
        with self.assertRaisesRegex(ValueError, "runtime.*caller"):
            runtime.generate_native_action(
                ({"role": "user"},),
                cuda_peak_measurement_owner="external",
            )
        self.assertEqual(runtime.torch.cuda.reset_calls, [])
        self.assertEqual(runtime.encode_calls, [])


if __name__ == "__main__":
    unittest.main()
