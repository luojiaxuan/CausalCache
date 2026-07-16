from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

from causalcache.policy.gui_owl_v2_1_runtime import (
    GUIOwlV21GenerationParseError,
    GUIOwlV21GenerationResult,
    GUIOwlV21OfficialToolsRuntime,
)
from causalcache.policy.gui_owl_v2_2_eager_runtime import (
    GUI_OWL_V2_2_EAGER_ATTENTION_IMPLEMENTATION,
    GUI_OWL_V2_2_EAGER_EXPECTED_GPU_NAME,
    GUI_OWL_V2_2_EAGER_EXPECTED_SOFTWARE_STACK,
    GUI_OWL_V2_2_EAGER_NUMERICAL_CONTROL_CLAIM,
    GUI_OWL_V2_2_EAGER_RUNTIME_ID,
    GUI_OWL_V2_2_EAGER_SEED,
    GUIOwlV22EagerRuntime,
    _configure_and_validate_numerical_controls,
    audit_gui_owl_v2_2_scientific_environment,
    gui_owl_v2_2_observed_attention,
    validate_gui_owl_v2_2_eager_attention,
)
from causalcache.spatial_reference_audit_v1 import (
    AUDITED_SCIENTIFIC_ENVIRONMENT_VARIABLES,
)


class _FakeDType:
    def __str__(self) -> str:
        return "torch.bfloat16"


class _FakeDevice:
    def __init__(self, value: str) -> None:
        self.value = value
        self.index = int(value.split(":", 1)[1])

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeDevice) and self.value == other.value


class _FakeCuda:
    def __init__(self) -> None:
        self.seed_calls: list[int] = []

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 2

    def manual_seed_all(self, value: int) -> None:
        self.seed_calls.append(value)

    def get_device_name(self, device: _FakeDevice) -> str:
        if not isinstance(device, _FakeDevice):
            raise TypeError("fake CUDA device identity drifted")
        return GUI_OWL_V2_2_EAGER_EXPECTED_GPU_NAME


class _FakeTorch(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
        self.__version__ = "2.11.0+cu130"
        self.version = types.SimpleNamespace(cuda="13.0")
        self.bfloat16 = _FakeDType()
        self.cuda = _FakeCuda()
        self.backends = types.SimpleNamespace(
            cudnn=types.SimpleNamespace(
                deterministic=False,
                benchmark=True,
                allow_tf32=True,
                version=lambda: 91900,
            ),
            cuda=types.SimpleNamespace(
                matmul=types.SimpleNamespace(allow_tf32=True),
            ),
        )
        self.deterministic_algorithms = True
        self.deterministic_warn_only = True
        self.matmul_precision = "medium"
        self.seed_calls: list[int] = []
        self.deterministic_calls: list[tuple[bool, bool]] = []

    def device(self, value: str) -> _FakeDevice:
        return _FakeDevice(value)

    def use_deterministic_algorithms(
        self,
        enabled: bool,
        *,
        warn_only: bool,
    ) -> None:
        self.deterministic_calls.append((enabled, warn_only))
        self.deterministic_algorithms = enabled
        self.deterministic_warn_only = warn_only

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.deterministic_algorithms

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self.deterministic_warn_only

    def manual_seed(self, value: int) -> None:
        self.seed_calls.append(value)

    def set_float32_matmul_precision(self, value: str) -> None:
        self.matmul_precision = value

    def get_float32_matmul_precision(self) -> str:
        return self.matmul_precision


class _FakeParameter:
    def __init__(self, *, device: _FakeDevice, dtype: object) -> None:
        self.device = device
        self.dtype = dtype
        self.requires_grad = False

    def is_floating_point(self) -> bool:
        return True


class _FakeModel:
    def __init__(self, torch: _FakeTorch, *, vision_attention: str = "eager") -> None:
        self.torch = torch
        self.device: _FakeDevice | None = None
        self.config = types.SimpleNamespace(
            _attn_implementation="eager",
            text_config=types.SimpleNamespace(_attn_implementation="eager"),
            vision_config=types.SimpleNamespace(
                _attn_implementation=vision_attention
            ),
        )
        self.generation_config = types.SimpleNamespace()
        self.parameter: _FakeParameter | None = None

    def to(self, device: _FakeDevice) -> _FakeModel:
        self.device = device
        self.parameter = _FakeParameter(device=device, dtype=self.torch.bfloat16)
        return self

    def eval(self) -> _FakeModel:
        return self

    def requires_grad_(self, value: bool) -> _FakeModel:
        if value:
            raise AssertionError("frozen fake model cannot enable gradients")
        return self

    def parameters(self) -> tuple[_FakeParameter, ...]:
        if self.parameter is None:
            raise AssertionError("fake model must move to a device first")
        return (self.parameter,)


class _FakeAutoModel:
    calls: list[tuple[str, dict[str, object]]] = []
    model: _FakeModel

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: object) -> _FakeModel:
        cls.calls.append((path, kwargs))
        return cls.model


class _FakeAutoProcessor:
    calls: list[tuple[str, dict[str, object]]] = []
    processor = types.SimpleNamespace(
        image_processor=types.SimpleNamespace(merge_size=2),
        tokenizer=types.SimpleNamespace(),
    )

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: object) -> object:
        cls.calls.append((path, kwargs))
        return cls.processor


def _identity() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_dir="/verified/model",
        model_repo="repo/model",
        model_revision="revision",
        snapshot_manifest_sha256="a" * 64,
        verified_model_file_count=3,
        verified_model_total_bytes=123,
        transformers_version="5.6.0",
        transformers_source_sha256={"modeling.py": "b" * 64},
    )


class GUIOwlV22EagerRuntimeContractTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeAutoModel.calls = []
        _FakeAutoProcessor.calls = []

    def test_attention_validation_requires_top_text_and_vision_eager(self) -> None:
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                _attn_implementation="eager",
                text_config=types.SimpleNamespace(_attn_implementation="eager"),
                vision_config=types.SimpleNamespace(_attn_implementation="eager"),
            )
        )
        observed = gui_owl_v2_2_observed_attention(model)
        self.assertEqual(
            observed,
            {"top": "eager", "text": "eager", "vision": "eager"},
        )
        validate_gui_owl_v2_2_eager_attention(observed)
        for invalid in (
            {"top": "eager", "text": "eager"},
            {"top": "eager", "text": "eager", "vision": None},
            {"top": "eager", "text": "sdpa", "vision": "eager"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                validate_gui_owl_v2_2_eager_attention(invalid)

    def test_scientific_environment_must_be_absent(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            result = audit_gui_owl_v2_2_scientific_environment()
        self.assertIs(result["all_absent"], True)
        self.assertEqual(result["present_names"], [])
        self.assertEqual(
            result["audited_names"],
            list(AUDITED_SCIENTIFIC_ENVIRONMENT_VARIABLES),
        )
        with mock.patch.dict(
            os.environ,
            {"NVIDIA_TF32_OVERRIDE": "0"},
            clear=True,
        ), self.assertRaisesRegex(RuntimeError, "NVIDIA_TF32_OVERRIDE"):
            audit_gui_owl_v2_2_scientific_environment()

    def test_numerical_controls_are_fixed_but_not_strict_determinism(self) -> None:
        torch = _FakeTorch()
        observed = _configure_and_validate_numerical_controls(torch)
        self.assertEqual(torch.deterministic_calls, [(False, False)])
        self.assertEqual(torch.seed_calls, [0])
        self.assertEqual(torch.cuda.seed_calls, [0])
        self.assertEqual(
            observed,
            {
                "deterministic_algorithms_enabled": False,
                "deterministic_warn_only_enabled": False,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest",
            },
        )

    def test_constructor_loads_exact_profile_and_records_metadata(self) -> None:
        torch = _FakeTorch()
        _FakeAutoModel.model = _FakeModel(torch)
        transformers = types.ModuleType("transformers")
        transformers.__version__ = "5.6.0"
        transformers.AutoModelForImageTextToText = _FakeAutoModel
        transformers.AutoProcessor = _FakeAutoProcessor
        identity = _identity()
        chat_identity = types.SimpleNamespace(
            file_sha256="c" * 64,
            text_sha256="d" * 64,
        )
        generation_tokens = types.SimpleNamespace(
            assistant_prefix_token_ids=(1, 2, 3),
            tool_call_open_token_id=4,
            tool_call_close_token_id=5,
            standard_eos_token_ids=(6, 7),
            pad_token_id=7,
        )
        module = "causalcache.policy.gui_owl_v2_2_eager_runtime"
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.dict(
                sys.modules,
                {"torch": torch, "transformers": transformers},
            ),
            mock.patch(f"{module}.platform.python_version", return_value="3.12.3"),
            mock.patch(f"{module}.verify_frozen_vision_runtime", return_value=identity),
            mock.patch(f"{module}._validate_model_identity") as validate_identity,
            mock.patch(
                f"{module}.validate_gui_owl_v2_1_chat_template",
                return_value=chat_identity,
            ),
            mock.patch(
                f"{module}.validate_gui_owl_v2_1_generation_tokens",
                return_value=generation_tokens,
            ),
        ):
            runtime = GUIOwlV22EagerRuntime(
                model_dir="/model",
                expected_snapshot_manifest="/manifest",
                device="cuda:1",
            )

        validate_identity.assert_called_once_with(_FakeAutoModel.model, identity)
        self.assertEqual(_FakeAutoProcessor.calls[0][0], identity.model_dir)
        _, load_kwargs = _FakeAutoModel.calls[0]
        self.assertIs(load_kwargs["dtype"], torch.bfloat16)
        self.assertEqual(
            load_kwargs["attn_implementation"],
            GUI_OWL_V2_2_EAGER_ATTENTION_IMPLEMENTATION,
        )
        self.assertIs(load_kwargs["low_cpu_mem_usage"], True)
        self.assertIs(load_kwargs["local_files_only"], True)
        self.assertEqual(
            runtime.metadata["runtime_profile_id"],
            GUI_OWL_V2_2_EAGER_RUNTIME_ID,
        )
        self.assertEqual(runtime.metadata["seed"], GUI_OWL_V2_2_EAGER_SEED)
        self.assertEqual(runtime.metadata["dtype"], "torch.bfloat16")
        self.assertEqual(runtime.metadata["gpu_name"], "NVIDIA H200")
        for key, expected in GUI_OWL_V2_2_EAGER_EXPECTED_SOFTWARE_STACK.items():
            self.assertEqual(runtime.metadata[key], expected)
        self.assertEqual(
            runtime.metadata["observed_attention_implementation"],
            {"top": "eager", "text": "eager", "vision": "eager"},
        )
        self.assertIs(runtime.metadata["deterministic_algorithms_enabled"], False)
        self.assertIs(runtime.metadata["cudnn_deterministic"], True)
        self.assertIs(runtime.metadata["cudnn_benchmark"], False)
        self.assertIs(runtime.metadata["cuda_matmul_allow_tf32"], False)
        self.assertIs(runtime.metadata["cudnn_allow_tf32"], False)
        self.assertEqual(runtime.metadata["float32_matmul_precision"], "highest")
        self.assertIs(runtime.metadata["strict_cuda_determinism_claimed"], False)
        self.assertEqual(
            runtime.metadata["numerical_control_claim"],
            GUI_OWL_V2_2_EAGER_NUMERICAL_CONTROL_CLAIM,
        )
        self.assertIs(
            runtime.metadata["scientific_environment_audit"]["all_absent"],
            True,
        )
        self.assertEqual(
            runtime.metadata["protocol_id"],
            "causalcache_restoration_v2_1_official_tool_interface",
        )

    def test_constructor_rejects_non_eager_nested_config(self) -> None:
        torch = _FakeTorch()
        _FakeAutoModel.model = _FakeModel(torch, vision_attention="sdpa")
        transformers = types.ModuleType("transformers")
        transformers.__version__ = "5.6.0"
        transformers.AutoModelForImageTextToText = _FakeAutoModel
        transformers.AutoProcessor = _FakeAutoProcessor
        module = "causalcache.policy.gui_owl_v2_2_eager_runtime"
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.dict(
                sys.modules,
                {"torch": torch, "transformers": transformers},
            ),
            mock.patch(f"{module}.platform.python_version", return_value="3.12.3"),
            mock.patch(
                f"{module}.verify_frozen_vision_runtime",
                return_value=_identity(),
            ),
            mock.patch(f"{module}._validate_model_identity"),
            self.assertRaisesRegex(RuntimeError, "top/text/vision"),
        ):
            GUIOwlV22EagerRuntime(
                model_dir="/model",
                expected_snapshot_manifest="/manifest",
                device="cuda:0",
            )

    def test_constructor_rejects_a_different_software_stack(self) -> None:
        torch = _FakeTorch()
        _FakeAutoModel.model = _FakeModel(torch)
        transformers = types.ModuleType("transformers")
        transformers.__version__ = "5.7.0"
        transformers.AutoModelForImageTextToText = _FakeAutoModel
        transformers.AutoProcessor = _FakeAutoProcessor
        module = "causalcache.policy.gui_owl_v2_2_eager_runtime"
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.dict(
                sys.modules,
                {"torch": torch, "transformers": transformers},
            ),
            mock.patch(f"{module}.platform.python_version", return_value="3.12.3"),
            self.assertRaisesRegex(RuntimeError, "software stack"),
        ):
            GUIOwlV22EagerRuntime(
                model_dir="/model",
                expected_snapshot_manifest="/manifest",
                device="cuda:0",
            )
        self.assertEqual(_FakeAutoModel.calls, [])

    def test_policy_interface_is_inherited_except_device_provenance_wrapper(self) -> None:
        self.assertEqual(
            GUIOwlV22EagerRuntime.__mro__[:3],
            (
                GUIOwlV22EagerRuntime,
                GUIOwlV21OfficialToolsRuntime,
                GUIOwlV21OfficialToolsRuntime.__mro__[1],
            ),
        )
        for name in (
            "_interface_metadata",
            "_encode_exact_batch",
            "prepare_native_message_shape",
            "teacher_forced_distance_logits",
            "maximum_context_tokens",
        ):
            with self.subTest(name=name):
                self.assertNotIn(name, GUIOwlV22EagerRuntime.__dict__)
                self.assertIs(
                    getattr(GUIOwlV22EagerRuntime, name),
                    getattr(GUIOwlV21OfficialToolsRuntime, name),
                )

        runtime = object.__new__(GUIOwlV22EagerRuntime)
        runtime.device = _FakeDevice("cuda:1")
        inherited = GUIOwlV21GenerationResult(
            output_text="<tool_call>{}</tool_call>",
            parsed_output=object(),
            metadata={"generated_tokens": 3},
        )
        with mock.patch.object(
            GUIOwlV21OfficialToolsRuntime,
            "generate_native_action",
            return_value=inherited,
        ) as generate:
            result = runtime.generate_native_action(({"role": "user"},))
        generate.assert_called_once_with(({"role": "user"},))
        self.assertEqual(result.output_text, inherited.output_text)
        self.assertIs(result.parsed_output, inherited.parsed_output)
        self.assertEqual(result.metadata, {"generated_tokens": 3, "device": "cuda:1"})

    def test_parse_failure_metadata_is_bound_to_worker_device(self) -> None:
        runtime = object.__new__(GUIOwlV22EagerRuntime)
        runtime.device = _FakeDevice("cuda:0")
        failure = GUIOwlV21GenerationParseError(
            output_text="invalid",
            metadata={"generated_tokens": 2},
            parse_error=ValueError("synthetic parse failure"),
        )
        with mock.patch.object(
            GUIOwlV21OfficialToolsRuntime,
            "generate_native_action",
            side_effect=failure,
        ), self.assertRaises(GUIOwlV21GenerationParseError) as captured:
            runtime.generate_native_action(({"role": "user"},))
        self.assertIs(captured.exception, failure)
        self.assertEqual(
            captured.exception.metadata,
            {"generated_tokens": 2, "device": "cuda:0"},
        )


if __name__ == "__main__":
    unittest.main()
