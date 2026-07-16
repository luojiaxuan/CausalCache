from __future__ import annotations

import json
import sys
import types
import unittest
from unittest import mock

import causalcache.policy.gui_owl_v2_2_vision_runtime as runtime_module
from causalcache.policy.gui_owl_v2_2_vision_runtime import (
    GPU_UUID_TYPE_PROFILE_V1,
    GPU_UUID_TYPE_PROFILE_V2,
    GUI_OWL_V2_2_VISION_RUNTIME_ID,
    GUI_OWL_V2_2_VISION_RUNTIME_UUID_TYPE_ONLY_V2_ID,
    GUIOwlV22VisionFeatureRuntime,
    _canonical_gpu_uuid,
    _nvidia_smi_gpu_identity,
    _validate_cpu_image_processor_output,
)
from causalcache.restoration_v2_baselines import BaselineSelection


class _FakeDType:
    pass


class _CUuuid:
    __module__ = "torch._C"

    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class _FakeDevice:
    def __init__(self, value: str) -> None:
        self.value = value
        self.type = value.split(":", 1)[0]
        self.index = int(value.split(":", 1)[1]) if ":" in value else None

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeDevice) and self.value == other.value


class _FakeTensor:
    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: object,
        device: _FakeDevice | None = None,
        data: object | None = None,
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device or _FakeDevice("cpu")
        self.data = data
        self.contiguous = contiguous
        self.to_calls: list[tuple[object, bool]] = []

    def is_contiguous(self) -> bool:
        return self.contiguous

    def detach(self) -> _FakeTensor:
        return self

    def to(
        self,
        device: object,
        non_blocking: bool = False,
    ) -> _FakeTensor:
        target = _FakeDevice(device) if isinstance(device, str) else device
        self.to_calls.append((target, non_blocking))
        return _FakeTensor(
            shape=self.shape,
            dtype=self.dtype,
            device=target,
            data=self.data,
            contiguous=self.contiguous,
        )

    def tolist(self) -> object:
        return self.data


class _FakeCuda:
    uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 2

    def get_device_name(self, _: _FakeDevice) -> str:
        return "NVIDIA H200"

    def get_device_properties(self, _: _FakeDevice) -> object:
        return types.SimpleNamespace(uuid=self.uuid)


class _FakeTorch(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
        self.Tensor = _FakeTensor
        self.bfloat16 = _FakeDType()
        self.float32 = _FakeDType()
        self.int64 = _FakeDType()
        self.cuda = _FakeCuda()
        self._C = types.SimpleNamespace(_CUuuid=_CUuuid)

    def device(self, value: str) -> _FakeDevice:
        return _FakeDevice(value)


class _FakeParameter:
    def __init__(self, *, device: _FakeDevice, dtype: object) -> None:
        self.device = device
        self.dtype = dtype
        self.requires_grad = False

    def is_floating_point(self) -> bool:
        return True


class _HookModule:
    def __init__(self) -> None:
        self.hooks: list[object] = []

    def register_forward_pre_hook(self, hook: object) -> object:
        self.hooks.append(hook)
        return types.SimpleNamespace(remove=lambda: None)

    def forward(self) -> None:
        return None

    def __call__(self) -> None:
        for hook in self.hooks:
            hook(self, ())


class Qwen3VLForConditionalGeneration(_HookModule):
    def __init__(self, torch: _FakeTorch) -> None:
        super().__init__()
        self.torch = torch
        self.device: _FakeDevice | None = None
        self.training = True
        self.model = types.SimpleNamespace(language_model=_HookModule())
        self.lm_head = _HookModule()

    def to(self, device: _FakeDevice) -> Qwen3VLForConditionalGeneration:
        self.device = device
        return self

    def eval(self) -> Qwen3VLForConditionalGeneration:
        self.training = False
        return self

    def requires_grad_(self, value: bool) -> Qwen3VLForConditionalGeneration:
        if value:
            raise AssertionError("fake model cannot enable gradients")
        return self

    def parameters(self) -> tuple[_FakeParameter, ...]:
        if self.device is None:
            raise AssertionError("fake model must move before parameter inspection")
        return (_FakeParameter(device=self.device, dtype=self.torch.bfloat16),)

    def generate(self) -> None:
        raise AssertionError("constructor must replace the fake generation method")

    def get_image_features(self) -> None:
        return None


class Qwen2VLImageProcessor:
    merge_size = 2
    size = {"shortest_edge": 2_621_440, "longest_edge": 2_621_440}

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.output: dict[str, object] = {}

    def __call__(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return self.output


class _FakeAutoImageProcessor:
    calls: list[tuple[str, dict[str, object]]] = []
    processor = Qwen2VLImageProcessor()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: object) -> Qwen2VLImageProcessor:
        cls.calls.append((path, kwargs))
        return cls.processor


class _FakeAutoModel:
    calls: list[tuple[str, dict[str, object]]] = []
    model: Qwen3VLForConditionalGeneration

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        **kwargs: object,
    ) -> Qwen3VLForConditionalGeneration:
        cls.calls.append((path, kwargs))
        return cls.model


class _FakePILImage:
    def __init__(self, mode: str = "RGB") -> None:
        self.mode = mode


def _identity() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_dir="/verified/model",
        model_repo="mPLUG/GUI-Owl-1.5-8B-Instruct",
        model_revision="revision",
        snapshot_manifest_sha256="a" * 64,
        verified_model_file_count=14,
        verified_model_total_bytes=123,
        transformers_version="5.6.0",
        transformers_source_sha256=(("modeling.py", "b" * 64),),
    )


class GUIOwlV22VisionFeatureRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.torch = _FakeTorch()
        _FakeAutoImageProcessor.calls = []
        _FakeAutoImageProcessor.processor = Qwen2VLImageProcessor()
        _FakeAutoModel.calls = []
        _FakeAutoModel.model = Qwen3VLForConditionalGeneration(self.torch)

    def _construct(
        self,
        *,
        pillow_version: str = "12.2.0",
        gpu_uuid_type_profile: str = GPU_UUID_TYPE_PROFILE_V1,
    ) -> GUIOwlV22VisionFeatureRuntime:
        transformers = types.ModuleType("transformers")
        transformers.AutoImageProcessor = _FakeAutoImageProcessor
        transformers.AutoModelForImageTextToText = _FakeAutoModel
        with (
            mock.patch.dict(
                sys.modules,
                {"torch": self.torch, "transformers": transformers},
            ),
            mock.patch.object(
                runtime_module,
                "audit_gui_owl_v2_2_scientific_environment",
                return_value={"audited_names": [], "present_names": [], "all_absent": True},
            ),
            mock.patch.object(
                runtime_module,
                "_validate_software_stack",
                return_value={
                    "python_version": "3.12.3",
                    "torch_version": "2.11.0+cu130",
                    "torch_cuda_version": "13.0",
                    "cudnn_version": 91900,
                    "transformers_version": "5.6.0",
                },
            ),
            mock.patch.object(
                runtime_module,
                "_configure_and_validate_numerical_controls",
                return_value={
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                    "float32_matmul_precision": "highest",
                },
            ),
            mock.patch.object(
                runtime_module,
                "verify_frozen_vision_runtime",
                return_value=_identity(),
            ),
            mock.patch.object(
                runtime_module.importlib.metadata,
                "version",
                return_value=pillow_version,
            ),
            mock.patch.object(
                runtime_module,
                "_nvidia_smi_gpu_identity",
                return_value={
                    "gpu_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                    "gpu_pci_bus_id": "00000000:41:00.0",
                    "nvidia_smi_index": 5,
                },
            ),
            mock.patch.object(runtime_module, "_validate_model_identity"),
            mock.patch.object(
                runtime_module,
                "gui_owl_v2_2_observed_attention",
                return_value={"top": "eager", "text": "eager", "vision": "eager"},
            ),
            mock.patch.object(runtime_module, "validate_gui_owl_v2_2_eager_attention"),
        ):
            return GUIOwlV22VisionFeatureRuntime(
                model_dir="/model",
                expected_snapshot_manifest="/manifest",
                device="cuda:1",
                expected_gpu_uuid="GPU-AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE",
                gpu_uuid_type_profile=gpu_uuid_type_profile,
            )

    @staticmethod
    def _pil_modules() -> dict[str, types.ModuleType]:
        image_module = types.ModuleType("PIL.Image")
        image_module.Image = _FakePILImage
        pil_module = types.ModuleType("PIL")
        pil_module.Image = image_module
        return {"PIL": pil_module, "PIL.Image": image_module}

    def _processor_tensors(self) -> tuple[_FakeTensor, _FakeTensor]:
        pixels = _FakeTensor(
            shape=(20, 1536),
            dtype=self.torch.float32,
        )
        grid = _FakeTensor(
            shape=(5, 3),
            dtype=self.torch.int64,
            data=[[1, 2, 2]] * 5,
        )
        return pixels, grid

    def test_constructor_uses_direct_image_processor_and_exact_eager_profile(self) -> None:
        runtime = self._construct()
        self.assertEqual(len(_FakeAutoImageProcessor.calls), 1)
        processor_path, processor_kwargs = _FakeAutoImageProcessor.calls[0]
        self.assertEqual(processor_path, "/verified/model")
        self.assertEqual(
            processor_kwargs,
            {
                "min_pixels": 2_621_440,
                "max_pixels": 2_621_440,
                "local_files_only": True,
            },
        )
        model_path, model_kwargs = _FakeAutoModel.calls[0]
        self.assertEqual(model_path, "/verified/model")
        self.assertIs(model_kwargs["dtype"], self.torch.bfloat16)
        self.assertEqual(model_kwargs["attn_implementation"], "eager")
        self.assertIs(model_kwargs["low_cpu_mem_usage"], True)
        self.assertIs(model_kwargs["local_files_only"], True)
        self.assertEqual(
            runtime.metadata["runtime_profile_id"],
            GUI_OWL_V2_2_VISION_RUNTIME_ID,
        )
        self.assertEqual(
            runtime.metadata["processor_interface"],
            "AutoImageProcessor.__call__",
        )
        self.assertIs(runtime.metadata["tokenizer_loaded"], False)
        self.assertIs(runtime.metadata["chat_template_called"], False)
        self.assertIs(runtime.metadata["cuda_matmul_allow_tf32"], False)
        self.assertIs(runtime.metadata["cudnn_allow_tf32"], False)
        self.assertEqual(runtime.metadata["pillow_version"], "12.2.0")
        self.assertEqual(
            runtime.metadata["image_processor_class"],
            "Qwen2VLImageProcessor",
        )
        self.assertEqual(
            runtime.metadata["processor_size"],
            {"shortest_edge": 2_621_440, "longest_edge": 2_621_440},
        )
        self.assertEqual(
            runtime.metadata["gpu_uuid"],
            "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        )
        self.assertEqual(runtime.metadata["gpu_pci_bus_id"], "00000000:41:00.0")
        self.assertEqual(runtime.metadata["logical_device_index"], 1)
        self.assertEqual(runtime.metadata["nvidia_smi_index"], 5)
        self.assertNotIn("model_dir", runtime.metadata)
        self.assertFalse(_FakeAutoModel.model.training)
        self.assertEqual(
            runtime.operation_counts,
            {
                "image_processor_batch_count": 0,
                "policy_vision_feature_forward_count": 0,
                "top_model_forward_count": 0,
                "language_model_forward_count": 0,
                "lm_head_forward_count": 0,
                "generation_count": 0,
            },
        )

    def test_gpu_identity_normalization_and_nvidia_smi_binding(self) -> None:
        self.assertEqual(
            _canonical_gpu_uuid("AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"),
            "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        )
        completed = types.SimpleNamespace(
            stdout=(
                "3, GPU-11111111-2222-3333-4444-555555555555, 00000000:21:00.0\n"
                "5, GPU-AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE, 00000000:41:00.0\n"
            )
        )
        with mock.patch.object(
            runtime_module.subprocess,
            "run",
            return_value=completed,
        ) as run:
            identity = _nvidia_smi_gpu_identity(
                "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
            )
        self.assertEqual(
            identity,
            {
                "gpu_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "gpu_pci_bus_id": "00000000:41:00.0",
                "nvidia_smi_index": 5,
            },
        )
        self.assertEqual(run.call_args.kwargs["timeout"], 10)

    def test_uuid_type_profiles_are_exact_and_default_remains_v1(self) -> None:
        expected = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        self.assertEqual(
            _canonical_gpu_uuid(expected.upper()),
            expected,
        )
        self.assertEqual(
            _canonical_gpu_uuid(expected.encode("ascii")),
            expected,
        )
        with self.assertRaisesRegex(ValueError, "v1 GPU UUID"):
            _canonical_gpu_uuid(_CUuuid(expected))

        self.assertEqual(
            _canonical_gpu_uuid(
                _CUuuid(expected.upper()),
                gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2,
                loaded_torch_c_cuuuid_type=_CUuuid,
            ),
            expected,
        )
        for value in (expected, expected.encode("ascii"), object()):
            with self.subTest(value_type=type(value)), self.assertRaises(ValueError):
                _canonical_gpu_uuid(
                    value,
                    gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2,
                    loaded_torch_c_cuuuid_type=_CUuuid,
                )
        with self.assertRaisesRegex(ValueError, "UUID format"):
            _canonical_gpu_uuid(
                _CUuuid("not-a-gpu-uuid"),
                gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2,
                loaded_torch_c_cuuuid_type=_CUuuid,
            )

        self.torch.cuda.uuid = _CUuuid(expected.upper())
        runtime = self._construct(gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2)
        self.assertEqual(runtime.metadata["gpu_uuid"], expected)
        self.assertEqual(
            runtime.metadata["runtime_profile_id"],
            GUI_OWL_V2_2_VISION_RUNTIME_UUID_TYPE_ONLY_V2_ID,
        )

    def test_uuid_v2_rejects_same_shape_spoof_and_wrong_loaded_type(self) -> None:
        expected = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        spoof_type = type(
            "_CUuuid",
            (),
            {
                "__module__": "torch._C",
                "__str__": lambda _: expected,
            },
        )
        with self.assertRaisesRegex(ValueError, "exact loaded torch"):
            _canonical_gpu_uuid(
                spoof_type(),
                gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2,
                loaded_torch_c_cuuuid_type=_CUuuid,
            )
        with self.assertRaisesRegex(ValueError, "requires the loaded torch"):
            _canonical_gpu_uuid(
                _CUuuid(expected),
                gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2,
            )
        self.torch.cuda.uuid = spoof_type()
        with self.assertRaisesRegex(ValueError, "exact loaded torch"):
            self._construct(gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2)

        self.setUp()
        self.torch.cuda.uuid = _CUuuid(expected)
        self.torch._C = types.SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError, "type is unavailable"):
            self._construct(gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2)

        self.setUp()
        self.torch.cuda.uuid = expected
        with self.assertRaisesRegex(ValueError, "exact loaded torch"):
            self._construct(gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2)

    def test_constructor_rejects_pillow_gpu_or_image_processor_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Pillow version"):
            self._construct(pillow_version="12.1.0")

        self.setUp()
        self.torch.cuda.uuid = "GPU-11111111-2222-3333-4444-555555555555"
        with self.assertRaisesRegex(RuntimeError, "UUID"):
            self._construct()

        self.setUp()

        class WrongImageProcessor(Qwen2VLImageProcessor):
            pass

        _FakeAutoImageProcessor.processor = WrongImageProcessor()
        with self.assertRaisesRegex(RuntimeError, "processor class"):
            self._construct()

        self.setUp()
        _FakeAutoImageProcessor.processor.size = {
            "shortest_edge": 1,
            "longest_edge": 2,
        }
        with self.assertRaisesRegex(ValueError, "processor size"):
            self._construct()

    def test_score_preprocesses_once_and_runs_two_feature_repeats(self) -> None:
        runtime = self._construct()
        pixels, grid = self._processor_tensors()
        runtime.image_processor.output = {
            "pixel_values": pixels,
            "image_grid_thw": grid,
        }
        images = [_FakePILImage() for _ in range(5)]
        batches = [
            types.SimpleNamespace(normalized_embeddings=object()),
            types.SimpleNamespace(normalized_embeddings=object()),
        ]
        selections = [
            BaselineSelection(
                scores_by_event_step=((1, 0.4), (2, 0.3), (3, 0.2), (4, 0.1)),
                ranked_event_step_ids=(1, 2, 3, 4),
                selected_event_step_ids=(1, 2),
            ),
            BaselineSelection(
                scores_by_event_step=((1, 0.400001), (2, 0.3), (3, 0.2), (4, 0.1)),
                ranked_event_step_ids=(1, 2, 3, 4),
                selected_event_step_ids=(1, 2),
            ),
        ]
        with (
            mock.patch.dict(sys.modules, self._pil_modules()),
            mock.patch.object(
                runtime_module,
                "extract_normalized_spatial_merger_embeddings",
                side_effect=batches,
            ) as extract,
            mock.patch.object(
                runtime_module,
                "frozen_policy_vision_similarity_from_batch",
                side_effect=selections,
            ) as similarity,
            mock.patch.object(
                runtime_module,
                "_normalized_norm_range",
                side_effect=((0.99999, 1.00001), (0.99998, 1.00002)),
            ),
        ):
            result = runtime.score_five_images(images, feature_repeats=2)

        self.assertEqual(len(runtime.image_processor.calls), 1)
        self.assertEqual(
            runtime.image_processor.calls[0],
            {"images": images, "return_tensors": "pt"},
        )
        self.assertEqual(extract.call_count, 2)
        self.assertEqual(similarity.call_count, 2)
        for call in extract.call_args_list:
            self.assertEqual(call.kwargs["pixel_values"].device, _FakeDevice("cuda:1"))
            self.assertEqual(call.kwargs["image_grid_thw"].device, _FakeDevice("cuda:1"))
        self.assertEqual(result["feature_repeats"], 2)
        self.assertEqual(len(result["repeat_results"]), 2)
        self.assertEqual(result["selected_event_step_ids"], [1, 2])
        self.assertTrue(result["same_device_replay"]["performed"])
        self.assertTrue(result["same_device_replay"]["ranking_equal"])
        self.assertTrue(result["same_device_replay"]["selection_equal"])
        self.assertAlmostEqual(
            result["same_device_replay"]["max_abs_score_difference"],
            0.000001,
        )
        self.assertEqual(
            result["operation_counts"],
            {
                "image_processor_batch_count": 1,
                "policy_vision_feature_forward_count": 2,
                "top_model_forward_count": 0,
                "language_model_forward_count": 0,
                "lm_head_forward_count": 0,
                "generation_count": 0,
            },
        )
        self.assertEqual(runtime.operation_counts["image_processor_batch_count"], 1)
        self.assertEqual(
            runtime.operation_counts["policy_vision_feature_forward_count"], 2
        )
        json.dumps(result, allow_nan=False, sort_keys=True)

    def test_requires_exactly_five_real_rgb_pil_images(self) -> None:
        runtime = self._construct()
        cases = (
            [_FakePILImage()] * 4,
            [_FakePILImage()] * 6,
            [_FakePILImage()] * 4 + [_FakePILImage("L")],
            [_FakePILImage()] * 4 + [object()],
        )
        with mock.patch.dict(sys.modules, self._pil_modules()):
            for images in cases:
                with self.subTest(length=len(images)), self.assertRaises(
                    (TypeError, ValueError)
                ):
                    runtime.score_five_images(images)
        self.assertEqual(runtime.image_processor.calls, [])
        for value in (0, 3, True):
            with self.subTest(feature_repeats=value), self.assertRaises(ValueError):
                runtime.score_five_images(
                    [_FakePILImage()] * 5,
                    feature_repeats=value,
                )

    def test_cpu_processor_output_is_exact_contiguous_and_shape_bound(self) -> None:
        pixels, grid = self._processor_tensors()
        _validate_cpu_image_processor_output(
            {"pixel_values": pixels, "image_grid_thw": grid},
            torch=self.torch,
        )
        invalid = (
            {
                "pixel_values": pixels,
                "image_grid_thw": grid,
                "input_ids": object(),
            },
            {
                "pixel_values": _FakeTensor(
                    shape=(20, 1536), dtype=self.torch.bfloat16
                ),
                "image_grid_thw": grid,
            },
            {
                "pixel_values": _FakeTensor(
                    shape=(20, 1536),
                    dtype=self.torch.float32,
                    contiguous=False,
                ),
                "image_grid_thw": grid,
            },
            {
                "pixel_values": _FakeTensor(
                    shape=(19, 1536), dtype=self.torch.float32
                ),
                "image_grid_thw": grid,
            },
            {
                "pixel_values": pixels,
                "image_grid_thw": _FakeTensor(
                    shape=(4, 3),
                    dtype=self.torch.int64,
                    data=[[1, 2, 2]] * 4,
                ),
            },
        )
        for encoded in invalid:
            with self.subTest(keys=tuple(encoded)), self.assertRaises(
                (TypeError, ValueError)
            ):
                _validate_cpu_image_processor_output(encoded, torch=self.torch)

    def test_forbidden_forward_and_generation_guards_poison_runtime(self) -> None:
        runtime = self._construct()
        targets = (
            ("top_model_forward_count", runtime.model),
            ("language_model_forward_count", runtime.model.model.language_model),
            ("lm_head_forward_count", runtime.model.lm_head),
            ("generation_count", runtime.model.generate),
        )
        for key, target in targets:
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                target()
            self.assertEqual(runtime.operation_counts[key], 1)
        with self.assertRaisesRegex(RuntimeError, "forbidden operation"):
            runtime.score_five_images([_FakePILImage()] * 5)

    def test_direct_forward_methods_are_guarded_not_only_module_calls(self) -> None:
        cases = (
            ("top_model_forward_count", lambda value: value.model.forward()),
            (
                "language_model_forward_count",
                lambda value: value.model.model.language_model.forward(),
            ),
            ("lm_head_forward_count", lambda value: value.model.lm_head.forward()),
        )
        for key, invoke in cases:
            runtime = self._construct()
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                invoke(runtime)
            self.assertEqual(runtime.operation_counts[key], 1)


if __name__ == "__main__":
    unittest.main()
