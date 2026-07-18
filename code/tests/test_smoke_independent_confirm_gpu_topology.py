from __future__ import annotations

import queue as queue_module
import tempfile
import unittest
from pathlib import Path
from typing import Any

import scripts.smoke_independent_confirm_gpu_topology as smoke


DEVICES = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
GPU_UUIDS = (
    "GPU-00000001-0000-0000-0000-000000000001",
    "GPU-00000002-0000-0000-0000-000000000002",
    "GPU-00000003-0000-0000-0000-000000000003",
    "GPU-00000004-0000-0000-0000-000000000004",
)
IMAGE_SHA = "a" * 64


def _runtime_metadata(*, device: str, gpu_uuid: str | None = None) -> dict[str, Any]:
    metadata = {
        "model_repo": "test/model",
        "model_revision": "f" * 40,
        "snapshot_manifest_sha256": "1" * 64,
        "verified_model_file_count": 3,
        "verified_model_total_bytes": 4096,
        "transformers_version": "5.6.0",
        "transformers_source_sha256": {"model.py": "2" * 64},
        "runtime_profile_id": "test-runtime",
        "dtype": "torch.bfloat16",
        "model_class": "TestModel",
        "requested_attention_implementation": "eager",
        "observed_attention_implementation": {
            "top": "eager",
            "text": "eager",
            "vision": "eager",
        },
        "device": device,
    }
    if gpu_uuid is not None:
        metadata["gpu_uuid"] = gpu_uuid
        metadata["image_processor_class"] = "TestImageProcessor"
    else:
        metadata["processor_class"] = "TestProcessor"
        metadata["protocol_id"] = "test-official-tools"
    return metadata


def _fake_vision_target(queue: Any, **kwargs: Any) -> None:
    queue.put(
        {
            "ok": True,
            "record": {
                "worker_id": kwargs["worker_id"],
                "device": kwargs["device"],
                "gpu_uuid": kwargs["expected_gpu_uuid"],
                "fixed_rgb_image_set_sha256": IMAGE_SHA,
                "runtime_identity": {"runtime_profile_id": "fake-vision"},
                "operation_counts": {
                    "image_processor_batch_count": 1,
                    "policy_vision_feature_forward_count": 1,
                    "generation_count": 0,
                },
            },
        }
    )


def _fake_teacher_target(queue: Any, **kwargs: Any) -> None:
    queue.put(
        {
            "ok": True,
            "record": {
                "worker_id": kwargs["worker_id"],
                "device": kwargs["device"],
                "gpu_uuid": kwargs["expected_gpu_uuid"],
                "fixed_rgb_image_set_sha256": IMAGE_SHA,
                "runtime_identity": {"runtime_profile_id": "fake-teacher"},
                "operation_counts": {
                    "teacher_forced_distance_logits_call_count": 1,
                    "finite_logits_validation_count": 1,
                    "policy_generation_call_count": 0,
                },
            },
        }
    )


class _FakeQueue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, value: Any) -> None:
        self.items.append(value)

    def get(self, timeout: float) -> Any:
        del timeout
        if not self.items:
            raise queue_module.Empty
        return self.items.pop(0)


class _FakeProcess:
    def __init__(
        self,
        *,
        target: Any,
        kwargs: dict[str, Any],
        name: str,
        method: str,
        events: list[str],
    ) -> None:
        self.target = target
        self.kwargs = kwargs
        self.name = name
        self.method = method
        self.events = events
        self.exitcode: int | None = None
        self.alive = False
        self.terminated = False
        self.killed = False

    def start(self) -> None:
        self.events.append(f"{self.method}:start:{self.name}")
        self.alive = True
        try:
            self.target(**self.kwargs)
        except BaseException:
            self.exitcode = 1
            self.alive = False
            raise
        self.exitcode = 0
        self.alive = False

    def join(self, timeout: float) -> None:
        del timeout
        self.events.append(f"{self.method}:join:{self.name}")

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False
        self.exitcode = -15

    def kill(self) -> None:
        self.killed = True
        self.alive = False
        self.exitcode = -9


class _FakeContext:
    def __init__(self, method: str, events: list[str]) -> None:
        self.method = method
        self.events = events
        self.processes: list[_FakeProcess] = []

    def get_start_method(self) -> str:
        return self.method

    def Queue(self) -> _FakeQueue:
        return _FakeQueue()

    def Process(self, **kwargs: Any) -> _FakeProcess:
        process = _FakeProcess(method=self.method, events=self.events, **kwargs)
        self.processes.append(process)
        return process


class _NeverQueue:
    def get(self, timeout: float) -> Any:
        del timeout
        raise queue_module.Empty


class _HungProcess:
    def __init__(self, name: str) -> None:
        self.name = name
        self.exitcode: int | None = None
        self.alive = True
        self.terminated = False
        self.joined = False

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False
        self.exitcode = -15

    def join(self, timeout: float) -> None:
        del timeout
        self.joined = True


class _StepClock:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return 0.0 if self.calls == 1 else 2.0


class _FakeVisionRuntime:
    def __init__(self, **kwargs: Any) -> None:
        self.device = kwargs["device"]
        self.gpu_uuid = kwargs["expected_gpu_uuid"]
        self.metadata = _runtime_metadata(
            device=self.device,
            gpu_uuid=self.gpu_uuid,
        )
        self.image_modes: list[str] = []

    def score_five_images(
        self, images: Any, *, feature_repeats: int
    ) -> dict[str, Any]:
        materialized = tuple(images)
        self.image_modes = [image.mode for image in materialized]
        if len(materialized) != 5 or feature_repeats != 1:
            raise AssertionError("vision smoke input drifted")
        return {
            "feature_repeats": 1,
            "operation_counts": {
                "image_processor_batch_count": 1,
                "policy_vision_feature_forward_count": 1,
                "top_model_forward_count": 0,
                "language_model_forward_count": 0,
                "lm_head_forward_count": 0,
                "generation_count": 0,
            },
        }


class _FakeFinite:
    def all(self) -> "_FakeFinite":
        return self

    def item(self) -> bool:
        return True


class _FakeCuda:
    def __init__(self) -> None:
        self.synchronized: list[str] = []

    def synchronize(self, device: str) -> None:
        self.synchronized.append(device)


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()

    def isfinite(self, logits: Any) -> _FakeFinite:
        del logits
        return _FakeFinite()


class _FakeLogits:
    def __init__(self, device: str) -> None:
        self.device = device
        self.dtype = "torch.bfloat16"
        self.requires_grad = False
        self.ndim = 3
        self.shape = (1, 9, 128)


class _FakeTeacherRuntime:
    def __init__(self, **kwargs: Any) -> None:
        self.device = kwargs["device"]
        self.torch = _FakeTorch()
        self.metadata = _runtime_metadata(device=self.device)

    def teacher_forced_distance_logits(
        self, messages_batch: Any, actions: Any
    ) -> tuple[_FakeLogits, dict[str, Any]]:
        messages = tuple(messages_batch)
        canonical_actions = tuple(actions)
        if len(messages) != 1 or len(canonical_actions) != 1:
            raise AssertionError("teacher microbatch drifted")
        if canonical_actions[0].action != "wait":
            raise AssertionError("teacher action must remain wait")
        image_blocks = [
            block
            for block in messages[0][1]["content"]
            if block.get("type") == "image"
        ]
        if len(image_blocks) != 5 or any(
            block["image"].mode != "RGB" for block in image_blocks
        ):
            raise AssertionError("teacher images drifted")
        return _FakeLogits(self.device), {
            "device": self.device,
            "dtype": "torch.bfloat16",
            "batch_size": 1,
            "samples": [{"image_count": 5}],
        }


class GPUTopologySmokeTest(unittest.TestCase):
    def _paths(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        model = root / "model"
        model.mkdir()
        manifest = root / "snapshot.json"
        manifest.write_text("{}\n", encoding="utf-8")
        return temporary, model, manifest

    def test_fake_context_runs_spawn_then_fork_and_binds_four_identities(self) -> None:
        temporary, model, manifest = self._paths()
        self.addCleanup(temporary.cleanup)
        events: list[str] = []
        spawn = _FakeContext("spawn", events)
        fork = _FakeContext("fork", events)
        receipt = smoke.run_gpu_topology_smoke(
            model_dir=model,
            snapshot_manifest=manifest,
            devices=DEVICES,
            gpu_uuids=GPU_UUIDS,
            phase_timeout_seconds=30,
            worker_termination_grace_seconds=5,
            spawn_context=spawn,
            fork_context=fork,
            vision_worker_target=_fake_vision_target,
            teacher_worker_target=_fake_teacher_target,
        )
        first_fork_start = next(
            index for index, event in enumerate(events) if event.startswith("fork:start")
        )
        spawn_joins = [
            index for index, event in enumerate(events) if event.startswith("spawn:join")
        ]
        self.assertEqual(len(spawn_joins), 4)
        self.assertLess(max(spawn_joins), first_fork_start)
        self.assertEqual(
            [(item["device"], item["gpu_uuid"]) for item in receipt["allocation"]],
            list(zip(DEVICES, GPU_UUIDS, strict=True)),
        )
        for phase in receipt["phases"].values():
            self.assertEqual(
                [(item["device"], item["gpu_uuid"]) for item in phase["workers"]],
                list(zip(DEVICES, GPU_UUIDS, strict=True)),
            )
        self.assertEqual(
            receipt["phase_order"],
            ["policy_vision_spawn", "teacher_forced_fork"],
        )
        self.assertEqual(
            receipt["controls"],
            {
                "phase_timeout_seconds": 30,
                "worker_termination_grace_seconds": 5,
            },
        )
        self.assertFalse(receipt["scope"]["confirm_data_accessed"])
        self.assertFalse(receipt["scope"]["policy_action_generated"])
        self.assertEqual(
            smoke.canonical_json_bytes(receipt),
            smoke.canonical_json_bytes(
                __import__("json").loads(smoke.canonical_json_bytes(receipt))
            ),
        )

    def test_timeout_terminates_and_joins_all_workers(self) -> None:
        processes = [_HungProcess(f"worker-{index}") for index in range(4)]
        with self.assertRaisesRegex(TimeoutError, "exceeded"):
            smoke._collect_phase(
                processes=processes,
                queue=_NeverQueue(),
                phase_label="hung phase",
                timeout_seconds=1,
                grace_seconds=1,
                clock=_StepClock(),
            )
        self.assertTrue(all(process.terminated for process in processes))
        self.assertTrue(all(process.joined for process in processes))
        self.assertTrue(all(not process.is_alive() for process in processes))

    def test_vision_worker_logic_is_cpu_testable_and_uses_five_fixed_rgb_images(self) -> None:
        record = smoke._vision_worker_logic(
            worker_id="worker-0",
            device=DEVICES[0],
            expected_gpu_uuid=GPU_UUIDS[0],
            model_dir="/model",
            snapshot_manifest="/snapshot.json",
            runtime_factory=_FakeVisionRuntime,
        )
        self.assertEqual(record["device"], DEVICES[0])
        self.assertEqual(record["gpu_uuid"], GPU_UUIDS[0])
        self.assertEqual(
            record["fixed_rgb_image_set_sha256"],
            smoke.FIXED_RGB_IMAGE_SET_SHA256,
        )
        self.assertEqual(
            record["operation_counts"]["policy_vision_feature_forward_count"], 1
        )
        self.assertEqual(record["operation_counts"]["generation_count"], 0)

    def test_teacher_worker_logic_is_cpu_testable_and_only_forces_wait(self) -> None:
        def validate_gpu(**kwargs: Any) -> dict[str, Any]:
            self.assertEqual(kwargs["device"], DEVICES[0])
            self.assertEqual(kwargs["expected_gpu_uuid"], GPU_UUIDS[0])
            return {"gpu_uuid": kwargs["expected_gpu_uuid"]}

        record = smoke._teacher_worker_logic(
            worker_id="worker-0",
            device=DEVICES[0],
            expected_gpu_uuid=GPU_UUIDS[0],
            model_dir="/model",
            snapshot_manifest="/snapshot.json",
            runtime_factory=_FakeTeacherRuntime,
            gpu_identity_validator=validate_gpu,
        )
        self.assertEqual(record["device"], DEVICES[0])
        self.assertEqual(
            record["operation_counts"]["teacher_forced_distance_logits_call_count"],
            1,
        )
        self.assertEqual(
            record["operation_counts"]["finite_logits_validation_count"], 1
        )
        self.assertEqual(record["operation_counts"]["policy_generation_call_count"], 0)

    def test_rejects_non_four_or_duplicate_allocations(self) -> None:
        with self.assertRaisesRegex(ValueError, "four unique"):
            smoke._validate_allocation(DEVICES[:3], GPU_UUIDS[:3])
        with self.assertRaisesRegex(ValueError, "four unique"):
            smoke._validate_allocation(
                (DEVICES[0], DEVICES[0], DEVICES[2], DEVICES[3]), GPU_UUIDS
            )

    def test_script_has_no_confirm_data_or_huggingface_client_dependency(self) -> None:
        source = Path(smoke.__file__).read_text(encoding="utf-8")
        self.assertNotIn("independent_confirm_data", source)
        self.assertNotIn("huggingface_hub", source)
        self.assertNotIn("HuggingFaceConfirmPublisher", source)


if __name__ == "__main__":
    unittest.main()
