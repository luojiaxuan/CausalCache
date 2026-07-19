from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import threading
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "code/scripts/run_set_utility_processor_freeze_v2.py"


def _load_runner() -> ModuleType:
    name = "_causalcache_test_processor_runner_v2"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load the v2 runner fixture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _png(mode: str) -> bytes:
    color = (5, 10, 15, 255) if mode == "RGBA" else (5, 10, 15)
    image = Image.new(mode, (4, 6), color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class _FakeInputIds:
    shape = (1, 123)


class _FakeProcessor:
    def apply_chat_template(self, *_: object, **__: object):
        return {"input_ids": _FakeInputIds()}


class _FakeDescriptor:
    observation_count = 11

    def to_payload(self) -> dict[str, object]:
        return {"worker_index": 0, "observation_count": self.observation_count}


def _set_processor_modeling_modules(
    monkeypatch: pytest.MonkeyPatch,
    *names: str,
) -> None:
    for name in tuple(sys.modules):
        if name.startswith("transformers.models.") and ".modeling_" in name:
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in names:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_processor_replay_decode_matches_v1_transient_rgb_semantics(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    runner = _load_runner()
    payload = _png(mode)
    before = hashlib.sha256(payload).hexdigest()
    captured: dict[str, object] = {}

    def build_messages(
        plan: object,
        *,
        image_bytes_loader: object,
        image_decoder: object,
    ) -> list[dict[str, object]]:
        del plan
        image = image_decoder(image_bytes_loader("fixture"))
        captured.update(
            {
                "mode": image.mode,
                "size": image.size,
                "pixels": image.tobytes(),
            }
        )
        return [{"role": "user", "content": []}]

    monkeypatch.setattr(
        runner._V1,
        "build_set_utility_gui_owl_v2_1_messages",
        build_messages,
    )
    runtime = runner.ProcessorLengthRuntimeV2.__new__(
        runner.ProcessorLengthRuntimeV2
    )
    runtime.processor = _FakeProcessor()
    _set_processor_modeling_modules(
        monkeypatch,
        "transformers.models.auto.modeling_auto",
    )

    length = runtime.length(object(), {"fixture": payload})

    with Image.open(io.BytesIO(payload)) as source:
        source.load()
        expected = source.convert("RGB")
    try:
        assert captured == {
            "mode": "RGB",
            "size": expected.size,
            "pixels": expected.tobytes(),
        }
    finally:
        expected.close()
    assert length == 123
    assert hashlib.sha256(payload).hexdigest() == before


def test_rgb_and_opaque_rgba_have_identical_replay_pixels_and_token_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    observed: list[bytes] = []

    def build_messages(
        plan: object,
        *,
        image_bytes_loader: object,
        image_decoder: object,
    ) -> list[dict[str, object]]:
        del plan
        image = image_decoder(image_bytes_loader("fixture"))
        observed.append(image.tobytes())
        return [{"role": "user", "content": []}]

    monkeypatch.setattr(
        runner._V1,
        "build_set_utility_gui_owl_v2_1_messages",
        build_messages,
    )
    runtime = runner.ProcessorLengthRuntimeV2.__new__(
        runner.ProcessorLengthRuntimeV2
    )
    runtime.processor = _FakeProcessor()
    _set_processor_modeling_modules(
        monkeypatch,
        "transformers.models.auto.modeling_auto",
    )

    lengths = [
        runtime.length(object(), {"fixture": _png(mode)})
        for mode in ("RGB", "RGBA")
    ]

    assert lengths == [123, 123]
    assert len(observed) == 2
    assert observed[0] == observed[1]


def test_v2_output_namespace_cannot_resume_v1_staging() -> None:
    runner = _load_runner()

    assert runner.OUTPUT_BASENAME_PREFIX == (
        "causalcache-set-utility-processor-freeze-v2-image-contract-repair-"
    )
    assert not "causalcache-set-utility-processor-freeze-v1-attempt".startswith(
        runner.OUTPUT_BASENAME_PREFIX
    )
    assert runner.OUTPUT_BASENAME_PREFIX + "a" * 7 == (
        "causalcache-set-utility-processor-freeze-v2-image-contract-repair-aaaaaaa"
    )


def test_bounded_parallel_map_runs_concurrently_and_preserves_order() -> None:
    runner = _load_runner()
    barrier = threading.Barrier(2)
    observed_threads: set[int] = set()
    lock = threading.Lock()

    def transform(value: int) -> int:
        with lock:
            observed_threads.add(threading.get_ident())
        if value < 2:
            barrier.wait(timeout=2)
        return value * 10

    result = list(
        runner._bounded_ordered_parallel_map(
            transform,
            range(6),
            max_workers=2,
        )
    )

    assert result == [0, 10, 20, 30, 40, 50]
    assert len(observed_threads) == 2


def test_bounded_parallel_map_rejects_invalid_concurrency() -> None:
    runner = _load_runner()
    with pytest.raises(ValueError, match="worker count must be positive"):
        list(
            runner._bounded_ordered_parallel_map(
                lambda value: value,
                [1],
                max_workers=0,
            )
        )


def test_processor_post_load_guard_accepts_only_exact_auto_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    allowed = "transformers.models.auto.modeling_auto"
    architecture = "transformers.models.qwen3_vl.modeling_qwen3_vl"

    _set_processor_modeling_modules(monkeypatch, allowed)
    runner.assert_processor_v2_post_load_import_state()

    _set_processor_modeling_modules(monkeypatch)
    with pytest.raises(RuntimeError, match="post-load modeling modules drifted"):
        runner.assert_processor_v2_post_load_import_state()

    _set_processor_modeling_modules(monkeypatch, allowed, architecture)
    with pytest.raises(RuntimeError, match="modeling_qwen3_vl"):
        runner.assert_processor_v2_post_load_import_state()


def test_processor_runtime_pool_uses_independent_auto_processors_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    _set_processor_modeling_modules(monkeypatch)
    created: list[object] = []

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, *_: object, **__: object) -> object:
            processor = object()
            created.append(processor)
            sys.modules["transformers.models.auto.modeling_auto"] = ModuleType(
                "transformers.models.auto.modeling_auto"
            )
            return processor

    transformers = sys.modules.get("transformers")
    if transformers is None:
        transformers = ModuleType("transformers")
        monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        transformers,
        "AutoProcessor",
        FakeAutoProcessor,
        raising=False,
    )

    first = runner.ProcessorLengthRuntimeV2(
        tmp_path,
        require_pristine_import_state=True,
    )
    second = runner.ProcessorLengthRuntimeV2(
        tmp_path,
        require_pristine_import_state=False,
    )

    assert first.processor is created[0]
    assert second.processor is created[1]
    assert first.processor is not second.processor

    _set_processor_modeling_modules(
        monkeypatch,
        "transformers.models.auto.modeling_auto",
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
    )
    with pytest.raises(RuntimeError, match="modeling_qwen3_vl"):
        runner.ProcessorLengthRuntimeV2(
            tmp_path,
            require_pristine_import_state=False,
        )


def test_processor_runtime_map_is_eight_way_ordered_and_slot_isolated() -> None:
    runner = _load_runner()
    runtimes = tuple(object() for _ in range(8))
    barrier = threading.Barrier(8)
    observed_runtime_ids: set[int] = set()
    lock = threading.Lock()

    def transform(runtime: object, value: int) -> int:
        with lock:
            observed_runtime_ids.add(id(runtime))
        if value < 8:
            barrier.wait(timeout=3)
        return value * 10

    result = list(
        runner._bounded_ordered_runtime_map(
            transform,
            range(24),
            runtimes=runtimes,
        )
    )

    assert result == [value * 10 for value in range(24)]
    assert observed_runtime_ids == {id(runtime) for runtime in runtimes}


def test_processor_runtime_pool_binds_eight_slots_and_pristine_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    calls: list[tuple[Path, bool, object]] = []

    def fake_runtime(
        model_dir: Path,
        *,
        require_pristine_import_state: bool,
    ) -> object:
        runtime = object()
        calls.append((model_dir, require_pristine_import_state, runtime))
        return runtime

    monkeypatch.setattr(runner, "ProcessorLengthRuntimeV2", fake_runtime)
    pool = runner._build_processor_runtime_pool(tmp_path, concurrency=8)

    assert len(pool) == 8
    assert len({id(runtime) for runtime in pool}) == 8
    assert [item[1] for item in calls] == [True, *([False] * 7)]
    with pytest.raises(ValueError, match="frozen contract"):
        runner._build_processor_runtime_pool(tmp_path, concurrency=7)


def test_processor_torch_threads_are_explicitly_set_and_getter_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    for key in runner.PROCESSOR_TORCH_AMBIENT_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    state = {"intraop": 112, "interop": 16}
    calls: list[tuple[str, int]] = []
    torch = ModuleType("torch")

    def set_num_threads(value: int) -> None:
        calls.append(("intraop", value))
        state["intraop"] = value

    def set_num_interop_threads(value: int) -> None:
        calls.append(("interop", value))
        state["interop"] = value

    torch.set_num_threads = set_num_threads
    torch.set_num_interop_threads = set_num_interop_threads
    torch.get_num_threads = lambda: state["intraop"]
    torch.get_num_interop_threads = lambda: state["interop"]
    monkeypatch.setitem(sys.modules, "torch", torch)
    contract = SimpleNamespace(
        data={
            "phases": {
                "auto_processor": {
                    "ambient_thread_environment_allowed": False,
                    "ambient_thread_environment_keys_removed": list(
                        runner.PROCESSOR_TORCH_AMBIENT_ENVIRONMENT_KEYS
                    ),
                    "torch_interop_thread_count": 1,
                    "torch_intraop_thread_count": 28,
                    "torch_thread_configuration_before_auto_processor_required": True,
                    "torch_thread_getter_verification_required": True,
                    "torch_thread_setter": "explicit_runtime_api",
                }
            }
        }
    )

    thread_contract = runner._processor_torch_thread_contract(contract)
    observed = runner._configure_processor_torch_threads(thread_contract)

    assert calls == [("intraop", 28), ("interop", 1)]
    assert observed == {
        "ambient_thread_environment_keys_present": [],
        "getter_verification_passed": True,
        "torch_interop_thread_count": 1,
        "torch_intraop_thread_count": 28,
    }


def test_processor_torch_threads_reject_ambient_or_getter_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    thread_contract = {
        "torch_interop_thread_count": 1,
        "torch_intraop_thread_count": 28,
    }
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    with pytest.raises(ValueError, match="ambient environment"):
        runner._configure_processor_torch_threads(thread_contract)
    monkeypatch.delenv("OMP_NUM_THREADS")
    for key in runner.PROCESSOR_TORCH_AMBIENT_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    torch = ModuleType("torch")
    torch.set_num_threads = lambda value: None
    torch.set_num_interop_threads = lambda value: None
    torch.get_num_threads = lambda: 27
    torch.get_num_interop_threads = lambda: 1
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(RuntimeError, match="getter verification failed"):
        runner._configure_processor_torch_threads(thread_contract)


def test_resumable_ocr_receipt_requires_exact_runtime_and_schema() -> None:
    runner = _load_runner()
    descriptor = _FakeDescriptor()
    identity = {
        "image_contract_sha256": "a" * 64,
        "runtime_packages": {"rapidocr": "3.8.4"},
    }
    receipt = {
        "backend_config_sha256": "b" * 64,
        "format_tally": {"PNG:RGB": 1, "PNG:RGBA": 10},
        "ocr_runtime_identity": identity,
        "shard": descriptor.to_payload(),
        "worker_index": 0,
    }
    runner._validate_resumable_ocr_receipt_v2(
        receipt,
        descriptor=descriptor,
        backend_config_sha256="b" * 64,
        ocr_runtime_identity=identity,
        worker_index=0,
    )

    mutations = []
    extra = deepcopy(receipt)
    extra["unexpected"] = True
    mutations.append(extra)
    wrong_backend = deepcopy(receipt)
    wrong_backend["backend_config_sha256"] = "c" * 64
    mutations.append(wrong_backend)
    wrong_identity = deepcopy(receipt)
    wrong_identity["ocr_runtime_identity"]["image_contract_sha256"] = "d" * 64
    mutations.append(wrong_identity)
    wrong_tally = deepcopy(receipt)
    wrong_tally["format_tally"] = {"PNG:RGBA": 10}
    mutations.append(wrong_tally)

    for mutated in mutations:
        with pytest.raises(ValueError, match="current runtime or shard"):
            runner._validate_resumable_ocr_receipt_v2(
                mutated,
                descriptor=descriptor,
                backend_config_sha256="b" * 64,
                ocr_runtime_identity=identity,
                worker_index=0,
            )
