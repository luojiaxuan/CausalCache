from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import threading
from copy import deepcopy
from pathlib import Path
from types import ModuleType

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
