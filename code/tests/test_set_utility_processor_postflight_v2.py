from __future__ import annotations

import io
import importlib.util
import json
import shutil
import sys
import threading
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

import causalcache.set_utility_processor_postflight_parallel_v1 as parallel_v1
import causalcache.set_utility_processor_postflight_v2 as postflight_v2

from causalcache.restoration_v2_text_backend import (
    build_ocr_record,
    load_backend_config,
    sha256_bytes,
)
from causalcache.set_utility_processor_artifacts import (
    canonical_json_bytes,
    canonical_pretty_json_bytes,
)
from causalcache.set_utility_processor_image_contract_v2 import (
    PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS,
)
from causalcache.set_utility_processor_postflight_v2 import (
    IMAGE_CONTRACT_PATH,
    RUNNER_PATH,
    V1_RUNNER_PATH,
    ProcessorFreezePostflightContextV2,
    VALIDATION_STATUS,
    _validate_global_format_tally,
    _validate_processor_thread_runtime_logs,
    _validate_record_without_image,
    validate_completed_processor_freeze_root_v2,
    validate_processor_only_source_v2,
)
from causalcache.set_utility_processor_postflight_parallel_v1 import (
    PROTOCOL_ID as PARALLEL_PROTOCOL_ID,
    VALIDATION_STATUS as PARALLEL_VALIDATION_STATUS,
    V2ParallelWorkerSemanticAudit,
    validate_completed_processor_freeze_root_parallel_v1,
    validate_v2_artifact_semantics_parallel,
)


ROOT = Path(__file__).resolve().parents[2]
BACKEND_SHA = "a" * 64
BACKEND_CONFIG_PATH = ROOT / "code/configs/restoration_v2_ocr_backend.json"


def _processor_thread_evidence_line(
    *,
    intraop: int = 28,
    interop: int = 1,
) -> bytes:
    return canonical_json_bytes(
        {
            "processor_torch_thread_runtime": {
                "ambient_thread_environment_keys_present": [],
                "getter_verification_passed": True,
                "torch_interop_thread_count": interop,
                "torch_intraop_thread_count": intraop,
            }
        }
    )


def _load_v1_postflight_fixture() -> ModuleType:
    name = "_causalcache_test_processor_postflight_v1_fixture"
    path = ROOT / "code/tests/test_set_utility_processor_postflight.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load the v1 postflight fixture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _png(mode: str) -> bytes:
    image = Image.new(mode, (5, 7), (1, 2, 3, 255) if mode == "RGBA" else (1, 2, 3))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _semantic_test_context() -> SimpleNamespace:
    schedule = _load_v1_postflight_fixture()._worker_schedule()
    return SimpleNamespace(
        structural_context=SimpleNamespace(worker_schedule=schedule),
        backend_config={},
        backend_config_sha256=BACKEND_SHA,
    )


def _worker_semantic_audit(
    worker_index: int,
    *,
    fully_validated_paths: tuple[tuple[str, str], ...] | None = None,
    terminal_paths: tuple[tuple[str, str], ...] | None = None,
    terminal_tally: tuple[tuple[str, int], ...] = (("PNG:RGBA", 1),),
) -> V2ParallelWorkerSemanticAudit:
    identity = (f"trajectory-{worker_index}", "images/observation.png")
    return V2ParallelWorkerSemanticAudit(
        worker_index=worker_index,
        terminal_tally=terminal_tally,
        fully_validated_paths=(
            (identity,)
            if fully_validated_paths is None
            else fully_validated_paths
        ),
        terminal_paths=(identity,) if terminal_paths is None else terminal_paths,
    )


def _patch_four_rgba_expected_tally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        postflight_v2,
        "PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS",
        {"PNG:RGBA": 4},
    )
    monkeypatch.setattr(
        postflight_v2,
        "PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_TOTAL",
        4,
    )


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, int, str], ...]:
    records = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            records.append(("directory", relative, 0, ""))
        else:
            payload = path.read_bytes()
            records.append(("file", relative, len(payload), sha256_bytes(payload)))
    return tuple(records)


def _serial_v2_semantics_reference(
    root: Path,
    context: ProcessorFreezePostflightContextV2,
) -> tuple[dict[str, int], int, int]:
    terminal_tallies = []
    fully_validated_paths: set[tuple[str, str]] = set()
    terminal_paths: set[tuple[str, str]] = set()
    for worker in context.structural_context.worker_schedule.workers:
        worker_tally: Counter[str] = Counter()
        shard = root / "substrate" / worker.filename
        for query in postflight_v2.iter_processor_query_artifact_records(
            shard,
            expected_worker=worker,
        ):
            for path, payload in query.image_payloads.items():
                identity = (query.trajectory_id, path)
                if identity in fully_validated_paths:
                    continue
                record = query.ocr_records_by_path[path]
                postflight_v2.validate_processor_ocr_record_v2(
                    record,
                    image_bytes=payload,
                    backend_config=context.backend_config,
                    backend_config_sha256=context.backend_config_sha256,
                )
                fully_validated_paths.add(identity)
            if query.query_kind != "terminal":
                continue
            for path, record in query.ocr_records_by_path.items():
                terminal_paths.add((query.trajectory_id, path))
                source_format, source_mode = _validate_record_without_image(
                    record,
                    expected_path=path,
                    backend_config_sha256=context.backend_config_sha256,
                )
                worker_tally[f"{source_format}:{source_mode}"] += 1
        terminal_tallies.append(dict(sorted(worker_tally.items())))
    if not fully_validated_paths.issubset(terminal_paths):
        raise ValueError("serial reference coverage drifted")
    return (
        _validate_global_format_tally(terminal_tallies),
        len(fully_validated_paths),
        len(terminal_paths - fully_validated_paths),
    )


def test_v2_semantic_overlay_runs_all_four_worker_tars_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _semantic_test_context()
    barrier = threading.Barrier(4, timeout=5.0)
    lock = threading.Lock()
    thread_ids: set[int] = set()
    calls: list[int] = []

    def inspect(
        root: Path,
        observed_context: object,
        worker: object,
    ) -> V2ParallelWorkerSemanticAudit:
        assert root == tmp_path
        assert observed_context is context
        with lock:
            thread_ids.add(threading.get_ident())
            calls.append(worker.worker_index)
        barrier.wait()
        return _worker_semantic_audit(worker.worker_index)

    monkeypatch.setattr(
        parallel_v1,
        "validate_v2_parallel_worker_artifact_semantics",
        inspect,
    )
    _patch_four_rgba_expected_tally(monkeypatch)

    result = validate_v2_artifact_semantics_parallel(tmp_path, context)

    assert result == ({"PNG:RGBA": 4}, 4, 0)
    assert sorted(calls) == [0, 1, 2, 3]
    assert len(thread_ids) == 4


def test_v2_semantic_overlay_aggregates_in_worker_order_after_reverse_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _semantic_test_context()
    releases = tuple(threading.Event() for _ in range(4))
    lock = threading.Lock()
    completion_order: list[int] = []
    observed_tallies: list[list[tuple[str, int]]] = []

    def inspect(
        root: Path,
        observed_context: object,
        worker: object,
    ) -> V2ParallelWorkerSemanticAudit:
        del root, observed_context
        worker_index = worker.worker_index
        if worker_index < 3:
            assert releases[worker_index + 1].wait(timeout=5.0)
        with lock:
            completion_order.append(worker_index)
        releases[worker_index].set()
        return _worker_semantic_audit(
            worker_index,
            terminal_tally=((f"worker-{worker_index}", worker_index + 1),),
        )

    def aggregate(tallies: object) -> dict[str, int]:
        observed_tallies.append(
            [tuple(tally.items())[0] for tally in tallies]
        )
        return {"ordered-worker-count": 4}

    monkeypatch.setattr(
        parallel_v1,
        "validate_v2_parallel_worker_artifact_semantics",
        inspect,
    )
    monkeypatch.setattr(postflight_v2, "_validate_global_format_tally", aggregate)

    result = validate_v2_artifact_semantics_parallel(tmp_path, context)

    assert completion_order == [3, 2, 1, 0]
    assert observed_tallies == [
        [
            ("worker-0", 1),
            ("worker-1", 2),
            ("worker-2", 3),
            ("worker-3", 4),
        ]
    ]
    assert result == ({"ordered-worker-count": 4}, 4, 0)


def test_v2_semantic_overlay_reports_lowest_worker_index_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _semantic_test_context()
    barrier = threading.Barrier(4, timeout=5.0)

    def inspect(
        root: Path,
        observed_context: object,
        worker: object,
    ) -> V2ParallelWorkerSemanticAudit:
        del root, observed_context
        barrier.wait()
        if worker.worker_index in {1, 3}:
            raise RuntimeError(f"worker-{worker.worker_index}-failure")
        return _worker_semantic_audit(worker.worker_index)

    monkeypatch.setattr(
        parallel_v1,
        "validate_v2_parallel_worker_artifact_semantics",
        inspect,
    )

    with pytest.raises(RuntimeError, match="worker-1-failure"):
        validate_v2_artifact_semantics_parallel(tmp_path, context)


def test_v2_semantic_worker_reads_only_its_exact_bound_tar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _semantic_test_context()
    lock = threading.Lock()
    calls: list[tuple[str, int]] = []

    def records(path: Path, *, expected_worker: object):
        worker_index = expected_worker.worker_index
        with lock:
            calls.append((path.name, worker_index))
        image_path = f"images/worker-{worker_index}.png"
        return iter(
            (
                SimpleNamespace(
                    image_payloads={image_path: b"image"},
                    ocr_records_by_path={image_path: {"worker": worker_index}},
                    query_kind="terminal",
                    trajectory_id=f"trajectory-{worker_index}",
                ),
            )
        )

    monkeypatch.setattr(
        parallel_v1,
        "iter_processor_query_artifact_records",
        records,
    )
    monkeypatch.setattr(
        parallel_v1,
        "validate_processor_ocr_record_v2",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        postflight_v2,
        "_validate_record_without_image",
        lambda *args, **kwargs: ("PNG", "RGBA"),
    )
    _patch_four_rgba_expected_tally(monkeypatch)

    result = validate_v2_artifact_semantics_parallel(tmp_path, context)

    expected = sorted(
        (worker.filename, worker.worker_index)
        for worker in context.structural_context.worker_schedule.workers
    )
    assert sorted(calls) == expected
    assert len(calls) == 4
    assert result == ({"PNG:RGBA": 4}, 4, 0)


def test_v2_semantic_overlay_rejects_cross_worker_path_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _semantic_test_context()
    shared = (("shared-trajectory", "images/shared.png"),)

    def inspect(
        root: Path,
        observed_context: object,
        worker: object,
    ) -> V2ParallelWorkerSemanticAudit:
        del root, observed_context
        if worker.worker_index in {0, 1}:
            return _worker_semantic_audit(
                worker.worker_index,
                fully_validated_paths=shared,
                terminal_paths=shared,
            )
        return _worker_semantic_audit(worker.worker_index)

    monkeypatch.setattr(
        parallel_v1,
        "validate_v2_parallel_worker_artifact_semantics",
        inspect,
    )

    with pytest.raises(ValueError, match="overlap across workers"):
        validate_v2_artifact_semantics_parallel(tmp_path, context)


def test_v2_semantic_overlay_rejects_worker_full_paths_outside_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _semantic_test_context()

    def inspect(
        root: Path,
        observed_context: object,
        worker: object,
    ) -> V2ParallelWorkerSemanticAudit:
        del root, observed_context
        if worker.worker_index == 0:
            return _worker_semantic_audit(
                worker.worker_index,
                fully_validated_paths=(("trajectory-0", "images/stored.png"),),
                terminal_paths=(("trajectory-0", "images/terminal.png"),),
            )
        return _worker_semantic_audit(worker.worker_index)

    monkeypatch.setattr(
        parallel_v1,
        "validate_v2_parallel_worker_artifact_semantics",
        inspect,
    )

    with pytest.raises(ValueError, match="escaped terminal OCR coverage"):
        validate_v2_artifact_semantics_parallel(tmp_path, context)


def test_v2_source_audit_recursively_binds_both_runners() -> None:
    result = validate_processor_only_source_v2(ROOT)

    assert result["status"] == (
        "VALID_PROCESSOR_ONLY_SOURCE_V2_IMAGE_CONTRACT_REPAIR"
    )
    assert set(result["bound_sources"]) == {
        "image_contract_v2",
        "runner_v1",
        "runner_v2",
    }
    assert result["forbidden_image_mutation_call_count"] == 0
    assert result["forbidden_v1_execution_helper_call_count"] == 0
    assert result["transient_rgb_convert_call_count"] == 1
    assert result["processor_concurrency_per_logical_worker"] == 8
    assert result["processor_post_load_modeling_module_allowlist"] == [
        "transformers.models.auto.modeling_auto"
    ]
    assert result["processor_torch_interop_threads"] == 1
    assert result["processor_torch_intraop_threads"] == 28
    assert result["processor_torch_ambient_environment_keys_removed"] == [
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TORCH_NUM_INTEROP_THREADS",
        "TORCH_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    assert result["processor_execution_source_contract"] == {
        "initializer_pre_load_guard_call_count": 1,
        "initializer_post_load_guard_call_count": 2,
        "length_pre_load_guard_call_count": 0,
        "length_post_load_guard_call_count": 1,
        "pool_builder_runtime_call_count": 1,
        "processor_concurrency_per_logical_worker": 8,
        "processor_post_load_modeling_module_allowlist": [
            "transformers.models.auto.modeling_auto"
        ],
        "processor_torch_ambient_environment_keys_removed": [
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "TORCH_NUM_INTEROP_THREADS",
            "TORCH_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ],
        "processor_torch_interop_threads": 1,
        "processor_torch_intraop_threads": 28,
        "runtime_map_bounded_map_call_count": 1,
        "launcher_environment_pop_call_count": 2,
        "thread_get_interop_call_count": 1,
        "thread_get_intraop_call_count": 1,
        "thread_set_interop_call_count": 1,
        "thread_set_intraop_call_count": 1,
        "worker_pool_builder_call_count": 1,
        "worker_runtime_map_call_count": 1,
        "worker_thread_configure_call_count": 1,
    }
    assert result["bound_sources"]["runner_v1"]["path"] == V1_RUNNER_PATH
    assert result["bound_sources"]["runner_v2"]["path"] == RUNNER_PATH
    assert result["bound_sources"]["image_contract_v2"]["path"] == (
        IMAGE_CONTRACT_PATH
    )


def test_v2_source_audit_rejects_direct_runner_image_mutation(
    tmp_path: Path,
) -> None:
    for relative in (RUNNER_PATH, V1_RUNNER_PATH, IMAGE_CONTRACT_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    runner = tmp_path / RUNNER_PATH
    runner.write_text(
        runner.read_text(encoding="utf-8") + "\nimage.save('forbidden.png')\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden image mutation"):
        validate_processor_only_source_v2(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    (
        "\nother.convert('RGB')\n",
        "\nother.convert('RGBA')\n",
        "\nother.putalpha(255)\n",
        "\nImageOps.exif_transpose(other)\n",
    ),
)
def test_v2_source_audit_rejects_extra_or_non_rgb_conversion(
    tmp_path: Path,
    mutation: str,
) -> None:
    for relative in (RUNNER_PATH, V1_RUNNER_PATH, IMAGE_CONTRACT_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    runner = tmp_path / RUNNER_PATH
    runner.write_text(
        runner.read_text(encoding="utf-8") + mutation,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden image mutation"):
        validate_processor_only_source_v2(tmp_path)


def test_v2_source_audit_requires_literal_rgb_at_the_exact_replay_site(
    tmp_path: Path,
) -> None:
    for relative in (RUNNER_PATH, V1_RUNNER_PATH, IMAGE_CONTRACT_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    runner = tmp_path / RUNNER_PATH
    source = runner.read_text(encoding="utf-8")
    assert source.count('source.convert("RGB")') == 1
    runner.write_text(
        source.replace('source.convert("RGB")', 'source.convert("RGBA")'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden image mutation"):
        validate_processor_only_source_v2(tmp_path)


def test_v2_source_audit_rejects_imported_image_mutation_alias(
    tmp_path: Path,
) -> None:
    for relative in (RUNNER_PATH, V1_RUNNER_PATH, IMAGE_CONTRACT_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    runner = tmp_path / RUNNER_PATH
    runner.write_text(
        runner.read_text(encoding="utf-8")
        + "\nfrom PIL.ImageOps import exif_transpose as x\nx(object())\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden image mutation"):
        validate_processor_only_source_v2(tmp_path)


def test_v2_source_audit_rejects_v1_helper_attribute_alias(
    tmp_path: Path,
) -> None:
    for relative in (RUNNER_PATH, V1_RUNNER_PATH, IMAGE_CONTRACT_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    runner = tmp_path / RUNNER_PATH
    runner.write_text(
        runner.read_text(encoding="utf-8") + "\nbad = _V1._run_ocr_worker\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden v1 image execution helper"):
        validate_processor_only_source_v2(tmp_path)


def test_v2_source_audit_rejects_v1_image_execution_helper_reuse(
    tmp_path: Path,
) -> None:
    for relative in (RUNNER_PATH, V1_RUNNER_PATH, IMAGE_CONTRACT_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    runner = tmp_path / RUNNER_PATH
    runner.write_text(
        runner.read_text(encoding="utf-8") + "\n_V1._run_ocr_worker(None, None, None)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden v1 image execution helper"):
        validate_processor_only_source_v2(tmp_path)


def test_processor_thread_runtime_logs_accept_one_first_line_and_text_tail(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    expected = _processor_thread_evidence_line()
    for worker_index in range(4):
        diagnostics = (
            b'warning: downstream diagnostic only\n{"warning":"not evidence"}\n'
            if worker_index == 0
            else b""
        )
        (logs / f"processor-worker-{worker_index:02d}.log").write_bytes(
            expected + b"\n" + diagnostics
        )

    result = _validate_processor_thread_runtime_logs(tmp_path)

    assert len(result) == 4
    assert [item["worker_index"] for item in result] == [0, 1, 2, 3]
    assert result[0]["diagnostic_line_count"] == 2
    assert all(item["status"].startswith("VALID_PROCESSOR") for item in result)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("warning_first", "canonical first line"),
        ("duplicate", "exactly one evidence marker"),
        ("wrong_intraop", "canonical first line"),
        ("non_utf8_tail", "must be UTF-8 text"),
        ("unsafe_tail", "unsafe control characters"),
        ("missing_worker", "regular file"),
    ],
)
def test_processor_thread_runtime_logs_fail_closed(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    expected = _processor_thread_evidence_line()
    for worker_index in range(4):
        (logs / f"processor-worker-{worker_index:02d}.log").write_bytes(
            expected + b"\n"
        )
    target = logs / "processor-worker-02.log"
    if mutation == "warning_first":
        target.write_bytes(b"warning before evidence\n" + expected + b"\n")
    elif mutation == "duplicate":
        target.write_bytes(expected + b"\n" + expected + b"\n")
    elif mutation == "wrong_intraop":
        target.write_bytes(_processor_thread_evidence_line(intraop=27) + b"\n")
    elif mutation == "non_utf8_tail":
        target.write_bytes(expected + b"\n\xff")
    elif mutation == "unsafe_tail":
        target.write_bytes(expected + b"\nwarning\x00")
    elif mutation == "missing_worker":
        target.unlink()
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(ValueError, match=match):
        _validate_processor_thread_runtime_logs(tmp_path)


def test_v2_source_audit_rejects_old_guard_after_processor_load(
    tmp_path: Path,
) -> None:
    for relative in (RUNNER_PATH, V1_RUNNER_PATH, IMAGE_CONTRACT_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    runner = tmp_path / RUNNER_PATH
    source = runner.read_text(encoding="utf-8")
    needle = (
        "            assert_processor_v2_post_load_import_state()\n\n\n"
        "def _build_processor_runtime_pool"
    )
    assert needle in source
    runner.write_text(
        source.replace(
            needle,
            "            assert_processor_only_import_state()\n\n\n"
            "def _build_processor_runtime_pool",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="guard or concurrency source contract"):
        validate_processor_only_source_v2(tmp_path)


def test_v2_source_audit_requires_bounded_ordered_processor_replay(
    tmp_path: Path,
) -> None:
    for relative in (RUNNER_PATH, V1_RUNNER_PATH, IMAGE_CONTRACT_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    runner = tmp_path / RUNNER_PATH
    source = runner.read_text(encoding="utf-8")
    needle = "        _bounded_ordered_runtime_map(\n"
    assert source.count(needle) == 1
    runner.write_text(
        source.replace(needle, "        _unbound_runtime_map(\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="guard or concurrency source contract"):
        validate_processor_only_source_v2(tmp_path)


def test_global_format_tally_requires_exact_census_counts() -> None:
    assert _validate_global_format_tally(
        (
            {"PNG:RGBA": 10_000, "PNG:RGB": 4},
            {"PNG:RGBA": 8_768, "PNG:RGB": 20},
        )
    ) == dict(sorted(PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS.items()))

    with pytest.raises(ValueError, match="frozen v2 census"):
        _validate_global_format_tally(
            ({"PNG:RGBA": 18_768, "PNG:RGB": 23},)
        )
    with pytest.raises(ValueError, match="escaped the v2 contract"):
        _validate_global_format_tally(
            ({"PNG:RGBA": 18_768, "PNG:RGB": 24, "JPEG:RGB": 1},)
        )


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_record_semantics_accept_exact_v2_union(mode: str) -> None:
    payload = _png(mode)
    path = f"images/fixture/{mode.lower()}.png"
    record = build_ocr_record(
        image_member_path=path,
        image_bytes=payload,
        backend_config_sha256=BACKEND_SHA,
        boxes=None,
        texts=None,
        scores=None,
    )

    assert _validate_record_without_image(
        record,
        expected_path=path,
        backend_config_sha256=BACKEND_SHA,
    ) == ("PNG", mode)
    assert record["image_sha256"] == sha256_bytes(payload)


def test_record_semantics_rejects_canonical_sha_tamper() -> None:
    payload = _png("RGB")
    path = "images/fixture/rgb.png"
    record = build_ocr_record(
        image_member_path=path,
        image_bytes=payload,
        backend_config_sha256=BACKEND_SHA,
        boxes=None,
        texts=None,
        scores=None,
    )
    record["canonical_ocr_record_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="canonical SHA256 drifted"):
        _validate_record_without_image(
            record,
            expected_path=path,
            backend_config_sha256=BACKEND_SHA,
        )


def test_record_semantics_rejects_extra_field_even_with_recomputed_hash() -> None:
    payload = _png("RGB")
    path = "images/fixture/rgb.png"
    record = build_ocr_record(
        image_member_path=path,
        image_bytes=payload,
        backend_config_sha256=BACKEND_SHA,
        boxes=None,
        texts=None,
        scores=None,
    )
    record["evil"] = "self-hashed garbage"
    unsigned = dict(record)
    unsigned.pop("canonical_ocr_record_sha256")
    record["canonical_ocr_record_sha256"] = sha256_bytes(
        canonical_json_bytes(unsigned)
    )

    with pytest.raises(ValueError, match="fields drifted"):
        _validate_record_without_image(
            record,
            expected_path=path,
            backend_config_sha256=BACKEND_SHA,
        )


def test_v2_postflight_wraps_v1_structure_and_rebuilds_mixed_mode_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _load_v1_postflight_fixture()
    original_builder = fixture._trajectory_record

    def full_record(assignment: object):
        base = original_builder(assignment)
        queries = []
        for query in base.queries:
            all_paths = tuple(query.ocr_records_by_path)

            def payload_for(path: str) -> bytes:
                step = int(path.rsplit("-", 1)[1].removesuffix(".png"))
                return _png("RGB" if step % 2 == 0 else "RGBA")

            ocr = {
                path: build_ocr_record(
                    image_member_path=path,
                    image_bytes=payload_for(path),
                    backend_config_sha256="7" * 64,
                    boxes=None,
                    texts=None,
                    scores=None,
                )
                for path in all_paths
            }
            queries.append(
                replace(
                    query,
                    image_payloads={
                        path: payload_for(path) for path in query.image_payloads
                    },
                    ocr_records_by_path=ocr,
                )
            )
        return replace(base, queries=tuple(queries))

    fixture._trajectory_record = full_record
    root, base_context, _, worker_tallies = fixture._write_root(tmp_path)
    renamed = root.with_name(
        "causalcache-set-utility-processor-freeze-v2-image-contract-repair-"
        + "d" * 7
    )
    root.rename(renamed)
    root = renamed
    for worker_index in range(4):
        diagnostics = b"warning: accepted diagnostic text\n" if worker_index == 0 else b""
        (root / "logs" / f"processor-worker-{worker_index:02d}.log").write_bytes(
            _processor_thread_evidence_line() + b"\n" + diagnostics
        )

    image_contract_sha = "6" * 64
    structural = replace(
        base_context,
        ocr_runtime_identity={
            **base_context.ocr_runtime_identity,
            "image_contract_sha256": image_contract_sha,
            "ocr_concurrency_per_logical_worker": 32,
        },
    )
    for worker_index in range(4):
        path = root / "receipts" / f"ocr-worker-{worker_index:02d}.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["ocr_runtime_identity"] = structural.ocr_runtime_identity
        path.write_bytes(canonical_json_bytes(receipt))
    identity_path = root / "run-identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["runtime_cli"]["output_root"] = str(root.resolve())
    identity_path.write_bytes(canonical_json_bytes(identity))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_identity_sha256"] = sha256_bytes(canonical_json_bytes(identity))
    manifest_path.write_bytes(canonical_pretty_json_bytes(manifest))

    expected_tally: Counter[str] = Counter()
    for tally in worker_tallies:
        expected_tally.update(tally)
    monkeypatch.setattr(
        postflight_v2,
        "PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS",
        dict(expected_tally),
    )
    monkeypatch.setattr(
        postflight_v2,
        "PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_TOTAL",
        sum(expected_tally.values()),
    )
    context = ProcessorFreezePostflightContextV2(
        structural_context=structural,
        backend_config=load_backend_config(BACKEND_CONFIG_PATH),
        backend_config_sha256="7" * 64,
        image_contract_sha256=image_contract_sha,
    )
    serial_semantics = _serial_v2_semantics_reference(root, context)
    tree_before = _tree_snapshot(root)

    result = validate_completed_processor_freeze_root_v2(root, context=context)
    parallel_result = validate_completed_processor_freeze_root_parallel_v1(
        root,
        context=context,
    )

    assert _tree_snapshot(root) == tree_before
    assert result["status"] == VALIDATION_STATUS
    assert result["processor_image_contract_id"].endswith(
        "png_rgb_allowlist_repair"
    )
    assert result["global_format_mode_tally"] == dict(sorted(expected_tally.items()))
    assert len(result["processor_thread_runtime_evidence"]) == 4
    assert result["processor_thread_runtime_evidence"][0][
        "diagnostic_line_count"
    ] == 1
    assert result["stored_image_ocr_validation_count"] > 0
    assert result["metadata_only_ocr_validation_count"] == 4
    assert (
        result["global_format_mode_tally"],
        result["stored_image_ocr_validation_count"],
        result["metadata_only_ocr_validation_count"],
    ) == serial_semantics
    assert parallel_result["status"] == PARALLEL_VALIDATION_STATUS
    assert parallel_result["protocol_id"] == PARALLEL_PROTOCOL_ID
    assert parallel_result["status"] != result["status"]
    assert (
        parallel_result["global_format_mode_tally"],
        parallel_result["stored_image_ocr_validation_count"],
        parallel_result["metadata_only_ocr_validation_count"],
    ) == serial_semantics
    assert parallel_result["parallel_semantic_worker_count"] == 4
    assert parallel_result["parallel_semantic_aggregation_order"] == [0, 1, 2, 3]

    arbitrary = root.with_name(
        "causalcache-set-utility-processor-freeze-v2-image-contract-repair-formal"
    )
    root.rename(arbitrary)
    with pytest.raises(ValueError, match="exact Git-bound namespace"):
        validate_completed_processor_freeze_root_v2(arbitrary, context=context)
