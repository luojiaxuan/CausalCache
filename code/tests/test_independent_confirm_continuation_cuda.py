from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest

import causalcache.independent_confirm_execution as execution
from causalcache.independent_confirm_artifact import PayloadPublicationReceipt
from causalcache.independent_confirm_runner import ConfirmWorkerAssignment


class _CudaWithoutBadFork:
    def __init__(self, *, initialized: bool) -> None:
        self.initialized = initialized
        self.initialized_call_count = 0
        self.forbidden_query_count = 0

    def is_initialized(self) -> bool:
        self.initialized_call_count += 1
        return self.initialized

    def is_available(self) -> bool:
        self.forbidden_query_count += 1
        raise AssertionError("fork guard must not query CUDA availability")

    def device_count(self) -> int:
        self.forbidden_query_count += 1
        raise AssertionError("fork guard must not query CUDA device count")


class _CudaWithBadFork(_CudaWithoutBadFork):
    def __init__(self, *, initialized: bool, bad_fork: bool) -> None:
        super().__init__(initialized=initialized)
        self.bad_fork = bad_fork
        self.bad_fork_call_count = 0

    def _is_in_bad_fork(self) -> bool:
        self.bad_fork_call_count += 1
        return self.bad_fork


class _TripwireCuda(_CudaWithBadFork):
    def __init__(self) -> None:
        super().__init__(initialized=False, bad_fork=False)
        self.available_calls = 0
        self.device_count_calls = 0
        self.lazy_init_calls = 0

    def is_available(self) -> bool:
        self.available_calls += 1
        return True

    def device_count(self) -> int:
        self.device_count_calls += 1
        return 4

    def _lazy_init(self) -> None:
        self.lazy_init_calls += 1


def test_cuda_clean_guard_uses_only_non_initializing_state_queries() -> None:
    cuda = _CudaWithoutBadFork(initialized=False)

    result = execution.assert_fork_parent_cuda_clean(
        SimpleNamespace(cuda=cuda)
    )

    assert dict(result) == {
        "cuda_initialized": False,
        "cuda_bad_fork": False,
        "bad_fork_check_available": False,
    }
    assert cuda.initialized_call_count == 1
    assert cuda.forbidden_query_count == 0


@pytest.mark.parametrize(
    ("initialized", "bad_fork"),
    ((True, False), (False, True), (True, True)),
)
def test_cuda_clean_guard_rejects_initialized_or_bad_fork_parent(
    initialized: bool,
    bad_fork: bool,
) -> None:
    cuda = _CudaWithBadFork(initialized=initialized, bad_fork=bad_fork)

    with pytest.raises(RuntimeError, match="CUDA-clean parent"):
        execution.assert_fork_parent_cuda_clean(SimpleNamespace(cuda=cuda))

    assert cuda.initialized_call_count == 1
    assert cuda.bad_fork_call_count == 1
    assert cuda.forbidden_query_count == 0


def test_real_restoration_path_guards_before_fork_context_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def reject_parent() -> None:
        events.append("cuda-guard")
        raise RuntimeError("parent poisoned")

    def forbidden_context(_method: str):
        events.append("context-created")
        pytest.fail("fork context was created before the CUDA guard passed")

    monkeypatch.setattr(execution.os, "name", "posix")
    monkeypatch.setattr(
        execution.multiprocessing,
        "get_all_start_methods",
        lambda: ("fork",),
    )
    monkeypatch.setattr(execution, "assert_fork_parent_cuda_clean", reject_parent)
    monkeypatch.setattr(execution.multiprocessing, "get_context", forbidden_context)
    assignments = tuple(
        ConfirmWorkerAssignment(
            worker_id=f"worker-{index}",
            ordinals=(),
            state_ids=(),
        )
        for index in range(4)
    )

    with pytest.raises(RuntimeError, match="parent poisoned"):
        execution.run_four_worker_restoration(
            assignments=assignments,
            receipt=object(),
            run_contract_sha256="a" * 64,
            payloads=object(),
            work_items=(),
            model_dir="/model",
            snapshot_manifest="/snapshot.json",
            devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
            gpu_uuids=(
                "GPU-00000000-0000-0000-0000-000000000000",
                "GPU-11111111-1111-1111-1111-111111111111",
                "GPU-22222222-2222-2222-2222-222222222222",
                "GPU-33333333-3333-3333-3333-333333333333",
            ),
            phase_timeout_seconds=1,
            worker_termination_grace_seconds=1,
        )

    assert events == ["cuda-guard"]


def test_cuda_tripwire_blocks_parent_init_and_restores_in_fork_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = _TripwireCuda()
    torch_module = SimpleNamespace(cuda=cuda)
    child_handlers = []
    monkeypatch.setattr(
        execution.os,
        "register_at_fork",
        lambda **kwargs: child_handlers.append(kwargs["after_in_child"]),
    )

    tripwire = execution.install_fork_parent_cuda_tripwire(torch_module)
    try:
        with pytest.raises(RuntimeError, match="pre-fork parent"):
            cuda.is_available()
        with pytest.raises(RuntimeError, match="pre-fork parent"):
            cuda.device_count()
        with pytest.raises(RuntimeError, match="pre-fork parent"):
            cuda._lazy_init()
        execution.assert_fork_parent_cuda_clean(
            torch_module,
            require_tripwire=True,
        )
        assert len(child_handlers) == 1
        child_handlers[0]()
        assert cuda.is_available() is True
        assert cuda.device_count() == 4
        cuda._lazy_init()
        assert cuda.available_calls == 1
        assert cuda.device_count_calls == 1
        assert cuda.lazy_init_calls == 1
    finally:
        execution.release_fork_parent_cuda_tripwire(tripwire)


def test_explicit_fork_context_cannot_bypass_cuda_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    class Context:
        def get_start_method(self):
            return "fork"

        def Queue(self):
            pytest.fail("queue was created before CUDA guard")

    monkeypatch.setattr(
        execution,
        "assert_fork_parent_cuda_clean",
        lambda **kwargs: events.append(kwargs) or (_ for _ in ()).throw(
            RuntimeError("guarded explicit context")
        ),
    )
    assignments = tuple(
        ConfirmWorkerAssignment(
            worker_id=f"worker-{index}",
            ordinals=(),
            state_ids=(),
        )
        for index in range(4)
    )
    with pytest.raises(RuntimeError, match="guarded explicit context"):
        execution.run_four_worker_restoration(
            assignments=assignments,
            receipt=object(),
            run_contract_sha256="a" * 64,
            payloads=object(),
            work_items=(),
            model_dir="/model",
            snapshot_manifest="/snapshot.json",
            devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
            gpu_uuids=(
                "GPU-00000000-0000-0000-0000-000000000000",
                "GPU-11111111-1111-1111-1111-111111111111",
                "GPU-22222222-2222-2222-2222-222222222222",
                "GPU-33333333-3333-3333-3333-333333333333",
            ),
            phase_timeout_seconds=1,
            worker_termination_grace_seconds=1,
            context=Context(),
            require_parent_cuda_tripwire=True,
        )
    assert events == [{"require_tripwire": True}]


def test_durable_terminals_are_mutually_exclusive(tmp_path) -> None:
    completed = execution.DurableConfirmOutput(tmp_path / "completed")
    completed.start_attempt({"status": "STARTED"})
    completed.write_completion({"status": "COMPLETED"})
    with pytest.raises(RuntimeError, match="completion terminal"):
        completed.write_failure({"status": "FAILED"})

    failed = execution.DurableConfirmOutput(tmp_path / "failed")
    failed.start_attempt({"status": "STARTED"})
    failed.write_failure({"status": "FAILED"})
    with pytest.raises(RuntimeError, match="failure terminal"):
        failed.write_completion({"status": "COMPLETED"})


class _NoMutationApi:
    def __init__(self) -> None:
        self.mutation_count = 0

    def create_repo(self, *_args, **_kwargs) -> None:
        self.mutation_count += 1
        pytest.fail("continuation adoption attempted to create a repository")

    def create_commit(self, *_args, **_kwargs) -> None:
        self.mutation_count += 1
        pytest.fail("continuation adoption attempted to create a commit")

    def create_tag(self, *_args, **_kwargs) -> None:
        self.mutation_count += 1
        pytest.fail("continuation adoption attempted to create a tag")


def test_publisher_adopts_existing_payload_without_enabling_payload_publish(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_commit = "1" * 40
    payload_commit = "2" * 40
    inventory_sha256 = "3" * 64
    expected_files = MappingProxyType({"sealed.bin": b"sealed\n"})
    seal = SimpleNamespace(
        files=expected_files,
        inventory_sha256=inventory_sha256,
    )
    receipt = PayloadPublicationReceipt(
        base_commit=base_commit,
        payload_commit=payload_commit,
        payload_inventory_sha256=inventory_sha256,
    )
    api = _NoMutationApi()
    download_calls = []

    def raw_download(**kwargs):
        download_calls.append(dict(kwargs))
        assert kwargs.pop("token") == "secret"
        return str(tmp_path / "unused")

    def adopt(**kwargs):
        assert kwargs["api"] is api
        assert kwargs["expected_base_commit"] == base_commit
        assert kwargs["expected_base_title"] == "initial commit"
        assert kwargs["expected_payload_commit"] == payload_commit
        assert kwargs["expected_payload_files"] is expected_files
        assert kwargs["fresh_parent"] == (
            tmp_path / "adopted-payload-fresh-replay"
        )
        kwargs["download_fn"](
            repo_id="private/repo",
            repo_type="dataset",
            filename="sealed.bin",
            revision=payload_commit,
            local_dir=tmp_path,
            force_download=True,
        )
        return SimpleNamespace(payload_seal=seal, payload_receipt=receipt)

    monkeypatch.setattr(
        execution,
        "adopt_artifact_existing_payload_commit",
        adopt,
    )
    publisher = object.__new__(execution.HuggingFaceConfirmPublisher)
    publisher._api = api
    publisher._cache_dir = tmp_path
    publisher._download_fn = raw_download
    publisher._operation_factory = object()
    publisher._payload_receipt = None
    publisher._payload_seal = None
    publisher._preflight_base_commit = None
    publisher._report_publication_attempted = False
    publisher._report_publication_audit = None
    publisher._token = "secret"

    result = publisher.adopt_existing_payload(
        expected_files=expected_files,
        expected_base_commit=base_commit,
        expected_base_title="initial commit",
        expected_payload_commit=payload_commit,
    )

    assert dict(result) == {
        "base_commit": base_commit,
        "payload_commit": payload_commit,
        "payload_inventory_sha256": inventory_sha256,
        "payload_file_count": 1,
        "byte_identical_fresh_replay": True,
        "remote_mutation_performed": False,
    }
    assert publisher._payload_seal is seal
    assert publisher._payload_receipt is receipt
    assert publisher._preflight_base_commit is None
    assert len(download_calls) == 1
    assert api.mutation_count == 0
    publication_audit = publisher.report_publication_audit()
    assert publication_audit["status"] == "REPORT_PUBLICATION_NOT_ATTEMPTED"
    assert publication_audit["payload_commit"] == payload_commit
    assert publication_audit["remote_mutation_count"] == 0
    with pytest.raises(RuntimeError, match="fresh publisher"):
        publisher.adopt_existing_payload(
            expected_files=expected_files,
            expected_base_commit=base_commit,
            expected_base_title="initial commit",
            expected_payload_commit=payload_commit,
        )
    with pytest.raises(ValueError, match="runner payload bytes"):
        publisher.publish_payload(expected_files, ())
    assert api.mutation_count == 0
