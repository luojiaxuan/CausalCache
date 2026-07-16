from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from causalcache.policy.gui_owl_v2_2_vision_runtime import (
    GUI_OWL_V2_2_VISION_RUNTIME_UUID_TYPE_ONLY_V2_ID,
)
from causalcache.restoration_v2_2_policy_vision_contract import (
    CANONICAL_CONFIG_PATH as V1_CONFIG_PATH,
)
from causalcache.restoration_v2_2_policy_vision_v2_contract import (
    CANONICAL_CONFIG_PATH,
    FORMAL_ATTEMPT_LEDGER_PATH,
    GPU_UUID_RUNTIME_TYPE_PROFILE,
    PROTOCOL_ID,
    RUN_STATUS,
    VALID_STATUS,
    RestorationV22PolicyVisionV2Contract,
)
from scripts.run_restoration_v2_2_policy_vision_baseline import (
    _expected_runtime_metadata,
    _claim_formal_attempt,
    _run_feature_workers,
    _source_execution,
    _validate_runner_protocol,
)
from scripts.run_restoration_v2_2_policy_vision_baseline_v2 import (
    V2_FORMAL_SOURCE_PATHS,
    V2_RUNNER_PROTOCOL,
    main,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


@dataclass(frozen=True)
class _Item:
    state_index: int


class _Future:
    def __init__(self, value: object) -> None:
        self.value = value

    def result(self) -> object:
        return self.value


class _Executor:
    submissions: list[dict[str, object]] = []

    def __init__(self, *, max_workers: int, mp_context: object) -> None:
        if max_workers != 2 or mp_context != "spawn-context":
            raise AssertionError("v2 runner did not preserve the two-worker schedule")

    def __enter__(self) -> "_Executor":
        type(self).submissions = []
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def submit(self, function: object, **kwargs: object) -> _Future:
        type(self).submissions.append(kwargs)
        return _Future({"worker_id": kwargs["worker_id"]})


class RunRestorationV22PolicyVisionBaselineV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = RestorationV22PolicyVisionV2Contract.load(
            CONFIG,
            repository_root=ROOT,
            validate_bound_sources=False,
        )

    def test_runner_identity_is_exactly_v2(self) -> None:
        protocol = V2_RUNNER_PROTOCOL
        self.assertEqual(protocol.protocol_id, PROTOCOL_ID)
        self.assertEqual(protocol.run_status, RUN_STATUS)
        self.assertEqual(protocol.valid_status, VALID_STATUS)
        self.assertIs(
            protocol.contract_class,
            RestorationV22PolicyVisionV2Contract,
        )
        self.assertEqual(
            protocol.runtime_profile_id,
            GUI_OWL_V2_2_VISION_RUNTIME_UUID_TYPE_ONLY_V2_ID,
        )
        self.assertEqual(
            protocol.gpu_uuid_type_profile,
            GPU_UUID_RUNTIME_TYPE_PROFILE,
        )
        self.assertTrue(protocol.include_repair_identity)
        self.assertTrue(protocol.formal_run_allowed)
        self.assertEqual(
            protocol.formal_attempt_ledger_path,
            FORMAL_ATTEMPT_LEDGER_PATH,
        )
        _validate_runner_protocol(protocol, self.contract)

    def test_attempt_claim_is_durable_and_exclusive(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "attempt.json"
            claim = {"schema_version": "1.0.0", "status": "CLAIMED"}
            recorded, digest = _claim_formal_attempt(path, claim)
            self.assertEqual(recorded, claim)
            self.assertEqual(len(digest), 64)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(FileExistsError, "already claimed"):
                _claim_formal_attempt(path, claim)

    def test_v2_formal_sources_bind_parent_repair_and_failed_attempt(self) -> None:
        self.assertEqual(len(V2_FORMAL_SOURCE_PATHS), len(set(V2_FORMAL_SOURCE_PATHS)))
        self.assertEqual(V2_FORMAL_SOURCE_PATHS[0], CANONICAL_CONFIG_PATH)
        for required in (
            V1_CONFIG_PATH,
            "code/causalcache/restoration_v2_2_policy_vision_v2_contract.py",
            "code/scripts/run_restoration_v2_2_policy_vision_baseline_v2.py",
            "code/scripts/validate_restoration_v2_2_policy_vision_v2_contract.py",
            "data/results/restoration_v2_2_policy_vision_baseline_v1_attempt/README.md",
            (
                "data/results/"
                "restoration_v2_2_policy_vision_baseline_v1_attempt/failure.json"
            ),
        ):
            self.assertIn(required, V2_FORMAL_SOURCE_PATHS)

    def test_feature_workers_receive_only_the_explicit_v2_type_profile(self) -> None:
        items = tuple(
            _Item(state_index=index)
            for index in range(2, 17)
        )
        with patch(
            "scripts.run_restoration_v2_2_policy_vision_baseline."
            "ProcessPoolExecutor",
            _Executor,
        ), patch(
            "scripts.run_restoration_v2_2_policy_vision_baseline."
            "multiprocessing.get_context",
            return_value="spawn-context",
        ):
            outputs = _run_feature_workers(
                work_items=items,
                derived_root=Path("/derived"),
                derived_payload_prefix="derived/restoration-v2-v1",
                expected_tar_member_count=210,
                model_dir=Path("/model"),
                snapshot_manifest=Path("/snapshot.json"),
                expected_gpu_uuids=(
                    "GPU-00000000-0000-0000-0000-000000000001",
                    "GPU-00000000-0000-0000-0000-000000000002",
                ),
                gpu_uuid_type_profile=GPU_UUID_RUNTIME_TYPE_PROFILE,
            )
        self.assertEqual([row["worker_id"] for row in outputs], ["even", "odd"])
        self.assertEqual(len(_Executor.submissions), 2)
        for submission in _Executor.submissions:
            self.assertEqual(
                submission["gpu_uuid_type_profile"],
                GPU_UUID_RUNTIME_TYPE_PROFILE,
            )
            self.assertFalse(
                any(
                    "label" in key or "ocr" in key or "goal" in key
                    for key in submission
                )
            )

    def test_runtime_metadata_and_summary_provenance_use_v2_identity(self) -> None:
        metadata = _expected_runtime_metadata(
            contract=self.contract,
            device="cuda:0",
            gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
            runtime_profile_id=V2_RUNNER_PROTOCOL.runtime_profile_id,
        )
        self.assertEqual(
            metadata["runtime_profile_id"],
            GUI_OWL_V2_2_VISION_RUNTIME_UUID_TYPE_ONLY_V2_ID,
        )
        closure = {
            "rule": "test-rule",
            "path_count": 7,
            "inventory_sha256": "a" * 64,
        }
        with patch(
            "scripts.run_restoration_v2_2_policy_vision_baseline."
            "_formal_python_source_closure",
            return_value=closure,
        ):
            source = _source_execution(
                contract=self.contract,
                source_commit="b" * 40,
                protocol=V2_RUNNER_PROTOCOL,
            )
        self.assertEqual(source["repair_identity"], self.contract.repair_identity)
        self.assertEqual(source["formal_source_paths"], list(V2_FORMAL_SOURCE_PATHS))
        self.assertEqual(source["formal_python_source_closure"], closure)

    def test_v2_entrypoint_rejects_the_v1_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "canonical config"):
            main(
                [
                    "validate",
                    "--repository-root",
                    str(ROOT),
                    "--contract",
                    str(ROOT / V1_CONFIG_PATH),
                    "--labels-archive",
                    str(ROOT / "unused-labels.tar"),
                    "--source-git-commit",
                    "a" * 40,
                ]
            )


if __name__ == "__main__":
    unittest.main()
