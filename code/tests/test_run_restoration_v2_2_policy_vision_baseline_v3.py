from __future__ import annotations

import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

from causalcache.policy.gui_owl_v2_2_vision_runtime import (
    GPU_UUID_TYPE_PROFILE_V2,
    GUI_OWL_V2_2_VISION_RUNTIME_UUID_SIZE_DICT_V3_ID,
    IMAGE_PROCESSOR_SIZE_PROFILE_V1,
    IMAGE_PROCESSOR_SIZE_PROFILE_V3,
)
from causalcache.restoration_v2_2_policy_vision_v3_contract import (
    CANONICAL_CONFIG_PATH,
    FORMAL_ATTEMPT_LEDGER_PATH,
    PROTOCOL_ID,
    RUN_STATUS,
    VALID_STATUS,
    RestorationV22PolicyVisionV3Contract,
)
from scripts.run_restoration_v2_2_policy_vision_baseline import (
    _expected_runtime_metadata,
    _run_feature_workers,
    _source_execution,
    _validate_runner_protocol,
)
from scripts.run_restoration_v2_2_policy_vision_baseline_v2 import (
    V2_FORMAL_SOURCE_PATHS,
)
from scripts.run_restoration_v2_2_policy_vision_baseline_v3 import (
    V3_FORMAL_SOURCE_PATHS,
    V3_RUNNER_PROTOCOL,
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
            raise AssertionError("v3 runner changed the two-worker schedule")

    def __enter__(self) -> "_Executor":
        type(self).submissions = []
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def submit(self, function: object, **kwargs: object) -> _Future:
        type(self).submissions.append(kwargs)
        return _Future({"worker_id": kwargs["worker_id"]})


class RunRestorationV22PolicyVisionBaselineV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = RestorationV22PolicyVisionV3Contract.load(
            CONFIG,
            repository_root=ROOT,
            validate_bound_sources=False,
        )

    def test_runner_identity_is_exactly_v3(self) -> None:
        protocol = V3_RUNNER_PROTOCOL
        self.assertEqual(protocol.protocol_id, PROTOCOL_ID)
        self.assertEqual(protocol.run_status, RUN_STATUS)
        self.assertEqual(protocol.valid_status, VALID_STATUS)
        self.assertIs(
            protocol.contract_class,
            RestorationV22PolicyVisionV3Contract,
        )
        self.assertEqual(
            protocol.runtime_profile_id,
            GUI_OWL_V2_2_VISION_RUNTIME_UUID_SIZE_DICT_V3_ID,
        )
        self.assertEqual(
            protocol.gpu_uuid_type_profile,
            GPU_UUID_TYPE_PROFILE_V2,
        )
        self.assertEqual(
            protocol.image_processor_size_profile,
            IMAGE_PROCESSOR_SIZE_PROFILE_V3,
        )
        self.assertTrue(protocol.formal_run_allowed)
        self.assertIsNone(protocol.run_tombstone_status)
        self.assertTrue(protocol.include_repair_identity)
        self.assertEqual(
            protocol.formal_attempt_ledger_path,
            FORMAL_ATTEMPT_LEDGER_PATH,
        )
        _validate_runner_protocol(protocol, self.contract)

    def test_v3_formal_sources_bind_v2_failure_and_all_parent_sources(self) -> None:
        self.assertEqual(len(V3_FORMAL_SOURCE_PATHS), len(set(V3_FORMAL_SOURCE_PATHS)))
        self.assertEqual(V3_FORMAL_SOURCE_PATHS[0], CANONICAL_CONFIG_PATH)
        for path in V2_FORMAL_SOURCE_PATHS:
            self.assertIn(path, V3_FORMAL_SOURCE_PATHS)
        for required in (
            "code/causalcache/restoration_v2_2_policy_vision_v3_contract.py",
            "code/scripts/run_restoration_v2_2_policy_vision_baseline_v3.py",
            "code/scripts/validate_restoration_v2_2_policy_vision_v3_contract.py",
            (
                "data/results/"
                "restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair_"
                "attempt/README.md"
            ),
            (
                "data/results/"
                "restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair_"
                "attempt/failure.json"
            ),
        ):
            self.assertIn(required, V3_FORMAL_SOURCE_PATHS)

    def test_feature_workers_receive_explicit_uuid_and_size_profiles(self) -> None:
        items = tuple(_Item(state_index=index) for index in range(2, 17))
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
                gpu_uuid_type_profile=(
                    V3_RUNNER_PROTOCOL.gpu_uuid_type_profile
                ),
                image_processor_size_profile=(
                    V3_RUNNER_PROTOCOL.image_processor_size_profile
                ),
            )
        self.assertEqual([row["worker_id"] for row in outputs], ["even", "odd"])
        self.assertEqual(len(_Executor.submissions), 2)
        for submission in _Executor.submissions:
            self.assertEqual(
                submission["gpu_uuid_type_profile"],
                GPU_UUID_TYPE_PROFILE_V2,
            )
            self.assertEqual(
                submission["image_processor_size_profile"],
                IMAGE_PROCESSOR_SIZE_PROFILE_V3,
            )
            self.assertFalse(
                any(
                    "label" in key or "ocr" in key or "goal" in key
                    for key in submission
                )
            )

    def test_runtime_metadata_and_summary_use_combined_v3_identity(self) -> None:
        metadata = _expected_runtime_metadata(
            contract=self.contract,
            device="cuda:0",
            gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
            runtime_profile_id=V3_RUNNER_PROTOCOL.runtime_profile_id,
        )
        self.assertEqual(
            metadata["runtime_profile_id"],
            GUI_OWL_V2_2_VISION_RUNTIME_UUID_SIZE_DICT_V3_ID,
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
                protocol=V3_RUNNER_PROTOCOL,
            )
        self.assertEqual(source["repair_identity"], self.contract.repair_identity)
        self.assertEqual(source["formal_source_paths"], list(V3_FORMAL_SOURCE_PATHS))
        self.assertEqual(source["formal_python_source_closure"], closure)

    def test_runner_rejects_non_v3_size_profile_or_disabled_run(self) -> None:
        for protocol in (
            replace(
                V3_RUNNER_PROTOCOL,
                image_processor_size_profile=IMAGE_PROCESSOR_SIZE_PROFILE_V1,
            ),
            replace(V3_RUNNER_PROTOCOL, formal_run_allowed=False),
            replace(
                V3_RUNNER_PROTOCOL,
                run_tombstone_status="INVALID_UNEXPECTED_V3_TOMBSTONE",
            ),
        ):
            with self.assertRaisesRegex(ValueError, "identity drifted"):
                _validate_runner_protocol(protocol, self.contract)

    def test_v3_entrypoint_rejects_the_v2_contract(self) -> None:
        v2_config = ROOT / V2_FORMAL_SOURCE_PATHS[0]
        with self.assertRaisesRegex(ValueError, "canonical config"):
            main(
                [
                    "validate",
                    "--repository-root",
                    str(ROOT),
                    "--contract",
                    str(v2_config),
                    "--labels-archive",
                    str(ROOT / "unused-labels.tar"),
                    "--source-git-commit",
                    "a" * 40,
                ]
            )


if __name__ == "__main__":
    unittest.main()
