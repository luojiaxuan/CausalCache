from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_spatial_reference_audit_v1 import (
    ATTEMPT_ID,
    CANONICAL_ATTEMPT_ROOT,
    CANONICAL_GLOBAL_LEDGER,
    PROFILE_ORDER,
    AttemptLayout,
    _begin_profile,
    _ensure_attempt_layout,
    _live_execution_identity,
    _load_json_object,
    _replace_json_durable,
    _run_state,
    _state_layout,
    _validated_attempt_identity,
    _write_json_exclusive,
)
from causalcache.data.guiodyssey_restoration_v2 import canonical_json_bytes
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.spatial_reference_audit_v1 import EXPECTED_RUNTIME_CONSTRAINTS


def _attempt_config() -> dict[str, object]:
    return {
        "attempt_identity": {
            "attempt_id": ATTEMPT_ID,
            "canonical_persistent_output_dir": str(CANONICAL_ATTEMPT_ROOT),
            "canonical_global_attempt_ledger": str(CANONICAL_GLOBAL_LEDGER),
            "canonical_host_alias": "hyper01",
            "canonical_host_hostname": "node-radixark-16-0001",
            "canonical_device": "cuda:0",
            "profile_order": list(PROFILE_ORDER),
            "alternate_output_or_ledger_allowed": False,
            "incomplete_attempt_retry_allowed": False,
            "profile_retry_allowed": False,
            "state_retry_allowed": False,
            "output_or_ledger_deletion_after_claim_allowed": False,
        }
    }


def _host() -> dict[str, object]:
    return {
        "alias": "hyper01",
        "hostname": "node-radixark-16-0001",
        "container_hostname": "a" * 12,
        "container_id": "a" * 64,
        "container_image_digest": "sha256:" + "b" * 64,
        "device": "cuda:0",
        "visible_gpu_count": 1,
        "cuda_visible_ordinal": 0,
        "nvidia_smi_gpu_index": 6,
        "gpu_name": "NVIDIA H200",
        "gpu_uuid": "GPU-test",
        "gpu_pci_bus_id": "00000000:01:00.0",
        "nvidia_smi_query": "query",
    }


class SpatialReferenceAuditRunnerTest(unittest.TestCase):
    def test_durable_json_is_exclusive_and_replaceable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.json"
            _write_json_exclusive(path, {"value": 1})
            self.assertEqual(_load_json_object(path), {"value": 1})
            with self.assertRaises(FileExistsError):
                _write_json_exclusive(path, {"value": 2})
            _replace_json_durable(path, {"value": 3})
            self.assertEqual(json.loads(path.read_bytes()), {"value": 3})

    def test_attempt_identity_is_read_from_and_bound_to_config(self) -> None:
        layout = _validated_attempt_identity(_attempt_config())
        self.assertEqual(layout.root, CANONICAL_ATTEMPT_ROOT)
        self.assertEqual(layout.global_ledger, CANONICAL_GLOBAL_LEDGER)
        drifted = _attempt_config()
        drifted["attempt_identity"]["profile_retry_allowed"] = True  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "identity"):
            _validated_attempt_identity(drifted)

    def test_live_identity_requires_one_h200_and_container_prefix(self) -> None:
        args = argparse.Namespace(
            container_id="a" * 64,
            container_image_digest=EXPECTED_RUNTIME_CONSTRAINTS[
                "container_image_digest"
            ],
            device="cuda:0",
            host_alias="hyper01",
            host_hostname="node-radixark-16-0001",
        )
        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout=(
                "6, NVIDIA H200, GPU-test, 00000000:01:00.0, 570.172.08\n"
            ),
            stderr="",
        )
        software = {
            key: EXPECTED_RUNTIME_CONSTRAINTS[key]
            for key in (
                "python_version",
                "torch_version",
                "torch_cuda_version",
                "cudnn_version",
                "transformers_version",
            )
        }
        environment_audit = {
            "audited_names": EXPECTED_RUNTIME_CONSTRAINTS[
                "audited_scientific_environment_variables"
            ],
            "present_names": [],
            "all_absent": True,
        }
        with mock.patch("socket.gethostname", return_value="a" * 12), mock.patch(
            "subprocess.run",
            return_value=completed,
        ), mock.patch(
            "scripts.run_spatial_reference_audit_v1._live_software_identity",
            return_value=software,
        ), mock.patch(
            "scripts.run_spatial_reference_audit_v1.audit_absent_scientific_environment",
            return_value=environment_audit,
        ):
            identity = _live_execution_identity(
                args,
                runtime_constraints=EXPECTED_RUNTIME_CONSTRAINTS,
            )
        self.assertEqual(identity["cuda_visible_ordinal"], 0)
        self.assertEqual(identity["nvidia_smi_gpu_index"], 6)
        self.assertEqual(identity["gpu_name"], "NVIDIA H200")
        self.assertEqual(identity["nvidia_driver_version"], "570.172.08")
        self.assertEqual(identity["software"], software)
        self.assertEqual(identity["scientific_environment_audit"], environment_audit)

        name_drift = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout=(
                "6, NVIDIA H200 NVL, GPU-test, 00000000:01:00.0, 570.172.08\n"
            ),
            stderr="",
        )
        with mock.patch("socket.gethostname", return_value="a" * 12), mock.patch(
            "subprocess.run",
            return_value=name_drift,
        ), mock.patch(
            "scripts.run_spatial_reference_audit_v1._live_software_identity",
            return_value=software,
        ), mock.patch(
            "scripts.run_spatial_reference_audit_v1.audit_absent_scientific_environment",
            return_value=environment_audit,
        ), self.assertRaisesRegex(RuntimeError, "exactly one NVIDIA H200"):
            _live_execution_identity(
                args,
                runtime_constraints=EXPECTED_RUNTIME_CONSTRAINTS,
            )

        completed.stdout += (
            "7, NVIDIA H200, GPU-other, 00000000:02:00.0, 570.172.08\n"
        )
        with mock.patch("socket.gethostname", return_value="a" * 12), mock.patch(
            "subprocess.run",
            return_value=completed,
        ), mock.patch(
            "scripts.run_spatial_reference_audit_v1._live_software_identity",
            return_value=software,
        ), mock.patch(
            "scripts.run_spatial_reference_audit_v1.audit_absent_scientific_environment",
            return_value=environment_audit,
        ), self.assertRaisesRegex(RuntimeError, "exactly one"):
            _live_execution_identity(
                args,
                runtime_constraints=EXPECTED_RUNTIME_CONSTRAINTS,
            )

        args.container_image_digest = "sha256:" + "b" * 64
        with self.assertRaisesRegex(ValueError, "frozen runtime"):
            _live_execution_identity(
                args,
                runtime_constraints=EXPECTED_RUNTIME_CONSTRAINTS,
            )

    def test_profile_order_and_duplicate_start_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            layout = AttemptLayout(
                root=parent / "attempt",
                global_ledger=parent / ".attempt.json",
                profiles_root=parent / "attempt/profiles",
                attempt_start=parent / "attempt/start.json",
            )
            with self.assertRaisesRegex(PermissionError, "first canonical"):
                _ensure_attempt_layout(
                    layout=layout,
                    profile_id=PROFILE_ORDER[1],
                    source_git_commit="c" * 40,
                    config_sha256="d" * 64,
                    host=_host(),
                    execution_inputs={"repository_root": "/repo"},
                    invocation_argv=["runner", "--profile-id", PROFILE_ORDER[1]],
                )
            self.assertFalse(layout.root.exists())

            ledger = _ensure_attempt_layout(
                layout=layout,
                profile_id=PROFILE_ORDER[0],
                source_git_commit="c" * 40,
                config_sha256="d" * 64,
                host=_host(),
                execution_inputs={"repository_root": "/repo"},
                invocation_argv=["runner", "--profile-id", PROFILE_ORDER[0]],
            )
            with self.assertRaisesRegex(PermissionError, "already claimed"):
                _ensure_attempt_layout(
                    layout=layout,
                    profile_id=PROFILE_ORDER[0],
                    source_git_commit="c" * 40,
                    config_sha256="d" * 64,
                    host=_host(),
                    execution_inputs={"repository_root": "/repo"},
                    invocation_argv=["runner", "--profile-id", PROFILE_ORDER[0]],
                )
            profile = types.SimpleNamespace(
                profile_id=PROFILE_ORDER[0],
                state_indices=(1, 10),
            )
            _begin_profile(
                layout=layout,
                ledger=ledger,
                profile=profile,
                invocation_argv=["runner", "--profile-id", PROFILE_ORDER[0]],
            )
            refreshed = _load_json_object(layout.global_ledger)
            with self.assertRaisesRegex(PermissionError, "already started"):
                _begin_profile(
                    layout=layout,
                    ledger=refreshed,
                    profile=profile,
                    invocation_argv=["runner", "--profile-id", PROFILE_ORDER[0]],
                )

    def test_state_uses_actual_ids_and_exact_four_teacher_forwards(self) -> None:
        class Runtime:
            def __init__(self) -> None:
                self.processor = types.SimpleNamespace(tokenizer=object())
                self.generation_calls = 0
                self.shared_calls = 0
                self.full_calls: list[int] = []

            def generate_native_action(self, messages: object) -> object:
                self.generation_calls += 1
                token_ids = [1, 2, 3]
                metadata = {
                    "generated_token_ids": token_ids,
                    "generated_token_ids_sha256": hashlib.sha256(
                        canonical_json_bytes(token_ids)
                    ).hexdigest(),
                }
                action = GUIOwlV2Action(action="click", coordinate=(10, 20))
                return types.SimpleNamespace(
                    output_text="output",
                    metadata=metadata,
                    parsed_output=types.SimpleNamespace(canonical_action=action),
                )

            def shared_prefix_parent_pair_diagnostics(
                self,
                messages: object,
                actions: object,
            ) -> dict[str, object]:
                self.shared_calls += 1
                return {
                    "first_divergent_token_index": 1,
                    "teacher_token_ids": [[1, 2], [1, 3]],
                }

            def full_parent_action_diagnostic(
                self,
                messages: object,
                action: object,
                *,
                divergence_index: int,
            ) -> dict[str, object]:
                self.full_calls.append(divergence_index)
                return {"divergence_index": divergence_index}

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            layout = AttemptLayout(
                root=parent / "attempt",
                global_ledger=parent / ".attempt.json",
                profiles_root=parent / "attempt/profiles",
                attempt_start=parent / "attempt/start.json",
            )
            profile = types.SimpleNamespace(
                profile_id=PROFILE_ORDER[0],
                state_indices=(1,),
                generation_repeat_count=2,
                shared_prefix_forward_repeat_count=2,
                full_parent_action_forward_count=2,
            )
            ledger = _ensure_attempt_layout(
                layout=layout,
                profile_id=profile.profile_id,
                source_git_commit="c" * 40,
                config_sha256="d" * 64,
                host=_host(),
                execution_inputs={"repository_root": "/repo"},
                invocation_argv=["runner"],
            )
            profile_layout = _begin_profile(
                layout=layout,
                ledger=ledger,
                profile=profile,
                invocation_argv=["runner"],
            )
            state = types.SimpleNamespace(
                index=1,
                role="v2_label_train",
                state_id="state-1",
                candidate_event_step_ids=(1, 2),
            )
            mismatch = types.SimpleNamespace(
                actions=(
                    {"action": "click", "coordinate": [10, 20]},
                    {"action": "click", "coordinate": [11, 20]},
                ),
                state_member_sha256="e" * 64,
                output_texts=("left", "right"),
                generated_token_ids_sha256=("f" * 64, "0" * 64),
            )
            runtime = Runtime()
            with mock.patch(
                "scripts.run_spatial_reference_audit_v1.build_full45_messages",
                return_value=[],
            ), mock.patch(
                "scripts.run_spatial_reference_audit_v1.retokenize_parent_mismatch",
                return_value={
                    "first_divergent_token_index": 1,
                    "token_ids": [[1, 2], [1, 3]],
                },
            ), mock.patch(
                "scripts.run_spatial_reference_audit_v1.coordinate_delta",
                return_value={"normalized_delta": {}},
            ):
                record = _run_state(
                    runtime=runtime,
                    artifact=object(),
                    state=state,
                    mismatch=mismatch,
                    profile=profile,
                    attempt_layout=layout,
                    profile_layout=profile_layout,
                    state_layout=_state_layout(
                        profile_layout,
                        ordinal=0,
                        state_index=1,
                    ),
                )
            self.assertEqual(runtime.generation_calls, 2)
            self.assertEqual(runtime.shared_calls, 2)
            self.assertEqual(runtime.full_calls, [1, 1])
            self.assertEqual(record["operation_counts"]["teacher_forwards"], 4)
            self.assertEqual(record["generations"][0]["generated_token_ids"], [1, 2, 3])
            terminal = _load_json_object(
                profile_layout.states_root / "000-001/terminal.json"
            )
            self.assertEqual(
                terminal["status"],
                "COMPLETED_SPATIAL_REFERENCE_AUDIT_STATE",
            )


if __name__ == "__main__":
    unittest.main()
