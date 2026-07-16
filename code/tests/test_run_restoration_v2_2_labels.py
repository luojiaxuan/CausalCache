from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.restoration_v2_2_eager_artifact import expected_worker_specs
from causalcache.restoration_v2_2_label_parent import V22LabelParentState
from causalcache.restoration_v2_2_label_contract import (
    V2_REPAIR_ATTEMPT_ID,
    V2_REPAIR_CONFIG_PATH,
    label_attempt_profile_for_config_path,
)
from scripts.run_restoration_v2_2_labels import (
    ATTEMPT_DIRECTORY,
    GLOBAL_INVALID_STATUS,
    LabelLayout,
    RUN_MANIFEST_FILENAME,
    STATE_DIRECTORY,
    WORKER_DIRECTORY,
    WORKER_LEDGER_FILENAME,
    _layout_from_existing,
    _replace_json_durable,
    _write_json_exclusive,
    claim_attempt,
    execute_attempt,
    execute_claimed_attempt,
    run_label_state_once,
    seal_invalid_attempt,
    validate_model_snapshot_preclaim,
)
from tests.test_restoration_v2_2_label_inputs import _artifact


class _FakeRuntime:
    def __init__(self, event_count: int) -> None:
        self.event_count = event_count
        self.call_count = 0
        self.next_mask = 0

    def teacher_forced_distance_logits(self, messages_batch, actions):
        self.call_count += 1
        if self.call_count == 1:
            marker = ("reference", (1 << self.event_count) - 1)
        elif self.call_count == 2:
            marker = ("repeat", (1 << self.event_count) - 1)
        else:
            marker = ("candidate", self.next_mask)
            self.next_mask += 1
        return marker, {
            "batch_size": 1,
            "samples": [{"image_count": len(messages_batch[0][1]["content"])}],
            "call_index": self.call_count,
        }


class _FakeDistanceBackend:
    def __init__(self, event_count: int, *, repeat_kl: float = 0.0) -> None:
        self.event_count = event_count
        self.repeat_kl = repeat_kl

    def prepare_reference(self, logits):
        self._reference = logits
        return logits

    def measure(self, reference, candidate):
        kind, mask = candidate
        if kind == "repeat":
            value = self.repeat_kl
        else:
            value = float(self.event_count - mask.bit_count())
        return SimpleNamespace(
            value=value,
            audit={"full_tensor_host_transfers": 0, "validation_scalar_host_reads": 0},
        )


class _FakeProcess:
    def __init__(self, *, name: str, exitcode: int) -> None:
        self.name = name
        self.exitcode = exitcode
        self._alive = False

    def start(self) -> None:
        self._alive = True

    def join(self) -> None:
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self._alive = False


class _FakeContext:
    def __init__(self, exitcodes) -> None:
        self.exitcodes = iter(exitcodes)

    def Process(self, *, target, args, name):
        del target, args
        return _FakeProcess(name=name, exitcode=next(self.exitcodes))


def _parent(state) -> V22LabelParentState:
    action = GUIOwlV2Action(action="wait")
    return V22LabelParentState(
        index=state.index,
        role=state.role,
        trajectory_id=state.trajectory_id,
        decision_step_id=state.decision_step_id,
        state_id=state.state_id,
        candidate_event_step_ids=state.candidate_event_step_ids,
        worker_id="even" if state.index % 2 == 0 else "odd",
        member_name=(
            f"workers/{'even' if state.index % 2 == 0 else 'odd'}"
            f"/states/{state.index:03d}.json"
        ),
        member_sha256="a" * 64,
        canonical_action=action,
        canonical_action_sha256="b" * 64,
    )


class RestorationV22LabelsRunnerTest(unittest.TestCase):
    @staticmethod
    def _snapshot_fixture(base: Path, *, directory_name: str = "GUI-Owl-1.5-8B-Instruct"):
        model = base / directory_name
        model.mkdir()
        payload = b"model-bytes"
        (model / "weights.bin").write_bytes(payload)
        manifest = {
            "repo": "mPLUG/GUI-Owl-1.5-8B-Instruct",
            "revision": "a" * 40,
            "files": [
                {
                    "path": "weights.bin",
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ],
        }
        manifest_path = base / "snapshot.json"
        serialized = json.dumps(manifest, sort_keys=True).encode("utf-8")
        manifest_path.write_bytes(serialized)
        (model / ".snapshot.json").write_bytes(serialized)

        def validator(*, model_dir, expected_snapshot_manifest):
            return SimpleNamespace(
                model_dir=str(Path(model_dir).resolve()),
                model_repo=manifest["repo"],
                model_revision=manifest["revision"],
                snapshot_manifest_sha256=hashlib.sha256(
                    Path(expected_snapshot_manifest).read_bytes()
                ).hexdigest(),
                verified_model_file_count=1,
                verified_model_total_bytes=len(payload),
            )

        return model, manifest_path, validator

    @staticmethod
    def _claim(base: Path) -> LabelLayout:
        return claim_attempt(
            run_contract={
                "attempt_identity": {
                    "attempt_id": "restoration-v2-2-eager-labels-v1"
                },
                "worker_topology": {
                    "workers": [spec.to_dict() for spec in expected_worker_specs()]
                },
            },
            output_dir=base / "raw",
            global_ledger=base / "attempt.json",
        )

    def test_state_kernel_builds_full_table_and_primary_labels_without_generation(self) -> None:
        artifact = _artifact()
        state = artifact.states[2]
        runtime = _FakeRuntime(event_count=4)
        record = run_label_state_once(
            artifact=artifact,
            state=state,
            parent=_parent(state),
            runtime=runtime,
            distance_backend=_FakeDistanceBackend(event_count=4),
            run_contract_sha256="c" * 64,
            image_decoder=lambda payload: payload.decode("utf-8"),
        )
        self.assertEqual(record["status"], "VALID_RESTORATION_V2_2_EAGER_LABEL_STATE")
        self.assertEqual(len(record["distance_rows"]), 16)
        self.assertEqual(len(record["deployment_conditional_edges"]), 16)
        self.assertEqual(len(record["full_hypercube_edges"]), 32)
        self.assertEqual(len(record["pair_interactions"]), 24)
        self.assertEqual(len(record["exact_permutation_attribution"]), 4)
        self.assertEqual(
            record["primary_exact_subset_oracle"]["coalition"],
            [1, 2],
        )
        self.assertEqual(record["operation_counts"]["generation_call_count"], 0)
        self.assertEqual(record["operation_counts"]["teacher_forward_call_count"], 17)
        self.assertEqual(record["operation_counts"]["kl_measurement_count"], 16)
        self.assertEqual(runtime.call_count, 17)
        self.assertEqual(record["distance_rows"][-1]["distance_kl"], 0.0)

    def test_state_kernel_rejects_repeat_noise_above_frozen_bound(self) -> None:
        artifact = _artifact()
        state = artifact.states[0]
        with self.assertRaisesRegex(RuntimeError, "noise bound"):
            run_label_state_once(
                artifact=artifact,
                state=state,
                parent=_parent(state),
                runtime=_FakeRuntime(event_count=2),
                distance_backend=_FakeDistanceBackend(
                    event_count=2,
                    repeat_kl=0.001,
                ),
                run_contract_sha256="c" * 64,
                image_decoder=lambda payload: payload.decode("utf-8"),
            )

    def test_global_claim_prebinds_two_workers_and_forbids_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            ledger = base / "attempt.json"
            contract = {
                "attempt_identity": {"attempt_id": "restoration-v2-2-eager-labels-v1"},
                "worker_topology": {
                    "workers": [spec.to_dict() for spec in expected_worker_specs()]
                },
            }
            layout = claim_attempt(
                run_contract=contract,
                output_dir=root,
                global_ledger=ledger,
            )
            self.assertIsInstance(layout, LabelLayout)
            self.assertTrue(ledger.is_file())
            self.assertTrue(all(path.is_file() for path in layout.worker_sibling_ledgers.values()))
            with self.assertRaises(FileExistsError):
                claim_attempt(
                    run_contract=contract,
                    output_dir=root,
                    global_ledger=ledger,
                )

    def test_claim_initialization_failure_is_durably_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            ledger = base / "attempt.json"
            contract = {
                "attempt_identity": {
                    "attempt_id": "restoration-v2-2-eager-labels-v1"
                },
                "worker_topology": {
                    "workers": [spec.to_dict() for spec in expected_worker_specs()]
                },
            }
            real_write = _write_json_exclusive

            def fail_manifest(path, value):
                if Path(path).name == RUN_MANIFEST_FILENAME:
                    raise OSError("injected manifest failure")
                return real_write(path, value)

            with mock.patch(
                "scripts.run_restoration_v2_2_labels._write_json_exclusive",
                side_effect=fail_manifest,
            ), self.assertRaisesRegex(OSError, "manifest failure"):
                claim_attempt(
                    run_contract=contract,
                    output_dir=root,
                    global_ledger=ledger,
                )
            persisted = json.loads(ledger.read_bytes())
            self.assertEqual(persisted["status"], GLOBAL_INVALID_STATUS)
            self.assertEqual(
                persisted["invalid_failure"]["stage"],
                "claim_initialization",
            )
            self.assertFalse(persisted["retry_allowed"])
            self.assertFalse(persisted["resume_allowed"])

    def test_invalid_seal_persists_failure_and_observed_high_water(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = self._claim(Path(directory))
            even = expected_worker_specs()[0]
            sibling = layout.worker_sibling_ledgers[even.worker_id]
            record = json.loads(sibling.read_bytes())
            record.update(
                {
                    "status": "LABEL_WORKER_RUNNING_NO_RETRY",
                    "attempted_state_indices": [0, 2],
                    "completed_state_indices": [0],
                }
            )
            _replace_json_durable(sibling, record)
            worker_root = layout.root / WORKER_DIRECTORY / even.worker_id
            _write_json_exclusive(worker_root / WORKER_LEDGER_FILENAME, record)
            _write_json_exclusive(
                worker_root / ATTEMPT_DIRECTORY / "002.json",
                {"state_index": 2},
            )
            _write_json_exclusive(
                worker_root / STATE_DIRECTORY / "000.json",
                {"state": {"index": 0}},
            )

            terminal = seal_invalid_attempt(
                layout,
                stage="worker_process",
                phase="exit",
                error=RuntimeError("worker failed"),
                process_failures=({"name": "even", "exitcode": 1},),
            )
            persisted = json.loads(layout.global_ledger.read_bytes())
            self.assertEqual(persisted, terminal)
            self.assertEqual(persisted["status"], GLOBAL_INVALID_STATUS)
            self.assertEqual(persisted["attempted_state_indices"], [0, 2])
            self.assertEqual(persisted["completed_state_indices"], [0])
            self.assertEqual(persisted["attempted_state_count"], 2)
            self.assertEqual(persisted["completed_state_count"], 1)
            self.assertFalse(persisted["retry_allowed"])
            self.assertFalse(persisted["resume_allowed"])
            self.assertEqual(
                persisted["worker_high_water"]["even"][
                    "observed_attempt_marker_indices"
                ],
                [2],
            )
            self.assertEqual(
                persisted["invalid_failure"]["worker_process_failures"],
                [{"name": "even", "exitcode": 1}],
            )
            with self.assertRaisesRegex(ValueError, "global claim differ"):
                _layout_from_existing(
                    root=layout.root,
                    global_ledger=layout.global_ledger,
                )

    def test_process_exit_failure_seals_claimed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = self._claim(Path(directory))
            with self.assertRaisesRegex(RuntimeError, "worker process failed"):
                execute_claimed_attempt(
                    layout,
                    SimpleNamespace(execution_argv=[]),
                    context=_FakeContext([1, 0]),
                )
            persisted = json.loads(layout.global_ledger.read_bytes())
            self.assertEqual(persisted["status"], GLOBAL_INVALID_STATUS)
            self.assertEqual(
                persisted["invalid_failure"]["stage"], "worker_process"
            )
            self.assertEqual(persisted["invalid_failure"]["phase"], "exit")
            self.assertEqual(
                persisted["invalid_failure"]["worker_process_failures"],
                [{"name": "causalcache-label-even", "exitcode": 1}],
            )

    def test_aggregate_failure_seals_claimed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = self._claim(Path(directory))

            def fail_aggregate(_: LabelLayout) -> dict:
                raise ValueError("aggregate drift")

            with self.assertRaisesRegex(ValueError, "aggregate drift"):
                execute_claimed_attempt(
                    layout,
                    SimpleNamespace(execution_argv=[]),
                    context=_FakeContext([0, 0]),
                    aggregate_fn=fail_aggregate,
                )
            persisted = json.loads(layout.global_ledger.read_bytes())
            self.assertEqual(persisted["status"], GLOBAL_INVALID_STATUS)
            self.assertEqual(persisted["invalid_failure"]["stage"], "aggregate")
            self.assertEqual(
                persisted["invalid_failure"]["phase"], "validation_and_write"
            )
            self.assertEqual(
                persisted["invalid_failure"]["exception_type"], "ValueError"
            )

    def test_model_snapshot_preclaim_rejects_missing_without_claim_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            ledger = base / "attempt.json"
            profile = replace(
                label_attempt_profile_for_config_path(V2_REPAIR_CONFIG_PATH),
                canonical_model_dir=base / "missing-model",
            )
            args = SimpleNamespace(
                repository_root=Path(__file__).resolve().parents[2],
                contract=Path("unused.json"),
                snapshot_manifest=(
                    Path(__file__).resolve().parents[2]
                    / "code/configs/gui_owl_1_5_8b_snapshot.json"
                ),
                model_dir=base / "missing-model",
                output_dir=root,
                global_ledger=ledger,
            )
            fake_contract = SimpleNamespace(profile=profile)
            with mock.patch(
                "scripts.run_restoration_v2_2_labels.RestorationV22LabelContract.load",
                return_value=fake_contract,
            ), self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                execute_attempt(args)
            self.assertFalse(root.exists())
            self.assertFalse(ledger.exists())

    def test_model_snapshot_preclaim_rejects_wrong_basename_symlink_and_partial(self) -> None:
        repair = label_attempt_profile_for_config_path(V2_REPAIR_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            wrong, manifest, validator = self._snapshot_fixture(
                base,
                directory_name="wrong-name",
            )
            with self.assertRaisesRegex(ValueError, "basename"):
                validate_model_snapshot_preclaim(
                    model_dir=wrong.resolve(),
                    snapshot_manifest=manifest,
                    profile=replace(repair, canonical_model_dir=None),
                    snapshot_validator=validator,
                )

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            model, manifest, validator = self._snapshot_fixture(base)
            linked = base / "linked-model"
            linked.symlink_to(model, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                validate_model_snapshot_preclaim(
                    model_dir=linked,
                    snapshot_manifest=manifest,
                    profile=replace(repair, canonical_model_dir=None),
                    snapshot_validator=validator,
                )

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            model, manifest, validator = self._snapshot_fixture(base)
            (model / "weights.bin").write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "missing or partial"):
                validate_model_snapshot_preclaim(
                    model_dir=model.resolve(),
                    snapshot_manifest=manifest,
                    profile=replace(repair, canonical_model_dir=model.resolve()),
                    snapshot_validator=validator,
                )

    def test_v2_claim_and_invalid_terminal_keep_repair_identity(self) -> None:
        profile = label_attempt_profile_for_config_path(V2_REPAIR_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            layout = claim_attempt(
                run_contract={
                    "attempt_identity": {
                        "attempt_id": profile.attempt_id,
                        "attempt_revision": profile.attempt_revision,
                        "supersedes_attempt_id": profile.supersedes_attempt_id,
                        "pass_outcome": profile.pass_outcome,
                        "invalid_outcome": profile.invalid_outcome,
                        "aggregate_status": profile.aggregate_status,
                    }
                },
                output_dir=base / "raw",
                global_ledger=base / "attempt.json",
            )
            manifest = json.loads(
                (layout.root / RUN_MANIFEST_FILENAME).read_bytes()
            )
            self.assertEqual(manifest["attempt_id"], V2_REPAIR_ATTEMPT_ID)
            for sibling in layout.worker_sibling_ledgers.values():
                value = json.loads(sibling.read_bytes())
                self.assertEqual(value["attempt_id"], V2_REPAIR_ATTEMPT_ID)
                self.assertEqual(value["attempt_revision"], "v2_preclaim_repair")
            terminal = seal_invalid_attempt(
                layout,
                stage="test",
                phase="identity",
                error=RuntimeError("injected"),
            )
            self.assertEqual(terminal["attempt_id"], V2_REPAIR_ATTEMPT_ID)
            self.assertEqual(terminal["outcome"], profile.invalid_outcome)


if __name__ == "__main__":
    unittest.main()
