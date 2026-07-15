from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from causalcache.data.restoration_v2_screening import ScreeningState
from causalcache.policy.gui_owl_v2_1 import parse_gui_owl_v2_1_output
from scripts.run_restoration_v2_1_interface_pilot import (
    AGGREGATE_FILENAME,
    ATTEMPT_DIRECTORY,
    CANONICAL_PILOT_CONTAINER_ID,
    CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
    CANONICAL_PILOT_DEVICE,
    CANONICAL_PILOT_HOST_ALIAS,
    CANONICAL_PILOT_HOST_HOSTNAME,
    INVALID_OUTCOME,
    NEGATIVE_OPERATION_COUNTS,
    NO_GO_OUTCOME,
    PASS_OUTCOME,
    AuthorizedPilot,
    PilotGate,
    RuntimeBindings,
    RUNTIME_METADATA_KEYS,
    _canonical_repository_path,
    _validate_absolute_production_paths,
    _validate_runtime_cli_identity,
    _pilot_projection,
    aggregate_pilot_gate,
    build_confirm_safe_full_history_messages,
    claim_interface_pilot,
    execute_production_pilot,
    run_interface_pilot,
    select_exact_development_pilot_states,
    validate_canonical_attempt_identity,
    validate_clean_pushed_main,
    validate_committed_source_blobs,
)


VALID_OUTPUT = (
    '<tool_call>\n{"name":"mobile_use","arguments":{"action":"wait"}}\n'
    "</tool_call>"
)


class _Image:
    size = (1080, 1920)


class _ParseError(ValueError):
    def __init__(self, output_text: str, metadata: dict[str, Any]) -> None:
        super().__init__("strict parse failed")
        self.output_text = output_text
        self.metadata = metadata
        self.parse_error_type = "ValueError"
        self.parse_error_message = "whole output mismatch"


def _metadata(output: str, *, sentinel: str = "preserved") -> dict[str, Any]:
    return {
        "protocol_id": "causalcache_restoration_v2_1_official_tool_interface",
        "generation_interface": "processor_apply_chat_template_official_tools_kwarg",
        "official_tools_argument_count": 1,
        "official_tool_schema_sha256": "1" * 64,
        "chat_template_file_sha256": "2" * 64,
        "chat_template_text_sha256": "3" * 64,
        "assistant_prefix_token_ids": [151644, 77091, 198],
        "tool_call_open_token_id": 151657,
        "tool_call_close_token_id": 151658,
        "generation_eos_token_id": 151658,
        "suppressed_standard_eos_token_ids": [151645, 151643],
        "generation_pad_token_id": 151643,
        "generation_num_beams": 1,
        "generation_num_return_sequences": 1,
        "generation_standard_eos_suppression": (
            "negative_infinity_via_transformers_suppress_tokens"
        ),
        "host_injected_tool_call_closer": False,
        "output_recovery_or_normalization": False,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "return_dict_in_generate": False,
        "max_new_tokens": 256,
        "decoded_output_utf8_sha256": hashlib.sha256(
            output.encode("utf-8")
        ).hexdigest(),
        "generated_tool_call_close_token_count": output.count("</tool_call>"),
        "final_generated_token_id": 151658 if output.endswith("</tool_call>") else None,
        "termination_reason": (
            "model_emitted_tool_call_close"
            if output.endswith("</tool_call>")
            else "max_new_tokens_without_tool_call_close"
        ),
        "model_emitted_tool_call_close": output.endswith("</tool_call>"),
        "test_sentinel": {"value": sentinel, "items": [1, 2, 3]},
    }


def _runtime_metadata(model_dir: str) -> dict[str, Any]:
    interface = _metadata(VALID_OUTPUT)
    metadata = {
        key: interface[key]
        for key in RUNTIME_METADATA_KEYS
        if key in interface
    }
    metadata.update(
        {
            "model_dir": str(Path(model_dir).resolve()),
            "model_repo": "repo",
            "model_revision": "revision",
            "snapshot_manifest_sha256": None,
            "verified_model_file_count": 10,
            "verified_model_total_bytes": 1024,
            "transformers_version": "test",
            "transformers_source_sha256": {"modeling": "4" * 64},
            "dtype": "bfloat16",
            "device": CANONICAL_PILOT_DEVICE,
            "frozen": True,
            "single_device": True,
            "processor_class": "TestProcessor",
            "model_class": "TestModel",
            "torch_version": "test",
            "target_effective_visual_tokens_per_image": 4096,
            "min_pixels": 4096 * 28 * 28,
            "max_pixels": 4096 * 28 * 28,
        }
    )
    if set(metadata) != RUNTIME_METADATA_KEYS:
        raise AssertionError("test runtime metadata fixture drifted")
    return metadata


def _states() -> tuple[ScreeningState, ...]:
    result = []
    for pilot_index in range(15):
        step = 4 + pilot_index % 3
        result.append(
            ScreeningState(
                index=30 + pilot_index,
                role="v2_development",
                trajectory_id=f"trajectory-{pilot_index // 3}",
                decision_step_id=step,
                candidate_event_step_ids=tuple(range(1, step - 1)),
            )
        )
    return tuple(result)


def _attempt_identity(root: Path) -> dict[str, Any]:
    attempt_id = "restoration-v2-1-interface-pilot-v1"
    return {
        "attempt_id": attempt_id,
        "canonical_persistent_output_dir": str(root.resolve()),
        "global_attempt_ledger": str(
            root.resolve().parent / f".{attempt_id}.attempt.json"
        ),
        "canonical_host_alias": CANONICAL_PILOT_HOST_ALIAS,
        "canonical_host_hostname": CANONICAL_PILOT_HOST_HOSTNAME,
        "canonical_container_id": CANONICAL_PILOT_CONTAINER_ID,
        "canonical_container_image_digest": CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
        "canonical_device": CANONICAL_PILOT_DEVICE,
        "cross_host_attempt_allowed": False,
        "alternate_output_dir_allowed": False,
        "output_directory_deletion_after_first_attempt_allowed": False,
    }


def _gate_mapping() -> dict[str, Any]:
    return {
        "fixed_state_denominator": 15,
        "required_exact_whole_output_parse_count": 15,
        "required_model_emitted_closer_count": 15,
        "required_androidworld_bridge_count": 15,
        "maximum_max_token_truncation_count": 0,
        "maximum_surrounding_prose_count": 0,
        "maximum_action_line_count": 0,
        "maximum_observation_count": 0,
        "maximum_second_json_count": 0,
        "maximum_second_tool_call_count": 0,
        "maximum_retry_count": 0,
        "maximum_top_up_count": 0,
        "pass_outcome": PASS_OUTCOME,
        "fail_outcome": NO_GO_OUTCOME,
        "invalid_outcome": INVALID_OUTCOME,
    }


def _run_contract(states: tuple[ScreeningState, ...]) -> dict[str, Any]:
    return {
        "states": [
            _pilot_projection(state, index) for index, state in enumerate(states)
        ],
        "negative_operation_counts": dict(NEGATIVE_OPERATION_COUNTS),
        "test_identity": "fixed-run",
    }


def _message_builder(artifact: Any, state: ScreeningState, decoder: Any) -> list[dict[str, Any]]:
    del artifact, state, decoder
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": _Image()},
                {"type": "text", "text": "fixed"},
            ],
        }
    ]


class _Runtime:
    def __init__(
        self,
        *,
        fail_index: int | None = None,
        runtime_error_index: int | None = None,
    ) -> None:
        self.calls = 0
        self.fail_index = fail_index
        self.runtime_error_index = runtime_error_index

    def generate_native_action(self, messages: Any) -> Any:
        del messages
        index = self.calls
        self.calls += 1
        if index == self.runtime_error_index:
            raise RuntimeError("CUDA runtime contract failed")
        if index == self.fail_index:
            raw = "Action: wait\n" + VALID_OUTPUT
            raise _ParseError(raw, _metadata(raw, sentinel=f"failure-{index}"))
        parsed = parse_gui_owl_v2_1_output(VALID_OUTPUT)
        return SimpleNamespace(
            output_text=VALID_OUTPUT,
            parsed_output=parsed,
            metadata=_metadata(VALID_OUTPUT, sentinel=f"state-{index}"),
        )


class InterfacePilotTest(unittest.TestCase):
    def test_production_paths_must_be_absolute_before_claim(self) -> None:
        values = {
            "repository_root": "/data/CausalCache",
            "contract": "/data/CausalCache/code/configs/pilot.json",
            "processor_preflight": "/data/artifacts/processor.json",
            "derived_artifact_root": "/data/artifacts/derived",
            "scientific_config": "/data/CausalCache/code/configs/scientific.json",
            "selection_manifest": "/data/CausalCache/data/selection.json",
            "ocr_backend_config": "/data/CausalCache/code/configs/ocr.json",
            "model_dir": "/data/artifacts/model",
            "output_dir": "/data/experiments/pilot",
        }
        _validate_absolute_production_paths(SimpleNamespace(**values))
        values["model_dir"] = "relative/model"
        with self.assertRaisesRegex(ValueError, "--model-dir"):
            _validate_absolute_production_paths(SimpleNamespace(**values))

    def test_internal_git_paths_are_joined_to_absolute_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            relative = "data/internal.json"
            path = root / relative
            path.parent.mkdir()
            path.write_text("{}")
            self.assertEqual(
                _canonical_repository_path(
                    root / relative,
                    repository_root=root,
                    relative_path=relative,
                    name="internal manifest",
                ),
                path,
            )
            with self.assertRaisesRegex(ValueError, "absolute path"):
                _canonical_repository_path(
                    relative,
                    repository_root=root,
                    relative_path=relative,
                    name="internal manifest",
                )

    def test_canonical_attempt_rejects_a_different_host(self) -> None:
        args = SimpleNamespace(
            host_alias="hyper01",
            host_hostname="node-radixark-16-0001",
            container_id=CANONICAL_PILOT_CONTAINER_ID,
            container_image_digest=CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
            device=CANONICAL_PILOT_DEVICE,
        )
        with patch(
            "scripts.run_restoration_v2_1_interface_pilot.socket.gethostname",
            return_value=CANONICAL_PILOT_CONTAINER_ID[:12],
        ), self.assertRaisesRegex(ValueError, "Hyper00"):
            _validate_runtime_cli_identity(
                args,
                observed={
                    "selected_device": "cuda:0",
                    "visible_cuda_device_count": 1,
                },
            )

    def test_canonical_attempt_rejects_different_container_image_or_device(self) -> None:
        base = {
            "host_alias": CANONICAL_PILOT_HOST_ALIAS,
            "host_hostname": CANONICAL_PILOT_HOST_HOSTNAME,
            "container_id": CANONICAL_PILOT_CONTAINER_ID,
            "container_image_digest": CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
            "device": CANONICAL_PILOT_DEVICE,
        }
        observed = {
            "selected_device": CANONICAL_PILOT_DEVICE,
            "visible_cuda_device_count": 1,
        }
        for field, value in (
            ("container_id", "a" * 64),
            ("container_image_digest", "sha256:" + "b" * 64),
            ("device", "cuda:1"),
        ):
            with self.subTest(field=field), patch(
                "scripts.run_restoration_v2_1_interface_pilot.socket.gethostname",
                return_value=CANONICAL_PILOT_CONTAINER_ID[:12],
            ), self.assertRaisesRegex(ValueError, "exact Hyper00"):
                values = dict(base)
                values[field] = value
                _validate_runtime_cli_identity(
                    SimpleNamespace(**values),
                    observed=observed,
                )

    def test_canonical_attempt_rejects_alternate_output_directory(self) -> None:
        contract = SimpleNamespace(
            data={
                "pilot_execution": {
                    "attempt_id": "restoration-v2-1-interface-pilot-v1",
                    "canonical_persistent_output_dir": (
                        "/data/experiments/causalcache/"
                        "restoration-v2-1-interface-pilot-v1"
                    ),
                    "canonical_host_alias": CANONICAL_PILOT_HOST_ALIAS,
                    "canonical_host_hostname": CANONICAL_PILOT_HOST_HOSTNAME,
                    "canonical_container_id": CANONICAL_PILOT_CONTAINER_ID,
                    "canonical_container_image_digest": (
                        CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST
                    ),
                    "canonical_device": CANONICAL_PILOT_DEVICE,
                    "cross_host_attempt_allowed": False,
                    "alternate_output_dir_allowed": False,
                    "output_directory_deletion_after_first_attempt_allowed": False,
                }
            }
        )
        identity = validate_canonical_attempt_identity(
            contract,
            "/data/experiments/causalcache/restoration-v2-1-interface-pilot-v1",
        )
        self.assertEqual(
            identity["attempt_id"],
            "restoration-v2-1-interface-pilot-v1",
        )
        with self.assertRaisesRegex(ValueError, "alternate"):
            validate_canonical_attempt_identity(contract, "/data/tmp/another-run")

    def test_only_exact_development_indices_thirty_through_forty_four_are_selected(self) -> None:
        development = _states()
        label_states = tuple(
            ScreeningState(
                index=index,
                role="v2_label_train",
                trajectory_id=f"label-{index // 3}",
                decision_step_id=4 + index % 3,
                candidate_event_step_ids=tuple(range(1, (4 + index % 3) - 1)),
            )
            for index in range(30)
        )
        artifact = SimpleNamespace(states=(*label_states, *development))
        projections = [
            _pilot_projection(state, index)
            for index, state in enumerate(development)
        ]
        contract = SimpleNamespace(
            data={
                "data": {
                    "pilot_state_ids": [state.state_id for state in development],
                    "pilot_state_projection_sha256": hashlib.sha256(
                        json.dumps(
                            projections,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            }
        )
        selected = select_exact_development_pilot_states(artifact, contract)
        self.assertEqual([state.index for state in selected], list(range(30, 45)))
        self.assertTrue(all(state.role == "v2_development" for state in selected))

        wrong = list(artifact.states)
        wrong[44] = ScreeningState(
            index=44,
            role="v2_label_train",
            trajectory_id=wrong[44].trajectory_id,
            decision_step_id=wrong[44].decision_step_id,
            candidate_event_step_ids=wrong[44].candidate_event_step_ids,
        )
        with self.assertRaisesRegex(PermissionError, "v2_development"):
            select_exact_development_pilot_states(
                SimpleNamespace(states=tuple(wrong)),
                contract,
            )

    def test_clean_pushed_main_accepts_git_sha1_and_pins_origin(self) -> None:
        commit = "a" * 40
        responses = {
            ("rev-parse", "--show-toplevel"): "/repo\n",
            ("branch", "--show-current"): "main\n",
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("rev-parse", "HEAD"): commit + "\n",
            ("rev-parse", "refs/remotes/origin/main"): commit + "\n",
            ("ls-remote", "--heads", "origin", "refs/heads/main"): (
                commit + "\trefs/heads/main\n"
            ),
            ("remote", "get-url", "origin"): (
                "https://github.com/luojiaxuan/CausalCache.git\n"
            ),
        }

        def fake_git(root: Path, arguments: Any, *, text: bool = True) -> str:
            del root, text
            return responses[tuple(arguments)]

        with patch(
            "scripts.run_restoration_v2_1_interface_pilot._run_git",
            side_effect=fake_git,
        ):
            identity = validate_clean_pushed_main("/repo")
        self.assertEqual(identity["commit"], commit)

        responses[("remote", "get-url", "origin")] = "git@example.invalid:repo.git\n"
        with patch(
            "scripts.run_restoration_v2_1_interface_pilot._run_git",
            side_effect=fake_git,
        ), self.assertRaisesRegex(ValueError, "canonical repository"):
            validate_clean_pushed_main("/repo")

    def test_live_source_must_equal_the_committed_git_blob(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            relative = "code/source.py"
            path = root / relative
            path.parent.mkdir()
            committed = b"VALUE = 1\n"
            path.write_bytes(committed)

            def fake_git(
                repository_root: Path,
                arguments: Any,
                *,
                text: bool = True,
            ) -> Any:
                self.assertEqual(repository_root, root)
                self.assertEqual(arguments, ("show", "a" * 40 + ":" + relative))
                self.assertIs(text, False)
                return committed

            with patch(
                "scripts.run_restoration_v2_1_interface_pilot._run_git",
                side_effect=fake_git,
            ):
                inventory = validate_committed_source_blobs(
                    repository_root=root,
                    git_commit="a" * 40,
                    paths=(relative,),
                )
                self.assertEqual(inventory[0]["sha256"], hashlib.sha256(committed).hexdigest())
                path.write_text("VALUE = 2\n")
                with self.assertRaisesRegex(ValueError, "committed Git blob"):
                    validate_committed_source_blobs(
                        repository_root=root,
                        git_commit="a" * 40,
                        paths=(relative,),
                    )

    def test_generation_reuses_exact_processor_audit_reference_builder(self) -> None:
        state = _states()[0]
        artifact = SimpleNamespace(states=_states())
        expected_messages = [{"role": "audited"}]

        def audited_builder(artifact_value: Any, spec: Any, *, image_decoder: Any) -> Any:
            self.assertIs(artifact_value, artifact)
            self.assertEqual(spec.prompt_index, 60)
            self.assertEqual(spec.fidelity, "reference")
            self.assertEqual(spec.state, state)
            self.assertEqual(spec.restored_event_step_ids, state.candidate_event_step_ids)
            self.assertEqual(image_decoder, "decoder")
            return expected_messages

        with patch(
            "scripts.run_restoration_v2_1_interface_pilot.build_v2_1_processor_messages",
            side_effect=audited_builder,
        ):
            observed = build_confirm_safe_full_history_messages(
                artifact,
                state,
                "decoder",
            )
        self.assertIs(observed, expected_messages)

    def _run(
        self,
        root: Path,
        runtime: _Runtime,
        *,
        resume: bool = False,
        integrity_guard: Any = None,
        message_builder: Any = _message_builder,
        run_contract: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        states = _states()
        return run_interface_pilot(
            artifact=SimpleNamespace(),
            states=states,
            runtime=runtime,
            parse_error_class=_ParseError,
            image_decoder=lambda payload: payload,
            gate=PilotGate.from_mapping(_gate_mapping()),
            run_contract=(run_contract or _run_contract(states)),
            output_dir=root,
            resume=resume,
            integrity_guard=(integrity_guard or (lambda: None)),
            message_builder=message_builder,
        )

    def test_exactly_fifteen_generation_calls_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = _Runtime()
            aggregate = self._run(Path(directory) / "pilot", runtime)
            self.assertEqual(runtime.calls, 15)
            self.assertEqual(aggregate["outcome"], PASS_OUTCOME)
            self.assertEqual(aggregate["metrics"]["generation_call_count"], 15)
            self.assertEqual(aggregate["metrics"]["exact_whole_output_parse_count"], 15)

    def test_one_scientific_failure_is_no_go_and_does_not_shrink_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = _Runtime(fail_index=4)
            aggregate = self._run(Path(directory) / "pilot", runtime)
            self.assertEqual(runtime.calls, 15)
            self.assertEqual(aggregate["outcome"], NO_GO_OUTCOME)
            self.assertEqual(aggregate["metrics"]["fixed_state_denominator"], 15)
            self.assertEqual(aggregate["metrics"]["exact_whole_output_parse_count"], 14)
            self.assertEqual(aggregate["metrics"]["action_line_count"], 1)
            self.assertEqual(aggregate["metrics"]["surrounding_prose_count"], 1)

    def test_runtime_failure_is_invalid_and_interrupted_attempt_cannot_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            runtime = _Runtime(runtime_error_index=2)
            aggregate = self._run(root, runtime)
            self.assertEqual(runtime.calls, 3)
            self.assertEqual(aggregate["outcome"], INVALID_OUTCOME)

            read_only_runtime = _Runtime()
            resumed = self._run(root, read_only_runtime, resume=True)
            self.assertEqual(resumed, aggregate)
            self.assertEqual(read_only_runtime.calls, 0)

            extra_state = root / "states" / "999.json"
            extra_state.write_text("{}")
            with self.assertRaisesRegex(ValueError, "directory inventory"):
                self._run(root, read_only_runtime, resume=True)
            extra_state.unlink()

            (root / AGGREGATE_FILENAME).unlink()
            recovered = self._run(root, read_only_runtime, resume=True)
            self.assertEqual(recovered["outcome"], INVALID_OUTCOME)
            self.assertEqual(
                recovered["invalid_failure"]["stage"],
                "resume_recovered_nonterminal_no_retry_claim",
            )
            self.assertEqual(read_only_runtime.calls, 0)
            self.assertTrue((root / ATTEMPT_DIRECTORY / "002.json").is_file())

    def test_invalid_image_dimensions_fail_before_generation(self) -> None:
        def invalid_messages(artifact: Any, state: Any, decoder: Any) -> Any:
            del artifact, state, decoder
            return [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": object()}],
                }
            ]

        with tempfile.TemporaryDirectory() as directory:
            runtime = _Runtime()
            aggregate = self._run(
                Path(directory) / "pilot",
                runtime,
                message_builder=invalid_messages,
            )
            self.assertEqual(aggregate["outcome"], INVALID_OUTCOME)
            self.assertEqual(runtime.calls, 0)
            self.assertEqual(
                aggregate["invalid_failure"]["category"],
                "CONTRACT_OR_RUNTIME",
            )

    def test_post_generation_integrity_drift_is_invalid_before_gate_write(self) -> None:
        guard_calls = 0

        def guard() -> None:
            nonlocal guard_calls
            guard_calls += 1
            if guard_calls == 31:
                raise ValueError("Git source drifted")

        with tempfile.TemporaryDirectory() as directory:
            runtime = _Runtime()
            aggregate = self._run(
                Path(directory) / "pilot",
                runtime,
                integrity_guard=guard,
            )
            self.assertEqual(runtime.calls, 15)
            self.assertEqual(guard_calls, 31)
            self.assertEqual(aggregate["outcome"], INVALID_OUTCOME)
            self.assertEqual(
                aggregate["invalid_failure"]["stage"],
                "post_generation_integrity",
            )

    def test_raw_generation_metadata_is_preserved_in_terminal_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            self._run(root, _Runtime())
            record = json.loads((root / "states" / "007.json").read_bytes())
            self.assertEqual(
                record["generation_metadata"]["test_sentinel"],
                {"value": "state-7", "items": [1, 2, 3]},
            )
            self.assertEqual(record["raw_output"], VALID_OUTPUT)

    def test_output_directory_and_terminal_state_files_are_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            aggregate = self._run(root, _Runtime())
            self.assertIsInstance(aggregate["started_at_utc"], str)
            self.assertIsInstance(aggregate["ended_at_utc"], str)
            self.assertGreaterEqual(aggregate["duration_seconds"], 0)
            with self.assertRaises(FileExistsError):
                self._run(root, _Runtime(), resume=False)
            resumed_runtime = _Runtime()
            aggregate = self._run(root, resumed_runtime, resume=True)
            self.assertEqual(aggregate["outcome"], PASS_OUTCOME)
            self.assertEqual(resumed_runtime.calls, 0)

    def test_completed_resume_requires_exact_inventory_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            self._run(root, _Runtime())
            extra = root / "states" / "999.json"
            extra.write_text("{}")
            with self.assertRaisesRegex(ValueError, "directory inventory"):
                self._run(root, _Runtime(), resume=True)
            extra.unlink()

            manifest_path = root / "run_manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["forged"] = True
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "run manifest identity"):
                self._run(root, _Runtime(), resume=True)

    def test_completed_resume_recomputes_exact_aggregate_schema_and_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            self._run(root, _Runtime())
            aggregate_path = root / AGGREGATE_FILENAME
            aggregate = json.loads(aggregate_path.read_bytes())
            aggregate["duration_seconds"] += 1
            aggregate["forged"] = True
            aggregate_path.write_text(json.dumps(aggregate))
            with self.assertRaisesRegex(ValueError, "exact raw denominator"):
                self._run(root, _Runtime(), resume=True)

    def test_global_attempt_ledger_survives_output_deletion_and_blocks_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root = parent / "canonical-pilot"
            states = _states()
            contract = _run_contract(states)
            contract["attempt_identity"] = _attempt_identity(root)
            ledger = Path(contract["attempt_identity"]["global_attempt_ledger"])
            first_runtime = _Runtime()
            first = self._run(
                root,
                first_runtime,
                run_contract=contract,
            )
            self.assertEqual(first["outcome"], PASS_OUTCOME)
            self.assertEqual(first_runtime.calls, 15)
            self.assertTrue(ledger.is_file())
            shutil.rmtree(root)

            second_runtime = _Runtime()
            with self.assertRaises(FileExistsError):
                self._run(
                    root,
                    second_runtime,
                    run_contract=contract,
                )
            self.assertEqual(second_runtime.calls, 0)

    def test_ledger_only_bootstrap_resume_becomes_packagable_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root = parent / "canonical-pilot"
            states = _states()
            contract = _run_contract(states)
            contract["attempt_identity"] = _attempt_identity(root)
            gate = PilotGate.from_mapping(_gate_mapping())
            claimed = claim_interface_pilot(
                states=states,
                gate=gate,
                run_contract=contract,
                output_dir=root,
                resume=False,
            )
            self.assertIsNone(claimed)
            ledger = Path(contract["attempt_identity"]["global_attempt_ledger"])
            self.assertTrue(ledger.is_file())
            shutil.rmtree(root)

            runtime = _Runtime()
            recovered = self._run(
                root,
                runtime,
                resume=True,
                run_contract=contract,
            )
            self.assertEqual(recovered["outcome"], INVALID_OUTCOME)
            self.assertEqual(recovered["completed_state_count"], 0)
            self.assertEqual(recovered["attempted_state_count"], 0)
            self.assertEqual(runtime.calls, 0)
            self.assertTrue((root / "run_manifest.json").is_file())
            self.assertTrue((root / AGGREGATE_FILENAME).is_file())

    def test_reducer_rejects_forged_top_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            self._run(root, _Runtime())
            states = _states()
            records = [
                json.loads((root / "states" / f"{index:03d}.json").read_bytes())
                for index in range(15)
            ]
            records[0]["top_up_count"] = 1
            contract = _run_contract(states)
            run_sha = hashlib.sha256(
                json.dumps(
                    contract,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "retry, top-up"):
                aggregate_pilot_gate(
                    records,
                    expected_states=states,
                    gate=PilotGate.from_mapping(_gate_mapping()),
                    run_contract_sha256=run_sha,
                    started_at_utc=records[0]["started_at_utc"],
                    ended_at_utc=records[-1]["ended_at_utc"],
                )

    def test_authorization_finishes_before_runtime_loader_or_constructor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events: list[str] = []
            states = _states()
            output_root = Path(directory).resolve() / "pilot"
            contract = SimpleNamespace(
                source_sha256="a" * 64,
                data={
                    "primary_policy": {"repo": "repo", "revision": "revision"},
                    "data": {"derived_artifact": {"repo": "dataset"}},
                    "pilot_gate": _gate_mapping(),
                },
            )
            artifact = SimpleNamespace(
                artifact_tree_sha256="b" * 64,
                artifact_manifest_sha256="c" * 64,
                screening_manifest_sha256="d" * 64,
            )
            authorized = AuthorizedPilot(
                repository_root=Path("/repo"),
                contract=contract,
                processor_audit={"valid": True},
                git_identity={"commit": "e" * 40},
                source_inventory=(),
                artifact=artifact,
                states=states,
                snapshot_manifest_path=Path("snapshot.json"),
                target_effective_visual_tokens_per_image=4096,
                runtime_identity={"selected_device": CANONICAL_PILOT_DEVICE},
                canonical_inputs={},
                attempt_identity=_attempt_identity(output_root),
            )

            def authorize(args: Any) -> AuthorizedPilot:
                del args
                events.append("authorized")
                return authorized

            def revalidate(value: AuthorizedPilot) -> None:
                self.assertIs(value, authorized)
                events.append("integrity")

            class Runtime:
                def __init__(self, **kwargs: Any) -> None:
                    del kwargs
                    self.assert_claim_exists()
                    events.append("constructor")
                    raise RuntimeError("CUDA out of memory during model load")

                @staticmethod
                def assert_claim_exists() -> None:
                    self.assertTrue(output_root.is_dir())
                    self.assertTrue((output_root / "run_manifest.json").is_file())
                    self.assertTrue(
                        Path(
                            authorized.attempt_identity["global_attempt_ledger"]
                        ).is_file()
                    )

            def load_runtime() -> RuntimeBindings:
                self.assertTrue((output_root / "run_manifest.json").is_file())
                events.append("runtime_imported")
                return RuntimeBindings(runtime_class=Runtime, parse_error_class=_ParseError)

            args = SimpleNamespace(
                model_dir=str(Path(directory) / "model"),
                device=CANONICAL_PILOT_DEVICE,
                output_dir=str(output_root),
                resume=False,
                execution_argv=["runner", "--fixed"],
            )
            result = execute_production_pilot(
                args,
                authorization_loader=authorize,
                integrity_revalidator=revalidate,
                runtime_bindings_loader=load_runtime,
            )
            self.assertEqual(result["outcome"], INVALID_OUTCOME)
            self.assertEqual(result["invalid_failure"]["category"], "OUT_OF_MEMORY")
            self.assertEqual(result["completed_state_count"], 0)
            self.assertEqual(result["attempted_state_count"], 0)
            self.assertFalse((output_root / "runtime_identity.json").exists())
            self.assertEqual(
                events,
                ["authorized", "integrity", "integrity", "runtime_imported", "constructor"],
            )

            args.resume = True
            args.execution_argv = ["runner", "--fixed", "--resume"]
            resumed = execute_production_pilot(
                args,
                authorization_loader=authorize,
                integrity_revalidator=revalidate,
                runtime_bindings_loader=lambda: self.fail(
                    "resume must not import the policy runtime"
                ),
            )
            self.assertEqual(resumed, result)

    def test_production_path_executes_real_runner_with_exact_attempt_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "pilot"
            states = _states()
            contract = SimpleNamespace(
                source_sha256="a" * 64,
                data={
                    "primary_policy": {"repo": "repo", "revision": "revision"},
                    "data": {"derived_artifact": {"repo": "dataset"}},
                    "pilot_gate": _gate_mapping(),
                },
            )
            artifact = SimpleNamespace(
                artifact_tree_sha256="b" * 64,
                artifact_manifest_sha256="c" * 64,
                screening_manifest_sha256="d" * 64,
            )
            authorized = AuthorizedPilot(
                repository_root=Path("/repo"),
                contract=contract,
                processor_audit={"valid": True},
                git_identity={"commit": "e" * 40},
                source_inventory=(),
                artifact=artifact,
                states=states,
                snapshot_manifest_path=Path("snapshot.json"),
                target_effective_visual_tokens_per_image=4096,
                runtime_identity={"selected_device": CANONICAL_PILOT_DEVICE},
                canonical_inputs={},
                attempt_identity=_attempt_identity(root),
            )
            instances: list[_Runtime] = []

            class Runtime(_Runtime):
                def __init__(self, **kwargs: Any) -> None:
                    model_dir = kwargs["model_dir"]
                    super().__init__()
                    self.metadata = _runtime_metadata(model_dir)
                    instances.append(self)

            def real_runner(**kwargs: Any) -> dict[str, Any]:
                kwargs["message_builder"] = _message_builder
                return run_interface_pilot(**kwargs)

            args = SimpleNamespace(
                model_dir=str(Path(directory) / "model"),
                device=CANONICAL_PILOT_DEVICE,
                output_dir=str(root),
                resume=False,
                execution_argv=["runner", "--fixed"],
            )
            aggregate = execute_production_pilot(
                args,
                authorization_loader=lambda _: authorized,
                integrity_revalidator=lambda _: None,
                runtime_bindings_loader=lambda: RuntimeBindings(
                    runtime_class=Runtime,
                    parse_error_class=_ParseError,
                ),
                run_executor=real_runner,
            )
            self.assertEqual(aggregate["outcome"], PASS_OUTCOME)
            self.assertEqual(instances[0].calls, 15)
            runtime_identity = json.loads(
                (root / "runtime_identity.json").read_bytes()
            )
            self.assertEqual(
                set(runtime_identity["runtime_metadata"]),
                RUNTIME_METADATA_KEYS,
            )
            manifest = json.loads((root / "run_manifest.json").read_bytes())
            self.assertEqual(
                manifest["run_contract"]["attempt_identity"],
                _attempt_identity(root),
            )
            args.resume = True
            args.execution_argv = ["runner", "--fixed", "--resume"]
            resumed = execute_production_pilot(
                args,
                authorization_loader=lambda _: authorized,
                integrity_revalidator=lambda _: None,
                runtime_bindings_loader=lambda: self.fail(
                    "completed resume must not import policy runtime"
                ),
                run_executor=real_runner,
            )
            self.assertEqual(resumed, aggregate)
            self.assertEqual(len(instances), 1)


if __name__ == "__main__":
    unittest.main()
