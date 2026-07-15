from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest import mock

from causalcache.data.restoration_v2_1_processor_inputs import (
    iter_v2_1_processor_prompt_specs,
)
from causalcache.data.restoration_v2_screening import (
    ScreeningState,
    ValidatedScreeningArtifact,
)
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    GUI_OWL_V2_1_SYSTEM_PROMPT,
    canonical_json_sha256,
    serialize_gui_owl_v2_1_teacher_target,
)
from causalcache.restoration_v2_1_contract import (
    ASSISTANT_PREFIX_TOKEN_IDS,
    CANONICAL_TOOL_SCHEMA_SHA256,
    CHAT_TEMPLATE_FILE_SHA256,
    CHAT_TEMPLATE_TEXT_SHA256,
    PAD_TOKEN_ID,
    RUNTIME_SOURCE_SHA256,
    STANDARD_EOS_TOKEN_IDS,
    TOOL_CALL_CLOSE_TOKEN_ID,
    TOOL_CALL_OPEN_TOKEN_ID,
)
from causalcache.restoration_v2_1_processor_audit import (
    CANONICAL_EVIDENCE_ARTIFACT_PATH,
    CANONICAL_EVIDENCE_HF_REPO,
    CANONICAL_EVIDENCE_HF_TAG,
    EXPECTED_MAXIMUM_CONTEXT_TOKENS,
    OFFICIAL_TOOL_BLOCK,
    OFFICIAL_TOOL_JSON,
    _validate_processor_only_source,
    build_processor_evidence_artifact_manifest,
    canonical_json_bytes,
    reduce_restoration_v2_1_processor_records,
    require_evidence_commit_ancestor,
    screening_state_projection,
    sha256_file,
    strict_json_object,
    validate_committed_processor_evidence_artifact,
    validate_stored_processor_reduction,
)
from scripts.audit_gui_owl_v2_1_processor import (
    _execution_runtime_identity,
    _require_absolute_paths,
    _validate_explicit_runtime_args,
    parse_args,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SELECTION = REPOSITORY_ROOT / "data/manifests/restoration_v2_selection.json"
V21_CONFIG = REPOSITORY_ROOT / "code/configs/causalcache_restoration_v2_1_pilot.json"
INTERFACE_SOURCE = REPOSITORY_ROOT / "code/causalcache/policy/gui_owl_v2_1.py"
AUDIT_SCRIPT = REPOSITORY_ROOT / "code/scripts/audit_gui_owl_v2_1_processor.py"
ASSISTANT_PREFIX = "<|im_start|>assistant\n"


def _states() -> list[dict[str, object]]:
    return screening_state_projection(strict_json_object(SELECTION, label="selection"))


def _bindings() -> dict[str, object]:
    return {
        "contract_sha256": sha256_file(V21_CONFIG),
        "selection_manifest_sha256": sha256_file(SELECTION),
        "policy_interface_source_sha256": sha256_file(INTERFACE_SOURCE),
        "runtime_source_sha256": RUNTIME_SOURCE_SHA256,
        "canonical_tool_schema_sha256": CANONICAL_TOOL_SCHEMA_SHA256,
        "chat_template_file_sha256": CHAT_TEMPLATE_FILE_SHA256,
        "chat_template_text_sha256": CHAT_TEMPLATE_TEXT_SHA256,
        "assistant_prefix_token_ids": list(ASSISTANT_PREFIX_TOKEN_IDS),
        "tool_call_open_token_id": TOOL_CALL_OPEN_TOKEN_ID,
        "tool_call_close_token_id": TOOL_CALL_CLOSE_TOKEN_ID,
        "standard_eos_token_ids": list(STANDARD_EOS_TOKEN_IDS),
        "pad_token_id": PAD_TOKEN_ID,
        "maximum_context_tokens": EXPECTED_MAXIMUM_CONTEXT_TOKENS,
    }


def _prompt_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for state in _states():
        for fidelity in ("reference", "summary_only"):
            prompt_index = len(records)
            image_count = (
                int(state["decision_step_id"]) - 1 if fidelity == "reference" else 1
            )
            grids = [[1, 2, 2] for _ in range(image_count)]
            input_ids = [100 + prompt_index] * 29 + list(ASSISTANT_PREFIX_TOKEN_IDS)
            inventory = {
                "attention_mask": {
                    "shape": [1, len(input_ids)],
                    "dtype": "torch.int64",
                    "device": "cpu",
                    "requires_grad": False,
                },
                "image_grid_thw": {
                    "shape": [image_count, 3],
                    "dtype": "torch.int64",
                    "device": "cpu",
                    "requires_grad": False,
                },
                "input_ids": {
                    "shape": [1, len(input_ids)],
                    "dtype": "torch.int64",
                    "device": "cpu",
                    "requires_grad": False,
                },
                "pixel_values": {
                    "shape": [4 * image_count, 1536],
                    "dtype": "torch.float32",
                    "device": "cpu",
                    "requires_grad": False,
                },
            }
            records.append(
                {
                    "prompt_index": prompt_index,
                    "state_index": state["index"],
                    "state_id": state["state_id"],
                    "role": state["role"],
                    "trajectory_id": state["trajectory_id"],
                    "decision_step_id": state["decision_step_id"],
                    "candidate_event_step_ids": state["candidate_event_step_ids"],
                    "fidelity": fidelity,
                    "restored_event_step_ids": (
                        state["candidate_event_step_ids"]
                        if fidelity == "reference"
                        else []
                    ),
                    "native_system_text": GUI_OWL_V2_1_SYSTEM_PROMPT,
                    "native_user_text_blocks": [
                        "synthetic record",
                        GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
                    ],
                    "rendered_prompt": (
                        "# Tools\n\nheader <tools></tools>\n"
                        f"{OFFICIAL_TOOL_BLOCK}\nbody\n{ASSISTANT_PREFIX}"
                    ),
                    "rendered_without_tools": f"header\nbody\n{ASSISTANT_PREFIX}",
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "image_grid_thw": grids,
                    "tensor_inventory": inventory,
                }
            )
    return records


def _actions() -> tuple[GUIOwlV2Action, ...]:
    return (
        GUIOwlV2Action(action="click", coordinate=(0, 999)),
        GUIOwlV2Action(action="long_press", coordinate=(500, 500)),
        GUIOwlV2Action(action="swipe", coordinate=(1, 2), coordinate2=(998, 997)),
        GUIOwlV2Action(action="type", text="Café"),
        GUIOwlV2Action(action="system_button", button="Back"),
        GUIOwlV2Action(action="open", text="设置"),
        GUIOwlV2Action(action="wait"),
        GUIOwlV2Action(action="answer", text="完成 ✅"),
        GUIOwlV2Action(action="terminate", status="success"),
    )


def _goldens() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, action in enumerate(_actions()):
        target = serialize_gui_owl_v2_1_teacher_target(action)
        target_ids = [TOOL_CALL_OPEN_TOKEN_ID, 500 + index, TOOL_CALL_CLOSE_TOKEN_ID]
        tool_calls = [
            {
                "type": "function",
                "function": {"name": "mobile_use", "arguments": action.arguments()},
            }
        ]
        records.append(
            {
                "action": action.action,
                "assistant_content": "",
                "assistant_tool_calls": tool_calls,
                "target_text": target,
                "rendered_conversation": (
                    f"prefix{ASSISTANT_PREFIX}{target}<|im_end|>\n"
                ),
                "assistant_prefix_token_ids": list(ASSISTANT_PREFIX_TOKEN_IDS),
                "target_token_ids": target_ids,
                "joint_token_ids": [*ASSISTANT_PREFIX_TOKEN_IDS, *target_ids],
            }
        )
    return records


def _git(path: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class RestorationV21ProcessorAuditTest(unittest.TestCase):
    def test_cli_records_frozen_container_and_cpu_only_provenance(self) -> None:
        container_id = "a" * 64
        image_digest = "sha256:" + "b" * 64
        argv = [
            "--repository-root",
            "/repo",
            "--derived-artifact-root",
            "/artifact",
            "--model-dir",
            "/model",
            "--snapshot-manifest",
            "/repo/snapshot.json",
            "--v2-config",
            "/repo/v2.json",
            "--v2-1-config",
            "/repo/v21.json",
            "--selection-manifest",
            "/repo/selection.json",
            "--ocr-backend-config",
            "/repo/ocr.json",
            "--artifact-binding-manifest",
            "/repo/artifact.json",
            "--host-alias",
            "hyper00",
            "--host-hostname",
            "node-radixark-16-0000",
            "--container-id",
            container_id,
            "--container-image-digest",
            image_digest,
            "--run-git-commit",
            "c" * 40,
            "--output",
            "/external/formal.json",
        ]
        args = parse_args(argv)
        _require_absolute_paths(args)
        args.repository_root = Path("relative-repo")
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            _require_absolute_paths(args)
        args.repository_root = Path("/repo")
        contract = SimpleNamespace(
            data={
                "pilot_execution": {
                    "canonical_host_alias": "hyper00",
                    "canonical_host_hostname": "node-radixark-16-0000",
                    "canonical_container_id": container_id,
                    "canonical_container_image_digest": image_digest,
                }
            }
        )
        with mock.patch(
            "scripts.audit_gui_owl_v2_1_processor.socket.gethostname",
            return_value="a" * 12,
        ):
            _validate_explicit_runtime_args(args, contract=contract)
            runtime = _execution_runtime_identity(
                args,
                started_at_utc="2026-07-15T00:00:00Z",
                ended_at_utc="2026-07-15T00:00:01Z",
                duration_seconds=1.0,
            )
        self.assertEqual(len(args.audit_argv), 32)
        self.assertEqual(runtime["processor_device"], "cpu")
        self.assertIs(runtime["gpu_operations_executed"], False)
        self.assertIsNone(runtime["policy_dtype"])
        self.assertIs(runtime["dtype_not_applicable"], True)
        self.assertIsNone(runtime["random_seed"])
        self.assertIs(runtime["seed_not_applicable"], True)

    def test_reducer_recomputes_exact_90_records_and_all_nine_goldens(self) -> None:
        reduction = reduce_restoration_v2_1_processor_records(
            prompt_records=_prompt_records(),
            teacher_golden_records=_goldens(),
            expected_state_projection=_states(),
            bindings=_bindings(),
        )

        summary = reduction["preflight_summary"]
        self.assertEqual(summary["prompt_count"], 90)
        self.assertEqual(summary["official_tools_injected_prompt_count"], 90)
        self.assertEqual(summary["image_count_distribution"], {"1": 45, "3": 15, "4": 15, "5": 15})
        self.assertEqual(summary["context_overflow_count"], 0)
        self.assertEqual(len(reduction["shape_records"]), 90)
        self.assertEqual(len(reduction["teacher_projection"]), 9)

    def test_stored_summary_is_not_trusted_and_record_tampering_fails(self) -> None:
        prompts = _prompt_records()
        goldens = _goldens()
        reduction = reduce_restoration_v2_1_processor_records(
            prompt_records=prompts,
            teacher_golden_records=goldens,
            expected_state_projection=_states(),
            bindings=_bindings(),
        )
        tampered_summary = copy.deepcopy(reduction)
        tampered_summary["preflight_summary"]["prompt_count"] = 89
        with self.assertRaisesRegex(ValueError, "record recomputation"):
            validate_stored_processor_reduction(
                tampered_summary,
                prompt_records=prompts,
                teacher_golden_records=goldens,
                expected_state_projection=_states(),
                bindings=_bindings(),
            )

        duplicated_tools = copy.deepcopy(prompts)
        duplicated_tools[0]["rendered_prompt"] += OFFICIAL_TOOL_BLOCK
        with self.assertRaisesRegex(ValueError, "canonical tools block"):
            reduce_restoration_v2_1_processor_records(
                prompt_records=duplicated_tools,
                teacher_golden_records=goldens,
                expected_state_projection=_states(),
                bindings=_bindings(),
            )

    def test_context_overflow_and_teacher_render_drift_fail_closed(self) -> None:
        prompts = _prompt_records()
        long_ids = [7] * (EXPECTED_MAXIMUM_CONTEXT_TOKENS - 255 - 3) + list(
            ASSISTANT_PREFIX_TOKEN_IDS
        )
        prompts[0]["input_ids"] = long_ids
        prompts[0]["attention_mask"] = [1] * len(long_ids)
        prompts[0]["tensor_inventory"]["input_ids"]["shape"] = [1, len(long_ids)]
        prompts[0]["tensor_inventory"]["attention_mask"]["shape"] = [1, len(long_ids)]
        with self.assertRaisesRegex(ValueError, "context overflow"):
            reduce_restoration_v2_1_processor_records(
                prompt_records=prompts,
                teacher_golden_records=_goldens(),
                expected_state_projection=_states(),
                bindings=_bindings(),
            )

        goldens = _goldens()
        goldens[3]["rendered_conversation"] = str(
            goldens[3]["rendered_conversation"]
        ).replace("Café", "Caf\\u00e9")
        with self.assertRaisesRegex(ValueError, "tool_calls render"):
            reduce_restoration_v2_1_processor_records(
                prompt_records=_prompt_records(),
                teacher_golden_records=goldens,
                expected_state_projection=_states(),
                bindings=_bindings(),
            )

    def test_official_jinja_insertion_order_and_raw_unicode_are_frozen(self) -> None:
        self.assertTrue(OFFICIAL_TOOL_JSON.startswith('{"type": "function"'))
        self.assertGreater(
            OFFICIAL_TOOL_JSON.index('"function":'),
            OFFICIAL_TOOL_JSON.index('"type":'),
        )
        type_target = serialize_gui_owl_v2_1_teacher_target(_actions()[3])
        open_target = serialize_gui_owl_v2_1_teacher_target(_actions()[5])
        answer_target = serialize_gui_owl_v2_1_teacher_target(_actions()[7])
        self.assertIn("Café", type_target)
        self.assertIn("设置", open_target)
        self.assertIn("完成 ✅", answer_target)
        self.assertNotIn("Caf\\u00e9", type_target)

    def test_confirm_safe_prompt_specs_are_exact_reference_then_summary(self) -> None:
        states = tuple(
            ScreeningState(
                index=index,
                role="v2_label_train" if index < 30 else "v2_development",
                trajectory_id=f"trajectory-{index // 3:02d}",
                decision_step_id=4 + index % 3,
                candidate_event_step_ids=tuple(range(1, 3 + index % 3)),
            )
            for index in range(45)
        )
        artifact = ValidatedScreeningArtifact(
            artifact_root=Path("/screening"),
            artifact_tree_sha256="1" * 64,
            artifact_manifest_sha256="2" * 64,
            screening_manifest_sha256="3" * 64,
            states=states,
            validation=MappingProxyType({}),
            _screening_manifest_json=b"{}",
            _screening_image_payloads=MappingProxyType({}),
        )
        specs = list(iter_v2_1_processor_prompt_specs(artifact))
        self.assertEqual(len(specs), 90)
        self.assertEqual(
            [spec.fidelity for spec in specs[:4]],
            ["reference", "summary_only", "reference", "summary_only"],
        )
        self.assertEqual(specs[0].restored_event_step_ids, states[0].candidate_event_step_ids)
        self.assertEqual(specs[1].restored_event_step_ids, ())
        self.assertNotIn("v2_confirm_primary", {spec.state.role for spec in specs})

    def test_cli_ast_has_no_model_import_forward_or_generation_call(self) -> None:
        _validate_processor_only_source(AUDIT_SCRIPT.read_bytes())
        source = AUDIT_SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("AutoModel", source)
        self.assertNotIn(
            "from causalcache.policy.gui_owl_v2_1_runtime",
            source,
        )
        forbidden_sources = (
            b"from transformers import AutoModel\n",
            b"from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21Runtime\n",
            b"policy.generate()\n",
        )
        for forbidden in forbidden_sources:
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ValueError):
                    _validate_processor_only_source(forbidden)

    def test_evidence_x_is_reusable_from_descendant_y_and_manifest_binds_raw(self) -> None:
        reduction = reduce_restoration_v2_1_processor_records(
            prompt_records=_prompt_records(),
            teacher_golden_records=_goldens(),
            expected_state_projection=_states(),
            bindings=_bindings(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            _git(root, "init", "-b", "main")
            _git(root, "config", "user.email", "audit@example.invalid")
            _git(root, "config", "user.name", "Audit Test")
            seed = root / "seed.txt"
            seed.write_text("x\n", encoding="utf-8")
            _git(root, "add", "seed.txt")
            _git(root, "commit", "-m", "source X")
            source_x = _git(root, "rev-parse", "HEAD")
            raw_value = {
                "run_git_commit": source_x,
                "reduction": reduction,
                "processor_identity": {
                    "classes": {
                        label: {
                            "name": f"{label}Class",
                            "module": "test.processor",
                            "source_path": f"/tmp/{label}.py",
                            "source_sha256": str(index + 1) * 64,
                        }
                        for index, label in enumerate(
                            (
                                "auto_processor",
                                "processor",
                                "tokenizer",
                                "image_processor",
                            )
                        )
                    }
                },
            }
            raw_path = Path(temporary) / "formal-result.json"
            raw_path.write_bytes(canonical_json_bytes(raw_value) + b"\n")
            manifest = build_processor_evidence_artifact_manifest(
                raw_value,
                raw_evidence_path=raw_path,
                hf_repo=CANONICAL_EVIDENCE_HF_REPO,
                hf_immutable_revision="a" * 40,
                hf_path="processor-preflight-v1/formal-result.json",
            )
            self.assertEqual(
                manifest["hf_artifact"]["tag"],
                CANONICAL_EVIDENCE_HF_TAG,
            )
            with self.assertRaisesRegex(ValueError, "immutable 40-hex"):
                build_processor_evidence_artifact_manifest(
                    raw_value,
                    raw_evidence_path=raw_path,
                    hf_repo=CANONICAL_EVIDENCE_HF_REPO,
                    hf_immutable_revision="main",
                    hf_path="processor-preflight-v1/formal-result.json",
                )
            manifest_path = root / CANONICAL_EVIDENCE_ARTIFACT_PATH
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
            _git(root, "add", CANONICAL_EVIDENCE_ARTIFACT_PATH)
            _git(root, "commit", "-m", "manifest Y")
            current_y = _git(root, "rev-parse", "HEAD")

            require_evidence_commit_ancestor(
                repository_root=root,
                evidence_git_commit=source_x,
                current_git_commit=current_y,
            )
            observed = validate_committed_processor_evidence_artifact(
                raw_value,
                repository_root=root,
                current_git_commit=current_y,
                raw_evidence_path=raw_path,
            )
            self.assertEqual(observed, manifest)
            with self.assertRaisesRegex(ValueError, "not an ancestor"):
                require_evidence_commit_ancestor(
                    repository_root=root,
                    evidence_git_commit=current_y,
                    current_git_commit=source_x,
                )
            raw_path.write_bytes(canonical_json_bytes(raw_value) + b" \n")
            with self.assertRaisesRegex(ValueError, "hash/size"):
                validate_committed_processor_evidence_artifact(
                    raw_value,
                    repository_root=root,
                    current_git_commit=current_y,
                    raw_evidence_path=raw_path,
                )


if __name__ == "__main__":
    unittest.main()
