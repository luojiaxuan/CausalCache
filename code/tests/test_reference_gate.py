import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from causalcache.reference_gate import (
    reduce_reference_gate,
    run_reference_gate,
    validate_hardware_anchor,
    validate_v04_manifest,
)
from scripts import run_independent_ui_tars_reference_gate as reference_gate_cli


def _config() -> dict:
    return {
        "protocol_id": "independent_reference_gate_v1",
        "source_pool": {
            "upstream_repo": "upstream/repo",
            "upstream_revision": "upstream-sha",
            "transport_repo": "transport/repo",
            "transport_revision": "transport-sha",
            "transport_files": ["mobile/shard-00000.parquet"],
            "excluded_source_ids": ["old-source"],
        },
        "eligibility": {
            "platform": "mobile",
            "terminal_status": "success",
            "minimum_decisions_per_trajectory": 4,
            "maximum_decisions_per_trajectory": 12,
            "source_id_regex": "^[A-Za-z0-9._-]+$",
            "parser_compatible_action_types": [
                "back",
                "home",
                "long_press",
                "stop",
                "swipe",
                "tap",
                "type_text",
                "wait",
            ],
        },
        "selection": {
            "trajectory_salt": "fixture-salt",
            "trajectory_hash_input": "UTF8(salt + NUL + source_id)",
            "trajectory_order": [
                "sha256_hex",
                "source_id",
                "transport_file",
                "transport_row_index",
            ],
            "algorithm": "shortest frozen prefixes",
            "required_action_types": ["tap", "swipe", "type_text"],
            "reference_gate": {
                "minimum_trajectories": 1,
                "minimum_decisions": 4,
                "minimum_distinct_app_labels": 1,
                "minimum_candidate_count_per_required_action_type": 1,
            },
            "oracle_pilot": {
                "minimum_trajectories": 1,
                "minimum_decisions": 4,
                "minimum_distinct_app_labels": 1,
                "minimum_candidate_count_per_required_action_type": 1,
            },
        },
        "artifact": {"repo": "owner/dataset", "revision": "artifact-sha"},
        "policy": {
            "repo": "ByteDance-Seed/UI-TARS-1.5-7B",
            "revision": "model-sha",
            "dtype": "bfloat16",
            "visual_tokens_per_image": 256,
            "max_new_tokens": 256,
        },
        "hardware_anchor": {
            "dataset_repo": "owner/old",
            "dataset_revision": "old-sha",
            "dataset_shard_sha256": "a" * 64,
            "expected_parsed": 3,
            "expected_matches": 2,
            "expected_decision_match_vector": [[2, False], [3, True], [4, True]],
        },
        "reference_gate": {
            "validation_mode": "full_history_executable_match",
            "minimum_overall_coverage": 0.5,
            "required_action_types": ["tap", "swipe", "type_text"],
            "pass_outcome": "REFERENCE_GATE_PASSED",
            "fail_outcome": "NO_GO_CURRENT_REFERENCE_STACK",
        },
    }


def _record(
    source_id: str,
    step_id: int,
    action_type: str,
    match: bool,
    *,
    parse_error: str | None = None,
) -> dict:
    return {
        "trajectory_source_id": source_id,
        "decision_step_id": step_id,
        "validated_action": {"action_type": action_type},
        "parse_error": parse_error,
        "executable_match": match,
    }


def _trajectory(source_id: str, split: str, app: str) -> dict:
    salt = _config()["selection"]["trajectory_salt"]
    action_types = ["tap", "swipe", "type_text", "tap"]
    decisions = []
    for step_id, action_type in enumerate(action_types, start=2):
        decisions.append(
            {
                "decision_step_id": step_id,
                "current_observation_path": f"images/{source_id}/{step_id}.png",
                "history_event_step_ids": list(range(1, step_id)),
                "validated_action": {
                    "action_type": action_type,
                    "target": "coordinate_bin:x1_y1" if action_type == "tap" else None,
                    "text_argument": "query" if action_type == "type_text" else None,
                    "text_case_sensitive": False,
                },
            }
        )
    return {
        "source_id": source_id,
        "split": split,
        "transport_file": "mobile/shard-00000.parquet",
        "transport_row_index": 0 if source_id == "source-a" else 1,
        "selection_sha256": hashlib.sha256(f"{salt}\0{source_id}".encode()).hexdigest(),
        "instruction": "Complete the task.",
        "platform": "mobile",
        "apps": [app],
        "normalized_app_labels": [app.casefold()],
        "terminal_status": "success",
        "events": [
            {
                "step_id": value,
                "observation_before_path": f"images/{source_id}/{value}.png",
                "observation_after_path": f"images/{source_id}/{value + 1}.png",
                "executed_action": {
                    "action_type": "tap",
                    "target": "coordinate_bin:x1_y1",
                    "text_argument": None,
                    "text_case_sensitive": False,
                },
                "source_tool_call": {"function": {"name": "tap", "arguments": {}}},
                "low_fidelity": {
                    "step_id": value,
                    "action_type": "tap",
                    "target_text_or_coordinate_bin": "coordinate_bin:x1_y1",
                    "deterministic_ui_delta": "not_available",
                    "result_status": "unknown",
                },
            }
            for value in range(1, 5)
        ],
        "decisions": decisions,
    }


def _manifest() -> dict:
    config = _config()
    trajectories = [_trajectory("source-a", "", "App A"), _trajectory("source-b", "", "App B")]
    trajectories.sort(
        key=lambda value: (
            value["selection_sha256"],
            value["source_id"],
            value["transport_file"],
            value["transport_row_index"],
        )
    )
    reference, oracle = trajectories
    reference["split"] = "reference_gate"
    oracle["split"] = "oracle_pilot"

    def split(trajectory: dict) -> dict:
        return {
            "source_ids": [trajectory["source_id"]],
            "trajectory_count": 1,
            "decision_count": 4,
            "distinct_app_labels": trajectory["normalized_app_labels"],
            "action_type_counts": {"tap": 2, "swipe": 1, "type_text": 1},
        }

    return {
        "schema_version": "0.4.0",
        "protocol_id": config["protocol_id"],
        "dataset_repo": config["artifact"]["repo"],
        "source": {
            "upstream_repo": config["source_pool"]["upstream_repo"],
            "upstream_revision": config["source_pool"]["upstream_revision"],
            "transport_repo": config["source_pool"]["transport_repo"],
            "transport_revision": config["source_pool"]["transport_revision"],
            "transport_files": [
                {
                    "transport_file": "mobile/shard-00000.parquet",
                    "sha256": "b" * 64,
                    "size_bytes": 10,
                    "row_count": 2,
                }
            ],
            "protocol_config_sha256": "c" * 64,
            "source_file_manifest_sha256": "d" * 64,
        },
        "selection": {
            "algorithm": config["selection"]["algorithm"],
            "trajectory_salt": config["selection"]["trajectory_salt"],
            "trajectory_hash_input": config["selection"]["trajectory_hash_input"],
            "trajectory_order": config["selection"]["trajectory_order"],
            "eligibility": config["eligibility"],
            "excluded_source_ids": config["source_pool"]["excluded_source_ids"],
            "total_source_rows": 2,
            "eligible_pool_count": 2,
            "eligible_pool_sha256": "e" * 64,
            "exclusion_counts": {},
        },
        "splits": {
            "reference_gate": split(reference),
            "oracle_pilot": split(oracle),
        },
        "trajectories": trajectories,
    }


class ReferenceGateReducerTest(unittest.TestCase):
    def test_exact_half_and_required_type_matches_pass(self) -> None:
        records = [
            _record("a", 2, "tap", True),
            _record("a", 3, "swipe", True),
            _record("a", 4, "type_text", True),
            _record("b", 2, "tap", False),
            _record("b", 3, "tap", False),
            _record("b", 4, "tap", False),
        ]
        result = reduce_reference_gate(records, config=_config())
        self.assertEqual(result["outcome"], "REFERENCE_GATE_PASSED")
        self.assertEqual(result["overall"]["coverage"], 0.5)
        self.assertTrue(result["gate"]["action_type_coverage_pass"])

    def test_action_type_failure_is_scientific_no_go(self) -> None:
        records = [
            _record("a", 2, "tap", True),
            _record("a", 3, "swipe", True),
            _record("a", 4, "type_text", False),
            _record("a", 5, "tap", True),
        ]
        result = reduce_reference_gate(records, config=_config())
        self.assertEqual(result["outcome"], "NO_GO_CURRENT_REFERENCE_STACK")
        self.assertTrue(result["gate"]["overall_coverage_pass"])
        self.assertFalse(result["gate"]["action_type_coverage_pass"])

    def test_parse_failure_counts_as_non_match(self) -> None:
        records = [
            _record("a", 2, "tap", True),
            _record("a", 3, "swipe", True),
            _record("a", 4, "type_text", False, parse_error="invalid output"),
        ]
        result = reduce_reference_gate(records, config=_config())
        self.assertEqual(result["overall"]["parsed"], 2)
        self.assertEqual(result["overall"]["matches"], 2)
        self.assertEqual(result["outcome"], "NO_GO_CURRENT_REFERENCE_STACK")

    def test_duplicate_decision_is_invalid(self) -> None:
        records = [_record("a", 2, "tap", True), _record("a", 2, "tap", True)]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            reduce_reference_gate(records, config=_config())


class ReferenceGateManifestTest(unittest.TestCase):
    def test_v04_multi_trajectory_manifest_passes(self) -> None:
        result = validate_v04_manifest(_manifest(), config=_config())
        self.assertEqual(result["splits"]["reference_gate"]["decisions"], 4)
        self.assertEqual(result["splits"]["oracle_pilot"]["trajectories"], 1)

    def test_excluded_source_and_overlap_are_invalid(self) -> None:
        excluded = _manifest()
        trajectory = excluded["trajectories"][0]
        old_id = trajectory["source_id"]
        _config_value = _config()
        _config_value["source_pool"]["excluded_source_ids"] = [old_id]
        with self.assertRaisesRegex(ValueError, "excluded"):
            validate_v04_manifest(excluded, config=_config_value)

        overlap = _manifest()
        overlap["splits"]["oracle_pilot"]["source_ids"] = overlap["splits"][
            "reference_gate"
        ]["source_ids"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_v04_manifest(overlap, config=_config())

    def test_missing_required_action_candidate_is_invalid(self) -> None:
        manifest = _manifest()
        source_id = manifest["splits"]["reference_gate"]["source_ids"][0]
        trajectory = next(
            value for value in manifest["trajectories"] if value["source_id"] == source_id
        )
        trajectory["decisions"][2]["validated_action"]["action_type"] = "tap"
        manifest["splits"]["reference_gate"]["action_type_counts"] = {
            "tap": 3,
            "swipe": 1,
            "type_text": 0,
        }
        with self.assertRaisesRegex(ValueError, "lacks"):
            validate_v04_manifest(manifest, config=_config())


class HardwareAnchorTest(unittest.TestCase):
    def test_anchor_provenance_and_vector_are_checked(self) -> None:
        config = _config()
        anchor = {
            "outcome": "HARDWARE_ANCHOR_PASSED",
            "dataset": {
                "repo": "owner/old",
                "revision": "old-sha",
                "shard_sha256": "a" * 64,
            },
            "model": {
                "snapshot": {
                    "repo": config["policy"]["repo"],
                    "revision": config["policy"]["revision"],
                },
                "dtype": "bfloat16",
            },
            "visual_tokens_per_image": 256,
            "max_new_tokens": 256,
            "overall": {"decisions": 3, "parsed": 3, "matches": 2},
            "decisions": [
                {"decision_step_id": 2, "executable_match": False},
                {"decision_step_id": 3, "executable_match": True},
                {"decision_step_id": 4, "executable_match": True},
            ],
            "runtime": {
                "raw_summary_sha256": "b" * 64,
                "gpu_monitor_sha256": "c" * 64,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchor.json"
            path.write_text(json.dumps(anchor), encoding="utf-8")
            result = validate_hardware_anchor(path, config=config)
            self.assertEqual(result["overall"]["matches"], 2)
            anchor["decisions"][-1]["executable_match"] = False
            path.write_text(json.dumps(anchor), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "vector"):
                validate_hardware_anchor(path, config=config)


class ReferenceGateOrderingTest(unittest.TestCase):
    def test_contract_failure_happens_before_runtime_construction(self) -> None:
        def forbidden_runtime(**_: object) -> object:
            self.fail("runtime must not be constructed after a contract failure")

        with patch(
            "causalcache.reference_gate.validate_reference_run_contract",
            side_effect=ValueError("bad frozen contract"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                config_path = Path(directory) / "config.json"
                config_path.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "bad frozen contract"):
                    run_reference_gate(
                        config_path=config_path,
                        dataset_tar=Path("missing.tar"),
                        hardware_anchor_summary=Path("missing-anchor.json"),
                        model_dir=Path("missing-model"),
                        device="cuda:0",
                        run_git_commit="commit",
                        container_image_digest="sha256:digest",
                        repo_root=Path("."),
                        argv=[],
                        runtime_factory=forbidden_runtime,
                    )


class ReferenceGateCLITest(unittest.TestCase):
    def _args(self, output_dir: Path) -> Namespace:
        return Namespace(
            config=Path("config.json"),
            dataset_tar=Path("dataset.tar"),
            hardware_anchor_summary=Path("anchor.json"),
            model_dir=Path("model"),
            device="cuda:0",
            output_dir=output_dir,
            run_git_commit="commit",
            container_image_digest="sha256:" + "a" * 64,
        )

    def test_scientific_no_go_writes_summary_without_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "result"
            result = {"outcome": "NO_GO_CURRENT_REFERENCE_STACK"}
            with (
                patch.object(reference_gate_cli, "parse_args", return_value=self._args(output_dir)),
                patch.object(
                    reference_gate_cli.subprocess,
                    "run",
                    return_value=SimpleNamespace(stdout=str(Path.cwd())),
                ),
                patch.object(reference_gate_cli, "run_reference_gate", return_value=result),
                patch("builtins.print"),
            ):
                reference_gate_cli.main()
            self.assertEqual(
                json.loads((output_dir / "summary.json").read_text()),
                result,
            )
            self.assertFalse((output_dir / "failure.json").exists())

    def test_operational_exception_writes_failure_and_reraises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "result"
            with (
                patch.object(reference_gate_cli, "parse_args", return_value=self._args(output_dir)),
                patch.object(
                    reference_gate_cli.subprocess,
                    "run",
                    return_value=SimpleNamespace(stdout=str(Path.cwd())),
                ),
                patch.object(
                    reference_gate_cli,
                    "run_reference_gate",
                    side_effect=ValueError("broken contract"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "broken contract"):
                    reference_gate_cli.main()
            failure = json.loads((output_dir / "failure.json").read_text())
            self.assertEqual(failure["status"], "INVALID")
            self.assertFalse((output_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
