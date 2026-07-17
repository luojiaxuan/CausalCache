from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from scripts import build_guiodyssey_restoration_v2_expansion as build_cli
from scripts import validate_guiodyssey_restoration_v2_expansion as validate_cli


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        label_expansion_config=root / "expansion-config.json",
        parent_selection_manifest=root / "parent.json",
        expansion_selection_manifest=root / "selection.json",
        v1_config=root / "v1.json",
        source_file_manifest=root / "source.json",
        source_root=root / "source",
        ocr_backend_config=root / "ocr-config.json",
        ocr_backend_manifest=root / "ocr-manifest.json",
        model_dir=root / "models",
        wheel_dir=root / "wheels",
        git_revision="a" * 40,
        output_dir=root / "output",
    )


def _input_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "label_expansion_config": args.label_expansion_config,
        "parent_selection_manifest": args.parent_selection_manifest,
        "expansion_selection_manifest": args.expansion_selection_manifest,
        "v1_config": args.v1_config,
        "source_file_manifest": args.source_file_manifest,
        "ocr_backend_config": args.ocr_backend_config,
        "ocr_backend_manifest": args.ocr_backend_manifest,
    }


class GUIOdysseyRestorationV2ExpansionBuildCLITest(unittest.TestCase):
    def test_cli_rejects_unregistered_dataset_repo_override(self) -> None:
        required = [
            "--label-expansion-config",
            "config.json",
            "--parent-selection-manifest",
            "parent.json",
            "--expansion-selection-manifest",
            "selection.json",
            "--v1-config",
            "v1.json",
            "--source-file-manifest",
            "source.json",
            "--source-root",
            "source",
            "--ocr-backend-config",
            "ocr.json",
            "--ocr-backend-manifest",
            "ocr-manifest.json",
            "--model-dir",
            "models",
            "--wheel-dir",
            "wheels",
            "--git-revision",
            "a" * 40,
            "--output-dir",
            "output",
        ]
        with self.assertRaises(SystemExit):
            build_cli.parse_args(required + ["--dataset-repo", "public/override"])

    def test_git_preflight_rejects_abbreviated_revision_without_subprocess(self) -> None:
        with mock.patch.object(build_cli.subprocess, "check_output") as check_output:
            with self.assertRaisesRegex(ValueError, "full 40-character"):
                build_cli._verify_git_checkout(Path("."), "abc123")
        check_output.assert_not_called()

    def test_git_preflight_requires_main_remote_advertisement_and_full_status(self) -> None:
        revision = "a" * 40
        outputs = [
            "main\n",
            f"{revision}\n",
            "",
            f"{revision}\n",
            f"{revision}\trefs/heads/main\n",
        ]
        with mock.patch.object(
            build_cli.subprocess, "check_output", side_effect=outputs
        ) as check_output:
            build_cli._verify_git_checkout(Path("."), revision)
        commands = [call.args[0] for call in check_output.call_args_list]
        self.assertIn(
            ["git", "status", "--porcelain", "--untracked-files=all"], commands
        )
        self.assertIn(
            [
                "git",
                "ls-remote",
                "--exit-code",
                "origin",
                "refs/heads/main",
            ],
            commands,
        )
        with mock.patch.object(
            build_cli.subprocess, "check_output", return_value="feature\n"
        ):
            with self.assertRaisesRegex(ValueError, "symbolic branch main"):
                build_cli._verify_git_checkout(Path("."), revision)

    def test_bad_source_rehash_state_leaves_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root)
            input_paths = _input_paths(args)
            for path in input_paths.values():
                path.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch.object(build_cli, "parse_args", return_value=args),
                mock.patch.object(build_cli, "_verify_git_checkout"),
                mock.patch.object(
                    build_cli,
                    "_require_canonical_inputs",
                    return_value=input_paths,
                ),
                mock.patch.object(
                    build_cli,
                    "_load_json_object",
                    return_value=(b"{}\n", {}),
                ),
                mock.patch.object(build_cli, "load_backend_config", return_value={}),
                mock.patch.object(build_cli, "validate_artifact_source"),
                mock.patch.object(build_cli, "validate_frozen_inputs"),
                mock.patch.object(
                    build_cli,
                    "source_dataset_identity_from_v1_config",
                    return_value={},
                ),
                mock.patch.object(
                    build_cli,
                    "load_expansion_source_pilots",
                    side_effect=ValueError("source Parquet SHA256 drifted"),
                ),
                mock.patch.object(build_cli, "runtime_identity_and_engine") as runtime,
                mock.patch.object(build_cli, "write_artifact") as write_artifact,
            ):
                with self.assertRaisesRegex(ValueError, "Parquet SHA256 drifted"):
                    build_cli.main([])
            self.assertFalse(args.output_dir.exists())
            runtime.assert_not_called()
            write_artifact.assert_not_called()

    def test_build_requires_formal_counts_and_full_postwrite_ocr_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root)
            input_paths = _input_paths(args)
            for path in input_paths.values():
                path.write_text("{}\n", encoding="utf-8")
            ocr_records = {"image": {"record": True}}
            aggregate = "9" * 64
            result = {
                "artifact_tree_sha256": "8" * 64,
                "ocr_record_aggregate_sha256": aggregate,
                "ocr_replay_record_count": 384,
                "ocr_replay_aggregate_sha256": aggregate,
            }
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(build_cli, "parse_args", return_value=args)
                )
                stack.enter_context(mock.patch.object(build_cli, "_verify_git_checkout"))
                stack.enter_context(
                    mock.patch.object(
                        build_cli,
                        "_require_canonical_inputs",
                        return_value=input_paths,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        build_cli,
                        "_load_json_object",
                        return_value=(b"{}\n", {}),
                    )
                )
                stack.enter_context(
                    mock.patch.object(build_cli, "load_backend_config", return_value={})
                )
                for name in (
                    "validate_artifact_source",
                    "validate_frozen_inputs",
                    "validate_formal_image_inventory",
                    "validate_ocr_runtime_identity",
                ):
                    stack.enter_context(mock.patch.object(build_cli, name))
                stack.enter_context(
                    mock.patch.object(
                        build_cli,
                        "source_dataset_identity_from_v1_config",
                        return_value={},
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        build_cli,
                        "load_expansion_source_pilots",
                        return_value=({"pilot": {}}, {"image": b"image"}),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        build_cli,
                        "build_ocr_backend_provenance",
                        return_value={"backend": True},
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        build_cli,
                        "required_image_member_paths",
                        return_value=("image",),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        build_cli,
                        "runtime_identity_and_engine",
                        return_value=({"runtime": True}, object()),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        build_cli,
                        "generate_ocr_records",
                        return_value=(ocr_records, aggregate),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        build_cli,
                        "build_trajectory_records",
                        return_value=([{"trajectory": True}], {"image": b"image"}),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        build_cli,
                        "_identity",
                        side_effect=lambda repository_root, path: {
                            "path": path.name,
                            "sha256": "7" * 64,
                        },
                    )
                )
                stack.enter_context(
                    mock.patch.object(build_cli, "_generator", return_value={})
                )
                write_artifact = stack.enter_context(
                    mock.patch.object(
                        build_cli,
                        "write_artifact",
                        return_value={"artifact_tree_sha256": "8" * 64},
                    )
                )
                validate_artifact = stack.enter_context(
                    mock.patch.object(
                        build_cli, "validate_artifact", return_value=result
                    )
                )
                stack.enter_context(mock.patch("builtins.print"))
                build_cli.main([])
            self.assertTrue(write_artifact.call_args.kwargs["formal_counts_enforced"])
            self.assertTrue(validate_artifact.call_args.kwargs["require_formal"])
            self.assertTrue(validate_artifact.call_args.kwargs["require_ocr_replay"])
            self.assertIsNotNone(validate_artifact.call_args.kwargs["ocr_engine"])


class GUIOdysseyRestorationV2ExpansionValidateCLITest(unittest.TestCase):
    def test_projection_preflight_rejects_symlink_before_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            root.mkdir()
            target = Path(directory) / "target"
            target.write_bytes(b"payload")
            (root / "linked").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "cannot contain symlinks"):
                validate_cli._reject_symlinks_and_special_entries(root)

    def test_validator_always_requests_formal_full_ocr_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root)
            input_paths = _input_paths(args)
            for path in input_paths.values():
                path.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch.object(validate_cli, "parse_args", return_value=args),
                mock.patch.object(validate_cli, "_verify_git_checkout"),
                mock.patch.object(
                    validate_cli,
                    "_require_canonical_inputs",
                    return_value=input_paths,
                ),
                mock.patch.object(validate_cli, "_preflight_artifact_projection"),
                mock.patch.object(
                    validate_cli,
                    "_load_json_object",
                    return_value=(b"{}\n", {}),
                ),
                mock.patch.object(validate_cli, "load_backend_config", return_value={}),
                mock.patch.object(validate_cli, "validate_artifact_source"),
                mock.patch.object(validate_cli, "validate_frozen_inputs"),
                mock.patch.object(
                    validate_cli,
                    "runtime_identity_and_engine",
                    return_value=({"runtime": True}, object()),
                ),
                mock.patch.object(validate_cli, "validate_ocr_runtime_identity"),
                mock.patch.object(
                    validate_cli,
                    "build_ocr_backend_provenance",
                    return_value={"backend": True},
                ),
                mock.patch.object(
                    validate_cli,
                    "source_dataset_identity_from_v1_config",
                    return_value={},
                ),
                mock.patch.object(
                    validate_cli,
                    "validate_artifact",
                    return_value={"outcome": "passed"},
                ) as validate_artifact,
                mock.patch("builtins.print") as output,
            ):
                validate_cli.main([])
            self.assertTrue(validate_artifact.call_args.kwargs["require_formal"])
            self.assertTrue(validate_artifact.call_args.kwargs["require_ocr_replay"])
            self.assertIsNotNone(validate_artifact.call_args.kwargs["ocr_engine"])
            rendered = output.call_args.args[0]
            self.assertEqual(json.loads(rendered)["outcome"], "passed")


if __name__ == "__main__":
    unittest.main()
