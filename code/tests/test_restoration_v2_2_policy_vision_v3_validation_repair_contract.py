from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract import (
    ARTIFACT_GIT_COMMIT,
    CANONICAL_CONFIG_PATH,
    FROZEN_CONFIG_SHA256,
    PRODUCER_CLOSURE_INVENTORY_SHA256,
    PRODUCER_CLOSURE_PATH_COUNT,
    PRODUCER_SOURCE_GIT_COMMIT,
    PolicyVisionV3ValidationRepairContract,
    validate_exact_artifact_directory,
)


ROOT = Path(__file__).resolve().parents[2]


class PolicyVisionV3ValidationRepairContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = PolicyVisionV3ValidationRepairContract.load(
            ROOT / CANONICAL_CONFIG_PATH,
            repository_root=ROOT,
        )

    def test_config_and_all_producer_bindings_are_exact(self) -> None:
        self.assertEqual(
            hashlib.sha256((ROOT / CANONICAL_CONFIG_PATH).read_bytes()).hexdigest(),
            FROZEN_CONFIG_SHA256,
        )
        self.assertEqual(
            self.contract.data["producer"]["producer_source_git_commit"],
            PRODUCER_SOURCE_GIT_COMMIT,
        )
        self.assertEqual(
            self.contract.data["producer"]["artifact_git_commit"],
            ARTIFACT_GIT_COMMIT,
        )
        self.assertEqual(
            self.contract.data["producer"]["formal_python_source_closure"],
            {
                "rule": (
                    "all_tracked_python_under_code_causalcache_and_code_scripts_"
                    "at_source_commit"
                ),
                "path_count": PRODUCER_CLOSURE_PATH_COUNT,
                "inventory_sha256": PRODUCER_CLOSURE_INVENTORY_SHA256,
            },
        )

    def test_source_freeze_requires_result_staging_and_ledger_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "attempt.json"
            seal = Path(temporary) / "completion.json"
            with (
                mock.patch(
                    "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract.FORMAL_ATTEMPT_LEDGER_PATH",
                    str(ledger),
                ),
                mock.patch(
                    "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract.FORMAL_COMPLETION_SEAL_PATH",
                    str(seal),
                ),
            ):
                self.contract.validate_outputs_absent(
                    ledger_path=ledger,
                    completion_seal_path=seal,
                )
                seal.write_text("occupied")
                with self.assertRaisesRegex(FileExistsError, "completion seal"):
                    self.contract.validate_outputs_absent(
                        ledger_path=ledger,
                        completion_seal_path=seal,
                    )
                seal.unlink()
                ledger.write_text("occupied")
                with self.assertRaisesRegex(FileExistsError, "ledger"):
                    self.contract.validate_outputs_absent(
                        ledger_path=ledger,
                        completion_seal_path=seal,
                    )

    def test_source_freeze_rejects_noncanonical_absence_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical_ledger = root / "canonical-attempt.json"
            canonical_seal = root / "canonical-completion.json"
            missing_ledger = root / "missing-attempt.json"
            missing_seal = root / "missing-completion.json"
            canonical_ledger.write_text("occupied")
            canonical_seal.write_text("occupied")
            with (
                mock.patch(
                    "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract.FORMAL_ATTEMPT_LEDGER_PATH",
                    str(canonical_ledger),
                ),
                mock.patch(
                    "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract.FORMAL_COMPLETION_SEAL_PATH",
                    str(canonical_seal),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "attempt ledger path"):
                    self.contract.validate_outputs_absent(
                        ledger_path=missing_ledger,
                        completion_seal_path=canonical_seal,
                    )
                with self.assertRaisesRegex(ValueError, "completion seal path"):
                    self.contract.validate_outputs_absent(
                        ledger_path=canonical_ledger,
                        completion_seal_path=missing_seal,
                    )

    def test_artifact_validator_rejects_tamper_extra_and_symlink(self) -> None:
        bindings = self.contract.data["producer_artifact"]["exact_files"]
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "artifact"
            shutil.copytree(self.contract.artifact_directory, copied)
            validate_exact_artifact_directory(copied, bindings)

            (copied / "extra.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "inventory"):
                validate_exact_artifact_directory(copied, bindings)
            (copied / "extra.json").unlink()

            readme = copied / "README.md"
            original = readme.read_bytes()
            readme.write_bytes(original + b"x")
            with self.assertRaisesRegex(ValueError, "bytes drifted"):
                validate_exact_artifact_directory(copied, bindings)
            readme.write_bytes(original)

            state_scores = copied / "state_scores.jsonl"
            target = copied / "state_scores.target"
            state_scores.rename(target)
            state_scores.symlink_to(target.name)
            with self.assertRaisesRegex(ValueError, "inventory"):
                validate_exact_artifact_directory(copied, bindings)

    def test_formal_source_separates_producer_and_validation_commits(self) -> None:
        validation_commit = "f" * 40
        expected_paths = self.contract.data["validation_source"][
            "formal_source_changed_paths_exact"
        ]
        closure = {
            "rule": "closure",
            "path_count": 1,
            "inventory_sha256": "0" * 64,
            "paths": ("code/causalcache/frozen.py",),
        }

        def fake_git_text(root: Path, *args: str) -> str:
            del root
            if args[:3] == ("diff", "--name-only", ARTIFACT_GIT_COMMIT):
                return "\n".join(expected_paths)
            if args[:3] == ("diff", "--name-only", PRODUCER_SOURCE_GIT_COMMIT):
                return ""
            if args == ("rev-parse", "HEAD") or args == (
                "rev-parse",
                "origin/main",
            ):
                return validation_commit
            if args == ("symbolic-ref", "--short", "HEAD"):
                return "main"
            if args[:2] == ("status", "--porcelain"):
                return ""
            raise AssertionError(args)

        with (
            mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._require_commit"
            ),
            mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._require_ancestor"
            ),
            mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._producer_python_closure",
                return_value=closure,
            ),
            mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._python_source_closure",
                return_value=closure,
            ),
            mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._git_text",
                side_effect=fake_git_text,
            ),
        ):
            result = self.contract.validate_formal_source(
                validation_commit,
                require_clean_pushed_main=True,
            )
        self.assertEqual(
            result["producer_source_git_commit"], PRODUCER_SOURCE_GIT_COMMIT
        )
        self.assertEqual(result["validation_source_git_commit"], validation_commit)
        with self.assertRaisesRegex(ValueError, "must differ"):
            self.contract.validate_formal_source(
                PRODUCER_SOURCE_GIT_COMMIT,
                require_clean_pushed_main=False,
            )

    def test_formal_source_rejects_old_module_or_changed_path_drift(self) -> None:
        validation_commit = "e" * 40
        closure = {
            "rule": "closure",
            "path_count": 1,
            "inventory_sha256": "0" * 64,
            "paths": ("code/scripts/frozen.py",),
        }
        with (
            mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._require_commit"
            ),
            mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._require_ancestor"
            ),
            mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._producer_python_closure",
                return_value=closure,
            ),
            mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._python_source_closure",
                return_value=closure,
            ),
            mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._git_text",
                return_value="unexpected/path.py",
            ),
        ):
            with self.assertRaisesRegex(ValueError, "changed paths"):
                self.contract.validate_formal_source(
                    validation_commit,
                    require_clean_pushed_main=False,
                )

    def test_staging_source_recheck_allows_only_the_exact_two_output_files(self) -> None:
        validation_commit = "d" * 40
        staging = (
            ROOT
            / "data/results/.restoration_v2_2_policy_vision_"
            "v3_validation_repair_v1.staging"
        )
        relative = staging.relative_to(ROOT).as_posix()
        exact_status = "\n".join(
            f"?? {relative}/{name}" for name in ("README.md", "summary.json")
        )

        def fake_git_text(root: Path, *args: str) -> str:
            del root
            if args[:2] == ("status", "--porcelain"):
                return exact_status
            if args == ("rev-parse", "HEAD") or args == (
                "rev-parse",
                "origin/main",
            ):
                return validation_commit
            if args == ("symbolic-ref", "--short", "HEAD"):
                return "main"
            if args[:2] == ("diff", "--name-only") or args[:3] == (
                "diff",
                "--cached",
                "--name-only",
            ):
                return ""
            raise AssertionError(args)

        with (
            mock.patch.object(
                PolicyVisionV3ValidationRepairContract,
                "validate_formal_source",
                return_value={"validation_source_git_commit": validation_commit},
            ),
            mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._git_text",
                side_effect=fake_git_text,
            ),
        ):
            result = self.contract.validate_source_during_staging(
                validation_commit,
                staging_directory=staging,
            )
        self.assertEqual(result["validation_source_git_commit"], validation_commit)

        with (
            mock.patch.object(
                PolicyVisionV3ValidationRepairContract,
                "validate_formal_source",
                return_value={"validation_source_git_commit": validation_commit},
            ),
            mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._git_text",
                return_value=f"{exact_status}\n?? unexpected.txt",
            ),
        ):
            with self.assertRaisesRegex(ValueError, "untracked inventory"):
                self.contract.validate_source_during_staging(
                    validation_commit,
                    staging_directory=staging,
                )

    def test_descendant_validation_freezes_full_validation_and_producer_closures(
        self,
    ) -> None:
        validation_commit = "c" * 40
        head = "d" * 40
        validation_closure = {
            "rule": "validation-closure",
            "path_count": 1,
            "inventory_sha256": "1" * 64,
            "paths": ("code/causalcache/validation.py",),
        }
        producer_closure = {
            "rule": "producer-closure",
            "path_count": 1,
            "inventory_sha256": "2" * 64,
            "paths": ("code/causalcache/producer.py",),
        }
        source = {
            "validation_source_git_commit": validation_commit,
            "validation_python_source_closure": {
                key: validation_closure[key]
                for key in ("rule", "path_count", "inventory_sha256")
            },
        }

        def run_check(*, validation_drift: bool, producer_drift: bool):
            def fake_git_text(root: Path, *args: str) -> str:
                del root
                if args == ("rev-parse", "HEAD") or args == (
                    "rev-parse",
                    "origin/main",
                ):
                    return head
                if args == ("symbolic-ref", "--short", "HEAD"):
                    return "main"
                if args[:2] == ("status", "--porcelain"):
                    return ""
                if args[:4] == (
                    "diff",
                    "--name-only",
                    validation_commit,
                    head,
                ):
                    return (
                        "code/causalcache/validation.py"
                        if validation_drift
                        else ""
                    )
                if args[:4] == (
                    "diff",
                    "--name-only",
                    PRODUCER_SOURCE_GIT_COMMIT,
                    head,
                ):
                    return (
                        "code/causalcache/producer.py" if producer_drift else ""
                    )
                raise AssertionError(args)

            with (
                mock.patch.object(
                    PolicyVisionV3ValidationRepairContract,
                    "validate_formal_source",
                    return_value=source,
                ),
                mock.patch(
                    "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._require_ancestor"
                ),
                mock.patch(
                    "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._python_source_closure",
                    return_value=validation_closure,
                ),
                mock.patch(
                    "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._producer_python_closure",
                    return_value=producer_closure,
                ),
                mock.patch(
                    "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._git_text",
                    side_effect=fake_git_text,
                ),
            ):
                return self.contract.validate_clean_pushed_descendant(
                    validation_commit
                )

        result = run_check(validation_drift=False, producer_drift=False)
        self.assertEqual(result["validation_descendant_git_commit"], head)
        with self.assertRaisesRegex(ValueError, "validation Python closure"):
            run_check(validation_drift=True, producer_drift=False)
        with self.assertRaisesRegex(ValueError, "producer Python closure"):
            run_check(validation_drift=False, producer_drift=True)

    def test_descendant_validation_rejects_new_python_closure_inventory(self) -> None:
        validation_commit = "c" * 40
        head = "d" * 40
        source_closure = {
            "rule": "closure",
            "path_count": 1,
            "inventory_sha256": "1" * 64,
            "paths": ("code/causalcache/validation.py",),
        }
        head_closure = {
            **source_closure,
            "path_count": 2,
            "inventory_sha256": "2" * 64,
            "paths": (
                "code/causalcache/validation.py",
                "code/causalcache/new_module.py",
            ),
        }
        source = {
            "validation_source_git_commit": validation_commit,
            "validation_python_source_closure": {
                key: source_closure[key]
                for key in ("rule", "path_count", "inventory_sha256")
            },
        }

        def fake_git_text(root: Path, *args: str) -> str:
            del root
            if args == ("rev-parse", "HEAD") or args == (
                "rev-parse",
                "origin/main",
            ):
                return head
            if args == ("symbolic-ref", "--short", "HEAD"):
                return "main"
            if args[:2] == ("status", "--porcelain"):
                return ""
            if args[:2] == ("diff", "--name-only"):
                return ""
            raise AssertionError(args)

        with (
            mock.patch.object(
                PolicyVisionV3ValidationRepairContract,
                "validate_formal_source",
                return_value=source,
            ),
            mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._require_ancestor"
            ),
            mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._python_source_closure",
                side_effect=(source_closure, head_closure),
            ),
            mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract._git_text",
                side_effect=fake_git_text,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "closure inventory"):
                self.contract.validate_clean_pushed_descendant(validation_commit)


if __name__ == "__main__":
    unittest.main()
