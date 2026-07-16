from __future__ import annotations

import argparse
import hashlib
import unittest
from pathlib import Path
from unittest import mock

from causalcache.restoration_v2_2_eager_contract import RestorationV22EagerContract
from scripts import validate_restoration_v2_2_eager_contract as validator


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/configs/causalcache_restoration_v2_2_eager.json"
HEAD = "1" * 40


def _git_response(
    repository_root: Path,
    arguments: tuple[str, ...],
    *,
    text: bool = True,
) -> str | bytes:
    values: dict[tuple[str, ...], str] = {
        ("rev-parse", "--show-toplevel"): f"{repository_root}\n",
        ("branch", "--show-current"): "main\n",
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
        ("rev-parse", "HEAD"): f"{HEAD}\n",
        ("rev-parse", "refs/remotes/origin/main"): f"{HEAD}\n",
        (
            "ls-remote",
            "--heads",
            "origin",
            "refs/heads/main",
        ): f"{HEAD}\trefs/heads/main\n",
        ("remote", "get-url", "origin"): f"{validator.CANONICAL_REMOTE_URL}\n",
    }
    if arguments[:1] == ("show",):
        relative = arguments[1].removeprefix("HEAD:")
        payload = (repository_root / relative).read_bytes()
        return payload if not text else payload.decode("utf-8")
    return values[arguments]


class RestorationV22EagerSourceFreezeValidatorTest(unittest.TestCase):
    def test_default_validation_does_not_claim_formal_source_freeze(self) -> None:
        result = validator.validate_from_args(
            argparse.Namespace(repository_root=ROOT, config=CONFIG)
        )
        self.assertIs(result["contract_valid"], True)
        self.assertEqual(
            result["source_freeze"],
            {
                "formal_source_freeze_established": False,
                "status": "NOT_REQUESTED",
                "git": None,
            },
        )

    @mock.patch.object(validator, "_run_git", side_effect=_git_response)
    def test_strict_mode_binds_every_inventory_file_to_head(
        self, run_git: mock.Mock
    ) -> None:
        contract = RestorationV22EagerContract.load(CONFIG, repository_root=ROOT)
        inventory = contract.data["scientific_inheritance"][
            "formal_run_source_inventory_paths"
        ]
        result = validator.validate_from_args(
            argparse.Namespace(
                repository_root=ROOT,
                config=CONFIG,
                require_clean_pushed_main=True,
            )
        )
        source_freeze = result["source_freeze"]
        self.assertIs(source_freeze["formal_source_freeze_established"], True)
        git = source_freeze["git"]
        self.assertEqual(git["commit"], HEAD)
        self.assertEqual(git["formal_inventory_file_count"], len(inventory))
        self.assertEqual(
            [record["path"] for record in git["formal_inventory"]], inventory
        )
        self.assertIn(
            "code/scripts/validate_restoration_v2_2_eager_contract.py",
            [record["path"] for record in git["formal_inventory"]],
        )
        show_calls = [
            call
            for call in run_git.call_args_list
            if call.args[1][:1] == ("show",)
        ]
        self.assertEqual(len(show_calls), len(inventory))

    def test_strict_mode_rejects_dirty_or_untracked_worktree(self) -> None:
        def dirty_git(
            repository_root: Path,
            arguments: tuple[str, ...],
            *,
            text: bool = True,
        ) -> str | bytes:
            if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
                return "?? untracked.py\n"
            return _git_response(repository_root, arguments, text=text)

        with mock.patch.object(validator, "_run_git", side_effect=dirty_git):
            with self.assertRaisesRegex(ValueError, "clean worktree"):
                validator.validate_clean_pushed_formal_inventory(
                    repository_root=ROOT,
                    inventory_paths=(
                        "code/scripts/validate_restoration_v2_2_eager_contract.py",
                    ),
                )

    def test_strict_mode_rejects_unpushed_head(self) -> None:
        def unpushed_git(
            repository_root: Path,
            arguments: tuple[str, ...],
            *,
            text: bool = True,
        ) -> str | bytes:
            if arguments == ("rev-parse", "refs/remotes/origin/main"):
                return f"{'2' * 40}\n"
            return _git_response(repository_root, arguments, text=text)

        with mock.patch.object(validator, "_run_git", side_effect=unpushed_git):
            with self.assertRaisesRegex(ValueError, "origin/main"):
                validator.validate_clean_pushed_formal_inventory(
                    repository_root=ROOT,
                    inventory_paths=(
                        "code/scripts/validate_restoration_v2_2_eager_contract.py",
                    ),
                )

    def test_strict_mode_rejects_working_file_that_differs_from_head_blob(self) -> None:
        def stale_blob_git(
            repository_root: Path,
            arguments: tuple[str, ...],
            *,
            text: bool = True,
        ) -> str | bytes:
            if arguments[:1] == ("show",):
                payload = b"not-the-working-file"
                return payload if not text else payload.decode("utf-8")
            return _git_response(repository_root, arguments, text=text)

        relative = "code/scripts/validate_restoration_v2_2_eager_contract.py"
        with mock.patch.object(validator, "_run_git", side_effect=stale_blob_git):
            with self.assertRaisesRegex(ValueError, "committed HEAD blob"):
                validator.validate_clean_pushed_formal_inventory(
                    repository_root=ROOT,
                    inventory_paths=(relative,),
                )

    def test_strict_mode_rejects_noncanonical_inventory_path(self) -> None:
        with mock.patch.object(validator, "_run_git", side_effect=_git_response):
            with self.assertRaisesRegex(ValueError, "not canonical"):
                validator.validate_clean_pushed_formal_inventory(
                    repository_root=ROOT,
                    inventory_paths=("code/scripts/../scripts/example.py",),
                )

    def test_strict_mode_rejects_repository_drift_during_validation(self) -> None:
        status_calls = 0

        def drifting_git(
            repository_root: Path,
            arguments: tuple[str, ...],
            *,
            text: bool = True,
        ) -> str | bytes:
            nonlocal status_calls
            if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
                status_calls += 1
                return "" if status_calls == 1 else " M changed.py\n"
            return _git_response(repository_root, arguments, text=text)

        relative = "code/scripts/validate_restoration_v2_2_eager_contract.py"
        with mock.patch.object(validator, "_run_git", side_effect=drifting_git):
            with self.assertRaisesRegex(ValueError, "drifted"):
                validator.validate_clean_pushed_formal_inventory(
                    repository_root=ROOT,
                    inventory_paths=(relative,),
                )

    def test_inventory_digest_is_over_ordered_path_and_blob_hash_records(self) -> None:
        relative = "code/scripts/validate_restoration_v2_2_eager_contract.py"
        with mock.patch.object(validator, "_run_git", side_effect=_git_response):
            result = validator.validate_clean_pushed_formal_inventory(
                repository_root=ROOT,
                inventory_paths=(relative,),
            )
        self.assertEqual(
            result["formal_inventory"][0]["sha256"],
            hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
