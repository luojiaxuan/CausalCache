import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class RepositoryLayoutTest(unittest.TestCase):
    def test_canonical_top_level_layout_exists(self) -> None:
        for relative_path in (
            "README.md",
            "paper/main.tex",
            "code/causalcache",
            "code/scripts",
            "code/tests",
            "code/configs",
            "code/configs/run_manifest.schema.json",
            "data/fixtures",
            "data/results",
            "docs/progress.md",
        ):
            self.assertTrue(
                (REPOSITORY_ROOT / relative_path).exists(),
                relative_path,
            )

    def test_legacy_top_level_directories_are_absent(self) -> None:
        for directory in ("causalcache", "scripts", "tests", "configs", "results"):
            self.assertFalse((REPOSITORY_ROOT / directory).exists(), directory)


if __name__ == "__main__":
    unittest.main()
