from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.package_aaai_code import ARCHIVE_ROOT, build_archive, verify_archive


class PackageAAAIcodeTest(unittest.TestCase):
    def test_archive_contains_only_code_and_metadata(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "code.zip"
            build_archive(repository_root, output)
            verify_archive(output)
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
            self.assertIn(f"{ARCHIVE_ROOT}/README.md", names)
            self.assertIn(
                f"{ARCHIVE_ROOT}/code/configs/causalcache_paper_evaluation.json",
                names,
            )
            self.assertFalse(any(name.startswith(f"{ARCHIVE_ROOT}/data/") for name in names))
            self.assertFalse(any(name.endswith((".jsonl", ".parquet", ".pdf", ".pt")) for name in names))


if __name__ == "__main__":
    unittest.main()
