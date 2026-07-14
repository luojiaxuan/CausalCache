import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.download_hf_snapshot import verify_file


class SnapshotDownloadTest(unittest.TestCase):
    def test_verify_file_checks_size_and_sha256(self) -> None:
        payload = b"pinned snapshot fixture"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bin"
            path.write_bytes(payload)
            specification = {
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            self.assertEqual(verify_file(path, specification), specification["sha256"])

            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_file(path, {"size": len(payload) + 1})


if __name__ == "__main__":
    unittest.main()
