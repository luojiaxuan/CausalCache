import tempfile
import unittest
from pathlib import Path

from scripts.prepare_androidworld_dockerfile import (
    COMPATIBLE_BASE_IMAGE,
    REMOVED_BASE_IMAGE,
    prepare_dockerfile,
)


class AndroidWorldDockerfileTest(unittest.TestCase):
    def test_replaces_only_the_removed_base_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Dockerfile"
            output = Path(directory) / "Dockerfile.causalcache"
            source.write_text(f"{REMOVED_BASE_IMAGE}\nRUN echo ready\n", encoding="utf-8")

            prepare_dockerfile(source, output)

            self.assertEqual(
                output.read_text(encoding="utf-8"),
                f"{COMPATIBLE_BASE_IMAGE}\nRUN echo ready\n",
            )

    def test_rejects_unexpected_upstream_dockerfile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Dockerfile"
            source.write_text("FROM debian:bookworm\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected exactly one"):
                prepare_dockerfile(source, Path(directory) / "output")


if __name__ == "__main__":
    unittest.main()
