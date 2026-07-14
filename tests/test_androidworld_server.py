import tempfile
import unittest
from pathlib import Path

from scripts.prepare_androidworld_server import (
    LEGACY_JSON_ACTION_IMPORT,
    PINNED_AGENT_JSON_ACTION_IMPORT,
    prepare_server,
)


class AndroidWorldServerTest(unittest.TestCase):
    def test_replaces_legacy_action_schema_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "android_server.py"
            source.write_text(
                f"{LEGACY_JSON_ACTION_IMPORT}\n\ndef endpoint():\n  return json_action.JSONAction()\n",
                encoding="utf-8",
            )

            prepare_server(source)

            self.assertIn(
                PINNED_AGENT_JSON_ACTION_IMPORT,
                source.read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                LEGACY_JSON_ACTION_IMPORT,
                source.read_text(encoding="utf-8"),
            )

    def test_rejects_already_patched_or_unexpected_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "android_server.py"
            source.write_text(PINNED_AGENT_JSON_ACTION_IMPORT, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                prepare_server(source)


if __name__ == "__main__":
    unittest.main()
