import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from causalcache.data.guiodyssey import build_pilot_manifest, canonicalize_tool_call, write_pilot_dataset
from causalcache.schema import ActionType


PNG = b"\x89PNG\r\n\x1a\n" + b"fixture"


class GUIOdysseyTest(unittest.TestCase):
    def test_tool_call_canonicalization(self) -> None:
        action, target = canonicalize_tool_call(
            {"function": {"name": "tap", "arguments": {"coordinate": [1000, 0], "clicks": 1}}}
        )
        self.assertEqual(action.action_type, ActionType.TAP)
        self.assertEqual(target, "coordinate_bin:x9_y0")

        swipe, swipe_target = canonicalize_tool_call(
            {
                "function": {
                    "name": "swipe",
                    "arguments": {
                        "start_coordinate": [603, 824],
                        "coordinate": [620, 304],
                    },
                }
            }
        )
        self.assertEqual(swipe.action_type, ActionType.SWIPE)
        self.assertEqual(swipe_target, "scroll:down")

    def test_manifest_omits_inline_reasoning(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Open settings and go home."},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "inline_reasoning", "text": "private expert rationale"},
                    {"type": "action_description", "text": "Open settings."},
                ],
                "tool_calls": [
                    {"function": {"name": "tap", "arguments": {"coordinate": [100, 200], "clicks": 1}}}
                ],
            },
            {"role": "user", "content": [{"type": "image", "index": 1}]},
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "Go home."}],
                "tool_calls": [
                    {"function": {"name": "system_button", "arguments": {"button": "Home"}}},
                    {"function": {"name": "terminate", "arguments": {"status": "success"}}},
                ],
            },
        ]
        row = {
            "images": [{"bytes": PNG, "path": None}, {"bytes": PNG, "path": None}],
            "messages": json.dumps(messages),
            "metadata": json.dumps(
                {
                    "platform": "mobile",
                    "others": {
                        "source_id": "fixture-001",
                        "apps": ["Settings"],
                        "device_name": "Fixture",
                        "resolution": [720, 1280],
                    },
                }
            ),
        }
        manifest, images = build_pilot_manifest(
            row,
            row_index=0,
            upstream_repo="upstream/repo",
            upstream_revision="upstream-sha",
            transport_repo="transport/repo",
            transport_revision="transport-sha",
            transport_file="mobile/train/shard.parquet",
            transport_file_sha256="fixture-sha256",
            hf_destination="owner/pilot",
        )
        serialized = json.dumps(manifest)
        self.assertNotIn("private expert rationale", serialized)
        self.assertEqual(len(manifest["trajectory"]["events"]), 1)
        self.assertEqual(len(manifest["trajectory"]["decisions"]), 1)
        self.assertEqual(manifest["trajectory"]["terminal_status"], "success")
        self.assertEqual(len(images), 2)

        failed_messages = json.loads(row["messages"])
        failed_messages[-1]["tool_calls"][-1]["function"]["arguments"]["status"] = "failure"
        failed_row = dict(row)
        failed_row["messages"] = json.dumps(failed_messages)
        with self.assertRaisesRegex(ValueError, "requires a successful recorded trajectory"):
            build_pilot_manifest(
                failed_row,
                row_index=0,
                upstream_repo="upstream/repo",
                upstream_revision="upstream-sha",
                transport_repo="transport/repo",
                transport_revision="transport-sha",
                transport_file="mobile/train/shard.parquet",
                transport_file_sha256="fixture-sha256",
                hf_destination="owner/pilot",
            )

        with tempfile.TemporaryDirectory() as directory:
            write_pilot_dataset(directory, manifest, images)
            tar_path = Path(directory) / "data" / "guiodyssey-pilot-00000.tar"
            with tarfile.open(tar_path) as archive:
                names = archive.getnames()
                self.assertEqual(names[0], "manifest.json")
                self.assertEqual(len(names), 3)
                manifest_member = archive.extractfile("manifest.json")
                self.assertIsNotNone(manifest_member)
                self.assertEqual(json.load(io.TextIOWrapper(manifest_member, encoding="utf-8")), manifest)


if __name__ == "__main__":
    unittest.main()
