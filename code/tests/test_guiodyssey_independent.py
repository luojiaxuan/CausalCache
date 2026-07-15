import copy
import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from causalcache.data.guiodyssey_independent import (
    ExclusionReason,
    SourceFileSpec,
    build_independent_manifest,
    candidate_sort_key,
    inspect_candidate,
    select_splits,
    source_file_specs,
    verify_local_source_files,
    write_independent_dataset,
)
from causalcache.reference_gate import validate_v04_manifest


PNG = b"\x89PNG\r\n\x1a\n" + b"independent-fixture"
TRANSPORT_FILE = "mobile/use/train/shard-00000-of-00610.parquet"


def _tool(name: str) -> dict:
    arguments = {
        "tap": {"coordinate": [200, 300], "clicks": 1},
        "swipe": {"start_coordinate": [500, 800], "coordinate": [500, 200]},
        "type": {"text": "query"},
        "open": {"text": "Settings"},
    }[name]
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def _row(source_id: str, app: str, *, actions: list[str] | None = None) -> dict:
    action_names = actions or ["tap", "tap", "swipe", "type", "tap"]
    messages = []
    for index, action_name in enumerate(action_names):
        user_content = [{"type": "image", "index": index}]
        if index == 0:
            user_content.append({"type": "text", "text": "Complete the fixture task."})
        messages.append({"role": "user", "content": user_content})
        tool_calls = [_tool(action_name)]
        if index == len(action_names) - 1:
            tool_calls.append(
                {
                    "type": "function",
                    "function": {"name": "terminate", "arguments": {"status": "success"}},
                }
            )
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": f"step {index}"}],
                "tool_calls": tool_calls,
            }
        )
    return {
        "messages": json.dumps(messages),
        "metadata": json.dumps(
            {
                "platform": "mobile",
                "others": {
                    "source_id": source_id,
                    "apps": [app],
                    "device_name": "Fixture",
                    "resolution": [720, 1280],
                },
            }
        ),
        "images": [{"bytes": PNG, "path": None} for _ in action_names],
    }


class IndependentGUIOdysseyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[2]
        cls.base_config = json.loads(
            (root / "code" / "configs" / "independent_reference_gate_v1.json").read_text(
                encoding="utf-8"
            )
        )

    def _config(self) -> dict:
        config = copy.deepcopy(self.base_config)
        config["source_pool"]["transport_files"] = [TRANSPORT_FILE]
        for split in ("reference_gate", "oracle_pilot"):
            config["selection"][split] = {
                "minimum_trajectories": 2,
                "minimum_decisions": 8,
                "minimum_distinct_app_labels": 2,
                "minimum_candidate_count_per_required_action_type": 1,
            }
        return config

    def _candidate(self, source_id: str, row_index: int, app: str, config: dict):
        result = inspect_candidate(
            _row(source_id, app),
            transport_file=TRANSPORT_FILE,
            transport_row_index=row_index,
            config=config,
        )
        self.assertIsNone(result.exclusion_reason)
        self.assertIsNotNone(result.candidate)
        return result.candidate

    def test_policy_blind_inspection_and_named_exclusions(self) -> None:
        config = self._config()
        row = _row("safe-id", "  ＢＢＣ\t News ")
        metadata = json.loads(row["metadata"])
        metadata["others"]["apps"].append("bbc news")
        row["metadata"] = json.dumps(metadata)
        result = inspect_candidate(
            row,
            transport_file=TRANSPORT_FILE,
            transport_row_index=0,
            config=config,
        )
        self.assertEqual(result.candidate.normalized_app_labels, ("bbc news",))
        self.assertEqual(result.candidate.decision_count, 4)
        self.assertEqual(
            result.candidate.action_counts(),
            {"swipe": 1, "tap": 2, "type_text": 1},
        )

        excluded = inspect_candidate(
            _row("0054832199799795", "Old"),
            transport_file=TRANSPORT_FILE,
            transport_row_index=1,
            config=config,
        )
        self.assertEqual(excluded.exclusion_reason, ExclusionReason.EXCLUDED_SOURCE_ID)

        unsafe = inspect_candidate(
            _row("../unsafe", "Unsafe"),
            transport_file=TRANSPORT_FILE,
            transport_row_index=2,
            config=config,
        )
        self.assertEqual(unsafe.exclusion_reason, ExclusionReason.UNSAFE_SOURCE_ID)

        incompatible = inspect_candidate(
            _row("open-action", "Open", actions=["tap", "open", "swipe", "type", "tap"]),
            transport_file=TRANSPORT_FILE,
            transport_row_index=3,
            config=config,
        )
        self.assertEqual(
            incompatible.exclusion_reason,
            ExclusionReason.PARSER_INCOMPATIBLE_ACTION_TYPE,
        )

        terminal_row = _row("early-terminal", "Terminal")
        terminal_messages = json.loads(terminal_row["messages"])
        terminal_call = terminal_messages[-1]["tool_calls"].pop()
        terminal_messages[3]["tool_calls"].append(terminal_call)
        terminal_row["messages"] = json.dumps(terminal_messages)
        early = inspect_candidate(
            terminal_row,
            transport_file=TRANSPORT_FILE,
            transport_row_index=4,
            config=config,
        )
        self.assertEqual(early.exclusion_reason, ExclusionReason.TERMINAL_SIGNAL_NOT_LAST)

    def test_shortest_prefix_splits_are_order_independent_and_disjoint(self) -> None:
        config = self._config()
        candidates = [
            self._candidate(f"source-{index}", index, f"app-{index}", config)
            for index in range(6)
        ]
        selected = select_splits(reversed(candidates), config=config)
        ordered = sorted(candidates, key=candidate_sort_key)
        self.assertEqual(list(selected.reference_gate), ordered[:2])
        self.assertEqual(list(selected.oracle_pilot), ordered[2:4])
        self.assertTrue(
            {item.source_id for item in selected.reference_gate}.isdisjoint(
                item.source_id for item in selected.oracle_pilot
            )
        )
        repeated = select_splits(candidates, config=config)
        self.assertEqual(selected, repeated)

    def test_manifest_and_tar_are_deterministic(self) -> None:
        config = self._config()
        rows = {
            (TRANSPORT_FILE, index): _row(f"source-{index}", f"app-{index}")
            for index in range(4)
        }
        candidates = [
            self._candidate(f"source-{index}", index, f"app-{index}", config)
            for index in range(4)
        ]
        selected = select_splits(candidates, config=config)
        spec = SourceFileSpec(TRANSPORT_FILE, 123, "a" * 64)
        manifest, images = build_independent_manifest(
            selected=selected,
            selected_rows=rows,
            config=config,
            source_specs=[spec],
            source_row_counts={TRANSPORT_FILE: 4},
            exclusion_counts={"platform_mismatch": 1},
            total_source_rows=5,
            protocol_config_sha256="b" * 64,
            source_file_manifest_sha256="c" * 64,
        )
        self.assertEqual(manifest["schema_version"], "0.4.0")
        self.assertEqual(manifest["splits"]["reference_gate"]["trajectory_count"], 2)
        self.assertEqual(manifest["splits"]["oracle_pilot"]["trajectory_count"], 2)
        self.assertEqual(
            manifest["splits"]["reference_gate"]["action_type_counts"],
            {"swipe": 2, "tap": 4, "type_text": 2},
        )
        validate_v04_manifest(manifest, config=config)
        for trajectory in manifest["trajectories"]:
            self.assertIn(trajectory["split"], {"reference_gate", "oracle_pilot"})
            self.assertEqual(len(trajectory["selection_sha256"]), 64)
            self.assertEqual(trajectory["transport_file"], TRANSPORT_FILE)

        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_tar = write_independent_dataset(
                Path(first),
                manifest=manifest,
                image_payloads=images,
                shard_relative_path="data/guiodyssey-independent-00000.tar",
            )
            second_tar = write_independent_dataset(
                Path(second),
                manifest=manifest,
                image_payloads=dict(reversed(list(images.items()))),
                shard_relative_path="data/guiodyssey-independent-00000.tar",
            )
            self.assertEqual(first_tar.read_bytes(), second_tar.read_bytes())
            with tarfile.open(first_tar) as archive:
                names = archive.getnames()
                self.assertEqual(names[0], "manifest.json")
                self.assertEqual(names[1:], sorted(names[1:]))
                embedded = archive.extractfile("manifest.json")
                self.assertIsNotNone(embedded)
                self.assertEqual(embedded.read(), (Path(first) / "manifest.json").read_bytes())

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unsafe tar member"):
                write_independent_dataset(
                    Path(directory),
                    manifest=manifest,
                    image_payloads={"../escape.png": PNG},
                    shard_relative_path="data/guiodyssey-independent-00000.tar",
                )

    def test_source_manifest_and_local_file_are_verified_before_rows(self) -> None:
        config = self._config()
        payload = b"pinned parquet fixture"
        digest = hashlib.sha256(payload).hexdigest()
        source_manifest = {
            "schema_version": "0.1.0",
            "protocol_id": config["protocol_id"],
            "repo": config["source_pool"]["transport_repo"],
            "revision": config["source_pool"]["transport_revision"],
            "total_bytes": len(payload),
            "files": [{"path": TRANSPORT_FILE, "size": len(payload), "sha256": digest}],
        }
        specs = source_file_specs(config, source_manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).joinpath(*TRANSPORT_FILE.split("/"))
            path.parent.mkdir(parents=True)
            path.write_bytes(payload)
            verify_local_source_files(Path(directory), specs)
            path.write_bytes(payload + b"corrupt")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_local_source_files(Path(directory), specs)


if __name__ == "__main__":
    unittest.main()
