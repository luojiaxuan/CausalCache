from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TuneOnPolicyScheduleTest(unittest.TestCase):
    def test_materializes_deduplicated_tune_only_coalitions(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            input_root = root / "input"
            input_root.mkdir()
            state = {
                "candidate_event_step_ids": [1, 2, 3, 4, 5],
                "logical_shard": 3,
                "role": "tune",
                "state_id": "7:decision:006",
                "trajectory_id": "7",
            }
            (input_root / "states.jsonl").write_text(
                json.dumps(state) + "\n", encoding="utf-8"
            )
            (input_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "content_sha256": "i" * 64,
                        "evaluation_labels_included": False,
                        "states_jsonl": "states.jsonl",
                    }
                ),
                encoding="utf-8",
            )
            selection_paths = []
            learned = (
                {"1": [1], "2": [1, 2], "3": [1, 2], "4": [1, 2, 4]},
                {"1": [5], "2": [3, 5], "3": [2, 3, 5], "4": [1, 2, 3, 5]},
            )
            recent = {"1": [5], "2": [4, 5], "3": [3, 4, 5], "4": [2, 3, 4, 5]}
            for index in range(2):
                path = root / f"selection-{index}.json"
                payload = {
                    "cache_content_sha256": "c" * 64,
                    "checkpoint_sha256": str(index) * 64,
                    "config_sha256": str(index + 2) * 64,
                    "content_sha256": chr(97 + index) * 64,
                    "input_content_sha256": "i" * 64,
                    "records": [
                        {
                            "candidate_event_ids": [1, 2, 3, 4, 5],
                            "learned": learned[index],
                            "logical_shard": 3,
                            "recent": recent,
                            "state_id": state["state_id"],
                            "trajectory_id": "7",
                        }
                    ],
                    "status": "COMPLETED_SET_UTILITY_TUNE_SELECTIONS",
                    "variant": f"variant_{index}",
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                selection_paths.append(path)
            output_root = root / "output"
            repository_root = Path(__file__).resolve().parents[2]
            command = [
                sys.executable,
                str(
                    repository_root
                    / "code/scripts/materialize_set_utility_tune_on_policy_schedules.py"
                ),
                "--input-root",
                str(input_root),
                "--selection",
                f"deepsets={selection_paths[0]}",
                "--selection",
                f"set_transformer={selection_paths[1]}",
                "--output-root",
                str(output_root),
            ]
            environment = {**os.environ, "PYTHONPATH": str(repository_root / "code")}
            completed = subprocess.run(command, env=environment, capture_output=True)
            self.assertEqual(
                completed.returncode,
                0,
                completed.stderr.decode("utf-8", errors="replace"),
            )
            row = json.loads(
                (
                    output_root / "schedule-shards/shard-003-of-256.jsonl"
                ).read_text(encoding="utf-8")
            )
            coalitions = [tuple(value["event_ids"]) for value in row["coalitions"]]
            self.assertEqual(coalitions[0], ())
            self.assertEqual(coalitions[-1], (1, 2, 3, 4, 5))
            self.assertEqual(len(coalitions), len(set(coalitions)))
            manifest = json.loads(
                (output_root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["state_count"], 1)
            self.assertEqual(
                manifest["status"],
                "COMPLETED_SET_UTILITY_TUNE_ON_POLICY_SCHEDULES",
            )


if __name__ == "__main__":
    unittest.main()
