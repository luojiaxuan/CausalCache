from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from causalcache.osworld_benchmark import (
    policy_endpoint_for_worker,
    smoke_policy_action,
    validate_osworld_benchmark_roster,
)


class OSWorldBenchmarkTests(unittest.TestCase):
    def test_policy_replica_assignment_is_round_robin(self) -> None:
        endpoints = ("http://a/act", "http://b/act")
        self.assertEqual(policy_endpoint_for_worker(endpoints, 0), endpoints[0])
        self.assertEqual(policy_endpoint_for_worker(endpoints, 3), endpoints[1])

    def test_smoke_policy_waits_then_stops(self) -> None:
        request = {
            "schema_version": "causalcache.osworld.policy_request.v1",
            "step_id": 1,
        }
        self.assertEqual(smoke_policy_action(request), {"type": "wait"})
        request["step_id"] = 2
        self.assertEqual(smoke_policy_action(request), {"type": "done"})

    def test_roster_difference_is_identity_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = root / "evaluation_examples/examples/multi_apps"
            examples.mkdir(parents=True)
            (root / "evaluation_examples/test_all.json").write_text(
                json.dumps({"multi_apps": ["a", "b", "c"]}), encoding="utf-8"
            )
            (root / "evaluation_examples/test_nogdrive.json").write_text(
                json.dumps({"multi_apps": ["a", "c"]}), encoding="utf-8"
            )
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=root, check=True
            )
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "fixture"], cwd=root, check=True
            )
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            config = {
                "osworld_revision": revision,
                "roster": {
                    "full_meta_path": "evaluation_examples/test_all.json",
                    "selected_meta_path": "evaluation_examples/test_nogdrive.json",
                    "expected_full_task_count": 3,
                    "expected_selected_task_count": 2,
                    "expected_excluded_task_count": 1,
                    "excluded_tasks": [
                        {"domain": "multi_apps", "task_id": "b"}
                    ],
                },
            }
            roster = validate_osworld_benchmark_roster(root, config)
            self.assertEqual(roster.excluded, (("multi_apps", "b"),))
            config["roster"]["excluded_tasks"][0]["task_id"] = "c"
            with self.assertRaisesRegex(ValueError, "identities drifted"):
                validate_osworld_benchmark_roster(root, config)


if __name__ == "__main__":
    unittest.main()
