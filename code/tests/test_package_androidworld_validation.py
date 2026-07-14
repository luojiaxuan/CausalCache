import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.package_androidworld_validation import package_validation


SOURCE_COMMIT = "a" * 40
POLICY_SLUG = "gui-owl-1.5-8b-think"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class AndroidWorldValidationPackageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plan_path = self.root / "plan.json"
        self.run_dir = self.root / "run"
        (self.run_dir / "episodes").mkdir(parents=True)
        self.instances = [
            {
                "task_type": "TaskAlpha",
                "task_index": 0,
                "goal": "第一步",
            },
            {
                "task_type": "TaskBeta",
                "task_index": 0,
                "goal": "second",
            },
            {
                "task_type": "TaskGamma",
                "task_index": 0,
                "goal": "third",
            },
            {
                "task_type": "TaskDelta",
                "task_index": 0,
                "goal": "fourth",
            },
        ]
        self.records_sha256 = hashlib.sha256(canonical_bytes(self.instances)).hexdigest()
        self.plan = {
            "schema_version": "0.1.0",
            "split": "validation",
            "task_instance_count": len(self.instances),
            "instance_records_sha256": self.records_sha256,
            "instances": self.instances,
        }
        self.plan_path.write_text(
            json.dumps(self.plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.run_contract = {
            "git_commit": SOURCE_COMMIT,
            "validation_plan_records_sha256": self.records_sha256,
            "model": {"repo": "owner/model", "revision": "model-sha"},
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def episode(
        self,
        plan_index: int,
        *,
        run_status: str = "complete",
        official_success: bool = False,
        termination_reason: str = "policy_terminated",
    ) -> dict:
        return {
            "schema_version": "0.1.0",
            "plan_index": plan_index,
            "instance": self.instances[plan_index],
            "run_contract": self.run_contract,
            "run_status": run_status,
            "model_step_count": 1,
            "parse_success_count": 1,
            "environment_success": official_success,
            "official_success": official_success,
            "termination_reason": termination_reason,
        }

    def episode_path(self, plan_index: int) -> Path:
        instance = self.instances[plan_index]
        return self.run_dir / "episodes" / (
            f"{plan_index:03d}-{instance['task_type']}-{instance['task_index']}.json"
        )

    def write_run(self, episodes: list[dict], *, early_stop: bool) -> dict:
        episodes = sorted(episodes, key=lambda episode: episode["plan_index"])
        for episode in episodes:
            self.episode_path(episode["plan_index"]).write_text(
                json.dumps(episode, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        completed = sum(episode["run_status"] == "complete" for episode in episodes)
        exceptions = sum(episode["run_status"] == "exception" for episode in episodes)
        official_successes = sum(episode["official_success"] for episode in episodes)
        outcomes: dict[str, int] = {}
        termination_reasons: dict[str, int] = {}
        for episode in episodes:
            reason = episode["termination_reason"]
            termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
            if episode["run_status"] == "exception":
                outcome = "infrastructure_failure"
            elif reason == "parse_error":
                outcome = "parse_failure"
            elif reason == "executor_error":
                outcome = "executor_failure"
            elif episode["official_success"]:
                outcome = "official_success"
            else:
                outcome = "terminal_failure"
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        summary = {
            "schema_version": "0.1.0",
            "split": "validation",
            "instance_records_sha256": self.records_sha256,
            "completed_episode_count": completed,
            "exception_episode_count": exceptions,
            "model_step_count": len(episodes),
            "parse_success_count": len(episodes),
            "parse_coverage": 1.0,
            "official_success_count": official_successes,
            "outcomes": dict(sorted(outcomes.items())),
            "termination_reasons": dict(sorted(termination_reasons.items())),
            "run_contract": self.run_contract,
            "episode_files": [
                f"episodes/{self.episode_path(episode['plan_index']).name}"
                for episode in episodes
            ],
        }
        if early_stop:
            unobserved = len(self.instances) - len(episodes)
            maximum_possible = official_successes + unobserved
            summary.update(
                {
                    "artifact_status": "valid_early_stopped_policy_rejection",
                    "plan_instance_count": len(self.instances),
                    "checkpoint_count": len(episodes),
                    "unobserved_instance_count": unobserved,
                    "official_success_rate_lower_bound": official_successes
                    / len(self.instances),
                    "minimum_required_official_success_count": 3,
                    "maximum_possible_official_success_count": maximum_possible,
                    "maximum_possible_official_success_rate": maximum_possible
                    / len(self.instances),
                    "early_stop_reason": "success_gate_mathematically_impossible",
                    "gates": {
                        "minimum_parse_coverage": 0.95,
                        "parse_gate_passed_on_observed_actions": True,
                        "minimum_official_success": 0.75,
                        "official_success_gate_passed": False,
                        "validation_gate_passed": False,
                    },
                }
            )
        else:
            success_rate = official_successes / len(self.instances)
            summary.update(
                {
                    "task_instance_count": len(self.instances),
                    "environment_success_count": official_successes,
                    "official_success_rate": success_rate,
                    "gates": {
                        "minimum_parse_coverage": 0.95,
                        "parse_gate_passed": True,
                        "minimum_official_success": 0.5,
                        "official_success_gate_passed": success_rate >= 0.5,
                        "validation_gate_passed": success_rate >= 0.5,
                    },
                }
            )
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary

    def package(self, output_dir: Path) -> dict:
        return package_validation(
            validation_run_dir=self.run_dir,
            validation_plan=self.plan_path,
            output_dir=output_dir,
            policy_slug=POLICY_SLUG,
            source_run_git_commit=SOURCE_COMMIT,
            hf_repo="gavinlaw/causalcache-androidworld-validation-mobile",
            hf_tag="v0.2.0-think",
        )

    def test_complete_package_is_deterministic_and_plan_ordered(self) -> None:
        episodes = [
            self.episode(0, official_success=True),
            self.episode(1, official_success=True),
            self.episode(2),
            self.episode(
                3,
                run_status="exception",
                termination_reason="exception",
            ),
        ]
        summary = self.write_run(episodes, early_stop=False)
        original_summary_bytes = (self.run_dir / "summary.json").read_bytes()
        first_output = self.root / "package-one"
        second_output = self.root / "package-two"
        first_manifest = self.package(first_output)
        second_manifest = self.package(second_output)

        shard_relative = Path("data") / POLICY_SLUG / "validation-00000-of-00001.jsonl.gz"
        summary_relative = Path("runs") / POLICY_SLUG / "summary.json"
        manifest_relative = Path("runs") / POLICY_SLUG / "payload_manifest.json"
        first_shard = (first_output / shard_relative).read_bytes()
        second_shard = (second_output / shard_relative).read_bytes()
        self.assertEqual(first_shard, second_shard)
        self.assertEqual(first_shard[3], 0)
        self.assertEqual(first_shard[4:8], b"\x00\x00\x00\x00")
        records = [json.loads(line) for line in gzip.decompress(first_shard).splitlines()]
        self.assertEqual([record["plan_index"] for record in records], [0, 1, 2, 3])
        self.assertIn("第一步".encode("utf-8"), gzip.decompress(first_shard))
        self.assertEqual((first_output / summary_relative).read_bytes(), original_summary_bytes)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_manifest["run_kind"], "complete")
        self.assertEqual(first_manifest["layout"]["trace_shard"], shard_relative.as_posix())
        self.assertEqual(first_manifest["layout"]["summary"], summary_relative.as_posix())
        self.assertEqual(first_manifest["layout"]["manifest"], manifest_relative.as_posix())
        self.assertEqual(
            first_manifest["files"]["trace_shard"]["sha256"],
            hashlib.sha256(first_shard).hexdigest(),
        )
        self.assertEqual(first_manifest["files"]["trace_shard"]["bytes"], len(first_shard))
        self.assertEqual(first_manifest["source_run_git_commit"], SOURCE_COMMIT)
        self.assertNotIn("revision", first_manifest["hf_destination"])
        self.assertEqual(
            json.loads((first_output / manifest_relative).read_text(encoding="utf-8")),
            first_manifest,
        )
        self.assertEqual(json.loads(original_summary_bytes), summary)

    def test_mathematically_decisive_early_stop_can_be_packaged(self) -> None:
        self.write_run([self.episode(0), self.episode(2)], early_stop=True)
        output = self.root / "early-package"
        manifest = self.package(output)
        self.assertEqual(manifest["run_kind"], "early_stopped_policy_rejection")
        self.assertEqual(manifest["record_count"], 2)
        shard = output / "data" / POLICY_SLUG / "validation-00000-of-00001.jsonl.gz"
        records = [json.loads(line) for line in gzip.decompress(shard.read_bytes()).splitlines()]
        self.assertEqual([record["plan_index"] for record in records], [0, 2])

    def test_rejects_mismatched_summary_episode_and_contracts(self) -> None:
        cases = ("summary_count", "duplicate_index", "instance", "source_commit", "extra_file")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as case_directory:
                case_root = Path(case_directory)
                old_run_dir = self.run_dir
                self.run_dir = case_root / "run"
                (self.run_dir / "episodes").mkdir(parents=True)
                self.write_run(
                    [self.episode(index) for index in range(len(self.instances))],
                    early_stop=False,
                )
                if case == "summary_count":
                    summary_path = self.run_dir / "summary.json"
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    summary["completed_episode_count"] -= 1
                    summary_path.write_text(json.dumps(summary), encoding="utf-8")
                elif case == "duplicate_index":
                    path = self.episode_path(1)
                    episode = json.loads(path.read_text(encoding="utf-8"))
                    episode["plan_index"] = 0
                    path.write_text(json.dumps(episode), encoding="utf-8")
                elif case == "instance":
                    path = self.episode_path(1)
                    episode = json.loads(path.read_text(encoding="utf-8"))
                    episode["instance"] = self.instances[0]
                    path.write_text(json.dumps(episode), encoding="utf-8")
                elif case == "source_commit":
                    summary_path = self.run_dir / "summary.json"
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    summary["run_contract"]["git_commit"] = "b" * 40
                    summary_path.write_text(json.dumps(summary), encoding="utf-8")
                else:
                    (self.run_dir / "episodes" / "unexpected.json").write_text(
                        "{}", encoding="utf-8"
                    )
                with self.assertRaises(ValueError):
                    self.package(case_root / "package")
                self.run_dir = old_run_dir

    def test_rejects_nonempty_output_and_unsafe_policy_slug(self) -> None:
        self.write_run(
            [self.episode(index) for index in range(len(self.instances))],
            early_stop=False,
        )
        output = self.root / "occupied"
        output.mkdir()
        (output / "keep.txt").write_text("do not overwrite", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.package(output)
        self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "do not overwrite")

        with self.assertRaisesRegex(ValueError, "policy slug"):
            package_validation(
                validation_run_dir=self.run_dir,
                validation_plan=self.plan_path,
                output_dir=self.root / "unsafe",
                policy_slug="../think",
                source_run_git_commit=SOURCE_COMMIT,
                hf_repo="gavinlaw/causalcache-androidworld-validation-mobile",
                hf_tag="v0.2.0-think",
            )


if __name__ == "__main__":
    unittest.main()
