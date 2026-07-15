import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.validate_restoration_v2_executor_dispatch import (
    FROZEN_ACTION_FIXTURE_SHA256,
    FROZEN_ANDROIDWORLD_REVISION,
    FROZEN_EXECUTION_HOST,
    FROZEN_INSPECTOR_EVIDENCE_TYPE,
    FROZEN_INTERFACE_MANIFEST_SHA256,
    FROZEN_LIVE_SOURCE_SHA256,
    FROZEN_SCIENTIFIC_CONTRACT_SHA256,
    FROZEN_SERVER_IMAGE,
    FROZEN_SERVER_IMAGE_ID,
    FROZEN_STACK_CONFIG_SHA256,
    NEGATIVE_ACTUATION_CONTROL,
    REPOSITORY_ROOT,
    dispatch_actions,
    expected_executor_message,
    load_dispatch_cases,
    validate_dispatch_evidence,
    write_summary,
)
from scripts.validate_restoration_v2_executor_evidence import (
    FROZEN_RUNTIME_IMAGE_ID,
    FROZEN_RUNTIME_REPO_DIGEST,
    _source_sha256,
    _validate_run_commit_sources,
    package,
    validate_summary,
)


def transport_record(
    body: dict[str, object] | None,
    *,
    status: int = 200,
    content_type: str = "application/json",
    started_at: str = "2026-07-15T00:00:20+00:00",
    finished_at: str = "2026-07-15T00:00:21+00:00",
) -> dict[str, object]:
    raw = (
        json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if body is not None
        else b"Internal Server Error"
    )
    record: dict[str, object] = {
        "started_at": started_at,
        "finished_at": finished_at,
        "http_status": status,
        "content_type": content_type,
        "response_bytes": len(raw),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "elapsed_seconds": 0.1,
        "raw_body_utf8": raw.decode("utf-8"),
    }
    if body is None:
        record["body_text"] = raw.decode("utf-8")
    else:
        record["json_body"] = body
    return record


class RestorationV2ExecutorEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_dispatch_cases(
            REPOSITORY_ROOT / "data/fixtures/gui_owl_v2_action_roundtrip.json"
        )
        cls.attempt_id = "rv2-20260715T000000Z-deadbeef"
        cls.run_commit = "a" * 40
        cls.runtime_id = "b" * 64
        cls.server_id = "c" * 64

    def setUp(self) -> None:
        self.run_commit_patch = mock.patch(
            "scripts.validate_restoration_v2_executor_evidence._validate_run_commit_sources"
        )
        self.run_commit_patch.start()

    def tearDown(self) -> None:
        self.run_commit_patch.stop()

    @staticmethod
    def _requester(
        base_url: str,
        path: str,
        *,
        method: str = "GET",
        params=None,
        body=None,
        timeout_seconds: float,
    ) -> dict[str, object]:
        if path == "/health":
            return transport_record({"status": "success"})
        if path == "/reset":
            return transport_record(
                {
                    "status": "success",
                    "message": "Environment reset with go_home=True.",
                }
            )
        if body == NEGATIVE_ACTUATION_CONTROL:
            return transport_record(None, status=500, content_type="text/plain")
        return transport_record(
            {
                "status": "success",
                "message": expected_executor_message(body),
            }
        )

    @staticmethod
    def _screenshot(base_url: str, *, timeout_seconds: float):
        return {
            "started_at": "2026-07-15T00:00:20+00:00",
            "finished_at": "2026-07-15T00:00:21+00:00",
            "http_status": 200,
            "content_type": "application/json",
            "response_bytes": 10,
            "response_sha256": "1" * 64,
            "width": 1080,
            "height": 2400,
            "elapsed_seconds": 0.1,
        }

    def _host_record(self, phase: str) -> dict[str, object]:
        if phase == "before":
            started_at = "2026-07-15T00:00:00+00:00"
            finished_at = "2026-07-15T00:00:05+00:00"
        else:
            started_at = "2026-07-15T00:01:05+00:00"
            finished_at = "2026-07-15T00:01:10+00:00"
        return {
            "schema_version": "1.0.0",
            "evidence_type": FROZEN_INSPECTOR_EVIDENCE_TYPE,
            "status": "passed",
            "attempt_id": self.attempt_id,
            "phase": phase,
            "started_at": started_at,
            "finished_at": finished_at,
            "host": FROZEN_EXECUTION_HOST,
            "inspector_source_sha256": _source_sha256()["host_inspector"],
            "runtime_container": {
                "name": "sglang-omni-jaxan-07150000",
                "id": self.runtime_id,
                "hostname": self.runtime_id[:12],
                "config_image": "hongccc/sglang-omni:dev",
                "image_id": FROZEN_RUNTIME_IMAGE_ID,
                "image_repo_digest": f"hongccc/sglang-omni@{FROZEN_RUNTIME_REPO_DIGEST}",
                "mounts": {
                    "/data": "/mnt/data6/jiaxuanluo/causalcache",
                    "/root/.cache/huggingface": "/mnt/data6/jiaxuanluo/causalcache/.cache/huggingface",
                },
                "running": True,
            },
            "server_container": {
                "name": "sglang-omni-jaxan-07150001",
                "id": self.server_id,
                "hostname": self.server_id[:12],
                "config_image": FROZEN_SERVER_IMAGE,
                "image_id": FROZEN_SERVER_IMAGE_ID,
                "host_port": 5003,
                "container_port": 5000,
                "running": True,
            },
            "live_source_sha256": FROZEN_LIVE_SOURCE_SHA256,
            "health": {
                **transport_record(
                    {"status": "success"},
                    started_at=started_at,
                    finished_at=finished_at,
                ),
                "url": "http://127.0.0.1:5003/health",
            },
        }

    def _canonical_summary(self) -> dict[str, object]:
        dispatch = dispatch_actions(
            base_url="http://172.17.0.1:5003",
            cases=self.cases,
            timeout_seconds=120.0,
            requester=self._requester,
            screenshot_requester=self._screenshot,
        )
        before = self._host_record("before")
        after = self._host_record("after")
        before_raw = json.dumps(before, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        after_raw = json.dumps(after, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        summary: dict[str, object] = {
            "schema_version": "1.0.0",
            "evidence_type": "restoration_v2_androidworld_executor_dispatch",
            "protocol_id": "causalcache_restoration_v2",
            "step": "androidworld_executor_dispatch_preflight",
            "status": "passed",
            "attempt_id": self.attempt_id,
            "started_at": "2026-07-15T00:00:10+00:00",
            "finished_at": "2026-07-15T00:01:00+00:00",
            "run_git_commit": self.run_commit,
            "source_sha256": _source_sha256(),
            "policy_output_generated": False,
            "restoration_label_generated": False,
            "interface_validation": {
                "scientific_contract_sha256": FROZEN_SCIENTIFIC_CONTRACT_SHA256,
                "action_fixture": {
                    "sha256": FROZEN_ACTION_FIXTURE_SHA256,
                    "valid_case_count": 14,
                    "invalid_case_count": 23,
                    "coordinate_scalar_checks": 6000,
                },
                "interface_manifest": {"sha256": FROZEN_INTERFACE_MANIFEST_SHA256},
                "androidworld_json_action_constructor_validation": {
                    "status": "passed",
                    "validated_case_count": 14,
                    "source_revision": FROZEN_ANDROIDWORLD_REVISION,
                    "run_git_commit": self.run_commit,
                    "container_image_digest": FROZEN_RUNTIME_REPO_DIGEST,
                    "module_path": "/pinned/android_world/agents/new_json_action.py",
                    "module_sha256": FROZEN_LIVE_SOURCE_SHA256[
                        "/android_world/agents/new_json_action.py"
                    ],
                },
            },
            "negative_control_constructor_validation": {
                "status": "passed",
                "payload": NEGATIVE_ACTUATION_CONTROL,
                "json_action_repr": "JSONAction(action_type='click')",
                "json_action_json_str": '{"action_type":"click"}',
                "module_path": "/pinned/android_world/agents/new_json_action.py",
                "module_sha256": FROZEN_LIVE_SOURCE_SHA256[
                    "/android_world/agents/new_json_action.py"
                ],
            },
            "server_provenance": {
                "host": FROZEN_EXECUTION_HOST,
                "base_url": "http://172.17.0.1:5003",
                "androidworld_source_revision": FROZEN_ANDROIDWORLD_REVISION,
                "stack_config_sha256": FROZEN_STACK_CONFIG_SHA256,
                "live_source_sha256": FROZEN_LIVE_SOURCE_SHA256,
                "runtime_container": {
                    "name": "sglang-omni-jaxan-07150000",
                    "id": self.runtime_id,
                    "image_id": FROZEN_RUNTIME_IMAGE_ID,
                    "image_repo_digest": FROZEN_RUNTIME_REPO_DIGEST,
                    "mounts": {
                        "/data": "/mnt/data6/jiaxuanluo/causalcache",
                        "/root/.cache/huggingface": "/mnt/data6/jiaxuanluo/causalcache/.cache/huggingface",
                    },
                },
                "server_container": {
                    "name": "sglang-omni-jaxan-07150001",
                    "id": self.server_id,
                    "image": FROZEN_SERVER_IMAGE,
                    "image_id": FROZEN_SERVER_IMAGE_ID,
                    "host_port": 5003,
                    "container_port": 5000,
                    "json_action_schema": "android_world.agents.new_json_action",
                    "host_health_url": "http://127.0.0.1:5003/health",
                },
                "host_inspection_before": before,
                "host_inspection_before_raw_utf8": before_raw,
                "host_inspection_sha256": hashlib.sha256(
                    before_raw.encode("utf-8")
                ).hexdigest(),
                "host_inspection_after": after,
                "host_inspection_after_raw_utf8": after_raw,
                "host_inspection_after_sha256": hashlib.sha256(
                    after_raw.encode("utf-8")
                ).hexdigest(),
            },
            "androidworld_executor_dispatch_validation": dispatch,
            "canonical_packaging": {
                "attempt_id": self.attempt_id,
                "packaged_at": "2026-07-15T00:01:15+00:00",
                "after_inspection_sha256": hashlib.sha256(
                    after_raw.encode("utf-8")
                ).hexdigest(),
                "offline_validator_sha256": _source_sha256()["offline_validator"],
            },
        }
        summary["offline_reduction"] = validate_dispatch_evidence(
            summary, cases=self.cases
        )
        raw_reconstruction = copy.deepcopy(summary)
        raw_reconstruction.pop("canonical_packaging")
        raw_provenance = raw_reconstruction["server_provenance"]
        raw_provenance.pop("host_inspection_after")
        raw_provenance.pop("host_inspection_after_raw_utf8")
        raw_provenance.pop("host_inspection_after_sha256")
        raw_summary = (
            json.dumps(raw_reconstruction, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        summary["canonical_packaging"]["raw_summary_raw_utf8"] = raw_summary
        summary["canonical_packaging"]["raw_summary_sha256"] = hashlib.sha256(
            raw_summary.encode("utf-8")
        ).hexdigest()
        return summary

    def test_offline_reducer_validates_full_provenance_and_time_bracket(self) -> None:
        summary = self._canonical_summary()
        self.assertEqual(validate_summary(summary)["status"], "PASSED_EXECUTOR_DISPATCH")

        wrong_constructor = copy.deepcopy(summary)
        wrong_constructor["interface_validation"][
            "androidworld_json_action_constructor_validation"
        ]["validated_case_count"] = 13
        with self.assertRaisesRegex(ValueError, "constructor"):
            validate_summary(wrong_constructor)

        changed_after = copy.deepcopy(summary)
        changed_after["server_provenance"]["host_inspection_after"][
            "server_container"
        ]["id"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "parsed after"):
            validate_summary(changed_after)

        stale_before = copy.deepcopy(summary)
        stale_before["started_at"] = "2026-07-15T01:00:00+00:00"
        stale_before["finished_at"] = "2026-07-15T01:01:00+00:00"
        with self.assertRaisesRegex(ValueError, "timestamps|stale"):
            validate_summary(stale_before)

        mismatched_top_level_port = copy.deepcopy(summary)
        mismatched_top_level_port["server_provenance"]["server_container"][
            "host_port"
        ] = 5002
        mismatched_top_level_port["server_provenance"]["base_url"] = (
            "http://172.17.0.1:5002"
        )
        with self.assertRaisesRegex(ValueError, "embedded live inspection"):
            validate_summary(mismatched_top_level_port)

        health_outside_interval = copy.deepcopy(summary)
        before = health_outside_interval["server_provenance"]["host_inspection_before"]
        before["health"]["started_at"] = "2026-07-15T00:00:06+00:00"
        before["health"]["finished_at"] = "2026-07-15T00:00:07+00:00"
        before_raw = json.dumps(before, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        health_outside_interval["server_provenance"][
            "host_inspection_before_raw_utf8"
        ] = before_raw
        health_outside_interval["server_provenance"]["host_inspection_sha256"] = (
            hashlib.sha256(before_raw.encode("utf-8")).hexdigest()
        )
        with self.assertRaisesRegex(ValueError, "outside its interval"):
            validate_summary(health_outside_interval)

    def test_run_commit_must_exist_and_contain_recorded_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            _validate_run_commit_sources("a" * 40, _source_sha256())

    def test_package_embeds_post_inspection_and_never_overwrites_outputs(self) -> None:
        canonical = self._canonical_summary()
        raw = copy.deepcopy(canonical)
        raw["server_provenance"].pop("host_inspection_after")
        raw["server_provenance"].pop("host_inspection_after_raw_utf8")
        raw["server_provenance"].pop("host_inspection_after_sha256")
        raw.pop("canonical_packaging")
        after = canonical["server_provenance"]["host_inspection_after"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / f"raw-{self.attempt_id}.json"
            after_path = root / f"after-{self.attempt_id}.json"
            raw_path.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            after_path.write_text(
                json.dumps(after, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            packaged = package(raw_path, after_path)
            self.assertEqual(validate_summary(packaged)["status"], "PASSED_EXECUTOR_DISPATCH")

            output_path = root / "immutable.json"
            write_summary(output_path, {"attempt_id": self.attempt_id})
            with self.assertRaises(FileExistsError):
                write_summary(output_path, {"attempt_id": self.attempt_id})


if __name__ == "__main__":
    unittest.main()
